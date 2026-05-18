#!/usr/bin/env python3
"""Normalize the official DialSim/LongDialQA v1.1 pickles.

The DialSim simulator keeps LongDialQA in large pickle files and samples one
question during each scene with Python's random module.  This adapter makes the
protocol explicit and reproducible:

* `sessions.jsonl`: one row per scene/session with parsed turns.
* `selected_qa.jsonl`: one row per seeded simulator question attempt.
* `manifest.json`: source hashes, protocol parameters, and summary counts.

The selected QA rows preserve the official multi-choice format and the online
ask point: the history available to an evaluated memory system is all previous
turns plus the current scene prefix through `ask_turn_index`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SHOWS = ("friends", "bigbang", "theoffice")
SHOW_NAMES = {
    "friends": "Friends",
    "bigbang": "The Big Bang Theory",
    "theoffice": "The Office",
}
CHATBOT = {
    "friends": "Ross",
    "bigbang": "Sheldon",
    "theoffice": "Michael",
}


@dataclass(frozen=True)
class ParsedTurn:
    turn_id: str
    turn_index: int
    speaker: str
    text: str
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/DialSim/v1.1"),
        help="Directory containing *_dialsim.pickle and oracle pickles.",
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("datasets/DialSim"),
        help="Directory containing the downloaded dialsim_v*.zip archives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/DialSim/longdialqa_normalized_v1.1_seed0"),
        help="Directory where normalized artifacts are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for deterministic replay of DialSim question sampling.",
    )
    parser.add_argument(
        "--tkg-ratio",
        type=float,
        default=0.7,
        help="Official simulator ratio for KG/temporal hard questions.",
    )
    parser.add_argument(
        "--shows",
        nargs="+",
        choices=SHOWS,
        default=list(SHOWS),
        help="Shows to normalize.",
    )
    parser.add_argument(
        "--max-selected-per-show",
        type=int,
        default=None,
        help="Optional cap on selected QA rows per show, after deterministic replay.",
    )
    parser.add_argument(
        "--max-sessions-per-show",
        type=int,
        default=None,
        help="Optional cap on sessions per show for smoke/debug adapters.",
    )
    parser.add_argument(
        "--include-available-turn-ids",
        action="store_true",
        help="Store the full visible-history turn id list in each QA row. This is large for full runs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    return sha256_file(path)


def git_commit(path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def season_episode(show: str, episode_key: str) -> tuple[int | None, int | None]:
    if show == "friends":
        m = re.search(r"S(\d+)E(\d+)", episode_key)
        if m:
            return int(m.group(1)), int(m.group(2))
    if show == "bigbang":
        m = re.search(r"Series_(\d+)Episode_(\d+)", episode_key)
        if m:
            return int(m.group(1)), int(m.group(2))
    if show == "theoffice":
        m = re.search(r"Season_(\d+)_Episode_(\d+)", episode_key)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def safe_episode_id(show: str, episode_key: str) -> str:
    season, episode = season_episode(show, episode_key)
    if season is not None and episode is not None:
        return f"{show}_s{season:02d}e{episode:02d}"
    stem = Path(episode_key).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{show}_{stem}"


def parse_turns(show: str, episode_key: str, scene_num: Any, script: str) -> list[ParsedTurn]:
    episode_id = safe_episode_id(show, episode_key)
    session_id = f"{episode_id}_scene{int(scene_num):03d}" if str(scene_num).isdigit() else f"{episode_id}_scene{scene_num}"
    post_utterances: list[str] = []
    temp_utter = ""

    for utter in script.split("\n"):
        if not utter.strip():
            continue
        if "Teleplay: " in utter or "Story: " in utter:
            continue
        if ":" in utter:
            temp_utter = utter.strip()
            post_utterances.append(temp_utter)
            continue
        if post_utterances:
            post_utterances.pop()
            temp_utter = f"{temp_utter}\n{utter.strip()}"
            post_utterances.append(temp_utter)

    turns: list[ParsedTurn] = []
    for idx, raw in enumerate(post_utterances, start=1):
        if ":" in raw:
            speaker, text = raw.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
        else:
            speaker = ""
            text = raw.strip()
        turn_id = f"{session_id}_turn{idx:04d}"
        turns.append(ParsedTurn(turn_id=turn_id, turn_index=idx, speaker=speaker, text=text, raw=raw))
    return turns


def option_letter(answer: str, options: list[str]) -> str:
    letters = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    for idx, option in enumerate(options[:5]):
        if str(answer).strip().lower() == str(option).strip().lower():
            return letters[idx]
    return ""


def question_text_for_character(question_obj: dict[str, Any], char_ask: str) -> str:
    questions = question_obj.get("questions") or {}
    if char_ask in questions:
        return questions[char_ask]
    if "default" in questions:
        return questions["default"]
    if questions:
        return questions[next(iter(questions))]
    return ""


def build_question_prompt(char_ask: str, question: str, options: list[str]) -> str:
    prompt = f"{char_ask}: {question}\n"
    labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    prompt += "\n".join(f"\t{label} {option}" for label, option in zip(labels, options))
    return prompt


def is_unanswerable(current_type: str, answer: str, options: list[str]) -> bool:
    if "unans" in current_type:
        return True
    if len(options) >= 5 and str(answer).strip().lower() == str(options[4]).strip().lower():
        return True
    return "i don't know" in str(answer).lower()


def hop_label(source: str, current_type: str) -> str:
    if source == "easy_q":
        return "single_hop"
    if "_" in current_type:
        return "two_hop_temporal"
    if current_type in {"past", "cur", "fu"}:
        return "single_hop_temporal"
    return "unknown"


def normalize_target_dates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.lower() in {"past", "future", "cur", "current"}:
            return []
        return [value]
    if isinstance(value, list):
        dates: list[str] = []
        for item in value:
            dates.extend(normalize_target_dates(item))
        return dates
    return []


def choose_question_candidates(
    *,
    rng: random.Random,
    epi: str,
    sc_num: Any,
    sess: dict[str, Any],
    oracle_tkg: dict[str, Any],
    oracle_fan: dict[str, Any],
    tkg_ratio: float,
) -> tuple[str, str, list[dict[str, Any]], list[Any], str]:
    """Replay the official question selection block for one scene.

    Returns `(source, current_type, target_question_list, target_dates_list, skip_reason)`.
    """
    cannot_tkg = 0
    cannot_fan = 0
    date = sess.get("date", "")
    date_splitted = str(date).replace(",", "").split()

    # The upstream simulator performs these exploratory random choices before
    # the final hard/easy branch.  They are unused but consume RNG state.
    try:
        question_dict = sess["hard_q"]
        final_tkg_list = [tkg for tkg in list(question_dict) if len(question_dict[tkg]) > 0]
        tkg_target_type = rng.choice(final_tkg_list)
        target_question = rng.choice(question_dict[tkg_target_type])
        _ = target_question
    except Exception:
        cannot_tkg = 1

    try:
        question_dict = sess["easy_q"]
        final_fan_list = [fan for fan in list(question_dict) if len(list(question_dict[fan])) > 0]
        fan_target_type = rng.choice(final_fan_list)
        fan_q_target_num = rng.choice(list(question_dict[fan_target_type]))
        target_question = question_dict[fan_target_type][fan_q_target_num]
        _ = target_question
    except Exception:
        cannot_fan = 1

    target_question_list: list[dict[str, Any]] = []
    target_dates_list: list[Any] = []

    rand_val = rng.random()
    if cannot_fan == 1 and cannot_tkg == 1:
        return "", "", [], [], "no_questions"

    if (cannot_fan == 1 and cannot_tkg == 0) or rand_val < tkg_ratio:
        question_dict = sess["hard_q"]
        final_tkg_list = []
        fu_num = 0
        not_fu_list = []
        for tkg in list(question_dict):
            if len(question_dict[tkg]) > 0:
                final_tkg_list.append(tkg)
                if "fu" in tkg:
                    fu_num += 1
                else:
                    not_fu_list.append(tkg)
        if not final_tkg_list:
            return "", "", [], [], "no_hard_questions"
        if len(not_fu_list) > 0:
            rng.shuffle(not_fu_list)
            while True:
                should_stop = 0
                for not_fu in not_fu_list:
                    if fu_num / len(final_tkg_list) < 0.215:
                        should_stop = 1
                        break
                    final_tkg_list.append(not_fu)
                if should_stop == 1:
                    break
        current_type = rng.choice(final_tkg_list)
        tkg_q_list = question_dict[current_type]
        for _ in range(20):
            target_question = rng.choice(tkg_q_list)
            questions = target_question.get("questions") or {}
            first_question = questions[next(iter(questions))] if questions else ""
            if len(date_splitted) >= 3 and (
                f"n {date_splitted[2]}" in first_question
                or f"{date_splitted[0]} {date_splitted[2]}" in first_question
            ):
                continue
            target_question_list.append(dict(target_question))
            try:
                target_dates_list.append(oracle_tkg[epi][sc_num][current_type][tkg_q_list.index(target_question)])
            except Exception:
                try:
                    target_dates_list.append(oracle_tkg[epi][sc_num][current_type][first_question])
                except Exception:
                    target_dates_list.append([])
        return "hard_q", current_type, target_question_list, target_dates_list, ""

    question_dict = sess["easy_q"]
    final_fan_list = []
    unans_num = 0
    ans_list = []
    for fan in list(question_dict):
        if len(list(question_dict[fan])) > 0:
            final_fan_list.append(fan)
            if "unans" in fan:
                unans_num += 1
            else:
                ans_list.append(fan)
    if not final_fan_list:
        return "", "", [], [], "no_easy_questions"
    if len(ans_list) > 0:
        rng.shuffle(ans_list)
        while True:
            should_stop = 0
            for ans_ele in ans_list:
                if unans_num / len(final_fan_list) < 0.27:
                    should_stop = 1
                    break
                final_fan_list.append(ans_ele)
            if should_stop == 1:
                break

    current_type = rng.choice(final_fan_list)
    fan_q_list = list(question_dict[current_type])
    for _ in range(20):
        fan_q_target_num = rng.choice(fan_q_list)
        target_question = dict(question_dict[current_type][fan_q_target_num])
        target_question["_official_question_id"] = fan_q_target_num
        target_question_list.append(target_question)
        if current_type in ["ans_w_time", "dont_know_unans_time"]:
            try:
                target_dates_list.append(oracle_fan[epi][sc_num][current_type][fan_q_target_num])
            except Exception:
                target_dates_list.append([])
        else:
            target_dates_list.append([])
    return "easy_q", current_type, target_question_list, target_dates_list, ""


def choose_ask_context(
    *,
    rng: random.Random,
    show: str,
    turns: list[ParsedTurn],
) -> tuple[str, str, int | None, str]:
    chatbot = CHATBOT[show]
    chatbot_utters = [turn.raw.strip() for turn in turns if f"{chatbot}:" in turn.raw]
    characters = [turn.speaker for turn in turns if turn.speaker]
    random_chatbot_utter = ""
    try:
        if len(chatbot_utters) > 1:
            chatbot_utters = chatbot_utters[1:]
        random_chatbot_utter = rng.choice(chatbot_utters)
        bot_indices = [idx for idx, turn in enumerate(turns) if random_chatbot_utter in turn.raw]
        range_indices = range(max(0, bot_indices[0] - 3), min(len(turns), bot_indices[0] + 3))
        close_chars = []
        for idx in range_indices:
            if turns[idx].speaker:
                close_chars.append(turns[idx].speaker)
        # The upstream script uses list(set(close_chars)), which is affected by
        # Python hash randomization.  Sort after de-duplication so a saved seed
        # produces identical artifacts across processes.
        characters = sorted(set(close_chars))
        close_chars = sorted(set(close_chars))
        for char_ in close_chars:
            if chatbot.lower() in char_.lower() or char_.lower() == "all":
                try:
                    characters.remove(char_)
                except ValueError:
                    pass
    except Exception:
        pass

    char_ask = rng.choice(characters) if characters else ""
    ask_turn_index = None
    if random_chatbot_utter and char_ask:
        for idx, turn in enumerate(turns):
            if random_chatbot_utter.lower() in turn.raw.lower():
                ask_turn_index = idx
                break
    if not char_ask:
        return "", random_chatbot_utter, ask_turn_index, "no_question_asker"
    if ask_turn_index is None:
        return char_ask, random_chatbot_utter, ask_turn_index, "no_chatbot_trigger"
    return char_ask, random_chatbot_utter, ask_turn_index, ""


def iter_sessions(show: str, data: dict[str, Any], max_sessions: int | None = None) -> Iterable[tuple[int, str, Any, dict[str, Any]]]:
    ordinal = 0
    for episode_key, epi_data in data.items():
        for scene_num, sess in epi_data.items():
            ordinal += 1
            if max_sessions is not None and ordinal > max_sessions:
                return
            yield ordinal, episode_key, scene_num, sess


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sessions_path = args.output_dir / "sessions.jsonl"
    selected_path = args.output_dir / "selected_qa.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    source_files: dict[str, dict[str, Any]] = {}
    for path in sorted(args.source_dir.glob("*.pickle")) + sorted(args.zip_dir.glob("dialsim_v*.zip")):
        source_files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    session_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "shows": {},
        "skip_reasons": Counter(),
        "selected_by_show": Counter(),
        "selected_by_source": Counter(),
        "selected_by_type": Counter(),
        "selected_by_answerability": Counter(),
        "selected_by_hop": Counter(),
    }

    for show_index, show in enumerate(args.shows):
        rng = random.Random(args.seed + show_index)
        dialsim_path = args.source_dir / f"{show}_dialsim.pickle"
        oracle_tkg_path = args.source_dir / f"{show}_oracle_tkg.pickle"
        oracle_fan_path = args.source_dir / f"{show}_oracle_fan.pickle"
        data = load_pickle(dialsim_path)
        oracle_tkg = load_pickle(oracle_tkg_path)
        oracle_fan = load_pickle(oracle_fan_path)

        show_stats = {
            "episodes": len(data),
            "sessions": 0,
            "turns": 0,
            "selected_qa": 0,
            "candidate_easy_counts": Counter(),
            "candidate_hard_counts": Counter(),
            "skip_reasons": Counter(),
            "seed": args.seed + show_index,
        }
        date_to_session_ids: dict[str, list[str]] = defaultdict(list)
        show_turn_ids_seen: list[str] = []
        selected_cap_reached = False
        before_date = ""
        conversation_num_for_date = 1

        # First pass for sessions and date index.
        session_cache: list[tuple[int, str, Any, dict[str, Any], list[ParsedTurn], str]] = []
        for ordinal, episode_key, scene_num, sess in iter_sessions(show, data, args.max_sessions_per_show):
            season, episode = season_episode(show, episode_key)
            episode_id = safe_episode_id(show, episode_key)
            session_id = f"{episode_id}_scene{int(scene_num):03d}" if str(scene_num).isdigit() else f"{episode_id}_scene{scene_num}"
            turns = parse_turns(show, episode_key, scene_num, sess.get("script", ""))
            date = sess.get("date", "")
            if before_date != date:
                conversation_num_for_date = 1
                before_date = date
            current_conversation_num = conversation_num_for_date
            conversation_num_for_date += 1
            date_to_session_ids[str(date)].append(session_id)
            show_turn_ids_seen.extend(turn.turn_id for turn in turns)
            show_stats["sessions"] += 1
            show_stats["turns"] += len(turns)

            for cat, val in (sess.get("easy_q") or {}).items():
                show_stats["candidate_easy_counts"][cat] += len(val)
            for cat, val in (sess.get("hard_q") or {}).items():
                show_stats["candidate_hard_counts"][cat] += len(val)

            session_cache.append((ordinal, episode_key, scene_num, sess, turns, session_id))
            session_rows.append(
                {
                    "show": show,
                    "show_name": SHOW_NAMES[show],
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "season": season,
                    "episode": episode,
                    "scene_id": session_id,
                    "scene_number": scene_num,
                    "session_ordinal": ordinal,
                    "date": date,
                    "conversation_number_for_date": current_conversation_num,
                    "speaker_names": sorted({turn.speaker for turn in turns if turn.speaker}),
                    "turn_count": len(turns),
                    "turns": [
                        {
                            "turn_id": turn.turn_id,
                            "turn_index": turn.turn_index,
                            "speaker": turn.speaker,
                            "text": turn.text,
                            "raw": turn.raw,
                        }
                        for turn in turns
                    ],
                }
            )

        # Second pass for deterministic selected QA.
        prior_turn_ids: list[str] = []
        for ordinal, episode_key, scene_num, sess, turns, session_id in session_cache:
            if args.max_selected_per_show is not None and show_stats["selected_qa"] >= args.max_selected_per_show:
                selected_cap_reached = True
                break

            source, current_type, candidates, target_dates, skip_reason = choose_question_candidates(
                rng=rng,
                epi=episode_key,
                sc_num=scene_num,
                sess=sess,
                oracle_tkg=oracle_tkg,
                oracle_fan=oracle_fan,
                tkg_ratio=args.tkg_ratio,
            )
            char_ask, random_chatbot_utter, ask_turn_index, ask_skip_reason = choose_ask_context(
                rng=rng,
                show=show,
                turns=turns,
            )
            skip_reason = skip_reason or ask_skip_reason

            if not candidates:
                show_stats["skip_reasons"][skip_reason or "no_candidates"] += 1
                stats["skip_reasons"][skip_reason or "no_candidates"] += 1
                prior_turn_ids.extend(turn.turn_id for turn in turns)
                continue

            real_tar_id = -1
            real_question = ""
            true_answer = ""
            options: list[str] = []
            evidence_target_dates: list[str] = []
            for tar_id, target_question in enumerate(candidates):
                real_question = question_text_for_character(target_question, char_ask)
                if not real_question:
                    continue
                true_answer = str(target_question.get("answer", ""))
                options = [str(option) for option in target_question.get("options", [])]
                if len(options) < 5:
                    continue
                evidence_target_dates = normalize_target_dates(target_dates[tar_id] if tar_id < len(target_dates) else [])
                real_tar_id = tar_id
                break

            gold_option = option_letter(true_answer, options)
            if ask_turn_index is None or not char_ask or real_tar_id < 0 or not gold_option:
                reason = skip_reason or "unusable_selected_question"
                show_stats["skip_reasons"][reason] += 1
                stats["skip_reasons"][reason] += 1
                prior_turn_ids.extend(turn.turn_id for turn in turns)
                continue

            answerable = not is_unanswerable(current_type, true_answer, options)
            hop = hop_label(source, current_type)
            evidence_session_ids = sorted(
                {
                    session
                    for date in evidence_target_dates
                    for session in date_to_session_ids.get(date, [])
                }
            )
            available_turn_count = len(prior_turn_ids) + ask_turn_index + 1
            selected_id = f"{show}:{ordinal:04d}:{source}:{current_type}:{show_stats['selected_qa']:04d}"
            selected_row = {
                "id": selected_id,
                "protocol": "dialsim_v1.1_seeded_official_selection",
                "seed": args.seed + show_index,
                "tkg_ratio": args.tkg_ratio,
                "show": show,
                "show_name": SHOW_NAMES[show],
                "episode_key": episode_key,
                "episode_id": safe_episode_id(show, episode_key),
                "season": season_episode(show, episode_key)[0],
                "episode": season_episode(show, episode_key)[1],
                "scene_id": session_id,
                "scene_number": scene_num,
                "session_ordinal": ordinal,
                "date": sess.get("date", ""),
                "chatbot": CHATBOT[show],
                "question_asker": char_ask,
                "question_source": source,
                "question_type": current_type,
                "hop_type": hop,
                "answerable": answerable,
                "question": real_question,
                "question_variants": candidates[real_tar_id].get("questions", {}),
                "options": options,
                "answer": true_answer,
                "gold_option": gold_option,
                "question_prompt": build_question_prompt(char_ask, real_question, options),
                "selected_candidate_index": real_tar_id,
                "official_question_id": candidates[real_tar_id].get("_official_question_id"),
                "target_dates_raw": target_dates[real_tar_id] if real_tar_id < len(target_dates) else [],
                "evidence_dates": evidence_target_dates,
                "evidence_scene_ids": evidence_session_ids,
                "evidence_turn_ids": [],
                "ask_turn_index": ask_turn_index + 1,
                "ask_turn_id": turns[ask_turn_index].turn_id,
                "ask_trigger_utterance": random_chatbot_utter,
                "available_turn_count": available_turn_count,
                "history_scope": {
                    "include_prior_sessions": True,
                    "end_scene_id": session_id,
                    "end_session_ordinal": ordinal,
                    "end_turn_index": ask_turn_index + 1,
                    "end_turn_id": turns[ask_turn_index].turn_id,
                },
                "current_scene_prefix_turn_ids": [turn.turn_id for turn in turns[: ask_turn_index + 1]],
                "speaker_names": sorted({turn.speaker for turn in turns if turn.speaker}),
            }
            if args.include_available_turn_ids:
                available_turn_ids = prior_turn_ids + [turn.turn_id for turn in turns[: ask_turn_index + 1]]
                selected_row["available_turn_ids"] = available_turn_ids
            selected_rows.append(selected_row)
            show_stats["selected_qa"] += 1
            stats["selected_by_show"][show] += 1
            stats["selected_by_source"][source] += 1
            stats["selected_by_type"][current_type] += 1
            stats["selected_by_answerability"]["answerable" if answerable else "unanswerable"] += 1
            stats["selected_by_hop"][hop] += 1
            prior_turn_ids.extend(turn.turn_id for turn in turns)

        if selected_cap_reached:
            show_stats["skip_reasons"]["selected_cap_reached"] += 1
        stats["shows"][show] = {
            **{
                k: v
                for k, v in show_stats.items()
                if k not in {"candidate_easy_counts", "candidate_hard_counts", "skip_reasons"}
            },
            "candidate_easy_counts": dict(show_stats["candidate_easy_counts"]),
            "candidate_hard_counts": dict(show_stats["candidate_hard_counts"]),
            "skip_reasons": dict(show_stats["skip_reasons"]),
        }

    write_jsonl(sessions_path, session_rows)
    write_jsonl(selected_path, selected_rows)

    manifest = {
        "schema_version": "longdialqa_dialsim_normalized_v1",
        "created_by": Path(__file__).name,
        "source": {
            "dataset": "DialSim/LongDialQA",
            "version": "v1.1",
            "official_repo": "https://github.com/jiho283/DialSim",
            "local_repo_commit": git_commit(Path("baseline/DialSim")),
            "files": source_files,
        },
        "protocol": {
            "name": "dialsim_v1.1_seeded_official_selection",
            "seed": args.seed,
            "per_show_seed": {show: args.seed + idx for idx, show in enumerate(args.shows)},
            "tkg_ratio": args.tkg_ratio,
            "shows": args.shows,
            "max_selected_per_show": args.max_selected_per_show,
            "max_sessions_per_show": args.max_sessions_per_show,
            "include_available_turn_ids": args.include_available_turn_ids,
            "question_format": "multi_choice_structured",
            "ask_history": "all previous turns plus current scene prefix through ask_turn_index",
            "selection_notes": [
                "Question source/type sampling mirrors baseline/DialSim/simulator.py, including unused preliminary random choices.",
                "The upstream close-speaker set is sorted here before random.choice so the saved seed is stable across PYTHONHASHSEED values.",
                "Python random is explicitly seeded per show; the upstream script did not set a seed.",
                "Evidence is session/date-level when oracle target dates are available; turn-level evidence is not present in the official pickle schema.",
            ],
        },
        "artifacts": {
            "sessions_jsonl": {
                "path": str(sessions_path),
                "rows": len(session_rows),
                "sha256": sha256_text_file(sessions_path),
            },
            "selected_qa_jsonl": {
                "path": str(selected_path),
                "rows": len(selected_rows),
                "sha256": sha256_text_file(selected_path),
            },
        },
        "stats": {
            "shows": stats["shows"],
            "selected_by_show": dict(stats["selected_by_show"]),
            "selected_by_source": dict(stats["selected_by_source"]),
            "selected_by_type": dict(stats["selected_by_type"]),
            "selected_by_answerability": dict(stats["selected_by_answerability"]),
            "selected_by_hop": dict(stats["selected_by_hop"]),
            "skip_reasons": dict(stats["skip_reasons"]),
            "total_sessions": len(session_rows),
            "total_selected_qa": len(selected_rows),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "selected_qa": len(selected_rows), "sessions": len(session_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
