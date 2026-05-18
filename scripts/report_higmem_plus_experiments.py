#!/usr/bin/env python3
"""Write EC-HiGMem experiment summary and bad-case reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CATEGORY_LABELS = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}


RUNS = {
    "5pct_baseline": Path("reproductions/higmem_plus/locomo10_5pct_fast/baseline_higmem"),
    "5pct_adaptive_iter1": Path("reproductions/higmem_plus/locomo10_5pct_fast/adaptive_routing"),
    "5pct_adaptive_iter2": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter2/adaptive_routing"),
    "5pct_selective_iter3": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter3/selective_adaptive_routing"),
    "5pct_cautious_iter4": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter4/cautious_adaptive_routing"),
    "5pct_graphwalk_iter5": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter5_graphwalk/graph_walk_plan_routing"),
    "5pct_slot_tree_iter6": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter6_slot_tree/answer_slot_tree_routing"),
    "5pct_evidence_frame_iter7": Path("reproductions/higmem_plus/locomo10_5pct_fast_iter7_evidence_frame/evidence_frame_routing"),
    "10pct_baseline": Path("reproductions/higmem_plus/locomo10_10pct_fast_confirm/baseline_higmem"),
    "10pct_selective_confirm": Path("reproductions/higmem_plus/locomo10_10pct_fast_confirm/selective_adaptive_routing"),
    "10pct_cautious_confirm": Path("reproductions/higmem_plus/locomo10_10pct_fast_cautious/cautious_adaptive_routing"),
    "10pct_evidence_frame_confirm": Path("reproductions/higmem_plus/locomo10_10pct_fast_evidence_frame/evidence_frame_routing"),
}


LONGDIALQA_RUNS = {
    "FullContext": Path("reproductions/higmem_plus/baselines_longdialqa_full/full_context/metrics.json"),
    "A-Mem": Path("reproductions/higmem_plus/baselines_longdialqa_full/a_mem/metrics.json"),
    "HiGMem": Path("reproductions/higmem_plus/baselines_longdialqa_full_sharded/higmem/metrics.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reproductions/higmem_plus"))
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def row_id_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["row_id"]): row for row in read_jsonl(run_dir / "raw_predictions.jsonl")}


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    overall = metrics["overall"]
    return (
        f"| {label} | {overall['count']} | {fmt(overall['f1'])} | {fmt(overall['bleu1'])} | "
        f"{fmt(overall['judge_accuracy_proxy'])} | {fmt(overall['evidence_support_rate'])} | "
        f"{fmt(overall['drill_down_rate'])} | {fmt(overall['avg_context_tokens'])} | {fmt(overall['avg_latency_seconds'])} |"
    )


def category_rows(
    baseline: dict[str, Any],
    improved: dict[str, Any],
    baseline_label: str,
    improved_label: str,
) -> list[str]:
    rows = [
        f"| Category | Count | {baseline_label} F1 | {improved_label} F1 | Delta F1 | "
        f"{baseline_label} Judge | {improved_label} Judge | Delta Judge | "
        f"{baseline_label} Support | {improved_label} Support | Delta Support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in sorted(baseline["by_category"]):
        left = baseline["by_category"][category]
        right = improved["by_category"].get(category, {})
        delta_f1 = float(right.get("f1", 0.0)) - float(left.get("f1", 0.0))
        delta_judge = float(right.get("judge_accuracy_proxy", 0.0)) - float(left.get("judge_accuracy_proxy", 0.0))
        delta_support = float(right.get("evidence_support_rate", 0.0)) - float(left.get("evidence_support_rate", 0.0))
        rows.append(
            f"| {category} | {left.get('count')} | {fmt(left.get('f1'))} | {fmt(right.get('f1'))} | {fmt(delta_f1)} | "
            f"{fmt(left.get('judge_accuracy_proxy'))} | {fmt(right.get('judge_accuracy_proxy'))} | {fmt(delta_judge)} | "
            f"{fmt(left.get('evidence_support_rate'))} | {fmt(right.get('evidence_support_rate'))} | {fmt(delta_support)} |"
        )
    return rows


def short(text: Any, limit: int = 180) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def case_table(title: str, rows: list[tuple[dict[str, Any], dict[str, Any]]], limit: int = 12) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Row | Category | Question | Reference | Baseline Prediction | EC Prediction | Baseline Support | EC Support |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for base, ec in rows[:limit]:
        lines.append(
            f"| {base.get('row_id')} | {base.get('category_name')} | {short(base.get('question'))} | "
            f"{short(base.get('reference'), 90)} | {short(base.get('prediction'), 90)} | "
            f"{short(ec.get('prediction'), 90)} | {base.get('evidence_support')} | {ec.get('evidence_support')} |"
        )
    if not rows:
        lines.append("| none | | | | | | | |")
    return lines


def compare_cases(base_dir: Path, ec_dir: Path) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    base_rows = row_id_map(base_dir)
    ec_rows = row_id_map(ec_dir)
    common_ids = sorted(set(base_rows) & set(ec_rows), key=lambda rid: tuple(int(part) for part in rid.split(":")))
    wins = []
    regressions = []
    support_gain = []
    support_loss = []
    for row_id in common_ids:
        base = base_rows[row_id]
        ec = ec_rows[row_id]
        if ec.get("answer_correct_proxy") and not base.get("answer_correct_proxy"):
            wins.append((base, ec))
        if base.get("answer_correct_proxy") and not ec.get("answer_correct_proxy"):
            regressions.append((base, ec))
        if ec.get("evidence_support") and not base.get("evidence_support"):
            support_gain.append((base, ec))
        if base.get("evidence_support") and not ec.get("evidence_support"):
            support_loss.append((base, ec))
    return {
        "wins": wins,
        "regressions": regressions,
        "support_gain": support_gain,
        "support_loss": support_loss,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {name: read_json(path / "metrics.json") for name, path in RUNS.items()}
    subset_5 = read_json(Path("datasets/subsets/locomo10_5pct_seed20260517_manifest.json"))
    subset_10 = read_json(Path("datasets/subsets/locomo10_10pct_seed20260517_manifest.json"))
    subset_20_path = Path("datasets/subsets/locomo10_20pct_seed20260517_manifest.json")
    subset_20 = read_json(subset_20_path) if subset_20_path.exists() else None

    report_lines = [
        "# EC-HiGMem Reproduction and Iteration Report",
        "",
        "Generated from local artifacts on 2026-05-18.",
        "",
        "## Baseline Reproduction: LongDialQA/DialSim",
        "",
        "| Method | Count | Accuracy | Strict Accuracy | Mean Token F1 | Evidence Recall | Mean Context Tokens | Artifact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, path in LONGDIALQA_RUNS.items():
        data = read_json(path)
        overall = data["overall"]
        report_lines.append(
            f"| {label} | {overall['count']} | {fmt(overall['accuracy'])} | {fmt(overall['strict_accuracy'])} | "
            f"{fmt(overall['mean_token_f1'])} | {fmt(overall.get('evidence_recall_any'))} | "
            f"{fmt(overall['mean_context_tokens_approx'])} | `{path}` |"
        )

    report_lines.extend(
        [
            "",
            "## LoCoMo Subsets",
            "",
            "| Subset | Seed | QA Count | Category Counts | SHA256 |",
            "| --- | ---: | ---: | --- | --- |",
            f"| 5% | {subset_5['seed']} | {subset_5['subset_qa_count']} | `{subset_5['per_category_subset_counts']}` | `{subset_5['subset_sha256']}` |",
            f"| 10% | {subset_10['seed']} | {subset_10['subset_qa_count']} | `{subset_10['per_category_subset_counts']}` | `{subset_10['subset_sha256']}` |",
            (
                f"| 20% | {subset_20['seed']} | {subset_20['subset_qa_count']} | `{subset_20['per_category_subset_counts']}` | `{subset_20['subset_sha256']}` |"
                if subset_20
                else "| 20% | not built | | | |"
            ),
            "",
            "The 20% subset is built for the original experiment spec; the executed LoCoMo runs use 5%/10% because the user explicitly requested faster iteration with 5% or 10% subsets.",
            "",
            "## LoCoMo 5% Iteration Results",
            "",
            "| Run | Count | F1 | BLEU1 | Judge Proxy | Evidence Support | Drill-down | Avg Context Tokens | Avg Latency s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in [
        "5pct_baseline",
        "5pct_adaptive_iter1",
        "5pct_adaptive_iter2",
        "5pct_selective_iter3",
        "5pct_cautious_iter4",
        "5pct_graphwalk_iter5",
        "5pct_slot_tree_iter6",
        "5pct_evidence_frame_iter7",
    ]:
        report_lines.append(metric_row(key, metrics[key]))
    report_lines.extend(
        [
            "",
            "### 5% Baseline vs Cautious Iteration by Category",
            "",
            *category_rows(metrics["5pct_baseline"], metrics["5pct_cautious_iter4"], "Baseline", "Cautious"),
            "",
            "### 5% Cautious vs Graph-Walk by Category",
            "",
            *category_rows(metrics["5pct_cautious_iter4"], metrics["5pct_graphwalk_iter5"], "Cautious", "GraphWalk"),
            "",
            "### 5% Cautious vs Answer-Slot Tree by Category",
            "",
            *category_rows(metrics["5pct_cautious_iter4"], metrics["5pct_slot_tree_iter6"], "Cautious", "SlotTree"),
            "",
            "### 5% Cautious vs Final Evidence Frame by Category",
            "",
            *category_rows(metrics["5pct_cautious_iter4"], metrics["5pct_evidence_frame_iter7"], "Cautious", "EvidenceFrame"),
            "",
            "## LoCoMo 10% Confirmation",
            "",
            "| Run | Count | F1 | BLEU1 | Judge Proxy | Evidence Support | Drill-down | Avg Context Tokens | Avg Latency s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            metric_row("10pct_baseline", metrics["10pct_baseline"]),
            metric_row("10pct_selective_confirm", metrics["10pct_selective_confirm"]),
            metric_row("10pct_cautious_confirm", metrics["10pct_cautious_confirm"]),
            metric_row("10pct_evidence_frame_confirm", metrics["10pct_evidence_frame_confirm"]),
            "",
            "### 10% Baseline vs Cautious by Category",
            "",
            *category_rows(metrics["10pct_baseline"], metrics["10pct_cautious_confirm"], "Baseline", "Cautious"),
            "",
            "### 10% Baseline vs Final Evidence Frame by Category",
            "",
            *category_rows(metrics["10pct_baseline"], metrics["10pct_evidence_frame_confirm"], "Baseline", "EvidenceFrame"),
            "",
            "## Interpretation",
            "",
            "- Final selected method is the universal evidence-frame route: it converts reasoning-heavy questions into answer-slot, intent, root-evidence, terminal-candidate, and minimal-bridge constraints without sample-specific entities or answer terms.",
            "- On 5%, final evidence-frame routing improves over HiGMem baseline by F1 +0.0830, BLEU1 +0.0730, Judge proxy +0.0707, and evidence support +0.0707.",
            "- On 10%, final evidence-frame routing confirms the gain: F1 +0.0351, BLEU1 +0.0356, Judge proxy +0.0050, and evidence support +0.0704 over the 10% HiGMem baseline.",
            "- A narrower answer-slot tree variant reached higher 5% F1, but used overly specific lexical expansions during development; it is kept as an exploratory ablation, not the final paper method.",
            "- Iteration 5 graph-walk evidence planning isolated a real temporal benefit: temporal F1 improved from cautious 0.2430 to 0.3328 and temporal Judge proxy from 0.1875 to 0.2500, but multi-hop regressed because broader graph paths introduced distractor facts.",
            "- Multi-hop remains harder than temporal: final evidence-frame improves 5% multi-hop F1 slightly versus cautious, but the main stable gain is evidence support and temporal reasoning.",
            "",
            "## Key Artifacts",
            "",
        ]
    )
    for key, path in RUNS.items():
        report_lines.append(f"- {key}: `{path}`")
    report_lines.extend(
        [
            "- EC implementation: `baseline/HiGMemPlus/`",
            "- Fast LoCoMo runner: `scripts/run_locomo_higmem_plus_fast.py`",
            "- Shard merger: `scripts/merge_locomo_higmem_plus_shards.py`",
            "- LongDialQA shard merger: `scripts/merge_longdialqa_baseline_shards.py`",
            "",
        ]
    )

    badcase_lines = [
        "# EC-HiGMem Bad Case Analysis",
        "",
        "Bad cases compare row-aligned HiGMem baseline and selective EC-HiGMem outputs.",
        "",
    ]
    for subset, base_key, ec_key in [
        ("5%", "5pct_baseline", "5pct_cautious_iter4"),
        ("5% graph-walk", "5pct_cautious_iter4", "5pct_graphwalk_iter5"),
        ("5% answer-slot tree", "5pct_cautious_iter4", "5pct_slot_tree_iter6"),
        ("5% final evidence-frame", "5pct_cautious_iter4", "5pct_evidence_frame_iter7"),
        ("10% selective", "10pct_baseline", "10pct_selective_confirm"),
        ("10% cautious", "10pct_baseline", "10pct_cautious_confirm"),
        ("10% final evidence-frame", "10pct_baseline", "10pct_evidence_frame_confirm"),
    ]:
        cases = compare_cases(RUNS[base_key], RUNS[ec_key])
        badcase_lines.extend(
            [
                f"## {subset} Summary",
                "",
                f"- EC wins over baseline: {len(cases['wins'])}",
                f"- EC regressions versus baseline: {len(cases['regressions'])}",
                f"- Evidence support gains: {len(cases['support_gain'])}",
                f"- Evidence support losses: {len(cases['support_loss'])}",
                "",
                *case_table(f"{subset} EC Wins", cases["wins"]),
                "",
                *case_table(f"{subset} EC Regressions", cases["regressions"]),
                "",
                *case_table(f"{subset} Evidence Support Gains", cases["support_gain"]),
                "",
            ]
        )

    report_path = args.output_dir / "ec_higmem_experiment_report_20260518.md"
    badcase_path = args.output_dir / "ec_higmem_bad_cases_20260518.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    badcase_path.write_text("\n".join(badcase_lines), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "bad_cases": str(badcase_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
