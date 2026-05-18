"""Evidence-component and episode store for HiGMemPlus."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .schemas import Episode, EvidenceComponent, RawTurn, TypedEdge
from .text import (
    extract_entities,
    extract_time_expr,
    infer_predicate,
    normalize_relative_time,
    rank_by_lexical,
    relation_type,
    split_atomic_sentences,
    stable_id,
    top_terms,
)


class EvidenceStore:
    """A deterministic evidence layer that sits beside upstream HiGMem."""

    def __init__(self, dataset: str = "") -> None:
        self.dataset = dataset
        self.turns: dict[str, RawTurn] = {}
        self.episodes: dict[str, Episode] = {}
        self.components: dict[str, EvidenceComponent] = {}
        self.edges: list[TypedEdge] = []
        self.components_by_turn: dict[str, list[str]] = defaultdict(list)
        self.components_by_episode: dict[str, list[str]] = defaultdict(list)
        self.episode_by_turn: dict[str, str] = {}

    def add_turn(self, turn: RawTurn) -> None:
        self.turns[turn.turn_id] = turn
        episode_id = turn.session_or_scene_id or turn.episode_id or stable_id("episode", turn.turn_id)
        if episode_id not in self.episodes:
            self.episodes[episode_id] = Episode(
                episode_id=episode_id,
                dataset=turn.dataset or self.dataset,
                session_or_scene_id=turn.session_or_scene_id,
                chronological_order=turn.chronological_order,
                participants=[],
                turns=[],
            )
        episode = self.episodes[episode_id]
        episode.turns.append(turn)
        if turn.speaker and turn.speaker not in episode.participants:
            episode.participants.append(turn.speaker)
        if not episode.summary:
            episode.summary = self._summarize_episode(episode)
        else:
            episode.summary = self._summarize_episode(episode)
        self.episode_by_turn[turn.turn_id] = episode_id

        for sentence_index, sentence in enumerate(split_atomic_sentences(turn.text)):
            component = self._component_from_sentence(turn, sentence, sentence_index, episode_id)
            self.components[component.component_id] = component
            self.components_by_turn[turn.turn_id].append(component.component_id)
            self.components_by_episode[episode_id].append(component.component_id)
            episode.linked_components.append(component.component_id)
            if component.event_id not in episode.linked_events:
                episode.linked_events.append(component.event_id)
        self._rebuild_local_edges(episode_id)

    def add_turns(self, turns: list[RawTurn]) -> None:
        for turn in turns:
            self.add_turn(turn)

    def _component_from_sentence(self, turn: RawTurn, sentence: str, sentence_index: int, episode_id: str) -> EvidenceComponent:
        entities = extract_entities(sentence)
        subject = entities[0] if entities else (turn.speaker or "")
        obj = entities[1] if len(entities) > 1 else ""
        time_expr = extract_time_expr(sentence)
        event_time = normalize_relative_time(time_expr, turn.timestamp)
        event_id = stable_id("event", episode_id, top_terms(sentence, 5))
        component_id = stable_id("component", turn.turn_id, sentence_index, sentence)
        return EvidenceComponent(
            component_id=component_id,
            event_id=event_id,
            text=sentence,
            subject=subject,
            predicate=infer_predicate(sentence),
            object=obj,
            time_expr=time_expr,
            event_time=event_time,
            mentioned_at=turn.timestamp,
            source_turn_ids=[turn.turn_id],
            source_session_or_episode_id=episode_id,
            confidence=1.0,
            metadata={
                "speaker": turn.speaker,
                "show": turn.show,
                "show_name": turn.show_name,
                "episode_id": turn.episode_id,
                "turn_index": turn.turn_index,
                "relation_type": relation_type(sentence),
                "raw_text_sha256": hashlib.sha256(turn.text.encode("utf-8")).hexdigest(),
            },
        )

    def _summarize_episode(self, episode: Episode) -> str:
        if not episode.turns:
            return ""
        lines = [turn.to_context_line() for turn in episode.turns[:3]]
        if len(episode.turns) > 6:
            lines.append("...")
        if len(episode.turns) > 3:
            lines.extend(turn.to_context_line() for turn in episode.turns[-3:])
        return "\n".join(lines)

    def _rebuild_local_edges(self, episode_id: str) -> None:
        component_ids = self.components_by_episode.get(episode_id, [])
        if len(component_ids) < 2:
            return
        existing = {(edge.source_component_id, edge.target_component_id, edge.edge_type) for edge in self.edges}
        recent_ids = component_ids[-8:]
        for left_id in recent_ids:
            left = self.components[left_id]
            for right_id in recent_ids:
                if left_id == right_id:
                    continue
                right = self.components[right_id]
                edge_type = self._edge_type(left, right)
                if not edge_type:
                    continue
                key = (left_id, right_id, edge_type)
                if key in existing:
                    continue
                self.edges.append(
                    TypedEdge(
                        edge_id=stable_id("edge", left_id, right_id, edge_type),
                        source_component_id=left_id,
                        target_component_id=right_id,
                        edge_type=edge_type,
                        relation_text=f"{left.subject or left.predicate} -> {right.subject or right.predicate}",
                        source_turn_ids=sorted(set(left.source_turn_ids + right.source_turn_ids)),
                        confidence=1.0,
                    )
                )
                existing.add(key)
            for right_id in [
                cid
                for cid in recent_ids
                if cid != left_id and set(self.components[cid].source_turn_ids) & set(left.source_turn_ids)
            ][:2]:
                key = (left_id, right_id, "source_of")
                if key in existing:
                    continue
                right = self.components[right_id]
                self.edges.append(
                    TypedEdge(
                        edge_id=stable_id("edge", left_id, right_id, "source_of"),
                        source_component_id=left_id,
                        target_component_id=right_id,
                        edge_type="source_of",
                        relation_text="same raw turn source",
                        source_turn_ids=sorted(set(left.source_turn_ids + right.source_turn_ids)),
                        confidence=1.0,
                    )
                )
                existing.add(key)

    def _edge_type(self, left: EvidenceComponent, right: EvidenceComponent) -> str:
        if left.source_session_or_episode_id != right.source_session_or_episode_id:
            return ""
        if self._is_conflict(left, right):
            return "conflict_with"
        if left.event_time or right.event_time or left.time_expr or right.time_expr:
            return "temporal_near"
        left_entities = {left.subject.lower(), left.object.lower()} - {""}
        right_entities = {right.subject.lower(), right.object.lower()} - {""}
        if left_entities & right_entities:
            return "same_entity"
        relation = str(left.metadata.get("relation_type") or "")
        if relation in {"causal", "preference/attribute", "social", "event-update"}:
            return relation
        return "same_session"

    def _is_conflict(self, left: EvidenceComponent, right: EvidenceComponent) -> bool:
        if not left.subject or not right.subject:
            return False
        if left.subject.lower() != right.subject.lower():
            return False
        if not left.object or not right.object or left.object.lower() == right.object.lower():
            return False
        return bool(left.predicate and left.predicate.lower() == right.predicate.lower())

    def search_components(self, query: str, limit: int = 10, relation_filter: set[str] | None = None) -> list[tuple[float, EvidenceComponent]]:
        candidates = []
        for component in self.components.values():
            if relation_filter and str(component.metadata.get("relation_type")) not in relation_filter:
                continue
            text = " ".join(
                [
                    component.text,
                    component.subject,
                    component.predicate,
                    component.object,
                    component.time_expr,
                    component.event_time,
                ]
            )
            candidates.append((text, component))
        return [(score, component) for score, _text, component in rank_by_lexical(query, candidates, limit)]

    def search_edges(self, query: str, limit: int = 10, edge_types: set[str] | None = None) -> list[tuple[float, TypedEdge]]:
        candidates = []
        for edge in self.edges:
            if edge_types and edge.edge_type not in edge_types:
                continue
            left = self.components.get(edge.source_component_id)
            right = self.components.get(edge.target_component_id)
            if not left or not right:
                continue
            text = f"{edge.edge_type} {edge.relation_text} {left.text} {right.text}"
            candidates.append((text, edge))
        return [(score, edge) for score, _text, edge in rank_by_lexical(query, candidates, limit)]

    def search_episodes(self, query: str, limit: int = 4) -> list[tuple[float, Episode]]:
        candidates = [(f"{episode.summary} {' '.join(episode.participants)}", episode) for episode in self.episodes.values()]
        return [(score, episode) for score, _text, episode in rank_by_lexical(query, candidates, limit)]

    def turns_for_component_ids(self, component_ids: list[str]) -> list[RawTurn]:
        seen = set()
        turns = []
        for component_id in component_ids:
            component = self.components.get(component_id)
            if not component:
                continue
            for turn_id in component.source_turn_ids:
                if turn_id in seen:
                    continue
                turn = self.turns.get(turn_id)
                if turn:
                    turns.append(turn)
                    seen.add(turn_id)
        turns.sort(key=lambda turn: (turn.chronological_order, turn.turn_index, turn.turn_id))
        return turns

    def episode_for_turn_id(self, turn_id: str) -> Episode | None:
        episode_id = self.episode_by_turn.get(turn_id)
        if not episode_id:
            return None
        return self.episodes.get(episode_id)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "turn_count": len(self.turns),
            "episode_count": len(self.episodes),
            "component_count": len(self.components),
            "edge_count": len(self.edges),
        }
