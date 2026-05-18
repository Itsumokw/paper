"""Shared data structures for non-invasive HiGMemPlus experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RawTurn:
    turn_id: str
    text: str
    speaker: str = ""
    timestamp: str = ""
    dataset: str = ""
    show: str = ""
    show_name: str = ""
    episode_id: str = ""
    session_or_scene_id: str = ""
    chronological_order: int = 0
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_context_line(self) -> str:
        parts = []
        if self.timestamp:
            parts.append(f"Date: {self.timestamp}")
        if self.session_or_scene_id:
            parts.append(f"Scene: {self.session_or_scene_id}")
        if self.turn_id:
            parts.append(f"Turn: {self.turn_id}")
        prefix = "; ".join(parts)
        speaker = self.speaker or "unknown"
        if prefix:
            return f"[{prefix}] Speaker {speaker} says: {self.text}"
        return f"Speaker {speaker} says: {self.text}"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    episode_id: str
    dataset: str
    session_or_scene_id: str
    chronological_order: int
    participants: list[str] = field(default_factory=list)
    turns: list[RawTurn] = field(default_factory=list)
    summary: str = ""
    linked_events: list[str] = field(default_factory=list)
    linked_components: list[str] = field(default_factory=list)

    def asdict(self, include_turns: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_turns:
            data["turns"] = [turn.turn_id for turn in self.turns]
        return data


@dataclass
class EvidenceComponent:
    component_id: str
    event_id: str
    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    time_expr: str = ""
    event_time: str = ""
    mentioned_at: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    source_session_or_episode_id: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TypedEdge:
    edge_id: str
    source_component_id: str
    target_component_id: str
    edge_type: str
    relation_text: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SufficiencyResult:
    sufficient: bool
    status: str = "SUPPORTED"
    missing_slots: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteDecision:
    question_type: str
    evidence_risk: str
    needed_layers: list[str]
    retrieval_budget: dict[str, int]
    route_name: str
    reasons: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    context: str
    evidence_records: list[dict[str, Any]]
    component_trace: list[dict[str, Any]] = field(default_factory=list)
    graph_trace: list[dict[str, Any]] = field(default_factory=list)
    repair_trace: list[dict[str, Any]] = field(default_factory=list)
    episode_trace: list[dict[str, Any]] = field(default_factory=list)
    route_trace: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
