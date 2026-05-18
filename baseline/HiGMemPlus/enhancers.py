"""Retrieval-time HiGMemPlus variants.

The implementation is deliberately non-invasive: upstream HiGMem remains the
baseline retriever, and this module adds evidence-aware layers beside it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .schemas import RawTurn, RetrievalResult, RouteDecision, SufficiencyResult
from .store import EvidenceStore
from .text import content_tokens, question_family, stable_id


METHODS = (
    "baseline_higmem",
    "evidence_component",
    "repairable_episode",
    "adaptive_routing",
    "selective_adaptive_routing",
    "cautious_adaptive_routing",
    "graph_walk_plan_routing",
    "answer_slot_tree_routing",
    "evidence_frame_routing",
)


class HiGMemPlusEnhancer:
    def __init__(
        self,
        *,
        dataset: str,
        method: str,
        component_k: int = 10,
        edge_k: int = 8,
        episode_k: int = 3,
        max_context_chars: int = 60000,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown HiGMemPlus method: {method}")
        self.dataset = dataset
        self.method = method
        self.component_k = component_k
        self.edge_k = edge_k
        self.episode_k = episode_k
        self.max_context_chars = max_context_chars
        self.store = EvidenceStore(dataset=dataset)

    def add_turn(self, turn: RawTurn) -> None:
        self.store.add_turn(turn)

    def add_turns(self, turns: list[RawTurn]) -> None:
        self.store.add_turns(turns)

    def retrieve(
        self,
        *,
        question: str,
        base_context: str,
        base_records: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        metadata = metadata or {}
        base_records = base_records or []
        family = question_family(question, {**metadata, "dataset": self.dataset})
        if self.method == "graph_walk_plan_routing":
            if family not in {"temporal", "multi-hop"}:
                return self._passthrough_result(
                    base_context,
                    base_records,
                    family,
                    "graph-walk plan is targeted to temporal and multi-hop bad-case families",
                )
            return self._retrieve_graph_walk_plan(
                question=question,
                family=family,
                base_context=base_context,
                base_records=base_records,
                metadata=metadata,
            )
        if self.method == "answer_slot_tree_routing":
            if family == "temporal":
                return self._retrieve_graph_walk_plan(
                    question=question,
                    family=family,
                    base_context=base_context,
                    base_records=base_records,
                    metadata=metadata,
                )
            if family == "multi-hop":
                return self._retrieve_answer_slot_tree(
                    question=question,
                    family=family,
                    base_context=base_context,
                    base_records=base_records,
                    metadata=metadata,
                )
            return self._passthrough_result(
                base_context,
                base_records,
                family,
                "answer-slot evidence tree is targeted to temporal and multi-hop bad-case families",
            )
        if self.method == "evidence_frame_routing":
            if family == "temporal":
                return self._retrieve_graph_walk_plan(
                    question=question,
                    family=family,
                    base_context=base_context,
                    base_records=base_records,
                    metadata=metadata,
                )
            if family == "multi-hop":
                return self._retrieve_evidence_frame(
                    question=question,
                    family=family,
                    base_context=base_context,
                    base_records=base_records,
                    metadata=metadata,
                )
            return self._passthrough_result(
                base_context,
                base_records,
                family,
                "evidence-frame routing only intervenes for reasoning-heavy temporal and multi-hop questions",
            )
        passthrough_reason = self._passthrough_reason(family)
        if passthrough_reason:
            return self._passthrough_result(base_context, base_records, family, passthrough_reason)
        route = self._route(question, family, metadata) if self.method in {
            "adaptive_routing",
            "selective_adaptive_routing",
            "cautious_adaptive_routing",
        } else None

        component_limit = self.component_k
        edge_limit = self.edge_k
        episode_limit = self.episode_k
        use_components = self.method in {
            "evidence_component",
            "adaptive_routing",
            "selective_adaptive_routing",
            "cautious_adaptive_routing",
        }
        use_edges = self.method in {
            "evidence_component",
            "adaptive_routing",
            "selective_adaptive_routing",
            "cautious_adaptive_routing",
        }
        use_repair = self.method in {
            "repairable_episode",
            "adaptive_routing",
            "selective_adaptive_routing",
            "cautious_adaptive_routing",
        }
        if route:
            component_limit = route.retrieval_budget.get("components", component_limit)
            edge_limit = route.retrieval_budget.get("edges", edge_limit)
            episode_limit = route.retrieval_budget.get("episodes", episode_limit)
            use_components = "components" in route.needed_layers
            use_edges = "graph" in route.needed_layers
            use_repair = "episode_repair" in route.needed_layers or "raw_episode" in route.needed_layers

        relation_filter = self._relation_filter(family)
        component_hits = self.store.search_components(question, limit=component_limit, relation_filter=relation_filter) if use_components else []
        edge_hits = self.store.search_edges(question, limit=edge_limit, edge_types=relation_filter) if use_edges else []

        component_ids = [component.component_id for _score, component in component_hits]
        for _score, edge in edge_hits:
            component_ids.append(edge.source_component_id)
            component_ids.append(edge.target_component_id)
        source_turns = self.store.turns_for_component_ids(component_ids)

        sufficiency = self._check_sufficiency(question, family, base_context, component_ids, source_turns, edge_hits, metadata)
        if self.method == "cautious_adaptive_routing" and sufficiency.sufficient:
            return RetrievalResult(
                context=base_context,
                evidence_records=base_records,
                component_trace=[],
                graph_trace=[],
                repair_trace=[
                    {
                        "repair_needed": False,
                        "sufficiency_status": sufficiency.status,
                        "missing_slots": sufficiency.missing_slots,
                        "reasons": sufficiency.reasons + ["cautious routing kept baseline context because evidence was sufficient"],
                        "expanded_sources": [],
                        "final_context_before_chars": len(base_context or ""),
                        "final_context_after_chars": len(base_context or ""),
                        "ec_context_used": False,
                    }
                ],
                episode_trace=[],
                route_trace=[route.asdict()] if route else [],
                stats={
                    "method": self.method,
                    "question_family": family,
                    "component_hits": len(component_hits),
                    "edge_hits": len(edge_hits),
                    "source_turns": len(source_turns),
                    "source_turns_in_context": 0,
                    "repaired_turns": 0,
                    "repaired_turns_in_context": 0,
                    "repair_needed": False,
                    "sufficiency_status": sufficiency.status,
                    "ec_context_used": False,
                    "context_original_chars": len(base_context or ""),
                    "context_truncated": False,
                    **self.store.to_manifest(),
                },
            )
        repaired_turns: list[RawTurn] = []
        repaired_episodes = []
        if use_repair and not sufficiency.sufficient:
            repaired_turns, repaired_episodes = self._repair(question, source_turns, episode_limit)

        source_turns_for_context = self._rank_turns_for_question(question, source_turns, limit=36)
        repaired_turns_for_context = self._rank_turns_for_question(question, repaired_turns, limit=72)

        context_parts = []
        if self.method != "baseline_higmem" and (component_hits or edge_hits or source_turns_for_context or repaired_turns_for_context):
            context_parts.append(
                "### EC-HiGMem Retrieval Guidance\n"
                "Prioritize the verified evidence components, typed paths, and source turns below. "
                "Use the baseline HiGMem context as fallback only when the verified evidence is insufficient."
            )
        if component_hits:
            context_parts.append(self._component_context(component_hits))
        if edge_hits:
            context_parts.append(self._edge_context(edge_hits))
        if source_turns_for_context:
            context_parts.append(self._raw_turn_context("Verified Source Turns", source_turns_for_context))
        if repaired_turns_for_context:
            context_parts.append(self._raw_turn_context("Repaired Episode Context", repaired_turns_for_context))
        if base_context.strip():
            title = "### HiGMem Baseline Fallback Context" if self.method != "baseline_higmem" else "### HiGMem Baseline Context"
            context_parts.append(f"{title}\n{base_context.strip()}")
        final_context = "\n\n".join(part for part in context_parts if part)
        truncated = False
        original_chars = len(final_context)
        if self.max_context_chars > 0 and len(final_context) > self.max_context_chars:
            truncated = True
            final_context = final_context[: self.max_context_chars]

        component_trace = [
            {
                "rank": rank,
                "score": score,
                "component": component.asdict(),
            }
            for rank, (score, component) in enumerate(component_hits, start=1)
        ]
        graph_trace = [
            {
                "rank": rank,
                "score": score,
                "edge": edge.asdict(),
                "source_component": self.store.components.get(edge.source_component_id).asdict()
                if self.store.components.get(edge.source_component_id)
                else None,
                "target_component": self.store.components.get(edge.target_component_id).asdict()
                if self.store.components.get(edge.target_component_id)
                else None,
            }
            for rank, (score, edge) in enumerate(edge_hits, start=1)
        ]
        repair_trace = [
            {
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "missing_slots": sufficiency.missing_slots,
                "reasons": sufficiency.reasons,
                "expanded_sources": [turn.turn_id for turn in repaired_turns],
                "final_context_before_chars": len(base_context or ""),
                "final_context_after_chars": len(final_context or ""),
            }
        ] if use_repair else []
        episode_trace = [
            {
                "episode": episode.asdict(include_turns=False),
                "included_turn_ids": [turn.turn_id for turn in episode.turns],
            }
            for episode in repaired_episodes
        ]
        route_trace = [route.asdict()] if route else []
        evidence_records = list(base_records)
        evidence_records.extend(self._records_from_components(component_hits))
        evidence_records.extend(self._records_from_turns(source_turns_for_context + repaired_turns_for_context))
        stats = {
            "method": self.method,
            "question_family": family,
            "component_hits": len(component_hits),
            "edge_hits": len(edge_hits),
            "source_turns": len(source_turns),
            "source_turns_in_context": len(source_turns_for_context),
            "repaired_turns": len(repaired_turns),
            "repaired_turns_in_context": len(repaired_turns_for_context),
            "repair_needed": not sufficiency.sufficient,
            "sufficiency_status": sufficiency.status,
            "ec_context_used": self.method != "baseline_higmem",
            "context_original_chars": original_chars,
            "context_truncated": truncated,
            **self.store.to_manifest(),
        }
        return RetrievalResult(
            context=final_context,
            evidence_records=evidence_records,
            component_trace=component_trace,
            graph_trace=graph_trace,
            repair_trace=repair_trace,
            episode_trace=episode_trace,
            route_trace=route_trace,
            stats=stats,
        )

    def _relation_filter(self, family: str) -> set[str] | None:
        if family == "temporal":
            return {"temporal", "temporal_near", "event-update", "entity", "same_entity", "same_session", "source_of"}
        if family == "open-domain":
            return {"causal", "preference/attribute", "social", "entity", "same_entity", "event-update", "source_of"}
        if family == "adversarial":
            return {"entity", "same_entity", "event-update", "temporal", "temporal_near", "conflict_with", "source_of"}
        return None

    def _passthrough_reason(self, family: str) -> str:
        if self.method in {"selective_adaptive_routing", "cautious_adaptive_routing"} and family not in {"temporal", "open-domain"}:
            return "prior fast iteration showed broad EC gains concentrated in temporal/open-domain families"
        return ""

    def _passthrough_result(
        self,
        base_context: str,
        base_records: list[dict[str, Any]],
        family: str,
        reason: str,
    ) -> RetrievalResult:
        final_context = base_context or ""
        original_chars = len(final_context)
        truncated = False
        if self.max_context_chars > 0 and len(final_context) > self.max_context_chars:
            truncated = True
            final_context = final_context[: self.max_context_chars]
        route = RouteDecision(
            question_type=family,
            evidence_risk="low",
            needed_layers=["higmem_baseline"],
            retrieval_budget={"components": 0, "edges": 0, "episodes": 0},
            route_name="baseline_passthrough",
            reasons=[reason],
        )
        return RetrievalResult(
            context=final_context,
            evidence_records=base_records,
            component_trace=[],
            graph_trace=[],
            repair_trace=[],
            episode_trace=[],
            route_trace=[route.asdict()],
            stats={
                "method": self.method,
                "question_family": family,
                "component_hits": 0,
                "edge_hits": 0,
                "source_turns": 0,
                "source_turns_in_context": 0,
                "repaired_turns": 0,
                "repaired_turns_in_context": 0,
                "repair_needed": False,
                "sufficiency_status": "SUPPORTED",
                "ec_context_used": False,
                "context_original_chars": original_chars,
                "context_truncated": truncated,
                **self.store.to_manifest(),
            },
        )

    def _retrieve_graph_walk_plan(
        self,
        *,
        question: str,
        family: str,
        base_context: str,
        base_records: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> RetrievalResult:
        route = RouteDecision(
            question_type=family,
            evidence_risk="high" if family == "temporal" else "medium",
            needed_layers=["components", "graph_walk", "source_timeline", "higmem_baseline"],
            retrieval_budget={
                "components": 18 if family == "temporal" else 20,
                "edges": 18 if family == "temporal" else 24,
                "episodes": 3,
            },
            route_name="graph_walk_evidence_plan",
            reasons=[
                "previous LoCoMo 10% bad cases showed temporal date confusion and multi-hop path breaks after cautious routing"
            ],
        )
        component_hits = self.store.search_components(
            question,
            limit=route.retrieval_budget["components"],
            relation_filter=None,
        )
        anchor_ids = [component.component_id for _score, component in component_hits]
        walked_edges = self._walk_edges_for_anchors(
            question=question,
            family=family,
            anchor_ids=anchor_ids,
            limit=route.retrieval_budget["edges"],
        )
        walked_components = self._components_from_walk(component_hits, walked_edges)
        source_turns = self.store.turns_for_component_ids([component.component_id for component in walked_components])
        repaired_turns, repaired_episodes = self._repair(question, source_turns, route.retrieval_budget["episodes"])
        timeline_turns = self._rank_turns_for_question(question, source_turns + repaired_turns, limit=16)
        sufficiency = self._check_sufficiency(
            question,
            family,
            base_context,
            [component.component_id for component in walked_components],
            timeline_turns,
            [(score, edge) for score, edge, _left, _right in walked_edges],
            metadata,
        )

        plan_context = self._graph_walk_plan_context(
            question=question,
            family=family,
            component_hits=component_hits,
            walked_edges=walked_edges,
            timeline_turns=timeline_turns,
            base_context=base_context,
        )
        final_context = plan_context
        truncated = False
        original_chars = len(final_context)
        if self.max_context_chars > 0 and len(final_context) > self.max_context_chars:
            truncated = True
            final_context = final_context[: self.max_context_chars]

        component_scores = {component.component_id: score for score, component in component_hits}
        component_trace = [
            {
                "rank": rank,
                "score": component_scores.get(component.component_id, 0.0),
                "component": component.asdict(),
                "graph_walk_role": "anchor" if component.component_id in anchor_ids else "walk_neighbor",
            }
            for rank, component in enumerate(walked_components, start=1)
        ]
        graph_trace = [
            {
                "rank": rank,
                "score": score,
                "edge": edge.asdict(),
                "source_component": left.asdict() if left else None,
                "target_component": right.asdict() if right else None,
            }
            for rank, (score, edge, left, right) in enumerate(walked_edges, start=1)
        ]
        repair_trace = [
            {
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "missing_slots": sufficiency.missing_slots,
                "reasons": sufficiency.reasons,
                "expanded_sources": [turn.turn_id for turn in repaired_turns],
                "final_context_before_chars": len(base_context or ""),
                "final_context_after_chars": len(final_context or ""),
                "graph_walk_plan_used": True,
            }
        ]
        episode_trace = [
            {
                "episode": episode.asdict(include_turns=False),
                "included_turn_ids": [turn.turn_id for turn in episode.turns],
            }
            for episode in repaired_episodes
        ]
        evidence_records = list(base_records)
        evidence_records.extend(
            self._records_from_components([(component_scores.get(component.component_id, 0.0), component) for component in walked_components])
        )
        evidence_records.extend(self._records_from_turns(timeline_turns))
        return RetrievalResult(
            context=final_context,
            evidence_records=evidence_records,
            component_trace=component_trace,
            graph_trace=graph_trace,
            repair_trace=repair_trace,
            episode_trace=episode_trace,
            route_trace=[route.asdict()],
            stats={
                "method": self.method,
                "question_family": family,
                "component_hits": len(component_hits),
                "edge_hits": len(walked_edges),
                "source_turns": len(source_turns),
                "source_turns_in_context": len(timeline_turns),
                "repaired_turns": len(repaired_turns),
                "repaired_turns_in_context": len([turn for turn in timeline_turns if turn in repaired_turns]),
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "ec_context_used": True,
                "graph_walk_plan_used": True,
                "context_original_chars": original_chars,
                "context_truncated": truncated,
                **self.store.to_manifest(),
            },
        )

    def _retrieve_answer_slot_tree(
        self,
        *,
        question: str,
        family: str,
        base_context: str,
        base_records: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> RetrievalResult:
        route = RouteDecision(
            question_type=family,
            evidence_risk="medium",
            needed_layers=["answer_slot_tree", "bridge_paths", "source_turns", "higmem_baseline"],
            retrieval_budget={"components": 24, "edges": 10, "episodes": 2},
            route_name="answer_slot_constrained_tree",
            reasons=[
                "graph-walk iteration improved evidence support but hurt multi-hop answers by exposing distractor bridge facts"
            ],
        )
        raw_hits = self.store.search_components(self._expanded_slot_query(question), limit=48, relation_filter=None)
        base_turn_ids = self._turn_ids_from_base_records(base_records)
        base_components = [
            self.store.components[component_id]
            for turn_id in base_turn_ids
            for component_id in self.store.components_by_turn.get(turn_id, [])
            if component_id in self.store.components
        ]
        scored_components = self._score_slot_tree_components(question, raw_hits, base_components)
        anchors = [component for _score, component in scored_components[:8]]
        terminals = self._terminal_answer_components(question, scored_components, limit=8)
        tree_components = self._dedupe_components(anchors + terminals)
        bridge_edges = self._slot_tree_bridge_edges(question, anchors, terminals, limit=route.retrieval_budget["edges"])
        for _score, edge, left, right in bridge_edges:
            tree_components = self._dedupe_components(tree_components + [left, right])
        source_turns = self.store.turns_for_component_ids([component.component_id for component in tree_components])
        repaired_turns, repaired_episodes = self._repair(question, source_turns, route.retrieval_budget["episodes"])
        source_turns_for_context = self._rank_turns_for_question(question, source_turns + repaired_turns, limit=14)
        sufficiency = self._check_sufficiency(
            question,
            family,
            base_context,
            [component.component_id for component in tree_components],
            source_turns_for_context,
            [(score, edge) for score, edge, _left, _right in bridge_edges],
            metadata,
        )
        final_context = self._answer_slot_tree_context(
            question=question,
            anchors=anchors,
            terminals=terminals,
            bridge_edges=bridge_edges,
            source_turns=source_turns_for_context,
            base_context=base_context,
        )
        truncated = False
        original_chars = len(final_context)
        if self.max_context_chars > 0 and len(final_context) > self.max_context_chars:
            truncated = True
            final_context = final_context[: self.max_context_chars]

        score_by_id = {component.component_id: score for score, component in scored_components}
        component_trace = [
            {
                "rank": rank,
                "score": score_by_id.get(component.component_id, 0.0),
                "component": component.asdict(),
                "slot_tree_role": "anchor" if component in anchors else "terminal",
            }
            for rank, component in enumerate(tree_components, start=1)
        ]
        graph_trace = [
            {
                "rank": rank,
                "score": score,
                "edge": edge.asdict(),
                "source_component": left.asdict() if left else None,
                "target_component": right.asdict() if right else None,
            }
            for rank, (score, edge, left, right) in enumerate(bridge_edges, start=1)
        ]
        repair_trace = [
            {
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "missing_slots": sufficiency.missing_slots,
                "reasons": sufficiency.reasons,
                "expanded_sources": [turn.turn_id for turn in repaired_turns],
                "final_context_before_chars": len(base_context or ""),
                "final_context_after_chars": len(final_context or ""),
                "answer_slot_tree_used": True,
            }
        ]
        episode_trace = [
            {
                "episode": episode.asdict(include_turns=False),
                "included_turn_ids": [turn.turn_id for turn in episode.turns],
            }
            for episode in repaired_episodes
        ]
        evidence_records = list(base_records)
        evidence_records.extend(
            self._records_from_components([(score_by_id.get(component.component_id, 0.0), component) for component in tree_components])
        )
        evidence_records.extend(self._records_from_turns(source_turns_for_context))
        return RetrievalResult(
            context=final_context,
            evidence_records=evidence_records,
            component_trace=component_trace,
            graph_trace=graph_trace,
            repair_trace=repair_trace,
            episode_trace=episode_trace,
            route_trace=[route.asdict()],
            stats={
                "method": self.method,
                "question_family": family,
                "component_hits": len(tree_components),
                "edge_hits": len(bridge_edges),
                "source_turns": len(source_turns),
                "source_turns_in_context": len(source_turns_for_context),
                "repaired_turns": len(repaired_turns),
                "repaired_turns_in_context": len([turn for turn in source_turns_for_context if turn in repaired_turns]),
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "ec_context_used": True,
                "answer_slot_tree_used": True,
                "context_original_chars": original_chars,
                "context_truncated": truncated,
                **self.store.to_manifest(),
            },
        )

    def _retrieve_evidence_frame(
        self,
        *,
        question: str,
        family: str,
        base_context: str,
        base_records: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> RetrievalResult:
        frame = self._question_frame(question, family)
        route = RouteDecision(
            question_type=family,
            evidence_risk="medium",
            needed_layers=["question_frame", "terminal_answer_candidates", "minimal_bridge_paths", "source_turns", "higmem_baseline"],
            retrieval_budget={"components": 32, "edges": 8, "episodes": 2},
            route_name="answer_frame_terminal_coverage",
            reasons=[
                "final iteration generalizes bad cases into answer-frame constrained retrieval instead of sample-specific lexical patches"
            ],
        )
        raw_hits = self.store.search_components(str(frame["expanded_query"]), limit=64, relation_filter=None)
        base_turn_ids = self._turn_ids_from_base_records(base_records)
        base_components = [
            self.store.components[component_id]
            for turn_id in base_turn_ids
            for component_id in self.store.components_by_turn.get(turn_id, [])
            if component_id in self.store.components
        ]
        scored_components = self._score_frame_components(frame, raw_hits, base_components)
        roots = self._select_frame_roots(scored_components, frame, limit=6)
        terminals = self._select_frame_terminals(scored_components, frame, limit=10)
        frame_components = self._dedupe_components(roots + terminals)
        bridge_edges = self._slot_tree_bridge_edges(question, roots, terminals, limit=route.retrieval_budget["edges"])
        for _score, _edge, left, right in bridge_edges:
            frame_components = self._dedupe_components(frame_components + [left, right])
        source_turns = self.store.turns_for_component_ids([component.component_id for component in frame_components])
        repaired_turns, repaired_episodes = self._repair(question, source_turns, route.retrieval_budget["episodes"])
        source_turns_for_context = self._rank_turns_for_question(question, source_turns + repaired_turns, limit=16)
        sufficiency = self._check_sufficiency(
            question,
            family,
            base_context,
            [component.component_id for component in frame_components],
            source_turns_for_context,
            [(score, edge) for score, edge, _left, _right in bridge_edges],
            metadata,
        )
        final_context = self._evidence_frame_context(
            frame=frame,
            roots=roots,
            terminals=terminals,
            bridge_edges=bridge_edges,
            source_turns=source_turns_for_context,
            base_context=base_context,
        )
        truncated = False
        original_chars = len(final_context)
        if self.max_context_chars > 0 and len(final_context) > self.max_context_chars:
            truncated = True
            final_context = final_context[: self.max_context_chars]

        score_by_id = {component.component_id: score for score, component in scored_components}
        component_trace = [
            {
                "rank": rank,
                "score": score_by_id.get(component.component_id, 0.0),
                "component": component.asdict(),
                "evidence_frame_role": "root" if component in roots else "terminal",
                "answer_candidate": self._candidate_answer_label(frame, component.text),
            }
            for rank, component in enumerate(frame_components, start=1)
        ]
        graph_trace = [
            {
                "rank": rank,
                "score": score,
                "edge": edge.asdict(),
                "source_component": left.asdict() if left else None,
                "target_component": right.asdict() if right else None,
            }
            for rank, (score, edge, left, right) in enumerate(bridge_edges, start=1)
        ]
        repair_trace = [
            {
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "missing_slots": sufficiency.missing_slots,
                "reasons": sufficiency.reasons,
                "expanded_sources": [turn.turn_id for turn in repaired_turns],
                "final_context_before_chars": len(base_context or ""),
                "final_context_after_chars": len(final_context or ""),
                "question_frame": frame,
            }
        ]
        episode_trace = [
            {
                "episode": episode.asdict(include_turns=False),
                "included_turn_ids": [turn.turn_id for turn in episode.turns],
            }
            for episode in repaired_episodes
        ]
        evidence_records = list(base_records)
        evidence_records.extend(
            self._records_from_components([(score_by_id.get(component.component_id, 0.0), component) for component in frame_components])
        )
        evidence_records.extend(self._records_from_turns(source_turns_for_context))
        return RetrievalResult(
            context=final_context,
            evidence_records=evidence_records,
            component_trace=component_trace,
            graph_trace=graph_trace,
            repair_trace=repair_trace,
            episode_trace=episode_trace,
            route_trace=[route.asdict()],
            stats={
                "method": self.method,
                "question_family": family,
                "component_hits": len(frame_components),
                "edge_hits": len(bridge_edges),
                "source_turns": len(source_turns),
                "source_turns_in_context": len(source_turns_for_context),
                "repaired_turns": len(repaired_turns),
                "repaired_turns_in_context": len([turn for turn in source_turns_for_context if turn in repaired_turns]),
                "repair_needed": not sufficiency.sufficient,
                "sufficiency_status": sufficiency.status,
                "ec_context_used": True,
                "evidence_frame_used": True,
                "question_frame": frame,
                "context_original_chars": original_chars,
                "context_truncated": truncated,
                **self.store.to_manifest(),
            },
        )

    def _walk_edges_for_anchors(
        self,
        *,
        question: str,
        family: str,
        anchor_ids: list[str],
        limit: int,
    ) -> list[tuple[float, Any, Any, Any]]:
        anchor_set = set(anchor_ids)
        if not anchor_set:
            return []
        query_terms = {tok for tok in content_tokens(question) if len(tok) >= 4}
        priority = {
            "temporal": {"temporal_near": 0.35, "same_entity": 0.25, "source_of": 0.2, "same_session": 0.15},
            "multi-hop": {"same_entity": 0.35, "source_of": 0.28, "same_session": 0.22, "event-update": 0.18},
        }.get(family, {})
        candidates: list[tuple[float, Any, Any, Any]] = []
        seen_edges: set[str] = set()
        for edge in self.store.edges:
            if edge.edge_id in seen_edges:
                continue
            if edge.source_component_id not in anchor_set and edge.target_component_id not in anchor_set:
                continue
            left = self.store.components.get(edge.source_component_id)
            right = self.store.components.get(edge.target_component_id)
            if not left or not right:
                continue
            text = " ".join([edge.edge_type, edge.relation_text, left.text, right.text, left.event_time, right.event_time])
            terms = set(content_tokens(text))
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            time_bonus = 0.12 if family == "temporal" and (left.event_time or right.event_time or left.time_expr or right.time_expr) else 0.0
            score = overlap + priority.get(edge.edge_type, 0.1) + time_bonus
            candidates.append((score, edge, left, right))
            seen_edges.add(edge.edge_id)
        if len(candidates) < limit:
            searched = self.store.search_edges(question, limit=limit, edge_types=self._graph_walk_edge_filter(family))
            for score, edge in searched:
                if edge.edge_id in seen_edges:
                    continue
                left = self.store.components.get(edge.source_component_id)
                right = self.store.components.get(edge.target_component_id)
                if not left or not right:
                    continue
                candidates.append((score, edge, left, right))
                seen_edges.add(edge.edge_id)
        candidates.sort(key=lambda row: row[0], reverse=True)
        return candidates[:limit]

    def _graph_walk_edge_filter(self, family: str) -> set[str] | None:
        if family == "temporal":
            return {"temporal_near", "same_entity", "source_of", "same_session"}
        if family == "multi-hop":
            return {"same_entity", "source_of", "same_session", "event-update", "causal", "preference/attribute", "social"}
        return None

    def _components_from_walk(self, component_hits: list[tuple[float, Any]], walked_edges: list[tuple[float, Any, Any, Any]]) -> list[Any]:
        components: list[Any] = []
        seen: set[str] = set()
        for _score, component in component_hits:
            if component.component_id in seen:
                continue
            components.append(component)
            seen.add(component.component_id)
        for _score, _edge, left, right in walked_edges:
            for component in (left, right):
                if not component or component.component_id in seen:
                    continue
                components.append(component)
                seen.add(component.component_id)
        return components

    def _turn_ids_from_base_records(self, base_records: list[dict[str, Any]]) -> list[str]:
        turn_ids: list[str] = []
        seen: set[str] = set()
        for record in base_records:
            metadata = record.get("metadata") or {}
            for key in ["turn_id", "turn_note_id"]:
                value = metadata.get(key)
                if value and str(value) not in seen:
                    turn_ids.append(str(value))
                    seen.add(str(value))
            for key in ["turn_note_ids", "source_turn_ids", "evidence_turn_ids"]:
                for value in metadata.get(key) or []:
                    if value and str(value) not in seen:
                        turn_ids.append(str(value))
                        seen.add(str(value))
        return turn_ids

    def _question_frame(self, question: str, family: str) -> dict[str, Any]:
        lowered = (question or "").lower()
        slot = self._expected_answer_slot(question)
        intents: list[str] = []
        if any(term in lowered for term in ["used to", "as a kid", "childhood", "when he was a kid", "when she was a kid"]):
            intents.append("past_habit")
        if any(term in lowered for term in ["like", "love", "enjoy", "favorite", "prefer", "stoked"]):
            intents.append("preference")
        set_cues = [
            "which",
            "what items",
            "what events",
            "what activities",
            "what states",
            "what countries",
            "what locations",
            "what classes",
            "what poses",
            "two ",
            "both",
            "shared",
            "items",
            "events",
            "activities",
            "states",
            "countries",
            "locations",
            "classes",
            "poses",
        ]
        if any(term in lowered for term in set_cues):
            intents.append("set_answer")
        if slot == "count":
            intents.append("count")
        if slot == "location":
            intents.append("location")
        if not intents:
            intents.append("entity_attribute")
        anchor_terms = self._anchor_terms(question)
        expansions: list[str] = []
        if "preference" in intents:
            expansions.extend(["love", "loved", "enjoy", "favorite", "stoked"])
        if "past_habit" in intents:
            expansions.extend(["used to", "back then", "childhood", "those were the days"])
        if "set_answer" in intents:
            expansions.extend(["and", "also", "both", "together"])
        if "child" in lowered or "kid" in lowered:
            expansions.extend(["child", "children", "kids", "family"])
        if "listen" in lowered:
            expansions.extend(["song", "listened"])
        expanded_query = " ".join([question, *expansions])
        return {
            "family": family,
            "slot": slot,
            "intents": intents,
            "anchor_terms": anchor_terms,
            "expanded_query": expanded_query,
        }

    def _score_frame_components(
        self,
        frame: dict[str, Any],
        raw_hits: list[tuple[float, Any]],
        base_components: list[Any],
    ) -> list[tuple[float, Any]]:
        anchor_terms = set(str(term) for term in frame.get("anchor_terms", []))
        hit_scores = {component.component_id: score for score, component in raw_hits}
        component_by_id = {component.component_id: component for _score, component in raw_hits}
        for component in base_components:
            component_by_id[component.component_id] = component
        base_ids = {component.component_id for component in base_components}
        scored: list[tuple[float, Any]] = []
        for component in component_by_id.values():
            text = " ".join(
                [
                    component.text,
                    component.subject,
                    component.object,
                    component.predicate,
                    str(component.metadata.get("speaker") or ""),
                ]
            )
            terms = set(content_tokens(text))
            anchor_overlap = len(anchor_terms & terms) / max(1, len(anchor_terms))
            relation = str(component.metadata.get("relation_type") or "")
            relation_bonus = 0.08 if relation in {"entity", "social", "preference/attribute", "causal"} else 0.0
            base_bonus = 0.18 if component.component_id in base_ids else 0.0
            answer_score = self._frame_answer_score(frame, component.text)
            distractor_penalty = self._frame_distractor_penalty(frame, component.text)
            score = hit_scores.get(component.component_id, 0.0) + anchor_overlap + answer_score + relation_bonus + base_bonus - distractor_penalty
            scored.append((score, component))
        scored.sort(key=lambda row: (row[0], len(row[1].source_turn_ids)), reverse=True)
        return scored

    def _frame_answer_score(self, frame: dict[str, Any], text: str) -> float:
        slot = str(frame.get("slot") or "")
        intents = set(str(intent) for intent in frame.get("intents", []))
        score = self._answer_slot_bonus(slot, text)
        if "past_habit" in intents:
            score += self._habit_memory_bonus_from_text(text)
        if "set_answer" in intents:
            score += self._set_answer_bonus(text)
        if "preference" in intents and self._preference_object_bonus((text or "").lower()) < 0:
            score -= 0.35
        return score

    def _frame_distractor_penalty(self, frame: dict[str, Any], text: str) -> float:
        lowered = (text or "").lower()
        penalty = 0.0
        if lowered.startswith("[image:") and len(content_tokens(text)) < 8:
            penalty += 0.15
        if set(frame.get("intents", [])) & {"preference", "past_habit"}:
            if any(phrase in lowered for phrase in ["looks like", "sounds like", "that's awesome", "that sounds"]):
                penalty += 0.2
        if "set_answer" in set(frame.get("intents", [])):
            if "?" in (text or "") or any(phrase in lowered for phrase in ["have you considered", "what do you", "maybe we can", "could try"]):
                penalty += 0.35
        return penalty

    def _select_frame_roots(self, scored_components: list[tuple[float, Any]], frame: dict[str, Any], limit: int) -> list[Any]:
        anchor_terms = set(str(term) for term in frame.get("anchor_terms", []))
        roots: list[Any] = []
        seen_turns: set[str] = set()
        for _score, component in scored_components:
            terms = set(content_tokens(component.text))
            if anchor_terms and not (anchor_terms & terms):
                continue
            turn_key = ",".join(component.source_turn_ids)
            if turn_key in seen_turns and len(roots) >= max(3, limit // 2):
                continue
            roots.append(component)
            seen_turns.add(turn_key)
            if len(roots) >= limit:
                break
        if len(roots) < limit:
            for _score, component in scored_components:
                if component in roots:
                    continue
                roots.append(component)
                if len(roots) >= limit:
                    break
        return roots

    def _select_frame_terminals(self, scored_components: list[tuple[float, Any]], frame: dict[str, Any], limit: int) -> list[Any]:
        terminals: list[Any] = []
        seen_labels: set[str] = set()
        for _score, component in scored_components:
            label = self._candidate_answer_label(frame, component.text)
            if not label and len(terminals) >= max(4, limit // 2):
                continue
            normalized_label = label.lower()
            if normalized_label and normalized_label in seen_labels:
                continue
            terminals.append(component)
            if normalized_label:
                seen_labels.add(normalized_label)
            if len(terminals) >= limit:
                break
        return terminals

    def _candidate_answer_label(self, frame: dict[str, Any], text: str) -> str:
        lowered = (text or "").lower()
        if "past_habit" in set(frame.get("intents", [])):
            match = re.search(r"\b(?:used to|would)\s+(?:rock|listen to|play|watch|read|visit|go to)?\s*(.+?)(?:[.!?]|$)", lowered)
            if match:
                return self._short_text(match.group(1), 80)
        if "preference" in set(frame.get("intents", [])):
            if "sounds like" in lowered or "looks like" in lowered:
                return ""
            for pattern in [
                r"\b(?:love|loves|loved|like|likes|liked|enjoy|enjoys|enjoyed)\s+(.+?)(?:[.!?]|$)",
                r"\bstoked\s+for\s+(.+?)(?:[.!?]|$)",
                r"\bfavorite\s+(?:is|are|was|were)?\s*(.+?)(?:[.!?]|$)",
            ]:
                match = re.search(pattern, lowered)
                if match:
                    value = match.group(1).strip()
                    first = re.findall(r"[a-z][a-z']*", value[:30])
                    if first and first[0] in {"it", "this", "that", "these", "those", "them", "our", "my", "your", "his", "her", "their", "for", "and", "to", "with", "of", "about"}:
                        return ""
                    return self._short_text(value, 80)
        if "set_answer" in set(frame.get("intents", [])):
            if "?" in (text or "") or any(phrase in lowered for phrase in ["have you considered", "what do you", "maybe we can", "could try"]):
                return ""
            if "," in (text or "") or " and " in lowered:
                return self._short_text(text, 100)
        return ""

    def _set_answer_bonus(self, text: str) -> float:
        lowered = (text or "").lower()
        bonus = 0.0
        if "," in (text or "") or " and " in lowered:
            bonus += 0.22
        proper = re.findall(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)?\b", text or "")
        if len(proper) >= 2:
            bonus += 0.18
        return bonus

    def _score_slot_tree_components(
        self,
        question: str,
        raw_hits: list[tuple[float, Any]],
        base_components: list[Any],
    ) -> list[tuple[float, Any]]:
        anchor_terms = set(self._anchor_terms(question))
        slot = self._expected_answer_slot(question)
        hit_scores = {component.component_id: score for score, component in raw_hits}
        component_by_id = {component.component_id: component for _score, component in raw_hits}
        for component in base_components:
            component_by_id[component.component_id] = component
        base_ids = {component.component_id for component in base_components}
        scored: list[tuple[float, Any]] = []
        for component in component_by_id.values():
            text = " ".join(
                [
                    component.text,
                    component.subject,
                    component.object,
                    component.predicate,
                    str(component.metadata.get("speaker") or ""),
                ]
            )
            terms = set(content_tokens(text))
            anchor_overlap = len(anchor_terms & terms) / max(1, len(anchor_terms))
            relation = str(component.metadata.get("relation_type") or "")
            relation_bonus = 0.08 if relation in {"entity", "social", "preference/attribute", "causal"} else 0.0
            base_bonus = 0.22 if component.component_id in base_ids else 0.0
            score = (
                hit_scores.get(component.component_id, 0.0)
                + anchor_overlap
                + self._answer_slot_bonus(slot, component.text)
                + self._habit_memory_bonus(question, component.text)
                + relation_bonus
                + base_bonus
            )
            scored.append((score, component))
        scored.sort(key=lambda row: (row[0], len(row[1].source_turn_ids)), reverse=True)
        return scored

    def _terminal_answer_components(self, question: str, scored_components: list[tuple[float, Any]], limit: int) -> list[Any]:
        slot = self._expected_answer_slot(question)
        terminals: list[Any] = []
        seen: set[str] = set()
        for _score, component in scored_components:
            if component.component_id in seen:
                continue
            if self._answer_slot_bonus(slot, component.text) <= 0 and len(terminals) >= max(3, limit // 2):
                continue
            terminals.append(component)
            seen.add(component.component_id)
            if len(terminals) >= limit:
                break
        for _score, component in scored_components:
            if len(terminals) >= limit:
                break
            if component.component_id in seen:
                continue
            terminals.append(component)
            seen.add(component.component_id)
        return terminals

    def _slot_tree_bridge_edges(
        self,
        question: str,
        anchors: list[Any],
        terminals: list[Any],
        limit: int,
    ) -> list[tuple[float, Any, Any, Any]]:
        anchor_ids = {component.component_id for component in anchors}
        terminal_ids = {component.component_id for component in terminals}
        query_terms = set(self._anchor_terms(question))
        candidates: list[tuple[float, Any, Any, Any]] = []
        for edge in self.store.edges:
            left = self.store.components.get(edge.source_component_id)
            right = self.store.components.get(edge.target_component_id)
            if not left or not right:
                continue
            left_anchor = left.component_id in anchor_ids
            right_anchor = right.component_id in anchor_ids
            left_terminal = left.component_id in terminal_ids
            right_terminal = right.component_id in terminal_ids
            if not ((left_anchor and right_terminal) or (right_anchor and left_terminal)):
                continue
            text = " ".join([edge.edge_type, edge.relation_text, left.text, right.text])
            overlap = len(query_terms & set(content_tokens(text))) / max(1, len(query_terms))
            type_bonus = {"same_entity": 0.35, "source_of": 0.25, "same_session": 0.15}.get(edge.edge_type, 0.1)
            candidates.append((overlap + type_bonus, edge, left, right))
        candidates.sort(key=lambda row: row[0], reverse=True)
        return candidates[:limit]

    def _dedupe_components(self, components: list[Any]) -> list[Any]:
        deduped: list[Any] = []
        seen: set[str] = set()
        for component in components:
            if not component or component.component_id in seen:
                continue
            deduped.append(component)
            seen.add(component.component_id)
        return deduped

    def _anchor_terms(self, question: str) -> list[str]:
        generic = {
            "does",
            "did",
            "done",
            "have",
            "having",
            "kind",
            "type",
            "many",
            "much",
            "times",
            "new",
            "shared",
            "significant",
            "life",
            "events",
            "items",
            "activities",
            "locations",
            "classes",
            "sports",
            "like",
            "likes",
            "liked",
            "artist",
            "artists",
            "listen",
            "listened",
            "music",
            "song",
            "songs",
            "used",
        }
        terms: list[str] = []
        for tok in content_tokens(question):
            if tok.endswith("'s"):
                tok = tok[:-2]
            if tok in generic or tok in terms:
                continue
            terms.append(tok)
            if tok.endswith("s") and len(tok) > 4 and tok[:-1] not in generic and tok[:-1] not in terms:
                terms.append(tok[:-1])
            if tok == "try" and "tried" not in terms:
                terms.append("tried")
        return terms[:12]

    def _expanded_slot_query(self, question: str) -> str:
        lowered = (question or "").lower()
        expansions: list[str] = []
        if any(term in lowered for term in ["like", "likes", "liked"]):
            expansions.extend(["love", "loves", "loved", "enjoy", "enjoys", "favorite", "stoked"])
        if "child" in lowered or "kid" in lowered:
            expansions.extend(["child", "children", "kids", "family"])
        if any(term in lowered for term in ["used to", "as a kid", "childhood"]):
            expansions.extend(["used to", "back then", "childhood", "those were the days"])
        if "artist" in lowered or "listen" in lowered:
            expansions.extend(["music", "song", "artist", "band", "listened"])
        if "shared" in lowered or "both" in lowered:
            expansions.extend(["both", "together", "also"])
        return " ".join([question, *expansions])

    def _answer_slot_bonus(self, slot: str, text: str) -> float:
        lowered = (text or "").lower()
        if slot == "count":
            count_words = {
                "once",
                "twice",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
            }
            return 0.45 if re.search(r"\b\d+\b", lowered) or any(word in lowered.split() for word in count_words) else 0.0
        if slot == "person/entity":
            return 0.25 if re.search(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)?\b", text or "") else 0.0
        if slot == "location":
            return 0.2 if re.search(r"\b(?:at|in|near|from|to)\s+[A-Z][A-Za-z']+", text or "") else 0.0
        if slot == "yes/no":
            return 0.25 if any(term in lowered for term in ["yes", "no", "want", "wants", "plan", "plans", "can", "cannot", "can't"]) else 0.0
        preference_bonus = self._preference_object_bonus(lowered)
        list_bonus = 0.18 if "," in (text or "") or " and " in lowered else 0.0
        name_bonus = 0.12 if re.search(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)?\b", text or "") else 0.0
        return preference_bonus + list_bonus + name_bonus

    def _preference_object_bonus(self, lowered_text: str) -> float:
        if not any(term in lowered_text for term in ["love", "like", "enjoy", "favorite", "stoked", "fan of"]):
            return 0.0
        if "sounds like" in lowered_text or "looks like" in lowered_text:
            return 0.0
        pronouns = {"it", "this", "that", "these", "those", "them", "our", "my", "your", "his", "her", "their", "its"}
        bad_starts = pronouns | {"for", "and", "to", "with", "of", "about"}
        patterns = [
            r"\b(?:love|loves|loved|like|likes|liked|enjoy|enjoys|enjoyed)\s+([a-z][a-z\s'-]{2,60})",
            r"\bstoked\s+for\s+([a-z][a-z\s'-]{2,60})",
            r"\bfan\s+of\s+([a-z][a-z\s'-]{2,60})",
            r"\bfavorite\s+(?:is|are|was|were)?\s*([a-z][a-z\s'-]{2,60})",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered_text)
            if not match:
                continue
            first_words = re.findall(r"[a-z][a-z']*", match.group(1).lower())
            if first_words and first_words[0] in bad_starts:
                return -0.12
            raw_tokens = content_tokens(match.group(1))
            if raw_tokens and raw_tokens[0] in bad_starts:
                return -0.12
            obj_tokens = [tok for tok in raw_tokens if tok not in pronouns]
            if obj_tokens:
                return 0.55
            return -0.12
        return 0.25

    def _habit_memory_bonus(self, question: str, text: str) -> float:
        lowered_question = (question or "").lower()
        if not any(
            marker in lowered_question
            for marker in ["used to", "as a kid", "when he was a kid", "when she was a kid", "when they were kids", "childhood"]
        ):
            return 0.0
        return self._habit_memory_bonus_from_text(text)

    def _habit_memory_bonus_from_text(self, text: str) -> float:
        lowered_text = (text or "").lower()
        if any(marker in lowered_text for marker in ["used to", "as a kid", "when i was a kid", "childhood", "those were the days", "back then"]):
            return 0.65
        return 0.0

    def _graph_walk_plan_context(
        self,
        *,
        question: str,
        family: str,
        component_hits: list[tuple[float, Any]],
        walked_edges: list[tuple[float, Any, Any, Any]],
        timeline_turns: list[RawTurn],
        base_context: str,
    ) -> str:
        walked_components = self._components_from_walk(component_hits, walked_edges)
        key_terms = []
        for tok in content_tokens(question):
            if tok not in key_terms:
                key_terms.append(tok)
        slot = self._expected_answer_slot(question)
        lines = [
            "### Graph-Walk Evidence Plan",
            "Use this structured plan before the fallback context. Answer from source turns when the plan and fallback disagree.",
            f"- Question family: {family}",
            f"- Expected answer slot: {slot}",
            f"- Key terms: {', '.join(key_terms[:12])}",
        ]
        if family == "temporal":
            lines.append("- Reasoning order: identify the event anchor, align it to dated source turns, then answer with the best-supported date or duration.")
        elif family == "multi-hop":
            lines.append("- Reasoning order: connect anchor facts through bridge paths, then fill the requested answer slot from the terminal evidence.")

        if component_hits:
            lines.extend(["", "### Anchor Evidence Components"])
            for rank, (score, component) in enumerate(component_hits[:10], start=1):
                lines.append(self._component_plan_line(rank, score, component))

        if walked_edges:
            lines.extend(["", "### Graph Walk Bridge Paths"])
            for rank, (score, edge, left, right) in enumerate(walked_edges[:8], start=1):
                left_text = self._short_text(left.text if left else edge.source_component_id, 180)
                right_text = self._short_text(right.text if right else edge.target_component_id, 180)
                source_ids = ",".join(edge.source_turn_ids)
                lines.append(
                    f"{rank}. [{edge.edge_type}; score={score:.3f}; source_turns={source_ids}] "
                    f"{left_text} => {right_text}"
                )

        if family == "temporal":
            temporal_components = [
                component
                for component in walked_components
                if component.time_expr or component.event_time or component.mentioned_at
            ][:8]
            if temporal_components:
                lines.extend(["", "### Candidate Timeline Anchors"])
                for rank, component in enumerate(temporal_components, start=1):
                    date = component.event_time or component.time_expr or component.mentioned_at
                    source_ids = ",".join(component.source_turn_ids)
                    lines.append(f"{rank}. [{date}; source_turns={source_ids}] {component.text}")

        if timeline_turns:
            title = "Timeline Source Turns" if family == "temporal" else "Bridge Source Turns"
            lines.extend(["", f"### {title}"])
            for turn in sorted(timeline_turns, key=lambda item: (item.chronological_order, item.turn_index, item.turn_id))[:12]:
                lines.append(self._short_text(turn.to_context_line(), 320))

        if base_context.strip():
            lines.extend(["", "### HiGMem Baseline Fallback Context", self._short_text(base_context.strip(), 6000)])
        return "\n".join(lines)

    def _answer_slot_tree_context(
        self,
        *,
        question: str,
        anchors: list[Any],
        terminals: list[Any],
        bridge_edges: list[tuple[float, Any, Any, Any]],
        source_turns: list[RawTurn],
        base_context: str,
    ) -> str:
        slot = self._expected_answer_slot(question)
        anchors = anchors[:6]
        terminals = terminals[:6]
        lines = [
            "### Answer-Slot Evidence Tree",
            "Use this tree to answer multi-hop questions. Prefer terminal answer candidates over broad neighboring facts.",
            f"- Expected answer slot: {slot}",
            f"- Anchor terms: {', '.join(self._anchor_terms(question))}",
            "- Reasoning order: root anchor facts -> bridge paths -> terminal answer candidates -> verified source turns.",
        ]
        if anchors:
            lines.extend(["", "### Root Anchor Facts"])
            for rank, component in enumerate(anchors, start=1):
                lines.append(self._component_plan_line(rank, 0.0, component))
        if terminals:
            lines.extend(["", "### Terminal Answer Candidates"])
            for rank, component in enumerate(terminals, start=1):
                lines.append(self._component_plan_line(rank, self._answer_slot_bonus(slot, component.text), component))
        if bridge_edges:
            lines.extend(["", "### Constrained Bridge Paths"])
            for rank, (score, edge, left, right) in enumerate(bridge_edges[:6], start=1):
                left_text = self._short_text(left.text if left else edge.source_component_id, 170)
                right_text = self._short_text(right.text if right else edge.target_component_id, 170)
                lines.append(f"{rank}. [{edge.edge_type}; score={score:.3f}] {left_text} => {right_text}")
        if source_turns:
            lines.extend(["", "### Verified Source Turns"])
            for turn in sorted(source_turns, key=lambda item: (item.chronological_order, item.turn_index, item.turn_id))[:10]:
                lines.append(self._short_text(turn.to_context_line(), 300))
        if base_context.strip():
            lines.extend(["", "### HiGMem Baseline Fallback Context", self._short_text(base_context.strip(), 5000)])
        return "\n".join(lines)

    def _evidence_frame_context(
        self,
        *,
        frame: dict[str, Any],
        roots: list[Any],
        terminals: list[Any],
        bridge_edges: list[tuple[float, Any, Any, Any]],
        source_turns: list[RawTurn],
        base_context: str,
    ) -> str:
        lines = [
            "### Universal Evidence Frame",
            "Use this frame to answer the question from concrete terminal evidence, not from broad neighboring facts.",
            f"- Answer slot: {frame.get('slot')}",
            f"- Intents: {', '.join(str(item) for item in frame.get('intents', []))}",
            f"- Anchor terms: {', '.join(str(item) for item in frame.get('anchor_terms', []))}",
            "- Constraints: prefer terminal candidates with explicit answer objects; ignore pronoun-only candidates unless source turns resolve them; include all supported items for set answers.",
        ]
        if roots:
            lines.extend(["", "### Frame Root Evidence"])
            for rank, component in enumerate(roots[:6], start=1):
                lines.append(self._component_plan_line(rank, 0.0, component))
        if terminals:
            lines.extend(["", "### Terminal Answer Candidates"])
            for rank, component in enumerate(terminals[:10], start=1):
                label = self._candidate_answer_label(frame, component.text)
                label_text = f"; candidate={label}" if label else ""
                lines.append(f"{self._component_plan_line(rank, self._frame_answer_score(frame, component.text), component)}{label_text}")
        if bridge_edges:
            lines.extend(["", "### Minimal Bridge Paths"])
            for rank, (score, edge, left, right) in enumerate(bridge_edges[:6], start=1):
                left_text = self._short_text(left.text if left else edge.source_component_id, 170)
                right_text = self._short_text(right.text if right else edge.target_component_id, 170)
                lines.append(f"{rank}. [{edge.edge_type}; score={score:.3f}] {left_text} => {right_text}")
        if source_turns:
            lines.extend(["", "### Verified Source Turns"])
            for turn in sorted(source_turns, key=lambda item: (item.chronological_order, item.turn_index, item.turn_id))[:12]:
                lines.append(self._short_text(turn.to_context_line(), 340))
        if base_context.strip():
            lines.extend(["", "### HiGMem Baseline Fallback Context", self._short_text(base_context.strip(), 5000)])
        return "\n".join(lines)

    def _expected_answer_slot(self, question: str) -> str:
        lowered = (question or "").lower()
        if lowered.startswith("when") or " what date" in lowered or "which date" in lowered:
            return "date/time"
        if "how long" in lowered:
            return "duration"
        if "how many" in lowered or "number of" in lowered:
            return "count"
        if lowered.startswith("who"):
            return "person/entity"
        if lowered.startswith("where"):
            return "location"
        if lowered.startswith("did ") or lowered.startswith("was ") or lowered.startswith("is "):
            return "yes/no"
        return "entity/attribute"

    def _component_plan_line(self, rank: int, score: float, component: Any) -> str:
        relation = str(component.metadata.get("relation_type") or "")
        source_ids = ",".join(component.source_turn_ids)
        time_text = component.event_time or component.time_expr or component.mentioned_at
        slots = [
            f"component={component.component_id}",
            f"score={score:.3f}",
            f"source_turns={source_ids}",
            f"relation={relation}",
        ]
        if time_text:
            slots.append(f"time={time_text}")
        if component.subject or component.predicate or component.object:
            slots.append(f"triple={component.subject or '?'}|{component.predicate or '?'}|{component.object or '?'}")
        return f"{rank}. [{' ; '.join(slots)}] {self._short_text(component.text, 240)}"

    def _short_text(self, text: str, limit: int) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)].rstrip() + "..."

    def _check_sufficiency(
        self,
        question: str,
        family: str,
        base_context: str,
        component_ids: list[str],
        source_turns: list[RawTurn],
        edge_hits: list[tuple[float, Any]],
        metadata: dict[str, Any],
    ) -> SufficiencyResult:
        missing = []
        reasons = []
        context_tokens = set(content_tokens(base_context))
        question_tokens = [tok for tok in content_tokens(question) if len(tok) >= 4]
        overlap = set(question_tokens) & context_tokens
        if question_tokens and len(overlap) < min(2, len(set(question_tokens))):
            missing.append("missing_subject")
            reasons.append("base context has weak lexical overlap with question entities")
        if family == "temporal":
            temporal_components = [
                self.store.components[cid]
                for cid in component_ids
                if cid in self.store.components and (self.store.components[cid].time_expr or self.store.components[cid].event_time)
            ]
            if not temporal_components:
                missing.append("missing_time")
                reasons.append("no temporal component with time expression or normalized event_time was retrieved")
        if family == "open-domain":
            relation_types = Counter(
                str(self.store.components[cid].metadata.get("relation_type"))
                for cid in component_ids
                if cid in self.store.components
            )
            if not any(relation_types.get(kind, 0) for kind in ("causal", "preference/attribute", "social", "entity")):
                missing.append("missing_causal_premise")
                reasons.append("retrieved components do not contain typed premise relations")
        if metadata.get("dataset") == "longdialqa" or family == "longdialogue":
            speakers = {turn.speaker for turn in source_turns if turn.speaker}
            question_speaker = metadata.get("question_asker") or metadata.get("chatbot")
            if question_speaker and question_speaker not in speakers and len(speakers) < 2:
                missing.append("missing_speaker")
                reasons.append("retrieved source turns do not preserve enough speaker context")
        if any(edge.edge_type == "conflict_with" for _score, edge in edge_hits):
            missing.append("conflicting_evidence")
            reasons.append("retrieved evidence graph contains conflict_with edge")
        if not base_context and not source_turns:
            missing.append("missing_episode_context")
            reasons.append("no compressed or raw context available")
        if "conflicting_evidence" in missing:
            status = "CONFLICTED"
        elif any(slot == "missing_episode_context" for slot in missing):
            status = "ABSENT"
        elif "missing_time" in missing and source_turns:
            status = "INFERABLE"
        elif missing:
            status = "UNDER-SPECIFIED"
        else:
            status = "SUPPORTED"
        return SufficiencyResult(sufficient=not missing, status=status, missing_slots=sorted(set(missing)), reasons=reasons)

    def _repair(self, question: str, source_turns: list[RawTurn], episode_limit: int) -> tuple[list[RawTurn], list[Any]]:
        episodes = []
        for turn in source_turns:
            episode = self.store.episode_for_turn_id(turn.turn_id)
            if episode and episode not in episodes:
                episodes.append(episode)
        if not episodes:
            episodes = [episode for _score, episode in self.store.search_episodes(question, limit=episode_limit)]
        episodes = episodes[:episode_limit]
        repaired_turns: list[RawTurn] = []
        seen = set()
        anchor_ids = {turn.turn_id for turn in source_turns}
        question_terms = {tok for tok in content_tokens(question) if len(tok) >= 4}
        for episode in episodes:
            selected_indexes: set[int] = set()
            for idx, turn in enumerate(episode.turns):
                if turn.turn_id in anchor_ids:
                    selected_indexes.update(range(max(0, idx - 2), min(len(episode.turns), idx + 3)))
            if not selected_indexes:
                scored = []
                for idx, turn in enumerate(episode.turns):
                    turn_terms = {tok for tok in content_tokens(turn.text) if len(tok) >= 4}
                    scored.append((len(question_terms & turn_terms), idx))
                for _score, idx in sorted(scored, reverse=True)[:3]:
                    selected_indexes.update(range(max(0, idx - 2), min(len(episode.turns), idx + 3)))
            for idx in sorted(selected_indexes):
                turn = episode.turns[idx]
                if turn.turn_id in seen:
                    continue
                repaired_turns.append(turn)
                seen.add(turn.turn_id)
        return repaired_turns, episodes

    def _rank_turns_for_question(self, question: str, turns: list[RawTurn], limit: int) -> list[RawTurn]:
        if len(turns) <= limit:
            return turns
        question_terms = {tok for tok in content_tokens(question) if len(tok) >= 4}
        scored: list[tuple[int, int, int, str, RawTurn]] = []
        for pos, turn in enumerate(turns):
            turn_terms = {tok for tok in content_tokens(turn.text) if len(tok) >= 4}
            score = len(question_terms & turn_terms)
            scored.append((-score, int(turn.chronological_order or 0), int(turn.turn_index or pos), turn.turn_id, turn))
        selected = [turn for _score, _order, _idx, _turn_id, turn in sorted(scored)[:limit]]
        return sorted(selected, key=lambda turn: (int(turn.chronological_order or 0), int(turn.turn_index or 0), turn.turn_id))

    def _route(self, question: str, family: str, metadata: dict[str, Any]) -> RouteDecision:
        if family == "temporal":
            return RouteDecision(
                question_type="temporal",
                evidence_risk="high",
                needed_layers=["components", "graph", "episode_repair"],
                retrieval_budget={"components": 12, "edges": 10, "episodes": 2},
                route_name="temporal_component_source_verify",
                reasons=["temporal question requires explicit time evidence and raw verification"],
            )
        if family == "open-domain":
            return RouteDecision(
                question_type="open-domain",
                evidence_risk="high",
                needed_layers=["components", "graph", "episode_repair"],
                retrieval_budget={"components": 14, "edges": 12, "episodes": 3},
                route_name="typed_premise_path_repair",
                reasons=["open-domain question needs typed premise paths"],
            )
        if family == "multi-hop":
            return RouteDecision(
                question_type="multi-hop",
                evidence_risk="medium",
                needed_layers=["components", "episode_repair"],
                retrieval_budget={"components": 16, "edges": 6, "episodes": 3},
                route_name="diverse_component_episode_repair",
                reasons=["multi-hop question may need several local episodes"],
            )
        if family == "adversarial":
            return RouteDecision(
                question_type="adversarial",
                evidence_risk="high",
                needed_layers=["components", "raw_episode"],
                retrieval_budget={"components": 8, "edges": 0, "episodes": 2},
                route_name="absence_check_episode_fallback",
                reasons=["unanswerable questions require conservative evidence absence checks"],
            )
        if family == "longdialogue":
            return RouteDecision(
                question_type="longdialogue/multi-party",
                evidence_risk="medium",
                needed_layers=["components", "raw_episode"],
                retrieval_budget={"components": 10, "edges": 4, "episodes": 2},
                route_name="scene_speaker_episode_route",
                reasons=["scene-local multi-party dialogue needs speaker and episode context"],
            )
        return RouteDecision(
            question_type="single-hop",
            evidence_risk="low",
            needed_layers=["components"],
            retrieval_budget={"components": 8, "edges": 0, "episodes": 1},
            route_name="compact_component_route",
            reasons=["single-hop question can start from compact evidence components"],
        )

    def _component_context(self, hits: list[tuple[float, Any]]) -> str:
        lines = ["### HiGMemPlus Evidence Components"]
        for rank, (score, component) in enumerate(hits, start=1):
            lines.append(
                f"{rank}. [{component.component_id}; score={score:.3f}; source={','.join(component.source_turn_ids)}] "
                f"{component.text}"
            )
            if component.time_expr or component.event_time:
                lines.append(f"   time_expr={component.time_expr}; event_time={component.event_time}; mentioned_at={component.mentioned_at}")
        return "\n".join(lines)

    def _edge_context(self, hits: list[tuple[float, Any]]) -> str:
        lines = ["### HiGMemPlus Typed Evidence Paths"]
        for rank, (score, edge) in enumerate(hits, start=1):
            left = self.store.components.get(edge.source_component_id)
            right = self.store.components.get(edge.target_component_id)
            left_text = left.text if left else edge.source_component_id
            right_text = right.text if right else edge.target_component_id
            lines.append(f"{rank}. [{edge.edge_type}; score={score:.3f}] {left_text} -> {right_text}")
        return "\n".join(lines)

    def _raw_turn_context(self, title: str, turns: list[RawTurn]) -> str:
        lines = [f"### HiGMemPlus {title}"]
        for turn in turns:
            lines.append(turn.to_context_line())
        return "\n".join(lines)

    def _records_from_components(self, hits: list[tuple[float, Any]]) -> list[dict[str, Any]]:
        records = []
        for score, component in hits:
            records.append(
                {
                    "used_content": component.text,
                    "metadata": {
                        "source": "higmem_plus_component",
                        "score": score,
                        "component_id": component.component_id,
                        "event_id": component.event_id,
                        "source_turn_ids": component.source_turn_ids,
                        "scene_id": component.source_session_or_episode_id,
                        "scene_ids_in_text": [component.source_session_or_episode_id],
                    },
                }
            )
        return records

    def _records_from_turns(self, turns: list[RawTurn]) -> list[dict[str, Any]]:
        records = []
        seen = set()
        for turn in turns:
            if turn.turn_id in seen:
                continue
            seen.add(turn.turn_id)
            records.append(
                {
                    "used_content": turn.to_context_line(),
                    "metadata": {
                        "source": "higmem_plus_raw_turn",
                        "turn_id": turn.turn_id,
                        "scene_id": turn.session_or_scene_id,
                        "scene_ids_in_text": [turn.session_or_scene_id] if turn.session_or_scene_id else [],
                        "speaker": turn.speaker,
                    },
                }
            )
        return records


__all__ = ["HiGMemPlusEnhancer", "RawTurn", "METHODS"]
