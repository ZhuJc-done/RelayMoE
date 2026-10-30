from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .catalog import CandidateTreeCatalog
from .model import (
    HierarchicalTopology,
    LogicalArc,
    RelayTree,
    TrafficClass,
    ensure_traffic_matches_topology,
)


@dataclass(frozen=True)
class SelectionResult:
    selected_trees: Mapping[str, RelayTree]
    edge_load_us: Mapping[str, float]
    theta_us: float
    doubling_rounds: int
    candidates_evaluated: int

    @property
    def max_edge_load_us(self) -> float:
        return max(self.edge_load_us.values(), default=0.0)


@dataclass(frozen=True)
class ScheduledAction:
    name: str
    phase: str
    traffic_name: str
    tree_name: str
    arc: LogicalArc
    physical_edges: tuple[str, ...]
    size_bytes: int
    dependencies: tuple[str, ...] = ()
    remaining_depth: int = 0
    priority_load_us: float = 0.0
    relay: bool = False
    reduction_inputs: int = 0


@dataclass(frozen=True)
class Stage:
    index: int
    actions: tuple[ScheduledAction, ...]

    @property
    def occupied_edges(self) -> set[str]:
        occupied: set[str] = set()
        for action in self.actions:
            occupied.update(action.physical_edges)
        return occupied


@dataclass(frozen=True)
class Schedule:
    phase: str
    stages: tuple[Stage, ...]
    # None means every physical edge is exclusive. A concrete set restricts
    # conflict avoidance to those edges; an empty set permits all sharing.
    exclusive_edges: frozenset[str] | None = None

    @property
    def actions(self) -> tuple[ScheduledAction, ...]:
        return tuple(action for stage in self.stages for action in stage.actions)

    def validate(self) -> None:
        completed: set[str] = set()
        seen: set[str] = set()
        for expected_index, stage in enumerate(self.stages):
            if stage.index != expected_index:
                raise ValueError(
                    f"{self.phase} stage index {stage.index} != {expected_index}"
                )
            occupied: set[str] = set()
            for action in stage.actions:
                if action.name in seen:
                    raise ValueError(f"action {action.name} appears more than once")
                if not set(action.dependencies).issubset(completed):
                    missing = set(action.dependencies) - completed
                    raise ValueError(
                        f"action {action.name} has unmet dependencies {sorted(missing)}"
                    )
                action_edges = set(action.physical_edges)
                if self.exclusive_edges is not None:
                    action_edges.intersection_update(self.exclusive_edges)
                conflict = occupied.intersection(action_edges)
                if conflict:
                    raise ValueError(
                        f"stage {stage.index} has edge conflicts {sorted(conflict)}"
                    )
                occupied.update(action_edges)
                seen.add(action.name)
            completed.update(action.name for action in stage.actions)


class PotentialGreedyScheduler:
    """Tree selection and causal, directed-edge-disjoint stage packing."""

    def __init__(
        self,
        topology: HierarchicalTopology,
        *,
        catalog: CandidateTreeCatalog | None = None,
        max_doublings: int = 16,
    ) -> None:
        self.topology = topology
        self.catalog = catalog or CandidateTreeCatalog()
        self.max_doublings = max_doublings

    def select(self, traffic: Iterable[TrafficClass]) -> SelectionResult:
        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        if not flows:
            raise ValueError("at least one traffic class is required")

        candidates_by_flow: dict[str, tuple[RelayTree, ...]] = {}
        for flow in flows:
            candidates = self.catalog.candidates(self.topology, flow)
            if not candidates:
                raise ValueError(f"traffic {flow.name} has no candidate tree")
            candidates_by_flow[flow.name] = candidates

        theta = self._initial_theta(flows, candidates_by_flow)
        ordered = sorted(flows, key=lambda flow: (-flow.volume_bytes, flow.name))
        candidates_evaluated = 0

        final_trees: dict[str, RelayTree] = {}
        final_loads: dict[str, float] = {}
        final_round = 0
        for doubling_round in range(self.max_doublings + 1):
            edge_loads = {edge_name: 0.0 for edge_name in self.topology.edges}
            selected: dict[str, RelayTree] = {}

            for flow in ordered:
                best_tree: RelayTree | None = None
                best_delta: dict[str, float] | None = None
                best_key: tuple[float, float, int, str] | None = None
                for tree in candidates_by_flow[flow.name]:
                    candidates_evaluated += 1
                    delta = tree.normalized_edge_increment_us(
                        self.topology, flow.volume_bytes
                    )
                    score = self._potential_increment(edge_loads, delta, theta)
                    max_after = max(
                        (
                            edge_loads[edge_name] + delta.get(edge_name, 0.0)
                            for edge_name in edge_loads
                        ),
                        default=0.0,
                    )
                    key = (score, max_after, len(tree.arcs), tree.name)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_tree = tree
                        best_delta = delta

                assert best_tree is not None and best_delta is not None
                selected[flow.name] = best_tree
                for edge_name, increment in best_delta.items():
                    edge_loads[edge_name] += increment

            max_load = max(edge_loads.values(), default=0.0)
            threshold = (1.0 + math.log(max(2, len(self.topology.edges)))) * theta
            final_trees = selected
            final_loads = edge_loads
            final_round = doubling_round
            if max_load <= threshold:
                break
            theta *= 2.0

        return SelectionResult(
            selected_trees=final_trees,
            edge_load_us=final_loads,
            theta_us=theta,
            doubling_rounds=final_round,
            candidates_evaluated=candidates_evaluated,
        )

    def build_dispatch_schedule(
        self,
        selection: SelectionResult,
        traffic: Iterable[TrafficClass],
        *,
        exclusive_edges: frozenset[str] | None = None,
    ) -> Schedule:
        flow_by_name = {flow.name: flow for flow in traffic}
        actions: list[ScheduledAction] = []
        for flow_name, tree in sorted(selection.selected_trees.items()):
            flow = flow_by_name[flow_name]
            parent_arcs = tree.parent_arcs()
            remaining_depth = tree.remaining_depths()
            for arc in tree.arcs:
                action_name = self._action_name("dispatch", flow_name, arc)
                dependency = parent_arcs.get(arc.src)
                dependencies = (
                    (self._action_name("dispatch", flow_name, dependency),)
                    if dependency is not None
                    else ()
                )
                physical_edges = self.topology.route(arc.src, arc.dst)
                actions.append(
                    ScheduledAction(
                        name=action_name,
                        phase="dispatch",
                        traffic_name=flow_name,
                        tree_name=tree.name,
                        arc=arc,
                        physical_edges=physical_edges,
                        size_bytes=flow.volume_bytes,
                        dependencies=dependencies,
                        remaining_depth=remaining_depth.get(arc.dst, 0),
                        priority_load_us=max(
                            (
                                selection.edge_load_us.get(edge_name, 0.0)
                                for edge_name in physical_edges
                            ),
                            default=0.0,
                        ),
                        relay=arc.src != tree.root,
                    )
                )

        schedule = self._pack_actions(
            "dispatch",
            actions,
            exclusive_edges=exclusive_edges,
        )
        schedule.validate()
        return schedule

    def build_combine_schedule(
        self,
        selection: SelectionResult,
        dispatch_schedule: Schedule,
    ) -> Schedule:
        dispatch_by_key = {
            (action.traffic_name, action.arc.src, action.arc.dst): action
            for action in dispatch_schedule.actions
        }
        combine_by_dispatch_name: dict[str, ScheduledAction] = {}

        for flow_name, tree in selection.selected_trees.items():
            children = tree.children()
            for arc in tree.arcs:
                dispatch_action = dispatch_by_key[(flow_name, arc.src, arc.dst)]
                dependency_names = tuple(
                    self._action_name(
                        "combine",
                        flow_name,
                        LogicalArc(arc.dst, child),
                    )
                    for child in children.get(arc.dst, ())
                )
                reverse_arc = LogicalArc(arc.dst, arc.src)
                combine_by_dispatch_name[dispatch_action.name] = ScheduledAction(
                    name=self._action_name("combine", flow_name, arc),
                    phase="combine",
                    traffic_name=flow_name,
                    tree_name=tree.name,
                    arc=reverse_arc,
                    physical_edges=self.topology.route(reverse_arc.src, reverse_arc.dst),
                    size_bytes=dispatch_action.size_bytes,
                    dependencies=dependency_names,
                    remaining_depth=0,
                    priority_load_us=dispatch_action.priority_load_us,
                    relay=bool(children.get(arc.dst)),
                    reduction_inputs=len(children.get(arc.dst, ())),
                )

        stages: list[Stage] = []
        for dispatch_stage in reversed(dispatch_schedule.stages):
            actions = tuple(
                combine_by_dispatch_name[action.name]
                for action in dispatch_stage.actions
            )
            stages.append(Stage(index=len(stages), actions=actions))
        schedule = Schedule(
            phase="combine",
            stages=tuple(stages),
            exclusive_edges=dispatch_schedule.exclusive_edges,
        )
        schedule.validate()
        return schedule

    def plan(
        self,
        traffic: Iterable[TrafficClass],
    ) -> tuple[SelectionResult, Schedule, Schedule]:
        flows = tuple(traffic)
        selection = self.select(flows)
        dispatch = self.build_dispatch_schedule(selection, flows)
        combine = self.build_combine_schedule(selection, dispatch)
        return selection, dispatch, combine

    def _initial_theta(
        self,
        flows: tuple[TrafficClass, ...],
        candidates_by_flow: Mapping[str, tuple[RelayTree, ...]],
    ) -> float:
        largest_unavoidable_step = 0.0
        for flow in flows:
            candidate_limits = []
            for tree in candidates_by_flow[flow.name]:
                delta = tree.normalized_edge_increment_us(
                    self.topology, flow.volume_bytes
                )
                candidate_limits.append(max(delta.values(), default=0.0))
            largest_unavoidable_step = max(
                largest_unavoidable_step, min(candidate_limits)
            )

        total_capacity_bytes_per_us = sum(
            edge.bytes_per_us for edge in self.topology.edges.values()
        )
        average_lower_bound = (
            sum(flow.volume_bytes for flow in flows) / total_capacity_bytes_per_us
        )
        return max(largest_unavoidable_step, average_lower_bound, 1e-9)

    @staticmethod
    def _potential_increment(
        current: Mapping[str, float],
        delta: Mapping[str, float],
        theta: float,
    ) -> float:
        score = 0.0
        for edge_name, increment in delta.items():
            current_scaled = min(current.get(edge_name, 0.0) / theta, 60.0)
            increment_scaled = min(increment / theta, 60.0)
            score += math.exp(current_scaled) * math.expm1(increment_scaled)
        return score

    def _pack_actions(
        self,
        phase: str,
        actions: Iterable[ScheduledAction],
        *,
        exclusive_edges: frozenset[str] | None = None,
    ) -> Schedule:
        pending = {action.name: action for action in actions}
        completed: set[str] = set()
        stages: list[Stage] = []

        while pending:
            ready = [
                action
                for action in pending.values()
                if set(action.dependencies).issubset(completed)
            ]
            if not ready:
                raise ValueError(
                    f"{phase} actions contain a dependency cycle: {sorted(pending)}"
                )
            ready.sort(
                key=lambda action: (
                    -action.remaining_depth,
                    -action.priority_load_us,
                    action.name,
                )
            )

            occupied: set[str] = set()
            chosen: list[ScheduledAction] = []
            for action in ready:
                action_edges = set(action.physical_edges)
                if exclusive_edges is not None:
                    action_edges.intersection_update(exclusive_edges)
                if occupied.isdisjoint(action_edges):
                    chosen.append(action)
                    occupied.update(action_edges)
            if not chosen:
                raise RuntimeError(f"unable to pack any ready {phase} action")

            stage = Stage(index=len(stages), actions=tuple(chosen))
            stages.append(stage)
            for action in chosen:
                completed.add(action.name)
                del pending[action.name]

        return Schedule(
            phase=phase,
            stages=tuple(stages),
            exclusive_edges=exclusive_edges,
        )

    @staticmethod
    def _action_name(phase: str, flow_name: str, arc: LogicalArc) -> str:
        if phase == "dispatch":
            src, dst = arc.src, arc.dst
        elif phase == "combine":
            # The input arc is the dispatch direction; the action runs in reverse.
            src, dst = arc.dst, arc.src
        else:
            raise ValueError(f"unknown phase {phase}")
        return f"{phase}:{flow_name}:{src}->{dst}"
