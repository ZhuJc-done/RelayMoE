from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .catalog import CandidateTreeCatalog, OfflineBinomialCatalog
from .model import (
    HierarchicalTopology,
    TrafficClass,
    ensure_traffic_matches_topology,
)
from .scenarios import TokenRouting
from .scheduler import (
    EventDrivenPhaseResult,
    PotentialGreedyScheduler,
    ScheduledAction,
    SelectionResult,
)


@dataclass(frozen=True)
class ExecutionDAG:
    """Backend-neutral action DAG consumed by a real communication runtime."""

    phase: str
    actions: tuple[ScheduledAction, ...]

    def validate(self) -> None:
        if self.phase not in {"dispatch", "combine"}:
            raise ValueError(f"unsupported RelayMoE phase {self.phase!r}")
        by_name = {action.name: action for action in self.actions}
        if len(by_name) != len(self.actions):
            raise ValueError(f"{self.phase} DAG contains duplicate action names")

        indegree = {name: 0 for name in by_name}
        dependents: dict[str, list[str]] = defaultdict(list)
        for action in self.actions:
            if action.phase != self.phase:
                raise ValueError(
                    f"{action.name} belongs to {action.phase}, not {self.phase}"
                )
            for dependency in action.dependencies:
                if dependency not in by_name:
                    raise ValueError(
                        f"{action.name} depends on unknown action {dependency}"
                    )
                indegree[action.name] += 1
                dependents[dependency].append(action.name)

        ready = [name for name, count in indegree.items() if count == 0]
        visited = 0
        while ready:
            name = ready.pop()
            visited += 1
            for child in dependents[name]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(self.actions):
            raise ValueError(f"{self.phase} execution graph contains a cycle")

    def actions_for_gpu(self, gpu_name: str) -> tuple[ScheduledAction, ...]:
        """Return local send/receive work for one EP rank."""

        return tuple(
            action
            for action in self.actions
            if action.arc.src == gpu_name or action.arc.dst == gpu_name
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "actions": [
                {
                    "name": action.name,
                    "traffic_name": action.traffic_name,
                    "src": action.arc.src,
                    "dst": action.arc.dst,
                    "physical_edges": list(action.physical_edges),
                    "size_bytes": action.size_bytes,
                    "dependencies": list(action.dependencies),
                    "remaining_depth": action.remaining_depth,
                    "relay": action.relay,
                    "reduction_inputs": action.reduction_inputs,
                }
                for action in self.actions
            ],
        }


@dataclass(frozen=True)
class TrafficClassLayout:
    """Token/expert membership needed to pack one aggregated traffic class."""

    traffic_name: str
    source: str
    token_indices: tuple[int, ...]
    expert_indices: tuple[tuple[int, ...], ...]
    expert_weights: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.token_indices) != len(self.expert_indices):
            raise ValueError("token_indices and expert_indices must align")
        if len(set(self.token_indices)) != len(self.token_indices):
            raise ValueError("traffic-class token indices must be unique")
        if self.expert_weights is not None:
            if len(self.expert_weights) != len(self.expert_indices):
                raise ValueError("expert weights must align with tokens")
            for experts, weights in zip(
                self.expert_indices,
                self.expert_weights,
            ):
                if len(experts) != len(weights):
                    raise ValueError(
                        "each token's expert IDs and weights must align"
                    )


@dataclass(frozen=True)
class RelayMoEPlan:
    """One post-gating RelayMoE plan shared by Dispatch and Combine."""

    traffic_classes: tuple[TrafficClass, ...]
    selection: SelectionResult
    dispatch: ExecutionDAG
    combine: ExecutionDAG
    useful_bytes: int
    planning_time_us: float
    traffic_layouts: Mapping[str, TrafficClassLayout] = field(
        default_factory=dict
    )
    cache_hit: bool = False

    @property
    def action_count(self) -> int:
        return len(self.dispatch.actions) + len(self.combine.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "traffic_classes": [asdict(flow) for flow in self.traffic_classes],
            "selected_trees": {
                flow_name: {
                    "name": tree.name,
                    "kind": tree.kind,
                    "root": tree.root,
                    "destinations": list(tree.destinations),
                    "arcs": [
                        {"src": arc.src, "dst": arc.dst}
                        for arc in tree.arcs
                    ],
                }
                for flow_name, tree in self.selection.selected_trees.items()
            },
            "theta_us": self.selection.theta_us,
            "doubling_rounds": self.selection.doubling_rounds,
            "candidates_evaluated": self.selection.candidates_evaluated,
            "useful_bytes": self.useful_bytes,
            "planning_time_us": self.planning_time_us,
            "cache_hit": self.cache_hit,
            "traffic_layouts": {
                traffic_name: asdict(layout)
                for traffic_name, layout in self.traffic_layouts.items()
            },
            "dispatch": self.dispatch.to_dict(),
            "combine": self.combine.to_dict(),
        }


@dataclass(frozen=True)
class RelayMoESimulationResult:
    dispatch: EventDrivenPhaseResult
    combine: EventDrivenPhaseResult

    @property
    def total_time_us(self) -> float:
        return self.dispatch.total_time_us + self.combine.total_time_us

    @property
    def total_physical_bytes(self) -> int:
        return self.dispatch.physical_bytes + self.combine.physical_bytes


@runtime_checkable
class RelayMoEBackend(Protocol):
    """Integration point for a CUDA/RDMA/NCCL communication implementation.

    The backend owns tensor packing, transport submission, completion events,
    and in-place partial reduction.  It must honor ``dependencies`` and acquire
    every action's directed ``physical_edges`` exclusively.
    """

    def execute(
        self,
        *,
        plan: RelayMoEPlan,
        phase: ExecutionDAG,
        tensors: Any,
    ) -> Any:
        ...


class RelayMoE:
    """Upper-layer API for post-gating planning and MoE execution.

    A production MoE layer calls ``plan_from_gating`` (or supplies already
    aggregated ``TrafficClass`` objects), executes Dispatch, runs experts, then
    executes Combine with the same immutable plan.
    """

    def __init__(
        self,
        topology: HierarchicalTopology,
        *,
        backend: RelayMoEBackend | None = None,
        catalog_path: str | Path | None = None,
        require_prebuilt_catalog: bool = False,
        max_doublings: int = 16,
        plan_cache_size: int = 8,
    ) -> None:
        if plan_cache_size < 0:
            raise ValueError("plan_cache_size must be non-negative")
        self.topology = topology
        self.backend = backend
        self.catalog_path = (
            None if catalog_path is None else Path(catalog_path)
        )
        if self.catalog_path is not None and self.catalog_path.exists():
            offline_catalog = OfflineBinomialCatalog.load(self.catalog_path)
        else:
            offline_catalog = OfflineBinomialCatalog()
        self.offline_catalog = offline_catalog
        self.scheduler = PotentialGreedyScheduler(
            topology,
            catalog=CandidateTreeCatalog(
                offline_catalog=self.offline_catalog,
            ),
            max_doublings=max_doublings,
        )
        self.require_prebuilt_catalog = require_prebuilt_catalog
        self.plan_cache_size = plan_cache_size
        self._plan_cache: OrderedDict[tuple[Any, ...], RelayMoEPlan] = (
            OrderedDict()
        )

    def prebuild_catalog(
        self,
        traffic: Iterable[TrafficClass],
        *,
        persist: bool = True,
    ) -> int:
        """Offline phase: materialize destination-distribution skeletons."""

        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        keys = tuple(
            dict.fromkeys(
                self._catalog_key(flow)
                for flow in flows
            )
        )
        self.offline_catalog.prebuild(keys)
        if persist and self.catalog_path is not None:
            self.offline_catalog.save(self.catalog_path)
        return len(keys)

    def save_catalog(self, path: str | Path | None = None) -> None:
        target = self.catalog_path if path is None else Path(path)
        if target is None:
            raise ValueError("save_catalog needs a path")
        self.offline_catalog.save(target)

    def plan(self, traffic: Iterable[TrafficClass]) -> RelayMoEPlan:
        """Online phases: exact catalog selection followed by DAG construction."""

        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        signature = tuple(
            (
                flow.name,
                flow.source,
                flow.destinations,
                flow.volume_bytes,
            )
            for flow in flows
        )
        cached = self._plan_cache.get(signature)
        if cached is not None:
            self._plan_cache.move_to_end(signature)
            return replace(cached, planning_time_us=0.0, cache_hit=True)

        if self.require_prebuilt_catalog:
            missing = [
                key
                for key in dict.fromkeys(
                    self._catalog_key(flow) for flow in flows
                )
                if not self.offline_catalog.contains(*key)
            ]
            if missing:
                raise RuntimeError(
                    "online planning requested a catalog key that was not "
                    "materialized offline; call prebuild_catalog first"
                )

        started_ns = perf_counter_ns()
        if flows:
            selection = self.scheduler.select(flows)
            dispatch_actions = self.scheduler.build_dispatch_actions(
                selection,
                flows,
            )
            combine_actions = self.scheduler.build_combine_actions(
                selection,
                dispatch_actions,
            )
        else:
            selection = SelectionResult(
                selected_trees={},
                edge_load_us={},
                theta_us=0.0,
                doubling_rounds=0,
                candidates_evaluated=0,
            )
            dispatch_actions = ()
            combine_actions = ()
        dispatch = ExecutionDAG("dispatch", dispatch_actions)
        combine = ExecutionDAG("combine", combine_actions)
        dispatch.validate()
        combine.validate()
        planning_time_us = (perf_counter_ns() - started_ns) / 1_000.0
        plan = RelayMoEPlan(
            traffic_classes=flows,
            selection=selection,
            dispatch=dispatch,
            combine=combine,
            useful_bytes=sum(
                len(flow.destinations) * flow.volume_bytes
                for flow in flows
            ),
            planning_time_us=planning_time_us,
        )
        if self.plan_cache_size > 0:
            self._plan_cache[signature] = plan
            self._plan_cache.move_to_end(signature)
            while len(self._plan_cache) > self.plan_cache_size:
                self._plan_cache.popitem(last=False)
        return plan

    def plan_from_token_routing(
        self,
        routing: TokenRouting,
        *,
        source_indices: tuple[int, ...] | None = None,
    ) -> RelayMoEPlan:
        started_ns = perf_counter_ns()
        if tuple(self.topology.gpus) != routing.gpu_names:
            raise ValueError("routing GPU order does not match topology")
        flows = routing.group_traffic_classes(
            source_indices=source_indices,
        )
        plan = self.plan(flows)
        flow_name_by_key = {
            (flow.source, flow.destinations): flow.name
            for flow in flows
        }
        gpu_index = {
            gpu_name: index
            for index, gpu_name in enumerate(routing.gpu_names)
        }
        expert_gpu_bits = [
            1 << gpu_index[gpu_name]
            for gpu_name in routing.expert_to_gpu
        ]
        sources = (
            tuple(range(routing.ep_size))
            if source_indices is None
            else source_indices
        )
        grouped: dict[
            str,
            list[tuple[int, tuple[int, ...]]],
        ] = defaultdict(list)
        for source_index in sources:
            source = routing.gpu_names[source_index]
            source_bit = 1 << source_index
            for token_index, route_mask in enumerate(
                routing.route_masks[source_index]
            ):
                experts: list[int] = []
                destination_mask = 0
                remaining = route_mask
                while remaining:
                    least_bit = remaining & -remaining
                    expert = least_bit.bit_length() - 1
                    experts.append(expert)
                    destination_mask |= expert_gpu_bits[expert]
                    remaining ^= least_bit
                destination_mask &= ~source_bit
                if not destination_mask:
                    continue
                destinations = tuple(
                    routing.gpu_names[index]
                    for index in range(routing.ep_size)
                    if destination_mask & (1 << index)
                )
                traffic_name = flow_name_by_key[(source, destinations)]
                grouped[traffic_name].append(
                    (token_index, tuple(experts))
                )
        layouts = {
            flow.name: TrafficClassLayout(
                traffic_name=flow.name,
                source=flow.source,
                token_indices=tuple(
                    token_index
                    for token_index, _ in grouped[flow.name]
                ),
                expert_indices=tuple(
                    experts for _, experts in grouped[flow.name]
                ),
            )
            for flow in flows
        }
        return replace(
            plan,
            traffic_layouts=layouts,
            planning_time_us=(perf_counter_ns() - started_ns) / 1_000.0,
        )

    def plan_from_gating(
        self,
        expert_indices_by_gpu: Sequence[
            Sequence[Sequence[int]]
        ],
        *,
        expert_to_gpu: Sequence[str],
        activation_bytes: int,
        expert_weights_by_gpu: Sequence[
            Sequence[Sequence[float]]
        ]
        | None = None,
    ) -> RelayMoEPlan:
        """Adapt post-gating Top-k expert IDs to RelayMoE traffic classes.

        This dependency-free adapter is suitable for integration tests.  A
        production GPU backend should aggregate ``(source, destination-mask)``
        counts on device and call :meth:`plan` with the compact classes to
        avoid synchronizing a large gate tensor through Python.
        """

        started_ns = perf_counter_ns()
        if activation_bytes <= 0:
            raise ValueError("activation_bytes must be positive")
        gpu_names = tuple(self.topology.gpus)
        if len(expert_indices_by_gpu) != len(gpu_names):
            raise ValueError(
                "expert_indices_by_gpu must contain one token batch per GPU"
            )
        if (
            expert_weights_by_gpu is not None
            and len(expert_weights_by_gpu) != len(gpu_names)
        ):
            raise ValueError(
                "expert_weights_by_gpu must contain one token batch per GPU"
            )
        placement = tuple(expert_to_gpu)
        if not placement:
            raise ValueError("expert_to_gpu cannot be empty")
        unknown = set(placement) - self.topology.gpus.keys()
        if unknown:
            raise ValueError(
                f"expert placement references unknown GPUs {sorted(unknown)}"
            )
        gpu_order = {gpu: index for index, gpu in enumerate(gpu_names)}

        per_source: list[
            tuple[
                str,
                list[
                    tuple[
                        tuple[str, ...],
                        list[
                            tuple[
                                int,
                                tuple[int, ...],
                                tuple[float, ...] | None,
                            ]
                        ],
                    ]
                ],
            ]
        ] = []
        for source_index, token_routes in enumerate(expert_indices_by_gpu):
            source = gpu_names[source_index]
            token_weights = (
                None
                if expert_weights_by_gpu is None
                else expert_weights_by_gpu[source_index]
            )
            if token_weights is not None and len(token_weights) != len(
                token_routes
            ):
                raise ValueError(
                    f"expert weights do not align on source {source}"
                )
            grouped_tokens: dict[
                tuple[str, ...],
                list[
                    tuple[
                        int,
                        tuple[int, ...],
                        tuple[float, ...] | None,
                    ]
                ],
            ] = defaultdict(list)
            for token_index, token_route in enumerate(token_routes):
                destinations = set()
                experts = tuple(int(raw_expert) for raw_expert in token_route)
                weights = (
                    None
                    if token_weights is None
                    else tuple(
                        float(weight)
                        for weight in token_weights[token_index]
                    )
                )
                if weights is not None and len(weights) != len(experts):
                    raise ValueError(
                        f"expert weights do not align for {source} "
                        f"token {token_index}"
                    )
                for expert in experts:
                    if expert < 0 or expert >= len(placement):
                        raise ValueError(
                            f"expert index {expert} is outside placement"
                        )
                    target = placement[expert]
                    if target != source:
                        destinations.add(target)
                if destinations:
                    key = tuple(
                        sorted(destinations, key=gpu_order.__getitem__)
                    )
                    grouped_tokens[key].append(
                        (token_index, experts, weights)
                    )
            per_source.append(
                (
                    source,
                    sorted(
                        grouped_tokens.items(),
                        key=lambda item: (-len(item[1]), item[0]),
                    ),
                )
            )

        interleaved: list[
            tuple[
                str,
                tuple[str, ...],
                list[
                    tuple[
                        int,
                        tuple[int, ...],
                        tuple[float, ...] | None,
                    ]
                ],
            ]
        ] = []
        max_classes = max(
            (len(items) for _, items in per_source),
            default=0,
        )
        for position in range(max_classes):
            for source, items in per_source:
                if position < len(items):
                    destinations, token_entries = items[position]
                    interleaved.append(
                        (source, destinations, token_entries)
                    )
        interleaved.sort(key=lambda item: -len(item[2]))
        flows = tuple(
            TrafficClass(
                name=f"tc{index:06d}",
                source=source,
                destinations=destinations,
                volume_bytes=len(token_entries) * activation_bytes,
            )
            for index, (source, destinations, token_entries) in enumerate(
                interleaved
            )
        )
        plan = self.plan(flows)
        layouts = {}
        for flow, (_, _, token_entries) in zip(flows, interleaved):
            weights = tuple(
                entry[2] for entry in token_entries
            )
            layouts[flow.name] = TrafficClassLayout(
                traffic_name=flow.name,
                source=flow.source,
                token_indices=tuple(entry[0] for entry in token_entries),
                expert_indices=tuple(entry[1] for entry in token_entries),
                expert_weights=(
                    None
                    if all(weight is None for weight in weights)
                    else tuple(
                        () if weight is None else weight
                        for weight in weights
                    )
                ),
            )
        return replace(
            plan,
            traffic_layouts=layouts,
            planning_time_us=(perf_counter_ns() - started_ns) / 1_000.0,
        )

    def simulate(
        self,
        plan: RelayMoEPlan,
        *,
        launch_overhead_us: float = 1.5,
        include_path_latency: bool = True,
        relay_overhead_us: float = 0.35,
        reduction_launch_overhead_us: float = 0.2,
        reduction_bandwidth_GBps: float = 200.0,
    ) -> RelayMoESimulationResult:
        dispatch = self.scheduler.simulate_event_driven_phase(
            plan.selection,
            plan.dispatch.actions,
            phase="dispatch",
            useful_bytes=plan.useful_bytes,
            launch_overhead_us=launch_overhead_us,
            include_path_latency=include_path_latency,
            relay_overhead_us=relay_overhead_us,
            reduction_launch_overhead_us=reduction_launch_overhead_us,
            reduction_bandwidth_GBps=reduction_bandwidth_GBps,
        )
        combine = self.scheduler.simulate_event_driven_phase(
            plan.selection,
            plan.combine.actions,
            phase="combine",
            useful_bytes=plan.useful_bytes,
            launch_overhead_us=launch_overhead_us,
            include_path_latency=include_path_latency,
            relay_overhead_us=relay_overhead_us,
            reduction_launch_overhead_us=reduction_launch_overhead_us,
            reduction_bandwidth_GBps=reduction_bandwidth_GBps,
        )
        return RelayMoESimulationResult(dispatch=dispatch, combine=combine)

    def dispatch(self, plan: RelayMoEPlan, hidden_states: Any) -> Any:
        return self._execute(plan, plan.dispatch, hidden_states)

    def combine(self, plan: RelayMoEPlan, expert_outputs: Any) -> Any:
        return self._execute(plan, plan.combine, expert_outputs)

    def _execute(
        self,
        plan: RelayMoEPlan,
        phase: ExecutionDAG,
        tensors: Any,
    ) -> Any:
        if self.backend is None:
            raise RuntimeError(
                "RelayMoE execution needs a RelayMoEBackend; use simulate() "
                "for the dependency-free performance model"
            )
        return self.backend.execute(
            plan=plan,
            phase=phase,
            tensors=tensors,
        )

    def _catalog_key(
        self,
        flow: TrafficClass,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        switches = self.topology.switches()
        switch_index = {switch: idx for idx, switch in enumerate(switches)}
        source_gpu = self.topology.gpus[flow.source]
        source_switch_index = switch_index[source_gpu.switch]
        by_switch: dict[str, int] = defaultdict(int)
        for destination in flow.destinations:
            by_switch[self.topology.gpus[destination].switch] += 1

        local_other = sorted(
            (
                switch
                for switch in switches
                if self.topology.switch_node(switch) == source_gpu.node
                and switch != source_gpu.switch
            ),
            key=lambda switch: (
                -by_switch[switch],
                (switch_index[switch] - source_switch_index) % len(switches),
            ),
        )
        remote = sorted(
            (
                switch
                for switch in switches
                if self.topology.switch_node(switch) != source_gpu.node
            ),
            key=lambda switch: (
                -by_switch[switch],
                (switch_index[switch] - source_switch_index) % len(switches),
            ),
        )
        canonical_switches = (
            source_gpu.switch,
            *local_other,
            *remote,
        )
        node_order = {source_gpu.node: 0}
        for switch in remote:
            node = self.topology.switch_node(switch)
            if node not in node_order:
                node_order[node] = len(node_order)
        canonical_nodes = tuple(
            node_order[self.topology.switch_node(switch)]
            for switch in canonical_switches
        )
        distribution = tuple(
            by_switch[switch] for switch in canonical_switches
        )
        nics_per_switch = max(
            1,
            len(
                self.topology.nic_tx_edges_by_switch.get(
                    source_gpu.switch,
                    (),
                )
            ),
        )
        return distribution, canonical_nodes, nics_per_switch
