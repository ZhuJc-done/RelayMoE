from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .model import HierarchicalTopology
from .scenarios import TokenRouting


@dataclass(frozen=True)
class TemplateArc:
    src: str
    dst: str
    expert_mask: int
    parent_arc: int | None


@dataclass(frozen=True)
class RouteTemplate:
    root: str
    arcs: tuple[TemplateArc, ...]


@dataclass(frozen=True)
class FluidFlow:
    flow_id: int
    name: str
    physical_edges: tuple[str, ...]
    size_bytes: int
    dependencies: tuple[int, ...]
    path_latency_us: float


@dataclass(frozen=True)
class TokenPhaseResult:
    phase: str
    total_time_us: float
    useful_bytes: int
    physical_bytes: int
    flow_count: int
    edge_bytes: Mapping[str, int]
    edge_peak_concurrency: Mapping[str, int]
    scheduling_steps: int = 1

    @property
    def effective_algobw_GBps(self) -> float:
        if self.total_time_us == 0:
            return 0.0
        return self.useful_bytes / self.total_time_us / 1_000.0

    @property
    def max_edge_concurrency(self) -> int:
        return max(self.edge_peak_concurrency.values(), default=0)

    @property
    def max_gpu_send_concurrency(self) -> int:
        return max(
            (
                concurrency
                for edge, concurrency in self.edge_peak_concurrency.items()
                if edge.startswith("gpu:") and edge.endswith(":up")
            ),
            default=0,
        )

    @property
    def max_gpu_receive_concurrency(self) -> int:
        return max(
            (
                concurrency
                for edge, concurrency in self.edge_peak_concurrency.items()
                if edge.startswith("gpu:") and edge.endswith(":down")
            ),
            default=0,
        )


@dataclass(frozen=True)
class TokenStrategyResult:
    strategy: str
    dispatch: TokenPhaseResult
    combine: TokenPhaseResult

    @property
    def total_time_us(self) -> float:
        return self.dispatch.total_time_us + self.combine.total_time_us

    @property
    def total_physical_bytes(self) -> int:
        return self.dispatch.physical_bytes + self.combine.physical_bytes


@dataclass(frozen=True)
class TokenSimulationConfig:
    launch_overhead_us: float = 1.5
    include_path_latency: bool = True
    tree_entries_per_node: int = 2
    pipeline_tokens: bool = True
    nccl_uncoordinated_contention_loss: float = 0.05

    def __post_init__(self) -> None:
        if self.launch_overhead_us < 0:
            raise ValueError("launch_overhead_us must be non-negative")
        if self.tree_entries_per_node < 1:
            raise ValueError("tree_entries_per_node must be positive")
        if not 0 <= self.nccl_uncoordinated_contention_loss < 1:
            raise ValueError(
                "nccl_uncoordinated_contention_loss must be in [0, 1)"
            )


def simulate_token_strategies(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    *,
    config: TokenSimulationConfig | None = None,
    source_indices: Iterable[int] | None = None,
) -> dict[str, TokenStrategyResult]:
    """Compare four routing strategies on one exact per-token routing trace.

    The trace is aggregated only after each token has independently selected
    its experts. One aggregate flow represents one stable logical stream, and
    bandwidth is dynamically redistributed whenever a flow completes.
    """

    simulation_config = config or TokenSimulationConfig()
    sources = tuple(
        range(routing.ep_size)
        if source_indices is None
        else source_indices
    )
    if any(source < 0 or source >= routing.ep_size for source in sources):
        raise ValueError("source index is outside the routing trace")
    if tuple(topology.gpus) != routing.gpu_names:
        raise ValueError("routing GPU order does not match topology")

    useful_bytes = sum(
        (
            route_mask
            & ~routing.gpu_expert_masks[source]
        ).bit_count()
        * routing.activation_bytes
        for source in sources
        for route_mask in routing.route_masks[source]
    )

    direct_templates = {
        source: _direct_template(topology, routing, source)
        for source in sources
    }
    deepep_templates = {
        source: _deepep_template(topology, routing, source)
        for source in sources
    }
    specifications = (
        ("nccl", direct_templates),
        ("collway", direct_templates),
        ("deepep", deepep_templates),
    )

    results: dict[str, TokenStrategyResult] = {}
    for strategy, templates in specifications:
        dispatch_flows = _compile_flows(
            topology,
            routing,
            templates,
            phase="dispatch",
            preserve_dependencies=(
                not simulation_config.pipeline_tokens
            ),
        )
        combine_flows = _compile_flows(
            topology,
            routing,
            templates,
            phase="combine",
            preserve_dependencies=(
                not simulation_config.pipeline_tokens
            ),
        )
        if strategy == "collway":
            dispatch = _simulate_collway_phase(
                topology,
                dispatch_flows,
                phase="dispatch",
                useful_bytes=useful_bytes,
                config=simulation_config,
            )
            combine = _simulate_collway_phase(
                topology,
                combine_flows,
                phase="combine",
                useful_bytes=useful_bytes,
                config=simulation_config,
            )
        else:
            dispatch = _simulate_phase(
                topology,
                dispatch_flows,
                phase="dispatch",
                useful_bytes=useful_bytes,
                exclusive_edges=frozenset(),
                config=simulation_config,
                uncoordinated_contention_loss=(
                    simulation_config.nccl_uncoordinated_contention_loss
                    if strategy == "nccl"
                    else 0.0
                ),
            )
        if simulation_config.pipeline_tokens and strategy != "collway":
            combine = _summarize_mirrored_phase(
                topology,
                combine_flows,
                phase="combine",
                useful_bytes=useful_bytes,
                total_time_us=dispatch.total_time_us,
                peak_concurrency=dispatch.max_edge_concurrency,
                scheduling_steps=dispatch.scheduling_steps,
            )
        elif strategy != "collway":
            combine = _simulate_phase(
                topology,
                combine_flows,
                phase="combine",
                useful_bytes=useful_bytes,
                exclusive_edges=frozenset(),
                config=simulation_config,
                uncoordinated_contention_loss=(
                    simulation_config.nccl_uncoordinated_contention_loss
                    if strategy == "nccl"
                    else 0.0
                ),
            )
        results[strategy] = TokenStrategyResult(
            strategy=strategy,
            dispatch=dispatch,
            combine=combine,
        )
    results["relaymoe"] = _simulate_selective_token_trees(
        topology,
        routing,
        sources=sources,
        useful_bytes=useful_bytes,
        config=simulation_config,
    )
    return results


def _direct_template(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    source_index: int,
) -> RouteTemplate:
    root = routing.gpu_names[source_index]
    parents = {
        gpu_name: root
        for gpu_name in topology.gpus
        if gpu_name != root
    }
    return _template_from_parents(topology, routing, root, parents)


def _deepep_template(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    source_index: int,
) -> RouteTemplate:
    root = routing.gpu_names[source_index]
    source_gpu = topology.gpus[root]
    source_node_gpus = _gpus_on_node(topology, source_gpu.node)
    local_index = source_node_gpus.index(root)
    parents: dict[str, str] = {}

    for node in _nodes(topology):
        node_gpus = _gpus_on_node(topology, node)
        if node == source_gpu.node:
            for gpu_name in node_gpus:
                if gpu_name != root:
                    parents[gpu_name] = root
            continue
        ingress = node_gpus[local_index % len(node_gpus)]
        parents[ingress] = root
        for gpu_name in node_gpus:
            if gpu_name != ingress:
                parents[gpu_name] = ingress
    return _template_from_parents(topology, routing, root, parents)


def _simulate_selective_token_trees(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    *,
    sources: tuple[int, ...],
    useful_bytes: int,
    config: TokenSimulationConfig,
) -> TokenStrategyResult:
    rounds = _aggregate_selective_tree_rounds(
        topology,
        routing,
        sources=sources,
        entries_per_node=config.tree_entries_per_node,
    )
    simulate_rounds = (
        _simulate_pipelined_tree_rounds
        if config.pipeline_tokens
        else _simulate_tree_rounds
    )
    dispatch = simulate_rounds(
        topology,
        rounds,
        phase="dispatch",
        useful_bytes=useful_bytes,
        config=config,
    )
    if config.pipeline_tokens:
        combine = _simulate_pipelined_tree_rounds(
            topology,
            rounds,
            phase="combine",
            useful_bytes=useful_bytes,
            config=config,
            mirrored_time_us=dispatch.total_time_us,
            mirrored_peak_concurrency=(
                dispatch.max_edge_concurrency
            ),
        )
    else:
        combine = simulate_rounds(
            topology,
            rounds,
            phase="combine",
            useful_bytes=useful_bytes,
            config=config,
        )
    return TokenStrategyResult(
        strategy="relaymoe",
        dispatch=dispatch,
        combine=combine,
    )


def _aggregate_selective_tree_rounds(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    *,
    sources: tuple[int, ...],
    entries_per_node: int,
) -> tuple[dict[tuple[str, str], int], ...]:
    gpu_index = {
        gpu_name: index
        for index, gpu_name in enumerate(routing.gpu_names)
    }
    expert_gpu_indices = tuple(
        gpu_index[gpu_name]
        for gpu_name in routing.expert_to_gpu
    )
    paths = {
        (src, dst): topology.route(src, dst)
        for src in topology.gpus
        for dst in topology.gpus
        if src != dst
    }
    rounds: list[dict[tuple[str, str], int]] = []

    for source in sources:
        root = routing.gpu_names[source]
        for token_index, route_mask in enumerate(
            routing.route_masks[source]
        ):
            destinations: set[str] = set()
            remaining = route_mask
            while remaining:
                least_bit = remaining & -remaining
                expert = least_bit.bit_length() - 1
                destination_index = expert_gpu_indices[expert]
                if destination_index != source:
                    destinations.add(routing.gpu_names[destination_index])
                remaining ^= least_bit
            if not destinations:
                continue

            stages = _selective_tree_stages(
                topology,
                root,
                destinations,
                entries_per_node=entries_per_node,
                paths=paths,
                relay_variant=token_index + source * routing.tokens_per_gpu,
            )
            while len(rounds) < len(stages):
                rounds.append(defaultdict(int))
            for stage_index, stage in enumerate(stages):
                for arc in stage:
                    rounds[stage_index][arc] += routing.activation_bytes
    return tuple(dict(stage) for stage in rounds)


def _selective_tree_stages(
    topology: HierarchicalTopology,
    root: str,
    destinations: set[str],
    *,
    entries_per_node: int,
    paths: Mapping[tuple[str, str], tuple[str, ...]],
    relay_variant: int,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    parents = _selective_tree_parents(
        topology,
        root,
        destinations,
        entries_per_node,
        relay_variant,
    )
    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in parents.items():
        children[parent].append(child)

    remaining_depth: dict[str, int] = {}

    def depth(gpu_name: str) -> int:
        if gpu_name in remaining_depth:
            return remaining_depth[gpu_name]
        remaining_depth[gpu_name] = max(
            (depth(child) + 1 for child in children.get(gpu_name, ())),
            default=0,
        )
        return remaining_depth[gpu_name]

    depth(root)
    pending = set(parents)
    completed: set[str] = {root}
    stages: list[tuple[tuple[str, str], ...]] = []
    while pending:
        ready = sorted(
            (
                child
                for child in pending
                if parents[child] in completed
            ),
            key=lambda child: (
                -remaining_depth[child],
                parents[child],
                child,
            ),
        )
        occupied: set[str] = set()
        selected: list[str] = []
        for child in ready:
            arc = (parents[child], child)
            physical_edges = paths[arc]
            if occupied.isdisjoint(physical_edges):
                selected.append(child)
                occupied.update(physical_edges)
        if not selected:
            raise RuntimeError("selective token tree packing deadlocked")
        stages.append(
            tuple((parents[child], child) for child in selected)
        )
        completed.update(selected)
        pending.difference_update(selected)
    return tuple(stages)


def _selective_tree_parents(
    topology: HierarchicalTopology,
    root: str,
    destinations: set[str],
    entries_per_node: int,
    relay_variant: int,
) -> dict[str, str]:
    source_node = topology.gpus[root].node
    source_switch = topology.gpus[root].switch
    parents: dict[str, str] = {}

    for node in _nodes(topology):
        by_switch: dict[str, list[str]] = defaultdict(list)
        for destination in destinations:
            gpu = topology.gpus[destination]
            if gpu.node == node:
                by_switch[gpu.switch].append(destination)
        if not by_switch:
            continue
        for switch_destinations in by_switch.values():
            switch_destinations.sort()

        active_switches = [
            switch
            for switch in _switches_on_node(topology, node)
            if switch in by_switch
        ]
        if node == source_node and source_switch in active_switches:
            active_switches = [
                source_switch,
                *[
                    switch
                    for switch in active_switches
                    if switch != source_switch
                ],
            ]
        elif len(active_switches) > 1:
            rotation = relay_variant % len(active_switches)
            active_switches = (
                active_switches[rotation:]
                + active_switches[:rotation]
            )
        groups = _balanced_groups(
            active_switches,
            min(entries_per_node, len(active_switches)),
        )

        for group in groups:
            entry_switch = group[0]
            if node == source_node and entry_switch == source_switch:
                entry = root
            else:
                entry_candidates = by_switch[entry_switch]
                entry = entry_candidates[
                    (relay_variant + len(entry_switch))
                    % len(entry_candidates)
                ]
                parents[entry] = root

            for switch_index, switch in enumerate(group):
                if switch_index == 0:
                    relay = entry
                else:
                    relay_candidates = by_switch[switch]
                    relay = relay_candidates[
                        (relay_variant + len(switch))
                        % len(relay_candidates)
                    ]
                    parents[relay] = entry
                for destination in by_switch[switch]:
                    if destination != relay:
                        parents[destination] = relay

    if set(parents) != destinations:
        raise RuntimeError("selective tree does not cover token destinations")
    return parents


def _simulate_tree_rounds(
    topology: HierarchicalTopology,
    rounds: tuple[dict[tuple[str, str], int], ...],
    *,
    phase: str,
    useful_bytes: int,
    config: TokenSimulationConfig,
) -> TokenPhaseResult:
    ordered_rounds = (
        rounds
        if phase == "dispatch"
        else tuple(reversed(rounds))
    )
    edge_bytes: dict[str, int] = defaultdict(int)
    edge_peak_concurrency: dict[str, int] = defaultdict(int)
    total_time_us = config.launch_overhead_us
    physical_bytes = 0
    flow_count = 0
    round_config = TokenSimulationConfig(
        launch_overhead_us=0.0,
        include_path_latency=config.include_path_latency,
        tree_entries_per_node=config.tree_entries_per_node,
    )

    for round_bytes in ordered_rounds:
        flows: list[FluidFlow] = []
        for flow_id, ((src, dst), size_bytes) in enumerate(
            sorted(round_bytes.items())
        ):
            if phase == "combine":
                src, dst = dst, src
            physical_edges = topology.route(src, dst)
            flows.append(
                FluidFlow(
                    flow_id=flow_id,
                    name=f"{phase}:round:{src}->{dst}",
                    physical_edges=physical_edges,
                    size_bytes=size_bytes,
                    dependencies=(),
                    path_latency_us=topology.route_latency_us(
                        physical_edges
                    ),
                )
            )
        round_result = _simulate_phase(
            topology,
            tuple(flows),
            phase=phase,
            useful_bytes=0,
            exclusive_edges=None,
            config=round_config,
        )
        total_time_us += round_result.total_time_us
        physical_bytes += round_result.physical_bytes
        flow_count += round_result.flow_count
        for edge_name, size_bytes in round_result.edge_bytes.items():
            edge_bytes[edge_name] += size_bytes
        for edge_name, concurrency in (
            round_result.edge_peak_concurrency.items()
        ):
            edge_peak_concurrency[edge_name] = max(
                edge_peak_concurrency[edge_name],
                concurrency,
            )

    return TokenPhaseResult(
        phase=phase,
        total_time_us=total_time_us,
        useful_bytes=useful_bytes,
        physical_bytes=physical_bytes,
        flow_count=flow_count,
        edge_bytes=dict(edge_bytes),
        edge_peak_concurrency=dict(edge_peak_concurrency),
    )


def _relaymoe_template(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    source_index: int,
    entries_per_node: int,
) -> RouteTemplate:
    root = routing.gpu_names[source_index]
    source_node = topology.gpus[root].node
    parents: dict[str, str] = {}

    for node in _nodes(topology):
        switches = _switches_on_node(topology, node)
        if node == source_node:
            root_switch = topology.gpus[root].switch
            switches = (
                [root_switch]
                + [switch for switch in switches if switch != root_switch]
            )
        group_count = min(entries_per_node, len(switches))
        groups = _balanced_groups(switches, group_count)

        for group_index, switch_group in enumerate(groups):
            entry_switch = switch_group[0]
            if node == source_node and group_index == 0:
                entry = root
            else:
                entry = _gpus_on_switch(topology, entry_switch)[0]
                parents[entry] = root

            for switch_index, switch in enumerate(switch_group):
                if switch_index == 0:
                    relay = entry
                else:
                    relay = _gpus_on_switch(topology, switch)[0]
                    parents[relay] = entry
                for gpu_name in _gpus_on_switch(topology, switch):
                    if gpu_name in (root, relay):
                        continue
                    parents[gpu_name] = relay

    return _template_from_parents(topology, routing, root, parents)


def _template_from_parents(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    root: str,
    parents: Mapping[str, str],
) -> RouteTemplate:
    expected = set(topology.gpus) - {root}
    if set(parents) != expected:
        missing = sorted(expected - set(parents))
        extra = sorted(set(parents) - expected)
        raise ValueError(
            f"template parent map mismatch: missing={missing}, extra={extra}"
        )

    children: dict[str, list[str]] = defaultdict(list)
    for child, parent in parents.items():
        children[parent].append(child)

    gpu_index = {
        gpu_name: index
        for index, gpu_name in enumerate(routing.gpu_names)
    }
    subtree_masks: dict[str, int] = {}

    def subtree_mask(gpu_name: str) -> int:
        if gpu_name in subtree_masks:
            return subtree_masks[gpu_name]
        mask = routing.gpu_expert_masks[gpu_index[gpu_name]]
        for child in children.get(gpu_name, ()):
            mask |= subtree_mask(child)
        subtree_masks[gpu_name] = mask
        return mask

    depths = {root: 0}
    pending = set(parents)
    while pending:
        progressed = False
        for child in tuple(pending):
            parent = parents[child]
            if parent in depths:
                depths[child] = depths[parent] + 1
                pending.remove(child)
                progressed = True
        if not progressed:
            raise ValueError("template parent map contains a cycle")

    ordered_nodes = sorted(
        parents,
        key=lambda gpu_name: (depths[gpu_name], gpu_index[gpu_name]),
    )
    arc_by_destination: dict[str, int] = {}
    arcs: list[TemplateArc] = []
    for destination in ordered_nodes:
        parent = parents[destination]
        parent_arc = (
            arc_by_destination[parent]
            if parent != root
            else None
        )
        arc_by_destination[destination] = len(arcs)
        arcs.append(
            TemplateArc(
                src=parent,
                dst=destination,
                expert_mask=subtree_mask(destination),
                parent_arc=parent_arc,
            )
        )
    return RouteTemplate(root=root, arcs=tuple(arcs))


def _compile_flows(
    topology: HierarchicalTopology,
    routing: TokenRouting,
    templates: Mapping[int, RouteTemplate],
    *,
    phase: str,
    preserve_dependencies: bool = True,
) -> tuple[FluidFlow, ...]:
    if phase not in {"dispatch", "combine"}:
        raise ValueError(f"unknown phase {phase}")

    flows: list[FluidFlow] = []
    for source_index, template in templates.items():
        sizes = [
            routing.tokens_selecting(source_index, arc.expert_mask)
            * routing.activation_bytes
            for arc in template.arcs
        ]
        flow_id_by_arc: dict[int, int] = {}
        for arc_index, size_bytes in enumerate(sizes):
            if size_bytes:
                flow_id_by_arc[arc_index] = len(flows) + len(flow_id_by_arc)

        child_arcs: dict[int, list[int]] = defaultdict(list)
        for arc_index, arc in enumerate(template.arcs):
            if arc.parent_arc is not None:
                child_arcs[arc.parent_arc].append(arc_index)

        template_flows: list[FluidFlow] = []
        for arc_index, arc in enumerate(template.arcs):
            if not sizes[arc_index]:
                continue
            flow_id = flow_id_by_arc[arc_index]
            if phase == "dispatch":
                src, dst = arc.src, arc.dst
                dependencies = (
                    (flow_id_by_arc[arc.parent_arc],)
                    if (
                        preserve_dependencies
                        and arc.parent_arc is not None
                    )
                    else ()
                )
            else:
                src, dst = arc.dst, arc.src
                dependencies = (
                    tuple(
                        flow_id_by_arc[child]
                        for child in child_arcs.get(arc_index, ())
                        if child in flow_id_by_arc
                    )
                    if preserve_dependencies
                    else ()
                )
            physical_edges = topology.route(src, dst)
            template_flows.append(
                FluidFlow(
                    flow_id=flow_id,
                    name=(
                        f"{phase}:s{source_index}:"
                        f"{src}->{dst}"
                    ),
                    physical_edges=physical_edges,
                    size_bytes=sizes[arc_index],
                    dependencies=dependencies,
                    path_latency_us=topology.route_latency_us(
                        physical_edges
                    ),
                )
            )
        flows.extend(template_flows)

    # IDs were assigned per template using the current global flow offset.
    for expected_id, flow in enumerate(flows):
        if flow.flow_id != expected_id:
            raise RuntimeError("compiled flow IDs are not contiguous")
    return tuple(flows)


def _simulate_collway_phase(
    topology: HierarchicalTopology,
    flows: tuple[FluidFlow, ...],
    *,
    phase: str,
    useful_bytes: int,
    config: TokenSimulationConfig,
) -> TokenPhaseResult:
    """Run direct flows as topology-aware one-to-one matching steps.

    Matching steps are persistent-kernel schedule slots, not separate kernel
    launches. Consequently, launch overhead and end-to-end path latency are
    charged once for the phase, while serialization time is accumulated over
    the slots.
    """

    if not flows:
        return TokenPhaseResult(
            phase=phase,
            total_time_us=0.0,
            useful_bytes=useful_bytes,
            physical_bytes=0,
            flow_count=0,
            edge_bytes={},
            edge_peak_concurrency={},
            scheduling_steps=0,
        )

    stages = _compile_collway_matching_stages(topology, flows)
    stage_config = TokenSimulationConfig(
        launch_overhead_us=0.0,
        include_path_latency=False,
        tree_entries_per_node=config.tree_entries_per_node,
        pipeline_tokens=config.pipeline_tokens,
        nccl_uncoordinated_contention_loss=0.0,
    )
    total_time_us = config.launch_overhead_us
    if config.include_path_latency:
        total_time_us += max(flow.path_latency_us for flow in flows)

    edge_bytes: dict[str, int] = defaultdict(int)
    edge_peak_concurrency: dict[str, int] = defaultdict(int)
    for flow in flows:
        for edge_name in flow.physical_edges:
            edge_bytes[edge_name] += flow.size_bytes

    for stage in stages:
        reindexed = tuple(
            FluidFlow(
                flow_id=flow_id,
                name=flow.name,
                physical_edges=flow.physical_edges,
                size_bytes=flow.size_bytes,
                dependencies=(),
                path_latency_us=0.0,
            )
            for flow_id, flow in enumerate(stage)
        )
        result = _simulate_phase(
            topology,
            reindexed,
            phase=phase,
            useful_bytes=0,
            exclusive_edges=frozenset(),
            config=stage_config,
        )
        total_time_us += result.total_time_us
        for edge_name, concurrency in result.edge_peak_concurrency.items():
            edge_peak_concurrency[edge_name] = max(
                edge_peak_concurrency[edge_name],
                concurrency,
            )

    return TokenPhaseResult(
        phase=phase,
        total_time_us=total_time_us,
        useful_bytes=useful_bytes,
        physical_bytes=sum(edge_bytes.values()),
        flow_count=len(flows),
        edge_bytes=dict(edge_bytes),
        edge_peak_concurrency=dict(edge_peak_concurrency),
        scheduling_steps=len(stages),
    )


def _compile_collway_matching_stages(
    topology: HierarchicalTopology,
    flows: tuple[FluidFlow, ...],
) -> tuple[tuple[FluidFlow, ...], ...]:
    """Factor direct traffic into topology-aware endpoint matchings.

    For power-of-two EP sizes, all bit-coordinate permutations are considered
    as compact Latin-square factorizations. The candidate with the smallest
    sum of per-step physical-link load lower bounds is selected. Other EP
    sizes use the analogous modular affine factorizations.
    """

    gpu_names = tuple(topology.gpus)
    gpu_index = {
        gpu_name: index for index, gpu_name in enumerate(gpu_names)
    }
    source_by_edge = {
        edge_name: gpu_name
        for gpu_name, edge_name in topology.gpu_up_edges.items()
    }
    destination_by_edge = {
        edge_name: gpu_name
        for gpu_name, edge_name in topology.gpu_down_edges.items()
    }
    pair_flows: dict[tuple[int, int], FluidFlow] = {}
    for flow in flows:
        if not flow.physical_edges:
            continue
        try:
            source = source_by_edge[flow.physical_edges[0]]
            destination = destination_by_edge[flow.physical_edges[-1]]
        except KeyError as exc:
            raise ValueError(
                "CollWay matching requires direct GPU-to-GPU flows"
            ) from exc
        pair = (gpu_index[source], gpu_index[destination])
        if pair in pair_flows:
            raise ValueError(f"duplicate CollWay direct pair: {pair}")
        pair_flows[pair] = flow

    size = len(gpu_names)
    if size & (size - 1) == 0:
        bit_count = size.bit_length() - 1
        candidates = (
            tuple(
                tuple(
                    stage ^ _permute_index_bits(source, permutation)
                    for source in range(size)
                )
                for stage in range(size)
            )
            for permutation in itertools.permutations(range(bit_count))
        )
    else:
        multipliers = tuple(
            multiplier
            for multiplier in range(1, size)
            if math.gcd(multiplier, size) == 1
        )
        candidates = (
            tuple(
                tuple(
                    (multiplier * source + stage) % size
                    for source in range(size)
                )
                for stage in range(size)
            )
            for multiplier in multipliers
        )

    best_key: tuple[float, tuple[tuple[int, ...], ...]] | None = None
    best_mapping: tuple[tuple[int, ...], ...] | None = None
    for mapping in candidates:
        key = (_collway_mapping_cost(topology, pair_flows, mapping), mapping)
        if best_key is None or key < best_key:
            best_key = key
            best_mapping = mapping
    if best_mapping is None:
        raise RuntimeError("CollWay matching planner found no factorization")

    stages = tuple(
        tuple(
            pair_flows[(source, destination)]
            for source, destination in enumerate(destinations)
            if (source, destination) in pair_flows
        )
        for destinations in best_mapping
    )
    nonempty_stages = tuple(stage for stage in stages if stage)
    scheduled = [
        flow.flow_id
        for stage in nonempty_stages
        for flow in stage
    ]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != {
        flow.flow_id for flow in flows
    }:
        raise RuntimeError("CollWay matching factorization lost or repeated a flow")
    return nonempty_stages


def _permute_index_bits(
    value: int,
    permutation: tuple[int, ...],
) -> int:
    return sum(
        ((value >> source_bit) & 1) << destination_bit
        for destination_bit, source_bit in enumerate(permutation)
    )


def _collway_mapping_cost(
    topology: HierarchicalTopology,
    pair_flows: Mapping[tuple[int, int], FluidFlow],
    mapping: tuple[tuple[int, ...], ...],
) -> float:
    total_us = 0.0
    for destinations in mapping:
        edge_loads: dict[str, int] = defaultdict(int)
        for source, destination in enumerate(destinations):
            flow = pair_flows.get((source, destination))
            if flow is None:
                continue
            for edge_name in flow.physical_edges:
                edge_loads[edge_name] += flow.size_bytes
        total_us += max(
            (
                size_bytes / topology.edges[edge_name].bytes_per_us
                for edge_name, size_bytes in edge_loads.items()
            ),
            default=0.0,
        )
    return total_us


def _summarize_mirrored_phase(
    topology: HierarchicalTopology,
    flows: tuple[FluidFlow, ...],
    *,
    phase: str,
    useful_bytes: int,
    total_time_us: float,
    peak_concurrency: int,
    scheduling_steps: int = 1,
) -> TokenPhaseResult:
    edge_bytes: dict[str, int] = defaultdict(int)
    for flow in flows:
        for edge_name in flow.physical_edges:
            edge_bytes[edge_name] += flow.size_bytes
    peak = (
        {next(iter(edge_bytes)): peak_concurrency}
        if edge_bytes
        else {}
    )
    return TokenPhaseResult(
        phase=phase,
        total_time_us=total_time_us,
        useful_bytes=useful_bytes,
        physical_bytes=sum(edge_bytes.values()),
        flow_count=len(flows),
        edge_bytes=dict(edge_bytes),
        edge_peak_concurrency=peak,
        scheduling_steps=scheduling_steps,
    )


def _simulate_phase(
    topology: HierarchicalTopology,
    flows: tuple[FluidFlow, ...],
    *,
    phase: str,
    useful_bytes: int,
    exclusive_edges: frozenset[str] | None,
    config: TokenSimulationConfig,
    uncoordinated_contention_loss: float = 0.0,
) -> TokenPhaseResult:
    if not flows:
        return TokenPhaseResult(
            phase=phase,
            total_time_us=0.0,
            useful_bytes=useful_bytes,
            physical_bytes=0,
            flow_count=0,
            edge_bytes={},
            edge_peak_concurrency={},
        )

    dependents: list[list[int]] = [[] for _ in flows]
    remaining_dependencies = [len(flow.dependencies) for flow in flows]
    for flow in flows:
        for dependency in flow.dependencies:
            dependents[dependency].append(flow.flow_id)
    critical_depth = _critical_depths(dependents)

    ready = {
        flow.flow_id
        for flow in flows
        if not flow.dependencies
    }
    active: dict[int, float] = {}
    arrivals: list[tuple[float, int]] = []
    arrived = 0
    occupied_exclusive: set[str] = set()
    flow_exclusive_resources = [
        (
            set(flow.physical_edges)
            if exclusive_edges is None
            else set(flow.physical_edges).intersection(exclusive_edges)
        )
        for flow in flows
    ]

    edge_bytes: dict[str, int] = defaultdict(int)
    for flow in flows:
        for edge_name in flow.physical_edges:
            edge_bytes[edge_name] += flow.size_bytes
    edge_peak_concurrency: dict[str, int] = defaultdict(int)

    current_time_us = config.launch_overhead_us
    epsilon = 1e-9
    while arrived < len(flows):
        for flow_id in sorted(
            tuple(ready),
            key=lambda item: (
                -critical_depth[item],
                -flows[item].size_bytes,
                item,
            ),
        ):
            resources = flow_exclusive_resources[flow_id]
            if occupied_exclusive.isdisjoint(resources):
                ready.remove(flow_id)
                active[flow_id] = float(flows[flow_id].size_bytes)
                occupied_exclusive.update(resources)

        edge_users: dict[str, list[int]] = defaultdict(list)
        for flow_id in active:
            for edge_name in flows[flow_id].physical_edges:
                edge_users[edge_name].append(flow_id)
        for edge_name, users in edge_users.items():
            edge_peak_concurrency[edge_name] = max(
                edge_peak_concurrency[edge_name],
                len(users),
            )

        rates = _progressive_max_min_rates(
            topology,
            active.keys(),
            flows,
            edge_users,
            edge_capacity_scales={
                edge_name: _uncoordinated_efficiency(
                    len(users),
                    uncoordinated_contention_loss,
                )
                for edge_name, users in edge_users.items()
                if topology.edges[edge_name].kind
                in {
                    "gpu_up",
                    "gpu_down",
                    "pxb",
                    "nic_tx",
                    "nic_rx",
                }
            },
        )
        finish_delta = min(
            (
                active[flow_id] / rates[flow_id]
                for flow_id in active
                if rates[flow_id] > 0
            ),
            default=math.inf,
        )
        arrival_delta = (
            max(0.0, arrivals[0][0] - current_time_us)
            if arrivals
            else math.inf
        )
        delta = min(finish_delta, arrival_delta)
        if math.isinf(delta):
            raise RuntimeError("fluid simulation deadlocked")

        if delta > 0:
            for flow_id, rate in rates.items():
                active[flow_id] = max(
                    0.0,
                    active[flow_id] - rate * delta,
                )
            current_time_us += delta

        finished = [
            flow_id
            for flow_id, remaining in active.items()
            if remaining <= epsilon
        ]
        for flow_id in finished:
            del active[flow_id]
            occupied_exclusive.difference_update(
                flow_exclusive_resources[flow_id]
            )
            latency = (
                flows[flow_id].path_latency_us
                if config.include_path_latency
                else 0.0
            )
            heapq.heappush(
                arrivals,
                (current_time_us + latency, flow_id),
            )

        while arrivals and arrivals[0][0] <= current_time_us + epsilon:
            _, flow_id = heapq.heappop(arrivals)
            arrived += 1
            for dependent in dependents[flow_id]:
                remaining_dependencies[dependent] -= 1
                if remaining_dependencies[dependent] == 0:
                    ready.add(dependent)

    return TokenPhaseResult(
        phase=phase,
        total_time_us=current_time_us,
        useful_bytes=useful_bytes,
        physical_bytes=sum(edge_bytes.values()),
        flow_count=len(flows),
        edge_bytes=dict(edge_bytes),
        edge_peak_concurrency=dict(edge_peak_concurrency),
    )


def _simulate_pipelined_tree_rounds(
    topology: HierarchicalTopology,
    rounds: tuple[dict[tuple[str, str], int], ...],
    *,
    phase: str,
    useful_bytes: int,
    config: TokenSimulationConfig,
    mirrored_time_us: float | None = None,
    mirrored_peak_concurrency: int = 0,
) -> TokenPhaseResult:
    """Fluid limit of fine-grained contention-free token time slicing.

    Independent 1 KiB tokens can pipeline across relay levels. Aggregating all
    bytes on each logical arc and applying max-min sharing is equivalent to
    infinitesimal conflict-free time slices, without charging one launch per
    token or holding a whole multi-token stream behind a global barrier.
    """

    pair_bytes: dict[tuple[str, str], int] = defaultdict(int)
    for round_bytes in rounds:
        for arc, size_bytes in round_bytes.items():
            pair_bytes[arc] += size_bytes

    flows: list[FluidFlow] = []
    for flow_id, ((src, dst), size_bytes) in enumerate(
        sorted(pair_bytes.items())
    ):
        if phase == "combine":
            src, dst = dst, src
        physical_edges = topology.route(src, dst)
        flows.append(
            FluidFlow(
                flow_id=flow_id,
                name=f"{phase}:pipeline:{src}->{dst}",
                physical_edges=physical_edges,
                size_bytes=size_bytes,
                dependencies=(),
                path_latency_us=topology.route_latency_us(physical_edges),
            )
        )
    compiled = tuple(flows)
    if mirrored_time_us is not None:
        return _summarize_mirrored_phase(
            topology,
            compiled,
            phase=phase,
            useful_bytes=useful_bytes,
            total_time_us=mirrored_time_us,
            peak_concurrency=mirrored_peak_concurrency,
        )
    return _simulate_phase(
        topology,
        compiled,
        phase=phase,
        useful_bytes=useful_bytes,
        # Fractional sharing represents fine-grained edge-disjoint time
        # slices; no two packets need to occupy an edge simultaneously.
        exclusive_edges=frozenset(),
        config=config,
    )


def _progressive_max_min_rates(
    topology: HierarchicalTopology,
    active_ids: Iterable[int],
    flows: tuple[FluidFlow, ...],
    edge_users: Mapping[str, list[int]],
    edge_capacity_scales: Mapping[str, float] | None = None,
) -> dict[int, float]:
    active = set(active_ids)
    if not active:
        return {}
    rates = {flow_id: 0.0 for flow_id in active}
    unfrozen = set(active)
    tolerance = 1e-9

    while unfrozen:
        best_increment = math.inf
        edge_increments: dict[str, float] = {}
        for edge_name, users in edge_users.items():
            unfrozen_users = [flow for flow in users if flow in unfrozen]
            if not unfrozen_users:
                continue
            used = sum(rates[flow] for flow in users)
            capacity_scale = (
                edge_capacity_scales.get(edge_name, 1.0)
                if edge_capacity_scales is not None
                else 1.0
            )
            residual = max(
                0.0,
                topology.edges[edge_name].bytes_per_us
                * capacity_scale
                - used,
            )
            increment = residual / len(unfrozen_users)
            edge_increments[edge_name] = increment
            best_increment = min(best_increment, increment)

        if math.isinf(best_increment):
            raise RuntimeError("active flow has no bandwidth-constrained edge")
        for flow_id in unfrozen:
            rates[flow_id] += best_increment

        saturated_edges = {
            edge_name
            for edge_name, increment in edge_increments.items()
            if increment <= best_increment + tolerance
        }
        frozen_now = {
            flow_id
            for edge_name in saturated_edges
            for flow_id in edge_users[edge_name]
            if flow_id in unfrozen
        }
        if not frozen_now:
            raise RuntimeError("max-min allocation made no progress")
        unfrozen.difference_update(frozen_now)
    return rates


def _uncoordinated_efficiency(
    concurrency: int,
    maximum_loss: float,
) -> float:
    if concurrency <= 1 or maximum_loss == 0:
        return 1.0
    return 1.0 - maximum_loss * (1.0 - 1.0 / concurrency)


def _critical_depths(dependents: list[list[int]]) -> list[int]:
    depths = [-1] * len(dependents)

    def visit(flow_id: int) -> int:
        if depths[flow_id] >= 0:
            return depths[flow_id]
        depths[flow_id] = max(
            (visit(child) + 1 for child in dependents[flow_id]),
            default=0,
        )
        return depths[flow_id]

    for flow_id in range(len(dependents)):
        visit(flow_id)
    return depths


def _nodes(topology: HierarchicalTopology) -> list[str]:
    return list(
        dict.fromkeys(gpu.node for gpu in topology.gpus.values())
    )


def _switches_on_node(
    topology: HierarchicalTopology,
    node: str,
) -> list[str]:
    return list(
        dict.fromkeys(
            gpu.switch
            for gpu in topology.gpus.values()
            if gpu.node == node
        )
    )


def _gpus_on_node(
    topology: HierarchicalTopology,
    node: str,
) -> list[str]:
    return [
        gpu.name
        for gpu in topology.gpus.values()
        if gpu.node == node
    ]


def _gpus_on_switch(
    topology: HierarchicalTopology,
    switch: str,
) -> list[str]:
    return [
        gpu.name
        for gpu in topology.gpus.values()
        if gpu.switch == switch
    ]


def _balanced_groups(
    values: list[str],
    group_count: int,
) -> list[list[str]]:
    quotient, remainder = divmod(len(values), group_count)
    groups: list[list[str]] = []
    start = 0
    for group_index in range(group_count):
        size = quotient + (1 if group_index < remainder else 0)
        groups.append(values[start : start + size])
        start += size
    return groups
