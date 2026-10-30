from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .model import (
    HierarchicalTopology,
    LogicalArc,
    RelayTree,
    TrafficClass,
    ensure_traffic_matches_topology,
)
from .scheduler import (
    PotentialGreedyScheduler,
    Schedule,
    SelectionResult,
)


@dataclass(frozen=True)
class BaselinePlan:
    selection: SelectionResult
    dispatch: Schedule
    combine: Schedule


class NCCLBaseline:
    """Unstaged direct point-to-point transfers with full link sharing.

    Multicast is expanded into independent source-to-destination unicasts.
    All ready transfers execute in one dependency round without fan-out,
    fan-in, or fabric conflict avoidance. The simulator accounts for the
    resulting bandwidth sharing on every directed physical edge.
    """

    def __init__(self, topology: HierarchicalTopology) -> None:
        self.topology = topology

    def plan(self, traffic: Iterable[TrafficClass]) -> BaselinePlan:
        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        trees = {
            flow.name: RelayTree(
                name=f"{flow.name}:nccl",
                traffic_name=flow.name,
                root=flow.source,
                destinations=flow.destinations,
                arcs=tuple(
                    sorted(
                        LogicalArc(flow.source, dst)
                        for dst in flow.destinations
                    )
                ),
                kind="nccl-direct",
            )
            for flow in flows
        }
        return _build_fixed_plan(
            self.topology,
            flows,
            trees,
            exclusive_edges=frozenset(),
        )


class CollWayBaseline:
    """Direct pairwise transfers with endpoint-conflict-free stage packing.

    This captures the core CollWay-like behavior used in this prototype:
    multicast is expanded into independent source-to-destination unicasts,
    while GPU uplink/downlink conflicts are split across stages to enforce at
    most one outgoing/incoming transfer per endpoint. Fabric edges can still
    be shared and are subject to bandwidth contention in the simulator.
    """

    def __init__(self, topology: HierarchicalTopology) -> None:
        self.topology = topology

    def plan(self, traffic: Iterable[TrafficClass]) -> BaselinePlan:
        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        trees = {
            flow.name: RelayTree(
                name=f"{flow.name}:collway",
                traffic_name=flow.name,
                root=flow.source,
                destinations=flow.destinations,
                arcs=tuple(
                    sorted(
                        LogicalArc(flow.source, dst)
                        for dst in flow.destinations
                    )
                ),
                kind="collway-direct",
            )
            for flow in flows
        }
        endpoint_edges = (
            frozenset(self.topology.gpu_up_edges.values())
            | frozenset(self.topology.gpu_down_edges.values())
        )
        return _build_fixed_plan(
            self.topology,
            flows,
            trees,
            exclusive_edges=endpoint_edges,
        )


class DeepEPBaseline:
    """Fixed node-star forwarding inspired by DeepEP's hierarchical path.

    For every remote destination node, the source sends one copy to the GPU
    with the same local index as the source. That ingress GPU forwards the
    activation to all destination GPUs in the node. Local-node destinations
    are sent directly. Combine reverses this tree and partially reduces at the
    ingress before returning one tensor to the source. Ready transfers execute
    concurrently without conflict-avoidance packing, exposing fan-out/fan-in
    and shared-fabric bandwidth contention.
    """

    def __init__(self, topology: HierarchicalTopology) -> None:
        self.topology = topology

    def plan(self, traffic: Iterable[TrafficClass]) -> BaselinePlan:
        flows = tuple(traffic)
        ensure_traffic_matches_topology(self.topology, flows)
        trees = {
            flow.name: self._tree_for_flow(flow)
            for flow in flows
        }
        return _build_fixed_plan(
            self.topology,
            flows,
            trees,
            exclusive_edges=frozenset(),
        )

    def _tree_for_flow(self, flow: TrafficClass) -> RelayTree:
        source_gpu = self.topology.gpus[flow.source]
        source_node_gpus = self._ordered_node_gpus(source_gpu.node)
        source_local_index = source_node_gpus.index(flow.source)

        destinations_by_node: dict[str, list[str]] = defaultdict(list)
        for destination in flow.destinations:
            destinations_by_node[self.topology.gpus[destination].node].append(
                destination
            )

        arcs: set[LogicalArc] = set()
        for node, destinations in sorted(destinations_by_node.items()):
            destinations.sort()
            if node == source_gpu.node:
                for destination in destinations:
                    arcs.add(LogicalArc(flow.source, destination))
                continue

            node_gpus = self._ordered_node_gpus(node)
            ingress = node_gpus[source_local_index % len(node_gpus)]
            arcs.add(LogicalArc(flow.source, ingress))
            for destination in destinations:
                if destination != ingress:
                    arcs.add(LogicalArc(ingress, destination))

        tree = RelayTree(
            name=f"{flow.name}:deepep",
            traffic_name=flow.name,
            root=flow.source,
            destinations=flow.destinations,
            arcs=tuple(sorted(arcs)),
            kind="deepep-node-star",
        )
        tree.validate(self.topology)
        return tree

    def _ordered_node_gpus(self, node: str) -> list[str]:
        result = [
            gpu.name for gpu in self.topology.gpus.values() if gpu.node == node
        ]
        if not result:
            raise ValueError(f"node {node} has no GPUs")
        return result


def _build_fixed_plan(
    topology: HierarchicalTopology,
    flows: tuple[TrafficClass, ...],
    trees: dict[str, RelayTree],
    *,
    exclusive_edges: frozenset[str] | None,
) -> BaselinePlan:
    edge_loads = {edge_name: 0.0 for edge_name in topology.edges}
    flow_by_name = {flow.name: flow for flow in flows}
    for flow_name, tree in trees.items():
        tree.validate(topology)
        increment = tree.normalized_edge_increment_us(
            topology,
            flow_by_name[flow_name].volume_bytes,
        )
        for edge_name, load_us in increment.items():
            edge_loads[edge_name] += load_us

    selection = SelectionResult(
        selected_trees=trees,
        edge_load_us=edge_loads,
        theta_us=max(edge_loads.values(), default=0.0),
        doubling_rounds=0,
        candidates_evaluated=len(trees),
    )
    packer = PotentialGreedyScheduler(topology)
    dispatch = packer.build_dispatch_schedule(
        selection,
        flows,
        exclusive_edges=exclusive_edges,
    )
    combine = packer.build_combine_schedule(selection, dispatch)
    return BaselinePlan(selection, dispatch, combine)
