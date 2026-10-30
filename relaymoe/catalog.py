from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import (
    HierarchicalTopology,
    LogicalArc,
    RelayTree,
    TrafficClass,
)


@dataclass
class CandidateTreeCatalog:
    """Generate a compact set of hierarchical relay-tree candidates.

    The prototype enumerates concrete trees rather than storing abstract
    skeletons.  This keeps the first simulator executable and deterministic;
    the API can later be backed by an offline skeleton catalog.
    """

    max_candidates: int = 16
    direct_only: bool = False

    def candidates(
        self,
        topology: HierarchicalTopology,
        traffic: TrafficClass,
    ) -> tuple[RelayTree, ...]:
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        raw: list[tuple[str, list[LogicalArc]]] = [
            (
                "direct",
                [LogicalArc(traffic.source, dst) for dst in traffic.destinations],
            )
        ]
        if not self.direct_only:
            raw.extend(self._switch_relay_candidates(topology, traffic))
            raw.extend(self._node_relay_candidates(topology, traffic))

        result: list[RelayTree] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for kind, arcs in raw:
            canonical = tuple(
                sorted({LogicalArc(arc.src, arc.dst) for arc in arcs})
            )
            key = tuple((arc.src, arc.dst) for arc in canonical)
            if key in seen:
                continue
            seen.add(key)
            tree = RelayTree(
                name=f"{traffic.name}:{kind}:{len(result)}",
                traffic_name=traffic.name,
                root=traffic.source,
                destinations=traffic.destinations,
                arcs=canonical,
                kind=kind,
            )
            tree.validate(topology)
            result.append(tree)
            if len(result) >= self.max_candidates:
                break
        return tuple(result)

    def _switch_relay_candidates(
        self,
        topology: HierarchicalTopology,
        traffic: TrafficClass,
    ) -> list[tuple[str, list[LogicalArc]]]:
        by_switch: dict[str, list[str]] = defaultdict(list)
        for dst in traffic.destinations:
            by_switch[topology.gpus[dst].switch].append(dst)
        groups = [
            (switch, sorted(destinations))
            for switch, destinations in sorted(by_switch.items())
        ]
        variant_count = min(
            self.max_candidates,
            max((len(destinations) for _, destinations in groups), default=1),
        )

        candidates: list[tuple[str, list[LogicalArc]]] = []
        for variant in range(variant_count):
            arcs: list[LogicalArc] = []
            for _, destinations in groups:
                relay = destinations[variant % len(destinations)]
                self._append_arc(arcs, traffic.source, relay)
                for dst in destinations:
                    if dst != relay:
                        self._append_arc(arcs, relay, dst)
            candidates.append((f"switch-relay-v{variant}", arcs))
        return candidates

    def _node_relay_candidates(
        self,
        topology: HierarchicalTopology,
        traffic: TrafficClass,
    ) -> list[tuple[str, list[LogicalArc]]]:
        by_node: dict[str, list[str]] = defaultdict(list)
        for dst in traffic.destinations:
            by_node[topology.gpus[dst].node].append(dst)
        node_groups = [
            (node, sorted(destinations))
            for node, destinations in sorted(by_node.items())
        ]
        variant_count = min(
            self.max_candidates,
            max((len(destinations) for _, destinations in node_groups), default=1),
        )

        candidates: list[tuple[str, list[LogicalArc]]] = []
        for variant in range(variant_count):
            arcs: list[LogicalArc] = []
            source_node = topology.gpus[traffic.source].node
            for node, node_destinations in node_groups:
                if node == source_node:
                    for destination in node_destinations:
                        self._append_arc(arcs, traffic.source, destination)
                    continue

                ingress = node_destinations[variant % len(node_destinations)]
                self._append_arc(arcs, traffic.source, ingress)

                by_switch: dict[str, list[str]] = defaultdict(list)
                for dst in node_destinations:
                    by_switch[topology.gpus[dst].switch].append(dst)

                for _, switch_destinations in sorted(by_switch.items()):
                    switch_destinations.sort()
                    if ingress in switch_destinations:
                        relay = ingress
                    else:
                        relay = switch_destinations[
                            variant % len(switch_destinations)
                        ]
                        self._append_arc(arcs, ingress, relay)
                    for dst in switch_destinations:
                        if dst != relay:
                            self._append_arc(arcs, relay, dst)
            candidates.append((f"node-relay-v{variant}", arcs))
        return candidates

    @staticmethod
    def _append_arc(arcs: list[LogicalArc], src: str, dst: str) -> None:
        if src != dst:
            arc = LogicalArc(src, dst)
            if arc not in arcs:
                arcs.append(arc)
