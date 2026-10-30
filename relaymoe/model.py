from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class PhysicalEdge:
    """One directed, bandwidth-constrained physical resource."""

    name: str
    src: str
    dst: str
    bandwidth_GBps: float
    latency_us: float = 0.0
    kind: str = "generic"

    def __post_init__(self) -> None:
        if self.bandwidth_GBps <= 0:
            raise ValueError(f"edge {self.name} must have positive bandwidth")
        if self.latency_us < 0:
            raise ValueError(f"edge {self.name} must have non-negative latency")

    @property
    def bytes_per_us(self) -> float:
        # Decimal GB/s: 1 GB/s = 1e9 bytes/s = 1000 bytes/us.
        return self.bandwidth_GBps * 1_000.0

    def serialization_time_us(self, size_bytes: int) -> float:
        return size_bytes / self.bytes_per_us


@dataclass(frozen=True, slots=True)
class Gpu:
    name: str
    node: str
    switch: str


@dataclass
class HierarchicalTopology:
    """Directed PCIe hierarchy used by both planning and simulation.

    A logical GPU-to-GPU transfer maps to:

    * same switch: source GPU uplink + destination GPU downlink;
    * same node, different switch: source uplink + directed PXB + downlink;
    * different node: source uplink + source NIC TX + network edge +
      destination NIC RX + destination downlink.
    """

    gpus: dict[str, Gpu] = field(default_factory=dict)
    edges: dict[str, PhysicalEdge] = field(default_factory=dict)
    gpu_up_edges: dict[str, str] = field(default_factory=dict)
    gpu_down_edges: dict[str, str] = field(default_factory=dict)
    pxb_edges: dict[tuple[str, str], str] = field(default_factory=dict)
    nic_tx_edges: dict[str, str] = field(default_factory=dict)
    nic_rx_edges: dict[str, str] = field(default_factory=dict)
    network_edges: dict[tuple[str, str], str] = field(default_factory=dict)

    def add_gpu(
        self,
        gpu: Gpu,
        *,
        gpu_bandwidth_GBps: float,
        gpu_latency_us: float = 0.25,
    ) -> None:
        if gpu.name in self.gpus:
            raise ValueError(f"duplicate GPU: {gpu.name}")
        self.gpus[gpu.name] = gpu

        up_name = f"gpu:{gpu.name}:up"
        down_name = f"gpu:{gpu.name}:down"
        self._add_edge(
            PhysicalEdge(
                up_name,
                gpu.name,
                gpu.switch,
                gpu_bandwidth_GBps,
                gpu_latency_us,
                "gpu_up",
            )
        )
        self._add_edge(
            PhysicalEdge(
                down_name,
                gpu.switch,
                gpu.name,
                gpu_bandwidth_GBps,
                gpu_latency_us,
                "gpu_down",
            )
        )
        self.gpu_up_edges[gpu.name] = up_name
        self.gpu_down_edges[gpu.name] = down_name

    def add_pxb_pair(
        self,
        switch_a: str,
        switch_b: str,
        *,
        bandwidth_GBps: float,
        latency_us: float = 0.4,
    ) -> None:
        if switch_a == switch_b:
            raise ValueError("PXB endpoints must be different switches")
        for src, dst in ((switch_a, switch_b), (switch_b, switch_a)):
            name = f"pxb:{src}->{dst}"
            self._add_edge(
                PhysicalEdge(name, src, dst, bandwidth_GBps, latency_us, "pxb")
            )
            self.pxb_edges[(src, dst)] = name

    def add_switch_nic(
        self,
        switch: str,
        *,
        bandwidth_GBps: float,
        latency_us: float = 0.8,
    ) -> None:
        if switch in self.nic_tx_edges or switch in self.nic_rx_edges:
            raise ValueError(f"NIC edges already exist for switch {switch}")
        fabric_name = f"fabric:{self.switch_node(switch)}"
        tx_name = f"nic:{switch}:tx"
        rx_name = f"nic:{switch}:rx"
        self._add_edge(
            PhysicalEdge(
                tx_name,
                switch,
                fabric_name,
                bandwidth_GBps,
                latency_us,
                "nic_tx",
            )
        )
        self._add_edge(
            PhysicalEdge(
                rx_name,
                fabric_name,
                switch,
                bandwidth_GBps,
                latency_us,
                "nic_rx",
            )
        )
        self.nic_tx_edges[switch] = tx_name
        self.nic_rx_edges[switch] = rx_name

    def add_network_pair(
        self,
        node_a: str,
        node_b: str,
        *,
        bandwidth_GBps: float,
        latency_us: float = 1.5,
    ) -> None:
        if node_a == node_b:
            raise ValueError("network endpoints must be different nodes")
        for src, dst in ((node_a, node_b), (node_b, node_a)):
            name = f"net:{src}->{dst}"
            self._add_edge(
                PhysicalEdge(
                    name,
                    f"fabric:{src}",
                    f"fabric:{dst}",
                    bandwidth_GBps,
                    latency_us,
                    "network",
                )
            )
            self.network_edges[(src, dst)] = name

    def _add_edge(self, edge: PhysicalEdge) -> None:
        if edge.name in self.edges:
            raise ValueError(f"duplicate physical edge: {edge.name}")
        self.edges[edge.name] = edge

    def switch_node(self, switch: str) -> str:
        nodes = {gpu.node for gpu in self.gpus.values() if gpu.switch == switch}
        if len(nodes) != 1:
            raise ValueError(
                f"switch {switch!r} must map to exactly one node, got {sorted(nodes)}"
            )
        return next(iter(nodes))

    def route(self, src_gpu: str, dst_gpu: str) -> tuple[str, ...]:
        if src_gpu == dst_gpu:
            return ()
        try:
            src = self.gpus[src_gpu]
            dst = self.gpus[dst_gpu]
        except KeyError as exc:
            raise KeyError(f"unknown GPU in route {src_gpu}->{dst_gpu}") from exc

        path = [self.gpu_up_edges[src.name]]
        if src.switch == dst.switch:
            path.append(self.gpu_down_edges[dst.name])
        elif src.node == dst.node:
            try:
                path.append(self.pxb_edges[(src.switch, dst.switch)])
            except KeyError as exc:
                raise ValueError(
                    f"missing PXB route {src.switch}->{dst.switch}"
                ) from exc
            path.append(self.gpu_down_edges[dst.name])
        else:
            try:
                path.extend(
                    (
                        self.nic_tx_edges[src.switch],
                        self.network_edges[(src.node, dst.node)],
                        self.nic_rx_edges[dst.switch],
                        self.gpu_down_edges[dst.name],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing cross-node route {src.name}->{dst.name}"
                ) from exc
        return tuple(path)

    def route_latency_us(self, edge_names: Iterable[str]) -> float:
        return sum(self.edges[name].latency_us for name in edge_names)

    def route_serialization_us(
        self, edge_names: Iterable[str], size_bytes: int
    ) -> float:
        names = tuple(edge_names)
        if not names:
            return 0.0
        return max(self.edges[name].serialization_time_us(size_bytes) for name in names)

    def switches(self) -> tuple[str, ...]:
        return tuple(sorted({gpu.switch for gpu in self.gpus.values()}))

    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted({gpu.node for gpu in self.gpus.values()}))

    def gpus_on_switch(self, switch: str) -> tuple[str, ...]:
        return tuple(
            sorted(gpu.name for gpu in self.gpus.values() if gpu.switch == switch)
        )

    def gpus_on_node(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(gpu.name for gpu in self.gpus.values() if gpu.node == node))

    def validate(self) -> None:
        if not self.gpus:
            raise ValueError("topology must contain at least one GPU")
        for gpu in self.gpus.values():
            if gpu.name not in self.gpu_up_edges or gpu.name not in self.gpu_down_edges:
                raise ValueError(f"GPU {gpu.name} is missing directed port edges")
            if gpu.switch not in self.nic_tx_edges or gpu.switch not in self.nic_rx_edges:
                raise ValueError(f"switch {gpu.switch} is missing NIC edges")
        for src in self.gpus.values():
            for dst in self.gpus.values():
                if src.name != dst.name:
                    self.route(src.name, dst.name)


@dataclass(frozen=True, slots=True)
class TrafficClass:
    """Aggregated tokens sharing a source, destination set and byte volume."""

    name: str
    source: str
    destinations: tuple[str, ...]
    volume_bytes: int

    def __post_init__(self) -> None:
        unique = tuple(dict.fromkeys(self.destinations))
        if not unique:
            raise ValueError(f"traffic class {self.name} needs a destination")
        if len(unique) != len(self.destinations):
            raise ValueError(f"traffic class {self.name} has duplicate destinations")
        if self.source in unique:
            raise ValueError(
                f"traffic class {self.name} must remove its source from destinations"
            )
        if self.volume_bytes <= 0:
            raise ValueError(f"traffic class {self.name} needs positive volume")


@dataclass(frozen=True, order=True, slots=True)
class LogicalArc:
    src: str
    dst: str

    def __post_init__(self) -> None:
        if self.src == self.dst:
            raise ValueError("logical tree arc cannot be a self-loop")


@dataclass(frozen=True)
class RelayTree:
    name: str
    traffic_name: str
    root: str
    destinations: tuple[str, ...]
    arcs: tuple[LogicalArc, ...]
    kind: str = "candidate"

    def nodes(self) -> set[str]:
        result = {self.root}
        for arc in self.arcs:
            result.add(arc.src)
            result.add(arc.dst)
        return result

    def children(self) -> Mapping[str, tuple[str, ...]]:
        result: dict[str, list[str]] = defaultdict(list)
        for arc in self.arcs:
            result[arc.src].append(arc.dst)
        return {parent: tuple(sorted(children)) for parent, children in result.items()}

    def parent_arcs(self) -> dict[str, LogicalArc]:
        result: dict[str, LogicalArc] = {}
        for arc in self.arcs:
            if arc.dst in result:
                raise ValueError(f"node {arc.dst} has multiple parents in {self.name}")
            result[arc.dst] = arc
        return result

    def depths(self) -> dict[str, int]:
        children = self.children()
        depths = {self.root: 0}
        queue = deque([self.root])
        while queue:
            parent = queue.popleft()
            for child in children.get(parent, ()):
                depths[child] = depths[parent] + 1
                queue.append(child)
        return depths

    def remaining_depths(self) -> dict[str, int]:
        children = self.children()
        result: dict[str, int] = {}

        def visit(node: str) -> int:
            if node in result:
                return result[node]
            child_depths = [visit(child) + 1 for child in children.get(node, ())]
            result[node] = max(child_depths, default=0)
            return result[node]

        visit(self.root)
        return result

    def validate(self, topology: HierarchicalTopology) -> None:
        if self.root not in topology.gpus:
            raise ValueError(f"tree {self.name} has unknown root {self.root}")
        if not self.arcs:
            raise ValueError(f"tree {self.name} has no arcs")
        for node in self.nodes():
            if node not in topology.gpus:
                raise ValueError(f"tree {self.name} references unknown GPU {node}")

        parents = self.parent_arcs()
        if self.root in parents:
            raise ValueError(f"tree {self.name} root has a parent")

        children = self.children()
        reached = {self.root}
        queue = deque([self.root])
        while queue:
            parent = queue.popleft()
            for child in children.get(parent, ()):
                if child in reached:
                    raise ValueError(f"tree {self.name} contains a cycle")
                reached.add(child)
                queue.append(child)

        if len(self.arcs) != len(reached) - 1:
            raise ValueError(f"tree {self.name} is disconnected or cyclic")
        missing = set(self.destinations) - reached
        if missing:
            raise ValueError(
                f"tree {self.name} does not cover destinations {sorted(missing)}"
            )
        for arc in self.arcs:
            topology.route(arc.src, arc.dst)

    def physical_edge_counts(
        self, topology: HierarchicalTopology
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for arc in self.arcs:
            for edge_name in topology.route(arc.src, arc.dst):
                counts[edge_name] += 1
        return dict(counts)

    def normalized_edge_increment_us(
        self,
        topology: HierarchicalTopology,
        volume_bytes: int,
    ) -> dict[str, float]:
        return {
            edge_name: count
            * topology.edges[edge_name].serialization_time_us(volume_bytes)
            for edge_name, count in self.physical_edge_counts(topology).items()
        }


def ensure_traffic_matches_topology(
    topology: HierarchicalTopology,
    traffic: Iterable[TrafficClass],
) -> None:
    seen_names: set[str] = set()
    for flow in traffic:
        if flow.name in seen_names:
            raise ValueError(f"duplicate traffic class name: {flow.name}")
        seen_names.add(flow.name)
        if flow.source not in topology.gpus:
            raise ValueError(f"traffic {flow.name} has unknown source {flow.source}")
        unknown = set(flow.destinations) - topology.gpus.keys()
        if unknown:
            raise ValueError(
                f"traffic {flow.name} has unknown destinations {sorted(unknown)}"
            )
