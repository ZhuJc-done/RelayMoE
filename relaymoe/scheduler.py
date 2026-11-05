from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .catalog import BinomialSkeleton, CandidateTreeCatalog, OfflineBinomialCatalog
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
    arc_physical_edges: Mapping[tuple[str, str, str], tuple[str, ...]] = field(
        default_factory=dict
    )
    arc_stage_index: Mapping[tuple[str, str, str], int] = field(
        default_factory=dict
    )

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


@dataclass(frozen=True)
class EventActionTiming:
    action_name: str
    ready_us: float
    start_us: float
    finish_us: float
    serialization_us: float
    path_latency_us: float
    preprocessing_us: float


@dataclass(frozen=True)
class EventDrivenPhaseResult:
    phase: str
    total_time_us: float
    useful_bytes: int
    physical_bytes: int
    action_count: int
    edge_bytes: Mapping[str, int]
    edge_busy_us: Mapping[str, float]
    max_edge_load_us: float
    action_timings: Mapping[str, EventActionTiming]
    max_concurrent_actions: int

    @property
    def effective_algobw_GBps(self) -> float:
        if self.total_time_us == 0:
            return 0.0
        return self.useful_bytes / self.total_time_us / 1_000.0


class PotentialGreedyScheduler:
    """Revision-572 Online Phase I (unpruned catalog binding and selection)
    and Online Phase II (Work-Conserving Event-Driven Scheduling)."""

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

        if self.catalog.direct_only:
            return self._select_concrete_candidates(flows)

        return self._select_unpruned_binomial(flows)

    def _select_unpruned_binomial(
        self, flows: tuple[TrafficClass, ...]
    ) -> SelectionResult:
        topology = self.topology
        switches = topology.switches()
        model_edges = topology.model_directed_edges()
        d_edges = max(2, len(model_edges))
        lambda_stop = 2.0 * math.log(d_edges) + 3.0

        gpu_pxb_bw = min(
            topology.edges[edge_name].bytes_per_us
            for edge_name in model_edges
            if topology.edges[edge_name].kind
            in {"gpu_up", "gpu_down", "pxb"}
        )
        sample_sw = switches[0]
        dev_tx_edges = topology.nic_tx_edges_by_switch.get(sample_sw, ())
        nic_model_edges = [
            topology.edges[edge_name].bytes_per_us
            for edge_name in model_edges
            if topology.edges[edge_name].kind in {"nic_dev_tx", "nic_dev_rx"}
        ]
        nic_bw = (
            min(nic_model_edges)
            if nic_model_edges
            else topology.edges[topology.nic_tx_edges[sample_sw]].bytes_per_us
        )
        nics_per_switch = max(1, len(dev_tx_edges))

        ordered = sorted(flows, key=lambda flow: -flow.volume_bytes)
        offline_catalog: OfflineBinomialCatalog = self.catalog.offline_catalog
        sw_index = {sw: idx for idx, sw in enumerate(switches)}
        sw_count = len(switches)

        # Prepare per-flow canonical switch mapping and O(1) skeleton lookup
        prepared_flows = []
        theta_0 = 0.0
        total_dest_bytes = 0
        for flow in ordered:
            total_dest_bytes += len(flow.destinations) * flow.volume_bytes
            src_sw = topology.gpus[flow.source].switch
            src_sw_idx = sw_index[src_sw]
            src_node = topology.gpus[flow.source].node
            by_switch: dict[str, list[str]] = defaultdict(list)
            for dst in flow.destinations:
                by_switch[topology.gpus[dst].switch].append(dst)
            for g_list in by_switch.values():
                g_list.sort()

            local_other = sorted(
                (
                    sw
                    for sw in switches
                    if topology.switch_node(sw) == src_node and sw != src_sw
                ),
                key=lambda sw: (
                    -len(by_switch.get(sw, ())),
                    (sw_index[sw] - src_sw_idx) % sw_count,
                ),
            )
            remote_sw = sorted(
                (
                    sw
                    for sw in switches
                    if topology.switch_node(sw) != src_node
                ),
                key=lambda sw: (
                    -len(by_switch.get(sw, ())),
                    (sw_index[sw] - src_sw_idx) % sw_count,
                ),
            )
            canon_to_actual = (src_sw, *local_other, *remote_sw)
            node_order = {src_node: 0}
            for sw in remote_sw:
                n_name = topology.switch_node(sw)
                if n_name not in node_order:
                    node_order[n_name] = len(node_order)
            canon_switch_to_node = tuple(
                node_order[topology.switch_node(sw)] for sw in canon_to_actual
            )
            m_canon = tuple(len(by_switch.get(sw, ())) for sw in canon_to_actual)
            skeletons = offline_catalog.lookup(
                m_canon,
                canon_switch_to_node,
                nics_per_switch,
            )
            if not skeletons:
                raise RuntimeError(
                    f"offline catalog returned no binomial skeleton for {flow.name}"
                )
            min_max_delta = min(
                skel.max_norm_us(
                    flow.volume_bytes,
                    gpu_pxb_bytes_per_us=gpu_pxb_bw,
                    nic_bytes_per_us=nic_bw,
                )
                for skel in skeletons
            )
            theta_0 = max(theta_0, min_max_delta)
            prepared_flows.append(
                (flow, src_sw, dict(by_switch), canon_to_actual, skeletons)
            )

        avg_down_lower_bound = (
            total_dest_bytes
            / (max(1, len(topology.gpus)) * gpu_pxb_bw * lambda_stop)
        )
        theta = max(theta_0, avg_down_lower_bound, 1e-9)
        candidates_evaluated = 0
        final_trees: dict[str, RelayTree] = {}
        final_loads: dict[str, float] = {}
        final_arc_edges: dict[tuple[str, str, str], tuple[str, ...]] = {}
        final_arc_stages: dict[tuple[str, str, str], int] = {}
        final_round = 0

        for doubling_round in range(self.max_doublings + 1):
            edge_loads = {edge_name: 0.0 for edge_name in topology.edges}
            inv_2theta = 1.0 / (2.0 * theta)
            selected: dict[str, RelayTree] = {}
            arc_edges_map: dict[tuple[str, str, str], tuple[str, ...]] = {}
            arc_stage_map: dict[tuple[str, str, str], int] = {}

            for (
                flow,
                src_sw,
                by_switch,
                canon_to_actual,
                skeletons,
            ) in prepared_flows:
                vol = flow.volume_bytes
                sorted_gpus_by_csw: dict[int, list[str]] = {}
                for csw, actual_sw in enumerate(canon_to_actual):
                    if actual_sw in by_switch:
                        g_list = by_switch[actual_sw]
                        if len(g_list) > 1:
                            sorted_gpus_by_csw[csw] = sorted(
                                g_list,
                                key=lambda g: (
                                    edge_loads[topology.gpu_up_edges[g]],
                                    edge_loads[topology.gpu_down_edges[g]],
                                    g,
                                ),
                            )
                        else:
                            sorted_gpus_by_csw[csw] = g_list

                best_skel: BinomialSkeleton | None = None
                best_idx = 0
                best_key: tuple[float, int] | None = None

                for idx, skel in enumerate(skeletons):
                    candidates_evaluated += 1
                    score = 0.0
                    if skel.root_out_deg > 0:
                        up_edge = topology.gpu_up_edges[flow.source]
                        inc = topology.edges[up_edge].serialization_time_us(
                            skel.root_out_deg * vol
                        )
                        score += math.exp(
                            min(60.0, edge_loads[up_edge] * inv_2theta)
                        ) * math.expm1(min(60.0, inc * inv_2theta))

                    for csw, nz_degs in skel.nonzero_role_out_degs_by_csw.items():
                        g_sorted = sorted_gpus_by_csw[csw]
                        for pos, deg in nz_degs:
                            up_edge = topology.gpu_up_edges[g_sorted[pos]]
                            inc = topology.edges[up_edge].serialization_time_us(
                                deg * vol
                            )
                            score += math.exp(
                                min(60.0, edge_loads[up_edge] * inv_2theta)
                            ) * math.expm1(min(60.0, inc * inv_2theta))

                    for (c_u, c_v), cnt in skel.pxb_canon_pairs:
                        pxb_edge = topology.pxb_edges[
                            (canon_to_actual[c_u], canon_to_actual[c_v])
                        ]
                        inc = topology.edges[pxb_edge].serialization_time_us(
                            cnt * vol
                        )
                        score += math.exp(
                            min(60.0, edge_loads[pxb_edge] * inv_2theta)
                        ) * math.expm1(min(60.0, inc * inv_2theta))

                    for csw, slot_cnts in skel.nic_tx_slot_counts.items():
                        actual_sw = canon_to_actual[csw]
                        tx_devs = sorted(
                            topology.nic_tx_edges_by_switch[actual_sw],
                            key=lambda e: (edge_loads[e], e),
                        )
                        for pos, (_, cnt) in enumerate(slot_cnts):
                            nic_edge = tx_devs[pos % len(tx_devs)]
                            inc = topology.edges[nic_edge].serialization_time_us(
                                cnt * vol
                            )
                            score += math.exp(
                                min(60.0, edge_loads[nic_edge] * inv_2theta)
                            ) * math.expm1(min(60.0, inc * inv_2theta))

                    for csw, slot_cnts in skel.nic_rx_slot_counts.items():
                        actual_sw = canon_to_actual[csw]
                        rx_devs = sorted(
                            topology.nic_rx_edges_by_switch[actual_sw],
                            key=lambda e: (edge_loads[e], e),
                        )
                        for pos, (_, cnt) in enumerate(slot_cnts):
                            nic_edge = rx_devs[pos % len(rx_devs)]
                            inc = topology.edges[nic_edge].serialization_time_us(
                                cnt * vol
                            )
                            score += math.exp(
                                min(60.0, edge_loads[nic_edge] * inv_2theta)
                            ) * math.expm1(min(60.0, inc * inv_2theta))

                    key = (score, idx)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_skel = skel
                        best_idx = idx

                assert best_skel is not None
                # Exact GPU & NIC binding for best_skel
                role_map: dict[tuple[int, int], str] = {(0, -1): flow.source}
                up_root = topology.gpu_up_edges[flow.source]
                edge_loads[up_root] += topology.edges[
                    up_root
                ].serialization_time_us(best_skel.root_out_deg * vol)
                for dst in flow.destinations:
                    down_edge = topology.gpu_down_edges[dst]
                    edge_loads[down_edge] += topology.edges[
                        down_edge
                    ].serialization_time_us(vol)

                for csw, roles in best_skel.role_order_by_csw.items():
                    g_sorted = sorted_gpus_by_csw[csw]
                    degs = best_skel.role_out_degs_by_csw[csw]
                    for pos, r_local in enumerate(roles):
                        g_name = g_sorted[pos]
                        role_map[(csw, r_local)] = g_name
                        if degs[pos] > 0:
                            up_edge = topology.gpu_up_edges[g_name]
                            edge_loads[up_edge] += topology.edges[
                                up_edge
                            ].serialization_time_us(degs[pos] * vol)

                for (c_u, c_v), cnt in best_skel.pxb_canon_pairs:
                    pxb_edge = topology.pxb_edges[
                        (canon_to_actual[c_u], canon_to_actual[c_v])
                    ]
                    edge_loads[pxb_edge] += topology.edges[
                        pxb_edge
                    ].serialization_time_us(cnt * vol)

                tx_slot_map: dict[tuple[int, int], str] = {}
                for csw, slot_cnts in best_skel.nic_tx_slot_counts.items():
                    actual_sw = canon_to_actual[csw]
                    tx_devs = sorted(
                        topology.nic_tx_edges_by_switch[actual_sw],
                        key=lambda e: (edge_loads[e], e),
                    )
                    for pos, (slot_id, cnt) in enumerate(slot_cnts):
                        chosen_tx = tx_devs[pos % len(tx_devs)]
                        tx_slot_map[(csw, slot_id)] = chosen_tx
                        edge_loads[chosen_tx] += topology.edges[
                            chosen_tx
                        ].serialization_time_us(cnt * vol)

                rx_slot_map: dict[tuple[int, int], str] = {}
                for csw, slot_cnts in best_skel.nic_rx_slot_counts.items():
                    actual_sw = canon_to_actual[csw]
                    rx_devs = sorted(
                        topology.nic_rx_edges_by_switch[actual_sw],
                        key=lambda e: (edge_loads[e], e),
                    )
                    for pos, (slot_id, cnt) in enumerate(slot_cnts):
                        chosen_rx = rx_devs[pos % len(rx_devs)]
                        rx_slot_map[(csw, slot_id)] = chosen_rx
                        edge_loads[chosen_rx] += topology.edges[
                            chosen_rx
                        ].serialization_time_us(cnt * vol)

                arcs: list[LogicalArc] = []
                for stage_index, stage in enumerate(best_skel.stages):
                    for sk_arc in stage:
                        src_gpu = role_map[sk_arc.src_role]
                        dst_gpu = role_map[sk_arc.dst_role]
                        arcs.append(LogicalArc(src_gpu, dst_gpu))
                        if sk_arc.tx_nic_slot is not None:
                            tx_e = tx_slot_map[
                                (sk_arc.src_role[0], sk_arc.tx_nic_slot)
                            ]
                            rx_e = rx_slot_map[
                                (sk_arc.dst_role[0], sk_arc.rx_nic_slot)
                            ]
                            phys = topology.route(
                                src_gpu,
                                dst_gpu,
                                src_nic_edge=tx_e,
                                dst_nic_edge=rx_e,
                                include_network_edge=False,
                            )
                        else:
                            phys = topology.route(src_gpu, dst_gpu)
                        arc_edges_map[(flow.name, src_gpu, dst_gpu)] = phys
                        arc_stage_map[(flow.name, src_gpu, dst_gpu)] = (
                            stage_index
                        )

                tree = RelayTree(
                    name=f"{flow.name}:binomial-relay-v{best_idx}",
                    traffic_name=flow.name,
                    root=flow.source,
                    destinations=flow.destinations,
                    arcs=tuple(arcs),
                    kind=f"binomial-relay-v{best_idx}",
                )
                tree.validate(topology)
                selected[flow.name] = tree

            max_model_load = max(
                (edge_loads[e] for e in model_edges), default=0.0
            )
            final_trees = selected
            final_loads = edge_loads
            final_arc_edges = arc_edges_map
            final_arc_stages = arc_stage_map
            final_round = doubling_round
            if max_model_load <= lambda_stop * theta:
                break
            theta *= 2.0

        return SelectionResult(
            selected_trees=final_trees,
            edge_load_us=final_loads,
            theta_us=theta,
            doubling_rounds=final_round,
            candidates_evaluated=candidates_evaluated,
            arc_physical_edges=final_arc_edges,
            arc_stage_index=final_arc_stages,
        )

    def _select_concrete_candidates(
        self, flows: tuple[TrafficClass, ...]
    ) -> SelectionResult:
        candidates_by_flow: dict[str, tuple[RelayTree, ...]] = {}
        for flow in flows:
            candidates = self.catalog.candidates(self.topology, flow)
            if not candidates:
                raise ValueError(f"traffic {flow.name} has no candidate tree")
            candidates_by_flow[flow.name] = candidates

        theta = self._initial_theta(flows, candidates_by_flow)
        ordered = sorted(flows, key=lambda flow: (-flow.volume_bytes, flow.name))
        candidates_evaluated = 0
        lambda_stop = 2.0 * math.log(max(2, len(self.topology.edges))) + 3.0

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
            final_trees = selected
            final_loads = edge_loads
            final_round = doubling_round
            if max_load <= lambda_stop * theta:
                break
            theta *= 2.0

        return SelectionResult(
            selected_trees=final_trees,
            edge_load_us=final_loads,
            theta_us=theta,
            doubling_rounds=final_round,
            candidates_evaluated=candidates_evaluated,
        )

    def build_dispatch_actions(
        self,
        selection: SelectionResult,
        traffic: Iterable[TrafficClass],
    ) -> tuple[ScheduledAction, ...]:
        """Build the Dispatch DAG A = union_{c in C} F_c* without barrier stage packing."""

        flow_by_name = {flow.name: flow for flow in traffic}
        actions: list[ScheduledAction] = []
        for flow_name, tree in sorted(selection.selected_trees.items()):
            flow = flow_by_name[flow_name]
            parent_arcs = tree.parent_arcs()
            remaining_depth = tree.remaining_depths()
            total_stages = max(1, len(tree.destinations).bit_length())
            for arc in tree.arcs:
                action_name = self._action_name("dispatch", flow_name, arc)
                dependency = parent_arcs.get(arc.src)
                dependencies = (
                    (self._action_name("dispatch", flow_name, dependency),)
                    if dependency is not None
                    else ()
                )
                custom_phys = selection.arc_physical_edges.get(
                    (flow_name, arc.src, arc.dst)
                )
                physical_edges = (
                    custom_phys
                    if custom_phys is not None
                    else self.topology.route(arc.src, arc.dst)
                )
                stage_idx = selection.arc_stage_index.get(
                    (flow_name, arc.src, arc.dst)
                )
                rem_depth = (
                    total_stages - stage_idx
                    if stage_idx is not None
                    else remaining_depth.get(arc.dst, 0)
                )
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
                        remaining_depth=rem_depth,
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
        return tuple(actions)

    def build_combine_actions(
        self,
        selection: SelectionResult,
        dispatch_actions: Iterable[ScheduledAction],
    ) -> tuple[ScheduledAction, ...]:
        """Build the reversed Combine DAG without barrier stage packing."""

        dispatch_by_key = {
            (action.traffic_name, action.arc.src, action.arc.dst): action
            for action in dispatch_actions
        }
        actions: list[ScheduledAction] = []
        for flow_name, tree in sorted(selection.selected_trees.items()):
            children = tree.children()
            depths = tree.depths()
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
                reverse_phys = self._reverse_physical_edges(
                    dispatch_action.physical_edges,
                    reverse_arc,
                )
                actions.append(
                    ScheduledAction(
                        name=self._action_name("combine", flow_name, arc),
                        phase="combine",
                        traffic_name=flow_name,
                        tree_name=tree.name,
                        arc=reverse_arc,
                        physical_edges=reverse_phys,
                        size_bytes=dispatch_action.size_bytes,
                        dependencies=dependency_names,
                        remaining_depth=depths.get(arc.dst, 1),
                        priority_load_us=dispatch_action.priority_load_us,
                        relay=bool(children.get(arc.dst)),
                        reduction_inputs=len(children.get(arc.dst, ())),
                    )
                )
        return tuple(actions)

    def build_dispatch_schedule(
        self,
        selection: SelectionResult,
        traffic: Iterable[TrafficClass],
        *,
        exclusive_edges: frozenset[str] | None = None,
    ) -> Schedule:
        actions = self.build_dispatch_actions(selection, traffic)
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
            depths = tree.depths()
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
                reverse_phys = self._reverse_physical_edges(
                    dispatch_action.physical_edges,
                    reverse_arc,
                )
                combine_by_dispatch_name[dispatch_action.name] = ScheduledAction(
                    name=self._action_name("combine", flow_name, arc),
                    phase="combine",
                    traffic_name=flow_name,
                    tree_name=tree.name,
                    arc=reverse_arc,
                    physical_edges=reverse_phys,
                    size_bytes=dispatch_action.size_bytes,
                    dependencies=dependency_names,
                    remaining_depth=depths.get(arc.dst, 1),
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

    def _reverse_physical_edges(
        self,
        forward_edges: tuple[str, ...],
        reverse_arc: LogicalArc,
    ) -> tuple[str, ...]:
        tx_edge: str | None = None
        rx_edge: str | None = None
        has_network = False
        for edge_name in forward_edges:
            kind = self.topology.edges[edge_name].kind
            if kind == "nic_dev_tx":
                rx_edge = edge_name[:-3] + ":rx"
            elif kind == "nic_dev_rx":
                tx_edge = edge_name[:-3] + ":tx"
            elif kind == "network":
                has_network = True
        if tx_edge is not None and rx_edge is not None:
            return self.topology.route(
                reverse_arc.src,
                reverse_arc.dst,
                src_nic_edge=tx_edge,
                dst_nic_edge=rx_edge,
                include_network_edge=has_network,
            )
        return self.topology.route(reverse_arc.src, reverse_arc.dst)

    def simulate_event_driven_phase(
        self,
        selection: SelectionResult,
        actions: Iterable[ScheduledAction],
        *,
        phase: str,
        useful_bytes: int,
        launch_overhead_us: float = 1.5,
        include_path_latency: bool = True,
        relay_overhead_us: float = 0.0,
        reduction_launch_overhead_us: float = 0.0,
        reduction_bandwidth_GBps: float = 200.0,
    ) -> EventDrivenPhaseResult:
        """Online Phase II: Work-Conserving Event-Driven Scheduling on DAG A."""

        del selection  # The selected paths and priorities are embedded in actions.
        if launch_overhead_us < 0 or relay_overhead_us < 0:
            raise ValueError("event-driven overheads must be non-negative")
        if reduction_launch_overhead_us < 0:
            raise ValueError("reduction launch overhead must be non-negative")
        if reduction_bandwidth_GBps <= 0:
            raise ValueError("reduction bandwidth must be positive")

        action_list = tuple(actions)
        if not action_list:
            return EventDrivenPhaseResult(
                phase=phase,
                total_time_us=0.0,
                useful_bytes=useful_bytes,
                physical_bytes=0,
                action_count=0,
                edge_bytes={},
                edge_busy_us={},
                max_edge_load_us=0.0,
                action_timings={},
                max_concurrent_actions=0,
            )

        index_by_name = {
            action.name: idx for idx, action in enumerate(action_list)
        }
        if len(index_by_name) != len(action_list):
            raise ValueError(f"{phase} contains duplicate action names")
        dependents: list[list[int]] = [[] for _ in action_list]
        rem_deps = [len(action.dependencies) for action in action_list]
        ser_durations: list[float] = []
        prop_latencies: list[float] = []
        preprocessing_us: list[float] = []

        edge_index: dict[str, int] = {}
        edge_bw_list: list[float] = []
        action_masks: list[int] = []
        action_edge_indices: list[tuple[int, ...]] = []
        action_sizes: list[int] = []
        action_depths: list[int] = []
        action_prios: list[float] = []

        for idx, action in enumerate(action_list):
            for dep_name in action.dependencies:
                if dep_name not in index_by_name:
                    raise ValueError(
                        f"{action.name} depends on unknown action {dep_name}"
                    )
                dependents[index_by_name[dep_name]].append(idx)
            mask = 0
            e_indices: list[int] = []
            ser_us = 0.0
            for edge_name in action.physical_edges:
                e_idx = edge_index.get(edge_name)
                if e_idx is None:
                    e_idx = len(edge_bw_list)
                    edge_index[edge_name] = e_idx
                    edge_bw_list.append(
                        self.topology.edges[edge_name].bytes_per_us
                    )
                mask |= 1 << e_idx
                e_indices.append(e_idx)
                edge_ser = action.size_bytes / edge_bw_list[e_idx]
                if edge_ser > ser_us:
                    ser_us = edge_ser
            action_masks.append(mask)
            action_edge_indices.append(tuple(e_indices))
            action_sizes.append(action.size_bytes)
            action_depths.append(action.remaining_depth)
            action_prios.append(action.priority_load_us)
            ser_durations.append(ser_us)
            hop_lat = (
                self.topology.route_latency_us(action.physical_edges)
                if include_path_latency
                else 0.0
            )
            prop_latencies.append(hop_lat)
            pre_us = 0.0
            if action.phase == "dispatch" and action.relay:
                pre_us += relay_overhead_us
            if action.phase == "combine" and action.reduction_inputs > 0:
                reduction_bytes = action.size_bytes * action.reduction_inputs
                pre_us += (
                    reduction_launch_overhead_us
                    + reduction_bytes
                    / (reduction_bandwidth_GBps * 1_000.0)
                )
            preprocessing_us.append(pre_us)

        sort_keys = [
            (
                -action_depths[idx],
                -action_prios[idx],
                -action_sizes[idx],
                idx,
            )
            for idx in range(len(action_list))
        ]
        ready: list[int] = []
        ready_needs_sort = True
        occupied_mask = 0
        edge_bytes_by_idx = [0] * len(edge_bw_list)
        # Event tuple: (time_us, kind, action_idx).
        # Resource release precedes completion/ready processing at equal time.
        RELEASE = 0
        COMPLETE = 1
        READY = 2
        events: list[tuple[float, int, int]] = []
        current_time_us = launch_overhead_us
        completed = 0
        active_actions = 0
        max_concurrent_actions = 0
        ready_at = [0.0] * len(action_list)
        start_at = [0.0] * len(action_list)
        finish_at = [0.0] * len(action_list)

        def make_ready(idx: int, dependency_finish_us: float) -> None:
            nonlocal ready_needs_sort
            timestamp = dependency_finish_us + preprocessing_us[idx]
            ready_at[idx] = timestamp
            if timestamp <= current_time_us + 1e-12:
                ready.append(idx)
                ready_needs_sort = True
            else:
                heapq.heappush(events, (timestamp, READY, idx))

        for idx, dep_cnt in enumerate(rem_deps):
            if dep_cnt == 0:
                make_ready(idx, launch_overhead_us)

        def try_launch() -> bool:
            nonlocal occupied_mask, ready_needs_sort
            nonlocal active_actions, max_concurrent_actions
            if not ready:
                return False
            if ready_needs_sort:
                ready.sort(key=sort_keys.__getitem__)
                ready_needs_sort = False
            waiting: list[int] = []
            launched = False
            for idx in ready:
                mask = action_masks[idx]
                if not (occupied_mask & mask):
                    launched = True
                    occupied_mask |= mask
                    active_actions += 1
                    max_concurrent_actions = max(
                        max_concurrent_actions,
                        active_actions,
                    )
                    sz = action_sizes[idx]
                    for e_idx in action_edge_indices[idx]:
                        edge_bytes_by_idx[e_idx] += sz
                    ser_us = ser_durations[idx]
                    start_at[idx] = current_time_us
                    release_us = current_time_us + ser_us
                    finish_us = release_us + prop_latencies[idx]
                    finish_at[idx] = finish_us
                    heapq.heappush(
                        events, (release_us, RELEASE, idx)
                    )
                    heapq.heappush(
                        events, (finish_us, COMPLETE, idx)
                    )
                else:
                    waiting.append(idx)
            ready[:] = waiting
            return launched

        while completed < len(action_list):
            while events and events[0][0] <= current_time_us + 1e-12:
                _, kind, idx = heapq.heappop(events)
                if kind == RELEASE:
                    occupied_mask &= ~action_masks[idx]
                elif kind == COMPLETE:
                    completed += 1
                    active_actions -= 1
                    for child_idx in dependents[idx]:
                        rem_deps[child_idx] -= 1
                        if rem_deps[child_idx] == 0:
                            make_ready(child_idx, current_time_us)
                else:
                    ready.append(idx)
                    ready_needs_sort = True

            try_launch()
            if completed >= len(action_list):
                break
            if not events:
                unresolved = [
                    action_list[idx].name
                    for idx, count in enumerate(rem_deps)
                    if count > 0
                ]
                raise RuntimeError(
                    f"{phase} event DAG deadlocked: {unresolved}"
                )
            current_time_us = events[0][0]

        edge_bytes = {
            edge_name: edge_bytes_by_idx[e_idx]
            for edge_name, e_idx in edge_index.items()
            if edge_bytes_by_idx[e_idx] > 0
        }
        edge_busy_us = {
            edge_name: edge_bytes_by_idx[e_idx] / edge_bw_list[e_idx]
            for edge_name, e_idx in edge_index.items()
            if edge_bytes_by_idx[e_idx] > 0
        }
        action_timings = {
            action.name: EventActionTiming(
                action_name=action.name,
                ready_us=ready_at[idx],
                start_us=start_at[idx],
                finish_us=finish_at[idx],
                serialization_us=ser_durations[idx],
                path_latency_us=prop_latencies[idx],
                preprocessing_us=preprocessing_us[idx],
            )
            for idx, action in enumerate(action_list)
        }
        return EventDrivenPhaseResult(
            phase=phase,
            total_time_us=current_time_us,
            useful_bytes=useful_bytes,
            physical_bytes=sum(edge_bytes_by_idx),
            action_count=len(action_list),
            edge_bytes=edge_bytes,
            edge_busy_us=edge_busy_us,
            max_edge_load_us=max(edge_busy_us.values(), default=0.0),
            action_timings=action_timings,
            max_concurrent_actions=max_concurrent_actions,
        )

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
        return max(largest_unavoidable_step, 1e-9)

    @staticmethod
    def _potential_increment(
        current: Mapping[str, float],
        delta: Mapping[str, float],
        theta: float,
    ) -> float:
        """Equation (7) of RelayMoE Revision 411 with denominator 2 * Theta."""

        score = 0.0
        two_theta = 2.0 * theta
        for edge_name, increment in delta.items():
            current_scaled = min(current.get(edge_name, 0.0) / two_theta, 60.0)
            increment_scaled = min(increment / two_theta, 60.0)
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
