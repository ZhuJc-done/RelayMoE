from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relaymoe.baselines import CollWayBaseline, DeepEPBaseline, NCCLBaseline
from relaymoe.catalog import CandidateTreeCatalog, OfflineBinomialCatalog
from relaymoe.model import Gpu, HierarchicalTopology, LogicalArc, TrafficClass
from relaymoe.runtime import RelayMoE, RelayMoEBackend
from relaymoe.scenarios import (
    TokenRouting,
    build_hierarchical_topology,
    generate_moe_traffic,
    generate_moe_workload,
    generate_token_routing,
)
from relaymoe.scheduler import PotentialGreedyScheduler
from relaymoe.simulator import StageSimulator
from relaymoe.token_simulator import (
    TokenSimulationConfig,
    simulate_token_strategies,
)


def _build_background_topology() -> HierarchicalTopology:
    topology = HierarchicalTopology()
    gpu_specs = (
        ("g0", "n0", "n0s0"),
        ("g1", "n1", "n1s0"),
        ("g2", "n1", "n1s0"),
        ("g3", "n1", "n1s1"),
        ("g4", "n1", "n1s1"),
        ("g5", "n1", "n1s2"),
        ("g6", "n1", "n1s2"),
    )
    for name, node, switch in gpu_specs:
        topology.add_gpu(
            Gpu(name=name, node=node, switch=switch),
            gpu_bandwidth_GBps=64.0,
            gpu_latency_us=0.0,
        )
    destination_switches = ("n1s0", "n1s1", "n1s2")
    for left in range(len(destination_switches)):
        for right in range(left + 1, len(destination_switches)):
            topology.add_pxb_pair(
                destination_switches[left],
                destination_switches[right],
                bandwidth_GBps=64.0,
                latency_us=0.0,
            )
    for switch in ("n0s0", *destination_switches):
        topology.add_switch_nic(
            switch,
            bandwidth_GBps=50.0,
            latency_us=0.0,
        )
    topology.add_network_pair(
        "n0",
        "n1",
        bandwidth_GBps=50.0,
        latency_us=0.0,
    )
    topology.validate()
    return topology


class TopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=2,
            gpus_per_switch=2,
        )

    def test_routes_follow_hierarchy(self) -> None:
        same_switch = self.topology.route("g0", "g1")
        same_node = self.topology.route("g0", "g2")
        cross_node = self.topology.route("g0", "g4")

        self.assertEqual(
            [self.topology.edges[name].kind for name in same_switch],
            ["gpu_up", "gpu_down"],
        )
        self.assertEqual(
            [self.topology.edges[name].kind for name in same_node],
            ["gpu_up", "pxb", "gpu_down"],
        )
        self.assertEqual(
            [self.topology.edges[name].kind for name in cross_node],
            ["gpu_up", "nic_tx", "network", "nic_rx", "gpu_down"],
        )


class MoEWorkloadTests(unittest.TestCase):
    def test_document_aligned_default_topology_and_workload(self) -> None:
        topology = build_hierarchical_topology()
        workload = generate_moe_workload(
            topology,
            classes=32,
            top_k=8,
            experts=32,
        )

        self.assertEqual(len(topology.gpus), 32)
        self.assertEqual(
            [len(topology.gpus_on_node(node)) for node in topology.nodes()],
            [16, 16],
        )
        self.assertEqual(workload.ep_size, 32)
        self.assertEqual(workload.expert_count, 32)
        self.assertEqual(workload.top_k, 8)
        self.assertEqual(set(workload.experts_per_gpu.values()), {1})
        self.assertEqual(len(workload.traffic_classes), 32)
        for flow in workload.traffic_classes:
            self.assertNotIn(flow.source, flow.destinations)
            self.assertLessEqual(len(flow.destinations), workload.top_k)

    def test_multiple_experts_on_one_gpu_are_deduplicated(self) -> None:
        topology = build_hierarchical_topology(
            nodes=1,
            switches_per_node=2,
            gpus_per_switch=2,
        )
        workload = generate_moe_workload(
            topology,
            classes=1,
            top_k=8,
            experts=8,
        )

        self.assertEqual(
            workload.traffic_classes[0].destinations,
            ("g1", "g2", "g3"),
        )

    def test_each_token_has_an_independent_exact_top_k_route(self) -> None:
        topology = build_hierarchical_topology()
        routing = generate_token_routing(
            topology,
            tokens_per_gpu=128,
            top_k=8,
            experts=32,
            activation_bytes=1024,
            seed=19,
        )

        self.assertEqual(routing.ep_size, 32)
        self.assertEqual(routing.token_count, 4096)
        self.assertEqual(routing.expert_assignments, 4096 * 8)
        for source_masks in routing.route_masks:
            self.assertGreater(len(set(source_masks)), 1)
            self.assertTrue(
                all(mask.bit_count() == 8 for mask in source_masks)
            )
        for source in range(routing.ep_size):
            self.assertEqual(
                sum(
                    bits.bit_count()
                    for bits in routing.expert_token_bits[source]
                ),
                routing.tokens_per_gpu * routing.top_k,
            )

    def test_background_motivation_example(self) -> None:
        topology = _build_background_topology()
        selected_mask = sum(1 << expert for expert in range(1, 7))
        source_bits = tuple(
            1 if expert in range(1, 7) else 0
            for expert in range(7)
        )
        routing = TokenRouting(
            gpu_names=tuple(topology.gpus),
            expert_to_gpu=tuple(topology.gpus),
            top_k=6,
            tokens_per_gpu=1,
            activation_bytes=64_000_000,
            route_masks=(
                (selected_mask,),
                *((0,) for _ in range(6)),
            ),
            expert_token_bits=(
                source_bits,
                *((tuple(0 for _ in range(7))) for _ in range(6)),
            ),
            seed=0,
        )
        results = simulate_token_strategies(
            topology,
            routing,
            config=TokenSimulationConfig(
                launch_overhead_us=0.0,
                relay_overhead_us=0.0,
                reduction_launch_overhead_us=0.0,
                include_path_latency=False,
                tree_entries_per_node=2,
                pipeline_tokens=False,
                nccl_uncoordinated_contention_loss=0.0,
            ),
            source_indices=(0,),
        )

        self.assertAlmostEqual(
            results["nccl"].dispatch.total_time_us,
            7680.0,
        )
        self.assertAlmostEqual(
            results["collway"].dispatch.total_time_us,
            7680.0,
        )
        self.assertAlmostEqual(
            results["deepep"].dispatch.total_time_us,
            6280.0,
        )
        self.assertAlmostEqual(
            results["relaymoe"].dispatch.total_time_us,
            3560.0,
        )

    def test_collway_compiles_endpoint_matching_steps(self) -> None:
        topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=2,
            gpus_per_switch=2,
        )
        routing = generate_token_routing(
            topology,
            tokens_per_gpu=128,
            top_k=4,
            experts=8,
            activation_bytes=1024,
            seed=23,
        )
        results = simulate_token_strategies(
            topology,
            routing,
            config=TokenSimulationConfig(
                nccl_uncoordinated_contention_loss=0.0,
            ),
        )

        nccl = results["nccl"].dispatch
        collway = results["collway"].dispatch
        self.assertGreater(nccl.max_gpu_send_concurrency, 1)
        self.assertGreater(nccl.max_gpu_receive_concurrency, 1)
        self.assertEqual(collway.max_gpu_send_concurrency, 1)
        self.assertEqual(collway.max_gpu_receive_concurrency, 1)
        self.assertGreater(collway.scheduling_steps, 1)
        self.assertEqual(collway.physical_bytes, nccl.physical_bytes)


class CandidateCatalogTests(unittest.TestCase):
    def test_candidates_are_valid_destination_relay_trees(self) -> None:
        topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=2,
            gpus_per_switch=2,
        )
        flow = TrafficClass(
            "remote",
            source="g0",
            destinations=("g4", "g5", "g6", "g7"),
            volume_bytes=1024 * 1024,
        )
        candidates = CandidateTreeCatalog().candidates(topology, flow)

        self.assertGreater(len(candidates), 1)
        for tree in candidates:
            tree.validate(topology)
            self.assertTrue(tree.nodes() - {flow.source} <= set(flow.destinations))

    def test_relay_tree_reuses_cross_node_trunk(self) -> None:
        topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=1,
            gpus_per_switch=2,
        )
        flow = TrafficClass(
            "multicast",
            source="g0",
            destinations=("g2", "g3"),
            volume_bytes=1024 * 1024,
        )
        candidates = CandidateTreeCatalog().candidates(topology, flow)
        network_edge = topology.network_edges[("n0", "n1")]
        direct = next(tree for tree in candidates if tree.kind == "direct")
        relayed = min(
            (tree for tree in candidates if tree.kind != "direct"),
            key=lambda tree: tree.physical_edge_counts(topology).get(
                network_edge, 0
            ),
        )

        self.assertEqual(direct.physical_edge_counts(topology)[network_edge], 2)
        self.assertEqual(relayed.physical_edge_counts(topology)[network_edge], 1)

    def test_binomial_skeleton_meets_min_stages_and_one_to_one_rules(self) -> None:
        catalog = OfflineBinomialCatalog()
        canon_switch_to_node = (0, 0, 0, 0, 1, 1, 1, 1)
        m_canon = (1, 1, 0, 1, 2, 1, 1, 0)
        skeletons = catalog.lookup(m_canon, canon_switch_to_node, 2)

        self.assertGreater(len(skeletons), 0)
        expected_stages = sum(m_canon).bit_length()
        for skel in skeletons:
            self.assertEqual(skel.stage_count, expected_stages)
            holders = {(0, -1)}
            for stage in skel.stages:
                senders = [arc.src_role for arc in stage]
                receivers = [arc.dst_role for arc in stage]
                self.assertEqual(len(senders), len(set(senders)))
                self.assertEqual(len(receivers), len(set(receivers)))
                self.assertTrue(set(senders).issubset(holders))
                self.assertTrue(set(receivers).isdisjoint(holders))
                pxb_pairs = [
                    (arc.src_role[0], arc.dst_role[0])
                    for arc in stage
                    if arc.src_role[0] != arc.dst_role[0]
                    and arc.tx_nic_slot is None
                ]
                tx_slots = [
                    (arc.src_role[0], arc.tx_nic_slot)
                    for arc in stage
                    if arc.tx_nic_slot is not None
                ]
                rx_slots = [
                    (arc.dst_role[0], arc.rx_nic_slot)
                    for arc in stage
                    if arc.rx_nic_slot is not None
                ]
                self.assertEqual(len(pxb_pairs), len(set(pxb_pairs)))
                self.assertEqual(len(tx_slots), len(set(tx_slots)))
                self.assertEqual(len(rx_slots), len(set(rx_slots)))
                holders.update(receivers)

    def test_document_aligned_catalog_is_not_silently_truncated(self) -> None:
        distribution = (1, 2, 1, 0, 1, 1, 1, 1)
        canonical_nodes = (0, 0, 0, 0, 1, 1, 1, 1)

        unbounded = OfflineBinomialCatalog().lookup(
            distribution,
            canonical_nodes,
            2,
        )
        approximate = OfflineBinomialCatalog(max_variants=16).lookup(
            distribution,
            canonical_nodes,
            2,
        )

        self.assertGreater(len(unbounded), 16)
        self.assertEqual(len(approximate), 16)

    def test_revision572_event_scheduler_preserves_resource_invariants(
        self,
    ) -> None:
        topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=4,
            gpus_per_switch=4,
        )
        routing = generate_token_routing(
            topology,
            tokens_per_gpu=128,
            top_k=8,
            experts=32,
            activation_bytes=1024,
            seed=11,
            routing_motifs_per_gpu=8,
        )
        results = simulate_token_strategies(topology, routing)

        relaymoe = results["relaymoe"]
        self.assertEqual(relaymoe.dispatch.max_edge_concurrency, 1)
        self.assertEqual(relaymoe.combine.max_edge_concurrency, 1)
        self.assertGreater(results["nccl"].dispatch.max_edge_concurrency, 1)
        self.assertGreater(relaymoe.planning_time_us, 0.0)
        self.assertEqual(
            relaymoe.dispatch.physical_bytes,
            sum(relaymoe.dispatch.edge_bytes.values()),
        )
        self.assertEqual(
            relaymoe.combine.physical_bytes,
            sum(relaymoe.combine.edge_bytes.values()),
        )
        self.assertEqual(
            relaymoe.dispatch.flow_count,
            relaymoe.combine.flow_count,
        )
        expected_max_load = max(
            (
                byte_count / topology.edges[edge_name].bytes_per_us
                for edge_name, byte_count
                in relaymoe.dispatch.edge_bytes.items()
            ),
            default=0.0,
        )
        self.assertAlmostEqual(
            relaymoe.dispatch.max_edge_load_us,
            expected_max_load,
        )

    def test_offline_catalog_round_trips_without_regeneration(self) -> None:
        catalog = OfflineBinomialCatalog()
        m_canon = (1, 1, 0, 2)
        canonical_nodes = (0, 0, 1, 1)
        expected = catalog.lookup(m_canon, canonical_nodes, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.json"
            catalog.save(path)
            restored = OfflineBinomialCatalog.load(path)
            corrupted = json.loads(path.read_text(encoding="utf-8"))
            corrupted["entries"][0]["m_canon"][0] += 1
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                OfflineBinomialCatalog.load(path)

        actual = restored.lookup(m_canon, canonical_nodes, 2)
        self.assertEqual(restored.entry_count, 1)
        self.assertEqual(restored.variant_count, len(expected))
        self.assertEqual(actual, expected)


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=2,
            gpus_per_switch=2,
        )

    def test_collway_uses_direct_arcs_and_valid_schedules(self) -> None:
        flow = TrafficClass(
            "direct",
            source="g0",
            destinations=("g1", "g4", "g7"),
            volume_bytes=1024 * 1024,
        )
        plan = CollWayBaseline(self.topology).plan((flow,))

        self.assertEqual(
            set(plan.selection.selected_trees[flow.name].arcs),
            {
                LogicalArc("g0", "g1"),
                LogicalArc("g0", "g4"),
                LogicalArc("g0", "g7"),
            },
        )
        plan.dispatch.validate()
        plan.combine.validate()
        self.assertIsNotNone(plan.dispatch.exclusive_edges)
        self.assertIn(
            self.topology.gpu_up_edges["g0"],
            plan.dispatch.exclusive_edges,
        )

    def test_nccl_is_single_stage_direct_with_fan_conflicts(self) -> None:
        flows = (
            TrafficClass(
                "fanout",
                source="g0",
                destinations=("g4", "g5", "g6"),
                volume_bytes=1024 * 1024,
            ),
            TrafficClass(
                "fanin",
                source="g1",
                destinations=("g4",),
                volume_bytes=1024 * 1024,
            ),
        )
        plan = NCCLBaseline(self.topology).plan(flows)
        result = StageSimulator(self.topology).simulate(
            plan.dispatch,
            plan.combine,
        )

        self.assertEqual(len(plan.dispatch.stages), 1)
        self.assertEqual(len(plan.combine.stages), 1)
        self.assertEqual(plan.dispatch.exclusive_edges, frozenset())
        self.assertTrue(
            all(
                tree.kind == "nccl-direct"
                and all(arc.src == tree.root for arc in tree.arcs)
                for tree in plan.selection.selected_trees.values()
            )
        )

        source_uplink = self.topology.gpu_up_edges["g0"]
        destination_downlink = self.topology.gpu_down_edges["g4"]
        self.assertEqual(
            result.dispatch.edge_peak_concurrency[source_uplink],
            3,
        )
        self.assertEqual(
            result.dispatch.edge_peak_concurrency[destination_downlink],
            2,
        )

    def test_deepep_uses_fixed_remote_ingress(self) -> None:
        flow = TrafficClass(
            "remote",
            source="g0",
            destinations=("g5", "g6", "g7"),
            volume_bytes=1024 * 1024,
        )
        plan = DeepEPBaseline(self.topology).plan((flow,))

        self.assertEqual(
            set(plan.selection.selected_trees[flow.name].arcs),
            {
                LogicalArc("g0", "g4"),
                LogicalArc("g4", "g5"),
                LogicalArc("g4", "g6"),
                LogicalArc("g4", "g7"),
            },
        )
        plan.dispatch.validate()
        plan.combine.validate()
        self.assertEqual(plan.dispatch.exclusive_edges, frozenset())

    def test_collway_allows_internal_fabric_contention(self) -> None:
        flows = (
            TrafficClass(
                "left",
                source="g0",
                destinations=("g4",),
                volume_bytes=1024 * 1024,
            ),
            TrafficClass(
                "right",
                source="g1",
                destinations=("g5",),
                volume_bytes=1024 * 1024,
            ),
        )
        plan = CollWayBaseline(self.topology).plan(flows)
        result = StageSimulator(self.topology).simulate(
            plan.dispatch,
            plan.combine,
        )

        shared_nic = self.topology.nic_tx_edges["n0s0"]
        shared_network = self.topology.network_edges[("n0", "n1")]
        self.assertEqual(len(plan.dispatch.stages), 1)
        self.assertEqual(
            result.dispatch.edge_peak_concurrency[shared_nic],
            2,
        )
        self.assertEqual(
            result.dispatch.edge_peak_concurrency[shared_network],
            2,
        )

    def test_deepep_shared_uplink_divides_bandwidth(self) -> None:
        flow = TrafficClass(
            "contended",
            source="g0",
            destinations=("g5", "g6"),
            volume_bytes=1024 * 1024,
        )
        plan = DeepEPBaseline(self.topology).plan((flow,))
        result = StageSimulator(self.topology).simulate(
            plan.dispatch,
            plan.combine,
        )

        uplink = self.topology.gpu_up_edges["g4"]
        base_serialization_us = self.topology.edges[
            uplink
        ].serialization_time_us(flow.volume_bytes)
        forward = next(
            action
            for stage in result.dispatch.stages
            for action in stage.actions
            if action.action_name.endswith("g4->g5")
        )

        self.assertEqual(
            result.dispatch.edge_peak_concurrency[uplink],
            2,
        )
        self.assertEqual(forward.max_edge_concurrency, 2)
        self.assertAlmostEqual(
            forward.serialization_us,
            2 * base_serialization_us,
        )


class _RecordingBackend:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def execute(self, *, plan, phase, tensors):
        self.phases.append(phase.phase)
        return {
            "phase": phase.phase,
            "actions": len(phase.actions),
            "input": tensors,
            "classes": len(plan.traffic_classes),
        }


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = build_hierarchical_topology(
            nodes=2,
            switches_per_node=2,
            gpus_per_switch=2,
        )

    def test_gating_api_builds_reusable_dispatch_and_combine_plan(self) -> None:
        gpu_names = tuple(self.topology.gpus)
        routes = []
        for source in range(len(gpu_names)):
            routes.append(
                (
                    (
                        (source + 1) % len(gpu_names),
                        (source + 4) % len(gpu_names),
                    ),
                    (
                        (source + 2) % len(gpu_names),
                        (source + 5) % len(gpu_names),
                    ),
                    (
                        (source + 1) % len(gpu_names),
                        (source + 4) % len(gpu_names),
                    ),
                )
            )

        backend = _RecordingBackend()
        self.assertIsInstance(backend, RelayMoEBackend)
        relaymoe = RelayMoE(self.topology, backend=backend)
        weights = tuple(
            tuple((0.6, 0.4) for _ in source_routes)
            for source_routes in routes
        )
        plan = relaymoe.plan_from_gating(
            routes,
            expert_to_gpu=gpu_names,
            activation_bytes=1024,
            expert_weights_by_gpu=weights,
        )

        plan.dispatch.validate()
        plan.combine.validate()
        json.dumps(plan.to_dict())
        self.assertGreater(len(plan.traffic_classes), 0)
        self.assertEqual(
            set(plan.traffic_layouts),
            {flow.name for flow in plan.traffic_classes},
        )
        self.assertTrue(
            all(
                layout.expert_weights is not None
                for layout in plan.traffic_layouts.values()
            )
        )
        self.assertEqual(
            len(plan.dispatch.actions),
            len(plan.combine.actions),
        )
        self.assertTrue(
            any(action.dependencies for action in plan.dispatch.actions)
        )
        cross_node_actions = [
            action
            for action in plan.dispatch.actions
            if self.topology.gpus[action.arc.src].node
            != self.topology.gpus[action.arc.dst].node
        ]
        self.assertTrue(cross_node_actions)
        for action in cross_node_actions:
            kinds = {
                self.topology.edges[edge].kind
                for edge in action.physical_edges
            }
            self.assertIn("nic_dev_tx", kinds)
            self.assertIn("nic_dev_rx", kinds)
            self.assertNotIn("network", kinds)

        simulation = relaymoe.simulate(
            plan,
            launch_overhead_us=0.0,
            include_path_latency=True,
        )
        timings = simulation.dispatch.action_timings
        for action in plan.dispatch.actions:
            for dependency in action.dependencies:
                self.assertGreaterEqual(
                    timings[action.name].ready_us + 1e-12,
                    timings[dependency].finish_us,
                )
        for edge_name in simulation.dispatch.edge_bytes:
            intervals = sorted(
                (
                    timings[action.name].start_us,
                    timings[action.name].start_us
                    + timings[action.name].serialization_us,
                )
                for action in plan.dispatch.actions
                if edge_name in action.physical_edges
            )
            for previous, current in zip(intervals, intervals[1:]):
                self.assertLessEqual(previous[1], current[0] + 1e-12)

        dispatched = relaymoe.dispatch(plan, "hidden")
        combined = relaymoe.combine(plan, "expert-output")
        self.assertEqual(backend.phases, ["dispatch", "combine"])
        self.assertEqual(dispatched["phase"], "dispatch")
        self.assertEqual(combined["phase"], "combine")

    def test_explicit_offline_phase_persists_catalog_for_online_only_use(
        self,
    ) -> None:
        traffic = (
            TrafficClass(
                name="remote",
                source="g0",
                destinations=("g4", "g5", "g6", "g7"),
                volume_bytes=4096,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "relaymoe-catalog.json"
            builder = RelayMoE(self.topology, catalog_path=path)
            self.assertEqual(builder.prebuild_catalog(traffic), 1)
            self.assertTrue(path.exists())

            online = RelayMoE(
                self.topology,
                catalog_path=path,
                require_prebuilt_catalog=True,
            )
            first = online.plan(traffic)
            second = online.plan(traffic)

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.planning_time_us, 0.0)
        self.assertEqual(
            first.selection.arc_physical_edges,
            second.selection.arc_physical_edges,
        )

    def test_online_only_mode_rejects_missing_offline_key(self) -> None:
        flow = TrafficClass(
            name="not-prebuilt",
            source="g0",
            destinations=("g4", "g6"),
            volume_bytes=1024,
        )
        relaymoe = RelayMoE(
            self.topology,
            require_prebuilt_catalog=True,
        )
        with self.assertRaisesRegex(RuntimeError, "prebuild_catalog"):
            relaymoe.plan((flow,))

    def test_token_trace_plan_preserves_token_to_expert_membership(self) -> None:
        routing = generate_token_routing(
            self.topology,
            tokens_per_gpu=8,
            top_k=4,
            experts=8,
            activation_bytes=2048,
            seed=31,
        )
        plan = RelayMoE(self.topology).plan_from_token_routing(routing)

        self.assertEqual(
            set(plan.traffic_layouts),
            {flow.name for flow in plan.traffic_classes},
        )
        for flow in plan.traffic_classes:
            layout = plan.traffic_layouts[flow.name]
            self.assertEqual(
                flow.volume_bytes,
                len(layout.token_indices) * routing.activation_bytes,
            )
            self.assertTrue(
                all(
                    len(experts) == routing.top_k
                    for experts in layout.expert_indices
                )
            )


class EndToEndTests(unittest.TestCase):
    def test_schedule_is_conflict_free_and_simulates(self) -> None:
        topology = build_hierarchical_topology()
        traffic = generate_moe_traffic(
            topology,
            classes=6,
            top_k=4,
            seed=11,
        )
        scheduler = PotentialGreedyScheduler(topology)
        selection, dispatch, combine = scheduler.plan(traffic)

        dispatch.validate()
        combine.validate()
        result = StageSimulator(topology).simulate(dispatch, combine)

        self.assertEqual(set(selection.selected_trees), {flow.name for flow in traffic})
        self.assertGreater(result.dispatch.total_time_us, 0)
        self.assertGreater(result.combine.total_time_us, 0)
        self.assertEqual(len(dispatch.actions), len(combine.actions))

    def test_relay_and_direct_plans_cover_identical_traffic(self) -> None:
        topology = build_hierarchical_topology()
        traffic = generate_moe_traffic(topology)

        direct_scheduler = PotentialGreedyScheduler(
            topology,
            catalog=CandidateTreeCatalog(max_candidates=1, direct_only=True),
        )
        direct_selection, direct_dispatch, direct_combine = (
            direct_scheduler.plan(traffic)
        )
        direct = StageSimulator(topology).simulate(
            direct_dispatch, direct_combine
        )

        tree_scheduler = PotentialGreedyScheduler(topology)
        tree_selection, tree_dispatch, tree_combine = tree_scheduler.plan(
            traffic
        )
        tree = StageSimulator(topology).simulate(tree_dispatch, tree_combine)

        expected_actions = sum(len(flow.destinations) for flow in traffic)
        self.assertEqual(len(direct_dispatch.actions), expected_actions)
        self.assertEqual(len(tree_dispatch.actions), expected_actions)
        self.assertEqual(len(direct_combine.actions), expected_actions)
        self.assertEqual(len(tree_combine.actions), expected_actions)
        self.assertGreater(direct.total_time_us, 0.0)
        self.assertGreater(tree.total_time_us, 0.0)
        for selection in (direct_selection, tree_selection):
            for flow in traffic:
                selected = selection.selected_trees[flow.name]
                selected.validate(topology)
                self.assertEqual(
                    set(flow.destinations),
                    selected.nodes() - {flow.source},
                )


if __name__ == "__main__":
    unittest.main()
