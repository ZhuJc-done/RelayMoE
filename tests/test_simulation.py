from __future__ import annotations

import unittest

from relaymoe.baselines import CollWayBaseline, DeepEPBaseline, NCCLBaseline
from relaymoe.catalog import CandidateTreeCatalog
from relaymoe.model import Gpu, HierarchicalTopology, LogicalArc, TrafficClass
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

    def test_default_relaymoe_model_beats_direct_baseline(self) -> None:
        topology = build_hierarchical_topology()
        traffic = generate_moe_traffic(topology)

        direct_scheduler = PotentialGreedyScheduler(
            topology,
            catalog=CandidateTreeCatalog(max_candidates=1, direct_only=True),
        )
        _, direct_dispatch, direct_combine = direct_scheduler.plan(traffic)
        direct = StageSimulator(topology).simulate(
            direct_dispatch, direct_combine
        )

        tree_scheduler = PotentialGreedyScheduler(topology)
        _, tree_dispatch, tree_combine = tree_scheduler.plan(traffic)
        tree = StageSimulator(topology).simulate(tree_dispatch, tree_combine)

        self.assertLess(tree.total_time_us, direct.total_time_us)
        self.assertLess(
            tree.total_physical_bytes,
            direct.total_physical_bytes,
        )


if __name__ == "__main__":
    unittest.main()
