from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass

from .baselines import CollWayBaseline, DeepEPBaseline, NCCLBaseline
from .catalog import CandidateTreeCatalog
from .scenarios import (
    build_hierarchical_topology,
    generate_moe_workload,
    generate_token_routing,
)
from .scheduler import PotentialGreedyScheduler, Schedule, SelectionResult
from .simulator import SimulationConfig, SimulationResult, StageSimulator
from .token_simulator import (
    TokenSimulationConfig,
    simulate_token_strategies,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    selection: SelectionResult
    dispatch: Schedule
    combine: Schedule
    simulation: SimulationResult

    def summary(self) -> dict:
        tree_kinds = Counter(
            tree.kind for tree in self.selection.selected_trees.values()
        )
        return {
            "strategy": self.name,
            "total_time_us": self.simulation.total_time_us,
            "dispatch_time_us": self.simulation.dispatch.total_time_us,
            "combine_time_us": self.simulation.combine.total_time_us,
            "dispatch_stages": len(self.dispatch.stages),
            "combine_stages": len(self.combine.stages),
            "actions": len(self.dispatch.actions) + len(self.combine.actions),
            "total_physical_bytes": self.simulation.total_physical_bytes,
            "max_dispatch_edge_busy_us": (
                self.simulation.dispatch.max_edge_busy_us
            ),
            "max_dispatch_edge_concurrency": (
                self.simulation.dispatch.max_edge_concurrency
            ),
            "max_combine_edge_concurrency": (
                self.simulation.combine.max_edge_concurrency
            ),
            "theta_us": self.selection.theta_us,
            "doubling_rounds": self.selection.doubling_rounds,
            "candidates_evaluated": self.selection.candidates_evaluated,
            "tree_kinds": dict(sorted(tree_kinds.items())),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact per-token RelayMoE communication simulator and "
            "compare NCCL-like, CollWay-like, DeepEP-like, and RelayMoE routes."
        )
    )
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--switches-per-node", type=int, default=4)
    parser.add_argument("--gpus-per-switch", type=int, default=4)
    parser.add_argument("--classes", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--experts",
        "--num-experts",
        dest="experts",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tokens-per-gpu", type=int, default=4096)
    parser.add_argument("--activation-bytes", type=int, default=1024)
    parser.add_argument("--tree-entries-per-node", type=int, default=2)
    parser.add_argument("--min-volume-mib", type=float, default=0.5)
    parser.add_argument("--max-volume-mib", type=float, default=4.0)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--gpu-bandwidth-gbps", type=float, default=64.0)
    parser.add_argument("--pxb-bandwidth-gbps", type=float, default=64.0)
    parser.add_argument("--nics-per-switch", type=int, default=2)
    parser.add_argument("--nic-bandwidth-gbps", type=float, default=50.0)
    parser.add_argument("--network-bandwidth-gbps", type=float, default=400.0)
    parser.add_argument("--stage-launch-us", type=float, default=1.5)
    parser.add_argument("--relay-overhead-us", type=float, default=0.35)
    parser.add_argument(
        "--nccl-contention-loss",
        type=float,
        default=0.05,
        metavar="FRACTION",
        help=(
            "synthetic upper-bound aggregate-goodput loss for uncoordinated "
            "NCCL fan conflicts; use 0 for the ideal fluid-capacity model"
        ),
    )
    parser.add_argument("--show-topology", action="store_true")
    parser.add_argument("--show-trees", action="store_true")
    parser.add_argument(
        "--show-stages",
        type=int,
        default=0,
        metavar="N",
        help="print the first N dispatch stages for each strategy",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--legacy-stage-model",
        action="store_true",
        help="run the earlier synthetic traffic-class stage model",
    )
    return parser


def run_experiment(args: argparse.Namespace):
    topology = build_hierarchical_topology(
        nodes=args.nodes,
        switches_per_node=args.switches_per_node,
        gpus_per_switch=args.gpus_per_switch,
        gpu_bandwidth_GBps=args.gpu_bandwidth_gbps,
        pxb_bandwidth_GBps=args.pxb_bandwidth_gbps,
        nic_bandwidth_GBps=(
            args.nics_per_switch * args.nic_bandwidth_gbps
        ),
        network_bandwidth_GBps=args.network_bandwidth_gbps,
    )
    workload = generate_moe_workload(
        topology,
        classes=args.classes,
        top_k=args.top_k,
        experts=args.experts,
        seed=args.seed,
        min_volume_mib=args.min_volume_mib,
        max_volume_mib=args.max_volume_mib,
    )
    traffic = workload.traffic_classes
    simulation_config = SimulationConfig(
        stage_launch_overhead_us=args.stage_launch_us,
        relay_kernel_overhead_us=args.relay_overhead_us,
    )

    nccl_plan = NCCLBaseline(topology).plan(traffic)
    nccl = _build_experiment(
        "nccl",
        topology,
        nccl_plan.selection,
        nccl_plan.dispatch,
        nccl_plan.combine,
        simulation_config,
    )
    collway_plan = CollWayBaseline(topology).plan(traffic)
    collway = _build_experiment(
        "collway",
        topology,
        collway_plan.selection,
        collway_plan.dispatch,
        collway_plan.combine,
        simulation_config,
    )
    deepep_plan = DeepEPBaseline(topology).plan(traffic)
    deepep = _build_experiment(
        "deepep",
        topology,
        deepep_plan.selection,
        deepep_plan.dispatch,
        deepep_plan.combine,
        simulation_config,
    )
    relaymoe = _run_strategy(
        "relaymoe",
        topology,
        traffic,
        CandidateTreeCatalog(max_candidates=args.max_candidates),
        simulation_config,
    )
    return topology, workload, nccl, collway, deepep, relaymoe


def _run_strategy(
    name,
    topology,
    traffic,
    catalog,
    simulation_config,
) -> Experiment:
    scheduler = PotentialGreedyScheduler(topology, catalog=catalog)
    selection, dispatch, combine = scheduler.plan(traffic)
    return _build_experiment(
        name,
        topology,
        selection,
        dispatch,
        combine,
        simulation_config,
    )


def _build_experiment(
    name,
    topology,
    selection,
    dispatch,
    combine,
    simulation_config,
) -> Experiment:
    simulation = StageSimulator(topology, simulation_config).simulate(
        dispatch, combine
    )
    return Experiment(name, selection, dispatch, combine, simulation)


def _run_token_cli(args: argparse.Namespace) -> int:
    topology = build_hierarchical_topology(
        nodes=args.nodes,
        switches_per_node=args.switches_per_node,
        gpus_per_switch=args.gpus_per_switch,
        gpu_bandwidth_GBps=args.gpu_bandwidth_gbps,
        pxb_bandwidth_GBps=args.pxb_bandwidth_gbps,
        nic_bandwidth_GBps=(
            args.nics_per_switch * args.nic_bandwidth_gbps
        ),
        network_bandwidth_GBps=args.network_bandwidth_gbps,
    )
    routing = generate_token_routing(
        topology,
        tokens_per_gpu=args.tokens_per_gpu,
        top_k=args.top_k,
        experts=args.experts,
        activation_bytes=args.activation_bytes,
        seed=args.seed,
    )
    results = simulate_token_strategies(
        topology,
        routing,
        config=TokenSimulationConfig(
            launch_overhead_us=args.stage_launch_us,
            include_path_latency=True,
            tree_entries_per_node=args.tree_entries_per_node,
            pipeline_tokens=True,
            nccl_uncoordinated_contention_loss=(
                args.nccl_contention_loss
            ),
        ),
    )
    summaries = {
        name: _token_summary(result)
        for name, result in results.items()
    }
    tree_time = results["relaymoe"].total_time_us
    speedups = {
        name: result.total_time_us / tree_time
        for name, result in results.items()
        if name != "relaymoe"
    }
    direct_bytes = results["nccl"].total_physical_bytes
    physical_ratios = {
        name: results["relaymoe"].total_physical_bytes
        / result.total_physical_bytes
        for name, result in results.items()
        if name != "relaymoe"
    }

    if args.as_json:
        print(
            json.dumps(
                {
                    "model": "exact-token-fluid-pipeline",
                    "contention_model": {
                        "nccl_uncoordinated_maximum_loss": (
                            args.nccl_contention_loss
                        ),
                        "calibration": "synthetic",
                        "collway": (
                            "topology-aware direct endpoint matchings"
                        ),
                    },
                    "topology": {
                        "nodes": args.nodes,
                        "switches_per_node": args.switches_per_node,
                        "gpus_per_switch": args.gpus_per_switch,
                        "gpus": routing.ep_size,
                        "nics_per_switch": args.nics_per_switch,
                        "nics_per_node": (
                            args.nics_per_switch
                            * args.switches_per_node
                        ),
                        "bandwidth_GBps": {
                            "gpu_port": args.gpu_bandwidth_gbps,
                            "pxb": args.pxb_bandwidth_gbps,
                            "nic_per_device": args.nic_bandwidth_gbps,
                            "nic_per_switch_aggregate": (
                                args.nics_per_switch
                                * args.nic_bandwidth_gbps
                            ),
                            "network_per_node_pair": (
                                args.network_bandwidth_gbps
                            ),
                        },
                    },
                    "workload": {
                        "ep_size": routing.ep_size,
                        "experts": routing.expert_count,
                        "tokens_per_gpu": routing.tokens_per_gpu,
                        "total_tokens": routing.token_count,
                        "gate_top_k": routing.top_k,
                        "expert_assignments": (
                            routing.expert_assignments
                        ),
                        "remote_expert_assignments": (
                            routing.remote_expert_assignments
                        ),
                        "activation_bytes": routing.activation_bytes,
                        "seed": routing.seed,
                    },
                    "strategies": summaries,
                    "relaymoe_speedups": speedups,
                    "relaymoe_physical_byte_ratios": physical_ratios,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        "Topology: "
        f"{args.nodes} nodes x {args.switches_per_node} switches x "
        f"{args.gpus_per_switch} GPUs = {routing.ep_size} GPUs"
    )
    print(
        "Bandwidths: "
        f"GPU/PXB {args.gpu_bandwidth_gbps:g}/"
        f"{args.pxb_bandwidth_gbps:g}, "
        f"NIC {args.nics_per_switch} x "
        f"{args.nic_bandwidth_gbps:g} per switch, "
        f"network {args.network_bandwidth_gbps:g} GB/s"
    )
    print(
        "MoE trace: "
        f"{routing.tokens_per_gpu} independently routed tokens/GPU, "
        f"{routing.token_count} total, Top-{routing.top_k}, "
        f"{routing.expert_count} experts, "
        f"{routing.activation_bytes} B/activation"
    )
    print(
        "Assignments: "
        f"{routing.expert_assignments} total, "
        f"{routing.remote_expert_assignments} require communication, "
        f"{results['nccl'].dispatch.useful_bytes / 2**20:.2f} MiB/phase"
    )
    print(
        "Execution model: dynamic max-min bandwidth with completion-time "
        "reallocation; fine-grained token pipeline; network-only."
    )
    print(
        "NCCL uncoordinated-contention loss ceiling: "
        f"{100 * args.nccl_contention_loss:.1f}% (synthetic; "
        "set --nccl-contention-loss 0 for ideal capacity)."
    )
    print()
    print(
        f"{'strategy':<10} {'dispatch(us)':>14} {'combine(us)':>13} "
        f"{'total(us)':>12} {'AlgoBW(GB/s)':>14} "
        f"{'physical(MiB)':>15}"
    )
    for name in ("nccl", "collway", "deepep", "relaymoe"):
        summary = summaries[name]
        print(
            f"{name:<10} "
            f"{summary['dispatch_time_us']:>14.2f} "
            f"{summary['combine_time_us']:>13.2f} "
            f"{summary['total_time_us']:>12.2f} "
            f"{summary['dispatch_algobw_GBps']:>14.2f} "
            f"{summary['total_physical_bytes'] / 2**20:>15.2f}"
        )
    print(
        "Dispatch endpoint fan-out/fan-in: "
        f"NCCL {summaries['nccl']['max_dispatch_gpu_send_concurrency']}/"
        f"{summaries['nccl']['max_dispatch_gpu_receive_concurrency']}, "
        f"CollWay "
        f"{summaries['collway']['max_dispatch_gpu_send_concurrency']}/"
        f"{summaries['collway']['max_dispatch_gpu_receive_concurrency']} "
        f"across {summaries['collway']['dispatch_scheduling_steps']} "
        "matching steps."
    )
    print()
    for name in ("nccl", "collway", "deepep"):
        print(
            f"RelayMoE speedup vs {name.upper()}-like: "
            f"{speedups[name]:.3f}x"
        )
    print(
        "RelayMoE physical-byte ratio: "
        f"{results['relaymoe'].total_physical_bytes / direct_bytes:.3f} "
        "vs direct"
    )
    print(
        "Note: reduction/compute is excluded equally for all strategies; "
        "the earlier traffic-class stage model remains available with "
        "--legacy-stage-model."
    )
    return 0


def _token_summary(result) -> dict:
    return {
        "dispatch_time_us": result.dispatch.total_time_us,
        "combine_time_us": result.combine.total_time_us,
        "total_time_us": result.total_time_us,
        "dispatch_algobw_GBps": (
            result.dispatch.effective_algobw_GBps
        ),
        "useful_bytes_per_phase": result.dispatch.useful_bytes,
        "total_physical_bytes": result.total_physical_bytes,
        "dispatch_flows": result.dispatch.flow_count,
        "dispatch_scheduling_steps": result.dispatch.scheduling_steps,
        "max_dispatch_gpu_send_concurrency": (
            result.dispatch.max_gpu_send_concurrency
        ),
        "max_dispatch_gpu_receive_concurrency": (
            result.dispatch.max_gpu_receive_concurrency
        ),
        "max_dispatch_active_streams": (
            result.dispatch.max_edge_concurrency
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    gpu_count = args.nodes * args.switches_per_node * args.gpus_per_switch
    if gpu_count < 2:
        parser.error("the demo requires at least two GPUs")
    if args.experts < 1:
        parser.error("--experts must be positive")
    if args.nics_per_switch < 1:
        parser.error("--nics-per-switch must be positive")
    if args.tokens_per_gpu < 1:
        parser.error("--tokens-per-gpu must be positive")
    if args.activation_bytes < 1:
        parser.error("--activation-bytes must be positive")
    if args.tree_entries_per_node < 1:
        parser.error("--tree-entries-per-node must be positive")
    if not 0 <= args.nccl_contention_loss < 1:
        parser.error("--nccl-contention-loss must be in [0, 1)")
    if not 1 <= args.top_k <= args.experts:
        parser.error("--top-k must not exceed --experts")
    if args.show_stages < 0:
        parser.error("--show-stages must be non-negative")
    bandwidths = (
        args.gpu_bandwidth_gbps,
        args.pxb_bandwidth_gbps,
        args.nic_bandwidth_gbps,
        args.network_bandwidth_gbps,
    )
    if any(value <= 0 for value in bandwidths):
        parser.error("all bandwidth values must be positive")

    if not args.legacy_stage_model:
        return _run_token_cli(args)

    topology, workload, nccl, collway, deepep, relaymoe = run_experiment(args)
    traffic = workload.traffic_classes
    placement_counts = workload.experts_per_gpu
    expert_counts = [
        placement_counts.get(gpu, 0)
        for gpu in topology.gpus
    ]
    nccl_summary = nccl.summary()
    collway_summary = collway.summary()
    deepep_summary = deepep.summary()
    relaymoe_summary = relaymoe.summary()
    speedup_vs_nccl = (
        nccl.simulation.total_time_us / relaymoe.simulation.total_time_us
    )
    speedup_vs_collway = (
        collway.simulation.total_time_us / relaymoe.simulation.total_time_us
    )
    speedup_vs_deepep = (
        deepep.simulation.total_time_us / relaymoe.simulation.total_time_us
    )
    byte_ratio_vs_nccl = (
        relaymoe.simulation.total_physical_bytes
        / nccl.simulation.total_physical_bytes
    )
    byte_ratio_vs_collway = (
        relaymoe.simulation.total_physical_bytes
        / collway.simulation.total_physical_bytes
    )
    byte_ratio_vs_deepep = (
        relaymoe.simulation.total_physical_bytes
        / deepep.simulation.total_physical_bytes
    )

    if args.as_json:
        print(
            json.dumps(
                {
                    "topology": {
                        "nodes": args.nodes,
                        "switches_per_node": args.switches_per_node,
                        "gpus_per_switch": args.gpus_per_switch,
                        "gpus": gpu_count,
                        "bandwidth_GBps": {
                            "gpu_port": args.gpu_bandwidth_gbps,
                            "pxb": args.pxb_bandwidth_gbps,
                            "nic_per_device": args.nic_bandwidth_gbps,
                            "nic_per_switch_aggregate": (
                                args.nics_per_switch
                                * args.nic_bandwidth_gbps
                            ),
                            "network_per_node_pair": (
                                args.network_bandwidth_gbps
                            ),
                        },
                        "nics_per_switch": args.nics_per_switch,
                        "nics_per_node": (
                            args.nics_per_switch
                            * args.switches_per_node
                        ),
                    },
                    "traffic": {
                        "classes": len(traffic),
                        "gate_top_k": workload.top_k,
                        "ep_size": workload.ep_size,
                        "experts": workload.expert_count,
                        "expert_placement": "round_robin",
                        "experts_per_gpu_min": min(expert_counts),
                        "experts_per_gpu_max": max(expert_counts),
                        "average_unique_remote_gpus": (
                            workload.average_destination_gpus
                        ),
                        "seed": args.seed,
                    },
                    "nccl": nccl_summary,
                    "collway": collway_summary,
                    "deepep": deepep_summary,
                    "relaymoe": relaymoe_summary,
                    "relaymoe_speedup_vs_nccl": speedup_vs_nccl,
                    "relaymoe_speedup_vs_collway": speedup_vs_collway,
                    "relaymoe_speedup_vs_deepep": speedup_vs_deepep,
                    "relaymoe_physical_byte_ratio_vs_nccl": (
                        byte_ratio_vs_nccl
                    ),
                    "relaymoe_physical_byte_ratio_vs_collway": (
                        byte_ratio_vs_collway
                    ),
                    "relaymoe_physical_byte_ratio_vs_deepep": (
                        byte_ratio_vs_deepep
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        "Topology: "
        f"{args.nodes} nodes x {args.switches_per_node} switches x "
        f"{args.gpus_per_switch} GPUs = {gpu_count} GPUs"
    )
    print(
        "Bandwidths: "
        f"GPU port {args.gpu_bandwidth_gbps:g}, "
        f"PXB {args.pxb_bandwidth_gbps:g}, "
        f"NIC {args.nics_per_switch} x "
        f"{args.nic_bandwidth_gbps:g} per switch "
        f"(aggregate "
        f"{args.nics_per_switch * args.nic_bandwidth_gbps:g}), "
        f"node-pair network {args.network_bandwidth_gbps:g} GB/s"
    )
    print(
        "Routes: same-switch=GPU up/down; cross-switch=GPU up+PXB+down; "
        "cross-node=GPU up+NIC TX+network+NIC RX+down"
    )
    print(
        f"MoE: EP size={workload.ep_size}, experts={workload.expert_count} "
        f"(round-robin {min(expert_counts)}-{max(expert_counts)}/GPU), "
        f"gate top-k={workload.top_k}"
    )
    print(
        f"Traffic: {len(traffic)} aggregated classes, "
        f"average unique remote GPUs="
        f"{workload.average_destination_gpus:.2f}, seed={args.seed}"
    )
    print()
    print(
        f"{'strategy':<10} {'total(us)':>12} {'dispatch':>12} {'combine':>12} "
        f"{'stages':>9} {'physical(MiB)':>15} {'peak-share':>11} "
        f"{'max-edge(us)':>13}"
    )
    for experiment, summary in (
        (nccl, nccl_summary),
        (collway, collway_summary),
        (deepep, deepep_summary),
        (relaymoe, relaymoe_summary),
    ):
        peak_share = max(
            summary["max_dispatch_edge_concurrency"],
            summary["max_combine_edge_concurrency"],
        )
        print(
            f"{experiment.name:<10} "
            f"{summary['total_time_us']:>12.2f} "
            f"{summary['dispatch_time_us']:>12.2f} "
            f"{summary['combine_time_us']:>12.2f} "
            f"{summary['dispatch_stages'] + summary['combine_stages']:>9d} "
            f"{summary['total_physical_bytes'] / 1024 / 1024:>15.2f} "
            f"{peak_share:>11d} "
            f"{summary['max_dispatch_edge_busy_us']:>13.2f}"
        )
    print("peak-share = maximum concurrent actions on one directed edge")
    print()
    print(f"RelayMoE speedup vs NCCL-like:    {speedup_vs_nccl:.3f}x")
    print(f"RelayMoE speedup vs CollWay-like: {speedup_vs_collway:.3f}x")
    print(f"RelayMoE speedup vs DeepEP-like:  {speedup_vs_deepep:.3f}x")
    print(
        "RelayMoE physical-byte ratio: "
        f"{byte_ratio_vs_nccl:.3f} vs NCCL-like, "
        f"{byte_ratio_vs_collway:.3f} vs CollWay-like, "
        f"{byte_ratio_vs_deepep:.3f} vs DeepEP-like"
    )
    print(f"Selected RelayMoE tree kinds: {relaymoe_summary['tree_kinds']}")

    if args.show_topology:
        print()
        print("Directed physical edges:")
        for edge in sorted(topology.edges.values(), key=lambda item: item.name):
            print(
                f"  {edge.name:<20} {edge.src:>10} -> {edge.dst:<10} "
                f"{edge.kind:<9} {edge.bandwidth_GBps:g} GB/s "
                f"{edge.latency_us:g} us"
            )

    if args.show_trees:
        for experiment in (nccl, collway, deepep, relaymoe):
            print()
            print(f"{experiment.name} trees:")
            for flow_name, tree in sorted(
                experiment.selection.selected_trees.items()
            ):
                arcs = ", ".join(
                    f"{arc.src}->{arc.dst}" for arc in tree.arcs
                )
                print(f"  {flow_name}: {tree.kind}: {arcs}")

    if args.show_stages:
        for experiment in (nccl, collway, deepep, relaymoe):
            print()
            print(f"{experiment.name} dispatch stages:")
            for stage in experiment.dispatch.stages[: args.show_stages]:
                actions = ", ".join(
                    f"{action.traffic_name}:{action.arc.src}->{action.arc.dst}"
                    for action in stage.actions
                )
                print(f"  stage {stage.index}: {actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
