from relaymoe import CandidateTreeCatalog, PotentialGreedyScheduler, StageSimulator
from relaymoe.model import TrafficClass
from relaymoe.scenarios import build_hierarchical_topology


def main() -> None:
    topology = build_hierarchical_topology(
        nodes=2,
        switches_per_node=2,
        gpus_per_switch=2,
        gpu_bandwidth_GBps=32.0,
        pxb_bandwidth_GBps=48.0,
        nic_bandwidth_GBps=50.0,
    )
    traffic = (
        TrafficClass(
            name="remote-fanout",
            source="g0",
            destinations=("g4", "g5", "g6", "g7"),
            volume_bytes=4 * 1024 * 1024,
        ),
        TrafficClass(
            name="mixed-fanout",
            source="g3",
            destinations=("g0", "g4", "g5", "g7"),
            volume_bytes=2 * 1024 * 1024,
        ),
    )

    scheduler = PotentialGreedyScheduler(
        topology,
        catalog=CandidateTreeCatalog(max_candidates=16),
    )
    selection, dispatch, combine = scheduler.plan(traffic)
    result = StageSimulator(topology).simulate(dispatch, combine)

    for flow_name, tree in sorted(selection.selected_trees.items()):
        arcs = ", ".join(f"{arc.src}->{arc.dst}" for arc in tree.arcs)
        print(f"{flow_name}: {tree.kind}: {arcs}")
    print(f"dispatch stages: {len(dispatch.stages)}")
    print(f"combine stages: {len(combine.stages)}")
    print(f"modeled total latency: {result.total_time_us:.2f} us")


if __name__ == "__main__":
    main()
