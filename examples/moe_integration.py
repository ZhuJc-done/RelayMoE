"""Backend-neutral RelayMoE integration point for an MoE inference layer."""

from __future__ import annotations

from typing import Any

from relaymoe import ExecutionDAG, RelayMoE, RelayMoEPlan
from relaymoe.scenarios import build_hierarchical_topology


class CommunicationBackend:
    """Replace this adapter with CUDA/RDMA kernels in a real deployment."""

    def execute(
        self,
        *,
        plan: RelayMoEPlan,
        phase: ExecutionDAG,
        tensors: Any,
    ) -> Any:
        # The native backend must:
        # 1. launch an action only after all action.dependencies complete;
        # 2. reserve action.physical_edges exclusively until transfer finishes;
        # 3. perform in-place partial reduction for Combine relay actions.
        print(
            f"{phase.phase}: {len(phase.actions)} actions, "
            f"{len(plan.traffic_classes)} traffic classes"
        )
        return tensors


def main() -> None:
    topology = build_hierarchical_topology(
        nodes=2,
        switches_per_node=2,
        gpus_per_switch=2,
    )
    gpu_names = tuple(topology.gpus)
    # Shape: [EP rank][local token][Top-k expert id]. In production, aggregate
    # destination-mask counts on GPU and call RelayMoE.plan with compact classes.
    expert_indices_by_gpu = tuple(
        tuple(
            (
                (rank + token + 1) % len(gpu_names),
                (rank + token + 4) % len(gpu_names),
            )
            for token in range(4)
        )
        for rank in range(len(gpu_names))
    )

    relaymoe = RelayMoE(topology, backend=CommunicationBackend())
    plan = relaymoe.plan_from_gating(
        expert_indices_by_gpu,
        expert_to_gpu=gpu_names,
        activation_bytes=1024,
    )

    dispatched = relaymoe.dispatch(plan, hidden_states="hidden-states")
    # expert_outputs = experts(dispatched)
    relaymoe.combine(plan, expert_outputs=dispatched)


if __name__ == "__main__":
    main()
