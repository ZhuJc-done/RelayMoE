from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .model import Gpu, HierarchicalTopology, TrafficClass


def build_hierarchical_topology(
    *,
    nodes: int = 2,
    switches_per_node: int = 4,
    gpus_per_switch: int = 4,
    gpu_bandwidth_GBps: float = 64.0,
    pxb_bandwidth_GBps: float = 64.0,
    nic_bandwidth_GBps: float = 100.0,
    network_bandwidth_GBps: float = 400.0,
    nics_per_switch: int = 2,
    per_nic_bandwidth_GBps: float | None = None,
) -> HierarchicalTopology:
    if (
        nodes < 1
        or switches_per_node < 1
        or gpus_per_switch < 1
        or nics_per_switch < 1
    ):
        raise ValueError("topology dimensions must be positive")

    topology = HierarchicalTopology()
    switches_by_node: dict[str, list[str]] = {}
    gpu_index = 0

    # GPUs must be registered before NIC edges because switch-to-node affinity
    # is inferred from the attached GPUs.
    for node_index in range(nodes):
        node = f"n{node_index}"
        switches: list[str] = []
        for switch_index in range(switches_per_node):
            switch = f"{node}s{switch_index}"
            switches.append(switch)
            for _ in range(gpus_per_switch):
                topology.add_gpu(
                    Gpu(name=f"g{gpu_index}", node=node, switch=switch),
                    gpu_bandwidth_GBps=gpu_bandwidth_GBps,
                )
                gpu_index += 1
        switches_by_node[node] = switches

    for switches in switches_by_node.values():
        for left in range(len(switches)):
            for right in range(left + 1, len(switches)):
                topology.add_pxb_pair(
                    switches[left],
                    switches[right],
                    bandwidth_GBps=pxb_bandwidth_GBps,
                )

    for switches in switches_by_node.values():
        for switch in switches:
            topology.add_switch_nic(
                switch,
                bandwidth_GBps=nic_bandwidth_GBps,
                nics_per_switch=nics_per_switch,
                per_nic_bandwidth_GBps=per_nic_bandwidth_GBps,
            )

    node_names = sorted(switches_by_node)
    for left in range(len(node_names)):
        for right in range(left + 1, len(node_names)):
            topology.add_network_pair(
                node_names[left],
                node_names[right],
                bandwidth_GBps=network_bandwidth_GBps,
            )

    topology.validate()
    return topology


@dataclass(frozen=True)
class MoEWorkload:
    """Synthetic post-gating MoE workload and its expert placement."""

    ep_size: int
    top_k: int
    expert_to_gpu: tuple[str, ...]
    traffic_classes: tuple[TrafficClass, ...]

    @property
    def expert_count(self) -> int:
        return len(self.expert_to_gpu)

    @property
    def experts_per_gpu(self) -> dict[str, int]:
        return dict(Counter(self.expert_to_gpu))

    @property
    def average_destination_gpus(self) -> float:
        return sum(
            len(flow.destinations) for flow in self.traffic_classes
        ) / len(self.traffic_classes)


@dataclass(frozen=True)
class TokenRouting:
    """Exact per-token Top-k routing trace with compact expert bitsets."""

    gpu_names: tuple[str, ...]
    expert_to_gpu: tuple[str, ...]
    top_k: int
    tokens_per_gpu: int
    activation_bytes: int
    route_masks: tuple[tuple[int, ...], ...]
    expert_token_bits: tuple[tuple[int, ...], ...]
    seed: int

    @property
    def ep_size(self) -> int:
        return len(self.gpu_names)

    @property
    def expert_count(self) -> int:
        return len(self.expert_to_gpu)

    @property
    def token_count(self) -> int:
        return self.ep_size * self.tokens_per_gpu

    @property
    def expert_assignments(self) -> int:
        return self.token_count * self.top_k

    @property
    def gpu_expert_masks(self) -> tuple[int, ...]:
        gpu_index = {
            gpu_name: index for index, gpu_name in enumerate(self.gpu_names)
        }
        masks = [0] * self.ep_size
        for expert, gpu_name in enumerate(self.expert_to_gpu):
            masks[gpu_index[gpu_name]] |= 1 << expert
        return tuple(masks)

    @property
    def remote_expert_assignments(self) -> int:
        local_masks = self.gpu_expert_masks
        return sum(
            (route_mask & ~local_masks[source]).bit_count()
            for source, source_masks in enumerate(self.route_masks)
            for route_mask in source_masks
        )

    @property
    def remote_destination_deliveries(self) -> int:
        """Total deduplicated remote GPU deliveries sum_{t in T} |D_t|."""

        gpu_index = {
            gpu_name: index for index, gpu_name in enumerate(self.gpu_names)
        }
        expert_gpu_bits = [
            1 << gpu_index[gpu_name] for gpu_name in self.expert_to_gpu
        ]
        total = 0
        for source, source_masks in enumerate(self.route_masks):
            source_bit = 1 << source
            for route_mask in source_masks:
                dest_gpu_mask = 0
                remaining = route_mask
                while remaining:
                    least_bit = remaining & -remaining
                    expert = least_bit.bit_length() - 1
                    dest_gpu_mask |= expert_gpu_bits[expert]
                    remaining ^= least_bit
                dest_gpu_mask &= ~source_bit
                total += dest_gpu_mask.bit_count()
        return total

    def group_traffic_classes(
        self,
        *,
        source_indices: tuple[int, ...] | None = None,
    ) -> tuple[TrafficClass, ...]:
        """Aggregate tokens with identical (source GPU, deduplicated D_t) into TrafficClass.

        Classes are ordered by descending byte volume V_c. Ties in V_c preserve
        natural token arrival order and interleave across source GPUs so no
        source GPU or high-bit destination GPU is systematically biased to the
        tail of online selection.
        """

        sources = (
            tuple(range(self.ep_size))
            if source_indices is None
            else source_indices
        )
        gpu_index = {
            gpu_name: index for index, gpu_name in enumerate(self.gpu_names)
        }
        expert_gpu_bits = [
            1 << gpu_index[gpu_name] for gpu_name in self.expert_to_gpu
        ]
        per_source_counts: list[tuple[int, list[tuple[int, int]]]] = []
        for source in sources:
            source_bit = 1 << source
            counts_for_source: dict[int, int] = {}
            for route_mask in self.route_masks[source]:
                dest_gpu_mask = 0
                remaining = route_mask
                while remaining:
                    least_bit = remaining & -remaining
                    expert = least_bit.bit_length() - 1
                    dest_gpu_mask |= expert_gpu_bits[expert]
                    remaining ^= least_bit
                dest_gpu_mask &= ~source_bit
                if dest_gpu_mask:
                    counts_for_source[dest_gpu_mask] = (
                        counts_for_source.get(dest_gpu_mask, 0) + 1
                    )
            sorted_for_source = sorted(
                counts_for_source.items(),
                key=lambda item: -item[1],
            )
            per_source_counts.append((source, sorted_for_source))

        interleaved: list[tuple[int, int, int]] = []
        max_len = max(
            (len(items) for _, items in per_source_counts),
            default=0,
        )
        for pos in range(max_len):
            for source, items in per_source_counts:
                if pos < len(items):
                    dest_gpu_mask, token_count = items[pos]
                    interleaved.append((source, dest_gpu_mask, token_count))
        interleaved.sort(key=lambda item: -item[2])

        classes: list[TrafficClass] = []
        for class_id, (source, dest_gpu_mask, token_count) in enumerate(
            interleaved
        ):
            destinations: list[str] = []
            remaining = dest_gpu_mask
            while remaining:
                least_bit = remaining & -remaining
                gpu_idx = least_bit.bit_length() - 1
                destinations.append(self.gpu_names[gpu_idx])
                remaining ^= least_bit
            classes.append(
                TrafficClass(
                    name=f"tc{class_id:04d}",
                    source=self.gpu_names[source],
                    destinations=tuple(destinations),
                    volume_bytes=token_count * self.activation_bytes,
                )
            )
        return tuple(classes)

    def tokens_selecting(
        self,
        source_index: int,
        expert_mask: int,
    ) -> int:
        selected_tokens = 0
        remaining = expert_mask
        while remaining:
            least_bit = remaining & -remaining
            expert = least_bit.bit_length() - 1
            selected_tokens |= self.expert_token_bits[source_index][expert]
            remaining ^= least_bit
        return selected_tokens.bit_count()


def generate_token_routing(
    topology: HierarchicalTopology,
    *,
    tokens_per_gpu: int = 4096,
    top_k: int = 8,
    experts: int | None = None,
    activation_bytes: int = 1024,
    seed: int = 7,
    routing_motifs_per_gpu: int = 0,
    motif_skew: float = 0.85,
) -> TokenRouting:
    """Generate per-token Top-k expert choices.

    When ``routing_motifs_per_gpu == 0`` (default), every token samples
    ``top_k`` experts uniformly without replacement. When
    ``routing_motifs_per_gpu > 0``, tokens on each GPU follow a Zipf-skewed
    distribution over ``routing_motifs_per_gpu`` Top-k expert combinations to
    model realistic MoE expert-combination locality.
    """

    gpu_names = tuple(topology.gpus)
    expert_count = experts if experts is not None else len(gpu_names)
    if len(gpu_names) < 2:
        raise ValueError("token routing requires at least two GPUs")
    if tokens_per_gpu < 1:
        raise ValueError("tokens_per_gpu must be positive")
    if expert_count < 1:
        raise ValueError("experts must be positive")
    if not 1 <= top_k <= expert_count:
        raise ValueError("top_k must be in [1, number_of_experts]")
    if activation_bytes < 1:
        raise ValueError("activation_bytes must be positive")
    if routing_motifs_per_gpu < 0:
        raise ValueError("routing_motifs_per_gpu must be non-negative")

    expert_to_gpu = tuple(
        gpu_names[expert % len(gpu_names)]
        for expert in range(expert_count)
    )
    all_experts = tuple(range(expert_count))
    route_masks: list[tuple[int, ...]] = []
    expert_token_bits: list[tuple[int, ...]] = []

    for source_index in range(len(gpu_names)):
        source_rng = random.Random(
            (seed + 1) * 1_000_003 + source_index * 97_409
        )
        source_masks: list[int] = []
        source_expert_bits = [0] * expert_count
        if routing_motifs_per_gpu > 0:
            motifs = [
                source_rng.sample(all_experts, top_k)
                for _ in range(routing_motifs_per_gpu)
            ]
            raw_weights = [
                1.0 / ((idx + 1) ** motif_skew)
                for idx in range(routing_motifs_per_gpu)
            ]
            total_w = sum(raw_weights)
            cdf: list[float] = []
            acc = 0.0
            for w in raw_weights:
                acc += w / total_w
                cdf.append(acc)
            for token_index in range(tokens_per_gpu):
                u = source_rng.random()
                chosen_idx = 0
                for idx, threshold in enumerate(cdf):
                    if u <= threshold:
                        chosen_idx = idx
                        break
                selected = motifs[chosen_idx]
                route_mask = 0
                token_bit = 1 << token_index
                for expert in selected:
                    route_mask |= 1 << expert
                    source_expert_bits[expert] |= token_bit
                source_masks.append(route_mask)
        else:
            for token_index in range(tokens_per_gpu):
                selected = source_rng.sample(all_experts, top_k)
                route_mask = 0
                token_bit = 1 << token_index
                for expert in selected:
                    route_mask |= 1 << expert
                    source_expert_bits[expert] |= token_bit
                source_masks.append(route_mask)
        route_masks.append(tuple(source_masks))
        expert_token_bits.append(tuple(source_expert_bits))

    return TokenRouting(
        gpu_names=gpu_names,
        expert_to_gpu=expert_to_gpu,
        top_k=top_k,
        tokens_per_gpu=tokens_per_gpu,
        activation_bytes=activation_bytes,
        route_masks=tuple(route_masks),
        expert_token_bits=tuple(expert_token_bits),
        seed=seed,
    )


def generate_moe_workload(
    topology: HierarchicalTopology,
    *,
    classes: int = 32,
    top_k: int = 8,
    experts: int | None = None,
    seed: int = 7,
    min_volume_mib: float = 0.5,
    max_volume_mib: float = 4.0,
    hotspot_skew: float = 3.0,
) -> MoEWorkload:
    gpu_names = tuple(topology.gpus)
    if len(gpu_names) < 2:
        raise ValueError("MoE traffic requires at least two GPUs")
    if classes < 1:
        raise ValueError("classes must be positive")
    expert_count = experts if experts is not None else len(gpu_names)
    if expert_count < 1:
        raise ValueError("experts must be positive")
    if not 1 <= top_k <= expert_count:
        raise ValueError("top_k must be in [1, number_of_experts]")
    if min_volume_mib <= 0 or max_volume_mib < min_volume_mib:
        raise ValueError("invalid traffic volume range")
    if hotspot_skew < 0:
        raise ValueError("hotspot_skew must be non-negative")

    rng = random.Random(seed)
    hotspot = gpu_names[-1]
    gpu_order = {gpu: index for index, gpu in enumerate(gpu_names)}
    expert_to_gpu = tuple(
        gpu_names[expert % len(gpu_names)]
        for expert in range(expert_count)
    )
    expert_ids = list(range(expert_count))
    result: list[TrafficClass] = []
    for class_index in range(classes):
        source = gpu_names[class_index % len(gpu_names)]
        weights: list[float] = []
        for target_gpu in expert_to_gpu:
            weight = 1.0
            if target_gpu == hotspot:
                weight += hotspot_skew
            if topology.gpus[target_gpu].node != topology.gpus[source].node:
                weight += 0.75
            if (
                topology.gpus[target_gpu].switch
                == topology.gpus[source].switch
            ):
                weight += 0.25
            weights.append(weight)

        selected_experts = _weighted_sample_without_replacement(
            rng,
            expert_ids,
            weights,
            top_k,
        )
        destinations_set = {
            expert_to_gpu[expert]
            for expert in selected_experts
            if expert_to_gpu[expert] != source
        }
        if not destinations_set:
            replacement = next(
                expert
                for expert, target_gpu in enumerate(expert_to_gpu)
                if target_gpu != source
            )
            selected_experts[-1] = replacement
            destinations_set.add(expert_to_gpu[replacement])
        destinations = tuple(
            sorted(destinations_set, key=gpu_order.__getitem__)
        )
        volume_mib = rng.uniform(min_volume_mib, max_volume_mib)
        result.append(
            TrafficClass(
                name=f"c{class_index:02d}",
                source=source,
                destinations=destinations,
                volume_bytes=max(1, int(volume_mib * 1024 * 1024)),
            )
        )
    return MoEWorkload(
        ep_size=len(gpu_names),
        top_k=top_k,
        expert_to_gpu=expert_to_gpu,
        traffic_classes=tuple(result),
    )


def generate_moe_traffic(
    topology: HierarchicalTopology,
    *,
    classes: int = 32,
    top_k: int = 8,
    experts: int | None = None,
    seed: int = 7,
    min_volume_mib: float = 0.5,
    max_volume_mib: float = 4.0,
    hotspot_skew: float = 3.0,
) -> tuple[TrafficClass, ...]:
    """Compatibility wrapper returning only aggregated traffic classes."""

    return generate_moe_workload(
        topology,
        classes=classes,
        top_k=top_k,
        experts=experts,
        seed=seed,
        min_volume_mib=min_volume_mib,
        max_volume_mib=max_volume_mib,
        hotspot_skew=hotspot_skew,
    ).traffic_classes


def _weighted_sample_without_replacement(
    rng: random.Random,
    values: list[int],
    weights: list[float],
    count: int,
) -> list[int]:
    remaining_values = list(values)
    remaining_weights = list(weights)
    selected: list[int] = []
    for _ in range(count):
        total = sum(remaining_weights)
        threshold = rng.random() * total
        cumulative = 0.0
        selected_index = len(remaining_values) - 1
        for index, weight in enumerate(remaining_weights):
            cumulative += weight
            if threshold <= cumulative:
                selected_index = index
                break
        selected.append(remaining_values.pop(selected_index))
        remaining_weights.pop(selected_index)
    return selected
