from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .model import (
    HierarchicalTopology,
    LogicalArc,
    RelayTree,
    TrafficClass,
)


@dataclass(frozen=True)
class BinomialSkeletonArc:
    """One directed domain-level role transfer in a binomial relay tree stage."""

    src_role: tuple[int, int]
    dst_role: tuple[int, int]
    tx_nic_slot: int | None = None
    rx_nic_slot: int | None = None


@dataclass(frozen=True)
class BinomialSkeleton:
    """Minimum-stage (H_c = ceil(log2(|D_c|+1))) domain-level binomial skeleton."""

    m_canon: tuple[int, ...]
    stages: tuple[tuple[BinomialSkeletonArc, ...], ...]
    root_out_deg: int
    active_canon_switches: tuple[int, ...]
    role_order_by_csw: Mapping[int, tuple[int, ...]]
    role_out_degs_by_csw: Mapping[int, tuple[int, ...]]
    nonzero_role_out_degs_by_csw: Mapping[int, tuple[tuple[int, int], ...]]
    pxb_canon_pairs: tuple[tuple[tuple[int, int], int], ...]
    nic_tx_slot_counts: Mapping[int, tuple[tuple[int, int], ...]]
    nic_rx_slot_counts: Mapping[int, tuple[tuple[int, int], ...]]
    max_gpu_pxb_count: int
    max_nic_count: int

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def max_norm_us(
        self,
        volume_bytes: int,
        *,
        gpu_pxb_bytes_per_us: float,
        nic_bytes_per_us: float,
    ) -> float:
        gpu_pxb_us = (
            self.max_gpu_pxb_count * volume_bytes / gpu_pxb_bytes_per_us
            if gpu_pxb_bytes_per_us > 0
            else 0.0
        )
        nic_us = (
            self.max_nic_count * volume_bytes / nic_bytes_per_us
            if self.max_nic_count > 0 and nic_bytes_per_us > 0
            else 0.0
        )
        return max(gpu_pxb_us, nic_us)


def _compile_binomial_skeleton(
    m_canon: tuple[int, ...],
    raw_stages: tuple[
        tuple[tuple[tuple[int, int], tuple[int, int], int | None, int | None], ...],
        ...,
    ],
    nics_per_switch: int,
) -> BinomialSkeleton:
    out_deg: dict[tuple[int, int], int] = defaultdict(int)
    pxb: dict[tuple[int, int], int] = defaultdict(int)
    tx_cnt: dict[int, list[int]] = defaultdict(lambda: [0] * nics_per_switch)
    rx_cnt: dict[int, list[int]] = defaultdict(lambda: [0] * nics_per_switch)
    typed_stages: list[tuple[BinomialSkeletonArc, ...]] = []

    for stage in raw_stages:
        typed_arcs: list[BinomialSkeletonArc] = []
        for u_role, v_role, tx_slot, rx_slot in stage:
            out_deg[u_role] += 1
            if u_role[0] != v_role[0]:
                if tx_slot is None:
                    pxb[(u_role[0], v_role[0])] += 1
                else:
                    tx_cnt[u_role[0]][tx_slot] += 1
                    rx_cnt[v_role[0]][rx_slot] += 1
            typed_arcs.append(
                BinomialSkeletonArc(
                    src_role=u_role,
                    dst_role=v_role,
                    tx_nic_slot=tx_slot,
                    rx_nic_slot=rx_slot,
                )
            )
        typed_stages.append(tuple(typed_arcs))

    root_out_deg = out_deg[(0, -1)]
    active_canon_switches = tuple(
        s for s, count in enumerate(m_canon) if count > 0
    )
    role_order_by_csw: dict[int, tuple[int, ...]] = {}
    role_out_degs_by_csw: dict[int, tuple[int, ...]] = {}
    nonzero_role_out_degs_by_csw: dict[int, tuple[tuple[int, int], ...]] = {}
    for s in active_canon_switches:
        roles = tuple(
            sorted(range(m_canon[s]), key=lambda i: (-out_deg[(s, i)], i))
        )
        degs = tuple(out_deg[(s, i)] for i in roles)
        role_order_by_csw[s] = roles
        role_out_degs_by_csw[s] = degs
        nz = tuple((idx, deg) for idx, deg in enumerate(degs) if deg > 0)
        if nz:
            nonzero_role_out_degs_by_csw[s] = nz

    nic_tx_slot_counts: dict[int, tuple[tuple[int, int], ...]] = {}
    for s, cnts in tx_cnt.items():
        slots = sorted(range(nics_per_switch), key=lambda r: (-cnts[r], r))
        nic_tx_slot_counts[s] = tuple(
            (slot_id, cnts[slot_id]) for slot_id in slots if cnts[slot_id] > 0
        )

    nic_rx_slot_counts: dict[int, tuple[tuple[int, int], ...]] = {}
    for s, cnts in rx_cnt.items():
        slots = sorted(range(nics_per_switch), key=lambda r: (-cnts[r], r))
        nic_rx_slot_counts[s] = tuple(
            (slot_id, cnts[slot_id]) for slot_id in slots if cnts[slot_id] > 0
        )

    max_gpu_count = max(
        [root_out_deg, 1]
        + [degs[0] for degs in role_out_degs_by_csw.values()]
    )
    max_pxb_count = max((cnt for _, cnt in pxb.items()), default=0)
    max_nic_count = max(
        [cnts[0][1] for cnts in nic_tx_slot_counts.values()]
        + [cnts[0][1] for cnts in nic_rx_slot_counts.values()],
        default=0,
    )

    return BinomialSkeleton(
        m_canon=m_canon,
        stages=tuple(typed_stages),
        root_out_deg=root_out_deg,
        active_canon_switches=active_canon_switches,
        role_order_by_csw=role_order_by_csw,
        role_out_degs_by_csw=role_out_degs_by_csw,
        nonzero_role_out_degs_by_csw=nonzero_role_out_degs_by_csw,
        pxb_canon_pairs=tuple(pxb.items()),
        nic_tx_slot_counts=nic_tx_slot_counts,
        nic_rx_slot_counts=nic_rx_slot_counts,
        max_gpu_pxb_count=max(max_gpu_count, max_pxb_count),
        max_nic_count=max_nic_count,
    )


def _build_binomial_skeleton_variant(
    m_canon: tuple[int, ...],
    canon_switch_to_node: tuple[int, ...],
    nics_per_switch: int,
    switch_order: tuple[int, ...],
    extra_cross_node_budget: int = 0,
    prefer_non_root: bool = False,
) -> tuple[
    tuple[tuple[tuple[int, int], tuple[int, int], int | None, int | None], ...],
    ...,
] | None:
    total_targets = sum(m_canon)
    if total_targets == 0:
        return ()
    min_stages = total_targets.bit_length()
    unreached = {
        s: [(s, i) for i in range(m_canon[s])]
        for s in range(len(m_canon))
        if m_canon[s] > 0
    }
    holders: list[tuple[int, int]] = [(0, -1)]
    holders_by_switch: dict[int, list[tuple[int, int]]] = defaultdict(list)
    holders_by_switch[0].append((0, -1))
    order_rank = {s: idx for idx, s in enumerate(switch_order)}
    remaining_extra_cn = extra_cross_node_budget
    out_deg: dict[tuple[int, int], int] = defaultdict(int)
    node_ids = tuple(dict.fromkeys(canon_switch_to_node))

    stages: list[
        tuple[tuple[tuple[int, int], tuple[int, int], int | None, int | None], ...]
    ] = []
    for stage_idx in range(1, min_stages + 1):
        rem_stages = min_stages - stage_idx
        factor = 1 << rem_stages

        used_senders: set[tuple[int, int]] = set()
        used_pxb: set[tuple[int, int]] = set()
        used_nic_tx: dict[int, int] = defaultdict(int)
        used_nic_rx: dict[int, int] = defaultdict(int)
        stage_arcs: list[
            tuple[tuple[int, int], tuple[int, int], int | None, int | None]
        ] = []

        new_h: dict[int, int] = defaultdict(int)
        rem_u = {
            s: len(unreached.get(s, ())) for s in range(len(m_canon))
        }

        def unused_holders_in_switch(s: int) -> int:
            return sum(
                1 for u in holders_by_switch[s] if u not in used_senders
            )

        def used_holders_in_switch(s: int) -> int:
            return sum(
                1 for u in holders_by_switch[s] if u in used_senders
            )

        def switch_capacity_remaining(s: int) -> int:
            return (
                unused_holders_in_switch(s) * (2 * factor - 1)
                + (used_holders_in_switch(s) + new_h[s]) * (factor - 1)
            )

        def switch_deficit(s: int) -> int:
            if rem_u[s] <= switch_capacity_remaining(s):
                return 0
            return math.ceil(
                (rem_u[s] - switch_capacity_remaining(s)) / factor
            )

        def node_capacity_remaining(node_id: int) -> int:
            return sum(
                switch_capacity_remaining(s)
                for s in range(len(m_canon))
                if canon_switch_to_node[s] == node_id
            )

        def node_unreached_remaining(node_id: int) -> int:
            return sum(
                rem_u[s]
                for s in range(len(m_canon))
                if canon_switch_to_node[s] == node_id
            )

        def node_deficit(node_id: int) -> int:
            u_n = node_unreached_remaining(node_id)
            cap_n = node_capacity_remaining(node_id)
            if u_n <= cap_n:
                return 0
            return math.ceil((u_n - cap_n) / factor)

        def can_send_cross_node_from_source_node(
            u: tuple[int, int], s_v: int
        ) -> bool:
            s_u, _ = u
            # Monotone hierarchy: cross-node hops only go from source node 0 to remote node > 0
            if (
                canon_switch_to_node[s_u] != 0
                or canon_switch_to_node[s_v] == 0
            ):
                return False
            if (
                used_nic_tx[s_u] >= nics_per_switch
                or used_nic_rx[s_v] >= nics_per_switch
            ):
                return False
            return (
                node_unreached_remaining(0)
                <= node_capacity_remaining(0) - factor
            )

        def can_link_same_node(s_u: int, s_v: int) -> bool:
            if canon_switch_to_node[s_u] != canon_switch_to_node[s_v]:
                return False
            if s_u == s_v:
                return True
            return (s_u, s_v) not in used_pxb

        def record_arc(
            u: tuple[int, int],
            v: tuple[int, int],
            is_optional_cn: bool = False,
        ) -> None:
            nonlocal remaining_extra_cn
            s_u, _ = u
            s_v, _ = v
            tx_slot: int | None = None
            rx_slot: int | None = None
            if s_u != s_v:
                if canon_switch_to_node[s_u] == canon_switch_to_node[s_v]:
                    used_pxb.add((s_u, s_v))
                else:
                    if is_optional_cn and remaining_extra_cn > 0:
                        remaining_extra_cn -= 1
                    tx_slot = used_nic_tx[s_u]
                    rx_slot = used_nic_rx[s_v]
                    used_nic_tx[s_u] += 1
                    used_nic_rx[s_v] += 1
            used_senders.add(u)
            out_deg[u] += 1
            new_h[s_v] += 1
            rem_u[s_v] -= 1
            stage_arcs.append((u, v, tx_slot, rx_slot))

        if (
            prefer_non_root
            and stage_idx == 1
            and node_unreached_remaining(0) > 0
        ):
            local_cands = [
                s
                for s in switch_order
                if canon_switch_to_node[s] == 0 and rem_u[s] > 0
            ]
            if local_cands:
                local_cands.sort(
                    key=lambda s: (s == 0, order_rank[s], -rem_u[s])
                )
                s_v = local_cands[0]
                if can_link_same_node(0, s_v) and all(
                    node_unreached_remaining(n) <= factor - 1
                    for n in node_ids
                    if n != 0
                ):
                    record_arc((0, -1), unreached[s_v].pop(0))

        # Step 1: Cross-node hops from source node 0 to remote nodes (> 0)
        for target_node in node_ids:
            if target_node == 0:
                continue
            while node_deficit(target_node) > 0 or (
                remaining_extra_cn > 0
                and any(
                    canon_switch_to_node[s] == target_node
                    and rem_u[s] > 0
                    and len(holders_by_switch[s]) + new_h[s] == 0
                    for s in switch_order
                )
            ):
                is_opt = node_deficit(target_node) == 0
                cand_switches = [
                    s
                    for s in switch_order
                    if canon_switch_to_node[s] == target_node and rem_u[s] > 0
                ]
                cand_switches.sort(
                    key=lambda s: (
                        len(holders_by_switch[s]) + new_h[s] > 0,
                        -switch_deficit(s),
                        order_rank[s],
                        -rem_u[s],
                    )
                )
                progressed = False
                for s_v in cand_switches:
                    senders = [
                        u
                        for u in holders
                        if u not in used_senders
                        and can_send_cross_node_from_source_node(u, s_v)
                    ]
                    if not senders:
                        continue
                    senders.sort(
                        key=lambda u: (
                            (1 if u == (0, -1) else 0)
                            if prefer_non_root
                            else (0 if u == (0, -1) else 1),
                            rem_u[u[0]]
                            > switch_capacity_remaining(u[0]) - factor,
                            out_deg[u],
                            order_rank[u[0]],
                            u[1],
                        )
                    )
                    record_arc(
                        senders[0],
                        unreached[s_v].pop(0),
                        is_optional_cn=is_opt,
                    )
                    progressed = True
                    break
                if not progressed:
                    break

        # Step 2: Same-node switch deficits via PXB
        while True:
            deficit_switches = [
                s for s in switch_order if switch_deficit(s) > 0
            ]
            if not deficit_switches:
                break
            deficit_switches.sort(
                key=lambda s: (
                    len(holders_by_switch[s]) + new_h[s] > 0,
                    -switch_deficit(s),
                    order_rank[s],
                    -rem_u[s],
                )
            )
            progressed = False
            for s_v in deficit_switches:
                senders = [
                    u
                    for u in holders
                    if u not in used_senders
                    and u[0] != s_v
                    and can_link_same_node(u[0], s_v)
                    and rem_u[u[0]] <= switch_capacity_remaining(u[0]) - factor
                ]
                if not senders and canon_switch_to_node[s_v] != 0:
                    senders = [
                        u
                        for u in holders
                        if u not in used_senders
                        and can_send_cross_node_from_source_node(u, s_v)
                    ]
                if not senders:
                    continue
                senders.sort(
                    key=lambda u: (
                        0
                        if canon_switch_to_node[u[0]]
                        == canon_switch_to_node[s_v]
                        else 1,
                        (1 if u == (0, -1) else 0) if prefer_non_root else 0,
                        out_deg[u],
                        order_rank[u[0]],
                        u[1],
                    )
                )
                record_arc(senders[0], unreached[s_v].pop(0))
                progressed = True
                break
            if not progressed:
                break

        # Step 3: Seed unseeded same-node switches via PXB
        for s_v in switch_order:
            if (
                rem_u[s_v] > 0
                and len(holders_by_switch[s_v]) + new_h[s_v] == 0
            ):
                senders = [
                    u
                    for u in holders
                    if u not in used_senders
                    and u[0] != s_v
                    and can_link_same_node(u[0], s_v)
                    and rem_u[u[0]] <= switch_capacity_remaining(u[0]) - factor
                ]
                if senders:
                    senders.sort(
                        key=lambda u: (
                            (1 if u == (0, -1) else 0)
                            if prefer_non_root
                            else 0,
                            out_deg[u],
                            order_rank[u[0]],
                            u[1],
                        )
                    )
                    record_arc(senders[0], unreached[s_v].pop(0))

        # Step 4: Intra-switch doubling (zero PXB and zero NIC usage)
        for s_v in switch_order:
            while rem_u[s_v] > 0:
                intra_senders = [
                    u
                    for u in holders
                    if u not in used_senders and u[0] == s_v
                ]
                if not intra_senders:
                    break
                intra_senders.sort(
                    key=lambda u: (
                        (1 if u == (0, -1) else 0) if prefer_non_root else 0,
                        out_deg[u],
                        u[1],
                    )
                )
                record_arc(intra_senders[0], unreached[s_v].pop(0))

        for _, v, _, _ in stage_arcs:
            holders.append(v)
            holders_by_switch[v[0]].append(v)
        stages.append(tuple(stage_arcs))

    if sum(rem_u.values()) > 0:
        return None
    return tuple(stages)


@dataclass
class OfflineBinomialCatalog:
    """Persistable switch-domain binomial skeleton catalog with O(1) lookup.

    ``max_variants=None`` materializes every variant produced by the offline
    enumerator.  A finite limit is intended only for explicit approximation
    experiments and must not be used by the document-aligned planner.
    """

    max_variants: int | None = None
    _cache: dict[
        tuple[tuple[int, ...], int, tuple[int, ...]],
        tuple[BinomialSkeleton, ...],
    ] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.max_variants is not None and self.max_variants < 1:
            raise ValueError("max_variants must be positive or None")

    @property
    def entry_count(self) -> int:
        return len(self._cache)

    @property
    def variant_count(self) -> int:
        return sum(len(skeletons) for skeletons in self._cache.values())

    def contains(
        self,
        m_canon: tuple[int, ...],
        canon_switch_to_node: tuple[int, ...],
        nics_per_switch: int,
    ) -> bool:
        key = (canon_switch_to_node, nics_per_switch, m_canon)
        return key in self._cache

    def prebuild(
        self,
        keys: Iterable[
            tuple[tuple[int, ...], tuple[int, ...], int]
        ],
    ) -> None:
        """Materialize a collection of canonical distribution keys offline."""

        for m_canon, canon_switch_to_node, nics_per_switch in keys:
            self.lookup(
                m_canon,
                canon_switch_to_node,
                nics_per_switch,
            )

    def save(self, path: str | Path) -> None:
        """Persist the materialized static lookup table as deterministic JSON."""

        entries = []
        for (
            canon_switch_to_node,
            nics_per_switch,
            m_canon,
        ), skeletons in sorted(self._cache.items()):
            entries.append(
                {
                    "canon_switch_to_node": list(canon_switch_to_node),
                    "nics_per_switch": nics_per_switch,
                    "m_canon": list(m_canon),
                    "skeletons": [
                        [
                            [
                                [
                                    list(arc.src_role),
                                    list(arc.dst_role),
                                    arc.tx_nic_slot,
                                    arc.rx_nic_slot,
                                ]
                                for arc in stage
                            ]
                            for stage in skeleton.stages
                        ]
                        for skeleton in skeletons
                    ],
                }
            )
        serialized_entries = json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload = {
            "format": "relaymoe-offline-binomial-catalog",
            "version": 1,
            "max_variants": self.max_variants,
            "sha256": hashlib.sha256(
                serialized_entries.encode("utf-8")
            ).hexdigest(),
            "entries": entries,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "OfflineBinomialCatalog":
        """Load a catalog created by :meth:`save` without regenerating trees."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            payload.get("format")
            != "relaymoe-offline-binomial-catalog"
            or payload.get("version") != 1
        ):
            raise ValueError("unsupported RelayMoE offline catalog format")
        entries = payload.get("entries", ())
        expected_checksum = payload.get("sha256")
        actual_checksum = hashlib.sha256(
            json.dumps(
                entries,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if expected_checksum != actual_checksum:
            raise ValueError("RelayMoE offline catalog checksum mismatch")
        catalog = cls(max_variants=payload.get("max_variants"))
        for entry in entries:
            m_canon = tuple(int(value) for value in entry["m_canon"])
            canon_switch_to_node = tuple(
                int(value) for value in entry["canon_switch_to_node"]
            )
            nics_per_switch = int(entry["nics_per_switch"])
            skeletons = []
            for serialized_stages in entry["skeletons"]:
                raw_stages = tuple(
                    tuple(
                        (
                            tuple(int(value) for value in serialized_arc[0]),
                            tuple(int(value) for value in serialized_arc[1]),
                            (
                                None
                                if serialized_arc[2] is None
                                else int(serialized_arc[2])
                            ),
                            (
                                None
                                if serialized_arc[3] is None
                                else int(serialized_arc[3])
                            ),
                        )
                        for serialized_arc in stage
                    )
                    for stage in serialized_stages
                )
                skeletons.append(
                    _compile_binomial_skeleton(
                        m_canon,
                        raw_stages,
                        nics_per_switch,
                    )
                )
            key = (canon_switch_to_node, nics_per_switch, m_canon)
            catalog._cache[key] = tuple(skeletons)
        return catalog

    def lookup(
        self,
        m_canon: tuple[int, ...],
        canon_switch_to_node: tuple[int, ...],
        nics_per_switch: int,
    ) -> tuple[BinomialSkeleton, ...]:
        key = (canon_switch_to_node, nics_per_switch, m_canon)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        switch_count = len(m_canon)
        active_switches = [s for s in range(switch_count) if m_canon[s] > 0]
        local_active = [
            s for s in active_switches if canon_switch_to_node[s] == 0
        ]
        remote_active = [
            s for s in active_switches if canon_switch_to_node[s] != 0
        ]
        remote_orders = (
            [
                remote_active[shift:] + remote_active[:shift]
                for shift in range(len(remote_active))
            ]
            if remote_active
            else [[]]
        )
        local_orders = (
            [
                local_active[shift:] + local_active[:shift]
                for shift in range(len(local_active))
            ]
            if local_active
            else [[]]
        )
        inactive = [s for s in range(switch_count) if m_canon[s] == 0]

        seen: set[tuple] = set()
        skeletons: list[BinomialSkeleton] = []
        for extra_cn in (0, 1):
            for prefer_non_root in (False, True):
                for remote_first in (True, False):
                    for r_idx, r_ord in enumerate(remote_orders):
                        for l_idx in range(len(local_orders)):
                            l_ord = local_orders[
                                (r_idx + l_idx) % len(local_orders)
                            ]
                            switch_order = tuple(
                                (
                                    r_ord + l_ord
                                    if remote_first
                                    else l_ord + r_ord
                                )
                                + inactive
                            )
                            raw = _build_binomial_skeleton_variant(
                                m_canon,
                                canon_switch_to_node,
                                nics_per_switch,
                                switch_order,
                                extra_cross_node_budget=extra_cn,
                                prefer_non_root=prefer_non_root,
                            )
                            if raw is not None and raw not in seen:
                                seen.add(raw)
                                skeletons.append(
                                    _compile_binomial_skeleton(
                                        m_canon, raw, nics_per_switch
                                    )
                                )
                                if (
                                    self.max_variants is not None
                                    and len(skeletons) >= self.max_variants
                                ):
                                    result = tuple(skeletons)
                                    self._cache[key] = result
                                    return result
        result = tuple(skeletons)
        self._cache[key] = result
        return result


@dataclass
class CandidateTreeCatalog:
    """Generate hierarchical and minimum-stage binomial relay-tree candidates."""

    max_candidates: int = 16
    direct_only: bool = False
    offline_catalog: OfflineBinomialCatalog = field(
        default_factory=OfflineBinomialCatalog
    )

    def candidates(
        self,
        topology: HierarchicalTopology,
        traffic: TrafficClass,
    ) -> tuple[RelayTree, ...]:
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")

        raw: list[tuple[str, list[LogicalArc]]] = []
        if not self.direct_only:
            raw.extend(self._binomial_relay_candidates(topology, traffic))
            raw.extend(self._switch_relay_candidates(topology, traffic))
            raw.extend(self._node_relay_candidates(topology, traffic))
        raw.append(
            (
                "direct",
                [LogicalArc(traffic.source, dst) for dst in traffic.destinations],
            )
        )

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

    def _binomial_relay_candidates(
        self,
        topology: HierarchicalTopology,
        traffic: TrafficClass,
    ) -> list[tuple[str, list[LogicalArc]]]:
        switches = topology.switches()
        src_switch = topology.gpus[traffic.source].switch
        src_node = topology.gpus[traffic.source].node
        by_switch: dict[str, list[str]] = defaultdict(list)
        for dst in traffic.destinations:
            by_switch[topology.gpus[dst].switch].append(dst)
        for g_list in by_switch.values():
            g_list.sort()

        local_other = sorted(
            (
                sw
                for sw in switches
                if topology.switch_node(sw) == src_node and sw != src_switch
            ),
            key=lambda sw: (-len(by_switch.get(sw, ())), sw),
        )
        remote_sw = sorted(
            (
                sw
                for sw in switches
                if topology.switch_node(sw) != src_node
            ),
            key=lambda sw: (-len(by_switch.get(sw, ())), sw),
        )
        canon_to_actual = (src_switch, *local_other, *remote_sw)
        node_order = {src_node: 0}
        for sw in remote_sw:
            n_name = topology.switch_node(sw)
            if n_name not in node_order:
                node_order[n_name] = len(node_order)
        canon_switch_to_node = tuple(
            node_order[topology.switch_node(sw)] for sw in canon_to_actual
        )
        m_canon = tuple(len(by_switch.get(sw, ())) for sw in canon_to_actual)
        nics_per_switch = max(
            1,
            len(topology.nic_tx_edges_by_switch.get(src_switch, ())),
        )
        skeletons = self.offline_catalog.lookup(
            m_canon, canon_switch_to_node, nics_per_switch
        )

        candidates: list[tuple[str, list[LogicalArc]]] = []
        for idx, skel in enumerate(skeletons):
            role_map: dict[tuple[int, int], str] = {(0, -1): traffic.source}
            for csw, roles in skel.role_order_by_csw.items():
                actual_sw = canon_to_actual[csw]
                g_list = by_switch[actual_sw]
                for pos, r_local in enumerate(roles):
                    role_map[(csw, r_local)] = g_list[pos]
            arcs: list[LogicalArc] = []
            for stage in skel.stages:
                for arc in stage:
                    self._append_arc(
                        arcs,
                        role_map[arc.src_role],
                        role_map[arc.dst_role],
                    )
            candidates.append((f"binomial-relay-v{idx}", arcs))
        return candidates

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
