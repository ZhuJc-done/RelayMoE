"""Distributed CUDA/NCCL reference backend for real RelayMoE correctness tests.

This initial executor is deliberately identified as a staged reference: it
uses globally synchronized conflict-free stages, not the event-driven native
executor required for the final performance claim. All payloads reside on GPU.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable

import torch
import torch.distributed as dist

from .catalog import CandidateTreeCatalog
from .model import Gpu, HierarchicalTopology
from .runtime import RelayMoE, RelayMoEPlan
from .scheduler import PotentialGreedyScheduler, SelectionResult


class IndependentTreeScheduler(PotentialGreedyScheduler):
    """Keep tree search identical while removing cross-flow load feedback."""

    def select(self, traffic):
        selections = [super(IndependentTreeScheduler, self).select([flow]) for flow in traffic]
        trees, arcs, stages, loads = {}, {}, {}, defaultdict(float)
        for selection in selections:
            trees.update(selection.selected_trees)
            arcs.update(selection.arc_physical_edges)
            stages.update(selection.arc_stage_index)
            for edge, value in selection.edge_load_us.items():
                loads[edge] += value
        return SelectionResult(trees, dict(loads),
                               max((p.theta_us for p in selections), default=0),
                               max((p.doubling_rounds for p in selections), default=0),
                               sum(p.candidates_evaluated for p in selections), arcs, stages)


def single_node_topology(world: int) -> HierarchicalTopology:
    """Model GPU grouping, with explicit NODE/SYS pair-specific bandwidths.

    The initial numbers are placeholders pending P2P calibration. Physical
    resources beyond endpoint ports remain a path model, not hardware counters.
    """
    topology = HierarchicalTopology()
    for rank in range(world):
        topology.add_gpu(Gpu(f"g{rank}", "relay2", f"s{rank // 4}"),
                         gpu_bandwidth_GBps=32.0)
    switches = topology.switches()
    for left, a in enumerate(switches):
        for b in switches[left + 1:]:
            same_numa = int(a[1:]) // 2 == int(b[1:]) // 2
            topology.add_pxb_pair(a, b, bandwidth_GBps=24.0 if same_numa else 16.0)
    # Current catalog schema requires NIC metadata even though routes are local.
    for switch in switches:
        topology.add_switch_nic(switch, bandwidth_GBps=100.0, nics_per_switch=2)
    topology.validate()
    return topology


@dataclass
class PreparedRouting:
    plan: RelayMoEPlan
    routes: list
    weights: list
    placement: tuple[int, ...]
    hidden_size: int
    dispatch_buffers: dict[str, torch.Tensor]
    combine_buffers: dict[str, torch.Tensor]
    combine_receives: dict[str, torch.Tensor]
    source_indices: dict[str, torch.Tensor]
    expert_rows: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]
    local_expert_rows: dict[int, tuple[torch.Tensor, torch.Tensor]]
    stages: dict[str, tuple]
    local_count: int
    prepare_us: float


@dataclass
class Dispatched:
    prepared: PreparedRouting
    local_hidden: torch.Tensor


@dataclass
class ExpertOutputs:
    prepared: PreparedRouting
    local_output: torch.Tensor


class TorchDistributedBackend:
    """Reference NCCL transport that implements the RelayMoE backend protocol.

    One process per GPU; use a Gloo metadata group and a CUDA/NCCL default group.
    The full route exchange and CPU planning are intentionally kept visible.
    """

    executor_name = "nccl_staged_reference"

    def __init__(self, topology=None, *, control_group=None, strategy="relay", executor="staged", build_expert_rows=True):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.topology = topology or single_node_topology(self.world)
        self.names = tuple(self.topology.gpus)
        self.rank_by_name = {name: i for i, name in enumerate(self.names)}
        self.name = self.names[self.rank]
        if len(self.names) != self.world:
            raise ValueError("Topology rank count mismatch")
        if strategy not in {"relay", "direct", "independent"}:
            raise ValueError(strategy)
        self.strategy, self.control_group = strategy, control_group
        self.build_expert_rows = build_expert_rows
        if executor not in {"staged", "event", "ipc", "nccl"}:
            raise ValueError(executor)
        if executor == "nccl" and strategy != "direct":
            raise ValueError("Bulk NCCL baseline requires direct strategy")
        self.executor = executor
        self.executor_name = "nccl_staged_reference" if executor == "staged" else "nccl_event_host"
        if executor == "nccl":
            self.executor_name = "nccl_direct_bulk"
        self.ipc_executor = None
        if executor == "ipc":
            from .ipc_backend import IpcExecutor
            self.ipc_executor = IpcExecutor(self.rank, self.world, control_group)
            self.executor_name = "cuda_ipc_native_host"
        self.pair_groups = {}
        self.transfer_streams = {}
        if executor == "event":
            # Pair communicators let independent transfers advance without the
            # global NCCL stream serializing unrelated actions. Group creation
            # order is collective and belongs to initialization, not a step.
            # Split all disjoint pairs in one collective call. Issuing a split
            # per edge makes even excluded ranks clone TCPStore connections;
            # 240 splits at EP16 exhaust the store's file descriptors.
            ring = list(range(self.world))
            if len(ring) % 2:
                ring.append(-1)
            for _ in range(len(ring) - 1):
                pairs = [sorted([ring[i], ring[-1-i]]) for i in range(len(ring)//2)
                         if -1 not in (ring[i], ring[-1-i])]
                for reverse in (False, True):
                    options = dist.ProcessGroupNCCL.Options()
                    options.config.max_ctas = 1
                    options.config.min_ctas = 1
                    options.config.split_share = 1
                    group = dist.split_group(split_ranks=pairs, pg_options=options)
                    mine = next((pair for pair in pairs if self.rank in pair), None)
                    if mine is not None:
                        a, b = mine[::-1] if reverse else mine
                        self.pair_groups[a, b] = group
                        self.transfer_streams[a, b] = torch.cuda.Stream()
                        marker = torch.ones(1, device=self.device)
                        dist.all_reduce(marker, group=group)
                        torch.cuda.synchronize()
                    dist.barrier(group=self.control_group)
                ring = [ring[0], ring[-1], *ring[1:-1]]
            torch.cuda.synchronize()
            dist.barrier(group=self.control_group)
        self.planner = RelayMoE(self.topology, plan_cache_size=0)
        if strategy == "direct":
            self.planner.scheduler.catalog = CandidateTreeCatalog(direct_only=True)
        elif strategy == "independent":
            self.planner.scheduler = IndependentTreeScheduler(
                self.topology, catalog=self.planner.scheduler.catalog)
        self.prepared: PreparedRouting | None = None

    def close(self):
        if self.ipc_executor is not None:
            self.ipc_executor.close()
        for group in self.pair_groups.values():
            dist.destroy_process_group(group)
        self.pair_groups.clear()
        self.transfer_streams.clear()

    def _rows(self, routes, weights):
        rows = defaultdict(list)
        gates = defaultdict(list)
        for row, (experts, values) in enumerate(zip(routes, weights)):
            for expert, weight in zip(experts, values):
                if self._placement[expert] == self.rank:
                    rows[expert].append(row)
                    gates[expert].append(weight)
        return {expert: (torch.tensor(ids, device=self.device, dtype=torch.int64),
                         torch.tensor(gates[expert], device=self.device, dtype=torch.float32))
                for expert, ids in rows.items()}

    def prepare(self, expert_ids, gate_weights, *, expert_to_rank, hidden_size,
                dtype=torch.bfloat16):
        """Gather routing, select a global plan, and allocate exact GPU buffers.

        CPU copies, object exchange and plan broadcast belong to measured online
        preparation; this API does not silently treat them as offline work.
        """
        start = perf_counter_ns()
        if expert_ids.ndim != 2 or expert_ids.shape != gate_weights.shape:
            raise ValueError("Expected aligned [tokens, top_k] tensors")
        self._placement = tuple(int(x) for x in expert_to_rank)
        if any(x < 0 or x >= self.world for x in self._placement):
            raise ValueError("Invalid expert placement")
        local = (expert_ids.detach().cpu().tolist(), gate_weights.detach().cpu().tolist())
        gathered = [None] * self.world
        dist.all_gather_object(gathered, local, group=self.control_group)
        routes, weights = [x[0] for x in gathered], [x[1] for x in gathered]
        packet = [None]
        if self.rank == 0:
            try:
                plan = self.planner.plan_from_gating(
                    routes, expert_to_gpu=[self.names[i] for i in self._placement],
                    activation_bytes=hidden_size * torch.empty((), dtype=dtype).element_size(),
                    expert_weights_by_gpu=weights,
                )
                packet[0] = (plan, None)
            except Exception as error:
                packet[0] = (None, repr(error))
        dist.broadcast_object_list(packet, src=0, group=self.control_group)
        plan, error = packet[0]
        if error:
            raise RuntimeError("Global planning failed: " + error)
        exclusive = frozenset() if self.strategy == "direct" else None
        stages = {}
        if self.executor == "staged":
            for phase in (plan.dispatch, plan.combine):
                schedule = self.planner.scheduler._pack_actions(
                    phase.phase, phase.actions, exclusive_edges=exclusive)
                schedule.validate()
                stages[phase.phase] = schedule.stages
        buffers, accumulators, receives, indices, expert_rows = {}, {}, {}, {}, {}
        for flow in plan.traffic_classes:
            if self.name not in plan.selection.selected_trees[flow.name].nodes():
                continue
            layout = plan.traffic_layouts[flow.name]
            shape = (len(layout.token_indices), hidden_size)
            buffers[flow.name] = torch.empty(shape, device=self.device, dtype=dtype)
            accumulators[flow.name] = torch.empty_like(buffers[flow.name])
            if flow.source == self.name:
                indices[flow.name] = torch.tensor(layout.token_indices, device=self.device, dtype=torch.int64)
            if self.build_expert_rows:
                expert_rows[flow.name] = self._rows(layout.expert_indices, layout.expert_weights)
        for action in plan.combine.actions:
            if action.arc.dst == self.name:
                receives[action.name] = torch.empty_like(buffers[action.traffic_name])
        local_rows = self._rows(routes[self.rank], weights[self.rank]) if self.build_expert_rows else {}
        self.prepared = PreparedRouting(
            plan, routes, weights, self._placement, hidden_size, buffers, accumulators,
            receives, indices, expert_rows, local_rows, stages, len(routes[self.rank]),
            (perf_counter_ns() - start) / 1000,
        )
        if self.ipc_executor is not None:
            self.ipc_executor.prepare(self.prepared, self.name, self.rank_by_name)
            self.prepared.prepare_us = (perf_counter_ns() - start) / 1000
        return self.prepared

    def execute(self, *, plan, phase, tensors):
        if self.prepared is None or self.prepared.plan is not plan:
            raise ValueError("prepare() must create this plan before execution")
        if phase.phase == "dispatch":
            return self.dispatch(tensors)
        return self.combine(tensors)

    def _transfer(self, prepared, phase):
        if self.ipc_executor is not None:
            return self.ipc_executor.execute(prepared, phase, self.strategy)
        if self.executor == "nccl":
            actions = (prepared.plan.dispatch if phase == "dispatch" else prepared.plan.combine).actions
            operations = []
            for action in actions:
                src, dst = self.rank_by_name[action.arc.src], self.rank_by_name[action.arc.dst]
                if self.rank == src:
                    data = (prepared.dispatch_buffers if phase == "dispatch" else prepared.combine_buffers)[action.traffic_name]
                    operations.append(dist.P2POp(dist.isend, data, dst))
                elif self.rank == dst:
                    data = (prepared.dispatch_buffers[action.traffic_name] if phase == "dispatch"
                            else prepared.combine_receives[action.name])
                    operations.append(dist.P2POp(dist.irecv, data, src))
            if operations:
                for work in dist.batch_isend_irecv(operations):
                    work.wait()
            if phase == "combine":
                for action in actions:
                    if action.arc.dst == self.name:
                        prepared.combine_buffers[action.traffic_name].add_(prepared.combine_receives[action.name])
            return
        if self.executor == "event":
            return self._transfer_events(prepared, phase)
        for stage in prepared.stages[phase]:
            operations = []
            for action in stage.actions:
                src, dst = self.rank_by_name[action.arc.src], self.rank_by_name[action.arc.dst]
                if self.rank == src:
                    data = (prepared.dispatch_buffers if phase == "dispatch" else prepared.combine_buffers)[action.traffic_name]
                    operations.append(dist.P2POp(dist.isend, data, dst))
                elif self.rank == dst:
                    data = (prepared.dispatch_buffers[action.traffic_name] if phase == "dispatch"
                            else prepared.combine_receives[action.name])
                    operations.append(dist.P2POp(dist.irecv, data, src))
            if operations:
                for request in dist.batch_isend_irecv(operations):
                    request.wait()
            if phase == "combine":
                for action in stage.actions:
                    if action.arc.dst == self.name:
                        prepared.combine_buffers[action.traffic_name].add_(prepared.combine_receives[action.name])
            # Explicit reference-only barrier preserves modeled shared-edge
            # exclusivity across ranks. Replace with native event dependencies
            # before publishing a work-conserving executor performance claim.
            torch.cuda.synchronize()
            dist.barrier(group=self.control_group)

    def _transfer_events(self, prepared, phase):
        """Central ready/resource scheduler driven by real CUDA completions.

        Gloo control traffic and Python overhead are real online costs. The
        physical path conflicts are model constraints; completion is observed
        on GPU, not inferred from a simulated transfer duration.
        """
        actions = (prepared.plan.dispatch if phase == "dispatch" else prepared.plan.combine).actions
        if not actions:
            return
        by_name = {action.name: i for i, action in enumerate(actions)}
        completion_tag = 151 if phase == "dispatch" else 152
        command_tag = 153 if phase == "dispatch" else 154
        current = torch.cuda.current_stream()
        pending_sends = []
        local_transfers = {}
        reduction_streams = {}
        local_complete = []
        receives = {}
        waiters = ThreadPoolExecutor(max_workers=self.world)
        command_buffer = torch.empty(1, dtype=torch.int64)
        command_work = None
        if self.rank:
            command_work = waiters.submit(dist.irecv(command_buffer, src=0, group=self.control_group, tag=command_tag).wait)

        def launch(index):
            action = actions[index]
            sending = action.arc.src == self.name
            peer = self.rank_by_name[action.arc.dst if sending else action.arc.src]
            pair = (self.rank_by_name[action.arc.src], self.rank_by_name[action.arc.dst])
            stream = self.transfer_streams[pair]
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                if sending:
                    data = (prepared.dispatch_buffers if phase == "dispatch" else prepared.combine_buffers)[action.traffic_name]
                    request = dist.isend(data, dst=peer, group=self.pair_groups[pair])
                else:
                    data = (prepared.dispatch_buffers[action.traffic_name] if phase == "dispatch"
                            else prepared.combine_receives[action.name])
                    request = dist.irecv(data, src=peer, group=self.pair_groups[pair])
                request.wait()  # Inserts stream wait; no CPU device synchronize.
                event = torch.cuda.Event()
                event.record(stream)
                if not sending and phase == "combine":
                    # Distinct incoming edges may complete concurrently. Keep
                    # transport concurrent while serializing writes to each
                    # accumulator; add_ is not an atomic cross-stream reduction.
                    reduction = reduction_streams.setdefault(action.traffic_name, torch.cuda.Stream())
                    reduction.wait_stream(current)
                    reduction.wait_event(event)
                    with torch.cuda.stream(reduction):
                        prepared.combine_buffers[action.traffic_name].add_(data)
                        event = torch.cuda.Event()
                        event.record(reduction)
            local_transfers[index] = (event, request)

        def poll_local():
            for index, (event, request) in list(local_transfers.items()):
                if event.query():
                    del local_transfers[index]
                    if self.rank == 0:
                        local_complete.append(index)
                    else:
                        message = torch.tensor([index], dtype=torch.int64)
                        work = dist.isend(message, dst=0, group=self.control_group, tag=completion_tag)
                        pending_sends.append((work, message))

        if self.rank == 0:
            for peer in range(1, self.world):
                if any(self.names[peer] in (a.arc.src, a.arc.dst) for a in actions):
                    count = sum(self.names[peer] in (a.arc.src, a.arc.dst) for a in actions)
                    buf = torch.empty(1, dtype=torch.int64)
                    receives[peer] = [waiters.submit(dist.irecv(buf, src=peer, group=self.control_group, tag=completion_tag).wait), buf, count]
            waiting = set(range(len(actions)))
            completed, active, occupied = set(), {}, set()
            deps = [set(by_name[name] for name in action.dependencies) for action in actions]
            order = sorted(range(len(actions)), key=lambda i: (-actions[i].remaining_depth,
                -actions[i].priority_load_us, -actions[i].size_bytes, actions[i].name))

            def finish(index):
                active[index] -= 1
                if active[index] == 0:
                    del active[index]
                    completed.add(index)
                    if self.strategy != "direct":
                        occupied.difference_update(actions[index].physical_edges)

            while len(completed) < len(actions):
                for index in order:
                    if index not in waiting or not deps[index].issubset(completed):
                        continue
                    action = actions[index]
                    if self.strategy != "direct" and not occupied.isdisjoint(action.physical_edges):
                        continue
                    waiting.remove(index)
                    active[index] = 2
                    if self.strategy != "direct":
                        occupied.update(action.physical_edges)
                    # Send both endpoints' launch commands before launching
                    # rank 0's NCCL side to avoid blocking its peer's progress.
                    endpoints = [self.rank_by_name[action.arc.src], self.rank_by_name[action.arc.dst]]
                    for endpoint in endpoints:
                        if endpoint:
                            message = torch.tensor([index], dtype=torch.int64)
                            work = dist.isend(message, dst=endpoint, group=self.control_group, tag=command_tag)
                            pending_sends.append((work, message))
                    if 0 in endpoints:
                        launch(index)
                poll_local()
                while local_complete:
                    finish(local_complete.pop())
                for peer, state in list(receives.items()):
                    work, buf, left = state
                    if work.done():
                        work.result()
                        finish(int(buf.item()))
                        left -= 1
                        if left:
                            buf = torch.empty(1, dtype=torch.int64)
                            receives[peer] = [waiters.submit(dist.irecv(buf, src=peer, group=self.control_group, tag=completion_tag).wait), buf, left]
                        else:
                            del receives[peer]
            for peer in range(1, self.world):
                message = torch.tensor([-1], dtype=torch.int64)
                pending_sends.append((dist.isend(message, dst=peer, group=self.control_group, tag=command_tag), message))
        else:
            while True:
                poll_local()
                if command_work.done():
                    command_work.result()
                    index = int(command_buffer.item())
                    if index == -1:
                        break
                    launch(index)
                    command_buffer = torch.empty(1, dtype=torch.int64)
                    command_work = waiters.submit(dist.irecv(command_buffer, src=0, group=self.control_group, tag=command_tag).wait)
        for work, message in pending_sends:
            work.wait()
        for stream in self.transfer_streams.values():
            current.wait_stream(stream)
        for stream in reduction_streams.values():
            current.wait_stream(stream)
        waiters.shutdown()

    def dispatch(self, hidden):
        prepared = self.prepared
        if prepared is None:
            raise RuntimeError("prepare required")
        if hidden.shape != (prepared.local_count, prepared.hidden_size):
            raise ValueError("Hidden shape mismatch")
        for flow in prepared.plan.traffic_classes:
            if flow.source == self.name:
                torch.index_select(hidden, 0, prepared.source_indices[flow.name],
                                   out=prepared.dispatch_buffers[flow.name])
        self._transfer(prepared, "dispatch")
        return Dispatched(prepared, hidden)

    @staticmethod
    def _expert_sum(hidden, rows, expert_fn):
        output = torch.zeros_like(hidden)
        for expert, (indices, weights) in rows.items():
            values = expert_fn(expert, hidden.index_select(0, indices))
            values = (values.float() * weights[:, None]).to(hidden.dtype)
            output.index_add_(0, indices, values)
        return output

    def experts(self, dispatched: Dispatched, expert_fn: Callable):
        prepared = dispatched.prepared
        local_output = self._expert_sum(dispatched.local_hidden, prepared.local_expert_rows, expert_fn)
        for flow in prepared.plan.traffic_classes:
            if flow.name not in prepared.combine_buffers:
                continue
            # Source-local experts are already included in local_output.
            if flow.source == self.name:
                prepared.combine_buffers[flow.name].zero_()
            else:
                prepared.combine_buffers[flow.name].copy_(self._expert_sum(
                    prepared.dispatch_buffers[flow.name], prepared.expert_rows[flow.name], expert_fn))
        return ExpertOutputs(prepared, local_output)

    def combine(self, outputs: ExpertOutputs):
        prepared = outputs.prepared
        self._transfer(prepared, "combine")
        result = outputs.local_output.clone()
        for flow in prepared.plan.traffic_classes:
            if flow.source == self.name:
                result.index_add_(0, prepared.source_indices[flow.name], prepared.combine_buffers[flow.name])
        return result
