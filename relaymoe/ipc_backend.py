"""Single-node native executor using CUDA IPC and shared-memory notifications.

Online routing/plan preparation remains visible in TorchDistributedBackend.
The native executor only replaces per-action Python/Gloo launch and completion.
"""
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

_extension = None


def extension():
    global _extension
    if _extension is None:
        os.environ.setdefault('MAX_JOBS', '4')
        _extension = load(name='relaymoe_ipc',
                          sources=[str(Path(__file__).parent / 'csrc' / 'ipc.cu')],
                          extra_cuda_cflags=['-O3'], extra_cflags=['-O3'], verbose=False)
    return _extension


class IpcExecutor:
    def __init__(self, rank, world, control):
        self.rank, self.world, self.control = rank, world, control
        self.ext = extension()
        token = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(token, src=0, group=control)
        self.path = '/dev/shm/relaymoe-' + token[0]
        if rank == 0:
            self.ext.create_shared(self.path)
        dist.barrier(group=control)
        self.native = self.ext.Executor(self.path, rank, world)
        self.storage = None
        self.capacity = 0
        self.tables = {}

    def close(self):
        dist.barrier(group=self.control)
        self.native.close()
        dist.barrier(group=self.control)
        if self.rank == 0:
            os.unlink(self.path)

    def prepare(self, prepared, name, rank_by_name):
        # One contiguous persistent allocation per rank. IPC descriptors are
        # exchanged only on capacity growth and that cost remains online.
        old = [prepared.dispatch_buffers, prepared.combine_buffers, prepared.combine_receives]
        offsets, size = [], 0
        for buffers in old:
            here = {}
            for key, tensor in buffers.items():
                size = (size + 255) // 256 * 256
                here[key] = (size, tuple(tensor.shape), tensor.dtype)
                size += tensor.numel() * tensor.element_size()
            offsets.append(here)
        grow = torch.tensor([int(size > self.capacity)], dtype=torch.int64)
        dist.all_reduce(grow, op=dist.ReduceOp.MAX, group=self.control)
        if grow.item() or self.storage is None:
            torch.cuda.synchronize()
            self.native.unmap()
            dist.barrier(group=self.control)
            self.capacity = max(1 << 20, 1 << max(0, size - 1).bit_length(), self.capacity)
            # Retain the owning tensor. Per-flow views must not keep dozens
            # of previous allocations alive across MoE layers.
            self.storage = self.ext.allocate(self.capacity)
            handle = self.ext.handle(self.storage)
            handles = [None] * self.world
            dist.all_gather_object(handles, handle, group=self.control)
            self.native.map(handles, self.storage.data_ptr())
        for target, table in zip(old, offsets):
            for key, (offset, shape, dtype) in table.items():
                nbytes = target[key].numel() * target[key].element_size()
                target[key] = self.storage[offset:offset+nbytes].view(dtype).view(shape)
        local_offsets = [{key: value[0] for key, value in table.items()} for table in offsets]
        all_offsets = [None] * self.world
        dist.all_gather_object(all_offsets, local_offsets, group=self.control)
        self.tables = {}
        for phase in (prepared.plan.dispatch, prepared.plan.combine):
            actions = phase.actions
            by_name = {a.name: i for i, a in enumerate(actions)}
            resources = {e: i for i, e in enumerate(sorted({e for a in actions for e in a.physical_edges}))}
            order = sorted(range(len(actions)), key=lambda i: (-actions[i].remaining_depth,
                -actions[i].priority_load_us, -actions[i].size_bytes, actions[i].name))
            rows = []
            for action in actions:
                src, dst = rank_by_name[action.arc.src], rank_by_name[action.arc.dst]
                part = 0 if phase.phase == 'dispatch' else 1
                src_offset = all_offsets[src][part][action.traffic_name]
                dst_offset = all_offsets[dst][part][action.traffic_name]
                scratch = all_offsets[dst][2].get(action.name, 0)
                rows.append((src, dst, src_offset, dst_offset, scratch, action.size_bytes,
                             [by_name[d] for d in action.dependencies],
                             [resources[e] for e in action.physical_edges]))
            self.tables[phase.phase] = self.ext.make_plan(rows, order, len(resources))

    def execute(self, prepared, phase, strategy):
        # All input packing or expert compute must complete before any peer
        # reads its allocation. The collective readiness barrier is inside
        # the measured call; no per-action global barrier is introduced.
        torch.cuda.synchronize()
        dist.barrier(group=self.control)
        dtype = next(iter(prepared.dispatch_buffers.values())).dtype if prepared.dispatch_buffers else torch.float32
        self.native.run(self.tables[phase], phase == 'combine', strategy != 'direct', dtype == torch.bfloat16)
