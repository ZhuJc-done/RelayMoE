"""Verify replicated routing reuse matches all-gather, including padded chunks."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from relaymoe.compact_backend import CompactBackend


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=timedelta(seconds=120))
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    experts = world * 8
    for method, executor in [('relay', 'batch'), ('direct', 'nccl')]:
        gathered = CompactBackend(control, method, executor=executor)
        replicated = CompactBackend(control, method, executor=executor)
        for total in [1, world - 1, world, world + 3, world * 8]:
            chunk = (total + world - 1) // world
            torch.manual_seed(123)
            ids = torch.stack([torch.randperm(experts, device='cuda')[:8] for _ in range(total)])
            weights = torch.full(ids.shape, 1 / 8, dtype=torch.float32, device='cuda')
            global_hidden = torch.randn(total, 16, dtype=torch.bfloat16, device='cuda')
            local_ids = torch.full((chunk, 8), -1, dtype=torch.int64, device='cuda')
            local_weights = torch.zeros((chunk, 8), device='cuda')
            hidden = torch.zeros(chunk, 16, dtype=torch.bfloat16, device='cuda')
            begin = rank * chunk
            count = max(0, min(chunk, total - begin))
            local_ids[:count] = ids[begin:begin+count]
            local_weights[:count] = weights[begin:begin+count]
            hidden[:count] = global_hidden[begin:begin+count]
            gathered.prepare(local_ids, local_weights, 16, experts)
            replicated.prepare(local_ids, local_weights, 16, experts, replicated_routing=(ids, weights))
            assert torch.equal(gathered.all_ids, replicated.all_ids)
            assert torch.equal(gathered.all_weights, replicated.all_weights)
            assert gathered.plan.trees() == replicated.plan.trees()
            assert gathered.plan.indices(rank) == replicated.plan.indices(rank)
            a, b = gathered.dispatch(hidden), replicated.dispatch(hidden)
            assert torch.equal(a, b)
            # Same expert-like output verifies combine and original token ordering.
            x = gathered.combine(a * 0.5)
            y = replicated.combine(b * 0.5)
            assert torch.equal(x, y)
            if rank == 0:
                print('replicated routing exact', method, total, flush=True)
        gathered.close()
        replicated.close()
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
