"""Stress IPC event reuse with asynchronous producers and changing tree depths."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from relaymoe.compact_backend import CompactBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=timedelta(seconds=120))
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    backend = CompactBackend(control, 'relay', executor='batch')
    records = []
    experts = world * 8
    for step in range(args.steps):
        tokens = [1, 9, 64, 17][step % 4]
        hidden = [16, 2048, 2048, 40][step % 4]
        topk = [1, 8, 16, 3][step % 4]
        g = torch.Generator().manual_seed(991 + rank * 10000 + step)
        ids = torch.stack([torch.randperm(experts, generator=g)[:topk] for _ in range(tokens)]).cuda()
        case = step % 8
        if case == 2:
            ids = torch.arange(0, experts, 8, device='cuda').expand(tokens, -1).contiguous()
        elif case == 3:
            ids.fill_(-1)
        elif case == 4:
            ids.fill_(rank * 8)
        elif case == 5 and rank == 0:
            ids.fill_(-1)
        weights = (ids >= 0).float() / ids.shape[1]
        backend.prepare(ids, weights, hidden, experts)
        # Different payload each call, generated on the CUDA stream immediately
        # before dispatch; no producer-side synchronize is inserted here.
        token_values = rank * 16 + ((torch.arange(tokens, device='cuda') + step) % 16)
        x = torch.empty((tokens, hidden), device='cuda', dtype=torch.bfloat16)
        x.copy_(token_values[:, None])
        received = backend.dispatch(x)
        expected = ((backend.expert_global // tokens) * 16 +
                    ((backend.expert_global % tokens + step) % 16)).to(torch.bfloat16)
        exact = torch.all(received == expected[:, None])
        # Each destination GPU contributes once if any selected expert lives
        # there. Small integer contributions are exactly representable in BF16.
        local = (backend.prepared_ids >= rank*8) & (backend.prepared_ids < (rank+1)*8)
        output = torch.empty_like(received)
        output.copy_(local.any(dim=1)[:, None].to(output.dtype) * (rank + 1))
        combined = backend.combine(output)
        destinations = torch.stack([((ids >= r*8) & (ids < (r+1)*8)).any(dim=1) * (r + 1)
                                    for r in range(world)]).sum(dim=0)
        combine_exact = torch.all(combined == destinations[:, None])
        valid = torch.tensor([int(exact.item()), int(combine_exact.item())], device='cuda')
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not bool(torch.all(valid == 1)):
            raise AssertionError((step, tokens, hidden, valid.tolist()))
        if rank == 0:
            records.append({'step': step, 'tokens': tokens, 'hidden': hidden,
                            'topk': ids.shape[1], 'case': case, 'dispatch_exact': True,
                            'combine_exact': True})
    backend.close()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'records': records,
            'source_sha256': {k: os.environ[k] for k in ('RELAYMOE_NATIVE_VARIANT_SHA256',
                                                       'RELAYMOE_BACKEND_VARIANT_SHA256') if k in os.environ}}, indent=2))
        print('Exact dispatch/combine reuse checks passed:', len(records), flush=True)
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
