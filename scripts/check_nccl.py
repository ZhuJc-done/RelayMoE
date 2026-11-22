"""Run with torchrun for a small all-reduce and point-to-point correctness check."""

from __future__ import annotations

from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        values = torch.full((1024,), float(rank + 1), device="cuda")
        dist.all_reduce(values)
        expected = world * (world + 1) // 2
        assert torch.all(values == expected).item()

        if world > 1:
            send = torch.full((1024,), float(rank), device="cuda")
            recv = torch.empty_like(send)
            requests = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, send, (rank + 1) % world),
                dist.P2POp(dist.irecv, recv, (rank - 1) % world),
            ])
            for request in requests:
                request.wait()
            torch.cuda.synchronize()
            assert torch.all(recv == (rank - 1) % world).item()
        dist.barrier()
        if rank == 0:
            print(json.dumps({"ranks": world, "all_reduce": "passed", "ring_send_recv": "passed"}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
