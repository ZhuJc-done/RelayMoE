"""Real GPU correctness matrix; run using torchrun on 2, 4, then 16 ranks."""
import argparse
from datetime import timedelta
import json
import os
import faulthandler
import signal
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from relaymoe.torch_backend import TorchDistributedBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor", choices=["staged", "event", "ipc", "nccl"], default="staged")
    args = parser.parse_args()
    faulthandler.register(signal.SIGUSR1)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=torch.device("cuda", rank))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    world = dist.get_world_size()
    # Initialize the global NCCL communicator before subset point-to-point calls.
    ping = torch.ones(1, device="cuda")
    dist.all_reduce(ping)
    records = []
    try:
        for strategy in (("direct",) if args.executor == "nccl" else ("direct", "relay", "independent")):
            backend = TorchDistributedBackend(control_group=control, strategy=strategy, executor=args.executor)
            for case in ("multicast", "fanout", "local", "empty_rank", "all_empty", "hotspot"):
                for dtype in (torch.float32, torch.bfloat16):
                    torch.manual_seed(123 + rank)
                    count = 0 if case == "all_empty" or (case == "empty_rank" and rank == 0) else 6
                    experts = world * 2
                    placement = [e // 2 for e in range(experts)]
                    routes = []
                    for token in range(count):
                        if case == "fanout":
                            route = list(range(experts))
                        elif case == "local":
                            route = [rank * 2, rank * 2 + 1]
                        elif case == "hotspot":
                            route = [0, 1, experts - 1]
                        else:
                            route = list(dict.fromkeys([rank * 2, rank * 2 + 1,
                                ((rank + 1) % world) * 2, ((rank + world - 1) % world) * 2 + 1]))
                        routes.append(route)
                    k = len(routes[0]) if routes else 2
                    ids = torch.tensor(routes, device="cuda", dtype=torch.int64).reshape(count, k)
                    weights = torch.full(ids.shape, 1 / k, device="cuda", dtype=torch.float32)
                    hidden = (torch.arange(count * 16, device="cuda", dtype=torch.float32)
                              .reshape(count, 16) / 128 + rank).to(dtype)
                    prepared = backend.prepare(ids, weights, expert_to_rank=placement, hidden_size=16, dtype=dtype)
                    dispatched = backend.execute(plan=prepared.plan, phase=prepared.plan.dispatch, tensors=hidden)
                    dispatch_ok = True
                    for flow in prepared.plan.traffic_classes:
                        if backend.name not in flow.destinations:
                            continue
                        source = backend.rank_by_name[flow.source]
                        indices = torch.tensor(prepared.plan.traffic_layouts[flow.name].token_indices, device="cuda")
                        source_count = len(prepared.routes[source])
                        expected = (torch.arange(source_count * 16, device="cuda", dtype=torch.float32)
                                    .reshape(source_count, 16) / 128 + source).to(dtype).index_select(0, indices)
                        dispatch_ok &= torch.equal(prepared.dispatch_buffers[flow.name], expected)
                    expert_fn = lambda expert, x: (x.float() * (1 + expert / 128) + expert / 64).to(dtype)
                    outputs = backend.experts(dispatched, expert_fn)
                    result = backend.execute(plan=prepared.plan, phase=prepared.plan.combine, tensors=outputs)
                    reference = torch.zeros_like(hidden, dtype=torch.float32)
                    for row, route in enumerate(routes):
                        for expert in route:
                            reference[row] += expert_fn(expert, hidden[row:row + 1])[0].float() / k
                    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
                    good = dispatch_ok and torch.allclose(result.float(), reference, atol=tolerance, rtol=tolerance)
                    max_error = (result.float() - reference).abs().max().item() if count else 0.0
                    report = {"rank": rank, "passed": good, "dispatch_exact": dispatch_ok, "max_abs_error": max_error}
                    reports = [None] * world
                    dist.all_gather_object(reports, report, group=control)
                    if not all(x["passed"] for x in reports):
                        raise AssertionError((strategy, case, str(dtype), reports))
                    if rank == 0:
                        record = {"strategy": strategy, "case": case, "dtype": str(dtype), "world": world,
                                  "max_abs_error": max(x["max_abs_error"] for x in reports),
                                  "passed": True, "actions": prepared.plan.action_count,
                                  "executor": backend.executor_name}
                        records.append(record)
                        print(json.dumps(record), flush=True)
            backend.close()
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"world": world, "passed": True, "cases": records}, indent=2))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
