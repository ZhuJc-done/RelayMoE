"""Bounded PCIe/NCCL path calibration with raw per-window timing samples."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist


def summary(values):
    ordered = sorted(values)
    windows = [statistics.mean(values[i:i + 20]) for i in range(0, len(values), 20)]
    tail = windows[-3:]
    return {"p50_us": statistics.median(values), "p95_window_us": ordered[int(.95 * (len(ordered) - 1))],
            "mean_us": statistics.mean(values),
            "cv": statistics.pstdev(values) / statistics.mean(values),
            "window_means_us": windows,
            "tail_window_drift": (max(tail) - min(tail)) / statistics.mean(tail)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    numa = rank // 8
    cpus = {numa * 64 + (rank % 8) * 2, numa * 64 + (rank % 8) * 2 + 1}
    cpus &= os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    world = dist.get_world_size()
    ping = torch.ones(1, device="cuda")
    dist.all_reduce(ping)
    cases = []
    for a, b, kind in [(0, 1, "PIX"), (0, 4, "NODE"), (0, 8, "SYS")]:
        cases.extend([(kind + "_forward", [(a, b)]), (kind + "_reverse", [(b, a)]),
                      (kind + "_bidirectional", [(a, b), (b, a)])])
    cases.extend([("NODE_concurrent", [(0, 4), (1, 5), (8, 12), (9, 13)]),
                  ("SYS_concurrent", [(0, 8), (1, 9), (4, 12), (5, 13)])])
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, pairs in cases:
            if any(max(pair) >= world for pair in pairs):
                continue
            for size in (4096, 65536, 1048576, 16777216):
                sends, receives = {}, {}
                for src, dst in pairs:
                    if rank == src:
                        sends[src, dst] = torch.full((size // 4,), float(src + 1), device="cuda")
                    if rank == dst:
                        receives[src, dst] = torch.empty((size // 4,), device="cuda")
                ops = []
                for src, dst in pairs:
                    if rank == src:
                        ops.append(dist.P2POp(dist.isend, sends[src, dst], dst))
                    if rank == dst:
                        ops.append(dist.P2POp(dist.irecv, receives[src, dst], src))

                def transfer():
                    if ops:
                        for req in dist.batch_isend_irecv(ops):
                            req.wait()

                inner = 10 if size <= 65536 else 1
                gpu_samples, wall_samples = [], []
                for step in range(args.warmup + args.steps):
                    dist.barrier(group=control)
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    clock = time.perf_counter_ns()
                    start.record()
                    for _ in range(inner):
                        transfer()
                    end.record()
                    end.synchronize()
                    wall_us = (time.perf_counter_ns() - clock) / 1000 / inner
                    if step >= args.warmup:
                        gpu_samples.append(start.elapsed_time(end) * 1000 / inner if ops else 0)
                        wall_samples.append(wall_us if ops else 0)
                correct = all(torch.all(data == src + 1).item() for (src, _), data in receives.items())
                collected = [None] * world
                dist.all_gather_object(collected, {"gpu": gpu_samples, "wall": wall_samples, "correct": correct}, group=control)
                if not all(row["correct"] for row in collected):
                    raise AssertionError("P2P payload mismatch")
                if rank == 0:
                    gpu = [max(row["gpu"][i] for row in collected) for i in range(args.steps)]
                    wall = [max(row["wall"][i] for row in collected) for i in range(args.steps)]
                    record = {"case": name, "pairs": pairs, "payload_bytes_per_transfer": size,
                              "inner_repeats": inner, "sample_unit": "per-transfer mean within timing window",
                              "warmup": args.warmup, "steps": args.steps, "gpu_us": gpu, "wall_us": wall,
                              "gpu_summary": summary(gpu), "wall_summary": summary(wall),
                              "aggregate_payload_GBps": len(pairs) * size / statistics.median(gpu) / 1000,
                              "payload_correct": True}
                    records.append(record)
                    print(json.dumps({k: v for k, v in record.items() if k not in {"gpu_us", "wall_us"}}), flush=True)
                    args.output.write_text(json.dumps(records, indent=2))
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
