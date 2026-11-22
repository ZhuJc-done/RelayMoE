"""Compare an isolated native-planner build with the frozen production build.

Does not use GPUs for data or timing; run between formal GPU measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.utils.cpp_extension import load
import relaymoe.compact_backend as compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(allowed)})
    baseline = compact.extension()
    name = 'relaymoe_variant_' + hashlib.sha256(args.source.read_bytes()).hexdigest()[:12]
    candidate = load(name=name, sources=[str(args.source)], extra_cuda_cflags=['-O3'],
                     extra_cflags=['-O3'], verbose=False)
    planners = {}
    original = compact.extension
    for name, extension in [('baseline', baseline), ('candidate', candidate)]:
        compact.extension = lambda ext=extension: ext
        planners[name] = compact.native_planner(16)
    compact.extension = original
    cases = [(1, 8, 256), (84, 8, 256), (670, 8, 256), (512, 8, 128), (84, 1, 256), (256, 15, 128)]
    evidence = []
    for tokens, topk, experts in cases:
        routes = []
        for seed in range(8):
            ids = torch.multinomial(torch.ones(16 * tokens, experts), topk, replacement=False,
                                    generator=torch.Generator().manual_seed(seed)).view(16, tokens, topk)
            if topk == 15:
                ids.fill_(-1)
                ids[0] = torch.arange(1, 16) * 8
            routes.append(ids)
        for mode in [0, 1, 2, 3]:
            for seed, ids in enumerate(routes):
                a = planners['baseline'].prepare(ids, experts, 2048, mode, -1)
                b = planners['candidate'].prepare(ids, experts, 2048, mode, -1)
                assert a.trees() == b.trees(), (tokens, topk, mode, seed, 'trees')
                assert a.capacities == b.capacities, (tokens, topk, mode, seed, 'capacities')
                for rank in [0, 3, 8, 15]:
                    assert a.indices(rank) == b.indices(rank), (tokens, topk, mode, seed, 'indices', rank)
                assert a.dispatch.stages == b.dispatch.stages
                assert a.combine.stages == b.combine.stages
            print('exact match', tokens, topk, experts, mode, flush=True)
        if topk == 15:
            continue
        records = {name: [] for name in planners}
        for block in range(12):
            order = list(planners)
            if block % 2:
                order.reverse()
            for name in order:
                for step in range(10):
                    ids = routes[(block * 10 + step) % len(routes)]
                    started = time.perf_counter_ns()
                    plan = planners[name].prepare(ids, experts, 2048, 1, 0)
                    elapsed = (time.perf_counter_ns() - started) / 1000
                    if block >= 2:
                        records[name].append({'full_us': elapsed, 'grouping_us': plan.grouping_us,
                                              'selection_us': plan.selection_us, 'layout_us': plan.layout_us})
        summary = {name: {field: statistics.median(r[field] for r in rows) for field in rows[0]}
                   for name, rows in records.items()}
        evidence.append({'tokens': tokens, 'topk': topk, 'experts': experts, 'records': records, 'summary': summary})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'candidate_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                                          'baseline_sha256': hashlib.sha256(Path('relaymoe/csrc/ipc.cu').read_bytes()).hexdigest(),
                                          'cases': evidence, 'complete': False}, indent=2))
        print(tokens, topk, summary, flush=True)
    payload = json.loads(args.output.read_text())
    payload['complete'] = True
    payload['exact_match_cases'] = len(cases) * 4 * 8
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
