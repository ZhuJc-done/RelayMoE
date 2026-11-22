"""Rebuild paper data from complete, version-matched paired experiments.

Best cases are selected across configurations using geometric means of per-seed
P50 ratios. Raw samples are never filtered by their measured latency.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st


METRICS = {
    'full': 'full_communication_us',
    'dispatch': 'dispatch_us',
    'combine': 'combine_us',
}
LABELS = {
    'nccl': 'NCCL all-to-all-v adapter',
    'collway_like': 'Repository CollWay-like (staged IPC)',
    'relay_batch': 'RelayMoE (batch IPC)',
    'direct_batch': 'Direct (same batch IPC)',
    'independent_batch': 'Independent trees (same batch IPC)',
}


def geomean(values):
    return math.exp(st.mean(math.log(v) for v in values))


def quantile(values, q):
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def stability(values):
    windows = [st.mean(values[i:i + 20]) for i in range(0, len(values), 20)
               if len(values[i:i + 20]) == 20]
    tail = windows[-3:]
    return {
        'tail_drift': (max(tail) - min(tail)) / st.mean(tail),
        # Operational definition of the plan's repeated-block fluctuation.
        'block_cv': st.pstdev(windows) / st.mean(windows),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output, required_seeds):
    output.mkdir(parents=True, exist_ok=True)
    raw_rows, comparisons, manifest, pending = [], [], [], []
    seen = set()
    for path in sorted(root.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            pending.append(path.name)
            continue
        if not isinstance(data, dict) or 'summaries' not in data or 'config' not in data:
            continue
        if not data.get('complete'):
            pending.append(path.name)
            continue
        config = data['config']
        family = config['name'].split('-')[0]
        workload = {k: v for k, v in config.items() if k not in ('seed', 'name')}
        workload.setdefault('topk', 8)
        workload.setdefault('experts', 256)
        workload.setdefault('skew', 0)
        version = hashlib.sha256(json.dumps(data['source_sha256'], sort_keys=True).encode()).hexdigest()
        condition = json.dumps({'family': family, **workload}, sort_keys=True)
        seed = config['seed']
        key = (version, condition, seed)
        if key in seen:
            raise ValueError(f'Duplicate workload/seed/version: {path}')
        seen.add(key)
        manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_sha256': data['source_sha256'], 'config': config,
                         'measurement': data['measurement']})
        methods = data['records']
        target = methods.get('relay_batch')
        if not target:
            continue
        reference_steps = [r['step'] for r in target]
        reference_bytes = [r['useful_bytes'] for r in target]
        for method, samples in methods.items():
            if [r['step'] for r in samples] != reference_steps:
                raise ValueError(f'Unpaired step IDs: {path}, {method}')
            if [r['useful_bytes'] for r in samples] != reference_bytes:
                raise ValueError(f'Unmatched semantic bytes: {path}, {method}')
            if len(samples) < 100:
                raise ValueError(f'Fewer than 100 measured steps: {path}, {method}')
            row = {'file': path.name, 'version': version, 'condition': condition,
                   'family': family, **workload, 'seed': seed, 'method': method,
                   'steps': len(samples), 'useful_bytes_per_rank_mean': st.mean(reference_bytes) / 16}
            for metric, field in METRICS.items():
                values = [r[field] for r in samples]
                row[f'{metric}_p50_ms'] = st.median(values) / 1000
                row[f'{metric}_p95_ms'] = quantile(values, .95) / 1000
                for k, v in stability(values).items():
                    row[f'{metric}_{k}'] = v
            for field in ['prepare_us', 'planner_us', 'selection_us', 'metadata_us', 'materialize_us']:
                row[field.replace('_us', '_mean_ms')] = st.mean(r[field] for r in samples) / 1000
            for phase in ['dispatch', 'combine']:
                row[f'{phase}_algobw_GBs_p50'] = st.median(r[f'{phase}_algobw_GBs'] for r in samples)
            raw_rows.append(row)
            if method == 'relay_batch':
                continue
            for metric, field in METRICS.items():
                baseline_values = [r[field] for r in samples]
                relay_values = [r[field] for r in target]
                bs, rs = stability(baseline_values), stability(relay_values)
                comparisons.append({
                    'version': version, 'condition': condition, 'family': family, **workload,
                    'seed': seed, 'baseline': method, 'metric': metric,
                    'speedup': st.median(baseline_values) / st.median(relay_values),
                    'baseline_p50_ms': st.median(baseline_values) / 1000,
                    'relay_p50_ms': st.median(relay_values) / 1000,
                    'p95_speedup': quantile(baseline_values, .95) / quantile(relay_values, .95),
                    'tail_pass': bs['tail_drift'] <= .03 and rs['tail_drift'] <= .03,
                    'block_pass': bs['block_cv'] <= .05 and rs['block_cv'] <= .05,
                })
    groups = {}
    for row in comparisons:
        key = tuple(row[k] for k in ['version', 'condition', 'baseline', 'metric'])
        groups.setdefault(key, []).append(row)
    aggregates = []
    for rows in groups.values():
        first = rows[0]
        ratios = [r['speedup'] for r in rows]
        seeds = sorted(r['seed'] for r in rows)
        speedup = geomean(ratios)
        aggregate = {k: first[k] for k in ['version', 'condition', 'family', 'tokens', 'topk',
                                          'experts', 'skew', 'baseline', 'metric']}
        aggregate.update(seeds=seeds, n_seeds=len(rows), speedup_geomean=speedup,
                         speedup_seed_min=min(ratios), speedup_seed_max=max(ratios),
                         latency_reduction_pct=100 * (1 - 1 / speedup),
                         baseline_p50_ms_geomean=geomean(r['baseline_p50_ms'] for r in rows),
                         relay_p50_ms_geomean=geomean(r['relay_p50_ms'] for r in rows),
                         p95_speedup_geomean=geomean(r['p95_speedup'] for r in rows),
                         all_seeds_tail_pass=all(r['tail_pass'] for r in rows),
                         all_seeds_block_pass=all(r['block_pass'] for r in rows))
        aggregate['eligible_best'] = (set(seeds) == set(required_seeds)
                                      and aggregate['all_seeds_tail_pass']
                                      and aggregate['all_seeds_block_pass'])
        aggregates.append(aggregate)
    best, candidates = [], []
    versions = sorted({r['version'] for r in aggregates})
    for version in versions:
        for baseline in ['nccl', 'collway_like', 'direct_batch', 'independent_batch']:
            for metric in METRICS:
                pool = [r for r in aggregates if r['version'] == version and r['baseline'] == baseline
                        and r['metric'] == metric and r['family'] != 'trace']
                if pool:
                    candidates.append(max(pool, key=lambda r: r['speedup_geomean']))
                verified = [r for r in pool if r['eligible_best']]
                if verified:
                    best.append(max(verified, key=lambda r: r['speedup_geomean']))
    policy = {
        'target': 'relay_batch', 'method_labels': LABELS, 'required_seeds': required_seeds,
        'best_selection': 'Maximum across workload configurations of geometric mean of per-seed P50 baseline/RelayMoE ratios; versions and metrics kept separate',
        'raw_sample_policy': 'All measured steps retained; no latency-dependent trimming',
        'stability': 'Both methods in every seed: last three complete 20-step window means range/mean <=3%; population CV of all complete 20-step window means <=5%',
        'stability_note': 'Live v7 auto-extension tests tail drift only. Block CV is computed here without changing raw runs.',
        'dispersion': 'Seed min/max, not confidence intervals',
        'bandwidth': 'Semantic remote bytes / (16 * phase wall time), GB/s; not PCIe hardware throughput',
        'scope': 'Communication-only; full includes online prepare + dispatch + combine and excludes expert GEMM',
    }
    payload = {'policy': policy, 'completed_configs': len(manifest), 'pending_files': pending,
               'manifest': manifest, 'method_rows': raw_rows, 'paired_seed_rows': comparisons,
               'aggregate_rows': aggregates, 'verified_best': best, 'provisional_candidates': candidates}
    (output / 'plot-data.json').write_text(json.dumps(payload, indent=2))
    (output / 'best-gains.json').write_text(json.dumps({'policy': policy, 'verified_best': best,
                                                      'provisional_candidates': candidates}, indent=2))
    write_csv(output / 'per-seed.csv', raw_rows)
    write_csv(output / 'paired-seed.csv', comparisons)
    write_csv(output / 'aggregate.csv', aggregates)
    lines = ['# RelayMoE result snapshot', '', f'Complete configurations: {len(manifest)}. Pending files: {len(pending)}.',
             '', 'Best selection uses per-seed P50 ratios, never the fastest individual step.',
             'NCCL denotes the matched all-to-all-v adapter; CollWay-like denotes the repository staged IPC implementation.',
             '', '## Verified best configurations', '']
    for title, rows in [('Verified', best), ('Provisional', candidates)]:
        if title == 'Provisional':
            lines += ['', '## Provisional candidates (not final claims)', '']
        if not rows:
            lines += ['No eligible result yet.']
        else:
            lines += ['| Baseline | Metric | T/rank | top-k | skew | seeds | Speedup | Latency reduction |',
                      '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for r in rows:
                lines.append(f"| {LABELS[r['baseline']]} | {r['metric']} | {r['tokens']} | {r['topk']} | {r['skew']} | {r['n_seeds']} | {r['speedup_geomean']:.3f}x | {r['latency_reduction_pct']:.2f}% |")
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed_configs': len(manifest), 'verified_best': len(best),
                      'provisional_candidates': len(candidates), 'output': str(output)}, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', default='123,234,345,456,567')
    args = parser.parse_args()
    analyze(args.root, args.output or args.root / 'analysis', [int(v) for v in args.seeds.split(',')])


if __name__ == '__main__':
    main()
