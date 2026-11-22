"""Attribute instrumented model transport time without trimming any samples."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

from summarize_model_layers import request_spans


def summary(rows, fields):
    return {key: {'mean': st.mean(r[key] for r in rows),
                  'p50': st.median(r[key] for r in rows),
                  'p95': sorted(r[key] for r in rows)[int(.95 * (len(rows) - 1))]}
            for key in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if not data.get('complete') or not data.get('diagnostic_profile'):
        raise ValueError('Require completed diagnostic profile, not a formal benchmark')
    root = args.result.with_suffix('')
    fields = ['prepare_us', 'dispatch_us', 'combine_us', 'full_moe_us', 'planner_us']
    profile_fields = [phase + '_' + part + '_us' for phase in ('dispatch', 'combine')
                      for part in ('device_sync', 'host_barrier', 'launch', 'native', 'python_boundary')]
    fields += profile_fields
    output = {'scope': 'Instrumented diagnostics only; all measured samples retained',
              'definitions': {
                  'device_sync': 'Wall time in cudaDeviceSynchronize; includes GPU work and host scheduling',
                  'host_barrier': 'CPU shared-memory readiness wait after local device synchronization',
                  'launch': 'Wall time in kernel submission, not GPU kernel duration',
                  'python_boundary': 'Outer Python call wall time minus native body, including binding/GIL boundary',
                  'critical_rank': 'Rank with largest summed full_moe time per generate call',
                  'groups': 'Within-method wall-time <=P50 / >=P90, diagnostic comparison only; no sample removal'},
              'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest(), 'methods': {}}
    for method, requests in data['records'].items():
        if method != 'relay:compact_batch':
            continue
        paths = [root / f'{method.replace(":", "-")}-rank{rank}.json' for rank in range(16)]
        ranks = [json.loads(p.read_text()) for p in paths]
        spans = request_spans(ranks[0], requests)
        for rank in ranks[1:]:
            if request_spans(rank, requests) != spans:
                raise ValueError('Unmatched rank layer spans')
        records, by_rank = [], [[] for _ in ranks]
        for i, request in enumerate(requests):
            if request['warmup']:
                continue
            begin, end = spans[i]
            totals = []
            for rank, layers in enumerate(ranks):
                values = {key.replace('_us', '_ms'): sum(row[key] for row in layers[begin:end]) / 1000
                          for key in fields}
                for phase in ('dispatch', 'combine'):
                    residual = values[phase+'_native_ms'] - sum(values[phase+'_'+part+'_ms']
                        for part in ('device_sync', 'host_barrier', 'launch'))
                    if residual < -.001:
                        raise ValueError('Profile components exceed native body')
                    values[phase+'_native_other_ms'] = residual
                    values[phase+'_packing_and_other_ms'] = values[phase+'_ms'] - (
                        values[phase+'_native_ms'] + values[phase+'_python_boundary_ms'])
                    if values[phase+'_packing_and_other_ms'] < -.001:
                        raise ValueError('Transfer profile exceeds phase wall time')
                values.update(rank=rank, step=request['step'], wall_ms=request['wall_seconds']*1000,
                              prompt_index=request['step'] % data['prompt_templates'])
                totals.append(values)
                by_rank[rank].append(values)
            critical = max(totals, key=lambda r: r['full_moe_ms'])
            records.append(critical)
        stats_fields = [key for key in records[0] if key.endswith('_ms')]
        walls = sorted(r['wall_ms'] for r in records)
        p50, p90 = st.median(walls), walls[int(.90 * (len(walls)-1))]
        groups = {}
        for name, selected in [('wall_le_p50', [r for r in records if r['wall_ms'] <= p50]),
                               ('wall_ge_p90', [r for r in records if r['wall_ms'] >= p90])]:
            groups[name] = {'count': len(selected), 'summary': summary(selected, stats_fields)}
        value = {'count': len(records), 'critical_rank_summary': summary(records, stats_fields),
                 'groups': groups, 'thresholds_ms': {'p50': p50, 'p90': p90}, 'records': records,
                 'per_rank_summary': {str(rank): summary(rows, stats_fields)
                                      for rank, rows in enumerate(by_rank)},
                 'layer_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        output['methods'][method] = value
        for group in ['all', 'wall_le_p50', 'wall_ge_p90']:
            stats = value['critical_rank_summary'] if group == 'all' else groups[group]['summary']
            print(group, {key: round(v['mean'], 3) for key, v in stats.items()}, flush=True)
    if not output['methods']:
        raise ValueError('No profiled Relay method')
    target = root / 'batch-profile-summary.json'
    target.write_text(json.dumps(output, indent=2))
    print(target, flush=True)


if __name__ == '__main__':
    main()
