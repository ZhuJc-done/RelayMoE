"""Aggregate all-rank per-layer timings into matched full-request diagnostics."""
import argparse
import json
from pathlib import Path
import statistics as st


def request_spans(rows, requests, layers=48):
    """A generate call may be scheduled as several forward passes."""
    order = [r['layer'] for r in rows[:layers]]
    if len(set(order)) != layers:
        raise ValueError('Expected one complete ordered MoE layer sequence')
    offset = 0
    spans = []
    for request in requests:
        begin = offset
        tokens = 0
        while tokens < request['input_tokens']:
            block = rows[offset:offset+layers]
            if [r['layer'] for r in block] != order:
                raise ValueError(f'Incomplete or inconsistent forward pass at row {offset}')
            counts = {r['total_tokens'] for r in block}
            if len(counts) != 1 or next(iter(counts)) <= 0:
                raise ValueError(f'Inconsistent token count at row {offset}')
            tokens += next(iter(counts))
            offset += layers
        if tokens != request['input_tokens']:
            raise ValueError('Forward passes do not sum to the generate call input tokens')
        spans.append((begin, offset))
    if offset != len(rows):
        raise ValueError('Unmatched layer samples after the final generate call')
    return spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    if not data['complete']:
        raise ValueError('Wait for a complete model run')
    root = args.result.with_suffix('')
    output = {'scope': 'Per-LLM.generate totals over every scheduled 48-layer forward pass, on the rank with the largest summed full_moe time; includes experts and restore-layout; excludes attention and other model work',
              'aggregation': 'Select one critical rank for each request, then use its components; mean components sum to mean MoE total. These are not GPU busy times.',
              'methods': {}}
    fields = ['prepare_us', 'dispatch_us', 'expert_us', 'combine_us', 'restore_layout_us',
              'full_moe_us', 'metadata_us', 'planner_us', 'grouping_us', 'selection_us',
              'layout_us', 'materialize_us']
    for method, requests in data['records'].items():
        if method == 'native':
            continue
        by_rank = [json.loads((root / f'{method.replace(":", "-")}-rank{rank}.json').read_text())
                   for rank in range(16)]
        layers = 48
        spans = request_spans(by_rank[0], requests, layers)
        for rows in by_rank[1:]:
            assert request_spans(rows, requests, layers) == spans
        records = []
        for i, request in enumerate(requests):
            if request['warmup']:
                continue
            begin, end = spans[i]
            totals = [{field: sum(row[field] for row in rows[begin:end]) / 1000
                       for field in fields} for rows in by_rank]
            rank = max(range(16), key=lambda r: totals[r]['full_moe_us'])
            row = {k.replace('_us', '_ms'): v for k, v in totals[rank].items()}
            row['adapter_setup_ms'] = row['full_moe_ms'] - sum(row[k] for k in
                ['prepare_ms', 'dispatch_ms', 'expert_ms', 'combine_ms', 'restore_layout_ms'])
            assert row['adapter_setup_ms'] >= -.001, row
            row.update(step=request['step'], critical_rank=rank, wall_ms=request['wall_seconds']*1000,
                       forward_passes=(end-begin)//layers)
            records.append(row)
        keys = [key for key in records[0] if key.endswith('_ms')]
        summary = {k: {'mean': st.mean(r[k] for r in records), 'p50': st.median(r[k] for r in records),
                        'p95': sorted(r[k] for r in records)[int(.95*(len(records)-1))]} for k in keys}
        output['methods'][method] = {'summary': summary, 'records': records, 'layers': layers}
        print(method, {k: round(v['mean'], 3) for k, v in summary.items()}, flush=True)
    target = root / 'layer-summary.json'
    target.write_text(json.dumps(output, indent=2))
    print(target, flush=True)


if __name__ == '__main__':
    main()
