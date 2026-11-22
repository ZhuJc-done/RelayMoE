"""Plot static fan-out and full-model results, with distinct timing scopes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_results import METHODS, save


def fanout(path, output):
    rows = json.loads(path.read_text())
    patterns = sorted({r['pattern'] for r in rows})
    payloads = sorted({r['payload_bytes'] for r in rows})
    output.mkdir(parents=True, exist_ok=True)
    flat = []
    for pattern in patterns:
        fig, axes = plt.subplots(len(payloads), 3, figsize=(11, 3 * len(payloads)),
                                 squeeze=False, constrained_layout=True)
        for i, payload in enumerate(payloads):
            selected = [r for r in rows if r['pattern'] == pattern and r['payload_bytes'] == payload]
            # At fan-out 15 both destination orderings are the same set.
            if not any(r['destinations'] == 15 for r in selected):
                selected += [r for r in rows if r['destinations'] == 15 and r['payload_bytes'] == payload]
            for method in ('nccl', 'direct_batch', 'relay_batch'):
                samples = sorted([r for r in selected if r['method'] == method], key=lambda r: r['destinations'])
                label, color, marker = METHODS[method]
                xs = [r['destinations'] for r in samples]
                for ax, phase in zip(axes[i, :2], ('dispatch', 'combine')):
                    ax.plot(xs, [r[phase]['median_us'] / 1000 for r in samples],
                            label=label, color=color, marker=marker)
                    ax.set_ylabel(phase.capitalize() + ' P50 (ms)'); ax.set_yscale('log')
                axes[i, 2].plot(xs, [r['source_injection_bytes'][0] / 1048576 for r in samples],
                                label=label, color=color, marker=marker)
                for r in samples:
                    flat.append({'pattern': pattern, 'payload_bytes': payload, 'method': method,
                                 'destinations': r['destinations'], 'dispatch_p50_us': r['dispatch']['median_us'],
                                 'combine_p50_us': r['combine']['median_us'],
                                 'source_bytes': r['source_injection_bytes'][0],
                                 'dispatch_stable': r['dispatch'].get('stable'),
                                 'combine_stable': r['combine'].get('stable')})
            for ax in axes[i]:
                ax.set_xlabel('Destination GPUs'); ax.set_xticks([1, 2, 4, 8, 15])
            axes[i, 2].set_ylabel('Source injection per dispatch (MiB)')
            axes[i, 0].set_title(f'{payload / 1024:g} KiB source payload')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'Static fan-out, {pattern.replace("_", " ")} destinations', fontsize=12)
        save(fig, output, 'fanout-' + pattern,
             'Plan constructed once before timing; source rank 0; no expert GEMM. '
             'Wall time includes packing, transport and required synchronization. All measured steps retained. '
             'Source injection is software-counted logical bytes, not PCIe hardware counters. '
             'The 15-destination point is shared because both patterns produce the same destination set.')
    with (output / 'fanout-data.csv').open('w', newline='') as stream:
        if flat:
            writer = csv.DictWriter(stream, list(flat[0])); writer.writeheader(); writer.writerows(flat)
    (output / 'fanout-provenance.json').write_text(json.dumps({
        'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sha256': rows[0].get('source_sha256') if rows else None}, indent=2))


def models(paths, output):
    output.mkdir(parents=True, exist_ok=True)
    flat = []
    for path in paths:
        data = json.loads(path.read_text())
        if (not data.get('complete') or data.get('logits_capture') or data.get('boundary_capture')
                or data.get('replicated_routing_validation') or data.get('diagnostic_profile')):
            continue
        groups = data['records']
        reference = None
        for method, samples in groups.items():
            samples = [r for r in samples if not r['warmup']]
            keys = [(r['step'], r['input_tokens']) for r in samples]
            if reference is None:
                reference = keys
            if keys != reference:
                raise ValueError(f'Unpaired model steps in {path}')
            values = [r['wall_seconds'] * 1000 for r in samples]
            flat.append({'file': path.name, 'batch': data['batch'], 'method': method,
                         'tokens': data['tokens_per_request'], 'steps': len(samples),
                         'p50_ms': st.median(values), 'p95_ms': sorted(values)[int(.95*(len(values)-1))],
                         'stable': data['stable_by_method'][method],
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    labels = {'native': 'Native vLLM\nreplicated tokens',
              'direct:compact_nccl': 'NCCL adapter\nsparse token chunks',
              'relay:compact_batch': 'RelayMoE batch\nsparse token chunks'}
    for batch in sorted({r['batch'] for r in flat}):
        rows = [r for r in flat if r['batch'] == batch]
        fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
        xs = list(range(len(rows)))
        colors = ['#D55E00' if r['method'].startswith('relay:') else '#0072B2' for r in rows]
        ax.bar(xs, [r['p50_ms'] for r in rows], color=colors, width=.6)
        ax.scatter(xs, [r['p95_ms'] for r in rows], color='black', marker='_', s=100, label='P95')
        for x, r in zip(xs, rows):
            ax.text(x, max(r['p50_ms'], r['p95_ms']) * 1.035,
                    f"P50 {r['p50_ms']:.1f}\n{'stable' if r['stable'] else 'variable'}", ha='center', fontsize=9)
        ax.set_xticks(xs, [labels.get(r['method'], r['method']) for r in rows])
        ax.set_ylabel('Full LLM.generate latency (ms)')
        ax.set_ylim(0, max(r['p95_ms'] for r in rows) * 1.22)
        ax.legend(); ax.set_title(f'Qwen3-30B-A3B, batch {batch}, {rows[0]["tokens"]}+1 tokens')
        save(fig, output, f'qwen3-batch{batch}',
             'Full pretrained model, TP16/DP1/EP16, BF16, eager, prefix cache off. '
             'Repeated prompt templates, paired steps and rotated method blocks. P50 bars, P95 ticks. '
             'Native vLLM uses replicated token input plus all-reduce; adapters use sparse token chunks '
             'plus an all-gather restoring the TP layout. Logits comparisons are outside this run.')
    if flat:
        with (output / 'model-data.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print('Plotted', len(flat), 'completed model method/config rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fanout', type=Path)
    parser.add_argument('--models', type=Path, nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    if args.fanout:
        fanout(args.fanout, args.output)
    if args.models:
        models(args.models, args.output)


if __name__ == '__main__':
    main()
