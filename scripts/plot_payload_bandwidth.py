"""Export payload/P50 bandwidth figures with raw-file provenance.

Each invocation handles one frozen source version. Dynamic payload is summarized
as mean semantic remote bytes per rank; it is not measured PCIe bus traffic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from analyze_results import METRICS, geomean, quantile, stability, write_csv
from plot_results import METHODS


PHASES = ('dispatch', 'combine', 'full')
MAIN = ('relay_batch', 'collway_like', 'nccl')
TITLES = ('Dispatch', 'Combine', 'Online round trip')


def extract(root):
    analysis_path = root / 'analysis' / 'plot-data.json'
    analysis = json.loads(analysis_path.read_text())
    rows, manifest = [], []
    versions = {row['version'] for row in analysis['method_rows']}
    if len(versions) != 1:
        raise ValueError('Export exactly one frozen source version at a time')
    for entry in analysis['manifest']:
        if entry['config']['name'].split('-')[0] not in ('size', 'pilot'):
            continue
        path = root / entry['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Raw file changed since analysis: {path}')
        data = json.loads(path.read_text())
        if not data['complete']:
            raise ValueError(f'Incomplete experiment: {path}')
        config = data['config']
        if config.get('skew', 0) != 0 or config.get('topk', 8) != 8 or 'trace' in config:
            raise ValueError('This figure is a uniform-expert dynamic top-8 size sweep')
        target = data['records']['relay_batch']
        reference = [(r['step'], r['useful_bytes']) for r in target]
        if len({r['step'] for r in target}) != len(target) or len(target) < 100:
            raise ValueError(f'Invalid measured step coverage: {path}')
        payloads = [r['useful_bytes'] / 16 for r in target]
        payload = st.mean(payloads)
        for method, samples in data['records'].items():
            if [(r['step'], r['useful_bytes']) for r in samples] != reference:
                raise ValueError(f'Unpaired workload: {path}, {method}')
            if any(len(r['rank_sent_bytes']) != 16 for r in samples):
                raise ValueError('Only the recorded 16-rank experiment is supported')
            for phase, field in METRICS.items():
                times = [r[field] for r in samples]
                if min(times) <= 0:
                    raise ValueError('Nonpositive latency')
                p50, p95 = st.median(times), quantile(times, .95)
                status = stability(times)
                factor = 2 if phase == 'full' else 1
                rows.append({
                    'file': path.name, 'raw_sha256': digest,
                    'source_version_sha256': next(iter(versions)),
                    'tokens_per_rank': config['tokens'], 'hidden_size': 2048,
                    'dtype': 'bfloat16', 'input_activation_bytes_per_rank': config['tokens'] * 4096,
                    'topk': config.get('topk', 8), 'experts': config.get('experts', 256),
                    'seed': config['seed'], 'warmup': data['warmup'], 'steps': len(samples),
                    'method': method, 'phase': phase,
                    'payload_bytes_per_rank_mean': payload,
                    'payload_bytes_per_rank_min': min(payloads),
                    'payload_bytes_per_rank_max': max(payloads),
                    'payload_KiB_per_rank': payload / 1024,
                    'latency_p50_ms': p50 / 1000, 'latency_p95_ms': p95 / 1000,
                    'bandwidth_numerator_factor': factor,
                    'bandwidth_payload_over_p50_GBs': factor * payload / (p50 * 1000),
                    'bandwidth_payload_over_p95_GBs': factor * payload / (p95 * 1000),
                    'prepare_mean_ms': st.mean(r['prepare_us'] for r in samples) / 1000,
                    'planner_mean_ms': st.mean(r['planner_us'] for r in samples) / 1000,
                    **status,
                    'stable': status['tail_drift'] <= .03 and status['block_cv'] <= .05,
                })
        manifest.append({**entry, 'path': str(path.resolve()),
                         'runner_sha256': data.get('runner_sha256'),
                         'measured_steps': len(target), 'warmup': data['warmup']})
    if not rows:
        raise ValueError('No completed measurements')
    if len({r['experts'] for r in rows}) != 1:
        raise ValueError('Do not mix expert counts in a size sweep')
    return rows, manifest, analysis


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row['tokens_per_rank'], row['method'], row['phase'])
        groups.setdefault(key, []).append(row)
    result = []
    for (tokens, method, phase), values in sorted(groups.items()):
        seeds = sorted(r['seed'] for r in values)
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed in a plotted condition')
        p50bw = [r['bandwidth_payload_over_p50_GBs'] for r in values]
        result.append({
            'tokens_per_rank': tokens, 'method': method, 'phase': phase,
            'seeds': ','.join(map(str, seeds)), 'n_seeds': len(values),
            'payload_KiB_per_rank': st.mean(r['payload_KiB_per_rank'] for r in values),
            'latency_p50_ms_geomean': geomean(r['latency_p50_ms'] for r in values),
            'latency_p95_ms_geomean': geomean(r['latency_p95_ms'] for r in values),
            'bandwidth_GBs_geomean': geomean(p50bw),
            'bandwidth_seed_min': min(p50bw), 'bandwidth_seed_max': max(p50bw),
            'bandwidth_p95_GBs_geomean': geomean(r['bandwidth_payload_over_p95_GBs'] for r in values),
            'all_seeds_stable': all(r['stable'] for r in values),
        })
    return result


def save(fig, output, name, caption):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    (output / f'{name}.caption.txt').write_text(caption + '\n')
    plt.close(fig)


def plot(rows, output, label, methods, name):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    styles = {'relay_batch': '-', 'collway_like': '--', 'nccl': '-.',
              'direct_batch': ':', 'independent_batch': '--'}
    single_seed = all(r['n_seeds'] == 1 for r in rows)
    for ax, phase, title, letter in zip(axes, PHASES, TITLES, 'abc'):
        for method in methods:
            selected = sorted((r for r in rows if r['method'] == method and r['phase'] == phase),
                              key=lambda r: r['payload_KiB_per_rank'])
            if not selected:
                continue
            x = [r['payload_KiB_per_rank'] for r in selected]
            y = [r['bandwidth_GBs_geomean'] for r in selected]
            method_label, color, marker = METHODS[method]
            ax.plot(x, y, label=method_label, color=color, marker=marker, markersize=5,
                    markerfacecolor='white', linewidth=1.7, linestyle=styles[method])
            if single_seed:
                lower = [r['bandwidth_p95_GBs_geomean'] for r in selected]
                ax.fill_between(x, lower, y, color=color, alpha=.1, linewidth=0)
            else:
                ax.errorbar(x, y,
                            yerr=[[v-r['bandwidth_seed_min'] for v, r in zip(y, selected)],
                                  [r['bandwidth_seed_max']-v for v, r in zip(y, selected)]],
                            fmt='none', ecolor=color, capsize=4, linewidth=1.2)
        ax.set_xscale('log', base=2)
        xs = sorted({r['payload_KiB_per_rank'] for r in rows if r['phase'] == phase})
        if len(xs) == 1:
            ax.set_xlim(xs[0] * .85, xs[0] * 1.18)
            ax.set_xticks(xs)
        else:
            ax.set_xticks([2 ** p for p in range(4, 17, 2) if xs[0] / 2 <= 2 ** p <= xs[-1] * 2])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.minorticks_off()
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Mean remote payload/rank (KiB)')
        ax.set_ylabel('Effective bandwidth (GB/s per rank)')
        ax.set_title(f'({letter}) {title}' + (' (2U / P50)' if phase == 'full' else ' (U / P50)'), fontsize=10)
        ax.grid(axis='y', alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=min(len(handles), 3), frameon=False, fontsize=9)
    counts = '/'.join(str(n) for n in sorted({r['n_seeds'] for r in rows}))
    extra = ('1 seed; shading: P50–P95 latency envelope' if single_seed else
             f'{counts} seeds; whiskers: seed min–max')
    fig.text(.5, -.01, f'16 GPUs, one PCIe node | Dynamic top-8 routing | {label} preview | {extra}',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.065, right=.99, top=.77, bottom=.19, wspace=.36)
    caption = (
        f'{label}: dynamic synthetic MoE routing on one 16-GPU PCIe node; BF16, hidden size 2048, '
        '256 experts, top-8, uniform expert probabilities. Each point retains all measured steps. '
        'U = mean across measured steps of global semantic remote bytes / 16; it counts one hidden '
        'vector per remote destination GPU, excludes local traffic and metadata, and is identical '
        'between methods at each paired step. This is not input activation size or hardware bus traffic. '
        'Dispatch and combine use U/P50 phase wall time. Online round trip uses 2U/P50 full wall time, '
        'including online prepare, dispatch and combine but excluding expert GEMM. Phase maxima are '
        'taken across ranks before percentiles; separate medians are not additive. GB is decimal; KiB is binary. '
        'P50 bandwidth is a ratio of mean payload to median latency, not the median of per-step bandwidth. '
        + ('One routing seed; shaded region spans U/P95 to U/P50 (2U for full), showing within-run latency '
           'variation, not a confidence interval or independent-run uncertainty. '
           if single_seed else
           'Per-seed values are aggregated geometrically; x is the arithmetic mean payload across seeds. '
           'Whiskers show min/max across seed estimates, not confidence intervals. ')
        + 'CollWay-like is the repository staged IPC implementation, not the full CollWay system. '
        'NCCL is the matched sparse all-to-all-v adapter. Stability outcomes and source hashes are '
        'in the CSV and manifest; this preview is not a validated-best claim.'
    )
    save(fig, output, name, caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'analysis' / 'payload-figures'
    output.mkdir(parents=True, exist_ok=True)
    rows, manifest, analysis = extract(args.root)
    aggregates = aggregate(rows)
    write_csv(output / 'key-data.csv', rows)
    write_csv(output / 'curve-data.csv', aggregates)
    write_csv(output / 'paired-speedups.csv',
              [r for r in analysis['aggregate_rows'] if r['family'] in ('size', 'pilot')])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    plot(aggregates, output, args.label, MAIN, 'payload-bandwidth')
    plot(aggregates, output, args.label,
         ('relay_batch', 'direct_batch', 'independent_batch'), 'payload-bandwidth-ablations')
    (output / 'manifest.json').write_text(json.dumps({
        'label': args.label, 'inputs': manifest,
        'analysis_sha256': hashlib.sha256((args.root / 'analysis/plot-data.json').read_bytes()).hexdigest(),
        'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plot_dependency_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('analyze_results.py', 'plot_results.py')},
        'raw_sample_policy': 'All measured samples retained; warmup excluded by benchmark writer',
        'bandwidth': 'Mean remote semantic bytes/rank / P50 phase seconds / 1e9; full numerator multiplied by 2',
        'source_versions_mixed': False, 'matplotlib': matplotlib.__version__,
    }, indent=2))
    print(json.dumps({'output': str(output), 'raw_files_verified': len(manifest),
                      'key_data_rows': len(rows), 'curve_rows': len(aggregates)}))


if __name__ == '__main__':
    main()
