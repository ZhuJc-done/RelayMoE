"""Export reproducible PNG/PDF/SVG figures from analyze_results.py output."""
import argparse
import json
import math
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METHODS = {
    'nccl': ('NCCL adapter', '#0072B2', 's'),
    'collway_like': ('CollWay-like (repository)', '#CC79A7', '^'),
    'relay_batch': ('RelayMoE', '#D55E00', 'o'),
    'direct_batch': ('Direct, same executor', '#009E73', 'D'),
    'independent_batch': ('Independent trees', '#666666', 'v'),
}


def geomean(values):
    return math.exp(st.mean(math.log(v) for v in values))


def save(fig, folder, name, caption):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{name}.{extension}', dpi=220, bbox_inches='tight', facecolor='white')
    (folder / f'{name}.caption.txt').write_text(caption + '\n')
    plt.close(fig)


def grouped_curve(ax, rows, method, field, xfield='tokens'):
    selected = [r for r in rows if r['method'] == method]
    groups = {}
    for row in selected:
        groups.setdefault(row[xfield], []).append(row[field])
    if not groups:
        return
    xs = sorted(groups)
    ys = [geomean(groups[x]) for x in xs]
    label, color, marker = METHODS[method]
    ax.errorbar(xs, ys, yerr=[[max(0, y - min(groups[x])) for x, y in zip(xs, ys)],
                            [max(0, max(groups[x]) - y) for x, y in zip(xs, ys)]],
                label=label, color=color, marker=marker, capsize=3, linewidth=1.5, markersize=5)


def plot(data, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    versions = sorted({r['version'] for r in data['method_rows']})
    if len(versions) != 1:
        raise ValueError('Plot one source version at a time')
    rows = data['method_rows']
    sizes = [r for r in rows if r['family'] == 'size']
    counts = sorted({len({q['seed'] for q in sizes if q['tokens'] == r['tokens']}) for r in sizes})
    subtitle = ('Completed data snapshot; seeds per size: ' + '/'.join(map(str, counts))
                + '; error bars: seed range')
    if sizes:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for method in METHODS:
            grouped_curve(axes[0], sizes, method, 'full_p50_ms')
            grouped_curve(axes[1], sizes, method, 'dispatch_algobw_GBs_p50')
        axes[0].set_ylabel('Full communication P50 (ms)')
        axes[1].set_ylabel('Dispatch semantic bandwidth (GB/s per rank)')
        for ax in axes:
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Tokens per rank (top-8)')
        axes[0].legend(fontsize=8)
        fig.suptitle('16-GPU PCIe communication\n' + subtitle, fontsize=11)
        save(fig, output, 'size-latency-bandwidth',
             'Dynamic routes; online prepare included in full time; all measured samples retained. '
             'Each seed contributes its median, aggregated geometrically. The bandwidth numerator is '
             'semantic remote payload, not measured PCIe bus bytes. CollWay-like uses the repository '
             'staged IPC executor. This snapshot includes unfinished seed coverage and unstable points.')
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
        for ax, metric, title in zip(axes, ('full', 'dispatch', 'combine'),
                                     ('Online communication', 'Dispatch', 'Combine')):
            for baseline in ('nccl', 'collway_like', 'direct_batch', 'independent_batch'):
                selected = sorted([r for r in data['aggregate_rows'] if r['family'] == 'size'
                                   and r['baseline'] == baseline and r['metric'] == metric],
                                  key=lambda r: r['tokens'])
                if not selected:
                    continue
                x = [r['tokens'] for r in selected]
                y = [r['speedup_geomean'] for r in selected]
                label, color, marker = METHODS[baseline]
                ax.errorbar(x, y, yerr=[[max(0, r['speedup_geomean'] - r['speedup_seed_min']) for r in selected],
                                      [max(0, r['speedup_seed_max'] - r['speedup_geomean']) for r in selected]],
                            label=label, color=color, marker=marker, linewidth=1.5, capsize=3)
            ax.axhline(1, color='black', linestyle='--', linewidth=.8)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Tokens per rank'); ax.set_title(title)
        axes[0].set_ylabel('Baseline latency / RelayMoE latency')
        axes[0].legend(fontsize=7)
        fig.suptitle('RelayMoE speedups — ' + subtitle, fontsize=10)
        save(fig, output, 'size-speedups',
             'Geometric mean of paired per-seed P50 latency ratios. Values above one favor RelayMoE. '
             'Bars show seed range, not confidence intervals. Transport and full online metrics are separate.')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        for method in ('nccl', 'relay_batch', 'direct_batch', 'independent_batch'):
            grouped_curve(axes[0], sizes, method, 'planner_mean_ms')
            grouped_curve(axes[1], sizes, method, 'selection_mean_ms')
        for ax, title in zip(axes, ('Native planning', 'Tree selection within planning')):
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Tokens per rank')
            ax.set_ylabel('Mean stage latency (ms)'); ax.set_title(title)
        axes[0].legend(fontsize=8)
        fig.suptitle('Online planning cost — ' + subtitle, fontsize=10)
        save(fig, output, 'planning-cost',
             'All measured steps, per-seed mean of stage-wise maximum across ranks. '
             'Selection is part of planning; panels must not be added together or to other phase medians. '
             'No full-plan cache; prebuilt skeleton catalog.')
    selected_best = data['verified_best'] or data['provisional_candidates']
    selected_best = [r for r in selected_best if r['baseline'] in ('nccl', 'collway_like')]
    if selected_best:
        baselines = [b for b in ('nccl', 'collway_like') if any(r['baseline'] == b for r in selected_best)]
        fig, grid = plt.subplots(1, len(baselines), figsize=(5 * len(baselines), 4.1),
                                squeeze=False, constrained_layout=True)
        axes = grid[0]
        for ax, baseline in zip(axes, baselines):
            selected = [r for r in selected_best if r['baseline'] == baseline]
            x = list(range(len(selected)))
            y = [r['speedup_geomean'] for r in selected]
            ax.bar(x, y, color='#D55E00', width=.55, alpha=.85)
            ax.errorbar(x, y, yerr=[[max(0, r['speedup_geomean'] - r['speedup_seed_min']) for r in selected],
                                  [max(0, r['speedup_seed_max'] - r['speedup_geomean']) for r in selected]],
                        fmt='none', color='black', capsize=3)
            ax.axhline(1, color='black', linestyle='--', linewidth=.8)
            ax.set_xticks(x, [dict(full='Online total', dispatch='Dispatch', combine='Combine')[r['metric']]
                              for r in selected])
            ax.set_ylim(0, max([1] + [r['speedup_seed_max'] for r in selected]) * 1.35)
            ax.set_title('vs ' + METHODS[baseline][0])
            for i, r in enumerate(selected):
                ax.text(i, r['speedup_seed_max'] + ax.get_ylim()[1] * .025,
                        f"{r['speedup_geomean']:.2f}x\nT={r['tokens']}, k={r['topk']}\n"
                        f"s={r['skew']}, n={r['n_seeds']}", ha='center', fontsize=8)
        axes[0].set_ylabel('Baseline P50 / RelayMoE P50')
        coverage = all(set(r['seeds']) == set(data['policy']['required_seeds']) for r in selected_best)
        status = ('Verified best' if data['verified_best'] else
                  'Provisional maxima\n' + ('stability criteria not met' if coverage else
                                             'seed coverage incomplete / stability unverified'))
        fig.suptitle(status, fontsize=11)
        save(fig, output, 'best-gains',
             status + '. Configurations are selected independently for each metric and baseline. '
             'T denotes tokens per rank; k top-k; s Zipf skew; n measured seeds. '
             'Each result uses every measured sample in its configuration, never the fastest step. '
             'Verified results require five specified seeds and both stability criteria.')
    skew = [r for r in rows if r['family'] == 'skew' or (r['family'] == 'size' and r['tokens'] == 84)]
    if len({r['skew'] for r in skew}) > 1:
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        for method in METHODS:
            grouped_curve(ax, skew, method, 'full_p50_ms', 'skew')
        ax.set_xlabel('Expert Zipf skew'); ax.set_ylabel('Full communication P50 (ms)'); ax.set_yscale('log')
        ax.legend(fontsize=8); ax.set_title('Dynamic routing at 84 tokens/rank')
        save(fig, output, 'routing-skew', 'Same token count and top-k; semantic bytes may change with skew. '
             'Geometric mean across available seeds, range error bars; all completed data shown.')
    (output / 'source.txt').write_text('Data: plot-data.json\nVersion: ' + versions[0] + '\n')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output or args.data.parent / 'figures')


if __name__ == '__main__':
    main()
