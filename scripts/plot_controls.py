"""Plot top-1 controls and static real-route replay from paired result analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_results import METHODS, save


def plot(data, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = data['method_rows']
    if len({r['version'] for r in rows}) != 1:
        raise ValueError('Controls must use one source version')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    for family, name, title in [('top1', 'top1-controls', 'Top-1: no shared remote destinations'),
                                 ('trace', 'real-route-replay', 'Static Qwen3 routing replay')]:
        selected = [r for r in rows if r['family'] == family]
        case_order = {r['file']: int(r['tokens'] if family == 'top1'
                                   else r['layer'].split('.')[2]) for r in selected}
        cases = sorted(case_order, key=case_order.get)
        if not cases:
            continue
        fig, axes = plt.subplots(1, len(cases), figsize=(4.3 * len(cases), 4.1),
                                 squeeze=False, constrained_layout=True)
        for ax, case in zip(axes[0], cases):
            group = {r['method']: r for r in selected if r['file'] == case}
            labels = []
            for i, method in enumerate(METHODS):
                if method not in group:
                    raise ValueError(f'Missing {method} in {case}')
                row = group[method]
                label, color, _ = METHODS[method]
                labels.append({'nccl': 'NCCL', 'collway_like': 'CollWay-like',
                               'relay_batch': 'RelayMoE', 'direct_batch': 'Direct',
                               'independent_batch': 'Independent'}[method])
                ax.scatter(i, row['full_p50_ms'], color=color, marker='o', s=55, label=label)
                ax.plot([i, i], [row['full_p50_ms'], row['full_p95_ms']], color=color, alpha=.5)
                ax.scatter(i, row['full_p95_ms'], marker='_', color='black', s=80)
                ax.annotate(f"{row['full_p50_ms']:.2f}", (i, row['full_p50_ms']),
                            xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
            row = next(iter(group.values()))
            subtitle = (f"T={row['tokens']}/rank" if family == 'top1'
                        else f"Layer {row['layer'].split('.')[2]} · {row['tokens']}/rank")
            ax.set_title(subtitle)
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right')
            ax.set_ylabel('Full communication latency (ms, log scale)')
            ax.set_yscale('log')
            ax.set_ylim(bottom=min(r['full_p50_ms'] for r in group.values()) * .55,
                        top=max(r['full_p95_ms'] for r in group.values()) * 1.7)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        scope = 'one workload seed' if family == 'top1' else 'one captured trace per layer'
        fig.suptitle(title + '\nP50 dots · P95 ticks · ' + scope, fontsize=11)
        save(fig, output, name,
             '16 PCIe GPUs; complete online prepare, dispatch and combine, no expert GEMM. '
             'All measured samples retained. Colored dots show P50, black ticks P95; logarithmic y axis. '
             'CollWay-like is the repository implementation. Stability is recorded in the accompanying CSV. '
             + ('Top-1 routes cannot share one token across multiple remote GPUs.' if family == 'top1'
                else 'A fixed route from a 512-token Qwen3 batch-1 prefill is replayed at each selected layer; '
                     'online planning remains timed and full-plan caching is disabled. '
                     'Repeated timings do not represent independent routing samples.'))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output or args.data.parent / 'figures')
