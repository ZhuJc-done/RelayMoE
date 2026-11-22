"""Aggregate matched per-seed results; preserve negative and unstable points."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    rows=[]
    for file in sorted(args.root.glob('*.json')):
        data=json.loads(file.read_text())
        if 'summaries' not in data:continue
        baseline=data['summaries']['nccl']['full_communication_us']['median_us']
        for method,s in data['summaries'].items():
            samples=data['records'][method]
            rows.append({'file':file.name,'method':method,**data['config'],
                'steps':len(samples),'complete':data.get('complete',False),
                'median_full_ms':s['full_communication_us']['median_us']/1000,
                'p95_full_ms':s['full_communication_us']['p95_us']/1000,
                'median_prepare_ms':s['prepare_us']['median_us']/1000,
                'median_planner_ms':s['planner_us']['median_us']/1000,
                'median_selection_ms':s['selection_us']['median_us']/1000,
                'median_dispatch_ms':s['dispatch_us']['median_us']/1000,
                'median_combine_ms':s['combine_us']['median_us']/1000,
                'speedup_vs_nccl':baseline/s['full_communication_us']['median_us'],
                'tail_drift':s['full_communication_us']['tail_drift'],
                'median_dispatch_algobw_GBs':statistics.median(x['dispatch_algobw_GBs'] for x in samples),
                'mean_useful_bytes_per_rank':statistics.mean(x['useful_bytes']/16 for x in samples)})
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (args.root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
    (args.root/'summary-rows.json').write_text(json.dumps(rows,indent=2))
    print('summarized',len(rows),'method/config rows',flush=True)


if __name__=='__main__':main()
