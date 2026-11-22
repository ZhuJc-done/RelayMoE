"""CPU online planning isolated from metadata transport and device allocation."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from relaymoe.compact_backend import native_planner
from scripts.bench_communication import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1)
    allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})
    start=time.perf_counter();planner=native_planner(16);init=time.perf_counter()-start
    result={'init_seconds_including_extension_load':init,'warmup':20,'steps':100,
            'measurement':'One CPU rank, hot catalog, dynamic routing, no full-plan cache; no metadata exchange or GPU materialization',
            'cases':[]}
    for tokens,experts in [(2,256),(91,256),(728,256),(512,128)]:
        routes=[torch.multinomial(torch.ones(16*tokens,experts),8,replacement=False,
            generator=torch.Generator().manual_seed(123+i)).view(16,tokens,8) for i in range(120)]
        rows={m:[] for m in ['direct','relay','independent']}
        for block in range(0,120,10):
            order=list(rows);offset=block//10%3;order=order[offset:]+order[:offset]
            for method in order:
                mode={'direct':0,'relay':1,'independent':2}[method]
                for i in range(block,block+10):
                    start=time.perf_counter_ns();plan=planner.prepare(routes[i],experts,2048,mode)
                    elapsed=(time.perf_counter_ns()-start)/1000
                    if i>=20:rows[method].append(dict(full_planner_us=elapsed,grouping_us=plan.grouping_us,
                        selection_us=plan.selection_us,layout_us=plan.layout_us,
                        candidates=plan.candidates,actions=plan.dispatch.action_count))
        summaries={m:{key:summarize([r[key] for r in rs]) for key in ['full_planner_us','grouping_us','selection_us','layout_us']}
                   for m,rs in rows.items()}
        case={'tokens_per_rank':tokens,'experts':experts,'summaries':summaries,'records':rows}
        result['cases'].append(case);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps({'tokens_per_rank':tokens,'experts':experts,'summaries':summaries}),flush=True)


if __name__=='__main__':main()
