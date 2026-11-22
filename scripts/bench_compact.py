"""Dynamic paired NCCL/IPC benchmark; full online costs and per-step evidence."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from relaymoe.compact_backend import CompactBackend
from scripts.bench_communication import summarize


def stable(values):
    windows=[statistics.mean(values[i:i+20]) for i in range(0,len(values),20) if len(values[i:i+20])==20]
    if len(windows)<3:return False
    tail=windows[-3:]
    return ((max(tail)-min(tail))/statistics.mean(tail)<=.03
            and statistics.pstdev(windows)/statistics.mean(windows)<=.05)


def metrics(b):
    m=dict(b.plan.metrics())
    sizes=m.pop('group_tokens')
    m.update(group_tokens_min=min(sizes,default=0),group_tokens_median=statistics.median(sizes) if sizes else 0,
             group_tokens_max=max(sizes,default=0),catalog_cache='prebuilt',full_plan_cache_hits=0,
             metadata_logical_bytes=b.world*(b.world-1)*b.tokens*b.topk*12,
             cpu_routing_copy_bytes_per_rank=b.world*b.tokens*b.topk*8,
             useful_bytes=b.plan.useful,candidates=b.plan.candidates,
             allocated_buffer_bytes_per_rank=b.capacity,required_buffer_bytes_by_rank=b.plan.capacities)
    m['planned_dispatch_stages']=len(b.plan.dispatch.stages)
    m['busiest_sender_rank']=max(range(b.world),key=lambda r:m['rank_sent_bytes'][r])
    m['busiest_receiver_rank']=max(range(b.world),key=lambda r:m['rank_received_bytes'][r])
    return m


def run_case(args,case,rank,control,backends):
    methods=list(backends);tokens=case['tokens'];topk=case.get('topk',8);experts=case.get('experts',256)
    records={m:[] for m in methods}
    output=args.output/(case['name']+'.json') if args.suite else args.output.with_suffix('.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text()).get('complete'):
        if rank==0:print('SKIP',output,flush=True)
        return
    probabilities=torch.arange(1,experts+1,dtype=torch.float64).pow(-case.get('skew',0))
    hidden=torch.randn(tokens,2048,device='cuda',dtype=torch.bfloat16)
    steps=args.steps;limit=args.max_steps;routes=[]
    trace=case.get('trace')
    if trace:
        payload=torch.load(Path(trace)/f'routing-rank{rank}.pt',weights_only=True)
        ids=payload[case['layer']]['ids'].to(device='cuda',dtype=torch.int64)
        gates=payload[case['layer']]['weights'].to(device='cuda',dtype=torch.float32)
        routes=[(ids,gates)]*(args.warmup+limit)
        hidden=torch.randn(ids.shape[0],2048,device='cuda',dtype=torch.bfloat16)
    else:
        for step in range(args.warmup+limit):
            g=torch.Generator().manual_seed(case['seed']+rank*100000+step)
            ids=torch.multinomial(probabilities.expand(tokens,-1),topk,replacement=False,generator=g).to('cuda')
            gates=torch.full(ids.shape,1/topk,device='cuda',dtype=torch.float32)
            routes.append((ids,gates))
    baseline_memory=torch.cuda.memory_allocated();block=0;summaries={}
    while block<args.warmup+steps:
        offset=block//10%len(methods);order=methods[offset:]+methods[:offset]
        for method in order:
            b=backends[method];b.measure_events=True
            for step in range(block,min(block+10,args.warmup+steps)):
                ids,weights=routes[step]
                dist.barrier(group=control);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                t0=time.perf_counter_ns();b.prepare(ids,weights,2048,experts)
                torch.cuda.synchronize();t1=time.perf_counter_ns();received=b.dispatch(hidden)
                torch.cuda.synchronize();t2=time.perf_counter_ns()
                out=torch.ones_like(received);torch.cuda.synchronize();t3=time.perf_counter_ns()
                result=b.combine(out);torch.cuda.synchronize();t4=time.perf_counter_ns()
                values=torch.tensor([(t1-t0)/1000,(t2-t1)/1000,(t4-t3)/1000,(t2-t0+t4-t3)/1000,
                    b.cuda_times['dispatch'],b.cuda_times['combine'],torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),baseline_memory,
                    *[b.stats[k] for k in ['metadata_us','planner_us','grouping_us','selection_us','layout_us','materialize_us']]],dtype=torch.float64)
                dist.all_reduce(values,op=dist.ReduceOp.MAX,group=control)
                if rank==0 and step>=args.warmup:
                    row=dict(zip(['prepare_us','dispatch_us','combine_us','full_communication_us',
                        'dispatch_cuda_timeline_us','combine_cuda_timeline_us','torch_peak_allocated_bytes',
                        'torch_peak_reserved_bytes','workload_base_torch_bytes','metadata_us','planner_us',
                        'grouping_us','selection_us','layout_us','materialize_us'],values.tolist()))
                    row.update(step=step-args.warmup,**metrics(b))
                    row['dispatch_algobw_GBs']=b.plan.useful/(b.world*row['dispatch_us']*1000)
                    row['combine_algobw_GBs']=b.plan.useful/(b.world*row['combine_us']*1000)
                    records[method].append(row)
                del result,out,received
            if rank==0:
                keys=['prepare_us','dispatch_us','combine_us','full_communication_us',
                      'dispatch_cuda_timeline_us','combine_cuda_timeline_us','metadata_us','planner_us',
                      'grouping_us','selection_us','layout_us','materialize_us']
                summaries={m:{k:summarize([r[k] for r in rows]) for k in keys} for m,rows in records.items() if rows}
                payload={'config':case,'warmup':args.warmup,'requested_steps':args.steps,'max_steps':limit,
                    'summaries':summaries,'records':records,'complete':False,
                    'measurement':'dynamic routes; no plan cache' if not trace else 'static real routing replay; online prepare included',
                    'cuda_timing':'CUDA elapsed timeline around transport, includes host submission gaps and IPC readiness barrier; not kernel busy time',
                    'memory':'torch peak is process-wide, shared persistent backends; explicit IPC cudaMalloc buffers reported separately',
                    'cut_bytes':'software path model, not hardware PCIe counters',
                    'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('relaymoe/compact_backend.py'),Path('relaymoe/csrc/ipc.cu')]}}
                output.write_text(json.dumps(payload,indent=2))
                print(case['name'],block,method,len(records[method]),flush=True)
        block+=10
        if block>=args.warmup+steps:
            extend=torch.zeros(1,dtype=torch.int64)
            if rank==0:
                unstable=any(not stable([r['full_communication_us'] for r in rows]) for rows in records.values())
                if unstable and steps<limit:extend.fill_(min(100,limit-steps))
            dist.broadcast(extend,src=0,group=control);steps+=int(extend.item())
    if rank==0:
        payload['complete']=True;payload['measured_steps']=steps
        payload['stable_tail_by_method']={m:s['full_communication_us']['tail_drift']<=.03 for m,s in summaries.items()}
        payload['stable_by_method']={m:stable([r['full_communication_us'] for r in rows]) for m,rows in records.items()}
        output.write_text(json.dumps(payload,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=2);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--max-steps',type=int,default=100);p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--skew',type=float,default=0)
    p.add_argument('--topk',type=int,default=8);p.add_argument('--suite',choices=['pilot','full','trace'])
    p.add_argument('--methods',default='nccl,collway_like,direct_batch,relay_batch,independent_batch')
    p.add_argument('--seeds',default='123,234,345,456,567',help='Seeds for the full suite')
    p.add_argument('--case-filter',help='Comma-separated exact case names from the selected suite')
    args=p.parse_args();assert args.max_steps>=args.steps
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    cpus={rank//8*64+(rank%8)*2,rank//8*64+(rank%8)*2+1}&os.sched_getaffinity(0)
    if cpus:os.sched_setaffinity(0,cpus)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(seconds=300))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=300))
    backends={m:CompactBackend(control,'direct' if m=='nccl' else m.removesuffix('_batch'),
                executor='nccl' if m=='nccl' else 'batch' if m.endswith('_batch') else 'ipc') for m in args.methods.split(',')}
    cases=[]
    if args.suite=='pilot':
        cases=[dict(name=f'pilot-t{t}',tokens=t,seed=123) for t in [2,91,728]]
    elif args.suite=='full':
        for seed in map(int,args.seeds.split(',')):
            cases.extend(dict(name=f'size-t{t}-s{seed}',tokens=t,seed=seed) for t in [1,5,21,84,335,670])
            cases.extend(dict(name=f'skew-{skew}-s{seed}',tokens=84,skew=skew,seed=seed) for skew in [.4,.6,.8])
            cases.extend(dict(name=f'top1-t{t}-s{seed}',tokens=t,topk=1,seed=seed) for t in [1,670])
    elif args.suite=='trace':
        trace='artifacts/single-node-16gpu/qwen3-smoke-relay'
        data=torch.load(Path(trace)/f'routing-rank{rank}.pt',weights_only=True)
        layers=list(data)
        for i in [0,len(layers)//2,len(layers)-1]:
            layer=layers[i];ids=data[layer]['ids'];cases.append(dict(name=f'trace-layer{i}',tokens=ids.shape[0],topk=ids.shape[1],
                experts=128,seed=123,trace=trace,layer=layer))
    else:cases=[dict(name='single',tokens=args.tokens,topk=args.topk,skew=args.skew,seed=args.seed)]
    if args.case_filter:
        selected=set(args.case_filter.split(','));available={c['name'] for c in cases}
        if selected-available:raise ValueError(f'Unknown cases: {sorted(selected-available)}')
        cases=[c for c in cases if c['name'] in selected]
    try:
        for case in cases:run_case(args,case,rank,control,backends)
    finally:
        for b in backends.values():b.close()
        dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
