"""Paired short communication experiments with all online preparation timed."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from relaymoe.torch_backend import TorchDistributedBackend, ExpertOutputs


def summarize(values):
    ordered = sorted(values)
    windows = [statistics.mean(values[i:i+20]) for i in range(0, len(values), 20)]
    tail = windows[-3:]
    return {'median_us': statistics.median(values), 'mean_us': statistics.mean(values),
            'p95_us': ordered[int(.95*(len(ordered)-1))],
            'window_means_us': windows,
            'tail_drift': (max(tail)-min(tail))/statistics.mean(tail)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--hidden', type=int, default=2048)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--skew', type=float, default=0)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--methods', default='direct:nccl,relay:ipc,direct:ipc,independent:ipc')
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    numa = rank // 8
    cpus = {numa*64+(rank%8)*2, numa*64+(rank%8)*2+1}
    cpus &= os.sched_getaffinity(0)
    if cpus: os.sched_setaffinity(0, cpus)
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=timedelta(seconds=180))
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=180))
    world = dist.get_world_size()
    dist.all_reduce(torch.ones(1, device='cuda'))
    methods = [tuple(x.split(':')) for x in args.methods.split(',')]
    backends = {method: TorchDistributedBackend(control_group=control, strategy=method[0], executor=method[1], build_expert_rows=False)
                for method in methods}
    experts = 256
    placement = [e // (experts//world) for e in range(experts)]
    probabilities = torch.arange(1,experts+1,dtype=torch.float64).pow(-args.skew)
    hidden = torch.randn(args.tokens,args.hidden,device='cuda',dtype=torch.bfloat16)
    records = {':'.join(m):[] for m in methods}
    routes = []
    for step in range(args.warmup+args.steps):
        generator = torch.Generator().manual_seed(args.seed+rank*100000+step)
        ids = torch.multinomial(probabilities.expand(args.tokens,-1),args.topk,
                                replacement=False,generator=generator).to('cuda')
        weights = torch.full(ids.shape,1/args.topk,device='cuda',dtype=torch.float32)
        routes.append((ids,weights))
    try:
        if args.profile and rank == 0:
            import cProfile
            profile = cProfile.Profile()
            profile.enable()
        # Rotate method order in each fixed 10-step block.
        for block in range(0,args.warmup+args.steps,10):
            order = methods[block//10%len(methods):]+methods[:block//10%len(methods)]
            for method in order:
                backend=backends[method]
                for step in range(block,min(block+10,args.warmup+args.steps)):
                    ids,weights=routes[step]
                    dist.barrier(group=control);torch.cuda.synchronize()
                    t0=time.perf_counter_ns()
                    prepared=backend.prepare(ids,weights,expert_to_rank=placement,
                                             hidden_size=args.hidden,dtype=hidden.dtype)
                    torch.cuda.synchronize();t1=time.perf_counter_ns()
                    backend.dispatch(hidden)
                    torch.cuda.synchronize();t2=time.perf_counter_ns()
                    # No GEMM in communication-only experiment. Initialize
                    # nonzero expert partials outside both timed phases.
                    for flow in prepared.plan.traffic_classes:
                        if flow.name in prepared.combine_buffers:
                            prepared.combine_buffers[flow.name].fill_(0 if flow.source==backend.name else 1)
                    outputs=ExpertOutputs(prepared,torch.zeros_like(hidden))
                    torch.cuda.synchronize();t3=time.perf_counter_ns()
                    backend.combine(outputs)
                    torch.cuda.synchronize();t4=time.perf_counter_ns()
                    stats=torch.tensor([(t1-t0)/1000,(t2-t1)/1000,(t4-t3)/1000,
                                       (t2-t0+t4-t3)/1000],dtype=torch.float64)
                    dist.all_reduce(stats,op=dist.ReduceOp.MAX,group=control)
                    if rank==0 and step>=args.warmup:
                        record=dict(zip(['prepare_us','dispatch_us','combine_us','full_communication_us'],stats.tolist()))
                        record.update(step=step-args.warmup,useful_bytes=prepared.plan.useful_bytes,
                                      flows=len(prepared.plan.traffic_classes),actions=prepared.plan.action_count,
                                      planner_us=prepared.plan.planning_time_us,
                                      candidates=prepared.plan.selection.candidates_evaluated)
                        records[':'.join(method)].append(record)
                if rank==0:
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    summaries={name:{metric:summarize([r[metric] for r in rows])
                        for metric in ['prepare_us','dispatch_us','combine_us','full_communication_us']}
                        for name,rows in records.items() if rows}
                    args.output.write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v
                        for k,v in vars(args).items()},'world':world,'records':records,'summaries':summaries},indent=2))
                    print(json.dumps({'block':block,'method':':'.join(method),'samples':len(records[':'.join(method)])}),flush=True)
    finally:
        if args.profile and rank == 0:
            import pstats
            profile.disable()
            profile.dump_stats(str(args.output.with_suffix('.prof')))
            pstats.Stats(profile).sort_stats('cumtime').print_stats(25)
        for backend in backends.values():backend.close()
        dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
