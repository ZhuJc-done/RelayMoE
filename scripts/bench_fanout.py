"""Static fan-out: source injection, tree rounds and forward/reverse transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import timedelta
import time
import statistics
import torch
import torch.distributed as dist
from relaymoe.compact_backend import CompactBackend
from scripts.bench_communication import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/single-node-16gpu/fanout.json'))
    p.add_argument('--steps',type=int,default=100);p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--max-steps',type=int,default=300);args=p.parse_args()
    if args.max_steps<args.steps:p.error('max-steps must be >= steps')
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
             [Path('relaymoe/compact_backend.py'),Path('relaymoe/csrc/ipc.cu'),Path(__file__)]}
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    cpus={rank//8*64+(rank%8)*2,rank//8*64+(rank%8)*2+1}&os.sched_getaffinity(0)
    if cpus:os.sched_setaffinity(0,cpus)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(seconds=180))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=180))
    methods=['nccl','direct_batch','relay_batch']
    backends={m:CompactBackend(control,'direct' if m=='nccl' else m.removesuffix('_batch'),executor='nccl' if m=='nccl' else 'batch') for m in methods}
    records=[]
    sets={'near_first':list(range(1,16)),'numa_spread':[8,4,12,1,9,5,13,2,10,6,14,3,11,7,15]}
    try:
      for destinations in [1,2,4,8,15]:
       for pattern,ranks in sets.items():
        if destinations==15 and pattern=='numa_spread':continue
        selected=ranks[:destinations]
        for payload in [65536,1048576]:
          tokens=payload//4096
          ids=torch.full((tokens,destinations),-1,device='cuda',dtype=torch.int64)
          weights=torch.zeros_like(ids,dtype=torch.float32)
          if rank==0:ids[:]=torch.tensor(selected,device='cuda')*8;weights.fill_(1/destinations)
          hidden=torch.randn(tokens,2048,device='cuda',dtype=torch.bfloat16)
          for b in backends.values():b.prepare(ids,weights,2048,128);b.measure_events=True
          values={m:[] for m in methods}
          block=0;steps=args.steps
          while block<args.warmup+steps:
            offset=block//10%len(methods)
            for method in methods[offset:]+methods[:offset]:
              b=backends[method]
              for step in range(block,min(block+10,args.warmup+steps)):
                dist.barrier(group=control);torch.cuda.synchronize();t0=time.perf_counter_ns()
                x=b.dispatch(hidden);torch.cuda.synchronize();t1=time.perf_counter_ns()
                out=torch.ones_like(x);torch.cuda.synchronize();t2=time.perf_counter_ns()
                result=b.combine(out);torch.cuda.synchronize();t3=time.perf_counter_ns()
                sample=torch.tensor([(t1-t0)/1000,(t3-t2)/1000,b.cuda_times['dispatch'],b.cuda_times['combine']],dtype=torch.float64)
                dist.all_reduce(sample,op=dist.ReduceOp.MAX,group=control)
                if step>=args.warmup:values[method].append(sample.tolist())
            block+=10
            if block>=args.warmup+steps:
              extend=torch.zeros(1,dtype=torch.int64)
              if rank==0 and steps<args.max_steps:
                unstable=False
                for samples in values.values():
                  for column in [0,1]:
                    result=summarize([v[column] for v in samples]);windows=result['window_means_us']
                    unstable|=result['tail_drift']>.03 or statistics.pstdev(windows)/statistics.mean(windows)>.05
                if unstable:extend.fill_(min(100,args.max_steps-steps))
              dist.broadcast(extend,src=0,group=control);steps+=int(extend.item())
          if rank==0:
            for method,b in backends.items():
                m=dict(b.plan.metrics());samples=values[method]
                row={'destinations':destinations,'destination_ranks':selected,'pattern':pattern,
                     'payload_bytes':payload,'method':method,**m,
                     'dispatch':summarize([v[0] for v in samples]),'combine':summarize([v[1] for v in samples]),
                     'raw_samples_us':samples,'columns':['dispatch','combine','dispatch_cuda_timeline','combine_cuda_timeline'],
                     'ideal_relay_rounds':destinations.bit_length(),
                     'warmup':args.warmup,'requested_steps':args.steps,'measured_steps':steps,
                     'source_sha256':sources,
                     'measurement':f'static routing; plan prepared once; {args.warmup} warmup +{steps} measured; rotated method blocks'}
                for phase in ['dispatch','combine']:
                    windows=row[phase]['window_means_us']
                    row[phase]['block_cv']=statistics.pstdev(windows)/statistics.mean(windows)
                    row[phase]['stable']=row[phase]['tail_drift']<=.03 and row[phase]['block_cv']<=.05
                records.append(row)
                print(json.dumps({k:row[k] for k in ['destinations','pattern','payload_bytes','method','dispatch','combine','source_injection_bytes']}),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(records,indent=2))
    finally:
        for b in backends.values():b.close()
        dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
