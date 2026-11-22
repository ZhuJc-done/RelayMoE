"""Alternate full-model methods in blocks without reloading expert weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def summarize(values):
    windows = [statistics.mean(values[i:i+20]) for i in range(0, len(values), 20)
               if len(values[i:i+20]) == 20]
    tail = windows[-3:]
    return {'steps': len(values), 'median_seconds': statistics.median(values),
            'p95_seconds': sorted(values)[int(.95*(len(values)-1))],
            'window_means_seconds': windows,
            'tail_drift': (max(tail)-min(tail))/statistics.mean(tail) if tail else None,
            'block_cv': statistics.pstdev(windows)/statistics.mean(windows) if windows else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=8192);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--max-steps',type=int,default=300)
    p.add_argument('--methods',default='native,direct:compact_nccl,relay:compact_batch')
    p.add_argument('--capture-logits',action='store_true',help='Correctness only; CPU capture changes timing')
    p.add_argument('--capture-boundary',action='store_true')
    args=p.parse_args()
    if args.max_steps<args.steps or args.steps<=0 or args.warmup<0:
        p.error('Require max-steps >= steps > 0 and warmup >= 0')
    source_paths=['relaymoe/compact_backend.py','relaymoe/csrc/ipc.cu','relaymoe/ipc_backend.py',
                  'relaymoe/vllm_adapter.py','scripts/bench_model_paired.py']
    source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_paths}
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    model='/data07/fy/relaymoe-models/Qwen3-30B-A3B'
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    segments=[tokenizer.encode(x,add_special_tokens=False) for x in [
        'Explain how data moves between experts in a mixture of experts model. ',
        'Describe a distributed system for processing language with several computers. ',
        'Compare the tradeoffs between bandwidth, latency, and memory when serving a model. ',
        'Give a detailed explanation of parallel computing and communication scheduling. ']]
    sequences=[(s*(args.tokens//len(s)+1))[:args.tokens] for s in segments]
    config=dict(model=model,dtype='bfloat16',tensor_parallel_size=16,enable_expert_parallel=True,
        enable_ep_weight_filter=True,all2all_backend='allgather_reducescatter',moe_backend='triton',
        distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
        enable_chunked_prefill=False,max_model_len=args.tokens+16,
        max_num_batched_tokens=max(args.tokens*args.batch,args.tokens+16),max_num_seqs=args.batch,
        gpu_memory_utilization=.50,disable_custom_all_reduce=True,seed=123,
        worker_extension_cls='relaymoe.vllm_adapter.RelayWorkerExtension')
    llm=LLM(**config);methods=args.methods.split(',')
    runtime_provenance=llm.collective_rpc('relaymoe_initialize_methods',args=([m for m in methods if m!='native'],))
    for worker in runtime_provenance:
        for name,source in worker['loaded_sources'].items():
            if source['sha256']!=source_sha256[name]:
                raise RuntimeError(f'Worker {worker["rank"]} loaded a different source: {source["path"]}')
    print(runtime_provenance,flush=True)
    if args.capture_logits:llm.collective_rpc('relaymoe_capture_logits')
    if args.capture_boundary:llm.collective_rpc('relaymoe_capture_boundary')
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
    records={m:[] for m in methods}
    block=0;steps=args.steps;summaries={}
    while block<steps+args.warmup:
        offset=block//10%len(methods);order=methods[offset:]+methods[:offset]
        for method in order:
            llm.collective_rpc('relaymoe_select_method',args=(method,))
            for step in range(block,min(block+10,steps+args.warmup)):
                prompts=[{'prompt_token_ids':sequences[(step+i)%len(sequences)]} for i in range(args.batch)]
                start=time.perf_counter();outputs=llm.generate(prompts,params,use_tqdm=False);elapsed=time.perf_counter()-start
                row={'step':step,'warmup':step<args.warmup,'wall_seconds':elapsed,
                     'input_tokens':sum(len(x.prompt_token_ids) for x in outputs),
                     'token_ids':[list(x.outputs[0].token_ids) for x in outputs],
                     'top_logprobs':[{str(k):v.logprob for k,v in x.outputs[0].logprobs[0].items()} for x in outputs]}
                assert row['input_tokens']==args.tokens*args.batch
                records[method].append(row)
                if args.capture_logits and step>=args.warmup:
                    path=args.output.with_suffix('')/method.replace(':','-')/f'step-{step}'
                    llm.collective_rpc('relaymoe_save_logits',args=(str(path),))
                if args.capture_boundary and step>=args.warmup:
                    path=args.output.with_suffix('')/method.replace(':','-')/f'step-{step}'
                    llm.collective_rpc('relaymoe_save_boundary',args=(str(path),method))
                print(json.dumps({'method':method,**row}),flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            summaries={m:summarize([r['wall_seconds'] for r in rows if not r['warmup']])
                       for m,rows in records.items() if any(not r['warmup'] for r in rows)}
            payload={'config':config,'records':records,'block_size':10,'batch':args.batch,
                'tokens_per_request':args.tokens,'warmup':args.warmup,'requested_steps':args.steps,
                'max_steps':args.max_steps,'summaries':summaries,'complete':False,
                'source_sha256':source_sha256,'runtime_provenance':runtime_provenance,
                'prompt_templates':4,'prefix_cache':False,
                'logits_capture':args.capture_logits,'boundary_capture':args.capture_boundary,
                'replicated_routing_validation':os.environ.get('RELAYMOE_VALIDATE_REPLICATED_ROUTING')=='1',
                'measurement':'LLM.generate wall time; 8192+1 prefill when tokens=8192; fixed repeated prompt templates; method order rotated every 10 steps',
                'native_layout':'TP16 DP1 EP16 replicated token input with local experts and all-reduce',
                'adapted_layout':'source token chunks, sparse dispatch, local experts, combine, all-gather to restore TP layout'}
            args.output.write_text(json.dumps(payload,indent=2))
        block+=10
        if block>=steps+args.warmup and steps<args.max_steps:
            unstable=any(s['tail_drift'] is None or s['tail_drift']>.03 or s['block_cv']>.05
                         for s in summaries.values())
            if unstable:
                steps+=min(100,args.max_steps-steps)
                print('EXTEND measured steps to',steps,flush=True)
    payload['complete']=True;payload['measured_steps']=steps
    payload['stable_by_method']={m:s['tail_drift'] is not None and s['tail_drift']<=.03 and s['block_cv']<=.05
                                 for m,s in summaries.items()}
    args.output.write_text(json.dumps(payload,indent=2))
    llm.collective_rpc('relaymoe_save_methods',args=(str(args.output.with_suffix('')),))
    llm.collective_rpc('relaymoe_close_methods')


if __name__=='__main__':main()
