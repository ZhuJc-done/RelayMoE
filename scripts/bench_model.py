"""Full Qwen3 prefill + first-token smoke/benchmark, with explicit vLLM settings."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data07/fy/relaymoe-models/Qwen3-30B-A3B')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', choices=['native', 'direct', 'relay', 'independent'], default='native')
    parser.add_argument('--executor', choices=['ipc', 'staged', 'event', 'nccl', 'compact', 'compact_nccl'], default='ipc')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--capture-logits', action='store_true')
    args = parser.parse_args()
    from vllm import LLM, SamplingParams
    import vllm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    segment = tokenizer.encode('Explain how data moves between experts in a mixture of experts model. ', add_special_tokens=False)
    prompts = [{'prompt_token_ids': (segment * (args.tokens // len(segment) + 1))[:args.tokens]}
               for _ in range(args.batch)]
    config = dict(model=args.model, dtype='bfloat16', tensor_parallel_size=16,
                  enable_expert_parallel=True, enable_ep_weight_filter=True,
                  all2all_backend='allgather_reducescatter', moe_backend='triton',
                  distributed_executor_backend='mp', enforce_eager=True,
                  enable_prefix_caching=False, enable_chunked_prefill=False,
                  max_model_len=args.tokens + 16,
                  max_num_batched_tokens=max(args.tokens * args.batch, args.tokens + 16),
                  max_num_seqs=args.batch, gpu_memory_utilization=.50,
                  disable_custom_all_reduce=True, seed=123)
    if args.strategy != 'native' or args.capture_logits:
        config['worker_extension_cls'] = 'relaymoe.vllm_adapter.RelayWorkerExtension'
    start = time.perf_counter()
    llm = LLM(**config)
    init_seconds = time.perf_counter() - start
    if args.strategy != 'native':
        configured = llm.collective_rpc('relaymoe_configure', args=(args.strategy, args.executor, args.trace))
        print(json.dumps({'configured': configured}), flush=True)
    if args.capture_logits:
        llm.collective_rpc('relaymoe_capture_logits')
    sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, logprobs=5)
    records = []
    for step in range(args.warmup + args.steps):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        record = {'step': step, 'warmup': step < args.warmup, 'wall_seconds': elapsed,
                  'input_tokens': sum(len(x.prompt_token_ids) for x in outputs),
                  'output_token_ids': [list(x.outputs[0].token_ids) for x in outputs],
                  'top_logprobs': [{str(k):v.logprob for k,v in x.outputs[0].logprobs[0].items()}
                                  for x in outputs]}
        assert record['input_tokens'] == args.tokens * args.batch
        assert all(len(x) == 1 for x in record['output_token_ids'])
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'vllm_version': vllm.__version__, 'config': config,
            'measurement': 'full model server-side prefill plus first token wall time',
            'init_seconds': init_seconds, 'records': records}, indent=2))
    if args.strategy != 'native':
        llm.collective_rpc('relaymoe_save', args=(str(args.output.with_suffix('')),))
    if args.capture_logits:
        llm.collective_rpc('relaymoe_save_logits', args=(str(args.output.with_suffix('')),))


if __name__ == '__main__':
    main()
