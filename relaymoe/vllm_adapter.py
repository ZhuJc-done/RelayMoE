"""Pinned vLLM 0.20 integration; preserve its real expert/attention kernels.

TP16/DP1 provides replicated inputs at the MoE boundary. The adapter assigns
each token to one source, runs sparse EP dispatch/combine, and all-gathers the
result to restore the required replicated TP layout. All three steps are timed.
"""
from pathlib import Path
import time

import torch
import torch.distributed as dist
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)

from .torch_backend import TorchDistributedBackend, ExpertOutputs


class RelayPrepareFinalize(MoEPrepareAndFinalizeNoDPEPModular):
    def __init__(self, backend, name, stats, trace):
        super().__init__()
        self.backend, self.name, self.stats, self.trace = backend, name, stats, trace

    def output_is_reduced(self):
        return True

    def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant=False):
        if a1.dtype != torch.bfloat16 or apply_router_weight_on_input or quant_config.quant_dtype is not None:
            raise ValueError('First adapter supports BF16, output-weighted experts only')
        backend = self.backend
        self.started = time.perf_counter_ns()
        self.total = a1.shape[0]
        chunk = (self.total + backend.world - 1) // backend.world
        self.chunk = chunk
        start, end = backend.rank * chunk, min((backend.rank + 1) * chunk, self.total)
        local = a1[start:end]
        ids, weights = topk_ids[start:end], topk_weights[start:end]
        if self.trace is not None and self.name not in self.trace:
            self.trace[self.name] = {'ids': ids.detach().cpu(), 'weights': weights.detach().cpu(),
                                     'total_tokens': self.total, 'hidden': a1.shape[1]}
        prepared = backend.prepare(ids, weights, expert_to_rank=[e // (num_experts // backend.world)
                                  for e in range(num_experts)], hidden_size=a1.shape[1], dtype=a1.dtype)
        prepare_end = time.perf_counter_ns()
        dispatched = backend.dispatch(local)
        torch.cuda.synchronize()
        dispatch_end = time.perf_counter_ns()
        chunks, all_ids, all_weights = [local], [ids], [weights]
        self.slices = []
        offset = local.shape[0]
        for flow in prepared.plan.traffic_classes:
            if backend.name not in flow.destinations:
                continue
            layout = prepared.plan.traffic_layouts[flow.name]
            data = prepared.dispatch_buffers[flow.name]
            chunks.append(data)
            all_ids.append(torch.tensor(layout.expert_indices, device=a1.device, dtype=ids.dtype))
            all_weights.append(torch.tensor(layout.expert_weights, device=a1.device, dtype=weights.dtype))
            self.slices.append((flow.name, offset, offset + data.shape[0]))
            offset += data.shape[0]
        self.prepared, self.local = prepared, local
        self.partial = {'layer': self.name, 'total_tokens': self.total,
                        'prepare_us': (prepare_end-self.started)/1000,
                        'dispatch_us': (dispatch_end-prepare_end)/1000,
                        'flows': len(prepared.plan.traffic_classes),
                        'actions': prepared.plan.action_count}
        return torch.cat(chunks), None, None, torch.cat(all_ids), torch.cat(all_weights)

    def finalize(self, output, fused_expert_output, topk_weights, topk_ids,
                 apply_router_weight_on_input, weight_and_reduce_impl):
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        out = weight_and_reduce_impl.apply(output=None, fused_expert_output=fused_expert_output,
            topk_weights=topk_weights, topk_ids=topk_ids, apply_router_weight_on_input=False)
        prepared, backend = self.prepared, self.backend
        for buffer in prepared.combine_buffers.values():
            buffer.zero_()
        for name, start, end in self.slices:
            prepared.combine_buffers[name].copy_(out[start:end])
        local_out = out[:self.local.shape[0]]
        combined = backend.combine(ExpertOutputs(prepared, local_out))
        padded = torch.zeros((self.chunk, output.shape[1]), device=output.device, dtype=output.dtype)
        padded[:combined.shape[0]].copy_(combined)
        gathered = torch.empty((self.chunk * backend.world, output.shape[1]), device=output.device, dtype=output.dtype)
        dist.all_gather_into_tensor(gathered, padded)
        output.copy_(gathered[:self.total])
        torch.cuda.synchronize()
        self.partial['full_moe_us'] = (time.perf_counter_ns()-self.started)/1000
        self.stats.append(self.partial)
        # The backend owns the current routing; keeping every layer's views
        # alive needlessly retains previous capacity allocations.
        self.prepared = None
        self.local = None


class RelayWorkerExtension:
    def relaymoe_capture_logits(self):
        self.relaymoe_last_logits = None
        model = self.model_runner.model
        original = model.compute_logits
        def capture(*args, **kwargs):
            logits = original(*args, **kwargs)
            if logits is not None:
                self.relaymoe_last_logits = logits.detach().float().cpu()
            return logits
        model.compute_logits = capture
        return dist.get_rank()

    def relaymoe_capture_boundary(self):
        self.relaymoe_boundary={}
        name=next(iter(self.relaymoe_native_adapters))
        adapters={'native':self.relaymoe_native_adapters[name],
                  **{m:entry[1][name] for m,entry in self.relaymoe_methods.items()}}
        for method,adapter in adapters.items():
            prepare,finalize=adapter.prepare,adapter.finalize
            def wrapped_prepare(*args,_method=method,_prepare=prepare,**kwargs):
                a1=args[0] if args else kwargs['a1']
                weights=args[1] if len(args)>1 else kwargs['topk_weights']
                ids=args[2] if len(args)>2 else kwargs['topk_ids']
                self.relaymoe_boundary[_method]={'input':a1.detach().cpu(),'ids':ids.detach().cpu(),
                    'weights':weights.detach().cpu(),'layer':name}
                return _prepare(*args,**kwargs)
            def wrapped_finalize(*args,_method=method,_finalize=finalize,**kwargs):
                result=_finalize(*args,**kwargs)
                output=args[0] if args else kwargs['output']
                captured=output.detach().clone()
                if _method=='native':dist.all_reduce(captured)
                self.relaymoe_boundary[_method]['output']=captured.cpu()
                return result
            adapter.prepare,adapter.finalize=wrapped_prepare,wrapped_finalize
        return dist.get_rank()

    def relaymoe_save_boundary(self,path,method):
        if dist.get_rank()==0:
            path=Path(path);path.mkdir(parents=True,exist_ok=True)
            torch.save(self.relaymoe_boundary[method],path/'boundary.pt')
        return dist.get_rank()

    def relaymoe_save_logits(self, path):
        if self.relaymoe_last_logits is not None:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.relaymoe_last_logits, path / f'logits-rank{dist.get_rank()}.pt')
        return dist.get_rank()

    def relaymoe_configure(self, strategy='relay', executor='ipc', trace=False):
        from vllm.distributed import get_ep_group
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        ep = get_ep_group()
        assert ep.world_size == dist.get_world_size() == 16
        self.relaymoe_stats = []
        self.relaymoe_trace = {} if trace else None
        if executor in ('compact','compact_nccl','compact_batch'):
            from .compact_backend import CompactBackend
            self.relaymoe_backend = CompactBackend(ep.cpu_group,strategy,executor='nccl' if executor=='compact_nccl' else 'batch' if executor=='compact_batch' else 'ipc')
        else:
            self.relaymoe_backend = TorchDistributedBackend(control_group=ep.cpu_group,
                                                            strategy=strategy, executor=executor,
                                                            build_expert_rows=False)
        count = 0
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, FusedMoE):
                kernel = module.quant_method.moe_kernel
                adapter = CompactPrepareFinalize if executor in ('compact','compact_nccl','compact_batch') else RelayPrepareFinalize
                kernel.impl.prepare_finalize = adapter(self.relaymoe_backend, name,
                                        self.relaymoe_stats, self.relaymoe_trace)
                count += 1
        return {'rank': ep.rank_in_group, 'layers': count, 'strategy': strategy, 'executor': executor}

    def relaymoe_initialize_methods(self, methods):
        import os
        rank=dist.get_rank()
        cpus={rank//8*64+(rank%8)*2,rank//8*64+(rank%8)*2+1} & os.sched_getaffinity(0)
        if cpus:os.sched_setaffinity(0,cpus)
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        self.relaymoe_methods = {}
        self.relaymoe_native_adapters = {name:module.quant_method.moe_kernel.impl.prepare_finalize
            for name,module in self.model_runner.model.named_modules() if isinstance(module,FusedMoE)}
        for method in methods:
            strategy,executor = method.split(':')
            self.relaymoe_configure(strategy,executor,False)
            adapters = {name:module.quant_method.moe_kernel.impl.prepare_finalize
                for name,module in self.model_runner.model.named_modules() if isinstance(module,FusedMoE)}
            self.relaymoe_methods[method] = (self.relaymoe_backend,adapters,self.relaymoe_stats)
        import hashlib
        from . import compact_backend, ipc_backend
        sources = {'relaymoe/vllm_adapter.py': Path(__file__),
                   'relaymoe/compact_backend.py': Path(compact_backend.__file__),
                   'relaymoe/ipc_backend.py': Path(ipc_backend.__file__),
                   'relaymoe/csrc/ipc.cu': Path(ipc_backend.__file__).parent / 'csrc/ipc.cu'}
        return {'rank':dist.get_rank(),'methods':methods,
                'loaded_sources': {name: {'path': str(path.resolve()),
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for name,path in sources.items()},
                'native_extensions': {method: {'path': backend.ext.__file__,
                    'sha256': hashlib.sha256(Path(backend.ext.__file__).read_bytes()).hexdigest()}
                    for method,(backend,_,_) in self.relaymoe_methods.items() if hasattr(backend,'ext')}}

    def relaymoe_select_method(self, method):
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        adapters = self.relaymoe_native_adapters if method=='native' else self.relaymoe_methods[method][1]
        for name,module in self.model_runner.model.named_modules():
            if isinstance(module,FusedMoE):module.quant_method.moe_kernel.impl.prepare_finalize=adapters[name]
        return dist.get_rank()

    def relaymoe_save_methods(self,path):
        import json
        root=Path(path);root.mkdir(parents=True,exist_ok=True)
        for method,(backend,adapters,stats) in self.relaymoe_methods.items():
            (root/f'{method.replace(":","-")}-rank{dist.get_rank()}.json').write_text(json.dumps(stats))
        return dist.get_rank()

    def relaymoe_close_methods(self):
        self.relaymoe_select_method('native')
        for backend,adapters,stats in self.relaymoe_methods.values():backend.close()
        self.relaymoe_methods.clear()
        return dist.get_rank()

    def relaymoe_save(self, path):
        rank = dist.get_rank()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        import json
        (path / f'layers-rank{rank}.json').write_text(json.dumps(self.relaymoe_stats))
        if self.relaymoe_trace is not None:
            torch.save(self.relaymoe_trace, path / f'routing-rank{rank}.pt')
        return {'rank': rank, 'records': len(self.relaymoe_stats)}


class CompactPrepareFinalize(RelayPrepareFinalize):
    def prepare(self, a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant=False):
        if a1.dtype != torch.bfloat16 or apply_router_weight_on_input or quant_config.quant_dtype is not None:
            raise ValueError('Compact adapter requires BF16 output-weighted experts')
        self.started=time.perf_counter_ns()
        b=self.backend;self.total=a1.shape[0];self.chunk=(self.total+b.world-1)//b.world
        start=b.rank*self.chunk;end=min(start+self.chunk,self.total)
        hidden=torch.zeros((self.chunk,a1.shape[1]),device=a1.device,dtype=a1.dtype)
        ids=torch.full((self.chunk,topk_ids.shape[1]),-1,device=a1.device,dtype=torch.int64)
        weights=torch.zeros_like(ids,dtype=torch.float32)
        valid=max(0,end-start)
        hidden[:valid]=a1[start:end];ids[:valid]=topk_ids[start:end];weights[:valid]=topk_weights[start:end]
        if self.trace is not None and self.name not in self.trace:
            self.trace[self.name]={'ids':ids.cpu(),'weights':weights.cpu(),'total_tokens':self.total,'hidden':a1.shape[1]}
        b.prepare(ids,weights,a1.shape[1],num_experts,replicated_routing=(topk_ids,topk_weights))
        t1=time.perf_counter_ns()
        received=b.dispatch(hidden)
        torch.cuda.synchronize()
        self.dispatch_finished=time.perf_counter_ns()
        self.partial={'layer':self.name,'total_tokens':self.total,**b.stats,
                      'dispatch_us':(self.dispatch_finished-t1)/1000}
        return received,None,None,b.prepared_ids.to(topk_ids.dtype),b.prepared_weights.to(topk_weights.dtype)

    def finalize(self,output,fused_expert_output,topk_weights,topk_ids,
                 apply_router_weight_on_input,weight_and_reduce_impl):
        if isinstance(weight_and_reduce_impl,TopKWeightAndReduceDelegate):
            weight_and_reduce_impl=TopKWeightAndReduceContiguous()
        out=weight_and_reduce_impl.apply(output=None,fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,topk_ids=topk_ids,apply_router_weight_on_input=False)
        torch.cuda.synchronize();expert_finished=time.perf_counter_ns()
        combined=self.backend.combine(out)
        torch.cuda.synchronize();combine_finished=time.perf_counter_ns()
        gathered=torch.empty((self.chunk*self.backend.world,output.shape[1]),device=output.device,dtype=output.dtype)
        dist.all_gather_into_tensor(gathered,combined)
        output.copy_(gathered[:self.total]);torch.cuda.synchronize()
        finished=time.perf_counter_ns()
        self.partial.update(expert_us=(expert_finished-self.dispatch_finished)/1000,
            combine_us=(combine_finished-expert_finished)/1000,
            restore_layout_us=(finished-combine_finished)/1000,full_moe_us=(finished-self.started)/1000)
        self.stats.append(self.partial)
