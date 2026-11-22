"""Single-node compact CPU planning + CUDA IPC transport implementation.

Fixed EP rank order and contiguous expert placement; no full-plan cache.
The complete small single-node skeleton catalog is prebuilt at initialization.
"""
from itertools import product
import os
import uuid
import time

import torch
import torch.distributed as dist
from .catalog import OfflineBinomialCatalog
from .ipc_backend import extension


def native_planner(world, bandwidths=(47.,47.,47.)):
    ext = extension()
    planner = ext.CompactPlanner(world, *bandwidths)
    catalog = OfflineBinomialCatalog()
    switches = (world+3)//4
    for counts in product(range(5), repeat=switches):
        if counts[0] > min(3,world-1) or sum(counts)==0:
            continue
        if any(counts[i]>max(0,min(4,world-i*4)) for i in range(1,switches)):
            continue
        variants = catalog.lookup(counts, (0,)*switches, 2)
        if not variants:
            continue
        key = sum(x*5**i for i,x in enumerate(counts))
        rows = []
        for s in variants:
            rows.append((s.root_out_deg,s.max_gpu_pxb_count,
                         [list(s.role_out_degs_by_csw.get(c,())) for c in range(switches)],
                         [list(s.role_order_by_csw.get(c,())) for c in range(switches)],
                         [[a.src_role[0],a.src_role[1],a.dst_role[0],a.dst_role[1],stage]
                          for stage,arcs in enumerate(s.stages) for a in arcs]))
        planner.add(key,rows)
    return planner


class CompactBackend:
    def __init__(self, control_group, strategy='relay', bandwidths=(47.,47.,47.), executor='ipc'):
        self.control = control_group
        self.rank,self.world = dist.get_rank(),dist.get_world_size()
        self.strategy = strategy
        self.executor = executor
        if executor == 'nccl' and strategy != 'direct':
            raise ValueError('NCCL all-to-all baseline requires direct strategy')
        self.ext = extension()
        self.planner = native_planner(self.world,bandwidths)
        self.capacity = 0
        self.storage = None
        self.stats = {}
        self.measure_events = False
        self.cuda_times = {}
        token = [uuid.uuid4().hex if self.rank==0 else None]
        dist.broadcast_object_list(token,src=0,group=self.control)
        self.path = '/dev/shm/relaymoe-compact-'+token[0]
        if self.rank==0:self.ext.create_shared(self.path)
        dist.barrier(group=self.control)
        self.native = self.ext.Executor(self.path,self.rank,self.world)

    def close(self):
        dist.barrier(group=self.control)
        self.native.close()
        dist.barrier(group=self.control)
        if self.rank==0:os.unlink(self.path)

    def prepare(self, ids, weights, hidden, experts, replicated_routing=None):
        started = time.perf_counter_ns()
        if ids.ndim!=2 or ids.shape!=weights.shape or hidden<=0 or experts%self.world:
            raise ValueError('Expected matching [tokens,topk] routing and divisible expert placement')
        self.tokens,self.hidden = ids.shape[0],hidden
        self.topk = ids.shape[1]
        # Equal token chunks; padded tokens are represented by expert=-1.
        if replicated_routing is None:
            self.all_ids = torch.empty((self.world*self.tokens,self.topk),dtype=torch.int64,device=ids.device)
            self.all_weights = torch.empty_like(self.all_ids,dtype=torch.float32)
            dist.all_gather_into_tensor(self.all_ids,ids.to(torch.int64).contiguous())
            dist.all_gather_into_tensor(self.all_weights,weights.float().contiguous())
        else:
            global_ids,global_weights=replicated_routing
            if global_ids.shape!=global_weights.shape or global_ids.ndim!=2 or global_ids.shape[1]!=self.topk:
                raise ValueError('Expected matching replicated [total_tokens,topk] routing')
            count=global_ids.shape[0]
            if (count+self.world-1)//self.world!=self.tokens:
                raise ValueError('Replicated routing does not match equal source chunks')
            if os.environ.get('RELAYMOE_VALIDATE_REPLICATED_ROUTING')=='1':
                check_ids=torch.empty((self.world*self.tokens,self.topk),dtype=torch.int64,device=ids.device)
                check_weights=torch.empty_like(check_ids,dtype=torch.float32)
                dist.all_gather_into_tensor(check_ids,ids.to(torch.int64).contiguous())
                dist.all_gather_into_tensor(check_weights,weights.float().contiguous())
                count=global_ids.shape[0]
                if not torch.equal(check_ids[:count],global_ids.to(torch.int64)) or not torch.equal(check_weights[:count],global_weights.float()):
                    raise ValueError('Replicated routing differs across tensor-parallel ranks')
            padded=self.world*self.tokens
            if count==padded:
                self.all_ids=global_ids.to(torch.int64).contiguous()
                self.all_weights=global_weights.float().contiguous()
            else:
                self.all_ids=torch.full((padded,self.topk),-1,dtype=torch.int64,device=ids.device)
                self.all_weights=torch.zeros((padded,self.topk),dtype=torch.float32,device=ids.device)
                self.all_ids[:count]=global_ids
                self.all_weights[:count]=global_weights
        host_ids=self.all_ids.cpu().view(self.world,self.tokens,self.topk)
        metadata_finished=time.perf_counter_ns()
        self.plan = self.planner.prepare(host_ids,experts,hidden,{'direct':0,'relay':1,'independent':2,'collway_like':3}[self.strategy],
                                         self.rank if self.executor=='batch' else -1)
        planner_finished=time.perf_counter_ns()
        cap=self.plan.capacities
        # Capacity changes are deterministic from the shared plan; no extra
        # object collective or flag all-reduce is needed.
        required=max(cap)
        if required>self.capacity or self.storage is None:
            torch.cuda.synchronize();self.native.unmap();dist.barrier(group=self.control)
            self.capacity=max(1<<20,1<<max(0,required-1).bit_length())
            self.storage=self.ext.allocate(self.capacity)
            if self.executor in ('ipc','batch'):
                handles=[None]*self.world
                dist.all_gather_object(handles,self.ext.handle(self.storage),group=self.control)
                self.native.map(handles,self.storage.data_ptr())
        indices=[torch.tensor(x,dtype=torch.int64,device=ids.device) for x in self.plan.indices(self.rank)]
        self.pack_src,self.pack_dst,self.expert_global,storage,expert_combine,self.reduce_src,self.reduce_dst=indices
        self.remote_storage=storage[self.tokens:]
        self.expert_combine=expert_combine[self.tokens:]
        self.buffer=self.storage[:self.capacity//(hidden*2)*(hidden*2)].view(torch.bfloat16).view(-1,hidden)
        self.prepared_ids=self.all_ids.index_select(0,self.expert_global)
        self.prepared_weights=self.all_weights.index_select(0,self.expert_global)
        self.nccl_tables={}
        self.batch_tables={}
        if self.executor=='batch':
            self.batch_tables={phase:self.ext.compact_batch_descriptors(self.plan,self.rank,phase=='combine')
                               for phase in ['dispatch','combine']}
        if self.executor=='nccl':
            for phase in ['dispatch','combine']:
                send,recv,send_counts,recv_counts=self.plan.peer_indices(self.rank,phase=='combine')
                self.nccl_tables[phase]=(torch.tensor(send,device=ids.device,dtype=torch.int64),
                    torch.tensor(recv,device=ids.device,dtype=torch.int64),send_counts,recv_counts)
        self.stats={'prepare_us':(time.perf_counter_ns()-started)/1000,
                    'metadata_us':(metadata_finished-started)/1000,
                    'planner_us':(planner_finished-metadata_finished)/1000,
                    'grouping_us':self.plan.grouping_us,'selection_us':self.plan.selection_us,'layout_us':self.plan.layout_us,
                    'materialize_us':(time.perf_counter_ns()-planner_finished)/1000,
                    'useful_bytes':self.plan.useful,'candidates':self.plan.candidates,
                    'routing_source':'replicated' if replicated_routing is not None else 'all_gather'}
        return self

    def dispatch(self,hidden):
        self.local=hidden
        self.buffer.index_copy_(0,self.pack_dst,hidden.index_select(0,self.pack_src))
        self.transfer('dispatch')
        return torch.cat((hidden,self.buffer.index_select(0,self.remote_storage)))

    def combine(self,out):
        # Zero accumulators, but do not overwrite dispatch data still used by
        # expert output. out is separately owned by the fused expert kernel.
        self.buffer.zero_()
        self.buffer.index_copy_(0,self.expert_combine,out[self.tokens:])
        self.transfer('combine')
        result=out[:self.tokens].clone()
        result.index_add_(0,self.reduce_dst,self.buffer.index_select(0,self.reduce_src))
        return result

    def transfer(self,phase):
        if self.measure_events:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
        if self.executor=='batch':
            self.native.run_batches(self.batch_tables[phase],phase=='combine')
        elif self.executor=='ipc':
            torch.cuda.synchronize();dist.barrier(group=self.control)
            self.native.run(getattr(self.plan,phase),phase=='combine',self.strategy in ('relay','independent'),True)
        else:
            send_ids,recv_ids,send_counts,recv_counts=self.nccl_tables[phase]
            send=self.buffer.index_select(0,send_ids)
            recv=torch.empty((len(recv_ids),self.hidden),device=send.device,dtype=send.dtype)
            dist.all_to_all_single(recv,send,output_split_sizes=recv_counts,input_split_sizes=send_counts)
            if phase=='dispatch':self.buffer.index_copy_(0,recv_ids,recv)
            else:self.buffer.index_add_(0,recv_ids,recv)
        if self.measure_events:
            end.record();end.synchronize()
            self.cuda_times[phase]=start.elapsed_time(end)*1000
