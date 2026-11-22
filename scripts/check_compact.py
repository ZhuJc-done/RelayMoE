"""Compact backend correctness with real BF16 matrix expert operations."""
import os
import json
import argparse
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from relaymoe.compact_backend import CompactBackend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--methods',default='nccl,direct,collway_like,relay,relay_batch,direct_batch,independent_batch,independent')
    args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=timedelta(seconds=120))
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=120))
    world=dist.get_world_size();experts=world*8;records=[]
    torch.manual_seed(123)
    matrices=torch.randn(experts,16,16,device='cuda',dtype=torch.bfloat16)/8
    for strategy in args.methods.split(','):
        backend=CompactBackend(control,'direct' if strategy=='nccl' else strategy.removesuffix('_batch'),
                               executor='nccl' if strategy=='nccl' else 'batch' if strategy.endswith('_batch') else 'ipc')
        for case in ['random','fanout','local','empty_rank','all_empty','hotspot']:
            torch.manual_seed(456+rank);count=8
            ids=torch.stack([torch.randperm(experts,device='cuda')[:min(8,experts)] for _ in range(count)])
            if case=='fanout':ids=torch.arange(0,experts,8,device='cuda').expand(count,-1).contiguous()
            if case=='local':ids=torch.arange(rank*8,rank*8+8,device='cuda').expand(count,-1).contiguous()
            if case=='hotspot':ids=torch.tensor([0,1,experts-1],device='cuda').expand(count,-1).contiguous()
            if case=='all_empty' or (case=='empty_rank' and rank==0):ids.fill_(-1)
            gates=torch.full(ids.shape,1/ids.shape[1],device='cuda')
            gates.masked_fill_(ids<0,0)
            hidden=torch.randn(count,16,device='cuda',dtype=torch.bfloat16)
            prepared=backend.prepare(ids,gates,16,experts)
            received=backend.dispatch(hidden)
            output=torch.zeros_like(received)
            for e in range(rank*8,(rank+1)*8):
                rows,ks=torch.where(prepared.prepared_ids==e)
                values=(received[rows]@matrices[e]).float()*prepared.prepared_weights[rows,ks,None]
                output.index_add_(0,rows,values.to(output.dtype))
            combined=backend.combine(output)
            reference=torch.zeros_like(hidden,dtype=torch.float32)
            for e in range(experts):
                rows,ks=torch.where(ids==e)
                reference.index_add_(0,rows,(hidden[rows]@matrices[e]).float()*gates[rows,ks,None])
            good=torch.allclose(combined.float(),reference,atol=.02,rtol=.02)
            error=(combined.float()-reference).abs().max().item()
            reports=[None]*world;dist.all_gather_object(reports,(good,error),group=control)
            assert all(x[0] for x in reports),(strategy,case,reports)
            if rank==0:
                row={'strategy':strategy,'case':case,'passed':True,'max_abs_error':max(x[1] for x in reports)}
                records.append(row);print(json.dumps(row),flush=True)
        backend.close()
    if rank==0:
        output=args.output or Path(f'artifacts/single-node-16gpu/correctness-compact-{world}.json')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(records,indent=2))
    dist.destroy_process_group(control);dist.destroy_process_group()


if __name__=='__main__':main()
