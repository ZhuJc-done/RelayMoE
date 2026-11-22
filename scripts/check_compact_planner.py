"""Compare native selected trees to the exact reference scheduler."""
import torch
from dataclasses import replace
from relaymoe.compact_backend import native_planner
from relaymoe.torch_backend import single_node_topology,IndependentTreeScheduler
from relaymoe.runtime import RelayMoE


def main():
    world=16;native=native_planner(world,bandwidths=(32.,24.,16.))
    for strategy,mode in [('relay',1),('independent',2)]:
        reference=RelayMoE(single_node_topology(world),plan_cache_size=0)
        if strategy=='independent':reference.scheduler=IndependentTreeScheduler(reference.topology,catalog=reference.scheduler.catalog)
        for seed in range(8):
            generator=torch.Generator().manual_seed(seed)
            ids=torch.multinomial(torch.ones(world*16,128),8,replacement=False,generator=generator).view(world,16,8)
            p=reference.plan_from_gating(ids.tolist(),expert_to_gpu=['g'+str(e//8) for e in range(128)],activation_bytes=4096)
            c=native.prepare(ids,128,2048,mode)
            trees=c.trees()
            assert len(trees)==len(p.traffic_classes)
            for flow,(src,mask,tokens,arcs) in zip(p.traffic_classes,trees):
                assert flow.source=='g'+str(src)
                assert list(p.traffic_layouts[flow.name].token_indices)==tokens
                actual={(a.src,a.dst) for a in p.selection.selected_trees[flow.name].arcs}
                expected={('g'+str(a[0]),'g'+str(a[1])) for a in arcs}
                assert actual==expected,(strategy,seed,flow.name,actual,expected)
            print(strategy,seed,'PASS',len(trees),flush=True)


if __name__=='__main__':main()
