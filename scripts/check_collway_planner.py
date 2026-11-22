"""Native CollWay-like must reproduce repository stage membership exactly."""
import torch
from relaymoe.baselines import CollWayBaseline
from relaymoe.compact_backend import native_planner
from relaymoe.model import TrafficClass
from relaymoe.torch_backend import single_node_topology


def main():
    planner=native_planner(16,bandwidths=(32.,24.,16.))
    reference=CollWayBaseline(single_node_topology(16))
    for seed in range(8):
        ids=torch.multinomial(torch.ones(16*8,128),8,replacement=False,
            generator=torch.Generator().manual_seed(seed)).view(16,8,8)
        native=planner.prepare(ids,128,2048,3)
        trees=native.trees()
        flows=[TrafficClass(name=f'tc{i:06d}',source=f'g{src}',destinations=tuple(f'g{d}' for d in range(16) if mask&(1<<d)),
                            volume_bytes=len(ts)*4096) for i,(src,mask,ts,arcs) in enumerate(trees)]
        expected=reference.plan(flows)
        keys=[]
        for i,(src,mask,ts,arcs) in enumerate(trees):
            keys.extend((f'tc{i:06d}',f'g{a[0]}',f'g{a[1]}') for a in arcs)
        for phase in ['dispatch','combine']:
            actual=[{keys[i] if phase=='dispatch' else (keys[i][0],keys[i][2],keys[i][1]) for i in stage}
                    for stage in getattr(native,phase).stages]
            wanted=[{(a.traffic_name,a.arc.src,a.arc.dst) for a in stage.actions} for stage in getattr(expected,phase).stages]
            assert actual==wanted,(seed,phase,len(actual),len(wanted))
        print('CollWay-like exact stages PASS',seed,len(native.dispatch.stages),flush=True)


if __name__=='__main__':main()
