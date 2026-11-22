"""Compare full vocabulary logits from the paired correctness-only run."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    records=[]
    for method in sorted(args.directory.iterdir()):
        if not method.is_dir() or method.name=='native':continue
        for step in sorted(method.glob('step-*')):
            ref_files=sorted((args.directory/'native'/step.name).glob('logits-rank*.pt'))
            got_files=sorted(step.glob('logits-rank*.pt'))
            if not ref_files or not got_files:raise RuntimeError(f'Missing logits: {step}')
            ref=torch.load(ref_files[0],weights_only=True).float()
            got=torch.load(got_files[0],weights_only=True).float()
            assert ref.shape==got.shape
            delta=got-ref
            finite=torch.isfinite(ref)&torch.isfinite(got)
            assert torch.equal(torch.isfinite(ref),torch.isfinite(got))
            x,y=ref[finite],got[finite];d=y-x
            p_ref=torch.softmax(ref,-1);logq=torch.log_softmax(got,-1)
            kl=torch.sum(torch.where(p_ref>0,p_ref*(torch.log_softmax(ref,-1)-logq),0),-1)
            row={'method':method.name,'step':step.name,'shape':list(ref.shape),
                 'max_abs_error':d.abs().max().item(),'relative_l2':(d.norm()/x.norm()).item(),
                 'mean_abs_error':d.abs().mean().item(),'kl_ref_to_method':kl.tolist(),
                 'reference_argmax':ref.argmax(-1).tolist(),'method_argmax':got.argmax(-1).tolist(),
                 'argmax_equal':torch.equal(ref.argmax(-1),got.argmax(-1)),
                 'available_rank_files':len(got_files),
                 'allclose_atol002_rtol002':torch.allclose(x,y,atol=.02,rtol=.02)}
            records.append(row);print(json.dumps(row),flush=True)
    (args.directory/'comparison.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
