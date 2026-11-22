"""Produce a compact, evidence-linked JSON summary from raw experiment files."""
import argparse
import json
from pathlib import Path
import statistics


def stats(values):
    windows=[statistics.mean(values[i:i+20]) for i in range(0,len(values),20)]
    tail=windows[-3:]
    return {'samples':len(values),'median':statistics.median(values),'mean':statistics.mean(values),
            'p95':sorted(values)[int(.95*(len(values)-1))],
            'tail_window_drift':(max(tail)-min(tail))/statistics.mean(tail),
            'window_means':windows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('artifacts/single-node-16gpu'));args=p.parse_args()
    models={};communications={};correctness={}
    for file in sorted(args.directory.glob('*.json')):
        data=json.loads(file.read_text())
        if file.name.startswith('qwen3-') and isinstance(data,dict) and 'records' in data:
            groups=data['records'] if isinstance(data['records'],dict) else {'single':data['records']}
            for method,rows in groups.items():
                samples=[x['wall_seconds'] for x in rows if not x['warmup']]
                if samples:models[f'{file.name}:{method}']={'seconds':stats(samples),'config':data['config'],
                    'logits_capture':data.get('logits_capture',False)}
        elif isinstance(data,dict) and 'summaries' in data:
            communications[file.name]=data['summaries']
        elif file.name.startswith('correctness-'):
            rows=data if isinstance(data,list) else data.get('cases',[])
            correctness[file.name]={'cases':len(rows),'passed':all(x['passed'] for x in rows),
                                    'max_abs_error':max((x.get('max_abs_error',0) for x in rows),default=0)}
    output={'models':models,'communications':communications,'correctness':correctness}
    (args.directory/'summary.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v['seconds'] for k,v in models.items()},indent=2))


if __name__=='__main__':main()
