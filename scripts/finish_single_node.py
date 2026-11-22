"""Finish the approved v9 coverage serially on relay2; never stop other GPU jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/data02/fy/relaymoe')
ARTIFACTS = ROOT / 'artifacts/single-node-16gpu'
METHODS = {'nccl', 'collway_like', 'direct_batch', 'relay_batch', 'independent_batch'}
SOURCE = {
    'relaymoe/compact_backend.py': '7fc763ad9413bf1264b641c48cfad338485069fdf562e73ee4a83cab2e94a9c3',
    'relaymoe/csrc/ipc.cu': 'a936467301a0221809dbc308ff729c534d9855b7c54c48fb84187b488de486ef',
}


def process_identity(pid):
    try:
        # Fields after the parenthesized comm start with field 3; starttime is 22.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def validate_result(path, methods):
    data = json.loads(path.read_text())
    if not data.get('complete') or set(data['records']) != methods:
        raise RuntimeError(f'Incomplete or mismatched methods: {path}')
    reference = None
    for method, records in data['records'].items():
        rows = [r for r in records if not r.get('warmup', False)]
        if not 100 <= len(rows) <= 300:
            raise RuntimeError(f'Unexpected sample count: {path}, {method}, {len(rows)}')
        steps = [r['step'] for r in rows]
        if reference is not None and reference != steps:
            raise RuntimeError(f'Unpaired steps: {path}, {method}')
        reference = steps
    for name, digest in SOURCE.items():
        if data['source_sha256'].get(name) != digest:
            raise RuntimeError(f'Unexpected source version: {path}, {name}')


def check_idle():
    rows = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                           '--format=csv,noheader'], check=True, capture_output=True,
                          text=True).stdout.strip()
    if rows:
        raise RuntimeError(f'GPU processes still present; no additional job started: {rows}')
    for name, digest in SOURCE.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen v9 source changed: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    status_path = ARTIFACTS / 'closeout-status.json'
    state = {'state': 'waiting_for_existing_sweep', 'wait_pid': args.wait_pid,
             'source_sha256': SOURCE, 'jobs': []}

    def save():
        state['updated_unix_seconds'] = time.time()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(status_path)

    try:
        identity = process_identity(args.wait_pid)
        state['wait_process_starttime'] = identity
        save()
        while identity is not None and process_identity(args.wait_pid) == identity:
            time.sleep(10)
        cases = [f'size-t{t}-s123' for t in [1, 5, 21, 84, 335, 670]]
        cases += [f'skew-{s}-s123' for s in [.4, .6, .8]]
        cases += [f'top1-t{t}-s123' for t in [1, 670]]
        for case in cases:
            validate_result(ARTIFACTS / 'final-v9' / f'{case}.json', METHODS)
        jobs = [
            ('trace', ['python3', '-m', 'torch.distributed.run', '--standalone',
                       '--nproc_per_node=16', 'scripts/bench_compact.py', '--suite', 'trace',
                       '--methods', 'nccl,collway_like,direct_batch,relay_batch,independent_batch',
                       '--warmup', '20', '--steps', '100', '--max-steps', '300',
                       '--output', 'artifacts/single-node-16gpu/trace-v9']),
            ('model_batch4', ['/data07/fy/relaymoe-cache/vllm-venv/bin/python',
                              '-m', 'scripts.bench_model_paired', '--batch', '4',
                              '--tokens', '8192', '--warmup', '20', '--steps', '100',
                              '--max-steps', '300', '--output',
                              'artifacts/single-node-16gpu/qwen3-batch-v9-b4-8192.json']),
        ]
        for name, command in jobs:
            check_idle()
            record = {'name': name, 'command': ['bash', 'docker/experiment.sh', *command],
                      'started_unix_seconds': time.time(),
                      'log': str(ARTIFACTS / f'closeout-{name}.log')}
            state['jobs'].append(record)
            state['state'] = 'running_' + name
            with open(record['log'], 'w') as stream:
                process = subprocess.Popen(record['command'], cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT)
                record['host_pid'] = process.pid
                save()
                result = process.wait()
            record.update(exit_code=result, finished_unix_seconds=time.time())
            save()
            if result:
                raise RuntimeError(f'{name} exited with {result}; inspect {record["log"]}')
            if name == 'trace':
                for layer in [0, 24, 47]:
                    validate_result(ARTIFACTS / 'trace-v9' / f'trace-layer{layer}.json', METHODS)
            else:
                validate_result(ARTIFACTS / 'qwen3-batch-v9-b4-8192.json',
                                {'native', 'direct:compact_nccl', 'relay:compact_batch'})
        check_idle()
        state['state'] = 'measurements_complete_gpu_idle'
        save()
        print(json.dumps(state), flush=True)
    except Exception as error:
        state.update(state='stopped_for_inspection', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
