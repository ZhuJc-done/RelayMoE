"""Run a validation/benchmark script with an isolated CUDA extension variant."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

from torch.utils.cpp_extension import load
import relaymoe.ipc_backend as ipc


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--backend-source', type=Path, help='Optional isolated compact_backend.py candidate')
parser.add_argument('script')
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
os.environ['RELAYMOE_NATIVE_VARIANT_SOURCE'] = str(args.source.resolve())
os.environ['RELAYMOE_NATIVE_VARIANT_SHA256'] = digest
ipc._extension = load(name='relaymoe_variant_' + digest[:12], sources=[str(args.source)],
                      extra_cuda_cflags=['-O3'], extra_cflags=['-O3'], verbose=False)
if args.backend_source:
    os.environ['RELAYMOE_BACKEND_VARIANT_SHA256'] = hashlib.sha256(args.backend_source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('relaymoe.compact_backend', args.backend_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
print(json.dumps({'native_variant_source': str(args.source), 'native_variant_sha256': digest}), flush=True)
sys.argv = [args.script] + args.arguments
runpy.run_path(args.script, run_name='__main__')
