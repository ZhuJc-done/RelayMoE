"""Record reproducible environment and file hashes without secrets."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import torch


def command(argv):
    r=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    return {'exit_code':r.returncode,'output':r.stdout}


root=Path('artifacts/single-node-16gpu');root.mkdir(parents=True,exist_ok=True)
files=list(Path('relaymoe').glob('*.py'))+list(Path('relaymoe/csrc').glob('*'))+list(Path('scripts').glob('*.py'))
data={'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,
      'torch_cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),'gpu_count':torch.cuda.device_count(),
      'nvidia_smi':command(['nvidia-smi']),'topology':command(['nvidia-smi','topo','-m']),
      'nvcc':command(['nvcc','--version']),'repo':command(['git','rev-parse','HEAD']),
      'status':command(['git','status','--short']),
      'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
      'measurement_env':{'NCCL_IB_DISABLE':'1','NCCL_P2P_LEVEL':'SYS','NCCL_NET_DISABLE_INTRA':'1',
                         'NCCL_NET_PLUGIN':'none','OMP_NUM_THREADS':'1','MAX_JOBS':'2'}}
(root/'environment-final.json').write_text(json.dumps(data,indent=2));print('saved',root/'environment-final.json')
