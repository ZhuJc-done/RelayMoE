"""Apply task-local resource limits before starting an experiment command."""
import os
import resource
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: python scripts/with_limits.py COMMAND [ARG ...]")
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
os.execvp(sys.argv[1], sys.argv[1:])
