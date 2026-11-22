#!/usr/bin/env bash
set -euo pipefail
# Start only the already provisioned RelayMoE daemon. Never changes system Docker.
config=/data07/fy/relaymoe-docker/daemon.json
socket=unix:///run/relaymoe-docker/docker.sock
[[ ! -f /.dockerenv ]] || { printf '请在宿主机运行。\n' >&2; exit 1; }
if docker -H "$socket" info >/dev/null 2>&1; then
    printf 'RelayMoE Docker 已运行：%s\n' "$socket"
    exit 0
fi
[[ -f "$config" ]] || { printf '尚未配置独立 daemon：%s\n' "$config" >&2; exit 1; }
[[ "$(id -u)" == 0 ]] || { printf '启动 daemon 需要宿主机 root。\n' >&2; exit 1; }
python3 - "$config" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
expected={'data-root':'/data07/fy/relaymoe-docker/data',
          'exec-root':'/run/relaymoe-docker/exec',
          'pidfile':'/run/relaymoe-docker/dockerd.pid',
          'hosts':['unix:///run/relaymoe-docker/docker.sock'],
          'containerd-namespace':'relaymoe',
          'containerd-plugins-namespace':'plugins.relaymoe',
          'bridge':'none','iptables':False,'ip6tables':False,
          'ip-forward':False,'ip-masq':False}
for k,v in expected.items():
    if c.get(k)!=v: raise SystemExit('独立 daemon 配置不符：'+k)
PY
install -d -m 700 /run/relaymoe-docker
dockerd --validate --config-file "$config"
nohup dockerd --config-file "$config" >>/data07/fy/relaymoe-docker/dockerd.log 2>&1 </dev/null &
for ((i=0; i<30; i++)); do
    if docker -H "$socket" info >/dev/null 2>&1; then
        printf 'RelayMoE Docker 已启动：%s\n' "$socket"
        exit 0
    fi
    sleep 1
done
printf '启动未完成，请查看 /data07/fy/relaymoe-docker/dockerd.log\n' >&2
exit 1
