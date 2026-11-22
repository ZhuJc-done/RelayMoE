"""Audit the finite single-node closeout scope and write a reproducible report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_results import analyze, stability


SOURCE = {
    'relaymoe/compact_backend.py': '7fc763ad9413bf1264b641c48cfad338485069fdf562e73ee4a83cab2e94a9c3',
    'relaymoe/csrc/ipc.cu': 'a936467301a0221809dbc308ff729c534d9855b7c54c48fb84187b488de486ef',
}
METHODS = ['nccl', 'collway_like', 'relay_batch', 'direct_batch', 'independent_batch']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/single-node-16gpu'))
    parser.add_argument('--report', type=Path, default=Path('docs/single-node-16gpu-results.md'))
    args = parser.parse_args()
    root = args.root
    audit = {'scope': 'User-requested single-node phase closeout: one seed of 11 synthetic configs, '
                      'three static real-route layers, v9 model batches 1 and 4; no further tuning',
             'source_sha256': SOURCE, 'files': [], 'checks': [],
             'deferred': ['Remaining four v9 workload seeds', 'Dense payload sweep',
                          'Full-model logits comparison', 'Multi-node and real DeepEP']}

    def read(relative):
        path = root / relative
        content = path.read_bytes()
        audit['files'].append({'file': relative, 'sha256': hashlib.sha256(content).hexdigest(),
                               'bytes': len(content)})
        return json.loads(content)

    def check(condition, description):
        if not condition:
            raise ValueError(description)
        audit['checks'].append(description)

    cases = [f'size-t{t}-s123' for t in [1, 5, 21, 84, 335, 670]]
    cases += [f'skew-{s}-s123' for s in [.4, .6, .8]]
    cases += [f'top1-t{t}-s123' for t in [1, 670]]
    pairs = [('final-v9', c) for c in cases]
    pairs += [('trace-v9', f'trace-layer{layer}') for layer in [0, 24, 47]]
    communication = []
    mechanisms = []
    for folder, case in pairs:
        data = read(f'{folder}/{case}.json')
        check(data.get('complete') and set(data['records']) == set(METHODS), case + ': five methods complete')
        check(data['source_sha256'] == SOURCE, case + ': frozen v9 sources')
        reference = None
        direct = data['records']['direct_batch']
        cut_names = sorted({cut for r in direct for cut in r['modeled_directed_cut_bytes']})
        direct_cut_means = {cut: st.mean(r['modeled_directed_cut_bytes'].get(cut, 0)
                                        for r in direct) for cut in cut_names}
        bottleneck = max(cut_names, key=direct_cut_means.get) if cut_names else None
        for method in METHODS:
            rows = data['records'][method]
            check(100 <= len(rows) <= 300, case + '/' + method + ': sample count')
            keys = [(r['step'], r['useful_bytes']) for r in rows]
            check(reference is None or keys == reference, case + '/' + method + ': paired steps and bytes')
            reference = keys
            values = [r['full_communication_us'] / 1000 for r in rows]
            test = stability(values)
            communication.append({'case': case, 'folder': folder, 'method': method, 'steps': len(rows),
                                  'full_p50_ms': st.median(values),
                                  'full_p95_ms': sorted(values)[int(.95 * (len(values) - 1))],
                                  'prepare_mean_ms': st.mean(r['prepare_us'] for r in rows) / 1000,
                                  'planner_mean_ms': st.mean(r['planner_us'] for r in rows) / 1000,
                                  'semantic_remote_bytes_per_rank_mean': st.mean(r['useful_bytes'] for r in rows) / 16,
                                  **test, 'stable': test['tail_drift'] <= .03 and test['block_cv'] <= .05})
            mechanisms.append({'case': case, 'method': method,
                               'source_injection_bytes_mean': st.mean(sum(r['source_injection_bytes']) for r in rows),
                               'logical_dispatch_bytes_mean': st.mean(r['logical_dispatch_bytes'] for r in rows),
                               'direct_reference_cut': bottleneck,
                               'modeled_bytes_on_direct_cut_mean': st.mean(r['modeled_directed_cut_bytes'].get(bottleneck, 0) for r in rows),
                               'modeled_max_cut_bytes_mean': st.mean(max(r['modeled_directed_cut_bytes'].values(), default=0) for r in rows),
                               'allocated_ipc_buffer_bytes_per_rank_max': max(r['allocated_buffer_bytes_per_rank'] for r in rows),
                               'required_ipc_buffer_bytes_max_rank_step': max(max(r['required_buffer_bytes_by_rank']) for r in rows),
                               'torch_peak_allocated_bytes_max_rank_step': max(r['torch_peak_allocated_bytes'] for r in rows),
                               'metadata_logical_bytes_mean': st.mean(r['metadata_logical_bytes'] for r in rows),
                               'flows_mean': st.mean(r['flows'] for r in rows),
                               'candidates_mean': st.mean(r['candidates'] for r in rows),
                               'dispatch_actions_mean': st.mean(r['dispatch_actions'] for r in rows),
                               'planned_dispatch_stages_mean': st.mean(r['planned_dispatch_stages'] for r in rows)})
    models = []
    for batch in [1, 4]:
        data = read(f'qwen3-batch-v9-b{batch}-8192.json')
        check(data.get('complete'), f'model batch {batch}: complete')
        check(all(data['source_sha256'][k] == v for k, v in SOURCE.items()),
              f'model batch {batch}: frozen transport source')
        check(not any(data.get(k, False) for k in ['logits_capture', 'boundary_capture',
                                                  'replicated_routing_validation', 'diagnostic_profile']),
              f'model batch {batch}: no diagnostic mode')
        reference = None
        check(set(data['records']) == {'native', 'direct:compact_nccl', 'relay:compact_batch'},
              f'model batch {batch}: three methods')
        for method, records in data['records'].items():
            rows = [r for r in records if not r['warmup']]
            keys = [(r['step'], r['input_tokens']) for r in rows]
            check(100 <= len(rows) <= 300 and all(r['input_tokens'] == 8192 * batch for r in rows),
                  f'model batch {batch}/{method}: complete inputs and sample count')
            check(reference is None or keys == reference, f'model batch {batch}/{method}: paired steps')
            reference = keys
            values = [r['wall_seconds'] * 1000 for r in rows]
            test = stability(values)
            models.append({'batch': batch, 'method': method, 'steps': len(rows),
                           'p50_ms': st.median(values), 'mean_ms': st.mean(values),
                           'p95_ms': sorted(values)[int(.95 * (len(values) - 1))],
                           **test, 'stable': test['tail_drift'] <= .03 and test['block_cv'] <= .05})
        if batch == 4:
            workers = data.get('runtime_provenance', [])
            check(len(workers) == 16 and {w['rank'] for w in workers} == set(range(16)),
                  'batch 4: all 16 worker runtime provenance records')
            check(all(w['loaded_sources'][k]['sha256'] == v for w in workers for k, v in SOURCE.items()),
                  'batch 4: actual loaded worker transport sources match')
        layers = read(f'qwen3-batch-v9-b{batch}-8192/layer-summary.json')
        check(set(layers['methods']) == {'direct:compact_nccl', 'relay:compact_batch'},
              f'batch {batch}: per-layer diagnostics aggregated')
    for world in [2, 4, 16]:
        correctness = read(f'correctness-{world}.json')
        check(correctness['passed'] and correctness['world'] == world,
              f'reference backend correctness: {world} ranks')
    correctness = read('correctness-v9-candidate.json')
    check(len(correctness) == 18 and all(r['passed'] for r in correctness), 'v9 BF16: 18 cases passed')
    hardware = read('p2p-intra-affinity-300.json')
    check(len(hardware) == 44 and all(r['payload_correct'] for r in hardware), 'P2P: 44 correct payload cases')
    fanout = read('fanout-v7.json')
    check(len(fanout) == 54, 'fan-out: 54 raw method/config rows')
    trace = read('qwen3-smoke-relay/routing-manifest.json')
    check(len(trace['files']) == 16 and trace['total_tokens'] == 512, 'static trace: 16 source rank files')
    for item in trace['files']:
        path = root / 'qwen3-smoke-relay' / Path(item['file']).name
        check(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'],
              f'static trace rank {item["rank"]}: original payload hash')
    status = read('closeout-status.json')
    check(status['state'] == 'measurements_complete_gpu_idle', 'closeout runner: all jobs exited and GPU idle')
    environment = read('environment-closeout.json')
    check(environment['gpu_count'] == 16, 'environment: 16 GPUs recorded')
    resources = read('environment-closeout-resource.json')
    check(resources['gpu_compute_query']['returncode'] == 0
          and not resources['gpu_compute_query']['stdout'].strip(),
          'final resource snapshot: no GPU compute processes')
    check(resources['repo_branch'] == 'main' and resources['repo_head'] ==
          '5e2f4f01827097b0451a97e7bc8a5ca273fe7ca1',
          'repository: unchanged Git base on main; uncommitted work recorded separately')
    for folder in ['final-v9', 'trace-v9']:
        analyze(root / folder, root / folder / 'analysis', [123, 234, 345, 456, 567])
    output = root / 'closeout'
    output.mkdir(exist_ok=True)
    for name, rows in [('communication', communication), ('models', models), ('mechanisms', mechanisms)]:
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    audit.update(result='finite_closeout_scope_complete', communication_cases=len(pairs),
                 model_configs=2, stability_is_separate_from_coverage=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# 单机 16 卡阶段收尾结果', '',
             '按用户最新要求结束本轮单机验证：冻结 v9，补齐有限实验清单，不再继续调优或扩展参数。',
             '模型保留在 `/data07/fy/relaymoe-models/Qwen3-30B-A3B`，代码与原始结果保留在',
             '`/data02/fy/relaymoe`。无 push。', '',
             '## 完成范围', '',
             '- 11 个合成配置：六档尺寸、Zipf 0.4/0.6/0.8、两档 top-1；每项五种方法。',
             '- 3 个静态真实路由回放：Qwen3 batch 1 / 512 tokens 的第 0、24、47 层。',
             '- Qwen3 完整模型 v9 batch 1 和 4，8192 输入 token/请求加首个输出 token。',
             '- P2P 44 点、fan-out 54 条原始记录、参考后端 2/4/16 卡以及 v9 BF16 正确性记录。',
             '- 全部原始样本保留。v9 通信只有 seed 123；其余四 seed 延后，v8 五 seed 结果独立保存。',
             '- 采样为 20 warmup +100 steps，不稳定则最多 300；覆盖完成不等于稳定性全部通过。', '',
             '## v9 完整通信 P50', '',
             '单位 ms，包含在线 prepare、dispatch、combine，不包含专家 GEMM。',
             'NCCL 是当前 all-to-all-v 适配器；CollWay-like 是仓库算法及执行器，不代表完整论文系统。', '',
             '| 配置 | NCCL | CollWay-like | RelayMoE | DirectBatch | IndependentBatch | NCCL/Relay |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for _, case in pairs:
        by_method = {r['method']: r for r in communication if r['case'] == case}
        numbers = [by_method[m]['full_p50_ms'] for m in METHODS]
        lines.append('| ' + case + ' | ' + ' | '.join(f'{x:.3f}' for x in numbers)
                     + f' | {numbers[0]/numbers[2]:.3f}× |')
    size_rows = [r for r in communication if r['case'].startswith('size-')]
    gains = []
    for case in {r['case'] for r in size_rows}:
        selected = {r['method']: r for r in size_rows if r['case'] == case}
        gains.append((selected['nccl']['full_p50_ms'] / selected['relay_batch']['full_p50_ms'], case))
    best_gain, best_case = max(gains)
    lines += ['', f'六档尺寸中，相对 NCCL 的最大已观测完整通信 P50 加速为 {best_gain:.3f}×',
              f'（{best_case}，单 seed 配置级比较，不是最快单步，也不是五 seed 稳定收益）。']
    lines += ['', '静态回放使用真实 expert IDs 和 gate weights，但重复同一份路由；在线 prepare 仍计时，',
              '整计划缓存关闭。这些结果不代表独立动态路由样本或完整模型时延。', '',
              '## 完整模型', '', '| Batch | 方法 | P50 (ms) | P95 (ms) | 均值 (ms) | 实测 steps | 稳定性通过 |',
              '| ---: | --- | ---: | ---: | ---: | ---: | --- |']
    labels = {'native': '原生 vLLM（复制 token 布局）', 'direct:compact_nccl': '匹配 NCCL 适配器',
              'relay:compact_batch': 'RelayMoE'}
    for row in models:
        lines.append(f"| {row['batch']} | {labels[row['method']]} | {row['p50_ms']:.3f} | "
                     f"{row['p95_ms']:.3f} | {row['mean_ms']:.3f} | {row['steps']} | {'是' if row['stable'] else '否'} |")
    for batch in [1, 4]:
        by_method = {r['method']: r for r in models if r['batch'] == batch}
        nccl, relay = by_method['direct:compact_nccl'], by_method['relay:compact_batch']
        lines += ['', f"Batch {batch}：RelayMoE 相对匹配 NCCL 的 P50 时延变化 "
                  f"{100*(relay['p50_ms']/nccl['p50_ms']-1):+.2f}%，P95 变化 "
                  f"{100*(relay['p95_ms']/nccl['p95_ms']-1):+.2f}%，均值变化 "
                  f"{100*(relay['mean_ms']/nccl['mean_ms']-1):+.2f}%（负值表示降低）。"]
    lines += ['', '## 硬件、在线成本与机制', '',
              '16 MiB 传输的聚合 payload 带宽如下；该硬件基准与 MoE 有效载荷带宽分开解释。', '',
              '| 路径 / 并发 | GB/s |', '| --- | ---: |']
    for row in hardware:
        if row['payload_bytes_per_transfer'] == 16777216:
            lines.append(f"| {row['case']} | {row['aggregate_payload_GBps']:.3f} |")
    lines += ['', '在线准备开销采用所有 step 的均值，不能与前面 P50 相加或相减。', '',
              '| 配置 | 方法 | prepare 均值 (ms) | planner 均值 (ms) | 源注入量相对 Direct 降低 |',
              '| --- | --- | ---: | ---: | ---: |']
    for case in ['size-t84-s123', 'skew-0.8-s123']:
        direct = next(r for r in mechanisms if r['case'] == case and r['method'] == 'direct_batch')
        for method in ['relay_batch', 'direct_batch', 'independent_batch']:
            row = next(r for r in communication if r['case'] == case and r['method'] == method)
            mechanism = next(r for r in mechanisms if r['case'] == case and r['method'] == method)
            reduction = 100 * (1 - mechanism['source_injection_bytes_mean'] / direct['source_injection_bytes_mean'])
            lines.append(f"| {case} | {method} | {row['prepare_mean_ms']:.3f} | "
                         f"{row['planner_mean_ms']:.3f} | {reduction:.2f}% |")
    for case in ['size-t84-s123', 'skew-0.8-s123']:
        selected = {r['method']: r for r in communication if r['case'] == case}
        check(all(selected[m]['full_p50_ms'] < selected['relay_batch']['full_p50_ms']
                  for m in ['direct_batch', 'independent_batch']),
              case + ': both ablations faster in observed P50')
    lines += ['', '这两个负载中，DirectBatch 和 IndependentBatch 的完整通信仍快于完整 RelayMoE。',
              '源注入量降低并未抵消所有在线开销，当前数据尚未证实中继与联合选树的净时延收益。',
              '静态 fan-out 的 d=15 源注入副本数为 15→4（减少 73.33%）；不把此结果解释为',
              '全局传输量变为 4 份或实测墙钟时间必然按 log N 缩短。']
    lines += ['', '原生 vLLM 在当前 TP16/DP1/EP16 布局使用复制 token、本地专家和 TP AllReduce；',
              '适配器使用源 token 分片、稀疏 dispatch/combine，再 AllGather 恢复布局。',
              '因此原生系统参照与匹配 NCCL 适配器分列，不能互换。全部方法保持相同 eager、',
              'prefix cache 关闭、BF16 与专家/attention kernel 配置。', '',
              '## 稳定性与结论边界', '',
              '通信五 seed 稳定收益尚未完成验证。CSV 保留 P50、P95、尾部窗口漂移和窗口 CV；',
              '稳定阈值为最后三个 20-step 窗口的均值极差/均值 ≤3%，全部完整窗口均值 CV ≤5%。',
              '不裁剪慢样本，不把单 seed 的最大收益改称跨场景普遍收益。',
              'DirectBatch 与 IndependentBatch 是相同执行器的消融；其结果决定机制是否带来净收益。',
              'Fan-out 的源注入量降低与全局传输量、端到端耗时是不同指标。', '',
              '## 归档与后续', '',
              '原始输入清单及 SHA256、检查结果见 `artifacts/single-node-16gpu/closeout/audit.json`。',
              '主要表格见该目录 `communication.csv`、`models.csv`、`mechanisms.csv`；逐 step 数据保存在原始 JSON。',
              '机制表的瓶颈 cut 由同配置 DirectBatch 的平均字节数确定，再在相同 cut 上比较各方法；',
              'cut 字节来自软件拓扑模型，不是 PCIe 硬件计数。Torch 显存峰值含同进程常驻后端，',
              '显式 IPC 缓冲区另列，不把两者混称各方法的独立总显存需求。',
              '图分别位于 `final-v9/analysis/`、`trace-v9/analysis/` 和 `model-v9-figures/`。',
              '本阶段不再追加实验。多 seed 复现、模型/拓扑扩展与第二台服务器上的 DeepEP 对比留待新阶段。',
              '测量结束时，实验 runner 已确认所有测量进程退出且 GPU 无计算进程；该状态是结束时快照，',
              '不表示服务器之后一直空闲。专属容器及环境保留，恢复收尾不再启动 GPU 实验。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n')
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'checks': len(audit['checks']), 'report': str(args.report), 'audit': str(output/'audit.json')}))


if __name__ == '__main__':
    main()
