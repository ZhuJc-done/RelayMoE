# 单机 16 卡收尾实验复现

固定版本：v9；Git 基点 `5e2f4f01827097b0451a97e7bc8a5ca273fe7ca1`，加未提交工作区实现。
单机收尾时工作区未 push，单凭该 Git 基点不能复现实验；以归档源码和 SHA256 为准。
2026-11-22 起用户授权将冻结实现提交至 `zjc/test_real_moe`，环境同步见 `two-node-sync.md`。
正式 transport 摘要：

```text
relaymoe/compact_backend.py
7fc763ad9413bf1264b641c48cfad338485069fdf562e73ee4a83cab2e94a9c3
relaymoe/csrc/ipc.cu
a936467301a0221809dbc308ff729c534d9855b7c54c48fb84187b488de486ef
```

以下命令从 relay2 宿主机 `/data02/fy/relaymoe` 执行，使用已配置的
`zjc_relaymoe_exp` 容器与 `zjc_relaymoe:latest` 镜像。模型在 `/data07/fy/relaymoe-models`，
缓存和 vLLM 虚拟环境在 `/data07/fy/relaymoe-cache`。运行前确认 16 张 GPU 空闲；
这些任务必须串行，不与其他用户 GPU 任务并行启动。命令不修改宿主驱动或时钟。

## 合成通信扫描

```bash
bash docker/experiment.sh python3 -m torch.distributed.run \
  --standalone --nproc_per_node=16 scripts/bench_compact.py \
  --suite full --seeds 123 \
  --methods nccl,collway_like,direct_batch,relay_batch,independent_batch \
  --warmup 20 --steps 100 --max-steps 300 \
  --output artifacts/single-node-16gpu/final-v9
```

六档 tokens/rank=1/5/21/84/335/670；top-8、256 experts、hidden 2048、BF16。
Zipf=0.4/0.6/0.8 使用 T84；top-1 使用 T1/T670。每配置 5 种方法，共 11 配置。
runner 跳过已标为 complete 的同名文件；重新测量应指定新的输出目录，不覆盖旧结果。

## 静态真实路由回放

```bash
bash docker/experiment.sh python3 -m torch.distributed.run \
  --standalone --nproc_per_node=16 scripts/bench_compact.py \
  --suite trace \
  --methods nccl,collway_like,direct_batch,relay_batch,independent_batch \
  --warmup 20 --steps 100 --max-steps 300 \
  --output artifacts/single-node-16gpu/trace-v9
```

输入为 `artifacts/single-node-16gpu/qwen3-smoke-relay/routing-rank{rank}.pt`，
取第 0、24、47 层。该输入来自完整 Qwen3-30B-A3B 的 batch 1、512-token prefill；
每 rank 32 tokens、top-8、128 experts。16 个输入文件摘要见同目录 `routing-manifest.json`。
保持原始 expert IDs/gate weights，每步重新 prepare，静态路由重复用于短时计量。
它不是动态 routing 多样性实验；随机 hidden 只作为传输 payload，不计算真实专家 GEMM。

## 完整模型 batch 4

```bash
bash docker/experiment.sh /data07/fy/relaymoe-cache/vllm-venv/bin/python \
  -m scripts.bench_model_paired --batch 4 --tokens 8192 \
  --warmup 20 --steps 100 --max-steps 300 \
  --output artifacts/single-node-16gpu/qwen3-batch-v9-b4-8192.json
```

batch 1 使用相同命令，将 `--batch` 改为 1 并使用新的输出文件。
三方法分别为原生 vLLM、匹配 NCCL、RelayMoE。四个固定 prompt 模板轮换；每 10 step
轮换方法顺序。真实 gating；关闭 prefix cache、chunked prefill 和 CUDA Graph；
TP16/DP1/EP16，BF16，FlashAttention2/Triton MoE。模型加载和 JIT 位于测量之前。
完整调用包含恢复 TP token 布局的 AllGather。原生布局不同，单列系统参照。

## 汇总与绘图

在原始 JSON 完成、模型逐层记录写出后执行：

```bash
python3 scripts/summarize_model_layers.py artifacts/single-node-16gpu/qwen3-batch-v9-b4-8192.json
python3 scripts/analyze_results.py artifacts/single-node-16gpu/final-v9
python3 scripts/analyze_results.py artifacts/single-node-16gpu/trace-v9
python3 scripts/plot_payload_bandwidth.py artifacts/single-node-16gpu/final-v9 --label v9
python3 scripts/plot_results.py artifacts/single-node-16gpu/final-v9/analysis/plot-data.json
python3 scripts/plot_controls.py artifacts/single-node-16gpu/final-v9/analysis/plot-data.json
python3 scripts/plot_controls.py artifacts/single-node-16gpu/trace-v9/analysis/plot-data.json
python3 scripts/plot_supplements.py \
  --models artifacts/single-node-16gpu/qwen3-batch-v9-b1-8192.json \
           artifacts/single-node-16gpu/qwen3-batch-v9-b4-8192.json \
  --output artifacts/single-node-16gpu/model-v9-figures
```

绘图需要 matplotlib。当前本地绘图解释器为 `/tmp/relaymoe-plot-venv/bin/python`；
复用该环境时不要重新安装 GPU 运行依赖。科学图同时导出 PNG/PDF/SVG 与图注。
单 seed 阴影为 P50/P95 时延换算的带宽范围，不是置信区间。
最终收尾检查 `scripts/summarize_closeout.py` 还要求环境及 GPU 退出快照，
不以输出了图片就视为全部测量完成。

## 采样与范围

所有样本保留，在线 prepare 纳入完整通信成本。稳定性目标为尾部三个 20-step 窗口
均值极差/均值不超过 3%，所有完整窗口均值 CV 不超过 5%；达到 300 步仍不稳定即记录，
不无限延长或只保留较快样本。初始化生成的树骨架库可复用，整计划缓存关闭。
软件估算 cut 字节与实际时延分列，不将模型估算宣称为硬件带宽计数。

本阶段按用户要求收尾，只完成 seed 123 的 v9 覆盖。v8 的五 seed 数据另存；
其余 v9 seeds、尺寸加密、全模型 logits、DeepEP 与双机测试推迟到后续阶段。
