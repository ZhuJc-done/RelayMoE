# 实验记录与论文绘图口径

用户要求：优化重点放在 RelayMoE 执行效率与带宽；论文主结果突出测试中可复现的
最佳收益，保留实验数据供后续绘图。本阶段为 relay2 单机 16 卡，暂不比较 logits。

## 数据层与呈现层

- 原始 JSON 保存所有计时 step、路由配置、种子、方法、版本摘要与阶段计时，
  不根据快慢删除样本。开发期回归和未完成运行保留为诊断资料。
- 汇总 CSV/JSON 由脚本重新生成，记录完成状态和稳定性；不同版本不混合平均。
- 主图突出最佳配置，图注写清负载、对照身份、时延口径和种子数。完整扫描和
  同执行器消融单独出图，支撑主结果的适用范围和机制解释。

## 最佳收益的定义

每个配置、每个 seed、每个方法先对全部实测 step 求 P50，再计算
`speedup = baseline_P50 / RelayMoE_P50`。相同配置五个种子的 speedup 用几何
平均汇总；各配置间取最大值。时延降低为 `1 - 1/speedup`。

Dispatch、Combine、含在线 prepare 的完整通信分别选取最佳配置；不能把某个
阶段的收益写成完整模型收益。误差线展示种子间最小/最大值，不标作置信区间。
选定配置的跨种子结果使用全部计划种子，不按有利的 seed 筛选。

五个指定种子为 123/234/345/456/567。正式最佳结果要求双方所有种子满足：
末三个完整 20-step 窗口均值的 `(max-min)/mean <=3%`，以及所有完整20-step
窗口均值的总体变异系数 `std/mean <=5%`。此处将原计划的“块均值波动”明确
操作化为总体变异系数。未满足的结果保留并标注，不加入“验证后的最佳收益”。

当前 v7 通信程序只按尾漂移决定是否延长（100→200→300），汇总脚本额外计算
块变异系数；运行中不修改其源文件。后续模型程序同时检查两个指标。
未完成种子覆盖时仍可生成预览图，但标为 provisional。

## 对照与指标

- NCCL：本项目相同稀疏输入布局的真实 all-to-all-v 适配器。
- CollWay-like：用户指定的仓库算法，端点独占贪心阶段、CUDA IPC 执行；不是
  完整 CollWay 系统的复现结果。
- Direct / Independent：与 RelayMoE 同一 batch IPC 执行器的内部消融。
- 模型原生 vLLM：TP16/DP1/EP16，复制 token 输入、局部 experts、all-reduce；
  与源 token 分片的 NCCL/Relay 适配器分别标注。
- AlgoBW：远端语义有效字节 / (16 × 阶段墙钟时间)，不称为 PCIe 硬件带宽。
- Fan-out 静态计划、真实路由回放、动态合成路由和完整模型分别记录，不能混称。

CollWay-like 已覆盖通信扫描。全模型阶段重点测原生 vLLM、匹配布局 NCCL、
RelayMoE batch 的真实端到端耗时，避免将通信层仿真算法的收益扩展为整系统结论。

## 复现汇总与绘图

```bash
python3 scripts/analyze_results.py artifacts/single-node-16gpu/final-v7
python3 scripts/plot_results.py artifacts/single-node-16gpu/final-v7/analysis/plot-data.json
python3 scripts/check_result_analysis.py
```

输出为 `analysis/per-seed.csv`、`paired-seed.csv`、`aggregate.csv`、
`best-gains.json`、含输入文件 SHA256 的 `plot-data.json`。图输出在
`analysis/figures/`，包含 PNG/PDF/SVG 和独立 caption 文本。
本地绘图依赖安装在独立 `/tmp/relaymoe-plot-venv`，不修改服务器运行环境。

以有效载荷为横轴的新图使用 `scripts/plot_payload_bandwidth.py`，详细口径和补测
顺序见 `docs/payload-bandwidth-plan.md`。注意新图为 `mean(payload)/P50(time)`，
旧图为 `median(payload/time)`；不得混标。两种统计均保留，原始数据不变。
