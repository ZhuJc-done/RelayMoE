# RelayMoE Simulation Prototype

这是一个可运行、无第三方依赖的 RelayMoE 通信仿真原型。默认路径包含：

1. 分层、全双工、有向 PCIe/NIC 拓扑；
2. 每个 token 独立随机 Top-k 的精确 gate trace；
3. NCCL、拓扑感知 matching 的 CollWay、DeepEP 与 RelayMoE 通信模板；
4. 流完成后立即回收带宽的动态 max-min 流体仿真；
5. 细粒度 token pipeline 与物理传输字节统计。

## Quick start

```bash
cd /home/tiger/RelayMoE
python3 -m relaymoe
python3 -m relaymoe --tokens-per-gpu 4096 --top-k 8
python3 -m relaymoe --nccl-contention-loss 0
python3 -m relaymoe --json
python3 -m relaymoe --legacy-stage-model
python3 -m examples.custom_scenario
```

也可以安装命令行入口：

```bash
python3 -m pip install -e .
relaymoe-sim --help
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 默认实验

默认构造：

- 2 个节点；
- 每节点 4 个 PCIe switch；
- 每 switch 4 张 GPU，共 32 张 GPU；
- EP size 为 32，32 个专家按 round-robin 放置，即每 GPU 1 个专家；
- 每 GPU 4096 个 token，共 131072 个 token；
- 每个 token 使用独立随机数无放回选择 8 个专家；
- 每份 token activation 为 1024 B；
- GPU 端口与 switch 间 PXB 均为 64 GB/s；
- 每个 switch 下挂 2 张 50 GB/s NIC，仿真为 100 GB/s 聚合 TX/RX；
- 每个有向节点对共享 400 GB/s 的网络边。


程序比较四种策略：

- `nccl`：每个 source-destination 直接点到点发送，全部 flow 无协调地并发；
- `collway`：物理字节仍与 direct 相同，但先把通信对分解成拓扑感知的
  one-to-one matching；每个 step 每张 GPU 至多一发一收；
- `deepep`：每个远端节点使用一个固定入口 GPU，跨节点只发一份，
  再由入口 GPU 在目标节点内转发；
- `relaymoe`：每个 token 根据自己的目的专家选择最多两个 node ingress，
  由真实 destination GPU 作为 switch relay，并进行细粒度流水。

```text
strategy     dispatch(us)   combine(us)    total(us)   AlgoBW(GB/s)   physical(MiB)
nccl               712.86        712.86      1425.73        1459.10         7806.07
collway            702.96        702.96      1405.92        1479.66         7806.07
deepep             618.65        618.65      1237.31        1681.29         6655.68
relaymoe             532.77        532.77      1065.53        1952.34         6230.06

RelayMoE speedup vs NCCL-like:       1.338x
RelayMoE speedup vs CollWay-like:    1.319x
RelayMoE speedup vs DeepEP-like:       1.161x
```
