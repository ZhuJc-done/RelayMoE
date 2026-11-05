# RelayMoE

- 可持久化的离线二项中继树目录；
- 在线 TrafficClass 聚合、势函数选树及 GPU/NIC 绑定；
- 保留父子依赖和具体 NIC 路径的事件驱动、有向链路独占调度；
- 对称反向 Combine DAG 与中间节点部分归约元数据；
- 面向真实 MoE 层的 `plan → dispatch → expert compute → combine` 接口；
- NCCL、CollWay、DeepEP 通信仿真基线。

## MoE 上层接口

```python
relaymoe = RelayMoE(topology, backend=my_cuda_rdma_backend)
plan = relaymoe.plan_from_gating(
    expert_indices_by_gpu,       # [EP rank][local token][Top-k]
    expert_to_gpu=expert_to_gpu,
    expert_weights_by_gpu=gate_weights,
    activation_bytes=hidden_bytes,
)

dispatched = relaymoe.dispatch(plan, hidden_states)
expert_outputs = experts(dispatched)
outputs = relaymoe.combine(plan, expert_outputs)
```

`RelayMoEPlan` 同时保留 token/expert membership、树边依赖、源/目的 GPU、
具体 NIC 路径以及 Combine 部分归约信息。真实 CUDA/RDMA 实现只需实现
`RelayMoEBackend.execute()`。完整示例位于 `examples/moe_integration.py`。

## 测试结果

测试配置：2 节点 × 16 GPU（EP=32），每节点 4 个 PCIe Switch、每个
Switch 4 张 GPU 和 2 张 50 GB/s NIC；每 GPU 512 tokens，Top-8，
32 experts，激活大小 1 KiB，`routing-motifs-per-gpu=8`，`seed=7`。

```text
策略          Dispatch(us)  Combine(us)  Total(us)  RelayMoE 加速比
CollWay-like       193.16       193.16     386.31          2.143x
DeepEP-like        122.06       122.06     244.12          1.354x
RelayMoE             90.88        89.43     180.30               -
```

复现命令：

```bash
python3 -m relaymoe \
  --tokens-per-gpu 512 \
  --routing-motifs-per-gpu 8 \
  --nccl-contention-loss 0
python3 -m unittest discover -s tests -v  # 23 tests
```
