# 当前实验方案

新训练只提供 MATD3 和普通 T 两个算法，共用 `configs/cooperative.yaml` 的物理系统、`single_dispatch_dr` 双动作设计以及 `peer_energy_summary` 邻居观测。默认算法为 MATD3。

| 显示名 | 命令与实验名 | 实现目录 | 算法配置 |
|---|---|---|---|
| MATD3 | `--algorithm matd3` 或 `--experiment matd3` | `algorithms/matd3_dispatch/` | `configs/algorithms/matd3_dispatch.yaml` |
| 普通 T | `--algorithm matd3_hierarchical_transformer` 或 `--experiment ordinary_t` | `algorithms/matd3_hierarchical_transformer/` | `configs/algorithms/matd3_hierarchical_transformer.yaml` |

`--algorithm matd3` 是 `matd3_dispatch` 的别名。也可用 `--policy mlp` 或 `--policy transformer` 选择相应实验方案。`algorithms/matd3/` 与 `configs/algorithms/matd3.yaml` 是永久基线源码参考，不作为第三个运行方案。

## 公平比较边界

两个插件为新运行解析出相同的环境配置和观测字段。每个 Actor 接收自身物理与设备状态，以及邻居的能量盈缺、充放电灵活度和上一时刻 P2P 净交易；邻居原始 SOC 不直接进入 Actor 观测。集中式 Critic 仍使用全局状态。两个算法的 Actor 架构不同：MATD3 使用普通 MLP，普通 T 使用层级时空 Transformer。

历史结果不随新默认值改写。图中 MATD3 运行 `MATD3_2026-09-18_15-18-57_seed42` 使用 `peer_energy_summary`，普通 T 运行 `MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42` 使用 `cross_microgrid`。两者的观测输入不同；旧普通 T 检查点不能作为新观测配置的检查点。要比较同一观测下的两个架构，须分别重新训练、验证和测试。

## 运行

```powershell
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer
```

两者使用共享流程：训练 → 验证选出 `best_model.pt` → 确定性测试 → 逐日联盟 MILP → Equal/Shapley 结算 → 摘要 → 仅从保存结果绘图。新结果分别写入 `outputs/working_runs/matd3_dispatch/runs/` 和 `outputs/working_runs/matd3_hierarchical_transformer/runs/`。历史 `outputs`、报告和分析文件保留原标签与内容。
