# 多微网协同调度

本仓库研究三个异构微网的新能源、储能、可控负荷、电网和 P2P 协同调度。当前新训练只保留 MATD3 和普通 T 两个学习算法。联盟 MILP、Equal/Shapley 结算与绘图是共享评估流程。

## 两个算法

| 显示名 | 运行命令 | 实现与参数 |
|---|---|---|
| MATD3 | `python scripts/run_experiment.py --algorithm matd3` | `algorithms/matd3_dispatch/`、`configs/algorithms/matd3_dispatch.yaml` |
| 普通 T | `python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer` | `algorithms/matd3_hierarchical_transformer/`、`configs/algorithms/matd3_hierarchical_transformer.yaml` |

默认算法是 MATD3。两者的新运行都使用 `configs/cooperative.yaml` 的物理系统、`single_dispatch_dr` 双动作设计和 `peer_energy_summary` 邻居观测。MATD3 使用普通 MLP Actor；普通 T 使用层级时空 Transformer Actor。集中式 Critic 使用全局状态。标准 `algorithms/matd3/` 和 `configs/algorithms/matd3.yaml` 作为永久源码参考保留。

图中历史比较的 MATD3 运行 `MATD3_2026-09-18_15-18-57_seed42` 与普通 T 运行 `MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42` 原样保留；当时普通 T 使用 `cross_microgrid` 观测，不能把该图当作新统一观测下的架构对比。旧检查点不会被重新标记或改写。

## 使用

```powershell
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer
pytest -q
```

也可使用 `--experiment matd3`、`--experiment ordinary_t`、`--policy mlp` 或 `--policy transformer`。算法超参数分别放在自己的 YAML；环境物理参数位于 `configs/cooperative.yaml`。详细实验边界见 [当前实验方案](docs/EXPERIMENT_PROFILES.md)。

共享流程依次执行训练、验证选择 `best_model.pt`、确定性正式测试、七个联盟的每日 24 小时 MILP、Equal/Shapley 结算、摘要以及从保存结果生成 PNG/PDF 图表。普通训练写入 `outputs/working_runs/<algorithm>/runs/<run_id>/`。历史 `outputs/`、`reports/` 和 `analysis/` 保留。

经过审查且需要长期保存的运行可用 `python scripts/manage_results.py promote --run <run_dir> --release <release_name>` 复制到 `outputs/releases/`。晋升不会删除源运行；正式结果不覆盖。

## 代码边界

- `algorithms/matd3/`：标准 Actor、Critic、Replay 和 MATD3 更新的永久参考实现。
- `algorithms/matd3_dispatch/`、`algorithms/matd3_hierarchical_transformer/`：当前运行算法。
- `envs/`、`core/`、`components/`：共享环境、观测、动作映射和设备物理。
- `experiments/`、`evaluation/`、`optimization/`：共享训练、评估、结算及 MILP 基准。

新实验算法遵循根目录 [AGENTS.md](AGENTS.md) 与 [开发说明](docs/ALGORITHM_DEVELOPMENT.md)，在独立包中实现。
