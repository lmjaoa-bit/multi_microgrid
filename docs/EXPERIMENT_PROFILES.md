# 当前实验方案

项目只提供 MATD3 和 CrossGate 两个运行方案，共用单调度动作 `single_dispatch`。
全局默认是 MATD3，配置在 `configs/experiment_defaults.yaml`；不再使用动作版本编号命名实验。

| 实验名 | 算法目录及配置标识 | Actor |
|---|---|---|
| matd3 | matd3_dispatch | 完整观测 → Linear(128) → ReLU → Linear(128) → ReLU → Linear(1) → Tanh |
| crossgate | crossgate | 本地/市场/同行分别编码 → 交叉门控 → 拼接 → 全连接网络 → 单调度输出 |

两者使用相同环境、观测、物理动作投影、Critic 信息和训练设置（包括梯度裁剪）。
该对比评估完整 Actor 架构，不能将收益差异仅归因于门控。

## 运行

```powershell
python scripts/run_experiment.py
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm crossgate
```

`--algorithm matd3` 是 `matd3_dispatch` 的便捷别名。
也可以使用 `--experiment matd3`、`--experiment crossgate`，或 `--policy mlp`、`--policy crossgate`。
命令行显式选择优先于全局默认；清除默认时仍回退到 MATD3 单调度方案。

```powershell
python scripts/manage_experiments.py --list
python scripts/manage_experiments.py --set-default matd3
```

## 单调度动作

每个微网输出 u ∈ [-1, 1]。通常可停机时，负值请求充电，零值停机，正值请求放电。
放电先满足本地负荷，剩余请求形成 ESS 报量；只实际释放成交的可选 ESS 电量。
终端可达性强制充放电时，整个动作区间映射到相应可行功率区间，此时零值不一定停机。
必要终端放电可以上网。报量受同行需求、交易价格可行性和链路容量约束。

Actor 输出一维，Critic 接收投影后的两个物理量：本地/必要调度与 ESS 报量。
环境执行、Replay 重建、目标动作与 Actor 更新共用 `core/single_dispatch.py`。
基础物理配置是 `configs/cooperative.yaml`，命名环境是 `cooperative_standard`。

## 验证与输出

训练 → 验证选择 best_model.pt → 确定性测试 → 每日联盟 MILP → Equal/Shapley 结算 → 摘要 → 保存结果绘图。
联盟 MILP 是保留的评估与结算工具，不是另一个训练算法。

```powershell
pytest -q
python scripts/run_experiment.py --algorithm-config configs/smoke/matd3_dispatch.yaml --no-dashboard
python scripts/run_experiment.py --algorithm crossgate --algorithm-config configs/smoke/crossgate.yaml --no-dashboard
```

新运行分别写入 `outputs/matd3_dispatch/runs/` 与 `outputs/crossgate/runs/`，每次运行独立保存配置、模型及图表。
`algorithms/matd3/` 与 `configs/algorithms/matd3.yaml` 仅作为永久参考保留，不列为第三个运行方案；参考网络和更新实现没有修改。
旧算法代码、专用配置和测试已移除。历史 outputs、reports 与 analysis 文件原样保留，不改写旧标签，不混入新结果。
旧检查点的算法标识仍保留历史含义，当前加载器不会将它们自动当作新方案检查点。
