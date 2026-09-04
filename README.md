# Multi-Microgrid Cooperative Optimization v1

本项目研究三个异构微网在同一初始条件下从独立运行转向联盟联合优化后，调度如何变化、系统收益增加多少，以及 MATD3 实际合作剩余如何通过 Equal Surplus Sharing 与 MILP-Shapley 贡献权重进行事后分配。

## 第一版边界

系统固定为 `3MG + RES + ESS + Load + External Grid`。协调手段为无损的合作型 P2P 能量共享。求解层包括标准 MATD3 和集中式社会规划 MILP；Nash Bargaining 只在调度完成后分配收益。

本版本不包含日前/实时市场、战略报价、竞价出清、常规机组、柔性负荷、潮流或网络约束。Shapley 不参与训练、奖励或物理调度。

## 研究流程

1. `optimization/independent_dispatch.py` 分别求解三个微网的反事实独立基准，P2P 关闭。
2. `envs/multi_microgrid_env.py` 执行合作型 MATD3 调度。MATD3 和 CrossGate 均使用一维有符号调度意图 `[signed_dispatch_intent]`；三个智能体接收相同团队奖励。
3. `optimization/cooperative_dispatch.py` 给出集中式联合优化的社会规划上界。
4. `evaluation/compare_independent_cooperative.py` 输出 ESS、SOC、外部电网、P2P 和协调重调度差值。
5. `evaluation/coalition_evaluator.py` 在每个完整测试日上分别求解 7 个非空联盟的 24h MILP value。
6. `evaluation/shapley_allocator.py` 计算三参与者 Shapley value 和贡献权重；`evaluation/settlement.py` 只按该权重分配 MATD3 实际合作剩余，并同时保留 Equal baseline。
7. `evaluation/plotting.py` 仅从已保存的 JSON/CSV 结果生成统一 PNG/PDF 图组。

## 关键约束

- 风光先服务本地负荷。Actor 请求充电后，本地风光盈余先满足该请求，剩余部分进入 P2P。
- P2P 先满足其他微网的负荷缺口，再满足其 Actor 已请求的 ESS 充电；P2P 可直接给负荷供电，也可给 ESS 充电。
- 调度先满足本地负荷，剩余放电请求形成 ESS 报量；未成交的可选电量保留在电池中。
- 同一时段禁止微网一边从外部电网购电一边向 P2P 出售，但历史时段由电网充入的能量允许以后跨微网出售。
- P2P 内部支付严格和为零，不进入联盟总收益；它只进入各微网的个体账本及事后结算。
- episode 末端按初始 SOC、容差和偏差电量施加配置化惩罚，MILP 也支持硬约束模式。
- Independent Baseline 是独立求解的反事实，不推进或修改合作环境状态。

## 快速使用

```powershell
python -m scripts.run_scenario_diagnostic
python -m scripts.run_independent_baseline
python -m scripts.run_cooperative_milp
python scripts/run_experiment.py
python scripts/run_experiment.py --algorithm matd3
python scripts/run_experiment.py --algorithm crossgate
python scripts/plot_results.py --run-dir outputs/matd3_dispatch/runs/MATD3_YYYY-MM-DD_HH-MM-SS_seed42
pytest -q
```

全局默认采用普通 MATD3 的单调度方案。CrossGate 使用相同动作和物理环境。
`--algorithm matd3` 对应实现 `algorithms/matd3_dispatch/`，参数位于 `configs/algorithms/matd3_dispatch.yaml`；
CrossGate 参数位于 `configs/algorithms/crossgate.yaml`。方案名称不再包含动作版本编号。
环境配置位于 `configs/cooperative.yaml`，完整说明见 [当前实验方案](docs/EXPERIMENT_PROFILES.md)。
历史训练结果、分析和报告保留原有名称。`algorithms/matd3/` 及其原参数文件仅作永久源代码参考。

新增隔离算法：

```powershell
python scripts/create_algorithm_variant.py --name matd3_attention
python scripts/run_experiment.py --algorithm matd3_attention
```

智能体必须遵守根目录 `AGENTS.md`；详细维护说明见 `docs/ALGORITHM_DEVELOPMENT.md`。

## 目录边界

- `envs/`：合作环境与能量共享。
- `algorithms/matd3/`：永久保留的标准 Actor、Twin Critic、Replay Buffer、目标网络和 TD3 更新。
- `algorithms/<new_algorithm>/`：独立实验算法及按需加载的 `plugin.py`。
- `configs/algorithms/`：每个算法独立的超参数，不与其他算法共享修改。
- `experiments/`：所有算法共用的实验输出布局。
- `optimization/`：公共约束、独立基准与合作社会规划 MILP。
- `allocation/`：Nash Bargaining 收益分配。
- `diagnostics/`：异构场景与合作机会诊断。
- `evaluation/`：所有算法共用的测试、MILP coalition、Equal/Shapley 结算和绘图。

训练结果统一写入：

```text
outputs/<algorithm>/runs/<run_id>/
```

`outputs/<algorithm>/latest_run.json` 指向最近一次运行；历史散落文件保存在 `legacy/`，不会与新 run 混在一起。
