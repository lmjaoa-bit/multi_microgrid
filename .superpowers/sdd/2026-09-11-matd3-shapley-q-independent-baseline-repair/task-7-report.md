# Task 7 验收报告：全量测试与端到端 Smoke

## 自动测试

- `pytest -q tests/test_matd3_shapley_q.py`：`157 passed in 29.41s`，无 skipped。
- `pytest -q`：`244 passed, 2 warnings in 48.42s`，0 failed。
- 两条 warning 是仓库既有测试主动覆盖的非有限 JSON 清理与非正 MILP cooperation gain。

## 端到端 Smoke

执行：

```text
python scripts/run_experiment.py --algorithm matd3_shapley_q --episodes 2
  --test-episodes 2 --validation-interval 1 --validation-days 1
  --warmup-steps 0 --batch-size 8 --no-dashboard
```

运行目录：

```text
C:\daima\multi_microgrid_rl_standalone\outputs\matd3_shapley_q\runs\MATD3SHAPLEYQ_2026-09-12_14-12-05_seed42
```

流水线完成 training → validation → best checkpoint → deterministic formal test →
daily coalition MILP → Equal/Shapley settlement → summary → saved-result plotting。

## 产物与身份审计

- training episodes=2，steps=48；两个 episode 均有 Critic 与 delayed Actor update 指标。
- validation records=2；best episode=2；selection metric=`validation_team_profit_mean`。
- `best_model.pt`、`final_model.pt` 均存在；formal `checkpoint_used` 精确指向
  `checkpoints/best_model.pt`。
- formal test days=2；hourly trajectory CSV=2。
- figures：PNG=15，PDF=15。
- `run_metadata.json`、`summary.json` 的 slug 均为 `matd3_shapley_q`；算法配置路径为
  `configs/algorithms/matd3_shapley_q.yaml`；run 位于自己的标准 outputs 目录。
- 必需 JSON/CSV/YAML 文件无缺失；正式 YAML 仍为 warmup=1000、batch=128，smoke override
  未写回配置。

## 数值与后处理审计

- 两条训练记录的 `Grand return - Independent return = Surplus return`：0 errors。
- 两条训练记录的 profit 对应恒等式：0 errors。
- 两条 formal daily CSV 的 return/profit 对应恒等式：0 errors。
- 已发生 update 的训练记录中，所有 Grand/Independent/Surplus、Qφ、Qφ sum、efficiency
  residual、grand-decomposition residual 均非 null 且 finite。
- summary 的六个 final-100 新字段和 formal baseline/surplus statistics 均非 null 且 finite。
- 两个测试日的七联盟 MILP 均成功：failure count=0；Shapley valid days=2，invalid=0；
  最大预算平衡误差 `1.13686837721616e-13`。

## 隔离检查

- `git diff d5d558f..HEAD -- algorithms/matd3` 无输出。
- `git diff --check d5d558f..HEAD` 通过。
- canonical MATD3 未修改；所有实验实现位于 `algorithms/matd3_shapley_q/` 与独立 YAML。
