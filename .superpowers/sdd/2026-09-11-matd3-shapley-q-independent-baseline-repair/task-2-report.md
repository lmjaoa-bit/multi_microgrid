# Task 2 实施报告：Fixed Independent Baseline / Common Scenario

## 状态

Task 2 已完成。实现保持 `algorithms/matd3/` canonical baseline 不变，也未提前接入
Task 3 之后的 replay、Bellman target、训练 pipeline 或正式测试输出。

## 实现摘要

- `MultiMicrogridEnv.reset()` 保存实际选中日期及原始 seed、deterministic、forecast-noise
  输入到 `last_reset_options`，不改变采样、预测或统一 `team_reward`。
- 新增 `IndependentBaselineEvaluator`：
  - 只支持 `heuristic` 与 `frozen_policy`；不支持的 mode 与缺失 checkpoint 立即失败。
  - 影子环境深拷贝 Grand 配置，仅关闭 P2P 与 ESS 跨微网合作。
  - 每个 episode 使用 Grand 的 `last_reset_options` reset，并在 rollout 前严格比较物理配置、
    context、全部 load/PV/wind 外生数组、local/public forecast arrays、forecast validity mask
    与初始 SOC。
  - 默认 heuristic 是无参数、确定性的本地 ESS 策略：前半日充电、后半日放电；P2P 为零，
    本地 ESS 与主网能量流保持真实有效。
  - frozen checkpoint 相对路径按 `env.config["_project_root"]` 解析；验证文件存在、
    `matd3_dispatch` slug、`single_dispatch_v1`、projection signature、state schema、
    observation schemas 和严格 actor state dict。
  - frozen actors 实际加载 checkpoint 权重，设置 evaluation mode 与
    `requires_grad=False`，所有推理包裹在 `torch.no_grad()` 中。
  - 缓存逐步 reward、profit、info 与 gamma-discounted return-to-go；越界或未初始化访问
    `get_step_data()` 报 `IndexError`；`episode_metrics()` 返回 baseline return/profit。
- `matd3_shapley_q.yaml` 默认 baseline 固定为无参数 heuristic，并明确 P2P disabled / frozen。

## TDD 证据

### RED

1. mode / missing checkpoint：`2 failed`；最初暴露 YAML 尚无 `shapley_q` 段，调整测试输入
   构造后，稳定失败于 `independent_baseline` 模块缺失。
2. reset lifecycle：`test_reset_records_effective_common_scenario_options` 因
   `last_reset_options` 缺失失败。
3. Common Scenario：因 `reset_episode` 缺失失败。
4. physics / step data / metrics / checkpoint / config 批次：`9 failed, 6 passed`，失败分别为
   trajectory API 缺失、checkpoint 未校验以及 YAML 默认段缺失。
5. mutation 补强：暂时撤下物理配置、projection 与 state-schema 防线后，聚焦测试得到
   `3 failed, 4 passed`，证明三条测试确实能捕捉对应回归。

### GREEN

- mode / missing checkpoint：`2 passed, 4 deselected`。
- Common Scenario：`1 passed`。
- 物理配置与 checkpoint mutation 补强：`7 passed, 15 deselected`。
- Task 2 聚焦集合：
  `pytest -q tests/test_matd3_shapley_q.py -k "baseline or common_scenario or independent or frozen_policy"`
  → `19 passed, 3 deselected`。
- 完整 Task 2 测试文件：`pytest -q tests/test_matd3_shapley_q.py`
  → `22 passed`。

## 最终验证

- `python -m py_compile envs/multi_microgrid_env.py algorithms/matd3_shapley_q/independent_baseline.py tests/test_matd3_shapley_q.py`
  → exit 0。
- `pytest -q` → `106 passed, 2 warnings in 29.90s`。两条 warning 均为仓库已有的预期
  RuntimeWarning（非有限 JSON 值归一化、非正 MILP cooperation gain）。
- `git diff --check` → exit 0，仅有 Git 的 LF→CRLF 工作树提示，无 whitespace error。
- `git diff d849caa -- algorithms/matd3/` → 无输出；canonical MATD3 未修改。
- 变更范围只有 Task 2 指定的四个源码/测试文件，以及本报告。

## 顾虑与后续边界

- 默认 heuristic 是固定、可审计且有真实能量流的 counterfactual，不代表经济最优独立调度；
  若要换成 learned baseline，必须提供严格兼容的 frozen `matd3_dispatch` checkpoint。
- 当前 Task 2 只生产 evaluator API。按任务要求没有提前把它接入 Shapley-Q replay、训练
  lifecycle 或 formal evaluation；这些属于 Task 3 及之后。因此此增量提交未运行端到端
  Shapley-Q smoke workflow，因为当前 pipeline 尚不会消费本 evaluator，运行它不能验证
  Task 2 的 baseline 生命周期。完整验收 smoke 应在后续 lifecycle 接入完成后执行。
