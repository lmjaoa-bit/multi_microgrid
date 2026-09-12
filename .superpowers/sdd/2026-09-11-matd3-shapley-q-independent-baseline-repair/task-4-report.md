# Task 4 实施报告：训练生命周期与诊断输出

## 实现

- 共享训练流水线通过公开可选 hook 调用 `on_episode_start`、
  `get_independent_step_data` 和 `episode_metrics`；普通算法没有这些方法时继续使用原 replay
  与原 episode schema。
- Shapley-Q replay 每步接收 Independent reward、Cooperation Surplus reward 以及当前/下一步
  Independent return-to-go；episode history 记录 Grand、Independent、Surplus 的 return/profit。
- `_diagnostic_means` 保存每个微网的 Qφ、Qφ 总和、surplus target、efficiency residual 与
  grand-decomposition residual。短 episode 没有 update 时保持 `None`，不伪造为 0。
- `training_summary.json` 新增最终 100 episode 的 Grand、Independent、Surplus、Shapley sum
  与两个 residual 均值。
- 增加隐藏的 `--warmup-steps`、`--batch-size` smoke 参数，只覆盖当前运行内存中的数值，
  不改写算法 YAML，并对负 warmup/非正 batch fail-fast。
- dashboard 对新字段做最小展示，缺失时显示 `N/A`；TensorBoard exporter 写入标准化的
  `reward/*` 与 `q/*` tags，同时保留原有通用输出。
- `Matd3ShapleyQ.episode_metrics()` 作为算法级公开适配层，不让共享流水线读取网络内部。

## TDD 与验证

- RED：新测试首次收集即因 `_algorithm_episode_metrics` 等公开 helper 不存在而失败。
- GREEN：Task 4 两个测试文件合计 `156 passed, 1 warning`。
- 全仓库：`238 passed, 2 warnings in 46.99s`；warning 均为仓库既有预期场景。
- `git diff --check` 通过；`algorithms/matd3/` 无差异。

## 边界

正式测试阶段对同日 Independent Baseline 的回调接入属于 Task 5；hard terminal SOC 和确定性
物理场景属于 Task 6；端到端产物验证属于 Task 7，本任务未提前改变这些职责。
