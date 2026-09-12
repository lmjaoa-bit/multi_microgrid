# Task 5 实施报告：Formal Test 匹配 Independent Baseline

## 实现

- `evaluate_policy_on_days` 增加默认值为 `None` 的 episode start/finalize callbacks；原有算法
  不提供 callbacks 时执行路径和输出保持不变。
- Shapley-Q 正式测试在每个 Grand episode reset 后，以同一 day、seed、deterministic 和
  forecast-noise 参数运行训练时同一个固定 Independent Baseline policy。
- 每日 metrics 和 CSV 新增 Grand、Independent、Cooperation Surplus 的 return/profit，
  并严格按 Grand 减 Independent 计算。
- JSON summary 新增 Independent 和 Surplus 的 return/profit 分布统计；paired daily 也保存
  相同字段。
- 在线固定 baseline 字段与现有 coalition MILP 的 `independent_total_profit` 保持不同命名，
  后者仍只用于 benchmark、settlement 和绘图。

## TDD 与验证

- RED：callback 测试因不支持 `episode_start_fn` 失败；真实一日 formal evaluation 因 CSV
  缺少 `cooperation_surplus_return` 失败。
- GREEN：两项精确测试通过；formal/evaluation/baseline 聚焦集合 `30 passed`。
- 全仓库 `pytest -q`：`240 passed, 2 warnings in 48.08s`。
- warning 均为既有预期场景；canonical `algorithms/matd3/` 无修改。
