# Task 6 实施报告：Hard Terminal SOC 与确定性物理语义

## 实现与验证

- 共享物理配置按修正说明切换为 `terminal_soc.mode: hard`、`tolerance: 1e-6`；三个微网
  的 initial/min SOC 均保持 0.1。
- hard 模式继续使用环境已有的可达域约束强制日末回到初始 SOC，但 `_soc_potential` 和
  terminal penalty 仅在显式 `penalty` 模式下参与 reward/cost。因此 hard 模式的每一步
  shaping 和 terminal penalty 都为 0。
- 原“无终端约束”测试改为显式设置 `mode: none`，保留旧语义并避免依赖全局默认。
- 新测试用真实 24 小时环境检查 hard 模式下逐步无 shaping/penalty、奖励有限、最终 SOC
  回到初始值。
- 临时确定性 CSV 构造真实正合作场景：Grand 有 P2P 流量，profit 和 return 均高于关闭
  合作的 Independent 环境。
- 同一确定性无合作配置的两次 rollout 得到完全一致的 reward/profit，并由 exact Shapley
  工具验证全部贡献为 0。
- empty coalition projector 使用 `[batch,3]` mask，验证本地 dispatch 保留、合作 offer
  清零、membership 全 false。

## TDD 证据

- 默认 hard 测试最初因配置仍为 `none` 失败；修正后 Task 6 聚焦集合为
  `17 passed, 147 deselected`。
- 全仓库 `pytest -q`：`244 passed, 2 warnings in 49.19s`。
- warning 均为既有预期场景；canonical `algorithms/matd3/` 无修改。
