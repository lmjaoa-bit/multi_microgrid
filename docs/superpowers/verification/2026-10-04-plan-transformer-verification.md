# Plan-Transformer 实现与验收记录

2026-10-04；在用户选择的当前目录、`codex/dueling-h24-study` 分支内执行。

## 已交付行为

独立插件 `matd3_plan_transformer` 使用两层 NOW query cross-attention，读取
24 个未来 token 和 2 个 peer token。当前动作仍为 ESS/DR 两个 intent；
24 步 SOC/实际 ESS 功率计划仅用于辅助监督。计划误差、SOC 动力学误差和
投影后首步一致性损失参与延迟 Actor 更新，两层共享注意力均收到辅助梯度。

教师只使用当前状态、当时可用的冻结预测和公开 EV cohort 先验。CEM 在有限
预算内保留最好可行候选，使用原有环境调度与广义成本；不提供最优性证书。
推理与正式测试不调用教师。首版明确拒绝训练 resume。

正式配置：`configs/algorithms/matd3_plan_transformer.yaml`。
短流程配置：`configs/smoke/matd3_plan_transformer.yaml`。
设计与执行计划见同目录体系内的 specs/plans 文档。

## 最终验证

- `pytest -q`：**698 passed, 4 warnings，193.51 秒**。四条警告类别与
  658 项通过的原始基线相同；Actor 的实际 loss/梯度检查均为有限值。
- 修正后的完整 CPU 短流程：3 个训练日，每轮 2 个验证日，2 个正式测试日。
  使用共享 `scripts/run_experiment.py`，保留 validation、MILP、结算与出图。
- 训练/验证阶段共 13.55 秒；教师累计约 7.01 秒。
- 教师求解 12 次，产生 12 条有效联合标签；实际辅助 Actor 更新 32 次。
  三个 Actor 的每轮平均辅助注意力梯度均大于 0。
- 正式测试采用第 1 轮选出的 `best_model.pt`，测试日期/seed 为
  `2018-01-28/42`、`2018-01-29/43`；每个测试日完整 24 小时。
- 每日 7 个联盟 MILP 全部 `kOptimal`；Equal/Shapley 预算守恒误差最大
  `1.71e-13`。正式测试教师求解次数为 0。
- 保存 144 条正式测试计划/注意力记录，含原始观测、checkpoint hash、日期、
  seed、微网、时刻、两个 mask、计划和逐层逐 head attention。
- 标准根目录为 **15 PNG + 15 PDF**；诊断子目录为 **12 PNG + 12 PDF**。
  重绘生成 54 个文件，既有 checkpoint/JSON/CSV 的 hash 不变。
- 已检查注意力热图和 SOC/ESS 对照图；无效 forecast 格留白，peer 标签可读，
  辅助计划与实际调度明确分开。
- 单独保存观测扰动研究成功，源 run 全部文件 hash 保持不变。
- 对真实已完成 run 尝试不支持的 resume，明确拒绝；其文件 hash 保持不变。
- 相对执行起点，canonical MATD3、MATD3 dispatch、普通 T、共享环境 YAML
  和永久 MATD3 YAML 无改动。四项既有未跟踪 oracle replay 文件保留。

最终 working run：

```text
outputs/working_runs/matd3_plan_transformer/runs/PLAN-TRANSFORMER_2026-10-04_19-55-34_seed42/
```

正式测试 checkpoint SHA256：

```text
c92b4ab40481594072cc73f61989abdc4e3c44e9662e486805f19f2409da8f03
```

独立研究：

```text
outputs/working_runs/matd3_plan_transformer/studies/PLAN-TRANSFORMER_2026-10-04_19-55-34_seed42_sensitivity/
```

## 终审与处理

独立终审未发现 Critical 问题；提出的三个 Important 问题均已有先失败、后
通过的回归检查并修复：HVAC 使用本地天气预测；重复诊断配置/不支持的续训在
改变证据前拒绝；绘图对照实际 daily CSV 检查日期、seed 和完整唯一小时覆盖。
同时修复了冻结场景 profile cache 与 materialized exogenous 不一致的问题。
更新三处写死插件名单的现有测试，保持原算法选择与默认 MATD3 断言。

初次工程 smoke 保留，最终验收使用上述修正后的新 run。未晋升 release、
未删除既有结果。源码与验收文档在当前分支提交并保留。

## 解释范围

此短运行只验证完整执行和监督接入。3 个训练日的计划与实际轨迹仍有明显
差异；不能据此声称收敛、性能改善或注意力已学会关键时段。热图与观测扰动
是诊断证据。多种子、无辅助监督消融和完整训练效果研究尚未执行。

完整测试日志及工程检查记录保存在本计划的本地 `.superpowers/sdd/` 工作目录。
