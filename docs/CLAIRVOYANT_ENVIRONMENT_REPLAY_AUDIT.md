# 全知 MILP：正式环境回放审计（第一阶段）

此分支从 \`codex/matd3-cross-microgrid-comparable-20260922\` 的提交 \`238b490f64ebac36969f84046ace1ee6b8985320\` 建立，只新增审计入口，不改 MATD3、Transformer、环境、历史收益和旧 coalition/Shapley MILP。

## 为什么先审计，暂不发布“可兑现的设备贡献”

交接包中的 source-aware revised MILP 是参考模型，并没有把 \`MultiMicrogridEnv._step_dr\` 的四阶段 \`CooperativeEnergySharing.allocate_structured\`、本地 ESS 优先、充电能源来源、供能不足时的 \`delivered_fraction\`、HVAC 舒适带不可达时的允许越界完整模拟。55 天 \`free\` 求解成功，但同一 55 天内 MATD3 固定动作完全成功 54 天、普通 Transformer 53 天，两算法共同可比较的仅 53 天。

特别是：
- \`2018-01-29\`：两个 RL 策略在动作 06:00 执行后的 HVAC 07:00 边界温度分别约 20.592898 / 20.692960 °C，低于占用舒适下界 21°C；原环境在物理能力不足时记录舒适不可达，候选 MILP 对该边界施加硬下限。
- \`2018-02-26\`：Transformer 在 01:00 的 MG3 负荷 202.713757 kW、主网输入 200 kW、无 P2P 输入及 ESS 放电，存在 2.713757 kW 缺供；原环境对设备按实际供能分数执行，候选 MILP 未建模该执行场景。后者是确定的建模差异，仍需核查是否还存在其他冲突。
- 共同比较的 53 天：\`RL实际成本 - 修订MILP(fixed_both)\` 日均 MATD3 约 59.064687、Transformer 约 59.034946。这是模型执行/结算对账余额，**不得标为纯 P2P 收益**。

旧放宽 oracle 55 天平均成本 555.342194，新 source-aware revised 55 天 \`free\` 平均成本 565.165128，只有 7 天旧/新目标在 0.02 内一致；二者不能跨版本拼接。旧放宽 oracle 存在 81 条高于 local-first 来源上界的出售记录。

## 从实际 GitHub 空间数据重放两份已保存策略

本分支的 \`scripts/audit_clairvoyant_environment_replay.py\` 不是第二个优化器：只读取交接包内原始 RL **raw action** 及逐小时实际结果，使用本分支现有 \`MultiMicrogridEnv\` / \`DataManager\` / \`CooperativeEnergySharing\`，对 55 天 × 2 策略进行三级核对：
1. GitHub 原始 \`data/spatial_market_2018_{test,train}.csv\` 经 \`DataManager\` 的基础负荷、实际风光是否与记录相符；
2. 重建四类报价/需求，调用原版 \`allocate_structured\`，检查每对微网逐小时 P2P 流、负荷购电、电池充电等；
3. 在新建的环境回合中逐小时送入保存的两维 raw action，核对实际负荷、ESS 充放电/SOC、P2P、主网、缺供/弃电、HVAC 状态、小时收益和全天收益。

请将原交接压缩包保存在本地，不需要解压，也不要把它提交到 GitHub。PowerShell 示例：

\`\`\`powershell
cd C:\daima\multi_microgrid_rl_standalone
git fetch origin
git switch --track origin/codex/clairvoyant-environment-replay-audit-20260922
python scripts/audit_clairvoyant_environment_replay.py --package "C:\替换成你的真实路径\全知MILP差距分解_交接数据包.zip" --output outputs/clairvoyant_environment_replay_audit
\`\`\`

产物：\`paired_replay_daily.csv\`（保留全部日期以及失败原因）、\`replay_mismatches.csv\`、\`summary.json\`。\`all_110_passed=true\` 才能报告“保存 raw action 在当前代码/数据配置上的重新执行与历史结果吻合”。否则先核查哪一级发生差异：Git dirty 的运行快照、测试日种子/噪声时序、原数据变动、P2P 价差或当前代码变化。不能为了使检查通过而更改容差、丢弃失败日期或用英国旧天气反推代替原始空间测试集。

即使 110 个 replay 全部通过，这**只证明保存策略动作的环境执行对账**；并不证明放宽 MILP 的计划可在该环境执行。后续要对 \`fixed_both\` 的账目逐时对齐后，才可在一个有明确可行性/状态覆盖的统一模型中分别运行 \`free/fixed_dr/fixed_ess/fixed_both\`，每一天保留 solver status、MIP gap 和物理违反/软违约记录。候选重算脚本中含硬编码容器路径与旧天气恢复，不能作为正式的 55 天结果入口。
