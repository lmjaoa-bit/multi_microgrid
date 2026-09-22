# 全知 MILP 差距分解：原始输入与清算第一道验收

研究基准：`codex/matd3-cross-microgrid-comparable-20260922`（父提交 `238b490f64ebac36969f84046ace1ee6b8985320`）。本分支新增独立审计脚本，不更改基准 MATD3、Transformer、环境或正式 coalition MILP。

## 运行

将交接包 `全知MILP差距分解_交接数据包.zip` 保存在本地任意目录。确保仓库内原始 `data/spatial_market_2018_test.csv` 和 `data/spatial_market_2018_train.csv` 存在。PowerShell：

```powershell
python scripts/audit_clairvoyant_gap_inputs.py --handoff-zip "C:\实际目录\全知MILP差距分解_交接数据包.zip"
```

逐日审计 55 天 × 2 个 RL 策略，读取同一交接包保存的小时轨迹，以当前仓库的 `DataManager` / `HeterogeneousProfileModel` 从原始训练、测试 CSV 生成基础负荷和可再生功率；使用原始每日逐微网电价和 RL 已保存的调度、报价字段重构原环境 `CooperativeEnergySharing.allocate_structured` 的报价/需求，再逐微网、逐有向边与 RL 已记录成交对账。不会调用旧版天气反推算法。

审计产物：`outputs/clairvoyant_gap_input_audit/original_input_p2p_audit_110.csv`、`original_input_p2p_mismatches.csv`、`audit_summary.json`。摘要保存配置和原始 CSV 的 SHA256；遇到输入/P2P 错位会非零退出。请保存审计摘要、逐日表和 mismatch 文件；**只有全部 110 条原始外生数据检查及 P2P 回放同时通过后**，才能进入相同动作的成本/设备状态审计。

## 不得越过的验收门槛

此脚本只认证数据输入与已执行报价条件下的撮合重放，**没有**验证 MILP 自由优化结果能否在环境中执行，也**没有**生成 `free/fixed_both/fixed_dr/fixed_ess` 的新目标值。已知修订模型 55 天自由优化成功，但 MATD3 和 Transformer 的固定动作共同完整解仅 53 天。

两例问题分别保留：`2018-01-29` 两种策略存在 HVAC 占用时间入口舒适软违约，修订 MILP 硬约束可能排除原环境轨迹；`2018-02-26` Transformer 存在限供，DR `delivered_fraction` 对后续设备状态有影响。修订模型 `outgoing + actual_load + battery_charge <= renewable + battery_discharge` 也可能排除环境中 P2P 给电池充电且同时向其他微网出售自有能源的情况，不能靠加严线性不等式修复。

后续应使用原环境重放 RL 原始 raw action，以小时粒度逐项认证 `actual_load_kw`、ESS SOC、DR 状态、P2P 流、主网、`delivered_fraction`、终端惩罚、联盟成本；若确有不一致，先定位来源，再为固定动作 MILP 定义与环境一致的软违约/供给来源模型、回归失败两天并重新求解 55 天四组场景。保留旧 555.342194 与修订版 565.165128 不同模型的版本标签；不得把修订模型内部 53 天 ESS/DR 差额解释为环境可实现的独立收益。
