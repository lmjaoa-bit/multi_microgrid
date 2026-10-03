# Critic T-Q 实现与验收（2026-10-02）

已建立独立的 `matd3_mlp_transformer_critic_q_guided`，显示名称 `Critic T-Q`。本轮完成结构实现、数值验证和共享完整短流程；尚未启动 5000 回合完整训练，也没有证据宣布优于 MATD3-Q。

## 结构与保持项

Actor 为两层 MLP，使用已完成诊断中的当前 T-Q 共同非学习 H8 特征，保留独立基础负荷和两同伴能量平衡预测。三个 Actor 参数量与当前 T-Q 的差异均低于 0.1%。Critic 保留现有完整 H24 全局状态，分成一个当前状态/有效联合动作 token 和 24 个预测时段 token；每个微网的 Q1/Q2 分别使用独立的 Transformer。

保留原 ESS 探索和双 Q 引导 pre-tanh 辅助目标、目标策略噪声、延迟策略更新、soft update、共享奖励及可微物理投影。HVAC 跨时段评价通过当前空气/热质量温度、设备状态、联合动作和 H24 预测建立联系。本轮没有增加真实未来、HVAC 专属新奖励或新服务约束，也没有载入离线拟合 Q 权重。

`algorithms/matd3/`、原普通 T/T-Q、环境、共享训练/评估/结算/绘图实现保持开工时内容。41 个受保护 Python 源文件的 SHA256 全部一致，诊断登记的两个 5000 回合来源模型及配置、历史、汇总、最佳检查点哈希也全部一致。

新算法运行时依赖正式普通 T 的共享更新和固定特征元数据，不依赖 `diagnostics` 或可删除的其他 Q 引导实验包。检查点具有自己的插件与网络合同，并在任何权重写入前核验全部网络状态、固定 buffer 和更新计数。

## 测试与独立审查

新增 `tests/test_mlp_transformer_critic_q_guidance.py` 共 19 项测试通过，覆盖：

- 当前 T-Q 固定特征一致性与参数预算；超过 H8 的预测不影响 Actor；无效预测不会进入计算。
- H24 字段无损分组、Q1/Q2 权重独立、日终全部预测被屏蔽时数值仍有限。
- MG2 DR 动作梯度与有限差分一致；实际 Actor/Critic/目标网络参数发生更新。
- 学习起步门槛、确定性测试与 ESS 探索隔离、ESS 饱和 logit 恢复和双 Q 门槛。
- 检查点往返、网络结构不兼容、后部网络损坏、固定特征 buffer 被篡改时加载拒绝且不会部分改写权重。
- 默认网络规模在 CUDA 上完成真实更新，并加载 CPU 保存的模型。
- 省略同伴预测开关时仍应用 H8 同伴默认输入，且不修改调用方配置。

独立审查确认共同特征一致、全部 9 个有效动作通道具有有限非零 Q 梯度、共享更新调用新的 ESS 辅助项，以及运行时实验依赖隔离。审查发现的同伴预测默认值不一致已通过失败回归测试复现、在本地配置副本中修复并通过验证。

修改前全量 `pytest -q` 为 **518 passed、3 failed**。本轮最终全量为 **537 passed、3 failed、4 warnings**，仍是以下三项已有失败，未宣称全量通过：

1. `tests/test_audit_saved_service_metrics.py::test_audit_is_read_only_and_marks_missing_historical_evidence`：历史审计源路径清单与预期不一致。
2. `tests/test_executed_action_equivalence.py::test_no_demand_region_has_zero_local_gradient`：无需求场景的共享投影在零点仍返回其他储能分量 0.5 的次梯度。
3. `tests/test_plain_t_power_features.py::test_power_normalizer_buffer_is_nonpersistent_and_moves_with_actor`：原 T 的 CPU/GPU 浮点输出差异超过现有容差。

插件清单测试只追加新 slug，原工作区已有的修改保持原样；共享物理投影与原 T 数值问题未混入本次结构实验修复。

## 共享短流程结果

最新验收目录：

`outputs/working_runs/matd3_mlp_transformer_critic_q_guided/runs/CRITICT-Q_2026-10-02_21-10-37_seed42/`

通过 `scripts/run_experiment.py` 加载新插件及其独立 smoke YAML，训练 3 回合、72 个环境步、65 次 Critic 更新及 32 轮延迟 Actor 更新；使用正式网络尺寸，smoke 仅减少 replay/起步与批量规模。3 次验证在每次 2 天上选优，第 2 回合的 `best_model.pt` 被正式测试读取，检查点 SHA256 与测试记录一致。

测试 2 天完成每日七个联盟 MILP、综合成本参考、Equal/Shapley 结算、设备服务指标、汇总和保存文件绘图。MILP 失败数 0，Shapley 有效日 2/2，结算最大预算平衡误差约 5.68e-14。`figures/` 有 15 PNG 和 15 PDF；服务数据覆盖完整，均写入新插件自己的 working-run 目录。

同配置的首次短流程结果也保留在相邻旧目录；显示名称缩短为 `Critic T-Q` 后再次完成短流程，避免汇总图的长 run ID 覆盖右列。汇总图已视觉抽检。3 回合收敛图沿用共享模板，最佳点文字与标题距离偏近，不能当成论文图；本轮未改动共享绘图逻辑。

短训模型日均综合成本约 1204.25，占用时段舒适率 0%，严格服务可行率 0%。这些数值仅证明服务退化被记录，不能用于结构排名或成功声明。

检查点、逐日表、设备轨迹、服务汇总、结算数据和全部图已保存。`verification/implementation_verification.json` 记录数值验收，`verification/pytest_final.txt` 保存全量测试原输出。没有推广到 releases，也未删除任何已有结果。

## 完整训练入口与后续判断

```powershell
python scripts/run_experiment.py --algorithm matd3_mlp_transformer_critic_q_guided
```

默认加载 `configs/algorithms/matd3_mlp_transformer_critic_q_guided.yaml`，CUDA，5000 回合，55 个正式测试日。下一步需用完整训练、多初始化和相同测试场景确认成本、碳、峰值、交易、HVAC 舒适、EV 离场与工业供货；特别关注 MG2 DR 及 MG1/MG3 储能是否退化。

组合与原 T-Q 的差异同时涉及两端结构。要回答 Transformer 放在哪端更值得，仍需保持 H8/H24 信息与训练预算一致的 Actor×Critic 结构对照。原 MATD3-Q Actor 使用 H24，其已有成绩可作实际性能目标，但不能作为同信息架构归因的唯一对照。
