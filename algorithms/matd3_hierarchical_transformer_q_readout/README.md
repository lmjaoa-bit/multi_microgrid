# 普通 T：Q 引导与设备预测读取

本变体继承当前 T-Q，保持共享物理环境、奖励、H8 预测窗口、Q 引导规则和正式训练预算。
当前 T-Q、普通 T 与规范 MATD3 包均不修改。

## 改动

- 每小时预测 token 追加原观测中的基础负荷，按原负荷尺度归一化。原有净负荷通道保留；
  默认预测输入从 8 维变为 9 维，输出仍为 32 维。
- 保留 14 个 token、两层四头共享 Transformer 和设备原始状态线性旁路。
- ESS 与 DR 分别以融合后的设备 token 为查询，对有效 H8 预测 token 执行一层四头注意力。
  两个 32 维上下文各自拼接到对应动作头，动作头输入从 128 维变为 160 维。
- 全部预测无效时两个新增上下文均严格置零；跨日预测不参与读取。
- 动作生成与 ESS Q 引导共用 `action_logits`，避免 Q 引导遗漏新分支。
- 新参数进入 Actor 优化器、目标 Actor、检查点和分支梯度诊断。

预测 token 中仍包含两个邻居的未来能量平衡。未新增占用日程、未来舒适边界或真实未来信息。
ESS/DR 共享主干；Q 引导依旧允许更新共享表示，未额外隔离共享梯度。

## 正式训练

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_q_readout
```

默认 5000 回合，运行设备为 CUDA。此变体应从头训练，不能加载原 T-Q 检查点。
正式配置位于 `configs/algorithms/matd3_hierarchical_transformer_q_readout.yaml`。
输出位于 `outputs/working_runs/matd3_hierarchical_transformer_q_readout/runs/`。

## 短流程验证

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_q_readout --algorithm-config configs/smoke/matd3_hierarchical_transformer_q_readout.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

短流程覆盖真实参数更新、验证选择、best_model 正式测试、联盟 MILP、Equal/Shapley 与图表。
短流程用于数值与流程验证，不作为性能提升证据。效果需由完整训练后的成本、HVAC 舒适和 EV 服务指标判断。

## 本次验证记录（2026-10-01）

- 新变体专项测试：12 项通过，包含 CUDA、无效预测遮罩、Q 引导、真实参数更新、诊断和检查点。
- 全量 `python -m pytest -q`：466 项通过、3 项失败，尚未达到全量通过。
- 三项失败在不加载新变体的独立测试进程中也能复现，本次未修改这些测试或其实现：
  - `test_audit_is_read_only_and_marks_missing_historical_evidence`：Windows 路径分隔符与预期不一致。
  - `test_no_demand_region_has_zero_local_gradient`：共享出清在无需求、零出力边界仍有非零梯度。
  - `test_power_normalizer_buffer_is_nonpersistent_and_moves_with_actor`：原普通 T 的 CPU/CUDA 输出差异超过测试容差。
- 最终短流程：3 回合训练、2 天验证、2 天正式测试；使用验证选中的 `best_model.pt`。
  联盟 MILP 无失败，Equal/Shapley 与汇总完成，生成 15 PNG 和 15 PDF。
- ESS/DR 新分支的梯度已在真实训练中记录；MG2 最后一回合平均梯度范数分别约为 0.000938、0.000913。
- 检查了规范 MATD3、当前普通 T/T-Q、共享环境配置等 15 个受保护文件，内容哈希与改动前一致。
- 最终短流程运行：`普通TQ引导与设备预测读取_2026-10-01_18-38-54_seed42`。
- 未启动 5000 回合完整训练，因此本记录不声称成本或舒适表现已经提升。
