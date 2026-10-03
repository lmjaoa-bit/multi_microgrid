# Critic T-Q H24：MLP Actor H24＋Transformer Critic H24

本独立实验将 `Critic T-Q` 的 Actor 最大预测窗口由 H8 扩至 H24，检验更长的 MLP 输入是否改善完整训练后的成本和设备服务。原 H8/H24 包、配置和训练结果保留。

插件：`matd3_mlp_transformer_critic_h24_q_guided`；显示名称：`Critic T-Q H24`。

## 实现与比较口径

- Actor 仍为两层 ReLU MLP，输出 ESS/DR 两个意图值。未来24步的本地净负荷、基础负荷、价格、MG2温度、时钟、预测有效性和两个同伴能量平衡进入固定特征。
- Actor 的基础负荷索引、时间特征和同伴特征均扩至24步，避免仅改变配置而遗漏部分通道。无效预测在进入 MLP 前置零。
- 继续采用原参数预算匹配规则。因输入维度增加，隐藏层宽度重新匹配：MG1/MG2/MG3 分别为 `[153,153]`、`[152,152]`、`[151,151]`；输入维度为265、271、286；参数量为64,568、64,906、66,593。因此这是预算相近的 H8/H24 与 H24/H24 对照，不能称为隐藏宽度完全相同的单变量对照。
- Critic 保持原来的 H24 Transformer twin Q：每个微网两个独立 Q，embedding=48、4 heads、2 layers、feedforward=128、dropout=0；每个 Q 为57,505参数。
- 保留共享动作投影、TD3 更新、replay、ESS随机探索概率0.10，以及双 Q 同时认可才触发的 ESS pre-tanh 辅助项（权重和收益门槛均为0.01）。
- 复用正式普通 T 的投影、训练更新和诊断接口。其初始化只允许 H8 同伴预测，因此先初始化共享机制，再在本包内构建 H24 固定特征参考并替换全部 Actor、目标网络和优化器。所有临时参考网络均不参与训练；最终检查点和 RewardV2 算法合同记录 H24。
- 本包有自己的 Actor、Critic、插件和配置，不依赖可删除的 H8 实验包或诊断代码。共享环境、奖励、训练、评估、结算与绘图流程不作修改。

共享环境仍在日终后屏蔽预测。H24表示最大读取窗口，剩余时段不足24小时会有无效预测；本实验不引入跨日状态或跨日回报。

## 完整训练

```powershell
Set-Location 'C:\daima\multi_microgrid_rl_standalone'
python scripts/run_experiment.py --algorithm matd3_mlp_transformer_critic_h24_q_guided --episodes 5000 --test-episodes 55 --device cuda --seed 42
```

默认加载 `configs/algorithms/matd3_mlp_transformer_critic_h24_q_guided.yaml`，其余正式训练设置与 H8/H24 版本一致。

结果写入 `outputs/working_runs/matd3_mlp_transformer_critic_h24_q_guided/runs/`，不会覆盖原 H8/H24 结果。正式测试读取验证选择的 `best_model.pt`，随后执行每日联盟 MILP、Equal/Shapley 结算、汇总和从保存数据绘图。

## 短流程验证

```powershell
python scripts/run_experiment.py --algorithm matd3_mlp_transformer_critic_h24_q_guided --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_mlp_transformer_critic_h24_q_guided.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

检查点绑定独立插件、H24特征、Actor/Critic结构、固定索引、normalizer、动作投影和奖励合同。H8检查点不能用作此版本的直接续训；本实验从新初始化开始。

## 结论边界

短流程和单元测试只验证实现可运行。之前 Transformer Actor H24 的负面结果不能直接决定 MLP Actor H24 的表现，也不能据窗口扩展预设本版本胜出。

完整训练后在相同日期和逐日随机场景上比较综合成本、碳排、峰值、交易、储能、MG2舒适度、EV离场和工业供货，并补重复训练种子。不得用55个测试日代替多种子证据。
