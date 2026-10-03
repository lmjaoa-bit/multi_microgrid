# MLP Actor＋Transformer Critic（Q 引导实验）

这个独立变体检验：保持当前 T-Q 的 H8 Actor 信息和 ESS 辅助机制，把可学习的注意力结构移到 H24 Critic 后，完整强化学习训练能否产生更好的闭环成本和设备服务。

显示名称为 `Critic T-Q`；插件名称为 `matd3_mlp_transformer_critic_q_guided`。

## 实现

- 三个独立 Actor 使用两层 ReLU MLP，输出 `[ESS intent, DR intent]` 的 tanh 值。
- 固定特征与当前 `matd3_hierarchical_transformer_q_readout` 的非学习编码输入一致：ESS 与电池参数、异构设备状态、供需平衡、交易、同伴摘要、H8 本地净负荷/价格/温度/时钟、两邻居能量平衡和单独的基础负荷预测。超过 H8 的观测字段不进入 Actor；无效预测在编码前置零。
- MLP 参数预算对齐上述 T-Q；默认 MG1/MG2/MG3 隐藏层分别为 `[199,199]`、`[198,198]`、`[195,195]`，参数量分别为 64,478、65,144、66,497。当前 T-Q 对应 64,514、65,090、66,530。
- 每个微网有两个独立 Transformer Q 网络。每个网络包含一个当前全局状态与有效联合动作 token，以及 24 个全局预测时间 token。所有全局字段按 schema 完整分组，包含 MG2 空气/热质量温度、占用、舒适度、需求响应状态和 H24 温度预测。
- 每个 Q 默认 embedding=48、4 heads、2 layers、feedforward=128、dropout=0，参数量 57,505。日终以后预测由有效性掩码排除；当前状态/动作 token 始终有效，防止所有 token 同时被屏蔽。
- 保持三维有效动作/微网的共享可微投影、TD3 双 Q 目标、延迟 Actor 更新、目标策略噪声、soft update、原 replay、原奖励和原物理约束。
- ESS 随机探索概率 0.10；满足电池方向被阻挡且双 Q 都认可候选的条件时，保留权重 0.01、收益门槛 0.01 的 ESS pre-tanh 辅助目标。DR 输出没有直接辅助梯度，但与 ESS 共享 MLP 隐藏层。

`algorithm.network` 描述 H8 固定特征与 T-Q 参数预算参考，不代表运行时 Actor 含 Transformer。`algorithm.critic_network` 才配置训练中的 Transformer。训练代码不依赖诊断模块或其他 Q 引导实验包；通过正式普通 T 父类复用共享学习、投影和诊断接口。

## 运行

先验证完整短流程，包括实际参数更新、验证、最佳模型选择、正式测试、每日联盟 MILP、Equal/Shapley 结算和保存结果绘图：

```powershell
python scripts/run_experiment.py --algorithm matd3_mlp_transformer_critic_q_guided --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_mlp_transformer_critic_q_guided.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

完整训练使用独立正式 YAML，默认 CUDA、5000 回合、55 个测试日：

```powershell
python scripts/run_experiment.py --algorithm matd3_mlp_transformer_critic_q_guided
```

输出写入 `outputs/working_runs/matd3_mlp_transformer_critic_q_guided/runs/`。正式测试读取验证选择的 `best_model.pt`，使用现有 `physical_constrained_objective` 选择规则；离线诊断的逐日服务筛选不会变成额外训练约束。图表只读取已经保存的 JSON/CSV。接受的结果仍需由用户显式推广。

检查点绑定插件、Actor/Critic 架构、H8/H24、特征与掩码语义、normalizer、schema、动作投影和 Reward V2 合同。加载前核验所有网络参数和固定特征 buffer，避免不兼容或损坏的后部网络导致先前网络被部分覆盖。原 T-Q、MATD3-Q 和离线拟合模型不是这个结构的初始化检查点。

## 判断边界

短流程只检验代码与数值合同，不检验收敛或性能。离线 Transformer Critic 的排序优势不等于动作梯度可靠，也不保证完整训练收益。

这组结构同时替换 Actor 和 Critic。与原 T-Q 比较可检验推荐组合的实际效果；判断结构变化来自哪一端，仍需同 H8/H24 输入、同预算的 MLP/Transformer Actor × MLP/Transformer Critic 对照。原 MATD3-Q Actor 使用 H24，与本实验的 H8 有信息差异。

完整训练应同时报告综合成本、碳排放、峰值、交易、HVAC 舒适度、EV 离场和工业供货，特别观察 MG2 DR 与 MG1/MG3 储能。保持原共享物理模型和奖励；如果后续需要修改它们，应单列跨算法研究。
