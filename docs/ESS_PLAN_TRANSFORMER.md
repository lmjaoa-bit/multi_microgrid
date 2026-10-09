# 储能规划 T

独立插件 `matd3_ess_plan_transformer`。保留当前 Dueling H24 Critic、DR MLP 的结构和语义、
MATD3 更新及 ESS Q 引导，新增 H24 ESS Transformer 计划路径。canonical MATD3 和共享环境未修改。

## 动作和计划

每个 Actor 只读取已有本地观测与公开 peer 预测。Transformer 解码剩余当日的 24 槽 signed intent，
槽 0 直接作为当前储能动作，无独立 ESS 动作头。未来无效槽不参与预测 attention 或监督。
名义申请功率和 SOC 按电池效率、瞬时 SOC 与功率限制逐步推导，不是两个独立预测头。
实际放电可能被清算削减，下一小时必须从环境返回的实际 SOC 重新规划。

推理 `forward` 只计算决定动作的网络路径；完整名义 SOC 展开在监督/诊断需要时执行。
网络存在显式计划不等于已学到优良经济策略，仍需同预算多种子比较。

## 训练专家

专家在 train split 的稀疏状态上捕获当前预测，隔离构造剩余日内场景。未揭示 EV 会话使用公开
条件先验，未来碳因子沿用当前值，不读取实际未来行。CTDE 专家使用当前联合状态；Actor
推理输入不增加私有 peer 信息。冻结当前策略在预测场景生成基线 DR 曲线。

紧凑 highspy 模型只优化 ESS：储能容量、SOC、效率、充放电互斥、源流守恒，禁止出口、
网电转售和 ESS 对 ESS。该模型的自由流分配近似实际比例清算，因此其解仅是候选。
候选在同一预测场景按原物理规则执行；残差超过 1e-6 或实际广义成本比基线高出 1e-5 则拒收。
接受标签包括申请 intent、同一名义模型推导的轨迹，以及分开保存的实际执行轨迹。
不能将全信息正式测试 MILP 轨迹用于标签，也不能把候选称为实际清算下的全局最优。

标签池反复采样，整段有效 intent/归一化功率/SOC 用 SmoothL1 监督，只回传 ESS 分支。
默认每 10 次 Actor 更新监督一次；第 50 轮起每 10 轮轮换 0/6/12/18 时捕获一个状态。
专家生成目标占非专家累计时间最多 15%，为辅助训练留余量；物化、基线、求解、核验和写标签
全部计时。实际单次调用可能超过剩余估计，之后预算会暂停生成，故总 20% 目标需要配对计时验证。

第一版只提供推理检查点，不支持训练恢复。检查点绑定网络、schema、奖励/物理合同和规划配置，
预检通过后才写入权重。专家及标签池不在推理路径中。

## 使用与证据

标准运行：

```powershell
python scripts/run_experiment.py --algorithm matd3_ess_plan_transformer
```

短流水线：

```powershell
python scripts/run_experiment.py --algorithm matd3_ess_plan_transformer --algorithm-config configs/smoke/matd3_ess_plan_transformer.yaml --validation-days 2 --device cpu
```

同结构 on/off 配对计时，输出必须是新目录：

```powershell
python scripts/benchmark_ess_plan_training.py --episodes 12 --pairs 3 --device cpu --output reports/ess_plan_runtime_new
```

计时包含标签生成、写入和实际辅助反传；监督关闭时没有专家调用。另报告旧 MLP 的网络时间。
6 天无专家预热单独报告，不隐藏任何专家生成。小样本结果不保证 5000 轮运行或不同硬件开销。
监督 on/off 可用同 slug 的临时配置将 `algorithm.ess_plan.enabled=false`，其余配置保持一致。

正常结果位于 `outputs/working_runs/matd3_ess_plan_transformer/runs/<run_id>/`。
`artifacts/auxiliary_plan/` 保存合同、逐标签证据、训练监督与正式测试专家调用摘要。
标准 best checkpoint、联盟 MILP、Equal/Shapley、摘要和季节图沿用唯一共享流水线。
短 smoke 只验证流程，不证明收益改善，不补造不存在的季节。
