# 全知 MILP：24小时完整设备联合规划

## 内容与使用边界

` scripts/clairvoyant_milp.py ` 是一个**独立的新 MILP 模型**，使用 `scipy.optimize.milp` 调用 HiGHS，保留现有算法、旧 MILP、训练命令及训练输出不变。它从仓库当前的 `configs/cooperative.yaml` 和 `data/spatial_market_2018_{train,test}.csv` 构造实际24小时的价格、负荷、风光和气象输入，不使用带噪声的 RL 预测。原模型的 `core.data_manager`、`core.profiles`、三个物理柔性设备类和 `optimization.common_constraints.build_dispatch_data` 仅用于**读取与转换同一环境数据**；新的决策变量、稀疏线性约束和目标函数全部在脚本中重新建立。

每一个自然日整体一次性求解所有3个微网的 ESS、MG1 工业磨机/料仓、MG2 HVAC 双状态热惯性/舒适温度、MG3 EV 电量与时段、外部电网和 P2P 联合调度。ESS 与电网双向运行、HVAC 冷热模式分别互斥；电池 SOC、设备状态转移、各微网能量守恒、同一微网对双向合计75 kW容量（实际数值读取配置）、末端偏离惩罚进入 MILP。目标函数是联盟实际购售电净成本 + ESS 衰减 + 弃电 + 缺供 + 末端任务惩罚。P2P 内部交易不应计作联盟的外部收入。

**重要：这是乐观的全信息集中规划参考，而非已经认证的当前环境严格可实现最优值。** 自由选取的双边 P2P 流与环境实际“本地优先→P2P交易池→电网结算”的具体出清顺序不同；虽然保留共用链路容量和本地新能源/储能出口来源约束，但允许进口外部电与对外 P2P 交易同时发生，可能产生环境无法复现的能量替代。因此 MILP 与 RL 的收益差距包含额外市场出清自由度及完全未来信息优势，不能全部归因于 Actor 不足。HVAC 舒适温度在本 MILP 中按严格可行约束处理；如果环境允许不可避免的一步越界，当前 MILP 不保证数学上严格包含环境的全部可行轨迹。任何求解失败都将作为失败记录，而不会混入55天平均值。

## 安装和放置文件

将压缩包中的 `scripts/clairvoyant_milp.py` 和 `scripts/test_clairvoyant_milp.py` 覆盖复制到**当前 GitHub 代码对应的项目根目录下的 `scripts/` 文件夹**。无需覆盖或修改任何旧文件。

在项目根目录的 PowerShell 中运行：

```powershell
python -m pip install "scipy>=1.11"
python scripts/test_clairvoyant_milp.py
```

测试只使用自构造的24小时三微网数据，不等于55天实际数据结果。

## 单日求解

```powershell
python scripts/clairvoyant_milp.py --day 2018-04-27
```

## 55天全知求解，与同物理配置的 RL 训练结果逐日配对

先确保本地 `outputs` 已同步到包含以下两个运行目录（名称与你此前配对成功的两次训练一致）；如果实际保存位置不同，只需改 CSV 路径。

```powershell
python scripts/clairvoyant_milp.py --all-test-days `
  --matd3-csv "outputs\working_runs\matd3_dispatch\runs\MATD3_2026-09-22_17-35-13_seed42\test_daily_metrics.csv" `
  --transformer-csv "outputs\working_runs\matd3_hierarchical_transformer\runs\MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42\test_daily_metrics.csv"
```

没有本地 RL CSV 时可以先只执行 `--all-test-days`，MILP 仍会独立运行。

正常生成目录（自动时间戳，不覆盖历史运行）：

```
outputs/working_runs/clairvoyant_milp/runs/MILP_YYYY-MM-DD_HH-MM-SS/
├── audit.json
├── daily_comparison.csv
└── hourly/
    ├── 2018-01-28.csv
    └── ...
```

`audit.json` 含完整性、MILP 均值、逐日求解状态、与两个 RL 的平均净收益差距和缺供量；`daily_comparison.csv` 含逐日 MILP 净收益、两种 RL 净收益、差距、各微网外部成本及设备末端代价；`hourly/*.csv` 含每个微网每小时储能、柔性负荷、SOC、购售电、P2P 交易、弃电、缺供及相应设备状态。

有任一测试日未证明最优或有数据/模型错误时，`audit.json` 标为 `incomplete`，不能公布不完整日期集合的55天平均值。可指定 `--time-limit 600` 增加单日求解时限。对于研究不允许缺供的规划，可以加 `--no-unmet`，但必须预期某些日期可能无法满足严格约束。MILP 默认按环境的 20/kWh 缺供成本计价，并在结果中明确报告缺供电量。

**声明：** 此代码已在独立合成24小时样本通过最优性、逐小时功率平衡和成本对账测试；此交付不包含在真实55天空间市场数据上的已求解数值，真实数据需要在包含完整当前仓库数据的运行环境中执行。
