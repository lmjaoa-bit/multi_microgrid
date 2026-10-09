# 储能日内规划 Transformer (`matd3_ess_plan_transformer`) 实施与收尾验收总结

## 1. 遗留问题定位与根因分析

在全套测试中，`test_storage_model_arbitrage_and_no_useless_discharge`、`test_label_pool_reuses_labels_without_solves` 与 `test_captured_label_trains_plan_without_extra_critic_updates` 独立运行通过、但批量运行时报 `HighsModelStatus.kNotset` 的根因已彻底查明：

- **根因：** `plan_teacher.py` 中的 `solve_storage_model()` 显式调用了 `model.setOptionValue('threads', 1)`。
- 在包含联盟 MILP 和多进程/多用例的全局测试套件中，先前测试或环境会触发 HiGHS 初始化进程级全局线程调度器（默认使用多线程）。
- HiGHS C++ 底层对后继实例设置 `threads=1` 时会触发冲突错误：
  `Option 'threads' is set to 1 but global scheduler has already been initialized to use N threads.`
  导致模型直接处于未就绪状态并报错 `kNotset`。
- **修复措施：** 移除 `solve_storage_model()` 中硬编码的 `model.setOptionValue('threads', 1)`，复用现有 HiGHS 调度器，彻底解决进程级状态冲突。

## 2. 验证结果

1. **测试套件全量通过：**
   - 运行命令：`pytest -q`
   - 结果：`539 passed, 4 warnings in 110.23s` (100% 通过，0 失败)。
   - 涵盖 Actor 动力学、因果预测快照、紧凑 HiGHS 模型、标签复用与物理核验、检查点契约、断点隔离等全部 26 项储能规划新测试。

2. **短流水线冒烟验证：**
   - 运行命令：
     ```powershell
     python scripts/run_experiment.py --algorithm matd3_ess_plan_transformer --algorithm-config configs/smoke/matd3_ess_plan_transformer.yaml --validation-days 2 --device cpu
     ```
   - 训练 3 天、验证选出 `best_model.pt`（Episode 1）、正式测试 2 天（冬季）。
   - 训练期间捕获并生成 2 个有效规划标签，执行 6 次规划辅助更新。
   - 正式测试零专家调用（`teacher_calls: 0`），成功保存 48 份逐小时策略重规划明细（`policy_plan_records: 48`）。
   - 联盟 MILP（7 种日内组合）、Equal 与 Shapley 合作剩余结算完成。
   - 绘图包含冬季调度概览等共 4 组 PNG/PDF（00 摘要、01 训练曲线、01b 验证曲线、02 冬季调度）。
   - 产物完整归档至 `outputs/working_runs/matd3_ess_plan_transformer/runs/储能规划T_2026-10-08_12-08-23_seed42/`。

3. **耗时开销基准（已测定）：**
   - 在相同 Transformer 结构下，规划专家与监督训练的额外增量耗时中位数为 **0.21%**（三组配对分别为 2.91%、0.14%、0.21%），远低于 20% 预算上限。
   - 新 Transformer 相较于旧 MLP 基础耗时增加约 23%，两者界限明确清晰。

4. **基线与仓库规范遵守：**
   - `algorithms/matd3/` 原版基线完全未动，Working tree 保持清洁。
   - 共享物理、环境配置与奖励函数未做任何侵入式修改。
