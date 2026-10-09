# 单个实验训练速度：AutoDL 实测与配置

日期：2026-10-07。服务器硬件、连接和同步约定见 SERVER_WORKFLOW.md。
当前目标是先减少一个实验的耗时；没有启动多个并行训练，也没有启动 5000 回合长训练。

## 测速方法与适用范围

首先逐一测试三个算法的 CPU/CUDA 与 1、2、4 算子线程，共 18 组。
随后使用相同初始化、随机种子和训练日，对主要候选做 3 回合预热、12 回合计时，
并补测 MATD3-Q 的 8 线程和 Dueling 的 16 线程。每个 12 回合计时组都有 288 次实际更新，
保持 batch_size=128、learning_starts=1000、网络规模、奖励、物理规则和更新频率。
PyTorch interop_threads 统一为 1。各组在同一台服务器上依次运行，避免多个测速进程争抢资源。

计时包括动作、环境、经验写入、反事实上下文及更新；不含预填充、预热、验证、学习曲线评估、
模型保存、正式测试、MILP、结算和绘图。反事实预填充阶段只填真实转移，预热与计时阶段启用
实际反事实上下文和教师回放，记录辅助更新事件。它不是整个 5000 回合实验的端到端耗时。
CUDA 在组件间同步，以便定位瓶颈；其耗时包含额外同步开销。

服务器使用 PyTorch 2.8.0+cu128，本机使用 2.11.0+cu128。跨机器的差异不能完全归因于硬件。
短测存在共享服务器和浮点计算波动，百分比是实测值，不是长期保证。

## 当前原生执行结果

| 算法 | 设备 / 算子线程 / interop | 秒/24小时回合 | 同配置优化前 | 耗时变化 |
|---|---|---:|---:|---:|
| MATD3-Q | CPU / 4 / 1 | 0.987 | 0.988 | 额外代码收益不明显 |
| Dueling H24 Q-guided | CPU / 8 / 1 | 4.321 | 4.572 | 减少约 5.5% |
| DR Counterfactual | CPU / 1 / 1 | 1.858 | 2.143 | 减少约 13.3% |

MATD3-Q 未优化时 CPU 1 线程为 1.243 秒/回合，切换到优化后的 CPU 4 线程约减少 20.6% 耗时。
Dueling 的优化前 CUDA 4 线程为 5.213 秒/回合，切换到 CPU 8 线程加代码优化约减少 17.1%。
增加到 8/16 线程没有进一步改善上述 Q/Dueling 的最快组合。
当前三个原生执行候选都以 CPU 为优先，不能从 GPU 显存或标称算力直接推断训练速度。
default 模式编译 Dueling 的 CUDA Critic 后为 4.659 秒/回合，
虽比 CUDA 原生 5.093 秒快约 8.5%，仍慢于 CPU 8 线程，因此最终不启用编译。

## 本轮代码优化

1. HVAC 从整行 pandas 读取改为对应列的标量读取，保留每日替换和帧内数据修改的行为。
2. 可微物理清算只跳过已由构造保证的中间量重复校验，公共输入验证保留。
   继续保留原来的 clamp 梯度节点；CPU/CUDA、float32/float64 的完整调度账目及梯度逐项一致。
3. Dueling 的同一状态、当前动作和五个候选动作共用一次 Transformer 状态编码。
   只在本次候选评分内复用，不跨更新缓存。候选 Q、指导损失、动作批准与 Actor 梯度通过浮点容差检查。
4. 反事实比较去掉一份额外整环境快照；每条参考/候选轨迹仍各自独立深拷贝后执行。
   教师收益、服务约束和在线环境隔离继续受测试保护。
5. 修复仓库测速工具：预填充达到学习启动阈值，调用反事实上下文协议，记录并核验实际更新次数。
   不能把诊断返回非空次数当作网络更新次数。

永久基线 algorithms/matd3/ 的 6 个 Python 文件与本轮优化前 SHA-256 一致。
没有改环境 YAML、奖励、学习率、网络宽度、batch size、训练预算或辅助损失权重。

## 服务器启动配置

服务器专用配置放在 configs/servers/autodl_8481c/，来自对应算法自己的 YAML，
仅覆盖 runtime.device、runtime.cpu_threads、runtime.interop_threads。本机普通默认 YAML 保留。
改变算法超参数后，应重新从算法自己的 YAML 生成服务器配置并核对差异。

单实验命令示例（在服务器项目目录内，任选一个执行）：

```bash
/root/miniconda3/bin/python -u scripts/run_experiment.py --algorithm matd3_dispatch_q_guided --algorithm-config configs/servers/autodl_8481c/matd3_dispatch_q_guided.yaml --no-dashboard
/root/miniconda3/bin/python -u scripts/run_experiment.py --algorithm matd3_mlp_dueling_critic_h24_q_guided --algorithm-config configs/servers/autodl_8481c/matd3_mlp_dueling_critic_h24_q_guided.yaml --no-dashboard
/root/miniconda3/bin/python -u scripts/run_experiment.py --algorithm matd3_dr_counterfactual --algorithm-config configs/servers/autodl_8481c/matd3_dr_counterfactual.yaml --no-dashboard
```

长训练使用 screen 保存日志，结果继续写入 outputs/working_runs/<slug>/runs/，
每次运行的配置、模型和数据在 artifacts/，图片在 figures/。上述命令没有在本轮启动长训练。

## 验证与证据

- 本机完整测试：513 passed，4 warnings，96.77 秒。
- 修复后的仓库测速工具也已在服务器验收：真实反事实上下文预填充 1000 步，
  随后 2 回合产生 48 次实际更新，1.936 秒/回合。此组预填充标签设置与外部比较探针不同，不混算提速百分比。
- 服务器完整测试：513 passed，4 warnings，119.37 秒。
- 三个算法各完成 3 个训练日、1 个验证日/次、1 个正式测试日，使用各自服务器 YAML，
  正式测试 best_model.pt 的路径和 SHA-256 已核对；7 联盟 MILP 无失败，Equal/Shapley 已保存。
  每次生成 00、01、01b 和冬季总览，共 4 PNG + 4 PDF；没有伪造缺失季节或学习曲线。
- 短流程保留 learning_starts=1000，因此 72 步的流程测试不代表学得策略；
  网络实际更新由独立计时组的 288 次更新和单元测试覆盖。短流程图片不用于算法性能汇报。
- reduce-overhead 编译候选出现 CUDA Graph 输出覆盖错误，未启用；default 模式成功完成 288 次更新，
  但 4.659 秒/回合仍不是最优。两个模式都只在外部临时测速进程中试验，未改训练默认行为。
- 原始计时证据：服务器 /root/autodl-tmp/single_speed_baseline_20261007/、
  single_speed_baseline_long_20261007/、single_speed_optimized_20261007/、single_speed_final_checks_20261007/。
- 服务器原代码备份：/root/autodl-tmp/single_speed_before_optimization_20261007.zip。
- 本机备份和原始证据归档：C:\daima\multi_microgrid_rl_standalone_backups\server_transfer_20261007\。
- 最终 239 个源码、数据、文档和配置文件的本机/服务器 SHA-256 一致；本轮后台会话均已结束。

线程参数控制 PyTorch 算子的并行度，当前环境的 24 个小时仍依次推进。
参考：[PyTorch 2.8 算子线程 API](https://docs.pytorch.org/docs/2.8/generated/torch.set_num_threads.html)。
