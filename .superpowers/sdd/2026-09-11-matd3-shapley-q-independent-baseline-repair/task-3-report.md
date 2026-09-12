# Task 3 实施报告：Cooperation Surplus Replay 与 Twin Marginal Critic

## 范围与状态

Task 3 已实现并验证。当前仓库实际起点是 `Matd3ShapleyQ(MATD3): pass`，且没有
Shapley replay 或 marginal critic；因此本任务在指定的 algorithm.py 中补齐算法局部基础设施，
新建指定的 replay_buffer.py，并扩展 tests/test_matd3_shapley_q.py。

父代理明确要求本任务同时实现供 Task 6 消费的 `alg.coalition_projector(state, raw, coalition)`；
`CoalitionAwareProjector` 和 `TwinMarginalCritic` 均放在 algorithm.py 中，没有新增其他生产文件。

未修改 canonical `algorithms/matd3/`，未修改环境统一 team_reward，也未提前实施 Task 4+
训练生命周期调用、dashboard、export、formal evaluation 或结果布局。

## 实现

- 独立 ShapleyReplayBuffer / ShapleyTransition 存储 Grand、Independent、Surplus reward，
  以及同一 transition 的当前/下一步 independent discounted return-to-go。
  所有五个 reward/reference 字段检查 finite；Surplus 必须与 Grand - Independent 一致。
  observations、actions 和 states 入库时复制；所有标量采样为 `[batch, 1]` float32 tensors。
- 继承 Matd3Dispatch 的单维 decentralized actors、同一物理 dispatch projector 和公共接口，
  替换成本算法的 3 组 centralized TwinMarginalCritic 与独立 replay。
- 每个 marginal critic 看见全局 state、predecessor membership mask，以及加入 active agent
  前后的全部物理 projected controls。对每个 active 的四种 predecessor 使用
  `1/3, 1/6, 1/6, 1/3` 精确聚合，得到两组 Q_phi。
- CoalitionAwareProjector 为所有 coalition 保留每个微网 local dispatch channel；
  cooperation offer 仅允许 coalition 内的 seller/buyer，并依据 coalition 内 eligible demand
  重新计算 cap。empty/singleton coalition 没有跨网 offer，plans 暴露 in_coalition。
- Target 为 `r_coop + gamma*(1-done)*sum_i min(Q_phi_i_A', Q_phi_i_B')`；
  当前两组 global sums 分别拟合这个相同 target，所有 critic optimizer 在共同 backward 后 step。
- Grand residual 为当前 `independent_return_to_go + sum_i min(Q_phi_i_A,Q_phi_i_B)`
  减去真实 Grand Bellman target；输出顶层 metrics 和 algorithm_diagnostics。
  Q_phi、sum、surplus target、efficiency residual 一起输出，并检查 loss/gradient/diagnostic finite。
- Actor i 最大化自己的第一组 Q_phi；其他 actor actions detach，active critic 暂时冻结。
  延迟 Actor update 和 target soft update 保留，Actor gradients 做 finite-aware clipping。
- `on_episode_start(env, **kwargs)` 调用 Task 2 evaluator reset；
  `get_independent_step_data(t)` 接入 Task 2 严格 bounds 与 return-to-go API。
  heuristic 和 frozen-policy 两种 baseline 均在真实环境轨迹训练测试中覆盖。
- checkpoint 记录 `surplus_q_v2`、`cooperation_surplus`、baseline mode/configuration，
  严格校验算法/action semantics、projection signature、state/observation schemas。
  拒绝 `full_enumeration_v1` 等不兼容 metadata；严格验证四组网络后再加载到 live modules。

## TDD 与验证证据

### RED

1. `pytest -q tests/test_matd3_shapley_q.py::test_replay_surplus_consistency`
   → 1 failed：ReplayBuffer.add 拒绝 independent_reward keyword。
2. `pytest -q tests/test_matd3_shapley_q.py::test_conservative_target_takes_twin_min_per_agent_before_sum`
   → 1 failed：缺少 _conservative_shapley_sum helper。
3. Replay finite/consistency、coalition projector、真实 update、Actor gradients 和 checkpoint
   新增测试在实现前运行 → 30 failed, 34 deselected，分别暴露新 replay 参数、projector/hook
   与 checkpoint metadata 缺失。
4. 补强 exact-weight 测试后，暂时把生产 predecessor weights 改为均匀 1/4：
   测试得到 0.75 而非 1.0，确认能发现错误；恢复标准 Shapley weights。
5. 补强 coalition 测试后，暂时强制所有请求使用 grand coalition：
   测试发现 empty coalition 的 member 标志错误；恢复 coalition 参数。
   测试还直接覆盖实际正报价场景，避免 charging-only 输入下 offer 天然为零的空验证。

### GREEN

- 首轮实现聚焦：32 passed, 32 deselected。
- 精确 predecessor / coalition projector 补强：2 passed, 63 deselected。
- 要求命令 `pytest -q tests/test_matd3_shapley_q.py -k "replay or conservative or gradient or checkpoint"`
  → 37 passed, 28 deselected in 16.35s。
- 完整算法文件 `pytest -q tests/test_matd3_shapley_q.py`
  → 65 passed in 22.58s。
- 本任务执行一次完整 `pytest -q`
  → 149 passed, 2 warnings in 40.93s。
  两条 warning 是既有非有限 JSON 值处理与非正 MILP cooperation gain 测试的预期警告。
- `git diff --check` → exit 0；仅 Git LF→CRLF 提示，没有 whitespace error。
- `git diff --name-only -- algorithms/matd3/` → 无输出。

### 数值验证重点

- 交叉双头常数 `(10,1,10)` 与 `(1,10,1)`：per-agent minimum 后 sum 必须为 3。
- 单步 gamma=0.5、Grand reward=5、Independent reward=2、Surplus=3：
  nonterminal target=4.5，terminal target=3；分别检查两组全局 MSE loss。
- current Q_phi sum=6、independent reference=12、next reference=7 时，
  nonterminal Grand residual=8、terminal residual=13；证明它没有被伪造为恒零。
- 两种 baseline 的真实 8 步环境轨迹：三个 Actor 的 gradient norm 都有限且 >0，
  三个 Actor 参数实际改变；frozen baseline 所有参数 grad=None 且 requires_grad=False。

## 自审与后续边界

- 检查了 raw action replay、physical projection、min/sum 顺序、共同 critic backward、
  Actor 图隔离、baseline 与真实 reference、checkpoint mutation 前校验以及 canonical 无差异。
- 此模型的 exact enumeration 指对 learned marginal heads 的精确 predecessor 加权；
  聚合 Bellman 损失本身不构成“各 head 已等于真实物理 Shapley value”的证明。
  算法学习效果和真实 coalition-value 对照仍需后续正式实验评估。
- 当前共享 pipeline 尚未接入新 replay 的必需 baseline 参数；按 Task 3 指定边界，
  本任务没有运行端到端 Shapley-Q smoke。Task 4 生命周期接入后再执行完整 smoke 验收。
- 本次独立 replay 使用严格必填 reference 字段，不保留旧 Grand-return 目标兼容路径。

## 提交

本报告随 Task 3 实现一起提交，提交信息为 `fix: train Shapley critics on cooperation surplus`。
