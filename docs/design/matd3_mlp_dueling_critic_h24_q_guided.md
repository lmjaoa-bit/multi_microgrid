# 设计文档与实施规范：Dueling Transformer Critic（H24, Q-guided）

| 项 | 内容 |
|---|---|
| 算法 slug | `matd3_mlp_dueling_critic_h24_q_guided` |
| 显示名 | `Dueling Critic T-Q H24` |
| 对照基线 | `matd3_mlp_transformer_critic_h24_q_guided`（下称 **H24**）、`matd3_dispatch_q_guided`（下称 **MATD3Q**） |
| 改动性质 | **单变量**：只换 Critic 结构；Actor、环境、奖励、动作投影、超参数全部与 H24 一致 |
| 实施者必读 | 仓库根目录的 `AGENTS.md`、`docs/ALGORITHM_DEVELOPMENT.md` |

---

## 1. 背景与动机

H24 与 MATD3Q 的 seed 42 对比（5000 episodes，55 个测试日）：

| 指标 | MATD3Q | H24 |
|---|---|---|
| 测试利润 | -855.4 | -833.7 |
| critic_loss（第 4500–5000 个 episode） | 0.30 | **0.07** |
| td_error_std（第 4500–5000 个 episode） | 0.37 | 0.17 |
| 储能动作梯度 RMS，即 ∂Q/∂a_dispatch（mg1/2/3） | .13/.13/.09 | **.06/.06/.04** |
| 储能动作饱和率（mg1/2/3） | .27/.29/.36 | **.49/.59/.46** |

诊断结论：H24 的 Transformer Critic 把容量用在了拟合**外生的状态价值**上。每天成本约 1050，而动作能影响的部分只有约 20–30。结果是 Q 值拟合得更准，但对动作更平滑，Actor 拿到的梯度变弱，动作逐渐贴到 tanh 边界。

结构原因：9 维联合动作和 103 维当前状态拼接后，经过 `Linear→LayerNorm→GELU` 压缩成 1 个 token，再和 24 个预测 token 一起做 attention 和 pooling，动作信号被稀释。

**本设计的目标**：把 Q 拆成"外生价值 V(s)"和"动作优势 A(s,a)"两部分。Transformer 只负责 V，A 由一个动作直接输入、不被归一化稀释的独立分支负责，从而恢复 ∂Q/∂a 的强度和质量。

## 2. 范围

### 2.1 本次要做
- 新建算法包 `algorithms/matd3_mlp_dueling_critic_h24_q_guided/` 和配置 `configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml`。
- 新的 Twin Critic：`DuelingTemporalCritic`。
- 只记录诊断量，不改变任何训练目标：V、A 的统计量，以及 ∂Q/∂a。
- 单元测试和冒烟测试，然后跑 3 个 seed 的正式实验。

### 2.2 明确不做（禁止）
- 不修改 `algorithms/matd3/`、`algorithms/matd3_hierarchical_transformer/`、`algorithms/matd3_mlp_transformer_critic_h24_q_guided/` 中的任何文件。
- 不修改 `experiments/`、`evaluation/`、`core/`、`envs/`、`components/`、`configs/cooperative.yaml`。
- 不修改共享的 update 逻辑（`algorithms/diagnostic_sampling.py`）。
- 不改奖励、动作语义、投影器，不加 reward shaping。
- 不新增 Critic 辅助损失。优势中心化完全通过网络结构实现，见 §3.4。
- 不调整 Actor 结构或任何超参数，保证单变量对比。

## 3. 网络设计

### 3.1 输入约定（沿用 H24，不改变）

| 张量 | 形状 | 说明 |
|---|---|---|
| `state` | `[B, 295]` | 全局状态，按 `self.state_schema` 排列，已归一化 |
| 当前特征 | `[B, 103]` | schema 中不匹配 `_t\+(\d+)$` 的列，用 `current_indices` 取出 |
| 预测 token | `[B, 24, 8]` | `t+1..t+24`，8 个通道的顺序必须完全一致，且包含 `forecast_valid` |
| `joint_action` | `[B, 9]` | **投影后的有效动作**，3 个 agent × `[ESS-local, ESS-trade, DR-effective]`，按 agent 依次展平；agent i 占第 `3i..3i+2` 维 |

schema 的解析与校验规则和 H24 `critic.py` 中的 `TemporalValue.__init__` 完全一致：必须覆盖完整的 H24、通道顺序一致、包含 `forecast_valid`、没有重复列。建议在新包里**复制**这段逻辑，不要 import H24 包（见 §4.1）。

### 3.2 单个 Q 网络：`DuelingTemporalValue`

```text
                    ┌────────── V 分支（Transformer，不含动作）──────────┐
state ─┬─ s_cur[103] ─ Linear(103→E) → LayerNorm → GELU ──┐
       │                                                   ├─ TransformerEncoder(L 层, H 头, FF, norm_first, dropout=0)
       └─ fc[24×8] ─(有效位 mask, NaN→0)→ Linear(8→E) + pos[24×E] ┘        │
                                                                           ▼
                         ctx = [out_token0 ‖ masked_mean(out_1..24)]   (2E 维)
                         V(s) = MLP(2E → 64 → 1, GELU)

                    ┌────────── A 分支（MLP，动作直接输入）──────────┐
x = [ LN_s(s_cur)[103] ‖ ctx_A[2E] ‖ a[9] ]
h1 = ReLU(Linear(103+2E+9 → Hₐ))
h2 = ReLU(Linear(Hₐ + 9 → Hₐ))        ← 动作再注入（action_reinjection）
A(s,a) = Linear(Hₐ → 1)

Q(s,a) = V(s) + A(s,a) − A(s,a₀)       ← a₀ = zeros(9)，见 §3.4
```

默认维度：E=48，H=4，L=2，FF=128，Hₐ=128。

| 设计点 | 规范 | 原因 |
|---|---|---|
| V 分支的当前 token | **不包含动作**，输入只有 `s_cur` | V 只拟合外生价值；动作信息不进入 Transformer |
| V 分支的 mask 和 pooling | 与 H24 完全相同：padding mask 中当前 token 始终可见；pooling 只对有效 token 取平均，分母 `clamp_min(1)` | 保持 V 的能力与 H24 等价，便于归因 |
| `LN_s` | 一个独立的 `nn.LayerNorm(103)`，**只作用于状态**，不作用于动作 | 动作不能被归一化稀释 |
| `ctx_A` | 默认是 `ctx.detach()`；由配置项 `advantage_network.detach_value_context` 控制 | A 可以读取时间上下文，但它的梯度不进入 Transformer 编码器，防止 A 去拟合外生价值 |
| `use_value_context=false` | 此时 `ctx_A` 不拼接，x 的维度变为 103+9 | 消融用 |
| 动作再注入 | `action_reinjection=true` 时第二层输入是 `[h1 ‖ a]`；为 false 时第二层是 `Linear(Hₐ→Hₐ)` | 保持 ∂A/∂a 的通路，防止动作信号在深层衰减 |
| A 输出层初始化 | weight 服从 `U(-3e-3, 3e-3)`，bias 为 0 | 训练初期 Q≈V，A 从小值开始学，避免早期噪声主导 |
| 激活函数 | A 分支用 ReLU（与 MATD3Q Critic 一致）；V 分支用 GELU（与 H24 一致） | 两个分支分别继承各自已验证过的设置 |

### 3.3 Twin Critic：`DuelingTwinCritic`

- 两个**完全独立**的 `DuelingTemporalValue`（`q1_network`、`q2_network`），V 和 A 都不共享参数。
- 对外接口**必须**与 H24 的 `TransformerTwinCritic` 完全一致，共享 update 只依赖下面两个方法：

```python
forward(state, joint_action) -> tuple[Tensor[B,1], Tensor[B,1]]   # (q1, q2)
q1(state, joint_action) -> Tensor[B,1]
```

- 额外暴露一个只用于诊断的方法，**不被共享 update 调用**：

```python
decompose(state, joint_action) -> dict[str, Tensor[B,1]]
# 键: "v1", "adv1", "q1", "v2", "adv2", "q2"
# 其中 adv = A(s,a) − A(s,a₀)
```

- 输入校验与 H24 一致：`state.ndim==2`，`state.shape[-1]==state_dim`，`joint_action.shape==(B, 9)`；不满足时抛 `ValueError`。

### 3.4 优势中心化：为什么选 `A(s,a) − A(s,a₀)`

- 如果没有中心化，V 和 A 无法区分：任意 `V+c, A−c` 给出相同的 Q，A 可能吞掉外生价值，又回到 H24 的问题。
- `a₀ = 0` 是**有效动作空间里的零点**：储能不动、DR 有效值为 0。它只是一个函数求值的参照点，**不会被执行**，也不需要可行。
- 性质：
  - `Q(s,a₀) ≡ V(s)` 严格成立，单元测试要检查这一点。
  - `∂Q/∂a = ∂A/∂a`：策略梯度完全由 A 分支提供。
  - 不需要投影器，也不改接口，计算量只多一次 A 分支前向（是个小 MLP）。
- 实现方式：把 `[a; a₀]` 拼成 `2B` 的 batch，一次前向得到 A 后再切成两半，避免两次前向。

### 3.5 参数量与速度预算
- 每个 Q 网络的 V 分支参数量约等于 H24 的 `TemporalValue`，A 分支约 `(103+96+9)·128 + (128+9)·128 + 128 ≈ 44k`。实施者需要在 PR 中报告两者的实际参数量。
- 秒/episode **不超过 H24 的 1.3 倍**（H24 ≈ 2.67s，参考值上限约 3.5s，同一台机器）。

## 4. 代码实施规范

### 4.1 目录与依赖

```text
algorithms/matd3_mlp_dueling_critic_h24_q_guided/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py        # 从 H24 包逐字复制 SemanticH24Features / SemanticMlpActor，文件头注明来源 commit
└── critic.py       # 新的 DuelingTemporalValue / DuelingTwinCritic
configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml
tests/test_matd3_mlp_dueling_critic_h24_q_guided.py
```

- **允许的依赖**：`algorithms.matd3_hierarchical_transformer.*`（正式插件，H24 本身也依赖它）、`algorithms.registry`、`core.*`、`torch`、`numpy`。
- **禁止**依赖 `algorithms.matd3_mlp_transformer_critic_h24_q_guided`。实验包之间不能互相依赖，否则 H24 被删除时本算法会一起失效（AGENTS §2、§7）。因此 `actor.py` 采用复制方式，并在测试中断言它与 H24 Actor 的参数量和输出相同（见 §6）。
- 先用脚手架生成，再做替换：

```powershell
python scripts/create_algorithm_variant.py --name matd3_mlp_dueling_critic_h24_q_guided
```

### 4.2 `algorithm.py`

以 H24 的 `algorithm.py` 为模板**复制**，类名为 `Matd3MlpDuelingCriticH24QGuided(Matd3HierarchicalTransformer)`，`SLUG = "matd3_mlp_dueling_critic_h24_q_guided"`。只允许下面这些差异，其他部分与 H24 逐行一致，PR 中需要附上与 H24 的 diff：

| # | 位置 | 改动 |
|---|---|---|
| D1 | import | 从本包的 `actor`、`critic` 导入 |
| D2 | Critic 配置解析 | 把 `critic_network` 换成下面的两级结构；出现未知键时抛 `ValueError`；所有维度必须是 int（bool 不算），`embed_dim % num_heads == 0`；布尔开关必须是 bool |
| D3 | Critic 构造 | 用 `DuelingTwinCritic(self.state_schema, len(self.agent_ids)*3, value=..., advantage=...)` 替换 `TransformerTwinCritic` |
| D4 | reward_v2 contract | `critic_feature_semantics="full_state_h24_dueling_v_transformer_a_mlp_zero_ref_v1"`；`critic_network_config` 写入完整的两级配置 |
| D5 | `_checkpoint_metadata` | `critic_network_config` 改为两级配置，其余键不变；`load` 的校验逻辑原样保留 |
| D6 | `actor_auxiliary_loss` | ESS Q-guidance 部分**逐字保留**；在返回之前追加 §4.4 的诊断量（`no_grad`，只在诊断步计算，**不加入 loss**） |

两级配置的默认值：

```python
VALUE_DEFAULTS = dict(forecast_horizon=24, embed_dim=48, num_heads=4, num_layers=2, feedforward_dim=128)
ADV_DEFAULTS   = dict(hidden_dim=128, use_value_context=True, detach_value_context=True,
                      action_reinjection=True, reference_action="zero")
# reference_action 当前只允许 "zero"，其他值抛 ValueError
```

### 4.3 `plugin.py`

```python
from algorithms.diagnostic_sampling import install_diagnostic_sampling  # 只有父类没装时才需要；见下方说明
from algorithms.matd3_mlp_dueling_critic_h24_q_guided.algorithm import Matd3MlpDuelingCriticH24QGuided
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="matd3_mlp_dueling_critic_h24_q_guided",
    display_name="Dueling Critic T-Q H24",
    algorithm_class=Matd3MlpDuelingCriticH24QGuided,
    default_config_path=PROJECT_ROOT / "configs" / "algorithms" / "matd3_mlp_dueling_critic_h24_q_guided.yaml",
    description="H24 semantic MLP actors with dueling critics: Transformer V(s) plus action-direct MLP A(s,a), zero-action centred; ESS Q guidance retained.",
    environment_overrides={"ess_sharing": {"observation_mode": "peer_energy_summary"}},
)
```

说明：H24 通过 `from algorithms.matd3_hierarchical_transformer.plugin import Matd3HierarchicalTransformer` 触发 `install_diagnostic_sampling`，共享的 update 和 select_actions 由此安装在父类上并被子类继承。**本算法采用同样的 import 方式**（在 `algorithm.py` 中从 `...matd3_hierarchical_transformer.plugin` 导入父类）。不要对子类再调用一次 `install_diagnostic_sampling`。如果确实需要，实施者应先确认不会重复包装 `__init__`。

### 4.4 诊断量（写入 `actor_diagnostics.csv`）

训练历史只记录白名单里的标量，而 `actor_auxiliary_loss` 返回的 metrics 会在诊断步被写进 `actor_diagnostics.csv`（现有的 `dispatch_q_guidance_*` 就是这样落盘的）。因此诊断量**放在 `actor_auxiliary_loss` 的 metrics 中返回**，不需要改流水线。

- 计算时机：`self.diagnostic_schedule.due(self.update_steps)` 为真时才计算；其他时候不计算，也不要返回这些键。共享 update 在 Actor 阶段之前已经执行了 `update_steps += 1`，所以这个判断和共享 update 内部的 `diagnostic_due` 一致。实施者需要用测试确认这一点。
- 输入：参数里的 `state`，以及用 `self.project_actions(state, joint_raw.detach())` 得到的有效动作。
- 所有统计量都在 `torch.no_grad()` 下计算，只有 dQ/da 需要局部 `enable_grad`，对 detach 后的动作副本求导。**不允许让 Critic 或 Actor 的参数产生梯度。**

| 键（均为标量 tensor） | 定义 |
|---|---|
| `critic_v_mean` | `v1.mean()` |
| `critic_adv_std` | `adv1.std(unbiased=False)` |
| `critic_adv_abs_mean` | `adv1.abs().mean()` |
| `critic_adv_share` | `adv1.std() / (q1.std() + 1e-8)` |
| `critic_dqda_dispatch_rms` | 对有效动作求导 `∂q1/∂a`，取 active agent 的 `[3i, 3i+1]` 两维（ESS-local、ESS-trade）的 RMS |
| `critic_dqda_dr_rms` | 同上，取第 `3i+2` 维 |
| `critic_v_adv_corr` | `corr(v1, adv1)`，用于检查 V 和 A 是否发生耦合 |

### 4.5 YAML（完整内容）

除了 `name` 和 `critic_network` 之外，**必须与 H24 的 YAML 逐项一致**，PR 中附上 diff。

```yaml
# H24 semantic MLP Actor + Dueling Critic (Transformer V, action-direct MLP A); ESS guidance retained.
algorithm:
  name: matd3_mlp_dueling_critic_h24_q_guided
  hidden_dims: [128, 128]
  actor_lr: 0.0002
  actor_max_grad_norm: 1.0
  critic_lr: 0.0003
  gamma: 1.0
  tau: 0.005
  policy_delay: 2
  target_policy_noise: 0.2
  target_noise_clip: 0.5
  batch_size: 128
  replay_capacity: 200000
  warmup_steps: 1000
  learning_starts: 1000
  exploration_noise: 0.15
  dispatch_random_action_prob: 0.1
  dispatch_q_guidance_weight: 0.01
  dispatch_q_guidance_margin: 0.01
  network:
    feature_semantics: physical_power_balance_v1
    embed_dim: 32
    num_heads: 4
    num_layers: 2
    hidden_dim: 64
    forecast_horizon: 24
    include_peer_forecast: true
    peer_forecast_features: [energy_balance]
  critic_network:
    value:
      forecast_horizon: 24
      embed_dim: 48
      num_heads: 4
      num_layers: 2
      feedforward_dim: 128
    advantage:
      hidden_dim: 128
      use_value_context: true
      detach_value_context: true
      action_reinjection: true
      reference_action: zero
# runtime / training / learning_curve / validation / logging: 从 H24 YAML 原样复制
```

## 5. 风险与应对

| 风险 | 现象（诊断量） | 应对（只能在本算法自己的 YAML 或代码中改） |
|---|---|---|
| A 塌缩为 0，退化成 V-only | `critic_adv_share` 低于 0.01 且持续下降，`dqda` 趋近 0 | 先检查初始化和动作再注入；再考虑 `detach_value_context=false` 的消融 |
| A 吞掉外生价值 | `critic_adv_share` 大于 0.8，`critic_v_adv_corr` 的绝对值很大 | 检查中心化实现，确认 `Q(s,0)==V(s)` 的测试通过 |
| ∂Q/∂a 过大，训练振荡 | `dqda` 是 MATD3Q 的 5 倍以上，`clip_applied_mean` 大于 0.3，Actor 损失抖动 | 记录下来即可，**不要**在主实验中调参；作为后续实验另开配置 |
| 与 Q-guidance 产生交互 | `dispatch_q_guidance_trigger_rate` 明显异常 | 主实验保持不变；后续可做一个 `dispatch_q_guidance_weight: 0` 的消融 |

## 6. 测试规范

新增 `tests/test_matd3_mlp_dueling_critic_h24_q_guided.py`，至少覆盖下面这些用例。用合成 schema 或真实 env 都可以，参考现有 H24 测试的写法。

| # | 用例 | 断言 |
|---|---|---|
| T1 | 插件按名称加载 | `get_algorithm_spec(slug)` 成功；YAML 中的 `algorithm.name == slug` |
| T2 | 接口形状 | `forward` 返回两个 `[B,1]`；`q1` 返回 `[B,1]`；形状不对时抛 `ValueError` |
| T3 | 中心化恒等式 | `torch.allclose(Q(s, zeros(9)), V(s), atol=1e-6)`，q1 和 q2 都要检查 |
| T4 | 初始化时动作敏感 | 随机 s、a，`∂q1/∂a` 有限且不全为 0 |
| T5 | V 与动作无关 | `decompose(s,a1)["v1"] == decompose(s,a2)["v1"]` |
| T6 | detach 生效 | `detach_value_context=true` 时，只对 `adv1.sum()` 反传，Transformer 编码器参数的梯度为 None 或全 0 |
| T7 | mask 鲁棒性 | 预测部分全为 NaN 或全无效，以及 t=23 时只剩 0 个有效 token，输出仍然有限 |
| T8 | 配置校验 | 未知键、bool 当 int 用、`embed_dim % num_heads != 0`、`reference_action != "zero"`，都抛 `ValueError` |
| T9 | checkpoint | save→load 往返后 Q 输出逐元素相等；改动 `critic_network_config` 后 load 抛 `ValueError` |
| T10 | Actor 与 H24 一致 | 在同一个 env 和 seed 下，本算法与 H24 每个 agent 的 Actor 参数量相同；加载同一份 Actor state_dict 后输出相同（只在测试中 import H24，不进入运行时依赖） |
| T11 | 诊断不影响训练 | 固定 seed，分别在 `diagnostic_interval` 很大和很小时跑相同的 N 步 update，Actor 和 Critic 的参数逐元素相等 |
| T12 | 诊断键落盘 | 冒烟运行后，`actor_diagnostics.csv` 包含 `critic_adv_share_mean`、`critic_dqda_dispatch_rms_mean` 等列 |

## 7. 实验规范

### 7.1 冒烟测试（必须先通过）

```powershell
pytest -q
python scripts/run_experiment.py --algorithm matd3_mlp_dueling_critic_h24_q_guided --episodes 3 --test-episodes 2 --seed 42 --no-dashboard
```

按 AGENTS §8 逐项确认：使用了本插件和本 YAML；正式测试用的是 `best_model.pt`；test、Equal/Shapley 结算、summary、绘图全部完成；生成了 15 张 PNG 和 15 张 PDF；结果写入 `outputs/working_runs/matd3_mlp_dueling_critic_h24_q_guided/runs/`；`git diff --stat algorithms/matd3/` 为空。

### 7.2 正式实验

| 算法 | seeds | 备注 |
|---|---|---|
| `matd3_mlp_dueling_critic_h24_q_guided` | 42, 43, 44 | 新算法 |
| `matd3_mlp_transformer_critic_h24_q_guided` | 43, 44 | seed 42 已有：`CRITICT-QH24_2026-10-03_08-43-59_seed42` |
| `matd3_dispatch_q_guided` | 43, 44 | seed 42 已有：`MATD3Q-GUIDEDDISPATCH_2026-09-29_17-35-36_seed42` |

```powershell
python scripts/run_experiment.py --algorithm <slug> --seed <seed> --device cuda --no-dashboard
```

- 所有运行使用同一个 `configs/cooperative.yaml`，`resolved_environment_hash` 必须一致（`745ca262…`）。
- 运行前工作区必须是干净的 Git 状态，`run_metadata.json` 中会记录 commit。
- 跑完后**不要** promote，由评审确认后再执行。

### 7.3 报告模板（实施者需提交）

1. **结果表**：每个算法 × seed 的测试利润、相对 idle 增益、strict feasible 天数比例、generalized cost；再给出均值 ± 标准差。
2. **配对检验**：Dueling 对 H24、Dueling 对 MATD3Q，在 55 天 × 3 seed 上的配对差值均值、95% bootstrap 置信区间、胜率。
3. **过程诊断**：训练第 0–500、2000–2500、4500–5000 个 episode 的均值，指标包括 `critic_loss`、`td_error_std`、`dispatch_action_grad_rms`、`dispatch_saturation_rate`、`dr_saturation_rate`，以及本算法新增的 `critic_adv_share`、`critic_dqda_*`。
4. **成本分解**：测试集上 grid、peak、curtail、comfort、ev 各项成本的均值对比，以及弃电量（kWh）。
5. 秒/episode、参数量。

## 8. 验收标准

### 8.1 工程验收（必须全部满足）
- [ ] T1–T12 全部通过，`pytest -q` 全绿
- [ ] 冒烟测试满足 AGENTS §8 的全部条款
- [ ] 与 H24 的 `algorithm.py` 和 YAML 的 diff 只包含 §4.2 和 §4.5 允许的差异
- [ ] 没有修改 §2.2 禁止的任何文件（附上 `git diff --stat`）
- [ ] 秒/episode 不超过 H24 的 1.3 倍

### 8.2 研究判定（3 个 seed，第 4500–5000 个 episode 和测试集）

| 判定 | 条件 |
|---|---|
| **机制成立** | `dispatch_action_grad_rms` 的 3-seed 均值不低于 MATD3Q 同期均值（约 0.11），**并且** `dispatch_saturation_rate` 均值不高于 0.35，**并且** `critic_adv_share` 落在 [0.02, 0.8] |
| **性能成立** | 测试利润的 3-seed 均值高于 H24 的 3-seed 均值，并且配对差值的 95% CI 下界大于 0 |
| **继续投入** | 机制成立且性能成立 → 进入下一阶段（Plan-Transformer Actor 加 MILP 蒸馏） |
| **部分成立** | 机制成立但性能不成立 → 瓶颈不在 Critic 的梯度，转去做动作语义上限测试（stage 0） |
| **不成立** | 机制不成立 → 先跑 §9 的消融 A1 和 A2，定位原因后再决定 |

## 9. 后续消融（不属于本次交付，只预留配置）

每个消融都是独立的 YAML，放在 `configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided/` 子目录下，用 `--algorithm-config` 指定，`algorithm.name` 保持不变：

| ID | 改动 | 回答的问题 |
|---|---|---|
| A1 | `detach_value_context: false` | 让 A 的梯度回流到编码器是否有帮助 |
| A2 | `use_value_context: false` | A 是否需要时间上下文 |
| A3 | `action_reinjection: false` | 动作再注入的贡献 |
| A4 | 去掉中心化（需要新增配置项 `reference_action: none`，仅用于消融） | 可辨识性的作用 |
| A5 | `dispatch_q_guidance_weight: 0` | 和 Q-guidance 之间的交互 |

## 10. 交付清单

- [ ] `algorithms/matd3_mlp_dueling_critic_h24_q_guided/`（5 个文件）
- [ ] `configs/algorithms/matd3_mlp_dueling_critic_h24_q_guided.yaml`
- [ ] `tests/test_matd3_mlp_dueling_critic_h24_q_guided.py`
- [ ] 冒烟运行目录路径，以及 AGENTS §8 自检记录
- [ ] 7 个正式运行目录路径（新算法 3 个 + H24 2 个 + MATD3Q 2 个）
- [ ] §7.3 报告（Markdown），以及 §8 的判定结论
