# Forecast Context 四组消融实验实施计划

> **For agentic workers:** 如由代理实施，使用 superpowers:executing-plans 按任务推进；只有用户明确选择代理并行时才使用 superpowers:subagent-driven-development。本文交接给其他实施者，本次不启动执行。

**Goal:** 实现一个可配置的新算法，以 2×2 对照分离同伴未来信息和预测窗口对收益的影响。

**Architecture:** 新增独立的 `matd3_forecast_context` 包，局部适配普通 T Actor 和算法接口。四组共用一份实现，使用完整的独立 YAML；训练、正式测试、结算和绘图复用现有管线。研究汇总只读取已保存结果。

**Tech Stack:** Python、PyTorch、NumPy/Pandas、PyYAML、pytest；沿用现有 MILP 和绘图依赖。

**Spec:** [设计与实施约束](../specs/2026-09-28-forecast-context-ablation-design.md)。实施前必须完整阅读设计及仓库 AGENTS.md。

## Global Constraints

- 新插件 slug 固定为 `matd3_forecast_context`；默认配置为 A。
- A=(8,false)，B=(8,true)，C=(24,false)，D=(24,true)。`algorithm.name` 均等于新 slug。
- 不修改 canonical MATD3、现有两个插件、共享物理/奖励/预测、更新规则、结算、绘图或默认实验。
- 原训练接口保持不变；新模型必须安装已有 grouped Actor 的诊断更新路径。
- 原始结果和 releases 不改写。工作结果遵循 `outputs/working_runs/<slug>/runs/`。
- 当前任务只有文档交接；下列命令供实施者后续执行，不表示已执行或验证通过。

## Review Focus

1. 23 点全部未来无效：前六个 token 保持有效，输出有限且不受无效预测影响（Task 2）。
2. A/C 或 B/D 权重形状一致：仍必须拒绝跨 H 加载（Task 3）。
3. plugin 安装器替换 update 且有继承标记：新类只包装一次、与普通 T 路径一致、旧类未被改变（Task 1、4）。
4. 同伴标识/次序/尺度变化：禁止静默重解释检查点；新字段有实际计算路径（Task 2、3）。
5. 重跑或缺组、日期重复、跨 seed 场景不一致：汇总显式报错或标记，不挑最优运行、不伪造独立样本（Task 5）。

## Task 0：冻结实施基线与检查工作区

**涉及文件：**设计文档、AGENTS.md、现有算法和管线，只读。

- [ ] 记录起始提交、Git 状态和相关文件 hash，保留已有未提交工作。
- [ ] 若需隔离开发，按仓库/宿主已有工作树机制创建或复用；不要覆盖他人的分支或运行目录。
- [ ] 核对新包、四个 YAML 路径尚未存在；存在时先核对来源，不直接重建覆盖。
- [ ] 确认 `--algorithm-config`、`--episodes`、`--test-episodes`、`--validation-days`、`--validation-interval` 和 `--no-dashboard` 仍为有效参数。
- [ ] 锁定本次算法配置基线；记录数据文件 SHA256。本文核对提交为 `cb42e9c04cb9d92360b72e705e4243f751e3680d`，不强制实施者回退当前仓库。

完成条件：已明确工作区及参考代码版本，未更改生产实现。

## Task 1：创建独立插件和完整配置

**创建文件：**`algorithms/matd3_forecast_context/{__init__,plugin,algorithm,actor}.py`；四份 YAML 路径见设计 §3；`tests/test_forecast_context_plugin.py`。

**接口：**类 `Matd3ForecastContext`，共享协议保持不变；插件唯一 `ALGORITHM_SPEC`；局部算法继承稳定 canonical `MATD3`，不继承现有实验包。

- [ ] 先增加 `test_explicit_plugin_load_and_default_group_a`：显式 slug 可加载，默认 YAML 为 A，display_name 和 environment_overrides 符合设计。新插件不存在时测试应失败。
- [ ] 增加 `test_four_configs_have_only_allowed_differences`：四份 YAML 的差异只能为 experiment.group、network.forecast_horizon、network.include_peer_forecast，其他值一致；算法名均为新 slug。
- [ ] 增加 `test_invalid_group_and_network_config_rejected`：错配组别、H=0/9/25、布尔 H、字符串 peer 开关、未知 feature 列表均拒绝。
- [ ] 运行上述测试，确认失败源自缺少实现/不符合契约，而不是导入环境问题。
- [ ] 执行脚手架命令并补齐本包内普通 T 适配；注意 `--base` 不会自动生成对应网络。

```powershell
python scripts/create_algorithm_variant.py --name matd3_forecast_context --base matd3_hierarchical_transformer
```

- [ ] 从锁定普通 T 实现局部建立 A 的起点，尽量保持构造顺序、模块名和随机初始化调用顺序，便于等价比较。
- [ ] 新插件只对新类调用 `install_diagnostic_sampling`，variant 为新 slug，plain_actor=False。保留 Actor branch 和参数缓存接口。
- [ ] 实现四份完整配置；不得增加 profile/default catalog 条目或自定义 YAML include。
- [ ] 增加 `test_plugin_import_is_idempotent_and_does_not_mutate_old_classes`：重复加载新插件不重复包装，旧算法类的方法对象和行为不被改写。
- [ ] 增加 `test_existing_public_selectors_unchanged`：默认和 `--algorithm matd3` 仍指向 matd3_dispatch，`--policy transformer` 仍能唯一选择 ordinary_t。
- [ ] 检查新生产包无现有普通 T 包的运行时依赖；移除新包的隔离目录测试中，MATD3 注册仍正常，不实际删除工作区文件。
- [ ] 运行 `pytest tests/test_forecast_context_plugin.py -q`；通过后提交本任务。

注意：Task 1 的最小 A 路径可以先完整工作；B/C/D 在 Task 2 才完成，不能在此宣称四组可训练。

## Task 2：实现 Forecast Context 输入构造

**文件：**`algorithms/matd3_forecast_context/actor.py`；`tests/test_forecast_context_actor.py`。

**接口固定：**

```python
class ForecastContextActor(nn.Module):
    # 构造参数沿用 HierarchicalTransformerActor，追加 keyword-only 参数：
    # include_peer_forecast: bool = False
    # peer_forecast_features: Sequence[str] = ("energy_balance",)

    def _build_forecast_features(self, obs: torch.Tensor) -> torch.Tensor:
        """[B, observation_dim] -> [B, H, 6 or 8], ordered per design."""

    def _build_tokens(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,6+H,32] tokens and [B,6+H] boolean padding mask."""

    def forward(self, obs: torch.Tensor, collect_diagnostics: bool = False) -> torch.Tensor:
        """Return [B,2] bounded raw actions; retain diagnostics protocol."""
```

- [ ] 先写参数化 `test_forecast_features_and_tokens_match_matrix`：A/B/C/D 的 H、feature_dim、token_count 分别为 (8,6,14)/(8,8,14)/(24,6,30)/(24,8,30)。
- [ ] 写 `test_physical_net_load_and_peer_sign_scales`：例如本地尺度 load=100,pv=500,wind=1000,power=1000，观测 .5/.2/.0 应得到 -.05；同伴字段 +.2/-.3 必须原值按固定顺序进入最后两列，不能用本地尺度再次归一化。
- [ ] 写 `test_future_hour_encoding`：当前 08:00、k=1 必须对应 09:00 sin/cos；检查 k=8/24 的角度公式。validity 与时间编码分开测试。
- [ ] 写 `test_peer_future_enabled_disabled_dependency`：B/D 更改有效同伴余额使 feature 对应列准确改变；A/C 完全不读取它们。
- [ ] 写 `test_horizon_24_reads_late_fields`：C/D 改 k=9 的本地及启用的 peer 字段，features 对应行改变；A/B 不受影响。
- [ ] 写 `test_missing_schema_fields_fail_fast`：缺有效标记、启用 peer 的某一时刻字段、HVAC 温度时均报清楚错误；非 HVAC 温度为零且不误取 hour_sin。
- [ ] 写 `test_padding_invariance_and_all_future_masked`：固定 eval 模型，改变 mask=false 对应的预测内容为不同有限数，动作 assert_close(atol=1e-6,rtol=1e-5)；全部未来屏蔽时无 NaN/Inf。
- [ ] 写 `test_peer_features_have_differentiable_path`：先对 feature 拼接做精确梯度测试，再用固定种子的小幅输入及受控非饱和权重/输出头确认新列能传播到 action。不要以任意已训练饱和模型都必须响应为断言。
- [ ] 写 `test_forecast_parameter_delta_is_128_per_actor`：保持其余结构时 B-A、D-C 每 Actor 参数差为 128；C-A 参数差为 0。
- [ ] 运行失败测试；按设计实现索引缓存、特征拼接和 Linear 输入维度。不加新的 token、位置编码、网络层或未来真实 DR 特征。
- [ ] 运行 `pytest tests/test_forecast_context_actor.py -q`；CUDA 可用时补 CPU/CUDA 前向一致性检查（atol=1e-5,rtol=1e-4），记录设备差异。
- [ ] 检查所有 index tensor/buffer 跟随 `.to(device)`，schema/尺度仍由当前环境构造，缓存不携带批次数据；通过后提交。

## Task 3：实现检查点版本契约

**文件：**`algorithms/matd3_forecast_context/algorithm.py`；`tests/test_forecast_context_checkpoint.py`。

**内部接口：**

```python
def _forecast_context_contract(self) -> dict[str, Any]: ...
def _validate_forecast_context_checkpoint(self, checkpoint: Mapping[str, Any]) -> None: ...
def save(self, path: str | Path) -> None: ...
def load(self, path: str | Path) -> None: ...
```

- [ ] 写 `test_checkpoint_roundtrip_each_group`：A/B/C/D 各自保存加载后，固定输入输出一致，update_steps、online/target 网络一致。
- [ ] 写 `test_cross_group_horizon_rejected_even_when_shapes_match`：A→C、C→A、B→D、D→B 必须在加载权重前抛出兼容性错误。
- [ ] 写 `test_peer_switch_order_scale_schema_and_architecture_rejected`：peer 开关、agent_order/peer_order、任一 normalizer、schema、head 数/层数/宽度发生变化均拒绝；错误应能指出差异项。
- [ ] 写 `test_legacy_checkpoint_rejected`：旧普通 T、MATD3、缺 version/contract 的新 slug 都不能静默加载。
- [ ] 写 `test_checkpoint_cannot_override_environment_contract`：篡改 checkpoint normalizer 后加载失败，当前实例尺度不发生改变。
- [ ] 先运行失败测试；按设计 §6 实现完整 contract、canonical JSON 签名与加载前校验，保留投影和特征语义校验。
- [ ] 运行 `pytest tests/test_forecast_context_checkpoint.py -q`；通过后提交。

## Task 4：A 组等价性与真实更新路径

**文件：**`tests/test_forecast_context_equivalence.py`，必要时修正本地新算法；禁止为通过测试修改旧算法。

**输入：**通过 registry 加载的现有普通 T 与新 A；相同环境、配置公共部分、权重、replay transitions 和随机状态。

- [ ] `test_a_actor_matches_reference_with_same_weights`：在训练/验证/测试来源的固定有限观测及日末边界上，将参考 Actor state_dict 直接复制给 A Actor，比较 tokens/mask/actions；CPU atol=1e-6,rtol=1e-5。该复制只属于测试，不放宽正式 load 身份校验。
- [ ] `test_a_initialization_matches_reference`：同 seed、相同构造调用路径下比较参数键/形状和初始动作；若不等，定位额外随机调用。禁止用不同随机初始化掩盖实现差异。
- [ ] `test_a_projection_and_replay_contract_match`：同一 state/raw action 的有效动作与 plans 数值相同；replay 类型、采样字段和 joint reward 语义一致。
- [ ] `test_a_controlled_update_matches_reference`：两个实例放入同一批真实 transitions，复制全部 online/target 网络和 optimizer 初始状态；分别恢复 Python/NumPy/Torch/采样器 RNG 状态后 update。覆盖一次 Critic-only 步和一次 Actor 更新步，比较损失及更新后参数；CPU atol=1e-6,rtol=1e-5。
- [ ] 同时覆盖 diagnostic_interval=20 的非采样步及诊断到期步，确认没有错误走回未包装旧 update。性能、日志字段和非有限计数分别检查。
- [ ] 对旧插件的包装状态独立取证，不能把两个实例错误绑定到同一修改后的类而获得虚假的相等。
- [ ] 不将与 9 月 28 日历史训练最终收益完全相等设为单元测试；历史代码、评估频率及硬件存在差异。
- [ ] 运行 `pytest tests/test_forecast_context_equivalence.py -q`；失败先定位数学/随机/安装路径，不随意放宽容差。通过后提交。

完成条件：A 是当前锁定普通 T 的可靠对照；否则不进入长训练。

## Task 5：实现只读研究汇总

**文件：**`scripts/summarize_forecast_context.py`；`tests/test_forecast_context_summary.py`。

**接口：**

```python
def summarize_study(run_dirs: Sequence[Path], *, study_id: str,
                    required_seeds: Sequence[int]) -> Path:
    """Read saved results; write a new reports/forecast_context/<study_id> directory."""
```

CLI 固定为 `--study-id`、可重复的 `--run`、`--require-seeds`。study_id 只允许 `[a-z0-9][a-z0-9_-]*`，解析后的目标必须留在 reports/forecast_context；已存在则拒绝覆盖。

- [ ] 写三 seed×四组、每组 55 个简化日记录的合成结果 fixture，不依赖真实大训练文件；正式 study 固定要求 55 天，不能为迁就测试放宽完整性判断。
- [ ] `test_factor_effects_and_date_pairing`：打乱日期顺序后，仍按 date 得到正确 B-A、C-A、D-C、D-B、D-C-B+A；不按 CSV 行号相减。
- [ ] `test_duplicate_missing_run_or_date_rejected`：重复 group×seed、缺少组/seed、重复/缺失日期、未完成状态、未使用 best checkpoint 均拒绝正式完整报告。
- [ ] `test_config_and_baseline_consistency`：同 seed 不同组出现环境 hash、参考收益、日期、训练预算/运行线程等不一致时拒绝或清楚标记不可比；只有设计白名单允许差异。
- [ ] `test_ratio_of_sums_and_seed_aggregation`：用逐日分母差异显著的 fixture 保证不是均值比值混淆；跨 seed std 使用 ddof=1；required_seeds=[42] 时 std 不伪造为已知零。
- [ ] `test_read_only_and_no_overwrite`：源文件 hash 不变，无策略或求解器执行；目标已存在时无覆盖。模拟未知公共 policy=`unregistered` 但合法 slug/config 时仍能识别。
- [ ] 实现设计 §8 五个输出文件；源字段缺失不填 0。仅从 hour CSV 计算辅助用电和成本，不重新跑环境。Actor H 从配置/新 checkpoint contract 读取，不误用共享观测的 forecast_horizon_hours。
- [ ] 对初始数据 hash、代码 hash、每组 config_snapshot 和选中的 run 路径做可追溯记录；记录研究前后采集时点。
- [ ] 运行 `pytest tests/test_forecast_context_summary.py -q`；通过后提交。

## Task 6：全套测试及四组短流程

- [ ] 运行 `pytest -q`，保存实际通过/失败/跳过结果及运行环境。存在失败时不得报告全部通过。
- [ ] 每组完成一次 3 回合训练、验证/测试各 2 天的短流程。另生成临时 smoke YAML，保存于 `reports/forecast_context/smoke_inputs/<唯一检查批次>/`，由对应正式配置完整派生，仅覆盖 batch_size=16、warmup_steps=24。正式四份配置保持原值。这样 72 个环境步能实际发生 Actor/Critic 更新，不会出现“只走流程、没有训练”的假验收。
- [ ] 四份 smoke YAML 保留正式组别和 H/peer 组合，使用文件路径/运行记录明确标记 smoke；正式汇总必须按 5000 回合/55 天预算规则拒绝纳入这些运行。检查最终训练历史及 checkpoint update_steps，确认更新确实发生。

示例 A（其余替换 algorithm-config）：

```powershell
python scripts/run_experiment.py --algorithm matd3_forecast_context --algorithm-config reports/forecast_context/smoke_inputs/check01/a.yaml --episodes 3 --validation-interval 1 --validation-days 2 --test-episodes 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

上面的 `check01/a.yaml` 必须先生成，批次目录重名时另取新名。短流程的 `--no-learning-curve` 仅减少工程检查开销；另对 A/D 各补一次启用学习曲线的短检查（interval_steps=24、eval_episodes=2），验证 initial evaluation 与新算法兼容。正式训练使用正式四份配置，不能继续传入 smoke YAML。

- [ ] 每组核实：选中插件及 YAML；best_model.pt 确实用于正式测试；Actor/Critic 权重发生更新；日指标与小时轨迹完整；七联盟 MILP 成功或明确报告失败；Equal/Shapley 守恒；summary 和绘图完成；15 PNG＋15 PDF；正常 working_runs 路径。
- [ ] 保存验证证据，但不把 smoke 的 2 天结果混入正式 study。
- [ ] 用 Git diff 核实受保护算法、原 YAML、shared pipeline、物理配置和 public selector catalog 没有变化。
- [ ] 完成一次实现评审并修正发现的问题；重新运行受影响测试，最终 `pytest -q` 及必要 smoke 在最终代码上通过。
- [ ] 提交工程完成版本，记录 clean commit；此时可交付代码，不代表 12 个长运行已完成。

## Task 7：正式 2×2×3 研究运行

前置条件：Task 1–6 通过，代码/配置/数据已冻结，执行者明确承担长训练预算。

- [ ] 先固定三个 seed `[42,43,44]`、四组配置及报告规则。每个 seed 内顺序可预先轮换减少机器状态偏差：42 用 A/B/C/D；43 用 B/C/D/A；44 用 C/D/A/B。单 GPU 顺序运行，避免争抢资源污染耗时比较。
- [ ] 运行示例（每次更改配置路径和 seed）：

```powershell
python scripts/run_experiment.py --algorithm matd3_forecast_context --algorithm-config configs/algorithms/matd3_forecast_context.yaml --device cuda --seed 42 --no-dashboard
python scripts/run_experiment.py --algorithm matd3_forecast_context --algorithm-config configs/algorithms/matd3_forecast_context/b_peer_h8.yaml --device cuda --seed 42 --no-dashboard
python scripts/run_experiment.py --algorithm matd3_forecast_context --algorithm-config configs/algorithms/matd3_forecast_context/c_local_h24.yaml --device cuda --seed 42 --no-dashboard
python scripts/run_experiment.py --algorithm matd3_forecast_context --algorithm-config configs/algorithms/matd3_forecast_context/d_peer_h24.yaml --device cuda --seed 42 --no-dashboard
```

- [ ] 每次记录返回的 run_directory，而非最终再读取 latest_run。完整参数写入每次运行记录。
- [ ] 每个正式运行必须 5000 回合、55 测试日，验证选 best、正式测试、MILP、结算、summary 和 30 个标准图文件完整。
- [ ] 对 seed43/44 按同样规则完成。失败或中断运行保留，记录工程原因；重跑纳入规则只取预先指定的首个完整有效重跑，不按收益挑选。
- [ ] 中途若修改算法/正式配置，则建立新的 study_id，旧结果不能无说明混合为同一研究。
- [ ] 研究结束复核数据 hash 与 runtime/环境配置一致。

## Task 8：汇总、结论与交接

- [ ] 使用 12 个显式 run 路径运行只读汇总；示例语法：`python scripts/summarize_forecast_context.py --study-id fc_2x2_seed42_43_44 --require-seeds 42 43 44 --run <A42路径> --run <B42路径> ...`。尖括号与省略号仅为说明，执行时必须替换为全部实际目录。
- [ ] 核对三 seed 的四个简单效果与交互项；确认日均收益可以与逐小时收益求和对账。
- [ ] 分开写“实现是否正确”“哪项输入有帮助”“服务/耗时的代价”“尚未证明什么”。结果为负或不一致时如实报告。
- [ ] 不将多个日样本当作多个训练种子；不把历史 MATD3 单 seed 与新四组的多 seed 结果混成严格架构排名。
- [ ] 如果仍需要与 MATD3 作正式外部比较，另以相同预算/runtime/种子新增 MATD3 对照运行，不改其实现；这额外 3 个运行不包含在本轮 12 个核心消融中。
- [ ] 交付新算法/配置、测试结果、smoke 证据、manifest、汇总 CSV、限制说明；列出所有未完成项。
- [ ] 仅在结果被接受保留后执行显式 promote；不覆盖 release，不删除 working source，不自行接受 dirty 风险。

## 实施者最终检查表

| 项目 | 必须提供的证据 |
|---|---|
| 算法隔离 | Git diff；新包无旧实验运行时 import |
| A 等价 | 前向、初始化、投影和受控更新测试结果 |
| 预测确实接入 | 字段顺序、特征/梯度路径、mask 测试 |
| 检查点安全 | 四组 roundtrip；跨 H/开关/schema/尺度拒绝测试 |
| 工程闭环 | pytest；四组 smoke；实际更新；正式测试 best；MILP/结算/30 图 |
| 研究完整性 | 四组×三 seed manifest，日期/配置/数据一致性 |
| 结论可信 | 配对效果、seed 级方差、服务指标、限制和失败运行记录 |
