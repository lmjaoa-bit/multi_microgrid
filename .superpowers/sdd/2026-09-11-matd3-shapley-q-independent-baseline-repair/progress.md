# SDD ledger — plan: docs/superpowers/plans/2026-09-11-matd3-shapley-q-independent-baseline-repair.md

## 执行环境

- Branch: `codex/matd3-shapleyq-baseline`
- Workspace: 当前 checkout（Gemini 未提交半成品已按用户授权清理）
- Clean baseline test evidence before implementation: `pytest -q` → 84 passed, 2 expected warnings
- Ruling: 不创建 linked worktree — 已切换到独立 `codex/` 分支且用户要求最快方案；错误代价是目录级隔离较弱，但分支可回退。

## 任务状态

- [x] Task 1: Characteristic Value 与 exact Shapley — implementation `27c9177`; archived report `9c2b363`; traceability fix `d849caa`; focused `4 passed`; full `88 passed, 2 warnings`; independent review APPROVED
- [x] Task 2: 固定 Independent Baseline 与 Common Scenario — implementation `9d45759`; review fixes `b44b400`; focused `11 passed`; task file `33 passed`; full `117 passed, 2 warnings`; independent re-review APPROVED
- [x] Task 3: Replay 与 per-agent Twin-min target — implementation `7d5b9e8`; direct review fixes `18e80cf`; dedicated `147 passed`; full `231 passed, 2 warnings`
- [x] Task 4: 训练日志、summary、dashboard、TensorBoard — `9ccee84`; focused `156 passed`; full `238 passed, 2 warnings`
- [x] Task 5: Formal Test baseline 输出 — `c06c3c5`; focused `30 passed`; full `240 passed, 2 warnings`
- [x] Task 6: hard SOC 与确定性物理场景 — `d23f6d5`; focused `17 passed`; full `244 passed, 2 warnings`
- [x] Task 7: 全量验证与 smoke workflow — dedicated `157 passed`; full `244 passed, 2 warnings`; smoke complete; PNG=15/PDF=15

## Pre-flight 冲突扫描

| 范围 | Producer → Consumer / 共享面 | 检查结果与裁定 |
|---|---|---|
| Task 1 自身 | 数学测试 → normalization/exact Shapley helper | 一致；期望值是手工 literal，不复用生产逻辑。 |
| Task 2 自身 | reset metadata → baseline scenario clone → physics/gradient tests | 一致；只同步场景，不同步 cooperation mask。 |
| Task 3 自身 | baseline return-to-go → replay → Bellman diagnostics | 一致；per-agent Twin min 顺序符合规格。 |
| Task 4 自身 | baseline step API → replay extras → history/summary/export | 发现原计划 Step 2 的 `-k` 表达式与测试名不匹配。Ruling: 实施时运行精确 node `test_algorithm_replay_extras_uses_fixed_baseline_step`；错误代价仅为计划命令偏差，无产品语义变化。 |
| Task 5 自身 | evaluator callbacks → daily rows → JSON/CSV summary | 一致；callbacks 默认 `None`，保持现有算法兼容。 |
| Task 6 自身 | 临时确定性数据 → 双环境 rollout → physics/SOC assertions | 一致；测试真实环境，不 mock physics。 |
| Task 7 自身 | 已实现代码 → focused/full/smoke/artifact audit | 发现最终 commit 可能为空。Ruling: 若前置任务已逐项提交，Task 7 只记录验证证据，不制造空提交；错误代价为少一个汇总 commit，不影响可审计性。 |
| Task 1 ↔ Task 2 | 共享 `tests/test_matd3_shapley_q.py` | 顺序执行；Task 2 基于 Task 1 已提交测试文件增量修改。 |
| Task 1 ↔ Task 3 | 共享 `tests/test_matd3_shapley_q.py` | 顺序执行；无接口冲突。 |
| Task 1 ↔ Task 4 | 共享 `tests/test_matd3_shapley_q.py` | Task 4 只增加 pipeline/output tests，不改数学 helper 语义。 |
| Task 1 ↔ Task 5 | 共享 `tests/test_matd3_shapley_q.py` | Task 5 只增加 evaluation lifecycle coverage。 |
| Task 1 ↔ Task 6 | `all_coalitions`/`exact_shapley_values` → no-cooperation physics test | 接口匹配；Task 6 消费 Task 1 pure helpers。 |
| Task 2 ↔ Task 3 | `get_step_data` → replay return-to-go fields；共享测试文件 | 接口字段名完全一致：reward/profit/return_to_go/next_return_to_go。 |
| Task 2 ↔ Task 4 | `get_step_data`、episode totals → training hook/history | 接口匹配；Task 4 不修改 baseline policy。 |
| Task 2 ↔ Task 5 | `episode_metrics`、`on_episode_start` → formal evaluation callbacks | 接口匹配；callback 使用 formal-test 的同一 seed/reset options。 |
| Task 2 ↔ Task 6 | 共享 `envs/multi_microgrid_env.py`、配置与测试 | Task 2 只增加 reset metadata；Task 6 验证 physics/SOC，不改变 metadata。 |
| Task 3 ↔ Task 4 | replay extra kwargs、diagnostic names → pipeline aggregation/export | 字段名匹配；grand_decomposition_residual 来自真实 Q reference。 |
| Task 3 ↔ Task 5 | algorithm checkpoint/baseline evaluator → formal load/evaluation | 新版 checkpoint 被 formal evaluator 加载；旧版明确拒绝。 |
| Task 3 ↔ Task 6 | 共享算法测试；Cooperation Surplus 定义 → physics sanity | 物理 return 差与 critic reward 差使用同一符号方向。 |
| Task 4 ↔ Task 5 | history/summary 与 formal daily output | 训练和正式测试使用同名 Grand/Independent/Surplus 字段；MILP baseline 保持独立命名。 |
| Task 4 ↔ Task 6 | 共享 metrics/算法测试 | Task 6 不改变输出字段；无冲突。 |
| Task 5 ↔ Task 6 | 共享正式测试所使用的 hard SOC/Common Scenario physics | Task 6 的配置是 Task 5 formal evaluation 的物理基础；一致。 |

## Rulings

- Ruling: Task 4 的 focused RED/GREEN 命令使用精确测试 node，不使用失配的 `-k` 表达式 — 防止零测试被误判为验证成功 — 若判断错误只会多运行一个单测。
- Ruling: Task 7 不要求空的汇总 commit — 每个实现任务已有独立 commit 和 review range — 若判断错误只影响提交分组，不影响代码。
- Ruling: 用户明确表示 Gemini 未提交半成品无需保留且已有备份；已逐文件删除未跟踪的 `matd3_shapley_q` 半成品并恢复 4 个已跟踪文件到 HEAD，规格与实施计划提交保留 — 这样满足 TDD 的干净起点；错误代价是不能直接复用 Gemini 代码，但用户接受且备份可恢复。
- Ruling: 清理后 Task 1 的算法目录与配置不再存在。Task 1 必须先创建会因 import/package 缺失而失败的测试，确认 RED 后运行仓库规定的 `python scripts/create_algorithm_variant.py --name matd3_shapley_q`，再实现 coalition helper；scaffold 产生的 `__init__.py`、`algorithm.py`、`plugin.py` 与 YAML 允许作为 Task 1 的必要额外文件 — 错误代价是 Task 1 review surface 变大，但满足 AGENTS.md 强制 scaffold 规则。
