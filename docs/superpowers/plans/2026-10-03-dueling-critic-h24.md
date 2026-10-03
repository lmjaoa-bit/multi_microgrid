# Dueling Critic H24 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task, with a fresh final review.

**Goal:** Implement the approved H24 dueling critic study and verify the complete shared experiment workflow.

**Architecture:** Copy the H24 semantic MLP actor and algorithm locally. Replace only the critic with independent Transformer V and action-direct MLP A branches, centred on zero effective action. Return sampled diagnostics through the existing auxiliary metrics hook.

**Tech Stack:** Python, PyTorch, NumPy, YAML, pytest, the existing shared pipeline and MILP evaluator.

**Spec:** `docs/design/matd3_mlp_dueling_critic_h24_q_guided.md` (user explicitly requested implementation).

## Global Constraints

User subsequently authorized resolving the three shared failures and committing the current research code. That authorization overrides the original protected-file restriction only for the audit path formatter, zero-demand dispatch derivative and exact-GELU inference fix. Canonical MATD3 and experimental comparator packages remain unchanged. Because the shared derivative affects training, all three algorithms should train seeds 42/43/44 freshly, for nine runs total; historical seed-42 results are preserved but not reused. The user's latest instruction reserves formal training to the user: deliver runnable commands/tools, do not launch formal experiments.

- Slug: `matd3_mlp_dueling_critic_h24_q_guided`; display: `Dueling Critic T-Q H24`.
- No changes to canonical MATD3, H24 comparator, shared update, experiments, evaluation or cooperative YAML; only the three explicitly authorized shared fixes are allowed.
- No runtime imports from the removable H24 comparator package.
- Actor source copied verbatim with provenance; Q-guidance calculation and checkpoint preflight retained.
- Default configuration differs from H24 only in algorithm name and critic settings.
- Formal experiments require clean Git state, CUDA, 5000 episodes, 55 test days, fresh seeds 42/43/44 for all three algorithms; user runs them, no promotion.

## Review Focus

- `DiagnosticSchedule.due` is stateful: consume its existing result without calling it twice or changing the shared schedule.
- Q-guidance has multiple early returns: diagnostics must run for disabled, unblocked and rejected guidance too.
- Single-sample diagnostics: variance and correlation must remain finite.
- Invalid forecast payload must not influence Q, including gradients and an entirely masked horizon.
- Checkpoint rejection must occur before mutating any network, including nested configuration mismatches.

## Task 1: Independent dueling networks and registration

**Files:** New five-file algorithm package, own formal YAML, `tests/test_matd3_mlp_dueling_critic_h24_q_guided.py`.

**Interfaces:** `DuelingTemporalValue(schema, action_dim=9, *, value=None, advantage=None)`; `DuelingTwinCritic.forward(state, joint_action) -> (q1, q2)`; `q1(...)`; `decompose(...) -> {v1,adv1,q1,v2,adv2,q2}`. `Matd3MlpDuelingCriticH24QGuided` implements the existing algorithm protocol.

- [x] Write T1–T10 and schema/ablation edge-case tests; run before implementation and confirm missing variant failure.
- [x] Run required scaffold, copy actor and H24 algorithm, replace only permitted sections, author local critic and plugin.
- [x] Validate nested settings: positive integer dimensions excluding bool, H24, head divisibility, strict bool switches, zero reference only, unknown keys rejected at both levels.
- [x] Verify zero-action centring, independent twins, action gradient, state-only V, context detach, invalid forecast masking and ablations.
- [x] Verify Actor equivalence, nested checkpoint round trip and atomic rejection.
- [x] Run focused tests; expected all passing.

## Task 2: Sampled diagnostics and update invariance

**Files:** New algorithm module and test file only.

**Interfaces:** Existing `actor_auxiliary_loss` returns unchanged guidance loss plus seven scalar diagnostics at shared diagnostic steps.

- [x] Write tests for seven metrics, all agents, shared scheduling, finite singleton statistics, early guidance returns and unchanged parameter gradients.
- [x] Confirm these tests fail before diagnostic implementation.
- [x] Preserve guidance calculation verbatim in its local helper and add no-grad decomposition plus action-only autograd; observe shared schedule cursor once per update.
- [x] Compare identical seeded updates with sparse/dense diagnostic intervals: Actor, Critic and targets exactly equal.
- [x] Verify metric aggregation into `actor_diagnostics.csv` through the existing writer.
- [x] Run focused tests; expected all passing.

## Task 3: Full pipeline acceptance and engineering evidence

**Files:** Own `configs/smoke/...yaml`, `docs/DUELING_CRITIC_H24_IMPLEMENTATION.md`; working outputs only.

- [x] Run `pytest -q`; record all failures by name if any.
- [x] Run default-config command with 3 training and 2 test days.
- [x] Run additional update-enabled smoke with learning starts 8, batch 8, capacity 256, diagnostics every policy step; validation and MILP remain enabled.
- [x] Verify plugin/YAML, best checkpoint, settlement, summaries, 15 PNG + 15 PDF, diagnostic columns and working-run layout.
- [x] Measure formal-size parameter counts and matched H24/Dueling seconds per episode after learning starts on this CUDA device; enforce ratio <= 1.3.
- [x] Attach H24 algorithm/config diffs and compare protected-file hashes with initial snapshot.
- [x] Dispatch one independent final code review, address material findings with regression tests.

## Task 4: Formal study and saved-result report

- [ ] Resolve clean-Git prerequisite without discarding or silently committing pre-existing user work.
- [ ] Verify both saved seed-42 baselines and environment hashes before reusing them.
- [ ] User runs nine fresh formal experiments sequentially using the shared CLI and unchanged formal YAMLs.
- [ ] Report every run path, per-seed metrics, mean/std, day-aligned paired bootstrap intervals and win rates, diagnostic windows, cost decomposition, runtime and parameter counts.
- [ ] Evaluate spec §8.2 only with complete evidence; leave results in working runs and do not promote.

## Execution record

The original working tree has pre-existing tracked edits and untracked research packages (including H24). Work stays in the current checkout so it uses these exact sources; no unrelated changes were discarded. At first commits were deferred; after explicit authorization the existing research sources are preserved in the delivery commit. Formal runs are reserved to the user. This plan is executed inline under the user's direct implementation request.

The spec's literal second `diagnostic_schedule.due(update_steps)` call conflicts with its requirement to match shared scheduling: `due()` advances `next_boundary`. The local hook will observe the cursor advance and cache the due update, preserving the shared algorithm and training numerics.

The default 3-day smoke cannot reach learning_starts=1000. A separate smoke YAML is necessary to verify update-time metrics while keeping the formal comparison untouched.

### Progress ledger

- Task 1: implemented; missing-plugin RED reproduced; all network, schema/config, Actor, checkpoint and dependency-isolation tests GREEN.
- Task 2: implemented; missing-diagnostic-metrics RED reproduced; all scheduling, gradient isolation, exact training-equivalence and CSV checks GREEN.
- Task 3: verification completed, overall acceptance blocked. New algorithm 50 tests GREEN; final full suite 608 passed / 3 existing shared-module failures. Both smoke runs completed; update-enabled smoke has exactly 15 PNG / 15 PDF, seven metrics and valid settlements. CUDA time ratio 1.0444; protected source hashes unchanged. Independent reviewer found no implementation issues.
- Task 4: not started; requires resolving shared-test failures under spec §2.2 and dirty Git under §7.2. Historical seed-42 metadata also records git_dirty=true, so its provenance cannot be assumed clean.
- Ruling: observe the shared diagnostic cursor rather than re-call stateful due() — second call consumes a result that has already been consumed — tests pin parity; cost if wrong is missing diagnostics, with no changes to training loss.
- Ruling: keep the user's existing checkout and defer commits — H24 sources and other dependencies are untracked and unrelated user edits must be preserved — formal reproducibility remains pending.
- Ruling: use an additional update-enabled smoke YAML — the default 72 steps cannot reach learning_starts=1000 — formal YAML unchanged; smoke data does not establish convergence.
- Ruling: complete full-suite verification but do not alter protected shared code to fix existing failures — user-approved design explicitly forbids it — full acceptance and formal experiments remain blocked.
- Final review: no actionable findings. Reviewer declined pending empirical results (now independently measured), formal research outcomes (pending), unrelated changes (preserved), and inherited checkpoint/resume behavior (unchanged).

## Authorized continuation

### Task 5: Shared fixes and reproducible source

- [x] Reproduce the three existing failures and add mixed-batch / peer-gradient and seeded float64-reference regressions.
- [x] Normalize audit manifest paths with `as_posix`, make zero residual local demand have zero local discharge gradient, and bypass inaccurate fused GELU inference using the original exact functional GELU.
- [x] Verify unchanged physical forward ledgers and unchanged Transformer training outputs/gradients; keep existing CPU/CUDA tolerances.
- [x] Run full pytest and three update-enabled shared-pipeline smoke runs with settlement and canonical 15 figure pairs.
- [x] Preserve all existing research source in a reviewed commit on `codex/dueling-h24-study`; no push, merge or release promotion.

### Task 6: User-operated formal queue and automatic saved-result report

- [x] Create a narrow CLI queue that delegates all training/evaluation to `scripts/run_experiment.py`, verifies clean/fixed Git state, environment/config hashes and best-model artifacts, and writes progress under ignored working outputs.
- [x] Add a saved-result-only Markdown report generator covering nine run paths, per-seed/aggregate results, paired day/seed bootstrap, diagnostic windows, costs, runtime and parameter counts, and spec §8.2 decisions only when all nine runs are valid.
- [x] Test rejection of dirty state, incomplete/mismatched results and unmatched pairs; test exact constant paired differences and partial-report status.
- [x] Run final tests/review, create a clean commit, and deliver the user's single-run and optional nine-run CLI commands. Do not launch formal training; tools update the report after verified runs and stop on failures/mismatches.
- [x] Record that formal performance conclusions await the user's formal runs. Do not substitute smoke data for research evidence.

### Authorized continuation progress

- Task 5 shared fixes: complete. Existing failures reproduced without the new plugin; retained old tolerances/expectations, added regression tests. 62 targeted tests pass, forward dispatch has 1,142,916 exactly equal scalar comparisons; standard exact-GELU training outputs and gradients match. Canonical MATD3 and the H24 experimental sources unchanged.
- Task 5 pipeline verification: complete. All three update-enabled 3/2-day smoke runs use their own YAML, test verified best-model hashes, complete MILP and Equal/Shapley, and each produce exactly 15 matching PNG/PDF pairs with the same environment hash.
- Task 6 tools: complete. Nine fresh fixed-source jobs, saved-result-only Markdown/JSON report, seed/date pairing, hierarchical bootstrap, diagnostic windows, costs, timings and counts. 18 targeted tests pass. Independent review found three queue lifecycle issues, each addressed with fail-first regressions; final review has no remaining P1/P2 finding.
- Final verification: **638 passed, 4 warnings, 169.93 s**. Warnings are the established nonfinite-JSON and invalid-Shapley fixtures plus slow list-to-tensor construction. Protected snapshot has only the two authorized changes in its 69 files; audit script is outside that original snapshot.
- Source preservation: existing research sources and this implementation are retained together on local branch `codex/dueling-h24-study`, committed before handoff; no push, merge or release promotion.
- Formal training: explicitly reserved to the user. No formal job launched. Research judgment remains pending until the user completes the nine-run cohort.
