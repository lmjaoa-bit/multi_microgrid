# Peak-Risk Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated predictive-DR Transformer variant that learns to reduce the remaining daily three-microgrid actual-load maximum while keeping training overhead low and preserving profit and physical feasibility.

**Architecture:** Each decentralized actor receives one derived peak-risk token. One small system-wide twin peak critic learns a no-discount max-return Bellman target from already projected actions; its conservative estimate enters the delayed actor loss with a ramped weight. All parent projections and replay batches are reused.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, pytest, YAML, existing MATD3 plugin and experiment pipeline.

**Spec:** `docs/superpowers/specs/2026-09-21-peak-risk-transformer-design.md`.

## Global Constraints

- Complete `docs/superpowers/plans/2026-09-21-training-reliability-recovery.md` first.
- Create `matd3_hierarchical_transformer_peak_risk`; do not edit `algorithms/matd3/`.
- Parent is corrected `matd3_hierarchical_transformer_predictive_dr`, contract `1.1` or its explicitly versioned reliability-only successor.
- Do not inherit `matd3_hierarchical_transformer_p2p_value`.
- Keep raw actions `[ESS intent, DR intent]`, physical projection, environment reward, settlement, and formal evaluation unchanged.
- Primary peak is simultaneous `sum(mg1..mg3 actual_load_kw)`, not grid import or per-microgrid peak sums.
- Exactly one shared twin peak critic; default hidden dimensions `[64, 64]`.
- No additional projector call, replay sample, environment rollout, MILP solve, or formal-test day in a normal update.
- Median update and 100-episode wall-time overhead target is at most 15%; over 20% blocks full training.
- Peak-critic interval 2 is allowed only if the design's learning-adequacy comparison passes; otherwise use interval 1.
- Formal test data is never used for training, checkpoint selection, or hyperparameter selection.

## Review Focus

- All-invalid forecast horizons must yield finite peak-token features and the documented validity indicator; Task 2 tests it.
- Token-disabled and zero-weight mode must execute the exact parent path, not a zero extra token; Tasks 4 and 5 test bitwise action/update parity.
- Terminal transitions, single-row batches, and a next predicted peak below current load must produce the exact max-return target; Task 3 tests each case.
- Peak critics must be frozen during actor updates while gradients still reach the active actor through physical projection; Task 5 tests both sides.
- Missing/nonfinite gate fields and mismatched test-day sets must fail closed instead of producing a favorable comparison; Task 6 tests these artifacts.

---

## File structure

**Create**

- `algorithms/matd3_hierarchical_transformer_peak_risk/__init__.py` — public export.
- `algorithms/matd3_hierarchical_transformer_peak_risk/plugin.py` — one `ALGORITHM_SPEC`.
- `algorithms/matd3_hierarchical_transformer_peak_risk/peak_risk.py` — configuration, token features, scale, target and weight helpers.
- `algorithms/matd3_hierarchical_transformer_peak_risk/actor.py` — 15-token predictive actor.
- `algorithms/matd3_hierarchical_transformer_peak_risk/critic.py` — one lightweight shared twin cost critic.
- `algorithms/matd3_hierarchical_transformer_peak_risk/algorithm.py` — construction, contracts, checkpoints, recovery hooks.
- `algorithms/matd3_hierarchical_transformer_peak_risk/training.py` — efficient integrated MATD3/peak update.
- `configs/algorithms/matd3_hierarchical_transformer_peak_risk.yaml` — isolated defaults.
- `scripts/check_peak_risk_gate.py` — saved-result-only effectiveness, safety, and speed gate.
- `tests/test_matd3_hierarchical_transformer_peak_risk.py` — unit/integration tests.
- `tests/test_peak_risk_gate.py` — artifact gate tests.

**Modify**

- `scripts/benchmark_training_runtime.py` — compare arbitrary registered parent/candidate algorithms and emit machine-readable timing.
- `experiments/training_pipeline.py` — preserve nested `peak` and `runtime` diagnostics without inspecting internals.
- `tests/test_runtime_optimizations.py`, `tests/test_metrics_recording.py` — performance/diagnostic regressions.

### Task 1: Create the isolated plugin, configuration, and contract validator

**Files:**
- Create: package `algorithms/matd3_hierarchical_transformer_peak_risk/`
- Create: `configs/algorithms/matd3_hierarchical_transformer_peak_risk.yaml`
- Create: `tests/test_matd3_hierarchical_transformer_peak_risk.py`

**Interfaces:**
- Produces class `Matd3HierarchicalTransformerPeakRisk`.
- Produces `validate_peak_risk_config(config: Mapping[str, Any]) -> dict[str, Any]`.
- Produces plugin slug `matd3_hierarchical_transformer_peak_risk`.

- [ ] **Step 1: Scaffold the package**

Run:

```powershell
python scripts/create_algorithm_variant.py --name matd3_hierarchical_transformer_peak_risk
```

Replace the generated inheritance with the corrected predictive-DR parent; keep exactly one `ALGORITHM_SPEC`.

- [ ] **Step 2: Add failing registry/config tests**

```python
def test_peak_risk_plugin_is_isolated():
    spec = get_algorithm_spec("matd3_hierarchical_transformer_peak_risk")
    assert spec.slug == "matd3_hierarchical_transformer_peak_risk"
    cfg = yaml.safe_load(spec.default_config_path.read_text(encoding="utf-8"))
    assert cfg["algorithm"]["name"] == spec.slug
    assert "p2p_auxiliary" not in cfg["algorithm"]
```

Parameterize invalid `critic_update_interval`, learning rate, gradient limit,
hidden dimensions, actor weights, warmup/ramp values, `use_discount`, and
`conservative_twin_mode`.

- [ ] **Step 3: Run tests and confirm missing plugin/validator failures**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "plugin or config" -v`

- [ ] **Step 4: Implement strict validation and initial YAML**

Use the predictive parent YAML values unchanged, add the exact `peak_risk` and
`training_recovery` keys from Design Section 10, and reject unknown peak-risk
keys so misspellings cannot silently disable a feature.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "plugin or config" -v`

```powershell
git add algorithms/matd3_hierarchical_transformer_peak_risk configs/algorithms/matd3_hierarchical_transformer_peak_risk.yaml tests/test_matd3_hierarchical_transformer_peak_risk.py
git commit -m "feat: scaffold peak-risk Transformer variant"
```

### Task 2: Implement finite peak-risk features and the fifteenth actor token

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_peak_risk/peak_risk.py`
- Create: `algorithms/matd3_hierarchical_transformer_peak_risk/actor.py`
- Test: `tests/test_matd3_hierarchical_transformer_peak_risk.py`

**Interfaces:**
- Produces `PeakRiskFeatureExtractor(nn.Module)` with `forward(observation: Tensor) -> Tensor[B, 14]`.
- Produces `PeakRiskActor(PredictiveDRActor)` with inherited `forward_latent` and `map_latent_actions` behavior.

- [ ] **Step 1: Add failing feature tests**

For MG1/MG2/MG3 fixtures, assert shape `[batch, 14]`, finiteness, normalized time-to-peak in `[0, 1]`, and correct valid-forecast fraction. Set all forecast-valid flags to zero and assert finite fallback features plus validity zero. Perturb a future realized outcome field not listed in the current observation schema and prove features are unchanged.

- [ ] **Step 2: Implement named-schema vectorized extraction**

Resolve indices in `__init__`, register index tensors as nonpersistent buffers, and compute all 14 Design Section 5.2 features without Python loops over batch rows. Use forecast-valid masks in mean/max calculations and a deterministic current-value fallback when every forecast row is invalid.

- [ ] **Step 3: Add a failing 15-token layout test**

```python
tokens, mask = actor._build_tokens(obs)
assert tokens.shape == (batch_size, 15, actor.embed_dim)
assert mask.shape == (batch_size, 15)
```

- [ ] **Step 4: Implement token insertion without parent edits**

Call `super()._build_tokens(observation)`, encode peak features,
add a dedicated peak-type embedding, apply the inherited `token_norm`, insert
the peak token after the trade token, and insert a `False` padding column at the
same position. Disabled mode is implemented by constructing the unmodified
`PredictiveDRActor` in Task 4, not by constructing this class with a zero token.

Extend `branch_modules()` with `peak_encoder` so existing gradient diagnostics
can identify it. Keep all feasible and predictive action mapping inherited.

- [ ] **Step 5: Run actor tests and commit**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "feature or token or parity" -v`

```powershell
git add algorithms/matd3_hierarchical_transformer_peak_risk/peak_risk.py algorithms/matd3_hierarchical_transformer_peak_risk/actor.py tests/test_matd3_hierarchical_transformer_peak_risk.py
git commit -m "feat: add peak-risk Transformer token"
```

### Task 3: Implement the shared twin peak critic and exact max-return target

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_peak_risk/critic.py`
- Modify: `algorithms/matd3_hierarchical_transformer_peak_risk/peak_risk.py`
- Test: `tests/test_matd3_hierarchical_transformer_peak_risk.py`

**Interfaces:**
- Produces `TwinPeakCritic(global_state_dim: int, joint_action_dim: int, hidden_dims: tuple[int, ...])` with `forward` and `peak1`.
- Produces `system_actual_load_kw(plans: Sequence[Mapping[str, Tensor]]) -> Tensor[B, 1]`.
- Produces `max_return_target(current: Tensor, next1: Tensor, next2: Tensor, done: Tensor) -> Tensor`.

- [ ] **Step 1: Add failing physical-cost tests**

Project randomized valid states/actions with `JointActionProjector`; assert
`system_actual_load_kw(plans)` equals the explicit sum of all
`plan["actual_load_kw"]`, has shape `[B, 1]`, and remains differentiable with
respect to raw actions.

- [ ] **Step 2: Add exact Bellman-target tests**

```python
current = torch.tensor([[0.7], [0.7], [0.7]])
next1 = torch.tensor([[0.4], [0.8], [0.9]])
next2 = torch.tensor([[0.5], [0.9], [0.8]])
done = torch.tensor([[0.0], [0.0], [1.0]])
actual = max_return_target(current, next1, next2, done)
assert torch.equal(actual, torch.tensor([[0.7], [0.9], [0.7]]))
```

- [ ] **Step 3: Implement helpers and twin network**

Use independent MLPs and normalized scalar outputs. Do not accept `gamma` in
`max_return_target`. Validate identical tensor shapes and finite inputs; raise a
field-specific `FloatingPointError` for nonfinite values.

- [ ] **Step 4: Add parameter-budget test**

Construct parent `[128,128]` and peak `[64,64]` twin critics at the real global
state/joint-action dimensions. Assert peak parameters are at most 40% of one
parent twin critic.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "critic or target or actual_load" -v`

```powershell
git add algorithms/matd3_hierarchical_transformer_peak_risk/critic.py algorithms/matd3_hierarchical_transformer_peak_risk/peak_risk.py tests/test_matd3_hierarchical_transformer_peak_risk.py
git commit -m "feat: add max-return system peak critic"
```

### Task 4: Construct the algorithm and isolate inference/recovery checkpoints

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_peak_risk/algorithm.py`
- Modify: package `__init__.py`, `plugin.py`
- Test: `tests/test_matd3_hierarchical_transformer_peak_risk.py`

**Interfaces:**
- Produces `Matd3HierarchicalTransformerPeakRisk.from_env(...)`.
- Overrides recovery hooks `_extra_training_state` and `_load_extra_training_state` from the reliability plan.
- Stores `peak_risk_contract_version = "1.0"`.

- [ ] **Step 1: Add failing construction and disabled-path tests**

Assert exactly three peak actors, exactly one online twin peak critic, exactly
one target copy, one optimizer, frozen target parameters, and derived positive
`system_peak_scale_kw`. Assert no P2P auxiliary modules exist. With
`token_enabled: false`, assert every actor is the exact `PredictiveDRActor`
class, load identical parent weights, and verify bitwise-equal latent actions.

- [ ] **Step 2: Implement construction**

Call the predictive parent initializer, validate peak config, replace actors
with `PeakRiskActor`, rebuild actor optimizers, create the shared online/target
peak critics and optimizer, derive the physical scale from load normalizers plus
configured DR maximum electrical powers, and cache configuration fingerprints.

- [ ] **Step 3: Add strict checkpoint tests**

Round-trip a valid checkpoint. Then independently corrupt slug, peak contract,
parent contract, peak config fingerprint, normalization signature, projection
signature, and schemas. Every corruption must raise before loading weights.

- [ ] **Step 4: Implement inference checkpoint and recovery hooks**

Inference `save/load` includes actor/reward critic state for normal evaluation
and peak critic state for contract completeness. Recovery hooks additionally
include peak optimizer, target peak critic, peak update count, and ramp cursor.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "construction or checkpoint or recovery" -v`

```powershell
git add algorithms/matd3_hierarchical_transformer_peak_risk tests/test_matd3_hierarchical_transformer_peak_risk.py
git commit -m "feat: isolate peak-risk algorithm checkpoints"
```

### Task 5: Integrate an efficient peak update into MATD3 training

**Files:**
- Create: `algorithms/matd3_hierarchical_transformer_peak_risk/training.py`
- Modify: `algorithms/matd3_hierarchical_transformer_peak_risk/algorithm.py`
- Modify: `experiments/training_pipeline.py`
- Test: `tests/test_matd3_hierarchical_transformer_peak_risk.py`
- Test: `tests/test_metrics_recording.py`

**Interfaces:**
- Produces `update_peak_risk(self, batch_size: int) -> dict[str, Any]`.
- Produces `peak_actor_weight(update_step, start, ramp, final) -> float`.
- Emits nested `peak` and `runtime` metrics from the algorithm-neutral update result.

- [ ] **Step 1: Add failing projector-call parity tests**

Wrap parent and candidate projectors with counters, populate identical replay,
and run actor-not-due and actor-due updates. Assert equal call counts in each
case and exactly one replay `sample` call.

- [ ] **Step 2: Add peak critic update and actor-weight schedule tests**

Assert no peak update before its cadence, one shared update when due, target
network movement only when due, finite Smooth L1 loss, and exact update-count
increments. Patch next-peak outputs to prove the target uses `max`, not `min`
or `gamma`. Assert `peak_actor_weight` is zero before `actor_start_update`,
half the final weight halfway through the ramp, exactly final at the end, and
monotonic at every integer boundary.

- [ ] **Step 3: Add actor gradient-boundary tests**

On an actor-due step, assert peak critic parameter gradients remain absent,
peak critic parameters do not change, active actor peak-token/action parameters
receive finite nonzero gradients, and peer actor parameters are unchanged by
the active actor's backward pass.

- [ ] **Step 4: Implement the fused update**

Copy the parent update into the child package and make only these scoped
changes: obtain current peak cost from existing `plans`; query target peak heads
with existing `target_effective`; update the shared peak critic on its cadence;
query frozen peak heads with each existing actor `projected` tensor; add the
ramped risk term before the single actor backward call; reuse all parent
diagnostic scheduling and finite checks.

- [ ] **Step 5: Preserve nested diagnostics generically**

Extend `_diagnostic_means` to recursively retain algorithm-supplied `peak` and
`runtime` mappings without knowing network fields. Add a test with missing
delayed updates to ensure missing metrics stay `None`, not zero.

- [ ] **Step 6: Add strict disabled-mode update parity**

With token disabled, peak critic disabled, and actor weight zero, initialize
parent and child identically, restore identical replay/RNG state, run one update,
and assert identical actor/reward-critic parameters and update metrics after
excluding the empty peak namespace.

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/test_matd3_hierarchical_transformer_peak_risk.py tests/test_metrics_recording.py -v`

```powershell
git add algorithms/matd3_hierarchical_transformer_peak_risk experiments/training_pipeline.py tests/test_matd3_hierarchical_transformer_peak_risk.py tests/test_metrics_recording.py
git commit -m "feat: train peak risk with reused MATD3 projections"
```

### Task 6: Add runtime/learning-adequacy benchmarks and the saved-result gate

**Files:**
- Modify: `scripts/benchmark_training_runtime.py`
- Create: `scripts/check_peak_risk_gate.py`
- Modify: `tests/test_runtime_optimizations.py`
- Create: `tests/test_peak_risk_gate.py`

**Interfaces:**
- Runtime CLI accepts registered parent and candidate slugs and emits JSON with median update/wall-time/memory/projector counts.
- Gate CLI consumes candidate and corrected-parent run directories plus runtime JSON; it never reruns an environment.

- [ ] **Step 1: Add failing generic benchmark tests**

Assert the benchmark can resolve both algorithms, warm up before timing, run at
least 1,000 updates and three repetitions in production mode, separate actor-due
from actor-not-due timings, count projector/sample calls, and emit finite JSON.

- [ ] **Step 2: Implement generic benchmark comparison**

Keep worker processes isolated, seed both algorithms identically, populate
equivalent replay data once per worker, synchronize CUDA around timed regions,
and report median plus individual repetitions. Do not enable different
diagnostic schedules between algorithms.

- [ ] **Step 3: Add a learning-adequacy comparison mode**

Feed identical fixed replay batches and initialization to peak update intervals
1 and 2 for 2,000 opportunities. Emit held-out Smooth L1 error, Spearman rank
correlation, actor peak-gradient norm, twin gap, and divergence flag. The gate
accepts interval 2 only under Design Section 9.2 thresholds.

- [ ] **Step 4: Add failing gate artifact tests**

Create candidate/parent fixtures with identical date sets and explicit paper,
DR-shift, safety, profit, and runtime metrics. Test each effectiveness/safety/
speed failure independently. Remove or make nonfinite every required field in a
parameterized test and assert fail-closed behavior. Give parent/candidate
different date sets and assert rejection.

- [ ] **Step 5: Implement the saved-result-only gate**

Require manifest/artifact files and hashes, read exact daily peak reduction
columns from `dr_shift_metrics.csv`, compute mean/median/positive rate/p10/worst
day/rebound-to-shed, compare mean profit with a zero-safe percentage function,
apply inherited feasibility gates, and apply the 15% target/20% hard speed
limits. Write a JSON report containing every input, threshold, and pass state.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_runtime_optimizations.py tests/test_peak_risk_gate.py -v`

```powershell
git add scripts/benchmark_training_runtime.py scripts/check_peak_risk_gate.py tests/test_runtime_optimizations.py tests/test_peak_risk_gate.py
git commit -m "feat: gate peak-risk effectiveness and runtime"
```

### Task 7: Complete repository, smoke, performance, and experiment-readiness verification

**Files:**
- Modify only through the owning earlier task if verification exposes a defect.

**Interfaces:**
- Consumes all earlier interfaces; produces the verified training commands and gate reports.

- [ ] **Step 1: Run focused stage-three tests**

Run:

```powershell
pytest tests/test_matd3_hierarchical_transformer_peak_risk.py tests/test_peak_risk_gate.py tests/test_runtime_optimizations.py tests/test_metrics_recording.py -q
```

- [ ] **Step 2: Run the entire suite**

Run: `pytest -q`

- [ ] **Step 3: Run the microbenchmark before any full training**

Run the benchmark for corrected predictive DR versus peak risk on the intended
training device. Block the full run if median overhead exceeds 20%, projector
counts differ, memory increases by more than 15%, or parameter budget fails.

- [ ] **Step 4: Resolve critic cadence without formal-test data**

Run the learning-adequacy comparison. Keep interval 2 only if all four adequacy
conditions pass; otherwise change the production YAML to interval 1, rerun
focused tests and the speed benchmark, and record the decision in the benchmark
JSON.

- [ ] **Step 5: Run a complete three-episode pipeline smoke**

Confirm the plugin/YAML, best checkpoint formal test, MILP, Equal/Shapley
settlement, summaries, 15 PNG and 15 PDF files, working-run layout, peak
diagnostics, and resumable training state.

- [ ] **Step 6: Verify source isolation**

Run:

```powershell
git diff --exit-code -- algorithms/matd3
git status --short
```

Expected: canonical MATD3 has no changes; only intentional stage-three and
reliability files appear.

- [ ] **Step 7: Hand off full-run order**

Run the corrected predictive-DR seed-42 baseline first, then the peak-risk
seed-42 run with identical training/test/validation settings, then execute
`check_peak_risk_gate.py` against saved outputs and runtime JSON. Do not promote
either run to a release until the user reviews the gate report.
