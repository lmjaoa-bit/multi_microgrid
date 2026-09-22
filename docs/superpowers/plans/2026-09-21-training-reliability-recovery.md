# Training Reliability and Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the known P2P false failure, make saved-result gates fail closed, and make supported Transformer training runs observable and resumable without materially slowing training.

**Architecture:** Numerical and artifact fixes remain local to their current components. A new optional recovery protocol lets the shared pipeline request opaque, algorithm-owned training snapshots without inspecting network internals; predictive-DR descendants implement it through a focused recovery helper with incremental replay chunks and atomic manifests.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, pytest, YAML/JSON, existing plugin experiment pipeline.

**Spec:** `docs/superpowers/specs/2026-09-21-peak-risk-transformer-design.md`, especially Section 12.

## Global Constraints

- Do not modify `algorithms/matd3/` or its checkpoint/public behavior.
- Do not change environment physics, reward, action space, P2P clearing, settlement, or formal-test data.
- Recovery is optional; algorithms without the recovery protocol continue to run normally and reject `--resume-run` clearly.
- `latest_training_state` is distinct from `best_model.pt` and `final_model.pt`.
- Recovery persistence is incremental and uncompressed; the manifest is atomically replaced last.
- Required saved metrics must be present, numeric, and finite; no favorable fallback values are allowed.
- A completed implementation must pass `pytest -q` and a normal plus interrupted/resumed pipeline smoke test.
- Canonical MATD3 must have a zero diff.

## Review Focus

- Float32 trade values reconstructed with large normalizers must accept roundoff near `1.2e-5 kWh` but reject a real `1e-3 kWh` mismatch; Task 1 pins both cases.
- A gate artifact with a missing, boolean-as-number, string, NaN, or infinity field must fail with the field name; Task 2 pins all five classes.
- A crash between chunk rename and manifest replacement must leave the previous recovery point loadable; Task 4 simulates the interruption.
- A full ring buffer must restore logical order, insertion index, capacity, and RNG state; Task 3 compares post-restore samples.
- Resuming with a different config, schema, algorithm slug, or normalization signature must fail before changing algorithm state; Tasks 3 and 4 pin these mismatches.

---

## File structure

**Create**

- `algorithms/training_recovery.py` — serialization helpers, incremental replay chunks, hashes, atomic manifest commit.
- `experiments/training_recovery.py` — optional protocol detection and pipeline progress metadata validation.
- `tests/test_training_recovery.py` — replay, RNG, atomicity, compatibility, and deterministic continuation tests.
- `tests/test_training_pipeline_recovery.py` — CLI/run lifecycle/resume integration tests.

**Modify**

- `algorithms/base.py` — separate optional `RecoverableAlgorithmProtocol`.
- `algorithms/runtime_helpers.py` — explicit logical replay export/import helpers for `FastReplayBuffer`.
- `algorithms/matd3_hierarchical_transformer_predictive_dr/algorithm.py` — recovery implementation for the corrected parent and descendants.
- `algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py` — scale-aware numerical tolerance.
- `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py` — strict normalization-signature load validation.
- `evaluation/paper_artifacts.py` — saved-hourly HVAC late-power metric.
- `scripts/check_p2p_value_gate.py` — fail-closed required metric reads.
- `experiments/training_pipeline.py` — `--resume-run`, status/error files, recovery save/load calls, early `latest_run.json`.
- `tests/test_matd3_hierarchical_transformer_p2p_value.py`, `tests/test_p2p_value_gate.py`, `tests/test_paper_artifacts.py` — focused regressions.

### Task 1: Fix P2P numerical tolerance and checkpoint validation

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/p2p_auxiliary.py`
- Modify: `algorithms/matd3_hierarchical_transformer_p2p_value/algorithm.py`
- Test: `tests/test_matd3_hierarchical_transformer_p2p_value.py`

**Interfaces:**
- Produces: `physical_consistency_tolerance(values: torch.Tensor, scale_kwh: float, absolute_tolerance_kwh: float) -> torch.Tensor`.
- Preserves: `extract_p2p_labels(...)` return type and all existing actor/update interfaces.

- [ ] **Step 1: Add failing roundoff and material-mismatch tests**

```python
def test_scaled_float32_roundoff_does_not_abort_label_extraction():
    # Reproduce the observed 1.1920928955078125e-05 kWh residual.
    scale = 901.0
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched"]
    offer = torch.tensor([0.1234567], dtype=torch.float32)
    accepted = torch.tensor([0.1000000], dtype=torch.float32)
    unmatched = offer - accepted + torch.tensor([1.1920928955078125e-05 / scale])
    obs = torch.stack([offer, accepted, unmatched], dim=1)
    labels = extract_p2p_labels(
        obs, schema, trade_scale=scale, consistency_tolerance_kwh=1e-5
    )
    assert labels.consistency_residual_kwh.abs().max() < 2e-5

def test_material_p2p_inconsistency_still_fails():
    scale = 901.0
    schema = ["previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched"]
    obs = torch.tensor([[10.0 / scale, 4.0 / scale, 5.999 / scale]], dtype=torch.float32)
    with pytest.raises(ValueError, match="P2P consistency error"):
        extract_p2p_labels(
            obs, schema, trade_scale=scale, consistency_tolerance_kwh=1e-5
        )
```

- [ ] **Step 2: Run the focused tests and confirm the first one fails**

Run: `pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -k "roundoff or material_p2p" -v`

Expected: roundoff case raises the current fixed-tolerance `ValueError`; material mismatch fails as intended.

- [ ] **Step 3: Implement a scale- and dtype-aware bound**

```python
def physical_consistency_tolerance(values, scale_kwh, absolute_tolerance_kwh):
    eps = torch.finfo(values.dtype).eps
    magnitude = values.detach().abs().amax(dim=-1) * float(scale_kwh)
    roundoff = 8.0 * eps * torch.maximum(
        magnitude, torch.ones_like(magnitude) * float(scale_kwh)
    )
    return torch.maximum(
        roundoff,
        torch.full_like(roundoff, float(absolute_tolerance_kwh)),
    )
```

Compare each residual against its row tolerance. Keep all negative-value and accepted-greater-than-offer checks unchanged.

- [ ] **Step 4: Add strict normalization-signature load tests and implementation**

```python
checkpoint = torch.load(path, weights_only=False)
checkpoint["normalization_signature"] = "0" * 64
torch.save(checkpoint, path)
with pytest.raises(ValueError, match="normalization_signature"):
    algorithm.load(path)
```

In `load`, require the key and compare it with `self.normalization_signature` before loading any state dict.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_matd3_hierarchical_transformer_p2p_value.py -v`

Expected: all P2P algorithm tests pass.

- [ ] **Step 6: Commit**

```powershell
git add algorithms/matd3_hierarchical_transformer_p2p_value tests/test_matd3_hierarchical_transformer_p2p_value.py
git commit -m "fix: tolerate scaled P2P roundoff safely"
```

### Task 2: Produce the HVAC late-power metric and make P2P gates fail closed

**Files:**
- Modify: `evaluation/paper_artifacts.py`
- Modify: `scripts/check_p2p_value_gate.py`
- Test: `tests/test_paper_artifacts.py`
- Test: `tests/test_p2p_value_gate.py`

**Interfaces:**
- Produces paper metric: `hvac_late_power_p95_kw`.
- Produces helper: `_require_finite_number(mapping, key, source) -> float` in the gate script.

- [ ] **Step 1: Add a failing saved-hourly metric test**

```python
def test_paper_metrics_include_hvac_late_power_p95():
    hourly = _hourly_frame()
    hourly["mg2_hvac_effective_signed_power_kw"] = 2.0
    hourly.loc[hourly["hour"].isin([22, 23]), "mg2_hvac_effective_signed_power_kw"] = [4.0, -6.0] * day_count
    metrics = calculate_paper_metrics(hourly, daily, _environment_config())
    values = metrics.set_index("metric_id")["value"]
    assert values["hvac_late_power_p95_kw"] == pytest.approx(6.0, rel=0.1)
```

Define late hours as the final two episode hours using `hour >= episode_length - 2`; use absolute signed HVAC electrical power and `np.quantile(..., 0.95)`.

- [ ] **Step 2: Run the metric test and confirm it fails**

Run: `pytest tests/test_paper_artifacts.py::test_paper_metrics_include_hvac_late_power_p95 -v`

Expected: the metric ID is absent.

- [ ] **Step 3: Implement the metric from saved data**

Require `mg2_hvac_effective_signed_power_kw`, select the last two hours of every saved day, reject an empty/nonfinite series, and append:

```python
_metric(
    "hvac_late_power_p95_kw", "MG2 HVAC", "Late HVAC absolute power p95",
    "kW", float(np.quantile(late_power.abs(), 0.95)), "lower",
)
```

- [ ] **Step 4: Add fail-closed gate tests**

Parameterize removal/replacement of `constraint_eligible`, `hvac_late_power_p95_kw`, and another required safety field with `None`, `True`, `"5"`, `nan`, and `inf`. Assert `evaluate_gate(...)` raises `ValueError` containing the exact field name.

- [ ] **Step 5: Implement strict readers and remove favorable defaults**

```python
def _require_finite_number(mapping, key, source):
    if key not in mapping or isinstance(mapping[key], bool):
        raise ValueError(f"{source} requires numeric {key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"{source} has non-finite {key}")
    return value
```

Require `constraint_eligible` to exist and be a real boolean. Replace `.get(..., 0.0)` for all required safety/economic metrics with strict reads.

- [ ] **Step 6: Run focused tests and commit**

Run: `pytest tests/test_paper_artifacts.py tests/test_p2p_value_gate.py -v`

```powershell
git add evaluation/paper_artifacts.py scripts/check_p2p_value_gate.py tests/test_paper_artifacts.py tests/test_p2p_value_gate.py
git commit -m "fix: make P2P result gates fail closed"
```

### Task 3: Define recovery protocol and exact FastReplayBuffer state restoration

**Files:**
- Modify: `algorithms/base.py`
- Modify: `algorithms/runtime_helpers.py`
- Create: `algorithms/training_recovery.py`
- Create: `tests/test_training_recovery.py`

**Interfaces:**
- Produces optional protocol:
  `save_training_state(directory: Path, progress: Mapping[str, Any]) -> Mapping[str, Any]` and
  `load_training_state(directory: Path) -> dict[str, Any]`.
- Produces replay methods: `logical_transitions() -> list[Transition]`, `state_metadata() -> dict[str, Any]`, and `restore_logical(transitions, metadata) -> None`.

- [ ] **Step 1: Write failing replay-order and RNG tests**

Fill a capacity-5 buffer with eight numbered transitions, save state, restore into a fresh buffer, and assert logical IDs are `[3, 4, 5, 6, 7]`, `_next_index` is identical, and the next three sampled batches have identical transition IDs.

- [ ] **Step 2: Run and confirm missing interfaces**

Run: `pytest tests/test_training_recovery.py -k replay -v`

Expected: missing logical export/restore methods.

- [ ] **Step 3: Implement logical replay state**

```python
def logical_transitions(self):
    return [self._logical_get(i) for i in range(len(self))]

def state_metadata(self):
    return {
        "capacity": self.capacity,
        "size": len(self.storage),
        "next_index": self._next_index,
        "total_added": self._total_added,
        "rng_state": copy.deepcopy(self.rng.bit_generator.state),
    }
```

Increment `_total_added` in `add`. `restore_logical` validates capacity/size,
rebuilds physical storage so the same logical ordering is preserved, restores
`_next_index`, `_total_added`, and the NumPy generator state.

- [ ] **Step 4: Add and test the optional protocol**

Define a separate `@runtime_checkable RecoverableAlgorithmProtocol`; do not add methods to `MultiAgentAlgorithm`, which would break existing plugins.

Test that a dummy recoverable algorithm satisfies the protocol and a normal MATD3-shaped dummy does not.

- [ ] **Step 5: Implement atomic helpers and fingerprints**

In `algorithms/training_recovery.py`, provide `_atomic_torch_save`, `_atomic_write_json`, SHA-256 hashing, transition chunk encoding/decoding, and `validate_fingerprints(expected, actual)`. All temporary files live inside the target recovery directory and are renamed with `Path.replace`.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_training_recovery.py -v`

```powershell
git add algorithms/base.py algorithms/runtime_helpers.py algorithms/training_recovery.py tests/test_training_recovery.py
git commit -m "feat: add optional training recovery protocol"
```

### Task 4: Add incremental algorithm snapshots and deterministic restore

**Files:**
- Modify: `algorithms/matd3_hierarchical_transformer_predictive_dr/algorithm.py`
- Modify: `algorithms/training_recovery.py`
- Test: `tests/test_training_recovery.py`

**Interfaces:**
- Implements the Task 3 protocol on `Matd3HierarchicalTransformerPredictiveDr`.
- Defines protected hooks `_extra_training_state() -> dict` and `_load_extra_training_state(state) -> None` for descendants such as peak-risk.

- [ ] **Step 1: Add a failing interrupted-versus-uninterrupted CPU test**

Train a small deterministic fixture for six updates. In the comparison path, train three updates, persist, construct a new algorithm, restore, and train three more. Assert equal sampled transition IDs, equal update counter, equal actor/critic tensors, and equal next selected noisy action.

- [ ] **Step 2: Add compatibility failure tests**

Modify one manifest field at a time for slug, config hash, state schema, projection signature, and normalization signature. Assert restore raises before any model parameter changes.

- [ ] **Step 3: Implement complete algorithm-owned state**

Persist actors/targets, reward critics/targets, actor/critic optimizers, update step, algorithm RNG, Torch CPU RNG, CUDA RNG list when present, replay metadata/chunks, and caller-supplied progress. Include the immutable fingerprints from Step 2.

- [ ] **Step 4: Implement incremental commit ordering**

Write only replay transitions added since the previous committed monotonic
`total_added` sequence number. Each chunk records a half-open sequence range
`[first_sequence_id, last_sequence_id)`. Hash each chunk. Rename chunk and state
payload first, then atomically replace `manifest.json`. On restore, load only
files named by the last valid manifest, verify contiguous ranges and hashes,
then retain the newest `capacity` transitions. This explicitly handles ring
overwrite without depending on physical list indices.

- [ ] **Step 5: Simulate an orphan chunk**

Create a valid snapshot, write an additional valid-looking chunk without updating the manifest, then restore. Assert the orphan is ignored and the previous state resumes exactly.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_training_recovery.py -v`

```powershell
git add algorithms/training_recovery.py algorithms/matd3_hierarchical_transformer_predictive_dr/algorithm.py tests/test_training_recovery.py
git commit -m "feat: persist resumable predictive DR training state"
```

### Task 5: Integrate run lifecycle and resume into the shared pipeline

**Files:**
- Create: `experiments/training_recovery.py`
- Modify: `experiments/training_pipeline.py`
- Create: `tests/test_training_pipeline_recovery.py`

**Interfaces:**
- CLI: `--resume-run PATH`.
- Artifacts: `run_status.json`, `training_error.json`, `checkpoints/training_state/manifest.json`.
- Status writer: `write_run_status(run_dir, status, **details) -> None`.

- [ ] **Step 1: Add parser and unsupported-algorithm tests**

Assert `build_parser().parse_args(["--resume-run", path])` returns an absolute-capable `Path`. Assert a nonrecoverable algorithm produces a clear `ValueError` before creating a new run.

- [ ] **Step 2: Add status transition tests**

Mock training/evaluation and assert the sequence is `RUNNING`, `TRAINED`, `EVALUATING`, `COMPLETED`. Force an exception and assert `FAILED` plus `training_error.json` with exception type, message, traceback, last episode/update, and durable recovery point.

- [ ] **Step 3: Implement early run registration and lifecycle writes**

Create the run directory, write config snapshot and metadata, immediately update `latest_run.json`, then write `RUNNING`. Use atomic JSON replacement for every lifecycle artifact.

- [ ] **Step 4: Implement resume validation and cursor restoration**

Resolve `--resume-run`, require it to lie under the selected algorithm's working `runs` directory, read its config snapshot, reject conflicting CLI algorithm/config/seed arguments, construct the algorithm, load recovery state, restore history/validation selector/counters, and continue at `completed_episode + 1` in the same run directory.

- [ ] **Step 5: Save durable state at configured episode boundaries**

After the episode history and validation state have been flushed, call `save_training_state`. Do not overwrite best/final inference checkpoints. Record the returned manifest path and episode in `run_status.json`.

- [ ] **Step 6: Run integration tests and commit**

Run: `pytest tests/test_training_pipeline_recovery.py tests/test_result_lifecycle.py -v`

```powershell
git add experiments/training_recovery.py experiments/training_pipeline.py tests/test_training_pipeline_recovery.py
git commit -m "feat: resume interrupted Transformer training runs"
```

### Task 6: Verify reliability, speed, and repository isolation

**Files:**
- Modify only if a verification failure identifies a scoped defect in Tasks 1-5.

**Interfaces:**
- Consumes all prior task interfaces; produces no new production API.

- [ ] **Step 1: Run focused regression suites**

Run:

```powershell
pytest tests/test_matd3_hierarchical_transformer_p2p_value.py tests/test_p2p_value_gate.py tests/test_paper_artifacts.py tests/test_training_recovery.py tests/test_training_pipeline_recovery.py -q
```

- [ ] **Step 2: Run the full suite**

Run: `pytest -q`

Expected: all tests pass with no xfail added to hide a regression.

- [ ] **Step 3: Run normal and resumed smoke experiments**

Run a 3-episode recoverable predictive-DR smoke, interrupt a separate run after a committed recovery point, resume it with `--resume-run`, and confirm formal test, MILP, settlement, summaries, and 15 PNG plus 15 PDF figures complete.

- [ ] **Step 4: Measure snapshot overhead**

Compare identical 100-episode runs with recovery enabled and disabled after warm-up. Record wall time in the implementation notes; snapshot overhead must be at most 3%.

- [ ] **Step 5: Confirm baseline integrity**

Run: `git diff --exit-code -- algorithms/matd3`

Expected: exit code 0.

- [ ] **Step 6: Close the verification gate**

If every command passes, make no additional commit. If a command fails, return
to the task that owns the failing behavior, add a regression test there, fix it,
rerun Steps 1-5, and use that task's commit message with a `fix:` prefix.
