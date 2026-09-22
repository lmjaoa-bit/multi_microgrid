# Peak-Risk Transformer implementation handoff

**Date:** 2026-09-22  
**Base branch/commit:** `codex/milp-benchmark` @ `ef216fb`  
**Implementation branch:** `codex/peak-risk-transformer-20260922`  
**Current scope status:** stage-three Peak-Risk Transformer implementation is complete through unit/integration tests, benchmark-worker smoke, full one-episode shared pipeline smoke, and same-directory recovery verification. Formal long-run effectiveness and production runtime gates still require local/GPU evidence.

## 1. Completed prerequisite work

The prerequisite work was intentionally kept separate from the stage-three algorithm:

| Commit | Delivery |
|---|---|
| `01c410b` | Scale-aware P2P label consistency tolerance and strict normalization-signature loading |
| `2fbe45a` | Fail-closed saved-result P2P gates and saved-hourly HVAC late-power metric |
| `d5aed9f` | Optional recoverable-algorithm protocol, replay logical-state persistence, atomic/chunk helpers |
| `1b1cf58` | Full predictive-DR training-state persistence: online/target models, optimizers, replay, counters and RNG state |
| `9efc54d` | Shared pipeline run lifecycle and same-directory `--resume-run` support |

New runs now write `run_status.json` and transition through `RUNNING -> TRAINED -> EVALUATING -> COMPLETED`; caught failures produce `FAILED` and `training_error.json`. Recovery manifests are stored below `checkpoints/training_state/`. Configuration, algorithm, seed, schema, normalization and projection contracts are checked before restoration.

Resume is supported only for runs created with the new recovery state. The old episode-3454 P2P run cannot be resumed because its replay, optimizer and RNG state were never saved.

## 2. Verification evidence

- Focused reliability/P2P suites: `132 passed`.
- Full repository suite: `412 passed`, with three pre-existing warnings and no hidden xfails.
- Three-episode end-to-end predictive-DR smoke run completed training, validation, best-checkpoint selection, deterministic formal test, coalition MILP evaluation, Equal/Shapley settlement calculations and saved-result plotting.
- Smoke artifacts included `best_model.pt`, `final_model.pt`, recovery manifest, test summary, daily metrics, and 24 PNG plus 24 PDF figures.
- Same-directory resume was exercised after a completed one-episode run: no training row was duplicated, `resume_count` became 1, and evaluation completed again from the saved best checkpoint.
- `algorithms/matd3/` remains unchanged.

Windows note: an artificially deep scratch output path can exceed the legacy path-length limit when `checkpoints/training_state/` is appended. Normal repository output paths are valid; use a short output root for smoke experiments.

## 3. How to resume a new interrupted predictive-DR run

Start normally:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_predictive_dr `
  --device cpu
```

Resume into the exact same run directory with the same algorithm, configuration, seed and episode count:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_predictive_dr `
  --resume-run outputs/working_runs/matd3_hierarchical_transformer_predictive_dr/runs/<RUN_ID> `
  --device cpu
```

If the original command explicitly set `--episodes`, `--test-episodes`, `--output-root`, `--config`, or `--algorithm-config`, repeat those values on resume.

## 4. Stage-three implementation delivered

The isolated package `algorithms/matd3_hierarchical_transformer_peak_risk/`
now contains:

- a 15-token predictive actor with one actor-visible peak-risk token;
- exactly one shared twin peak critic;
- no-discount max-return target over three-microgrid simultaneous actual load;
- delayed actor regularization using the conservative max of the twin peak heads;
- strict checkpoint and recovery contracts including peak networks, optimizer,
  update count, observation/state schemas, normalizers and projection signature;
- fused training that reuses the parent's replay sample and projected actions;
- nested `peak` and `runtime` diagnostics in the shared training history;
- a saved-result-only effectiveness/safety gate and a controlled parent/candidate
  runtime benchmark mode.

No files under `algorithms/matd3/` were modified. The P2P-value auxiliary head
is not inherited.

Verification on GitHub Actions has exercised:

1. focused Peak-Risk tests;
2. parameter budget and exact max-return target;
3. token-disabled exact parent action path;
4. token/Critic-disabled full update parity with the predictive parent;
5. peak-critic cadence, actor-gradient path and recovery round-trip;
6. benchmark worker smoke for parent and candidate;
7. the full one-episode train -> validation -> best checkpoint -> formal test ->
   MILP -> settlement -> plotting pipeline;
8. same-directory resume with no duplicated training row;
9. the full repository regression suite.

The remaining evidence is intentionally not fabricated by CI smoke tests:

- a controlled production timing comparison with at least 1000 updates,
  3 repetitions and 100 end-to-end episodes per candidate;
- a full stage-three training run;
- saved-result comparison against the corrected predictive-DR parent on the
  exact same formal test dates;
- final Peak-Risk gate PASS on peak reduction, rebound, profit, feasibility and
  runtime overhead.

## 5. Commands for local production verification

Run the controlled runtime comparison first:

```powershell
python scripts/benchmark_training_runtime.py `
  --compare-parent matd3_hierarchical_transformer_predictive_dr `
  --compare-candidate matd3_hierarchical_transformer_peak_risk `
  --compare-device cuda `
  --compare-updates 1000 `
  --compare-repetitions 3 `
  --compare-episodes 100 `
  --output outputs/runtime_benchmarks/peak_risk_vs_predictive.json
```

Do not start the 5000-episode experiment if the measured median update overhead
exceeds 20%, projector/replay parity is false, or the other hard runtime budgets
fail.

Then train the candidate with its registered production configuration:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_peak_risk `
  --device cuda `
  --seed 42
```

If interrupted, resume into the same run directory and repeat the original
algorithm/config/seed/episode arguments:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_peak_risk `
  --resume-run outputs/working_runs/matd3_hierarchical_transformer_peak_risk/runs/<RUN_ID> `
  --device cuda `
  --seed 42
```

After both the corrected predictive parent and Peak-Risk candidate have complete
saved formal-test artifacts for the same dates, run the fail-closed gate:

```powershell
python scripts/check_peak_risk_gate.py `
  --candidate outputs/working_runs/matd3_hierarchical_transformer_peak_risk/runs/<PEAK_RUN> `
  --parent outputs/working_runs/matd3_hierarchical_transformer_predictive_dr/runs/<PARENT_RUN> `
  --runtime-json outputs/runtime_benchmarks/peak_risk_vs_predictive.json `
  --output outputs/runtime_benchmarks/peak_risk_gate.json
```

## 6. Final verification evidence

GitHub Actions run `35680046413` completed successfully against the production
code after the peak-critic budget correction.

- Focused Peak-Risk/gate/runtime/constraint tests: passed.
- Benchmark worker smoke for corrected predictive parent and Peak-Risk candidate: passed.
- Full repository regression: `447 passed, 4 skipped, 3 warnings`.
- Full one-episode shared pipeline smoke: passed.
- Same-directory recovery/resume smoke with no duplicated training row: passed.
- Measured Peak-Risk critic parameter fraction in the worker smoke:
  `0.3939371710` (39.39%), below the 40% hard limit.
- The latest branch commits after that successful code run are documentation-only;
  no algorithm/runtime/test code changed after the tested `ee56da9` code state.

## 2026-09-22 implementation reconciliation: peak-critic width

The design draft proposed `critic_hidden_dims: [64, 64]` while also making the
peak critic parameter budget a hard requirement of at most 40% of one parent
centralized twin critic. At the repository's real global-state and joint-action
dimensions, independent twin `[64,64]` peak networks contain 46,338 parameters
versus 109,058 for one parent twin critic, or 42.49%.

The production implementation therefore uses `[60,60]`, which retains the
approved two-hidden-layer independent-twin architecture while reducing the
ratio to 42,962 / 109,058 = 39.40%. The 40% budget is treated as the stricter
requirement because it is part of the training-overhead acceptance contract.
