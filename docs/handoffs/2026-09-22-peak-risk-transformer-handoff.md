# Peak-Risk Transformer implementation handoff

**Date:** 2026-09-22  
**Branch:** `codex/milp-benchmark`  
**Current scope status:** reliability prerequisite is implemented; peak-risk Transformer production code has **not** been started.

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

## 4. Stage-three work deliberately not implemented

No files were created under `algorithms/matd3_hierarchical_transformer_peak_risk/`; no peak-risk YAML, critic, actor token, training update, gate, or runtime benchmark change exists yet.

The approved design and executable task plan are:

- `docs/superpowers/specs/2026-09-21-peak-risk-transformer-design.md`
- `docs/superpowers/plans/2026-09-21-peak-risk-transformer.md`

The next implementer should execute the plan in task order. In particular:

1. use the corrected `matd3_hierarchical_transformer_predictive_dr` as parent;
2. scaffold the isolated slug `matd3_hierarchical_transformer_peak_risk`;
3. do not inherit the P2P-value auxiliary algorithm;
4. add exactly one actor peak-risk token and one shared lightweight twin peak critic;
5. preserve the raw action space, feasible/predictive DR mapping, reward, environment, settlement and canonical MATD3 baseline;
6. preserve exact parent behavior when the token and peak actor weight are disabled;
7. reuse the parent's projected actions and replay batch—no extra projector call, replay sample, rollout, MILP solve or test day;
8. include peak modules and optimizer state in both strict checkpoint validation and the new recovery hooks;
9. enforce the saved-result effectiveness/safety gate and the specified runtime overhead gate before a long training run.

## 5. First commands for the next implementer

Read the repository instructions and both documents above, then begin Task 1 with tests first:

```powershell
python scripts/create_algorithm_variant.py --name matd3_hierarchical_transformer_peak_risk
pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -k "plugin or config" -v
```

Do not launch a long experiment until every plan task, `pytest -q`, the short full pipeline smoke test, baseline-integrity check, and runtime gate pass.

