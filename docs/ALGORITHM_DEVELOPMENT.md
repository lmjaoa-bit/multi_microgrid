# Algorithm Development Guide

## Current formal algorithms

The shared formal environment is `configs/cooperative.yaml`. The two current
algorithms are:

- `matd3_dispatch` (CLI alias `matd3`), displayed as MATD3.
- `matd3_hierarchical_transformer`, displayed as 普通 T in stored metadata.
  Plotting renders this as `Ordinary Transformer` when a CJK-capable font is
  unavailable; the run ID and metadata are not renamed.

Both algorithms use the same `single_dispatch_dr` physical action contract,
`peer_energy_summary` observations, and the same Reward V2 environment.

## Reward V2 contract

The formal reward version is `generalized_system_cost_v2`. For one complete
24-hour episode, the generalized system cost is the sum of:

1. external-grid import cost;
2. battery degradation cost;
3. renewable-curtailment cost;
4. grid-import carbon cost;
5. daily team-load peak cost;
6. industrial-mill unserved-service cost;
7. HVAC comfort degree-hour cost;
8. EV departure shortfall cost;
9. unmet-load cost.

The three agents receive the same team reward

`team_reward = -generalized_system_cost / reward_scale`.

Accounting profit remains a separate ledger. Equal/Shapley settlement and all
seven coalition values continue to use accounting profit; Reward V2 does not
replace those values.

The locked formal coefficients are:

| Quantity | Value |
|---|---:|
| Curtailment | 0.05 / kWh |
| Carbon | 0.10 / kg |
| Daily peak | 1.0 / kW |
| Mill unserved service | 100.0 / t |
| HVAC discomfort | 10.0 / (°C·h) |
| EV departure shortfall | 3.0 / kWh |
| Reward scale | 100.0 |

The calibration evidence SHA256 recorded by the formal configuration is:

`9d1461793ae112719dedccb2204b17814006f4b700ee1b67c009e2a68c7a562b`.

Do not tune these coefficients from the formal test split. Any future change
must return to a diagnostic `calibration_status`, use train/validation
evidence, and lock a new calibration-report hash before a formal run.

## Physical semantics

Reward V2 formal runs use:

- no external-grid export (`export_limit_kw=0` for all microgrids);
- renewable-load-first dispatch;
- no terminal SOC restoration target;
- MG1 instantaneous silo/production physics retained;
- MG2 `soft_comfort`: comfort is a measured service cost, not a future
  viability projection;
- MG3 `no_deadline_minimum`: the environment does not force future charging;
  departure shortfall is settled as a service cost.

The effective action/projector semantics are
`single_dispatch_dr_v4_instantaneous_physical`.

## Ordinary Transformer forecast context

The current ordinary Transformer Actor uses an 8-hour forecast-token window.
Each H8 forecast token includes local forecast context plus both peers'
`energy_balance` forecast channels. `include_peer_forecast=true` and
`peer_forecast_features=[energy_balance]`.

H8 refers only to the Actor's formal Transformer window. The shared observation
builder still keeps its wider future-horizon contract; H8 does not redefine the
episode or the environment forecast store.

The historical `matd3_forecast_context` 2×2 ablation is intentionally left
unchanged and must not be silently migrated to the current formal contract.

## Checkpoint selection

Formal Reward V2 validation uses
`physical_constrained_objective` with `team_reward_mean` as the objective.
Eligibility checks physical feasibility only:

- physical feasible-day rate;
- power/source/P2P balance residuals;
- device-bound violations;
- zero grid export;
- zero unmet energy.

HVAC comfort and EV service are soft Reward V2 objectives, so they are not
reintroduced as hard checkpoint filters.

Reward V2 checkpoints bind the reward/model contract, including gamma,
observation schemas, normalizers, action semantics, and (for the ordinary
Transformer) peer order, mask semantics and forecast-context metadata. A
mismatch fails before model state is mutated.

## Reference optimization and settlement

Each formal test day may run eight optimization solves from the same
materialized day/scenario:

- seven accounting-profit coalitions for Equal/Shapley;
- one full-coalition generalized-cost reference for Reward V2.

The generalized reference records solver status, incumbent objective, dual
bound, MIP gap, optimality flag, scenario fingerprint, initial-state
fingerprint, and an independently replayed trajectory cost. A feasible
incumbent is never presented as a lower bound.

## Reproducible smoke commands

From the repository root:

```powershell
python scripts/run_experiment.py --algorithm matd3 --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_dispatch.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer --config configs/cooperative.yaml --algorithm-config configs/smoke/matd3_hierarchical_transformer.yaml --episodes 3 --test-episodes 2 --validation-interval 1 --validation-days 2 --no-learning-curve --no-dashboard --device cpu --seed 42
```

Do not add `--no-milp` or `--no-validation` to formal acceptance smoke
runs. The 3-episode smoke verifies the pipeline and numerical contracts only;
it is not a convergence or performance experiment.

For a full training run, use the corresponding formal algorithm YAML rather
than the smoke YAML. This implementation does not automatically start a
5000-episode run.

## Saved-result regeneration

Figures are consumers of saved CSV/JSON outputs. To regenerate the canonical
figures without rerunning the environment, model, or MILP:

```powershell
python scripts/generate_results.py --run_dir <existing_run_directory>
```

A formal run must produce exactly 15 PNG and 15 PDF files under `figures/`.
Under Reward V2, the training plot labels the optimized quantity as
`Generalized Return`; accounting profit is shown separately and is not
relabeled as reward.

## Plan-Transformer experiment

Run the isolated H24 actor through the same experiment workflow:

```powershell
python scripts/run_experiment.py --algorithm matd3_plan_transformer
python scripts/run_experiment.py --algorithm matd3_plan_transformer --algorithm-config configs/smoke/matd3_plan_transformer.yaml --validation-days 2
python scripts/probe_plan_transformer.py --run <completed_working_run> --output <new_study_directory>
```

The current two ESS/DR intents execute through the existing physical projector.
The auxiliary head predicts SOC after each step and actual ESS discharge minus
charge; its 24 slots are supervision outputs. NOW cross-attends valid forecast
slots and two peer summaries through two layers. Forecast slot k refers to t+k;
plan slot j refers to dispatch t+j and resulting SOC at t+j+1.

The CEM teacher uses current state and frozen currently available predictions in
a sanitized copy of the environment. It runs the same physical dispatch and
generalized-cost calculation, retaining the best feasible candidate found within
its budget. This is approximate optimization without an optimality certificate.
Future carbon persists at its observed value. Unrevealed EV sessions use the
public cohort prior; realized future session samples and future episode rows
never enter labels. Teacher sampling has a separate RNG and does not change the
training environment. Inference and formal testing do not run the teacher.

Version one supports inference checkpoints and explicitly rejects training
resume: the teacher pool, cache, counters, and RNG are not recovery snapshots.
Auxiliary evidence lives in each run's `auxiliary_plan/`; additional plots live
in `figures/plan_diagnostics/`, alongside the unchanged 15 standard PNG/PDF pairs.
These plots read saved files only. Missing supervision fails training acceptance.

For the no-supervision ablation, copy this plugin's YAML to a temporary file and
set `algorithm.auxiliary_plan.lambda_plan`, `lambda_dyn`, and `lambda_first` to
zero. Keep the network, seed, environment and RL hyperparameters identical and
use `--algorithm-config` with the same slug. This disables all teacher capture.
Assess effects with paired multi-seed training/validation/test studies. A short
smoke run verifies execution only; attention heatmaps and controlled forecast
probes do not establish performance improvements or causal credit assignment.
