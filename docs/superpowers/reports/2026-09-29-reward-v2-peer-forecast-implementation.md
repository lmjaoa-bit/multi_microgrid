# Reward V2 + Ordinary T Peer Forecast Implementation Report

Date: 2026-09-29  
Branch: `代号：噜噜`  
Implementation base: `d085ff71be9ab560854ad5798272e51c2496f2f4`  
Last code commit covered by final end-to-end smoke: `23943b85344c2769da22f2b4e1ff7b62a368d969`  
Documentation-only commits after that smoke: `docs/ALGORITHM_DEVELOPMENT.md` and this report.

## 1. Outcome

The 2026-09-29 Reward V2 + peer-forecast design and execution plan has been implemented.

The formal shared environment now uses:

- `generalized_system_cost_v2`;
- zero external-grid export;
- renewable-load-first dispatch;
- no terminal SOC restoration target;
- MG1 instantaneous silo/production feasibility retained;
- MG2 soft comfort semantics with no future HVAC viability projection in the formal path;
- MG3 no-deadline-minimum semantics, with departure shortfall settled as a service cost;
- a shared team reward derived from generalized system cost;
- current team-load peak and current grid-carbon factor in actor/critic public state;
- physical-only checkpoint eligibility;
- `gamma=1.0` for the complete 24-hour finite-day objective.

The ordinary Transformer formally uses:

- 8-hour forecast token window;
- both peers' future `energy_balance` channels;
- shared forecast-valid masking;
- explicit forecast-context version/checkpoint metadata;
- no runtime dependency on the historical `matd3_forecast_context` ablation plugin.

## 2. Locked Reward V2 coefficients

The locked formal coefficients in `configs/cooperative.yaml` are:

| Component | Coefficient |
|---|---:|
| Renewable curtailment | 0.05 / kWh |
| Grid carbon | 0.10 / kg |
| Daily team-load peak | 1.0 / kW |
| MG1 unserved downstream service | 100.0 / t |
| MG2 comfort violation | 10.0 / (°C·h) |
| MG3 EV departure shortfall | 3.0 / kWh |
| Unmet load | existing 20.0 / kWh |
| Reward scale | 100.0 |

The exact calibration evidence bound into the formal Reward V2 contract is:

`9d1461793ae112719dedccb2204b17814006f4b700ee1b67c009e2a68c7a562b`

A locked Reward V2 configuration now requires a valid 64-character calibration-report SHA256.

## 3. Calibration evidence

Calibration used train/validation natural days only; the formal test split was not used to tune coefficients.

The final evidence bundle used 8 train days and 8 validation days with:

- device-specific idle policy;
- current-state service-priority policy;
- a 3-episode short MATD3 diagnostic policy;
- seed 42.

Key validation evidence:

| Metric | Idle | Service-priority |
|---|---:|---:|
| Generalized system cost mean | 2056.35 | 1228.85 |
| Team accounting profit mean | -1505.21 | -880.00 |
| Comfort violation mean | 23.46 °C·h/day | ~0 °C·h/day |
| EV departure shortfall mean | 238.25 kWh/day | ~0 kWh/day |
| Curtailment mean | 2625.90 kWh/day | 2460.60 kWh/day |
| Daily team-load peak mean | 249.69 kW | 272.08 kW |

The service-priority policy deliberately uses a one-step minimum-power thermostat and does not invoke the removed HVAC future-viability LP.

For EV service calibration, the selected validation days had:

- maximum observed buy price: 0.95 / kWh;
- maximum observed carbon factor: 0.7 kg/kWh;
- EV charging efficiency: 0.93;
- worst-case one-hour marginal energy + carbon + peak cost:
  approximately 2.172 per battery-side kWh.

The locked EV shortfall cost of 3.0 / kWh is therefore above that diagnostic worst-case marginal service cost.

## 4. Short numerical stability evidence

The 3-episode Reward V2 MATD3 diagnostic used `gamma=1.0`.

Observed finite ranges:

- Critic loss: 1.178–5.054
- Actor loss: 0.305–1.039
- Critic gradient norm: 3.866–7.670
- Actor gradient norm: 0.00861–0.01840
- Q1 mean: -1.059 to -0.273
- Q2 mean: -1.082 to -0.325
- target-Q mean: -1.565 to -1.058
- TD-error mean: -1.176 to 0.043
- TD-error std: 0.651–0.805

No non-finite values or diagnostic explosion threshold violations were observed.

This is a numerical smoke result only; it is not convergence or policy-performance evidence.

## 5. Reward and accounting separation

Reward V2 does not overwrite accounting profit.

The environment and outputs now keep separate quantities for:

- accounting/economic profit;
- raw Reward V2 physical/service components;
- generalized system cost;
- team reward.

P2P transfer payments remain internal accounting transfers and are not added as a team reward bonus.

Power balance is still an environment/settlement invariant, not a reward term.

## 6. Physical/projection changes

### HVAC

The formal HVAC path:

- keeps only physical heating/cooling power bounds;
- no longer calls the future HiGHS viability LP;
- no longer clips the action to one-step comfort feasibility;
- evolves the 2R2C thermal state normally;
- records boundary-sampled comfort violation in °C·h.

### EV

The formal EV path:

- preserves connection/availability, remaining-energy and charger-power bounds;
- removes the future reachability/deadline minimum charging lower bound;
- allows zero charging while connected;
- settles unmet departure demand as an explicit Reward V2 service cost.

### MG1

The current instantaneous silo bound remains in place because it is an immediate physical/operational bound, not a future reachability optimizer.

## 7. Ordinary Transformer peer forecast

The formal ordinary Transformer Actor now uses 8-dimensional forecast features per future token:

1. local base net load;
2. buy price;
3. temperature;
4. valid flag;
5. future hour sine;
6. future hour cosine;
7. peer 0 future energy balance;
8. peer 1 future energy balance.

Token count remains 14:

- 4 local semantic tokens;
- 2 current peer tokens;
- 8 forecast tokens.

Invalid forecast steps are zeroed before encoding, including the peer future channels.

The formal version is recorded as:

`actor_forecast_context_version = peer_balance_h8_v2`

The archived 2×2 forecast-context ablation implementation/configuration is unchanged.

## 8. Checkpoint contract and selection

Formal Reward V2 checkpoint selection now uses:

- mode: `physical_constrained_objective`;
- objective: `team_reward_mean`.

Hard eligibility contains only physical feasibility:

- physical feasible-day rate;
- power balance;
- source balance;
- P2P balance;
- device bounds;
- zero grid export;
- zero unmet energy.

HVAC comfort and EV completion are intentionally not reintroduced as 100% hard checkpoint filters.

Reward V2 checkpoints bind and validate, before network mutation:

- reward semantics and coefficients;
- calibration evidence hash;
- gamma;
- action semantics;
- observation/state schemas;
- normalizers;
- peer order;
- forecast context version;
- peer forecast features;
- mask semantics.

Old 6-dimensional ordinary-T checkpoints do not silently load into the new 8-dimensional forecast encoder.

## 9. Generalized MILP reference vs settlement MILP

The seven Equal/Shapley coalition values still solve the accounting-profit objective.

Reward V2 adds one separate full-coalition generalized-cost reference for each test day. The eight solves reuse the same materialized day/scenario.

The generalized reference includes:

- zero grid export;
- grid import cost;
- ESS degradation;
- curtailment;
- carbon;
- daily load peak;
- soft HVAC comfort slack;
- EV departure shortfall.

It records:

- solver status;
- feasible incumbent objective;
- dual bound when available;
- MIP gap;
- optimality flag;
- scenario fingerprint;
- initial-state fingerprint;
- independently replayed trajectory cost.

A feasible incumbent is not relabeled as a mathematical lower bound.

## 10. Output and plotting contract

Formal outputs now distinguish:

- Generalized Return;
- Generalized System Cost;
- Accounting Profit.

The training plot no longer labels Reward V2 return as profit.

Saved-result figure regeneration uses:

`python scripts/generate_results.py --run_dir <run_directory>`

and does not execute the environment, policy or MILP again.

The canonical result contract remains exactly:

- 15 PNG figures;
- 15 PDF figures.

For CI environments without CJK fonts, the stored method name remains unchanged, while plots render `普通 T` as the font-safe label `Ordinary Transformer`.

## 11. Verification

### DR/full regression workflow

Successful workflow: `DR environment validation`, run `36524790800`.

Results:

- targeted DR/compatibility suite: **67 passed**, 2 warnings;
- full regression suite: **431 passed**, 4 warnings.

### Final two-algorithm end-to-end smoke

Successful workflow: `Renewable-first two-algorithm smoke`, run `36526687422`, against code commit `23943b85344c2769da22f2b4e1ff7b62a368d969`.

Both MATD3 and ordinary Transformer completed:

- 3 training episodes;
- validation with the new Reward V2 selection contract;
- 2 formal test days;
- seven accounting coalition solves per day;
- one generalized reference solve per day;
- checkpoint save/load validation;
- best-checkpoint evaluation;
- config/checkpoint hashing;
- Equal/Shapley settlement;
- saved-result-only figure regeneration;
- smoke artifact upload.

The final smoke explicitly verified:

- `generalized_system_cost_v2`;
- zero export;
- action semantics `single_dispatch_dr_v4_instantaneous_physical`;
- generalized reference solve/success counts;
- best-model SHA256 used for evaluation;
- environment and algorithm config SHA256 values;
- ordinary Transformer H8 + peer `energy_balance`;
- no protected historical/canonical path changes.

### Protected-path audit

Comparison from implementation base `d085ff71be9ab560854ad5798272e51c2496f2f4` found no changes in:

- `algorithms/matd3/`;
- `configs/algorithms/matd3.yaml`;
- `algorithms/matd3_forecast_context/`;
- `configs/algorithms/matd3_forecast_context.yaml`;
- retained release/output directories.

## 12. Known benign warnings

Generalized-reference CSVs are a union of heterogeneous MG-specific columns.
For example, MG1 rows do not have HVAC or EV fields. Pandas therefore writes
those structurally inapplicable NaN fields as empty CSV cells and emits a
runtime warning. The generalized objective replay is NaN-safe and the smoke
workflow succeeds; these empty cells are not numerical failures in the solved
trajectory.

## 13. Explicit non-actions

This implementation did **not**:

- start a 5000-episode formal training run;
- add a P2P-volume reward;
- switch the ordinary Transformer to H24;
- modify the canonical `algorithms/matd3/` baseline;
- modify the archived forecast-context ablation;
- restore ESS terminal SOC constraints;
- use the formal test split to tune Reward V2 coefficients.

The repository is ready for a new formal MATD3 / ordinary-T long training run
under the locked Reward V2 contract.
