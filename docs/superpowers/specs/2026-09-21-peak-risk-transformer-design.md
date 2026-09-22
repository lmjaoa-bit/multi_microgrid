# Peak-Risk Transformer and Training Reliability Design

**Date:** 2026-09-21

**Status:** Approved by user on 2026-09-22

**Target algorithm slug:** `matd3_hierarchical_transformer_peak_risk`

**Parent algorithm:** corrected `matd3_hierarchical_transformer_predictive_dr`

**Initial experiment mode:** stage-three peak-shaving ablation, seed 42

## 1. Decision summary

The third-stage algorithm uses the approved **Option A** architecture:

1. add one peak-risk semantic token to each decentralized Transformer actor;
2. add one system-wide centralized twin peak critic;
3. train the peak critic with a max-return Bellman target representing the
   maximum three-microgrid actual load remaining in the episode;
4. add a small, ramped peak-risk term to the delayed actor objective;
5. preserve the shared reward, environment, raw action space, physical action
   projection, evaluation definitions, and canonical MATD3 baseline;
6. do not inherit the stage-two P2P-value auxiliary head.

Training speed is a first-class acceptance requirement. The implementation must
reuse tensors already produced by the parent update, use only one small shared
peak critic, skip unnecessary peak updates, and keep measured end-to-end
training overhead within the budgets in Section 13.

Reliability fixes are a separate prerequisite workstream. They are delivered in
the same overall implementation plan but are not represented as part of the
stage-three algorithm improvement.

## 2. Motivation and precise objective

The current Transformer can shift flexible load away from some expensive or
high-load hours, but it does not explicitly optimize the daily simultaneous
system peak. Existing saved results show that shifted energy can return in a
concentrated rebound period. The result can improve local scheduling while
making the daily three-microgrid maximum actual load worse.

The primary metric is the same system actual-load peak used by the existing
`dr_shift_metrics` artifact:

```text
system_actual_load_kw[t]
  = actual_load_kw[mg1, t]
  + actual_load_kw[mg2, t]
  + actual_load_kw[mg3, t]

daily_system_peak_kw
  = max_t system_actual_load_kw[t]
```

Grid-import peak is retained as a secondary diagnostic. It must not replace the
primary metric, because ESS dispatch, renewable production, and P2P exchange
can change grid import without changing actual consumption.

The stage-three learning objective is:

> Preserve the corrected feasible and predictive DR behavior while teaching
> each actor how its current action contributes to the maximum system actual
> load expected over the remainder of the episode.

## 3. Scope and non-goals

### 3.1 In scope

- an isolated stage-three algorithm package and YAML configuration;
- one new actor-side peak-risk token;
- one shared centralized twin peak critic and target copy;
- a max-return Bellman target with no reward modification;
- a ramped peak penalty in delayed actor updates;
- peak-specific diagnostics, checkpoint isolation, and saved-result gates;
- speed benchmarks and an explicit maximum training-overhead gate;
- prerequisite fixes for P2P numerical tolerance, fail-closed gates,
  checkpoint validation, run failure reporting, and resumable training.

### 3.2 Explicit non-goals

This work must not:

- modify `algorithms/matd3/` or its checkpoint contract;
- change the environment reward, tariff, physical model, P2P clearing, or
  settlement;
- change the raw actor action space `[ESS intent, DR intent]`;
- add an independent P2P offer action;
- inherit or silently combine the P2P-value auxiliary head;
- use formal-test episodes for gradient updates, checkpoint selection, or
  hyperparameter tuning;
- add CVaR, quantile critics, adaptive Lagrange multipliers, or multiple peak
  horizons in the first stage-three version;
- claim that the earlier episode-3454 P2P run can be resumed, because it did not
  save optimizer, replay, and random-number-generator state.

## 4. Isolation and inheritance

Create:

```text
algorithms/matd3_hierarchical_transformer_peak_risk/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py
├── critic.py
├── peak_risk.py
└── training.py

configs/algorithms/matd3_hierarchical_transformer_peak_risk.yaml
scripts/check_peak_risk_gate.py
```

The inheritance path is:

```text
matd3_hierarchical_transformer
  -> matd3_hierarchical_transformer_feasible_dr
  -> corrected matd3_hierarchical_transformer_predictive_dr
  -> matd3_hierarchical_transformer_peak_risk
```

The new plugin exposes exactly one `ALGORITHM_SPEC` and continues to use:

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_peak_risk
```

Results are written only below:

```text
outputs/working_runs/matd3_hierarchical_transformer_peak_risk/runs/<run_id>/
```

The checkpoint must store and strictly validate:

- `algorithm_slug`;
- a new immutable peak-risk contract version;
- parent predictive-DR contract version;
- action and projection semantics;
- projection signature;
- observation and global-state schemas;
- normalization signature;
- complete peak-risk configuration fingerprint.

## 5. Peak-risk actor token

### 5.1 Token count and policy path

The child actor adds one token to the existing 14-token actor, resulting in 15
semantic tokens. The raw observation vector is unchanged. The token is derived
only from fields already available to the decentralized actor at decision time.

The token is inserted before the Transformer stack and receives its own token
type embedding. The ESS and DR readout heads keep their inherited definitions.
The new token may influence both actions through attention, but it must never
write an action directly or bypass feasible DR mapping.

When `peak_risk.token_enabled: false` and the actor-risk weight is zero, the
child must instantiate the exact parent actor path rather than inserting a zero
token. This is required for a strict zero-feature parent-equivalence test; a
zero-valued extra token is not equivalent because it still changes attention.

### 5.2 Token features

`peak_risk.py` provides a deterministic, vectorized feature extractor. The
first contract contains these dimensionless or explicitly normalized features:

1. current local net-load pressure;
2. current peer-adjusted system-demand proxy;
3. previous system grid-buy demand;
4. mean forecast net load over the available eight-hour horizon;
5. maximum forecast net load over that horizon;
6. forecast rise from the current value to the maximum;
7. normalized time until the forecast maximum;
8. local feasible DR lower bound;
9. local feasible DR upper bound;
10. remaining DR recovery or completion pressure;
11. local ESS discharge headroom;
12. aggregate peer deficit and surplus pressure;
13. valid-forecast fraction;
14. a categorical device/deadline urgency feature appropriate to mill, HVAC,
    or EV operation.

The extractor must use named schema lookup rather than positional constants.
Missing required fields are configuration errors. Invalid optional forecast
values use the same documented fallback as the parent predictor and set the
validity feature accordingly. Every returned feature must be finite.

### 5.3 Token encoder

Use a small encoder consistent with the existing actor:

```text
Linear(peak_feature_dim, hidden_dim)
LayerNorm(hidden_dim)
GELU
Linear(hidden_dim, embed_dim)
```

No separate Transformer layer is added. The existing two-layer Transformer is
used with a sequence length of 15.

## 6. System-wide twin peak critic

### 6.1 One shared critic, not one critic per agent

The peak target is a single system-wide physical quantity. Therefore, the
algorithm owns exactly one online `TwinPeakCritic` and one target copy. It must
not create three peak critics. All three delayed actor updates query the same
critic while replacing only the active agent's differentiable action, matching
the existing CTDE update pattern.

This choice is both semantically correct and important for speed: compared with
per-agent peak critics, it removes two-thirds of the additional peak-network
forward and backward passes.

### 6.2 Inputs and outputs

The critic receives:

```text
concat(global_state, flattened_projected_joint_action)
```

The projected action has the existing three effective channels per microgrid.
The critic outputs two nonnegative scalar estimates:

```text
peak1_normalized
peak2_normalized
```

Use two independent MLPs with default hidden dimensions `[64, 64]`, GELU or
ReLU activations, and one scalar output each. The network is deliberately
smaller than the parent reward critics (`[128, 128]`). A positive output
parameterization such as `Softplus` is allowed only if its initialization is
tested to avoid saturated gradients.

### 6.3 Physical peak normalizer

Targets are computed in physical kW and divided by a deterministic
`system_peak_scale_kw`. The scale is derived once from the environment's load
normalizers and maximum DR electrical powers, must be positive and finite, and
is stored in the checkpoint contract. It is not recomputed silently at load
time.

The metric logger converts predictions back to kW before reporting them.

## 7. Exact learning target

The existing differentiable joint-action projector already returns, for each
microgrid:

```text
plans[i]["actual_load_kw"]
plans[i]["flexible_load_kw"]
plans[i]["base_load_kw"]
```

For each replayed transition:

```text
current_peak_cost_kw
  = sum(plans[i]["actual_load_kw"] for i in agents)

c_t = current_peak_cost_kw / system_peak_scale_kw
```

With parent target actions and their already projected next actions:

```text
next_peak_1, next_peak_2
  = target_peak_critic(next_state, next_projected_joint_action)

conservative_next_peak = max(next_peak_1, next_peak_2)

y_peak = where(
  done,
  c_t,
  max(c_t, conservative_next_peak),
)
```

There is no discount factor in this max-return recursion. Applying `gamma`
would change the meaning from a physical remaining maximum to a discounted
surrogate.

Both online peak heads minimize Smooth L1 loss against the detached target:

```text
L_peak_critic
  = SmoothL1(peak1, y_peak)
  + SmoothL1(peak2, y_peak)
```

The use of the larger target prediction is intentionally conservative for a
cost critic. The target is clipped only by validated physical scale limits, not
by minibatch statistics.

## 8. Actor objective

The inherited reward critic remains unchanged. For the active agent:

```text
L_profit = -Q_reward_1(state, projected_joint_action).mean()

predicted_peak_risk
  = max(
      PeakCritic_1(state, projected_joint_action),
      PeakCritic_2(state, projected_joint_action),
    ).mean()

L_actor = L_profit + peak_weight(update_step) * predicted_peak_risk
```

During the actor update:

- freeze both the reward critic and peak critic parameters;
- preserve gradients through the active actor and differentiable projector;
- detach other agents' actions;
- use the same projected joint-action tensor for reward and peak queries;
- unfreeze critics in `finally` blocks even when an exception occurs;
- apply the inherited actor gradient clipping after combining both losses.

The peak weight ramps only after the peak critic has received enough updates:

```text
peak_weight(step) = 0                                      before actor_start_step
peak_weight(step) = final_weight * ramp_fraction(step)     during ramp
peak_weight(step) = final_weight                           after ramp
```

No peak term is added to environment reward or reward-critic targets.

## 9. Speed-preserving update design

### 9.1 Mandatory tensor reuse

The implementation must not add projector calls to the normal update path.
Specifically:

- reuse `effective, plans` from the parent behavior-action projection to obtain
  `c_t`;
- reuse `target_effective` from the parent target-Q calculation for the target
  peak critic;
- reuse each active actor's `projected` tensor for both reward-Q and peak-risk
  actor losses;
- compute physical current load directly from the returned `plans` rather than
  decoding global state a second time;
- retain no diagnostic graph when diagnostics are not due.

Tests instrument the projector and assert the same projector-call count as the
corrected parent update for both actor-due and actor-not-due steps.

### 9.2 Reduced peak update cadence

The default peak critic is updated once every two MATD3 update steps:

```yaml
peak_critic_update_interval: 2
```

It uses the already sampled parent minibatch; no second replay sample is
allowed. Peak-critic target networks are soft-updated only when the peak critic
is updated. Actor peak regularization begins only after a configurable minimum
number of successful peak-critic updates.

This cadence is configurable for diagnostics, but changing it requires a new
run configuration fingerprint. It is the only speed optimization in this
design that can change learning dynamics, so it requires a pre-run learning
adequacy check. On the same fixed replay batches and initialization, compare
intervals 1 and 2 over at least 2,000 peak-critic update opportunities. Interval
2 is accepted only when all of the following hold:

- its held-out peak-target Smooth L1 error is no more than 5% above interval 1;
- its prediction/target rank correlation is no more than 0.02 below interval 1;
- its actor peak-gradient norm is finite and at least 90% of interval 1 after
  accounting for the configured actor weight;
- neither configuration shows divergence or a growing twin gap.

If any condition fails, the production YAML uses interval 1. This comparison
uses training/validation replay only and never formal-test data. Other speed
optimizations in this section are semantics-preserving and do not require this
trade-off check.

### 9.3 Lightweight network and metric sampling

- peak hidden dimensions default to `[64, 64]`;
- one system-wide twin critic is used;
- peak diagnostics follow the existing `diagnostic_interval` and do not force
  CPU synchronization on every update;
- detailed per-agent gradient metrics are collected only when diagnostics are
  due;
- formal evaluation calls actors only; it does not run the peak critic;
- automatic mixed precision is not introduced in version 1 because it adds a
  separate numerical variable and complicates reproducibility.

### 9.4 No extra environment rollouts

Peak labels are reconstructed from the state, projected action, and existing
projector plans. Training must not execute an additional environment step,
MILP solve, validation episode, or formal-test episode for peak supervision.

## 10. Initial configuration contract

The new YAML inherits the corrected predictive-DR settings and adds:

```yaml
algorithm:
  name: matd3_hierarchical_transformer_peak_risk
  peak_risk:
    enabled: true
    token_enabled: true
    critic_enabled: true
    contract_version: "1.0"

    critic_hidden_dims: [64, 64]
    critic_lr: 0.0003
    critic_update_interval: 2
    critic_loss: smooth_l1
    critic_max_grad_norm: 10.0
    conservative_twin_mode: max
    use_discount: false

    critic_warmup_updates: 1000
    actor_start_update: 5000
    actor_weight_initial: 0.0
    actor_weight_final: 0.02
    actor_weight_ramp_updates: 15000

    reuse_parent_projection: true
    diagnostics_only_on_schedule: true

  training_recovery:
    enabled: true
    snapshot_interval_episodes: 100
    incremental_replay_chunks: true
    compression: none
```

Validation rejects:

- a nonpositive update interval, learning rate, gradient limit, hidden width,
  physical peak scale, or ramp length;
- `conservative_twin_mode` other than `max` in contract 1.0;
- `use_discount: true`;
- actor peak regularization when the peak critic is disabled;
- disabling parent projection reuse in the production configuration;
- an algorithm name that does not equal the plugin slug;
- nonfinite weights or a negative final actor weight.

`actor_weight_final: 0.02` is an initial engineering default, not a guaranteed
optimum. If the full run violates the profit guardrail while reducing peak, the
next run may lower this value based on validation results only.

## 11. Diagnostics and saved artifacts

The training history records, on the normal diagnostic schedule:

```text
peak/current_system_load_mean_kw
peak/current_system_load_p95_kw
peak/target_future_max_mean_kw
peak/predicted_future_max_mean_kw
peak/critic_loss
peak/critic1_loss
peak/critic2_loss
peak/critic_grad_norm
peak/twin_gap_mean_kw
peak/actor_penalty
peak/actor_weight
peak/actor_peak_gradient_norm
peak/update_count
peak/nonfinite_count
runtime/updates_per_second
runtime/update_wall_time_ms
```

Formal saved-result analysis includes:

```text
mean_peak_load_reduction_kw
median_peak_load_reduction_kw
positive_peak_reduction_day_rate
p10_peak_load_reduction_kw
worst_day_peak_increase_kw
rebound_to_shed_ratio
system_grid_import_peak_kw
```

All gates read saved JSON or CSV files. They must not rerun policy evaluation or
the environment.

## 12. Reliability prerequisite workstream

Reliability changes are implemented and tested separately before the stage-three
full run.

### 12.1 P2P numerical tolerance

Replace the fixed physical consistency comparison with:

```text
effective_tolerance_kwh = max(
  configured_absolute_tolerance_kwh,
  dtype_and_scale_roundoff_bound,
)
```

The roundoff bound is calculated from input dtype epsilon, trade normalizer,
and reconstructed value magnitude. The episode-3454 residual of approximately
`1.1920929e-05 kWh` must pass, while deliberately material inconsistencies must
still fail.

### 12.2 Fail-closed saved-result gates

Required metrics must be present, numeric, and finite. In particular:

- missing `constraint_eligible` must fail rather than default to `true`;
- missing `hvac_late_power_p95_kw` must fail rather than default to zero;
- the HVAC late-power metric must be computed from saved hourly trajectories;
- no required safety metric may acquire a favorable fallback value.

### 12.3 Checkpoint normalization validation

P2P checkpoint loading must validate the stored normalization signature in
addition to checking the normalizer dictionary. A mismatch is a hard error.

### 12.4 Run lifecycle and failure report

Each run maintains an atomically replaced `run_status.json` with states:

```text
RUNNING -> TRAINED -> EVALUATING -> COMPLETED
                 \-> FAILED
```

Failures additionally produce `training_error.json` containing exception type,
message, traceback, last completed episode, update step, last durable recovery
point, algorithm slug, and config fingerprint.

`latest_run.json` is updated at run creation, not only after completion, so an
active run can be located reliably.

### 12.5 Fast resumable training

Add an optional public recovery protocol without changing canonical MATD3's
existing `save` or `load` contract. The shared pipeline may feature-detect the
explicit recovery protocol but must not inspect private actor or replay fields.

The CLI accepts:

```powershell
python scripts/run_experiment.py `
  --algorithm matd3_hierarchical_transformer_peak_risk `
  --resume-run <absolute-run-directory>
```

Recovery state includes:

- online and target actors, reward critics, and peak critics;
- all optimizer states;
- update counters and completed episode;
- replay contents, logical ordering, capacity, and insertion position;
- NumPy, algorithm RNG, PyTorch CPU, and PyTorch CUDA RNG states;
- training and validation histories and checkpoint-selection state;
- environment seed schedule cursor;
- immutable algorithm, config, schema, projection, and normalization
  fingerprints.

To protect training speed, replay persistence is incremental:

1. new transitions since the last durable point are written to an uncompressed
   contiguous chunk in a temporary path;
2. the model/optimizer/RNG metadata snapshot is written separately;
3. file hashes and logical replay ranges are written to a manifest;
4. data files are atomically renamed first;
5. the manifest is atomically replaced last and is the commit point;
6. resume ignores any orphan temporary file or uncommitted chunk.

Uncompressed chunks are deliberate: compression saves disk space but adds CPU
time during training. Snapshot work occurs only at the configured episode
boundary and must satisfy the overhead budget in Section 13.

## 13. Performance acceptance requirements

Performance is measured against the corrected predictive-DR parent on the same
machine, device, batch size, seed, replay contents, and diagnostic schedule.
Timing must exclude initial import and one-time CUDA warm-up.

### 13.1 Microbenchmark

Run at least 1,000 update calls on a fixed populated replay buffer, report the
median of three measurements, and separately measure actor-due and
actor-not-due paths.

Required:

- median update-time overhead no greater than 15%;
- hard maximum accepted overhead no greater than 20%;
- no additional projector calls;
- peak critic parameters no greater than 40% of one parent twin reward critic;
- GPU peak-memory increase no greater than 15% when CUDA is used.

If the 15% target is missed, optimize tensor reuse, logging synchronization,
and critic width before changing the learning objective. Crossing 20% blocks a
full 5,000-episode run.

### 13.2 End-to-end smoke benchmark

On identical 100-episode smoke configurations:

- wall-clock training overhead must be no greater than 15%;
- recovery snapshot overhead must be no greater than 3% of training wall time;
- deterministic evaluation overhead must be no greater than 5%;
- the speed comparison must not disable diagnostics or validation on only one
  side.

Speed gates are guardrails, not permission to reduce the parent batch size,
validation coverage, training episodes, or formal test days.

## 14. Effectiveness and safety gates

Compare the stage-three run with a new full corrected predictive-DR parent run
using the same seed and experiment settings. Old runs with a known contract bug
may be informative but are not the sole acceptance baseline.

### 14.1 Primary effectiveness gates

- mean actual-load peak reduction greater than `0 kW`;
- median actual-load peak reduction greater than `0 kW`;
- positive peak-reduction day rate at least `60%`;
- rebound-to-shed ratio materially lower than the corrected parent;
- mean team profit no more than `2%` below the corrected parent.

### 14.2 Inherited safety gates

- strict feasible day rate equals `1.0`;
- unmet energy remains zero within the established tolerance;
- EV departure success remains `1.0`;
- mill completion/production success remains at least `0.995`;
- HVAC comfort remains at least `0.995`;
- HVAC terminal joint success does not regress below the corrected parent gate;
- MG2 and MG3 projection rates remain zero within tolerance;
- all actions, losses, gradients, predictions, and saved metrics are finite.

### 14.3 Interpretation rules

- Peak improves and profit passes: stage three succeeds.
- Peak improves but profit falls by more than 2%: direction is valid, lower the
  actor peak weight using validation data and rerun.
- Mean improves but tail days worsen: do not claim robust peak control; a later
  version may add a tail-risk objective.
- Peak does not improve while critic loss decreases: investigate actor peak
  gradient strength and token use before increasing network size.
- Feasibility regresses: treat as an implementation defect, not a trade-off.
- Speed exceeds the hard budget: do not launch the full run.

## 15. Verification matrix

### 15.1 Unit tests

- token features use named fields, remain finite, and contain no future realized
  outcome leakage;
- token-disabled plus zero-weight mode is bitwise-equivalent to the parent
  action path for identical weights and inputs;
- randomized projected plans produce exactly the same summed actual load as the
  peak-cost extractor;
- terminal max-return target equals current load;
- nonterminal target equals the maximum of current and conservative next peak;
- no discount is applied;
- both peak heads receive gradients during peak-critic updates;
- actor peak loss reaches the active actor through the projector while peer
  actor actions remain detached;
- frozen critic parameters receive no gradients or optimizer updates;
- projector call count equals the parent call count;
- disabled peak critic produces exact parent update behavior;
- malformed or incompatible checkpoint and recovery manifests fail closed;
- P2P floating-point roundoff passes while material inconsistencies fail;
- missing gate metrics fail closed;
- interruption and resume reproduce an uninterrupted CPU reference trajectory.

### 15.2 Repository verification

Before implementation is declared complete:

1. run `pytest -q`;
2. run the update microbenchmark;
3. run a 1-3 episode complete pipeline smoke test;
4. run an interrupted-then-resumed smoke test;
5. confirm the selected plugin and its own YAML were used;
6. confirm `best_model.pt` is used for formal testing;
7. confirm formal test, coalition MILP, Equal/Shapley settlement, summary, and
   plotting complete;
8. confirm the expected 15 PNG and 15 PDF figures unless the run explicitly
   requests otherwise;
9. confirm outputs are under the stage-three working-run directory;
10. confirm no file below `algorithms/matd3/` changed.

## 16. Experiment sequence

The implementation and experiment order is:

```text
1. Reliability fixes and their regression tests
2. Stage-three isolated scaffold and contract
3. Peak token and parent-equivalence tests
4. Shared peak critic and max-return target tests
5. Integrated efficient training update
6. Checkpoint, recovery, diagnostics, and gate checker
7. Full repository tests
8. Speed microbenchmark
9. Short complete pipeline smoke test
10. Corrected predictive-DR seed-42 baseline run
11. Peak-risk seed-42 full run
12. Saved-result effectiveness, safety, and speed gate comparison
```

The full stage-three run must not start if reliability tests, projector-call
parity, finite-gradient tests, or the hard speed budget fail.

## 17. Expected outcome and limitations

The architecture gives the actor a direct action-conditioned signal for the
remaining daily maximum while preserving the economic reward objective. The
single shared critic and tensor reuse make the expected runtime increase small
relative to Transformer inference, environment simulation, validation, and
post-training MILP evaluation.

This design improves the probability of reducing rebound peaks; it does not
guarantee improvement before training. The result is accepted only through the
saved-result gates above. A one-seed run is sufficient for an engineering-stage
decision, but any later paper-level claim requires multiple seeds and confidence
intervals.
