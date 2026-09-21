# P2P Value Auxiliary Head Transformer Design

**Date:** 2026-09-21

**Status:** Proposed design, pending user review

**Target algorithm slug:** `matd3_hierarchical_transformer_p2p_value`

**Parent algorithm:** `matd3_hierarchical_transformer_predictive_dr`

**Initial experiment mode:** P2P auxiliary-head ablation, seed 42

## 1. Purpose

This document specifies the second stage of the approved Transformer improvement
roadmap. The new isolated algorithm adds supervised P2P transaction-value
representation learning to the corrected predictive-DR parent while preserving
the existing action space, physical projection, reward, environment, settlement,
and shared experiment workflow.

The motivating observation is that a large fraction of ESS P2P offers remain
unmatched. The current actor already receives market history and peer energy
information, but its policy loss only receives a delayed scalar value signal
through the centralized critic. The new auxiliary task gives the shared
Transformer representation a direct, action-conditioned learning signal about
whether an ESS offer is likely to clear and how much of it is likely to clear.

The stage is successful only if it improves realized P2P matching without
gaming the acceptance ratio by suppressing offers and without regressing
physical feasibility, terminal tasks, HVAC behavior, or operating cost.

## 2. Scope and non-goals

### 2.1 In scope

- one new isolated algorithm package and one isolated YAML configuration;
- an actor-side P2P auxiliary trunk with two prediction heads;
- labels derived from already stored next observations;
- a masked supervised auxiliary loss added to delayed actor updates;
- strict checkpoint isolation and auxiliary diagnostics;
- saved-result gates for P2P improvement and inherited safety constraints.

### 2.2 Explicit non-goals

This stage must not:

- add a P2P offer action or change the two-dimensional raw action space;
- change P2P clearing, pair capacities, prices, settlement, or physical dispatch;
- modify environment rewards or add a peak-shaving reward;
- add the stage-three peak-risk token or peak auxiliary critic;
- change replay-buffer public interfaces or shared pipeline call signatures;
- edit canonical `algorithms/matd3/`;
- edit the corrected parent predictive-DR implementation in place;
- use formal-test data for training, hyperparameter selection, or checkpoint
  selection.

## 3. Package and configuration isolation

Create:

```text
algorithms/matd3_hierarchical_transformer_p2p_value/
├── __init__.py
├── plugin.py
├── algorithm.py
├── actor.py
├── p2p_auxiliary.py
└── training.py

configs/algorithms/matd3_hierarchical_transformer_p2p_value.yaml
```

The plugin exposes exactly one `ALGORITHM_SPEC`. The algorithm continues to use
the repository-wide experiment command:

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_p2p_value
```

Normal results are written only below:

```text
outputs/working_runs/matd3_hierarchical_transformer_p2p_value/runs/<run_id>/
```

## 4. Preserved parent behavior

The first implementation accepts exactly the corrected predictive-DR contract
version `1.1`. Supporting any later parent contract requires an explicit child
contract change and new compatibility tests. The child preserves:

- predictive HVAC mapping and configuration;
- feasible MG2 and MG3 DR mapping;
- EV catch-up floor and unavailable-EV raw action `-1.0`;
- latent-space exploration and target-policy smoothing;
- actor and critic optimizer defaults;
- centralized twin critics and physical joint-action projector;
- replay capacity, update cadence, policy delay, and target-network updates;
- validation constraints and formal-test workflow.

For identical inherited weights, observations, and latent noise, the child
actor's two policy actions must match the parent actor exactly before any
training update, whether the auxiliary objective is enabled or disabled.
Auxiliary modules may add parameters but may not alter the inference action
path.

## 5. Auxiliary architecture

### 5.1 Shared representations

The child actor reuses inherited modules and constructs the same attended token
tensor as the parent. It exposes, within the child package only, these attended
representations:

```text
ess_rep
supply_rep
trade_rep
peer_rep = mean(attended peer tokens)
```

The main ESS and DR readouts must execute the same operations, in the same order,
as the parent. The child must not modify the parent actor merely to expose these
representations. A focused parity test protects against drift in the duplicated
child readout path.

### 5.2 Behavior-action conditioning

The auxiliary predictor is conditioned on the replayed behavior ESS action, not
the current policy action:

```text
aux_input = concat(
  ess_rep,
  supply_rep,
  trade_rep,
  peer_rep,
  behavior_ess_action.detach(),
)
```

Detaching the behavior action is mandatory. The auxiliary objective may improve
the shared market representation, but it may not reduce its own loss by changing
the action used to define the recorded outcome.

### 5.3 Dual-head predictor

Use one small shared auxiliary trunk and two output heads:

```text
aux_trunk:
  Linear(4 * embed_dim + 1, hidden_dim)
  LayerNorm(hidden_dim)
  GELU

any_fill_head:
  Linear(hidden_dim, 1)              # raw logit

fill_ratio_head:
  Linear(hidden_dim, 1)
  Sigmoid                            # [0, 1]
```

The outputs are:

```text
any_fill_logit
conditional_fill_ratio
```

The auxiliary outputs are never returned by `select_actions` and never enter the
environment or projector.

## 6. Supervision from existing replay data

No replay schema change is required. For a transition from time `t` to `t+1`,
the next observation already contains the realized outcome of the action at
time `t`:

```text
previous_ess_p2p_offer
previous_ess_p2p_accepted
previous_ess_p2p_unmatched
```

These observation fields are scale-normalized by the same positive per-agent
`trade` normalizer. The label extractor must first restore physical kWh using
the checkpointed normalizer. This makes thresholds and consistency diagnostics
explicitly physical and keeps the contract safe if normalization code is later
refactored.

For each agent and sampled transition:

```text
trade_scale = normalizers_by_microgrid[agent]["trade"]
offer_kwh = next_observation[previous_ess_p2p_offer] * trade_scale
accepted_kwh = next_observation[previous_ess_p2p_accepted] * trade_scale
unmatched_kwh = next_observation[previous_ess_p2p_unmatched] * trade_scale

offered_mask = offer_kwh > offer_epsilon_kwh
any_fill_target = accepted_kwh > offer_epsilon_kwh
fill_ratio_target = clamp(
  accepted_kwh / max(offer_kwh, offer_epsilon_kwh), 0, 1
)
```

Only `offered_mask` rows contribute to either auxiliary loss. Zero-offer rows do
not become easy negative examples.

The label extractor must reject nonfinite values and materially inconsistent
outcomes such as negative offer, negative accepted energy, or accepted energy
greater than offer beyond numerical tolerance. The physical consistency
residual `offer_kwh - accepted_kwh - unmatched_kwh` is recorded diagnostically.

## 7. Auxiliary loss and scheduling

For offered rows:

```text
L_any = BCEWithLogits(any_fill_logit, any_fill_target)
L_fill = SmoothL1(conditional_fill_ratio, fill_ratio_target)

L_aux = any_fill_loss_weight * L_any
        + fill_ratio_loss_weight * L_fill
```

The delayed actor update uses:

```text
L_actor_total = L_policy + auxiliary_weight(update_step) * L_aux
```

The auxiliary weight ramps linearly:

```text
auxiliary_weight(update_step)
  = loss_weight_max * min(update_step / ramp_updates, 1)
```

First-run defaults:

```yaml
p2p_auxiliary:
  enabled: true
  hidden_dim: 64
  loss_weight_max: 0.05
  ramp_updates: 5000
  any_fill_loss_weight: 1.0
  fill_ratio_loss_weight: 1.0
  offer_epsilon_kwh: 1.0e-6
  consistency_tolerance_kwh: 1.0e-5
  detach_behavior_action: true
```

The first contract rejects disabling behavior-action detachment, negative loss
weights, nonpositive epsilon or tolerance, nonpositive hidden size, and
nonpositive ramp length.

If a sampled agent batch contains no offered rows, its auxiliary loss is a
differentiable zero and all auxiliary sample-count diagnostics are zero. The
training step must remain finite.

## 8. Gradient boundaries

The auxiliary graph may update:

- all shared observation-token encoders, including ESS, DR, supply, trade,
  peer, and forecast encoders, because attended representations are globally
  coupled by self-attention;
- token normalization and type embeddings;
- shared Transformer layers;
- auxiliary trunk and both auxiliary heads.

The auxiliary graph must not directly traverse:

- ESS action readout head or ESS linear skip;
- DR action readout head or DR linear skip;
- predictive HVAC transform;
- feasible DR transform;
- centralized critics;
- replayed behavior actions.

The ordinary policy loss continues to update the complete policy path. A focused
gradient test must demonstrate nonzero auxiliary gradients in the auxiliary head
and at least one shared market encoder while showing zero auxiliary-only gradient
in both action readout heads. Because the objective intentionally trains a shared
Transformer, indirect changes to DR and forecast representations are permitted;
the inherited HVAC, EV, MG1, and feasibility gates are mandatory protection
against harmful transfer.

## 9. Training integration

The child package owns its update implementation. It follows the corrected
predictive parent's critic update, target-action mapping, projection, delayed
actor update, gradient clipping, diagnostics, and soft-update behavior.

During an actor update for agent `mg`:

1. build current mapped policy actions exactly as the parent does;
2. compute ordinary centralized `-Q1` policy loss;
3. extract P2P labels from that agent's next observation;
4. compute auxiliary predictions using the current observation and replayed
   behavior ESS action;
5. add the scheduled masked auxiliary loss;
6. backpropagate once through the combined loss;
7. clip gradients once and perform one actor optimizer step;
8. soft-update actor and critic target networks exactly once.

Auxiliary learning must not alter target-Q construction or critic losses.

## 10. Configuration contract

The isolated YAML inherits the corrected parent experiment's network,
optimizer, predictive-HVAC, feasible-DR, diagnostics, runtime, training, and
validation settings. It changes only:

```yaml
algorithm:
  name: matd3_hierarchical_transformer_p2p_value
  p2p_auxiliary:
    enabled: true
    hidden_dim: 64
    loss_weight_max: 0.05
    ramp_updates: 5000
    any_fill_loss_weight: 1.0
    fill_ratio_loss_weight: 1.0
    offer_epsilon_kwh: 1.0e-6
    consistency_tolerance_kwh: 1.0e-5
    detach_behavior_action: true
```

The environment and physical configuration remain `configs/cooperative.yaml`.

## 11. Checkpoint isolation

Use a new slug and an explicit P2P auxiliary contract version. Every checkpoint
must contain:

```text
algorithm_slug
action_semantics
parent_predictive_contract_version
predictive_hvac_config
feasible_dr_config
p2p_auxiliary_contract_version
p2p_auxiliary_config
observation_schema_by_agent
normalizers_by_microgrid
actors and target_actors, including auxiliary heads
critics and target_critics
update_steps
```

Loading rejects missing or mismatched values for any contract field. Parent,
V1, V2, predictive-only, and stage-three checkpoints are incompatible.

## 12. Diagnostics

At the existing sparse diagnostic cadence, record per-agent and aggregate:

```text
p2p_auxiliary_weight
p2p_aux_any_fill_loss
p2p_aux_fill_ratio_loss
p2p_aux_total_loss
p2p_aux_offered_sample_rate
p2p_aux_offered_sample_count
p2p_aux_positive_fill_rate
p2p_aux_fill_ratio_mae
p2p_aux_predicted_fill_mean
p2p_aux_actual_fill_mean
p2p_aux_offer_consistency_error_mean
p2p_aux_shared_gradient_norm
p2p_aux_head_gradient_norm
```

Diagnostics are observational and may not change sampling, actions, gradients,
or optimizer cadence.

Formal evaluation continues to report:

- ESS offer energy;
- ESS accepted energy;
- ESS unmatched energy;
- energy acceptance rate, system-wide and by microgrid;
- offered-step, any-fill-step, and full-fill-step rates;
- total P2P energy and settlement outcomes.

## 13. Failure handling

- Missing required observation fields fail actor construction and name every
  missing field.
- Nonfinite labels or predictions raise a descriptive runtime error.
- Accepted energy materially above offered energy is a hard invariant failure.
- A no-offer batch returns differentiable zero auxiliary loss.
- A disabled auxiliary head is allowed only in tests for exact parent parity;
  the first training configuration requires it to be enabled.
- Auxiliary loss, policy loss, and gradients are checked for finiteness before
  optimizer steps.

## 14. Tests

### 14.1 Label extraction

- exact labels for zero, partial, and full fill;
- zero-offer rows are masked;
- scale-normalized offer, accepted, and unmatched values are restored to the
  correct physical kWh values before labels and residuals are computed;
- nonfinite, negative, and accepted-above-offer inputs are rejected;
- offer-minus-accepted-minus-unmatched consistency is measured.

### 14.2 Actor and mapping preservation

- identical inherited state dictionaries produce exactly equal parent and child
  policy actions with the auxiliary loss disabled;
- MG3 feasible mapping remains enabled;
- unavailable EV output remains raw `-1.0`;
- MG2 and MG3 projection remain zero on deterministic closed-loop fixtures;
- predictive HVAC action mapping is unchanged.

### 14.3 Auxiliary heads and gradients

- any-fill logits and fill ratios have correct shapes;
- fill ratios remain within `[0, 1]`;
- auxiliary-only gradients reach the auxiliary head and shared encoders;
- auxiliary-only gradients do not reach ESS or DR action heads;
- replay behavior actions receive no gradient;
- an all-zero-offer batch stays finite and produces zero loss.

### 14.4 Training update

- combined actor loss equals policy loss when scheduled auxiliary weight is zero;
- auxiliary weight ramps to the configured maximum;
- one update changes auxiliary and shared parameters but preserves optimizer and
  target-update cadence;
- target-Q and critic-loss calculations match the parent for identical batches;
- sparse diagnostics do not alter numerical updates.

### 14.5 Checkpoints

- identical round-trip succeeds;
- wrong slug, parent version, auxiliary version, auxiliary configuration,
  feasible configuration, predictive configuration, or schema is rejected;
- error messages name the mismatched contract field.

### 14.6 Repository regression

- full `pytest -q` passes;
- canonical MATD3 and all existing experimental packages remain unchanged;
- the shared pipeline completes a two-to-three-episode smoke test through
  validation, best checkpoint, formal test, MILP, settlement, summaries, and
  saved-result plotting.

## 15. Experiment protocol

### 15.1 Pre-training verification

Run focused unit tests, the full suite, deterministic mapping checks, and a
two-to-three-episode CPU smoke test. Confirm:

- the new plugin and its own YAML are selected;
- checkpoint contract fields are present;
- `best_model.pt` is used for formal testing;
- MG2 and MG3 projection are zero;
- figures and settlement artifacts are complete;
- outputs are isolated under the new slug;
- the working tree used for a full experiment is clean and committed.

### 15.2 First full experiment

Run one seed-42 experiment using the same 5000-episode length, validation
cadence, test days, environment hash, and physical hash as the approved parent
comparisons. Do not tune the auxiliary loss using formal-test results.

This stage is an internal algorithm ablation, not a paper result. Multi-seed
training is optional and must not begin before the seed-42 safety gates pass.

## 16. Acceptance criteria

### 16.1 Primary P2P gates

Use the frozen formal result
`MATD3HIERARCHICALTRANSFORMER_2026-09-20_15-09-10_seed42` as the external
reference and retain the corrected predictive parent as the architectural
parent.

The seed-42 candidate must satisfy all of:

| Metric | Requirement |
|---|---:|
| ESS offer energy acceptance rate | >= 36.49% |
| Improvement over predictive exploratory run | >= 5 percentage points |
| ESS accepted energy | >= 95% of frozen reference |
| ESS offer energy | >= 80% of frozen reference |
| Total P2P energy | >= 95% of frozen reference |

The volume guards prevent a trivial acceptance-rate improvement caused by
stopping offers.

### 16.2 Inherited safety gates

| Area | Metric | Requirement |
|---|---|---:|
| Reliability | Strict feasible-day rate | 100% |
| Reliability | Mean unmet energy | 0 kWh/day |
| MG2 | Projection rate | 0 |
| MG3 | Projection rate | 0 |
| MG3 | EV departure success | 100% |
| HVAC | Occupied comfort rate | >= 99.5% |
| HVAC | Joint terminal success | no worse than predictive parent by > 5 pp |
| HVAC | Late-power P95 | no worse than predictive parent by > 5% |
| MG1 | Terminal success or production completion | >= 99.5% |

### 16.3 Economic guardrails

| Metric | Requirement |
|---|---:|
| Mean total operating cost | <= frozen reference * 1.02 |
| Peak grid import | <= frozen reference * 1.05 |
| Renewable utilization | no decrease greater than 0.5 percentage points |

Any safety-gate failure produces an overall `FAIL`. A P2P primary-gate failure
means the auxiliary hypothesis is not supported even if average reward improves.

## 17. Interpretation rules

Evidence supporting the hypothesis has this signature:

- auxiliary fill predictions become calibrated on offered samples;
- acceptance rate rises while accepted and offered energy volumes remain stable;
- unmatched ESS energy decreases;
- actor actions, feasibility, HVAC, EV, and cost remain within guardrails.

The hypothesis is not supported if:

- prediction loss falls but realized acceptance does not improve;
- acceptance rises only because offer volume collapses;
- auxiliary gradients dominate policy gradients;
- P2P improves at the expense of unmet load, terminal tasks, or peak import;
- results require changing rewards or adding a P2P action.

In those cases, do not increase the auxiliary weight repeatedly. Diagnose label
calibration, action conditioning, and whether the current two-action policy has
sufficient authority before considering a new architecture.

## 18. Definition of done

Implementation is complete only when:

1. the isolated package and configuration exist;
2. label, actor, gradient, update, checkpoint, and preservation tests pass;
3. full `pytest -q` passes;
4. the shared CPU smoke test completes through plotting and settlement;
5. MG2/MG3 mapping and the predictive-HVAC contract remain intact;
6. canonical MATD3 and existing experimental packages remain unchanged;
7. implementation and verification evidence are committed independently from
   this design document;
8. the first full training run, if requested separately, starts from that clean
   committed state.
