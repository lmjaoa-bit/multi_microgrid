# MATD3-ShapleyQ Independent Baseline Repair Design

## Goal

Complete the partially implemented `matd3_shapley_q` algorithm so that Shapley-Q decomposes only the long-term cooperation surplus:

```text
Q_coop = Q_grand - Q_ind
sum_i Q_phi_i = Q_coop
Q_grand = Q_ind + sum_i Q_phi_i
```

The shared environment continues to emit one `team_reward` for the grand coalition. Shapley values remain long-term credit assignments and are never injected into immediate per-agent rewards.

## Scope and invariants

The repair stays inside the existing experimental package and shared algorithm-neutral lifecycle hooks. It preserves:

- three decentralized actors;
- three centralized Twin Marginal-Contribution Critics;
- exact three-agent predecessor enumeration and weights `1/3, 1/6, 1/6, 1/3`;
- single-dispatch action semantics;
- local operation for microgrids outside a coalition;
- hard terminal SOC with `SOC_start = SOC_end = 0.1`;
- the shared training, validation, formal-test, settlement, and plotting workflow.

The canonical implementation under `algorithms/matd3/` will not change.

## Independent baseline

`IndependentBaselineEvaluator` is a fixed counterfactual evaluator. Its environment uses the same physical configuration as the grand-coalition environment, except that all cross-microgrid P2P and ESS cooperation are disabled. Local renewable generation, load, ESS dispatch, and main-grid exchange remain active.

The default mode is a deterministic heuristic because the repository currently has no compatible trained no-cooperation MATD3 checkpoint. The heuristic has no learnable parameters and therefore cannot receive gradients or be manipulated by the Shapley actors.

An optional `frozen_policy` mode remains supported, but it must require an explicit compatible checkpoint. Missing, incompatible, or incomplete checkpoints fail at construction rather than silently leaving a random policy in place. Loaded parameters have `requires_grad=False`, the module stays in evaluation mode, and inference runs under `torch.no_grad()`.

For every episode, the baseline and grand branches use the same day, seed, deterministic/noisy-forecast settings, exogenous load/PV/wind arrays, price context, and initial SOC. Tests compare the complete scenario inputs rather than only one load array.

The evaluator caches per-step independent rewards, profits, and discounted return-to-go references for diagnostics. Out-of-range or uninitialized reward access raises an error instead of returning zero.

## Replay and value targets

The Shapley replay transition stores the raw authoritative values:

- `team_reward`;
- `independent_reward`;
- `cooperation_surplus_reward`;
- observations, state, action, next observations/state, and terminal flag.

The buffer enforces:

```text
cooperation_surplus_reward == team_reward - independent_reward
```

If discounted independent references are used for the grand-decomposition diagnostic, they are stored alongside the transition with the same episode/time identity.

The Shapley Bellman target is:

```text
y_coop = r_coop + gamma * (1 - done) * sum_i min(Q_phi_i_A', Q_phi_i_B')
```

The Twin minimum is therefore applied per agent before summation. Both current global twin sums are fitted to `y_coop`; they no longer fit the grand-coalition return.

Actor `i` continues to maximize its own `Q_phi_i_A`. Independent-baseline computation is outside every actor and critic graph.

## Characteristic-value semantics

The coalition utility API distinguishes physical value `J(S)` from normalized characteristic value:

```text
V(S) = J(S) - J(empty)
```

`J(empty)` represents three independently operating microgrids and is not forced to zero. Pure exact-Shapley helpers normalize real coalition-value mappings and calculate contributions from marginal differences. This gives executable coverage for `V(empty)=0`, Shapley efficiency, no-cooperation zero contributions, and positive cooperation.

`CoalitionAwareProjector` only masks cooperation channels. It preserves every microgrid's local dispatch channel for all coalition masks, including the empty coalition.

## Diagnostics and output

Training episode records include:

- grand team return/profit;
- independent baseline return/profit;
- cooperation-surplus return/profit;
- `shapley_q_mg1`, `shapley_q_mg2`, `shapley_q_mg3`, and their sum;
- cooperation-surplus Q target;
- efficiency residual;
- grand-decomposition residual when an independent return-to-go reference is available.

Validation continues to select `best_model.pt` solely by deterministic grand-coalition team profit. This preserves the real optimization objective.

Formal testing evaluates the selected policy and the same configured independent-baseline policy on matched days and seeds. Its daily CSV and JSON summary report grand, independent, and cooperation-surplus results. Existing daily coalition MILP evaluation and Equal/Shapley settlement remain separate post-processing and are not used as the online baseline reward.

The TensorBoard exporter maps the new record fields to the requested reward and Q namespaces. The terminal dashboard displays the essential grand/independent/surplus and Shapley efficiency diagnostics without changing saved data semantics.

Only residuals with a real reference are emitted. No tautological quantity is labeled as a learned grand-decomposition residual.

## Checkpoint compatibility

New checkpoints identify the cooperation-surplus objective explicitly and record baseline mode/configuration metadata. Loading an older `full_enumeration_v1` checkpoint is rejected because its critic objective may decompose `Q_grand` rather than `Q_grand - Q_ind`.

Checkpoint loading still validates action semantics, projection signature, state schema, and observation schemas.

## Error handling

The implementation fails early for:

- a missing or incompatible frozen-baseline checkpoint;
- unsupported baseline modes;
- inconsistent replay rewards;
- baseline/grand scenario mismatches;
- unavailable baseline step values;
- old Shapley-Q checkpoint semantics;
- non-finite losses, gradients, rewards, or diagnostics.

## Testing strategy

Tests are written or strengthened before production changes and must demonstrate the intended failure first. Coverage includes:

1. Empty-coalition physics: local ESS and grid operation remain available while all P2P flows are zero.
2. Characteristic normalization: the implementation computes `V(empty)=0` from a nonzero `J(empty)`.
3. Exact Shapley efficiency: contributions sum to `J(grand)-J(empty)`.
4. No-cooperation case: equal physical coalition values yield zero surplus and zero Shapley contributions.
5. Positive deterministic cooperation case: a renewable-rich seller and high-price deficit buyer produce positive surplus and positive total contribution.
6. Frozen-baseline isolation: all baseline gradients remain `None`, while every actor obtains a finite nonzero gradient in an update designed to exercise it.
7. Common scenario: timestamps, physical exogenous arrays, prices, forecasts, forecast masks/noise, seed, and initial SOC match.
8. Replay consistency and invalid-input rejection.
9. Per-agent Twin-min ordering in the cooperation target.
10. Training/formal-test output fields and strict checkpoint compatibility.

## Verification and acceptance

Before completion is reported:

1. Run the focused MATD3-ShapleyQ tests.
2. Run `pytest -q` for the full repository.
3. Run a 1-3 day `scripts/run_experiment.py --algorithm matd3_shapley_q` smoke workflow with learning enabled early enough to exercise critic and actor updates.
4. Confirm the plugin and `configs/algorithms/matd3_shapley_q.yaml` were selected.
5. Confirm formal testing used `best_model.pt`.
6. Confirm test evaluation, Equal/Shapley settlement, summaries, and saved-result-only plotting completed.
7. Confirm exactly 15 PNG and 15 PDF figures were produced.
8. Confirm the run is under `outputs/matd3_shapley_q/runs/`.
9. Confirm no file under `algorithms/matd3/` changed.
