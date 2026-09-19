# MATD3-Transformer-Critic (37 state tokens + 6 action queries)

This is an isolated MATD3 variant. Three original two-output MLP actors,
team reward, three independent centralized twin critics, replay format, DR-first
differentiable projection, eight-hour forecast and common training pipeline are
unchanged. Only the architecture **inside** each critic changes.

## Normal training

    git switch codex/matd3-transformer-critic-8h
    python scripts/run_experiment.py --algorithm matd3_transformer_critic --device cuda

The requested plugin uses its own configs/algorithms/matd3_transformer_critic.yaml
and the shared configs/cooperative.yaml. Outputs go to
outputs/working_runs/matd3_transformer_critic/runs/<run_id>/.
Standard validation, best checkpoint formal test, coalition MILP,
Equal/Shapley settlement and saved-result-only figures remain unchanged.

## Architecture

- 3 ESS, 3 heterogeneous DR, 3 current supply, 3 trade and 1 market token.
- 24 microgrid-by-horizon forecast tokens (3 microgrids x 8 hours).
- One 32-dim, 4-head, 1-layer state self-attention encoder per independent Q.
- Six device queries (ESS/DR for each MG) include all nine projected joint
  actions and cross-attend to every valid state token.
- The Q head concatenates the six action-conditioned representations,
  current-state skip features, and all nine projected effective actions.
- Forecast features are excluded from the raw Q skip to prevent padded
  end-of-day forecast values from leaking into the value estimate.
- No future SOC/DR trajectories, realized trades or post-action state inputs.
  Actors still receive only their configured decentralized observations.

## Compute budget

Existing fast replay, packed minibatch transfers, sparse diagnostics, policy
delay and target-policy smoothing are reused. Normal updates never request
attention weights or perform auxiliary environment rollouts. Encoder dropout
is zero to keep the Q evaluation deterministic for a fixed state/action.
The optional Q accuracy check runs only after training.

## Post-training Q accuracy (separate optional check)

After the standard training command completes, find its run directory in
outputs/working_runs/matd3_transformer_critic/latest_run.json and run:

    python -m algorithms.matd3_transformer_critic.q_accuracy --run "outputs/working_runs/matd3_transformer_critic/runs/<run_id>" --split test --days 3 --seeds-per-day 2 --channel ess --device cuda

Repeat with --channel dr for flexible-load action checks. The diagnostic loads
checkpoints/best_model.pt, resets matching day/seed/anchor pairs, changes only
the focal MG's first ESS/DR raw action, uses the existing physical projector,
and then follows the same frozen policy to the episode end. Both candidates
use common random seeds. It compares the scaled, discounted TEAM REWARD
against each per-MG critic's min(Q1,Q2); currency profit is NOT an equivalent
target and should not be used in its place.

Outputs within the same run directory:

- critic_q_accuracy.json: Q/return MAE, action-gap MAE, preference agreement,
  twin-Q gap and sample counts.
- critic_q_accuracy_cases.csv: scenario, action, Q1, Q2, min-Q and realized
  discounted team reward.
- critic_q_action_pairs.csv: predicted and realized candidate action gaps.
- critic_future_action_probe.csv: separate peer-forecast input perturbation test
  of the **predicted action gap**; this is model sensitivity, NOT realized
  return accuracy or a physically consistent forecast counterfactual.

Interpretation: realized return is a noisy sample of the Q expectation.
Repeated matched seeds reduce variance; first-action changes propagate
through subsequent system states and the frozen policy. Min-Q also has a
built-in conservative bias. This test does NOT by itself prove that
attention uses future information causally. Run controlled physical forecast
scenario/ablation checks separately. Do not select checkpoints or tune model
hyperparameters using test-split Q accuracy.

Sampled TD errors and Q statistics remain in standard training history.
Encoder/cross-attention gradient norms are written into q_statistics[mg] at
existing diagnostic boundaries, NOT collected at every training step.
