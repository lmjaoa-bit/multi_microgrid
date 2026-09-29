# MG1 dispatch saturation: bounded recovery experiment

## Diagnosis

`dispatch_saturation_rate` measures raw actor output magnitude, not executed
battery power. MG1 can output nearly +1 while an empty battery executes zero.
The physical projector is flat over the unavailable discharge direction, and
the saturated tanh further attenuates gradients. More exploration can collect
charging transitions without restoring the actor's local gradient.

The saved unique-day counterfactual probe found nine surplus-charge states
across twelve distinct training dates where both critics and the measured
counterfactual continuation preferred charging. The saved Q curve shows a raw
output of approximately 0.999990, logit 6.1103, and tanh derivative 1.97e-5;
the unavailable positive dispatch region has zero physical-action gradient.
These probes support this mechanism for the observed MG1 behavior; they do not
establish that every saturated action is erroneous.

## Change

`matd3_dispatch_q_guided` is an independent experimental plugin. It retains the
1000-transition learning delay and 10% dispatch exploration from the first
experiment. Only when a dispatch direction is physically unavailable, five
candidate intents are evaluated through the existing projector. Both critics
must predict improvement above 0.01 relative to their own current-action Q.
A weight-0.01 auxiliary loss moves the pre-tanh dispatch logit toward the
accepted candidate. Physics, rewards, and deterministic action execution are
unchanged. The canonical MATD3 implementation and default algorithm are unchanged.

## Comparison

Both variants trained for 300 episodes with seeds 42, 43, and 44. Validation ran
every 50 episodes on three days. Each formal test used its validation-selected
best_model.pt on the same three dates, 2018-01-28 through 2018-01-30.

| Seed | Exploration-only test mean cost | Q-guided test mean cost | Exploration-only MG1 active days | Q-guided MG1 active days |
| --- | ---: | ---: | ---: | ---: |
| 42 | 1481.78 | 1380.72 | 0/3 | 3/3 |
| 43 | 1486.61 | 1372.49 | 0/3 | 3/3 |
| 44 | 1455.52 | 1317.95 | 0/3 | 3/3 |

An active day here requires both positive charging and positive discharging.
The average test cost fell by approximately 8.0%. These are three seeds on
three shared test dates, not nine independent dates. Validation cost improved
only for seed 43; seeds 42 and 44 were worse. This is evidence of recovery from
the observed idle behavior, not proof of better converged performance. The
exploration-only seed 42 actor also partially recovered later in training,
although its validation-selected checkpoint remained idle on the test days.

All six runs completed test, coalition MILP (zero failures), Equal/Shapley
settlement, summaries, and 15 PNG plus 15 PDF figures under working_runs.
The separate two-episode smoke test also completed the full pipeline.
Exact run paths, checkpoint hashes, daily charge/discharge, costs, and replay
actor saturation measurements are in dispatch_q_guided_comparison.json.

## Verification and use

The focused test selection passed 31 tests. Full `pytest -q` yielded 446 passed
and three failures: Windows path separators in saved-service audit, the
zero-demand minimum-tie subgradient, and ordinary-transformer CPU/GPU numeric
tolerance. These remain unresolved; the full suite is not green.

Run the independent experiment with:

```powershell
python scripts/run_experiment.py --algorithm matd3_dispatch_q_guided
```

Keep guidance weight at 0.01 and margin at 0.01 for the next longer training
comparison rather than tuning against these test dates. Set guidance weight to
zero for an ablation. Evaluate broader held-out dates and multiple seeds before
changing the default. Judge recovery by deterministic physical charging and
discharging, and judge quality by cost and service constraints, not saturation
alone. Critic agreement is not a guarantee of real improvement.
