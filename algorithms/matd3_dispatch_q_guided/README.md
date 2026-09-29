# MATD3 dispatch Q guidance

Run through the shared pipeline:

```powershell
python scripts/run_experiment.py --algorithm matd3_dispatch_q_guided
```

The flat actor, twin critics and physical dispatch use the current MATD3
dispatch baseline. The experiment collects 1000 transitions before learning
and replaces each dispatch intent with a uniform random intent with probability
0.10 when exploration noise is enabled. Deterministic evaluation has neither
Gaussian nor uniform action noise.

During an actor update, an auxiliary loss can recover a dispatch intent whose
direction is unavailable at the battery's charge/discharge boundary:

1. Keep all other agent actions and the active agent's DR action fixed.
2. Project five dispatch candidates through the normal physical projector.
3. Compute each critic's improvement relative to its own current-action value.
4. Choose the largest minimum improvement; require both improvements to exceed
   `dispatch_q_guidance_margin` (0.01 in reward units).
5. Regress the dispatch pre-tanh logit toward the selected finite logit target.
   Add this squared error to the normal actor loss with
   `dispatch_q_guidance_weight` (0.01), normalized by the full batch size.

Candidate selection and targets are detached. Only dispatch gets a direct
auxiliary output-head gradient; shared actor features can also affect DR.
The auxiliary loss never changes environment rewards, physical constraints,
replay actions or target-policy smoothing. Set its weight to zero for an
ablation. Critic agreement is a filter, not a guarantee of true improvement.

Saved actor diagnostics include the trigger fraction, weighted auxiliary loss
and accepted conservative Q improvement. Assess deterministic charging,
discharging, total system cost and service constraints together; a lower raw
action saturation rate alone is not evidence of a better controller.

This package depends on the current dispatch baseline, not on the separate
`matd3_dispatch_exploration` experiment. Checkpoints carry this plugin's own slug.
