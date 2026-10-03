# Ordinary T with Q-guided ESS recovery

Run this independent experiment through the shared pipeline:

```powershell
python scripts/run_experiment.py --algorithm matd3_hierarchical_transformer_q_guided
```

The Transformer, critics, physical projector, rewards, and environment match
ordinary T. The experiment waits for 1000 replay transitions before learning
and replaces the ESS dispatch action with a uniform random intent with
probability 0.10 during noisy training. Evaluation remains deterministic.

When an ESS intent points toward a physically unavailable direction, five
candidate dispatch intents are evaluated through the regular projector. Both
critics must predict improvement above `dispatch_q_guidance_margin`; the best
conservative candidate then supplies a small auxiliary target to the ESS
pre-tanh logit, weighted by `dispatch_q_guidance_weight`. The default weight
and margin are 0.01. The DR output head receives no direct auxiliary gradient,
although it shares Transformer features with ESS.

This mechanism changes only policy learning. It does not force physical battery
power, change reward, or alter constraints. Critic agreement is a filter, not a
guarantee of real improvement. Set the auxiliary weight to zero for an
ablation, and compare deterministic MG2 battery throughput, raw action,
saturation, cost, and service constraints over multiple seeds.
