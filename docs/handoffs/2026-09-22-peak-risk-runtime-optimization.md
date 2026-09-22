# Peak-Risk Transformer CUDA runtime optimization (2026-09-22)

Base: `codex/peak-risk-transformer-20260922` @ `79af46a`.
Candidate: `codex/peak-risk-runtime-optimization-20260922`.

## User-provided pre-optimization CUDA benchmark

- 3 paired repetitions, 1000 update calls and 100 rollout episodes each.
- Update median overhead +27.7323%; actor-due +15.2662%; actor-not-due +24.9087%.
- End-to-end training overhead +18.3542%; deterministic inference overhead +16.7939%.
- Peak Critic parameter ratio 0.393937; projector/replay counts matched.
- These are **pre-optimization** measurements; do not treat the optimization as a performance pass until it is measured on the same GPU.

## Changes and contract

- `peak_risk.assert_finite`: CUDA uses PyTorch's stream-ordered
  `torch._assert_async(torch.isfinite(...).all(), message)`. CPU and builds
  without this operation retain a synchronous `FloatingPointError` check.
  On CUDA, failures surface as CUDA errors at the next synchronization; the
  training process must abort rather than continue after a device assertion.
- The target check aggregates four tensors into a single assertion. The system
  actual-load check verifies the summed three-load tensor on CUDA and preserves
  per-load checking on CPU. Shape checks are unchanged.
- The observation feature extractor precomputes its forecast gather indices,
  validity flags and Python flag-presence decision; removes dead fallback
  calculations; retains the same 14 features and the same 15-Token Actor.
- Peak loss and norm still undergo a check on every peak-critic update.
  The gradient clipping formula/threshold and the reward-critic/Actor
  optimization remain unchanged.
- No changes to peak critic size [60,60], loss weight, update interval,
  replay sampling, projection count, checkpoint format or recovery contract.

## Local GPU acceptance benchmark (PowerShell)

Run the normal paired comparison on the **optimized branch**:

```powershell
git fetch origin
git switch codex/peak-risk-runtime-optimization-20260922
git pull origin codex/peak-risk-runtime-optimization-20260922

python scripts/benchmark_training_runtime.py `
  --compare-parent matd3_hierarchical_transformer_predictive_dr `
  --compare-candidate matd3_hierarchical_transformer_peak_risk `
  --compare-device cuda `
  --compare-updates 1000 `
  --compare-repetitions 3 `
  --compare-episodes 100 `
  --output outputs/runtime_benchmarks/peak_risk_optimized.json
```

Compare the report against the pre-optimization JSON on the *same host and GPU*.
Do not assert the <=15% target / <=20% hard cap before that test succeeds.

Additionally measure the Actor penalty branch *without* 5000+ replay warmup
updates:

```powershell
python scripts/benchmark_training_runtime.py `
  --compare-parent matd3_hierarchical_transformer_predictive_dr `
  --compare-candidate matd3_hierarchical_transformer_peak_risk `
  --compare-device cuda `
  --compare-updates 1000 `
  --compare-repetitions 3 `
  --compare-episodes 100 `
  --compare-active-peak `
  --output outputs/runtime_benchmarks/peak_risk_optimized_active.json
```

The active-stage option sets both algorithms' update counters to the same
mid-ramp step and advances only the candidate's `peak_critic_updates` counter
past the warmup threshold, to exercise active peak loss computation. It does
**not** update the peak critic through thousands of steps; it is a throughput
probe, not evidence of critic accuracy, convergence or real long-run training
cost. Formal training and aligned test days remain necessary to establish
whether the optimized algorithm actually shaves peaks.
