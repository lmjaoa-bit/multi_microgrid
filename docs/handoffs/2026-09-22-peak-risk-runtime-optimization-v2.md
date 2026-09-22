# Peak-Risk Transformer runtime optimization v2 (2026-09-22)

Branch: `codex/peak-risk-runtime-optimization-v2-20260922`, based on
`codex/peak-risk-runtime-optimization-20260922` @ `0fc231d5`.

## Why this change

The user's first post-v1 local CUDA comparison showed
`median_update_overhead_pct=14.24%` (legacy pooled median),
`end_to_end_training_overhead_pct=26.62%`, and
`deterministic_evaluation_overhead_pct=45.43%`. Parent timings differed
substantially between repetitions. The mixed policy-due/non-due median is
not a valid end-to-end update acceptance statistic, and ratio-of-medians can
disagree with paired runtime comparisons.

## Changes

1. For CUDA `select_actions` with the 15-token Actor, prepare the 14 peak-risk
   observation features **on CPU** from the original local/public observation
   using a schema-identical, parameter-free `PeakRiskFeatureExtractor`.
   Concatenate raw float32 observation and 14 features on CPU and send **one**
   packed tensor per agent to CUDA. The GPU runs the same peak encoder, 15-token
   Transformer, ESS/DR readouts, latent exploration and physical action map.
   This avoids repeated GPU launches for scalar-like peak feature operations
   at every environment step. It is not used in the training or target Actor
   batch path. Actor/checkpoint weights and config fingerprints are unchanged.
2. An inference-only Actor entrypoint accepts precomputed 14-dimensional
   features with strict shape/device/dtype checks and no mutable per-call
   state. It reuses the same token ordering and Transformer/readout weights.
   CPU parity is exact under the test fixture; CUDA parity test uses tolerant
   floating-point comparison and requires a local GPU to execute.
3. Benchmark retains the old pooled median as **informational only**.
   Each worker now reports policy-due, not-due and total actual mean update
   times. The update acceptance metric is the median of the 3+ matched
   `candidate/parent - 1` overheads using the *mean of all updates* in
   each repetition. End-to-end training and inference overheads also report
   median paired percent changes and the full paired arrays. The gate
   selects the weighted update metric on v2 benchmark reports and fails closed
   if it is missing.

## Local CUDA validation (PowerShell)

```powershell
git fetch origin
git switch codex/peak-risk-runtime-optimization-v2-20260922
git pull origin codex/peak-risk-runtime-optimization-v2-20260922

pytest tests/test_matd3_hierarchical_transformer_peak_risk.py -q

python scripts/benchmark_training_runtime.py `
  --compare-parent matd3_hierarchical_transformer_predictive_dr `
  --compare-candidate matd3_hierarchical_transformer_peak_risk `
  --compare-device cuda `
  --compare-updates 1000 `
  --compare-repetitions 3 `
  --compare-episodes 100 `
  --output outputs/runtime_benchmarks/peak_risk_v2_cuda.json
```

The pytest command includes a CUDA-only feature/action parity test when
`torch.cuda.is_available()`; CPU-only CI skips it. Compare the parent and
candidate **paired** overhead arrays in the JSON, alongside absolute timing.
The historical `median_update_overhead_pct` is not the acceptance statistic;
use `weighted_update_overhead_pct` and its explicit pass/hard-pass booleans.

To probe Actor peak penalty already active, repeat benchmark with
`--compare-active-peak` and a different output path. It advances counters
only; it does **not** establish critic accuracy or convergence. Do not claim
a GPU performance improvement, <=15% end-to-end overhead, or peak-shaving
effectiveness until those have actually been measured on the target GPU
and verified on aligned formal test days.
