# Forecast Context Study: fc_2x2_seed42_review_20260929

Required seeds: 42

This report contains one training seed only. Cross-seed variance is unknown and std_across_seeds is intentionally left blank; treat the results as exploratory rather than a multi-seed confirmation.

Factor definitions:

- B-A: add peer future balance at H=8.
- C-A: extend local forecast horizon from 8 to 24 without peer future.
- D-C: add peer future balance at H=24.
- D-B: extend horizon from 8 to 24 with peer future enabled.
- interaction: D-C-B+A.

The primary effect is the date-paired team-profit difference. Cooperation value capture is computed as a ratio of sums, not an average of daily ratios. The shared 55-day test set is not a new blind holdout, and MILP remains a full-information reference rather than pure learning error.

Included runs:

- group A, seed 42: C:\daima\multi_microgrid_rl_standalone\outputs\working_runs\matd3_forecast_context\runs\普通T预测上下文实验_2026-09-28_16-15-11_seed42
- group B, seed 42: C:\daima\multi_microgrid_rl_standalone\outputs\working_runs\matd3_forecast_context\runs\普通T预测上下文实验_2026-09-28_20-20-11_seed42
- group C, seed 42: C:\daima\multi_microgrid_rl_standalone\outputs\working_runs\matd3_forecast_context\runs\普通T预测上下文实验_2026-09-29_00-07-56_seed42
- group D, seed 42: C:\daima\multi_microgrid_rl_standalone\outputs\working_runs\matd3_forecast_context\runs\普通T预测上下文实验_2026-09-29_03-16-26_seed42

Source runs and checkpoints are read-only. This report does not rerun policies, environments, settlement, or MILP.
