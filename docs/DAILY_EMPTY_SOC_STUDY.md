# Independent daily dispatch with empty usable storage

The default `configs/cooperative.yaml` defines a shared cross-algorithm study in
which each natural day resets every battery to its configured initial SOC. Physical
dispatch enforces only current-step SOC, power, efficiency, and energy-balance
limits. It never forces charging or discharging to make a future terminal target
reachable.

Terminal mode is `penalty`: an end-of-day miss is priced after the final physical
transition, while P0 observation fields expose the SOC gap, time remaining,
required average power, and normalized terminal pressure. The target therefore
remains learnable information rather than an environment-selected future action.

The centralized critic state now includes each microgrid's next four hourly buy
price forecasts, normalized exactly like the actor's `price_t+k` features and read
from the same forecast provider. With the default three microgrids, its dimension
changes from 66 to 78. This is a shared information change, not an actor improvement.

Old checkpoints/results belong to a different study and cannot be reused as new
training results. Retrain using the shared pipeline and each algorithm's own YAML.
Canonical `algorithms/matd3/` is unchanged. Compare methods under the same initial
SOC and terminal mode, and distinguish full-day MILP information from four-hour
policy forecasts. The existing modified Transformer actor designs are independent
of this environment/state change.
