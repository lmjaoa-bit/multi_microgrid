# Independent daily dispatch with empty usable storage

The default `configs/cooperative.yaml` now defines a shared cross-algorithm study:
each natural day resets every battery to its minimum SOC (0.10), and terminal SOC
mode is `none`. Energy below minimum SOC remains inaccessible. End-of-day energy
has no salvage value and is not carried into the next independent episode.

`none` disables terminal penalties, SOC shaping, and terminal reachability limits
in environment observations, centralized state, physical dispatch, and MILP/MPC.
SOC bounds, power limits, efficiency, degradation and energy balance remain active.
The existing `terminal_soc_gap` observation slot is zero in this mode to preserve
actor schemas; it does not represent a target. Legacy `penalty` and `hard` modes
remain available for historical configurations; their shaping behavior is unchanged.

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
