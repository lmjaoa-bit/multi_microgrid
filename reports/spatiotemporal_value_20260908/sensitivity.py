from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.data_manager import DataManager
from optimization.common_constraints import build_dispatch_data, load_config, solve_dispatch


CONFIG = ROOT / "configs" / "cooperative.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    config = load_config(CONFIG)
    manager = DataManager(config, split="test", seed=int(config["env"]["seed"]))
    unrestricted = pd.read_csv(OUTPUT_DIR / "daily_counterfactuals.csv").set_index("day")
    rows = []
    for index, raw_day in enumerate(manager.valid_days, start=1):
        day = str(raw_day.date())
        _, data = build_dispatch_data(config, split="test", day=day)
        result = solve_dispatch(
            config,
            data,
            data.agent_ids,
            cooperative=True,
            p2p_export_source="renewable_only",
        )
        unrestricted_cost = float(unrestricted.loc[day, "cost_spatiotemporal"])
        restricted_cost = float(result.total_cost)
        rows.append(
            {
                "day": day,
                "unrestricted_cost": unrestricted_cost,
                "renewable_only_p2p_cost": restricted_cost,
                "stored_energy_peer_export_value": restricted_cost - unrestricted_cost,
                "renewable_only_p2p_energy_kwh": float(result.trajectories["p2p_export_kw"].sum()),
                "renewable_only_battery_throughput_kwh": float(
                    result.trajectories["battery_charge_kw"].sum()
                    + result.trajectories["battery_discharge_kw"].sum()
                ),
            }
        )
        print(f"{day}: {index}/{len(manager.valid_days)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "stored_energy_peer_export_sensitivity.csv", index=False)
    x = frame["stored_energy_peer_export_value"].to_numpy()
    rng = np.random.default_rng(42)
    bootstrap_index = rng.integers(0, len(x), size=(20_000, len(x)))
    bootstrap_means = x[bootstrap_index].mean(axis=1)
    summary = {
        "mean_value_per_day": float(x.mean()),
        "median_value_per_day": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "positive_day_rate": float(np.mean(x > 1e-7)),
        "share_of_total_spatiotemporal_gain": float(
            x.mean() / unrestricted["total_spatiotemporal_gain"].mean()
        ),
    }
    (OUTPUT_DIR / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
