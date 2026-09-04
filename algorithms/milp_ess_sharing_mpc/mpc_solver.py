from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimization.common_constraints import DispatchData, solve_dispatch


@dataclass(frozen=True)
class MpcFirstStep:
    status: str
    charge_kw: dict[str, float]
    discharge_kw: dict[str, float]
    p2p_export_kw: dict[str, float]


def solve_receding_horizon(env: object, *, forecast_horizon: int = 4) -> MpcFirstStep:
    """Solve current plus four forecast hours and return only the first control."""
    if env.episode_df is None or env.t >= env.episode_length:
        raise RuntimeError("Reset the environment before requesting an MPC action")
    remaining = int(env.episode_length) - int(env.t)
    steps = min(remaining, 1 + int(forecast_horizon))
    ids = list(env.agent_ids)
    current = env._exogenous_at(env.t)
    load = {mg: np.zeros(steps, dtype=np.float64) for mg in ids}
    renewable = {mg: np.zeros(steps, dtype=np.float64) for mg in ids}
    buy_price = {mg: np.zeros(steps, dtype=np.float64) for mg in ids}
    sell_price = {mg: np.zeros(steps, dtype=np.float64) for mg in ids}
    timestamps: list[str] = []

    for offset in range(steps):
        row_index = min(int(env.t) + offset, len(env.episode_context_df) - 1)
        timestamps.append(str(env.episode_context_df.iloc[row_index]["timestamp"]))
        if offset == 0:
            row = env.episode_df.iloc[int(env.t)]
            for mg in ids:
                buy_price[mg][offset], sell_price[mg][offset] = env.data.prices_at(row, mg)
                load[mg][offset] = float(current[mg]["load_kw"])
                renewable[mg][offset] = float(current[mg]["renewable_kw"])
        else:
            for mg in ids:
                forecast = env.forecasts.combined(mg, int(env.t), offset)
                load[mg][offset] = float(forecast["load_kw"])
                renewable[mg][offset] = float(forecast["pv_kw"] + forecast["wind_kw"])
                buy_price[mg][offset] = float(forecast["grid_price_per_kwh"])
                sell_price[mg][offset] = float(forecast["grid_sell_price_per_kwh"])

    data = DispatchData(
        timestamps=timestamps,
        agent_ids=ids,
        load_kw=load,
        renewable_kw=renewable,
        buy_price=buy_price,
        sell_price=sell_price,
    )
    current_soc = {mg: float(env.cores[mg].battery.soc) for mg in ids}
    terminal_target = {mg: float(env.cores[mg].battery.initial_soc) for mg in ids}
    result = solve_dispatch(
        env.config,
        data,
        ids,
        cooperative=True,
        solver_output=False,
        initial_soc_by_microgrid=current_soc,
        terminal_target_soc_by_microgrid=terminal_target,
        apply_terminal_condition=steps == remaining,
    )
    first = result.trajectories.groupby("microgrid", sort=False).first()
    return MpcFirstStep(
        status=result.status,
        charge_kw={mg: float(first.loc[mg, "battery_charge_kw"]) for mg in ids},
        discharge_kw={mg: float(first.loc[mg, "battery_discharge_kw"]) for mg in ids},
        p2p_export_kw={mg: float(first.loc[mg, "p2p_export_kw"]) for mg in ids},
    )
