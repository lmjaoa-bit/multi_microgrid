from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import highspy
import numpy as np


@dataclass(frozen=True)
class HVACForecastSnapshot:
    snapshot_id: str
    boundary_timestamps: tuple[str, ...]
    outside_temperature_c: np.ndarray
    ghi_w_per_m2: np.ndarray
    base_load_kw: np.ndarray
    interval_occupied: np.ndarray
    comfort_low_c: np.ndarray
    comfort_high_c: np.ndarray

    def __post_init__(self) -> None:
        h = len(self.outside_temperature_c)
        if len(self.boundary_timestamps) != h + 1:
            raise ValueError("HVAC snapshot requires H+1 boundary timestamps")
        for name in (
            "ghi_w_per_m2", "base_load_kw", "interval_occupied",
            "comfort_low_c", "comfort_high_c",
        ):
            if len(getattr(self, name)) != h:
                raise ValueError(f"HVAC snapshot {name} must have length H")
        if h and not all(
            np.isfinite(np.asarray(getattr(self, name), dtype=np.float64)).all()
            for name in (
                "outside_temperature_c", "ghi_w_per_m2", "base_load_kw",
                "comfort_low_c", "comfort_high_c",
            )
        ):
            raise ValueError("HVAC snapshot contains non-finite values")


@dataclass(frozen=True)
class HVACViabilityResult:
    min_signed_power_kw: float
    max_signed_power_kw: float
    feasible: bool
    status: str
    current_min_slack_c: float
    future_min_slack_sum_c: float
    snapshot_id: str


class HVACViabilitySolverError(RuntimeError):
    pass


def thermal_linear_coefficients(
    config: Mapping[str, Any], delta_t: float
) -> tuple[np.ndarray, np.ndarray, float]:
    dt = float(delta_t)
    r_oa = float(config["r_out_air_c_per_kw"])
    r_am = float(config["r_air_mass_c_per_kw"])
    c_air = float(config["c_air_kwh_per_c"])
    c_mass = float(config["c_mass_kwh_per_c"])
    a = np.asarray([
        [
            1.0 - dt / c_air * (1.0 / r_oa + 1.0 / r_am),
            dt / (c_air * r_am),
        ],
        [
            dt / (c_mass * r_am),
            1.0 - dt / (c_mass * r_am),
        ],
    ], dtype=np.float64)
    b = np.asarray([dt / c_air, 0.0], dtype=np.float64)
    return a, b, dt


def _electric_from_q(q: float, config: Mapping[str, Any]) -> float:
    if q >= 0.0:
        return float(q / float(config["cop_heat"]))
    return float(q / float(config["cop_cool"]))


def _build_model(
    initial_air_c: float,
    initial_mass_c: float,
    snapshot: HVACForecastSnapshot,
    *,
    config: Mapping[str, Any],
    delta_t: float,
    relaxed: bool,
):
    h_count = len(snapshot.outside_temperature_c)
    model = highspy.Highs()
    model.setOptionValue("output_flag", False)
    inf = highspy.kHighsInf
    q_min = -float(config["cooling_max_kw"]) * float(config["cop_cool"])
    q_max = float(config["heating_max_kw"]) * float(config["cop_heat"])
    q = [model.addVariable(lb=q_min, ub=q_max, name=f"q_{k}") for k in range(h_count)]
    air = [
        model.addVariable(
            lb=float(initial_air_c) if k == 0 else -inf,
            ub=float(initial_air_c) if k == 0 else inf,
            name=f"air_{k}",
        )
        for k in range(h_count + 1)
    ]
    mass = [
        model.addVariable(
            lb=float(initial_mass_c) if k == 0 else -inf,
            ub=float(initial_mass_c) if k == 0 else inf,
            name=f"mass_{k}",
        )
        for k in range(h_count + 1)
    ]
    slack = (
        [model.addVariable(lb=0.0, ub=inf, name=f"slack_{k}") for k in range(h_count)]
        if relaxed else None
    )
    a, bq, dt = thermal_linear_coefficients(config, delta_t)
    solar_gain = float(config.get("solar_gain_kw_per_wm2", 0.025))
    solar_air_fraction = float(config.get("solar_air_fraction", 0.20))
    occ_fraction = float(config.get("internal_gain_occupied_fraction", 0.50))
    unocc_fraction = float(config.get("internal_gain_unoccupied_fraction", 0.20))
    c_air = float(config["c_air_kwh_per_c"])
    c_mass = float(config["c_mass_kwh_per_c"])
    r_oa = float(config["r_out_air_c_per_kw"])

    for k in range(h_count):
        solar = solar_gain * float(snapshot.ghi_w_per_m2[k])
        q_air = solar_air_fraction * solar
        q_mass = (1.0 - solar_air_fraction) * solar
        q_internal = float(snapshot.base_load_kw[k]) * (
            occ_fraction if bool(snapshot.interval_occupied[k]) else unocc_fraction
        )
        b_air = dt / c_air * (
            float(snapshot.outside_temperature_c[k]) / r_oa
            + q_internal + q_air
        )
        b_mass = dt / c_mass * q_mass
        model.addConstr(
            air[k + 1]
            == a[0, 0] * air[k] + a[0, 1] * mass[k]
            + bq[0] * q[k] + b_air
        )
        model.addConstr(
            mass[k + 1]
            == a[1, 0] * air[k] + a[1, 1] * mass[k] + b_mass
        )
        low = float(snapshot.comfort_low_c[k])
        high = float(snapshot.comfort_high_c[k])
        if slack is None:
            model.addConstr(air[k + 1] >= low)
            model.addConstr(air[k + 1] <= high)
        else:
            model.addConstr(air[k + 1] + slack[k] >= low)
            model.addConstr(air[k + 1] - slack[k] <= high)
    return model, q, air, mass, slack


def _solve_status(model: highspy.Highs, objective, *, maximize: bool) -> str:
    if maximize:
        model.maximize(objective)
    else:
        model.minimize(objective)
    return model.modelStatusToString(model.getModelStatus()).lower()


def solve_hvac_viability(
    initial_air_c: float,
    initial_mass_c: float,
    snapshot: HVACForecastSnapshot,
    *,
    config: Mapping[str, Any],
    delta_t: float,
) -> HVACViabilityResult:
    h_count = len(snapshot.outside_temperature_c)
    if h_count == 0:
        return HVACViabilityResult(
            0.0, 0.0, True, "terminal", 0.0, 0.0, snapshot.snapshot_id
        )

    try:
        hard, q, _, _, _ = _build_model(
            initial_air_c, initial_mass_c, snapshot,
            config=config, delta_t=delta_t, relaxed=False,
        )
        status = _solve_status(hard, q[0], maximize=False)
        if "optimal" in status:
            q_low = float(hard.val(q[0]))
            status_hi = _solve_status(hard, q[0], maximize=True)
            if "optimal" not in status_hi:
                raise HVACViabilitySolverError(
                    f"HVAC max-bound solve failed: {status_hi}"
                )
            q_high = float(hard.val(q[0]))
            return HVACViabilityResult(
                _electric_from_q(q_low, config),
                _electric_from_q(q_high, config),
                True, "optimal", 0.0, 0.0, snapshot.snapshot_id,
            )
        if "infeasible" not in status:
            raise HVACViabilitySolverError(
                f"HVAC feasibility solve failed: {status}"
            )

        relaxed, rq, _, _, slack = _build_model(
            initial_air_c, initial_mass_c, snapshot,
            config=config, delta_t=delta_t, relaxed=True,
        )
        assert slack is not None
        s0 = slack[0]
        status = _solve_status(relaxed, s0, maximize=False)
        if "optimal" not in status:
            raise HVACViabilitySolverError(
                f"HVAC current-slack solve failed: {status}"
            )
        current_slack = max(0.0, float(relaxed.val(s0)))
        relaxed.addConstr(s0 <= current_slack + 1e-7)

        future_expr = None
        for item in slack[1:]:
            future_expr = item if future_expr is None else future_expr + item
        future_slack = 0.0
        if future_expr is not None:
            status = _solve_status(relaxed, future_expr, maximize=False)
            if "optimal" not in status:
                raise HVACViabilitySolverError(
                    f"HVAC future-slack solve failed: {status}"
                )
            future_slack = max(0.0, float(sum(relaxed.val(x) for x in slack[1:])))
            relaxed.addConstr(future_expr <= future_slack + 1e-7)

        status = _solve_status(relaxed, rq[0], maximize=False)
        if "optimal" not in status:
            raise HVACViabilitySolverError(
                f"HVAC relaxed min-bound solve failed: {status}"
            )
        q_low = float(relaxed.val(rq[0]))
        status = _solve_status(relaxed, rq[0], maximize=True)
        if "optimal" not in status:
            raise HVACViabilitySolverError(
                f"HVAC relaxed max-bound solve failed: {status}"
            )
        q_high = float(relaxed.val(rq[0]))
        return HVACViabilityResult(
            _electric_from_q(q_low, config),
            _electric_from_q(q_high, config),
            False, "degraded_infeasible",
            current_slack, future_slack, snapshot.snapshot_id,
        )
    except HVACViabilitySolverError:
        raise
    except Exception as exc:
        raise HVACViabilitySolverError(
            f"HVAC viability solver failure: {type(exc).__name__}: {exc}"
        ) from exc
