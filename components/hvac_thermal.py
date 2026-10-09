"""Shared 2R2C transition for simulation and linear optimization.

Inputs are thermal kW entering the air and mass nodes, held constant during
one control interval. Missing discretization retains historical snapshots.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def thermal_discretization(config: Mapping[str, Any]) -> str:
    mode = str(config.get("thermal_discretization", "legacy_euler"))
    if mode not in {"legacy_euler", "exact_zoh"}:
        raise ValueError(f"Unsupported HVAC thermal_discretization: {mode!r}")
    return mode


def thermal_transition(config: Mapping[str, Any], delta_t: float) -> tuple[np.ndarray, np.ndarray]:
    """Return F,G such that T_next = F @ T + G @ [air_input, mass_input]."""
    mode = thermal_discretization(config)
    r_oa, r_am, c_air, c_mass, dt = (
        float(config["r_out_air_c_per_kw"]), float(config["r_air_mass_c_per_kw"]),
        float(config["c_air_kwh_per_c"]), float(config["c_mass_kwh_per_c"]), float(delta_t),
    )
    values = np.array([r_oa, r_am, c_air, c_mass, dt])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("HVAC thermal parameters and delta_t must be finite and positive")
    conductance = np.array([[-(1/r_oa + 1/r_am), 1/r_am], [1/r_am, -1/r_am]])
    capacities = np.array([c_air, c_mass])
    if mode == "legacy_euler":
        return np.eye(2) + dt * conductance / capacities[:, None], np.diag(dt / capacities)

    # Scaling temperature by sqrt(C) makes the passive heat operator symmetric.
    # Its real eigendecomposition gives an exact matrix exponential and input
    # integral using NumPy only; expm1 preserves precision for small intervals.
    root_capacity = np.sqrt(capacities)
    symmetric = conductance / np.outer(root_capacity, root_capacity)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    exponentials = np.exp(dt * eigenvalues)
    integrals = np.divide(np.expm1(dt * eigenvalues), eigenvalues,
                          out=np.full_like(eigenvalues, dt), where=eigenvalues != 0)
    exponential = (eigenvectors * exponentials) @ eigenvectors.T
    integral = (eigenvectors * integrals) @ eigenvectors.T
    f = exponential * root_capacity[None, :] / root_capacity[:, None]
    g = integral / np.outer(root_capacity, root_capacity)
    return f, g
