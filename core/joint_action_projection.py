from __future__ import annotations

from typing import Any

import torch

from core.single_dispatch import SingleDispatchProjector


class JointActionProjector:
    """Differentiable raw [ESS, DR] -> effective [ESS-local, ESS-trade, DR].

    DR is projected first. Its electrical load is then added to the base-load
    channel used by the existing local-first ESS projector. This mirrors the
    environment execution order and keeps rollout, replay and actor gradients on
    the same physical semantics.
    """

    semantics = "single_dispatch_dr_v2"

    def __init__(self, config, schema, normalizers, price_norm):
        self.ids = [str(mg["id"]) for mg in config["microgrids"]]
        if len(self.ids) != 3:
            raise ValueError("JointActionProjector requires exactly three microgrids")
        self.index = {name: i for i, name in enumerate(schema)}
        self.norm = normalizers
        self.ess = SingleDispatchProjector(config, schema, normalizers, price_norm)
        self.dr = {}
        for mg_cfg in config["microgrids"]:
            mg = str(mg_cfg["id"])
            for suffix in ("dr_min", "dr_max", "load"):
                name = f"{mg}_{suffix}"
                if name not in self.index:
                    raise ValueError(f"Global state is missing required field: {name}")
            dr_cfg = dict(mg_cfg.get("dr", {}))
            kind = str(dr_cfg.get("type", ""))
            if kind == "industrial_mill":
                scale_kw = float(dr_cfg["rated_power_kw"])
            elif kind == "commercial_hvac":
                heat = float(dr_cfg["heating_max_kw"])
                cool = float(dr_cfg["cooling_max_kw"])
                if abs(heat - cool) > 1e-9:
                    raise ValueError("Differentiable HVAC projection currently requires symmetric electric limits")
                scale_kw = heat
            elif kind == "ev_charging":
                scale_kw = float(dr_cfg["max_charge_kw"])
            else:
                raise ValueError(f"{mg}: unsupported DR type {kind!r}")
            if scale_kw <= 0.0:
                raise ValueError(f"{mg}: DR electrical power scale must be positive")
            self.dr[mg] = {"type": kind, "scale_kw": scale_kw,
                           "cohorts": bool(dr_cfg.get("cohorts"))}

    def _dr_requested(self, raw: torch.Tensor, agent_index: int) -> torch.Tensor:
        if self.dr[self.ids[agent_index]]["type"] == "commercial_hvac":
            return raw.clamp(-1.0, 1.0)
        return (raw.clamp(-1.0, 1.0) + 1.0) * 0.5

    def _flexible_load_kw(self, mg: str, effective: torch.Tensor) -> torch.Tensor:
        scale = float(self.dr[mg]["scale_kw"])
        if self.dr[mg]["type"] == "commercial_hvac":
            return effective.abs() * scale
        return effective * scale

    def __call__(self, state: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if raw.ndim != 3 or raw.shape[1:] != (len(self.ids), 2):
            raise ValueError("single_dispatch_dr requires raw controls [batch, agents, 2]")
        if state.ndim != 2 or state.shape[0] != raw.shape[0]:
            raise ValueError("State and action batch dimensions must match")

        dr_effective: list[torch.Tensor] = []
        dr_plans: list[dict[str, Any]] = []
        ess_state = state.clone()
        for i, mg in enumerate(self.ids):
            requested = self._dr_requested(raw[:, i, 1], i)
            lower = state[:, self.index[f"{mg}_dr_min"]]
            upper = state[:, self.index[f"{mg}_dr_max"]]
            if torch.any(lower > upper + 1e-7):
                raise ValueError(f"{mg}: invalid DR feasible interval in global state")
            effective = (lower + requested * (upper - lower)
                         if self.dr[mg]["cohorts"] else
                         torch.maximum(lower, torch.minimum(requested, upper)))
            flexible_kw = self._flexible_load_kw(mg, effective)
            load_index = self.index[f"{mg}_load"]
            base_kw = state[:, load_index] * float(self.norm[mg]["load"])
            actual_kw = base_kw + flexible_kw
            ess_state[:, load_index] = actual_kw / float(self.norm[mg]["load"])
            dr_effective.append(effective)
            dr_plans.append({
                "dr_requested": requested,
                "dr_effective": effective,
                "dr_min": lower,
                "dr_max": upper,
                "dr_projection_delta": effective - requested,
                "dr_projected": (effective - requested).abs() > 1e-7,
                "flexible_load_kw": flexible_kw,
                "base_load_kw": base_kw,
                "actual_load_kw": actual_kw,
            })

        ess_effective, ess_plans = self.ess(ess_state, raw[..., 0:1])
        plans: list[dict[str, Any]] = []
        for i in range(len(self.ids)):
            plan = dict(ess_plans[i])
            plan.update(dr_plans[i])
            plans.append(plan)

        dr_channel = torch.stack(dr_effective, dim=1).unsqueeze(-1)
        return torch.cat([ess_effective, dr_channel], dim=-1), plans
