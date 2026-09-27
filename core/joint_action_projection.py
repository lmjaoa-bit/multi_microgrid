from __future__ import annotations

from typing import Any

import torch

from core.renewable_first_dispatch import clear_renewable_first
from core.single_dispatch import SingleDispatchProjector


class JointActionProjector:
    """Differentiable raw [ESS, DR] -> effective [ESS-local, ESS-trade, DR].

    DR is projected first. Its electrical load is then added to the base-load
    channel used by the existing local-first ESS projector. This mirrors the
    environment execution order and keeps rollout, replay and actor gradients on
    the same physical semantics.
    """

    LEGACY_SEMANTICS = "single_dispatch_dr_v2"
    RENEWABLE_FIRST_SEMANTICS = "single_dispatch_dr_v3"

    def __init__(self, config, schema, normalizers, price_norm):
        self.ids = [str(mg["id"]) for mg in config["microgrids"]]
        if len(self.ids) != 3:
            raise ValueError("JointActionProjector requires exactly three microgrids")
        self.index = {name: i for i, name in enumerate(schema)}
        self.norm = normalizers
        self.dispatch_mode = str(
            config.get("dispatch", {}).get("priority_mode", "legacy_local_first")
        )
        if self.dispatch_mode not in {"legacy_local_first", "renewable_load_first"}:
            raise ValueError(f"Unsupported dispatch.priority_mode: {self.dispatch_mode!r}")
        self.semantics = (
            self.RENEWABLE_FIRST_SEMANTICS
            if self.dispatch_mode == "renewable_load_first"
            else self.LEGACY_SEMANTICS
        )
        self.ess = SingleDispatchProjector(config, schema, normalizers, price_norm)
        self.p2p_enabled = bool(
            config.get("env", {}).get("mode", "cooperative") == "cooperative"
            and config.get("p2p", {}).get("enabled", True)
        )
        self.batteries = {
            str(mg["id"]): dict(mg["battery"]) for mg in config["microgrids"]
        }
        self.dr = {}
        for mg_cfg in config["microgrids"]:
            mg = str(mg_cfg["id"])
            load_name = f"{mg}_load"
            if load_name not in self.index:
                raise ValueError(f"Global state is missing required field: {load_name}")
            dr_cfg = dict(mg_cfg.get("dr", {}))
            kind = str(dr_cfg.get("type", ""))
            if kind == "ev_charging" and dr_cfg.get("cohorts"):
                min_name = f"{mg}_ev_feasible_min"
                max_name = f"{mg}_ev_feasible_max"
            else:
                min_name = f"{mg}_dr_min"
                max_name = f"{mg}_dr_max"
            for name in (min_name, max_name):
                if name not in self.index:
                    raise ValueError(f"Global state is missing required field: {name}")
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
            self.dr[mg] = {
                "type": kind,
                "scale_kw": scale_kw,
                "cohorts": bool(dr_cfg.get("cohorts")),
                "min_name": min_name,
                "max_name": max_name,
            }

    def _dr_requested(self, raw: torch.Tensor, agent_index: int) -> torch.Tensor:
        if self.dr[self.ids[agent_index]]["type"] == "commercial_hvac":
            return raw.clamp(-1.0, 1.0)
        return (raw.clamp(-1.0, 1.0) + 1.0) * 0.5

    def _flexible_load_kw(self, mg: str, effective: torch.Tensor) -> torch.Tensor:
        scale = float(self.dr[mg]["scale_kw"])
        if self.dr[mg]["type"] == "commercial_hvac":
            return effective.abs() * scale
        return effective * scale


    def _renewable_first(
        self,
        state: torch.Tensor,
        raw: torch.Tensor,
        dr_effective: list[torch.Tensor],
        dr_plans: list[dict[str, Any]],
        ess_state: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        charge_requests, discharge_requests = [], []
        renewables, loads = [], []
        for i, mg in enumerate(self.ids):
            cfg = self.batteries[mg]
            def col(name: str) -> torch.Tensor:
                return ess_state[:, self.index[f"{mg}_{name}"]]
            ch_max = col("feasible_charge_max").clamp_min(0.0) * float(cfg["max_charge_kw"])
            dis_max = col("feasible_discharge_max").clamp_min(0.0) * float(cfg["max_discharge_kw"])
            u = raw[:, i, 0].clamp(-1.0, 1.0)
            charge_requests.append((-u).clamp_min(0.0) * ch_max)
            discharge_requests.append(u.clamp_min(0.0) * dis_max)
            renewables.append(
                col("pv") * float(self.norm[mg]["pv"])
                + col("wind") * float(self.norm[mg]["wind"])
            )
            # __call__ already rewrites ess_state[*_load] to base + projected
            # flexible load before entering the renewable-first clearing path.
            loads.append(col("load") * float(self.norm[mg]["load"]))

        charge_request = torch.stack(charge_requests, dim=1)
        discharge_request = torch.stack(discharge_requests, dim=1)
        ledger = clear_renewable_first(
            torch.stack(renewables, dim=1),
            torch.stack(loads, dim=1),
            charge_request,
            discharge_request,
            p2p_enabled=self.p2p_enabled,
        )
        ess_peer_sell = ledger["ess_peer_load_kw"].sum(dim=2)
        effective, plans = [], []
        for i, mg in enumerate(self.ids):
            cfg = self.batteries[mg]
            scale = max(
                float(cfg["max_charge_kw"]),
                float(cfg["max_discharge_kw"]),
                1e-12,
            )
            local_net = (
                ledger["ess_local_load_kw"][:, i]
                - ledger["actual_charge_kw"][:, i]
            ) / scale
            peer_sell = ess_peer_sell[:, i] / scale
            effective.append(torch.stack([local_net, peer_sell], dim=-1))
            plan = dict(dr_plans[i])
            plan.update({
                "charge": ledger["actual_charge_kw"][:, i],
                "local": ledger["ess_local_load_kw"][:, i],
                "remaining": ledger["ess_offer_uncapped_kw"][:, i],
                "cap": ledger["ess_offer_cap_kw"][:, i],
                "offer": ledger["ess_offer_kw"][:, i],
                "uncapped_offer": ledger["ess_offer_uncapped_kw"][:, i],
                "requested_ratio": torch.where(
                    ledger["ess_offer_uncapped_kw"][:, i] > 1e-12,
                    ledger["ess_offer_kw"][:, i]
                    / ledger["ess_offer_uncapped_kw"][:, i].clamp_min(1e-12),
                    torch.zeros_like(ledger["ess_offer_kw"][:, i]),
                ),
                "requested_charge_kw": charge_request[:, i],
                "requested_discharge_kw": discharge_request[:, i],
                "requested_power": discharge_request[:, i] - charge_request[:, i],
                "local_priority_reserve": torch.zeros_like(
                    ledger["ess_local_load_kw"][:, i]
                ),
                "mask": ledger["ess_offer_cap_kw"][:, i] > 1e-9,
                "ess_offer_projection_reduction_kw":
                    ledger["ess_offer_projection_reduction_kw"][:, i],
                "ess_p2p_unmatched_kw": ledger["ess_unmatched_kw"][:, i],
                "actual_discharge_kw": ledger["actual_discharge_kw"][:, i],
                "actual_charge_kw": ledger["actual_charge_kw"][:, i],
                "ess_peer_sell_kw": ess_peer_sell[:, i],
                "dispatch_ledger": ledger,
            })
            plans.append(plan)
        dr_channel = torch.stack(dr_effective, dim=1).unsqueeze(-1)
        return torch.cat(
            [torch.stack(effective, dim=1), dr_channel], dim=-1
        ), plans

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
            lower = state[:, self.index[self.dr[mg]["min_name"]]]
            upper = state[:, self.index[self.dr[mg]["max_name"]]]
            if torch.any(lower > upper + 1e-7):
                raise ValueError(f"{mg}: invalid DR feasible interval in global state")
            if self.dr[mg]["cohorts"]:
                effective = lower + requested * (upper - lower)
                mapping_delta = effective - requested
                safety_correction = torch.zeros_like(effective)
                safety_projected = torch.zeros_like(effective, dtype=torch.bool)
            else:
                effective = torch.maximum(lower, torch.minimum(requested, upper))
                mapping_delta = torch.zeros_like(effective)
                safety_correction = effective - requested
                safety_projected = safety_correction.abs() > 1e-7
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
                "dr_mapping_delta": mapping_delta,
                "dr_mapping_active": mapping_delta.abs() > 1e-7,
                "dr_safety_correction": safety_correction,
                "dr_safety_projected": safety_projected,
                "dr_projection_delta": safety_correction,
                "dr_projected": safety_projected,
                "flexible_load_kw": flexible_kw,
                "base_load_kw": base_kw,
                "actual_load_kw": actual_kw,
            })

        if self.dispatch_mode == "renewable_load_first":
            return self._renewable_first(
                state, raw, dr_effective, dr_plans, ess_state
            )

        ess_effective, ess_plans = self.ess(ess_state, raw[..., 0:1])
        plans: list[dict[str, Any]] = []
        for i in range(len(self.ids)):
            plan = dict(ess_plans[i])
            plan.update(dr_plans[i])
            plans.append(plan)

        dr_channel = torch.stack(dr_effective, dim=1).unsqueeze(-1)
        return torch.cat([ess_effective, dr_channel], dim=-1), plans
