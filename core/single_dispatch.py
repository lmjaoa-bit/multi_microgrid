"""Differentiable local-first dispatch shared by rollout and learning."""
from __future__ import annotations

import torch


class SingleDispatchProjector:
    semantics = "single_dispatch_v1"

    def __init__(self, config, schema, normalizers, price_norm):
        self.ids = [str(mg["id"]) for mg in config["microgrids"]]
        self.index = {name: i for i, name in enumerate(schema)}
        self.norm, self.price_norm = normalizers, price_norm
        self.batteries = {str(mg["id"]): dict(mg["battery"]) for mg in config["microgrids"]}
        self.pair_capacity = float(config["p2p"]["pair_capacity_kw"])
        self.enabled = (config.get("env", {}).get("mode", "cooperative") == "cooperative"
                        and config["p2p"].get("enabled", True)
                        and config.get("ess_sharing", {}).get("ess_p2p_enabled", True))

    def __call__(self, state, raw):
        if raw.ndim != 3 or raw.shape[1:] != (len(self.ids), 1):
            raise ValueError("single_dispatch requires raw controls [batch, agents, 1]")
        if state.ndim != 2 or state.shape[0] != raw.shape[0]:
            raise ValueError("State and action batch dimensions must match")
        a = raw.clamp(-1., 1.)
        plans = []
        for i, mg in enumerate(self.ids):
            cfg, norm = self.batteries[mg], self.norm[mg]
            def col(name):
                return state[:, self.index[f"{mg}_{name}"]]
            ch_max = col("feasible_charge_max").clamp_min(0) * cfg["max_charge_kw"]
            ch_min = col("feasible_charge_min").clamp_min(0) * cfg["max_charge_kw"]
            dis_max = col("feasible_discharge_max").clamp_min(0) * cfg["max_discharge_kw"]
            dis_min = col("feasible_discharge_min").clamp_min(0) * cfg["max_discharge_kw"]
            u = a[:, i, 0]
            # Ordinary operation preserves zero. When zero is physically
            # infeasible, use the full scalar interval for the forced mode.
            power = torch.where(u >= 0, u * dis_max, u * ch_max)
            forced_charge, forced_discharge = ch_min > 0, dis_min > 0
            charge_power = -ch_max + (u + 1) * .5 * (ch_max - ch_min).clamp_min(0)
            discharge_power = dis_min + (u + 1) * .5 * (dis_max - dis_min).clamp_min(0)
            power = torch.where(forced_charge, charge_power,
                                torch.where(forced_discharge, discharge_power, power))
            charge, discharge = (-power).clamp_min(0), power.clamp_min(0)
            net = col("load") * norm["load"] - col("pv") * norm["pv"] - col("wind") * norm["wind"]
            deficit, surplus = net.clamp_min(0), (-net).clamp_min(0)
            local = torch.minimum(discharge, deficit)
            # This mandatory component survives an absent buyer. Everything
            # above it is an offer, and unmatched offered power stays stored.
            mandatory = (dis_min - local).clamp_min(0)
            offer_raw = (discharge - local - mandatory).clamp_min(0)
            remaining = (dis_max - local - mandatory).clamp_min(0)
            demand = (deficit - local).clamp_min(0) + (charge - surplus).clamp_min(0)
            if f"{mg}_grid_buy_price" in self.index:
                buy, sell = col("grid_buy_price") * norm["price"], col("grid_sell_price") * norm["price"]
            else:
                buy = state[:, self.index["grid_buy_price"]] * self.price_norm
                sell = state[:, self.index["grid_sell_price"]] * self.price_norm
            plans.append(dict(charge=charge, local=local, mandatory_export=mandatory,
                              remaining=remaining, demand=demand, buy=buy, sell=sell,
                              deficit=deficit, surplus=surplus, dis_max=dis_max,
                              charging=power < 0, local_priority_reserve=torch.zeros_like(local),
                              requested_power=power, uncapped_offer=offer_raw,
                              forced_dispatch=forced_charge | forced_discharge))
        # All local decisions are now final. There is no second pass that
        # increases local discharge after looking at peer demand.
        capped_demands = [p["demand"].clamp(max=self.pair_capacity) for p in plans]
        effective = []
        for i, mg in enumerate(self.ids):
            p = plans[i]
            zero = torch.zeros_like(p["local"])
            peer_cap = zero
            if self.enabled:
                for j, other in enumerate(plans):
                    if i != j:
                        peer_cap = peer_cap + torch.where(other["buy"] + 1e-10 >= p["sell"],
                                                         capped_demands[j], zero)
            # Local-first holds by construction: uncapped_offer > 0 implies
            # that this agent already covers its local deficit and is not charging.
            cap = torch.where(p["charging"] | (p["deficit"] - p["local"] > 1e-7), zero,
                              torch.minimum(p["remaining"], peer_cap))
            offer = torch.minimum(p["uncapped_offer"], cap)
            ratio = p["uncapped_offer"] / p["remaining"].clamp_min(1e-12)
            p.update(cap=cap, mask=cap > 1e-9, offer=offer, requested_ratio=ratio, peer_cap=peer_cap)
            scale = max(self.batteries[mg]["max_charge_kw"], self.batteries[mg]["max_discharge_kw"])
            effective.append(torch.stack([(p["local"] + p["mandatory_export"] - p["charge"]) / scale,
                                          offer / scale], dim=-1))
        return torch.stack(effective, dim=1), plans
