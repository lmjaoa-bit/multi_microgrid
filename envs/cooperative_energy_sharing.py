from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CooperativeSharingResult:
    by_microgrid: dict[str, dict[str, float]]
    pair_flows_kwh: dict[str, float]
    total_p2p_kwh: float
    transfer_price_per_kwh: float


class CooperativeEnergySharing:
    """Deterministic, lossless sharing pool coordinated from physical imbalances."""

    def __init__(self, delta_t: float = 1.0, enabled: bool = True,
                 pair_capacity_kw: float = float("inf")) -> None:
        self.delta_t = float(delta_t)
        self.enabled = bool(enabled)
        self.pair_capacity_kw = float(pair_capacity_kw)
        if self.pair_capacity_kw <= 0.0:
            raise ValueError("pair_capacity_kw must be positive")

    def allocate(self, net_power_kw: Mapping[str, float], grid_sell_price: float,
                 grid_buy_price: float) -> CooperativeSharingResult:
        ids = sorted(str(key) for key in net_power_kw)
        by = {mg: {"p2p_buy_kw": 0.0, "p2p_sell_kw": 0.0,
                   "p2p_net_kw": 0.0, "p2p_transfer_payment": 0.0} for mg in ids}
        flows: dict[str, float] = {}
        if self.enabled:
            sellers = {mg: max(0.0, float(net_power_kw[mg])) for mg in ids}
            buyers = {mg: max(0.0, -float(net_power_kw[mg])) for mg in ids}
            for buyer in ids:
                for seller in ids:
                    if buyer == seller:
                        continue
                    power = min(buyers[buyer], sellers[seller], self.pair_capacity_kw)
                    if power <= 1e-12:
                        continue
                    energy = power * self.delta_t
                    key = f"{seller}->{buyer}"
                    flows[key] = flows.get(key, 0.0) + energy
                    by[seller]["p2p_sell_kw"] += power
                    by[buyer]["p2p_buy_kw"] += power
                    sellers[seller] -= power
                    buyers[buyer] -= power
        price = 0.5 * (float(grid_sell_price) + float(grid_buy_price))
        for mg in ids:
            by[mg]["p2p_net_kw"] = by[mg]["p2p_buy_kw"] - by[mg]["p2p_sell_kw"]
            by[mg]["p2p_transfer_payment"] = (
                by[mg]["p2p_sell_kw"] - by[mg]["p2p_buy_kw"]
            ) * self.delta_t * price
        total = sum(flows.values())
        if abs(sum(item["p2p_net_kw"] for item in by.values())) > 1e-8:
            raise RuntimeError("Cooperative P2P energy is not conserved.")
        if abs(sum(item["p2p_transfer_payment"] for item in by.values())) > 1e-8:
            raise RuntimeError("Cooperative P2P transfer payments are not budget balanced.")
        return CooperativeSharingResult(by, flows, float(total), float(price))

    def allocate_structured(
        self,
        renewable_offers_kw: Mapping[str, float],
        ess_offers_kw: Mapping[str, float],
        load_demands_kw: Mapping[str, float],
        charge_demands_kw: Mapping[str, float],
        grid_sell_price: float,
        grid_buy_price: float,
    ) -> CooperativeSharingResult:
        """Clear source- and use-aware offers without microgrid-id priority.

        Load demand is cleared before battery charging demand. Within each stage,
        renewable offers clear before ESS offers. A proportional iterative matcher
        removes the fixed-id preference of the legacy residual pool while enforcing
        the cumulative capacity of every directed pair.
        """
        ids = sorted(str(key) for key in renewable_offers_kw)
        expected = set(ids)
        for values in (ess_offers_kw, load_demands_kw, charge_demands_kw):
            if set(str(key) for key in values) != expected:
                raise ValueError("Structured P2P inputs must contain the same microgrid ids")
        by = {
            mg: {
                "p2p_buy_kw": 0.0, "p2p_sell_kw": 0.0,
                "p2p_net_kw": 0.0, "p2p_transfer_payment": 0.0,
                "renewable_p2p_sell_kw": 0.0, "ess_p2p_sell_kw": 0.0,
                "p2p_to_load_kw": 0.0, "p2p_to_battery_kw": 0.0,
                "renewable_p2p_offer_kw": max(0.0, float(renewable_offers_kw[mg])),
                "ess_p2p_offer_kw": max(0.0, float(ess_offers_kw[mg])),
                "ess_p2p_unmatched_kw": 0.0,
            }
            for mg in ids
        }
        pair_remaining = {
            (seller, buyer): self.pair_capacity_kw
            for seller in ids for buyer in ids if seller != buyer
        }
        flows_kw: dict[str, float] = {}
        tolerance = 1e-10

        def clear_stage(
            supplies_input: Mapping[str, float],
            demands_input: Mapping[str, float],
            supply_field: str,
            demand_field: str,
        ) -> None:
            if not self.enabled:
                return
            supplies = {mg: max(0.0, float(supplies_input[mg])) for mg in ids}
            demands = {mg: max(0.0, float(demands_input[mg])) for mg in ids}
            for _ in range(max(1, len(ids) * len(ids) * 4)):
                proposals: dict[tuple[str, str], float] = {}
                for seller in ids:
                    if supplies[seller] <= tolerance:
                        continue
                    buyers = [
                        buyer for buyer in ids
                        if buyer != seller and demands[buyer] > tolerance
                        and pair_remaining[(seller, buyer)] > tolerance
                    ]
                    demand_total = sum(demands[buyer] for buyer in buyers)
                    if demand_total <= tolerance:
                        continue
                    for buyer in buyers:
                        proportional = supplies[seller] * demands[buyer] / demand_total
                        proposals[(seller, buyer)] = min(
                            proportional, demands[buyer], pair_remaining[(seller, buyer)]
                        )
                if not proposals:
                    break
                for buyer in ids:
                    keys = [key for key in proposals if key[1] == buyer]
                    offered = sum(proposals[key] for key in keys)
                    if offered > demands[buyer] + tolerance:
                        scale = demands[buyer] / offered
                        for key in keys:
                            proposals[key] *= scale
                progress = 0.0
                for (seller, buyer), power in proposals.items():
                    power = min(power, supplies[seller], demands[buyer], pair_remaining[(seller, buyer)])
                    if power <= tolerance:
                        continue
                    supplies[seller] -= power
                    demands[buyer] -= power
                    pair_remaining[(seller, buyer)] -= power
                    by[seller][supply_field] += power
                    by[seller]["p2p_sell_kw"] += power
                    by[buyer][demand_field] += power
                    by[buyer]["p2p_buy_kw"] += power
                    key = f"{seller}->{buyer}"
                    flows_kw[key] = flows_kw.get(key, 0.0) + power
                    progress += power
                if progress <= tolerance:
                    break

        remaining_load = {mg: max(0.0, float(load_demands_kw[mg])) for mg in ids}
        remaining_charge = {mg: max(0.0, float(charge_demands_kw[mg])) for mg in ids}
        remaining_res = {mg: max(0.0, float(renewable_offers_kw[mg])) for mg in ids}
        remaining_ess = {mg: max(0.0, float(ess_offers_kw[mg])) for mg in ids}

        def stage(supply: dict[str, float], demand: dict[str, float], sf: str, df: str) -> None:
            before_sell = {mg: by[mg][sf] for mg in ids}
            before_buy = {mg: by[mg][df] for mg in ids}
            clear_stage(supply, demand, sf, df)
            for mg in ids:
                supply[mg] = max(0.0, supply[mg] - (by[mg][sf] - before_sell[mg]))
                demand[mg] = max(0.0, demand[mg] - (by[mg][df] - before_buy[mg]))

        stage(remaining_res, remaining_load, "renewable_p2p_sell_kw", "p2p_to_load_kw")
        stage(remaining_ess, remaining_load, "ess_p2p_sell_kw", "p2p_to_load_kw")
        stage(remaining_res, remaining_charge, "renewable_p2p_sell_kw", "p2p_to_battery_kw")
        stage(remaining_ess, remaining_charge, "ess_p2p_sell_kw", "p2p_to_battery_kw")

        price = 0.5 * (float(grid_sell_price) + float(grid_buy_price))
        pair_flows = {key: power * self.delta_t for key, power in flows_kw.items()}
        for mg in ids:
            by[mg]["p2p_net_kw"] = by[mg]["p2p_buy_kw"] - by[mg]["p2p_sell_kw"]
            by[mg]["p2p_transfer_payment"] = (
                by[mg]["p2p_sell_kw"] - by[mg]["p2p_buy_kw"]
            ) * self.delta_t * price
            by[mg]["ess_p2p_unmatched_kw"] = max(
                0.0, by[mg]["ess_p2p_offer_kw"] - by[mg]["ess_p2p_sell_kw"]
            )
        total = float(sum(pair_flows.values()))
        if abs(sum(item["p2p_net_kw"] for item in by.values())) > 1e-8:
            raise RuntimeError("Structured P2P energy is not conserved.")
        if abs(sum(item["p2p_transfer_payment"] for item in by.values())) > 1e-8:
            raise RuntimeError("Structured P2P payments are not budget balanced.")
        return CooperativeSharingResult(by, pair_flows, total, float(price))
