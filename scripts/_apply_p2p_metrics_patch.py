from pathlib import Path

path = Path("utils/metrics_collector.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one anchor, found {count}: {old[:80]!r}")
    text = text.replace(old, new, 1)


replace_once(
    "import numpy as np\n",
    "import numpy as np\n\nfrom core.tariffs import prices_from_row\n",
)

replace_once(
    '        self.p2p_total = 0.0\n        self.p2p_payment = {mg: 0.0 for mg in ids}\n',
    '''        self.p2p_total = 0.0
        self.p2p_unconstrained_total = 0.0
        self.p2p_blocked_by_capacity = 0.0
        self.physical_pairs = {
            f"{ids[i]}_{ids[j]}": (ids[i], ids[j])
            for i in range(len(ids)) for j in range(i + 1, len(ids))
        }
        self.pair_saturation_steps = {key: 0 for key in self.physical_pairs}
        self._last_p2p_capacity_diag: dict[str, Any] = {
            "unconstrained_total_energy_kwh": 0.0,
            "blocked_by_pair_capacity_kwh": 0.0,
            "pair_usage_kw": {key: 0.0 for key in self.physical_pairs},
            "pair_saturated": {key: False for key in self.physical_pairs},
        }
        self.p2p_payment = {mg: 0.0 for mg in ids}
''',
)

marker = "    def _constraint_step(self, info: Mapping[str, Any]) -> dict[str, Any]:\n"
helper = '''    def _p2p_capacity_diagnostics(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """Re-clear the same DR offers without pair limits for diagnostics only."""
        p2p = info["p2p"]
        actual_total = float(p2p.get("total_p2p_kwh", 0.0))
        pair_capacity = float(getattr(self.env.energy_pool, "pair_capacity_kw", float("inf")))
        pair_usage = {key: 0.0 for key in self.physical_pairs}
        flows = p2p.get("pair_flows_kwh", {})
        dt = max(float(self.env.delta_t), EPSILON)
        for key, (first, second) in self.physical_pairs.items():
            energy = float(flows.get(f"{first}->{second}", 0.0)) + float(
                flows.get(f"{second}->{first}", 0.0)
            )
            pair_usage[key] = energy / dt
        pair_saturated = {
            key: bool(
                math.isfinite(pair_capacity)
                and pair_usage[key] >= pair_capacity - self.tolerance
            )
            for key in self.physical_pairs
        }

        if not self.dr_enabled or not hasattr(
            self.env.energy_pool, "allocate_structured_unconstrained"
        ):
            return {
                "unconstrained_total_energy_kwh": actual_total,
                "blocked_by_pair_capacity_kwh": 0.0,
                "pair_usage_kw": pair_usage,
                "pair_saturated": pair_saturated,
            }

        plans = info["plans"]
        required = (
            "renewable_p2p_offer_kw", "ess_p2p_offer_kw",
            "p2p_load_demand_kw", "p2p_charge_demand_kw",
        )
        if any(any(field not in plans[mg] for field in required) for mg in self.agent_ids):
            return {
                "unconstrained_total_energy_kwh": actual_total,
                "blocked_by_pair_capacity_kwh": 0.0,
                "pair_usage_kw": pair_usage,
                "pair_saturated": pair_saturated,
            }

        if self.env.episode_df is None:
            raise RuntimeError("P2P capacity diagnostics require an active episode")
        row_index = min(max(self.steps - 1, 0), len(self.env.episode_df) - 1)
        row = self.env.episode_df.iloc[row_index]
        prices = {mg: prices_from_row(row, mg) for mg in self.agent_ids}
        unconstrained = self.env.energy_pool.allocate_structured_unconstrained(
            {mg: float(plans[mg]["renewable_p2p_offer_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["ess_p2p_offer_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["p2p_load_demand_kw"]) for mg in self.agent_ids},
            {mg: float(plans[mg]["p2p_charge_demand_kw"]) for mg in self.agent_ids},
            {mg: prices[mg][1] for mg in self.agent_ids},
            {mg: prices[mg][0] for mg in self.agent_ids},
        )
        unconstrained_total = float(unconstrained.total_p2p_kwh)
        return {
            "unconstrained_total_energy_kwh": unconstrained_total,
            "blocked_by_pair_capacity_kwh": max(0.0, unconstrained_total - actual_total),
            "pair_usage_kw": pair_usage,
            "pair_saturated": pair_saturated,
        }

'''
replace_once(marker, helper + marker)

replace_once(
    '        self.p2p_total += float(p2p.get("total_p2p_kwh", 0.0))\n        for key, energy in p2p.get("pair_flows_kwh", {}).items():\n',
    '''        self.p2p_total += float(p2p.get("total_p2p_kwh", 0.0))
        capacity_diag = self._p2p_capacity_diagnostics(info)
        self._last_p2p_capacity_diag = capacity_diag
        self.p2p_unconstrained_total += float(
            capacity_diag["unconstrained_total_energy_kwh"]
        )
        self.p2p_blocked_by_capacity += float(
            capacity_diag["blocked_by_pair_capacity_kwh"]
        )
        for key, saturated in capacity_diag["pair_saturated"].items():
            self.pair_saturation_steps[key] += int(bool(saturated))
        for key, energy in p2p.get("pair_flows_kwh", {}).items():
''',
)

replace_once(
    '        row: dict[str, Any] = {\n            "date": timestamp[:10], "hour": hour, "timestep": self.steps - 1,\n        }\n',
    '''        row: dict[str, Any] = {
            "date": timestamp[:10], "hour": hour, "timestep": self.steps - 1,
            "p2p_unconstrained_total_energy_kwh": float(
                self._last_p2p_capacity_diag["unconstrained_total_energy_kwh"]
            ),
            "p2p_blocked_by_pair_capacity_kwh": float(
                self._last_p2p_capacity_diag["blocked_by_pair_capacity_kwh"]
            ),
        }
        for pair, usage in self._last_p2p_capacity_diag["pair_usage_kw"].items():
            row[f"p2p_{pair}_usage_kw"] = float(usage)
            row[f"p2p_{pair}_saturated"] = int(bool(
                self._last_p2p_capacity_diag["pair_saturated"][pair]
            ))
''',
)

replace_once(
    '        p2p_stats = {\n            "total_energy_kwh": float(self.p2p_total),\n',
    '''        pair_capacity = float(
            getattr(self.env.energy_pool, "pair_capacity_kw", float("inf"))
        )
        p2p_stats = {
            "total_energy_kwh": float(self.p2p_total),
            "pair_capacity_kw": pair_capacity,
            "unconstrained_total_energy_kwh": float(self.p2p_unconstrained_total),
            "blocked_by_pair_capacity_kwh": float(self.p2p_blocked_by_capacity),
            "blocked_energy_rate": safe_ratio(
                self.p2p_blocked_by_capacity, self.p2p_unconstrained_total
            ),
            "pair_saturation_rate_by_pair": {
                pair: safe_ratio(count, self.steps)
                for pair, count in self.pair_saturation_steps.items()
            },
            "pair_saturation_step_count_by_pair": dict(self.pair_saturation_steps),
''',
)

path.write_text(text, encoding="utf-8")
