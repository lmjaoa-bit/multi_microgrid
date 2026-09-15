from pathlib import Path

path = Path("envs/multi_microgrid_env.py")
text = path.read_text(encoding="utf-8")

old = '''            plan.update({
                "base_load_kw": float(base_current[mg]["load_kw"]),
                "flexible_load_kw": float(dr_plans[mg]["flexible_load_kw"]),
                "actual_load_kw": float(actual_current[mg]["load_kw"]),
                "load_kw": float(actual_current[mg]["load_kw"]),
                "raw_ess_action": float(decoded[mg]["battery"]),
                "raw_dr_action": float(decoded[mg]["dr"]),
                "effective_ess_dispatch": float(effective[0, i, 0]),
                "effective_local_action": float(effective[0, i, 0]),
                "effective_trade_action": float(effective[0, i, 1]),
                "effective_dr_action": effective_dr,
                "ess_projection_delta": float(effective[0, i, 0]) - float(decoded[mg]["battery"]),
                "dr_projection_delta": dr_delta,
                "ess_projected": bool(direct["forced_dispatch"] or not direct["mask"]),
                "dr_projected": bool(abs(dr_delta) > 1e-7),
                "dr_requested_normalized": requested_dr,
'''

new = '''            raw_ess_action = float(decoded[mg]["battery"])
            battery_cfg = self.cfg_by_id[mg]["battery"]
            max_charge_kw = float(battery_cfg["max_charge_kw"])
            max_discharge_kw = float(battery_cfg["max_discharge_kw"])
            raw_ess_requested_power_kw = (
                raw_ess_action * max_discharge_kw
                if raw_ess_action >= 0.0
                else raw_ess_action * max_charge_kw
            )
            projected_ess_requested_power_kw = float(direct["requested_power"])
            ess_projection_delta_kw = (
                projected_ess_requested_power_kw - raw_ess_requested_power_kw
            )
            ess_power_scale_kw = max(max_charge_kw, max_discharge_kw, 1e-12)
            plan.update({
                "base_load_kw": float(base_current[mg]["load_kw"]),
                "flexible_load_kw": float(dr_plans[mg]["flexible_load_kw"]),
                "actual_load_kw": float(actual_current[mg]["load_kw"]),
                "load_kw": float(actual_current[mg]["load_kw"]),
                "raw_ess_action": raw_ess_action,
                "raw_dr_action": float(decoded[mg]["dr"]),
                "raw_ess_requested_power_kw": raw_ess_requested_power_kw,
                "projected_ess_requested_power_kw": projected_ess_requested_power_kw,
                "effective_ess_dispatch": float(effective[0, i, 0]),
                "effective_local_action": float(effective[0, i, 0]),
                "effective_trade_action": float(effective[0, i, 1]),
                "effective_dr_action": effective_dr,
                "ess_projection_delta_kw": ess_projection_delta_kw,
                "ess_projection_delta": ess_projection_delta_kw / ess_power_scale_kw,
                "dr_projection_delta": dr_delta,
                "ess_projected": bool(abs(ess_projection_delta_kw) > 1e-7),
                "dr_projected": bool(abs(dr_delta) > 1e-7),
                "dr_requested_normalized": requested_dr,
'''

count = text.count(old)
if count != 1:
    raise SystemExit(f"expected one ESS diagnostic anchor, found {count}")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
