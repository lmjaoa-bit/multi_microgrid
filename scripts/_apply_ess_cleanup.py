from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8")


def replace_once(rel: str, old: str, new: str, marker: str) -> None:
    text = read(rel)
    if old in text:
        write(rel, text.replace(old, new, 1))
        return
    if marker in text:
        return
    raise RuntimeError(f"{rel}: replacement source not found")


replace_once(
    "envs/multi_microgrid_env.py",
    '''                floor, ceiling = battery.reachable_soc_envelope(
                    index,
                    self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=False,
                )
                neutral = battery.map_action(0.0, floor, ceiling)
''',
    '''                neutral = battery.map_action(0.0)
''',
    "neutral = battery.map_action(0.0)",
)
replace_once(
    "envs/multi_microgrid_env.py",
    '''                tolerance=tolerance,
                exogenous=actual_current[mg],
                enforce_terminal=False,
''',
    '''                tolerance=tolerance,
                exogenous=actual_current[mg],
''',
    "exogenous=actual_current[mg],\n            )",
)
replace_once(
    "envs/multi_microgrid_env.py",
    '''                tolerance=tolerance,
                enforce_terminal=self.config["terminal_soc"]["mode"] != "none",
                exogenous=current[mg],
''',
    '''                tolerance=tolerance,
                exogenous=current[mg],
''',
    "tolerance=tolerance,\n                exogenous=current[mg],",
)

replace_once(
    "core/global_state.py",
    '''            remaining_hours = max(self.episode_length - int(t), 0) * float(battery.delta_t)
            soc_gap = float(battery.soc) - float(battery.initial_soc)
            if remaining_hours > 0.0:
                if soc_gap >= 0.0:
                    required_kw = soc_gap * float(battery.capacity_kwh) * float(battery.eta_discharge) / remaining_hours
                    required_ratio = required_kw / max(float(battery.max_discharge_kw), 1e-12)
                else:
                    required_kw = -(-soc_gap) * float(battery.capacity_kwh) / max(float(battery.eta_charge) * remaining_hours, 1e-12)
                    required_ratio = required_kw / max(float(battery.max_charge_kw), 1e-12)
            else:
                required_ratio = 0.0
''',
    '''            terminal_enabled = self.config["terminal_soc"]["mode"] != "none"
            remaining_hours = max(self.episode_length - int(t), 0) * float(battery.delta_t)
            soc_gap = (float(battery.soc) - float(battery.initial_soc)) if terminal_enabled else 0.0
            if terminal_enabled and remaining_hours > 0.0:
                if soc_gap >= 0.0:
                    required_kw = soc_gap * float(battery.capacity_kwh) * float(battery.eta_discharge) / remaining_hours
                    required_ratio = required_kw / max(float(battery.max_discharge_kw), 1e-12)
                else:
                    required_kw = -(-soc_gap) * float(battery.capacity_kwh) / max(float(battery.eta_charge) * remaining_hours, 1e-12)
                    required_ratio = required_kw / max(float(battery.max_charge_kw), 1e-12)
            else:
                required_ratio = 0.0
            p0_time_ratio = (
                remaining_hours / max(float(self.episode_length), 1.0)
                if terminal_enabled else 0.0
            )
''',
    "p0_time_ratio = (",
)
replace_once(
    "core/global_state.py",
    '''                           previous["p2p_net_kwh"] / n["trade"], soc_gap,
                           remaining_hours / max(float(self.episode_length), 1.0),
                           required_ratio, abs(required_ratio)])
''',
    '''                           previous["p2p_net_kwh"] / n["trade"], soc_gap,
                           p0_time_ratio, required_ratio, abs(required_ratio)])
''',
    "p0_time_ratio, required_ratio",
)
replace_once(
    "core/global_state.py",
    '''                floor, ceiling = battery.reachable_soc_envelope(
                    t, self.episode_length,
                    tolerance=float(self.config["terminal_soc"]["tolerance"]),
                    enforce_terminal=False)
                neutral = battery.map_action(0.0, floor, ceiling)
''',
    '''                neutral = battery.map_action(0.0)
''',
    "neutral = battery.map_action(0.0)",
)

replace_once(
    "tests/test_instantaneous_default_and_p0.py",
    '''    for cls in (MATD3, Matd3Dispatch, PNN, Matd3Transformer, PnnTransformer):
        signature = inspect.signature(cls.update)
        assert "collect_diagnostics" in signature.parameters
''',
    '''    for cls in (Matd3Dispatch, PNN, Matd3Transformer, PnnTransformer):
        signature = inspect.signature(cls.update)
        assert "collect_diagnostics" in signature.parameters, cls.__name__
''',
    "assert \"collect_diagnostics\" in signature.parameters, cls.__name__",
)
replace_once(
    "tests/test_instantaneous_default_and_p0.py",
    '''    battery = env.cores["mg1"].battery
    battery.soc = battery.soc_min
''',
    '''    battery = env.cores["mg1"].battery
    assert not hasattr(battery, "reachable_soc_envelope")
    battery.soc = battery.soc_min
''',
    "assert not hasattr(battery, \"reachable_soc_envelope\")",
)

replace_once(
    "tests/test_single_dispatch.py",
    '''def test_forced_terminal_export_without_market():
    env, _ = build(disabled=True, terminal_mode="penalty")
    env.reset(seed=1, deterministic=True)
    env.t = 23
    for mg in env.agent_ids:
        env.cores[mg].battery.soc = .7
        for field in ("load_kw", "pv_kw", "wind_kw", "renewable_kw"):
            env._episode_exogenous[mg][field][23] = 0.
    env._last_global_state = env._build_global_state()
    _, _, done, info = env.step({mg: [-1.] for mg in env.agent_ids})
    assert done
    for mg in env.agent_ids:
        b = env.cores[mg].battery
        assert info["plans"][mg]["mandatory_export_kw"] > 0
        assert info["plans"][mg]["dispatch_forced"]
        assert info["plans"][mg]["ess_p2p_offer_kw"] == 0
        assert b.soc <= b.initial_soc + .030001
''',
    '''def test_terminal_penalty_does_not_force_last_hour_export_without_market():
    env, _ = build(disabled=True, terminal_mode="penalty")
    env.reset(seed=1, deterministic=True)
    env.t = 23
    for mg in env.agent_ids:
        env.cores[mg].battery.soc = .7
        for field in ("load_kw", "pv_kw", "wind_kw", "renewable_kw"):
            env._episode_exogenous[mg][field][23] = 0.
    env._last_global_state = env._build_global_state()
    _, _, done, info = env.step({mg: [-1.] for mg in env.agent_ids})
    assert done
    for mg in env.agent_ids:
        b = env.cores[mg].battery
        assert info["plans"][mg]["mandatory_export_kw"] == pytest.approx(0.0)
        assert not info["plans"][mg]["dispatch_forced"]
        assert info["plans"][mg]["ess_p2p_offer_kw"] == pytest.approx(0.0)
        assert abs(b.soc - b.initial_soc) > env.config["terminal_soc"]["tolerance"]
        assert info["terminal_soc_penalty"][mg] > 0.0
''',
    "test_terminal_penalty_does_not_force_last_hour_export_without_market",
)

print("remaining ESS strong-projection hooks removed")
