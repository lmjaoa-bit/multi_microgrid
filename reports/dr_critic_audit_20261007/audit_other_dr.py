"""Diagnose mill/EV critics using the unchanged episode-1900 HVAC-audit snapshot.

This script makes no learning updates. Effective-action duplicates and forced
EV/no-availability states are excluded from value-ranking comparisons.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HVAC = ROOT / "reports/hvac_audit_20261007"
spec = importlib.util.spec_from_file_location("hvac_audit", HVAC / "audit.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
np, pd, torch = base.np, base.pd, base.torch


def write_json(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_source():
    source = json.loads((HVAC / "source_manifest.json").read_text())
    for relative, expected in source["source_hashes"].items():
        assert base.digest(ROOT / relative) == expected, f"Source changed: {relative}"
    for relative, expected in source["snapshot_hashes"].items():
        assert base.digest(HVAC / "source_snapshot" / relative) == expected
    return source


def baseline():
    source = verify_source()
    env, model, config = base.load()
    days = [str(x.date()) for x in env.data.valid_days]
    rows = []
    for index, day in enumerate(days):
        obs, _ = env.reset(day=day, seed=base.SEED + index, deterministic=True, forecast_noise=True)
        while True:
            t = env.t
            actions = model.select_actions(obs, noise_std=0.)
            records = []
            for agent in ["mg1", "mg3"]:
                device = env.dr_devices[agent]
                plan = device.plan(float(actions[agent][1]), t)
                low, high = device.feasible_bounds(t)
                if agent == "mg1":
                    low *= device.rated_power_kw
                    high *= device.rated_power_kw
                state = device.state(t)
                records.append(dict(date=day, hour=t, agent=agent,
                    raw=float(actions[agent][1]), power_kw=plan["flexible_load_kw"],
                    low_kw=low, high_kw=high, span_kw=high-low,
                    silo_t=state.get("silo_level_t", 0.),
                    connected_remaining_kwh=state.get("ev_remaining_battery_kwh", 0.),
                    hours_to_departure=state.get("ev_hours_to_departure", 0.),
                    deadline_pressure=state.get("deadline_pressure", 0.)))
            obs, rewards, done, info = env.step(actions)
            for record in records:
                agent = record["agent"]
                after = info["dr_devices"][agent]
                record.update(reward=rewards[agent], generalized_cost=info["generalized_system_cost"],
                    downstream_unserved_t=after.get("downstream_unserved_t", 0.),
                    inventory_min_violation_t=after.get("inventory_min_violation_t", 0.),
                    inventory_max_violation_t=after.get("inventory_max_violation_t", 0.),
                    ev_departure_shortfall_kwh=after.get("ev_departure_shortfall_kwh", 0.),
                    **info["reward_v2_breakdown"])
                rows.append(record)
            if done:
                break
        if (index+1) % 10 == 0:
            print(f"baseline {index+1}/{len(days)} days", flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 2640 and not frame.duplicated(["date", "hour", "agent"]).any()
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    old = pd.read_csv(HVAC / "hourly_forecast_true.csv")
    aligned = frame[frame.agent == "mg1"].merge(old[["date", "hour", "reward"]], on=["date", "hour"], suffixes=("", "_hvac"), validate="one_to_one")
    error = float(abs(aligned.reward-aligned.reward_hvac).max())
    assert error < 1e-10
    frame.to_csv(OUT / "dr_hourly.csv", index=False)
    code = ["components/industrial_mill.py", "components/ev_charging.py", "components/ev_sessions.py"]
    write_json("source_manifest.json", dict(parent_audit=str(HVAC), checkpoint_sha256=source["checkpoint_sha256"],
        best_episode=source["best_validation_episode"], source_run=source["source_run"],
        additional_source_hashes={x: base.digest(ROOT/x) for x in code},
        reference_replay_reward_max_error=error, split="test", days=55, seed=base.SEED,
        action_noise_std=0., forecast_noise=True, learning_updates=0))
    print("baseline complete, exact reward match", error, flush=True)


def select_states(frame):
    selections = []
    frame["month"] = pd.to_datetime(frame.date).dt.month
    for (agent, month), group in frame.groupby(["agent", "month"]):
        g = group[group.span_kw >= 2.].sort_values(["date", "hour"]).copy()
        g["ramp"] = g.groupby("date").power_kw.diff().abs().fillna(0.)
        if agent == "mg1":
            urgent = g.sort_values("silo_t")
            urgent_label = "low_inventory"
        else:
            urgent = g.sort_values(["hours_to_departure", "deadline_pressure"], ascending=[True, False])
            urgent_label = "near_departure"
        chosen = set()
        cuts = [("small", g.sort_values("power_kw")),
                ("large", g.sort_values("power_kw", ascending=False)),
                (urgent_label, urgent),
                ("large_ramp", g.sort_values("ramp", ascending=False))]
        for label, candidates in cuts:
            for _, row in candidates.iterrows():
                key = (row.date, int(row.hour))
                if key not in chosen:
                    selections.append(dict(agent=agent, month=int(month), date=row.date, hour=int(row.hour), kind=label))
                    chosen.add(key)
                    break
    return selections


def rollout(env, model, first):
    trial = copy.deepcopy(env)
    actions = {mg: a.copy() for mg, a in first.items()}
    total = 0.; discount = 1.; costs = {}; service = []
    while True:
        obs, rewards, done, info = trial.step(actions)
        total += discount * float(rewards["mg1"])
        discount *= model.gamma
        for name, value in info["reward_v2_breakdown"].items():
            if name.startswith("cost_"):
                costs[name] = costs.get(name, 0.) + value
        for mg in model.agent_ids:
            device = info["dr_devices"][mg]
            service.extend(float(device.get(k, 0.)) for k in ["comfort_degree_hours", "downstream_unserved_t", "inventory_min_violation_t", "inventory_max_violation_t"])
            service.append(float(info["settlements"][mg]["unmet_load_kw"]))
        if done:
            break
        actions = model.select_actions(obs, noise_std=0.)
    ev = trial.dr_devices["mg3"]
    events = {item["group_id"]: item["shortfall_kwh"] for item in ev.departure_events}
    for session in sorted(ev.sessions, key=lambda x: x["group_id"]):
        service.append(float(events[session["group_id"]]))
    return dict(total=total, costs=costs, service=np.asarray(service),
        ev_shortfall=sum(events.values()), final_inventory=trial.dr_devices["mg1"].silo_t)


def candidate_actions(env, agent, original):
    device = env.dr_devices[agent]
    current_raw = float(original[agent][1])
    current_power = device.plan(current_raw, env.t)["flexible_load_kw"]
    low, high = device.feasible_bounds(env.t)
    if agent == "mg1":
        scale = device.rated_power_kw
        to_raw = lambda power: 2*power/scale-1
    else:
        to_raw = lambda power: 2*(power-low)/(high-low)-1
    probes = [(current_raw, "reference")]
    probes.extend((x, "grid") for x in [-1., -.5, 0., .5, 1.])
    for delta in [-1., 1.]:
        probes.append((float(np.clip(to_raw(current_power+delta), -1, 1)), "local"))
    if agent == "mg1":
        cfg = env.cfg_by_id[agent]["dr"]
        probes.append((2*float(cfg["downstream_consumption_tph"])/float(cfg["rated_production_tph"])-1, "steady_production"))
    unique = []
    for raw, kind in probes:
        raw = float(np.float32(raw))
        power = device.plan(raw, env.t)["flexible_load_kw"]
        if any(abs(power-previous[1]) < 1e-5 for previous in unique):
            continue
        unique.append((raw, power, kind))
    return unique


def probes():
    verify_source()
    manifest = json.loads((OUT/"source_manifest.json").read_text())
    for file, expected in manifest["additional_source_hashes"].items():
        assert base.digest(ROOT/file) == expected
    frame = pd.read_csv(OUT/"dr_hourly.csv")
    selections = select_states(frame)
    write_json("probe_selection.json", selections)
    env, model, config = base.load()
    days = [str(x.date()) for x in env.data.valid_days]
    wanted = {}
    for item in selections:
        wanted.setdefault(item["date"], {}).setdefault(item["hour"], []).append(item)
    rows = []; completed = 0
    for day, hours in sorted(wanted.items()):
        obs, _ = env.reset(day=day, seed=base.SEED+days.index(day), deterministic=True, forecast_noise=True)
        while True:
            t = env.t
            actions = model.select_actions(obs, noise_std=0.)
            state = env.global_state()
            if t in hours:
                for item in hours[t]:
                    agent = item["agent"]; agent_index = model.agent_ids.index(agent)
                    candidates = candidate_actions(env, agent, actions)
                    raw_joint = np.stack([actions[mg] for mg in model.agent_ids])
                    candidate_joint = np.repeat(raw_joint[None], len(candidates), 0)
                    candidate_joint[:, agent_index, 1] = [x[0] for x in candidates]
                    states = torch.tensor(np.repeat(state[None], len(candidates), 0))
                    with torch.no_grad():
                        effective, _ = model.project_actions(states, torch.tensor(candidate_joint))
                        q1, q2 = model.critics[agent](states, effective.flatten(1))
                    scale = float(model.projector.dr[agent]["scale_kw"])
                    projected_power = effective[:, agent_index, 2].numpy()*scale
                    assert np.max(abs(projected_power-np.asarray([x[1] for x in candidates]))) < 1e-4
                    reference = rollout(env, model, actions)
                    live_state = env.global_state().copy()
                    for index, (raw, power, kind) in enumerate(candidates):
                        changed = {mg: a.copy() for mg, a in actions.items()}
                        changed[agent][1] = raw
                        alternative = reference if index == 0 else rollout(env, model, changed)
                        rows.append(dict(date=day, hour=t, agent=agent, selection_kind=item["kind"], candidate_kind=kind,
                            reference_raw=candidates[0][0], candidate_raw=raw,
                            reference_kw=candidates[0][1], candidate_kw=power,
                            power_delta_kw=power-candidates[0][1],
                            q1=float(q1[index]), q2=float(q2[index]),
                            dq1=float(q1[index]-q1[0]), dq2=float(q2[index]-q2[0]),
                            dqmin=float(torch.minimum(q1[index],q2[index])-torch.minimum(q1[0],q2[0])),
                            reference_return=reference["total"], candidate_return=alternative["total"],
                            gain=alternative["total"]-reference["total"],
                            service_no_worse=bool(np.all(alternative["service"] <= reference["service"]+1e-6)),
                            reference_ev_shortfall=reference["ev_shortfall"], candidate_ev_shortfall=alternative["ev_shortfall"],
                            reference_final_inventory=reference["final_inventory"], candidate_final_inventory=alternative["final_inventory"],
                            **{f"delta_{k}": alternative["costs"][k]-reference["costs"][k] for k in reference["costs"]}))
                    assert env.t == t and np.array_equal(env.global_state(), live_state)
                    completed += 1
                    pd.DataFrame(rows).to_csv(OUT/"counterfactual_probes.csv", index=False)
                    print(f"probe {completed}/{len(selections)} {agent} {day} h{t}: {len(candidates)} effective candidates", flush=True)
            obs, _, done, _ = env.step(actions)
            if done:
                break
    print("probes complete", len(rows), flush=True)


def summarize():
    source = verify_source()
    manifest = json.loads((OUT/"source_manifest.json").read_text())
    for file, expected in manifest["additional_source_hashes"].items():
        assert base.digest(ROOT/file) == expected
    frame = pd.read_csv(OUT/"dr_hourly.csv")
    probes = pd.read_csv(OUT/"counterfactual_probes.csv")
    assert not probes.duplicated(["date", "hour", "agent", "candidate_kw"]).any()
    errors = []
    for (agent, day, hour), group in probes.groupby(["agent", "date", "hour"]):
        baseline_suffix = frame[(frame.agent == agent)&(frame.date == day)&(frame.hour >= hour)].reward.sum()
        errors.append(abs(group.reference_return.iloc[0]-baseline_suffix))
    cost_cols = [c for c in probes if c.startswith("delta_cost_")]
    cost_error = float(abs(probes.gain+probes[cost_cols].sum(axis=1)/100).max())
    assert max(errors) < 1e-10 and cost_error < 1e-10
    results = []
    for agent, group in probes.groupby("agent"):
        local = group[np.isclose(abs(group.power_delta_kw),1.,atol=1e-5,rtol=0)]
        meaningful = local[local.gain.abs() > .001]
        broad = group[group.gain.abs() > .001]
        hourly = frame[frame.agent == agent]
        all_changed = group[group.power_delta_kw.abs() > 1e-5]
        results.append(dict(agent=agent, sampled_states=len(group.groupby(["date","hour"])),
            effective_candidate_rows=len(group), controllable_hours=int(sum(hourly.span_kw > 1e-5)),
            forced_hours=int(sum(hourly.span_kw <= 1e-5)),
            local_comparisons=len(local), meaningful_local_comparisons=len(meaningful),
            q1_wrong_sign=int(sum(meaningful.dq1*meaningful.gain < 0)),
            q2_wrong_sign=int(sum(meaningful.dq2*meaningful.gain < 0)),
            qmin_wrong_sign=int(sum(meaningful.dqmin*meaningful.gain < 0)),
            both_wrong_sign=int(sum((meaningful.dq1*meaningful.gain < 0)&(meaningful.dq2*meaningful.gain < 0))),
            broad_meaningful_comparisons=len(broad), broad_q1_wrong_sign=int(sum(broad.dq1*broad.gain < 0)),
            local_actual_median_abs_gain=float(local.gain.abs().median()),
            local_q1_median_abs_error=float((local.dq1-local.gain).abs().median()),
            safe_improvements_critic_rejects=int(sum((all_changed.gain > .001)&all_changed.service_no_worse&(all_changed.dq1 < 0))),
            harmful_actions_critic_prefers=int(sum((all_changed.gain < -.001)&(all_changed.dq1 > 0)))))
    wrong = probes[(probes.gain.abs() > .001)&(probes.dq1*probes.gain < 0)]
    wrong.to_csv(OUT/"q1_wrong_examples.csv", index=False)
    summary = dict(checkpoint_sha256=source["checkpoint_sha256"], best_episode=source["best_validation_episode"],
        days=55, action_noise_std=0., forecast_noise=True,
        results=results,
        checks=dict(parent_reference_reward_max_error=manifest["reference_replay_reward_max_error"],
            counterfactual_reference_suffix_max_error=max(errors), reward_cost_identity_max_error=cost_error,
            source_hashes_unchanged=True, effective_actions_deduplicated=True),
        caveats=["Same frozen intermediate episode-1900 model as HVAC audit; not the final trained policy.",
            "48 controllable states per agent, four state categories per month, not a random population error-rate sample.",
            "Only actual effective ±1 kW changes are counted as local comparisons; forced EV actions and clipped duplicates are excluded.",
            "All returns use the common team reward and unchanged deterministic policy until day end; TD3 targets include smoothing noise.",
            "Service checks compare each remaining hour's comfort, mill unserved material and inventory violations, unmet electricity, and each EV cohort's final departure deficit.",
            "Mill terminal_mode=none: changing final inventory within bounds is permitted by the current objective."])
    write_json("audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage", choices=["baseline", "probes", "summarize"])
    stage=parser.parse_args().stage
    {"baseline":baseline,"probes":probes,"summarize":summarize}[stage]()
