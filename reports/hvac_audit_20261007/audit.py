"""Read-only HVAC diagnosis. Run from repository root; no learning updates.

MKL_THREADING_LAYER=SEQUENTIAL and OMP_NUM_THREADS=1 avoid duplicate OpenMP.
Results are diagnostic evidence, never formal-test or release artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
SLUG = "matd3_mlp_dueling_critic_h24_q_guided"
SEED = 42


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze():
    run = Path(json.loads((ROOT / "outputs/working_runs" / SLUG / "latest_run.json").read_text())["run_directory"])
    artifacts = run / "artifacts"
    target = OUT / "source_snapshot"
    target.mkdir(exist_ok=True)
    checkpoint = artifacts / "checkpoints/best_model.pt"
    for attempt in range(4):
        before = digest(checkpoint)
        content = checkpoint.read_bytes()
        after = digest(checkpoint)
        if before == after == hashlib.sha256(content).hexdigest():
            break
    else:
        raise RuntimeError("Checkpoint changed during snapshot; cannot diagnose consistently")
    (target / "best_model.pt").write_bytes(content)
    for name in ["config_snapshot.yaml", "run_metadata.json", "validation_history.json", "run_status.json"]:
        (target / name).write_bytes((artifacts / name).read_bytes())
    checkpoint_payload = torch.load(target / "best_model.pt", map_location="cpu", weights_only=True)
    validation = json.loads((target / "validation_history.json").read_text())
    code_paths = list((ROOT / "algorithms/matd3").glob("*.py")) + list((ROOT / "algorithms" / SLUG).glob("*.py"))
    code_paths += [ROOT / x for x in ["algorithms/diagnostic_sampling.py", "components/commercial_hvac.py", "components/hvac_thermal.py", "core/reward_v2.py", "core/joint_action_projection.py", "envs/multi_microgrid_env.py"]]
    config = yaml.safe_load((target / "config_snapshot.yaml").read_text(encoding="utf-8"))
    data_paths = [ROOT / config["environment"]["data"][f"{split}_path"] for split in ["train", "val", "test"]]
    manifest = dict(source_run=str(run), checkpoint_sha256=before,
        checkpoint_update_steps=checkpoint_payload.get("update_steps"),
        best_validation_episode=next((x["episode"] for x in reversed(validation) if x.get("is_best")), None),
        last_validation_episode=validation[-1]["episode"],
        status_at_capture=json.loads((target / "run_status.json").read_text())["status"],
        seed=SEED, policy_noise_std=0.0, split="test", training_started_by_audit=False,
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in code_paths + data_paths},
        snapshot_hashes={p.name: digest(p) for p in target.iterdir()},
        note="Frozen intermediate best model from active training; audit evaluation is not formal test. Historical plot refers to removed feasible-actor version.")
    write_json("source_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def load():
    snapshot = yaml.safe_load((OUT / "source_snapshot/config_snapshot.yaml").read_text(encoding="utf-8"))
    env = MultiMicrogridEnv(snapshot["environment"], split="test")
    model = get_algorithm_spec(SLUG).create(env, snapshot["algorithm"], device="cpu", seed=SEED)
    model.load(OUT / "source_snapshot/best_model.pt")
    for module in (model.actors, model.critics, model.target_actors, model.target_critics):
        module.eval()
    return env, model, snapshot


def day_metrics(frame, threshold=1.0):
    rows = []
    for day, group in frame.groupby("date", sort=True):
        group = group.sort_values("hour")
        assert len(group) == 24 and group.hour.nunique() == 24
        power = group.power_kw.to_numpy()
        signs = np.where(power > threshold, 1, np.where(power < -threshold, -1, 0))
        nonzero = signs[signs != 0]
        rows.append(dict(date=day, threshold_kw=threshold,
            both_modes=bool((signs == 1).any() and (signs == -1).any()),
            adjacent_reversals=int(np.sum(signs[1:] * signs[:-1] == -1)),
            reversals_ignoring_off=int(np.sum(nonzero[1:] * nonzero[:-1] == -1)),
            sign_flips_raw=int(np.sum(power[1:] * power[:-1] < 0)),
            micro_hours=int(np.sum((np.abs(power) > 1e-6) & (np.abs(power) < threshold))),
            off_hours=int(np.sum(np.abs(power) <= 1e-6)),
            variation_kw=float(np.abs(np.diff(power)).sum()),
            hvac_energy_kwh=float(np.abs(power).sum()),
            comfort_degree_hours=float(group.comfort_degree_hours.sum()),
            generalized_cost=float(group.generalized_cost.sum()),
            reward=float(group.reward.sum())))
    return pd.DataFrame(rows)


def baseline():
    if not (OUT / "source_manifest.json").exists():
        freeze()
    env, model, snapshot = load()
    days = [str(day.date()) for day in env.data.valid_days]
    all_rows = []
    for noise in [True, False]:
        rows = []
        for i, day in enumerate(days):
            obs, _ = env.reset(day=day, seed=SEED+i, deterministic=True, forecast_noise=noise)
            done = False
            while not done:
                t = env.t
                actions = model.select_actions(obs, noise_std=0.0)
                plan = env.dr_devices["mg2"].plan(float(actions["mg2"][1]), t)
                exo = env._exogenous_at(t)["mg2"]
                obs, rewards, done, info = env.step(actions)
                rows.append(dict(date=day, hour=t, forecast_noise=noise,
                    raw=float(actions["mg2"][1]), power_kw=plan["effective_signed_power_kw"],
                    air_before=plan["t_air_before_c"], air_after=info["dr_devices"]["mg2"]["t_air_c"],
                    mass_before=plan["t_mass_before_c"],
                    comfort_lower=info["dr_devices"]["mg2"]["comfort_lower_c"],
                    comfort_upper=info["dr_devices"]["mg2"]["comfort_upper_c"],
                    comfort_degree_hours=info["dr_devices"]["mg2"]["comfort_degree_hours"],
                    reward=float(rewards["mg2"]), generalized_cost=info["generalized_system_cost"],
                    **info["reward_v2_breakdown"]))
            if (i+1) % 10 == 0:
                print(f"baseline forecast_noise={noise} {i+1}/{len(days)} days", flush=True)
        frame = pd.DataFrame(rows)
        assert not frame.duplicated(["date", "hour"]).any()
        assert np.isfinite(frame.select_dtypes(include="number")).all().all()
        frame.to_csv(OUT / f"hourly_forecast_{str(noise).lower()}.csv", index=False)
        metrics = pd.concat([day_metrics(frame, x) for x in [0.1, 1.0, 3.0]])
        metrics["forecast_noise"] = noise
        metrics.to_csv(OUT / f"daily_forecast_{str(noise).lower()}.csv", index=False)
        all_rows.append(metrics)
    all_metrics = pd.concat(all_rows)
    summary = []
    for (noise, threshold), group in all_metrics.groupby(["forecast_noise", "threshold_kw"]):
        summary.append(dict(forecast_noise=bool(noise), threshold_kw=threshold, days=len(group),
            both_modes_days=int(group.both_modes.sum()), adjacent_reversals=int(group.adjacent_reversals.sum()),
            reversals_ignoring_off=int(group.reversals_ignoring_off.sum()),
            raw_sign_flips=int(group.sign_flips_raw.sum()), micro_hours=int(group.micro_hours.sum()),
            off_hours=int(group.off_hours.sum()), mean_daily_cost=float(group.generalized_cost.mean()),
            mean_comfort_degree_hours=float(group.comfort_degree_hours.mean())))
    write_json("baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def historical():
    summaries=[]
    for slug in ["matd3_dispatch_q_guided", "matd3_dr_counterfactual"]:
        for run in sorted((ROOT / "outputs/working_runs" / slug / "runs").iterdir()):
            if not run.is_dir(): continue
            a=run/"artifacts" if (run/"artifacts").exists() else run
            paths=sorted((a/"test_hourly_trajectories").glob("*.csv"))
            if len(paths)!=55: continue
            cfg=yaml.safe_load((a/"config_snapshot.yaml").read_text(encoding="utf-8"))
            frames=[]
            for path in paths:
                f=pd.read_csv(path)
                cols=[c for c in f.columns if "mg2" in c and any(x in c for x in ["heat", "cool", "signed", "comfort", "dr_raw"])]
                if not frames: print("historical columns",str(run),cols,flush=True)
                frames.append(f)
            frame=pd.concat(frames,ignore_index=True)
            frame.to_csv(OUT/f"historical_{run.name}.csv",index=False)
            summaries.append(dict(run=str(run),slug=slug,days=len(paths),rows=len(frame),
                thermal_discretization=next(x["dr"].get("thermal_discretization","legacy_euler") for x in cfg["environment"]["microgrids"] if x["id"]=="mg2"),
                episodes=cfg["algorithm"].get("training",{}).get("episodes"),
                source_files={str(p.relative_to(ROOT)):digest(p) for p in paths+[a/"config_snapshot.yaml",a/"test_summary.json"]}))
    write_json("historical_sources.json",summaries)


def archived():
    archive=ROOT.parent/"multi_microgrid_rl_standalone_backups/artifacts_20261007_184857/research_artifacts_before_cleanup.zip"
    name="DUELINGFEASIBLEACTOR_2026-10-05_11-07-08_seed42"
    prefix=f"outputs/working_runs/matd3_dueling_feasible_actor/runs/{name}/"
    hashes={};frames=[]
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if member.startswith(prefix+"test_hourly_trajectories/") and member.endswith(".csv"):
                content=z.read(member)
                hashes[member]=hashlib.sha256(content).hexdigest()
                import io
                frames.append(pd.read_csv(io.BytesIO(content)))
        for file in ["config_snapshot.yaml","run_metadata.json","test_summary.json"]:
            content=z.read(prefix+file)
            (OUT/("archived_"+file)).write_bytes(content)
            hashes[prefix+file]=hashlib.sha256(content).hexdigest()
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(OUT/f"historical_{name}.csv",index=False)
    write_json("archived_sources.json",dict(archive=str(archive),source_run=name,days=len(frames),rows=len(frame),member_hashes=hashes))
    print("archived source",name,"days",len(frames),"columns",[x for x in frame.columns if "mg2" in x and any(s in x for s in ["signed","heat","cool"])],flush=True)


def deadband():
    env,model,snapshot=load()
    days=[str(x.date()) for x in env.data.valid_days]
    frames=[]
    for threshold in [1.,3.]:
        rows=[]
        for i,day in enumerate(days):
            obs,_=env.reset(day=day,seed=SEED+i,deterministic=True,forecast_noise=True)
            while True:
                t=env.t
                actions=model.select_actions(obs,noise_std=0.)
                original=float(actions["mg2"][1])
                if abs(original*35)<threshold:actions["mg2"][1]=0.
                plan=env.dr_devices["mg2"].plan(float(actions["mg2"][1]),t)
                obs,rewards,done,info=env.step(actions)
                rows.append(dict(date=day,hour=t,threshold_kw=threshold,original_raw=original,
                    raw=float(actions["mg2"][1]),power_kw=plan["effective_signed_power_kw"],
                    comfort_degree_hours=info["dr_devices"]["mg2"]["comfort_degree_hours"],
                    generalized_cost=info["generalized_system_cost"],reward=float(rewards["mg2"])))
                if done:break
            if (i+1)%10==0:print(f"deadband {threshold} kW {i+1}/{len(days)}",flush=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(OUT/f"deadband_{int(threshold)}kw_hourly.csv",index=False)
        metrics=day_metrics(frame,1.)
        metrics["intervention_threshold_kw"]=threshold
        frames.append(metrics)
    pd.concat(frames).to_csv(OUT/"deadband_daily.csv",index=False)
    print("deadband complete",flush=True)


def rollout(env, model, first):
    trial = copy.deepcopy(env)
    actions = {mg: a.copy() for mg,a in first.items()}
    total=0.0
    discount=1.0
    costs={}
    service=[]
    steps=0
    while True:
        obs,rewards,done,info=trial.step(actions)
        total+=discount*float(rewards["mg2"])
        discount*=model.gamma
        for key,value in info["reward_v2_breakdown"].items():
            if key.startswith("cost_"): costs[key]=costs.get(key,0.0)+value
        service.append(info["dr_devices"]["mg2"]["comfort_degree_hours"])
        steps+=1
        if done: break
        actions=model.select_actions(obs,noise_std=0.0)
    return total,costs,np.asarray(service),steps


def probes():
    env,model,snapshot=load()
    frame=pd.read_csv(OUT/"hourly_forecast_true.csv")
    frame["month"]=pd.to_datetime(frame.date).dt.month
    # Four states per month: actual reversal, small action, large action, late day.
    # First choose one reversal state if present; no fabricated switches.
    selections=[]
    for month,group in frame.groupby("month"):
        ordered=group.sort_values(["date","hour"]).copy()
        previous=ordered.groupby("date").power_kw.shift(1)
        ordered["previous_raw"]=ordered.groupby("date").raw.shift(1)
        reversal=ordered[(previous*ordered.power_kw<0)&(previous.abs()>1)&(ordered.power_kw.abs()>1)]
        chosen={}
        for label,candidates in [
            ("reversal",reversal.sort_values("power_kw",key=lambda x:x.abs(),ascending=False)),
            ("small",ordered.sort_values("power_kw",key=lambda x:x.abs())),
            ("large",ordered.sort_values("power_kw",key=lambda x:x.abs(),ascending=False)),
            ("late",ordered[ordered.hour>=20].sort_values("power_kw",key=lambda x:x.abs()))]:
            for _,r in candidates.iterrows():
                key=(r.date,int(r.hour))
                if key not in chosen:
                    chosen[key]=dict(date=r.date,hour=int(r.hour),kind=label)
                    break
        selections.extend(chosen.values())
    write_json("probe_selection.json",selections)
    rows=[]
    rng=np.random.default_rng(20261007)
    all_days=[str(x.date()) for x in env.data.valid_days]
    requested={}
    for item in selections:requested.setdefault(item["date"],{})[item["hour"]]=item["kind"]
    for day,times in sorted(requested.items()):
        obs,_=env.reset(day=day,seed=SEED+all_days.index(day),deterministic=True,forecast_noise=True)
        prev=None
        while True:
            t=env.t
            actions=model.select_actions(obs,noise_std=0.0)
            if t in times:
                state=env.global_state()
                raw=float(actions["mg2"][1])
                candidates=[-1.,-.5,-.15,-.05,0.,.05,.15,.5,1.,raw,
                    float(np.clip(raw-1/35,-1,1)),float(np.clip(raw+1/35,-1,1))]
                if prev is not None:candidates.append(prev)
                candidates=list(dict.fromkeys(candidates))
                joint=np.stack([actions[mg] for mg in model.agent_ids])
                raw_batch=np.repeat(joint[None],len(candidates),axis=0)
                raw_batch[:,1,1]=candidates
                states=torch.tensor(np.repeat(state[None],len(candidates),axis=0),dtype=torch.float32)
                with torch.no_grad():
                    effective,_=model.project_actions(states,torch.tensor(raw_batch))
                    q1,q2=model.critics["mg2"](states,effective.flatten(1))
                baseline_index=candidates.index(raw)
                base_return,base_cost,base_service,_=rollout(env,model,actions)
                live_t=env.t;live_state=env.global_state().copy()
                # TD3 target smoothing: next-state target-policy analysis at this state.
                # Compare identical states; all-channel and HVAC-only noise separated.
                with torch.no_grad():
                    target=np.stack([model.target_actors[mg](torch.tensor(obs[mg])[None]).numpy()[0] for mg in model.agent_ids])
                    for mode in ["hvac_only","all_channels"]:
                        batch=np.repeat(target[None],512,axis=0)
                        noise=np.clip(rng.normal(0,model.target_policy_noise,batch.shape),-model.target_noise_clip,model.target_noise_clip)
                        if mode=="hvac_only":
                            mask=np.zeros_like(noise);mask[:,1,1]=noise[:,1,1];noise=mask
                        batch=np.clip(batch+noise,-1,1)
                        sample_states=torch.tensor(np.repeat(state[None],513,axis=0))
                        sample_raw=torch.tensor(np.concatenate([target[None],batch]),dtype=torch.float32)
                        projected,_=model.project_actions(sample_states,sample_raw)
                        tq1,tq2=model.target_critics["mg2"](sample_states,projected.flatten(1))
                        tq=torch.minimum(tq1,tq2).numpy().ravel()
                        prefix=f"noise_{mode}_"
                        # store once on every candidate for grouping convenience
                        globals_noise={prefix+"q_shift":float(tq[1:].mean()-tq[0]),prefix+"q_std":float(tq[1:].std()),
                            prefix+"sign_cross_rate":float(np.mean(batch[:,1,1]*target[1,1]<0)),
                            prefix+"target_raw":float(target[1,1])}
                        if mode=="hvac_only":noise_metrics=globals_noise
                        else:noise_metrics.update(globals_noise)
                for index,candidate in enumerate(candidates):
                    changed={mg:a.copy() for mg,a in actions.items()};changed["mg2"][1]=candidate
                    ret,cost,service,steps=(base_return,base_cost,base_service,len(base_service)) if index==baseline_index else rollout(env,model,changed)
                    rows.append(dict(date=day,hour=t,kind=times[t],reference_raw=raw,candidate_raw=candidate,
                        candidate_kw=candidate*35,reference_kw=raw*35,previous_raw=prev,
                        q1=float(q1[index]),q2=float(q2[index]),
                        dq1=float(q1[index]-q1[baseline_index]),dq2=float(q2[index]-q2[baseline_index]),
                        dqmin=float(min(q1[index],q2[index])-min(q1[baseline_index],q2[baseline_index])),
                        reference_return=base_return,candidate_return=ret,gain=ret-base_return,
                        comfort_gain=float(base_service.sum()-service.sum()),
                        comfort_no_worse=bool(np.all(service<=base_service+1e-6)),
                        rollout_steps=steps,**{f"delta_{k}":cost[k]-base_cost[k] for k in cost},**noise_metrics))
                assert env.t==live_t and np.array_equal(live_state,env.global_state())
                pd.DataFrame(rows).to_csv(OUT/"counterfactual_probes.csv",index=False)
                print(f"probe {day} hour {t}: {len(candidates)} candidates, total states {len(set((r['date'],r['hour']) for r in rows))}",flush=True)
            prev=float(actions["mg2"][1])
            obs,_,done,_=env.step(actions)
            if done:break
    print("probes complete",len(selections),len(rows),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["baseline","historical","archived","deadband","probes"])
    stage=parser.parse_args().stage
    {"baseline":baseline,"historical":historical,"archived":archived,"deadband":deadband,"probes":probes}[stage]()
