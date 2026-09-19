"""Post-training Q accuracy check using the selected best_model.pt checkpoint.

Do not confuse a small Bellman TD error with accurate realized returns. This
script replays matched prefixes under common random seeds, changes only the
focal agent's FIRST raw ESS/DR action, then follows the SAME frozen policy
to episode end. It compares Q(s, projected_joint_action) with the actual
discounted team reward over that same horizon. Run explicitly AFTER training;
the standard test/MILP/settlement/plot pipeline is not altered.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithms.registry import get_algorithm_spec
from envs.multi_microgrid_env import MultiMicrogridEnv
from experiments.training_pipeline import PROJECT_ROOT


def _prediction(algorithm, state, actions) -> dict[str, tuple[float, float, float]]:
    with torch.inference_mode():
        s = torch.as_tensor(state, dtype=torch.float32, device=algorithm.device)[None]
        raw = torch.as_tensor(
            np.stack([actions[mg] for mg in algorithm.agent_ids], axis=0),
            dtype=torch.float32, device=algorithm.device,
        )[None]
        projected, _ = algorithm.project_actions(s, raw)
        flat = projected.flatten(1)
        out = {}
        for mg in algorithm.agent_ids:
            q1, q2 = algorithm.critics[mg](s, flat)
            left, right = float(q1.item()), float(q2.item())
            out[mg] = (left, right, min(left, right))
        return out


def _reset_to_anchor(env, algorithm, *, day: str, seed: int, step: int, forecast_noise: bool):
    observations, info = env.reset(
        day=day, seed=seed, deterministic=True, forecast_noise=forecast_noise,
    )
    for _ in range(step):
        actions = algorithm.select_actions(observations, noise_std=0.0)
        observations, _, done, info = env.step(actions)
        if done:
            raise ValueError(f"Anchor step {step} exceeds episode length")
    return observations, env.global_state()


def _rollout(
    env, algorithm, *, day: str, seed: int, step: int, forecast_noise: bool,
    focal: str, channel: int, requested_action: float,
):
    observations, state = _reset_to_anchor(
        env, algorithm, day=day, seed=seed, step=step, forecast_noise=forecast_noise,
    )
    actions = algorithm.select_actions(observations, noise_std=0.0)
    original = float(actions[focal][channel])
    actions[focal] = actions[focal].copy()
    actions[focal][channel] = np.float32(requested_action)
    q_values = _prediction(algorithm, state, actions)

    discounted, discount = 0.0, 1.0
    done = False
    while not done:
        observations, rewards, done, _ = env.step(actions)
        discounted += discount * float(rewards[algorithm.agent_ids[0]])
        discount *= algorithm.gamma
        if not done:
            actions = algorithm.select_actions(observations, noise_std=0.0)
    return q_values, discounted, original, state


def diagnose(
    run_dir: Path, *, split: str = "test", days: int = 3,
    anchor_steps: tuple[int, ...] = (0, 8, 16),
    seeds_per_day: int = 2, action_delta: float = 0.6,
    forecast_noise: bool = True, focal: str | None = None,
    channel: int = 0, device: str = "cpu",
) -> dict:
    run_dir = Path(run_dir).resolve()
    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    if snapshot["algorithm"]["algorithm"]["name"] != "matd3_transformer_critic":
        raise ValueError("Run was not produced by matd3_transformer_critic")
    config = dict(snapshot["environment"])
    config["_project_root"] = str(PROJECT_ROOT)
    spec = get_algorithm_spec("matd3_transformer_critic")
    if split not in ("val", "test"):
        raise ValueError("Q accuracy checks require an independent val/test split")
    if days < 1 or seeds_per_day < 1 or not (0 < action_delta <= 2):
        raise ValueError("days, seeds_per_day and action_delta must be positive")
    if channel not in (0, 1):
        raise ValueError("channel must be 0 (ESS) or 1 (DR)")
    env = MultiMicrogridEnv(config, split=split)
    algorithm = spec.create(env, snapshot["algorithm"], device=device)
    algorithm.load(run_dir / "checkpoints" / "best_model.pt")
    focal = focal or algorithm.agent_ids[0]
    if focal not in algorithm.agent_ids:
        raise ValueError(f"Unknown focal microgrid: {focal}")
    dates = [str(day.date()) for day in env.data.valid_days[:days]]
    if not dates:
        raise ValueError(f"No valid days in {split} split")
    rows, paired = [], []
    skipped_physically_identical = 0
    for date_index, day in enumerate(dates):
        for trial in range(seeds_per_day):
            seed = 900_000 + date_index * 100 + trial
            for step in anchor_steps:
                if step < 0 or step >= env.episode_length:
                    raise ValueError(f"Anchor {step} outside episode length {env.episode_length}")
                baseline_obs, baseline_state = _reset_to_anchor(
                    env, algorithm, day=day, seed=seed,
                    step=step, forecast_noise=forecast_noise,
                )
                base_action = algorithm.select_actions(baseline_obs, noise_std=0.0)
                center = float(base_action[focal][channel])
                requested = (
                    max(-1.0, center - action_delta),
                    min(1.0, center + action_delta),
                )
                if abs(requested[1] - requested[0]) < 1e-7:
                    continue
                # Different raw commands can project to the same physical action
                # at ESS/DR limits. They are not meaningful candidate comparisons.
                with torch.no_grad():
                    s = torch.as_tensor(baseline_state, dtype=torch.float32, device=algorithm.device)[None]
                    effective = []
                    for value in requested:
                        candidate = {
                            mg: action.copy() for mg, action in base_action.items()
                        }
                        candidate[focal][channel] = np.float32(value)
                        raw = torch.as_tensor(
                            np.stack([candidate[mg] for mg in algorithm.agent_ids]),
                            dtype=torch.float32, device=algorithm.device,
                        )[None]
                        projected, _ = algorithm.project_actions(s, raw)
                        effective.append(projected)
                    distance = float((effective[1] - effective[0]).abs().max())
                if distance < 1e-7:
                    skipped_physically_identical += 1
                    continue
                pair = []
                for candidate_index, value in enumerate(requested):
                    predictions, realized, _, _ = _rollout(
                        env, algorithm, day=day, seed=seed, step=step,
                        forecast_noise=forecast_noise, focal=focal,
                        channel=channel, requested_action=value,
                    )
                    for critic_mg, (q1, q2, qmin) in predictions.items():
                        rows.append({
                            "day": day, "scenario_seed": seed, "anchor_step": step,
                            "focal": focal, "channel": "ess" if channel == 0 else "dr",
                            "candidate": candidate_index,
                            "raw_action": value, "effective_action_distance": distance,
                            "critic_mg": critic_mg,
                            "q1": q1, "q2": q2, "qmin": qmin,
                            "realized_discounted_team_reward": realized,
                            "qmin_absolute_error": abs(qmin - realized),
                            "q1_q2_abs_gap": abs(q1 - q2),
                        })
                    pair.append((predictions, realized))
                if len(pair) == 2:
                    (first_q, first_r), (second_q, second_r) = pair
                    for critic_mg in algorithm.agent_ids:
                        predicted_gap = second_q[critic_mg][2] - first_q[critic_mg][2]
                        realized_gap = second_r - first_r
                        paired.append({
                            "day": day, "scenario_seed": seed, "anchor_step": step,
                            "critic_mg": critic_mg,
                            "predicted_action_gap": predicted_gap,
                            "realized_action_gap": realized_gap,
                            "action_gap_absolute_error": abs(predicted_gap - realized_gap),
                            "same_preference": int(
                                predicted_gap * realized_gap > 0
                                or abs(predicted_gap) < 1e-8 and abs(realized_gap) < 1e-8
                            ),
                        })
    if not rows:
        raise ValueError("No valid Q accuracy samples")
    def mean(values):
        return float(np.mean(values)) if values else None
    summary = {
        "algorithm_slug": "matd3_transformer_critic",
        "checkpoint": "checkpoints/best_model.pt",
        "split": split,
        "forecast_noise": forecast_noise,
        "focal": focal,
        "action_channel": "ess" if channel == 0 else "dr",
        "action_delta": action_delta,
        "days": dates,
        "seeds_per_day": seeds_per_day,
        "anchor_steps": list(anchor_steps),
        "definition": (
            "Q(s,a) is compared to the discounted realized TEAM REWARD after "
            "one modified projected action followed by the frozen best-model policy. "
            "For a stochastic environment, realized return is a noisy sample "
            "of the Q expectation. This test does not prove causal use of forecasts."
        ),
        "q_return_mae": mean([x["qmin_absolute_error"] for x in rows]),
        "q1_q2_abs_gap": mean([x["q1_q2_abs_gap"] for x in rows]),
        "q_action_gap_mae": mean([x["action_gap_absolute_error"] for x in paired]),
        "q_return_rank_agreement": mean([x["same_preference"] for x in paired]),
        "q_samples": len(rows),
        "paired_cases": len(paired),
        "skipped_physically_identical_candidate_pairs": skipped_physically_identical,
        "by_critic": {
            mg: {
                "q_return_mae": mean([x["qmin_absolute_error"] for x in rows if x["critic_mg"] == mg]),
                "q_action_gap_mae": mean([x["action_gap_absolute_error"] for x in paired if x["critic_mg"] == mg]),
                "q_return_rank_agreement": mean([x["same_preference"] for x in paired if x["critic_mg"] == mg]),
            } for mg in algorithm.agent_ids
        },
    }
    with (run_dir / "critic_q_accuracy_cases.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (run_dir / "critic_q_action_pairs.csv").open("w", newline="", encoding="utf-8") as fp:
        if paired:
            writer = csv.DictWriter(fp, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
    (run_dir / "critic_q_accuracy.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate critic Q against realized returns")
    parser.add_argument("--run", type=Path, required=True, help="Completed working-run directory")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--seeds-per-day", type=int, default=2)
    parser.add_argument("--anchor-steps", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--action-delta", type=float, default=0.6)
    parser.add_argument("--focal", default=None)
    parser.add_argument("--channel", choices=("ess", "dr"), default="ess")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-forecast-noise", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(
        args.run, split=args.split, days=args.days,
        anchor_steps=tuple(args.anchor_steps),
        seeds_per_day=args.seeds_per_day, action_delta=args.action_delta,
        focal=args.focal, channel=0 if args.channel == "ess" else 1,
        device=args.device, forecast_noise=not args.no_forecast_noise,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
