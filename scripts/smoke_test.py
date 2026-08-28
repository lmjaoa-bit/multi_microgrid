from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402


def run(p2p_enabled: bool):
    env = MultiMicrogridEnv(ROOT / "configs" / "multi_microgrid.yaml", split="train")
    env.p2p_enabled = p2p_enabled
    obs, info = env.reset(day=str(env.data.valid_days[0].date()))
    totals = {"trade": 0.0, "curtail": 0.0, "cost": 0.0}
    done = False
    # battery idle, generator off, midpoint bid
    actions = {mg: np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32) for mg in env.agent_ids}
    while not done:
        obs, rewards, done, info = env.step(actions)
        totals["trade"] += float(info["market"]["total_trade_kwh"])
        totals["curtail"] += sum(float(s["renewable_curtailment_kw"]) * env.delta_t for s in info["settlements"].values())
        totals["cost"] += sum(float(c["total_cost"]) for c in info["costs"].values())
    return totals


if __name__ == "__main__":
    off = run(False)
    on = run(True)
    print("No P2P:", off)
    print("P2P   :", on)
