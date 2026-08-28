from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.conditional_matd3 import (  # noqa: E402
    ConditionalMATD3Trainer,
    load_runtime_config,
)
from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402


if __name__ == "__main__":
    env = MultiMicrogridEnv(load_runtime_config(), split="train")
    trainer = ConditionalMATD3Trainer.from_env(env, device="cpu")
    # A smoke test only: force early updates after a tiny warm-up. This is not a
    # research training run and should not be used as a performance result.
    trainer.warmup_steps = 24
    trainer.batch_size = 16
    for day in env.data.valid_days[:2]:
        result = trainer.run_episode(day=str(day.date()), training=True, seed=123)
        print(day.date(), result["market_trade_kwh"], result["renewable_curtailment_kwh"])
    print("replay_size:", len(trainer.replay))
    print("last_update_keys:", sorted(result["last_update"]))
