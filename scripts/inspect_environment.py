from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.multi_microgrid_env import MultiMicrogridEnv  # noqa: E402


env = MultiMicrogridEnv(ROOT / "configs" / "multi_microgrid.yaml")
obs, info = env.reset(day=str(env.data.valid_days[0].date()))
print("Agents:", env.agent_ids)
print("Action dim per agent:", env.action_dim)
print("Observation dims:", env.observation_dims)
print("Global state dim:", env.global_state_dim)
print("Global state schema:", env.global_state_schema)
for mg in env.agent_ids:
    print(f"\n{mg} observation schema:")
    print(env.obs_builder.schema(mg))
