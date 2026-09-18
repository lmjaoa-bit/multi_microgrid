from algorithms.milp.solver import MilpBenchmark
from algorithms.registry import AlgorithmSpec, PROJECT_ROOT

ALGORITHM_SPEC = AlgorithmSpec(
    slug="milp",
    display_name="MILP",
    algorithm_class=MilpBenchmark,
    default_config_path=PROJECT_ROOT / "configs/algorithms/milp.yaml",
    description="Offline mixed-integer linear optimization benchmark for DRL comparison.",
    environment_overrides={"env": {"mode": "cooperative"}},
)
