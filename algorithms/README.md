# Algorithm package independence

Each directory directly below `algorithms/` is one independently maintained
algorithm. A complete algorithm consists of:

```text
algorithms/<algorithm_name>/       implementation, networks, replay and trainer
configs/algorithms/<algorithm_name>.yaml
scripts/train_<algorithm_name>.py
tests/                              algorithm-specific tests
```

The environment and physical system are shared through `envs/`, `core/`,
`components/`, `market/` and generic `utils/`. Algorithm packages may use those
shared, algorithm-neutral modules.

An improved algorithm must own every implementation detail that differs from
its predecessor. Do not subclass or import another directory under
`algorithms/` merely to reuse actor, critic, replay, update or trainer logic.
Copy the required implementation into the new package and evolve it there.
Deleting one algorithm directory and its matching config/training script must
not break any other algorithm.

CLI options are temporary overrides. Persistent experiment defaults—including
episodes, warm-up, network sizes, learning rates, exploration noise and target
smoothing—belong in the algorithm's YAML file. Each run copies both source
configs and the composed effective runtime config into its output directory.
