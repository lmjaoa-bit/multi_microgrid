from __future__ import annotations

from typing import Any, Mapping


class MilpBenchmark:
    """Offline MILP benchmark interface.

    The model formulation is intentionally separated from the DRL pipeline.
    This class will expose the same evaluation contract after the optimization
    model is connected to the environment data and objective terms.
    """

    def __init__(self, env: Any = None, config: Mapping[str, Any] | None = None, **_: Any) -> None:
        self.env = env
        self.config = dict(config or {})
        self.replay = _EmptyReplay()

    @classmethod
    def from_env(cls, env: Any, config: Mapping[str, Any], **kwargs: Any) -> "MilpBenchmark":
        return cls(env=env, config=config, **kwargs)

    def select_actions(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError("MILP solve routine is under construction.")

    def update(self, *_: Any, **__: Any) -> None:
        return None

    def save(self, *_: Any, **__: Any) -> None:
        return None

    def load(self, *_: Any, **__: Any) -> None:
        return None


class _EmptyReplay:
    def add(self, *_: Any, **__: Any) -> None:
        return None
