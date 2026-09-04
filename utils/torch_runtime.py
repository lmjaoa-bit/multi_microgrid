"""Opt-in runtime settings; absent configuration preserves historical defaults."""
from numbers import Integral

import torch


def configure_torch_runtime(config, device):
    requested = config.get("runtime", {}).get("cpu_threads")
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, Integral) or requested < 1:
            raise ValueError("runtime.cpu_threads must be a positive integer")
        if torch.device(device).type == "cpu":
            torch.set_num_threads(int(requested))
    return {"cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}
