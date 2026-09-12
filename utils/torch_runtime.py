"""Opt-in runtime settings; absent configuration preserves historical defaults."""
from numbers import Integral

import torch


def configure_torch_runtime(config, device):
    requested = config.get("runtime", {}).get("cpu_threads")
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, Integral) or requested < 1:
            raise ValueError("runtime.cpu_threads must be a positive integer")
        # Environment dynamics and projection still execute CPU tensor work
        # when Actor/Critic modules live on CUDA.  Apply the CPU setting for
        # every device to avoid severe tiny-op thread-pool overhead.
        torch.device(device)  # Validate the requested device syntax.
        torch.set_num_threads(int(requested))
    return {"cpu_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}
