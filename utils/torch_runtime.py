"""Opt-in PyTorch runtime settings shared by training and runtime benchmarks."""
from numbers import Integral

import torch


def _positive_int(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def configure_torch_runtime(config, device):
    """Apply CPU intra/inter-op settings before model construction on any device.

    CPU thread settings still matter when the model is on CUDA because Python-side tensor
    preparation and any CPU PyTorch work share the same process. Inter-op threads can only be
    changed before PyTorch parallel work starts; callers should invoke this at process startup.
    """
    runtime = config.get("runtime", {})
    requested_threads = _positive_int(runtime.get("cpu_threads"), "runtime.cpu_threads")
    requested_interop = _positive_int(
        runtime.get("interop_threads"), "runtime.interop_threads"
    )

    if requested_threads is not None:
        torch.set_num_threads(requested_threads)
    if requested_interop is not None:
        current = torch.get_num_interop_threads()
        if current != requested_interop:
            try:
                torch.set_num_interop_threads(requested_interop)
            except RuntimeError as exc:
                raise RuntimeError(
                    "runtime.interop_threads must be configured before PyTorch parallel work starts"
                ) from exc

    return {
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
