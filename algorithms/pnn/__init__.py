from algorithms.pnn.algorithm import PNN
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations

install_single_dispatch_runtime_optimizations(
    PNN,
    variant="pnn",
)

__all__ = ["PNN"]
