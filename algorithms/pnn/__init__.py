from algorithms.pnn.algorithm import PNN
from algorithms.pnn.runtime import optimized_pnn_update
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations

install_single_dispatch_runtime_optimizations(
    PNN,
    variant="pnn",
)
PNN.update = optimized_pnn_update

__all__ = ["PNN"]
