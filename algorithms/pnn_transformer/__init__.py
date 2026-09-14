from algorithms.pnn_transformer.algorithm import PnnTransformer
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations

install_single_dispatch_runtime_optimizations(
    PnnTransformer,
    variant="pnn_transformer",
)

__all__ = ["PnnTransformer"]
