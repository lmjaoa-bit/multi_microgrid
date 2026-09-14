from algorithms.matd3_transformer.algorithm import Matd3Transformer
from algorithms.runtime_helpers import install_single_dispatch_runtime_optimizations

install_single_dispatch_runtime_optimizations(
    Matd3Transformer,
    variant="matd3_transformer",
)

__all__ = ["Matd3Transformer"]
