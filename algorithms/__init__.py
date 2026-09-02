"""Multi-agent reinforcement-learning algorithms for the standalone project."""
from algorithms.base import MultiAgentAlgorithm, ReplayBufferProtocol
from algorithms.registry import AlgorithmSpec, available_algorithm_names, get_algorithm_spec

__all__ = [
    "AlgorithmSpec",
    "MultiAgentAlgorithm",
    "ReplayBufferProtocol",
    "available_algorithm_names",
    "get_algorithm_spec",
]
