from __future__ import annotations

from algorithms.matd3 import MATD3


class Matd3ShapleyQ(MATD3):
    """Experimental Matd3ShapleyQ variant. Keep algorithms/matd3 unchanged."""

    # Override only the behavior under experiment. If network definitions change,
    # place the new actor/critic modules in this package and import them here.
    pass
