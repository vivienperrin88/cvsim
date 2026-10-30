"""Gaussian states, operations, channels and measurements (layer 2).

S04 delivers :class:`GaussianState`; the symplectic library, channels and measurements follow in
S05–S07, after which the feature ``gaussian`` is listed in :data:`cvsim.FEATURES`.
"""

from __future__ import annotations

from .states import PURE_TOL, GaussianState

__all__ = ["PURE_TOL", "GaussianState"]
