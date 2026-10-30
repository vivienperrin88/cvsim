"""Gaussian states, operations, channels and measurements (layer 2).

* :mod:`.states` — :class:`GaussianState` and its figures of merit (S04);
* :mod:`.symplectic` — gate matrices, quadratic Hamiltonians, Clements meshes, state preparation,
  mode-basis changes (S05);
* :mod:`.ops` — gates applied to states (S05).

Channels and measurements follow in S06–S07, after which the feature ``gaussian`` is listed in
:data:`cvsim.FEATURES`.
"""

from __future__ import annotations

from . import ops, symplectic
from .states import PURE_TOL, GaussianState

__all__ = ["PURE_TOL", "GaussianState", "ops", "symplectic"]
