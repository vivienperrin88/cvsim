"""Gaussian states, operations, channels and measurements (layer 2).

* :mod:`.states` — :class:`GaussianState` and its figures of merit (S04);
* :mod:`.symplectic` — gate matrices, quadratic Hamiltonians, Clements meshes, state preparation,
  mode-basis changes (S05);
* :mod:`.ops` — gates applied to states (S05);
* :mod:`.channels` — :class:`GaussianChannel` ``(X, Y)`` and phase diffusion (S06);
* :mod:`.ledger` — :class:`NoiseLedger`, the exact noise budget of a chain (S06).

Measurements follow in S07, after which the feature ``gaussian`` is listed in
:data:`cvsim.FEATURES`.
"""

from __future__ import annotations

from . import channels, ledger, ops, symplectic
from .channels import GaussianChannel
from .ledger import NoiseLedger
from .states import PURE_TOL, GaussianState

__all__ = [
    "PURE_TOL",
    "GaussianChannel",
    "GaussianState",
    "NoiseLedger",
    "channels",
    "ledger",
    "ops",
    "symplectic",
]
