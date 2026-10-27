"""Physics-agnostic numerical core.

* :mod:`~qttlind.core.conventions` -- bit order and truncation rule
* :mod:`~qttlind.core.linalg`      -- the dense kernel, single seam to LAPACK
* :mod:`~qttlind.core.tt`          -- :class:`TensorTrain`
* :mod:`~qttlind.core.mpo`         -- :class:`MPO`, MPO-state products

Nothing else in the package may call ``numpy.linalg`` on a hot path: go through ``linalg``.
"""

from . import conventions, linalg
from .mpo import MPO
from .tt import TensorTrain

__all__ = ["MPO", "TensorTrain", "conventions", "linalg"]
