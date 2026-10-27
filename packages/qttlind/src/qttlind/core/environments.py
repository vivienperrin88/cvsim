"""Environments: partial contractions of ``<phi| W |psi>`` used by sweeping algorithms.

For a bra ``phi``, an MPO ``W`` and a ket ``psi`` of the same length, the **left environment**
at cut ``k`` is the contraction of sites ``0..k-1``; it is a 3-index array ``(r_phi, r_w, r_psi)``.
The right environment at cut ``k`` contracts sites ``k+1..L-1`` and has axes
``(l_phi, l_w, l_psi)``. Both start from the trivial ``1`` of shape ``(1, 1, 1)``.

Contraction order (``L . A``, then ``. W``, then ``. conj(phi)``) costs
``O(d chi^3 D + d^2 chi^2 D^2)`` per site. Environments are cached by
:class:`EnvironmentCache`, which recomputes only what an update invalidates.

Used by the variational MPO fit (:func:`qttlind.core.mpo.apply_mpo` with ``method="fit"``), the
effective Hamiltonians of TDVP (:mod:`qttlind.closed.effective`) and the ALS linear solver of
Crank--Nicolson.
"""

from __future__ import annotations

import numpy as np

from ..errors import ShapeError
from . import linalg
from .mpo import MPO
from .tt import TensorTrain

__all__ = ["EnvironmentCache", "extend_left", "extend_left_overlap", "extend_right", "extend_right_overlap"]


def extend_left(env: np.ndarray, bra: np.ndarray, op: np.ndarray, ket: np.ndarray) -> np.ndarray:
    """Left environment one site further.

    ``L'[r_phi, r_w, r_a] = sum conj(bra[l_phi, o, r_phi]) L[l_phi, l_w, l_a] W[l_w, o, i, r_w] ket[l_a, i, r_a]``.
    """
    t = linalg.contract(env, [2], ket, [0])  # (l_phi, l_w, i, r_a)
    t = linalg.contract(t, [1, 2], op, [0, 2])  # (l_phi, r_a, o, r_w)
    t = linalg.contract(bra.conj(), [0, 1], t, [0, 2])  # (r_phi, r_a, r_w)
    return t.transpose(0, 2, 1)


def extend_right(env: np.ndarray, bra: np.ndarray, op: np.ndarray, ket: np.ndarray) -> np.ndarray:
    """Right environment one site further.

    ``R'[l_phi, l_w, l_a] = sum conj(bra[l_phi, o, r_phi]) W[l_w, o, i, r_w] ket[l_a, i, r_a] R[r_phi, r_w, r_a]``.
    """
    t = linalg.contract(ket, [2], env, [2])  # (l_a, i, r_phi, r_w)
    t = linalg.contract(op, [2, 3], t, [1, 3])  # (l_w, o, l_a, r_phi)
    t = linalg.contract(bra.conj(), [1, 2], t, [1, 3])  # (l_phi, l_w, l_a)
    return t


def extend_left_overlap(env: np.ndarray, bra: np.ndarray, ket: np.ndarray) -> np.ndarray:
    """Overlap version (no operator): ``E'[r_phi, r_a] = sum conj(bra) E ket``."""
    t = linalg.contract(env, [1], ket, [0])  # (l_phi, s, r_a)
    return linalg.contract(bra.conj(), [0, 1], t, [0, 1])


def extend_right_overlap(env: np.ndarray, bra: np.ndarray, ket: np.ndarray) -> np.ndarray:
    t = linalg.contract(ket, [2], env, [1])  # (l_a, s, r_phi)
    return linalg.contract(bra.conj(), [1, 2], t, [1, 2])  # (l_phi, l_a)


class EnvironmentCache:
    """Left and right environments of ``<bra| op |ket>``, computed lazily and cached.

    ``left(k)`` contracts sites ``0..k-1``; ``right(k)`` contracts sites ``k+1..L-1``.
    :meth:`update_bra` / :meth:`update_ket` replace one core and invalidate the environments
    that depend on it: ``left(j)`` for ``j > k`` and ``right(j)`` for ``j < k``.
    """

    def __init__(self, bra: TensorTrain, op: MPO, ket: TensorTrain):
        if not (bra.length == op.length == ket.length):
            raise ShapeError(f"lengths differ: bra {bra.length}, op {op.length}, ket {ket.length}")
        if op.dims_out != bra.dims or op.dims_in != ket.dims:
            raise ShapeError(f"op dims {op.dims_out} x {op.dims_in} do not match bra {bra.dims} / ket {ket.dims}")
        self.bra = list(bra.cores)
        self.op = list(op.cores)
        self.ket = list(ket.cores)
        self.length = bra.length
        one = np.ones((1, 1, 1), dtype=np.complex128)
        self._left: dict[int, np.ndarray] = {0: one}
        self._right: dict[int, np.ndarray] = {self.length - 1: one}

    def left(self, k: int) -> np.ndarray:
        if k in self._left:
            return self._left[k]
        start = max(j for j in self._left if j < k)
        env = self._left[start]
        for j in range(start, k):
            env = extend_left(env, self.bra[j], self.op[j], self.ket[j])
            self._left[j + 1] = env
        return env

    def right(self, k: int) -> np.ndarray:
        if k in self._right:
            return self._right[k]
        start = min(j for j in self._right if j > k)
        env = self._right[start]
        for j in range(start, k, -1):
            env = extend_right(env, self.bra[j], self.op[j], self.ket[j])
            self._right[j - 1] = env
        return env

    def _invalidate(self, k: int) -> None:
        self._left = {j: e for j, e in self._left.items() if j <= k}
        self._right = {j: e for j, e in self._right.items() if j >= k}

    def update_bra(self, k: int, core: np.ndarray) -> None:
        self.bra[k] = core
        self._invalidate(k)

    def update_ket(self, k: int, core: np.ndarray) -> None:
        self.ket[k] = core
        self._invalidate(k)

    def update_both(self, k: int, core: np.ndarray) -> None:
        """Same core in bra and ket (TDVP: ``<psi|H|psi>``)."""
        self.bra[k] = core
        self.ket[k] = core
        self._invalidate(k)

    def value(self) -> complex:
        """The full contraction ``<bra|op|ket>``."""
        env = extend_left(self.left(self.length - 1), self.bra[-1], self.op[-1], self.ket[-1])
        return complex(env.reshape(-1)[0])
