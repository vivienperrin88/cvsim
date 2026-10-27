"""Effective (projected) Hamiltonians of a tensor train, from environments.

With the state in canonical form centred at site ``k``, the projected Schrodinger equation for
core ``k`` is ``i dA/dt = H_eff A`` with

    (H_eff A)[l, o, r] = sum L[l, l_w, l_a] W[l_w, o, i, r_w] A[l_a, i, r_a] R[r, r_w, r_a],

the same contraction as the variational fit, with bra = ket = psi. The two-site version acts on
merged cores ``(l, i1, i2, r)``; the bond (zero-site) version acts on the bond matrix between two
sites. The matvecs never build ``H_eff`` as a matrix.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core import linalg

__all__ = ["h_eff_bond", "h_eff_one_site", "h_eff_two_site"]


def h_eff_one_site(L: np.ndarray, W: np.ndarray, R: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def matvec(A: np.ndarray) -> np.ndarray:
        t = linalg.contract(L, [2], A, [0])  # (l, l_w, i, r_a)
        t = linalg.contract(t, [1, 2], W, [0, 2])  # (l, r_a, o, r_w)
        return linalg.contract(t, [1, 3], R, [2, 1]).reshape(L.shape[0], W.shape[1], R.shape[0])

    return matvec


def h_eff_two_site(L: np.ndarray, W1: np.ndarray, W2: np.ndarray, R: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def matvec(theta: np.ndarray) -> np.ndarray:
        t = linalg.contract(L, [2], theta, [0])  # (l, l_w, i1, i2, r_a)
        t = linalg.contract(t, [1, 2], W1, [0, 2])  # (l, i2, r_a, o1, w)
        t = linalg.contract(t, [1, 4], W2, [2, 0])  # (l, r_a, o1, o2, r_w)
        return linalg.contract(t, [1, 4], R, [2, 1])  # (l, o1, o2, r)

    return matvec


def h_eff_bond(L: np.ndarray, R: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """``(H C)[l, r] = sum L[l, w, l_a] C[l_a, r_a] R[r, w, r_a]`` (L at cut k+1, R at cut k)."""

    def matvec(C: np.ndarray) -> np.ndarray:
        t = linalg.contract(L, [2], C, [0])  # (l, w, r_a)
        return linalg.contract(t, [1, 2], R, [1, 2])  # (l, r)

    return matvec
