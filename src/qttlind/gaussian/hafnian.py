"""Photon-counting statistics: hafnians and click probabilities.

**Why hafnians.** Expanding the Bargmann function (:mod:`.fock`), the coefficient of ``z^N`` in
``exp(z^T A z / 2 + b^T z)`` is a sum over all ways of pairing the ``|N|`` "photons" (edges,
weight ``A_ij``) or leaving one alone (loops, weight ``b_i``): the *loop hafnian* of the matrix
obtained by repeating index ``i`` ``N_i`` times and putting ``b`` on the diagonal. Hence::

    rho_{mn} = e^c lhaf(A~_{(m, n)}) / sqrt(m! n!),     p(n) = rho_{nn}.

For zero-mean states ``b = 0`` and the loop hafnian reduces to the hafnian. This is the
Gaussian boson sampling formula of Hamilton et al., PRL 119, 170501 (2017).

**Power-trace algorithm** (Björklund, Gupt, Quesada, J. Exp. Algorithmics 24, 2019). For a
``2n x 2n`` symmetric ``A`` with zero diagonal::

    haf(A) = sum_{S subset [n]} (-1)^{n - |S|} [lambda^n] exp( sum_{k=1}^{n} tr((A X)_S^k) lambda^k / (2k) ),

where ``X`` swaps the two members of each consecutive pair ``(2i, 2i+1)`` and ``(.)_S`` keeps the
pairs in ``S``. Loops add ``(1/2) v_S^T X_S (A X)_S^{k-1} v_S`` to each term (``v`` = diagonal).
Cost ``O(n^3 2^n)`` instead of the ``(2n-1)!!`` terms of the definition.

**Threshold ("click") detectors** only tell vacuum from not-vacuum. By inclusion-exclusion on
vacuum probabilities of reduced states::

    P(clicks exactly on S) = sum_{Z subset S} (-1)^{|Z|} p0(modes not in S, plus Z),

which for zero-mean states is the *Torontonian* of Quesada, Arrazola, Killoran, PRA 98, 062322
(2018).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import factorial

import numpy as np

from ..errors import InputError, ShapeError
from .fock import bargmann
from .measure import vacuum_probability
from .states import GaussianState

__all__ = [
    "MAX_HAFNIAN_SIZE",
    "click_probability",
    "hafnian",
    "loop_hafnian",
    "pattern_probability",
    "torontonian",
]

#: Matrices larger than this are refused (2^(n/2) subsets: 2^20 already takes minutes in Python).
MAX_HAFNIAN_SIZE = 40


def _exp_series_coefficient(g: np.ndarray, n: int) -> complex:
    """``[lambda^n] exp(sum_{k>=1} g[k-1] lambda^k)`` by the recursion ``m e_m = sum_k k g_k e_{m-k}``."""
    e = np.zeros(n + 1, dtype=complex)
    e[0] = 1.0
    for m in range(1, n + 1):
        e[m] = sum(k * g[k - 1] * e[m - k] for k in range(1, m + 1)) / m
    return complex(e[n])


def _check_square_symmetric(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ShapeError(f"expected a square matrix, got shape {A.shape}")
    if A.shape[0] > MAX_HAFNIAN_SIZE:
        raise InputError(f"matrix of size {A.shape[0]} exceeds MAX_HAFNIAN_SIZE = {MAX_HAFNIAN_SIZE}")
    if not np.allclose(A, A.T, atol=1e-12 * max(1.0, np.abs(A).max(initial=0.0))):
        raise InputError("hafnians are defined for symmetric matrices")
    return A


def _power_trace_sum(A: np.ndarray, loops: np.ndarray | None) -> complex:
    """Shared kernel of :func:`hafnian` and :func:`loop_hafnian` (``A`` even-sized, zero diagonal)."""
    m = A.shape[0]
    n = m // 2
    X = np.kron(np.eye(n), np.array([[0, 1], [1, 0]]))
    total = 0.0 + 0j
    for size in range(n + 1):
        sign = (-1) ** (n - size)
        for S in combinations(range(n), size):
            idx = [i for s in S for i in (2 * s, 2 * s + 1)]
            if not idx:
                total += sign * (1.0 if n == 0 else 0.0)
                continue
            B = A[np.ix_(idx, idx)] @ X[np.ix_(idx, idx)]
            g = np.zeros(n, dtype=complex)
            Bk = np.eye(len(idx), dtype=complex)
            for k in range(1, n + 1):
                if loops is not None:  # (1/2) v^T X B^{k-1} v  uses B^{k-1}
                    v = loops[idx]
                    g[k - 1] += 0.5 * (v @ X[np.ix_(idx, idx)] @ Bk @ v)
                Bk = Bk @ B
                g[k - 1] += np.trace(Bk) / (2 * k)
            total += sign * _exp_series_coefficient(g, n)
    return complex(total)


def hafnian(A: np.ndarray) -> complex:
    """Hafnian (sum over perfect matchings) of a symmetric matrix; the diagonal is ignored."""
    A = _check_square_symmetric(A)
    m = A.shape[0]
    if m == 0:
        return 1.0 + 0j
    if m % 2:
        return 0.0 + 0j
    A = A - np.diag(np.diag(A))
    return _power_trace_sum(A, None)


def loop_hafnian(A: np.ndarray) -> complex:
    """Loop hafnian (matchings with self-loops weighted by the diagonal) of a symmetric matrix."""
    A = _check_square_symmetric(A)
    m = A.shape[0]
    if m == 0:
        return 1.0 + 0j
    if m % 2:
        P = np.zeros((m + 1, m + 1), dtype=complex)
        P[:m, :m] = A
        P[m, m] = 1.0
        A = P
    v = np.diag(A).copy()
    return _power_trace_sum(A - np.diag(v), v)


def torontonian(O: np.ndarray) -> complex:
    """``Tor(O) = sum_{Z subset [n]} (-1)^{n - |Z|} / sqrt(det(I - O_Z))`` for a ``2n x 2n`` matrix.

    ``O_Z`` keeps rows and columns ``Z`` and ``n + Z`` (the ``a`` and ``a^dag`` blocks of mode set ``Z``).
    """
    O = np.asarray(O, dtype=complex)
    if O.ndim != 2 or O.shape[0] != O.shape[1] or O.shape[0] % 2:
        raise ShapeError(f"expected a 2n x 2n matrix, got shape {O.shape}")
    n = O.shape[0] // 2
    total = 0.0 + 0j
    for size in range(n + 1):
        for Z in combinations(range(n), size):
            idx = list(Z) + [n + z for z in Z]
            sub = np.eye(len(idx)) - O[np.ix_(idx, idx)]
            total += (-1) ** (n - size) / np.sqrt(np.linalg.det(sub) + 0j)
    return complex(total)


def pattern_probability(state: GaussianState, pattern: Sequence[int]) -> float:
    """Probability of detecting ``pattern[k]`` photons in mode ``k`` (photon-number-resolving detectors)."""
    M = state.num_modes
    pattern = [int(n) for n in pattern]
    if len(pattern) != M or any(n < 0 for n in pattern):
        raise InputError(f"pattern must list {M} non-negative photon numbers, got {pattern}")
    A, b, c = bargmann(state)
    idx = [k for k in range(M) for _ in range(pattern[k])]
    idx = idx + [M + k for k in idx]  # bra copies of the same photons
    sub = A[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, b[idx])
    value = np.exp(c) * loop_hafnian(sub) / np.prod([factorial(n) for n in pattern])
    return float(value.real)


def click_probability(state: GaussianState, clicks: Sequence[int]) -> float:
    """Probability that threshold detectors click exactly on the modes with ``clicks[k] == 1``."""
    M = state.num_modes
    clicks = [int(c) for c in clicks]
    if len(clicks) != M or any(c not in (0, 1) for c in clicks):
        raise InputError(f"clicks must list {M} values in {{0, 1}}, got {clicks}")
    S = [k for k in range(M) if clicks[k]]
    silent = [k for k in range(M) if not clicks[k]]
    total = 0.0
    for size in range(len(S) + 1):
        for Z in combinations(S, size):
            vac = silent + list(Z)
            total += (-1) ** size * (vacuum_probability(state, vac) if vac else 1.0)
    return float(total)
