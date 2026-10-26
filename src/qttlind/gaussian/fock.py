"""From phase space to Fock space: the Bargmann recursion.

**Bargmann representation.** For a density matrix, the function
``F(z) = sum_{m,n} rho_{mn} conj(alpha)^m beta^n / sqrt(m! n!)`` of ``z = (conj(alpha), beta)``
(``2M`` complex variables: first the ket indices, then the bra indices) is, for a Gaussian state,
the exponential of a quadratic form::

    F(z) = exp(z^T A z / 2 + b^T z + c).

Derivation (ħ = 1): ``F(conj(alpha), alpha) = pi^M Q(alpha) e^{|alpha|^2}`` with the Husimi
function ``Q``; reading off the quadratic form with ``sigma = W V W^dag``,
``Q_cov = sigma + I/2``, ``mu = W d`` and ``X = [[0, I], [I, 0]]`` gives::

    K = Q_cov^{-1} X (symmetric),   A = X - K,   b = K X mu,
    c = -mu^T X K X mu / 2 - ln det(Q_cov) / 2.

**Recursion.** Writing ``F = sum_N G_N z^N / sqrt(N!)`` and differentiating once::

    G_{N + e_i} = (b_i G_N + sum_j A_ij sqrt(N_j) G_{N - e_j}) / sqrt(N_i + 1),   G_0 = e^c,

so every Fock amplitude follows from its neighbours with ``2M + 1`` multiplications, without
factorials or hafnians. The same recursion gives kets of pure states (``M`` variables).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ..errors import InputError, ShapeError
from .phase_space import complex_basis
from .states import GaussianState

__all__ = [
    "MAX_FOCK_ENTRIES",
    "Bargmann",
    "bargmann",
    "bargmann_ket",
    "fock_coefficients",
    "fock_density_matrix",
    "fock_ket",
]

#: Refuse recursions that would allocate more than this many complex numbers (16 * 2^25 = 512 MB).
MAX_FOCK_ENTRIES = 2**25


class Bargmann(NamedTuple):
    """``F(z) = exp(z^T A z / 2 + b^T z + c)``."""

    A: np.ndarray
    b: np.ndarray
    c: complex


def bargmann(state: GaussianState) -> Bargmann:
    """Bargmann triple of the density matrix (``2M`` variables: ket indices first)."""
    M = state.num_modes
    W = complex_basis(M)
    sigma = W @ state.cov @ W.conj().T
    mu = W @ state.mean
    X = np.block([[np.zeros((M, M)), np.eye(M)], [np.eye(M), np.zeros((M, M))]])
    Qc = sigma + np.eye(2 * M) / 2
    K = np.linalg.solve(Qc, X)
    K = (K + K.T) / 2
    A = X - K
    b = K @ (X @ mu)
    sign, logdet = np.linalg.slogdet(Qc)
    c = -0.5 * (X @ mu) @ K @ (X @ mu) - 0.5 * (logdet + np.log(sign))
    return Bargmann(A, b, complex(c))


def bargmann_ket(state: GaussianState, atol: float = 1e-9) -> Bargmann:
    """Bargmann triple of the ket of a **pure** state (``M`` variables).

    For a pure state ``rho_{mn} = psi_m conj(psi_n)``: ``A`` is block diagonal
    ``A_ket (+) conj(A_ket)``, ``b = (b_ket, conj(b_ket))``, and ``Re c = 2 Re c_ket``. The global
    phase is fixed by choosing ``psi_0 = <0|psi> > 0``.

    Raises
    ------
    InputError
        If the state is not pure within ``atol``.
    """
    if not state.is_pure(atol):
        raise InputError(f"bargmann_ket needs a pure state, got purity {state.purity():.6g}")
    M = state.num_modes
    full = bargmann(state)
    return Bargmann(full.A[:M, :M], full.b[:M], complex(full.c.real / 2))


def fock_coefficients(A: np.ndarray, b: np.ndarray, c: complex, shape: Sequence[int]) -> np.ndarray:
    """All coefficients ``G_N`` with ``0 <= N_i < shape[i]``, by the Bargmann recursion.

    Visits multi-indices in C order: for ``N != 0`` the first non-zero entry ``i`` gives the
    predecessor ``N - e_i``, which (like every ``N - e_i - e_j``) has already been computed.
    """
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)
    shape = tuple(int(s) for s in shape)
    n = len(shape)
    if A.shape != (n, n) or b.shape != (n,):
        raise ShapeError(f"A must be {n} x {n} and b of length {n} for shape {shape}, got {A.shape}, {b.shape}")
    if any(s < 1 for s in shape):
        raise InputError(f"every cutoff must be >= 1, got {shape}")
    total = int(np.prod(shape))
    if total > MAX_FOCK_ENTRIES:
        raise InputError(f"{total} Fock coefficients exceed MAX_FOCK_ENTRIES = {MAX_FOCK_ENTRIES}")
    G = np.zeros(shape, dtype=complex)
    G[(0,) * n] = np.exp(c)
    sqrt = np.sqrt(np.arange(max(shape) + 1))
    for N in np.ndindex(*shape):
        if not any(N):
            continue
        i = next(k for k, v in enumerate(N) if v)
        P = list(N)
        P[i] -= 1  # predecessor N - e_i
        val = b[i] * G[tuple(P)]
        for j in range(n):
            if P[j]:
                Q = list(P)
                Q[j] -= 1
                val += A[i, j] * sqrt[P[j]] * G[tuple(Q)]
        G[N] = val / sqrt[N[i]]
    return G


def fock_density_matrix(state: GaussianState, cutoffs: int | Sequence[int]) -> np.ndarray:
    """Dense density matrix in the truncated Fock basis (Kronecker order, mode 0 slowest)."""
    M = state.num_modes
    cut = [int(cutoffs)] * M if np.isscalar(cutoffs) else [int(k) for k in cutoffs]
    if len(cut) != M:
        raise ShapeError(f"{len(cut)} cutoffs for {M} modes")
    A, b, c = bargmann(state)
    G = fock_coefficients(A, b, c, cut + cut)
    D = int(np.prod(cut))
    return G.reshape(D, D)


def fock_ket(state: GaussianState, cutoffs: int | Sequence[int]) -> np.ndarray:
    """Dense ket of a pure state in the truncated Fock basis (``psi_0 > 0`` phase convention)."""
    M = state.num_modes
    cut = [int(cutoffs)] * M if np.isscalar(cutoffs) else [int(k) for k in cutoffs]
    if len(cut) != M:
        raise ShapeError(f"{len(cut)} cutoffs for {M} modes")
    A, b, c = bargmann_ket(state)
    return fock_coefficients(A, b, c, cut).reshape(-1)
