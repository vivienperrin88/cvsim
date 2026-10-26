"""Phase-space conventions for Gaussian states.

hbar = 1 throughout (CONVENTIONS.md, "Continuous variables"):

* quadratures ``x = (a + a^dag)/sqrt(2)``, ``p = (a - a^dag)/(i sqrt(2))``, ``[x, p] = i``;
* quadrature vector ``r = (x_1, ..., x_M, p_1, ..., p_M)`` ("xxpp" ordering);
* covariance ``V_ij = <{dr_i, dr_j}>/2``; the vacuum has ``V = I/2``;
* canonical commutators ``[r_i, r_j] = i Omega_ij`` with ``Omega = [[0, I], [-I, 0]]``.

A linear (Bogoliubov) transformation of the mode operators ``a -> A a + B a^dag`` acts on the
quadratures as a real **symplectic** matrix ``S`` (``S Omega S^T = Omega``). Gates are written in
that Bogoliubov form, which is the one you can check by hand against ``U^dag a U``.

Nothing in this subpackage imports the tensor-network layers: the Gaussian backend is exact,
costs O(M^3) for M modes, and is the right tool whenever a circuit stays Gaussian.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ShapeError

__all__ = [
    "check_covariance",
    "complex_basis",
    "is_symplectic",
    "mode_indices",
    "omega",
    "symplectic_eigenvalues",
    "symplectic_from_bogoliubov",
]

#: Default absolute tolerance for physicality checks (covariances are O(1) numbers).
ATOL = 1e-10


def omega(M: int) -> np.ndarray:
    """The symplectic form ``[[0, I_M], [-I_M, 0]]`` (xxpp ordering)."""
    z = np.zeros((M, M))
    eye = np.eye(M)
    return np.block([[z, eye], [-eye, z]])


def mode_indices(modes: Sequence[int], M: int) -> list[int]:
    """Quadrature indices ``[x_k..., p_k...]`` of the listed modes, for sub-block extraction."""
    modes = list(modes)
    if not modes or any(not 0 <= k < M for k in modes) or len(set(modes)) != len(modes):
        raise InputError(f"modes must be distinct integers in 0..{M - 1}, got {modes}")
    return modes + [M + k for k in modes]


def complex_basis(M: int) -> np.ndarray:
    """Unitary ``W`` with ``(a_1..a_M, a^dag_1..a^dag_M) = W r``.

    ``W = [[I, iI], [I, -iI]] / sqrt(2)``. The complex covariance is ``sigma = W V W^dag``.
    """
    eye = np.eye(M)
    return np.block([[eye, 1j * eye], [eye, -1j * eye]]) / np.sqrt(2)


def symplectic_from_bogoliubov(A: npt.ArrayLike, B: npt.ArrayLike) -> np.ndarray:
    """Real symplectic matrix of the mode transformation ``a -> A a + B a^dag``.

    For the mean amplitudes ``alpha = <a>``: ``alpha -> A alpha + B conj(alpha)``. Writing
    ``alpha = (x + i p)/sqrt(2)`` gives (xxpp ordering)::

        S = [[Re(A + B), -Im(A - B)],
             [Im(A + B),  Re(A - B)]]

    ``A`` and ``B`` are ``M x M`` complex matrices; ``S`` is ``2M x 2M``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    B = np.atleast_2d(np.asarray(B, dtype=complex))
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ShapeError(f"A and B must be square and of the same shape, got {A.shape} and {B.shape}")
    return np.block([[(A + B).real, -(A - B).imag], [(A + B).imag, (A - B).real]])


def is_symplectic(S: np.ndarray, atol: float = ATOL) -> bool:
    """``S Omega S^T == Omega`` within ``atol``."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] % 2:
        return False
    W = omega(S.shape[0] // 2)
    return bool(np.allclose(S @ W @ S.T, W, atol=atol))


def symplectic_eigenvalues(V: np.ndarray) -> np.ndarray:
    """Williamson's symplectic eigenvalues of a covariance matrix, sorted increasingly.

    They are the moduli of the eigenvalues of ``i Omega V``, which come in pairs ``+-nu``.
    Numerically stable route: the Hermitian matrix ``i V^{1/2} Omega V^{1/2}`` has the same
    spectrum. The state is physical iff every ``nu >= 1/2``; ``nu = 1/2`` for every pure state.
    """
    V = np.asarray(V, dtype=float)
    M = V.shape[0] // 2
    w, U = np.linalg.eigh((V + V.T) / 2)
    if w.min() <= 0:
        raise InputError(f"covariance must be positive definite, smallest eigenvalue {w.min():.3e}")
    sqrtV = (U * np.sqrt(w)) @ U.T
    ev = np.linalg.eigvalsh(1j * sqrtV @ omega(M) @ sqrtV)
    return np.sort(np.abs(ev))[::2]  # pairs (nu1, nu1, nu2, nu2, ...)


def check_covariance(V: np.ndarray, atol: float = ATOL) -> np.ndarray:
    """Validate a covariance matrix and return it as a symmetric float array.

    Raises
    ------
    ShapeError
        Not a square ``2M x 2M`` array.
    InputError
        Not symmetric, or violates the uncertainty principle ``V + i Omega / 2 >= 0``
        (message gives the most negative eigenvalue).
    """
    V = np.asarray(V)
    if np.iscomplexobj(V):
        if np.abs(V.imag).max(initial=0.0) > atol:
            raise InputError("covariance matrix must be real")
        V = V.real
    V = V.astype(float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2 or V.shape[0] == 0:
        raise ShapeError(f"covariance must be a non-empty 2M x 2M array, got shape {V.shape}")
    if not np.allclose(V, V.T, atol=atol):
        raise InputError(f"covariance must be symmetric (max asymmetry {np.abs(V - V.T).max():.3e})")
    M = V.shape[0] // 2
    lowest = np.linalg.eigvalsh(V + 0.5j * omega(M)).min()
    if lowest < -atol:
        raise InputError(f"covariance violates the uncertainty principle: V + i Omega/2 has eigenvalue {lowest:.3e}")
    return (V + V.T) / 2
