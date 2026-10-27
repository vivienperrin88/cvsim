"""Gaussian states.

A Gaussian state of ``M`` modes is fully described by its mean vector ``d`` (length ``2M``) and
covariance matrix ``V`` (``2M x 2M``), in the conventions of :mod:`.phase_space`. Everything
else (photon numbers, purity, entropy, Wigner function, Fock amplitudes) is a function of
``(d, V)``.

:class:`GaussianState` is an immutable value object: operations return new states.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InputError, ShapeError
from .phase_space import ATOL, check_covariance, mode_indices, symplectic_eigenvalues

__all__ = ["GaussianState"]


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class GaussianState:
    """Mean vector and covariance matrix of an ``M``-mode Gaussian state (xxpp ordering).

    Parameters
    ----------
    mean : array_like, shape (2M,)
    cov : array_like, shape (2M, 2M)
    validate : bool
        Check symmetry and the uncertainty principle (default). Internal code that preserves
        physicality by construction passes ``False`` to skip the eigenvalue computation.
    """

    __slots__ = ("cov", "mean")
    mean: np.ndarray
    cov: np.ndarray

    def __init__(self, mean: np.ndarray, cov: np.ndarray, *, validate: bool = True, atol: float = ATOL):
        cov = check_covariance(cov, atol) if validate else np.array(cov, dtype=float)
        mean = np.array(mean, dtype=float).reshape(-1)
        if mean.shape[0] != cov.shape[0]:
            raise ShapeError(f"mean has length {mean.shape[0]} but covariance is {cov.shape[0]} x {cov.shape[0]}")
        mean.setflags(write=False)
        cov = np.array(cov, dtype=float)
        cov.setflags(write=False)
        self.mean = mean
        self.cov = cov

    # ------------------------------------------------------------------ constructors
    @classmethod
    def vacuum(cls, M: int = 1) -> GaussianState:
        """``M``-mode vacuum: ``d = 0``, ``V = I/2``."""
        if M < 1:
            raise InputError(f"number of modes must be >= 1, got {M}")
        return cls(np.zeros(2 * M), np.eye(2 * M) / 2, validate=False)

    @classmethod
    def coherent(cls, alpha: complex | Sequence[complex]) -> GaussianState:
        """Coherent state(s) ``|alpha_1, ..., alpha_M>``: ``d = sqrt(2) (Re alpha, Im alpha)``."""
        a = np.atleast_1d(np.asarray(alpha, dtype=complex))
        if a.ndim != 1:
            raise ShapeError(f"alpha must be a scalar or a 1-D sequence, got shape {a.shape}")
        return cls(np.sqrt(2) * np.concatenate([a.real, a.imag]), np.eye(2 * a.size) / 2, validate=False)

    @classmethod
    def thermal(cls, nbar: float | Sequence[float]) -> GaussianState:
        """Thermal state(s) with mean photon numbers ``nbar``: ``V = (nbar + 1/2) I``."""
        n = np.atleast_1d(np.asarray(nbar, dtype=float))
        if n.ndim != 1 or np.any(n < 0):
            raise InputError(f"nbar must be >= 0, got {nbar}")
        diag = np.concatenate([n, n]) + 0.5
        return cls(np.zeros(2 * n.size), np.diag(diag), validate=False)

    @classmethod
    def squeezed_vacuum(cls, r: float, phi: float = 0.0) -> GaussianState:
        """``S(r, phi)|0>`` with ``S = exp((conj(z) a^2 - z a^dag^2)/2)``, ``z = r e^{i phi}``.

        ``phi = 0`` squeezes ``x``: ``Var(x) = e^{-2r}/2``. The squeezing axis is rotated by
        ``phi/2``: ``V = R(phi/2) diag(e^{-2r}, e^{2r}) R(phi/2)^T / 2``.
        """
        Rm = _rot(phi / 2)
        return cls(np.zeros(2), Rm @ np.diag([np.exp(-2 * r), np.exp(2 * r)]) @ Rm.T / 2, validate=False)

    @classmethod
    def product(cls, *states: GaussianState) -> GaussianState:
        """Tensor product ``rho_1 (x) rho_2 (x) ...`` (mode order: as listed).

        Careful with the xxpp ordering: the x-blocks of all states come first, then the p-blocks.
        """
        if not states:
            raise InputError("product needs at least one state")
        Ms = [s.num_modes for s in states]
        M = sum(Ms)
        mean = np.zeros(2 * M)
        cov = np.zeros((2 * M, 2 * M))
        offset = 0
        for s, m in zip(states, Ms):
            idx = list(range(offset, offset + m)) + list(range(M + offset, M + offset + m))
            mean[idx] = s.mean
            cov[np.ix_(idx, idx)] = s.cov
            offset += m
        return cls(mean, cov, validate=False)

    # ------------------------------------------------------------------ basic properties
    @property
    def num_modes(self) -> int:
        return self.mean.shape[0] // 2

    def __repr__(self) -> str:
        return f"GaussianState(modes={self.num_modes}, purity={self.purity():.6g})"

    def reduce(self, modes: Sequence[int]) -> GaussianState:
        """Reduced state of the listed modes (partial trace = keep the sub-blocks), in that order."""
        idx = mode_indices(modes, self.num_modes)
        return GaussianState(self.mean[idx], self.cov[np.ix_(idx, idx)], validate=False)

    def purity(self) -> float:
        """``Tr rho^2 = 1 / (2^M sqrt(det V))`` (hbar = 1)."""
        _, logdet = np.linalg.slogdet(self.cov)
        return float(np.exp(-self.num_modes * np.log(2) - 0.5 * logdet))

    def is_pure(self, atol: float = 1e-9) -> bool:
        return abs(self.purity() - 1.0) < atol

    def mean_photons(self) -> np.ndarray:
        """``<n_k> = (V_xx + V_pp - 1)/2 + (d_x^2 + d_p^2)/2`` for every mode ``k``."""
        M = self.num_modes
        V, d = self.cov, self.mean
        diag = np.diag(V)
        return (diag[:M] + diag[M:] - 1) / 2 + (d[:M] ** 2 + d[M:] ** 2) / 2

    def entropy(self) -> float:
        """Von Neumann entropy in nats: ``sum_k g(nu_k)``, ``g(nu) = (nu+1/2) ln(nu+1/2) - (nu-1/2) ln(nu-1/2)``."""
        total = 0.0
        for nu in symplectic_eigenvalues(self.cov):
            a, b = nu + 0.5, nu - 0.5
            total += a * np.log(a) - (b * np.log(b) if b > 1e-14 else 0.0)
        return float(total)
