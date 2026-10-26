"""Gaussian measurements, overlaps and phase-space functions.

**Conditioning is a Schur complement.** Split the modes into a measured part ``A`` and the
rest ``B``. Measuring a Gaussian observable on ``A`` leaves ``B`` Gaussian, with

    V_B' = V_B - C (Sigma)^{-1} C^T,      d_B' = d_B + C Sigma^{-1} (outcome - mean_A),

where ``C`` is the cross-covariance between ``B`` and the measured variables and ``Sigma`` the
covariance of the measured variables *including the detector noise*:

* homodyne of ``x_phi = cos(phi) x + sin(phi) p`` on one mode: one variable, ``Sigma = u^T V_A u``;
* heterodyne (projection on coherent states): both quadratures, ``Sigma = V_A + I/2``.

The covariance update does not depend on the outcome; the mean does. Outcomes are drawn from the
marginal distribution, which is Gaussian with mean ``mean_A`` and covariance ``Sigma``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ..errors import InputError
from .phase_space import mode_indices, symplectic_eigenvalues
from .states import GaussianState

__all__ = [
    "HeterodyneResult",
    "HomodyneResult",
    "fidelity",
    "heterodyne",
    "homodyne",
    "homodyne_density",
    "log_negativity",
    "overlap",
    "vacuum_probability",
    "wigner",
]


class HomodyneResult(NamedTuple):
    outcome: float  # measured value of x_phi
    density: float  # probability density of that outcome
    state: GaussianState | None  # conditional state of the other modes (None if none are left)


class HeterodyneResult(NamedTuple):
    outcome: complex  # measured alpha
    density: float  # Husimi Q value: probability density with respect to d^2 alpha
    state: GaussianState | None


def _rest(state: GaussianState, mode: int) -> list[int]:
    return [k for k in range(state.num_modes) if k != mode]


def homodyne_density(state: GaussianState, q: float | np.ndarray, mode: int = 0, phi: float = 0.0) -> np.ndarray:
    """Marginal probability density of ``x_phi = cos(phi) x + sin(phi) p`` on ``mode`` at ``q``."""
    idx = mode_indices([mode], state.num_modes)
    u = np.array([np.cos(phi), np.sin(phi)])
    mu = u @ state.mean[idx]
    var = u @ state.cov[np.ix_(idx, idx)] @ u
    q = np.asarray(q, dtype=float)
    return np.exp(-((q - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def homodyne(
    state: GaussianState,
    mode: int = 0,
    phi: float = 0.0,
    outcome: float | None = None,
    rng: np.random.Generator | None = None,
) -> HomodyneResult:
    """Measure ``x_phi`` on ``mode``: sample (or impose) the outcome, condition the other modes.

    Pass ``outcome`` to post-select a value, or ``rng`` to sample one; exactly one is required.
    """
    if (outcome is None) == (rng is None):
        raise InputError("pass exactly one of `outcome` (post-selection) or `rng` (sampling)")
    M = state.num_modes
    idx = mode_indices([mode], M)
    u = np.array([np.cos(phi), np.sin(phi)])
    mu = float(u @ state.mean[idx])
    var = float(u @ state.cov[np.ix_(idx, idx)] @ u)
    q = float(rng.normal(mu, np.sqrt(var))) if rng is not None else float(outcome)  # type: ignore[arg-type]
    density = float(np.exp(-((q - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var))
    rest = _rest(state, mode)
    if not rest:
        return HomodyneResult(q, density, None)
    jdx = mode_indices(rest, M)
    C = state.cov[np.ix_(jdx, idx)] @ u  # cross-covariance with x_phi
    cov = state.cov[np.ix_(jdx, jdx)] - np.outer(C, C) / var
    mean = state.mean[jdx] + C * (q - mu) / var
    return HomodyneResult(q, density, GaussianState(mean, cov, validate=False))


def heterodyne(
    state: GaussianState,
    mode: int = 0,
    outcome: complex | None = None,
    rng: np.random.Generator | None = None,
) -> HeterodyneResult:
    """Project ``mode`` on a coherent state ``|alpha>``; ``alpha`` sampled from the Husimi Q function.

    In quadrature units the outcome ``r = sqrt(2) (Re alpha, Im alpha)`` is Gaussian with mean
    ``d_A`` and covariance ``V_A + I/2``; the density with respect to ``d^2 alpha`` is twice the
    density with respect to ``d^2 r`` (Jacobian of ``r = sqrt(2) alpha``).
    """
    if (outcome is None) == (rng is None):
        raise InputError("pass exactly one of `outcome` (post-selection) or `rng` (sampling)")
    M = state.num_modes
    idx = mode_indices([mode], M)
    Sigma = state.cov[np.ix_(idx, idx)] + np.eye(2) / 2
    dA = state.mean[idx]
    if rng is not None:
        r = rng.multivariate_normal(dA, Sigma)
    else:
        assert outcome is not None  # guaranteed by the check above
        r = np.sqrt(2) * np.array([np.real(outcome), np.imag(outcome)])
    alpha = complex(r[0], r[1]) / np.sqrt(2)
    diff = r - dA
    density = float(
        2 * np.exp(-0.5 * diff @ np.linalg.solve(Sigma, diff)) / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    )
    rest = _rest(state, mode)
    if not rest:
        return HeterodyneResult(alpha, density, None)
    jdx = mode_indices(rest, M)
    C = state.cov[np.ix_(jdx, idx)]
    K = C @ np.linalg.inv(Sigma)
    cov = state.cov[np.ix_(jdx, jdx)] - K @ C.T
    mean = state.mean[jdx] + K @ diff
    return HeterodyneResult(alpha, density, GaussianState(mean, cov, validate=False))


def vacuum_probability(state: GaussianState, modes: Sequence[int] | None = None) -> float:
    """Probability that every listed mode is in vacuum (all modes if ``None``).

    ``p0 = exp(-d^T (V + I/2)^{-1} d / 2) / sqrt(det(V + I/2))`` on the reduced state: the
    Husimi function at the origin, times ``pi^m``.
    """
    red = state if modes is None else state.reduce(modes)
    Q = red.cov + np.eye(red.cov.shape[0]) / 2
    d = red.mean
    return float(np.exp(-0.5 * d @ np.linalg.solve(Q, d)) / np.sqrt(np.linalg.det(Q)))


def overlap(s1: GaussianState, s2: GaussianState) -> float:
    """``Tr(rho_1 rho_2) = exp(-delta^T (V1 + V2)^{-1} delta / 2) / sqrt(det(V1 + V2))``.

    Equals the fidelity when at least one of the two states is pure.
    """
    if s1.num_modes != s2.num_modes:
        raise InputError(f"states have {s1.num_modes} and {s2.num_modes} modes")
    Vs = s1.cov + s2.cov
    delta = s1.mean - s2.mean
    return float(np.exp(-0.5 * delta @ np.linalg.solve(Vs, delta)) / np.sqrt(np.linalg.det(Vs)))


def fidelity(s1: GaussianState, s2: GaussianState, atol: float = 1e-9) -> float:
    """Uhlmann fidelity ``F = (Tr sqrt(sqrt(rho1) rho2 sqrt(rho1)))^2`` when one state is pure.

    Raises :class:`InputError` if neither state is pure (the general mixed-state formula is not
    implemented: use the Fock representation (:mod:`.fock`) for small cases).
    """
    if not (s1.is_pure(atol) or s2.is_pure(atol)):
        raise InputError("fidelity is implemented when at least one state is pure")
    return overlap(s1, s2)


def wigner(state: GaussianState, x: np.ndarray, p: np.ndarray, mode: int = 0) -> np.ndarray:
    """Wigner function ``W(x, p)`` of one mode on the grid ``x`` (columns) by ``p`` (rows).

    ``W(r) = exp(-(r - d)^T V^{-1} (r - d) / 2) / (2 pi sqrt(det V))``; ``int W dx dp = 1``.
    """
    red = state.reduce([mode])
    X, P = np.meshgrid(np.asarray(x, float), np.asarray(p, float))
    R = np.stack([X - red.mean[0], P - red.mean[1]], axis=-1)
    Vi = np.linalg.inv(red.cov)
    quad = np.einsum("...i,ij,...j->...", R, Vi, R)
    return np.exp(-quad / 2) / (2 * np.pi * np.sqrt(np.linalg.det(red.cov)))


def log_negativity(state: GaussianState, modes_a: Sequence[int]) -> float:
    """Logarithmic negativity ``E_N = sum_k max(0, -log2(2 nu~_k))`` of the bipartition ``A | rest``.

    Partial transposition of ``A`` flips the sign of its momenta: ``V~ = P V P``. ``nu~`` are the
    symplectic eigenvalues of ``V~``. For the two-mode squeezed vacuum ``E_N = 2 r / ln 2``.
    """
    M = state.num_modes
    flips = np.ones(2 * M)
    for k in modes_a:
        if not 0 <= k < M:
            raise InputError(f"mode {k} out of range 0..{M - 1}")
        flips[M + k] = -1.0
    Vt = flips[:, None] * state.cov * flips[None, :]
    nu = symplectic_eigenvalues(Vt)
    return float(np.sum(np.maximum(0.0, -np.log2(2 * nu))))
