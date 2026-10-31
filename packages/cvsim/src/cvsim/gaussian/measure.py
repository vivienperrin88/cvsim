"""Gaussian measurements: homodyne (any angle, any local-oscillator mode), heterodyne, general-dyne.

**One model for all of them.** A measurement on modes ``A`` records ``y = L r_A + ξ``: ``L``
(``k x 2|A|``) selects the measured quadratures and ``ξ ~ N(0, N)`` is the noise of the
measurement itself. The outcome is Gaussian with mean ``μ = L d_A`` and covariance
``Σ = L V_A Lᵀ + N``, and the other modes ``B`` are left in the Gaussian state

    V_B' = V_B − K Σ Kᵀ,    d_B' = d_B + K (y − μ),    K = V_{BA} Lᵀ Σ⁻¹

(a Schur complement: the covariance update does not depend on the outcome). The measured modes
are consumed.

* **Homodyne** of ``x_φ = cos φ x + sin φ p``: ``L = (cos φ, sin φ)``, ideal ``N = 0``.
* **Heterodyne** (projection on coherent states): ``L = I``, ideal ``N = I/2``; the outcome is
  reported as ``α = (y_x + i y_p)/√2`` and its density is per ``d²α``.
* **Detector imperfections** (CONVENTIONS.md, "Measurements"): efficiency ``η`` and electronic
  noise ``v_el`` (shot-noise units, referred to each photodiode pair) are modelled as loss ``η``
  and added noise before an ideal detector, with the outcome rescaled by ``1/√η`` so that it
  estimates the quadrature at the detector input. This adds ``(1 − η + v_el)/(2η)`` to a homodyne
  and ``(1 − η + v_el)/η`` per quadrature to a heterodyne (each arm sees half the signal).
* **Local oscillator in any mode** (:func:`homodyne_lo`): the measured mode is
  ``b = Σ_j conj(f_j) a_j`` for a unit vector ``f``; the remaining state is returned in a basis
  completing ``f``, given with the result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core import linalg as la
from ..core._types import ComplexArray, FloatArray
from ..errors import InputError, ShapeError
from .states import GaussianState
from .symplectic import change_basis

__all__ = [
    "MeasurementResult",
    "general_dyne",
    "heterodyne",
    "homodyne",
    "homodyne_density",
    "homodyne_lo",
    "homodyne_noise",
    "sample_homodyne",
]


class MeasurementResult(NamedTuple):
    """Outcome, its probability density, and the conditional state of the unmeasured modes.

    ``outcome`` is a float (homodyne), a complex ``α`` (heterodyne) or an array (several
    quadratures); ``density`` is per unit of that outcome (per ``d²α`` for heterodyne);
    ``state`` is ``None`` when no mode is left; ``basis`` (``homodyne_lo`` only) holds the mode
    functions of the remaining modes as columns, in the old basis.
    """

    outcome: float | complex | FloatArray
    density: float
    state: GaussianState | None
    mean: FloatArray
    cov: FloatArray
    basis: ComplexArray | None = None


# ------------------------------------------------------------------------------- detector model
def _check_detector(efficiency: float, electronic: float) -> None:
    if not 0.0 < efficiency <= 1.0:
        raise InputError(f"efficiency must be in (0, 1], got {efficiency}")
    if not electronic >= 0 or not math.isfinite(electronic):
        raise InputError(f"electronic noise must be finite and >= 0 (shot-noise units), got {electronic}")


def homodyne_noise(efficiency: float = 1.0, electronic: float = 0.0, heterodyne: bool = False) -> float:
    """Variance the detector adds to each estimated quadrature (ħ = 1 units).

    Homodyne ``(1 − η + v_el)/(2η)``; heterodyne ``½ + (1 − η + v_el)/η`` (vacuum port included).

    Parameters
    ----------
    efficiency
        Detection efficiency ``η`` in ``(0, 1]``.
    electronic
        Electronic noise ``v_el`` in shot-noise units.
    heterodyne
        Heterodyne instead of homodyne.
    """
    _check_detector(efficiency, electronic)
    excess = (1.0 - efficiency + electronic) / efficiency
    return cv.VACUUM_VARIANCE + excess if heterodyne else excess / 2


# ------------------------------------------------------------------------------- core
def _gaussian_logpdf(y: FloatArray, mu: FloatArray, Sigma: FloatArray) -> float:
    diff = y - mu
    k = int(diff.shape[0])
    return -0.5 * float(diff @ la.solve(Sigma, diff)) - 0.5 * la.logdet_pd(Sigma) - 0.5 * k * math.log(2 * math.pi)


def general_dyne(
    state: GaussianState,
    modes: Sequence[int],
    L: npt.ArrayLike,
    noise: npt.ArrayLike | None = None,
    outcome: npt.ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> MeasurementResult:
    """Measure ``y = L r_A + ξ`` on ``modes``; sample the outcome (``rng``) or impose it (``outcome``).

    Parameters
    ----------
    state
        State before the measurement.
    modes
        Measured modes ``A`` (consumed).
    L
        ``k x 2|A|`` matrix on the xxpp quadratures of ``modes``.
    noise
        ``k x k`` covariance of the measurement noise ``ξ`` (default 0: projective).
    outcome
        Post-selected value of ``y`` (length ``k``).
    rng
        Generator to sample ``y``. Exactly one of ``outcome`` and ``rng`` is required.

    Raises
    ------
    InputError
        Both or neither of ``outcome``/``rng``; a singular outcome covariance.
    """
    if (outcome is None) == (rng is None):
        raise InputError("pass exactly one of `outcome` (post-selection) or `rng` (sampling)")
    M = state.num_modes
    idx = cv.mode_indices(modes, M)
    L_ = np.atleast_2d(np.asarray(L, dtype=float))
    if L_.shape[1] != len(idx):
        raise ShapeError(f"L must have {len(idx)} columns (xxpp of {len(modes)} modes), got shape {L_.shape}")
    k = L_.shape[0]
    N = np.zeros((k, k)) if noise is None else np.atleast_2d(np.asarray(noise, dtype=float))
    if N.shape != (k, k):
        raise ShapeError(f"noise must be {k} x {k}, got shape {N.shape}")
    mu = L_ @ state.mean[idx]
    Sigma = L_ @ state.cov[np.ix_(idx, idx)] @ L_.T + N
    Sigma = (Sigma + Sigma.T) / 2
    if la.eigvalsh(Sigma)[0] <= 1e-14 * max(1.0, float(np.abs(Sigma).max())):
        raise InputError("outcome covariance is singular (an infinitely squeezed quadrature?)")
    if rng is not None:
        y = mu + la.cholesky(Sigma) @ rng.standard_normal(k)
    else:
        y = np.atleast_1d(np.asarray(outcome, dtype=float))
        if y.shape != (k,):
            raise ShapeError(f"outcome must have length {k}, got shape {y.shape}")
    density = math.exp(_gaussian_logpdf(y, mu, Sigma))
    rest = [m for m in range(M) if m not in set(int(a) for a in modes)]
    if not rest:
        return MeasurementResult(y, density, None, mu, Sigma)
    jdx = cv.mode_indices(rest, M)
    C = state.cov[np.ix_(jdx, idx)] @ L_.T  # Cov(r_B, y)
    K = la.solve(Sigma, C.T).T
    V = state.cov[np.ix_(jdx, jdx)] - K @ C.T
    d = state.mean[jdx] + K @ (y - mu)
    return MeasurementResult(y, density, GaussianState(d, (V + V.T) / 2, validate=False), mu, Sigma)


# ------------------------------------------------------------------------------- homodyne / heterodyne
def homodyne(
    state: GaussianState,
    mode: int | Sequence[int] = 0,
    phi: float | Sequence[float] = 0.0,
    *,
    efficiency: float = 1.0,
    electronic: float = 0.0,
    outcome: float | npt.ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> MeasurementResult:
    """Homodyne of ``x_φ`` on one mode, or simultaneous homodynes on several modes.

    Parameters
    ----------
    state
        State before the measurement.
    mode
        Mode index, or several (measured jointly; outcome is then an array).
    phi
        Local-oscillator phase(s), one per mode or one for all.
    efficiency
        Detection efficiency ``η`` (outcome rescaled by ``1/√η``).
    electronic
        Electronic noise in shot-noise units.
    outcome
        Post-selected value(s).
    rng
        Generator to sample the outcome.
    """
    modes = [int(mode)] if isinstance(mode, (int, np.integer)) else [int(m) for m in mode]
    phis = np.broadcast_to(np.asarray(phi, dtype=float), (len(modes),))
    n = len(modes)
    L = np.zeros((n, 2 * n))
    for j, ph in enumerate(phis):
        L[j, [j, n + j]] = cv.homodyne_vector(float(ph))
    N = homodyne_noise(efficiency, electronic) * np.eye(n)
    res = general_dyne(state, modes, L, N, outcome=None if outcome is None else np.atleast_1d(outcome), rng=rng)
    if isinstance(mode, (int, np.integer)):
        return res._replace(outcome=float(np.asarray(res.outcome)[0]))
    return res


def heterodyne(
    state: GaussianState,
    mode: int = 0,
    *,
    efficiency: float = 1.0,
    electronic: float = 0.0,
    outcome: complex | None = None,
    rng: np.random.Generator | None = None,
) -> MeasurementResult:
    """Heterodyne (coherent-state projection) of ``mode``; outcome ``α``, density per ``d²α``.

    Parameters
    ----------
    state
        State before the measurement.
    mode
        Mode index.
    efficiency
        Detection efficiency of each arm.
    electronic
        Electronic noise of each arm, shot-noise units.
    outcome
        Post-selected ``α``.
    rng
        Generator to sample ``α`` (from the Husimi function, broadened by the detector).
    """
    N = homodyne_noise(efficiency, electronic, heterodyne=True) * np.eye(2)
    y = None if outcome is None else math.sqrt(2.0) * np.array([complex(outcome).real, complex(outcome).imag])
    res = general_dyne(state, [mode], np.eye(2), N, outcome=y, rng=rng)
    yo = np.asarray(res.outcome)
    # y = √2 (Re α, Im α): d²y = 2 d²α
    return res._replace(outcome=complex(yo[0], yo[1]) / math.sqrt(2.0), density=2.0 * res.density)


def homodyne_density(
    state: GaussianState,
    q: npt.ArrayLike,
    mode: int = 0,
    phi: float = 0.0,
    *,
    efficiency: float = 1.0,
    electronic: float = 0.0,
) -> FloatArray:
    """Marginal probability density of the homodyne outcome at ``q`` (array-valued).

    Parameters
    ----------
    state
        The state.
    q
        Outcome value(s).
    mode
        Mode index.
    phi
        Local-oscillator phase.
    efficiency, electronic
        Detector model (as :func:`homodyne`).
    """
    idx = cv.mode_indices([mode], state.num_modes)
    u = cv.homodyne_vector(phi)
    mu = float(u @ state.mean[idx])
    var = float(u @ state.cov[np.ix_(idx, idx)] @ u) + homodyne_noise(efficiency, electronic)
    qq = np.asarray(q, dtype=float)
    return np.asarray(np.exp(-((qq - mu) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var), dtype=float)


def homodyne_lo(
    state: GaussianState,
    lo: npt.ArrayLike,
    phi: float = 0.0,
    *,
    efficiency: float = 1.0,
    electronic: float = 0.0,
    outcome: float | None = None,
    rng: np.random.Generator | None = None,
) -> MeasurementResult:
    """Homodyne with the local oscillator in the mode ``b = Σ_j conj(f_j) a_j`` (e.g. a shaped pulse).

    The modes are rotated to a basis whose first mode is ``b`` (``f`` completed to a unitary
    ``U`` by QR); ``b`` is measured and the state of the other ``M − 1`` modes is returned in that
    basis, with ``basis = U[:, 1:]`` (their mode functions in the old basis).

    Parameters
    ----------
    state
        State before the measurement.
    lo
        Mode function ``f`` of the local oscillator (complex, length ``M``, unit norm).
    phi
        Local-oscillator phase.
    efficiency, electronic
        Detector model (as :func:`homodyne`).
    outcome
        Post-selected value.
    rng
        Generator to sample the outcome.
    """
    f = np.asarray(lo, dtype=complex).reshape(-1)
    M = state.num_modes
    if f.shape != (M,):
        raise ShapeError(f"lo must have one entry per mode ({M}), got shape {f.shape}")
    if not math.isclose(math.sqrt(float(np.sum(np.abs(f) ** 2))), 1.0, abs_tol=1e-10):
        raise InputError("lo must have unit norm (model mode mismatch as loss, or normalise it)")
    Z = np.column_stack([f, np.eye(M, dtype=complex)])
    Q, _ = la.qr(Z)
    U = np.asarray(Q[:, :M], dtype=complex)
    U[:, 0] = f  # QR fixes the first column up to a phase; restore f exactly
    rotated = change_basis(state, U)
    res = homodyne(rotated, 0, phi, efficiency=efficiency, electronic=electronic, outcome=outcome, rng=rng)
    return res._replace(basis=U[:, 1:] if M > 1 else None)


def sample_homodyne(
    state: GaussianState,
    modes: Sequence[int],
    phis: float | Sequence[float],
    shots: int,
    rng: np.random.Generator,
    *,
    efficiency: float = 1.0,
    electronic: float = 0.0,
) -> FloatArray:
    """``shots`` joint samples of simultaneous homodynes on ``modes`` (no conditioning), shape ``(shots, len(modes))``.

    Parameters
    ----------
    state
        The state.
    modes
        Measured modes.
    phis
        Local-oscillator phases, one per mode or one for all.
    shots
        Number of samples.
    rng
        Random generator.
    efficiency, electronic
        Detector model (as :func:`homodyne`).
    """
    if shots < 1:
        raise InputError(f"shots must be >= 1, got {shots}")
    ms = [int(m) for m in modes]
    idx = cv.mode_indices(ms, state.num_modes)
    n = len(ms)
    phs = np.broadcast_to(np.asarray(phis, dtype=float), (n,))
    L = np.zeros((n, 2 * n))
    for j, ph in enumerate(phs):
        L[j, [j, n + j]] = cv.homodyne_vector(float(ph))
    mu = L @ state.mean[idx]
    Sigma = L @ state.cov[np.ix_(idx, idx)] @ L.T + homodyne_noise(efficiency, electronic) * np.eye(n)
    return np.asarray(mu + rng.standard_normal((shots, n)) @ la.cholesky((Sigma + Sigma.T) / 2).T, dtype=float)
