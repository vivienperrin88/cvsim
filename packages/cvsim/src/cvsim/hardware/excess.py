"""Excess antisqueezing: noise beyond what loss explains, its models and its calibration.

Loss alone takes a pure squeezed state ``V = ½ diag(e^{−2r}, e^{2r})`` to
``V_∓ = η e^{∓2r} + 1 − η`` (shot-noise units). Measured antisqueezing is often larger. Two
models are offered, both booked in the noise ledger with kind ``"excess"`` (CONVENTIONS.md,
"Sources"):

* **phase-sensitive added noise** (component ``"excess antisqueezing"``): ``Y = diag(0, ε_k)`` on
  the antisqueezed quadrature of each supermode at the source output, a random displacement of
  ``p`` (completely positive). It is specified by ``antisqueezing_db``, the excess of supermode 0
  over the antisqueezing the source would have without it; supermode ``k`` gets
  ``ε_k = ε₀ r_k/r₀`` (``scaling="proportional"``, default) or ``ε₀`` (``"constant"``);
* **gain fluctuations** (component ``"gain fluctuations"``): the squeezing parameter of each
  supermode fluctuates from pulse to pulse, ``r ~ N(r_k, (g r_k)²)`` with ``g = gain_rms``. The
  state is then a mixture of squeezed states, not Gaussian; cvsim keeps its exact second moments
  (``⟨e^{±2r}⟩ = e^{±2r_k + 2σ_k²}``) and warns with
  :class:`~cvsim.errors.ApproximationWarning`, as for phase diffusion.

Phase-lock noise (S13) and local-oscillator mismatch (S12) are separate models.

:func:`calibrate` inverts a measurement: from squeezing and antisqueezing in dB it returns the
squeezing parameter and the efficiency under pure loss, or, given the efficiency, the squeezing
parameter and the excess antisqueezing at the measurement point.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..core import _checks as ck
from ..core._types import FloatArray
from ..errors import CalibrationError, InputError

__all__ = [
    "SCALINGS",
    "Calibration",
    "ExcessNoise",
    "antisqueezing_noise",
    "calibrate",
    "gain_average",
    "gain_fluctuation_noise",
]

#: How the phase-sensitive excess noise scales across supermodes.
SCALINGS = ("proportional", "constant")

#: Gauss–Hermite nodes used to average a smooth function of a Gaussian squeezing parameter.
_GH_NODES = 48


@dataclass(frozen=True)
class ExcessNoise:
    """Excess-noise settings of a source (both off by default).

    Parameters
    ----------
    antisqueezing_db
        Excess antisqueezing of supermode 0 (dB over the source's antisqueezing without it), >= 0.
    scaling
        ``"proportional"`` (``ε_k ∝ r_k``) or ``"constant"``, see :data:`SCALINGS`.
    gain_rms
        Relative rms of the squeezing parameter, ``σ_k = gain_rms · r_k``, >= 0.
    """

    antisqueezing_db: float = 0.0
    scaling: str = "proportional"
    gain_rms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("antisqueezing_db", "gain_rms"):
            object.__setattr__(self, name, ck.non_negative(name, getattr(self, name)))
        if self.scaling not in SCALINGS:
            raise InputError(f"scaling must be one of {SCALINGS}, got {self.scaling!r}")

    @property
    def active(self) -> bool:
        """Whether any excess noise is switched on."""
        return self.antisqueezing_db > 0 or self.gain_rms > 0


def _as_r(r: npt.ArrayLike) -> FloatArray:
    r_ = np.atleast_1d(np.asarray(r, dtype=float))
    if r_.ndim != 1 or not np.all(np.isfinite(r_)):
        raise InputError(f"squeezing parameters must be a finite 1-D array, got shape {r_.shape}")
    return r_


def gain_fluctuation_noise(r: npt.ArrayLike, gain_rms: float) -> FloatArray:
    """Added covariance of squeezed vacua whose ``r_k`` fluctuate, ``r ~ N(r_k, (gain_rms r_k)²)``.

    Returns the diagonal ``(Y_x..., Y_p...)`` (xxpp, ħ = 1):
    ``Y = ½ (e^{∓2r_k}) (e^{2σ_k²} − 1)``, the exact difference between the averaged and the
    nominal covariance of the squeezed vacuum.

    Parameters
    ----------
    r
        Nominal squeezing parameters ``r_k`` (x squeezed).
    gain_rms
        Relative rms, >= 0.
    """
    r_ = _as_r(r)
    g = ck.non_negative("gain_rms", gain_rms)
    grow = np.expm1(2.0 * (g * r_) ** 2)
    return np.concatenate([0.5 * np.exp(-2 * r_) * grow, 0.5 * np.exp(2 * r_) * grow])


def gain_average(fn: Callable[[FloatArray], FloatArray], r: npt.ArrayLike, gain_rms: float) -> FloatArray:
    """``E[fn(r)]`` over ``r ~ N(r_k, (gain_rms r_k)²)``, element-wise in ``k``, by Gauss–Hermite
    quadrature (48 nodes: exact for polynomials of degree 95, to round-off for the smooth
    functions of ``r`` of the source models).

    Parameters
    ----------
    fn
        Vectorised function of an array of squeezing parameters, returning an array whose last axis
        is ``k`` (the same shape, or e.g. ``(2, k)`` to average two quantities in one pass).
    r
        Nominal squeezing parameters.
    gain_rms
        Relative rms, >= 0.
    """
    r_ = _as_r(r)
    g = ck.non_negative("gain_rms", gain_rms)
    x, w = np.polynomial.hermite_e.hermegauss(_GH_NODES)  # weight exp(−x²/2)
    w = w / w.sum()
    total: FloatArray | float = 0.0
    for xi, wi in zip(x, w):
        total = total + wi * np.asarray(fn(r_ * (1.0 + g * xi)), dtype=float)
    return np.asarray(total, dtype=float)


def antisqueezing_noise(antisqueezed: npt.ArrayLike, r: npt.ArrayLike, excess: ExcessNoise) -> FloatArray:
    """Added ``p`` variances ``ε_k`` (ħ = 1) of the phase-sensitive excess model.

    ``ε₀ = V_p,0 (10^{antisqueezing_db/10} − 1)`` raises the antisqueezed variance ``V_p,0`` of
    supermode 0 by ``antisqueezing_db``; the others follow ``excess.scaling``.

    Parameters
    ----------
    antisqueezed
        Antisqueezed variances ``V_p,k`` at the point where the noise is added (ħ = 1).
    r
        Squeezing parameters ``r_k`` (for the proportional scaling).
    excess
        The settings.
    """
    vp = _as_r(antisqueezed)
    r_ = _as_r(r)
    if vp.shape != r_.shape:
        raise InputError(f"need one variance per squeezing parameter, got {vp.size} and {r_.size}")
    eps0 = float(vp[0]) * math.expm1(excess.antisqueezing_db * math.log(10.0) / 10.0)
    if excess.scaling == "constant":
        return np.full_like(r_, eps0)
    if r_[0] <= 0:
        return np.zeros_like(r_)
    return np.asarray(eps0 * r_ / r_[0], dtype=float)


class Calibration(NamedTuple):
    """Result of :func:`calibrate` (variances in shot-noise units)."""

    r: float
    efficiency: float
    excess: float
    excess_db: float


def _fit_pure_loss(squeezing_db: float, antisqueezing_db: float) -> Calibration:
    """``r`` and ``η`` of the pure-loss model through ``V₋``, ``V₊``, without cancellation.

    With ``x_∓ = ∓ dB · ln10/10`` (``V₋ = e^{−x₋}``, ``V₊ = e^{x₊}``): ``d_p = V₊ − 1 = expm1(x₊)``,
    ``d_m = 1 − V₋ = −expm1(−x₋)`` and, with ``δ = (x₊ − x₋)/2 = ½ ln(V₋V₊)``, ``μ = (x₊ + x₋)/2``,
    ``V₊ + V₋ − 2 = 2 (2 e^δ sinh²(μ/2) + expm1(δ))``: a sum of non-negative terms when ``δ >= 0``.
    """
    ln10_10 = math.log(10.0) / 10.0
    xp, xm = antisqueezing_db * ln10_10, -squeezing_db * ln10_10
    dp, dm = math.expm1(xp), -math.expm1(-xm)
    delta, mu = 0.5 * (xp - xm), 0.5 * (xp + xm)
    excess_sum = 2.0 * (2.0 * math.exp(delta) * math.sinh(0.5 * mu) ** 2 + math.expm1(delta))
    if excess_sum <= 0.0 or dp * dm >= excess_sum:  # within tolerance of a pure state
        return Calibration(0.5 * mu, 1.0, 0.0, 0.0)
    # e^{2r} = d_p/d_m = 1 + (V₊ + V₋ − 2)/d_m
    r = 0.5 * math.log1p(excess_sum / dm) if dm >= sys.float_info.min else math.inf
    eta = dp * dm / excess_sum
    if not (math.isfinite(r) and eta >= sys.float_info.min):
        raise CalibrationError(
            f"squeezing {squeezing_db} dB is too weak against antisqueezing {antisqueezing_db} dB to fit r and η "
            f"in double precision (1 − V₋ = {dm:.3e}): give the efficiency"
        )
    return Calibration(r, eta, 0.0, 0.0)


def calibrate(
    squeezing_db: float, antisqueezing_db: float, efficiency: float | None = None, rtol: float = 1e-9
) -> Calibration:
    """Squeezing parameter, efficiency and excess antisqueezing of a measured squeezed mode.

    With ``V_∓ = 10^{dB/10}`` (shot-noise units) and the pure-loss model
    ``V_∓ = η e^{∓2r} + 1 − η``:

    * ``efficiency=None``: solve for both, ``e^{2r} = (V₊ − 1)/(1 − V₋)``,
      ``η = (V₊ − 1)/(e^{2r} − 1) = (V₊ − 1)(1 − V₋)/(V₊ + V₋ − 2)``; always possible for physical
      data (``V₋V₊ >= 1`` gives ``η <= 1``), with no excess. The differences are formed from the dB
      values without cancellation, so weak squeezing keeps full precision. Data within the
      tolerance ``rtol`` of a pure state (``V₋V₊`` in ``[1 − rtol, 1]``, or ``η`` rounding above 1) are
      read as pure: ``η = 1``, ``r = ¼ ln(V₊/V₋)``;
    * ``efficiency`` given (independently calibrated, e.g. escape × propagation × detection):
      ``r`` from the squeezing, and the excess ``V₊ − (η e^{2r} + 1 − η)`` in shot-noise units and
      in dB (``10 log10`` of the ratio). It is the excess at the measurement point; the
      phase-sensitive model of :class:`ExcessNoise` adds it there.

    Parameters
    ----------
    squeezing_db
        Squeezed-quadrature noise relative to shot noise (dB), < 0.
    antisqueezing_db
        Antisqueezed-quadrature noise relative to shot noise (dB), > 0.
    efficiency
        Total efficiency ``η`` in ``(0, 1]``, or ``None`` to fit it.
    rtol
        Relative tolerance of the consistency checks; an excess below ``rtol · V₊`` is reported as 0.

    Raises
    ------
    CalibrationError
        The data violate the uncertainty principle (``V₋ V₊ < 1``), the squeezing is deeper than
        ``efficiency`` allows (``V₋ <= 1 − η``), or the antisqueezing is below the loss-only
        prediction.
    """
    s, a = ck.finite("squeezing_db", squeezing_db), ck.finite("antisqueezing_db", antisqueezing_db)
    if not s < 0 < a:
        raise InputError(f"need squeezing_db < 0 < antisqueezing_db, got {squeezing_db} and {antisqueezing_db}")
    tol = ck.in_range("rtol", rtol, 0.0, 1.0, hi_open=True)
    vm, vp = 10.0 ** (s / 10.0), 10.0 ** (a / 10.0)
    if vm * vp < 1.0 - tol:
        raise CalibrationError(
            f"squeezing {s} dB and antisqueezing {a} dB violate the uncertainty principle (V₋V₊ = {vm * vp:.6f} < 1)"
        )
    if efficiency is None:
        return _fit_pure_loss(s, a)
    eta = ck.in_range("efficiency", efficiency, 0.0, 1.0, lo_open=True)
    # without cancellation: d_m = 1 − V₋ = η(1 − e^{−2r}), d_p = V₊ − 1; the loss-only prediction
    # is V₊ = 1 + η d_m/(η − d_m)
    ln10_10 = math.log(10.0) / 10.0
    dm, dp = -math.expm1(s * ln10_10), math.expm1(a * ln10_10)
    if dm >= eta:
        raise CalibrationError(
            f"squeezing {s} dB is deeper than an efficiency of {eta} allows (at most "
            f"{10 * math.log10(1 - eta) if eta < 1 else -math.inf:.3f} dB)"
        )
    r = -0.5 * math.log1p(-dm / eta)
    predicted_excess = eta * dm / (eta - dm)  # predicted V₊ − 1
    predicted = 1.0 + predicted_excess
    excess = dp - predicted_excess
    if excess < -tol * vp:
        raise CalibrationError(
            f"antisqueezing {a} dB is below the loss-only prediction {10 * math.log10(predicted):.4f} dB for "
            f"efficiency {eta}: the efficiency is too low or the squeezing mis-measured"
        )
    if excess <= tol * vp:  # round-off of a loss-only measurement
        return Calibration(r, eta, 0.0, 0.0)
    return Calibration(r, eta, excess, 10.0 * math.log1p(excess / predicted) / math.log(10.0))
