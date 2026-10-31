"""Squeezed-light sources: synchronously pumped OPO and single-pass waveguide (S11).

Both sources take the :class:`~cvsim.hardware.jsa.Supermodes` of their crystal's JSA and produce
the state of the kept supermodes, each squeezed in ``x``, as a
:class:`~cvsim.gaussian.NoiseLedger` in which every imperfection is a booked step
(CONVENTIONS.md, "Sources"):

1. gate ``⊗_k S(r_k)``: the ideal squeezing (component ``"<source> squeezing"``);
2. ``"gain fluctuations"`` (kind ``"excess"``), if :class:`~cvsim.hardware.excess.ExcessNoise`
   asks for them: exact second moments of the fluctuating squeezing;
3. the source's losses (kind ``"loss"``): ``"OPO escape"``; or ``"waveguide propagation loss"``
   then ``"waveguide output coupling"``;
4. ``"excess antisqueezing"`` (kind ``"excess"``), if asked: added ``p`` noise at the output.

**SPOPO** below threshold (Patera et al. 2010; Collett and Gardiner 1984). The supermodes of the
cavity are those of the single-pass JSA; supermode ``k`` sees the pump parameter
``σ_k = σ s_k/s₀`` (``σ = √(P/P_thr)``, threshold of supermode 0 at ``σ = 1``). At analysis
(sideband) frequency ``Ω``, with cavity half width ``γ`` and escape efficiency ``η``, the output
spectra in shot-noise units are

    S_∓(Ω) = 1 ∓ η · 4σ_k / ((1 ± σ_k)² + (Ω/γ)²).

They are exactly pure squeezing ``e^{2r_k(Ω)} = ((1 + σ_k)² + x²)/((1 − σ_k)² + x²)``,
``x = Ω/γ``, followed by a loss ``η``, which is how the ledger books them. A cavity of
repetition rate (free spectral range) ``f``, output coupler ``T`` and other round-trip losses
``L`` has (good-cavity limit, ``T, L ≪ 1``) ``η = T/(T + L)`` and ``γ = f (T + L)/2`` (rad/s).

**Single-pass waveguide** with distributed loss. Squeezing builds up uniformly along the length
``ℓ`` in each supermode (``r_k/ℓ`` per metre, no time ordering) while the power attenuation
coefficient ``κ`` acts on every frequency alike, so the two commute with the supermode basis and
the exact solution of ``dV/dz = A V + V Aᵀ + (κ/2) I`` is, per supermode, the gate ``S(r_k)``
followed by the channel

    X = √η I,  η = e^{−κℓ},   Y = (κℓ/2) diag((1 − e^{−(2r + κℓ)})/(2r + κℓ), (e^{2r − κℓ} − 1)/(2r − κℓ)),

booked as ``"waveguide propagation loss"``; ``effective_transmissivity`` gives the lumped loss
that would cost the same squeezing. A lumped output coupling (facet and collection) follows.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core import units
from ..core._types import FloatArray
from ..errors import ApproximationWarning, InputError
from ..gaussian import GaussianChannel, GaussianState, NoiseLedger
from .excess import ExcessNoise, antisqueezing_noise, gain_average, gain_fluctuation_noise
from .jsa import Supermodes

__all__ = ["SPOPO", "SourceOutput", "Spectrum", "Waveguide", "waveguide_propagation"]

#: Default analysis frequency of the SPOPO, Ω = 2π × 1 MHz (rad/s).
DEFAULT_ANALYSIS_FREQUENCY = units.angular(1.0 * units.MHZ)


def _check(name: str, value: float, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False) -> float:
    v = float(value)
    ok = math.isfinite(v) and (v > lo if lo_open else v >= lo) and (v < hi if hi_open else v <= hi)
    if not ok:
        interval = f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}"
        raise InputError(f"{name} must be in {interval}, got {value}")
    return v


def _diag_channel(x: FloatArray, y: FloatArray) -> GaussianChannel:
    """Channel with diagonal ``X`` and ``Y`` (xxpp vectors of length ``2m``)."""
    return GaussianChannel(np.diag(x), np.diag(y))


def _relative_expm1(z: FloatArray) -> FloatArray:
    """``(e^z − 1)/z``, continuous at 0."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-8
    safe = np.where(small, 1.0, z)
    return np.where(small, 1.0 + z / 2.0, np.expm1(safe) / safe)


def waveguide_propagation(r: npt.ArrayLike, attenuation_length: float) -> tuple[FloatArray, FloatArray]:
    """Exact ``(Y_x, Y_p)`` per supermode of squeezing ``r`` built up over a length of loss ``κℓ``.

    Together with ``X = e^{−κℓ/2} diag(e^{−r}, e^{r})`` this is the solution of
    ``dV/dz = A V + V Aᵀ + (κ/2) I``, ``A = diag(−r/ℓ − κ/2, r/ℓ − κ/2)``, for any input:
    ``Y_x = (κℓ/2)(1 − e^{−(2r + κℓ)})/(2r + κℓ)``, ``Y_p = (κℓ/2)(e^{2r − κℓ} − 1)/(2r − κℓ)``.

    Parameters
    ----------
    r
        Squeezing parameters accumulated over the length (any sign).
    attenuation_length
        ``κℓ`` (power attenuation coefficient × length), >= 0.
    """
    r_ = np.atleast_1d(np.asarray(r, dtype=float))
    a = _check("attenuation_length", attenuation_length, 0.0, math.inf, hi_open=True)
    yx = 0.5 * a * _relative_expm1(-(2 * r_ + a))
    yp = 0.5 * a * _relative_expm1(2 * r_ - a)
    return yx, yp


class Spectrum(NamedTuple):
    """Squeezed and antisqueezed noise spectra in shot-noise units, ``(modes, frequencies)``."""

    omega: FloatArray
    squeezed: FloatArray
    antisqueezed: FloatArray


@dataclass(frozen=True)
class SourceOutput:
    """The state a source emits, in its supermode basis, with its noise ledger.

    Attributes
    ----------
    name
        Source name (prefix of the squeezing component).
    supermodes
        The supermodes (mode functions on the frequency grid).
    squeezing
        Ideal squeezing parameters ``r_k`` of the kept supermodes (before any imperfection).
    ledger
        Every step from the vacuum to the output (see the module docstring).
    """

    name: str
    supermodes: Supermodes
    squeezing: FloatArray
    ledger: NoiseLedger

    @property
    def state(self) -> GaussianState:
        """The output state of the kept supermodes (supermode ``k`` is mode ``k``)."""
        return self.ledger.state

    @property
    def num_modes(self) -> int:
        """Number of supermodes in the state."""
        return int(self.squeezing.size)

    def quadrature_db(self) -> tuple[FloatArray, FloatArray]:
        """Squeezed (``x``) and antisqueezed (``p``) noise of each supermode, dB relative to shot noise."""
        V = self.state.cov
        M = self.num_modes
        return cv.db_from_variance(np.diag(V)[:M]), cv.db_from_variance(np.diag(V)[M:])

    def in_pixel_basis(self) -> GaussianState:
        """The same state on the frequency pixels of the grid (the supermodes left out in vacuum).

        Pixel ``j`` is ``a_j = √δ a(ω₀ + Ω_j)``; the state is ``V = S(U) (V_sm ⊕ I/2) S(U)ᵀ`` with
        ``U`` the full Takagi unitary (:func:`cvsim.gaussian.symplectic.change_basis` takes it back).
        """
        sm = self.supermodes
        n, M = sm.grid.size, self.num_modes
        full = self.state.tensor(GaussianState.vacuum(n - M)) if n > M else self.state
        S = cv.interferometer(sm.unitary)
        V = S @ full.cov @ S.T
        return GaussianState(S @ full.mean, (V + V.T) / 2, validate=False)


def _start(name: str, r: FloatArray, excess: ExcessNoise) -> NoiseLedger:
    M = r.size
    ledger = NoiseLedger.start(GaussianState.vacuum(M)).gate(
        np.diag(np.concatenate([np.exp(-r), np.exp(r)])), component=f"{name} squeezing"
    )
    if excess.gain_rms > 0:
        warnings.warn(
            "gain fluctuations: the output is a mixture of squeezed states; cvsim keeps its exact second "
            "moments (moment-matched Gaussian)",
            ApproximationWarning,
            stacklevel=3,
        )
        ledger = ledger.channel(
            GaussianChannel.additive_noise(np.diag(gain_fluctuation_noise(r, excess.gain_rms))),
            component="gain fluctuations",
            kind="excess",
        )
    return ledger


def _finish(ledger: NoiseLedger, r: FloatArray, excess: ExcessNoise) -> NoiseLedger:
    if excess.antisqueezing_db <= 0:
        return ledger
    M = r.size
    vp = np.diag(ledger.state.cov)[M:]
    eps = antisqueezing_noise(vp, r, excess)
    return ledger.channel(
        GaussianChannel.additive_noise(np.diag(np.concatenate([np.zeros(M), eps]))),
        component="excess antisqueezing",
        kind="excess",
    )


def _uniform_loss(eta: float, M: int) -> GaussianChannel:
    return GaussianChannel(math.sqrt(eta) * np.eye(2 * M), 0.5 * (1.0 - eta) * np.eye(2 * M))


# ------------------------------------------------------------------------------- SPOPO
@dataclass(frozen=True)
class SPOPO:
    """Synchronously pumped OPO below threshold.

    Parameters
    ----------
    supermodes
        Supermodes of the crystal's JSA (the cavity's supermodes).
    escape_efficiency
        ``η = T/(T + L)`` in ``(0, 1]``.
    linewidth
        Cavity half width at half maximum ``γ`` (rad/s), > 0.
    pump
        ``σ = √(P/P_thr)`` in ``[0, 1)``.
    """

    supermodes: Supermodes
    escape_efficiency: float
    linewidth: float
    pump: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.supermodes, Supermodes):
            raise InputError(f"supermodes must be Supermodes, got {type(self.supermodes).__name__}")
        _check("escape_efficiency", self.escape_efficiency, 0.0, 1.0, lo_open=True)
        _check("linewidth", self.linewidth, 0.0, math.inf, lo_open=True, hi_open=True)
        if not (math.isfinite(float(self.pump)) and 0 <= self.pump < 1):
            raise InputError(
                f"pump σ = √(P/P_thr) must be in [0, 1) (below threshold; above threshold is not modelled), "
                f"got {self.pump}"
            )

    @classmethod
    def from_cavity(
        cls,
        supermodes: Supermodes,
        repetition_rate: float = 76.0 * units.MHZ,
        transmission: float = 0.20,
        loss: float = 0.01,
        pump: float = 0.5,
    ) -> SPOPO:
        """From the cavity's parameters (good-cavity limit): ``η = T/(T + L)``, ``γ = f (T + L)/2``.

        Parameters
        ----------
        supermodes
            Supermodes of the crystal's JSA.
        repetition_rate
            Pump repetition rate = cavity free spectral range ``f`` (Hz), > 0.
        transmission
            Output-coupler transmission ``T`` in ``(0, 1)``.
        loss
            Other round-trip losses ``L`` in ``[0, 1)``.
        pump
            ``σ = √(P/P_thr)`` in ``[0, 1)``.
        """
        f = _check("repetition_rate", repetition_rate, 0.0, math.inf, lo_open=True, hi_open=True)
        T = _check("transmission", transmission, 0.0, 1.0, lo_open=True, hi_open=True)
        L = _check("loss", loss, 0.0, 1.0, hi_open=True)
        return cls(supermodes, T / (T + L), f * (T + L) / 2.0, pump)

    @property
    def pump_parameters(self) -> FloatArray:
        """``σ_k = σ s_k/s₀`` of the kept supermodes."""
        s = self.supermodes.coefficients[: self.supermodes.kept]
        return np.asarray(self.pump * s / s[0], dtype=float)

    def squeezing_parameters(self, omega: float = DEFAULT_ANALYSIS_FREQUENCY) -> FloatArray:
        """``r_k(Ω) = ½ ln(((1 + σ_k)² + x²)/((1 − σ_k)² + x²))``, ``x = Ω/γ``, of the kept supermodes.

        Parameters
        ----------
        omega
            Analysis frequency ``Ω`` (rad/s).
        """
        w = float(omega)
        if not math.isfinite(w):
            raise InputError(f"omega must be finite, got {omega}")
        x2 = (w / self.linewidth) ** 2
        sk = self.pump_parameters
        return np.asarray(0.5 * (np.log1p(sk * (2 + sk) + x2) - np.log((1 - sk) ** 2 + x2)), dtype=float)

    def spectrum(self, omega: npt.ArrayLike) -> Spectrum:
        """Output noise spectra ``S_∓(Ω)`` (shot-noise units) of the kept supermodes (no excess noise).

        Parameters
        ----------
        omega
            Analysis frequencies ``Ω`` (rad/s).
        """
        w = np.atleast_1d(np.asarray(omega, dtype=float))
        if w.ndim != 1 or not np.all(np.isfinite(w)):
            raise InputError("omega must be a finite 1-D array")
        x2 = (w[None, :] / self.linewidth) ** 2
        sk = self.pump_parameters[:, None]
        eta = self.escape_efficiency
        sq = 1.0 - eta * 4 * sk / ((1 + sk) ** 2 + x2)
        anti = 1.0 + eta * 4 * sk / ((1 - sk) ** 2 + x2)
        return Spectrum(w, sq, anti)

    def threshold_power_ratio(self) -> float:
        """``P/P_thr = σ²``."""
        return float(self.pump) ** 2

    def output(self, omega: float = DEFAULT_ANALYSIS_FREQUENCY, excess: ExcessNoise | None = None) -> SourceOutput:
        """State of the kept supermodes at analysis frequency ``Ω``, with its ledger.

        Parameters
        ----------
        omega
            Analysis frequency ``Ω`` (rad/s), default 2π × 1 MHz.
        excess
            Excess-noise settings (default none).
        """
        ex = ExcessNoise() if excess is None else excess
        r = self.squeezing_parameters(omega)
        ledger = _start("OPO", r, ex)
        ledger = ledger.channel(_uniform_loss(self.escape_efficiency, r.size), component="OPO escape", kind="loss")
        return SourceOutput("OPO", self.supermodes, r, _finish(ledger, r, ex))


# ------------------------------------------------------------------------------- waveguide
@dataclass(frozen=True)
class Waveguide:
    """Single-pass χ⁽²⁾ waveguide with distributed propagation loss.

    Parameters
    ----------
    supermodes
        Supermodes of the waveguide's JSA (computed with the dispersion of this length).
    length
        Length ``ℓ`` (m), > 0.
    loss_db_per_cm
        Propagation loss (dB/cm), >= 0.
    r0
        Squeezing parameter of supermode 0 for a lossless guide (nepers), >= 0.
    output_coupling
        Lumped transmission of the output facet and collection, in ``(0, 1]``.
    """

    supermodes: Supermodes
    length: float = 10.0 * units.MM
    loss_db_per_cm: float = 0.2
    r0: float = 1.0
    output_coupling: float = 0.9

    def __post_init__(self) -> None:
        if not isinstance(self.supermodes, Supermodes):
            raise InputError(f"supermodes must be Supermodes, got {type(self.supermodes).__name__}")
        _check("length", self.length, 0.0, math.inf, lo_open=True, hi_open=True)
        _check("loss_db_per_cm", self.loss_db_per_cm, 0.0, math.inf, hi_open=True)
        _check("r0", self.r0, 0.0, math.inf, hi_open=True)
        _check("output_coupling", self.output_coupling, 0.0, 1.0, lo_open=True)

    @property
    def attenuation_length(self) -> float:
        """``κℓ``: the guide's total propagation loss in nepers of power."""
        return units.attenuation_from_db_per_cm(self.loss_db_per_cm) * float(self.length)

    @property
    def propagation_transmissivity(self) -> float:
        """``η = e^{−κℓ}``."""
        return math.exp(-self.attenuation_length)

    @property
    def squeezing_parameters(self) -> FloatArray:
        """Lossless squeezing ``r_k = r₀ s_k/s₀`` of the kept supermodes."""
        return self.supermodes.squeezing(self.r0)

    def effective_transmissivity(self) -> FloatArray:
        """Lumped transmissivity that, after lossless squeezing ``r_k``, costs the same squeezing as
        the distributed loss: ``η_eff = (1 − 2V_x)/(1 − e^{−2r_k})`` (``nan`` where ``r_k = 0``)."""
        r = self.squeezing_parameters
        yx, _ = waveguide_propagation(r, self.attenuation_length)
        vx = 0.5 * self.propagation_transmissivity * np.exp(-2 * r) + yx
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(r > 0, (1 - 2 * vx) / -np.expm1(-2 * r), np.nan)

    def output(self, excess: ExcessNoise | None = None) -> SourceOutput:
        """State of the kept supermodes at the output, with its ledger.

        Parameters
        ----------
        excess
            Excess-noise settings (default none). With gain fluctuations the propagation-loss noise
            is averaged over the fluctuating ``r`` as well (Gauss–Hermite), so the second moments stay
            exact.
        """
        ex = ExcessNoise() if excess is None else excess
        r = self.squeezing_parameters
        M = r.size
        a = self.attenuation_length
        if ex.gain_rms > 0:
            yx = gain_average(lambda q: waveguide_propagation(q, a)[0], r, ex.gain_rms)
            yp = gain_average(lambda q: waveguide_propagation(q, a)[1], r, ex.gain_rms)
        else:
            yx, yp = waveguide_propagation(r, a)
        ledger = _start("waveguide", r, ex)
        x = math.sqrt(self.propagation_transmissivity) * np.ones(2 * M)
        ledger = ledger.channel(
            _diag_channel(x, np.concatenate([yx, yp])), component="waveguide propagation loss", kind="loss"
        )
        ledger = ledger.channel(
            _uniform_loss(self.output_coupling, M), component="waveguide output coupling", kind="loss"
        )
        return SourceOutput("waveguide", self.supermodes, r, _finish(ledger, r, ex))
