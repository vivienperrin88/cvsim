"""Joint spectral amplitude (JSA) of degenerate down-conversion and its Schmidt supermodes.

A χ⁽²⁾ crystal pumped around ``2ω₀`` couples pairs of frequencies ``ω₀ + Ω_s`` and ``ω₀ + Ω_i``
with the amplitude (CONVENTIONS.md, "Sources")

    f(Ω_s, Ω_i) = α(Ω_s + Ω_i) · φ(Ω_s, Ω_i),

the product of the pump spectral amplitude ``α`` and the phase-matching function ``φ``. In
degenerate type-0 or type-I down-conversion signal and idler are the same field, so ``f`` is
symmetric. Phase matching is parametrised by the crystal's dispersion only (no Sellmeier data):

    Δk L = τ Ω₊ + ½ β_p Ω₊² − ½ β_s (Ω_s² + Ω_i²),     Ω₊ = Ω_s + Ω_i,

with ``τ = L (k'_p − k'_s)`` the group-velocity mismatch (s), ``β_p = L k''_p`` and
``β_s = L k''_s`` the group-delay dispersions of pump and signal (s²). The first-order term depends
on ``Ω₊`` only; the signal GVD ``β_s`` alone sets the bandwidth along ``Ω_s − Ω_i``, so it must be
non-zero for a finite number of modes.

On a uniform grid of detunings (spacing ``δ``) the pixel modes ``a_j = √δ a(ω₀ + Ω_j)`` see the
symmetric matrix ``J_jk = f(Ω_j, Ω_k)``, normalised to ``‖J‖_F = 1``. Its Takagi factorisation
``J = U diag(s) Uᵀ`` (:func:`cvsim.core.linalg.takagi`) gives the Schmidt coefficients ``s_k``
(``Σ s_k² = 1``, descending) and the **supermodes** ``b_k = Σ_j conj(U_jk) a_j``: column ``k`` of
``U`` is the mode function of supermode ``k`` (the convention of
:func:`cvsim.gaussian.symplectic.change_basis`), sampled on the grid; ``U_jk/√δ`` is the continuous
function, ``∫|u_k|² dΩ = 1``. The Schmidt number is ``K = 1/Σ s_k⁴``.

Undepleted pump, no time ordering: a source of gain ``G`` produces the state
``exp(½ Σ_jk (conj(Z_jk) a_j a_k − Z_jk a_j† a_k†))|0⟩`` with ``Z = G J``, i.e. supermode ``k``
squeezed in ``x`` by ``r_k = G s_k``. Sources (:mod:`cvsim.hardware.sources`) give the gain
through the squeezing ``r₀`` of supermode 0: ``r_k = r₀ s_k/s₀``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..core import _checks as ck
from ..core import linalg as la
from ..core import units
from ..core._types import ComplexArray, FloatArray
from ..errors import ApproximationWarning, InputError

__all__ = [
    "SCHMIDT_RTOL",
    "SINC_GAUSSIAN_GAMMA",
    "FrequencyGrid",
    "PhaseMatching",
    "Pump",
    "Supermodes",
    "joint_spectral_amplitude",
    "schmidt_decomposition",
    "supermodes",
]

#: Width parameter of the Gaussian approximation of ``sinc``: ``sinc x ≈ exp(−γ x²)``, both ½ at
#: ``x ≈ 1.8955`` (equal amplitude FWHM; U'Ren et al. 2005).
SINC_GAUSSIAN_GAMMA = 0.193

#: Degeneracy tolerance of the Schmidt decomposition. Schmidt coefficients span many decades;
#: Takagi's default (``1e-8``) would merge every coefficient below ``1e-8 s₀`` into the null group,
#: costing a reconstruction error of that size. Much smaller values split clusters of tiny,
#: nearly equal coefficients whose singular vectors are ill-determined, and ``U`` loses unitarity
#: (``3e-11`` at ``1e-12`` on a 240-point sinc JSA). ``1e-11`` keeps both errors near ``5e-12``;
#: exactly degenerate coefficients (the sideband pairs of the CW limit) still group.
SCHMIDT_RTOL = 1e-11

_Spectrum = Callable[[FloatArray], ComplexArray]
_Joint = Callable[[FloatArray, FloatArray], ComplexArray]


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


# ------------------------------------------------------------------------------- frequency grid
class FrequencyGrid:
    """Uniform grid of detunings ``Ω_j`` (rad/s) from the degenerate frequency ``ω₀``.

    Parameters
    ----------
    detunings
        Strictly increasing, uniformly spaced detunings (rad/s), at least 2.
    """

    __slots__ = ("_detunings",)

    def __init__(self, detunings: npt.ArrayLike):
        d = np.array(detunings, dtype=float).reshape(-1)
        if d.size < 2 or not np.all(np.isfinite(d)):
            raise InputError(f"a frequency grid needs at least 2 finite detunings, got {d.size}")
        steps = np.diff(d)
        if steps.min() <= 0 or (steps.max() - steps.min()) > 1e-9 * steps.mean():
            raise InputError("detunings must be strictly increasing and uniformly spaced")
        self._detunings = _readonly(d)

    @classmethod
    def centered(cls, n: int, half_span: float) -> FrequencyGrid:
        """``n`` points from ``−half_span`` to ``+half_span`` (symmetric: ``Ω_j + Ω_{n−1−j} = 0``).

        Parameters
        ----------
        n
            Number of points, >= 2.
        half_span
            Largest detuning (rad/s), > 0.
        """
        n_ = ck.integer("n", n, 2)
        h = ck.positive("half_span", half_span)
        return cls(np.linspace(-h, h, n_))

    @classmethod
    def auto(
        cls,
        pump: Pump,
        phase_matching: PhaseMatching,
        n: int | None = None,
        extent: float = 4.0,
        points_per_width: int = 2,
        max_points: int = 2048,
    ) -> FrequencyGrid:
        """A centred grid that contains and resolves the JSA.

        ``w₊`` is the JSA's half width along the diagonal ``Ω_s = Ω_i`` (the narrower of the pump's
        and the phase matching's) and ``w₋`` along the anti-diagonal ``Ω_s = −Ω_i`` (phase matching
        only); the support lies within ``|Ω| <= |c|/2 + extent (w₊ + w₋)``, the half span, with ``c``
        the pump's ``center`` (the diagonal sees the pump at ``Ω₊ = 2Ω``). The step resolves the
        scale ``w_r`` of the JSA's main lobe: the smaller of ``w₋``, the phase matching's
        ``half_width_sum`` and half the pump's ``resolution_width`` (its HWHM, or less for a chirped
        pump). By default the number of points puts ``points_per_width`` points across ``w_r`` (at
        most ``max_points``); a grid coarser than that raises an
        :class:`~cvsim.errors.ApproximationWarning`. The side lobes of a ``sinc`` phase matching
        narrow away from the origin and are not resolved to the same standard (pass ``n`` for that).

        Parameters
        ----------
        pump
            Pump spectrum.
        phase_matching
            Phase-matching function.
        n
            Number of points (default: from the resolution).
        extent
            Half span in units of ``w₊ + w₋``, > 0.
        points_per_width
            Resolution target: points across ``w_r``, an integer >= 1.
        max_points
            Cap on the automatic number of points, an integer >= 2.
        """
        ext = ck.positive("extent", extent)
        ppw = ck.integer("points_per_width", points_per_width, 1)
        cap = ck.integer("max_points", max_points, 2)
        w_plus = min(pump.half_width / 2.0, phase_matching.half_width_sum)
        w_minus = phase_matching.half_width_difference
        if not math.isfinite(w_minus):
            raise InputError(
                "the phase matching is unbounded along Ω_s − Ω_i (signal GVD β_s = 0): give gvd_signal "
                "or an explicit FrequencyGrid"
            )
        w_res = min(pump.resolution_width / 2.0, phase_matching.half_width_sum, w_minus)
        half = abs(pump.center) / 2.0 + ext * (w_plus + w_minus)
        if n is None:
            n = min(cap, max(64, math.ceil(2 * half * ppw / w_res) + 1))
        grid = cls.centered(n, half)
        if grid.step > w_res / ppw * (1 + 1e-12):
            warnings.warn(
                f"FrequencyGrid.auto: step {grid.step:.3e} rad/s does not resolve the JSA's shortest scale "
                f"{w_res:.3e} rad/s with {ppw} points; increase n (now {n})",
                ApproximationWarning,
                stacklevel=2,
            )
        return grid

    @property
    def detunings(self) -> FloatArray:
        """Detunings ``Ω_j`` (rad/s, read-only)."""
        return self._detunings

    @property
    def step(self) -> float:
        """Spacing ``δ`` (rad/s)."""
        d = self._detunings
        return float((d[-1] - d[0]) / (d.size - 1))

    @property
    def size(self) -> int:
        """Number of points."""
        return int(self._detunings.size)

    def __repr__(self) -> str:
        d = self._detunings
        return f"FrequencyGrid(n={d.size}, from {d[0]:.4e} to {d[-1]:.4e} rad/s)"


# ------------------------------------------------------------------------------- pump
@dataclass(frozen=True)
class Pump:
    """Spectral amplitude ``α(Ω₊)`` of the pump, as a function of its detuning ``Ω₊`` from ``2ω₀``.

    Build it with :meth:`gaussian` or :meth:`sampled`. Three numbers size grids
    (:meth:`FrequencyGrid.auto`), all in rad/s:

    * ``center`` and ``half_width``: the midpoint and half the distance of the outermost points
      where ``|α|²`` crosses half its maximum, so the span covers every peak above half maximum
      (for a single peak, its centre and its half width at half maximum, HWHM);
    * ``resolution_width``: the shortest scale of ``α``, which sets the grid's step: the narrowest
      HWHM of those peaks, or ``1/max|dφ/dΩ₊|`` over them when the spectral phase ``φ = arg α``
      turns faster (a chirp). ``0`` (default) means ``half_width``.
    """

    amplitude: _Spectrum
    half_width: float
    description: str = ""
    resolution_width: float = 0.0
    center: float = 0.0

    def __post_init__(self) -> None:
        h = ck.positive("half_width", self.half_width)
        r = h if self.resolution_width == 0 else ck.positive("resolution_width", self.resolution_width)
        object.__setattr__(self, "half_width", h)
        object.__setattr__(self, "resolution_width", min(h, r))
        object.__setattr__(self, "center", ck.finite("center", self.center))

    @classmethod
    def gaussian(cls, fwhm: float, chirp: float = 0.0) -> Pump:
        """Gaussian pump of intensity FWHM ``fwhm``, with an optional quadratic spectral phase.

        ``α(Ω) = exp(−Ω²/(2σ²) + i chirp Ω²/2)``, ``σ = fwhm/(2√ln 2)``
        (:func:`cvsim.core.units.amplitude_sigma_from_fwhm`); ``chirp`` is the group-delay
        dispersion of the pump pulse (s²). A wavelength FWHM converts with
        :func:`cvsim.core.units.angular_bandwidth`.

        Parameters
        ----------
        fwhm
            Intensity FWHM of the pump spectrum (rad/s), > 0.
        chirp
            Quadratic spectral phase (s²).
        """
        sigma = units.amplitude_sigma_from_fwhm(fwhm)
        c = ck.finite("chirp", chirp)
        h = float(fwhm) / 2.0

        def amplitude(w: FloatArray) -> ComplexArray:
            return np.exp(-(w**2) / (2 * sigma**2) + 0.5j * c * w**2)

        # φ = c Ω²/2 turns at |dφ/dΩ| = |c| h at the half-maximum points Ω = ±h; 1/(|c| h) < h iff |c| h² > 1
        resolution = 1.0 / (abs(c) * h) if abs(c) * h * h > 1.0 else h
        return cls(amplitude, h, f"gaussian(fwhm={float(fwhm):.4e}, chirp={c:.4e})", resolution)

    @classmethod
    def sampled(cls, detunings: npt.ArrayLike, amplitudes: npt.ArrayLike) -> Pump:
        """A measured or shaped pump spectrum, interpolated linearly (real and imaginary parts), zero
        outside the samples.

        The widths of :class:`Pump` are those of the interpolated spectrum: the half-maximum
        crossings are solved exactly on the linear interpolation (the sample range bounds a peak
        that reaches it), and the phase rate is taken between consecutive samples over each peak
        above half maximum, including the intervals of its crossings.

        Parameters
        ----------
        detunings
            Strictly increasing pump detunings ``Ω₊`` (rad/s).
        amplitudes
            Complex spectral amplitudes at those detunings (any normalisation).
        """
        w = np.array(detunings, dtype=float).reshape(-1)
        a = np.array(amplitudes, dtype=complex).reshape(-1)
        if w.size < 2 or w.shape != a.shape:
            raise InputError(f"need >= 2 detunings and as many amplitudes, got {w.size} and {a.size}")
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(a))) or np.any(np.diff(w) <= 0):
            raise InputError("detunings must be finite and strictly increasing, amplitudes finite")
        if not np.any(a):
            raise InputError("the pump amplitude is zero everywhere")
        w.setflags(write=False)
        a.setflags(write=False)

        def amplitude(x: FloatArray) -> ComplexArray:
            re = np.interp(x, w, a.real, left=0.0, right=0.0)
            im = np.interp(x, w, a.imag, left=0.0, right=0.0)
            return np.asarray(re + 1j * im, dtype=complex)

        center, half, resolution = _sampled_widths(w, a)
        return cls(amplitude, half, f"sampled({w.size} points)", resolution, center)


def _half_crossing(w0: float, w1: float, a0: complex, a1: complex, level: float) -> float:
    """The detuning in ``[w0, w1]`` where ``|a0 + t (a1 − a0)|² = level`` (one crossing exists)."""
    d = a1 - a0
    A = abs(d) ** 2
    B = 2.0 * (a0.conjugate() * d).real
    C = abs(a0) ** 2 - level
    disc = math.sqrt(max(B * B - 4 * A * C, 0.0))
    # the two roots, written to avoid cancellation; keep the one in [0, 1]
    q = -0.5 * (B + math.copysign(disc, B))
    roots = [q / A, C / q] if q != 0 else [0.0]
    t = min(roots, key=lambda x: abs(x - min(max(x, 0.0), 1.0)))
    return w0 + min(max(t, 0.0), 1.0) * (w1 - w0)


def _sampled_widths(w: FloatArray, a: ComplexArray) -> tuple[float, float, float]:
    """``(center, half_width, resolution_width)`` of a linearly interpolated spectrum (see :class:`Pump`)."""
    intensity = np.abs(a) ** 2
    level = 0.5 * float(intensity.max())
    above = np.concatenate([[False], intensity >= level, [False]])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    runs = list(zip(edges[0::2], edges[1::2] - 1))  # inclusive sample ranges of the peaks >= half max
    n = w.size
    rate = 0.0  # fastest phase turn (rad per rad/s) between consecutive samples over the peaks
    peaks = []
    for i, j in runs:
        left = float(w[i]) if i == 0 else _half_crossing(float(w[i - 1]), float(w[i]), a[i - 1], a[i], level)
        right = float(w[j]) if j == n - 1 else _half_crossing(float(w[j]), float(w[j + 1]), a[j], a[j + 1], level)
        peaks.append((left, right))
        lo, hi = max(i - 1, 0), min(j + 1, n - 1)  # include the intervals of the crossings
        turn = np.abs(np.angle(a[lo + 1 : hi + 1] * np.conj(a[lo:hi]))) / np.diff(w[lo : hi + 1])
        rate = max(rate, float(turn.max()))
    left, right = peaks[0][0], peaks[-1][1]
    half = 0.5 * (right - left)
    resolution = min(0.5 * (hi_ - lo_) for lo_, hi_ in peaks)
    if rate * resolution > 1.0:
        resolution = 1.0 / rate
    return 0.5 * (left + right), half, resolution


# ------------------------------------------------------------------------------- phase matching
@dataclass(frozen=True)
class PhaseMatching:
    """Phase-matching function ``φ(Ω_s, Ω_i)`` of a crystal of given dispersion.

    Build it with :meth:`sinc` or :meth:`gaussian`. ``half_width_sum`` and
    ``half_width_difference`` are the detunings closest to the origin where ``|ΔkL/2| = π`` along
    the diagonal ``Ω_s = Ω_i`` and the anti-diagonal ``Ω_s = −Ω_i`` (``inf`` when unbounded), used to
    size grids: the first zero of the ``sinc`` shape, where the ``gaussian`` shape has fallen to
    ``exp(−γπ²) ≈ 0.149`` of its maximum, for both shapes.
    """

    function: _Joint
    half_width_sum: float
    half_width_difference: float
    gvm: float
    gvd_signal: float
    gvd_pump: float
    shape: str

    @staticmethod
    def _dk_l(gvm: float, gvd_signal: float, gvd_pump: float) -> Callable[[FloatArray, FloatArray], FloatArray]:
        def dk_l(ws: FloatArray, wi: FloatArray) -> FloatArray:
            wp = ws + wi
            return np.asarray(gvm * wp + 0.5 * gvd_pump * wp**2 - 0.5 * gvd_signal * (ws**2 + wi**2), dtype=float)

        return dk_l

    @staticmethod
    def _widths(gvm: float, gvd_signal: float, gvd_pump: float) -> tuple[float, float]:
        # first zero of sinc(ΔkL/2): |ΔkL/2| = π. Diagonal Ω_s = Ω_i = Ω: ΔkL = 2τΩ + bΩ², b = 2β_p − β_s;
        # it reaches 2π first on the side where both terms have the same sign, at the positive root
        # of |b|Ω² + 2|τ|Ω = 2π, Ω = 2π/(|τ| + √(τ² + 2π|b|)) (π/|τ| for b = 0, √(2π/|b|) for τ = 0).
        # Anti-diagonal Ω_s = −Ω_i = Ω: ΔkL = −β_s Ω².
        b = 2 * gvd_pump - gvd_signal
        t = abs(gvm)
        w_sum = 2 * math.pi / (t + math.sqrt(t * t + 2 * math.pi * abs(b))) if (gvm or b) else math.inf
        diff = math.sqrt(2 * math.pi / abs(gvd_signal)) if gvd_signal else math.inf
        return w_sum, diff

    @classmethod
    def _build(cls, gvm: float, gvd_signal: float, gvd_pump: float, shape: str) -> PhaseMatching:
        t, bs, bp = ck.finite("gvm", gvm), ck.finite("gvd_signal", gvd_signal), ck.finite("gvd_pump", gvd_pump)
        dk_l = cls._dk_l(t, bs, bp)
        if shape == "sinc":

            def function(ws: FloatArray, wi: FloatArray) -> ComplexArray:
                x = dk_l(ws, wi) / 2
                return np.asarray(np.sinc(x / np.pi) * np.exp(1j * x), dtype=complex)

        else:

            def function(ws: FloatArray, wi: FloatArray) -> ComplexArray:
                x = dk_l(ws, wi) / 2
                return np.exp(-SINC_GAUSSIAN_GAMMA * x**2 + 1j * x)

        w_sum, w_diff = cls._widths(t, bs, bp)
        return cls(function, w_sum, w_diff, t, bs, bp, shape)

    @classmethod
    def sinc(cls, gvm: float, gvd_signal: float, gvd_pump: float = 0.0) -> PhaseMatching:
        """Uniformly poled (or birefringent) crystal: ``φ = sinc(ΔkL/2) e^{iΔkL/2}``.

        Parameters
        ----------
        gvm
            Group-velocity mismatch ``τ = L (k'_p − k'_s)`` (s).
        gvd_signal
            Group-delay dispersion ``β_s = L k''_s`` at the signal (s²); non-zero for a finite JSA.
        gvd_pump
            Group-delay dispersion ``β_p = L k''_p`` at the pump (s²).
        """
        return cls._build(gvm, gvd_signal, gvd_pump, "sinc")

    @classmethod
    def gaussian(cls, gvm: float, gvd_signal: float, gvd_pump: float = 0.0) -> PhaseMatching:
        """Apodised poling, or the Gaussian approximation of :meth:`sinc`:
        ``φ = exp(−γ (ΔkL/2)²) e^{iΔkL/2}``, ``γ = 0.193`` (same amplitude FWHM as the sinc).

        Parameters
        ----------
        gvm
            Group-velocity mismatch ``τ`` (s).
        gvd_signal
            Group-delay dispersion ``β_s`` at the signal (s²).
        gvd_pump
            Group-delay dispersion ``β_p`` at the pump (s²).
        """
        return cls._build(gvm, gvd_signal, gvd_pump, "gaussian")


# ------------------------------------------------------------------------------- JSA and supermodes
def joint_spectral_amplitude(pump: Pump, phase_matching: PhaseMatching, grid: FrequencyGrid) -> ComplexArray:
    """``f(Ω_j, Ω_k) = α(Ω_j + Ω_k) φ(Ω_j, Ω_k)`` on the grid (not normalised).

    Parameters
    ----------
    pump
        Pump spectrum.
    phase_matching
        Phase-matching function.
    grid
        Detunings of signal and idler.
    """
    w = grid.detunings
    ws, wi = np.meshgrid(w, w, indexing="ij")
    f = pump.amplitude(ws + wi) * phase_matching.function(ws, wi)
    return np.asarray(f, dtype=complex)


@dataclass(frozen=True, eq=False)
class Supermodes:
    """Schmidt decomposition of a degenerate JSA on a frequency grid.

    Compared and hashed by identity (it holds arrays, like :class:`FrequencyGrid`): two
    decompositions of the same JSA are distinct objects.

    Attributes
    ----------
    grid
        The frequency grid.
    jsa
        The JSA as a pixel matrix ``J``, normalised to ``‖J‖_F = 1`` (read-only).
    coefficients
        All Schmidt coefficients ``s_k`` (descending, ``Σ s_k² = 1``).
    unitary
        Full Takagi unitary ``U`` (``J = U diag(s) Uᵀ``): column ``k`` is supermode ``k`` on the grid.
    kept
        Number of supermodes kept (the leading ones).
    """

    grid: FrequencyGrid
    jsa: ComplexArray
    coefficients: FloatArray
    unitary: ComplexArray
    kept: int
    groups: list[npt.NDArray[np.int64]] = field(repr=False, default_factory=list)

    @property
    def schmidt_number(self) -> float:
        """``K = 1/Σ s_k⁴`` (from every coefficient; 1 for a separable JSA)."""
        return float(1.0 / np.sum(self.coefficients**4))

    @property
    def modes(self) -> ComplexArray:
        """Kept supermodes as orthonormal pixel vectors, ``grid.size x kept`` (columns)."""
        return self.unitary[:, : self.kept]

    @property
    def functions(self) -> ComplexArray:
        """Kept supermodes as functions of the detuning, ``U_jk/√δ`` (``∫|u_k|² dΩ = 1``)."""
        return self.modes / math.sqrt(self.grid.step)

    @property
    def dropped_weight(self) -> float:
        """``1 − Σ_{k < kept} s_k²``: the share of the JSA in the supermodes left out."""
        return float(max(0.0, 1.0 - np.sum(self.coefficients[: self.kept] ** 2)))

    def squeezing(self, r0: float) -> FloatArray:
        """Squeezing parameters ``r_k = r₀ s_k/s₀`` of the kept supermodes for a gain giving ``r₀`` to
        supermode 0.

        Parameters
        ----------
        r0
            Squeezing parameter of supermode 0 (nepers), >= 0.
        """
        r = ck.non_negative("r0", r0)
        return np.asarray(r * self.coefficients[: self.kept] / self.coefficients[0], dtype=float)

    def coupling(self, r0: float) -> ComplexArray:
        """Pixel-basis squeezing matrix ``Z = (r₀/s₀) J``: the source prepares
        ``exp(½ Σ (conj(Z_jk) a_j a_k − Z_jk a_j† a_k†))|0⟩`` (every supermode, none dropped).

        Parameters
        ----------
        r0
            Squeezing parameter of supermode 0 (nepers), >= 0.
        """
        r = ck.non_negative("r0", r0)
        return np.asarray(r / self.coefficients[0] * self.jsa, dtype=complex)


def schmidt_decomposition(
    jsa: npt.ArrayLike,
    grid: FrequencyGrid,
    *,
    tol: float = 1e-8,
    max_modes: int = 32,
    edge_tol: float = 1e-3,
    rtol: float = SCHMIDT_RTOL,
) -> Supermodes:
    """Supermodes of a symmetric (degenerate) JSA sampled on ``grid``.

    The matrix is normalised to ``‖J‖_F = 1`` and Takagi-factorised; the leading supermodes are kept
    until their weight ``Σ s_k²`` reaches ``1 − tol``, at most ``max_modes``. An
    :class:`~cvsim.errors.ApproximationWarning` is raised when the JSA is not contained in the grid
    (``|J|²`` on the boundary above ``edge_tol`` times its maximum): widen the grid.

    Parameters
    ----------
    jsa
        ``n x n`` complex symmetric matrix ``f(Ω_j, Ω_k)`` (e.g. from :func:`joint_spectral_amplitude`).
    grid
        The grid it is sampled on (``n`` points).
    tol
        Weight that may be dropped, in ``[0, 1)``; 0 keeps every supermode (up to ``max_modes``).
    max_modes
        Largest number of supermodes kept, >= 1.
    edge_tol
        Largest boundary intensity, relative to the maximum, accepted silently.
    rtol
        Degeneracy tolerance of the Takagi factorisation (default :data:`SCHMIDT_RTOL`).

    Raises
    ------
    InputError
        The JSA is not symmetric (non-degenerate type-II down-conversion is not modelled), is zero,
        or does not match the grid.
    """
    J: ComplexArray = np.array(jsa, dtype=complex)
    n = grid.size
    if J.shape != (n, n):
        raise InputError(f"jsa must be {n} x {n} to match the grid, got shape {J.shape}")
    if not np.all(np.isfinite(J)):
        raise InputError("jsa must be finite")
    norm = float(np.sqrt(np.sum(np.abs(J) ** 2)))
    if norm == 0.0:
        raise InputError("the JSA is zero on the grid (pump and phase matching do not overlap)")
    J = np.asarray(J / norm, dtype=complex)
    asym = float(np.abs(J - J.T).max())
    if asym > 1e-10:
        raise InputError(
            f"the JSA is not symmetric (max |J − Jᵀ| = {asym:.2e} for ‖J‖ = 1): only degenerate type-0/I "
            "down-conversion is modelled (type-II is deferred)"
        )
    tol = ck.in_range("tol", tol, 0.0, 1.0, hi_open=True)
    max_modes = ck.integer("max_modes", max_modes, 1)
    edge_tol = ck.non_negative("edge_tol", edge_tol)
    J = np.asarray((J + J.T) / 2, dtype=complex)
    J.setflags(write=False)
    intensity = np.abs(J) ** 2
    edge = max(intensity[0].max(), intensity[-1].max(), intensity[:, 0].max(), intensity[:, -1].max())
    if edge > edge_tol * intensity.max():
        warnings.warn(
            f"schmidt_decomposition: the JSA reaches the grid boundary (|J|² there = {edge / intensity.max():.2e} "
            "of its maximum); widen the grid",
            ApproximationWarning,
            stacklevel=2,
        )
    tk = la.takagi(J, rtol=rtol)
    s = np.asarray(tk.s, dtype=float)
    weight = np.cumsum(s**2)
    kept = n if tol == 0 else int(np.searchsorted(weight, (1.0 - tol) * weight[-1]) + 1)
    kept = min(kept, int(max_modes), n)
    U = np.asarray(tk.U, dtype=complex)
    for a in (s, U):
        a.setflags(write=False)
    return Supermodes(grid, J, s, U, kept, tk.groups)


def supermodes(
    pump: Pump,
    phase_matching: PhaseMatching,
    grid: FrequencyGrid | None = None,
    *,
    tol: float = 1e-8,
    max_modes: int = 32,
) -> Supermodes:
    """JSA of ``pump`` and ``phase_matching`` and its Schmidt supermodes, in one call.

    Parameters
    ----------
    pump
        Pump spectrum.
    phase_matching
        Phase-matching function.
    grid
        Frequency grid (default :meth:`FrequencyGrid.auto`).
    tol
        Weight that may be dropped.
    max_modes
        Largest number of supermodes kept.
    """
    g = FrequencyGrid.auto(pump, phase_matching) if grid is None else grid
    return schmidt_decomposition(joint_spectral_amplitude(pump, phase_matching, g), g, tol=tol, max_modes=max_modes)
