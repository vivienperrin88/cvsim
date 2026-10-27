"""Quadrature grids and position-space states.

A single mode in the **position representation** is the wavefunction ``psi(x)`` sampled on a
grid of ``N = 2**bits`` points, stored as a quantics tensor train of the amplitudes
``c_j = sqrt(dx) psi(x_j)`` (so that ``sum_j |c_j|^2 = int |psi|^2 dx = 1``)::

    x_j = (j - N/2) dx,     j = 0..N-1,     dx = L / N.

The discrete Fourier transform (:mod:`.fourier`) maps it to the momentum grid ``p_k = (k - N/2) dp`` with
``dp = 2 pi / L``. The default box ``L = sqrt(2 pi N)`` is **self-dual** (``dx = dp``): the x and p
grids coincide, and the grid resolves Hermite functions up to ``n ~ pi N / 4`` minus a safety
margin (see :meth:`QuadratureGrid.max_photon_number`).

States here are built by sampling a function and compressing it (``bosons.states.from_function``), so
``bits`` is limited by ``MAX_DENSE_BITS``; :func:`qttlind.core.tci.quantics_tci` handles larger grids.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..bosons.states import DEFAULT_TOL, MAX_DENSE_BITS, from_function
from ..core.tt import TensorTrain
from ..errors import CutoffWarning, InputError
from ..gaussian.states import GaussianState

__all__ = [
    "QuadratureGrid",
    "cat",
    "check_resolution",
    "coherent",
    "fock",
    "from_gaussian",
    "from_wavefunction",
    "gkp",
    "hermite_functions",
    "squeezed_vacuum",
    "wavefunction",
]


@dataclass(frozen=True)
class QuadratureGrid:
    """A grid of ``2**bits`` points on ``[-L/2, L/2)``; ``length=None`` selects the self-dual box."""

    bits: int
    length: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bits, int | np.integer) or not 1 <= self.bits <= 62:
            raise InputError(f"bits must be an integer in 1..62, got {self.bits!r}")
        if self.length is None:
            object.__setattr__(self, "length", float(np.sqrt(2 * np.pi * 2**self.bits)))
        elif not self.length > 0:
            raise InputError(f"box length must be > 0, got {self.length}")
        else:
            object.__setattr__(self, "length", float(self.length))

    @property
    def size(self) -> int:
        return 1 << self.bits

    @property
    def L(self) -> float:
        assert self.length is not None  # set in __post_init__
        return self.length

    @property
    def dx(self) -> float:
        return self.L / self.size

    @property
    def dp(self) -> float:
        return 2 * np.pi / self.L

    @property
    def self_dual(self) -> bool:
        return bool(np.isclose(self.dx, self.dp))

    def x(self) -> np.ndarray:
        """Dense array of the grid points (guarded: ``bits <= MAX_DENSE_BITS``)."""
        return (self._index() - self.size / 2) * self.dx

    def p(self) -> np.ndarray:
        """Dense array of the momentum grid points ``(k - N/2) dp``."""
        return (self._index() - self.size / 2) * self.dp

    def _index(self) -> np.ndarray:
        if self.bits > MAX_DENSE_BITS:
            raise InputError(f"{self.bits} bits: dense grid arrays are limited to {MAX_DENSE_BITS} bits")
        return np.arange(self.size, dtype=float)

    def max_photon_number(self, margin: float = 6.0) -> int:
        """Largest ``n`` whose Hermite function the grid represents faithfully.

        Two conditions on ``phi_n``: its classical turning point ``sqrt(2n + 1)`` lies ``margin``
        inside the box (``<= L/2 - margin``), and its largest wavenumber ``sqrt(2n + 1)`` is below
        the Nyquist limit of both grids (``pi/dx`` in x, ``pi/dp`` in p) minus the same margin.
        """
        kmax = min(self.L / 2, np.pi / self.dx, np.pi / self.dp) - margin
        if kmax <= 1:
            return 0
        return int(np.floor((kmax**2 - 1) / 2))


def check_resolution(grid: QuadratureGrid, nmax: float, what: str = "state") -> None:
    """Warn (:class:`~qttlind.errors.CutoffWarning`) when a state with photon numbers up to ``nmax``
    is not resolved by ``grid``."""
    limit = grid.max_photon_number()
    if nmax > limit:
        warnings.warn(
            f"{what} reaches photon numbers ~{nmax:.0f} but a {grid.bits}-bit grid of length {grid.L:.3g} "
            f"resolves n <= {limit}: increase bits or adjust the box length",
            CutoffWarning,
            stacklevel=3,
        )


def hermite_functions(nmax: int, x: np.ndarray) -> np.ndarray:
    """``phi_n(x) = <x|n>`` for ``n < nmax``, shape ``(nmax, len(x))``.

    The three-term recurrence on *normalised* functions,
    ``phi_{n+1} = sqrt(2/(n+1)) x phi_n - sqrt(n/(n+1)) phi_{n-1}``, never forms ``H_n(x)`` or ``n!``
    and is stable for thousands of terms.
    """
    if nmax < 1:
        raise InputError(f"nmax must be >= 1, got {nmax}")
    x = np.asarray(x, dtype=float)
    phi = np.zeros((nmax, x.size))
    phi[0] = np.pi**-0.25 * np.exp(-(x**2) / 2)
    if nmax > 1:
        phi[1] = np.sqrt(2.0) * x * phi[0]
    for n in range(1, nmax - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * x * phi[n] - np.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def from_wavefunction(
    psi: Callable[[np.ndarray], np.ndarray],
    grid: QuadratureGrid,
    tol: float = DEFAULT_TOL,
    chi_max: int | None = None,
    normalize: bool = True,
) -> TensorTrain:
    """Quantics tensor train of ``c_j = sqrt(dx) psi(x_j)``, normalised by default."""
    x0, dx = -grid.L / 2, grid.dx
    tt = from_function(lambda j: np.sqrt(dx) * np.asarray(psi(x0 + j * dx), dtype=complex), grid.bits, tol, chi_max)
    if normalize:
        nrm = tt.norm()
        if nrm == 0:
            raise InputError("wavefunction vanishes on the grid")
        tt = tt * (1.0 / nrm)
    return tt


def wavefunction(tt: TensorTrain, grid: QuadratureGrid) -> np.ndarray:
    """Dense ``psi(x_j) = c_j / sqrt(dx)`` (guarded through the grid)."""
    if tt.length != grid.bits:
        raise InputError(f"tensor train has {tt.length} sites but the grid has {grid.bits} bits")
    grid.x()  # raises for grids too large to densify
    return tt.to_dense() / np.sqrt(grid.dx)


def fock(n: int, grid: QuadratureGrid, tol: float = DEFAULT_TOL) -> TensorTrain:
    """The Fock state ``|n>`` in position space: the Hermite function ``phi_n``."""
    if n < 0:
        raise InputError(f"photon number must be >= 0, got {n}")
    check_resolution(grid, n, f"Fock state |{n}>")
    return from_wavefunction(lambda x: hermite_functions(n + 1, x)[n], grid, tol)


def coherent(alpha: complex, grid: QuadratureGrid, tol: float = DEFAULT_TOL) -> TensorTrain:
    """``D(alpha)|0>``: ``psi(x) = pi^{-1/4} exp(-(x - x0)^2/2 + i p0 x - i x0 p0/2)``,
    ``(x0, p0) = sqrt(2) (Re alpha, Im alpha)``. Same phase as the Fock-space coherent state."""
    x0, p0 = np.sqrt(2) * np.real(alpha), np.sqrt(2) * np.imag(alpha)
    check_resolution(grid, abs(alpha) ** 2 + 6 * abs(alpha) + 6, "coherent state")
    return from_wavefunction(
        lambda x: np.pi**-0.25 * np.exp(-((x - x0) ** 2) / 2 + 1j * p0 * x - 0.5j * x0 * p0), grid, tol
    )


def from_gaussian(state: GaussianState, grid: QuadratureGrid, tol: float = DEFAULT_TOL) -> TensorTrain:
    """Wavefunction of a **pure single-mode** Gaussian state ``(d, V)``.

    ``psi(x) ~ exp(-a (x - x0)^2 / 2 + i p0 x - i x0 p0 / 2)`` with ``a = (1 - 2 i V_xp) / (2 V_xx)``
    (the local momentum ``-Im(a) x`` reproduces the x-p correlation). The global phase is a
    convention; it coincides with :func:`coherent` when ``V = I/2``.
    """
    if state.num_modes != 1 or not state.is_pure():
        raise InputError("from_gaussian needs a pure single-mode Gaussian state")
    (x0, p0), V = state.mean, state.cov
    a = (1 - 2j * V[0, 1]) / (2 * V[0, 0])
    check_resolution(grid, float(state.mean_photons()[0]) * 3 + 10, "Gaussian state")
    return from_wavefunction(lambda x: np.exp(-a * (x - x0) ** 2 / 2 + 1j * p0 * x - 0.5j * x0 * p0), grid, tol)


def squeezed_vacuum(r: float, grid: QuadratureGrid, phi: float = 0.0, tol: float = DEFAULT_TOL) -> TensorTrain:
    return from_gaussian(GaussianState.squeezed_vacuum(r, phi), grid, tol)


def cat(alpha: complex, grid: QuadratureGrid, parity: int = 1, tol: float = DEFAULT_TOL) -> TensorTrain:
    """``(|alpha> + parity |-alpha>)`` normalised, in position space."""
    if parity not in (1, -1):
        raise InputError(f"parity must be +1 or -1, got {parity}")
    if parity == -1 and alpha == 0:
        raise InputError("the odd cat state with alpha = 0 is the zero vector")
    x0, p0 = np.sqrt(2) * np.real(alpha), np.sqrt(2) * np.imag(alpha)

    def psi(x: np.ndarray) -> np.ndarray:
        plus = np.exp(-((x - x0) ** 2) / 2 + 1j * p0 * x - 0.5j * x0 * p0)
        minus = np.exp(-((x + x0) ** 2) / 2 - 1j * p0 * x - 0.5j * x0 * p0)
        return plus + parity * minus

    check_resolution(grid, abs(alpha) ** 2 + 6 * abs(alpha) + 6, "cat state")
    return from_wavefunction(psi, grid, tol)


def gkp(
    delta: float, grid: QuadratureGrid, logical: int = 0, tol: float = DEFAULT_TOL, spacing: float = np.sqrt(np.pi)
) -> TensorTrain:
    """Finite-energy square GKP state ``|mu_delta>``, ``mu = logical`` in {0, 1}.

    Gaussian peaks of width ``delta`` at ``x_s = (2 s + mu) sqrt(pi)`` under a Gaussian envelope
    ``exp(-delta^2 x_s^2 / 2)``::

        psi(x) ~ sum_s exp(-delta^2 x_s^2 / 2) exp(-(x - x_s)^2 / (2 delta^2)).

    This is the standard "Gaussian envelope" approximation of ``exp(-delta^2 n)|mu_GKP>``; its
    squeezing in dB is ``-10 log10(2 delta^2)``. Mean photon number ``~ 1/(2 delta^2)``.
    """
    if logical not in (0, 1):
        raise InputError(f"logical must be 0 or 1, got {logical}")
    if not delta > 0:
        raise InputError(f"delta must be > 0, got {delta}")
    if grid.dx > delta / 4:
        warnings.warn(
            f"grid spacing dx = {grid.dx:.3g} does not resolve GKP peaks of width {delta:.3g} (want dx <= delta/4)",
            CutoffWarning,
            stacklevel=2,
        )
    smax = int(np.ceil((grid.L / 2 + 8 * delta) / (2 * spacing))) + 1
    centres = (2 * np.arange(-smax, smax + 1) + logical) * spacing

    def psi(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=complex)
        for xs in centres:
            out += np.exp(-(delta**2) * xs**2 / 2) * np.exp(-((x - xs) ** 2) / (2 * delta**2))
        return out

    check_resolution(grid, 1 / (2 * delta**2) * 3, "GKP state")
    return from_wavefunction(psi, grid, tol)
