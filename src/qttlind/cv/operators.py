"""Quadrature and ladder operators in position space; the Hermite transform.

In position space ``x`` is diagonal, and ``x_j = (j - N/2) dx`` is affine in the bits: an exact
**rank-2** diagonal MPO. The momentum is diagonal in momentum space, so::

    p = F^dag diag(p_k) F       (spectral accuracy, MPO rank ~ 13 after compression),

with the bit-reversed Fourier MPO of :mod:`.fourier`: ``F = P F^`` where ``P`` reverses the sites, hence
``p = F^^dag (P diag(p) P) F^`` and ``P diag(p) P`` is again an affine diagonal (with the bit
weights reversed). Everything else is algebra: ``a = (x + i p)/sqrt(2)``,
``n = (x^2 + p^2 - 1)/2``, all of MPO rank ~ 15 for any number of bits. The Lindblad solver
of :mod:`qttlind.lindblad` can therefore run directly in position space.

**Fock <-> position.** ``H[j, n] = sqrt(dx) phi_n(x_j)`` maps Fock amplitudes to position
amplitudes. Its MPO rank is close to the number of Fock states kept (the Hermite functions are
genuinely different for every ``n``): measured rank ~ ``2^{fock_bits}``. A change of basis is
therefore O(N), not O(log N): do it rarely -- to prepare an input given in Fock space, or to read
out photon statistics -- and do the dynamics in one basis.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import CutoffWarning, InputError
from .fourier import dft_mpo, reverse_sites
from .grids import QuadratureGrid, hermite_functions

__all__ = [
    "MAX_HERMITE_BITS",
    "OPERATOR_TOL",
    "affine_tt",
    "annihilation",
    "creation",
    "fock_to_position",
    "hermite_mpo",
    "momentum",
    "number",
    "pad_fock",
    "position",
    "position_to_fock",
    "quadrature",
]

#: Discarded weight when compressing operator MPOs (relative error 1e-12).
OPERATOR_TOL = 1e-24

#: The Hermite transform is built densely (4^bits entries): refuse beyond this.
MAX_HERMITE_BITS = 11


def affine_tt(bits: int, slope: float, offset: float) -> TensorTrain:
    """Exact rank-2 tensor train of ``f(j) = offset + slope * j``, ``j`` on ``bits`` bits (MSB first).

    Same transfer-matrix idea as ``bosons.operators.number_tt``: channel 0 = "constant so far",
    channel 1 = "sum complete"; the offset enters on the first core.
    """
    if bits < 1:
        raise InputError(f"bits must be >= 1, got {bits}")
    cores = []
    for i in range(bits):
        w = slope * 2.0 ** (bits - 1 - i)
        c = np.zeros((2, 2, 2), dtype=complex)
        for s in (0, 1):
            c[:, s, :] = [[1.0, w * s], [0.0, 1.0]]
        cores.append(c)
    first = cores[0][:1].copy()
    first[0, :, 1] += offset  # f = offset + ...
    cores[0] = first
    cores[-1] = cores[-1][..., 1:]
    return TensorTrain(cores, copy=False)


def _diag(f: TensorTrain) -> MPO:
    cores = []
    for a in f.cores:
        l_, d, r_ = a.shape
        w = np.zeros((l_, d, d, r_), dtype=complex)
        for s in range(d):
            w[:, s, s, :] = a[:, s, :]
        cores.append(w)
    return MPO(cores, copy=False)


def position(grid: QuadratureGrid) -> MPO:
    """``x = diag(x_j)``, exact rank 2."""
    N = grid.size
    return _diag(affine_tt(grid.bits, grid.dx, -N / 2 * grid.dx))


def momentum(grid: QuadratureGrid, tol: float = OPERATOR_TOL) -> MPO:
    """``p = F^^dag diag_rev(p_k) F^`` in position space (spectral), compressed."""
    N = grid.size
    F = dft_mpo(grid.bits)
    Dp = _diag(reverse_sites(affine_tt(grid.bits, grid.dp, -N / 2 * grid.dp)))
    P, _ = (F.dagger() @ (Dp @ F)).compress(tol)
    return P


def quadrature(grid: QuadratureGrid, phi: float, tol: float = OPERATOR_TOL) -> MPO:
    """``x_phi = cos(phi) x + sin(phi) p``."""
    op, _ = (position(grid) * float(np.cos(phi)) + momentum(grid, tol) * float(np.sin(phi))).compress(tol)
    return op


def annihilation(grid: QuadratureGrid, tol: float = OPERATOR_TOL) -> MPO:
    """``a = (x + i p)/sqrt(2)`` in position space."""
    op, _ = ((position(grid) + 1j * momentum(grid, tol)) * (1 / np.sqrt(2))).compress(tol)
    return op


def creation(grid: QuadratureGrid, tol: float = OPERATOR_TOL) -> MPO:
    return annihilation(grid, tol).dagger()


def number(grid: QuadratureGrid, tol: float = OPERATOR_TOL) -> MPO:
    """``n = (x^2 + p^2 - 1)/2`` in position space."""
    X = position(grid)
    P = momentum(grid, tol)
    X2, _ = (X @ X).compress(tol)
    P2, _ = (P @ P).compress(tol)
    op, _ = ((X2 + P2 - MPO.identity([2] * grid.bits)) * 0.5).compress(tol)
    return op


# ----------------------------------------------------------------------------- Fock <-> position


def hermite_mpo(grid: QuadratureGrid, fock_bits: int, tol: float = OPERATOR_TOL) -> MPO:
    """Square MPO on ``grid.bits`` sites: Fock amplitudes (padded to ``grid.bits`` bits) -> position amplitudes.

    Columns ``n >= 2^{fock_bits}`` are zero. Built densely: ``grid.bits <= MAX_HERMITE_BITS``.
    """
    if grid.bits > MAX_HERMITE_BITS:
        raise InputError(f"hermite_mpo is built densely: grid.bits <= {MAX_HERMITE_BITS}, got {grid.bits}")
    if not 1 <= fock_bits <= grid.bits:
        raise InputError(f"fock_bits must be in 1..{grid.bits}, got {fock_bits}")
    nf = 2**fock_bits
    if nf - 1 > grid.max_photon_number():
        warnings.warn(
            f"the grid resolves n <= {grid.max_photon_number()} but {nf} Fock states were requested",
            CutoffWarning,
            stacklevel=2,
        )
    H = np.zeros((grid.size, grid.size), dtype=complex)
    H[:, :nf] = np.sqrt(grid.dx) * hermite_functions(nf, grid.x()).T
    return MPO.from_dense(H, [2] * grid.bits, tol=tol)


def pad_fock(psi: TensorTrain, bits: int) -> TensorTrain:
    """Write a Fock-space tensor train on more bits: prepend ``|0>`` cores (the new MSBs are 0)."""
    extra = bits - psi.length
    if extra < 0:
        raise InputError(f"cannot pad {psi.length} bits down to {bits}")
    e0 = np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1)
    return TensorTrain([e0] * extra + list(psi.cores))


def fock_to_position(
    psi_fock: TensorTrain, grid: QuadratureGrid, tol: float = OPERATOR_TOL, chi_max: int | None = None
) -> TensorTrain:
    """Position amplitudes of a Fock-space state (``psi_fock.length <= grid.bits``)."""
    H = hermite_mpo(grid, psi_fock.length, tol)
    psi_fock, _ = psi_fock.compress(tol)  # the MPO rank multiplies the state rank: compress first
    out, _ = H.apply(pad_fock(psi_fock, grid.bits)).compress(tol, chi_max)
    return out


def position_to_fock(
    psi_x: TensorTrain, grid: QuadratureGrid, fock_bits: int, tol: float = OPERATOR_TOL, warn_above: float = 1e-8
) -> TensorTrain:
    """Fock amplitudes ``<n|psi>``, ``n < 2^fock_bits``; warns if more than ``warn_above`` of the norm is lost."""
    H = hermite_mpo(grid, fock_bits, tol)
    psi_x, _ = psi_x.compress(tol)  # the MPO rank multiplies the state rank: compress first
    full, _ = H.dagger().apply(psi_x).compress(tol)
    extra = grid.bits - fock_bits
    left = np.ones((1,), dtype=complex)
    for c in full.cores[:extra]:
        left = left @ c[:, 0, :]  # project the padding bit on 0
    cores = [c.copy() for c in full.cores[extra:]]
    cores[0] = np.einsum("l,lsr->sr", left, cores[0])[None]
    out = TensorTrain(cores, copy=False)
    lost = max(0.0, psi_x.norm() ** 2 - out.norm() ** 2)
    if lost > warn_above * psi_x.norm() ** 2:
        warnings.warn(f"{lost:.2e} of the norm lies above n = {2**fock_bits - 1}", CutoffWarning, stacklevel=2)
    return out
