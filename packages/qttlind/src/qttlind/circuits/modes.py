"""CV modes in one of two bases, Fock or position.

:class:`CVMode` is an adapter: it turns the two layer-2 packages (:mod:`qttlind.bosons` for Fock
space, :mod:`qttlind.cv` for quadrature grids) into the one interface the comb machinery expects.
Everything above it -- Hamiltonians, jump operators, the comb, the Kraus integrator, the
executors -- is basis-agnostic: an MPO is an MPO (decision D16).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..bosons import operators as bops
from ..bosons import states as bstates
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..cv import grids as cvgrids
from ..cv import operators as cvops
from ..errors import InputError
from ..gaussian.fock import fock_ket
from ..gaussian.states import GaussianState

__all__ = ["CVMode"]

#: Discarded weight for the operators built here.
MODE_TOL = 1e-24
#: Largest number of levels for which dense single-mode matrices are formed.
MAX_DENSE_LEVELS = 4096


@dataclass(frozen=True)
class CVMode:
    """A bosonic mode with ``2**bits`` basis states: Fock levels or the points of a quadrature grid."""

    name: str
    basis: Literal["fock", "x"]
    bits: int
    grid: cvgrids.QuadratureGrid | None = None

    def __post_init__(self) -> None:
        if self.basis not in ("fock", "x"):
            raise InputError(f"basis must be 'fock' or 'x', got {self.basis!r}")
        if not isinstance(self.bits, (int, np.integer)) or not 1 <= self.bits <= 30:
            raise InputError(f"bits must be an integer in 1..30, got {self.bits!r}")
        if self.basis == "x":
            if self.grid is None:
                raise InputError(f"mode {self.name!r}: a position-basis mode needs a QuadratureGrid")
            if self.grid.bits != self.bits:
                raise InputError(f"mode {self.name!r}: grid has {self.grid.bits} bits, the mode {self.bits}")
        elif self.grid is not None:
            raise InputError(f"mode {self.name!r}: a Fock-basis mode takes no grid")

    @classmethod
    def fock(cls, name: str, bits: int) -> CVMode:
        return cls(name, "fock", bits)

    @classmethod
    def position(cls, name: str, bits: int, length: float | None = None) -> CVMode:
        """Position basis on ``QuadratureGrid(bits, length)`` (self-dual box by default)."""
        return cls(name, "x", bits, cvgrids.QuadratureGrid(bits, length))

    @property
    def dim(self) -> int:
        return 2**self.bits

    # ------------------------------------------------------------------ operators (cached)
    def annihilation(self) -> MPO:
        return _annihilation(self)

    def creation(self) -> MPO:
        return _annihilation(self).dagger()

    def number(self) -> MPO:
        return _number(self)

    def quadratures(self) -> tuple[MPO, MPO]:
        """``(x, p)`` with ``x = (a + a^dag)/sqrt 2`` (hbar = 1)."""
        return _quadratures(self)

    def parity(self) -> MPO:
        """``(-1)^n`` (exact in Fock space; ``psi(x) -> psi(-x)`` on a symmetric grid otherwise)."""
        return _parity(self)

    def dense(self, op: MPO) -> np.ndarray:
        if self.dim > MAX_DENSE_LEVELS:
            raise InputError(f"{self.dim} levels: dense single-mode matrices are limited to {MAX_DENSE_LEVELS}")
        return op.to_dense()

    # ------------------------------------------------------------------ states
    def vacuum(self) -> TensorTrain:
        return self.fock_state(0)

    def fock_state(self, n: int) -> TensorTrain:
        """``|n>`` in this mode's basis."""
        if self.basis == "fock":
            if not 0 <= n < self.dim:
                raise InputError(f"|{n}> does not fit in {self.dim} levels")
            return bstates.fock(n, self.bits)
        return cvgrids.fock(n, self.grid)  # type: ignore[arg-type]

    def state_from_gaussian(self, state: GaussianState, tol: float = 1e-20) -> TensorTrain:
        """Quantics amplitudes of a pure single-mode Gaussian state (Bargmann or wavefunction)."""
        if state.num_modes != 1 or not state.is_pure():
            raise InputError("state_from_gaussian needs a pure single-mode Gaussian state")
        if self.basis == "x":
            return cvgrids.from_gaussian(state, self.grid, tol)  # type: ignore[arg-type]
        if self.dim > 2**16:
            raise InputError("Fock-basis Gaussian states are built densely: at most 16 bits")
        v = fock_ket(state, self.dim)
        return TensorTrain.from_dense(np.asarray(v, dtype=np.complex128), [2] * self.bits, tol=tol)

    def to_fock(self, psi: TensorTrain, fock_bits: int) -> TensorTrain:
        """Fock amplitudes ``<n|psi>``, ``n < 2**fock_bits`` (truncation or zero-padding for Fock modes)."""
        if self.basis == "x":
            return cvops.position_to_fock(psi, self.grid, fock_bits)  # type: ignore[arg-type]
        if fock_bits == self.bits:
            return psi.copy()
        v = psi.to_dense() if self.bits <= 20 else None
        if v is None:
            raise InputError("changing the Fock cutoff is dense: at most 20 bits")
        out = np.zeros(2**fock_bits, dtype=np.complex128)
        k = min(v.size, out.size)
        out[:k] = v[:k]
        return TensorTrain.from_dense(out, [2] * fock_bits, tol=1e-28)

    def fock_basis_matrix(self, nmax: int) -> np.ndarray:
        """Rows ``<n|j>`` (``n < nmax``) mapping this basis to Fock amplitudes (dense)."""
        if self.basis == "fock":
            return np.eye(nmax, self.dim, dtype=np.complex128)
        g = self.grid
        assert g is not None
        return cvgrids.hermite_functions(nmax, g.x()).astype(np.complex128) * np.sqrt(g.dx)

    def fock_cutoff(self) -> int:
        """Number of Fock levels the mode represents faithfully."""
        if self.basis == "fock":
            return self.dim
        return min(self.dim, self.grid.max_photon_number() + 1)  # type: ignore[union-attr]

    def fock_to_basis_mpo(self, fock_bits: int) -> MPO | None:
        """MPO mapping zero-padded Fock amplitudes (``fock_bits`` significant bits) to this basis."""
        if self.basis == "fock":
            return None
        return _hermite(self, fock_bits)


# ---------------------------------------------------------------------------------- cached builders
@functools.lru_cache(maxsize=128)
def _annihilation(mode: CVMode) -> MPO:
    if mode.basis == "fock":
        return bops.annihilation(mode.bits)
    return cvops.annihilation(mode.grid)  # type: ignore[arg-type]


@functools.lru_cache(maxsize=128)
def _number(mode: CVMode) -> MPO:
    if mode.basis == "fock":
        return bops.number(mode.bits)
    return cvops.number(mode.grid)  # type: ignore[arg-type]


@functools.lru_cache(maxsize=128)
def _quadratures(mode: CVMode) -> tuple[MPO, MPO]:
    if mode.basis == "x":
        return cvops.position(mode.grid), cvops.momentum(mode.grid)  # type: ignore[arg-type]
    a = _annihilation(mode)
    ad = a.dagger()
    x = ((a + ad) * (1 / np.sqrt(2))).compress(MODE_TOL)[0]
    p = ((a - ad) * (-1j / np.sqrt(2))).compress(MODE_TOL)[0]
    return x, p


@functools.lru_cache(maxsize=128)
def _parity(mode: CVMode) -> MPO:
    if mode.basis == "fock":
        return bops.parity(mode.bits)
    # psi(x) -> psi(-x): on the grid x_j = (j - N/2) dx, j -> N - j (mod N); exact for states that
    # vanish at the box edge (the point j = 0 maps to itself)
    N = mode.dim
    if N > MAX_DENSE_LEVELS:
        raise InputError("position-basis parity is built densely: at most 12 bits")
    P = np.zeros((N, N), dtype=np.complex128)
    j = np.arange(N)
    P[(N - j) % N, j] = 1.0
    return MPO.from_dense(P, [2] * mode.bits, tol=MODE_TOL)


@functools.lru_cache(maxsize=64)
def _hermite(mode: CVMode, fock_bits: int) -> MPO:
    return cvops.hermite_mpo(mode.grid, fock_bits)  # type: ignore[arg-type]
