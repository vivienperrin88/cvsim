"""State specifications, resolved once the mode's number of bits is known.

``system.product_state(memory=ql.states.cat(2.0), buffer=ql.states.fock(0))``: each spec builds
the normalised quantics tensor train of one mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..bosons import states as bstates
from ..core.tt import TensorTrain
from ..errors import InputError

__all__ = ["StateSpec", "cat", "coherent", "fock", "vector"]

#: Discarded weight used to build coherent and cat states.
STATE_TOL = 1e-24


class StateSpec(Protocol):
    def build(self, bits: int) -> TensorTrain: ...


@dataclass(frozen=True)
class _Fock:
    k: int

    def build(self, bits: int) -> TensorTrain:
        if not 0 <= self.k < 2**bits:
            raise InputError(f"fock({self.k}) does not fit in 2**{bits} levels")
        return bstates.fock(self.k, bits)


@dataclass(frozen=True)
class _Coherent:
    alpha: complex

    def build(self, bits: int) -> TensorTrain:
        return bstates.coherent(self.alpha, bits, tol=STATE_TOL)


@dataclass(frozen=True)
class _Cat:
    alpha: complex
    parity: int

    def build(self, bits: int) -> TensorTrain:
        return bstates.cat(self.alpha, bits, parity=self.parity, tol=STATE_TOL)


@dataclass(frozen=True)
class _Vector:
    amplitudes: np.ndarray = field(compare=False)

    def build(self, bits: int) -> TensorTrain:
        v = np.asarray(self.amplitudes, dtype=np.complex128).reshape(-1)
        N = 2**bits
        if v.size > N:
            raise InputError(f"a vector of {v.size} amplitudes does not fit in 2**{bits} levels")
        full = np.zeros(N, dtype=np.complex128)
        full[: v.size] = v
        nrm = np.linalg.norm(full)
        if nrm == 0:
            raise InputError("the state vector is zero")
        return TensorTrain.from_dense(full / nrm, [2] * bits, tol=STATE_TOL)


def fock(k: int) -> StateSpec:
    """Fock state ``|k>`` (level ``k`` of a custom basis)."""
    return _Fock(int(k))


def coherent(alpha: complex) -> StateSpec:
    """Coherent state ``|alpha>``, truncated to the mode's levels and renormalised."""
    return _Coherent(complex(alpha))


def cat(alpha: complex, parity: int = +1) -> StateSpec:
    """Cat state ``|alpha> + parity |-alpha>``, normalised."""
    if parity not in (1, -1):
        raise InputError(f"parity must be +1 or -1, got {parity}")
    return _Cat(complex(alpha), int(parity))


def vector(amplitudes: np.ndarray) -> StateSpec:
    """Arbitrary amplitudes on the first levels (zero-padded, normalised)."""
    return _Vector(np.asarray(amplitudes))
