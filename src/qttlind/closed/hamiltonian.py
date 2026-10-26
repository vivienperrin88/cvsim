"""Time-dependent MPO Hamiltonians ``H(t) = sum_k f_k(t) H_k``.

Each term is a coefficient (a constant or a callable of ``t``) times an MPO. The constant terms
are summed and compressed once; :meth:`TimeDependentMPO.at` adds the time-dependent ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..core.mpo import MPO, apply_mpo
from ..core.tt import TensorTrain
from ..errors import InputError, ShapeError

__all__ = ["Coefficient", "TimeDependentMPO"]

Coefficient = complex | float | Callable[[float], complex]

#: Discarded weight when summing MPO terms (relative error 1e-12).
SUM_TOL = 1e-24


def _value(c: Coefficient, t: float) -> complex:
    return complex(c(t)) if callable(c) else complex(c)


class TimeDependentMPO:
    """``H(t) = sum_k f_k(t) H_k`` with MPO terms of identical dimensions.

    Parameters
    ----------
    terms
        Sequence of ``(coefficient, MPO)``; a coefficient is a number or a callable ``f(t)``.
    """

    def __init__(self, terms: Sequence[tuple[Coefficient, MPO]], *, tol: float = SUM_TOL):
        terms = list(terms)
        if not terms:
            raise InputError("a Hamiltonian needs at least one term")
        dims = {(op.dims_out, op.dims_in) for _, op in terms}
        if len(dims) != 1:
            raise InputError(f"all terms must act on the same dimensions, got {sorted(dims)}")
        self.terms = terms
        self.tol = tol
        const = [(complex(c), op) for c, op in terms if not callable(c)]
        self.dynamic = [(c, op) for c, op in terms if callable(c)]
        self._const: MPO | None = None
        if const:
            acc = const[0][1] * const[0][0]
            for c, op in const[1:]:
                acc, _ = (acc + op * c).compress(tol)
            self._const, _ = acc.compress(tol)
        self.dims = terms[0][1].dims_in

    @classmethod
    def constant(cls, H: MPO) -> TimeDependentMPO:
        return cls([(1.0, H)])

    @property
    def is_time_dependent(self) -> bool:
        return bool(self.dynamic)

    @property
    def length(self) -> int:
        return len(self.dims)

    def coefficients(self, t: float) -> list[complex]:
        return [_value(c, t) for c, _ in self.terms]

    def at(self, t: float, tol: float | None = None) -> MPO:
        """The MPO ``H(t)`` (sum of the terms, compressed)."""
        tol = self.tol if tol is None else tol
        acc = self._const
        for c, op in self.dynamic:
            term = op * _value(c, t)
            acc = term if acc is None else (acc + term).compress(tol)[0]
        assert acc is not None
        return acc

    def apply(
        self, t: float, psi: TensorTrain, *, tol: float = 0.0, chi_max: int | None = None, method: str = "zipup"
    ) -> TensorTrain:
        """``H(t) |psi>``, compressed."""
        if psi.dims != self.dims:
            raise ShapeError(f"state dims {psi.dims} do not match the Hamiltonian's {self.dims}")
        out, _ = apply_mpo(self.at(t), psi, method=method, tol=tol, chi_max=chi_max)
        return out
