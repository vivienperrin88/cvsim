"""Global integrators on tensor trains: RK4 and Crank--Nicolson (paper Eqs. 17-24).

Every vector operation becomes a tensor-train operation followed by recompression. Both
integrators follow the step protocol ``step(H, psi, t, h, *, tol, chi_max) -> TensorTrain``
used by :func:`qttlind.closed.evolve.evolve`.

* RK4 is explicit: stable only for ``h ||H|| <~ 2.8`` where ``||H||`` is the **largest** energy of
  the truncated space (``~ omega N``, or ``K N^2`` with a Kerr term) -- stiffness.
* Crank--Nicolson ``(1 + i h/2 H) psi' = (1 - i h/2 H) psi`` is unconditionally stable and second
  order; the linear system is solved by two-site ALS (:mod:`.linear_solve`).
"""

from __future__ import annotations

from ..core.mpo import MPO, apply_mpo
from ..core.tt import TensorTrain
from .hamiltonian import TimeDependentMPO
from .linear_solve import als_solve

__all__ = ["crank_nicolson_step", "rk4_step"]


def _axpy(*terms: tuple[complex, TensorTrain], tol: float, chi_max: int | None) -> TensorTrain:
    acc = terms[0][1] * terms[0][0]
    for c, v in terms[1:]:
        acc, _ = (acc + v * c).compress(tol, chi_max)
    return acc


def rk4_step(
    H: TimeDependentMPO, psi: TensorTrain, t: float, h: float, *, tol: float = 0.0, chi_max: int | None = None
) -> TensorTrain:
    """One classical fourth-order Runge--Kutta step of ``i dpsi/dt = H(t) psi``."""

    def f(tt: float, v: TensorTrain) -> TensorTrain:
        return H.apply(tt, v, tol=tol, chi_max=chi_max) * (-1j)

    k1 = f(t, psi)
    k2 = f(t + h / 2, _axpy((1, psi), (h / 2, k1), tol=tol, chi_max=chi_max))
    k3 = f(t + h / 2, _axpy((1, psi), (h / 2, k2), tol=tol, chi_max=chi_max))
    k4 = f(t + h, _axpy((1, psi), (h, k3), tol=tol, chi_max=chi_max))
    return _axpy((1, psi), (h / 6, k1), (h / 3, k2), (h / 3, k3), (h / 6, k4), tol=tol, chi_max=chi_max)


def crank_nicolson_step(
    H: TimeDependentMPO, psi: TensorTrain, t: float, h: float, *, tol: float = 0.0, chi_max: int | None = None
) -> TensorTrain:
    """One Crank--Nicolson step with ``H`` frozen at the midpoint ``t + h/2``."""
    Hm = H.at(t + h / 2)
    ident = MPO.identity(list(psi.dims))
    rhs, _ = apply_mpo((ident + Hm * (-0.5j * h)).compress(1e-28)[0], psi, tol=tol, chi_max=chi_max)
    A, _ = (ident + Hm * (0.5j * h)).compress(1e-28)
    return als_solve(A, rhs, rhs, tol=tol, chi_max=chi_max)
