"""Closed-system integrators on quantics tensor trains.

* :mod:`.hamiltonian` -- ``H(t) = sum_k f_k(t) H_k`` with MPO terms (:class:`TimeDependentMPO`)
* :mod:`.integrators` -- RK4 and Crank--Nicolson in the tensor-train manifold
* :mod:`.effective` -- one-site, two-site and bond effective Hamiltonians
* :mod:`.tdvp` -- two-site TDVP and the fourth-order Magnus-TDVP of the paper (Fig. 6)
* :mod:`.evolve` -- the driver :func:`evolve` and :class:`ClosedResult`
"""

from .evolve import METHODS, ClosedResult, evolve, make_stepper
from .hamiltonian import TimeDependentMPO
from .integrators import crank_nicolson_step, rk4_step
from .tdvp import MagnusTDVP, tdvp_frozen_step, tdvp_step

__all__ = [
    "METHODS",
    "ClosedResult",
    "MagnusTDVP",
    "TimeDependentMPO",
    "crank_nicolson_step",
    "evolve",
    "make_stepper",
    "rk4_step",
    "tdvp_frozen_step",
    "tdvp_step",
]
