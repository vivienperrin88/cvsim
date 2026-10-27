"""Time-dependent variational principle (two-site, symmetric) and Magnus-TDVP (paper Sec. III).

**TDVP by projector splitting** (Haegeman et al., PRB 94, 165116, 2016). A half step ``h/2`` left
to right, centre on the current pair: evolve the merged two-site core forward with the two-site
effective Hamiltonian (Krylov), split it by a truncated SVD (ranks adapt to ``tol``), and evolve
the new centre core **backwards** with the one-site effective Hamiltonian; then the mirror sweep
right to left. The symmetric composition is second order; it is exact (up to Krylov accuracy)
when the bond dimensions can reach their maximum.

**Magnus midpoint** (Eqs. 27-31). For ``H(t) = sum_k f_k(t) H_k`` a TDVP step with ``H`` frozen
makes an ``O(h^2)`` local error. The fourth-order Magnus Hamiltonian of the step,

    H_Omega = (H(t1) + H(t2)) / 2 + i sqrt(3) h / 12 [H(t1), H(t2)],   t_{1,2} = t + (3 -+ sqrt 3) h / 6,

with ``[H(t1), H(t2)] = sum_{k<l} (f_k(t1) f_l(t2) - f_l(t1) f_k(t2)) [H_k, H_l]`` needs the
commutators of the terms only, computed once by :class:`MagnusTDVP`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import linalg
from ..core.environments import EnvironmentCache
from ..core.krylov import expm_krylov_adaptive as expm_krylov
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import ShapeError
from .effective import h_eff_one_site, h_eff_two_site
from .hamiltonian import TimeDependentMPO, _value

__all__ = ["MagnusTDVP", "tdvp_frozen_step", "tdvp_step"]


def tdvp_step(
    H: MPO,
    psi: TensorTrain,
    h: float,
    *,
    tol: float = 0.0,
    chi_max: int | None = None,
    krylov_dim: int = 30,
    krylov_tol: float = 1e-12,
) -> TensorTrain:
    """One symmetric two-site TDVP step of ``i dpsi/dt = H psi`` (``H`` Hermitian, time-independent)."""
    if H.dims_in != psi.dims:
        raise ShapeError(f"Hamiltonian dims {H.dims_in} do not match state dims {psi.dims}")
    L = psi.length
    kw: dict[str, Any] = {"krylov_dim": krylov_dim, "tol": krylov_tol}
    if L == 1:
        one = np.ones((1, 1, 1), dtype=np.complex128)
        A, _ = expm_krylov(h_eff_one_site(one, H.cores[0], one), psi.cores[0], -1j * h, **kw)
        return TensorTrain([A], center=0, copy=False)
    x = psi.canonicalize(0)
    cache = EnvironmentCache(x, H, x)
    half = h / 2
    for k in range(L - 1):  # left to right
        theta = linalg.contract(cache.ket[k], [2], cache.ket[k + 1], [0])
        mv = h_eff_two_site(cache.left(k), H.cores[k], H.cores[k + 1], cache.right(k + 1))
        theta, _ = expm_krylov(mv, theta, -1j * half, **kw)
        res = linalg.svd(theta, split=2, tol=tol, chi_max=chi_max)
        cache.update_both(k, res.U)
        right = res.S[:, None, None] * res.Vh
        if k + 1 < L - 1:
            mv1 = h_eff_one_site(cache.left(k + 1), H.cores[k + 1], cache.right(k + 1))
            right, _ = expm_krylov(mv1, right, 1j * half, **kw)
        cache.update_both(k + 1, right)
    for k in range(L - 2, -1, -1):  # right to left
        theta = linalg.contract(cache.ket[k], [2], cache.ket[k + 1], [0])
        mv = h_eff_two_site(cache.left(k), H.cores[k], H.cores[k + 1], cache.right(k + 1))
        theta, _ = expm_krylov(mv, theta, -1j * half, **kw)
        res = linalg.svd(theta, split=2, tol=tol, chi_max=chi_max)
        cache.update_both(k + 1, res.Vh)
        left = res.U * res.S
        if k > 0:
            mv1 = h_eff_one_site(cache.left(k), H.cores[k], cache.right(k))
            left, _ = expm_krylov(mv1, left, 1j * half, **kw)
        cache.update_both(k, left)
    return TensorTrain(cache.ket, center=0)


def tdvp_frozen_step(
    H: TimeDependentMPO, psi: TensorTrain, t: float, h: float, *, tol: float = 0.0, chi_max: int | None = None
) -> TensorTrain:
    """Plain TDVP for time-dependent ``H``: ``H`` frozen at the start of the step (first order)."""
    return tdvp_step(H.at(t), psi, h, tol=tol, chi_max=chi_max)


class MagnusTDVP:
    """TDVP with the fourth-order Magnus midpoint Hamiltonian of each step.

    ``MagnusTDVP(H)`` for a :class:`TimeDependentMPO`, or ``MagnusTDVP.from_pair(H0, H1, f)`` for
    ``H(t) = H0 + f(t) H1``. The commutators of the terms are computed once (``commutator_count``).
    """

    def __init__(self, H: TimeDependentMPO, *, tol_mpo: float = 1e-24):
        self.H = H
        self.tol_mpo = tol_mpo
        self.commutator_count = 0
        self._comm: list[tuple[int, int, MPO]] = []
        terms = H.terms
        for k in range(len(terms)):
            for l in range(k + 1, len(terms)):
                if not (callable(terms[k][0]) or callable(terms[l][0])):
                    continue  # constant-constant pairs cancel in [H(t1), H(t2)]
                A, B = terms[k][1], terms[l][1]
                C, _ = ((A @ B) - (B @ A)).compress(tol_mpo)
                self._comm.append((k, l, C))
                self.commutator_count += 1

    @classmethod
    def from_pair(cls, H0: MPO, H1: MPO, f, *, tol_mpo: float = 1e-24) -> MagnusTDVP:
        return cls(TimeDependentMPO([(1.0, H0), (f, H1)]), tol_mpo=tol_mpo)

    def magnus_hamiltonian(self, t: float, h: float) -> MPO:
        t1 = t + (3 - np.sqrt(3)) / 6 * h
        t2 = t + (3 + np.sqrt(3)) / 6 * h
        terms = self.H.terms
        f1 = [_value(c, t1) for c, _ in terms]
        f2 = [_value(c, t2) for c, _ in terms]
        acc: MPO | None = None
        for (_, op), a, b in zip(terms, f1, f2):
            term = op * ((a + b) / 2)
            acc = term if acc is None else (acc + term).compress(self.tol_mpo)[0]
        pref = 1j * np.sqrt(3) * h / 12
        for k, l, C in self._comm:
            w = f1[k] * f2[l] - f1[l] * f2[k]
            if w != 0:
                assert acc is not None
                acc, _ = (acc + C * (pref * w)).compress(self.tol_mpo)
        assert acc is not None
        return acc

    def step(
        self,
        H: TimeDependentMPO | None,
        psi: TensorTrain,
        t: float,
        h: float,
        *,
        tol: float = 0.0,
        chi_max: int | None = None,
        **tdvp_kwargs,
    ) -> TensorTrain:
        """Step protocol: ``H`` is ignored (the Hamiltonian is fixed at construction)."""
        return tdvp_step(self.magnus_hamiltonian(t, h), psi, h, tol=tol, chi_max=chi_max, **tdvp_kwargs)
