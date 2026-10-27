"""Alternating least squares (two-site, Galerkin) solution of ``A x = b`` on tensor trains.

At the pair of sites ``(k, k+1)``, with ``x`` in canonical form centred there, the Galerkin
condition ``P_k (A x - b) = 0`` is a small linear system for the merged core ``theta``:

    (L_A W_k W_{k+1} R_A) theta = L_b b_k b_{k+1} R_b,

solved by GMRES with ``theta`` as the initial guess, then split by a truncated SVD (ranks adapt
to ``tol``). For ``A = 1 + i(h/2) H`` (Crank--Nicolson) the Hermitian part of ``A`` is the
identity, so the local problems are well conditioned and the sweeps converge in a few passes.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla

from ..core import linalg
from ..core.environments import EnvironmentCache
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import ShapeError
from .effective import h_eff_two_site

__all__ = ["als_solve"]


def _two_site_projection(Lk, W1, W2, Rk, a1, a2) -> np.ndarray:
    t = linalg.contract(Lk, [2], a1, [0])  # (l, l_w, i, r_a)
    t = linalg.contract(t, [1, 2], W1, [0, 2])  # (l, r_a, o1, r_w)
    t = linalg.contract(t, [1], a2, [0])  # (l, o1, r_w, i2, r_a2)
    t = linalg.contract(t, [2, 3], W2, [0, 2])  # (l, o1, r_a2, o2, r_w2)
    return linalg.contract(t, [2, 4], Rk, [2, 1])  # (l, o1, o2, r)


def als_solve(
    A: MPO,
    b: TensorTrain,
    x0: TensorTrain,
    *,
    tol: float = 0.0,
    chi_max: int | None = None,
    max_sweeps: int = 4,
    conv_tol: float = 1e-10,
    gmres_tol: float = 1e-12,
) -> TensorTrain:
    """Approximate solution of ``A x = b`` (two-site ALS sweeps started from ``x0``)."""
    if A.dims_in != b.dims or b.dims != x0.dims:
        raise ShapeError(f"dims differ: A {A.dims_in}, b {b.dims}, x0 {x0.dims}")
    L = A.length
    ident = MPO.identity(list(b.dims))
    if L == 1:
        M = A.to_dense()
        return TensorTrain([np.linalg.solve(M, b.to_dense()).reshape(1, -1, 1)], copy=False)
    x = x0.canonicalize(0)
    cA = EnvironmentCache(x, A, x)
    cb = EnvironmentCache(x, ident, b)
    previous = None
    for _ in range(max_sweeps):
        for direction in (+1, -1):
            sites = range(L - 1) if direction > 0 else range(L - 2, -1, -1)
            for k in sites:
                rhs = _two_site_projection(
                    cb.left(k), ident.cores[k], ident.cores[k + 1], cb.right(k + 1), b.cores[k], b.cores[k + 1]
                )
                mv = h_eff_two_site(cA.left(k), A.cores[k], A.cores[k + 1], cA.right(k + 1))
                theta0 = linalg.contract(cA.bra[k], [2], cA.bra[k + 1], [0])
                shape = rhs.shape
                n = int(np.prod(shape))
                op = spla.LinearOperator(
                    (n, n), matvec=lambda v, mv=mv, shape=shape: mv(v.reshape(shape)).reshape(-1), dtype=np.complex128
                )
                sol, _info = spla.gmres(
                    op, rhs.reshape(-1), x0=theta0.reshape(-1), rtol=gmres_tol, atol=0.0, restart=min(n, 40), maxiter=20
                )
                res = linalg.svd(sol.reshape(shape), split=2, tol=tol, chi_max=chi_max)
                if direction > 0:
                    left, right = res.U, res.S[:, None, None] * res.Vh
                else:
                    left, right = res.U * res.S, res.Vh
                for c in (cA,):
                    c.update_both(k, left)
                    c.update_both(k + 1, right)
                cb.update_bra(k, left)
                cb.update_bra(k + 1, right)
        norm2 = float(np.real(np.vdot(cA.bra[0], cA.bra[0])))
        if previous is not None and abs(norm2 - previous) <= conv_tol * max(norm2, 1e-300):
            break
        previous = norm2
    return TensorTrain(cA.bra, center=0)
