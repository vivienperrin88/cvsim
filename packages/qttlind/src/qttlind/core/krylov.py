"""Krylov-subspace exponential ``exp(tau A) v`` from matrix-vector products (Lanczos / Arnoldi).

The Krylov space ``K_m = span{v, Av, ..., A^{m-1} v}`` has an orthonormal basis ``V_m`` in which
``A`` is upper Hessenberg (tridiagonal when ``A`` is Hermitian): ``V_m^dag A V_m = T_m``. Then

    exp(tau A) v  ~=  ||v|| V_m exp(tau T_m) e_1,

exact when ``K_m`` is invariant. The a-posteriori error estimate is
``||v|| h_{m+1,m} |(exp(tau T_m) e_1)_m|``: the dimension grows until it is below ``tol``.
Full re-orthogonalisation keeps the basis orthonormal in floating point (``m <= ~40`` here).

``matvec`` acts on arrays of any shape (the core of a tensor train, typically): vectors are
flattened internally and the result has the shape of ``v``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ..errors import ConvergenceError, InputError

__all__ = ["KrylovInfo", "expm_krylov", "expm_krylov_adaptive"]


@dataclass(frozen=True)
class KrylovInfo:
    dim: int  # Krylov dimension used
    error: float  # a-posteriori error estimate
    happy_breakdown: bool  # the Krylov space became invariant (result exact)


def expm_krylov(
    matvec: Callable[[np.ndarray], np.ndarray],
    v: np.ndarray,
    tau: complex,
    *,
    krylov_dim: int = 30,
    tol: float = 1e-12,
    hermitian: bool = True,
) -> tuple[np.ndarray, KrylovInfo]:
    """``exp(tau A) v`` with ``A`` given by ``matvec``.

    Parameters
    ----------
    hermitian
        ``A`` Hermitian (Lanczos, tridiagonal ``T``); otherwise Arnoldi. For a Schrodinger step
        pass ``matvec = H`` and ``tau = -1j * dt``.
    tol
        Absolute error target relative to ``||v||``.

    Raises
    ------
    ConvergenceError
        If the error estimate is still above ``tol`` at ``krylov_dim``.
    """
    if krylov_dim < 1:
        raise InputError(f"krylov_dim must be >= 1, got {krylov_dim}")
    shape = v.shape
    v0 = np.asarray(v, dtype=np.complex128).reshape(-1)
    beta = float(np.linalg.norm(v0))
    if beta == 0.0:
        return np.zeros(shape, dtype=np.complex128), KrylovInfo(0, 0.0, True)
    n = v0.size
    mmax = min(krylov_dim, n)
    V = np.zeros((mmax + 1, n), dtype=np.complex128)
    Hm = np.zeros((mmax + 1, mmax), dtype=np.complex128)
    V[0] = v0 / beta
    err = np.inf
    for j in range(mmax):
        w = np.asarray(matvec(V[j].reshape(shape)), dtype=np.complex128).reshape(-1)
        # full (twice-applied) Gram-Schmidt against the whole basis
        for _ in range(2):
            coeffs = V[: j + 1].conj() @ w
            w = w - coeffs @ V[: j + 1]
            Hm[: j + 1, j] += coeffs
        if hermitian:  # enforce the tridiagonal structure (the rest is round-off)
            Hm[: max(0, j - 1), j] = 0.0
        hnext = float(np.linalg.norm(w))
        m = j + 1
        T = Hm[:m, :m]
        if hermitian:
            T = (T + T.conj().T) / 2
        y = scipy.linalg.expm(tau * T)[:, 0]
        if hnext <= 1e-14 * max(1.0, float(np.abs(T).max())):  # invariant subspace: exact
            out = beta * (y @ V[:m])
            return out.reshape(shape), KrylovInfo(m, 0.0, True)
        err = hnext * abs(y[-1])
        if err <= tol or m == n:
            out = beta * (y @ V[:m])
            return out.reshape(shape), KrylovInfo(m, float(err), m == n)
        Hm[m, j] = hnext
        V[m] = w / hnext
    raise ConvergenceError(
        f"Krylov exponential did not reach tol={tol:.1e} with krylov_dim={krylov_dim} (estimate {err:.2e});"
        " increase krylov_dim or reduce the time step"
    )


def expm_krylov_adaptive(
    matvec: Callable[[np.ndarray], np.ndarray],
    v: np.ndarray,
    tau: complex,
    *,
    krylov_dim: int = 30,
    tol: float = 1e-12,
    hermitian: bool = True,
    max_splits: int = 12,
) -> tuple[np.ndarray, KrylovInfo]:
    """:func:`expm_krylov`, splitting ``tau`` into equal sub-steps when the Krylov budget is exceeded.

    ``exp(tau A) = exp(tau A / 2)^2``: each failure doubles the number of sub-steps (at most
    ``2**max_splits``); the error target is shared between the sub-steps.
    """
    pieces = 1
    for _ in range(max_splits + 1):
        try:
            w = v
            used = 0
            for _ in range(pieces):
                w, info = expm_krylov(
                    matvec, w, tau / pieces, krylov_dim=krylov_dim, tol=tol / pieces, hermitian=hermitian
                )
                used = max(used, info.dim)
            return w, KrylovInfo(used, info.error, info.happy_breakdown)
        except ConvergenceError:
            pieces *= 2
    raise ConvergenceError(f"Krylov exponential failed even with {pieces // 2} sub-steps")
