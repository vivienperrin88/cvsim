"""Quantics tensor cross interpolation (TCI): a tensor train of ``f(n)`` from few evaluations.

A function ``f`` on ``n = 0 .. 2^R - 1`` (bits MSB first) is a tensor ``F[s_1, ..., s_R]``. TCI
builds its tensor train from ``O(R chi^2)`` evaluations instead of ``2^R``, by interpolating every
unfolding on a few rows (left multi-indices ``I_k``) and columns (right multi-indices ``J_k``):

    F ~= prod_k  T_k P_k^{-1},    T_k = F[I_{k-1}, s_k, J_k],    P_k = F[I_k, J_k].

Two-site sweeps update the pivots of bond ``k`` by a fully pivoted cross approximation (a
rank-revealing LU) of ``Pi = F[I_{k-1}, s_k, s_{k+1}, J_{k+1}]``, adding pivots where the current
interpolation is worst, until the largest residual is below ``sqrt(tol)`` times the largest
``|f|`` seen. See Núñez Fernández et al., SciPost Phys. 18, 104 (2025), Secs. 3-4.

``tol`` follows the library convention (a discarded relative weight): the max-norm target is
``sqrt(tol)``. Evaluations are cached, so revisited pivots are free; randomness (the initial
pivot search) comes from an explicit ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..errors import InputError
from .tt import TensorTrain

__all__ = ["TCIInfo", "quantics_tci"]


@dataclass
class TCIInfo:
    n_evals: int = 0  # number of distinct function evaluations
    errors: list[float] = field(default_factory=list)  # relative max-norm residual after each sweep
    ranks: tuple[int, ...] = ()
    converged: bool = False


class _CachedFunction:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], bits: int):
        self.f = f
        self.bits = bits
        self.cache: dict[int, complex] = {}
        self.max_abs = 0.0

    def __call__(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=np.int64)
        flat = n.reshape(-1)
        missing = np.array(sorted({int(x) for x in flat if int(x) not in self.cache}), dtype=np.int64)
        if missing.size:
            vals = np.asarray(self.f(missing), dtype=np.complex128).reshape(-1)
            if vals.shape != missing.shape:
                raise InputError(f"f must return one value per index: got shape {vals.shape} for {missing.shape}")
            for k, v in zip(missing.tolist(), vals.tolist()):
                self.cache[k] = v
            self.max_abs = max(self.max_abs, float(np.abs(vals).max()))
        return np.array([self.cache[int(x)] for x in flat], dtype=np.complex128).reshape(n.shape)


def _cross(Pi: np.ndarray, eps_abs: float, chi_max: int | None) -> tuple[list[int], list[int], float]:
    """Fully pivoted greedy cross (rank-revealing LU) of a small matrix: pivot rows, columns, residual."""
    E = Pi.copy()
    rows: list[int] = []
    cols: list[int] = []
    cap = min(Pi.shape) if chi_max is None else min(chi_max, *Pi.shape)
    while len(rows) < cap:
        r, c = np.unravel_index(int(np.argmax(np.abs(E))), E.shape)
        piv = E[r, c]
        if abs(piv) <= eps_abs and rows:
            break
        if piv == 0:
            break
        rows.append(int(r))
        cols.append(int(c))
        E = E - np.outer(E[:, c], E[r, :]) / piv
    return rows, cols, float(np.abs(E).max()) if E.size else 0.0


def quantics_tci(
    f: Callable[[np.ndarray], np.ndarray],
    bits: int,
    *,
    tol: float = 1e-20,
    chi_max: int | None = None,
    max_sweeps: int = 20,
    seed: int = 0,
    n_initial: int = 1000,
) -> tuple[TensorTrain, TCIInfo]:
    """Tensor train of ``f(n)``, ``n`` on ``bits`` bits, by two-site TCI.

    Parameters
    ----------
    f
        Vectorised: called with an int64 array of indices, returns values of the same shape.
    tol
        Discarded relative weight; the max-norm target is ``sqrt(tol) * max|f|``.
    seed
        Seeds the random search for the initial pivot.
    """
    if not 1 <= bits <= 62:
        raise InputError(f"bits must be in 1..62, got {bits}")
    F = _CachedFunction(f, bits)
    R = bits
    weights = [1 << (R - 1 - k) for k in range(R)]  # value of bit k (site k)
    rng = np.random.default_rng(seed)
    size = 1 << R
    if size <= n_initial:
        candidates = np.arange(size, dtype=np.int64)
    else:
        candidates = np.unique(np.concatenate([[0, size - 1], rng.integers(0, size, n_initial)]))
    vals = F(candidates)
    start = int(candidates[int(np.argmax(np.abs(vals)))])
    if F.max_abs == 0:
        cores = [np.zeros((1, 2, 1), dtype=np.complex128) for _ in range(R)]
        return TensorTrain(cores, copy=False), TCIInfo(len(F.cache), [0.0], (1,) * (R - 1), True)
    sbits = [(start >> (R - 1 - k)) & 1 for k in range(R)]
    # left multi-indices of length k as integer prefix values (n restricted to the first k bits);
    # right multi-indices covering sites k..R-1 as integer suffix values
    Ileft: list[list[int]] = [[sum(sbits[j] * weights[j] for j in range(k))] for k in range(R + 1)]
    Jright: list[list[int]] = [[sum(sbits[j] * weights[j] for j in range(k, R))] for k in range(R + 1)]
    info = TCIInfo()
    eps = float(np.sqrt(tol))
    if R == 1:
        T = F(np.array([0, 1])).reshape(1, 2, 1)
        return TensorTrain([T], copy=False), TCIInfo(len(F.cache), [0.0], (), True)

    def pi_matrix(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        I = np.array(Ileft[k], dtype=np.int64)
        J = np.array(Jright[k + 2], dtype=np.int64)
        s1 = np.array([0, weights[k]], dtype=np.int64)
        s2 = np.array([0, weights[k + 1]], dtype=np.int64)
        rows = (I[:, None] + s1[None, :]).reshape(-1)  # (i, s_k)
        cols = (s2[:, None] + J[None, :]).reshape(-1)  # (s_{k+1}, j)
        return F(rows[:, None] + cols[None, :]), rows, cols

    last_ranks: tuple[int, ...] = ()
    for _sweep in range(max_sweeps):
        err = 0.0
        order = list(range(R - 1)) + list(range(R - 3, -1, -1))
        for k in order:
            Pi, rows, cols = pi_matrix(k)
            scale = max(F.max_abs, 1e-300)
            r_idx, c_idx, resid = _cross(Pi, eps * scale, chi_max)
            err = max(err, resid / scale)
            Ileft[k + 1] = [int(rows[i]) for i in r_idx]
            Jright[k + 1] = [int(cols[j]) for j in c_idx]
        ranks = tuple(len(Ileft[k + 1]) for k in range(R - 1))
        info.errors.append(err)
        if err <= eps and ranks == last_ranks:
            info.converged = True
            break
        last_ranks = ranks
    # assemble T_k P_k^{-1}
    cores = []
    for k in range(R):
        I = np.array(Ileft[k], dtype=np.int64)
        J = np.array(Jright[k + 1], dtype=np.int64)
        s = np.array([0, weights[k]], dtype=np.int64)
        T = F(I[:, None, None] + s[None, :, None] + J[None, None, :])  # (|I_k|, 2, |J_{k+1}|)
        if k < R - 1:
            Inext = np.array(Ileft[k + 1], dtype=np.int64)
            P = F(Inext[:, None] + J[None, :])  # (r, r)
            Tm = T.reshape(-1, T.shape[2])
            A = np.linalg.lstsq(P.T, Tm.T, rcond=None)[0].T  # T P^{-1}
            cores.append(A.reshape(T.shape))
        else:
            cores.append(T)
    info.n_evals = len(F.cache)
    info.ranks = tuple(c.shape[2] for c in cores[:-1])
    return TensorTrain(cores, copy=False), info
