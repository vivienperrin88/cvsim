"""Gates, channels and projections on the purified comb, for the circuit executors.

* :func:`expm_mpo` -- ``exp(X)`` of a single-mode MPO (dense ``expm`` up to 2^10 levels, else
  scaling-and-squaring of a Taylor polynomial in MPO arithmetic).
* :func:`expm_terms` -- ``exp(X) |Psi>`` for ``X`` a sum of local terms touching several modes
  (beam splitters, two-mode squeezers): a Taylor series in ``m`` sub-steps. The sub-step is chosen
  from a bound ``B >= ||X||`` so that ``||X|| / m <= TAU_NORM``: then the series converges to
  machine precision on the *whole* truncated space, so compression noise on high levels is never
  amplified (the same stability argument as for the Kraus steps).
* :func:`apply_kraus` -- ``rho -> sum_i K_i rho K_i^dag`` as a matrix addition of ``K_i Psi``.
* :func:`rank_one` -- the MPO ``|ket><bra|`` of two tensor trains (projections, resets).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import scipy.linalg

from ..core.mpo import MPO, apply_mpo
from ..core.tt import TensorTrain
from ..errors import InputError
from ..lindblad.comb import PurifiedComb
from ..lindblad.kraus import LocalTerm, apply_terms

__all__ = ["apply_kraus", "expm_mpo", "expm_terms", "op_norm", "rank_one"]

#: Largest ``||X|| / m`` of a Taylor sub-step.
TAU_NORM = 3.0
#: Dense matrix exponentials up to this many levels.
DENSE_EXPM_LEVELS = 1024


def op_norm(op: MPO, *, iterations: int = 60) -> float:
    """Spectral norm of a single-mode MPO (dense up to 4096 levels, power iteration on TTs beyond)."""
    D = int(np.prod(op.dims_in))
    if D <= 4096:
        A = op.to_dense()
        if D <= 512:
            return float(np.linalg.norm(A, 2))
        v = np.random.default_rng(0).normal(size=D) + 0j
        s = 0.0
        for _ in range(iterations):
            w = A.conj().T @ (A @ v)
            s = float(np.linalg.norm(w))
            v = w / s
        return math.sqrt(s) * 1.01
    rng = np.random.default_rng(0)
    vt = TensorTrain.product([rng.normal(size=d) + 0j for d in op.dims_in])
    s = 1.0
    opd = op.dagger()
    for _ in range(iterations):
        wt, _ = apply_mpo(op, vt, tol=1e-16)
        wt, _ = apply_mpo(opd, wt, tol=1e-16)
        s = wt.norm()
        vt = wt * (1.0 / s)
    return math.sqrt(s) * 1.05


@functools.lru_cache(maxsize=256)
def _norm_cached(key: int, op: MPO) -> float:
    return op_norm(op)


def _norm(op: MPO) -> float:
    return _norm_cached(id(op), op)


def expm_mpo(X: MPO, *, tol: float = 1e-24, chi_max: int | None = None) -> MPO:
    """``exp(X)`` of a single-mode MPO."""
    D = int(np.prod(X.dims_in))
    if D <= DENSE_EXPM_LEVELS:
        return MPO.from_dense(scipy.linalg.expm(X.to_dense()), X.dims_out, X.dims_in, tol=tol, chi_max=chi_max)
    B = _norm(X)
    squarings = max(0, math.ceil(math.log2(max(B, 1e-300) / 0.5)))
    Y = X * (1.0 / 2**squarings)
    I = MPO.identity(X.dims_in)
    K = next(k for k in range(1, 60) if 0.5**k / math.factorial(k) < 1e-18)
    T = I
    for k in range(K, 0, -1):  # Horner: I + Y/1 (I + Y/2 (I + ...))
        T, _ = (I + (Y @ T) * (1.0 / k)).compress(tol, chi_max)
    for _ in range(squarings):
        T, _ = (T @ T).compress(tol, chi_max)
    return T


def _term_bound(terms: Sequence[LocalTerm]) -> float:
    total = 0.0
    for t in terms:
        b = abs(t.value(0.0))
        for op in t.ops.values():
            b *= _norm(op)
        total += b
    return total


def expm_terms(
    state: PurifiedComb,
    X: Sequence[LocalTerm],
    *,
    tol: float = 1e-14,
    caps: Mapping[str, int] | None = None,
    bound: float | None = None,
) -> tuple[PurifiedComb, dict]:
    """``exp(X) |Psi>`` for ``X`` a sum of local terms (constant coefficients).

    Returns the new state and ``{"substeps", "max_order", "bound"}``.
    """
    B = _term_bound(X) if bound is None else float(bound)
    if B == 0:
        return state, {"substeps": 0, "max_order": 0, "bound": 0.0}
    m = max(1, math.ceil(B / TAU_NORM))
    tau = B / m
    Kmax = next(k for k in range(1, 200) if tau**k / math.factorial(k) < 1e-17)
    Xs = [LocalTerm(t.value(0.0) / m, t.ops) for t in X]
    psi = state
    eps = math.sqrt(max(tol, 1e-32)) * 1e-2
    used = 0
    for _ in range(m):
        acc = psi
        term = psi
        nrm0 = math.sqrt(psi.trace())
        for k in range(1, Kmax + 1):
            nxt = apply_terms(term, Xs, 0.0, tol=tol, caps=caps)
            if nxt is None:
                break
            term = nxt.scale(1.0 / k)
            acc, _ = acc.vector_add(term).compress(tol, caps)
            used = max(used, k)
            if k >= tau and math.sqrt(term.trace()) < eps * nrm0:
                break
        psi = acc
    return psi, {"substeps": m, "max_order": used, "bound": B}


def rank_one(ket: TensorTrain, bra: TensorTrain) -> MPO:
    """``|ket><bra|`` as an MPO (bond dimensions multiply)."""
    if ket.dims != bra.dims:
        raise InputError(f"ket on {ket.dims} and bra on {bra.dims} differ")
    cores = []
    for A, B in zip(ket.cores, bra.cores):
        W = np.einsum("aob,cid->acoibd", A, B.conj())
        cores.append(W.reshape(A.shape[0] * B.shape[0], A.shape[1], B.shape[1], A.shape[2] * B.shape[2]))
    return MPO(cores, copy=False)


def apply_kraus(
    state: PurifiedComb,
    kraus: Sequence[Mapping[int, MPO] | tuple[complex, Mapping[int, MPO]]],
    *,
    tol: float = 1e-14,
    caps: Mapping[str, int] | None = None,
) -> PurifiedComb:
    """``rho -> sum_i K_i rho K_i^dag`` with ``K_i = c_i prod_modes ops`` (no renormalisation)."""
    caps = dict(caps or {})
    chi_mu = caps.pop("chi_mu", None)
    acc: PurifiedComb | None = None
    for K in kraus:
        c, ops = (1.0, K) if isinstance(K, Mapping) else K
        if c == 0:
            continue
        B = state.apply_local(ops, tol=tol, caps=caps).scale(c)
        if acc is None:
            acc = B
        else:
            acc, _ = acc.matrix_add(B).compress(tol, caps)
            acc, _ = acc.truncate_mu(tol, chi_mu)
    if acc is None:
        raise InputError("empty Kraus set")
    acc, _ = acc.truncate_mu(tol, chi_mu)
    acc, _ = acc.compress(tol, caps)
    return acc
