"""Bosonic operators as quantics MPOs.

Matrices act on a single mode truncated to ``N = 2**bits`` Fock states (MSB first).

* :func:`shift` -- the exact rank-2 "binary adder" ``T_{n, n+m} = 1``.
* :func:`diag` -- promote a quantics tensor train ``f(n)`` to ``diag(f)``.
* :func:`number`, :func:`falling_factorial` -- exact polynomial diagonals.
* :func:`annihilation`, :func:`creation`, :func:`annihilation_power` -- ``a = T diag(sqrt n)``.
* :func:`parity` -- ``(-1)^n`` depends on the last bit only: rank 1.
"""

from __future__ import annotations

import numpy as np

from ..core.conventions import bits_of
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import InputError
from .states import from_function

__all__ = [
    "annihilation",
    "annihilation_power",
    "creation",
    "diag",
    "falling_factorial",
    "normal_power",
    "number",
    "number_tt",
    "parity",
    "shift",
]

#: Default discarded weight for operator construction (relative error ~1e-12, Table I).
OPERATOR_TOL = 1e-24


# ----------------------------------------------------------------------------- the shift


def shift(bits: int, m: int = 1) -> MPO:
    """Exact MPO of ``T`` with ``T[n, n'] = 1`` iff ``n' = n + m`` (so ``(T psi)_n = psi_{n+m}``).

    Bond index = carry bit, propagating from the least to the most significant bit. Site ``i``
    receives carry ``c_r`` from the right and adds ``sigma_i + m_i + c_r = sigma'_i + 2 c_l``.
    Boundary conditions: no carry enters the LSB, and a carry leaving the MSB (overflow,
    ``n + m >= N``) is dropped -- those rows of ``T`` are zero.
    """
    if not 0 <= m < 2**bits:
        raise InputError(f"shift amount must be in 0..{2**bits - 1}, got {m}")
    mb = bits_of(m, bits)
    cores = []
    for i in range(bits):
        w = np.zeros((2, 2, 2, 2), dtype=np.complex128)  # (c_l, sigma, sigma', c_r)
        for c_r in (0, 1):
            for s in (0, 1):
                total = s + mb[i] + c_r
                w[total // 2, s, total % 2, c_r] = 1.0
        cores.append(w)
    cores[0] = cores[0][:1]  # carry out of the MSB must be 0
    cores[-1] = cores[-1][..., :1]  # carry into the LSB is 0
    return MPO(cores, copy=False)


# ----------------------------------------------------------------------------- diagonals


def diag(f: TensorTrain) -> MPO:
    """``diag(f)`` as an MPO: ``W[l, s, s', r] = A[l, s, r] delta_{s s'}`` (same ranks as ``f``)."""
    cores = []
    for a in f.cores:
        l_, d, r_ = a.shape
        w = np.zeros((l_, d, d, r_), dtype=np.complex128)
        for s in range(d):
            w[:, s, s, :] = a[:, s, :]
        cores.append(w)
    return MPO(cores, copy=False)


def number_tt(bits: int) -> TensorTrain:
    """Exact rank-2 tensor train of ``f(n) = n``.

    ``n = sum_i 2^(bits-i) sigma_i`` is a sum of one-site terms; each core is the 2x2 transfer
    matrix ``[[1, w_i sigma], [0, 1]]`` that either keeps "nothing added yet" or adds the weight.
    """
    cores = []
    for i in range(bits):
        weight = 2.0 ** (bits - 1 - i)
        c = np.zeros((2, 2, 2), dtype=np.complex128)  # (l, sigma, r)
        for s in (0, 1):
            c[:, s, :] = [[1.0, weight * s], [0.0, 1.0]]
        cores.append(c)
    cores[0] = cores[0][:1]  # start in state "nothing added"
    cores[-1] = cores[-1][..., 1:]  # end in state "sum complete"
    return TensorTrain(cores, copy=False)


def number(bits: int) -> MPO:
    """``a^dagger a = diag(n)``, exact rank 2."""
    return diag(number_tt(bits))


def falling_factorial(m: int, bits: int) -> TensorTrain:
    """Exact tensor train of the falling factorial ``(n)_m`` (rank ``m + 1``).

    ``(n)_m = n (n-1) (n-2) ... (n-m+1)``.

    Built as a Hadamard product of the rank-2 trains of ``n - p`` (``p = 0..m-1``), recompressed
    without truncation after each factor: a degree-``m`` polynomial has quantics rank ``m + 1``.
    """
    if m < 0:
        raise InputError(f"m must be >= 0, got {m}")
    ones = TensorTrain.product([np.ones(2)] * bits)
    out = ones
    n = number_tt(bits)
    for p in range(m):
        factor, _ = (n - p * ones).compress(tol=1e-28)  # drop round-off, keep rank 2
        out, _ = out.hadamard(factor).compress(tol=1e-28)
    return out


def normal_power(m: int, bits: int) -> MPO:
    """``(a^dagger)^m a^m = diag((n)_m)``, exact rank ``m + 1``."""
    return diag(falling_factorial(m, bits))


def annihilation_power(m: int, bits: int, tol: float = OPERATOR_TOL, chi_max: int | None = None) -> MPO:
    """``a^m``: ``<n| a^m |n'> = delta_{n+m, n'} sqrt(n'!/(n'-m)!)`` = shift(m) @ diag(sqrt((n)_m))."""
    if not 1 <= m < 2**bits:
        raise InputError(f"m must be in 1..{2**bits - 1}, got {m}")

    def sqrt_falling(n: np.ndarray) -> np.ndarray:
        out = np.ones(n.shape)
        for p in range(m):
            out *= np.clip(n - p, 0, None)
        return np.sqrt(out)

    d = diag(from_function(sqrt_falling, bits, tol=tol, chi_max=chi_max))
    op, _ = (shift(bits, m) @ d).compress(tol=tol, chi_max=chi_max)
    return op


def annihilation(bits: int, tol: float = OPERATOR_TOL, chi_max: int | None = None) -> MPO:
    """``a`` on ``2**bits`` Fock states (paper Sec. II B 1)."""
    return annihilation_power(1, bits, tol, chi_max)


def creation(bits: int, tol: float = OPERATOR_TOL, chi_max: int | None = None) -> MPO:
    """``a^dagger``, as the Hermitian conjugate of :func:`annihilation`."""
    return annihilation(bits, tol, chi_max).dagger()


def parity(bits: int) -> MPO:
    """``(-1)^{a^dagger a}``. Since ``(-1)^n = (-1)^{sigma_R}``, this is a rank-1 product operator."""
    local = [np.eye(2)] * (bits - 1) + [np.diag([1.0, -1.0])]
    return MPO([w.reshape(1, 2, 2, 1) for w in local])
