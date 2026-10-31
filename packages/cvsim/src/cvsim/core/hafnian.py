"""Matrix functions of photon counting: permanent, hafnian, loop hafnian, torontonian.

Written independently of qttlind and The Walrus (decision D2); each has an exact second
algorithm for cross-checking.

* **Permanent** — Glynn's formula with a Gray-code walk over ``δ ∈ {±1}^{n−1}``, ``O(n 2^n)``.
* **Hafnian** — the power-trace formula of Björklund, Cygan and Husfeldt (2019),
  ``O(n³ 2^{n/2})``: ``haf(A) = Σ_{Z ⊆ [n/2]} (−1)^{n/2−|Z|} f(X A_Z)`` with ``f(B)`` the
  ``λ^{n/2}`` coefficient of ``exp(Σ_k Tr(B^k) λ^k / (2k))``; ``method="recursive"`` sums perfect
  matchings by dynamic programming over subsets (``O(n 2^n)``), an independent check.
* **Loop hafnian** — the same subset dynamic programming with self-loops (``A_ii`` pairs ``i``
  with itself).
* **Torontonian** — the definition ``Tor(O) = Σ_{Z ⊆ [m]} (−1)^{m−|Z|} / √det(I − O_Z)``, with
  ``O`` of size ``2m`` (xxpp-like ``(a, a†)`` blocks) and ``O_Z`` the rows and columns of the modes
  in ``Z``.

Sizes are those of photon-counting problems: the subset dynamic programming is meant for up to
~20 rows, the power trace for a few tens.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ShapeError
from . import linalg as la

__all__ = ["hafnian", "loop_hafnian", "permanent", "torontonian"]


def _square(A: npt.ArrayLike, name: str) -> np.ndarray:
    M = np.asarray(A, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ShapeError(f"{name} needs a square matrix, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise InputError(f"{name}: matrix has non-finite entries")
    return M


def _symmetric(A: npt.ArrayLike, name: str) -> np.ndarray:
    M = _square(A, name)
    if not np.allclose(M, M.T, atol=1e-12 * max(1.0, float(np.abs(M).max(initial=0.0)))):
        raise InputError(f"{name} needs a symmetric matrix")
    return M


def _real_if_real(z: complex, A: np.ndarray) -> complex | float:
    return float(z.real) if not np.iscomplexobj(A) or np.all(A.imag == 0) else complex(z)


# ------------------------------------------------------------------------------- permanent
def permanent(A: npt.ArrayLike) -> complex | float:
    """Permanent ``Σ_σ Π_i A_{i σ(i)}`` (Glynn's formula with a Gray-code walk).

    Parameters
    ----------
    A
        Square matrix (``0 x 0`` has permanent 1).
    """
    M = _square(A, "permanent")
    n = M.shape[0]
    if n == 0:
        return 1.0
    row = M.sum(axis=0)  # Σ_i δ_i A_ij with all δ = +1
    total = np.prod(row)
    sign = 1.0
    delta = np.ones(n)
    for k in range(1, 2 ** (n - 1)):
        j = (k & -k).bit_length() - 1  # Gray code: flip δ_{j+1}
        i = j + 1
        delta[i] = -delta[i]
        row = row + 2 * delta[i] * M[i]
        sign = -sign
        total = total + sign * np.prod(row)
    return _real_if_real(complex(total / 2 ** (n - 1)), np.asarray(A))


# ------------------------------------------------------------------------------- hafnian
def _matchings(n: int, pair: Callable[[int, int], complex], loop: Callable[[int], complex] | None) -> complex:
    """Σ over perfect matchings (with self-loops if ``loop``) by DP on subsets: the lowest
    remaining index is paired with a partner (or itself)."""

    @functools.cache
    def f(mask: int) -> complex:
        if mask == 0:
            return 1.0
        i = (mask & -mask).bit_length() - 1
        rest = mask & ~(1 << i)
        total = loop(i) * f(rest) if loop is not None else 0.0
        m = rest
        while m:
            j = (m & -m).bit_length() - 1
            total += pair(i, j) * f(rest & ~(1 << j))
            m &= m - 1
        return complex(total)

    return f((1 << n) - 1)


def _power_trace_hafnian(A: np.ndarray) -> complex:
    n = A.shape[0]
    half = n // 2
    total = 0j
    swap = np.array([[0, 1], [1, 0]], dtype=complex)
    for Z in itertools.product((0, 1), repeat=half):
        idx = [r for k, z in enumerate(Z) if z for r in (2 * k, 2 * k + 1)]
        if not idx:
            coeff = 1.0 if half == 0 else 0.0
        else:
            B = np.kron(np.eye(len(idx) // 2), swap) @ A[np.ix_(idx, idx)]
            lam = la.eigvals(B)
            traces = [np.sum(lam ** (k + 1)) / (2 * (k + 1)) for k in range(half)]
            # λ^half coefficient of exp(Σ_k traces[k-1] λ^k): exp of a power series, term by term
            e = np.zeros(half + 1, dtype=complex)
            e[0] = 1.0
            for m in range(1, half + 1):
                e[m] = sum(k * traces[k - 1] * e[m - k] for k in range(1, m + 1)) / m
            coeff = e[half]
        total += (-1) ** (half - sum(Z)) * coeff
    return total


def hafnian(A: npt.ArrayLike, method: str = "power_trace") -> complex | float:
    """Hafnian ``Σ_{perfect matchings} Π_{(i,j)} A_ij`` of a symmetric matrix (0 for odd size).

    Parameters
    ----------
    A
        Symmetric ``n x n`` matrix (its diagonal is ignored).
    method
        ``"power_trace"`` (Björklund et al.) or ``"recursive"`` (dynamic programming over subsets).
    """
    M = _symmetric(A, "hafnian")
    n = M.shape[0]
    if n % 2:
        return 0.0
    if n == 0:
        return 1.0
    if method == "power_trace":
        value = _power_trace_hafnian(M)
    elif method == "recursive":
        value = _matchings(n, lambda i, j: M[i, j], None)
    else:
        raise InputError(f"method must be 'power_trace' or 'recursive', got {method!r}")
    return _real_if_real(value, np.asarray(A))


def loop_hafnian(A: npt.ArrayLike, diag: npt.ArrayLike | None = None) -> complex | float:
    """Loop hafnian: sum over matchings where an index may also be paired with itself (weight ``A_ii``,
    or ``diag[i]`` when given).

    Parameters
    ----------
    A
        Symmetric ``n x n`` matrix.
    diag
        Self-loop weights replacing the diagonal of ``A``.
    """
    M = _symmetric(A, "loop_hafnian")
    n = M.shape[0]
    d = np.diag(M).copy() if diag is None else np.asarray(diag, dtype=complex).reshape(-1)
    if d.shape != (n,):
        raise ShapeError(f"diag must have length {n}, got {d.shape}")
    if n == 0:
        return 1.0
    value = _matchings(n, lambda i, j: M[i, j], lambda i: d[i])
    real = not (np.iscomplexobj(A) and np.any(np.asarray(A).imag)) and not (
        diag is not None and np.iscomplexobj(diag) and np.any(np.asarray(diag).imag)
    )
    return float(value.real) if real else complex(value)


# ------------------------------------------------------------------------------- torontonian
def torontonian(O: npt.ArrayLike) -> complex | float:
    """Torontonian ``Σ_{Z ⊆ [m]} (−1)^{m−|Z|} / √det(I − O_Z)`` of a ``2m x 2m`` matrix.

    Its blocks are ordered like ``(a_1..a_m, a_1†..a_m†)``; ``O_Z`` keeps the rows and columns of
    the modes in ``Z`` in both halves.

    Parameters
    ----------
    O
        ``2m x 2m`` matrix (``O = I − Q⁻¹`` for click statistics).
    """
    M = _square(O, "torontonian")
    if M.shape[0] % 2:
        raise ShapeError(f"torontonian needs a 2m x 2m matrix, got shape {M.shape}")
    m = M.shape[0] // 2
    total = 0j
    for Z in itertools.product((0, 1), repeat=m):
        modes = [k for k, z in enumerate(Z) if z]
        idx = modes + [m + k for k in modes]
        det = 1.0 + 0j if not idx else complex(np.prod(la.eigvals(np.eye(len(idx)) - M[np.ix_(idx, idx)])))
        total += (-1) ** (m - len(modes)) / np.sqrt(det)
    return _real_if_real(total, np.asarray(O))
