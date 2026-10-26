"""Conventions shared by the whole code base, as tested code.

* Quantics index map (paper Eq. 4): bits most significant first,
  ``n = sum_{i=1..R} 2^(R-i) sigma_i``; sigma_1 sits next to the backbone.
* Truncation rule: ``tol`` is the discarded relative weight per bond.

The dense oracle (:mod:`qttlind.reference`) implements the same rules independently;
``tests/test_conventions.py`` checks that both agree.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..errors import InputError

__all__ = ["MAX_BITS", "bits_array", "bits_of", "index_of", "keep_count"]

#: Largest supported number of bits per mode (2^R must fit in a signed 64-bit integer).
MAX_BITS = 62


def _check_bits(bits: int) -> None:
    if not isinstance(bits, (int, np.integer)) or isinstance(bits, bool):
        raise InputError(f"number of bits must be an integer, got {type(bits).__name__}")
    if not 1 <= bits <= MAX_BITS:
        raise InputError(f"number of bits must be in 1..={MAX_BITS}, got {bits}")


def bits_of(n: int, bits: int) -> list[int]:
    """Binary digits of ``n`` on ``bits`` bits, most significant first.

    Parameters
    ----------
    n : int
        Integer in ``0 .. 2**bits - 1``.
    bits : int
        Number of bits, ``1 <= bits <= MAX_BITS``.

    Returns
    -------
    list of int
        ``[sigma_1, ..., sigma_R]`` with ``n = sum_i 2**(R-i) * sigma_i``.

    Raises
    ------
    InputError
        If ``bits`` is out of range or ``n`` does not fit.

    Examples
    --------
    >>> bits_of(5, 4)
    [0, 1, 0, 1]
    """
    _check_bits(bits)
    n = int(n)
    if not 0 <= n < 1 << bits:
        raise InputError(f"{n} does not fit in {bits} bits")
    return [(n >> (bits - 1 - i)) & 1 for i in range(bits)]


def index_of(sigma: Iterable[int]) -> int:
    """Inverse of :func:`bits_of`: the integer whose digits (MSB first) are ``sigma``.

    Raises
    ------
    InputError
        If ``sigma`` is empty, too long, or contains a digit other than 0 or 1.

    Examples
    --------
    >>> index_of([1, 0, 1, 0])
    10
    """
    digits = [int(s) for s in sigma]
    _check_bits(len(digits))
    n = 0
    for s in digits:
        if s not in (0, 1):
            raise InputError(f"quantics digits must be 0 or 1, got {s}")
        n = 2 * n + s
    return n


def bits_array(n: np.ndarray | Iterable[int], bits: int) -> np.ndarray:
    """Vectorised :func:`bits_of`: an array of integers -> array of digits, shape ``n.shape + (bits,)``.

    The digits are returned as ``int8`` (MSB first along the last axis).

    Examples
    --------
    >>> bits_array([0, 5], 4).tolist()
    [[0, 0, 0, 0], [0, 1, 0, 1]]
    """
    _check_bits(bits)
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        raise InputError(f"bits_array needs integers, got dtype {n.dtype}")
    n = n.astype(np.int64)
    if n.size and (n.min() < 0 or n.max() >= 1 << bits):
        raise InputError(f"values must be in 0..{(1 << bits) - 1}")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    return ((n[..., None] >> shifts) & 1).astype(np.int8)


def keep_count(s: np.ndarray | Iterable[float], tol: float, chi_max: int | None = None) -> tuple[int, float]:
    """Number of singular values to keep, and the relative weight actually discarded.

    For ``s`` sorted non-increasingly, keep the smallest ``k >= 1`` with
    ``sum_{j>=k} s_j^2 / sum_j s_j^2 <= tol``, then cap ``k`` at ``max(1, chi_max)``.
    An empty spectrum keeps 0 and an all-zero one keeps 1, both discarding 0.

    On the purification index the discarded weight is the trace-norm error made on
    ``rho = Psi Psi^dagger``.

    Raises
    ------
    InputError
        If ``tol`` is negative or not finite, or ``s`` is not a sorted 1-D array of finite,
        non-negative numbers.

    Examples
    --------
    >>> keep_count([3.0, 2.0, 1.0], tol=0.1)
    (2, 0.07142857142857142)
    """
    if not (np.isfinite(tol) and tol >= 0):
        raise InputError(f"tol must be finite and >= 0, got {tol}")
    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise InputError("singular values must be a 1-D sequence")
    if s.size == 0:
        return 0, 0.0
    if not (np.all(np.isfinite(s)) and np.all(s >= 0)):
        raise InputError("singular values must be finite and >= 0")
    if np.any(np.diff(s) > 0):
        raise InputError("singular values must be sorted non-increasingly")
    w = s * s
    total = float(w.sum())
    if total == 0.0:
        return 1, 0.0
    # tail[k] = sum_{j>=k} w_j, summed from the small end; tail is non-increasing, so the
    # smallest admissible k is the number of entries still above the threshold.
    tail = np.cumsum(w[::-1])[::-1]
    keep = max(1, int(np.count_nonzero(tail > tol * total)))
    if chi_max is not None:
        keep = min(keep, max(1, int(chi_max)))
    return keep, float(w[keep:].sum() / total)
