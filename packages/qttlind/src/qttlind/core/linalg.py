"""Dense linear-algebra kernel: the single seam between qttlind and LAPACK.

Every tensor contraction, QR and truncated SVD on a hot path goes through this module. Nothing
else in the package calls ``numpy.linalg`` directly on a hot path, so that

* the time spent inside LAPACK can be measured in one place (:func:`timed`), and
* a compiled kernel could replace one of these functions later without touching anything else.

Conventions
-----------
Tensors are ``complex128`` NumPy arrays. A tensor is turned into a matrix by *fusing* its first
``split`` axes into rows and the remaining axes into columns (C order).
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.linalg

from ..errors import InputError, ShapeError
from .conventions import keep_count

__all__ = ["SVD", "LinalgStats", "contract", "fuse", "qr", "svd", "timed"]


# ----------------------------------------------------------------------------- timing


@dataclass
class LinalgStats:
    """Number of kernel calls and seconds spent inside them while a :func:`timed` block is open."""

    calls: int = 0
    seconds: float = 0.0


_ACTIVE: list[LinalgStats] = []


@contextmanager
def timed() -> Iterator[LinalgStats]:
    """Collect kernel statistics for the duration of a ``with`` block (blocks may nest).

    Examples
    --------
    >>> import numpy as np
    >>> with timed() as stats:
    ...     _ = svd(np.eye(3), split=1)
    >>> stats.calls
    1
    """
    stats = LinalgStats()
    _ACTIVE.append(stats)
    try:
        yield stats
    finally:
        _ACTIVE.remove(stats)


def _record(seconds: float) -> None:
    """Add one kernel call lasting ``seconds`` to every open :func:`timed` block."""
    for stats in _ACTIVE:
        stats.calls += 1
        stats.seconds += seconds


# ----------------------------------------------------------------------------- reshaping


def _normalize_axes(axes: Sequence[int], ndim: int, name: str) -> list[int]:
    """Map possibly negative axes to ``0..ndim-1``; reject out-of-range or repeated axes."""
    out = []
    for ax in axes:
        if not -ndim <= ax < ndim:
            raise ShapeError(f"{name}: axis {ax} out of range for a tensor with {ndim} axes")
        out.append(ax % ndim)
    if len(set(out)) != len(out):
        raise ShapeError(f"{name}: repeated axes {list(axes)}")
    return out


def fuse(t: np.ndarray, split: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """View ``t`` as a matrix: axes ``[:split]`` become rows, axes ``[split:]`` become columns.

    Returns
    -------
    matrix, left_shape, right_shape
        ``matrix.shape == (prod(left_shape), prod(right_shape))``.

    Raises
    ------
    ShapeError
        If ``split`` is not in ``0..t.ndim``.
    """
    if not 0 <= split <= t.ndim:
        raise ShapeError(f"split must be in 0..{t.ndim}, got {split}")
    left, right = t.shape[:split], t.shape[split:]
    return t.reshape(int(np.prod(left)), int(np.prod(right))), left, right


def contract(a: np.ndarray, axes_a: Sequence[int], b: np.ndarray, axes_b: Sequence[int]) -> np.ndarray:
    """Sum over pairs of axes ``(axes_a[k], axes_b[k])`` -- the library's only contraction primitive.

    The result carries the free axes of ``a`` (in order) followed by the free axes of ``b``,
    exactly like ``numpy.tensordot(a, b, (axes_a, axes_b))``. Implemented as
    transpose -> reshape -> ``matmul`` -> reshape so that the work is a single GEMM call.

    Raises
    ------
    ShapeError
        If the axis lists differ in length, contain invalid or repeated axes, or pair axes of
        different dimensions.
    """
    t0 = time.perf_counter()
    if len(axes_a) != len(axes_b):
        raise ShapeError(f"contract: {len(axes_a)} axes of a paired with {len(axes_b)} axes of b")
    ax_a = _normalize_axes(axes_a, a.ndim, "contract(a)")
    ax_b = _normalize_axes(axes_b, b.ndim, "contract(b)")
    for i, j in zip(ax_a, ax_b):
        if a.shape[i] != b.shape[j]:
            raise ShapeError(f"contract: axis {i} of a (dim {a.shape[i]}) paired with axis {j} of b (dim {b.shape[j]})")
    free_a = [i for i in range(a.ndim) if i not in ax_a]
    free_b = [j for j in range(b.ndim) if j not in ax_b]
    am = a.transpose(free_a + ax_a)
    bm = b.transpose(ax_b + free_b)
    k = int(np.prod([a.shape[i] for i in ax_a]))
    out_shape = tuple(a.shape[i] for i in free_a) + tuple(b.shape[j] for j in free_b)
    res = (am.reshape(-1, k) @ bm.reshape(k, -1)).reshape(out_shape)
    _record(time.perf_counter() - t0)
    return res


# ----------------------------------------------------------------------------- factorisations


def qr(t: np.ndarray, split: int) -> tuple[np.ndarray, np.ndarray]:
    """Thin QR of ``t`` fused at ``split``: ``t = contract(Q, [-1], R, [0])``.

    Returns
    -------
    Q : ndarray, shape ``left_shape + (k,)``
        Isometry: fused to a matrix, ``Q^dagger Q = 1``.
    R : ndarray, shape ``(k,) + right_shape``
        With ``k = min(prod(left_shape), prod(right_shape))``.
    """
    t0 = time.perf_counter()
    m, left, right = fuse(t, split)
    q, r = np.linalg.qr(m, mode="reduced")
    k = q.shape[1]
    _record(time.perf_counter() - t0)
    return q.reshape(left + (k,)), r.reshape((k,) + right)


class SVD(NamedTuple):
    """Truncated SVD ``t ~= contract(U * S, [-1], Vh, [0])``."""

    U: np.ndarray
    S: np.ndarray
    Vh: np.ndarray
    discarded: float


def _svd_matrix(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``numpy.linalg.svd`` (LAPACK gesdd), falling back to the slower, more robust gesvd."""
    try:
        u, s, vh = np.linalg.svd(m, full_matrices=False)
    except np.linalg.LinAlgError:
        u, s, vh = scipy.linalg.svd(m, full_matrices=False, lapack_driver="gesvd")
    return u, s, vh


def svd(t: np.ndarray, split: int, tol: float = 0.0, chi_max: int | None = None) -> SVD:
    """Truncated SVD of ``t`` fused at ``split``, using the shared truncation rule.

    Parameters
    ----------
    t : ndarray
    split : int
        Number of leading axes fused into rows.
    tol : float
        Maximal discarded relative weight (see :func:`~qttlind.core.conventions.keep_count`).
    chi_max : int, optional
        Hard cap on the number of kept singular values.

    Returns
    -------
    SVD
        ``U`` of shape ``left_shape + (k,)``, ``S`` of shape ``(k,)``, ``Vh`` of shape
        ``(k,) + right_shape`` and the discarded relative weight.
    """
    if chi_max is not None and chi_max < 1:
        raise InputError(f"chi_max must be >= 1, got {chi_max}")
    t0 = time.perf_counter()
    m, left, right = fuse(t, split)
    u, s, vh = _svd_matrix(m)
    k, discarded = keep_count(s, tol, chi_max)
    _record(time.perf_counter() - t0)
    return SVD(u[:, :k].reshape(left + (k,)), s[:k], vh[:k].reshape((k,) + right), discarded)
