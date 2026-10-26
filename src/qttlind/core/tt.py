"""Tensor trains (matrix product states).

A :class:`TensorTrain` of length ``L`` represents a vector with ``L`` indices
``psi[s_1, ..., s_L]`` as a product of 3-index cores ``A_i[l, s, r]``::

    psi[s_1..s_L] = A_1[0, s_1, :] @ A_2[:, s_2, :] @ ... @ A_L[:, s_L, 0]

Core axes are always ``(left bond, physical, right bond)``; the outer bonds have dimension 1.
Index ``s_1`` is the most significant one: ``to_dense()`` returns the vector in C order.

Design choices:

* Cores are plain ``complex128`` arrays; the class only adds validation and algorithms.
* Methods never modify ``self``: they return new objects (cheap to reason about, easy to test).
* ``center`` records the orthogonality centre when it is known (``None`` otherwise): every core
  left of it is a left isometry and every core right of it a right isometry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InputError, ShapeError
from . import linalg

__all__ = ["TensorTrain"]


def _validate_cores(cores: Sequence[np.ndarray]) -> None:
    """Raise :class:`ShapeError` unless ``cores`` form a valid tensor train."""
    if len(cores) == 0:
        raise ShapeError("a tensor train needs at least one core")
    for i, c in enumerate(cores):
        if c.ndim != 3:
            raise ShapeError(f"core {i} must have 3 axes (left, phys, right), got shape {c.shape}")
        if min(c.shape) < 1:
            raise ShapeError(f"core {i} has an empty axis: shape {c.shape}")
    if cores[0].shape[0] != 1:
        raise ShapeError(f"first core must have left bond 1, got {cores[0].shape[0]}")
    if cores[-1].shape[2] != 1:
        raise ShapeError(f"last core must have right bond 1, got {cores[-1].shape[2]}")
    for i in range(len(cores) - 1):
        if cores[i].shape[2] != cores[i + 1].shape[0]:
            raise ShapeError(
                f"bond {i}: core {i} has right bond {cores[i].shape[2]} but core {i + 1} "
                f"has left bond {cores[i + 1].shape[0]}"
            )


class TensorTrain:
    """A vector stored as a tensor train (see the module docstring).

    Parameters
    ----------
    cores : sequence of array_like, each of shape ``(left, phys, right)``
    center : int, optional
        Orthogonality centre, if the caller guarantees one.
    copy : bool
        Copy the cores (default). Pass ``False`` only for cores the caller will not reuse.
    """

    __slots__ = ("center", "cores")
    cores: list[np.ndarray]  # (left, phys, right) complex128 arrays
    center: int | None  # orthogonality centre, if known

    def __init__(self, cores: Sequence[np.ndarray], center: int | None = None, *, copy: bool = True):
        if copy:
            arrays = [np.array(c, dtype=np.complex128) for c in cores]
        else:  # no copy unless a dtype conversion forces one
            arrays = [np.asarray(c, dtype=np.complex128) for c in cores]
        _validate_cores(arrays)
        if center is not None and not 0 <= center < len(arrays):
            raise InputError(f"center must be in 0..{len(arrays) - 1} or None, got {center}")
        self.cores = arrays
        self.center = center

    # ------------------------------------------------------------------ basic properties
    @property
    def length(self) -> int:
        """Number of cores ``L``."""
        return len(self.cores)

    @property
    def dims(self) -> tuple[int, ...]:
        """Physical dimensions ``(d_1, ..., d_L)``."""
        return tuple(c.shape[1] for c in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Inner bond dimensions ``(chi_1, ..., chi_{L-1})``."""
        return tuple(c.shape[2] for c in self.cores[:-1])

    @property
    def max_rank(self) -> int:
        """Largest inner bond dimension (1 for a product state or a single core)."""
        return max(self.ranks, default=1)

    @property
    def size(self) -> int:
        """Total number of stored complex numbers."""
        return sum(c.size for c in self.cores)

    def __repr__(self) -> str:
        return f"TensorTrain(length={self.length}, dims={self.dims}, ranks={self.ranks}, center={self.center})"

    def copy(self) -> TensorTrain:
        """Deep copy."""
        return TensorTrain(self.cores, self.center, copy=True)

    # ------------------------------------------------------------------ dense conversion
    @classmethod
    def from_dense(
        cls, v: np.ndarray, dims: Sequence[int], tol: float = 0.0, chi_max: int | None = None
    ) -> TensorTrain:
        """TT-SVD: compress a dense vector of length ``prod(dims)`` into a tensor train.

        Sweeps left to right; each step splits off one physical index with a truncated SVD.
        The result is left-canonical with ``center = L - 1``.

        Raises
        ------
        ShapeError
            If ``v.size != prod(dims)``.
        """
        dims = tuple(int(d) for d in dims)
        if len(dims) == 0 or min(dims) < 1:
            raise InputError(f"dims must be a non-empty sequence of positive integers, got {dims}")
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        if v.size != int(np.prod(dims)):
            raise ShapeError(f"vector of size {v.size} does not match dims {dims} (product {int(np.prod(dims))})")
        cores = []
        rest = v.reshape(1, -1)
        for d in dims[:-1]:
            left = rest.shape[0]
            res = linalg.svd(rest.reshape(left, d, -1), split=2, tol=tol, chi_max=chi_max)
            cores.append(res.U)  # (left, d, k)
            rest = res.S[:, None] * res.Vh  # (k, remaining)
        cores.append(rest.reshape(rest.shape[0], dims[-1], 1))
        return cls(cores, center=len(dims) - 1, copy=False)

    def to_dense(self) -> np.ndarray:
        """Contract all cores into the dense vector of length ``prod(dims)`` (C order, ``s_1`` slowest)."""
        out = self.cores[0]
        for c in self.cores[1:]:
            out = linalg.contract(out, [-1], c, [0])
        return out.reshape(-1)

    @classmethod
    def product(cls, vectors: Sequence[np.ndarray]) -> TensorTrain:
        """Rank-1 tensor train ``v_1 (x) v_2 (x) ... (x) v_L`` from local vectors."""
        if len(vectors) == 0:
            raise InputError("product needs at least one local vector")
        cores = []
        for i, v in enumerate(vectors):
            v = np.asarray(v, dtype=np.complex128)
            if v.ndim != 1 or v.size == 0:
                raise ShapeError(f"local vector {i} must be a non-empty 1-D array, got shape {v.shape}")
            cores.append(v.reshape(1, -1, 1))
        return cls(cores, copy=False)

    # ------------------------------------------------------------------ canonical forms
    def canonicalize(self, center: int) -> TensorTrain:
        """Return an equivalent tensor train whose orthogonality centre is ``center``.

        QR sweeps make cores ``0..center-1`` left isometries and cores ``center+1..L-1`` right
        isometries; the norm of the whole vector then sits in core ``center``.
        """
        if not 0 <= center < self.length:
            raise InputError(f"center must be in 0..{self.length - 1}, got {center}")
        cores = [c.copy() for c in self.cores]
        for i in range(center):  # left sweep: A = Q R, push R right
            q, r = linalg.qr(cores[i], split=2)
            cores[i] = q
            cores[i + 1] = linalg.contract(r, [1], cores[i + 1], [0])
        for i in range(self.length - 1, center, -1):  # right sweep: A = L Q, push L left
            c = cores[i]
            l_, d, r_ = c.shape
            q, r = linalg.qr(c.reshape(l_, d * r_).conj().T, split=1)  # A^dag = Q R
            k = q.shape[1]
            cores[i] = q.conj().T.reshape(k, d, r_)  # A = R^dag Q^dag
            cores[i - 1] = linalg.contract(cores[i - 1], [2], r.conj().T, [0])
        return TensorTrain(cores, center=center, copy=False)

    def is_left_isometry(self, i: int, atol: float = 1e-10) -> bool:
        """``sum_{l,s} conj(A[l,s,r]) A[l,s,r'] == delta_{r r'}`` for core ``i``."""
        c = self.cores[i]
        m = c.reshape(-1, c.shape[2])
        return bool(np.allclose(m.conj().T @ m, np.eye(c.shape[2]), atol=atol))

    def is_right_isometry(self, i: int, atol: float = 1e-10) -> bool:
        """``sum_{s,r} A[l,s,r] conj(A[l',s,r]) == delta_{l l'}`` for core ``i``."""
        c = self.cores[i]
        m = c.reshape(c.shape[0], -1)
        return bool(np.allclose(m @ m.conj().T, np.eye(c.shape[0]), atol=atol))

    def inner(self, other: TensorTrain) -> complex:
        """``<self|other> = sum conj(self) * other``, contracted site by site (cost ~ L d chi^3)."""
        if self.dims != other.dims:
            raise ShapeError(f"inner: dims {self.dims} and {other.dims} differ")
        env = np.ones((1, 1), dtype=np.complex128)
        for a, b in zip(self.cores, other.cores):
            t = linalg.contract(env, [1], b, [0])  # (l_a, s, r_b)
            env = linalg.contract(a.conj(), [0, 1], t, [0, 1])  # (r_a, r_b)
        return complex(env[0, 0])

    def norm(self) -> float:
        """Euclidean norm; O(size of one core) when the centre is known."""
        if self.center is not None:
            return float(np.linalg.norm(self.cores[self.center]))
        return float(np.sqrt(max(self.inner(self).real, 0.0)))

    def compress(self, tol: float = 0.0, chi_max: int | None = None) -> tuple[TensorTrain, float]:
        """Truncate every bond with the shared rule; return ``(compressed, discarded)``.

        Canonicalise to the last site, then sweep right to left with truncated SVDs (each bond
        is then truncated in an orthonormal frame, so its singular values are the true Schmidt
        values). ``discarded`` is the sum over bonds of the discarded relative weights, an
        upper-bound estimate of the squared relative error. The result has ``center = 0``.
        """
        tt = self.canonicalize(self.length - 1)
        cores = tt.cores
        discarded = 0.0
        for i in range(self.length - 1, 0, -1):
            res = linalg.svd(cores[i], split=1, tol=tol, chi_max=chi_max)
            cores[i] = res.Vh  # (k, d, r)
            cores[i - 1] = linalg.contract(cores[i - 1], [2], res.U * res.S, [0])
            discarded += res.discarded
        return TensorTrain(cores, center=0, copy=False), discarded

    # ------------------------------------------------------------------ algebra
    def __mul__(self, scalar: complex) -> TensorTrain:
        """Multiply the represented vector by a scalar (scales one core only)."""
        if not np.isscalar(scalar):
            return NotImplemented
        cores = [c.copy() for c in self.cores]
        k = self.center if self.center is not None else 0
        cores[k] *= scalar
        return TensorTrain(cores, center=self.center, copy=False)

    __rmul__ = __mul__

    def __add__(self, other: TensorTrain) -> TensorTrain:
        """Vector addition by direct sum of the cores (bond dimensions add)."""
        if not isinstance(other, TensorTrain):
            return NotImplemented
        if self.dims != other.dims:
            raise ShapeError(f"cannot add tensor trains with dims {self.dims} and {other.dims}")
        if self.length == 1:
            return TensorTrain([self.cores[0] + other.cores[0]], copy=False)
        cores = []
        for i, (a, b) in enumerate(zip(self.cores, other.cores)):
            la, d, ra = a.shape
            lb, _, rb = b.shape
            if i == 0:
                c = np.concatenate([a, b], axis=2)  # (1, d, ra + rb)
            elif i == self.length - 1:
                c = np.concatenate([a, b], axis=0)  # (la + lb, d, 1)
            else:
                c = np.zeros((la + lb, d, ra + rb), dtype=np.complex128)
                c[:la, :, :ra] = a
                c[la:, :, ra:] = b
            cores.append(c)
        return TensorTrain(cores, copy=False)

    def __neg__(self) -> TensorTrain:
        return self * -1.0

    def __sub__(self, other: TensorTrain) -> TensorTrain:
        return self + (-other)

    def hadamard(self, other: TensorTrain) -> TensorTrain:
        """Element-wise product of the represented vectors (bond dimensions multiply)."""
        if self.dims != other.dims:
            raise ShapeError(f"hadamard: dims {self.dims} and {other.dims} differ")
        cores = []
        for a, b in zip(self.cores, other.cores):
            la, d, ra = a.shape
            lb, _, rb = b.shape
            cores.append(np.einsum("asb,csd->acsbd", a, b).reshape(la * lb, d, ra * rb))
        return TensorTrain(cores, copy=False)
