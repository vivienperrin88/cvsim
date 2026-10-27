"""Matrix product operators.

An :class:`MPO` of length ``L`` represents a matrix ``A[(o_1..o_L), (i_1..i_L)]`` as a product of
4-index cores ``W_k[l, o, i, r]`` (left bond, output index, input index, right bond)::

    A[o, i] = W_1[0, o_1, i_1, :] @ W_2[:, o_2, i_2, :] @ ... @ W_L[:, o_L, i_L, 0]

Applied to a :class:`~qttlind.core.tt.TensorTrain`, the input indices are contracted with the
physical indices of the state. As for tensor trains, index 1 is the most significant one.

Trick used throughout: an MPO is a tensor train whose physical index is the fused pair
``(o, i)``, so compression and dense conversion reuse :class:`TensorTrain`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InputError, ShapeError
from . import linalg
from .tt import TensorTrain

__all__ = ["MPO", "apply_mpo"]


class MPO:
    """A matrix stored as a matrix product operator (see the module docstring)."""

    __slots__ = ("cores",)
    cores: list[np.ndarray]  # (left, out, in, right) complex128 arrays

    def __init__(self, cores: Sequence[np.ndarray], *, copy: bool = True):
        if copy:
            arrays = [np.array(c, dtype=np.complex128) for c in cores]
        else:
            arrays = [np.asarray(c, dtype=np.complex128) for c in cores]
        if len(arrays) == 0:
            raise ShapeError("an MPO needs at least one core")
        for k, c in enumerate(arrays):
            if c.ndim != 4:
                raise ShapeError(f"MPO core {k} must have 4 axes (left, out, in, right), got shape {c.shape}")
        if arrays[0].shape[0] != 1 or arrays[-1].shape[3] != 1:
            raise ShapeError("outer bonds of an MPO must have dimension 1")
        for k in range(len(arrays) - 1):
            if arrays[k].shape[3] != arrays[k + 1].shape[0]:
                raise ShapeError(
                    f"MPO bond {k}: core {k} has right bond {arrays[k].shape[3]} but core {k + 1} "
                    f"has left bond {arrays[k + 1].shape[0]}"
                )
        self.cores = arrays

    # ------------------------------------------------------------------ properties
    @property
    def length(self) -> int:
        return len(self.cores)

    @property
    def dims_out(self) -> tuple[int, ...]:
        return tuple(c.shape[1] for c in self.cores)

    @property
    def dims_in(self) -> tuple[int, ...]:
        return tuple(c.shape[2] for c in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(c.shape[3] for c in self.cores[:-1])

    @property
    def max_rank(self) -> int:
        return max(self.ranks, default=1)

    def __repr__(self) -> str:
        return f"MPO(length={self.length}, dims_out={self.dims_out}, dims_in={self.dims_in}, ranks={self.ranks})"

    # ------------------------------------------------------------------ TT view (the fusing trick)
    def as_tensor_train(self) -> TensorTrain:
        """The same data as a tensor train with fused physical index ``(o, i)``."""
        return TensorTrain([c.reshape(c.shape[0], c.shape[1] * c.shape[2], c.shape[3]) for c in self.cores])

    @classmethod
    def from_tensor_train(cls, tt: TensorTrain, dims_out: Sequence[int], dims_in: Sequence[int]) -> MPO:
        """Inverse of :meth:`as_tensor_train`."""
        if tuple(o * i for o, i in zip(dims_out, dims_in)) != tt.dims or len(dims_out) != tt.length:
            raise ShapeError(f"tensor train dims {tt.dims} do not factor as {tuple(dims_out)} x {tuple(dims_in)}")
        return cls([c.reshape(c.shape[0], o, i, c.shape[2]) for c, o, i in zip(tt.cores, dims_out, dims_in)])

    # ------------------------------------------------------------------ dense conversion
    def to_dense(self) -> np.ndarray:
        """The full matrix of shape ``(prod(dims_out), prod(dims_in))``."""
        out = self.cores[0]
        for c in self.cores[1:]:
            out = linalg.contract(out, [-1], c, [0])
        L = self.length
        out = out.reshape([d for pair in zip(self.dims_out, self.dims_in) for d in pair])
        out = out.transpose(list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2)))
        return out.reshape(int(np.prod(self.dims_out)), int(np.prod(self.dims_in)))

    @classmethod
    def from_dense(
        cls,
        a: np.ndarray,
        dims_out: Sequence[int],
        dims_in: Sequence[int] | None = None,
        tol: float = 0.0,
        chi_max: int | None = None,
    ) -> MPO:
        """Compress a dense matrix into an MPO (interleave ``(o_k, i_k)``, then TT-SVD)."""
        dims_out = tuple(int(d) for d in dims_out)
        dims_in = dims_out if dims_in is None else tuple(int(d) for d in dims_in)
        if len(dims_in) != len(dims_out):
            raise ShapeError(f"dims_out {dims_out} and dims_in {dims_in} have different lengths")
        a = np.asarray(a, dtype=np.complex128)
        if a.shape != (int(np.prod(dims_out)), int(np.prod(dims_in))):
            raise ShapeError(f"matrix of shape {a.shape} does not match dims {dims_out} x {dims_in}")
        L = len(dims_out)
        t = a.reshape(dims_out + dims_in)
        t = t.transpose([x for k in range(L) for x in (k, L + k)])
        tt = TensorTrain.from_dense(t.reshape(-1), [o * i for o, i in zip(dims_out, dims_in)], tol, chi_max)
        return cls.from_tensor_train(tt, dims_out, dims_in)

    @classmethod
    def identity(cls, dims: Sequence[int]) -> MPO:
        """Rank-1 identity MPO."""
        return cls([np.eye(d, dtype=np.complex128).reshape(1, d, d, 1) for d in dims], copy=False)

    # ------------------------------------------------------------------ algebra
    def apply(self, tt: TensorTrain) -> TensorTrain:
        """Exact product ``A |psi>``: new bond dimension = MPO rank x state rank (no truncation)."""
        if self.dims_in != tt.dims:
            raise ShapeError(f"MPO input dims {self.dims_in} do not match state dims {tt.dims}")
        cores = []
        for w, a in zip(self.cores, tt.cores):
            lw, o, _, rw = w.shape
            la, _, ra = a.shape
            c = linalg.contract(w, [2], a, [1])  # (lw, o, rw, la, ra)
            c = c.transpose(0, 3, 1, 2, 4)  # (lw, la, o, rw, ra)
            cores.append(c.reshape(lw * la, o, rw * ra))
        return TensorTrain(cores, copy=False)

    def __matmul__(self, other: MPO) -> MPO:
        """Operator product ``self @ other`` (ranks multiply)."""
        if not isinstance(other, MPO):
            return NotImplemented
        if self.dims_in != other.dims_out:
            raise ShapeError(f"cannot multiply: inner dims {self.dims_in} vs {other.dims_out}")
        cores = []
        for w1, w2 in zip(self.cores, other.cores):
            l1, o, _, r1 = w1.shape
            l2, _, i, r2 = w2.shape
            c = linalg.contract(w1, [2], w2, [1])  # (l1, o, r1, l2, i, r2)
            c = c.transpose(0, 3, 1, 4, 2, 5)  # (l1, l2, o, i, r1, r2)
            cores.append(c.reshape(l1 * l2, o, i, r1 * r2))
        return MPO(cores, copy=False)

    def __add__(self, other: MPO) -> MPO:
        """Operator sum (ranks add), via the tensor-train direct sum."""
        if not isinstance(other, MPO):
            return NotImplemented
        if (self.dims_out, self.dims_in) != (other.dims_out, other.dims_in):
            raise ShapeError("cannot add MPOs with different dimensions")
        return MPO.from_tensor_train(self.as_tensor_train() + other.as_tensor_train(), self.dims_out, self.dims_in)

    def __mul__(self, scalar: complex) -> MPO:
        if not np.isscalar(scalar):
            return NotImplemented
        cores = [c.copy() for c in self.cores]
        cores[0] *= scalar
        return MPO(cores, copy=False)

    __rmul__ = __mul__

    def __neg__(self) -> MPO:
        return self * -1.0

    def __sub__(self, other: MPO) -> MPO:
        return self + (-other)

    def dagger(self) -> MPO:
        """Hermitian conjugate: conjugate every core and swap output and input indices."""
        return MPO([c.conj().transpose(0, 2, 1, 3) for c in self.cores], copy=False)

    def compress(self, tol: float = 0.0, chi_max: int | None = None) -> tuple[MPO, float]:
        """Compress through the tensor-train view. ``tol`` is a relative weight in the Frobenius norm."""
        tt, discarded = self.as_tensor_train().compress(tol, chi_max)
        return MPO.from_tensor_train(tt, self.dims_out, self.dims_in), discarded

    def check_dims(self, dims: Sequence[int]) -> None:
        """Raise :class:`InputError` unless the MPO is square with dims ``dims``."""
        if self.dims_out != tuple(dims) or self.dims_in != tuple(dims):
            raise InputError(f"expected a square MPO on dims {tuple(dims)}, got {self.dims_out} x {self.dims_in}")


# ============================================================================= applying an MPO with compression
# Three algorithms behind one function:
#
# * ``naive``: exact product (rank D chi), then compression -- O(L d D^3 chi^3);
# * ``zipup`` (Stoudenmire & White 2010): contract and truncate site by site, carrying the
#   remainder to the right -- O(L d D^2 chi^3), then one final compression;
# * ``fit``: variational two-site sweeps minimising ||phi - W psi||, started from zip-up --
#   optimal for the final rank.

_APPLY_METHODS = ("naive", "zipup", "fit")


def _apply_naive(op: MPO, psi: TensorTrain, tol: float, chi_max: int | None) -> tuple[TensorTrain, float]:
    return op.apply(psi).compress(tol, chi_max)


def _apply_zipup(op: MPO, psi: TensorTrain, tol: float, chi_max: int | None) -> tuple[TensorTrain, float]:
    L = op.length
    if L == 1:
        return _apply_naive(op, psi, tol, chi_max)
    a = psi.canonicalize(0).cores  # right-canonical remainder: truncations happen in a near-orthonormal frame
    tol_zip = tol / 10
    chi_zip = None if chi_max is None else 2 * chi_max
    X = np.ones((1, 1, 1), dtype=np.complex128)  # (n, l_w, l_a)
    cores: list[np.ndarray] = []
    for k in range(L - 1):
        T = linalg.contract(X, [2], a[k], [0])  # (n, l_w, s, r_a)
        T = linalg.contract(T, [1, 2], op.cores[k], [0, 2])  # (n, r_a, o, r_w)
        T = T.transpose(0, 2, 3, 1)  # (n, o, r_w, r_a)
        res = linalg.svd(T, split=2, tol=tol_zip, chi_max=chi_zip)
        cores.append(res.U)  # (n, o, k)
        X = res.S[:, None, None] * res.Vh  # (k, r_w, r_a)
    T = linalg.contract(X, [2], a[L - 1], [0])  # (n, l_w, s, 1)
    T = linalg.contract(T, [1, 2], op.cores[L - 1], [0, 2])  # (n, 1, o, 1)
    cores.append(T.reshape(T.shape[0], T.shape[2], 1))
    return TensorTrain(cores, copy=False).compress(tol, chi_max)


def _apply_fit(
    op: MPO,
    psi: TensorTrain,
    tol: float,
    chi_max: int | None,
    max_sweeps: int,
    conv_tol: float,
    strict: bool,
) -> tuple[TensorTrain, float]:
    import warnings

    from ..errors import ConvergenceError, QttlindWarning
    from .environments import EnvironmentCache

    L = op.length
    phi, discarded = _apply_zipup(op, psi, tol, chi_max)
    if L == 1:
        return phi, discarded
    phi = phi.canonicalize(0)
    cache = EnvironmentCache(phi, op, psi)
    history: list[float] = []
    previous = None
    for _sweep in range(max_sweeps):
        discarded = 0.0
        for direction in (+1, -1):
            sites = range(L - 1) if direction > 0 else range(L - 2, -1, -1)
            for k in sites:
                Lk, Rk = cache.left(k), cache.right(k + 1)
                # projection of W psi on the two-site tangent space at (k, k+1)
                t = linalg.contract(Lk, [2], psi.cores[k], [0])  # (l_phi, l_w, i, r_a)
                t = linalg.contract(t, [1, 2], op.cores[k], [0, 2])  # (l_phi, r_a, o1, r_w)
                t = linalg.contract(t, [1], psi.cores[k + 1], [0])  # (l_phi, o1, r_w, i2, r_a2)
                t = linalg.contract(t, [2, 3], op.cores[k + 1], [0, 2])  # (l_phi, o1, r_a2, o2, r_w2)
                t = linalg.contract(t, [2, 4], Rk, [2, 1])  # (l_phi, o1, o2, r_phi)
                res = linalg.svd(t, split=2, tol=tol, chi_max=chi_max)
                discarded += res.discarded
                if direction > 0:
                    left_core, right_core = res.U, res.S[:, None, None] * res.Vh
                else:
                    left_core, right_core = res.U * res.S, res.Vh
                cache.update_bra(k, left_core)
                cache.update_bra(k + 1, right_core)
        cores = cache.bra
        norm2 = float(np.real(np.vdot(cores[0], cores[0])))  # centre is at 0 after the backward sweep
        history.append(norm2)
        if previous is not None and abs(norm2 - previous) <= conv_tol * max(norm2, 1e-300):
            return TensorTrain(cores, center=0), discarded
        previous = norm2
    msg = f"variational MPO fit did not converge in {max_sweeps} sweeps (norm^2 history {history})"
    if strict:
        raise ConvergenceError(msg)
    warnings.warn(msg, QttlindWarning, stacklevel=3)
    return TensorTrain(cache.bra, center=0), discarded


def apply_mpo(
    op: MPO,
    psi: TensorTrain,
    *,
    method: str = "zipup",
    tol: float = 0.0,
    chi_max: int | None = None,
    max_sweeps: int = 10,
    conv_tol: float = 1e-8,
    strict: bool = True,
) -> tuple[TensorTrain, float]:
    """``W|psi>`` compressed with the shared truncation rule; returns ``(result, discarded weight)``.

    Parameters
    ----------
    method : {"naive", "zipup", "fit"}
        ``zipup`` is the default: a factor ``D`` cheaper than ``naive`` for MPOs of rank ``D``.
        ``fit`` refines the zip-up result by variational two-site sweeps (optimal for a given rank).
    max_sweeps, conv_tol, strict
        Only for ``fit``: budget, convergence criterion on the relative change of ``||phi||^2``,
        and whether a budget overrun raises :class:`~qttlind.errors.ConvergenceError` (default)
        or warns and returns the last iterate.

    Raises
    ------
    ShapeError
        If ``op.dims_in != psi.dims``.
    InputError
        For an unknown method.
    """
    if method not in _APPLY_METHODS:
        raise InputError(f"unknown method {method!r}; choose one of {_APPLY_METHODS}")
    if op.dims_in != psi.dims:
        raise ShapeError(f"MPO input dims {op.dims_in} do not match state dims {psi.dims}")
    if method == "naive":
        return _apply_naive(op, psi, tol, chi_max)
    if method == "zipup":
        return _apply_zipup(op, psi, tol, chi_max)
    return _apply_fit(op, psi, tol, chi_max, max_sweeps, conv_tol, strict)
