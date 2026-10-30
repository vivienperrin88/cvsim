"""Dense linear algebra: the single seam between cvsim and LAPACK (decision D3).

No other cvsim module calls ``numpy.linalg`` or ``scipy.linalg`` (enforced by
``tests/test_guards.py``), so that the time spent inside LAPACK is measured in one place
(:func:`timed`), failures are handled in one place (``gesdd`` → ``gesvd`` fallback, explicit
errors for non-positive matrices), and a compiled kernel could replace one routine later.

Two groups of functions:

* **kernels** -- :func:`eigh`, :func:`eigvalsh`, :func:`svd`, :func:`solve`, :func:`inv`,
  :func:`cholesky`, :func:`logdet_pd`, :func:`sqrtm_psd`, :func:`expm`, :func:`polar`, :func:`qr`;
* **symplectic decompositions** (xxpp, ``Ω = [[0, I], [−I, 0]]``) --
  :func:`symplectic_eigenvalues`, :func:`williamson`, :func:`takagi`, :func:`bloch_messiah`,
  :func:`nearest_physical_covariance`, and :func:`degenerate_groups`, which every decomposition
  reports: inside a group of equal values the factors are defined only up to a gauge, see
  ``CONVENTIONS.md`` ("Decompositions").

Example
-------
>>> import numpy as np
>>> from cvsim.core import conventions as cv, linalg
>>> V = cv.squeezing(0.5) @ cv.squeezing(0.5).T / 2 * 1.5   # a squeezed thermal state
>>> nu, S, groups = linalg.williamson(V)
>>> nu.round(6)
array([0.75])
>>> bool(np.allclose(S @ np.diag([0.75, 0.75]) @ S.T, V))
True
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import NamedTuple, ParamSpec, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.linalg

from ..errors import ConvergenceError, InputError, PhysicalityError, ShapeError
from ._types import ComplexArray, FloatArray, IntArray
from .conventions import ATOL, VACUUM_VARIANCE, omega

__all__ = [
    "BlochMessiah",
    "LinalgStats",
    "Takagi",
    "Williamson",
    "bloch_messiah",
    "cholesky",
    "degenerate_groups",
    "eigh",
    "eigvals",
    "eigvalsh",
    "expm",
    "inv",
    "logdet_pd",
    "nearest_physical_covariance",
    "polar",
    "qr",
    "solve",
    "sqrtm_psd",
    "svd",
    "symplectic_eigenvalues",
    "takagi",
    "timed",
    "williamson",
]

#: Relative tolerance under which two eigenvalues / singular values are treated as equal.
DEGENERACY_RTOL = 1e-8

P = ParamSpec("P")
R = TypeVar("R")


# ------------------------------------------------------------------------------- timing
@dataclass
class LinalgStats:
    """Kernel calls and seconds spent inside them while a :func:`timed` block is open."""

    calls: int = 0
    seconds: float = 0.0
    by_kernel: dict[str, int] = field(default_factory=dict)


_ACTIVE: list[LinalgStats] = []


@contextmanager
def timed() -> Iterator[LinalgStats]:
    """Collect kernel statistics for the duration of a ``with`` block (blocks may nest).

    Examples
    --------
    >>> import numpy as np
    >>> with timed() as stats:
    ...     _ = eigvalsh(np.eye(3))
    >>> stats.calls, stats.by_kernel
    (1, {'eigvalsh': 1})
    """
    stats = LinalgStats()
    _ACTIVE.append(stats)
    try:
        yield stats
    finally:
        _ACTIVE.remove(stats)


def _kernel(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: count and time a kernel in every open :func:`timed` block."""

    def wrap(fn: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                for stats in _ACTIVE:
                    stats.calls += 1
                    stats.seconds += dt
                    stats.by_kernel[name] = stats.by_kernel.get(name, 0) + 1

        inner.__name__ = fn.__name__
        inner.__qualname__ = fn.__qualname__
        inner.__doc__ = fn.__doc__
        inner.__module__ = fn.__module__
        return inner

    return wrap


# ------------------------------------------------------------------------------- validation
def _square(A: npt.ArrayLike, name: str = "matrix") -> np.ndarray:
    M = np.asarray(A)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ShapeError(f"{name} must be a square 2-D array, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise InputError(f"{name} has non-finite entries")
    return M


def _hermitian(A: npt.ArrayLike, name: str = "matrix", atol: float = ATOL) -> np.ndarray:
    M = _square(A, name)
    scale = max(1.0, float(np.abs(M).max(initial=0.0)))
    asym = float(np.abs(M - M.conj().T).max(initial=0.0))
    if asym > atol * scale:
        raise InputError(f"{name} must be symmetric/Hermitian (max asymmetry {asym:.2e})")
    return np.asarray((M + M.conj().T) / 2)


def _covariance(V: npt.ArrayLike) -> FloatArray:
    V_ = _hermitian(V, "covariance")
    if np.iscomplexobj(V_):
        if np.abs(V_.imag).max(initial=0.0) > ATOL:
            raise InputError("covariance must be real")
        V_ = V_.real
    if V_.shape[0] % 2 or V_.shape[0] == 0:
        raise ShapeError(f"covariance must be 2M x 2M, got shape {V_.shape}")
    return np.asarray(V_, dtype=float)


# ------------------------------------------------------------------------------- kernels
@_kernel("eigh")
def eigh(A: npt.ArrayLike) -> tuple[FloatArray, np.ndarray]:
    """Eigen-decomposition ``A = U diag(w) U†`` of a Hermitian matrix, ``w`` ascending.

    Parameters
    ----------
    A
        Real symmetric or complex Hermitian matrix (symmetrised after a check).

    Raises
    ------
    InputError
        ``A`` is not Hermitian or has non-finite entries.
    """
    w, U = np.linalg.eigh(_hermitian(A))
    return np.asarray(w, dtype=float), U


@_kernel("eigvalsh")
def eigvalsh(A: npt.ArrayLike) -> FloatArray:
    """Eigenvalues of a Hermitian matrix, ascending.

    Parameters
    ----------
    A
        Real symmetric or complex Hermitian matrix.
    """
    return np.asarray(np.linalg.eigvalsh(_hermitian(A)), dtype=float)


@_kernel("eigvals")
def eigvals(A: npt.ArrayLike) -> np.ndarray:
    """Eigenvalues of a general square matrix (complex, in LAPACK's order).

    Parameters
    ----------
    A
        Square real or complex matrix.
    """
    return np.asarray(np.linalg.eigvals(_square(A)), dtype=complex)


@_kernel("svd")
def svd(A: npt.ArrayLike, full_matrices: bool = False) -> tuple[np.ndarray, FloatArray, np.ndarray]:
    """``A = U diag(s) Vh`` with ``s`` descending; falls back from ``gesdd`` to ``gesvd``.

    Parameters
    ----------
    A
        Any 2-D array.
    full_matrices
        Square ``U`` and ``Vh`` instead of the thin factors.

    Raises
    ------
    ConvergenceError
        Neither LAPACK driver converged.
    """
    M = np.asarray(A)
    if M.ndim != 2:
        raise ShapeError(f"svd needs a 2-D array, got shape {M.shape}")
    try:
        U, s, Vh = scipy.linalg.svd(M, full_matrices=full_matrices, lapack_driver="gesdd", check_finite=True)
    except np.linalg.LinAlgError:
        try:
            U, s, Vh = scipy.linalg.svd(M, full_matrices=full_matrices, lapack_driver="gesvd", check_finite=True)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - needs a pathological matrix
            raise ConvergenceError(f"SVD did not converge with gesdd nor gesvd: {exc}") from exc
    except ValueError as exc:
        raise InputError(f"svd: {exc}") from exc
    return U, np.asarray(s, dtype=float), Vh


@_kernel("solve")
def solve(A: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Solution ``x`` of ``A x = b``.

    Parameters
    ----------
    A
        Square non-singular matrix.
    b
        Right-hand side(s).

    Raises
    ------
    InputError
        ``A`` is singular.
    """
    try:
        return np.asarray(np.linalg.solve(_square(A), np.asarray(b)))
    except np.linalg.LinAlgError as exc:
        raise InputError(f"singular matrix: {exc}") from exc


@_kernel("inv")
def inv(A: npt.ArrayLike) -> np.ndarray:
    """Inverse of a square non-singular matrix.

    Parameters
    ----------
    A
        Square non-singular matrix.

    Raises
    ------
    InputError
        ``A`` is singular.
    """
    try:
        return np.asarray(np.linalg.inv(_square(A)))
    except np.linalg.LinAlgError as exc:
        raise InputError(f"singular matrix: {exc}") from exc


@_kernel("cholesky")
def cholesky(A: npt.ArrayLike) -> np.ndarray:
    """Lower-triangular ``L`` with ``A = L L†``.

    Parameters
    ----------
    A
        Hermitian positive-definite matrix.

    Raises
    ------
    InputError
        ``A`` is not positive definite.
    """
    try:
        return np.asarray(np.linalg.cholesky(_hermitian(A)))
    except np.linalg.LinAlgError as exc:
        raise InputError("matrix is not positive definite") from exc


def logdet_pd(A: npt.ArrayLike) -> float:
    """``log det A`` of a positive-definite matrix, computed stably from its Cholesky factor.

    Parameters
    ----------
    A
        Hermitian positive-definite matrix.
    """
    L = cholesky(A)
    return float(2.0 * np.sum(np.log(np.abs(np.diag(L)))))


def sqrtm_psd(A: npt.ArrayLike, atol: float = ATOL) -> np.ndarray:
    """Principal square root of a positive-semidefinite matrix (eigenvalues down to ``−atol`` are clipped to 0).

    Parameters
    ----------
    A
        Hermitian positive-semidefinite matrix.
    atol
        Tolerance for slightly negative eigenvalues from round-off (relative to ``max(1, ‖A‖)``).

    Raises
    ------
    InputError
        ``A`` has an eigenvalue below ``−atol``.
    """
    w, U = eigh(A)
    scale = max(1.0, float(np.abs(w).max(initial=0.0)))
    if w.min(initial=0.0) < -atol * scale:
        raise InputError(f"matrix is not positive semidefinite (smallest eigenvalue {w.min():.3e})")
    return np.asarray((U * np.sqrt(np.clip(w, 0.0, None))) @ U.conj().T)


@_kernel("expm")
def expm(A: npt.ArrayLike) -> np.ndarray:
    """Matrix exponential (Padé approximation with scaling and squaring).

    Parameters
    ----------
    A
        Square matrix.
    """
    return np.asarray(scipy.linalg.expm(_square(A)))


@_kernel("qr")
def qr(A: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Thin QR decomposition ``A = Q R``.

    Parameters
    ----------
    A
        Any 2-D array.
    """
    M = np.asarray(A)
    if M.ndim != 2:
        raise ShapeError(f"qr needs a 2-D array, got shape {M.shape}")
    Q, Rm = np.linalg.qr(M)
    return np.asarray(Q), np.asarray(Rm)


def polar(A: npt.ArrayLike, side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """Polar decomposition: ``A = P U`` (``side="left"``) or ``A = U P`` (``"right"``).

    ``U`` is unitary (orthogonal for real ``A``), ``P`` Hermitian positive semidefinite.

    Parameters
    ----------
    A
        Square matrix.
    side
        ``"left"`` or ``"right"``.

    Returns
    -------
    (U, P)
    """
    if side not in ("left", "right"):
        raise InputError(f"side must be 'left' or 'right', got {side!r}")
    W, s, Vh = svd(_square(A))
    U = W @ Vh
    P = (W * s) @ W.conj().T if side == "left" else (Vh.conj().T * s) @ Vh
    return U, P


# ------------------------------------------------------------------------------- degeneracy
def degenerate_groups(values: npt.ArrayLike, rtol: float = DEGENERACY_RTOL) -> list[IntArray]:
    """Index groups of numerically equal values, in the order given (values must be sorted).

    Two neighbours ``v_i``, ``v_{i+1}`` belong to the same group when
    ``|v_{i+1} − v_i| <= rtol · max(1, |v_i|, |v_{i+1}|)``.

    Parameters
    ----------
    values
        Monotonic 1-D array.
    rtol
        Relative tolerance.

    Examples
    --------
    >>> [g.tolist() for g in degenerate_groups([0.5, 0.5, 0.7, 1.0, 1.0 + 1e-12])]
    [[0, 1], [2], [3, 4]]
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    groups: list[list[int]] = []
    for i, x in enumerate(v):
        if groups and abs(x - v[i - 1]) <= rtol * max(1.0, abs(x), abs(v[i - 1])):
            groups[-1].append(i)
        else:
            groups.append([i])
    return [np.asarray(g, dtype=np.int64) for g in groups]


# ------------------------------------------------------------------------------- gauge fixing
def _procrustes(cols: np.ndarray, ref: np.ndarray, kind: str) -> np.ndarray:
    """The rotation ``Q`` of the gauge group making ``cols @ Q`` closest to ``k`` reference columns.

    ``cols`` (``n x k``) span a degenerate subspace; ``ref`` (``n x N``) is the reference basis.
    The ``k`` reference columns with the largest weight in that subspace are selected (ties broken
    by index; kept in reference order), and ``Q`` maximises ``Re Tr(Q† cols† ref_sel)``: the polar
    factor of ``cols† ref_sel`` (``kind="unitary"``, gauge group ``U(k)``) or of its real part
    (``kind="orthogonal"``, gauge group ``O(k)``). A one-dimensional group only fixes a sign or a
    phase: for the identity reference, its largest component is made real-positive (``U(1)``) or
    of positive real part (``O(1)``).
    """
    k = cols.shape[1]
    ov = cols.conj().T @ ref  # k x N
    weight = np.sum(np.abs(ov) ** 2, axis=0)
    sel = np.sort(np.argsort(-weight, kind="stable")[:k])
    G = ov[:, sel]
    if kind == "orthogonal":
        G = G.real
    Q, _ = polar(G, side="right")
    return np.asarray(Q)


def _as_complex_modes(reference: npt.ArrayLike | None, M: int, rows: int) -> np.ndarray:
    """Reference basis as complex columns: ``None`` → the input modes; a ``2M x 2M`` real matrix
    (e.g. a previous ``O1`` or ``S``) → ``x-columns − i·p-columns`` of its first ``M`` pairs; a
    complex ``rows x M`` matrix → itself."""
    if reference is None:
        if rows == M:
            return np.eye(M, dtype=complex)
        eye = np.eye(2 * M)
        return eye[:, :M] - 1j * eye[:, M:]
    R = np.asarray(reference)
    if R.shape == (2 * M, 2 * M) and not np.iscomplexobj(R):
        if rows == M:
            return R[:M, :M] + 1j * R[M:, :M]
        return R[:, :M] - 1j * R[:, M:]
    if R.shape == (rows, M):
        return R.astype(complex)
    raise ShapeError(f"reference must be a real {2 * M}x{2 * M} or a complex {rows}x{M} matrix, got {R.shape}")


# ------------------------------------------------------------------------------- symplectic decompositions
def symplectic_eigenvalues(V: npt.ArrayLike) -> FloatArray:
    """Williamson's symplectic eigenvalues ``ν_k`` of a positive-definite covariance, ascending.

    They are the positive eigenvalues of the Hermitian matrix ``i V^{1/2} Ω V^{1/2}`` (which has
    the same spectrum as ``iΩV``). A covariance is physical iff every ``ν_k >= 1/2``.

    Parameters
    ----------
    V
        Real symmetric positive-definite ``2M x 2M`` matrix (xxpp).

    Raises
    ------
    InputError
        ``V`` is not positive definite.
    """
    V_ = _covariance(V)
    M = V_.shape[0] // 2
    w = eigvalsh(V_)
    if w[0] <= 0:
        raise InputError(f"covariance must be positive definite (smallest eigenvalue {w[0]:.3e})")
    s = sqrtm_psd(V_)
    ev = eigvalsh(1j * s @ omega(M) @ s)
    return np.sort(ev[M:])


class Williamson(NamedTuple):
    """``V = S diag(ν, ν) Sᵀ`` with ``S`` symplectic and ``ν`` ascending."""

    nu: FloatArray
    S: FloatArray
    groups: list[IntArray]


def williamson(V: npt.ArrayLike, reference: npt.ArrayLike | None = None, rtol: float = DEGENERACY_RTOL) -> Williamson:
    """Williamson normal form of a positive-definite covariance: ``V = S diag(ν, ν) Sᵀ``.

    Construction: ``A = V^{-1/2} Ω V^{-1/2}`` is real antisymmetric; the Hermitian matrix ``iA``
    has eigenvalues ``±1/ν_k``. Orthonormal eigenvectors ``z_k`` for ``+1/ν_k`` give real
    orthonormal ``u_k = √2 Re z_k``, ``w_k = √2 Im z_k`` with ``A u_k = w_k / ν_k``; with
    ``K = [u | −w]`` (so that ``Kᵀ A K = [[0, D⁻¹], [−D⁻¹, 0]]``), ``S = V^{1/2} K D^{-1/2}``,
    ``D = diag(ν, ν)``. ``eigh`` returns *some* orthonormal basis of a
    degenerate eigenspace, so inside a group of equal ``ν`` the columns of ``S`` are defined up
    to a passive (orthogonal symplectic) rotation of that group -- the gauge reported in
    ``groups``.

    Parameters
    ----------
    V
        Real symmetric positive-definite ``2M x 2M`` matrix (xxpp).

    Returns
    -------
    Williamson
        ``nu`` (ascending), ``S`` and the degeneracy ``groups`` of ``nu``.

    Raises
    ------
    InputError
        ``V`` is not positive definite.
    """
    V_ = _covariance(V)
    M = V_.shape[0] // 2
    w, U = eigh(V_)
    if w[0] <= 0:
        raise InputError(f"covariance must be positive definite (smallest eigenvalue {w[0]:.3e})")
    half = (U * np.sqrt(w)) @ U.T
    inv_half = (U / np.sqrt(w)) @ U.T
    lam, Z = eigh(1j * inv_half @ omega(M) @ inv_half)  # ascending: M negative, then M positive
    lam, Z = lam[M:][::-1], Z[:, M:][:, ::-1]  # +1/ν, largest first, so that ν ascends
    nu = 1.0 / lam
    # iA z = z/ν  =>  A u = w/ν, A w = −u/ν; the pair (u, −w) puts +1/ν in the upper-right block
    K = np.hstack([math.sqrt(2) * Z.real, -math.sqrt(2) * Z.imag])
    S = half @ K @ np.diag(1.0 / np.sqrt(np.concatenate([nu, nu])))
    groups = degenerate_groups(nu, rtol)
    c = S[:, :M] - 1j * S[:, M:]
    ref = _as_complex_modes(reference, M, 2 * M)
    for g in groups:
        c[:, g] = c[:, g] @ _procrustes(c[:, g], ref, "unitary")
    S = np.hstack([c.real, -c.imag])
    return Williamson(nu, np.asarray(S, dtype=float), groups)


class Takagi(NamedTuple):
    """``A = U diag(s) Uᵀ`` with ``U`` unitary and ``s >= 0`` descending."""

    s: FloatArray
    U: ComplexArray
    groups: list[IntArray]


def takagi(
    A: npt.ArrayLike, reference: npt.ArrayLike | None = None, rtol: float = DEGENERACY_RTOL, atol: float = ATOL
) -> Takagi:
    """Takagi (Autonne–Takagi) factorisation of a complex symmetric matrix: ``A = U diag(s) Uᵀ``.

    From the SVD ``A = W diag(s) V†``, symmetry gives ``A = conj(V) diag(s) Wᵀ``, so
    ``Z = W† conj(V)`` is unitary, symmetric and commutes with ``diag(s)`` (block diagonal on the
    degenerate groups of ``s``). Then ``U = W Z^{1/2}`` with the principal square root, itself
    symmetric, computed block by block.

    **Gauge.** Inside a group of equal ``s > 0`` the columns of ``U`` are defined up to a real
    orthogonal rotation ``O(k)``; for ``s = 0`` up to any unitary. The rotation is fixed by
    orthogonal Procrustes against ``reference`` (CONVENTIONS.md, "Decompositions").

    Parameters
    ----------
    A
        Complex symmetric ``n x n`` matrix.
    reference
        Complex ``n x n`` basis the groups are aligned with (``None``: the identity).
    rtol
        Relative tolerance under which singular values form one group.
    atol
        Tolerance of the symmetry check (relative to ``max(1, ‖A‖)``).

    Raises
    ------
    InputError
        ``A`` is not symmetric.
    """
    A_ = _square(A).astype(complex)
    scale = max(1.0, float(np.abs(A_).max(initial=0.0)))
    if np.abs(A_ - A_.T).max(initial=0.0) > atol * scale:
        raise InputError("Takagi factorisation needs a complex symmetric matrix (A = Aᵀ)")
    A_ = (A_ + A_.T) / 2
    W, s, Vh = svd(A_)
    n = A_.shape[0]
    groups = degenerate_groups(s, rtol)
    Z = W.conj().T @ Vh.T  # W† conj(V): V = Vh†, so conj(V) = Vhᵀ
    U = np.array(W, dtype=complex)
    ref = _as_complex_modes(reference, n, n)
    for g in groups:
        block = Z[np.ix_(g, g)]
        U[:, g] = W[:, g] @ _sqrtm_unitary(block)
        kind = "unitary" if s[g[0]] <= rtol * max(1.0, float(s[0])) else "orthogonal"
        U[:, g] = U[:, g] @ _procrustes(U[:, g], ref, kind)
    return Takagi(s, U, groups)


def _sqrtm_unitary(Z: np.ndarray) -> np.ndarray:
    """Principal square root of a unitary matrix (eigenvalues ``e^{iθ}``, ``θ ∈ (−π, π]``)."""
    if Z.shape == (1, 1):
        return np.sqrt(Z.astype(complex))
    T, Q = scipy.linalg.schur(Z.astype(complex), output="complex")  # Z normal: T diagonal
    d = np.diag(T)
    return np.asarray((Q * np.sqrt(d / np.abs(d))) @ Q.conj().T)


class BlochMessiah(NamedTuple):
    """``S = O1 diag(e^{−r}, e^{r}) O2`` with ``O1``, ``O2`` orthogonal symplectic and ``r >= 0`` descending."""

    O1: FloatArray
    r: FloatArray
    O2: FloatArray
    groups: list[IntArray]


def bloch_messiah(
    S: npt.ArrayLike, reference: npt.ArrayLike | None = None, rtol: float = DEGENERACY_RTOL, atol: float = ATOL
) -> BlochMessiah:
    """Bloch–Messiah (Euler) decomposition of a symplectic matrix.

    ``S = P O`` (left polar decomposition): ``O`` is orthogonal symplectic and ``P = (S Sᵀ)^{1/2}``
    symmetric positive symplectic. An eigenvector ``v`` of ``P`` with eigenvalue ``e^{−r} < 1``
    has the partner ``Ωᵀ v`` with eigenvalue ``e^{r}``: an orthonormal basis ``X`` of the
    eigenvalues below 1 and ``ΩᵀX`` give ``O1 = [X | ΩᵀX]``. The eigenvalue-1 subspace (``r = 0``)
    is invariant under Ω and is completed by a symplectic Gram–Schmidt (pairs ``w``, ``Ωᵀw``).
    Then ``O2 = O1ᵀ O``. Both factors are orthogonal symplectic to round-off; ``S`` is reproduced
    to round-off except that squeezing below ``rtol`` is reported as ``r = 0`` (an error of at most
    about ``rtol · ‖S‖``). ``r`` is sorted descending: the most squeezed mode first. The columns
    of ``O1`` are the *supermodes*: ``O1 = [[Re U, −Im U], [Im U, Re U]]`` with ``U[:, k]`` the
    complex mode function of supermode ``k`` in the input-mode basis.

    **Gauge.** Inside a group of equal ``r > 0`` the supermodes are defined up to a real orthogonal
    rotation ``O(k)``; for ``r = 0`` up to any passive ``U(k)``. The rotation is fixed by orthogonal
    Procrustes against ``reference`` (CONVENTIONS.md, "Decompositions"), so supermodes do not jump
    with the basis LAPACK returns; passing the previous ``O1`` keeps a parameter sweep continuous.

    Parameters
    ----------
    S
        Real symplectic ``2M x 2M`` matrix (xxpp).
    reference
        Basis the groups are aligned with: ``None`` (the input modes), a previous ``O1``, or a
        complex ``M x M`` matrix of mode functions (e.g. Hermite–Gauss modes).
    rtol
        Relative tolerance under which squeezing parameters form one group (also the threshold
        below which ``r`` counts as zero).
    atol
        Tolerance of the symplecticity check.

    Raises
    ------
    InputError
        ``S`` is not symplectic.
    """
    S_ = np.asarray(_square(S, "symplectic matrix"), dtype=float)
    n = S_.shape[0]
    if n % 2:
        raise ShapeError(f"symplectic matrix must be 2M x 2M, got shape {S_.shape}")
    M = n // 2
    Om = omega(M)
    scale = max(1.0, float(np.abs(S_).max()) ** 2)
    if np.abs(S_ @ Om @ S_.T - Om).max() > atol * scale:
        raise InputError("bloch_messiah needs a symplectic matrix (S Ω Sᵀ = Ω)")
    Opol, Pm = polar(S_, side="left")
    lam, E = eigh((Pm + Pm.T) / 2)  # ascending: e^{-r_1} <= ... <= 1 <= ... <= e^{r_1}
    r_all = -np.log(lam)
    below = np.flatnonzero(r_all > rtol)  # strictly squeezed directions (eigenvalue < 1)
    unit = np.flatnonzero(np.abs(r_all) <= rtol)
    # Symplectic Gram–Schmidt over [squeezed eigenvectors, eigenvalue-1 eigenvectors]: keeps the
    # squeezed ones (in order) and completes with a Lagrangian basis of the r = 0 subspace. It also
    # restores exact isotropy (Xᵀ Ω X = 0) when tiny r leave the eigenvectors of P poorly separated
    # from their partners (eigenvector error ~ eps / gap), so O1 is orthogonal symplectic to round-off.
    X = _lagrangian_basis(np.hstack([E[:, below], E[:, unit]]), Om, M)
    r = np.concatenate([r_all[below], np.zeros(M - below.size)])
    if X.shape[1] != M:  # pragma: no cover - a symplectic P always has paired spectra
        raise PhysicalityError("spectrum of (S Sᵀ)^{1/2} is not paired: S is not symplectic")
    groups = degenerate_groups(r, rtol)
    U = X[:M] + 1j * X[M:]  # supermode functions: O1 = [[Re U, −Im U], [Im U, Re U]]
    ref = _as_complex_modes(reference, M, M)
    for g in groups:
        kind = "unitary" if r[g[0]] <= rtol else "orthogonal"
        U[:, g] = U[:, g] @ _procrustes(U[:, g], ref, kind)
    X = np.vstack([U.real, U.imag])
    O1 = np.hstack([X, Om.T @ X])
    # O2 = O1ᵀ O: exactly orthogonal symplectic (a product of two such matrices). Then
    # O1 Z O2 = (O1 Z O1ᵀ) O reproduces S = P O to round-off, except for squeezing below ``rtol``,
    # which is reported as r = 0 (reconstruction error <= ~rtol · ‖S‖).
    O2 = O1.T @ Opol
    return BlochMessiah(O1, r, O2, groups)


def _lagrangian_basis(E: np.ndarray, Om: np.ndarray, k: int) -> np.ndarray:
    """``k`` orthonormal vectors ``w_j`` with ``w_iᵀ Ω w_j = 0``, taken from the candidates ``E`` in order.

    Symplectic Gram–Schmidt (twice, for numerical orthogonality): project each candidate off the
    span of the previous ``w`` and ``Ωᵀw``, keep it if what remains is not negligible, normalise;
    ``{w_j, Ωᵀ w_j}`` is then an orthonormal basis of the span.
    """
    basis: list[np.ndarray] = []
    for col in range(E.shape[1]):
        if len(basis) == k:
            break
        v = E[:, col].copy()
        for _ in range(2):  # "twice is enough" (Kahan): re-orthogonalise once
            for w in basis:
                v -= (w @ v) * w
                pw = Om.T @ w
                v -= (pw @ v) * pw
        norm = float(np.sqrt(v @ v))
        if norm > 1e-6:
            basis.append(v / norm)
    return np.column_stack(basis)


def nearest_physical_covariance(V: npt.ArrayLike, floor: float = VACUUM_VARIANCE) -> FloatArray:
    """A physical covariance obtained from ``V`` by *adding* the least noise in its symplectic basis.

    Write ``V = S diag(ν, ν) Sᵀ`` (Williamson) and raise every ``ν_k < 1/2`` to ``1/2``:
    ``V' = S diag(ν', ν') Sᵀ``. ``V' − V = S diag(ν' − ν) Sᵀ >= 0``: the correction is pure added
    noise, zero when ``V`` is already physical. (This is not the Frobenius-nearest physical
    matrix, which needs a semidefinite program.) A ``V`` that is not even positive definite is
    first made so by raising its eigenvalues to ``floor · 10⁻³``.

    Parameters
    ----------
    V
        Real symmetric ``2M x 2M`` matrix (e.g. a covariance estimated from noisy data).
    floor
        The minimal symplectic eigenvalue, ``1/2`` (vacuum) by default.
    """
    V_ = _covariance(V)
    w, U = eigh(V_)
    if w[0] <= 0:
        V_ = (U * np.clip(w, floor * 1e-3, None)) @ U.T
    nu, S, _ = williamson(V_)
    if nu[0] >= floor:
        return V_
    nu2 = np.maximum(nu, floor)
    out = S @ np.diag(np.concatenate([nu2, nu2])) @ S.T
    return np.asarray((out + out.T) / 2, dtype=float)
