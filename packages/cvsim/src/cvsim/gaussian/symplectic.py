"""The symplectic library: gate matrices, quadratic Hamiltonians, interferometers and their meshes,
state preparation, mode-basis changes.

Every gate matrix is the one of CONVENTIONS.md ("Gates"), defined once in
:mod:`cvsim.core.conventions` and re-exported here; this module adds what is built from them.

* :func:`from_hamiltonian` — ``U = exp(−i t H)``, ``H = ½ rᵀ H r``: ``S = exp(t Ω H)`` (Heisenberg
  equation ``ṙ = Ω H r``). With ``H = −I`` this is the rotation ``R(t)``.
* :func:`clements` — any ``N x N`` unitary as ``N(N−1)/2`` Mach–Zehnder cells in a rectangular mesh
  followed by output phases (Clements et al., Optica 3, 1460, 2016). A cell on modes ``(m, m+1)``
  is ``T(θ, φ) = BS(θ, 0) · R_m(φ)``, matrix ``[[e^{iφ} cos θ, −sin θ], [e^{iφ} sin θ, cos θ]]``;
  :meth:`Clements.gates` lists the physical gates in time order.
* :func:`preparation` — any Gaussian state as ``D(d) · I(U₁) · Sq(r) · I(U₂)`` applied to thermal
  modes ``ν`` (Williamson, then Bloch–Messiah of its symplectic factor); ``U₂`` is the identity
  for a pure state (it acts on the vacuum).
* :func:`change_basis` — the state in another mode basis (supermodes, pulse shapes): new mode
  ``k`` is ``b_k = Σ_j conj(U_jk) a_j``, i.e. column ``k`` of ``U`` is its mode function in the old
  basis; moments transform with ``S(U)ᵀ``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core import linalg as la
from ..core._types import ComplexArray, FloatArray
from ..core.conventions import (
    beamsplitter,
    cx,
    cz,
    displacement_vector,
    embed,
    fourier,
    interferometer,
    is_symplectic,
    rotation,
    shear,
    squeezing,
    two_mode_squeezing,
)
from ..errors import InputError, ShapeError
from .states import GaussianState

__all__ = [
    "Clements",
    "Preparation",
    "beamsplitter",
    "change_basis",
    "clements",
    "cx",
    "cz",
    "displacement_vector",
    "embed",
    "fourier",
    "from_hamiltonian",
    "interferometer",
    "is_passive",
    "is_symplectic",
    "mzi",
    "preparation",
    "random_interferometer",
    "random_symplectic",
    "rotation",
    "shear",
    "squeezing",
    "two_mode_squeezing",
    "unitary_from_passive",
]


# ------------------------------------------------------------------------------- Hamiltonians
def from_hamiltonian(H: npt.ArrayLike, t: float = 1.0) -> FloatArray:
    """Symplectic matrix of ``exp(−i t H)`` for the quadratic Hamiltonian ``H = ½ rᵀ H r``.

    Parameters
    ----------
    H
        Real symmetric ``2M x 2M`` matrix (xxpp).
    t
        Evolution time.

    Returns
    -------
    ndarray
        ``S = expm(t Ω H)``.
    """
    H_ = np.asarray(H, dtype=float)
    if H_.ndim != 2 or H_.shape[0] != H_.shape[1] or H_.shape[0] % 2 or H_.shape[0] == 0:
        raise ShapeError(f"H must be 2M x 2M, got shape {H_.shape}")
    if not np.all(np.isfinite(H_)) or not math.isfinite(t):
        raise InputError("H and t must be finite")
    if np.abs(H_ - H_.T).max() > cv.ATOL * max(1.0, float(np.abs(H_).max())):
        raise InputError("H must be symmetric")
    M = H_.shape[0] // 2
    return np.asarray(la.expm(t * cv.omega(M) @ ((H_ + H_.T) / 2)).real, dtype=float)


# ------------------------------------------------------------------------------- passive maps
def is_passive(S: npt.ArrayLike, atol: float = cv.ATOL) -> bool:
    """``S`` is orthogonal symplectic (photon-number preserving), i.e. an interferometer.

    Parameters
    ----------
    S
        Real ``2M x 2M`` matrix.
    atol
        Tolerance on ``SᵀS = I`` and symplecticity.
    """
    S_ = np.asarray(S, dtype=float)
    if S_.ndim != 2 or S_.shape[0] != S_.shape[1] or S_.shape[0] % 2:
        return False
    return bool(np.allclose(S_.T @ S_, np.eye(S_.shape[0]), atol=atol, rtol=0)) and is_symplectic(S_, atol)


def unitary_from_passive(S: npt.ArrayLike, atol: float = cv.ATOL) -> ComplexArray:
    """The unitary ``U`` with ``interferometer(U) = S``: ``U = A + iB`` for ``S = [[A, −B], [B, A]]``.

    Parameters
    ----------
    S
        Orthogonal symplectic ``2M x 2M`` matrix.
    atol
        Tolerance of the passivity check.

    Raises
    ------
    InputError
        ``S`` is not passive.
    """
    S_ = np.asarray(S, dtype=float)
    if not is_passive(S_, atol):
        raise InputError("S is not passive (orthogonal symplectic)")
    M = S_.shape[0] // 2
    return np.asarray(S_[:M, :M] + 1j * S_[M:, :M], dtype=complex)


def random_interferometer(rng: np.random.Generator, M: int) -> ComplexArray:
    """Haar-random ``M x M`` unitary (QR of a complex Ginibre matrix, phases fixed by ``diag R > 0``).

    Parameters
    ----------
    rng
        Random generator.
    M
        Number of modes.
    """
    if M < 1:
        raise InputError(f"M must be >= 1, got {M}")
    Z = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
    Q, R = la.qr(Z)
    d = np.diag(R)
    return np.asarray(Q * (d / np.abs(d)), dtype=complex)


def random_symplectic(rng: np.random.Generator, M: int, r_max: float = 1.0) -> FloatArray:
    """Random symplectic matrix ``I(U₁) · diag(e^{−r}, e^{r}) · I(U₂)``, Haar ``U``, ``r ~ U[0, r_max]``.

    Parameters
    ----------
    rng
        Random generator.
    M
        Number of modes.
    r_max
        Largest squeezing parameter.
    """
    r = rng.uniform(0.0, r_max, M)
    Z = np.diag(np.exp(np.concatenate([-r, r])))
    S = interferometer(random_interferometer(rng, M)) @ Z @ interferometer(random_interferometer(rng, M))
    return np.asarray(S, dtype=float)


# ------------------------------------------------------------------------------- Clements mesh
def mzi(theta: float, phi: float) -> ComplexArray:
    """Mesh cell ``T(θ, φ) = BS(θ, 0) R_m(φ)`` on modes ``(m, m+1)``: ``[[e^{iφ}c, −s], [e^{iφ}s, c]]``.

    Parameters
    ----------
    theta
        Beam-splitter angle (transmissivity ``cos² θ``).
    phi
        Phase on the first mode, applied before the beam splitter.
    """
    c, s, e = math.cos(theta), math.sin(theta), complex(math.cos(phi), math.sin(phi))
    return np.array([[e * c, -s], [e * s, c]], dtype=complex)


class Clements(NamedTuple):
    """A rectangular mesh: ``U = diag(phases) · T_K ⋯ T_1``.

    ``cells[k] = (m, θ, φ)`` is the cell on modes ``(m, m+1)``, in time order.
    """

    cells: list[tuple[int, float, float]]
    phases: ComplexArray

    @property
    def num_modes(self) -> int:
        """Number of modes ``N``."""
        return int(self.phases.shape[0])

    def unitary(self) -> ComplexArray:
        """The unitary the mesh implements."""
        N = self.num_modes
        U = np.eye(N, dtype=complex)
        for m, theta, phi in self.cells:
            U[[m, m + 1], :] = mzi(theta, phi) @ U[[m, m + 1], :]
        return np.asarray(self.phases[:, None] * U, dtype=complex)

    def depth(self) -> int:
        """Number of layers when cells on disjoint mode pairs run in parallel (``N`` for a full mesh)."""
        busy = np.zeros(self.num_modes, dtype=int)
        for m, _, _ in self.cells:
            busy[m] = busy[m + 1] = max(busy[m], busy[m + 1]) + 1
        return int(busy.max(initial=0))

    def gates(self) -> list[tuple[str, tuple[float, ...], tuple[int, ...]]]:
        """Physical gates in time order: ``("R", (φ,), (m,))``, ``("BS", (θ, 0.0), (m, m+1))`` per cell,
        then ``("R", (arg phase_k,), (k,))``; gate parameters follow CONVENTIONS.md."""
        out: list[tuple[str, tuple[float, ...], tuple[int, ...]]] = []
        for m, theta, phi in self.cells:
            out.append(("R", (phi,), (m,)))
            out.append(("BS", (theta, 0.0), (m, m + 1)))
        out.extend(("R", (float(np.angle(p)),), (k,)) for k, p in enumerate(self.phases))
        return out


def _null_angles(keep: complex, kill: complex, sign: float) -> tuple[float, float]:
    """``θ = atan2(|kill|, |keep|)`` and the phase ``arg(sign · kill / keep)`` (0 when undefined)."""
    theta = math.atan2(abs(kill), abs(keep))
    phi = float(np.angle(sign * kill * np.conj(keep))) if abs(kill) > 0 and abs(keep) > 0 else 0.0
    return theta, phi


def clements(U: npt.ArrayLike, atol: float = 1e-10) -> Clements:
    """Decompose a unitary into a rectangular Mach–Zehnder mesh (Clements et al. 2016).

    Elements are nulled alternately by cells applied on the right (columns) and on the left (rows)
    of ``U``; the left cells are then moved through the diagonal using
    ``T(θ, φ)† D = D' T(θ, arg(−d₁/d₂))``, which leaves ``N(N−1)/2`` cells in a mesh of depth
    ``N`` and one output phase per mode.

    Parameters
    ----------
    U
        ``N x N`` unitary.
    atol
        Tolerance of the unitarity check.

    Raises
    ------
    InputError
        ``U`` is not unitary.
    """
    V = np.array(U, dtype=complex)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] == 0:
        raise ShapeError(f"U must be a non-empty square matrix, got shape {V.shape}")
    N = V.shape[0]
    if not np.allclose(V.conj().T @ V, np.eye(N), atol=atol, rtol=0):
        raise InputError("U is not unitary")
    right: list[tuple[int, float, float]] = []  # V ← V T†, nulls V[row, m]
    left: list[tuple[int, float, float]] = []  # V ← T V, nulls V[m+1, col]
    for i in range(N - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                row, m = N - 1 - j, i - j
                # (V T†)[row, m] = e^{−iφ} c V[row, m] − s V[row, m+1] = 0
                theta, phi = _null_angles(V[row, m + 1], V[row, m], 1.0)
                V[:, [m, m + 1]] = V[:, [m, m + 1]] @ mzi(theta, phi).conj().T
                right.append((m, theta, phi))
        else:
            for j in range(1, i + 2):
                m, col = N + j - i - 3, j - 1
                # (T V)[m+1, col] = e^{iφ} s V[m, col] + c V[m+1, col] = 0
                theta, phi = _null_angles(V[m, col], V[m + 1, col], -1.0)
                V[[m, m + 1], :] = mzi(theta, phi) @ V[[m, m + 1], :]
                left.append((m, theta, phi))
    # V = L U R with L = T_k ⋯ T_1 (left cells), R = T†_1 T†_2 ⋯ (right cells), V = D diagonal.
    # U = L⁻¹ D R⁻¹ = T†_1 ⋯ T†_k D · T_r ⋯ T_1: move each T†_j through D, rightmost first.
    d = np.diag(V).copy()
    moved: list[tuple[int, float, float]] = []
    for m, theta, phi in reversed(left):
        d1, d2 = d[m], d[m + 1]
        moved.append((m, theta, float(np.angle(-d1 / d2))))
        d[m], d[m + 1] = -np.exp(-1j * phi) * d2, d2
    # U = D' · T'_1 ⋯ T'_k · T_r ⋯ T_1. Time order: right cells T_1 … T_r, then T'_k … T'_1 (= moved).
    cells = right + moved
    return Clements(cells, np.asarray(d, dtype=complex))


# ------------------------------------------------------------------------------- preparation
class Preparation(NamedTuple):
    """``state = D(d) · I(U1) · Sq(r) · I(U2) · ⊗_k thermal(ν_k)`` (``Sq`` squeezes ``x`` of each mode)."""

    nu: FloatArray
    U2: ComplexArray
    r: FloatArray
    U1: ComplexArray
    mean: FloatArray

    def symplectic(self) -> FloatArray:
        """``S = I(U1) · diag(e^{−r}, e^{r}) · I(U2)``."""
        Z = np.diag(np.exp(np.concatenate([-self.r, self.r])))
        return np.asarray(interferometer(self.U1) @ Z @ interferometer(self.U2), dtype=float)

    def state(self) -> GaussianState:
        """The state the preparation produces."""
        return GaussianState.from_symplectic(self.symplectic(), nu=self.nu, mean=self.mean)


def preparation(state: GaussianState, pure_tol: float = 1e-12) -> Preparation:
    """Recipe preparing ``state`` from thermal modes: Williamson ``V = S diag(ν, ν) Sᵀ``, then
    Bloch–Messiah ``S = O1 diag(e^{−r}, e^{r}) O2``.

    Parameters
    ----------
    state
        Target state.
    pure_tol
        If every ``ν − ½`` is below this, ``ν = ½`` exactly and ``U2 = I`` (it acts on the vacuum).
    """
    w = la.williamson(state.cov)
    bm = la.bloch_messiah(w.S)
    M = state.num_modes
    nu = np.asarray(w.nu, dtype=float)
    U2 = unitary_from_passive(bm.O2, atol=1e-8)
    if np.all(nu - cv.VACUUM_VARIANCE < pure_tol * max(1.0, float(np.abs(w.S).max()) ** 2)):
        nu = np.full(M, cv.VACUUM_VARIANCE)
        U2 = np.eye(M, dtype=complex)
    return Preparation(nu, U2, np.asarray(bm.r, dtype=float), unitary_from_passive(bm.O1, atol=1e-8), state.mean)


# ------------------------------------------------------------------------------- mode bases
def change_basis(state: GaussianState, U: npt.ArrayLike, modes: Sequence[int] | None = None) -> GaussianState:
    """The same state described in another mode basis.

    New mode ``k`` is ``b_k = Σ_j conj(U_jk) a_j``: column ``k`` of ``U`` is its mode function in
    the old basis (e.g. supermodes ``BlochMessiah.O1``'s ``U``, or Hermite–Gauss pulse shapes).
    Moments transform as ``d' = S(U)ᵀ d``, ``V' = S(U)ᵀ V S(U)``.

    Parameters
    ----------
    state
        State in the old basis.
    U
        Unitary whose columns are the new modes (on ``modes``).
    modes
        Old modes spanned by the new ones (default all); the others are unchanged.
    """
    U_ = np.asarray(U, dtype=complex)
    M = state.num_modes
    sub = list(range(M)) if modes is None else list(modes)
    S = embed(interferometer(U_), sub, M)
    return GaussianState(S.T @ state.mean, S.T @ state.cov @ S, validate=False)
