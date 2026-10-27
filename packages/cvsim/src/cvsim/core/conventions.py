"""Phase-space conventions, as code.

These are the rules of ``CONVENTIONS.md`` ("Continuous variables"), identical to qttlind's
(decision D4). Every rule marked **[tested]** there is implemented a second time, independently,
by the dense Fock-space oracle :mod:`cvsim.reference.fock_dense` (gates as matrix exponentials of
their generators) and the two are compared by ``tests/test_conventions.py``; the contract tests
compare them with qttlind as well.

* ħ = 1; ``x = (a + a†)/√2``, ``p = (a − a†)/(i√2)``, ``[x, p] = i``; vacuum covariance ``I/2``.
* Quadrature vector ``r = (x_1..x_M, p_1..p_M)`` ("xxpp"); covariance ``V_ij = ⟨{Δr_i, Δr_j}⟩/2``;
  ``[r_i, r_j] = i Ω_ij`` with ``Ω = [[0, I], [−I, 0]]``.
* A gate ``U`` with ``U† a U = A a + B a†`` acts as ``μ → S μ``, ``V → S V Sᵀ`` with
  ``S = [[Re(A+B), −Im(A−B)], [Im(A+B), Re(A−B)]]`` (:func:`symplectic_from_bogoliubov`).
* Gate generators are Strawberry Fields' (so published circuits translate), with the Fourier gate
  of the MBQC literature: ``F = R(π/2)``, ``F† x F = −p`` (qttlind's ``F`` is its inverse).

Example
-------
>>> import numpy as np
>>> from cvsim.core import conventions as cv
>>> np.diag(cv.squeezing(1.0)).round(4)      # φ = 0 squeezes x: diag(e^-r, e^r)
array([0.3679, 2.7183])
>>> round(float(cv.db_from_r(1.0)), 3)        # dB per neper
8.686
>>> cv.homodyne_vector(np.pi / 2).round(12)   # x_φ at φ = π/2 is p
array([0., 1.])
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..errors import InputError, ShapeError
from ._types import ComplexArray, FloatArray

__all__ = [
    "ATOL",
    "DB_PER_NEPER",
    "HBAR",
    "VACUUM_VARIANCE",
    "beamsplitter",
    "complex_basis",
    "convert_hbar",
    "cx",
    "cz",
    "db_from_r",
    "db_from_variance",
    "displacement_vector",
    "embed",
    "fourier",
    "from_snu",
    "from_xpxp",
    "homodyne_vector",
    "interferometer",
    "is_symplectic",
    "loss_xy",
    "mode_indices",
    "omega",
    "r_from_db",
    "rotation",
    "shear",
    "squeezing",
    "symplectic_from_bogoliubov",
    "to_snu",
    "to_xpxp",
    "two_mode_squeezing",
    "variance_from_db",
    "xpxp_permutation",
]

#: Reduced Planck constant of the internal unit system.
HBAR = 1.0
#: Variance of each quadrature of the vacuum (``ħ/2``).
VACUUM_VARIANCE = 0.5
#: Default absolute tolerance for physicality and symplecticity checks (covariances are O(1)).
ATOL = 1e-10
#: Decibels per neper of squeezing: ``20 log10(e)``; ``r = 1`` is 8.686 dB.
DB_PER_NEPER = 20.0 * math.log10(math.e)


# ------------------------------------------------------------------------------- validation helpers
def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise InputError(f"{name} must be a finite real number, got {value!r}")
    return v


def _num_modes(M: int) -> int:
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M < 1:
        raise InputError(f"number of modes must be an integer >= 1, got {M!r}")
    return int(M)


# ------------------------------------------------------------------------------- ordering
def omega(M: int) -> FloatArray:
    """The symplectic form ``[[0, I_M], [−I_M, 0]]`` (xxpp ordering).

    Parameters
    ----------
    M
        Number of modes.
    """
    M = _num_modes(M)
    z = np.zeros((M, M))
    eye = np.eye(M)
    return np.block([[z, eye], [-eye, z]])


def mode_indices(modes: Sequence[int], M: int) -> list[int]:
    """Quadrature indices ``[x_k..., p_k...]`` of ``modes`` in an ``M``-mode xxpp vector.

    Parameters
    ----------
    modes
        Distinct mode numbers in ``0..M-1``.
    M
        Total number of modes.

    Raises
    ------
    InputError
        Empty, repeated or out-of-range modes.
    """
    M = _num_modes(M)
    modes = [int(k) for k in modes]
    if not modes or len(set(modes)) != len(modes) or any(not 0 <= k < M for k in modes):
        raise InputError(f"modes must be distinct integers in 0..{M - 1}, got {modes}")
    return modes + [M + k for k in modes]


def xpxp_permutation(M: int) -> list[int]:
    """Index list ``P`` such that ``r_xpxp = r_xxpp[P]`` (``(x1, p1, x2, p2, ...)``).

    Parameters
    ----------
    M
        Number of modes.
    """
    M = _num_modes(M)
    return [k // 2 + (M if k % 2 else 0) for k in range(2 * M)]


def _check_moments(mean: npt.ArrayLike, cov: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    mu = np.asarray(mean, dtype=float).reshape(-1)
    V = np.asarray(cov, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2 or V.shape[0] == 0:
        raise ShapeError(f"covariance must be a non-empty 2M x 2M array, got shape {V.shape}")
    if mu.shape[0] != V.shape[0]:
        raise ShapeError(f"mean has length {mu.shape[0]} but the covariance is {V.shape[0]} x {V.shape[0]}")
    return mu, V


def to_xpxp(mean: npt.ArrayLike, cov: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Reorder xxpp moments to xpxp ``(x1, p1, x2, p2, ...)``.

    Parameters
    ----------
    mean, cov
        Mean vector ``(2M,)`` and covariance ``(2M, 2M)`` in xxpp order.
    """
    mu, V = _check_moments(mean, cov)
    P = xpxp_permutation(mu.shape[0] // 2)
    return mu[P], V[np.ix_(P, P)]


def from_xpxp(mean: npt.ArrayLike, cov: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Reorder xpxp moments to xxpp (inverse of :func:`to_xpxp`).

    Parameters
    ----------
    mean, cov
        Mean vector ``(2M,)`` and covariance ``(2M, 2M)`` in xpxp order.
    """
    mu, V = _check_moments(mean, cov)
    inv = np.argsort(xpxp_permutation(mu.shape[0] // 2))
    return mu[inv], V[np.ix_(inv, inv)]


def complex_basis(M: int) -> ComplexArray:
    """Unitary ``W`` with ``(a_1..a_M, a†_1..a†_M) = W r``: ``W = [[I, iI], [I, −iI]]/√2``.

    Parameters
    ----------
    M
        Number of modes.
    """
    M = _num_modes(M)
    eye = np.eye(M)
    return np.block([[eye, 1j * eye], [eye, -1j * eye]]).astype(np.complex128) / math.sqrt(2)


# ------------------------------------------------------------------------------- units
def convert_hbar(
    mean: npt.ArrayLike, cov: npt.ArrayLike, hbar_from: float, hbar_to: float
) -> tuple[FloatArray, FloatArray]:
    """Rescale moments between conventions ``x = √(ħ/2)(a + a†)`` of different ħ.

    The vacuum variance is ``ħ/2``: the covariance scales by ``ħ_to/ħ_from``, the mean by its
    square root. Strawberry Fields and The Walrus use ``ħ = 2`` (vacuum ``V = I``).

    Parameters
    ----------
    mean, cov
        Moments in the ``hbar_from`` convention.
    hbar_from, hbar_to
        Positive values of ħ.
    """
    mu, V = _check_moments(mean, cov)
    a, b = _finite("hbar_from", hbar_from), _finite("hbar_to", hbar_to)
    if a <= 0 or b <= 0:
        raise InputError(f"hbar must be > 0, got hbar_from={hbar_from}, hbar_to={hbar_to}")
    return mu * math.sqrt(b / a), V * (b / a)


def to_snu(variance: npt.ArrayLike) -> FloatArray:
    """Variances in shot-noise units (vacuum = 1): ``V / (ħ/2)``.

    Parameters
    ----------
    variance
        Variance(s) or covariance matrix in the internal ħ = 1 units.
    """
    return np.asarray(variance, dtype=float) / VACUUM_VARIANCE


def from_snu(variance: npt.ArrayLike) -> FloatArray:
    """Inverse of :func:`to_snu`.

    Parameters
    ----------
    variance
        Variance(s) in shot-noise units.
    """
    return np.asarray(variance, dtype=float) * VACUUM_VARIANCE


def db_from_variance(variance: npt.ArrayLike) -> FloatArray:
    """Noise level relative to the vacuum in dB: ``10 log10(V / (ħ/2))``.

    Negative values are squeezing (``−3 dB`` = half the vacuum noise), positive values excess noise
    (anti-squeezing).

    Parameters
    ----------
    variance
        Positive quadrature variance(s), ħ = 1 units.
    """
    v = np.asarray(variance, dtype=float)
    if np.any(~(v > 0)):
        raise InputError(f"variances must be > 0 to be expressed in dB, got {variance!r}")
    return 10.0 * np.log10(v / VACUUM_VARIANCE)


def variance_from_db(db: npt.ArrayLike) -> FloatArray:
    """Inverse of :func:`db_from_variance`.

    Parameters
    ----------
    db
        Noise level(s) relative to the vacuum, in dB.
    """
    return VACUUM_VARIANCE * 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def db_from_r(r: npt.ArrayLike) -> FloatArray:
    """Squeezing in dB of a pure squeezed vacuum with parameter ``r``: ``20 log10(e) r``.

    Parameters
    ----------
    r
        Squeezing parameter(s) (nepers).
    """
    return DB_PER_NEPER * np.asarray(r, dtype=float)


def r_from_db(db: npt.ArrayLike) -> FloatArray:
    """Inverse of :func:`db_from_r`.

    Parameters
    ----------
    db
        Squeezing in dB (positive numbers).
    """
    return np.asarray(db, dtype=float) / DB_PER_NEPER


# ------------------------------------------------------------------------------- symplectic algebra
def symplectic_from_bogoliubov(A: npt.ArrayLike, B: npt.ArrayLike) -> FloatArray:
    """Real symplectic matrix of the mode transformation ``U† a U = A a + B a†``.

    For mean amplitudes ``α = ⟨a⟩``: ``α → A α + B ᾱ``; with ``α = (x + ip)/√2`` (xxpp)::

        S = [[Re(A + B), −Im(A − B)],
             [Im(A + B),  Re(A − B)]]

    Parameters
    ----------
    A, B
        ``M x M`` complex matrices (scalars for one mode).

    Raises
    ------
    ShapeError
        ``A`` and ``B`` are not square matrices of the same shape.
    """
    A_ = np.atleast_2d(np.asarray(A, dtype=complex))
    B_ = np.atleast_2d(np.asarray(B, dtype=complex))
    if A_.ndim != 2 or A_.shape != B_.shape or A_.shape[0] != A_.shape[1]:
        raise ShapeError(f"A and B must be square and of the same shape, got {A_.shape} and {B_.shape}")
    return np.block([[(A_ + B_).real, -(A_ - B_).imag], [(A_ + B_).imag, (A_ - B_).real]])


def is_symplectic(S: npt.ArrayLike, atol: float = ATOL) -> bool:
    """Whether ``S Ω Sᵀ = Ω`` within ``atol``.

    Parameters
    ----------
    S
        A real ``2M x 2M`` matrix.
    atol
        Absolute tolerance on every entry.
    """
    S_ = np.asarray(S, dtype=float)
    if S_.ndim != 2 or S_.shape[0] != S_.shape[1] or S_.shape[0] % 2 or S_.shape[0] == 0:
        return False
    W = omega(S_.shape[0] // 2)
    return bool(np.allclose(S_ @ W @ S_.T, W, rtol=0.0, atol=atol))


def embed(S: npt.ArrayLike, modes: Sequence[int], M: int) -> FloatArray:
    """Extend a ``2m x 2m`` matrix acting on ``modes`` to all ``M`` modes (identity elsewhere).

    Parameters
    ----------
    S
        Matrix acting on ``len(modes)`` modes, xxpp order of those modes.
    modes
        The modes it acts on, in the order of its rows.
    M
        Total number of modes.

    Raises
    ------
    ShapeError
        ``S`` does not act on ``len(modes)`` modes.
    """
    idx = mode_indices(modes, M)
    S_ = np.asarray(S, dtype=float)
    if S_.shape != (len(idx), len(idx)):
        raise ShapeError(f"matrix of shape {S_.shape} does not act on {len(modes)} mode(s)")
    full = np.eye(2 * M)
    full[np.ix_(idx, idx)] = S_
    return full


# ------------------------------------------------------------------------------- gates
def displacement_vector(alpha: complex | Sequence[complex]) -> FloatArray:
    """Mean shift of ``D(α) = exp(α a† − ᾱ a)`` on each mode: ``√2 (Re α, Im α)`` (xxpp).

    Parameters
    ----------
    alpha
        One complex amplitude per mode.
    """
    a = np.atleast_1d(np.asarray(alpha, dtype=complex))
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise InputError(f"alpha must be finite complex number(s) in a 1-D sequence, got {alpha!r}")
    return math.sqrt(2) * np.concatenate([a.real, a.imag])


def rotation(theta: float) -> FloatArray:
    """``R(θ) = exp(iθn)``: ``U† a U = e^{iθ} a``; the mean turns counter-clockwise by ``θ``.

    Parameters
    ----------
    theta
        Rotation angle (radians).
    """
    th = _finite("theta", theta)
    return symplectic_from_bogoliubov(np.exp(1j * th), 0.0)


def squeezing(r: float, phi: float = 0.0) -> FloatArray:
    """``S(r, φ) = exp((z̄a² − z a†²)/2)``, ``z = r e^{iφ}``: ``U† a U = cosh r a − e^{iφ} sinh r a†``.

    ``φ = 0`` gives ``diag(e^{−r}, e^{r})`` (x is squeezed); in general the squeezed quadrature
    is ``x_{φ/2} = cos(φ/2) x + sin(φ/2) p``.

    Parameters
    ----------
    r
        Squeezing parameter (nepers; negative values swap the axes).
    phi
        Squeezing phase.
    """
    r_, ph = _finite("r", r), _finite("phi", phi)
    return symplectic_from_bogoliubov(math.cosh(r_), -np.exp(1j * ph) * math.sinh(r_))


def beamsplitter(theta: float, phi: float = 0.0) -> FloatArray:
    """``BS(θ, φ) = exp(θ(e^{iφ} a b† − e^{−iφ} a† b))`` on modes ``(a, b)``; transmissivity ``cos²θ``.

    ``U† a U = cos θ a − e^{−iφ} sin θ b`` and ``U† b U = e^{iφ} sin θ a + cos θ b``.

    Parameters
    ----------
    theta
        Mixing angle.
    phi
        Phase.
    """
    th, ph = _finite("theta", theta), _finite("phi", phi)
    t, s = math.cos(th), math.sin(th)
    U = np.array([[t, -np.exp(-1j * ph) * s], [np.exp(1j * ph) * s, t]])
    return symplectic_from_bogoliubov(U, np.zeros((2, 2)))


def two_mode_squeezing(r: float, phi: float = 0.0) -> FloatArray:
    """``S2(r, φ) = exp(z̄ ab − z a†b†)``: ``U† a U = cosh r a − e^{iφ} sinh r b†`` (and ``a ↔ b``).

    Parameters
    ----------
    r
        Squeezing parameter.
    phi
        Phase.
    """
    r_, ph = _finite("r", r), _finite("phi", phi)
    A = math.cosh(r_) * np.eye(2)
    B = -np.exp(1j * ph) * math.sinh(r_) * np.array([[0.0, 1.0], [1.0, 0.0]])
    return symplectic_from_bogoliubov(A, B)


def interferometer(U: npt.ArrayLike, atol: float = ATOL) -> FloatArray:
    """Passive linear optics ``U† a_j U = Σ_k U_jk a_k`` (row ``j`` = output mode ``j``).

    Parameters
    ----------
    U
        ``M x M`` unitary matrix.
    atol
        Tolerance of the unitarity check.

    Raises
    ------
    InputError
        ``U`` is not square and unitary.
    """
    U_ = np.atleast_2d(np.asarray(U, dtype=complex))
    if (
        U_.ndim != 2
        or U_.shape[0] != U_.shape[1]
        or not np.allclose(U_.conj().T @ U_, np.eye(U_.shape[0]), rtol=0.0, atol=atol)
    ):
        raise InputError(f"interferometer matrix must be square and unitary, got shape {U_.shape}")
    return symplectic_from_bogoliubov(U_, np.zeros_like(U_))


def cz(A: npt.ArrayLike) -> FloatArray:
    """``CZ(A) = exp(i/2 Σ_jk A_jk x_j x_k)``: ``p → p + A x``, ``S = [[I, 0], [A, I]]``.

    For two modes, ``A = [[0, g], [g, 0]]`` is ``exp(i g x_a x_b)``. A graph with weighted
    adjacency matrix ``A`` gives the nullifiers ``p − A x``.

    Parameters
    ----------
    A
        Real symmetric ``M x M`` matrix (diagonal entries are local shears).

    Raises
    ------
    InputError
        ``A`` is not real, square and symmetric.
    """
    A_ = np.atleast_2d(np.asarray(A))
    if np.iscomplexobj(A_) or A_.ndim != 2 or A_.shape[0] != A_.shape[1] or not np.allclose(A_, A_.T, atol=ATOL):
        raise InputError(f"CZ needs a real symmetric square matrix, got shape {A_.shape}")
    A_ = A_.astype(float)
    if not np.all(np.isfinite(A_)):
        raise InputError("CZ weights must be finite")
    M = A_.shape[0]
    eye, z = np.eye(M), np.zeros((M, M))
    return np.block([[eye, z], [A_, eye]])


def cx(g: float) -> FloatArray:
    """``CX(g) = exp(−i g x_a p_b)`` on modes ``(a, b)``: ``x_b → x_b + g x_a``, ``p_a → p_a − g p_b``.

    Parameters
    ----------
    g
        Gain.
    """
    g_ = _finite("g", g)
    S = np.eye(4)
    S[1, 0] = g_  # x_b += g x_a
    S[2, 3] = -g_  # p_a -= g p_b
    return S


def shear(s: float) -> FloatArray:
    """``P(s) = exp(i s x²/2)``: ``p → p + s x``, ``S = [[1, 0], [s, 1]]``.

    Parameters
    ----------
    s
        Shear strength.
    """
    s_ = _finite("s", s)
    return np.array([[1.0, 0.0], [s_, 1.0]])


def fourier() -> FloatArray:
    """``F = R(π/2)`` (MBQC convention): ``F† x F = −p``, ``F† p F = x``; qttlind's ``F`` is ``F†``."""
    return np.array([[0.0, -1.0], [1.0, 0.0]])


# ------------------------------------------------------------------------------- measurements and channels
def homodyne_vector(phi: float) -> FloatArray:
    """``u`` such that the homodyne quadrature is ``x_φ = u · (x, p) = cos φ x + sin φ p``.

    Measuring ``x_φ`` is equivalent to applying ``R(−φ)`` and measuring ``x``.

    Parameters
    ----------
    phi
        Local-oscillator phase.
    """
    ph = _finite("phi", phi)
    return np.array([math.cos(ph), math.sin(ph)])


def loss_xy(eta: float, nbar: float = 0.0) -> tuple[FloatArray, FloatArray]:
    """``(X, Y)`` of a beam splitter of transmissivity ``η`` with a thermal environment of ``n̄`` photons.

    ``X = √η I``, ``Y = (1 − η)(n̄ + ½) I``; the Lindblad equation with ``L = √κ a`` during ``t`` is
    this channel with ``η = e^{−κt}`` (and ``n̄ = 0``).

    Parameters
    ----------
    eta
        Transmissivity in ``[0, 1]``.
    nbar
        Mean thermal photon number of the environment, ``>= 0``.

    Raises
    ------
    InputError
        ``eta`` outside ``[0, 1]`` or ``nbar < 0``.
    """
    e, n = _finite("eta", eta), _finite("nbar", nbar)
    if not 0.0 <= e <= 1.0:
        raise InputError(f"transmissivity must be in [0, 1], got {eta}")
    if n < 0.0:
        raise InputError(f"nbar must be >= 0, got {nbar}")
    return math.sqrt(e) * np.eye(2), (1.0 - e) * (n + VACUUM_VARIANCE) * np.eye(2)
