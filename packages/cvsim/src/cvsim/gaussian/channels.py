"""Gaussian channels ``d → X d``, ``V → X V Xᵀ + Y`` (CONVENTIONS.md, "Channels").

:class:`GaussianChannel` holds ``(X, Y)`` on ``m`` modes, checks complete positivity
``Y + (i/2)(Ω − X Ω Xᵀ) >= 0`` and composes (``a.then(b)``: ``a`` first, then ``b``:
``X = X_b X_a``, ``Y = X_b Y_a X_bᵀ + Y_b``). Constructors:

* ``loss(η, n̄)``: beam splitter of transmissivity ``η`` with a thermal environment,
  ``X = √η I``, ``Y = (1 − η)(n̄ + ½) I``;
* ``amplifier(G, n̄)``: phase-insensitive, ``X = √G I``, ``Y = (G − 1)(n̄ + ½) I``;
* ``additive_noise(Σ)``: random displacements of covariance ``Σ`` (``X = I``, ``Y = Σ``);
* ``from_transfer(T)``: lossy linear optics ``a → T a + …`` with ``‖T‖ <= 1`` and vacuum
  environment, ``X = S(T)``, ``Y = ½(I − X Xᵀ)`` (The Walrus' ``passive_transformation``);
* ``from_ancilla(S, ancilla)``: a symplectic ``S`` on system ⊕ ancilla, ancilla in a Gaussian
  state, traced out: ``X = S_ss``, ``Y = S_sa V_a S_saᵀ``, and the shift ``S_sa d_a``;
* ``from_symplectic(S)``: a unitary gate as a channel (``Y = 0``).

Phase diffusion is not a Gaussian channel (it is linear in ``V`` but not a congruence);
:func:`phase_diffusion` returns the Gaussian state with the exact first and second moments of the
dephased state and warns with :class:`~cvsim.errors.ApproximationWarning`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core import linalg as la
from ..core._types import FloatArray
from ..errors import ApproximationWarning, InputError, PhysicalityError, ShapeError
from .states import GaussianState

__all__ = ["GaussianChannel", "apply_channel", "dephase_mean", "dephase_moments", "phase_diffusion"]


def _square_even(A: npt.ArrayLike, name: str) -> FloatArray:
    A_ = np.asarray(A)
    if A_.ndim != 2 or A_.shape[0] != A_.shape[1] or A_.shape[0] % 2 or A_.shape[0] == 0:
        raise ShapeError(f"{name} must be a non-empty 2m x 2m array, got shape {A_.shape}")
    if np.iscomplexobj(A_) or not np.all(np.isfinite(A_)):
        raise InputError(f"{name} must be real and finite")
    return np.array(A_, dtype=float)


class GaussianChannel:
    """A Gaussian channel ``(X, Y, c)`` on ``m`` modes: ``d → X d + c``, ``V → X V Xᵀ + Y``.

    Parameters
    ----------
    X
        Real ``2m x 2m`` matrix.
    Y
        Real symmetric ``2m x 2m`` matrix.
    c
        Deterministic shift (default 0), e.g. from a displaced ancilla.
    validate
        Check complete positivity (default).
    atol
        Tolerance of the check, relative to ``max(1, ‖X‖², ‖Y‖)``.

    Raises
    ------
    PhysicalityError
        ``Y + (i/2)(Ω − X Ω Xᵀ)`` has an eigenvalue below ``−atol · scale``.
    """

    __slots__ = ("_X", "_Y", "_c")

    def __init__(
        self,
        X: npt.ArrayLike,
        Y: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        *,
        validate: bool = True,
        atol: float = cv.ATOL,
    ):
        X_, Y_ = _square_even(X, "X"), _square_even(Y, "Y")
        if X_.shape != Y_.shape:
            raise ShapeError(f"X and Y must have the same shape, got {X_.shape} and {Y_.shape}")
        c_ = np.zeros(X_.shape[0]) if c is None else np.asarray(c, dtype=float).reshape(-1)
        if c_.shape != (X_.shape[0],):
            raise ShapeError(f"c must have length {X_.shape[0]}, got {c_.shape}")
        if validate:
            scale = max(1.0, float(np.abs(X_).max()) ** 2, float(np.abs(Y_).max()))
            if np.abs(Y_ - Y_.T).max() > atol * scale:
                raise InputError("Y must be symmetric")
            Y_ = (Y_ + Y_.T) / 2
            m = X_.shape[0] // 2
            Om = cv.omega(m)
            lowest = float(la.eigvalsh(Y_ + 0.5j * (Om - X_ @ Om @ X_.T))[0])
            if lowest < -atol * scale:
                raise PhysicalityError(
                    f"channel is not completely positive: Y + i(Ω − XΩXᵀ)/2 has eigenvalue {lowest:.3e}"
                )
        for a in (X_, Y_, c_):
            a.setflags(write=False)
        self._X, self._Y, self._c = X_, Y_, c_

    # ------------------------------------------------------------------------- access
    @property
    def X(self) -> FloatArray:
        """``X`` (read-only)."""
        return self._X

    @property
    def Y(self) -> FloatArray:
        """``Y`` (read-only)."""
        return self._Y

    @property
    def c(self) -> FloatArray:
        """Deterministic shift of the mean (read-only)."""
        return self._c

    @property
    def num_modes(self) -> int:
        """Number of modes ``m``."""
        return int(self._X.shape[0]) // 2

    def __repr__(self) -> str:
        return f"GaussianChannel(modes={self.num_modes}, unitary={self.is_unitary()})"

    def is_unitary(self, atol: float = 1e-10) -> bool:
        """``Y = 0`` and ``X`` symplectic: a gate.

        Parameters
        ----------
        atol
            Tolerance.
        """
        return bool(np.abs(self.Y).max() <= atol) and cv.is_symplectic(self.X, atol)

    # ------------------------------------------------------------------------- constructors
    @classmethod
    def identity(cls, m: int = 1) -> GaussianChannel:
        """The identity channel on ``m`` modes.

        Parameters
        ----------
        m
            Number of modes.
        """
        return cls(np.eye(2 * m), np.zeros((2 * m, 2 * m)), validate=False)

    @classmethod
    def from_symplectic(cls, S: npt.ArrayLike) -> GaussianChannel:
        """A gate ``S`` as a channel (``Y = 0``).

        Parameters
        ----------
        S
            Symplectic matrix.
        """
        S_ = _square_even(S, "S")
        if not cv.is_symplectic(S_, cv.ATOL * max(1.0, float(np.abs(S_).max()) ** 2)):
            raise InputError("S is not symplectic")
        return cls(S_, np.zeros_like(S_), validate=False)

    @classmethod
    def loss(cls, eta: float, nbar: float = 0.0) -> GaussianChannel:
        """Single-mode loss: transmissivity ``η`` into a thermal environment of ``n̄`` photons.

        Parameters
        ----------
        eta
            Transmissivity in ``[0, 1]``.
        nbar
            Environment photons (>= 0).
        """
        if not 0.0 <= eta <= 1.0:
            raise InputError(f"transmissivity must be in [0, 1], got {eta}")
        if not nbar >= 0:
            raise InputError(f"nbar must be >= 0, got {nbar}")
        return cls(math.sqrt(eta) * np.eye(2), (1 - eta) * (nbar + 0.5) * np.eye(2), validate=False)

    @classmethod
    def amplifier(cls, gain: float, nbar: float = 0.0) -> GaussianChannel:
        """Single-mode phase-insensitive amplifier of gain ``G >= 1`` (idler with ``n̄`` photons).

        Parameters
        ----------
        gain
            Power gain ``G >= 1``.
        nbar
            Idler photons (>= 0).
        """
        if not gain >= 1.0 or not math.isfinite(gain):
            raise InputError(f"amplifier gain must be >= 1, got {gain}")
        if not nbar >= 0:
            raise InputError(f"nbar must be >= 0, got {nbar}")
        return cls(math.sqrt(gain) * np.eye(2), (gain - 1) * (nbar + 0.5) * np.eye(2), validate=False)

    @classmethod
    def additive_noise(cls, cov: float | npt.ArrayLike) -> GaussianChannel:
        """Random displacements with covariance ``Σ`` (a scalar ``σ²`` means ``σ² I`` on one mode).

        Parameters
        ----------
        cov
            ``σ²`` (single mode) or a ``2m x 2m`` positive-semidefinite matrix.
        """
        if np.ndim(cov) == 0:
            s2 = float(cov)  # type: ignore[arg-type]
            if not s2 >= 0:
                raise InputError(f"noise variance must be >= 0, got {s2}")
            Y = s2 * np.eye(2)
        else:
            Y = _square_even(cov, "cov")
            if la.eigvalsh(Y)[0] < -cv.ATOL * max(1.0, float(np.abs(Y).max())):
                raise InputError("noise covariance must be positive semidefinite")
        return cls(np.eye(Y.shape[0]), Y, validate=False)

    @classmethod
    def from_transfer(cls, T: npt.ArrayLike) -> GaussianChannel:
        """Lossy linear optics ``a → T a`` (plus vacuum), ``‖T‖ <= 1``: ``X = S(T)``, ``Y = ½(I − XXᵀ)``.

        Parameters
        ----------
        T
            Complex ``m x m`` transfer matrix with singular values ``<= 1``.
        """
        T_ = np.asarray(T, dtype=complex)
        if T_.ndim != 2 or T_.shape[0] != T_.shape[1] or T_.shape[0] == 0:
            raise ShapeError(f"T must be a non-empty square matrix, got shape {T_.shape}")
        if la.svd(T_)[1][0] > 1 + 1e-12:
            raise PhysicalityError("transfer matrix has a singular value > 1 (it would need gain)")
        X = np.block([[T_.real, -T_.imag], [T_.imag, T_.real]])
        return cls(X, cv.VACUUM_VARIANCE * (np.eye(X.shape[0]) - X @ X.T))

    @classmethod
    def from_ancilla(cls, S: npt.ArrayLike, ancilla: GaussianState) -> GaussianChannel:
        """Couple the system to an ancilla with a gate ``S`` and trace the ancilla out.

        ``S`` acts on the system modes ``0..m−1`` followed by the ancilla modes (xxpp of all of
        them). ``X = S_ss``, ``Y = S_sa V_a S_saᵀ``, shift ``c = S_sa d_a``.

        Parameters
        ----------
        S
            Symplectic matrix on ``m + k`` modes.
        ancilla
            State of the ``k`` ancilla modes.
        """
        S_ = _square_even(S, "S")
        n = S_.shape[0] // 2
        k = ancilla.num_modes
        m = n - k
        if m < 1:
            raise ShapeError(f"S acts on {n} modes, leaving none for the system besides {k} ancilla modes")
        if not cv.is_symplectic(S_, cv.ATOL * max(1.0, float(np.abs(S_).max()) ** 2)):
            raise InputError("S is not symplectic")
        sys_idx = cv.mode_indices(range(m), n)
        anc_idx = cv.mode_indices(range(m, n), n)
        Sss = S_[np.ix_(sys_idx, sys_idx)]
        Ssa = S_[np.ix_(sys_idx, anc_idx)]
        Y = Ssa @ ancilla.cov @ Ssa.T
        return cls(Sss, (Y + Y.T) / 2, Ssa @ ancilla.mean)

    # ------------------------------------------------------------------------- algebra
    def then(self, other: GaussianChannel) -> GaussianChannel:
        """``self`` first, then ``other``: ``X = X₂X₁``, ``Y = X₂Y₁X₂ᵀ + Y₂``, ``c = X₂c₁ + c₂``.

        Parameters
        ----------
        other
            Channel on the same modes.
        """
        if not isinstance(other, GaussianChannel) or other.num_modes != self.num_modes:
            raise ShapeError("then() needs a GaussianChannel on the same number of modes")
        X2 = other.X
        Y = X2 @ self.Y @ X2.T + other.Y
        return GaussianChannel(X2 @ self.X, (Y + Y.T) / 2, X2 @ self.c + other.c, validate=False)

    def tensor(self, other: GaussianChannel) -> GaussianChannel:
        """``self ⊗ other`` (the modes of ``other`` follow, xxpp).

        Parameters
        ----------
        other
            Second channel.
        """
        m, k = self.num_modes, other.num_modes
        return self.embed(range(m), m + k).then(other.embed(range(m, m + k), m + k))

    def embed(self, modes: Sequence[int], M: int) -> GaussianChannel:
        """The channel acting on ``modes`` of an ``M``-mode system (identity elsewhere).

        Parameters
        ----------
        modes
            Target modes, in the order of the channel's modes.
        M
            Total number of modes.
        """
        idx = cv.mode_indices(modes, M)
        if len(idx) != self.X.shape[0]:
            raise ShapeError(f"channel acts on {self.num_modes} mode(s), got {len(modes)}")
        X = np.eye(2 * M)
        Y = np.zeros((2 * M, 2 * M))
        c = np.zeros(2 * M)
        X[np.ix_(idx, idx)] = self.X
        Y[np.ix_(idx, idx)] = self.Y
        c[idx] = self.c
        return GaussianChannel(X, Y, c, validate=False)

    def apply(self, state: GaussianState, modes: Sequence[int] | None = None) -> GaussianState:
        """Apply to ``state`` on ``modes`` (all modes, in order, by default).

        Parameters
        ----------
        state
            Input state.
        modes
            Modes the channel acts on.
        """
        M = state.num_modes
        ch = self if modes is None and self.num_modes == M else self.embed(range(M) if modes is None else modes, M)
        V = ch.X @ state.cov @ ch.X.T + ch.Y
        return GaussianState(ch.X @ state.mean + ch.c, (V + V.T) / 2, validate=False)


def apply_channel(state: GaussianState, channel: GaussianChannel, modes: Sequence[int] | None = None) -> GaussianState:
    """``channel.apply(state, modes)``, as a function (the style of :mod:`cvsim.gaussian.ops`).

    Parameters
    ----------
    state
        Input state.
    channel
        The channel.
    modes
        Modes it acts on (all by default).
    """
    return channel.apply(state, modes)


def phase_diffusion(state: GaussianState, sigma: float, mode: int = 0) -> GaussianState:
    """Moment-matched Gaussian approximation of random phase ``φ ~ N(0, σ²)`` on ``mode``.

    The dephased state ``∫ R(φ) ρ R(φ)† p(φ) dφ`` is not Gaussian; this returns the Gaussian state
    with its exact first and second moments: ``⟨r_m⟩ → e^{−σ²/2}⟨r_m⟩``, the raw second moment of
    ``mode`` keeps its isotropic part and its anisotropic part decays as ``e^{−2σ²}``, and
    correlations with other modes decay as ``e^{−σ²/2}``. Warns with ``ApproximationWarning``.

    Parameters
    ----------
    state
        Input state.
    sigma
        Phase standard deviation (radians).
    mode
        Mode index.
    """
    if not sigma >= 0 or not math.isfinite(sigma):
        raise InputError(f"sigma must be finite and >= 0, got {sigma}")
    cv.mode_indices([mode], state.num_modes)  # validates the mode
    if sigma > 0:
        warnings.warn(
            "phase_diffusion: moment-matched Gaussian approximation of a non-Gaussian state",
            ApproximationWarning,
            stacklevel=2,
        )
    d = dephase_mean(state.mean, sigma, mode)
    V = dephase_moments(state.cov + np.outer(state.mean, state.mean), sigma, mode) - np.outer(d, d)
    return GaussianState(d, (V + V.T) / 2)


def dephase_mean(d: npt.ArrayLike, sigma: float, mode: int) -> FloatArray:
    """Mean after random phase ``φ ~ N(0, σ²)`` on ``mode``: ``d_m → e^{−σ²/2} d_m``.

    Parameters
    ----------
    d
        Mean vector (xxpp).
    sigma
        Phase standard deviation.
    mode
        Mode index.
    """
    out = np.array(d, dtype=float)
    out[cv.mode_indices([mode], out.shape[0] // 2)] *= math.exp(-(sigma**2) / 2)
    return out


def dephase_moments(G: npt.ArrayLike, sigma: float, mode: int) -> FloatArray:
    """The linear map ``E_φ[R_m(φ) G R_m(φ)ᵀ]`` on a symmetric ``2M x 2M`` matrix, ``φ ~ N(0, σ²)``.

    On the ``mode`` block the isotropic part is kept and the anisotropic part decays as
    ``e^{−2σ²}``; its correlations with the other modes decay as ``e^{−σ²/2}``.

    Parameters
    ----------
    G
        Symmetric matrix (raw second moments, or a noise contribution).
    sigma
        Phase standard deviation.
    mode
        Mode index.
    """
    Gn = np.array(G, dtype=float)
    M = Gn.shape[0] // 2
    J = cv.mode_indices([mode], M)
    other = [i for i in range(2 * M) if i not in J]
    k1, k2 = math.exp(-(sigma**2) / 2), math.exp(-2 * sigma**2)
    B = Gn[np.ix_(J, J)]
    iso = 0.5 * np.trace(B) * np.eye(2)
    Gn[np.ix_(J, J)] = iso + k2 * (B - iso)
    Gn[np.ix_(J, other)] *= k1
    Gn[np.ix_(other, J)] *= k1
    return Gn
