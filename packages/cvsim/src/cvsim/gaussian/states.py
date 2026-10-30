"""Gaussian states: mean vector and covariance matrix, and every figure of merit computed from them.

An ``M``-mode Gaussian state is its mean ``d = ⟨r⟩`` (length ``2M``) and covariance
``V = ⟨{Δr, Δrᵀ}⟩/2`` (``2M x 2M``), in the conventions of CONVENTIONS.md (ħ = 1, xxpp, vacuum
``V = I/2``). :class:`GaussianState` is an immutable value: its arrays are read-only and every
operation returns a new state.

Formulas (ħ = 1; ``ν_k`` the symplectic eigenvalues, ``Ω`` the symplectic form):

* purity ``Tr ρ² = 1 / (2^M √det V)``; entropy ``Σ_k g(ν_k)``,
  ``g(ν) = (ν + ½) ln(ν + ½) − (ν − ½) ln(ν − ½)`` (nats);
* overlap ``Tr ρ₁ρ₂ = exp(−½ δᵀ(V₁ + V₂)⁻¹δ) / √det(V₁ + V₂)``, ``δ = d₂ − d₁``;
* Uhlmann fidelity ``F = (Tr √(√ρ₁ ρ₂ √ρ₁))²``. **Pure modes are removed exactly first:** in the
  Williamson basis of ``ρ₁`` a pure mode is the vacuum, and ``F(τ ⊗ |0⟩⟨0|, ρ₂) = p₀ F(τ, ρ̃₂)``
  (``p₀`` the vacuum probability of ``ρ₂`` on those modes, ``ρ̃₂`` the conditional state), then
  the same from ``ρ₂``'s side. Without this, round-off ``ε`` in a pure mode (``ν = ½ + ε``)
  enters the closed form below as ``√ε``. What is left has no pure mode and uses Banchi,
  Braunstein and Pirandola, PRL 115, 260501 (2015):
  ``√F = F_tot^{1/4} det(V₁ + V₂)^{−1/4} exp(−¼ δᵀ(V₁ + V₂)⁻¹δ)`` with
  ``V_aux = Ωᵀ(V₁ + V₂)⁻¹(Ω/4 + V₂ΩV₁)`` and
  ``F_tot = det[2(√(1 + (V_aux Ω)⁻²/4) + 1) V_aux]``, evaluated on the eigenvalues ``λ`` of
  ``V_aux Ω`` as ``Π_λ 2(√(1 + 1/(4λ²)) + 1) λ`` (no matrix square root);
* logarithmic negativity ``E_N = Σ_k max(0, −log(2 ν̃_k))`` with ``ν̃`` the symplectic
  eigenvalues of the partially transposed covariance (momenta of the transposed modes flipped);
* Wigner function ``W(r) = exp(−½ (r − d)ᵀV⁻¹(r − d)) / ((2π)^M √det V)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core import linalg as la
from ..core._types import ComplexArray, FloatArray
from ..errors import InputError, PhysicalityError, ShapeError

__all__ = ["PURE_TOL", "GaussianState"]


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _as_1d(
    values: complex | float | Sequence[complex] | Sequence[float] | npt.ArrayLike, name: str, dtype: type
) -> np.ndarray:
    a: np.ndarray = np.atleast_1d(np.asarray(values, dtype=dtype))
    if a.ndim != 1 or a.size == 0:
        raise ShapeError(f"{name} must be a scalar or a non-empty 1-D sequence, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise InputError(f"{name} has non-finite entries")
    return a


def _g(nu: FloatArray) -> FloatArray:
    """Entropy of a thermal mode of symplectic eigenvalue ``ν`` (nats); ``g(½) = 0``."""
    a, b = nu + 0.5, nu - 0.5
    b_log_b = np.where(b > 1e-15, b * np.log(np.where(b > 1e-15, b, 1.0)), 0.0)
    return np.asarray(a * np.log(a) - b_log_b, dtype=float)


#: Default threshold under which ``ν − ½`` counts as a pure mode in :meth:`GaussianState.fidelity`.
PURE_TOL = 1e-12


def _remove_pure_modes(
    V1: FloatArray, d1: FloatArray, V2: FloatArray, d2: FloatArray, pure_tol: float
) -> tuple[float | None, tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None]:
    """Map both states to the Williamson basis of state 1 and project out its pure modes.

    Returns ``(p0, rest)``: ``p0`` the vacuum probability of state 2 on the pure modes of state 1
    (``None`` if it has none), ``rest`` the moments ``(V1, d1, V2, d2)`` of the remaining modes,
    state 2 conditioned on that vacuum (``None`` if state 1 is entirely pure).
    """
    M = d1.shape[0] // 2
    w = la.williamson(V1)
    Si = la.inv(w.S)
    V2p = Si @ V2 @ Si.T
    V2p = (V2p + V2p.T) / 2
    dp = Si @ (d2 - d1)
    threshold = pure_tol * max(1.0, float(np.abs(w.S).max()) ** 2)
    pure = [k for k in range(M) if w.nu[k] - cv.VACUUM_VARIANCE < threshold]
    mixed = [k for k in range(M) if k not in pure]
    tau = np.concatenate([w.nu[mixed], w.nu[mixed]])
    if not pure:
        return None, (np.diag(tau), np.zeros(2 * M), V2p, dp)
    iP = cv.mode_indices(pure, M)
    Q = V2p[np.ix_(iP, iP)] + np.eye(len(iP)) * cv.VACUUM_VARIANCE
    p0 = math.exp(-0.5 * float(dp[iP] @ la.solve(Q, dp[iP])) - 0.5 * la.logdet_pd(Q))
    if not mixed:
        return p0, None
    iR = cv.mode_indices(mixed, M)
    C = V2p[np.ix_(iR, iP)]
    K = la.solve(Q, C.T).T  # C Q⁻¹ (Q symmetric)
    Vc = V2p[np.ix_(iR, iR)] - K @ C.T
    dc = dp[iR] - K @ dp[iP]
    return p0, (np.diag(tau), np.zeros(len(iR)), (Vc + Vc.T) / 2, dc)


def _banchi(V1: FloatArray, d1: FloatArray, V2: FloatArray, d2: FloatArray) -> float:
    """Closed form of Banchi, Braunstein and Pirandola (2015), on the eigenvalues of ``V_aux Ω``."""
    M = d1.shape[0] // 2
    Om = cv.omega(M)
    Vs = V1 + V2
    delta = d2 - d1
    V_aux = Om.T @ la.solve(Vs, Om / 4 + V2 @ Om @ V1)
    lam = la.eigvals(V_aux @ Om)
    f_tot4 = complex(np.prod(2.0 * (np.sqrt(1.0 + 1.0 / (4.0 * lam**2)) + 1.0) * lam))
    # Pure modes have been removed; what remains can only be round-off, amplified near ν = ½.
    if abs(f_tot4.imag) > 1e-6 * max(1.0, abs(f_tot4.real)) or f_tot4.real <= 0:
        raise PhysicalityError(f"fidelity: F_tot^4 = {f_tot4:.3e} is not real positive")
    log_root = 0.25 * math.log(f_tot4.real) - 0.25 * la.logdet_pd(Vs) - 0.25 * float(delta @ la.solve(Vs, delta))
    return math.exp(2.0 * log_root)


def _fidelity_moments(
    V1: FloatArray, d1: FloatArray, V2: FloatArray, d2: FloatArray, pure_tol: float, swapped: bool
) -> float:
    """Fidelity by alternately removing the pure modes of each state, then the closed form.

    Each removal drops at least one mode; ``swapped`` records that state 1 has just been found
    to have no pure modes, so that the roles are exchanged at most once between removals.
    """
    p0, rest = _remove_pure_modes(V1, d1, V2, d2, pure_tol)
    if rest is None:  # state 1 is pure: F = ⟨ψ|ρ₂|ψ⟩ = p0
        return float(p0) if p0 is not None else 0.0
    if p0 is None:  # no pure mode in state 1
        if swapped:
            return _banchi(*rest)
        Va, da, Vb, db = rest
        return _fidelity_moments(Vb, db, Va, da, pure_tol, swapped=True)
    if p0 == 0.0:
        return 0.0
    return p0 * _fidelity_moments(*rest, pure_tol, swapped=False)


class GaussianState:
    """Mean vector and covariance matrix of an ``M``-mode Gaussian state (xxpp, ħ = 1).

    Parameters
    ----------
    mean
        Mean vector ``d``, length ``2M``.
    cov
        Covariance ``V``, real symmetric ``2M x 2M``.
    validate
        Check symmetry and the uncertainty principle ``V + iΩ/2 >= 0`` (default). Code that
        preserves physicality by construction passes ``False``.
    atol
        Tolerance of the physicality check, relative to ``max(1, ‖V‖)``.

    Raises
    ------
    ShapeError
        Inconsistent shapes.
    InputError
        Non-finite, complex or asymmetric covariance.
    PhysicalityError
        ``V + iΩ/2`` has an eigenvalue below ``−atol · max(1, ‖V‖)``.
    """

    __slots__ = ("_cov", "_mean")
    _mean: np.ndarray
    _cov: np.ndarray

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike, *, validate: bool = True, atol: float = cv.ATOL):
        V = np.asarray(cov)
        if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] % 2 or V.shape[0] == 0:
            raise ShapeError(f"covariance must be a non-empty 2M x 2M array, got shape {V.shape}")
        d = np.asarray(mean)
        if d.ndim != 1 or d.shape[0] != V.shape[0]:
            raise ShapeError(f"mean must have length {V.shape[0]} to match the covariance, got shape {d.shape}")
        if validate:
            if np.iscomplexobj(V) or np.iscomplexobj(d):
                raise InputError("mean and covariance must be real")
            if not (np.all(np.isfinite(V)) and np.all(np.isfinite(d))):
                raise InputError("mean and covariance must be finite")
            scale = max(1.0, float(np.abs(V).max()))
            asym = float(np.abs(V - V.T).max())
            if asym > atol * scale:
                raise InputError(f"covariance must be symmetric (max asymmetry {asym:.3e})")
            V = (V + V.T) / 2
            M = V.shape[0] // 2
            lowest = float(la.eigvalsh(V + 0.5j * cv.omega(M))[0])
            if lowest < -atol * scale:
                raise PhysicalityError(
                    f"covariance violates the uncertainty principle: V + iΩ/2 has eigenvalue {lowest:.3e} "
                    f"(tolerance {atol * scale:.1e}); see core.linalg.nearest_physical_covariance"
                )
        self._mean = _readonly(d)
        self._cov = _readonly(V)

    # ------------------------------------------------------------------------- constructors
    @classmethod
    def vacuum(cls, M: int = 1) -> GaussianState:
        """``M``-mode vacuum: ``d = 0``, ``V = I/2``.

        Parameters
        ----------
        M
            Number of modes (>= 1).
        """
        if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M < 1:
            raise InputError(f"number of modes must be an integer >= 1, got {M!r}")
        return cls(np.zeros(2 * M), np.eye(2 * M) * cv.VACUUM_VARIANCE, validate=False)

    @classmethod
    def coherent(cls, alpha: complex | Sequence[complex]) -> GaussianState:
        """Coherent state ``|α₁, …, α_M⟩``: ``d = √2 (Re α, Im α)``, ``V = I/2``.

        Parameters
        ----------
        alpha
            One complex amplitude per mode.
        """
        a = _as_1d(alpha, "alpha", complex)
        return cls(cv.displacement_vector(a.tolist()), np.eye(2 * a.size) * cv.VACUUM_VARIANCE, validate=False)

    @classmethod
    def thermal(cls, nbar: float | Sequence[float]) -> GaussianState:
        """Thermal state(s) of mean photon numbers ``n̄``: ``V = (n̄ + ½) I``.

        Parameters
        ----------
        nbar
            One mean photon number (>= 0) per mode.
        """
        n = _as_1d(nbar, "nbar", float)
        if np.any(n < 0):
            raise InputError(f"nbar must be >= 0, got {n.tolist()}")
        return cls(np.zeros(2 * n.size), np.diag(np.concatenate([n, n]) + 0.5), validate=False)

    @classmethod
    def squeezed(cls, r: float | Sequence[float], phi: float | Sequence[float] = 0.0) -> GaussianState:
        """Squeezed vacuum ``S(r, φ)|0⟩`` on each mode (gate ``S`` of CONVENTIONS.md).

        ``φ = 0`` squeezes ``x``: ``Var x = e^{−2r}/2``; the squeezed quadrature is ``x_{φ/2}``.

        Parameters
        ----------
        r
            Squeezing parameter(s), one per mode.
        phi
            Squeezing angle(s), a scalar or one per mode.
        """
        rs = _as_1d(r, "r", float)
        phis = _as_1d(phi, "phi", float)
        if phis.size not in (1, rs.size):
            raise ShapeError(f"phi must be a scalar or have one entry per mode ({rs.size}), got {phis.size}")
        phis = np.broadcast_to(phis, rs.shape)
        M = rs.size
        S = np.eye(2 * M)
        for k, (rk, pk) in enumerate(zip(rs, phis)):
            S = cv.embed(cv.squeezing(float(rk), float(pk)), [k], M) @ S
        return cls.from_symplectic(S)

    @classmethod
    def two_mode_squeezed(cls, r: float, phi: float = 0.0) -> GaussianState:
        """Two-mode squeezed vacuum ``S2(r, φ)|0, 0⟩`` (gate ``S2`` of CONVENTIONS.md).

        Parameters
        ----------
        r
            Squeezing parameter; each mode has ``⟨n⟩ = sinh² r``.
        phi
            Squeezing phase.
        """
        return cls.from_symplectic(cv.two_mode_squeezing(r, phi))

    @classmethod
    def from_symplectic(
        cls, S: npt.ArrayLike, nu: float | Sequence[float] | None = None, mean: npt.ArrayLike | None = None
    ) -> GaussianState:
        """The state ``V = S diag(ν, ν) Sᵀ`` with mean ``d``: a symplectic map applied to thermal modes.

        Parameters
        ----------
        S
            Real symplectic ``2M x 2M`` matrix.
        nu
            Symplectic eigenvalues (>= 1/2), a scalar or one per mode; default ``1/2`` (pure).
        mean
            Mean vector; default zero.

        Raises
        ------
        InputError
            ``S`` is not symplectic or some ``ν < 1/2``.
        """
        S_ = np.asarray(S, dtype=float)
        if S_.ndim != 2 or S_.shape[0] != S_.shape[1] or S_.shape[0] % 2 or S_.shape[0] == 0:
            raise ShapeError(f"S must be 2M x 2M, got shape {S_.shape}")
        M = S_.shape[0] // 2
        if not cv.is_symplectic(S_, cv.ATOL * max(1.0, float(np.abs(S_).max()) ** 2)):
            raise InputError("S is not symplectic")
        nus = np.full(M, cv.VACUUM_VARIANCE) if nu is None else np.broadcast_to(_as_1d(nu, "nu", float), (M,))
        if np.any(nus < cv.VACUUM_VARIANCE - cv.ATOL):
            raise PhysicalityError(f"symplectic eigenvalues must be >= 1/2, got {np.asarray(nus).tolist()}")
        d = np.zeros(2 * M) if mean is None else np.asarray(mean, dtype=float)
        V = (S_ * np.concatenate([nus, nus])) @ S_.T
        return cls(d, (V + V.T) / 2, validate=mean is not None)

    @classmethod
    def product(cls, *states: GaussianState) -> GaussianState:
        """Tensor product ``ρ₁ ⊗ ρ₂ ⊗ …``, modes in the order listed (xxpp: all x first).

        Parameters
        ----------
        *states
            The factors.
        """
        if not states:
            raise InputError("product needs at least one state")
        M = sum(s.num_modes for s in states)
        d, V = np.zeros(2 * M), np.zeros((2 * M, 2 * M))
        offset = 0
        for s in states:
            idx = cv.mode_indices(range(offset, offset + s.num_modes), M)
            d[idx] = s.mean
            V[np.ix_(idx, idx)] = s.cov
            offset += s.num_modes
        return cls(d, V, validate=False)

    @classmethod
    def from_hbar(cls, mean: npt.ArrayLike, cov: npt.ArrayLike, hbar: float) -> GaussianState:
        """A state given in the convention ``x = √(ħ/2)(a + a†)`` (Strawberry Fields, The Walrus: ħ = 2).

        Parameters
        ----------
        mean, cov
            Moments in that convention.
        hbar
            Its value of ħ.
        """
        d, V = cv.convert_hbar(mean, cov, hbar_from=hbar, hbar_to=cv.HBAR)
        return cls(d, V)

    @classmethod
    def from_xpxp(cls, mean: npt.ArrayLike, cov: npt.ArrayLike) -> GaussianState:
        """A state given in the ``(x₁, p₁, x₂, p₂, …)`` ordering.

        Parameters
        ----------
        mean, cov
            Moments in xpxp ordering.
        """
        d, V = cv.from_xpxp(mean, cov)
        return cls(d, V)

    # ------------------------------------------------------------------------- basic access
    @property
    def mean(self) -> FloatArray:
        """Mean vector ``d`` (read-only), xxpp."""
        return self._mean

    @property
    def cov(self) -> FloatArray:
        """Covariance ``V`` (read-only), xxpp."""
        return self._cov

    @property
    def num_modes(self) -> int:
        """Number of modes ``M``."""
        return int(self._mean.shape[0]) // 2

    def __repr__(self) -> str:
        photons = float(self.mean_photons().sum())
        return f"GaussianState(modes={self.num_modes}, purity={self.purity():.6g}, photons={photons:.6g})"

    def isclose(self, other: GaussianState, atol: float = 1e-10) -> bool:
        """Same number of modes and moments equal within ``atol``.

        Parameters
        ----------
        other
            State to compare with.
        atol
            Absolute tolerance on every entry of ``d`` and ``V``.
        """
        return (
            self.num_modes == other.num_modes
            and bool(np.allclose(self.mean, other.mean, atol=atol, rtol=0))
            and bool(np.allclose(self.cov, other.cov, atol=atol, rtol=0))
        )

    def _check_same_modes(self, other: GaussianState) -> None:
        if not isinstance(other, GaussianState):
            raise InputError(f"expected a GaussianState, got {type(other).__name__}")
        if other.num_modes != self.num_modes:
            raise ShapeError(f"states have {self.num_modes} and {other.num_modes} modes")

    # ------------------------------------------------------------------------- structure
    def reduce(self, modes: Sequence[int]) -> GaussianState:
        """Reduced state of ``modes`` (partial trace over the others), in the order listed.

        Parameters
        ----------
        modes
            Distinct mode indices to keep.
        """
        idx = cv.mode_indices(modes, self.num_modes)
        return GaussianState(self.mean[idx], self.cov[np.ix_(idx, idx)], validate=False)

    def tensor(self, other: GaussianState) -> GaussianState:
        """``self ⊗ other`` (the modes of ``other`` follow).

        Parameters
        ----------
        other
            Second factor.
        """
        return GaussianState.product(self, other)

    def to_hbar(self, hbar: float) -> tuple[FloatArray, FloatArray]:
        """Moments ``(d, V)`` in the convention of another ħ (vacuum ``V = ħ/2 · I``).

        Parameters
        ----------
        hbar
            Target ħ (2 for Strawberry Fields and The Walrus).
        """
        return cv.convert_hbar(self.mean, self.cov, hbar_from=cv.HBAR, hbar_to=hbar)

    def to_xpxp(self) -> tuple[FloatArray, FloatArray]:
        """Moments ``(d, V)`` in the ``(x₁, p₁, x₂, p₂, …)`` ordering."""
        return cv.to_xpxp(self.mean, self.cov)

    # ------------------------------------------------------------------------- mixedness
    def symplectic_eigenvalues(self) -> FloatArray:
        """Williamson's symplectic eigenvalues ``ν_k >= 1/2``, ascending."""
        return la.symplectic_eigenvalues(self.cov)

    def purity(self) -> float:
        """``Tr ρ² = 1 / (2^M √det V)``."""
        return math.exp(-self.num_modes * math.log(2.0) - 0.5 * la.logdet_pd(self.cov))

    def is_pure(self, atol: float = 1e-9) -> bool:
        """``|Tr ρ² − 1| < atol``.

        Parameters
        ----------
        atol
            Tolerance on the purity.
        """
        return abs(self.purity() - 1.0) < atol

    def entropy(self, base: float = math.e) -> float:
        """Von Neumann entropy ``Σ_k g(ν_k)``; nats by default.

        Parameters
        ----------
        base
            Logarithm base (``2`` for bits).
        """
        if not base > 1:
            raise InputError(f"base must be > 1, got {base}")
        return float(np.sum(_g(self.symplectic_eigenvalues())) / math.log(base))

    # ------------------------------------------------------------------------- moments
    def complex_moments(self) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
        """``(α, N, A)`` with ``α_j = ⟨a_j⟩``, ``N_jk = ⟨a_j† a_k⟩``, ``A_jk = ⟨a_j a_k⟩``.

        From ``G = ⟨r rᵀ⟩ = V + d dᵀ + iΩ/2`` and ``a = (x + ip)/√2``.
        """
        M = self.num_modes
        d = self.mean
        G = self.cov + np.outer(d, d) + 0.5j * cv.omega(M)
        X, P = slice(0, M), slice(M, 2 * M)
        alpha = (d[X] + 1j * d[P]) / math.sqrt(2.0)
        N = (G[X, X] + 1j * G[X, P] - 1j * G[P, X] + G[P, P]) / 2
        A = (G[X, X] + 1j * G[X, P] + 1j * G[P, X] - G[P, P]) / 2
        return np.asarray(alpha, dtype=complex), np.asarray(N, dtype=complex), np.asarray(A, dtype=complex)

    def mean_photons(self) -> FloatArray:
        """``⟨n_k⟩ = (V_xx + V_pp − 1)/2 + (d_x² + d_p²)/2`` for every mode."""
        M = self.num_modes
        v, d = np.diag(self.cov), self.mean
        return np.asarray((v[:M] + v[M:] - 1.0) / 2 + (d[:M] ** 2 + d[M:] ** 2) / 2, dtype=float)

    def photon_covariance(self) -> FloatArray:
        """``Cov(n_j, n_k) = ½ Σ V[J, K]² + d_Jᵀ V[J, K] d_K − ¼ δ_jk``, ``J = (x_j, p_j)``.

        Wick's theorem for the Weyl-symmetric quadratic forms ``x² + p²``, minus the ordering
        correction ``¼ δ_jk`` (vacuum: ``Var n = 0``; thermal: ``n̄² + n̄``).
        """
        M = self.num_modes
        V, d = self.cov, self.mean
        C = np.empty((M, M))
        for j in range(M):
            J = [j, M + j]
            for k in range(j, M):
                K = [k, M + k]
                B = V[np.ix_(J, K)]
                C[j, k] = C[k, j] = 0.5 * float(np.sum(B**2)) + float(d[J] @ B @ d[K]) - (0.25 if j == k else 0.0)
        return C

    # ------------------------------------------------------------------------- distances
    def overlap(self, other: GaussianState) -> float:
        """``Tr ρ₁ρ₂ = exp(−½ δᵀ(V₁ + V₂)⁻¹δ) / √det(V₁ + V₂)``.

        Parameters
        ----------
        other
            Second state (same number of modes).
        """
        self._check_same_modes(other)
        Vs = self.cov + other.cov
        delta = other.mean - self.mean
        return min(1.0, math.exp(-0.5 * float(delta @ la.solve(Vs, delta)) - 0.5 * la.logdet_pd(Vs)))

    def fidelity(self, other: GaussianState, pure_tol: float = PURE_TOL) -> float:
        """Uhlmann fidelity ``(Tr √(√ρ₁ ρ₂ √ρ₁))²`` for any two Gaussian states.

        Pure modes are removed exactly before the closed form of Banchi, Braunstein and Pirandola
        (2015) is used (module docstring, "Fidelity"): in the Williamson basis of ``ρ₁``, a pure
        mode is the vacuum, and ``F(τ ⊗ |0⟩⟨0|, ρ₂) = p₀ F(τ, ρ̃₂)`` with ``p₀`` the probability
        of vacuum of ``ρ₂`` on those modes and ``ρ̃₂`` the state conditioned on it. The same is done
        from ``ρ₂``'s side; a fully pure state leaves ``p₀``, the overlap.

        Accuracy: 1e-13 to 1e-11 relative, growing with the squeezing, also with pure modes
        entangled with mixed ones (stress run: worst 5e-12 over 2000 random 1–3-mode pairs).
        A mode counts as pure when ``ν − ½ < pure_tol · max(1, ‖S‖²)`` (``S`` its Williamson
        factor): the covariance does not determine ``ν − ½`` more finely than that. The fidelity
        varies like ``√(ν − ½)`` there, so a state mixed by less than the threshold is answered to
        about ``√pure_tol``; for modes just above the threshold the closed form gives ``√ε``.

        Parameters
        ----------
        other
            Second state (same number of modes).
        pure_tol
            Threshold below which ``ν − ½`` counts as zero (relative to ``max(1, ‖S‖²)``).
        """
        self._check_same_modes(other)
        if not pure_tol >= 0:
            raise InputError(f"pure_tol must be >= 0, got {pure_tol}")
        f = _fidelity_moments(self.cov, self.mean, other.cov, other.mean, pure_tol, swapped=False)
        return min(1.0, f)

    # ------------------------------------------------------------------------- entanglement
    def log_negativity(self, modes: Sequence[int], base: float = 2.0) -> float:
        """Logarithmic negativity ``E_N = Σ_k max(0, −log(2ν̃_k))`` of the bipartition ``modes | rest``.

        Base 2 by default, as qttlind (two-mode squeezed vacuum: ``E_N = 2r / ln 2``).

        Parameters
        ----------
        modes
            Modes of one side (the partial transpose flips their momenta).
        base
            Logarithm base (``math.e`` for nats).
        """
        if not base > 1:
            raise InputError(f"base must be > 1, got {base}")
        M = self.num_modes
        idx = cv.mode_indices(modes, M)
        flips = np.ones(2 * M)
        flips[[i for i in idx if i >= M]] = -1.0
        Vt = flips[:, None] * self.cov * flips[None, :]
        nu = la.symplectic_eigenvalues(Vt)
        return float(np.sum(np.maximum(0.0, -np.log(2.0 * nu))) / math.log(base))

    # ------------------------------------------------------------------------- phase space
    def wigner(self, points: npt.ArrayLike) -> FloatArray:
        """Wigner function at phase-space points.

        Parameters
        ----------
        points
            Array of shape ``(..., 2M)`` (xxpp coordinates).

        Returns
        -------
        ndarray
            ``W`` at each point, shape ``points.shape[:-1]``.
        """
        r = np.asarray(points, dtype=float)
        n = 2 * self.num_modes
        if r.ndim == 0 or r.shape[-1] != n:
            raise ShapeError(f"points must have last dimension {n}, got shape {r.shape}")
        Vi = la.inv(self.cov)
        diff = r - self.mean
        quad = np.einsum("...i,ij,...j->...", diff, Vi, diff)
        log_norm = self.num_modes * math.log(2 * math.pi) + 0.5 * la.logdet_pd(self.cov)
        return np.asarray(np.exp(-0.5 * quad - log_norm), dtype=float)

    def wigner_grid(self, x: npt.ArrayLike, p: npt.ArrayLike, mode: int = 0) -> FloatArray:
        """Single-mode Wigner function of ``mode`` on the grid ``x`` (columns) by ``p`` (rows).

        Parameters
        ----------
        x, p
            1-D coordinate arrays.
        mode
            Mode index (the others are traced out).
        """
        X, P = np.meshgrid(np.asarray(x, dtype=float), np.asarray(p, dtype=float))
        return self.reduce([mode]).wigner(np.stack([X, P], axis=-1))

    def vacuum_probability(self, modes: Sequence[int] | None = None) -> float:
        """Probability that every listed mode is empty (all modes by default).

        ``p₀ = exp(−½ dᵀ(V + I/2)⁻¹d) / √det(V + I/2)`` on the reduced state.

        Parameters
        ----------
        modes
            Modes to test; ``None`` for all.
        """
        red = self if modes is None else self.reduce(modes)
        Q = red.cov + np.eye(red.cov.shape[0]) * cv.VACUUM_VARIANCE
        d = red.mean
        return math.exp(-0.5 * float(d @ la.solve(Q, d)) - 0.5 * la.logdet_pd(Q))
