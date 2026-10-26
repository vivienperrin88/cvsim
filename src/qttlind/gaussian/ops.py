"""Gaussian gates and channels.

**Gates.** A Gaussian unitary acts on a state as ``d -> S d + delta`` and ``V -> S V S^T``. Each
gate below is defined by its generator (the same as Strawberry Fields, see
``qttlind.reference.cv_dense``) and implemented through its Heisenberg-picture Bogoliubov form
``U^dag a U = A a + B a^dag``, which fixes ``S`` via
:func:`~.phase_space.symplectic_from_bogoliubov`:

=================  ==========================================  ===================================
gate               generator                                   ``U^dag a U``
=================  ==========================================  ===================================
``rotation``       ``exp(i theta n)``                          ``e^{i theta} a``
``squeezing``      ``exp((conj(z) a^2 - z a^dag^2)/2)``        ``cosh(r) a - e^{i phi} sinh(r) a^dag``
``beamsplitter``   ``exp(theta(e^{i phi} a b^dag - h.c.))``    ``cos(theta) a - e^{-i phi} sin(theta) b``
``two_mode_sq.``   ``exp(conj(z) a b - z a^dag b^dag)``        ``cosh(r) a - e^{i phi} sinh(r) b^dag``
=================  ==========================================  ===================================

**Channels.** A Gaussian channel acts as ``d -> X d`` and ``V -> X V X^T + Y``; it is completely
positive iff ``Y + i Omega/2 - i X Omega X^T/2 >= 0``. Pure loss of transmissivity ``eta`` is
``X = sqrt(eta) I``, ``Y = (1 - eta)/2 I``: the Lindblad equation with ``L = sqrt(kappa) a``
during a time ``t`` is exactly this channel with ``eta = exp(-kappa t)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InputError, ShapeError
from .phase_space import ATOL, is_symplectic, mode_indices, omega, symplectic_from_bogoliubov
from .states import GaussianState

__all__ = [
    "amplifier",
    "apply_channel",
    "apply_symplectic",
    "beamsplitter",
    "beamsplitter_matrix",
    "check_channel",
    "displace",
    "interferometer",
    "interferometer_matrix",
    "loss",
    "rotate",
    "rotation_matrix",
    "squeeze",
    "squeezing_matrix",
    "thermal_noise",
    "two_mode_squeeze",
    "two_mode_squeezing_matrix",
]

# ----------------------------------------------------------------------------- symplectic matrices


def rotation_matrix(theta: float) -> np.ndarray:
    """Symplectic matrix of ``R(theta) = exp(i theta n)``: a counter-clockwise rotation."""
    return symplectic_from_bogoliubov(np.exp(1j * theta), 0.0)


def squeezing_matrix(r: float, phi: float = 0.0) -> np.ndarray:
    """Symplectic matrix of ``S(r, phi)``; ``phi = 0`` gives ``diag(e^{-r}, e^{r})``."""
    return symplectic_from_bogoliubov(np.cosh(r), -np.exp(1j * phi) * np.sinh(r))


def beamsplitter_matrix(theta: float, phi: float = 0.0) -> np.ndarray:
    """4 x 4 symplectic matrix (modes ``a, b``) of ``BS(theta, phi)``; transmissivity ``cos^2 theta``."""
    t, s = np.cos(theta), np.sin(theta)
    U = np.array([[t, -np.exp(-1j * phi) * s], [np.exp(1j * phi) * s, t]])
    return symplectic_from_bogoliubov(U, np.zeros((2, 2)))


def two_mode_squeezing_matrix(r: float, phi: float = 0.0) -> np.ndarray:
    """4 x 4 symplectic matrix of ``S2(r, phi)``."""
    A = np.cosh(r) * np.eye(2)
    B = -np.exp(1j * phi) * np.sinh(r) * np.array([[0, 1], [1, 0]])
    return symplectic_from_bogoliubov(A, B)


def interferometer_matrix(U: np.ndarray, atol: float = ATOL) -> np.ndarray:
    """Symplectic matrix of the passive interferometer ``a -> U a`` (``U`` unitary)."""
    U = np.atleast_2d(np.asarray(U, dtype=complex))
    if U.shape[0] != U.shape[1] or not np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=atol):
        raise InputError("interferometer matrix must be square and unitary")
    return symplectic_from_bogoliubov(U, np.zeros_like(U))


# ----------------------------------------------------------------------------- applying gates


def _embed(S: np.ndarray, modes: Sequence[int], M: int) -> np.ndarray:
    """Extend a ``2m x 2m`` matrix acting on ``modes`` to all ``M`` modes (identity elsewhere)."""
    idx = mode_indices(modes, M)
    if S.shape != (len(idx), len(idx)):
        raise ShapeError(f"matrix of shape {S.shape} does not act on {len(modes)} mode(s)")
    full = np.eye(2 * M)
    full[np.ix_(idx, idx)] = S
    return full


def apply_symplectic(state: GaussianState, S: np.ndarray, modes: Sequence[int] | None = None) -> GaussianState:
    """``d -> S d``, ``V -> S V S^T`` with ``S`` acting on ``modes`` (all modes if ``None``)."""
    M = state.num_modes
    S = np.asarray(S, dtype=float)
    full = _embed(S, range(M) if modes is None else modes, M)
    if not is_symplectic(full):
        raise InputError("matrix is not symplectic (S Omega S^T != Omega)")
    return GaussianState(full @ state.mean, full @ state.cov @ full.T, validate=False)


def displace(state: GaussianState, alpha: complex, mode: int = 0) -> GaussianState:
    """``D(alpha)`` on one mode: ``d -> d + sqrt(2) (Re alpha, Im alpha)``."""
    M = state.num_modes
    idx = mode_indices([mode], M)
    mean = state.mean.copy()
    mean[idx] += np.sqrt(2) * np.array([np.real(alpha), np.imag(alpha)])
    return GaussianState(mean, state.cov, validate=False)


def rotate(state: GaussianState, theta: float, mode: int = 0) -> GaussianState:
    return apply_symplectic(state, rotation_matrix(theta), [mode])


def squeeze(state: GaussianState, r: float, phi: float = 0.0, mode: int = 0) -> GaussianState:
    return apply_symplectic(state, squeezing_matrix(r, phi), [mode])


def beamsplitter(state: GaussianState, theta: float, phi: float = 0.0, modes: Sequence[int] = (0, 1)) -> GaussianState:
    return apply_symplectic(state, beamsplitter_matrix(theta, phi), modes)


def two_mode_squeeze(state: GaussianState, r: float, phi: float = 0.0, modes: Sequence[int] = (0, 1)) -> GaussianState:
    return apply_symplectic(state, two_mode_squeezing_matrix(r, phi), modes)


def interferometer(state: GaussianState, U: np.ndarray, modes: Sequence[int] | None = None) -> GaussianState:
    return apply_symplectic(state, interferometer_matrix(U), modes)


# ----------------------------------------------------------------------------- channels


def check_channel(X: np.ndarray, Y: np.ndarray, atol: float = ATOL) -> None:
    """Raise :class:`InputError` unless ``(X, Y)`` is a completely positive Gaussian channel."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[0] != X.shape[1] or X.shape[0] % 2:
        raise ShapeError(f"X and Y must be 2m x 2m arrays of the same shape, got {X.shape} and {Y.shape}")
    if not np.allclose(Y, Y.T, atol=atol):
        raise InputError("Y must be symmetric")
    W = omega(X.shape[0] // 2)
    lowest = np.linalg.eigvalsh(Y + 0.5j * W - 0.5j * X @ W @ X.T).min()
    if lowest < -atol:
        raise InputError(
            f"channel is not completely positive: Y + i(Omega - X Omega X^T)/2 has eigenvalue {lowest:.3e}"
        )


def apply_channel(
    state: GaussianState, X: np.ndarray, Y: np.ndarray, modes: Sequence[int] | None = None
) -> GaussianState:
    """``d -> X d``, ``V -> X V X^T + Y`` on ``modes`` (all if ``None``), after :func:`check_channel`."""
    M = state.num_modes
    modes = list(range(M)) if modes is None else list(modes)
    check_channel(X, Y)
    Xf = _embed(np.asarray(X, dtype=float), modes, M)
    Yf = np.zeros((2 * M, 2 * M))
    idx = mode_indices(modes, M)
    Yf[np.ix_(idx, idx)] = Y
    return GaussianState(Xf @ state.mean, Xf @ state.cov @ Xf.T + Yf, validate=False)


def loss(state: GaussianState, eta: float, mode: int = 0, nbar: float = 0.0) -> GaussianState:
    """Beam splitter of transmissivity ``eta`` with a thermal environment of ``nbar`` photons."""
    if not 0.0 <= eta <= 1.0:
        raise InputError(f"transmissivity must be in [0, 1], got {eta}")
    if nbar < 0:
        raise InputError(f"nbar must be >= 0, got {nbar}")
    X = np.sqrt(eta) * np.eye(2)
    Y = (1 - eta) * (nbar + 0.5) * np.eye(2)
    return apply_channel(state, X, Y, [mode])


def amplifier(state: GaussianState, gain: float, mode: int = 0, nbar: float = 0.0) -> GaussianState:
    """Phase-insensitive amplifier of gain ``G >= 1``: ``X = sqrt(G) I``, ``Y = (G - 1)(nbar + 1/2) I``."""
    if gain < 1.0:
        raise InputError(f"amplifier gain must be >= 1, got {gain}")
    X = np.sqrt(gain) * np.eye(2)
    Y = (gain - 1) * (nbar + 0.5) * np.eye(2)
    return apply_channel(state, X, Y, [mode])


def thermal_noise(state: GaussianState, sigma2: float, mode: int = 0) -> GaussianState:
    """Additive Gaussian noise (random displacements of variance ``sigma2`` per quadrature)."""
    if sigma2 < 0:
        raise InputError(f"noise variance must be >= 0, got {sigma2}")
    return apply_channel(state, np.eye(2), sigma2 * np.eye(2), [mode])
