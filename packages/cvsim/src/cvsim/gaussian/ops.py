"""Gates applied to Gaussian states: ``d → S d + c``, ``V → S V Sᵀ`` (CONVENTIONS.md, "Gates").

Every function returns a new :class:`~cvsim.gaussian.states.GaussianState`. Mode arguments list
the modes a gate acts on, in the order of its matrix; the gate matrices are those of
:mod:`cvsim.gaussian.symplectic`. A sequence of gates ``g₁, g₂, …`` applied in that order has the
symplectic matrix ``S = ⋯ S₂ S₁``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..errors import InputError
from . import symplectic as sy
from .states import GaussianState

__all__ = [
    "apply_symplectic",
    "beamsplitter",
    "cx",
    "cz",
    "displace",
    "fourier",
    "interferometer",
    "rotate",
    "shear",
    "squeeze",
    "two_mode_squeeze",
]


def _modes(modes: int | Sequence[int]) -> list[int]:
    return [int(modes)] if isinstance(modes, (int, np.integer)) else [int(m) for m in modes]


def apply_symplectic(state: GaussianState, S: npt.ArrayLike, modes: Sequence[int] | None = None) -> GaussianState:
    """``d → S d``, ``V → S V Sᵀ`` with ``S`` acting on ``modes`` (all modes by default).

    Parameters
    ----------
    state
        Input state.
    S
        Symplectic matrix on ``len(modes)`` modes (xxpp of those modes).
    modes
        Modes ``S`` acts on, in the order of its rows.

    Raises
    ------
    InputError
        ``S`` is not symplectic.
    """
    S_ = np.asarray(S, dtype=float)
    if not cv.is_symplectic(S_, cv.ATOL * max(1.0, float(np.abs(S_).max(initial=0.0)) ** 2)):
        raise InputError("matrix is not symplectic")
    M = state.num_modes
    full = S_ if modes is None and S_.shape == (2 * M, 2 * M) else cv.embed(S_, range(M) if modes is None else modes, M)
    V = full @ state.cov @ full.T
    return GaussianState(full @ state.mean, (V + V.T) / 2, validate=False)


def displace(state: GaussianState, alpha: complex | Sequence[complex], modes: int | Sequence[int] = 0) -> GaussianState:
    """Displacement ``D(α)``: ``d → d + √2 (Re α, Im α)`` on ``modes``.

    Parameters
    ----------
    state
        Input state.
    alpha
        Complex amplitude(s), one per mode.
    modes
        Mode index or indices.
    """
    ms = _modes(modes)
    a = np.atleast_1d(np.asarray(alpha, dtype=complex))
    if a.shape != (len(ms),):
        raise InputError(f"need one amplitude per mode ({len(ms)}), got shape {a.shape}")
    shift = np.zeros(2 * state.num_modes)
    shift[cv.mode_indices(ms, state.num_modes)] = cv.displacement_vector(a.tolist())
    return GaussianState(state.mean + shift, state.cov, validate=False)


def rotate(state: GaussianState, theta: float, mode: int = 0) -> GaussianState:
    """Phase rotation ``R(θ) = exp(iθn)`` (counter-clockwise in phase space).

    Parameters
    ----------
    state
        Input state.
    theta
        Angle.
    mode
        Mode index.
    """
    return apply_symplectic(state, sy.rotation(theta), [mode])


def squeeze(state: GaussianState, r: float, phi: float = 0.0, mode: int = 0) -> GaussianState:
    """Squeezing ``S(r, φ)`` (``φ = 0`` squeezes ``x``).

    Parameters
    ----------
    state
        Input state.
    r
        Squeezing parameter.
    phi
        Squeezing angle.
    mode
        Mode index.
    """
    return apply_symplectic(state, sy.squeezing(r, phi), [mode])


def beamsplitter(state: GaussianState, theta: float, phi: float = 0.0, modes: Sequence[int] = (0, 1)) -> GaussianState:
    """Beam splitter ``BS(θ, φ)`` of transmissivity ``cos² θ``.

    Parameters
    ----------
    state
        Input state.
    theta, phi
        Mixing angle and phase.
    modes
        The two modes ``(a, b)``.
    """
    return apply_symplectic(state, sy.beamsplitter(theta, phi), modes)


def two_mode_squeeze(state: GaussianState, r: float, phi: float = 0.0, modes: Sequence[int] = (0, 1)) -> GaussianState:
    """Two-mode squeezing ``S2(r, φ)``.

    Parameters
    ----------
    state
        Input state.
    r, phi
        Squeezing parameter and phase.
    modes
        The two modes.
    """
    return apply_symplectic(state, sy.two_mode_squeezing(r, phi), modes)


def interferometer(state: GaussianState, U: npt.ArrayLike, modes: Sequence[int] | None = None) -> GaussianState:
    """Passive linear optics ``a_j → Σ_k U_jk a_k``.

    Parameters
    ----------
    state
        Input state.
    U
        Unitary on ``modes``.
    modes
        Modes it acts on (all by default).
    """
    return apply_symplectic(state, sy.interferometer(U), modes)


def cz(state: GaussianState, A: npt.ArrayLike, modes: Sequence[int] | None = None) -> GaussianState:
    """Controlled-phase network ``exp(i/2 Σ A_jk x_j x_k)``: ``p → p + A x`` (graph-state entangler).

    Parameters
    ----------
    state
        Input state.
    A
        Real symmetric weighted adjacency matrix on ``modes``.
    modes
        Modes it acts on (all by default).
    """
    return apply_symplectic(state, sy.cz(A), modes)


def cx(state: GaussianState, g: float, modes: Sequence[int] = (0, 1)) -> GaussianState:
    """``CX(g) = exp(−i g x_a p_b)``: ``x_b → x_b + g x_a``, ``p_a → p_a − g p_b``.

    Parameters
    ----------
    state
        Input state.
    g
        Gain.
    modes
        Control and target ``(a, b)``.
    """
    return apply_symplectic(state, sy.cx(g), modes)


def shear(state: GaussianState, s: float, mode: int = 0) -> GaussianState:
    """Shear ``P(s) = exp(i s x²/2)``: ``p → p + s x``.

    Parameters
    ----------
    state
        Input state.
    s
        Shear strength.
    mode
        Mode index.
    """
    return apply_symplectic(state, sy.shear(s), [mode])


def fourier(state: GaussianState, mode: int = 0) -> GaussianState:
    """Fourier gate ``F = R(π/2)``: ``x → −p``, ``p → x``.

    Parameters
    ----------
    state
        Input state.
    mode
        Mode index.
    """
    return apply_symplectic(state, sy.fourier(), [mode])
