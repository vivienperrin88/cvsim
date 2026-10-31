"""Photon counting on Gaussian states: Fock-basis elements, PNR and click statistics, detector
models, heralding.

Notation (ħ = 1). ``β = W d = (α, ᾱ)`` and ``σ = W V W†`` are the mean and covariance in the
complex basis ``(a, a†)`` (``W`` of CONVENTIONS.md); ``Q = σ + I/2`` is the covariance of the Husimi
function; ``X = [[0, I], [I, 0]]``; ``A = X (I − Q⁻¹)`` and ``γ = X Q⁻¹ β``. Then

* ``⟨n|ρ|m⟩ = T · lhaf(A_{(m ⊕ n)}; γ_{(m ⊕ n)}) / √(n! m!)`` with ``T = e^{−½ β† Q⁻¹ β} / √det Q``,
  where ``A_{(m ⊕ n)}`` repeats row and column ``j`` ``m_j`` times in the first half and ``n_j`` times
  in the second (Hamilton et al., PRL 119, 170501; Kruse et al., PRA 100, 032326);
* the elements are computed together by the recurrence of that generating function,
  ``√(k_i + 1) g_{k + e_i} = γ_i g_k + Σ_j A_ij √k_j g_{k − e_j}`` (:func:`fock_density_matrix`);
* a single pattern ``p(n) = T lhaf(A_{(n ⊕ n)}; γ_{(n ⊕ n)}) / n!`` (:func:`pattern_probability`);
* click statistics by inclusion–exclusion over vacuum probabilities,
  ``P(clicks on S only) = Σ_{Z ⊆ S} (−1)^{|Z|} P₀(Sᶜ ∪ Z)`` — the torontonian for zero mean,
  valid with displacement too (:func:`click_probability`).

**Detector models** (the same as MrMustard's): efficiency ``η`` is loss before an ideal detector
(applied exactly to the Gaussian state); PNR dark counts are Poissonian with mean ``ν`` added to
the count; a click detector fires spuriously with probability ``p_dark``. Distributions are
returned up to a photon-number ``cutoff``; their sum falls short of 1 by the population above it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from ..core import conventions as cv
from ..core import linalg as la
from ..core.hafnian import loop_hafnian
from ..errors import InputError
from .channels import GaussianChannel
from .states import GaussianState

__all__ = [
    "HeraldResult",
    "click_distribution",
    "click_probability",
    "fock_density_matrix",
    "herald",
    "husimi_parameters",
    "pattern_probability",
    "pnr_distribution",
]


def husimi_parameters(state: GaussianState) -> tuple[np.ndarray, np.ndarray, complex]:
    """``(A, γ, T)`` of the module docstring: the Fock-basis generating function of ``state``.

    Parameters
    ----------
    state
        Gaussian state.
    """
    M = state.num_modes
    W = cv.complex_basis(M)
    beta = W @ state.mean
    sigma = W @ state.cov @ W.conj().T
    Q = sigma + np.eye(2 * M) * cv.VACUUM_VARIANCE
    Qi = la.inv(Q)
    X = np.block([[np.zeros((M, M)), np.eye(M)], [np.eye(M), np.zeros((M, M))]])
    A = X @ (np.eye(2 * M) - Qi)
    gamma = X @ Qi @ beta
    detQ = complex(np.prod(la.eigvals(Q)))
    T = np.exp(-0.5 * (beta.conj() @ Qi @ beta)) / np.sqrt(detQ)
    return (A + A.T) / 2, gamma, complex(T)


def _check_modes(state: GaussianState, pattern: Sequence[int]) -> np.ndarray:
    n = np.asarray(pattern)
    if n.shape != (state.num_modes,) or not np.issubdtype(n.dtype, np.integer) or np.any(n < 0):
        raise InputError(f"pattern must hold {state.num_modes} non-negative integers, got {list(pattern)}")
    return n.astype(int)


def pattern_probability(state: GaussianState, pattern: Sequence[int]) -> float:
    """Probability of the photon-number pattern ``n`` (one count per mode).

    Parameters
    ----------
    state
        Gaussian state.
    pattern
        Photon number of every mode.
    """
    n = _check_modes(state, pattern)
    A, gamma, T = husimi_parameters(state)
    M = state.num_modes
    idx = [j for j in range(M) for _ in range(n[j])]
    idx = idx + [M + j for j in idx]
    lh = loop_hafnian(A[np.ix_(idx, idx)], gamma[idx]) if idx else 1.0
    return float(np.real(T * lh)) / float(np.prod([math.factorial(int(k)) for k in n]))


def fock_density_matrix(state: GaussianState, cutoff: int) -> np.ndarray:
    """Density matrix in the Fock basis, photon numbers ``0..cutoff−1`` per mode.

    Returns a ``cutoff^M x cutoff^M`` matrix, mode 0 most significant (the layout of
    ``cvsim.backends.FockState``).

    Parameters
    ----------
    state
        Gaussian state.
    cutoff
        Levels per mode.
    """
    if not isinstance(cutoff, (int, np.integer)) or cutoff < 1:
        raise InputError(f"cutoff must be a positive integer, got {cutoff!r}")
    A, gamma, T = husimi_parameters(state)
    M = state.num_modes
    D = 2 * M
    g = np.zeros((cutoff,) * D, dtype=complex)
    g[(0,) * D] = T
    sq = np.sqrt(np.arange(cutoff + 1))
    for total in range(1, D * (cutoff - 1) + 1):
        for k in _compositions(total, D, cutoff - 1):
            i = next(j for j, kj in enumerate(k) if kj > 0)
            kp = list(k)
            kp[i] -= 1
            val = gamma[i] * g[tuple(kp)]
            for j in range(D):
                if kp[j] > 0:
                    kpp = list(kp)
                    kpp[j] -= 1
                    val += A[i, j] * sq[kp[j]] * g[tuple(kpp)]
            g[k] = val / sq[k[i]]
    # g[m (first half), n (second half)] = ⟨n|ρ|m⟩: ket index = second half
    order = [*range(M, D), *range(M)]
    return g.transpose(order).reshape(cutoff**M, cutoff**M)


def _compositions(total: int, parts: int, cap: int) -> Iterator[tuple[int, ...]]:
    """All tuples of ``parts`` integers in ``[0, cap]`` summing to ``total``."""
    if parts == 1:
        if total <= cap:
            yield (total,)
        return
    for first in range(min(total, cap) + 1):
        for rest in _compositions(total - first, parts - 1, cap):
            yield (first, *rest)


# ------------------------------------------------------------------------------- distributions and detectors
def _lossy(state: GaussianState, modes: Sequence[int], efficiency: float) -> GaussianState:
    if not 0.0 < efficiency <= 1.0:
        raise InputError(f"efficiency must be in (0, 1], got {efficiency}")
    out = state
    if efficiency < 1.0:
        for m in modes:
            out = GaussianChannel.loss(efficiency).apply(out, [m])
    return out


def _poisson(nu: float, cutoff: int) -> np.ndarray:
    k = np.arange(cutoff)
    return np.asarray(np.exp(-nu) * nu**k / np.array([math.factorial(int(x)) for x in k], dtype=float))


def pnr_distribution(
    state: GaussianState, modes: Sequence[int], cutoff: int, efficiency: float = 1.0, dark_counts: float = 0.0
) -> np.ndarray:
    """Joint photon-count distribution of ``modes``, counts ``0..cutoff−1``, shape ``(cutoff,)*len(modes)``.

    Parameters
    ----------
    state
        Gaussian state.
    modes
        Detected modes (the others are traced out).
    cutoff
        Largest count + 1.
    efficiency
        Detection efficiency (loss before an ideal detector).
    dark_counts
        Mean number of dark counts per detector (Poissonian, added to the count).
    """
    if not dark_counts >= 0:
        raise InputError(f"dark_counts must be >= 0, got {dark_counts}")
    red = _lossy(state, modes, efficiency).reduce(list(modes))
    k = len(modes)
    probs = np.real(np.diag(fock_density_matrix(red, cutoff))).reshape((cutoff,) * k)
    if dark_counts > 0:
        kernel = _poisson(dark_counts, cutoff)
        for axis in range(k):
            probs = np.apply_along_axis(lambda v: np.convolve(v, kernel)[:cutoff], axis, probs)
    return np.asarray(np.clip(probs, 0.0, None))


def click_probability(state: GaussianState, clicks: Sequence[int], dark_prob: float = 0.0) -> float:
    """Probability that exactly the modes with ``clicks[j] = 1`` click (threshold detectors on every mode).

    Parameters
    ----------
    state
        Gaussian state (detection efficiency: apply a loss channel first, or use :func:`click_distribution`).
    clicks
        0/1 per mode.
    dark_prob
        Probability that a detector clicks with no light.
    """
    c = _check_modes(state, clicks)
    if np.any(c > 1):
        raise InputError(f"clicks must be 0 or 1, got {list(clicks)}")
    if not 0.0 <= dark_prob < 1.0:
        raise InputError(f"dark_prob must be in [0, 1), got {dark_prob}")
    M = state.num_modes
    S = [j for j in range(M) if c[j]]
    off = [j for j in range(M) if not c[j]]
    total = 0.0
    for r in range(len(S) + 1):
        for Z in itertools.combinations(S, r):
            quiet = sorted([*off, *Z])
            p0 = state.vacuum_probability(quiet) if quiet else 1.0
            total += (-1) ** r * (1.0 - dark_prob) ** len(quiet) * p0
    return max(0.0, float(total))


def click_distribution(
    state: GaussianState, modes: Sequence[int], efficiency: float = 1.0, dark_prob: float = 0.0
) -> np.ndarray:
    """Joint click distribution of ``modes``, shape ``(2,)*len(modes)`` (index 1 = click).

    Parameters
    ----------
    state
        Gaussian state.
    modes
        Detected modes.
    efficiency
        Detection efficiency.
    dark_prob
        Dark-click probability per detector.
    """
    red = _lossy(state, modes, efficiency).reduce(list(modes))
    k = len(modes)
    out = np.zeros((2,) * k)
    for pattern in itertools.product((0, 1), repeat=k):
        out[pattern] = click_probability(red, pattern, dark_prob)
    return out


class HeraldResult(NamedTuple):
    """Heralded state of the unmeasured modes (``modes``, Fock basis, ``cutoff`` levels each) and
    the probability of the heralding pattern."""

    probability: float
    dm: np.ndarray
    modes: tuple[int, ...]
    cutoff: int


def herald(
    state: GaussianState,
    modes: Sequence[int],
    pattern: Sequence[int],
    cutoff: int,
    efficiency: float = 1.0,
    dark_counts: float = 0.0,
) -> HeraldResult:
    """Condition ``state`` on PNR outcome ``pattern`` on ``modes``; return the (non-Gaussian)
    state of the other modes in the Fock basis.

    Parameters
    ----------
    state
        Gaussian state.
    modes
        Heralding modes.
    pattern
        Counts observed on ``modes``.
    cutoff
        Fock levels for every mode (must exceed every count).
    efficiency
        Detection efficiency of the heralding detectors.
    dark_counts
        Mean dark counts per heralding detector.
    """
    ms = [int(m) for m in modes]
    pat = [int(n) for n in pattern]
    if len(pat) != len(ms) or any(n < 0 for n in pat):
        raise InputError(f"pattern must give one non-negative count per heralding mode, got {pattern}")
    if max(pat, default=0) >= cutoff:
        raise InputError(f"cutoff {cutoff} must exceed every heralded count {pat}")
    if not dark_counts >= 0:
        raise InputError(f"dark_counts must be >= 0, got {dark_counts}")
    M = state.num_modes
    rest = [m for m in range(M) if m not in ms]
    if not rest:
        raise InputError("herald needs at least one unmeasured mode")
    lossy = _lossy(state, ms, efficiency)
    order = [*ms, *rest]
    dm = fock_density_matrix(lossy.reduce(order), cutoff)
    k, r = len(ms), len(rest)
    t = dm.reshape((cutoff,) * (2 * (k + r)))
    kernel = _poisson(dark_counts, cutoff) if dark_counts > 0 else np.eye(cutoff)[0]
    # POVM element for count n: Σ_j P(dark = n − j) |j⟩⟨j|
    out = np.zeros((cutoff,) * (2 * r), dtype=complex)
    for js in itertools.product(*(range(n + 1) for n in pat)):
        w = float(np.prod([kernel[n - j] for n, j in zip(pat, js)]))
        if w == 0.0:
            continue
        index: tuple[int | slice, ...] = (*js, *(slice(None),) * r, *js, *(slice(None),) * r)
        out = out + w * np.asarray(t[index], dtype=complex)
    flat = out.reshape(cutoff**r, cutoff**r)
    p = float(np.real(np.trace(flat)))
    return HeraldResult(p, flat / p if p > 0 else flat, tuple(rest), int(cutoff))
