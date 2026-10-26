"""Measurements and sampling on quantics states.

**Probabilities are tensor trains too.** ``P_j = |c_j|^2`` is the Hadamard product of a tensor
train with its complex conjugate (rank squared, then compressed), and any diagonal observable
``sum_j f_j P_j`` is one more Hadamard product and a sum over all indices -- the sum of a tensor
train contracts every core with ``(1, 1)``: ``O(R chi^2)``.

**Perfect sampling, most significant bit first.** In the canonical form with centre 0, every
core to the right of site ``k`` is a right isometry, so the marginal of the first ``k + 1`` bits
is::

    p(s_1..s_{k+1}) = || e_{s_1..s_k} A_{k+1}[:, s_{k+1}, :] ||^2,

where ``e`` is the (normalised) row vector accumulated along the bits already drawn. Drawing
the bits one after the other gives *exact* samples of ``|c_j|^2`` -- first the coarse position
(which half of the box), then finer and finer: the quantics hierarchy is the sampling order.
Drawing only the first ``grid.bits`` sites of a longer tensor train samples the **marginal** of
the first mode: this is a homodyne measurement of mode 0 of a two-mode chain, and the
conditional state of mode 1 is what remains after fixing those sites.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..bosons.states import DEFAULT_TOL
from ..core.conventions import bits_of
from ..core.tt import TensorTrain
from ..errors import InputError
from .fourier import to_momentum
from .grids import QuadratureGrid, hermite_functions
from .operators import affine_tt

__all__ = [
    "HomodyneResult",
    "conj",
    "expectation_diagonal",
    "fock_probabilities",
    "homodyne",
    "interval_probability",
    "probabilities",
    "quadrature_moments",
    "sample",
    "total",
]


class HomodyneResult(NamedTuple):
    outcome: float  # grid point x_j that was measured
    density: float  # probability density at that point (probability / dx)
    state: TensorTrain | None  # normalised conditional state of the remaining sites (None if none)


def conj(tt: TensorTrain) -> TensorTrain:
    """Element-wise complex conjugate (conjugate every core)."""
    return TensorTrain([c.conj() for c in tt.cores], tt.center)


def total(tt: TensorTrain) -> complex:
    """``sum_j t_j``: contract every core with the all-ones vector."""
    env = np.ones((1,), dtype=complex)
    for c in tt.cores:
        env = env @ c.sum(axis=1)
    return complex(env[0])


def probabilities(psi: TensorTrain, tol: float = DEFAULT_TOL) -> TensorTrain:
    """``|c_j|^2`` as a compressed tensor train (real up to rounding)."""
    out, _ = conj(psi).hadamard(psi).compress(tol)
    return out


def expectation_diagonal(psi: TensorTrain, f: TensorTrain, tol: float = DEFAULT_TOL) -> float:
    """``sum_j f_j |c_j|^2 / sum_j |c_j|^2`` for a diagonal observable given as a tensor train ``f``."""
    P = probabilities(psi, tol)
    return float((total(P.hadamard(f)) / total(P)).real)


def quadrature_moments(psi: TensorTrain, grid: QuadratureGrid) -> tuple[float, float, float, float]:
    """``(<x>, Var x, <p>, Var p)``; ``p`` through the quantics Fourier transform."""
    out = []
    for amps, step in ((psi, grid.dx), (to_momentum(psi), grid.dp)):
        q = affine_tt(grid.bits, step, -grid.size / 2 * step)
        mean = expectation_diagonal(amps, q)
        second = expectation_diagonal(amps, q.hadamard(q))
        out += [mean, second - mean**2]
    return out[0], out[1], out[2], out[3]


def interval_probability(psi: TensorTrain, indicator: TensorTrain) -> float:
    """Probability of the set where ``indicator`` (a 0/1 tensor train) is 1, e.g. GKP bins."""
    return expectation_diagonal(psi, indicator)


def sample(psi: TensorTrain, shots: int, rng: np.random.Generator, nsites: int | None = None) -> np.ndarray:
    """``shots`` exact samples of the index ``j`` of the first ``nsites`` sites (all by default).

    Returns integers in ``0 .. 2^nsites - 1`` (bits read MSB first) drawn from the marginal of
    ``|c|^2`` on those sites.
    """
    n = psi.length if nsites is None else int(nsites)
    if not 1 <= n <= psi.length:
        raise InputError(f"nsites must be in 1..{psi.length}, got {nsites}")
    if any(d != 2 for d in psi.dims[:n]):
        raise InputError("sample expects binary (quantics) sites")
    tt = psi.canonicalize(0)
    out = np.zeros(shots, dtype=np.int64)
    for shot in range(shots):
        env = np.ones((1,), dtype=complex)
        value = 0
        for k in range(n):
            v = [env @ tt.cores[k][:, s, :] for s in (0, 1)]
            w = np.array([np.vdot(x, x).real for x in v])
            s = int(rng.random() * w.sum() >= w[0])
            env = v[s] / np.sqrt(w[s])
            value = 2 * value + s
        out[shot] = value
    return out


def homodyne(
    psi: TensorTrain,
    grid: QuadratureGrid,
    rng: np.random.Generator | None = None,
    outcome: float | None = None,
) -> HomodyneResult:
    """Measure ``x`` of the mode on the first ``grid.bits`` sites of ``psi``.

    Pass ``rng`` to sample the outcome, or ``outcome`` (a position, snapped to the nearest grid
    point) to post-select it. The remaining sites, if any, form the conditional state.
    """
    if (rng is None) == (outcome is None):
        raise InputError("pass exactly one of `rng` (sampling) or `outcome` (post-selection)")
    R = grid.bits
    if psi.length < R:
        raise InputError(f"state has {psi.length} sites, fewer than the {R} bits of the measured mode")
    if rng is not None:
        j = int(sample(psi, 1, rng, R)[0])
    else:
        j = int(np.clip(np.rint(float(outcome) / grid.dx + grid.size / 2), 0, grid.size - 1))  # type: ignore[arg-type]
    env = np.ones((1,), dtype=complex)
    for k, s in enumerate(bits_of(j, R)):
        env = env @ psi.cores[k][:, s, :]
    x_j = (j - grid.size / 2) * grid.dx
    norm2 = psi.norm() ** 2
    if psi.length == R:
        return HomodyneResult(x_j, float(abs(env[0]) ** 2 / norm2 / grid.dx), None)
    cores = [c.copy() for c in psi.cores[R:]]
    cores[0] = np.einsum("l,lsr->sr", env, cores[0])[None]
    rest = TensorTrain(cores, copy=False)
    p = rest.norm() ** 2 / norm2
    if p == 0:
        raise InputError("the post-selected outcome has probability zero")
    return HomodyneResult(x_j, float(p / grid.dx), rest * (1.0 / rest.norm()))


def fock_probabilities(psi: TensorTrain, grid: QuadratureGrid, nmax: int, tol: float = DEFAULT_TOL) -> np.ndarray:
    """``|<n|psi>|^2`` for ``n < nmax`` (photon-number-resolving detection of a position-space state)."""
    x = grid.x()
    rows = hermite_functions(nmax, x) * np.sqrt(grid.dx)
    norm2 = psi.norm() ** 2
    out = np.empty(nmax)
    for n in range(nmax):
        phi = TensorTrain.from_dense(rows[n].astype(complex), [2] * grid.bits, tol)
        out[n] = abs(phi.inner(psi)) ** 2 / norm2
    return out
