"""Phase-space gates on quantics states.

**Diagonal gates.** ``exp(i f(x))`` is diagonal in position space and ``exp(i f(p))`` in momentum
space (one Fourier transform away). Measured fact: the *operator* ``exp(i s x^2/2)`` over the whole
box has quantics rank growing like ``2^{R/2}``, but the *state* it produces from a localised
wavefunction keeps a small rank. We therefore multiply element-wise (``TensorTrain.hadamard``) and
compress immediately; the intermediate rank is the product of the two ranks.

**Linear phases are rank 1.** ``exp(i c x_j)`` with ``x_j`` affine in the bits factorises over
the bits: displacements cost two rank-1 products and two Fourier transforms::

    D(alpha) = exp(i p0 x) exp(-i x0 p) exp(-i x0 p0 / 2),   (x0, p0) = sqrt(2) (Re alpha, Im alpha).

**Gaussian gates from shears.** With the shears ``U_x(s) = exp(i s x^2/2)`` (``p -> p + s x``,
matrix ``M_x(s) = [[1, 0], [s, 1]]``) and ``U_p(t) = exp(i t p^2/2)`` (``x -> x - t p``,
``M_p(t) = [[1, -t], [0, 1]]``), every 2 x 2 symplectic ``S = [[a, b], [c, d]]`` factorises exactly:

* if ``c != 0``: ``S = M_p((1 - a)/c) M_x(c) M_p((1 - d)/c)``;
* if ``b != 0``: ``S = M_x((d - 1)/b) M_p(-b) M_x((a - 1)/b)``;
* always: ``S = M_x(s0) S'`` with ``s0 = (c - c')/a`` (``c' = +-1``), then factorise ``S'``.

The three are exact but not equally good *on a grid*: a large shear spreads the state over many
momenta and aliases. :func:`shear_decomposition` returns the candidate with the smallest
largest parameter. Rotations, squeezers and every single-mode Gaussian unitary are therefore at
most four diagonal phases and a few Fourier transforms, on any grid -- and the tests check
them against the Gaussian backend, including random ones drawn by Hypothesis.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..bosons.states import from_function
from ..core.tt import TensorTrain
from ..errors import InputError
from ..gaussian.ops import rotation_matrix, squeezing_matrix
from ..gaussian.phase_space import is_symplectic
from .fourier import to_momentum, to_position
from .grids import QuadratureGrid

__all__ = [
    "GATE_TOL",
    "cubic_phase",
    "displace",
    "kerr",
    "linear_phase",
    "phase",
    "rotate",
    "shear_decomposition",
    "shear_p",
    "shear_x",
    "squeeze",
    "symplectic",
]

#: Discarded weight after every gate (relative error 1e-11 per bond).
GATE_TOL = 1e-22


def linear_phase(bits: int, slope: float, offset: float) -> TensorTrain:
    """Exact rank-1 tensor train of ``exp(i (offset + slope * j))``, ``j`` on ``bits`` bits."""
    vecs = [np.array([1.0, np.exp(1j * slope * 2.0 ** (bits - 1 - i))]) for i in range(bits)]
    vecs[0] = vecs[0] * np.exp(1j * offset)
    return TensorTrain.product(vecs)


def _multiply(psi: TensorTrain, f: TensorTrain, tol: float) -> TensorTrain:
    out, _ = psi.hadamard(f).compress(tol)
    return out


def phase(
    psi: TensorTrain,
    f: Callable[[np.ndarray], np.ndarray],
    grid: QuadratureGrid,
    basis: str = "x",
    tol: float = GATE_TOL,
) -> TensorTrain:
    """``exp(i f(x))`` (``basis="x"``) or ``exp(i f(p))`` (``basis="p"``) on a position-space state."""
    if basis not in ("x", "p"):
        raise InputError(f"basis must be 'x' or 'p', got {basis!r}")
    step = grid.dx if basis == "x" else grid.dp
    q0 = -grid.size / 2 * step
    ph = from_function(lambda j: np.exp(1j * f(q0 + j * step)), grid.bits, tol)
    if basis == "x":
        return _multiply(psi, ph, tol)
    return to_position(_multiply(to_momentum(psi, tol), ph, tol), tol)


def displace(psi: TensorTrain, alpha: complex, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """``D(alpha)``: a rank-1 phase in momentum space, then a rank-1 phase in position space."""
    x0, p0 = np.sqrt(2) * np.real(alpha), np.sqrt(2) * np.imag(alpha)
    N = grid.size
    out = psi
    if x0 != 0:
        kick = linear_phase(grid.bits, -x0 * grid.dp, x0 * N / 2 * grid.dp)  # exp(-i x0 p_k)
        out = to_position(_multiply(to_momentum(out, tol), kick, tol), tol)
    if p0 != 0:
        out = _multiply(out, linear_phase(grid.bits, p0 * grid.dx, -p0 * N / 2 * grid.dx), tol)  # exp(i p0 x_j)
    return out * np.exp(-0.5j * x0 * p0)


def shear_x(psi: TensorTrain, s: float, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """``exp(i s x^2 / 2)``: ``p -> p + s x``."""
    return phase(psi, lambda x: s * x**2 / 2, grid, "x", tol)


def shear_p(psi: TensorTrain, t: float, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """``exp(i t p^2 / 2)``: ``x -> x - t p``."""
    return phase(psi, lambda p: t * p**2 / 2, grid, "p", tol)


def shear_decomposition(S: np.ndarray) -> list[tuple[str, float]]:
    """Shears ``[("p" | "x", value), ...]`` in the order they are **applied**, whose product is ``S``.

    Among the exact factorisations of the module docstring, return the one whose largest
    parameter is smallest (large shears alias on a finite grid).
    """
    S = np.asarray(S, dtype=float)
    if S.shape != (2, 2) or not is_symplectic(S, atol=1e-9):
        raise InputError("expected a 2 x 2 symplectic matrix (det = 1)")

    def three(M: np.ndarray) -> list[list[tuple[str, float]]]:
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        out = []
        if abs(c) > 1e-12:  # applied right to left: M_p(t2) first
            out.append([("p", (1 - d) / c), ("x", c), ("p", (1 - a) / c)])
        if abs(b) > 1e-12:
            out.append([("x", (a - 1) / b), ("p", -b), ("x", (d - 1) / b)])
        return out

    candidates = three(S)
    if abs(S[0, 0]) > 1e-12:
        for target in (1.0, -1.0):
            s0 = (S[1, 0] - target) / S[0, 0]
            rest = np.array([[1.0, 0.0], [-s0, 1.0]]) @ S  # M_x(-s0) S has c' = target
            candidates += [steps + [("x", s0)] for steps in three(rest)]
    best = min(candidates, key=lambda steps: max(abs(v) for _, v in steps))
    return [(kind, float(v)) for kind, v in best if v != 0.0]


def symplectic(psi: TensorTrain, S: np.ndarray, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """Apply the single-mode Gaussian unitary with symplectic matrix ``S`` (up to a global phase)."""
    out = psi
    for kind, value in shear_decomposition(S):
        out = shear_x(out, value, grid, tol) if kind == "x" else shear_p(out, value, grid, tol)
    return out


def rotate(psi: TensorTrain, theta: float, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """``R(theta) = exp(i theta n)`` up to a global phase."""
    return symplectic(psi, rotation_matrix(theta), grid, tol)


def squeeze(psi: TensorTrain, r: float, grid: QuadratureGrid, phi: float = 0.0, tol: float = GATE_TOL) -> TensorTrain:
    """``S(r, phi)`` up to a global phase."""
    return symplectic(psi, squeezing_matrix(r, phi), grid, tol)


def cubic_phase(psi: TensorTrain, gamma: float, grid: QuadratureGrid, tol: float = GATE_TOL) -> TensorTrain:
    """``V(gamma) = exp(i gamma x^3 / 3)``: the standard non-Gaussian resource gate."""
    return phase(psi, lambda x: gamma * x**3 / 3, grid, "x", tol)


def kerr(psi_fock: TensorTrain, kappa: float, tol: float = GATE_TOL) -> TensorTrain:
    """``K(kappa) = exp(i kappa n^2)`` on a **Fock-space** tensor train (diagonal there)."""
    ph = from_function(lambda n: np.exp(1j * kappa * n.astype(float) ** 2), psi_fock.length, tol)
    return _multiply(psi_fock, ph, tol)
