"""Comparison with The Walrus: same decompositions, different gauge (CONVENTIONS.md, "Comparison with
other libraries").

The Walrus (``thewalrus.decompositions``; a dependency of Strawberry Fields and MrMustard, see
CONVENTIONS.md for what each of those does) and ``cvsim.core.linalg`` compute the same mathematical
objects. They may differ only by the freedom each decomposition leaves:

* ordering (The Walrus returns Williamson's ``ν`` in LAPACK's Schur order, cvsim ascending);
* the squeezed/anti-squeezed block order of Bloch–Messiah (The Walrus ``diag(e^{r}, e^{−r})``,
  cvsim ``diag(e^{−r}, e^{r})``): a fixed rotation by π/2 of every supermode;
* inside each degenerate group, the gauge rotation that cvsim fixes by Procrustes and The Walrus
  leaves to LAPACK.

Each test computes the relating matrix and checks that it has exactly that form, so any other
difference -- a transposed factor, a wrong sign convention, a missing permutation -- fails. Values
(ν, r, s) must agree to round-off; factor relations to ``FACTOR`` (they go through two
decompositions of the same matrix, each accurate to ~1e-13 for these sizes and conditions).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import scipy.linalg as sla
from harness import ROUNDOFF, assert_same, random_unitary

from cvsim.core import conventions as cv
from cvsim.core import linalg as cl

pytestmark = pytest.mark.thewalrus

FACTOR = 1e-9  # relations between factors computed by two different algorithms


@pytest.fixture(scope="module")
def tw():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus.decompositions")


# ------------------------------------------------------------------------------- inputs
def random_symplectic(rng: np.random.Generator, M: int, scale: float = 0.6) -> np.ndarray:
    """``expm(Ω H)`` with a random symmetric ``H``: a generic (active) symplectic matrix."""
    H = rng.normal(size=(2 * M, 2 * M))
    return sla.expm(cv.omega(M) @ (H + H.T) * (scale / 2))


def passive(U: np.ndarray) -> np.ndarray:
    return np.block([[U.real, -U.imag], [U.imag, U.real]])


def covariance(rng: np.random.Generator, nu: np.ndarray) -> np.ndarray:
    S = random_symplectic(rng, len(nu))
    return S @ np.diag(np.concatenate([nu, nu])) @ S.T


# ------------------------------------------------------------------------------- gauge checks
def assert_orthogonal_symplectic(K: np.ndarray, what: str) -> None:
    M = K.shape[0] // 2
    assert_same(K @ K.T, np.eye(2 * M), FACTOR, f"{what}: orthogonal")
    assert_same(K.T @ cv.omega(M) @ K, cv.omega(M), FACTOR, f"{what}: symplectic")


def assert_within_groups(K: np.ndarray, groups: list[np.ndarray], what: str) -> None:
    """``K`` (2M x 2M, xxpp) acts inside the degenerate groups only: zero between different groups."""
    M = K.shape[0] // 2
    label = np.empty(M, dtype=int)
    for i, g in enumerate(groups):
        label[g] = i
    lab2 = np.concatenate([label, label])
    outside = lab2[:, None] != lab2[None, :]
    assert np.abs(K[outside]).max(initial=0.0) <= FACTOR, f"{what}: mixes different degeneracy groups"


# ------------------------------------------------------------------------------- Williamson
@pytest.mark.parametrize(
    "nu",
    [
        np.array([0.6, 1.3, 2.1]),  # generic
        np.array([0.5, 0.5, 0.5]),  # pure state: one group, gauge U(3)
        np.array([0.7, 0.7, 1.9]),  # partly degenerate
    ],
    ids=["generic", "pure", "degenerate"],
)
def test_williamson_same_up_to_ordering_and_gauge(tw, rng, nu):
    V = covariance(rng, nu)
    ours = cl.williamson(V)
    Db, S_tw = tw.williamson(V)
    nu_tw = np.diag(Db)[: len(nu)]
    perm = np.argsort(nu_tw, kind="stable")
    assert_same(nu_tw[perm], ours.nu, ROUNDOFF, "symplectic eigenvalues")
    assert_same(S_tw @ Db @ S_tw.T, V, FACTOR, "The Walrus reconstructs V = S D Sᵀ")
    # relating matrix: S_tw (sorted) = S_ours K, with K a passive rotation inside each group
    M = len(nu)
    cols = np.concatenate([perm, perm + M])
    K = cl.solve(ours.S, S_tw[:, cols])
    assert_orthogonal_symplectic(K, "Williamson gauge")
    assert_within_groups(K, ours.groups, "Williamson gauge")


def test_symplectic_eigenvalues_and_hbar(tw, rng):
    """The Walrus works in ħ = 2: the same state has ν doubled, vacuum ν = 1."""
    V = covariance(rng, np.array([0.55, 0.9, 1.4]))
    _, V2 = cv.convert_hbar(np.zeros(6), V, hbar_from=1.0, hbar_to=2.0)
    assert_same(np.sort(tw.symplectic_eigenvals(V)), cl.symplectic_eigenvalues(V), ROUNDOFF, "ν, same matrix")
    assert_same(np.sort(tw.symplectic_eigenvals(V2)), 2 * cl.symplectic_eigenvalues(V), ROUNDOFF, "ν, ħ = 2")
    assert_same(tw.symplectic_eigenvals(np.eye(4)), np.ones(2), ROUNDOFF, "vacuum, ħ = 2")


# ------------------------------------------------------------------------------- Bloch–Messiah
def _bloch_messiah_relation(tw, S):
    ours = cl.bloch_messiah(S)
    O, D, Q = tw.blochmessiah(S)
    M = S.shape[0] // 2
    assert_same(O @ D @ Q, S, FACTOR, "The Walrus reconstructs S = O D Q")
    # The Walrus: D = diag(e^{r}, e^{−r}), r descending -- anti-squeezed quadrature in the first block.
    assert_same(np.diag(D), np.concatenate([np.exp(ours.r), np.exp(-ours.r)]), FACTOR, "squeezing")
    # Rotating every supermode by π/2 swaps the blocks: Ω diag(e^{r}, e^{−r}) Ωᵀ = diag(e^{−r}, e^{r}).
    K = cv.omega(M).T @ ours.O1.T @ O
    assert_orthogonal_symplectic(K, "Bloch–Messiah gauge")
    assert_within_groups(K, ours.groups, "Bloch–Messiah gauge")
    D_ours = np.diag(np.concatenate([np.exp(-ours.r), np.exp(ours.r)]))
    assert_same(K @ D_ours, D_ours @ K, FACTOR, "gauge commutes with the squeezer")
    return ours, (O, D, Q)


def test_bloch_messiah_generic(tw, rng):
    _bloch_messiah_relation(tw, random_symplectic(rng, 3))


def test_bloch_messiah_degenerate_two_mode_squeezer(tw, rng):
    """A two-mode squeezer has two equal squeezing parameters: the supermodes form one O(2) group."""
    r = 0.5
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    S = sla.block_diag(np.cosh(r) * np.eye(2) + np.sinh(r) * X, np.cosh(r) * np.eye(2) - np.sinh(r) * X)
    U = passive(random_unitary(rng, 2))
    ours, _ = _bloch_messiah_relation(tw, U @ S @ U.T)
    assert len(ours.groups) == 1 and len(ours.groups[0]) == 2


def test_bloch_messiah_passive_input(tw, rng):
    """No squeezing: both libraries put the whole matrix in the second factor (SF puts it in the first)."""
    S = passive(random_unitary(rng, 3))
    ours = cl.bloch_messiah(S)
    O, D, Q = tw.blochmessiah(S)
    assert_same(ours.r, np.zeros(3), ROUNDOFF, "r")
    for name, A in [("cvsim O1", ours.O1), ("The Walrus O", O), ("The Walrus D", D)]:
        assert_same(A, np.eye(6), FACTOR, name)
    assert_same(ours.O2, S, FACTOR, "cvsim O2")
    assert_same(Q, S, FACTOR, "The Walrus Q")


# ------------------------------------------------------------------------------- Takagi
@pytest.mark.parametrize("kind", ["complex", "real", "degenerate"])
def test_takagi_same_up_to_gauge(tw, rng, kind):
    n = 4
    if kind == "degenerate":
        U0 = random_unitary(rng, n)
        A = U0 @ np.diag([2.0, 2.0, 1.0, 0.3]) @ U0.T
    else:
        A = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if kind == "complex" else 0)
        A = A + A.T
    ours = cl.takagi(A)
    s_tw, U_tw = tw.takagi(A)
    assert_same(s_tw, ours.s, ROUNDOFF * max(1.0, float(ours.s[0])), "Takagi values, descending in both")
    assert_same(U_tw @ np.diag(s_tw) @ U_tw.T, A, FACTOR, "The Walrus reconstructs A = U diag(s) Uᵀ")
    # U_tw = U_ours k with k real orthogonal inside each group of equal s > 0 (±1 when non-degenerate)
    k = ours.U.conj().T @ U_tw
    assert_same(k.imag, np.zeros((n, n)), FACTOR, "Takagi gauge is real")
    assert_same(k.real @ k.real.T, np.eye(n), FACTOR, "Takagi gauge is orthogonal")
    label = np.empty(n, dtype=int)
    for i, g in enumerate(ours.groups):
        label[g] = i
    assert np.abs(k[label[:, None] != label[None, :]]).max(initial=0.0) <= FACTOR, "gauge mixes groups"
