"""Quadrature and ladder operators in position space; the Hermite transform."""

import numpy as np
import pytest

from qttlind.bosons import operators as bops
from qttlind.bosons import states as bs
from qttlind.cv import grids as gr
from qttlind.cv import operators as op
from qttlind.errors import CutoffWarning, InputError


def dist(a, b):
    """Norm of a difference of tensor trains. ``(a - b).norm()`` through ``inner`` cancels
    catastrophically; compressing first puts the norm in one core."""
    return (a - b).compress(0.0)[0].norm()


@pytest.fixture(scope="module")
def grid():
    return gr.QuadratureGrid(10)


def test_affine_tt(grid):
    f = op.affine_tt(6, 0.5, -3.0)
    assert f.max_rank == 2
    np.testing.assert_allclose(f.to_dense(), -3.0 + 0.5 * np.arange(64))


def test_position_is_exact_rank_two(grid):
    X = op.position(grid)
    assert X.max_rank == 2
    np.testing.assert_allclose(np.diag(X.to_dense()), grid.x())


def test_momentum_is_low_rank_for_any_grid():
    ranks = [op.momentum(gr.QuadratureGrid(R)).max_rank for R in (8, 12, 16)]
    assert max(ranks) <= 24


def test_canonical_commutator_on_a_band_limited_state(grid):
    psi = gr.coherent(0.7 + 0.4j, grid)
    X, P = op.position(grid), op.momentum(grid)
    assert dist(X.apply(P.apply(psi)) - P.apply(X.apply(psi)), psi * 1j) < 1e-8


@pytest.mark.parametrize("n", [0, 1, 5, 30, 100])
def test_number_and_annihilation_on_hermite_functions(grid, n):
    phi = gr.fock(n, grid, tol=1e-28)
    assert dist(op.number(grid).apply(phi), phi * n) < 1e-6 * max(1, n)
    if n:
        assert dist(op.annihilation(grid).apply(phi), gr.fock(n - 1, grid, tol=1e-28) * np.sqrt(n)) < 1e-6


def test_ladder_operators_have_small_flat_rank():
    for R in (8, 12):
        g = gr.QuadratureGrid(R)
        assert op.annihilation(g).max_rank <= 24
        assert op.number(g).max_rank <= 24


def test_creation_is_the_adjoint(grid):
    a = op.annihilation(grid)
    ad = op.creation(grid)
    psi, phi = gr.coherent(0.5, grid), gr.fock(2, grid)
    assert phi.inner(ad.apply(psi)) == pytest.approx(np.conj(psi.inner(a.apply(phi))), abs=1e-10)


def test_quadrature_of_coherent_state(grid):
    alpha = 1.0 + 0.5j
    psi = gr.coherent(alpha, grid)
    phi = np.pi / 3
    val = psi.inner(op.quadrature(grid, phi).apply(psi)).real
    assert val == pytest.approx(np.sqrt(2) * (np.cos(phi) * alpha.real + np.sin(phi) * alpha.imag), abs=1e-9)


def test_position_space_operators_agree_with_fock_space_operators(grid):
    """a in position space = H a_Fock H^dag on the span of the Fock states kept."""
    fb = 6
    psi_f = bs.coherent(1.2 - 0.3j, fb)
    a_fock = bops.annihilation(fb).apply(psi_f)
    via_x = op.position_to_fock(op.annihilation(grid).apply(op.fock_to_position(psi_f, grid)), grid, fb)
    assert dist(via_x, a_fock) < 1e-8


def test_hermite_transform_round_trip(grid):
    psi_f = bs.coherent(1.5 + 0.5j, 6)
    px = op.fock_to_position(psi_f, grid)
    assert dist(px, gr.coherent(1.5 + 0.5j, grid)) < 1e-8
    assert dist(op.position_to_fock(px, grid, 6), psi_f) < 1e-8


def test_hermite_transform_rank_is_large():
    """The design caveat of the CV extension: changing basis is O(N) -- the MPO rank is ~ the number of Fock states."""
    H = op.hermite_mpo(gr.QuadratureGrid(8), 6)
    assert H.max_rank >= 48


def test_position_to_fock_warns_when_the_state_does_not_fit(grid):
    with pytest.warns(CutoffWarning):
        op.position_to_fock(gr.coherent(4.0, grid), grid, 3)  # |alpha|^2 = 16 photons on 8 levels


def test_pad_fock():
    psi = bs.fock(3, 2)
    padded = op.pad_fock(psi, 5)
    np.testing.assert_array_equal(padded.to_dense(), np.eye(32)[3])
    with pytest.raises(InputError):
        op.pad_fock(psi, 1)


def test_hermite_mpo_guards():
    with pytest.raises(InputError):
        op.hermite_mpo(gr.QuadratureGrid(op.MAX_HERMITE_BITS + 1), 4)
    with pytest.raises(InputError):
        op.hermite_mpo(gr.QuadratureGrid(6), 7)
