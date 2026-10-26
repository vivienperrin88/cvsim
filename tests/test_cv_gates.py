"""Phase-space gates on quantics states, cross-checked with the Gaussian backend."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qttlind.bosons import states as bs
from qttlind.cv import gates as ga
from qttlind.cv import grids as gr
from qttlind.errors import InputError
from qttlind.gaussian import GaussianState, ops
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr


def infidelity(a, b):
    return 1 - abs(a.inner(b)) ** 2 / (a.norm() ** 2 * b.norm() ** 2)


@pytest.fixture(scope="module")
def grid():
    return gr.QuadratureGrid(10)


def test_linear_phase_is_rank_one():
    f = ga.linear_phase(6, 0.3, -1.0)
    assert f.max_rank == 1
    np.testing.assert_allclose(f.to_dense(), np.exp(1j * (-1.0 + 0.3 * np.arange(64))), atol=1e-13)


@pytest.mark.parametrize("alpha", [0.8 - 0.5j, -1.2, 0.7j])
def test_displacement_of_vacuum_is_the_coherent_state_including_phase(grid, alpha):
    out = ga.displace(gr.coherent(0.0, grid), alpha, grid)
    assert (out - gr.coherent(alpha, grid)).compress(0.0)[0].norm() < 1e-8


def test_displacements_compose_with_the_weyl_phase(grid):
    """D(a) D(b) = exp(i Im(a conj(b))) D(a + b)."""
    a, b = 0.4 + 0.3j, -0.2 + 0.5j
    psi = gr.squeezed_vacuum(0.3, grid)
    lhs = ga.displace(ga.displace(psi, b, grid), a, grid)
    rhs = ga.displace(psi, a + b, grid) * np.exp(1j * np.imag(a * np.conj(b)))
    assert (lhs - rhs).compress(0.0)[0].norm() < 1e-8


GATES = {
    "R(0.3)": lambda: ops.rotation_matrix(0.3),
    "R(2.0)": lambda: ops.rotation_matrix(2.0),
    "R(pi/2)": lambda: ops.rotation_matrix(np.pi / 2),
    "S(0.5)": lambda: ops.squeezing_matrix(0.5),
    "S(0.8,1.1)": lambda: ops.squeezing_matrix(0.8, 1.1),
}


@pytest.mark.parametrize("name", list(GATES))
def test_symplectic_gates_match_the_gaussian_backend(grid, name):
    S = GATES[name]()  # built inside the test: parametrize arguments are evaluated at collection time
    s0 = ops.displace(GaussianState.vacuum(), 0.6 + 0.3j)
    out = ga.symplectic(gr.from_gaussian(s0, grid), S, grid)
    assert infidelity(out, gr.from_gaussian(ops.apply_symplectic(s0, S), grid)) < 1e-10


@settings(max_examples=25, deadline=None)
@given(r=st.floats(0.0, 0.8), phi=st.floats(-3, 3), theta=st.floats(-3, 3))
def test_random_gaussian_unitaries(r, phi, theta):
    grid = gr.QuadratureGrid(10)
    S = ops.rotation_matrix(theta) @ ops.squeezing_matrix(r, phi)
    s0 = GaussianState.coherent(0.4 - 0.2j)
    out = ga.symplectic(gr.from_gaussian(s0, grid), S, grid)
    assert infidelity(out, gr.from_gaussian(ops.apply_symplectic(s0, S), grid)) < 1e-9


def test_shear_decomposition_reconstructs_the_matrix():
    M = {"x": lambda v: np.array([[1.0, 0.0], [v, 1.0]]), "p": lambda v: np.array([[1.0, -v], [0.0, 1.0]])}
    for S in (ops.squeezing_matrix(0.7), ops.rotation_matrix(0.1), ops.rotation_matrix(1.3), np.eye(2)):
        total = np.eye(2)
        for kind, v in ga.shear_decomposition(S):
            total = M[kind](v) @ total  # applied in order: later shears multiply on the left
        np.testing.assert_allclose(total, S, atol=1e-12)
    with pytest.raises(InputError):
        ga.shear_decomposition(np.diag([2.0, 2.0]))


def test_shear_decomposition_keeps_the_shears_mild():
    """The case Hypothesis found: R(-0.93) S(0.7) needs shears ~10 on one route but < 2 on another."""
    S = ops.rotation_matrix(-0.934) @ ops.squeezing_matrix(0.705, 0.064)
    assert max(abs(v) for _, v in ga.shear_decomposition(S)) < 2.5


def test_rotation_works_on_non_self_dual_grids():
    g = gr.QuadratureGrid(10, length=60.0)
    s0 = GaussianState.coherent(0.8)
    out = ga.rotate(gr.from_gaussian(s0, g), 0.9, g)
    assert infidelity(out, gr.from_gaussian(ops.rotate(s0, 0.9), g)) < 1e-9


def test_squeezed_rotated_state_rank_stays_small():
    g = gr.QuadratureGrid(14)
    out = ga.squeeze(gr.coherent(0.5, g), 0.8, g, phi=0.7)
    assert out.max_rank <= 16


def test_cubic_phase_matches_the_dense_oracle():
    g = gr.QuadratureGrid(11)
    out = ga.cubic_phase(gr.coherent(0.0, g), 0.1, g)
    N = 120
    k = cd.cubic_phase(N, 0.1) @ cd.fock_ket(N, 0)
    exact = cd.position_wavefunction(k, g.x()) * np.sqrt(g.dx)
    assert 1 - abs(np.vdot(exact, out.to_dense())) ** 2 / np.vdot(exact, exact).real < 1e-9


def test_cubic_phase_state_rank_is_small_although_the_gate_is_not():
    g = gr.QuadratureGrid(14)
    out = ga.cubic_phase(gr.squeezed_vacuum(0.5, g), 0.5, g)
    assert out.max_rank <= 16


def test_kerr_gate_matches_the_dense_oracle():
    out = ga.kerr(bs.coherent(2.0, 7), np.pi / 4)
    np.testing.assert_allclose(out.to_dense(), cd.kerr(128, np.pi / 4) @ dr.coherent(128, 2.0), atol=1e-10)


def test_kerr_half_period_makes_a_cat():
    """Yurke & Stoler (1986): exp(i pi n^2 / 2)|alpha> = (e^{i pi/4}|alpha> + e^{-i pi/4}|-alpha>) / sqrt(2)."""
    alpha = 2.0
    out = ga.kerr(bs.coherent(alpha, 7), np.pi / 2)  # exp(i pi n^2 / 2)
    k = out.to_dense()
    target = (
        np.exp(1j * np.pi / 4) * dr.coherent(128, alpha) + np.exp(-1j * np.pi / 4) * dr.coherent(128, -alpha)
    ) / np.sqrt(2)
    assert abs(np.vdot(target, k)) == pytest.approx(1.0, abs=1e-8)


def test_phase_validates_basis(grid):
    with pytest.raises(InputError):
        ga.phase(gr.coherent(0.0, grid), np.cos, grid, basis="q")
