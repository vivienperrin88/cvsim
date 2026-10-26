"""Quadrature grids and position-space states (qttlind.cv.grids)."""

import dataclasses
import warnings

import numpy as np
import pytest

from qttlind.cv import grids as gr
from qttlind.errors import CutoffWarning, InputError
from qttlind.gaussian import GaussianState, ops
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr


def test_self_dual_grid():
    g = gr.QuadratureGrid(10)
    assert g.size == 1024 and g.self_dual
    assert g.dx == pytest.approx(np.sqrt(2 * np.pi / 1024))
    x = g.x()
    assert x[0] == pytest.approx(-g.L / 2) and x[512] == 0.0
    np.testing.assert_allclose(g.p(), x)


def test_custom_grid_and_validation():
    g = gr.QuadratureGrid(8, length=20.0)
    assert g.dx == pytest.approx(20 / 256) and g.dp == pytest.approx(2 * np.pi / 20) and not g.self_dual
    with pytest.raises(InputError):
        gr.QuadratureGrid(0)
    with pytest.raises(InputError):
        gr.QuadratureGrid(8, length=-1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.bits = 9  # type: ignore[misc]


def test_max_photon_number_grows_linearly_with_the_grid_size():
    small, big = gr.QuadratureGrid(8).max_photon_number(), gr.QuadratureGrid(12).max_photon_number()
    assert 20 < small < np.pi * 256 / 4  # n_max ~ pi N / 4 minus the safety margin
    assert big > 12 * small  # 16x the points: more than 12x the photons


def test_hermite_functions_are_orthonormal_on_the_grid():
    g = gr.QuadratureGrid(10)
    nmax = g.max_photon_number()
    phi = gr.hermite_functions(nmax + 1, g.x())
    G = phi @ phi.T * g.dx
    np.testing.assert_allclose(G, np.eye(nmax + 1), atol=1e-10)


def test_hermite_functions_match_oracle():
    x = np.linspace(-5, 5, 41)
    np.testing.assert_allclose(gr.hermite_functions(30, x), cd.hermite_functions(30, x), atol=1e-14)


def test_from_wavefunction_normalises_and_validates():
    g = gr.QuadratureGrid(8)
    tt = gr.from_wavefunction(lambda x: np.exp(-(x**2)), g)
    assert tt.norm() == pytest.approx(1.0)
    with pytest.raises(InputError):
        gr.from_wavefunction(lambda x: 0 * x, g)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 0.8 - 1.3j])
def test_coherent_state_matches_the_fock_space_oracle(alpha):
    g = gr.QuadratureGrid(10)
    psi = gr.coherent(alpha, g)
    exact = cd.position_wavefunction(dr.coherent(80, alpha), g.x()) * np.sqrt(g.dx)
    np.testing.assert_allclose(psi.to_dense(), exact, atol=1e-9)  # same global phase too


@pytest.mark.parametrize("r,phi,alpha", [(0.5, 0.0, 0.0), (0.7, 1.1, 0.0), (0.4, -0.6, 0.5 + 0.5j)])
def test_from_gaussian_matches_the_oracle_up_to_a_phase(r, phi, alpha):
    g = gr.QuadratureGrid(10)
    s = ops.displace(ops.squeeze(GaussianState.vacuum(), r, phi), alpha)
    psi = gr.from_gaussian(s, g)
    k = cd.displacement(120, alpha) @ cd.squeezing(120, r, phi) @ cd.fock_ket(120, 0)
    exact = cd.position_wavefunction(k, g.x()) * np.sqrt(g.dx)
    assert abs(np.vdot(exact, psi.to_dense())) == pytest.approx(1.0, abs=1e-9)


def test_from_gaussian_rejects_mixed_or_multimode():
    g = gr.QuadratureGrid(8)
    with pytest.raises(InputError):
        gr.from_gaussian(GaussianState.thermal(0.5), g)
    with pytest.raises(InputError):
        gr.from_gaussian(GaussianState.vacuum(2), g)


def test_fock_and_cat_states():
    g = gr.QuadratureGrid(10)
    one = gr.fock(1, g)
    np.testing.assert_allclose(one.to_dense(), gr.hermite_functions(2, g.x())[1] * np.sqrt(g.dx), atol=1e-10)
    odd = gr.cat(1.5, g, parity=-1)
    exact = cd.position_wavefunction(dr.cat(80, 1.5, -1), g.x()) * np.sqrt(g.dx)
    np.testing.assert_allclose(odd.to_dense(), exact, atol=1e-9)
    with pytest.raises(InputError):
        gr.cat(0.0, g, parity=-1)
    with pytest.raises(InputError):
        gr.cat(1.0, g, parity=2)


def test_gkp_states_are_periodic_and_orthogonal():
    g = gr.QuadratureGrid(12)
    zero = gr.gkp(0.25, g, logical=0)
    one = gr.gkp(0.25, g, logical=1)
    assert abs(zero.inner(one)) < 2e-5  # physical overlap of neighbouring peaks ~ exp(-pi / (4 delta^2))
    w = np.abs(gr.wavefunction(zero, g)) ** 2
    x = g.x()
    peak = x[np.argmax(w)]
    assert peak == pytest.approx(0.0, abs=g.dx)
    nxt = x[(x > 1.5) & (x < 5)][np.argmax(w[(x > 1.5) & (x < 5)])]
    assert nxt == pytest.approx(2 * np.sqrt(np.pi), abs=2 * g.dx)


def test_gkp_rank_stays_small_and_flat():
    """The design measurement of the CV extension: GKP ranks ~ 10-20 in position space, independent of the grid size."""
    ranks = [gr.gkp(0.2, gr.QuadratureGrid(R), tol=1e-16).max_rank for R in (12, 14, 16)]
    assert max(ranks) <= 24
    assert ranks[-1] <= ranks[0] + 4


def test_squeezed_state_rank_is_flat():
    ranks = [gr.squeezed_vacuum(1.0, gr.QuadratureGrid(R), tol=1e-16).max_rank for R in (10, 14, 18)]
    assert max(ranks) <= 12


def test_resolution_warnings():
    g = gr.QuadratureGrid(6)
    with pytest.warns(CutoffWarning):
        gr.fock(60, g)
    with pytest.warns(CutoffWarning):
        gr.gkp(0.05, g)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gr.coherent(1.0, gr.QuadratureGrid(10))  # resolved: no warning


def test_dense_guard():
    with pytest.raises(InputError):
        gr.QuadratureGrid(40).x()
