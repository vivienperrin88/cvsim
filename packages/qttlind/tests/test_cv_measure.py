"""Measurements and sampling on quantics states (qttlind.cv.measure)."""

import numpy as np
import pytest

import qttlind as ql
from qttlind.bosons import states as bs
from qttlind.core.tt import TensorTrain
from qttlind.cv import grids as gr
from qttlind.cv import measure as me
from qttlind.errors import InputError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian import measure as gm
from qttlind.reference import cv_dense as cd


def gaussian_wavefunction_2modes(state, grid):
    """Dense two-mode pure Gaussian wavefunction psi(x1, x2) ~ exp(-(y^T A y)/2 + i p0.x), A = Vxx^{-1}(I/2 - i Vxp)."""
    V, d = state.cov, state.mean
    Vxx, Vxp = V[:2, :2], V[:2, 2:]
    A = np.linalg.solve(Vxx, 0.5 * np.eye(2) - 1j * Vxp)
    xs = grid.x()
    X1, X2 = np.meshgrid(xs, xs, indexing="ij")
    Y = np.stack([X1 - d[0], X2 - d[1]], axis=-1)
    q = np.einsum("...i,ij,...j->...", Y, A, Y)
    w = (np.exp(-q / 2 + 1j * (d[2] * X1 + d[3] * X2)) * grid.dx).reshape(-1)
    return TensorTrain.from_dense(w / np.linalg.norm(w), [2] * (2 * grid.bits), 1e-22)


@pytest.fixture(scope="module")
def grid():
    return gr.QuadratureGrid(9)


def test_total_and_probabilities(grid):
    psi = gr.coherent(0.5 - 0.5j, grid)
    P = me.probabilities(psi)
    assert me.total(P).real == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(P.to_dense().real, np.abs(psi.to_dense()) ** 2, rtol=0, atol=1e-11)


def test_quadrature_moments_match_the_gaussian_backend(grid):
    s = ops.squeeze(ops.displace(GaussianState.vacuum(), 0.7 - 0.2j), 0.4, 0.6)
    mx, vx, mp, vp = me.quadrature_moments(gr.from_gaussian(s, grid), grid)
    np.testing.assert_allclose([mx, vx, mp, vp], [s.mean[0], s.cov[0, 0], s.mean[1], s.cov[1, 1]], atol=1e-8)


def test_sampling_reproduces_the_distribution(grid, rng):
    """Chi-square-free check: the empirical CDF stays within the Kolmogorov 1e-4 band of the exact one."""
    s = ops.squeeze(GaussianState.coherent(0.8), 0.3)
    psi = gr.from_gaussian(s, grid)
    samples = me.sample(psi, 3000, rng)
    exact = np.cumsum(np.abs(psi.to_dense()) ** 2)
    empirical = np.searchsorted(np.sort(samples), np.arange(grid.size), side="right") / samples.size
    assert np.abs(empirical - exact).max() < 1.95 / np.sqrt(samples.size)  # KS at significance ~1e-3


def test_sampling_a_prefix_samples_the_marginal(rng):
    g = gr.QuadratureGrid(6)
    two = gaussian_wavefunction_2modes(ops.two_mode_squeeze(GaussianState.coherent([0.6, 0.0]), 0.4), g)
    P = (np.abs(two.to_dense()) ** 2).reshape(g.size, g.size).sum(axis=1)
    samples = me.sample(two, 3000, rng, nsites=g.bits)
    assert samples.max() < g.size
    empirical = np.bincount(samples, minlength=g.size) / samples.size
    assert np.sum(empirical * np.arange(g.size)) == pytest.approx(np.sum(P * np.arange(g.size)), abs=0.4)


def test_homodyne_on_a_two_mode_chain_matches_the_gaussian_backend():
    g = gr.QuadratureGrid(8)
    s = ops.beamsplitter(ops.two_mode_squeeze(GaussianState.vacuum(2), 0.5), 0.3)
    psi = gaussian_wavefunction_2modes(s, g)
    res = me.homodyne(psi, g, outcome=0.8)
    ref = gm.homodyne(s, 0, 0.0, outcome=res.outcome)
    assert res.density == pytest.approx(ref.density, rel=1e-8)
    assert 1 - abs(gr.from_gaussian(ref.state, g).inner(res.state)) ** 2 < 1e-10


def test_homodyne_single_mode_density_and_validation(grid, rng):
    psi = gr.coherent(0.0, grid)
    res = me.homodyne(psi, grid, outcome=0.0)
    assert res.state is None
    assert res.density == pytest.approx(1 / np.sqrt(np.pi), rel=1e-8)
    with pytest.raises(InputError):
        me.homodyne(psi, grid)
    with pytest.raises(InputError):
        me.homodyne(psi, grid, rng=rng, outcome=0.0)


def test_fock_probabilities_of_a_squeezed_state():
    g = gr.QuadratureGrid(10)
    probs = me.fock_probabilities(gr.squeezed_vacuum(0.5, g), g, 8)
    np.testing.assert_allclose(probs, np.abs(cd.squeezed_vacuum_amplitudes(8, 0.5)) ** 2, atol=1e-10)


def test_gkp_logical_readout_by_binning():
    """Binning x into intervals of width sqrt(pi) around even/odd multiples of sqrt(pi) reads the GKP qubit."""
    g = gr.QuadratureGrid(12)
    x0 = -g.L / 2

    def bins(logical):
        def ind(j):
            k = np.rint((x0 + j * g.dx) / np.sqrt(np.pi)).astype(int)
            return (np.mod(k, 2) == logical).astype(complex)

        return bs.from_function(ind, g.bits, 1e-20)

    zero = gr.gkp(0.25, g, logical=0)
    p_correct = me.interval_probability(zero, bins(0))
    assert p_correct > 0.99
    assert p_correct + me.interval_probability(zero, bins(1)) == pytest.approx(1.0, abs=1e-10)


def test_sample_validation(grid, rng):
    with pytest.raises(InputError):
        me.sample(gr.coherent(0.0, grid), 1, rng, nsites=0)


def test_feature_registered():
    assert "cv" in ql.FEATURES
