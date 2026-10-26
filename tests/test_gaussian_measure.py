"""Gaussian measurements and phase-space functions (qttlind.gaussian.measure)."""

import numpy as np
import pytest

from qttlind.errors import InputError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian import measure as gm
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr

N2 = 26


def dm(k):
    return np.outer(k, k.conj())


def two_mode():
    """A correlated, displaced two-mode pure state and its dense ket."""
    s = ops.beamsplitter(ops.two_mode_squeeze(GaussianState.coherent([0.3, -0.2j]), 0.3, 0.2), 0.6, 0.4)
    v0 = np.kron(cd.displacement(N2, 0.3) @ cd.fock_ket(N2, 0), cd.displacement(N2, -0.2j) @ cd.fock_ket(N2, 0))
    k = cd.beamsplitter(N2, N2, 0.6, 0.4) @ cd.two_mode_squeezing(N2, N2, 0.3, 0.2) @ v0
    return s, k


@pytest.mark.parametrize("phi", [0.0, 0.7, np.pi / 2])
def test_homodyne_density_matches_dense(phi):
    s = ops.squeeze(GaussianState.coherent(0.5 + 0.3j), 0.4, 0.2)
    N = 60
    k = cd.rotation(N, -phi) @ cd.squeezing(N, 0.4, 0.2) @ cd.displacement(N, 0.5 + 0.3j) @ cd.fock_ket(N, 0)
    q = np.linspace(-3, 3, 13)
    dense = np.abs(cd.position_wavefunction(k, q)) ** 2
    np.testing.assert_allclose(gm.homodyne_density(s, q, 0, phi), dense, atol=1e-9)


@pytest.mark.parametrize("phi,q", [(0.0, 0.4), (1.0, -0.8)])
def test_homodyne_conditional_state_matches_dense_projection(phi, q):
    s, k = two_mode()
    res = gm.homodyne(s, mode=0, phi=phi, outcome=q)
    # dense: rotate mode 0 by -phi, project it on <q|
    k = np.kron(cd.rotation(N2, -phi), np.eye(N2)) @ k
    phi_q = cd.hermite_functions(N2, np.array([q]))[:, 0]
    kb = phi_q @ k.reshape(N2, N2)  # <q| (x) 1
    prob_density = np.vdot(kb, kb).real
    m, V = cd.moments(dm(kb / np.sqrt(prob_density)), [N2])
    assert res.density == pytest.approx(prob_density, abs=1e-7)
    np.testing.assert_allclose(res.state.mean, m, atol=1e-6)
    np.testing.assert_allclose(res.state.cov, V, atol=1e-6)


def test_homodyne_requires_exactly_one_of_outcome_and_rng():
    s = GaussianState.vacuum(2)
    with pytest.raises(InputError):
        gm.homodyne(s)
    with pytest.raises(InputError):
        gm.homodyne(s, outcome=0.0, rng=np.random.default_rng(0))


def test_homodyne_sampling_statistics(rng):
    """Mean and variance of 4000 samples within 5 standard errors (false failure rate ~1e-6)."""
    s = ops.squeeze(GaussianState.coherent(1.0), 0.3)
    xs = np.array([gm.homodyne(s, rng=rng).outcome for _ in range(4000)])
    mu, var = s.mean[0], s.cov[0, 0]
    assert abs(xs.mean() - mu) < 5 * np.sqrt(var / xs.size)
    assert abs(xs.var() - var) < 5 * var * np.sqrt(2 / xs.size)


def test_homodyne_on_the_last_mode_returns_no_state():
    res = gm.homodyne(GaussianState.vacuum(), outcome=0.0)
    assert res.state is None
    assert res.density == pytest.approx(1 / np.sqrt(np.pi))


def test_heterodyne_density_is_the_husimi_function():
    s = ops.squeeze(GaussianState.coherent(0.4 - 0.5j), 0.5, 1.0)
    N = 60
    k = cd.squeezing(N, 0.5, 1.0) @ cd.displacement(N, 0.4 - 0.5j) @ cd.fock_ket(N, 0)
    for alpha in (0.0, 0.3 + 0.2j, -1.0j):
        husimi = abs(np.vdot(dr.coherent(N, alpha), k)) ** 2 / np.pi
        assert gm.heterodyne(s, outcome=alpha).density == pytest.approx(husimi, abs=1e-10)


def test_heterodyne_conditional_state_matches_dense_projection():
    s, k = two_mode()
    alpha = 0.2 - 0.4j
    res = gm.heterodyne(s, mode=0, outcome=alpha)
    kb = dr.coherent(N2, alpha).conj() @ k.reshape(N2, N2)
    m, V = cd.moments(dm(kb / np.linalg.norm(kb)), [N2])
    np.testing.assert_allclose(res.state.mean, m, atol=1e-6)
    np.testing.assert_allclose(res.state.cov, V, atol=1e-6)


def test_vacuum_probability():
    s, k = two_mode()
    P = np.abs(k.reshape(N2, N2)) ** 2
    assert gm.vacuum_probability(s) == pytest.approx(P[0, 0], abs=1e-10)
    assert gm.vacuum_probability(s, [1]) == pytest.approx(P[:, 0].sum(), abs=1e-9)


def test_overlap_and_fidelity():
    a = GaussianState.coherent(0.5)
    b = ops.squeeze(GaussianState.coherent(0.2j), 0.3)
    N = 50
    ka = dr.coherent(N, 0.5)
    kb = cd.squeezing(N, 0.3) @ cd.displacement(N, 0.2j) @ cd.fock_ket(N, 0)
    assert gm.overlap(a, b) == pytest.approx(abs(np.vdot(ka, kb)) ** 2, abs=1e-10)
    assert gm.fidelity(a, b) == pytest.approx(gm.overlap(a, b))
    with pytest.raises(InputError):
        gm.fidelity(GaussianState.thermal(1.0), GaussianState.thermal(2.0))


def test_wigner_matches_the_parity_formula_and_is_normalised():
    s = ops.squeeze(GaussianState.coherent(0.6 + 0.2j), 0.4, 0.5)
    N = 50
    rho = dm(cd.squeezing(N, 0.4, 0.5) @ cd.displacement(N, 0.6 + 0.2j) @ cd.fock_ket(N, 0))
    x = np.array([-0.5, 0.3, 1.2])
    p = np.array([0.0, 0.4])
    W = gm.wigner(s, x, p)
    alphas = (x[None, :] + 1j * p[:, None]) / np.sqrt(2)
    np.testing.assert_allclose(W, cd.wigner_parity(rho, alphas) / 2, atol=1e-9)
    g = np.linspace(-8, 8, 321)
    Wg = gm.wigner(s, g, g)
    assert Wg.sum() * (g[1] - g[0]) ** 2 == pytest.approx(1.0, abs=1e-6)


def test_log_negativity_of_two_mode_squeezing():
    r = 0.45
    s = ops.two_mode_squeeze(GaussianState.vacuum(2), r)
    assert gm.log_negativity(s, [0]) == pytest.approx(2 * r / np.log(2))
    assert (
        gm.log_negativity(GaussianState.product(GaussianState.squeezed_vacuum(1.0), GaussianState.vacuum()), [0]) == 0.0
    )


def test_log_negativity_matches_the_dense_partial_transpose():
    s, k = two_mode()
    s = ops.loss(s, 0.7, mode=1)
    rho = cd.loss_channel(dm(k), 0.7, 1, [N2, N2])
    rt = rho.reshape(N2, N2, N2, N2).transpose(0, 3, 2, 1).reshape(N2 * N2, N2 * N2)
    dense = np.log2(np.abs(np.linalg.eigvalsh((rt + rt.conj().T) / 2)).sum())
    assert gm.log_negativity(s, [0]) == pytest.approx(dense, abs=1e-6)
