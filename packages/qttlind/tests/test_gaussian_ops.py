"""Gaussian gates and channels (qttlind.gaussian.ops), against the dense oracle."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qttlind.errors import InputError, ShapeError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian.phase_space import is_symplectic
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr

N = 40  # single-mode Fock cutoff of the oracle
N2 = 14  # per-mode cutoff for two-mode checks


def dm(k):
    return np.outer(k, k.conj())


def assert_moments(state, rho, dims, atol=1e-8):
    m, V = cd.moments(rho, dims)
    np.testing.assert_allclose(state.mean, m, atol=atol)
    np.testing.assert_allclose(state.cov, V, atol=atol)


@pytest.mark.parametrize("theta", [0.0, 0.4, -2.5])
def test_rotation_and_displacement(theta):
    alpha = 0.6 - 0.2j
    k = cd.rotation(N, theta) @ cd.displacement(N, alpha) @ cd.fock_ket(N, 0)
    s = ops.rotate(ops.displace(GaussianState.vacuum(), alpha), theta)
    assert_moments(s, dm(k), [N])


@pytest.mark.parametrize("r,phi", [(0.2, 0.0), (0.5, 0.8), (0.4, -1.9)])
def test_squeezing_of_a_coherent_state(r, phi):
    k = cd.squeezing(N, r, phi) @ cd.displacement(N, 0.5) @ cd.fock_ket(N, 0)
    s = ops.squeeze(GaussianState.coherent(0.5), r, phi)
    assert_moments(s, dm(k), [N])


@pytest.mark.parametrize("theta,phi", [(0.3, 0.0), (np.pi / 4, 0.7), (1.1, -0.4)])
def test_beamsplitter(theta, phi):
    v0 = np.kron(cd.displacement(N2, 0.9) @ cd.fock_ket(N2, 0), cd.displacement(N2, 0.4j) @ cd.fock_ket(N2, 0))
    k = cd.beamsplitter(N2, N2, theta, phi) @ v0
    s = ops.beamsplitter(GaussianState.coherent([0.9, 0.4j]), theta, phi)
    assert_moments(s, dm(k), [N2, N2], atol=1e-7)


@pytest.mark.parametrize("r,phi", [(0.2, 0.0), (0.35, 1.0)])
def test_two_mode_squeezing(r, phi):
    v0 = np.kron(cd.fock_ket(N2, 0), cd.fock_ket(N2, 0))
    k = cd.two_mode_squeezing(N2, N2, r, phi) @ v0
    s = ops.two_mode_squeeze(GaussianState.vacuum(2), r, phi)
    assert_moments(s, dm(k), [N2, N2], atol=1e-7)


def test_two_mode_squeezed_vacuum_marginal_is_thermal():
    r = 0.6
    s = ops.two_mode_squeeze(GaussianState.vacuum(2), r)
    np.testing.assert_allclose(s.reduce([1]).cov, GaussianState.thermal(np.sinh(r) ** 2).cov, atol=1e-12)


def test_interferometer_from_beamsplitters():
    theta, phi = 0.4, 0.3
    t, sn = np.cos(theta), np.sin(theta)
    U = np.array([[t, -np.exp(-1j * phi) * sn], [np.exp(1j * phi) * sn, t]])
    np.testing.assert_allclose(ops.interferometer_matrix(U), ops.beamsplitter_matrix(theta, phi), atol=1e-15)
    with pytest.raises(InputError):
        ops.interferometer_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_random_interferometers_are_symplectic_and_conserve_photons(seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    U, _ = np.linalg.qr(Z)
    S = ops.interferometer_matrix(U)
    assert is_symplectic(S)
    s = ops.interferometer(GaussianState.coherent([1.0, 0.5j, -0.3, 0.0]), U)
    assert s.mean_photons().sum() == pytest.approx(1.0 + 0.25 + 0.09)


def test_gate_composition_is_matrix_multiplication():
    s0 = GaussianState.coherent([0.3, -0.2j])
    a = ops.beamsplitter(ops.squeeze(s0, 0.4, 0.1, mode=1), 0.7)
    full_sq = np.eye(4)
    full_sq[np.ix_([1, 3], [1, 3])] = ops.squeezing_matrix(0.4, 0.1)
    S = ops.beamsplitter_matrix(0.7) @ full_sq
    b = ops.apply_symplectic(s0, S)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-14)
    np.testing.assert_allclose(a.cov, b.cov, atol=1e-14)


def test_apply_symplectic_errors():
    with pytest.raises(ShapeError):
        ops.apply_symplectic(GaussianState.vacuum(2), np.eye(2), [0, 1])
    with pytest.raises(InputError, match="symplectic"):
        ops.apply_symplectic(GaussianState.vacuum(), np.diag([2.0, 2.0]))


@pytest.mark.parametrize("eta", [0.0, 0.35, 1.0])
def test_loss_matches_the_kraus_channel(eta):
    k = cd.squeezing(N, 0.4) @ cd.displacement(N, 0.8) @ cd.fock_ket(N, 0)
    rho = cd.loss_channel(dm(k), eta, 0, [N])
    s = ops.loss(ops.squeeze(GaussianState.coherent(0.8), 0.4), eta)
    assert_moments(s, rho, [N])


@pytest.mark.parametrize("kappa,t", [(0.5, 1.0), (2.0, 0.3)])
def test_lindblad_loss_is_the_loss_channel(kappa, t):
    k = cd.squeezing(N, 0.4) @ cd.displacement(N, 0.6 - 0.2j) @ cd.fock_ket(N, 0)
    rho_t = dr.exact_rho(dm(k), np.zeros((N, N)), [np.sqrt(kappa) * dr.destroy(N)], t)
    s = ops.loss(ops.squeeze(GaussianState.coherent(0.6 - 0.2j), 0.4), np.exp(-kappa * t))
    assert_moments(s, rho_t, [N])


def test_thermal_loss_and_amplifier_moments():
    s = ops.loss(GaussianState.vacuum(), 0.3, nbar=2.0)
    np.testing.assert_allclose(s.cov, (0.3 * 0.5 + 0.7 * 2.5) * np.eye(2))
    a = ops.amplifier(GaussianState.vacuum(), 2.0)
    assert a.mean_photons()[0] == pytest.approx(1.0)  # (G - 1) noise photons
    n = ops.thermal_noise(GaussianState.vacuum(), 0.25)
    np.testing.assert_allclose(n.cov, 0.75 * np.eye(2))


def test_channel_validation():
    with pytest.raises(InputError, match="completely positive"):
        ops.check_channel(np.eye(2) * 2.0, np.zeros((2, 2)))  # noiseless amplification is unphysical
    with pytest.raises(InputError):
        ops.loss(GaussianState.vacuum(), 1.5)
    with pytest.raises(InputError):
        ops.amplifier(GaussianState.vacuum(), 0.5)
    ops.check_channel(np.sqrt(0.5) * np.eye(2), 0.25 * np.eye(2))  # pure loss: fine


def test_loss_composes():
    s = GaussianState.squeezed_vacuum(0.8)
    a = ops.loss(ops.loss(s, 0.6), 0.5)
    b = ops.loss(s, 0.3)
    np.testing.assert_allclose(a.cov, b.cov, atol=1e-14)
