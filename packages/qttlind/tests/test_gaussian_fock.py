"""Fock amplitudes of Gaussian states by the Bargmann recursion (qttlind.gaussian.fock)."""

import numpy as np
import pytest

from qttlind.errors import InputError, ShapeError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian import fock as gf
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr

N = 60


def dm(k):
    return np.outer(k, k.conj())


def test_bargmann_of_a_coherent_state():
    gamma = 0.7 - 0.3j
    A, b, c = gf.bargmann(GaussianState.coherent(gamma))
    np.testing.assert_allclose(A, 0, atol=1e-14)
    np.testing.assert_allclose(b, [gamma, np.conj(gamma)], atol=1e-14)
    assert c == pytest.approx(-(abs(gamma) ** 2))


def test_bargmann_ket_of_squeezed_vacuum():
    r, phi = 0.6, 0.4
    A, _, c = gf.bargmann_ket(GaussianState.squeezed_vacuum(r, phi))
    assert A[0, 0] == pytest.approx(-np.exp(1j * phi) * np.tanh(r))
    assert np.exp(c) == pytest.approx(1 / np.sqrt(np.cosh(r)))


def test_recursion_reproduces_a_known_series():
    """exp(a z^2/2 + b z): G_n = coefficient * sqrt(n!) -- Hermite-like closed form for n <= 3."""
    a, b = 0.3, 0.5
    G = gf.fock_coefficients(np.array([[a]]), np.array([b]), 0.0, [4])
    exact = np.array([1, b, (b * b + a) / np.sqrt(2), (b**3 + 3 * a * b) / np.sqrt(6)])
    np.testing.assert_allclose(G, exact, atol=1e-15)


@pytest.mark.parametrize("r,phi,alpha", [(0.0, 0.0, 0.8), (0.5, 0.0, 0.0), (0.5, 0.3, 0.4 - 0.2j)])
def test_fock_ket_matches_dense(r, phi, alpha):
    s = ops.displace(ops.squeeze(GaussianState.vacuum(), r, phi), alpha)
    k = cd.displacement(N, alpha) @ cd.squeezing(N, r, phi) @ cd.fock_ket(N, 0)
    k = k / (k[0] / abs(k[0]))  # same phase convention: <0|psi> > 0
    np.testing.assert_allclose(gf.fock_ket(s, 25), k[:25], atol=1e-9)


@pytest.mark.parametrize("eta", [1.0, 0.8, 0.5])
def test_fock_density_matrix_of_lossy_states(eta):
    s = ops.loss(ops.displace(ops.squeeze(GaussianState.vacuum(), 0.5, 0.3), 0.4 - 0.2j), eta)
    k = cd.displacement(N, 0.4 - 0.2j) @ cd.squeezing(N, 0.5, 0.3) @ cd.fock_ket(N, 0)
    rho = cd.loss_channel(dm(k), eta, 0, [N])
    np.testing.assert_allclose(gf.fock_density_matrix(s, 25), rho[:25, :25], atol=1e-10)


def test_fock_density_matrix_reproduces_the_moments_of_a_thermal_noise_state():
    s = ops.loss(ops.squeeze(GaussianState.coherent(0.3j), 0.4), 0.6, nbar=0.5)
    m, V = cd.moments(gf.fock_density_matrix(s, 50), [50])
    np.testing.assert_allclose(m, s.mean, atol=1e-8)
    np.testing.assert_allclose(V, s.cov, atol=1e-8)


def test_two_mode_density_matrix_matches_dense():
    s = ops.beamsplitter(ops.two_mode_squeeze(GaussianState.vacuum(2), 0.3, 0.2), 0.5, 0.1)
    n = 14
    k = (
        cd.beamsplitter(n, n, 0.5, 0.1)
        @ cd.two_mode_squeezing(n, n, 0.3, 0.2)
        @ np.kron(cd.fock_ket(n, 0), cd.fock_ket(n, 0))
    )
    R = dm(k).reshape(n, n, n, n)[:6, :6, :6, :6].reshape(36, 36)
    np.testing.assert_allclose(gf.fock_density_matrix(s, 6), R, atol=1e-12)


def test_density_matrix_is_hermitian_with_unit_trace():
    s = ops.loss(ops.squeeze(GaussianState.coherent(0.5), 0.4), 0.7)
    rho = gf.fock_density_matrix(s, 40)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-10)
    assert np.linalg.eigvalsh(rho).min() > -1e-12


def test_thermal_state_diagonal():
    rho = gf.fock_density_matrix(GaussianState.thermal(0.5), 10)
    np.testing.assert_allclose(rho, cd.thermal_dm(10, 0.5), atol=1e-14)


def test_coherent_ket_matches_oracle():
    np.testing.assert_allclose(gf.fock_ket(GaussianState.coherent(1.2), 30), dr.coherent(N, 1.2)[:30], atol=1e-12)


def test_errors():
    with pytest.raises(InputError, match="pure"):
        gf.bargmann_ket(GaussianState.thermal(0.3))
    with pytest.raises(ShapeError):
        gf.fock_coefficients(np.zeros((2, 2)), np.zeros(2), 0.0, [3])
    with pytest.raises(InputError):
        gf.fock_coefficients(np.zeros((1, 1)), np.zeros(1), 0.0, [2**26])
    with pytest.raises(ShapeError):
        gf.fock_ket(GaussianState.vacuum(2), [3])
