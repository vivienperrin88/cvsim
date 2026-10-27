"""Phase space and Gaussian states (qttlind.gaussian.phase_space, .states)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qttlind.errors import InputError, ShapeError
from qttlind.gaussian import GaussianState
from qttlind.gaussian import phase_space as ps
from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr


def dm(k):
    return np.outer(k, k.conj())


# ----------------------------------------------------------------------------- phase space


def test_omega_and_complex_basis():
    W = ps.omega(2)
    np.testing.assert_array_equal(W @ W, -np.eye(4))
    U = ps.complex_basis(3)
    np.testing.assert_allclose(U @ U.conj().T, np.eye(6), atol=1e-15)


def test_mode_indices_validates():
    assert ps.mode_indices([1], 3) == [1, 4]
    with pytest.raises(InputError):
        ps.mode_indices([3], 3)
    with pytest.raises(InputError):
        ps.mode_indices([0, 0], 3)


def test_bogoliubov_of_squeezing_is_diagonal():
    r = 0.4
    S = ps.symplectic_from_bogoliubov(np.cosh(r), -np.sinh(r))
    np.testing.assert_allclose(S, np.diag([np.exp(-r), np.exp(r)]), atol=1e-15)


@settings(max_examples=30, deadline=None)
@given(r=st.floats(0, 1.5), th=st.floats(-3, 3), ph=st.floats(-3, 3))
def test_bogoliubov_transformations_are_symplectic(r, th, ph):
    """Any A = e^{i th} cosh(r), B = -e^{i (th + ph)} sinh(r) preserves [a, a^dag] = 1, hence Omega."""
    A = np.exp(1j * th) * np.cosh(r)
    B = -np.exp(1j * (th + ph)) * np.sinh(r)
    assert ps.is_symplectic(ps.symplectic_from_bogoliubov(A, B))


def test_is_symplectic_rejects():
    assert not ps.is_symplectic(np.diag([2.0, 2.0]))
    assert not ps.is_symplectic(np.eye(3))


def test_symplectic_eigenvalues_thermal_and_pure():
    np.testing.assert_allclose(ps.symplectic_eigenvalues(np.diag([2.5, 0.5, 2.5, 0.5])), [0.5, 2.5])
    S = ps.symplectic_from_bogoliubov(np.cosh(0.7), -np.sinh(0.7) * np.exp(0.3j))
    np.testing.assert_allclose(ps.symplectic_eigenvalues(S @ S.T / 2), [0.5], atol=1e-12)


def test_check_covariance_errors():
    with pytest.raises(ShapeError):
        ps.check_covariance(np.eye(3))
    with pytest.raises(InputError, match="symmetric"):
        ps.check_covariance(np.array([[1.0, 0.2], [0.0, 1.0]]))
    with pytest.raises(InputError, match="uncertainty"):
        ps.check_covariance(np.eye(2) * 0.3)  # below the vacuum
    ps.check_covariance(np.eye(2) / 2)


# ----------------------------------------------------------------------------- states


def test_vacuum_and_coherent():
    v = GaussianState.vacuum(3)
    assert v.num_modes == 3 and v.purity() == pytest.approx(1.0)
    c = GaussianState.coherent([1 + 1j, -0.5])
    np.testing.assert_allclose(c.mean, np.sqrt(2) * np.array([1, -0.5, 1, 0]))
    np.testing.assert_allclose(c.mean_photons(), [2.0, 0.25])


def test_states_are_immutable():
    v = GaussianState.vacuum()
    with pytest.raises(ValueError):
        v.mean[0] = 1.0


def test_constructor_validates():
    with pytest.raises(ShapeError):
        GaussianState(np.zeros(3), np.eye(2) / 2)
    with pytest.raises(InputError):
        GaussianState(np.zeros(2), np.eye(2) * 0.1)
    with pytest.raises(InputError):
        GaussianState.thermal(-1.0)
    with pytest.raises(InputError):
        GaussianState.vacuum(0)


@pytest.mark.parametrize("r,phi", [(0.3, 0.0), (0.8, 1.1), (1.2, -2.0)])
def test_squeezed_vacuum_matches_the_dense_oracle(r, phi):
    N = 120
    m, V = cd.moments(dm(cd.squeezing(N, r, phi) @ cd.fock_ket(N, 0)), [N])
    s = GaussianState.squeezed_vacuum(r, phi)
    np.testing.assert_allclose(s.mean, m, atol=1e-10)
    np.testing.assert_allclose(s.cov, V, atol=1e-8)
    assert s.mean_photons()[0] == pytest.approx(np.sinh(r) ** 2)


def test_thermal_state_moments_and_entropy():
    nbar = 0.7
    s = GaussianState.thermal(nbar)
    _, V = cd.moments(cd.thermal_dm(80, nbar), [80])
    np.testing.assert_allclose(s.cov, V, atol=1e-10)
    exact = (nbar + 1) * np.log(nbar + 1) - nbar * np.log(nbar)
    assert s.entropy() == pytest.approx(exact)
    assert s.purity() == pytest.approx(1 / (2 * nbar + 1))


def test_product_interleaves_xxpp_blocks():
    a = GaussianState.coherent(1.0)
    b = GaussianState.squeezed_vacuum(0.5)
    ab = GaussianState.product(a, b)
    assert ab.num_modes == 2
    np.testing.assert_allclose(ab.reduce([0]).mean, a.mean)
    np.testing.assert_allclose(ab.reduce([1]).cov, b.cov)
    np.testing.assert_allclose(ab.cov[0, 1], 0.0)  # no correlations in a product
    ba = ab.reduce([1, 0])
    np.testing.assert_allclose(ba.reduce([0]).cov, b.cov)


def test_purity_and_entropy_of_pure_states():
    s = GaussianState.product(GaussianState.squeezed_vacuum(1.0, 0.3), GaussianState.coherent(2j))
    assert s.is_pure()
    assert s.entropy() == pytest.approx(0.0, abs=1e-9)


def test_purity_does_not_underflow_for_many_modes():
    s = GaussianState.thermal([1.0] * 600)  # det V = 1.5^1200 overflows a double
    assert s.purity() == pytest.approx(3.0**-600, rel=1e-9)


def test_repr():
    assert "modes=2" in repr(GaussianState.vacuum(2))


def test_coherent_state_photon_number_matches_dense():
    alpha = 1.3 - 0.4j
    k = dr.coherent(60, alpha)
    n_dense = np.vdot(k, np.arange(60) * k).real
    assert GaussianState.coherent(alpha).mean_photons()[0] == pytest.approx(n_dense, abs=1e-10)
