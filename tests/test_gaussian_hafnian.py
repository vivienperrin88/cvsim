"""Hafnians and photon-counting statistics (qttlind.gaussian.hafnian)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import qttlind as ql
from qttlind.errors import InputError, ShapeError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian import hafnian as hf
from qttlind.gaussian.phase_space import complex_basis
from qttlind.reference import cv_dense as cd

n2 = 14


def random_symmetric(rng, m, complex_=True):
    A = rng.normal(size=(m, m)) + (1j * rng.normal(size=(m, m)) if complex_ else 0)
    return A + A.T


def test_exp_series_coefficient():
    # exp(lambda) -> 1/n!;  exp(lambda + lambda^2) -> [lambda^3] = 1/6 + 1
    assert hf._exp_series_coefficient(np.array([1.0, 0.0, 0.0]), 3) == pytest.approx(1 / 6)
    assert hf._exp_series_coefficient(np.array([1.0, 1.0, 0.0]), 3) == pytest.approx(1 / 6 + 1)


def test_small_hafnians_by_hand():
    assert hf.hafnian(np.array([[0, 3], [3, 0]])) == pytest.approx(3)
    assert hf.hafnian(np.ones((4, 4))) == pytest.approx(3)  # three perfect matchings of K4
    assert hf.hafnian(np.ones((6, 6))) == pytest.approx(15)  # 5!! = 15
    assert hf.hafnian(np.ones((3, 3))) == 0
    assert hf.hafnian(np.zeros((0, 0))) == 1
    assert hf.loop_hafnian(np.ones((3, 3))) == pytest.approx(4)  # 3 (edge + loop) + 1 (three loops)


@pytest.mark.parametrize("m", [2, 4, 6, 8, 10])
def test_hafnian_matches_brute_force(rng, m):
    A = random_symmetric(rng, m)
    assert hf.hafnian(A) == pytest.approx(cd.hafnian_recursive(A - np.diag(np.diag(A))), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 6, 8])
def test_loop_hafnian_matches_brute_force(rng, m):
    A = random_symmetric(rng, m)
    assert hf.loop_hafnian(A) == pytest.approx(cd.loop_hafnian_recursive(A), rel=1e-10, abs=1e-10)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_hafnian_is_invariant_under_simultaneous_permutation(seed):
    rng = np.random.default_rng(seed)
    A = random_symmetric(rng, 6)
    perm = rng.permutation(6)
    assert hf.hafnian(A[np.ix_(perm, perm)]) == pytest.approx(hf.hafnian(A), rel=1e-9, abs=1e-9)


def test_input_validation():
    with pytest.raises(ShapeError):
        hf.hafnian(np.ones((2, 3)))
    with pytest.raises(InputError, match="symmetric"):
        hf.hafnian(np.array([[0, 1], [2, 0]]))
    with pytest.raises(InputError):
        hf.hafnian(np.ones((hf.MAX_HAFNIAN_SIZE + 2,) * 2))


def two_mode_state(displaced=False):
    s = ops.beamsplitter(ops.two_mode_squeeze(GaussianState.vacuum(2), 0.3, 0.2), 0.5, 0.1)
    v0 = np.kron(cd.fock_ket(n2, 0), cd.fock_ket(n2, 0))
    k = cd.beamsplitter(n2, n2, 0.5, 0.1) @ cd.two_mode_squeezing(n2, n2, 0.3, 0.2) @ v0
    if displaced:
        s = ops.displace(s, 0.3, 1)
        k = np.kron(np.eye(n2), cd.displacement(n2, 0.3)) @ k
    return s, np.abs(k.reshape(n2, n2)) ** 2


@pytest.mark.parametrize("pattern", [[0, 0], [1, 1], [2, 0], [2, 1], [3, 1], [2, 2]])
def test_pattern_probabilities_zero_mean(pattern):
    s, P = two_mode_state()
    assert hf.pattern_probability(s, pattern) == pytest.approx(P[tuple(pattern)], abs=1e-12)


@pytest.mark.parametrize("pattern", [[1, 0], [1, 2], [0, 3], [2, 2]])
def test_pattern_probabilities_with_displacement(pattern):
    s, P = two_mode_state(displaced=True)
    assert hf.pattern_probability(s, pattern) == pytest.approx(P[tuple(pattern)], abs=1e-11)


def test_squeezed_vacuum_photon_statistics_closed_form():
    r = 0.7
    s = GaussianState.squeezed_vacuum(r)
    probs = [hf.pattern_probability(s, [n]) for n in range(8)]
    exact = np.abs(cd.squeezed_vacuum_amplitudes(8, r)) ** 2
    np.testing.assert_allclose(probs, exact, atol=1e-14)


@pytest.mark.parametrize("clicks", [[0, 0], [1, 0], [0, 1], [1, 1]])
def test_click_probabilities(clicks):
    s, P = two_mode_state(displaced=True)
    mask = [(np.arange(n2) > 0) == bool(c) for c in clicks]
    exact = P[np.ix_(mask[0], mask[1])].sum()
    assert hf.click_probability(s, clicks) == pytest.approx(exact, abs=2e-9)  # oracle cutoff error ~1e-9


def test_click_probabilities_sum_to_one():
    s = ops.interferometer(GaussianState.product(*[GaussianState.squeezed_vacuum(0.4)] * 3), np.eye(3))
    total = sum(hf.click_probability(s, [a, b, c]) for a in (0, 1) for b in (0, 1) for c in (0, 1))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_torontonian_gives_the_all_click_probability():
    """Quesada et al. (2018): P(all click) = Tor(O) / sqrt(det Q) with O = I - Q^{-1}, zero mean."""
    s, _ = two_mode_state()
    W = complex_basis(2)
    Q = W @ s.cov @ W.conj().T + np.eye(4) / 2
    O = np.eye(4) - np.linalg.inv(Q)
    value = hf.torontonian(O) / np.sqrt(np.linalg.det(Q))
    assert value.real == pytest.approx(hf.click_probability(s, [1, 1]), abs=1e-12)


def test_feature_registered():
    assert "gaussian" in ql.FEATURES
