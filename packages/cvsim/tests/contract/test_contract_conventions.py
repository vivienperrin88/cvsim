"""Contract: cvsim's conventions are qttlind's, number for number (CONVENTIONS.md, decision D4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from harness import EXACT, ROUNDOFF, assert_same, qttlind_module, random_unitary
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv

pytestmark = pytest.mark.qttlind

angles = st.floats(-math.pi, math.pi)
params = st.floats(-2.0, 2.0)
FAST = settings(max_examples=100, deadline=None)


@pytest.fixture(scope="module")
def ps():
    return qttlind_module("gaussian.phase_space")


@pytest.fixture(scope="module")
def ops():
    return qttlind_module("gaussian.ops")


@pytest.fixture(scope="module")
def states():
    return qttlind_module("gaussian.states")


@pytest.mark.parametrize("M", [1, 2, 5])
def test_omega(ps, M):
    assert_same(cv.omega(M), ps.omega(M), 0.0, "Ω")


def test_mode_indices(ps):
    assert cv.mode_indices([2, 0], 4) == ps.mode_indices([2, 0], 4)


@pytest.mark.parametrize("M", [1, 3])
def test_complex_basis(ps, M):
    assert_same(cv.complex_basis(M), ps.complex_basis(M), EXACT, "W")


@FAST
@given(seed=st.integers(0, 2**32 - 1), M=st.integers(1, 4))
def test_bogoliubov_to_symplectic(seed, M):
    ps = qttlind_module("gaussian.phase_space")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
    B = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
    assert_same(cv.symplectic_from_bogoliubov(A, B), ps.symplectic_from_bogoliubov(A, B), EXACT, "S(A, B)")


@FAST
@given(theta=angles)
def test_rotation(theta):
    ops = qttlind_module("gaussian.ops")
    assert_same(cv.rotation(theta), ops.rotation_matrix(theta), EXACT, "R")


@FAST
@given(r=params, phi=angles)
def test_squeezing(r, phi):
    ops = qttlind_module("gaussian.ops")
    assert_same(cv.squeezing(r, phi), ops.squeezing_matrix(r, phi), EXACT * math.cosh(r), "S")


@FAST
@given(theta=angles, phi=angles)
def test_beamsplitter(theta, phi):
    ops = qttlind_module("gaussian.ops")
    assert_same(cv.beamsplitter(theta, phi), ops.beamsplitter_matrix(theta, phi), EXACT, "BS")


@FAST
@given(r=params, phi=angles)
def test_two_mode_squeezing(r, phi):
    ops = qttlind_module("gaussian.ops")
    assert_same(cv.two_mode_squeezing(r, phi), ops.two_mode_squeezing_matrix(r, phi), EXACT * math.cosh(r), "S2")


@FAST
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 6))
def test_interferometer(seed, n):
    ops = qttlind_module("gaussian.ops")
    U = random_unitary(np.random.default_rng(seed), n)
    assert_same(cv.interferometer(U), ops.interferometer_matrix(U), EXACT, "interferometer")


def test_fourier_is_the_inverse_of_qttlind_F(ops):
    """cvsim: F = R(π/2) (MBQC literature); qttlind: F = R(−π/2) (unitary DFT of wavefunctions)."""
    assert_same(cv.fourier(), ops.rotation_matrix(math.pi / 2), EXACT, "F")
    assert_same(cv.fourier() @ ops.rotation_matrix(-math.pi / 2), np.eye(2), EXACT, "F · F_qttlind")


@FAST
@given(re=params, im=params)
def test_displacement(re, im):
    ops = qttlind_module("gaussian.ops")
    states = qttlind_module("gaussian.states")
    out = ops.displace(states.GaussianState.vacuum(1), complex(re, im))
    assert_same(cv.displacement_vector(complex(re, im)), out.mean, EXACT, "D(α) mean")


@FAST
@given(r=params, phi=angles)
def test_squeezed_vacuum_covariance(r, phi):
    states = qttlind_module("gaussian.states")
    S = cv.squeezing(r, phi)
    assert_same(
        S @ S.T * cv.VACUUM_VARIANCE,
        states.GaussianState.squeezed_vacuum(r, phi).cov,
        ROUNDOFF * math.exp(2 * abs(r)),
        "V",
    )


@FAST
@given(eta=st.floats(0, 1), nbar=st.floats(0, 5), seed=st.integers(0, 2**32 - 1))
def test_loss_channel(eta, nbar, seed):
    ops = qttlind_module("gaussian.ops")
    states = qttlind_module("gaussian.states")
    rng = np.random.default_rng(seed)
    st0 = ops.squeeze(states.GaussianState.coherent(complex(*rng.normal(size=2))), 0.5, 0.3)
    X, Y = cv.loss_xy(eta, nbar)
    out = ops.loss(st0, eta, 0, nbar)
    assert_same(X @ st0.mean, out.mean, ROUNDOFF, "loss mean")
    assert_same(X @ st0.cov @ X.T + Y, out.cov, ROUNDOFF * (1 + nbar), "loss cov")


@FAST
@given(phi=angles, q=st.floats(-3, 3))
def test_homodyne_angle(phi, q):
    measure = qttlind_module("gaussian.measure")
    states = qttlind_module("gaussian.states")
    st0 = states.GaussianState(np.array([0.4, -0.9]), np.array([[0.8, 0.3], [0.3, 0.6]]))
    u = cv.homodyne_vector(phi)
    mu, var = u @ st0.mean, u @ st0.cov @ u
    expected = math.exp(-((q - mu) ** 2) / (2 * var)) / math.sqrt(2 * math.pi * var)
    assert_same(expected, measure.homodyne_density(st0, q, 0, phi), ROUNDOFF, "homodyne density")


def test_db_per_neper():
    hardware = qttlind_module("circuits.hardware")
    for r in (0.1, 1.0, 2.3):
        assert_same(cv.db_from_r(r), hardware.squeezing_db(r), EXACT, "dB")


def test_vacuum_variance():
    states = qttlind_module("gaussian.states")
    assert_same(cv.VACUUM_VARIANCE * np.eye(4), states.GaussianState.vacuum(2).cov, 0.0, "vacuum")
