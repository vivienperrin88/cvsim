"""The dense CV oracle (qttlind.reference.cv_dense) -- provided, checked against closed forms.

These tests pass from the start: the oracle is the fixed point every later CV test relies on.
"""

import numpy as np
import pytest

from qttlind.reference import cv_dense as cd
from qttlind.reference import dense_ref as dr

N = 50


def dm(k):
    return np.outer(k, k.conj())


def test_quadratures_commute_canonically_far_below_the_cutoff():
    x, p = cd.quadratures(N)
    c = x @ p - p @ x
    np.testing.assert_allclose(c[: N - 1, : N - 1], 1j * np.eye(N - 1), atol=1e-12)  # last level: truncation


def test_vacuum_moments():
    m, V = cd.moments(dm(cd.fock_ket(N, 0)), [N])
    np.testing.assert_allclose(m, 0, atol=1e-14)
    np.testing.assert_allclose(V, np.eye(2) / 2, atol=1e-14)


def test_displacement_of_vacuum_is_the_coherent_state():
    np.testing.assert_allclose(
        cd.displacement(N, 0.8 - 0.3j) @ cd.fock_ket(N, 0), dr.coherent(N, 0.8 - 0.3j), atol=1e-10
    )


@pytest.mark.parametrize("r,phi", [(0.3, 0.0), (0.7, 1.2)])
def test_squeezing_matches_the_closed_form(r, phi):
    k = cd.squeezing(80, r, phi) @ cd.fock_ket(80, 0)
    np.testing.assert_allclose(k[:40], cd.squeezed_vacuum_amplitudes(40, r, phi), atol=1e-10)


def test_beamsplitter_hong_ou_mandel():
    """|1,1> through a 50:50 beam splitter never gives one photon in each output."""
    n = 4
    k = cd.beamsplitter(n, n, np.pi / 4) @ np.kron(cd.fock_ket(n, 1), cd.fock_ket(n, 1))
    P = np.abs(k.reshape(n, n)) ** 2
    assert P[1, 1] == pytest.approx(0.0, abs=1e-14)
    assert P[2, 0] == pytest.approx(0.5) and P[0, 2] == pytest.approx(0.5)


def test_loss_channel_is_trace_preserving_and_shrinks_the_mean():
    k = dr.coherent(N, 1.2)
    rho = cd.loss_channel(dm(k), 0.6, 0, [N])
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-10)
    m, _ = cd.moments(rho, [N])
    assert m[0] == pytest.approx(np.sqrt(0.6) * np.sqrt(2) * 1.2, abs=1e-9)


def test_position_wavefunction_of_coherent_state():
    x = np.linspace(-6, 6, 101)
    alpha = 0.5 + 0.4j
    x0, p0 = np.sqrt(2) * alpha.real, np.sqrt(2) * alpha.imag
    exact = np.pi**-0.25 * np.exp(-((x - x0) ** 2) / 2 + 1j * p0 * x - 0.5j * x0 * p0)
    np.testing.assert_allclose(cd.position_wavefunction(dr.coherent(60, alpha), x), exact, atol=1e-10)


def test_wigner_of_vacuum_and_single_photon():
    w0 = cd.wigner_parity(dm(cd.fock_ket(20, 0)), np.array([0.0, 0.5]))
    np.testing.assert_allclose(w0, (2 / np.pi) * np.exp(-2 * np.array([0.0, 0.25])), atol=1e-12)
    w1 = cd.wigner_parity(dm(cd.fock_ket(20, 1)), np.array([0.0]))
    assert w1[0] == pytest.approx(-2 / np.pi, abs=1e-12)  # maximal negativity at the origin


def test_brute_force_hafnians():
    A = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float)
    assert cd.hafnian_recursive(A) == pytest.approx(1 * 6 + 2 * 5 + 3 * 4)
    assert cd.hafnian_recursive(np.ones((3, 3))) == 0
    L = A + np.diag([1.0, 2.0, 3.0, 4.0])
    d = [1.0, 2.0, 3.0, 4.0]
    one_edge = sum(
        A[i, j] * np.prod([d[k] for k in range(4) if k not in (i, j)]) for i in range(4) for j in range(i + 1, 4)
    )
    expected = 28 + one_edge + np.prod(d)  # perfect matchings + one edge and two loops + four loops
    assert cd.loop_hafnian_recursive(L) == pytest.approx(expected)
