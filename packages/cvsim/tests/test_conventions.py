"""The conventions, implemented twice and compared.

``cvsim.core.conventions`` writes each rule down; the oracle ``cvsim.reference.fock_dense``
*measures* it in a truncated Fock space (gates as ``expm`` of their generators, moments as traces).
Three kinds of tests (as in qttlind):

* example tests    -- hand-checked values;
* comparison tests -- conventions vs oracle on random parameters (Hypothesis);
* property tests   -- invariants (symplecticity, round trips, inverses).

Tolerances. The oracle's only error is the Fock truncation, which the chosen cutoffs push below
1e-10 for the parameter ranges drawn here (|r| <= 0.6, |α| <= 1, ...); comparisons use
``atol = 1e-9``. Pure algebra (round trips, permutations) is compared to 1e-12 or exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg as sla
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.errors import InputError, ShapeError
from cvsim.reference import fock_dense as fd

ORACLE_ATOL = 1e-9
HEAVY = settings(max_examples=12, deadline=None)  # each example is a matrix exponential
LIGHT = settings(max_examples=60, deadline=None)

angles = st.floats(-math.pi, math.pi)
small = st.floats(-0.6, 0.6)


def _random_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


# ------------------------------------------------------------------------------- ordering and Ω
class TestOrdering:
    @pytest.mark.parametrize("M", [1, 2, 3])
    def test_omega_matches_the_commutators(self, M):
        assert np.allclose(cv.omega(M), fd.omega_from_commutators(M, 3), atol=1e-14)

    def test_omega_example(self):
        assert np.array_equal(cv.omega(1), np.array([[0.0, 1.0], [-1.0, 0.0]]))

    def test_mode_indices(self):
        assert cv.mode_indices([2, 0], 3) == [2, 0, 5, 3]

    @pytest.mark.parametrize("modes", [[], [0, 0], [3], [-1]])
    def test_mode_indices_rejects(self, modes):
        with pytest.raises(InputError, match=r"distinct integers in 0\.\.2"):
            cv.mode_indices(modes, 3)

    @pytest.mark.parametrize("M", [0, -1, 1.5, True, "2"])
    def test_number_of_modes_is_validated(self, M):
        with pytest.raises(InputError, match="number of modes"):
            cv.omega(M)

    def test_xpxp_permutation_example(self):
        assert cv.xpxp_permutation(2) == [0, 2, 1, 3]
        assert cv.xpxp_permutation(3) == [0, 3, 1, 4, 2, 5]

    @LIGHT
    @given(M=st.integers(1, 6), seed=st.integers(0, 2**32 - 1))
    def test_xpxp_round_trip(self, M, seed):
        rng = np.random.default_rng(seed)
        mu, V = rng.normal(size=2 * M), rng.normal(size=(2 * M, 2 * M))
        m2, V2 = cv.from_xpxp(*cv.to_xpxp(mu, V))
        assert np.array_equal(m2, mu) and np.array_equal(V2, V)

    def test_xpxp_puts_each_mode_together(self):
        mu = np.array([1.0, 2.0, 10.0, 20.0])  # x1 x2 p1 p2
        assert np.array_equal(cv.to_xpxp(mu, np.eye(4))[0], [1.0, 10.0, 2.0, 20.0])

    @pytest.mark.parametrize(
        ("mean", "cov"), [(np.zeros(3), np.eye(3)), (np.zeros(2), np.eye(4)), (np.zeros(0), np.zeros((0, 0)))]
    )
    def test_moment_shapes_are_validated(self, mean, cov):
        with pytest.raises(ShapeError):
            cv.to_xpxp(mean, cov)

    @pytest.mark.parametrize("M", [1, 2])
    def test_complex_basis_maps_quadratures_to_ladder_operators(self, M):
        N = 4
        r = fd.quadratures(M, N)
        W = cv.complex_basis(M)
        a = fd.mode_operators(M, N)
        for i in range(M):
            assert np.allclose(sum(W[i, j] * r[j] for j in range(2 * M)), a[i], atol=1e-14)
            assert np.allclose(sum(W[M + i, j] * r[j] for j in range(2 * M)), a[i].conj().T, atol=1e-14)
        assert np.allclose(W.conj().T @ W, np.eye(2 * M), atol=1e-14)


# ------------------------------------------------------------------------------- vacuum and units
class TestUnits:
    def test_vacuum_moments(self):
        mean, cov = fd.moments(fd.dm(fd.vacuum(2, 6)), 2, 6)
        assert np.allclose(mean, 0, atol=1e-14)
        assert np.allclose(cov, cv.VACUUM_VARIANCE * np.eye(4), atol=1e-14)
        assert cv.HBAR == 2 * cv.VACUUM_VARIANCE

    def test_hbar_two_vacuum_is_identity(self):
        _, V = cv.convert_hbar(np.zeros(2), np.eye(2) / 2, 1, 2)
        assert np.allclose(V, np.eye(2))

    @LIGHT
    @given(h1=st.floats(0.1, 4), h2=st.floats(0.1, 4), seed=st.integers(0, 2**32 - 1))
    def test_hbar_round_trip_and_scaling(self, h1, h2, seed):
        rng = np.random.default_rng(seed)
        mu, V = rng.normal(size=4), rng.normal(size=(4, 4))
        m2, V2 = cv.convert_hbar(mu, V, h1, h2)
        assert np.allclose(V2, V * h2 / h1, rtol=1e-12)
        assert np.allclose(m2, mu * math.sqrt(h2 / h1), rtol=1e-12)
        m3, V3 = cv.convert_hbar(m2, V2, h2, h1)
        assert np.allclose(m3, mu, rtol=1e-12) and np.allclose(V3, V, rtol=1e-12)

    @pytest.mark.parametrize(("h1", "h2"), [(0, 1), (1, -2), (math.nan, 1)])
    def test_hbar_must_be_positive(self, h1, h2):
        with pytest.raises(InputError):
            cv.convert_hbar(np.zeros(2), np.eye(2), h1, h2)

    def test_snu(self):
        assert cv.to_snu(0.5) == 1.0 and cv.from_snu(1.0) == 0.5

    def test_db_examples(self):
        assert cv.db_from_variance(0.25) == pytest.approx(-10 * math.log10(2))  # half the vacuum noise
        assert cv.db_from_r(1.0) == pytest.approx(8.685889638)
        assert pytest.approx(20 / math.log(10)) == cv.DB_PER_NEPER

    @LIGHT
    @given(r=st.floats(-3, 3))
    def test_db_of_a_pure_squeezed_vacuum(self, r):
        """The squeezed variance e^{-2r}/2 is -db_from_r(r) dB below the vacuum."""
        assert cv.db_from_variance(math.exp(-2 * r) / 2) == pytest.approx(-cv.db_from_r(r), abs=1e-12)
        assert cv.r_from_db(cv.db_from_r(r)) == pytest.approx(r, abs=1e-14)
        assert cv.variance_from_db(cv.db_from_variance(math.exp(2 * r) / 2)) == pytest.approx(math.exp(2 * r) / 2)

    @pytest.mark.parametrize("v", [0.0, -1.0, math.nan])
    def test_db_needs_positive_variance(self, v):
        with pytest.raises(InputError, match="> 0"):
            cv.db_from_variance(v)


# ------------------------------------------------------------------------------- gates: conventions vs oracle
def _check_gate(S, U, M, N, atol=ORACLE_ATOL):
    S_oracle, c = fd.heisenberg_map(U, M, N)
    assert np.allclose(S, S_oracle, atol=atol), np.abs(S - S_oracle).max()
    assert np.allclose(c, 0, atol=atol)
    assert cv.is_symplectic(S)


class TestGates:
    @HEAVY
    @given(theta=angles)
    def test_rotation(self, theta):
        _check_gate(cv.rotation(theta), fd.rotation(theta, 1, 30), 1, 30)

    def test_rotation_turns_the_mean_counter_clockwise(self):
        assert np.allclose(cv.rotation(math.pi / 2) @ [1.0, 0.0], [0.0, 1.0])

    @HEAVY
    @given(r=small, phi=angles)
    def test_squeezing(self, r, phi):
        _check_gate(cv.squeezing(r, phi), fd.squeezing(r, phi, 1, 60), 1, 60)

    def test_squeezing_example(self):
        assert np.allclose(cv.squeezing(0.5), np.diag([math.exp(-0.5), math.exp(0.5)]))

    @HEAVY
    @given(theta=angles, phi=angles)
    def test_beamsplitter(self, theta, phi):
        _check_gate(cv.beamsplitter(theta, phi), fd.beamsplitter(theta, phi, 2, 8), 2, 8)

    def test_beamsplitter_transmissivity(self):
        S = cv.beamsplitter(0.3)
        assert S[0, 0] ** 2 == pytest.approx(math.cos(0.3) ** 2)

    @HEAVY
    @given(r=st.floats(-0.4, 0.4), phi=angles)
    def test_two_mode_squeezing(self, r, phi):
        _check_gate(cv.two_mode_squeezing(r, phi), fd.two_mode_squeezing(r, phi, 2, 16), 2, 16)

    @HEAVY
    @given(seed=st.integers(0, 2**32 - 1))
    def test_interferometer(self, seed):
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        H = (H + H.conj().T) / 2
        a = fd.mode_operators(2, 8)
        gen = sum(H[j, k] * a[j].conj().T @ a[k] for j in range(2) for k in range(2))
        # U = exp(i a†Ha) gives U† a U = e^{iH} a
        _check_gate(cv.interferometer(sla.expm(1j * H)), sla.expm(1j * gen), 2, 8)

    @HEAVY
    @given(seed=st.integers(0, 2**32 - 1), M=st.integers(2, 3))
    def test_cz_graph(self, seed, M):
        rng = np.random.default_rng(seed)
        A = rng.uniform(-0.5, 0.5, size=(M, M))
        A = (A + A.T) / 2
        # three modes need a large cutoff: U is applied to 1 + 2M vectors only (expm_multiply, whose
        # own accuracy is ~1e-9 here, hence the looser tolerance)
        N = 20 if M == 2 else 14
        S_oracle, c = fd.heisenberg_map_from_generator(fd.cz_generator(A, N), M, N)
        assert np.allclose(cv.cz(A), S_oracle, atol=1e-8), np.abs(cv.cz(A) - S_oracle).max()
        assert np.allclose(c, 0, atol=1e-8)

    def test_dense_and_generator_routes_agree(self):
        A = np.array([[0.1, 0.4], [0.4, -0.2]])
        S1, _ = fd.heisenberg_map(fd.cz(A, 16), 2, 16)
        S2, _ = fd.heisenberg_map_from_generator(fd.cz_generator(A, 16), 2, 16)
        assert np.allclose(S1, S2, atol=1e-8)

    def test_cz_two_modes_is_exp_i_g_xa_xb(self):
        g = 0.7
        S = cv.cz([[0, g], [g, 0]])
        assert S[2, 1] == g and S[3, 0] == g  # p_a += g x_b, p_b += g x_a
        r = fd.quadratures(2, 20)
        _check_gate(S, sla.expm(1j * g * r[0] @ r[1]), 2, 20)

    @HEAVY
    @given(g=small)
    def test_cx(self, g):
        _check_gate(cv.cx(g), fd.cx(g, 2, 20), 2, 20)

    @HEAVY
    @given(s=small)
    def test_shear(self, s):
        _check_gate(cv.shear(s), fd.shear(s, 1, 60), 1, 60)

    def test_fourier_is_rotation_by_plus_pi_over_two(self):
        F = cv.fourier()
        assert np.allclose(F, cv.rotation(math.pi / 2), atol=1e-15)
        _check_gate(F, fd.rotation(math.pi / 2, 1, 10), 1, 10)
        # Heisenberg picture: U† x U = −p (row 0 of S), U† p U = x
        assert np.array_equal(F[0], [0.0, -1.0]) and np.array_equal(F[1], [1.0, 0.0])

    @HEAVY
    @given(re=st.floats(-1, 1), im=st.floats(-1, 1))
    def test_displacement(self, re, im):
        alpha = complex(re, im)
        S, c = fd.heisenberg_map(fd.displacement(alpha, 1, 40), 1, 40)
        assert np.allclose(S, np.eye(2), atol=ORACLE_ATOL)
        assert np.allclose(c, cv.displacement_vector(alpha), atol=ORACLE_ATOL)

    def test_displacement_of_several_modes(self):
        assert np.allclose(cv.displacement_vector([1, 2j]), math.sqrt(2) * np.array([1, 0, 0, 2]))

    def test_heisenberg_map_rejects_non_gaussian_gates(self):
        n = fd.mode_operators(1, 20)[0]
        kerr = sla.expm(0.3j * (n.conj().T @ n) @ (n.conj().T @ n))
        with pytest.raises(ValueError, match="not a real linear map"):
            fd.heisenberg_map(kerr, 1, 20)
        x = fd.quadratures(1, 30)[0]
        with pytest.raises(ValueError, match="not a real linear map"):  # cubic phase
            fd.heisenberg_map_from_generator(0.2j * x @ x @ x / 3, 1, 30)

    def test_a_too_small_cutoff_is_detected(self):
        with pytest.raises(ValueError, match="cutoff is too small"):
            fd.heisenberg_map(fd.squeezing(1.5, 0.0, 1, 6), 1, 6)

    def test_bogoliubov_rejects_bad_shapes(self):
        with pytest.raises(ShapeError):
            cv.symplectic_from_bogoliubov(np.eye(2), np.eye(3))

    @pytest.mark.parametrize(
        "call",
        [
            lambda: cv.rotation(math.inf),
            lambda: cv.squeezing(math.nan),
            lambda: cv.beamsplitter(0.1, math.inf),
            lambda: cv.two_mode_squeezing("r"),
            lambda: cv.cx(math.nan),
            lambda: cv.shear(math.inf),
            lambda: cv.homodyne_vector(math.nan),
            lambda: cv.displacement_vector(complex(math.nan, 0)),
            lambda: cv.displacement_vector([[1, 2]]),
        ],
    )
    def test_parameters_must_be_finite(self, call):
        with pytest.raises((InputError, ValueError)):
            call()

    @pytest.mark.parametrize(
        "A", [np.array([[0, 1], [2, 0]]), np.array([[0, 1j], [1j, 0]]), np.ones((2, 3)), np.array([[math.inf]])]
    )
    def test_cz_needs_real_symmetric_finite(self, A):
        with pytest.raises(InputError):
            cv.cz(A)

    def test_interferometer_needs_unitary(self):
        with pytest.raises(InputError, match="unitary"):
            cv.interferometer([[1, 1], [0, 1]])

    @LIGHT
    @given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 5))
    def test_interferometers_are_symplectic_and_orthogonal(self, seed, n):
        S = cv.interferometer(_random_unitary(np.random.default_rng(seed), n))
        assert cv.is_symplectic(S) and np.allclose(S @ S.T, np.eye(2 * n), atol=1e-12)

    @pytest.mark.parametrize("bad", [np.eye(3), np.ones((2, 2, 2)), np.zeros((0, 0)), 2 * np.eye(2)])
    def test_is_symplectic_negative_cases(self, bad):
        assert not cv.is_symplectic(bad)

    def test_embed(self):
        S = cv.embed(cv.squeezing(0.3), [1], 3)
        assert S.shape == (6, 6) and S[1, 1] == pytest.approx(math.exp(-0.3)) and S[0, 0] == 1
        assert cv.is_symplectic(cv.embed(cv.beamsplitter(0.4, 0.2), [2, 0], 3))
        with pytest.raises(ShapeError):
            cv.embed(np.eye(4), [0], 2)


# ------------------------------------------------------------------------------- states, measurements, channels
class TestSqueezedVacuum:
    @HEAVY
    @given(r=small, phi=angles)
    def test_covariance_and_squeezed_axis(self, r, phi):
        N = 60
        _, V = fd.moments(fd.dm(fd.squeezing(r, phi, 1, N) @ fd.vacuum(1, N)), 1, N)
        S = cv.squeezing(r, phi)
        assert np.allclose(V, S @ S.T / 2, atol=ORACLE_ATOL)
        u = cv.homodyne_vector(phi / 2)
        assert u @ V @ u == pytest.approx(math.exp(-2 * r) / 2, abs=ORACLE_ATOL)


class TestHomodyne:
    @HEAVY
    @given(phi=angles, re=st.floats(-1, 1), im=st.floats(-1, 1))
    def test_density_of_x_phi(self, phi, re, im):
        """The oracle density of x_φ is Gaussian with mean u·μ and variance uᵀVu, u = homodyne_vector(φ)."""
        N = 40
        state = fd.squeezing(0.3, 0.7, 1, N) @ fd.coherent([complex(re, im)], N)
        rho = fd.dm(state)
        mean, V = fd.moments(rho, 1, N)
        u = cv.homodyne_vector(phi)
        q = np.linspace(-4, 4, 9)
        mu, var = u @ mean, u @ V @ u
        expected = np.exp(-((q - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        assert np.allclose(fd.homodyne_density(rho, phi, q, N), expected, atol=1e-8)

    def test_x_phi_is_x_after_rotating_by_minus_phi(self):
        N, phi = 30, 0.9
        rho = fd.dm(fd.coherent([0.4 + 0.8j], N))
        R = fd.rotation(-phi, 1, N)
        q = np.linspace(-3, 3, 7)
        assert np.allclose(fd.homodyne_density(rho, phi, q, N), fd.homodyne_density(R @ rho @ R.conj().T, 0, q, N))

    def test_x_at_pi_over_two_is_p(self):
        assert np.allclose(cv.homodyne_vector(math.pi / 2), [0, 1], atol=1e-16)


class TestLoss:
    @HEAVY
    @given(eta=st.floats(0, 1))
    def test_pure_loss_matches_kraus(self, eta):
        N = 40
        rho = fd.dm(fd.squeezing(0.4, 0.3, 1, N) @ fd.coherent([0.6 - 0.2j], N))
        mean, V = fd.moments(rho, 1, N)
        m2, V2 = fd.moments(fd.apply_kraus(rho, fd.loss_kraus(eta, N), 1, N), 1, N)
        X, Y = cv.loss_xy(eta)
        assert np.allclose(m2, X @ mean, atol=ORACLE_ATOL)
        assert np.allclose(V2, X @ V @ X.T + Y, atol=ORACLE_ATOL)

    def test_lindblad_loss_is_the_channel_with_eta_exp_minus_kappa_t(self):
        N, kappa, t = 14, 0.8, 0.9
        rho = fd.dm(fd.coherent([0.7 + 0.3j], N))
        out = fd.lindblad_loss(rho, kappa, t, N)
        assert np.allclose(out, fd.apply_kraus(rho, fd.loss_kraus(math.exp(-kappa * t), N), 1, N), atol=1e-10)

    @pytest.mark.parametrize(("eta", "nbar"), [(0.7, 0.3), (0.2, 1.5), (1.0, 2.0)])
    def test_thermal_loss_is_a_beamsplitter_with_a_thermal_environment(self, eta, nbar):
        N = 30  # the thermal tail (n̄/(1+n̄))^N and the beam splitter's cutoff stay below 1e-6
        sys = fd.dm(fd.coherent([0.5], N))
        p = (nbar / (1 + nbar)) ** np.arange(N) / (1 + nbar)
        env = np.diag(p / p.sum()).astype(complex)
        U = fd.beamsplitter(math.acos(math.sqrt(eta)), 0.0, 2, N)
        out = U @ np.kron(sys, env) @ U.conj().T
        reduced = np.einsum("ikjk->ij", out.reshape(N, N, N, N))
        mean, V = fd.moments(reduced, 1, N)
        X, Y = cv.loss_xy(eta, nbar)
        m0, V0 = fd.moments(sys, 1, N)
        assert np.allclose(mean, X @ m0, atol=1e-6)
        assert np.allclose(V, X @ V0 @ X.T + Y, atol=1e-6)

    @pytest.mark.parametrize(("eta", "nbar"), [(-0.1, 0), (1.1, 0), (0.5, -1), (math.nan, 0)])
    def test_rejects(self, eta, nbar):
        with pytest.raises(InputError):
            cv.loss_xy(eta, nbar)


# ------------------------------------------------------------------------------- the oracle checks itself
class TestOracle:
    def test_coherent_ket_two_ways(self):
        N = 30
        assert np.allclose(
            fd.coherent([0.5 - 0.3j], N), fd.displacement(0.5 - 0.3j, 1, N) @ fd.vacuum(1, N), atol=1e-10
        )

    def test_fock_ket(self):
        v = fd.fock([1, 2], 3)
        assert v[1 * 3 + 2] == 1 and np.count_nonzero(v) == 1

    def test_hermite_functions_are_orthonormal(self):
        q = np.linspace(-12, 12, 4001)
        psi = fd.hermite_functions(8, q)
        G = psi @ psi.T * (q[1] - q[0])
        assert np.allclose(G, np.eye(8), atol=1e-10)
        assert np.allclose(fd.hermite_functions(1, q), psi[:1])

    def test_wigner_of_vacuum_and_one_photon(self):
        N = 20
        assert fd.wigner(fd.dm(fd.vacuum(1, N)), 0, 0, N) == pytest.approx(1 / math.pi)
        assert fd.wigner(fd.dm(fd.fock([1], N)), 0, 0, N) == pytest.approx(-1 / math.pi)
        x = 0.7
        assert fd.wigner(fd.dm(fd.vacuum(1, N)), x, 0, N) == pytest.approx(math.exp(-(x**2)) / math.pi, abs=1e-10)
