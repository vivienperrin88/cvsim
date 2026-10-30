"""cvsim.gaussian.symplectic and cvsim.gaussian.ops.

* ``from_hamiltonian`` against the Fock oracle (Heisenberg map of ``exp(−itH)`` measured from the
  generator) and against the closed-form gates of ``core.conventions``;
* the Clements mesh on Haar-random unitaries (Hypothesis, up to 10 modes) and on the degenerate
  cases where nulling angles are undefined (identity, permutations, diagonal phases, the DFT):
  reconstruction to 1e-12, ``N(N−1)/2`` cells, depth ``N``, and its gate list run through
  ``ops`` reproduces ``interferometer(U)``;
* ``preparation`` and ``change_basis`` round trips and invariants;
* ``ops``: gate application against the Fock oracle (moments of ``U ρ U†``), composition order,
  validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.errors import InputError, ShapeError
from cvsim.gaussian import GaussianState, ops
from cvsim.gaussian import symplectic as sy
from cvsim.reference import fock_dense as fd

SETTINGS = settings(max_examples=40, deadline=None)
seeds = st.integers(0, 2**32 - 1)


# ------------------------------------------------------------------------------- Hamiltonians
class TestFromHamiltonian:
    def test_rotation_and_squeezing_closed_forms(self):
        theta = 0.7
        assert np.allclose(sy.from_hamiltonian(-np.eye(2), theta), cv.rotation(theta), atol=1e-14)
        # S(r, 0) = exp(r(a² − a†²)/2) = exp(i r (xp + px)/2) = exp(−i r H), H = ½ rᵀ [[0, −1], [−1, 0]] r
        r = 0.4
        assert np.allclose(sy.from_hamiltonian(-np.array([[0.0, 1.0], [1.0, 0.0]]), r), cv.squeezing(r), atol=1e-14)

    def test_against_fock_oracle(self, rng):
        """Two modes, random quadratic H: the oracle measures U† r U of U = exp(−i t Ĥ)."""
        M, N, t = 2, 14, 0.3
        H = rng.normal(size=(4, 4)) * 0.5
        H = (H + H.T) / 2
        r = fd.quadratures(M, N)
        Hop = sum(0.5 * H[a, b] * (r[a] @ r[b]) for a in range(4) for b in range(4))
        S_oracle, c = fd.heisenberg_map_from_generator(-1j * t * Hop, M, N)
        assert np.allclose(sy.from_hamiltonian(H, t), S_oracle, atol=1e-9)
        assert np.allclose(c, 0, atol=1e-10)

    @SETTINGS
    @given(seed=seeds, M=st.integers(1, 4))
    def test_group_property(self, seed, M):
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(2 * M, 2 * M))
        H = H + H.T
        S1, S2, S12 = sy.from_hamiltonian(H, 0.2), sy.from_hamiltonian(H, 0.3), sy.from_hamiltonian(H, 0.5)
        assert cv.is_symplectic(S12, 1e-9)
        assert np.allclose(S2 @ S1, S12, atol=1e-10 * np.abs(S12).max())
        assert np.allclose(sy.from_hamiltonian(H, -0.5) @ S12, np.eye(2 * M), atol=1e-9 * np.abs(S12).max() ** 2)

    @pytest.mark.parametrize(
        ("H", "exc"),
        [
            (np.eye(3), ShapeError),
            (np.array([[0.0, 1.0], [0.0, 0.0]]), InputError),
            (np.full((2, 2), np.inf), InputError),
        ],
    )
    def test_rejects(self, H, exc):
        with pytest.raises(exc):
            sy.from_hamiltonian(H)


# ------------------------------------------------------------------------------- passive maps
class TestPassive:
    @SETTINGS
    @given(seed=seeds, M=st.integers(1, 6))
    def test_random_interferometer_round_trip(self, seed, M):
        rng = np.random.default_rng(seed)
        U = sy.random_interferometer(rng, M)
        assert np.allclose(U.conj().T @ U, np.eye(M), atol=1e-12)
        S = sy.interferometer(U)
        assert sy.is_passive(S)
        assert np.allclose(sy.unitary_from_passive(S), U, atol=1e-14)

    def test_active_is_not_passive(self, rng):
        S = sy.random_symplectic(rng, 2, r_max=1.0)
        assert cv.is_symplectic(S, 1e-10) and not sy.is_passive(S)
        with pytest.raises(InputError, match="passive"):
            sy.unitary_from_passive(S)
        assert not sy.is_passive(np.eye(3))

    def test_random_interferometer_rejects_zero_modes(self, rng):
        with pytest.raises(InputError):
            sy.random_interferometer(rng, 0)


# ------------------------------------------------------------------------------- Clements
def run_gates(state: GaussianState, gates) -> GaussianState:
    for name, par, modes in gates:
        if name == "R":
            state = ops.rotate(state, par[0], modes[0])
        else:
            assert name == "BS"
            state = ops.beamsplitter(state, *par, modes=modes)
    return state


def _check_mesh(U: np.ndarray) -> None:
    N = U.shape[0]
    c = sy.clements(U)
    assert len(c.cells) == N * (N - 1) // 2
    assert c.depth() <= N
    assert np.allclose(np.abs(c.phases), 1.0, atol=1e-14)
    assert np.abs(c.unitary() - U).max() < 1e-12
    for m, theta, phi in c.cells:
        assert 0 <= m < N - 1 and 0 <= theta <= math.pi / 2 + 1e-15 and abs(phi) <= math.pi + 1e-15


class TestClements:
    @SETTINGS
    @given(seed=seeds, N=st.integers(1, 10))
    def test_haar_random(self, seed, N):
        _check_mesh(sy.random_interferometer(np.random.default_rng(seed), N))

    @pytest.mark.parametrize("N", [2, 3, 4, 7])
    def test_degenerate_unitaries(self, N):
        """Zero entries make nulling angles undefined: identity, permutations, phases, DFT."""
        rng = np.random.default_rng(N)
        _check_mesh(np.eye(N, dtype=complex))
        _check_mesh(np.eye(N, dtype=complex)[::-1])
        _check_mesh(np.eye(N, dtype=complex)[rng.permutation(N)])
        _check_mesh(np.diag(np.exp(1j * rng.uniform(-3, 3, N))))
        k = np.arange(N)
        _check_mesh(np.exp(2j * np.pi * np.outer(k, k) / N) / np.sqrt(N))

    def test_full_mesh_depth_is_n(self, rng):
        assert sy.clements(sy.random_interferometer(rng, 6)).depth() == 6

    def test_gate_list_reproduces_the_interferometer(self, rng):
        U = sy.random_interferometer(rng, 4)
        state = GaussianState.from_symplectic(
            sy.random_symplectic(rng, 4), nu=[0.5, 0.7, 1.0, 2.0], mean=rng.normal(size=8)
        )
        assert run_gates(state, sy.clements(U).gates()).isclose(ops.interferometer(state, U), 1e-10)

    def test_mzi_cell_is_bs_after_phase(self):
        theta, phi = 0.4, 1.1
        expected = cv.beamsplitter(theta, 0.0) @ cv.embed(cv.rotation(phi), [0], 2)
        assert np.allclose(cv.interferometer(sy.mzi(theta, phi)), expected, atol=1e-15)

    @pytest.mark.parametrize("U", [np.array([[1.0, 1.0], [0.0, 1.0]]), np.ones((2, 3)), np.zeros((0, 0))])
    def test_rejects(self, U):
        with pytest.raises(InputError):
            sy.clements(U)


# ------------------------------------------------------------------------------- preparation and bases
@st.composite
def states(draw, M=None, pure=None):
    M = draw(st.integers(1, 4)) if M is None else M
    rng = np.random.default_rng(draw(seeds))
    pure = draw(st.booleans()) if pure is None else pure
    nu = 0.5 + (0.0 if pure else rng.uniform(0.05, 1.5, M))
    return GaussianState.from_symplectic(sy.random_symplectic(rng, M, 1.2), nu=nu, mean=rng.normal(size=2 * M))


class TestPreparation:
    @SETTINGS
    @given(s=states())
    def test_round_trip(self, s):
        p = sy.preparation(s)
        c = max(1.0, float(np.abs(s.cov).max()))
        assert p.state().isclose(s, 1e-9 * c)
        assert np.all(np.diff(p.r) <= 1e-12) and np.all(p.r >= 0)
        assert np.all(np.diff(p.nu) >= -1e-12) and np.all(p.nu >= 0.5 - 1e-12)
        for U in (p.U1, p.U2):
            assert np.allclose(U.conj().T @ U, np.eye(s.num_modes), atol=1e-9)

    @SETTINGS
    @given(s=states(pure=True))
    def test_pure_state_needs_no_second_interferometer(self, s):
        p = sy.preparation(s)
        assert np.array_equal(p.nu, np.full(s.num_modes, 0.5)) and np.array_equal(p.U2, np.eye(s.num_modes))
        assert p.state().isclose(s, 1e-9 * max(1.0, float(np.abs(s.cov).max())))

    def test_squeezed_interferometer_output(self, rng):
        s = ops.interferometer(GaussianState.squeezed([0.3, 0.6]), sy.random_interferometer(rng, 2))
        assert np.allclose(sy.preparation(s).r, [0.6, 0.3], atol=1e-12)


class TestChangeBasis:
    @SETTINGS
    @given(s=states())
    def test_round_trip_and_invariants(self, s):
        rng = np.random.default_rng(s.num_modes)
        U = sy.random_interferometer(rng, s.num_modes)
        t = sy.change_basis(s, U)
        assert sy.change_basis(t, U.conj().T).isclose(s, 1e-10 * max(1.0, float(np.abs(s.cov).max())))
        assert math.isclose(t.purity(), s.purity(), rel_tol=1e-9)
        assert math.isclose(t.mean_photons().sum(), s.mean_photons().sum(), rel_tol=1e-10, abs_tol=1e-10)

    def test_mode_function_convention(self):
        """b = (a₀ + a₁)/√2: a coherent state |α, α⟩ has all its amplitude in b₀ = √2 α."""
        alpha = 0.3 + 0.2j
        U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # column k = mode function of b_k
        t = sy.change_basis(GaussianState.coherent([alpha, alpha]), U)
        assert np.allclose(t.complex_moments()[0], [math.sqrt(2) * alpha, 0], atol=1e-14)

    def test_supermodes_diagonalise_a_pure_state(self, rng):
        from cvsim.core import linalg as la

        s = ops.interferometer(GaussianState.squeezed([0.2, 0.5, 0.8]), sy.random_interferometer(rng, 3))
        bm = la.bloch_messiah(la.williamson(s.cov).S)
        t = sy.change_basis(s, sy.unitary_from_passive(bm.O1))
        expected = np.diag(np.concatenate([np.exp(-2 * bm.r), np.exp(2 * bm.r)]) / 2)
        assert np.allclose(t.cov, expected, atol=1e-12)

    def test_subset_of_modes(self):
        s = GaussianState.coherent([0.1, 0.2, 0.3])
        U = np.array([[0, 1], [1, 0]], dtype=complex)
        t = sy.change_basis(s, U, modes=[0, 2])
        assert np.allclose(t.complex_moments()[0], [0.3, 0.2, 0.1])


# ------------------------------------------------------------------------------- ops
class TestOps:
    def test_gates_on_states_match_the_fock_oracle(self):
        """Squeeze, displace, beam splitter, rotation applied to a density matrix and to the moments."""
        M, N = 2, 22
        psi = fd.vacuum(M, N)
        psi = fd.squeezing(0.3, 0.4, M, N, mode=0) @ psi
        psi = fd.displacement(0.2 - 0.1j, M, N, mode=1) @ psi
        psi = fd.beamsplitter(0.6, 0.3, M, N, modes=(0, 1)) @ psi
        psi = fd.rotation(0.9, M, N, mode=1) @ psi
        d, V = fd.moments(fd.dm(psi), M, N)
        s = GaussianState.vacuum(2)
        s = ops.squeeze(s, 0.3, 0.4, mode=0)
        s = ops.displace(s, 0.2 - 0.1j, modes=1)
        s = ops.beamsplitter(s, 0.6, 0.3, modes=(0, 1))
        s = ops.rotate(s, 0.9, mode=1)
        assert np.allclose(s.mean, d, atol=1e-10) and np.allclose(s.cov, V, atol=1e-10)

    def test_every_gate_is_its_matrix(self, rng):
        s = GaussianState.from_symplectic(sy.random_symplectic(rng, 3), nu=[0.6, 0.9, 1.4], mean=rng.normal(size=6))
        A = np.array([[0.0, 0.5], [0.5, 0.0]])
        cases = [
            (ops.rotate(s, 0.3, 2), cv.embed(cv.rotation(0.3), [2], 3)),
            (ops.squeeze(s, 0.2, 0.5, 1), cv.embed(cv.squeezing(0.2, 0.5), [1], 3)),
            (ops.beamsplitter(s, 0.4, 0.1, (2, 0)), cv.embed(cv.beamsplitter(0.4, 0.1), [2, 0], 3)),
            (ops.two_mode_squeeze(s, 0.3, 0.2, (0, 1)), cv.embed(cv.two_mode_squeezing(0.3, 0.2), [0, 1], 3)),
            (ops.cz(s, A, [0, 2]), cv.embed(cv.cz(A), [0, 2], 3)),
            (ops.cx(s, 0.7, (1, 2)), cv.embed(cv.cx(0.7), [1, 2], 3)),
            (ops.shear(s, -0.4, 0), cv.embed(cv.shear(-0.4), [0], 3)),
            (ops.fourier(s, 1), cv.embed(cv.fourier(), [1], 3)),
        ]
        for out, S in cases:
            assert out.isclose(GaussianState(S @ s.mean, S @ s.cov @ S.T), 1e-12)

    def test_composition_order(self, rng):
        s = GaussianState.coherent([0.3, -0.2j])
        S1, S2 = cv.beamsplitter(0.3, 0.2), cv.embed(cv.squeezing(0.5), [0], 2)
        a = ops.apply_symplectic(ops.apply_symplectic(s, S1), S2)
        assert a.isclose(ops.apply_symplectic(s, S2 @ S1), 1e-14)

    def test_displace_several_modes(self):
        s = ops.displace(GaussianState.vacuum(3), [0.1j, 0.2], modes=[2, 0])
        assert np.allclose(s.complex_moments()[0], [0.2, 0, 0.1j])

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: ops.apply_symplectic(s, np.diag([2.0, 2.0])),
            lambda s: ops.displace(s, [0.1, 0.2], modes=[0]),
            lambda s: ops.rotate(s, 0.1, mode=5),
        ],
    )
    def test_rejects(self, call):
        with pytest.raises(InputError):
            call(GaussianState.vacuum(2))
