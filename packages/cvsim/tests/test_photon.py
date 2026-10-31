"""Photon counting: matrix functions (cvsim.core.hafnian) and statistics (cvsim.gaussian.photon).

Matrix functions against brute-force sums over permutations and matchings; statistics against
closed forms (thermal, coherent, squeezed and two-mode squeezed states, Poissonian detector
models), against each other (pattern probabilities, Fock elements, torontonian, click
inclusion–exclusion), and against the dense Fock backend, whose truncation error is measured
(heralded block agreement 2.7e-8 at cutoff 24, 2.4e-12 at 32).
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.backends import FockBackend
from cvsim.core import conventions as cv
from cvsim.core import hafnian as hf
from cvsim.core import linalg as la
from cvsim.errors import CutoffWarning, InputError, ShapeError
from cvsim.gaussian import GaussianState, ops
from cvsim.gaussian import photon as ph
from cvsim.gaussian import symplectic as sy
from cvsim.ir import Program, op

SETTINGS = settings(max_examples=25, deadline=None)


# ------------------------------------------------------------------------------- brute force
def brute_permanent(A):
    n = A.shape[0]
    return sum(np.prod([A[i, s[i]] for i in range(n)]) for s in itertools.permutations(range(n)))


def matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for k in range(1, len(items)):
        rest = items[1:k] + items[k + 1 :]
        for m in matchings(rest):
            yield [(a, items[k]), *m]


def loop_matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for m in loop_matchings(items[1:]):
        yield [(a, a), *m]
    for k in range(1, len(items)):
        for m in loop_matchings(items[1:k] + items[k + 1 :]):
            yield [(a, items[k]), *m]


def brute_hafnian(A, loops=False):
    gen = loop_matchings if loops else matchings
    return sum(np.prod([A[i, j] for i, j in m]) for m in gen(list(range(A.shape[0]))))


def complex_symmetric(rng, n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return A + A.T


class TestMatrixFunctions:
    @SETTINGS
    @given(seed=st.integers(0, 2**32 - 1), n=st.integers(0, 6))
    def test_permanent(self, seed, n):
        B = np.random.default_rng(seed).normal(size=(n, n)) + 1j * np.random.default_rng(seed + 1).normal(size=(n, n))
        assert abs(hf.permanent(B) - (brute_permanent(B) if n else 1.0)) < 1e-10 * max(
            1.0, abs(brute_permanent(B)) if n else 1.0
        )

    @SETTINGS
    @given(seed=st.integers(0, 2**32 - 1), n=st.integers(0, 8))
    def test_hafnians(self, seed, n):
        A = complex_symmetric(np.random.default_rng(seed), n)
        exact = brute_hafnian(A) if n % 2 == 0 else 0.0
        scale = max(1.0, abs(exact))
        assert abs(hf.hafnian(A) - exact) < 1e-10 * scale
        assert abs(hf.hafnian(A, method="recursive") - exact) < 1e-12 * scale
        lexact = brute_hafnian(A, loops=True)
        assert abs(hf.loop_hafnian(A) - lexact) < 1e-12 * max(1.0, abs(lexact))

    def test_bipartite_hafnian_is_a_permanent(self, rng):
        B = rng.normal(size=(4, 4))
        A = np.block([[np.zeros((4, 4)), B], [B.T, np.zeros((4, 4))]])
        assert math.isclose(hf.hafnian(A), hf.permanent(B), rel_tol=1e-12)
        assert isinstance(hf.hafnian(A), float) and isinstance(hf.hafnian(A + 0j), float)

    def test_loop_hafnian_diagonal_argument(self, rng):
        A = complex_symmetric(rng, 4)
        d = rng.normal(size=4)
        B = A.copy()
        np.fill_diagonal(B, d)
        assert abs(hf.loop_hafnian(A, diag=d) - hf.loop_hafnian(B)) < 1e-12

    def test_torontonian_is_the_zero_mean_click_probability(self, rng):
        s = GaussianState.from_symplectic(sy.random_symplectic(rng, 3, 0.6), nu=[0.5, 0.7, 0.9])
        A, _, T = ph.husimi_parameters(s)
        M = 3
        W = cv.complex_basis(M)
        Q = W @ s.cov @ W.conj().T + np.eye(2 * M) / 2
        O = np.eye(2 * M) - la.inv(Q)
        for S in [(0,), (0, 2), (0, 1, 2)]:
            idx = [*S, *(M + k for k in S)]
            rest = [k for k in range(M) if k not in S]
            # clicks exactly on S: vacuum on the rest, then the torontonian of the conditional block
            pattern = [1 if k in S else 0 for k in range(M)]
            if not rest:
                expected = hf.torontonian(O[np.ix_(idx, idx)]) / math.sqrt(abs(np.prod(la.eigvals(Q))))
                assert math.isclose(ph.click_probability(s, pattern), float(np.real(expected)), rel_tol=1e-10)
        del A, T

    @pytest.mark.parametrize(
        ("call", "exc"),
        [
            (lambda: hf.permanent(np.ones((2, 3))), ShapeError),
            (lambda: hf.hafnian(np.array([[0, 1], [2, 0]])), InputError),
            (lambda: hf.hafnian(np.eye(2), method="glynn"), InputError),
            (lambda: hf.loop_hafnian(np.eye(2), diag=[1, 2, 3]), ShapeError),
            (lambda: hf.torontonian(np.eye(3)), ShapeError),
            (lambda: hf.permanent(np.array([[np.nan]])), InputError),
        ],
    )
    def test_validation(self, call, exc):
        with pytest.raises(exc):
            call()

    def test_trivial_sizes(self):
        assert hf.permanent(np.zeros((0, 0))) == 1.0 and hf.hafnian(np.zeros((0, 0))) == 1.0
        assert hf.loop_hafnian(np.zeros((0, 0))) == 1.0 and hf.hafnian(np.ones((3, 3))) == 0.0


# ------------------------------------------------------------------------------- closed forms
class TestClosedForms:
    def test_thermal_coherent_squeezed(self):
        nb, alpha, r = 0.4, 0.7 - 0.2j, 0.6
        th, co, sq = GaussianState.thermal(nb), GaussianState.coherent(alpha), GaussianState.squeezed(r, 0.3)
        for n in range(6):
            assert math.isclose(ph.pattern_probability(th, [n]), nb**n / (1 + nb) ** (n + 1), rel_tol=1e-12)
            mu = abs(alpha) ** 2
            assert math.isclose(
                ph.pattern_probability(co, [n]), math.exp(-mu) * mu**n / math.factorial(n), rel_tol=1e-12
            )
            if n % 2:
                assert ph.pattern_probability(sq, [n]) < 1e-15
            else:
                k = n // 2
                p = math.tanh(r) ** (2 * k) * math.factorial(2 * k) / (4**k * math.factorial(k) ** 2 * math.cosh(r))
                assert math.isclose(ph.pattern_probability(sq, [n]), p, rel_tol=1e-12)

    def test_two_mode_squeezed_heralds_fock_states(self):
        r = 0.5
        s = GaussianState.two_mode_squeezed(r, 0.4)
        for n in range(4):
            h = ph.herald(s, [0], [n], cutoff=8)
            assert math.isclose(h.probability, math.tanh(r) ** (2 * n) / math.cosh(r) ** 2, rel_tol=1e-12)
            assert math.isclose(h.dm[n, n].real, 1.0, abs_tol=1e-12) and h.modes == (1,)

    def test_detector_models_on_a_coherent_state(self):
        """Efficiency η and Poissonian dark counts ν keep a coherent state Poissonian: mean η|α|² + ν."""
        alpha, eta, nu, pd = 0.9, 0.7, 0.05, 0.02
        s = GaussianState.coherent([alpha, 0.0])
        dist = ph.pnr_distribution(s, [0], 12, efficiency=eta, dark_counts=nu)
        mu = eta * alpha**2 + nu
        assert np.allclose(dist, [math.exp(-mu) * mu**k / math.factorial(k) for k in range(12)], atol=1e-12)
        clicks = ph.click_distribution(s, [0], efficiency=eta, dark_prob=pd)
        assert math.isclose(clicks[0], (1 - pd) * math.exp(-eta * alpha**2), rel_tol=1e-12)
        assert math.isclose(clicks.sum(), 1.0, rel_tol=1e-12)


# ------------------------------------------------------------------------------- consistency
@st.composite
def states(draw, M=2):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 0.4), nu=0.5 + rng.uniform(0, 0.3, M), mean=rng.normal(0, 0.4, 2 * M)
    )


class TestConsistency:
    @SETTINGS
    @given(s=states())
    def test_density_matrix_patterns_and_clicks(self, s):
        N = 7
        dm = ph.fock_density_matrix(s, N)
        assert np.allclose(dm, dm.conj().T, atol=1e-14)
        diag = np.real(np.diag(dm)).reshape(N, N)
        for pattern in [(0, 0), (1, 0), (2, 3), (4, 1)]:
            assert math.isclose(ph.pattern_probability(s, pattern), diag[pattern], rel_tol=1e-10, abs_tol=1e-15)
        assert math.isclose(diag[0, 0], s.vacuum_probability(), rel_tol=1e-12)
        # clicks on mode 0 only: Σ_{n0 >= 1} p(n0, 0), truncated at N
        assert ph.click_probability(s, [1, 0]) >= diag[1:, 0].sum() - 1e-14
        assert np.allclose(ph.click_distribution(s, [0, 1]).sum(), 1.0, atol=1e-12)

    def test_heralding_against_the_fock_backend(self):
        """PNR outcome 1 on one arm of a squeezed beam split: exact elements vs the dense backend."""
        N = 32
        prog = (
            Program(2)
            >> op("Squeezed", 0, r=0.5)
            >> op("BS", (0, 1), theta=0.3)
            >> op("PNR", 1, efficiency=0.8, dark_counts=0.05, outcome=1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CutoffWarning)
            res = FockBackend(N).run(prog)
        s = ops.beamsplitter(GaussianState.squeezed(0.5).tensor(GaussianState.vacuum(1)), 0.3, 0.0, (0, 1))
        h = ph.herald(s, [1], [1], cutoff=N, efficiency=0.8, dark_counts=0.05)
        assert math.isclose(h.probability, res.outcomes[0].density, rel_tol=1e-9)
        assert np.allclose(h.dm[:12, :12], res.state.dm[:12, :12], atol=1e-10)

    def test_threshold_against_the_fock_backend(self):
        prog = (
            Program(2)
            >> op("Squeezed", 0, r=0.5)
            >> op("BS", (0, 1), theta=0.3)
            >> op("Threshold", 1, efficiency=0.8, dark_prob=0.01, outcome=1)
        )
        res = FockBackend(28).run(prog)
        s = ops.beamsplitter(GaussianState.squeezed(0.5).tensor(GaussianState.vacuum(1)), 0.3, 0.0, (0, 1))
        assert math.isclose(res.outcomes[0].density, ph.click_distribution(s, [1], 0.8, 0.01)[1], rel_tol=1e-9)

    def test_fock_backend_pnr_sampling(self, rng):
        prog = Program(1) >> op("Coherent", 0, alpha=1.0) >> op("PNR", 0, efficiency=0.5)
        draws = [FockBackend(14).run(prog, rng=rng).outcomes[0].value for _ in range(4000)]
        assert abs(np.mean(draws) - 0.5) < 0.04  # Poisson(0.5)
        t = Program(1) >> op("Coherent", 0, alpha=1.0) >> op("Threshold", 0)
        clicks = [FockBackend(14).run(t, rng=rng).outcomes[0].value for _ in range(2000)]
        assert abs(np.mean(clicks) - (1 - math.exp(-1))) < 0.04


class TestValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s: ph.pattern_probability(s, [1]),
            lambda s: ph.pattern_probability(s, [1, -1]),
            lambda s: ph.click_probability(s, [2, 0]),
            lambda s: ph.click_probability(s, [1, 0], dark_prob=1.0),
            lambda s: ph.fock_density_matrix(s, 0),
            lambda s: ph.pnr_distribution(s, [0], 4, efficiency=0.0),
            lambda s: ph.pnr_distribution(s, [0], 4, dark_counts=-1.0),
            lambda s: ph.herald(s, [0], [5], cutoff=5),
            lambda s: ph.herald(s, [0, 1], [1, 1], cutoff=4),
            lambda s: ph.herald(s, [0], [1, 2], cutoff=4),
            lambda s: ph.herald(s, [0], [1], cutoff=4, dark_counts=-0.1),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises(InputError):
            call(GaussianState.vacuum(2))

    def test_fock_backend_outcome_range(self):
        with pytest.raises(InputError, match="out of range"):
            FockBackend(4).run(Program(1) >> op("Threshold", 0, outcome=2))
