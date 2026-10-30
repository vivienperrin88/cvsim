"""cvsim.gaussian.states: GaussianState against the Fock-space oracle and its own identities.

Oracle: ``cvsim.reference.fock_dense`` builds the same states as density matrices (gates as
exponentials of their generators, loss as Kraus maps) and every figure of merit is recomputed there
from its definition: purity ``Tr ρ²``, entropy from the eigenvalues, fidelity
``(Tr √(√ρ σ √ρ))²`` with Hermitian square roots, ``Tr ρσ``, ``Tr ρ n``, Wigner from displaced
parity, negativity from the trace norm of the partial transpose.

Tolerances. Truncation: one mode at N = 40, two modes at N = 22 with ``⟨n⟩ < 0.4`` per mode; the
neglected Fock tail is below 1e-12 (checked by the oracle's trace). Agreement is required to 1e-9
(entropy: logarithms of tiny eigenvalues) or 1e-10 (everything else). The square-root oracle of the
fidelity is only good to ~1e-8 (see ``TOL_FID_ORACLE``), so the fidelity is also checked against
two closed forms (thermal states; Scutaru's single-mode formula, also on products) to 1e-11.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.errors import InputError, PhysicalityError, ShapeError
from cvsim.gaussian import GaussianState
from cvsim.reference import fock_dense as fd

SETTINGS = settings(max_examples=40, deadline=None)
N1, N2 = 40, 22
TOL, TOL_LOG = 1e-10, 1e-9
#: The square-root oracle of the fidelity is limited by round-off eigenvalues ε ~ 1e-17 of the
#: density matrices, whose square roots (~3e-9) enter the trace norm: 1e-7 is the honest limit
#: there. The fidelity formula itself is checked to 1e-10 against closed forms (TestFidelity).
TOL_FID_ORACLE = 1e-7


# ------------------------------------------------------------------------------- oracle helpers
def herm_sqrt(rho: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh((rho + rho.conj().T) / 2)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.conj().T


def oracle_fidelity(r1: np.ndarray, r2: np.ndarray) -> float:
    s = herm_sqrt(r1)
    w = np.linalg.eigvalsh(s @ r2 @ s)
    return float(np.sum(np.sqrt(np.clip(w, 0, None))) ** 2)


def oracle_entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-16]
    return float(-np.sum(w * np.log(w)))


def lossy_single(r: float, phi: float, alpha: complex, eta: float) -> np.ndarray:
    psi = fd.displacement(alpha, 1, N1) @ fd.squeezing(r, phi, 1, N1) @ fd.vacuum(1, N1)
    return fd.apply_kraus(fd.dm(psi), fd.loss_kraus(eta, N1), 1, N1)


def lossy_tms(r: float, alphas: tuple[complex, complex], etas: tuple[float, float]) -> np.ndarray:
    psi = fd.two_mode_squeezing(r, 0.4, 2, N2) @ fd.vacuum(2, N2)
    psi = fd.displacement(alphas[1], 2, N2, mode=1) @ fd.displacement(alphas[0], 2, N2, mode=0) @ psi
    rho = fd.dm(psi)
    for mode, eta in enumerate(etas):
        rho = fd.apply_kraus(rho, fd.loss_kraus(eta, N2), 2, N2, mode=mode)
    return rho


def gaussian_of(rho: np.ndarray, M: int, N: int) -> GaussianState:
    d, V = fd.moments(rho, M, N)
    return GaussianState(d, V)


SINGLE = {
    "lossy squeezed coherent": lossy_single(0.4, 0.3, 0.5 + 0.2j, 0.7),
    "other lossy squeezed": lossy_single(0.2, 1.1, 0.1 - 0.3j, 0.8),
    "pure squeezed coherent": lossy_single(0.3, -0.6, -0.2 + 0.4j, 1.0),
}
TWO = {
    "lossy TMS": lossy_tms(0.3, (0.2 + 0.1j, -0.1j), (0.8, 0.6)),
    "other lossy TMS": lossy_tms(0.25, (-0.1, 0.15 + 0.05j), (0.9, 0.75)),
}


@pytest.fixture(scope="module", params=list(SINGLE))
def single(request):
    rho = SINGLE[request.param]
    return rho, gaussian_of(rho, 1, N1)


@pytest.fixture(scope="module", params=list(TWO))
def two(request):
    rho = TWO[request.param]
    return rho, gaussian_of(rho, 2, N2)


def test_oracle_truncation_is_negligible():
    for rho in [*SINGLE.values(), *TWO.values()]:
        assert abs(np.trace(rho).real - 1) < 1e-12


# ------------------------------------------------------------------------------- constructors
class TestConstructors:
    def test_vacuum(self):
        s = GaussianState.vacuum(3)
        assert s.num_modes == 3 and np.array_equal(s.cov, np.eye(6) / 2) and not s.mean.any()

    def test_coherent_matches_oracle(self):
        alpha = [0.3 - 0.2j, -0.5 + 0.1j]
        s = GaussianState.coherent(alpha)
        d, V = fd.moments(fd.dm(fd.coherent(alpha, N2)), 2, N2)
        assert np.allclose(s.mean, d, atol=TOL) and np.allclose(s.cov, V, atol=TOL)

    def test_thermal_matches_oracle(self):
        nbar = 0.3
        p = (nbar / (1 + nbar)) ** np.arange(N1) / (1 + nbar)
        d, V = fd.moments(np.diag(p / p.sum()).astype(complex), 1, N1)
        s = GaussianState.thermal(nbar)
        assert np.allclose(s.cov, V, atol=1e-9) and np.allclose(s.mean, d, atol=TOL)

    @pytest.mark.parametrize(("r", "phi"), [(0.4, 0.0), (0.3, 1.2), (0.5, -2.0)])
    def test_squeezed_matches_oracle(self, r, phi):
        d, V = fd.moments(fd.dm(fd.squeezing(r, phi, 1, N1) @ fd.vacuum(1, N1)), 1, N1)
        s = GaussianState.squeezed(r, phi)
        assert np.allclose(s.cov, V, atol=TOL) and np.allclose(s.mean, d, atol=TOL)

    def test_multimode_squeezed_is_a_product(self):
        s = GaussianState.squeezed([0.2, 0.5], [0.0, 1.0])
        t = GaussianState.product(GaussianState.squeezed(0.2), GaussianState.squeezed(0.5, 1.0))
        assert s.isclose(t, 1e-14)

    def test_two_mode_squeezed_matches_oracle(self):
        r, phi = 0.3, 0.7
        d, V = fd.moments(fd.dm(fd.two_mode_squeezing(r, phi, 2, N2) @ fd.vacuum(2, N2)), 2, N2)
        s = GaussianState.two_mode_squeezed(r, phi)
        assert np.allclose(s.cov, V, atol=TOL) and np.allclose(s.mean, d, atol=TOL)
        assert np.allclose(s.mean_photons(), np.sinh(r) ** 2)

    def test_from_symplectic_with_thermal_modes(self, rng):
        S = cv.interferometer(np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0])
        S = S @ cv.embed(cv.squeezing(0.4), [0], 2)
        s = GaussianState.from_symplectic(S, nu=[0.5, 1.5], mean=[0.1, 0.2, 0.3, 0.4])
        assert np.allclose(s.symplectic_eigenvalues(), [0.5, 1.5])
        assert np.allclose(s.mean, [0.1, 0.2, 0.3, 0.4])
        assert math.isclose(s.purity(), 1 / 3, rel_tol=1e-12)

    def test_product_ordering_is_xxpp(self):
        a, b = GaussianState.coherent(0.1 + 0.2j), GaussianState.coherent(-0.3 + 0.4j)
        s = a.tensor(b)
        assert np.allclose(s.mean, np.sqrt(2) * np.array([0.1, -0.3, 0.2, 0.4]))
        assert s.reduce([1]).isclose(b) and s.reduce([0]).isclose(a)

    def test_hbar_and_xpxp_round_trips(self, rng):
        s = GaussianState.from_symplectic(cv.two_mode_squeezing(0.3, 0.2), nu=[0.7, 1.0], mean=rng.normal(size=4))
        d2, V2 = s.to_hbar(2.0)
        assert np.allclose(np.diag(GaussianState.vacuum(2).to_hbar(2.0)[1]), 1.0)
        assert GaussianState.from_hbar(d2, V2, hbar=2.0).isclose(s, 1e-13)
        assert GaussianState.from_xpxp(*s.to_xpxp()).isclose(s, 1e-14)


# ------------------------------------------------------------------------------- figures of merit vs oracle
class TestAgainstOracle:
    def test_purity_and_entropy(self, single):
        rho, s = single
        assert math.isclose(s.purity(), np.trace(rho @ rho).real, abs_tol=TOL)
        assert math.isclose(s.entropy(), oracle_entropy(rho), abs_tol=TOL_LOG)
        assert math.isclose(s.entropy(base=2), oracle_entropy(rho) / math.log(2), abs_tol=TOL_LOG)

    def test_two_mode_purity_and_entropy(self, two):
        rho, s = two
        assert math.isclose(s.purity(), np.trace(rho @ rho).real, abs_tol=TOL)
        assert math.isclose(s.entropy(), oracle_entropy(rho), abs_tol=TOL_LOG)

    def test_complex_moments(self, two):
        rho, s = two
        a = fd.mode_operators(2, N2)
        alpha, Nm, A = s.complex_moments()
        for j in range(2):
            assert abs(alpha[j] - np.trace(rho @ a[j])) < TOL
            for k in range(2):
                assert abs(Nm[j, k] - np.trace(rho @ a[j].conj().T @ a[k])) < TOL
                assert abs(A[j, k] - np.trace(rho @ a[j] @ a[k])) < TOL

    def test_photon_moments(self, two):
        rho, s = two
        a = fd.mode_operators(2, N2)
        n = [ak.conj().T @ ak for ak in a]
        mean = np.array([np.trace(rho @ nk).real for nk in n])
        cov = np.array(
            [[np.trace(rho @ nj @ nk).real - mean[j] * mean[k] for k, nk in enumerate(n)] for j, nj in enumerate(n)]
        )
        assert np.allclose(s.mean_photons(), mean, atol=TOL)
        assert np.allclose(s.photon_covariance(), cov, atol=TOL)

    def test_overlap_and_fidelity_mixed(self):
        (r1, r2) = SINGLE["lossy squeezed coherent"], SINGLE["other lossy squeezed"]
        s1, s2 = gaussian_of(r1, 1, N1), gaussian_of(r2, 1, N1)
        assert math.isclose(s1.overlap(s2), np.trace(r1 @ r2).real, abs_tol=TOL)
        assert math.isclose(s1.fidelity(s2), oracle_fidelity(r1, r2), abs_tol=TOL_FID_ORACLE)
        assert math.isclose(s2.fidelity(s1), s1.fidelity(s2), abs_tol=1e-12)

    def test_fidelity_two_mode_mixed(self):
        r1, r2 = TWO["lossy TMS"], TWO["other lossy TMS"]
        s1, s2 = gaussian_of(r1, 2, N2), gaussian_of(r2, 2, N2)
        assert math.isclose(s1.fidelity(s2), oracle_fidelity(r1, r2), abs_tol=TOL_FID_ORACLE)

    def test_fidelity_pure_and_mixed(self):
        r1, r2 = SINGLE["pure squeezed coherent"], SINGLE["lossy squeezed coherent"]
        s1, s2 = gaussian_of(r1, 1, N1), gaussian_of(r2, 1, N1)
        assert s1.is_pure(1e-9) and not s2.is_pure()
        psi = fd.displacement(-0.2 + 0.4j, 1, N1) @ fd.squeezing(0.3, -0.6, 1, N1) @ fd.vacuum(1, N1)
        assert math.isclose(s1.fidelity(s2), (psi.conj() @ r2 @ psi).real, abs_tol=TOL)  # ⟨ψ|ρ|ψ⟩, exact
        assert math.isclose(s1.fidelity(s2), oracle_fidelity(r1, r2), abs_tol=TOL_FID_ORACLE)

    def test_wigner(self, single):
        rho, s = single
        for x, p in [(0.0, 0.0), (0.3, -0.2), (-1.1, 0.7)]:
            assert math.isclose(float(s.wigner([x, p])), fd.wigner(rho, x, p, N1), abs_tol=TOL)

    def test_vacuum_probability(self, two):
        rho, s = two
        assert math.isclose(s.vacuum_probability(), rho[0, 0].real, abs_tol=TOL)
        r4 = rho.reshape(N2, N2, N2, N2)
        p0_mode1 = np.einsum("aiaj->ij", r4)[0, 0].real  # trace out mode 0, vacuum of mode 1
        assert math.isclose(s.vacuum_probability([1]), p0_mode1, abs_tol=TOL)

    def test_log_negativity(self, two):
        rho, s = two
        pt = rho.reshape(N2, N2, N2, N2).transpose(2, 1, 0, 3).reshape(N2 * N2, N2 * N2)
        trace_norm = np.abs(np.linalg.eigvalsh((pt + pt.conj().T) / 2)).sum()
        assert math.isclose(s.log_negativity([0]), math.log2(trace_norm), abs_tol=TOL_LOG)
        assert math.isclose(s.log_negativity([1], base=math.e), math.log(trace_norm), abs_tol=TOL_LOG)


# ------------------------------------------------------------------------------- identities (Hypothesis)
@st.composite
def states(draw, M=None, mixed=None):
    """Random state: passive · squeezers · passive applied to thermal (or vacuum) modes, random mean."""
    M = draw(st.integers(1, 4)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))

    def passive():
        return cv.interferometer(np.linalg.qr(rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M)))[0])

    r = rng.uniform(0, 1.2, M)
    S = passive() @ np.diag(np.exp(np.concatenate([-r, r]))) @ passive()
    mixed = draw(st.booleans()) if mixed is None else mixed
    nu = 0.5 + (rng.uniform(0.05, 1.5, M) if mixed else np.zeros(M))
    return GaussianState.from_symplectic(S, nu=nu, mean=rng.normal(size=2 * M))


class TestIdentities:
    @SETTINGS
    @given(s=states())
    def test_purity_entropy_and_eigenvalues(self, s):
        nu = s.symplectic_eigenvalues()
        assert math.isclose(s.purity(), float(np.prod(1 / (2 * nu))), rel_tol=1e-9)
        assert (s.entropy() < 1e-9) == s.is_pure(1e-9)
        assert s.entropy() >= -1e-12

    @SETTINGS
    @given(s=states(), data=st.data())
    def test_fidelity_properties(self, s, data):
        t = data.draw(states(M=s.num_modes))
        f = s.fidelity(t)
        assert 0 <= f <= 1
        assert math.isclose(f, t.fidelity(s), rel_tol=1e-7, abs_tol=1e-12)
        assert math.isclose(s.fidelity(s), 1.0, abs_tol=1e-8)
        # invariance under a common symplectic map and displacement
        S = cv.embed(cv.squeezing(0.3, 0.4), [0], s.num_modes)
        shift = np.arange(2 * s.num_modes) * 0.1
        u = GaussianState(S @ s.mean + shift, S @ s.cov @ S.T)
        v = GaussianState(S @ t.mean + shift, S @ t.cov @ S.T)
        assert math.isclose(u.fidelity(v), f, rel_tol=1e-7, abs_tol=1e-12)

    @SETTINGS
    @given(s=states())
    def test_product_reduce_and_photons(self, s):
        v = GaussianState.thermal(0.4)
        p = s.tensor(v)
        assert p.reduce(list(range(s.num_modes))).isclose(s, 1e-14)
        assert p.reduce([s.num_modes]).isclose(v, 1e-14)
        C = p.photon_covariance()
        assert np.allclose(C[-1, :-1], 0) and math.isclose(C[-1, -1], 0.4**2 + 0.4)
        assert np.allclose(np.diag(s.complex_moments()[1]).real, s.mean_photons())

    @SETTINGS
    @given(s=states())
    def test_overlap_is_fidelity_for_a_pure_partner(self, s):
        pure = GaussianState.from_symplectic(cv.embed(cv.squeezing(0.2), [0], s.num_modes), mean=s.mean * 0.5)
        assert math.isclose(pure.fidelity(s), pure.overlap(s), rel_tol=1e-12)

    def test_tms_log_negativity_closed_form(self):
        r = 0.7
        s = GaussianState.two_mode_squeezed(r)
        assert math.isclose(s.log_negativity([0]), 2 * r / math.log(2), rel_tol=1e-12)
        assert s.log_negativity([0]) == pytest.approx(s.log_negativity([1]))
        assert GaussianState.squeezed([0.5, 0.2]).log_negativity([0]) < 1e-12  # product state

    def test_wigner_is_normalised(self):
        s = GaussianState.from_symplectic(cv.squeezing(0.4, 0.5), nu=0.8, mean=[0.3, -0.2])
        x = np.linspace(-12, 12, 1201)
        W = s.wigner_grid(x, x)
        assert math.isclose(W.sum() * (x[1] - x[0]) ** 2, 1.0, abs_tol=1e-10)
        assert W.shape == (1201, 1201)

    def test_multimode_wigner_factorises(self):
        a, b = GaussianState.squeezed(0.3, 0.2), GaussianState.coherent(0.4j)
        pt = np.array([0.1, -0.4, 0.2, 0.5])  # (x0, x1, p0, p1)
        assert math.isclose(
            float(a.tensor(b).wigner(pt)), float(a.wigner(pt[[0, 2]]) * b.wigner(pt[[1, 3]])), rel_tol=1e-12
        )


# ------------------------------------------------------------------------------- fidelity closed forms
def scutaru(s: GaussianState, t: GaussianState) -> float:
    """Single-mode fidelity (Scutaru 1998), vacuum V = I/2: ``F = e^{−½δᵀΣ⁻¹δ} / (√(Δ + Λ) − √Λ)``,
    ``Σ = V₁ + V₂``, ``Δ = det Σ``, ``Λ = 4 (det V₁ − ¼)(det V₂ − ¼)``."""
    Sig = s.cov + t.cov
    delta = t.mean - s.mean
    D = np.linalg.det(Sig)
    L = max(0.0, 4 * (np.linalg.det(s.cov) - 0.25) * (np.linalg.det(t.cov) - 0.25))  # 0 for a pure state
    return float(np.exp(-0.5 * delta @ np.linalg.solve(Sig, delta)) / (np.sqrt(D + L) - np.sqrt(L)))


class TestFidelity:
    @pytest.mark.parametrize(("n1", "n2"), [(0.1, 0.1), (0.3, 1.7), (2.0, 0.05)])
    def test_thermal_closed_form(self, n1, n2):
        exact = 1 / (math.sqrt((n1 + 1) * (n2 + 1)) - math.sqrt(n1 * n2)) ** 2
        assert math.isclose(GaussianState.thermal(n1).fidelity(GaussianState.thermal(n2)), exact, rel_tol=1e-12)

    @SETTINGS
    @given(s=states(M=1, mixed=True), t=states(M=1, mixed=True))
    def test_single_mode_matches_scutaru(self, s, t):
        assert math.isclose(s.fidelity(t), scutaru(s, t), rel_tol=1e-11, abs_tol=1e-14)

    @SETTINGS
    @given(a=states(M=1), b=states(M=1), c=states(M=1), d=states(M=1))
    def test_product_states_multiply(self, a, b, c, d):
        """A genuinely two-mode evaluation of the formula against the product of single-mode closed forms."""
        f = a.tensor(b).fidelity(c.tensor(d))
        # pure factors inside mixed products: the √ε sensitivity of the fidelity (docstring) -> 3e-7
        pure_factor = any(x.is_pure() for x in (a, b, c, d))
        assert math.isclose(f, scutaru(a, c) * scutaru(b, d), rel_tol=3e-7 if pure_factor else 1e-11, abs_tol=1e-14)

    def test_fidelity_never_exceeds_one(self):
        s = GaussianState.from_symplectic(cv.squeezing(1.0, 0.3), nu=1.3, mean=[1.0, -2.0])
        assert s.fidelity(s) <= 1.0 and math.isclose(s.fidelity(s), 1.0, abs_tol=1e-10)


# ------------------------------------------------------------------------------- validation
class TestValidation:
    def test_uncertainty_principle(self):
        with pytest.raises(PhysicalityError, match="uncertainty principle"):
            GaussianState(np.zeros(2), np.diag([0.2, 0.5]))
        GaussianState(np.zeros(2), np.diag([0.25, 1.0]))  # minimum uncertainty is fine

    def test_physicality_error_is_an_input_error(self):
        with pytest.raises(InputError):
            GaussianState(np.zeros(2), np.diag([0.2, 0.5]))

    @pytest.mark.parametrize(
        ("mean", "cov", "exc"),
        [
            (np.zeros(2), np.eye(3), ShapeError),
            (np.zeros(3), np.eye(2), ShapeError),
            (np.zeros(2), np.array([[1.0, 0.3], [0.0, 1.0]]), InputError),
            (np.zeros(2), np.eye(2) * (1 + 0j), InputError),
            (np.zeros(2), np.array([[np.nan, 0], [0, 1]]), InputError),
        ],
    )
    def test_rejects(self, mean, cov, exc):
        with pytest.raises(exc):
            GaussianState(mean, cov)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: GaussianState.vacuum(0),
            lambda: GaussianState.thermal(-0.1),
            lambda: GaussianState.coherent([]),
            lambda: GaussianState.from_symplectic(np.array([[1.0, 1.0], [0.0, 2.0]])),
            lambda: GaussianState.from_symplectic(np.eye(2), nu=0.4),
            lambda: GaussianState.vacuum(1).entropy(base=1),
            lambda: GaussianState.vacuum(2).log_negativity([0], base=0.5),
            lambda: GaussianState.vacuum(1).fidelity(GaussianState.vacuum(2)),
            lambda: GaussianState.vacuum(1).overlap("vacuum"),
            lambda: GaussianState.vacuum(2).reduce([0, 0]),
            lambda: GaussianState.vacuum(2).wigner([0.0, 0.0]),
            lambda: GaussianState.squeezed([0.1, 0.2, 0.3], [0.0, 1.0]),
        ],
    )
    def test_invalid_calls(self, call):
        with pytest.raises(InputError):
            call()

    def test_arrays_are_read_only(self):
        s = GaussianState.coherent(0.3)
        with pytest.raises(ValueError):
            s.mean[0] = 1.0
        with pytest.raises(ValueError):
            s.cov[0, 0] = 1.0

    def test_input_arrays_are_copied(self):
        d, V = np.zeros(2), np.eye(2) / 2
        s = GaussianState(d, V)
        V[0, 0] = 5.0
        assert s.cov[0, 0] == 0.5

    def test_repr(self):
        assert repr(GaussianState.vacuum(2)) == "GaussianState(modes=2, purity=1, photons=0)"
