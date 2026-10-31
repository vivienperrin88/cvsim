"""cvsim.gaussian.measure.

* Against the Fock oracle: homodyne densities and conditional states (the unmeasured mode of a
  lossy two-mode squeezed state projected with ``⟨q|R(−φ)``), heterodyne densities (the Husimi
  function ``⟨α|ρ|α⟩/π``).
* Detector model: homodyne with efficiency ``η`` and electronic noise equals loss and added noise
  before an ideal homodyne with the outcome rescaled by ``√η``; heterodyne equals the explicit
  eight-port detector (50:50 beam splitter with a vacuum port, two lossy noisy homodynes).
* Local oscillator in any mode, sampling statistics, validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.errors import InputError, ShapeError
from cvsim.gaussian import GaussianChannel as C
from cvsim.gaussian import GaussianState, measure, ops
from cvsim.gaussian import symplectic as sy
from cvsim.reference import fock_dense as fd

SETTINGS = settings(max_examples=30, deadline=None)
N2 = 24


@pytest.fixture(scope="module")
def lossy_tms():
    rho = fd.dm(fd.displacement(0.2 - 0.1j, 2, N2, mode=1) @ fd.two_mode_squeezing(0.35, 0.3, 2, N2) @ fd.vacuum(2, N2))
    rho = fd.apply_kraus(rho, fd.loss_kraus(0.8, N2), 2, N2, mode=1)
    return rho, GaussianState(*fd.moments(rho, 2, N2))


def oracle_condition(rho: np.ndarray, phi: float, q: float) -> tuple[float, np.ndarray]:
    """Density of x_φ = q on mode 0 and the conditional density matrix of mode 1."""
    n = np.arange(N2)
    r4 = rho.reshape(N2, N2, N2, N2) * np.exp(-1j * phi * (n[:, None, None, None] - n[None, None, :, None]))
    h = fd.hermite_functions(N2, q)[:, 0]
    cond = np.einsum("a,aibj,b->ij", h, r4, h)
    p = float(np.trace(cond).real)
    return p, cond / p


@st.composite
def states(draw, M=None):
    M = draw(st.integers(1, 4)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 1.0), nu=0.5 + rng.uniform(0, 0.8, M), mean=rng.normal(size=2 * M)
    )


# ------------------------------------------------------------------------------- oracle
class TestAgainstOracle:
    @pytest.mark.parametrize(("phi", "q"), [(0.0, 0.3), (0.7, -0.9), (math.pi / 2, 1.4)])
    def test_homodyne_density_and_conditional_state(self, lossy_tms, phi, q):
        rho, s = lossy_tms
        p, cond = oracle_condition(rho, phi, q)
        res = measure.homodyne(s, 0, phi, outcome=q)
        assert math.isclose(res.density, p, abs_tol=1e-10)
        assert math.isclose(float(measure.homodyne_density(s, q, 0, phi)), p, abs_tol=1e-10)
        assert res.state is not None and res.state.isclose(GaussianState(*fd.moments(cond, 1, N2)), 1e-9)

    def test_heterodyne_density_is_the_husimi_function(self, lossy_tms):
        rho, s = lossy_tms
        red = np.einsum("aiaj->ij", rho.reshape(N2, N2, N2, N2))  # mode 1
        for alpha in [0.0, 0.3 - 0.2j, -0.5 + 0.4j]:
            ket = fd.coherent([alpha], N2)
            husimi = float((ket.conj() @ red @ ket).real) / math.pi
            assert math.isclose(measure.heterodyne(s.reduce([1]), 0, outcome=alpha).density, husimi, abs_tol=1e-10)


# ------------------------------------------------------------------------------- detector model
class TestDetectorModel:
    @SETTINGS
    @given(s=states(M=3), eta=st.floats(0.1, 1.0), vel=st.floats(0.0, 0.3), phi=st.floats(-3, 3), q=st.floats(-2, 2))
    def test_homodyne_efficiency_is_loss_plus_noise(self, s, eta, vel, phi, q):
        a = measure.homodyne(s, 1, phi, efficiency=eta, electronic=vel, outcome=q)
        t = C.additive_noise(vel / 2).apply(C.loss(eta).apply(s, [1]), [1])
        b = measure.homodyne(t, 1, phi, outcome=q * math.sqrt(eta))
        assert a.state is not None and b.state is not None and a.state.isclose(b.state, 1e-10)
        assert math.isclose(a.density, b.density * math.sqrt(eta), rel_tol=1e-10)

    @SETTINGS
    @given(s=states(M=2), eta=st.floats(0.2, 1.0), vel=st.floats(0.0, 0.3), re=st.floats(-1, 1), im=st.floats(-1, 1))
    def test_heterodyne_is_the_eight_port_detector(self, s, eta, vel, re, im):
        alpha = complex(re, im)
        h = measure.heterodyne(s, 0, efficiency=eta, electronic=vel, outcome=alpha)
        x = s.tensor(GaussianState.vacuum(1))  # vacuum port = mode 2
        x = ops.beamsplitter(x, math.pi / 4, 0.0, (0, 2))
        for m in (0, 2):
            x = C.additive_noise(vel / 2).apply(C.loss(eta).apply(x, [m]), [m])
        y = math.sqrt(2) * np.array([alpha.real, alpha.imag])
        raw = math.sqrt(eta / 2) * y  # each arm sees x/√2 and p/√2, attenuated by √η
        e = measure.homodyne(x, [0, 2], [0.0, math.pi / 2], outcome=raw)
        assert h.state is not None and e.state is not None and h.state.isclose(e.state, 1e-10)
        # densities: d²α = d²y/2 and d²raw = (η/2) d²y
        assert math.isclose(h.density, e.density * eta, rel_tol=1e-10)

    def test_noise_values(self):
        assert measure.homodyne_noise() == 0.0
        assert measure.homodyne_noise(heterodyne=True) == 0.5
        assert math.isclose(measure.homodyne_noise(0.8, 0.02), 0.22 / 1.6)
        assert math.isclose(measure.homodyne_noise(0.8, 0.02, heterodyne=True), 0.5 + 0.22 / 0.8)


# ------------------------------------------------------------------------------- general behaviour
class TestBehaviour:
    @SETTINGS
    @given(s=states(M=3))
    def test_general_dyne_reduces_to_homodyne_and_heterodyne(self, s):
        phi, q = 0.4, 0.25
        a = measure.homodyne(s, 2, phi, outcome=q)
        b = measure.general_dyne(s, [2], [[math.cos(phi), math.sin(phi)]], outcome=[q])
        assert a.state.isclose(b.state, 1e-12) and math.isclose(a.density, b.density, rel_tol=1e-12)
        h = measure.heterodyne(s, 1, outcome=0.1 + 0.2j)
        g = measure.general_dyne(s, [1], np.eye(2), 0.5 * np.eye(2), outcome=math.sqrt(2) * np.array([0.1, 0.2]))
        assert h.state.isclose(g.state, 1e-12) and math.isclose(h.density, 2 * g.density, rel_tol=1e-12)

    def test_conditional_covariance_does_not_depend_on_the_outcome(self):
        s = GaussianState.two_mode_squeezed(0.8)
        a = measure.homodyne(s, 0, 0.0, outcome=-1.0).state
        b = measure.homodyne(s, 0, 0.0, outcome=2.0).state
        assert np.allclose(a.cov, b.cov) and not np.allclose(a.mean, b.mean)
        # conditioning a pure state on a projective homodyne outcome leaves a pure state
        assert a.is_pure(1e-10)

    def test_last_mode_leaves_no_state(self):
        res = measure.homodyne(GaussianState.coherent(0.5), 0, 0.0, outcome=0.7)
        assert res.state is None and math.isclose(res.mean[0], math.sqrt(2) * 0.5)

    def test_sampling_statistics(self, rng):
        s = GaussianState.from_symplectic(
            sy.random_symplectic(rng, 3), nu=[0.6, 0.8, 1.0], mean=[0.3, -0.2, 0.1, 0.0, 0.5, -0.4]
        )
        samples = measure.sample_homodyne(s, [0, 2], [0.2, 1.1], 40000, rng, efficiency=0.9, electronic=0.05)
        res = measure.homodyne(s, [0, 2], [0.2, 1.1], efficiency=0.9, electronic=0.05, outcome=[0.0, 0.0])
        assert np.allclose(samples.mean(axis=0), res.mean, atol=0.03)
        assert np.allclose(np.cov(samples.T), res.cov, rtol=0.05, atol=0.02)
        draws = np.array([measure.homodyne(s, 1, 0.5, rng=rng).outcome for _ in range(4000)])
        assert abs(draws.mean() - measure.homodyne(s, 1, 0.5, outcome=0.0).mean[0]) < 0.06

    def test_heterodyne_sampling_returns_alpha(self, rng):
        res = measure.heterodyne(GaussianState.coherent(0.4 + 0.3j), 0, rng=rng)
        assert isinstance(res.outcome, complex)


class TestLocalOscillator:
    def test_lo_on_an_input_mode_is_plain_homodyne(self, rng):
        s = GaussianState.from_symplectic(sy.random_symplectic(rng, 3), nu=[0.5, 0.9, 1.3], mean=rng.normal(size=6))
        lo = np.zeros(3, dtype=complex)
        lo[0] = 1.0
        a = measure.homodyne_lo(s, lo, 0.3, outcome=0.2)
        b = measure.homodyne(s, 0, 0.3, outcome=0.2)
        W = a.basis[1:, :]  # the remaining mode functions, on modes 1, 2
        assert np.allclose(a.basis[0], 0) and math.isclose(a.density, b.density, rel_tol=1e-12)
        assert a.state.isclose(sy.change_basis(b.state, W), 1e-12)

    def test_lo_in_a_superposition_mode(self):
        alpha = 0.3 + 0.1j
        s = GaussianState.coherent([alpha, alpha])
        res = measure.homodyne_lo(s, np.array([1, 1]) / math.sqrt(2), 0.0, outcome=0.0)
        assert math.isclose(res.mean[0], 2 * alpha.real, abs_tol=1e-14)  # b = √2 α, x_b = √2 Re b
        assert np.allclose(res.state.complex_moments()[0], 0, atol=1e-14)  # the orthogonal mode is empty

    def test_lo_validation(self):
        s = GaussianState.vacuum(2)
        with pytest.raises(InputError, match="unit norm"):
            measure.homodyne_lo(s, [1.0, 1.0], outcome=0.0)
        with pytest.raises(ShapeError):
            measure.homodyne_lo(s, [1.0], outcome=0.0)


class TestValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s, rng: measure.homodyne(s, 0, 0.0),
            lambda s, rng: measure.homodyne(s, 0, 0.0, outcome=0.1, rng=rng),
            lambda s, rng: measure.homodyne(s, 0, 0.0, efficiency=0.0, outcome=0.1),
            lambda s, rng: measure.homodyne(s, 0, 0.0, electronic=-0.1, outcome=0.1),
            lambda s, rng: measure.general_dyne(s, [0], [[0.0, 0.0]], outcome=[0.0]),
            lambda s, rng: measure.general_dyne(s, [0], np.eye(3), outcome=[0.0, 0.0, 0.0]),
            lambda s, rng: measure.general_dyne(s, [0], np.eye(2), np.eye(3), outcome=[0.0, 0.0]),
            lambda s, rng: measure.general_dyne(s, [0], np.eye(2), outcome=[0.0]),
            lambda s, rng: measure.sample_homodyne(s, [0], 0.0, 0, rng),
            lambda s, rng: measure.homodyne(s, 5, 0.0, outcome=0.0),
        ],
    )
    def test_invalid(self, call, rng):
        with pytest.raises(InputError):
            call(GaussianState.vacuum(2), rng)
