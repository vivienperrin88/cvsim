"""cvsim.gaussian.channels and cvsim.gaussian.ledger.

Channels against the Fock oracle (loss by its Kraus map, the amplifier as a two-mode squeezer with
a vacuum idler traced out, phase diffusion as exact dephasing ``ρ_nm e^{−σ²(n−m)²/2}``, whose
first and second moments the moment-matched state must reproduce exactly), against closed-form
identities (ancilla constructions, compositions), and the complete-positivity check against
known violations. The ledger by its defining identity ``signal + Σ C_k = V_out`` and by
agreement with applying the same chain directly, on Hypothesis-generated chains.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.errors import ApproximationWarning, InputError, PhysicalityError, ShapeError
from cvsim.gaussian import GaussianChannel as C
from cvsim.gaussian import GaussianState, NoiseLedger, channels, ops
from cvsim.gaussian import symplectic as sy
from cvsim.reference import fock_dense as fd

SETTINGS = settings(max_examples=40, deadline=None)
N1 = 40


def single_mode_rho():
    psi = fd.displacement(0.4 + 0.2j, 1, N1) @ fd.squeezing(0.4, 0.6, 1, N1) @ fd.vacuum(1, N1)
    return fd.dm(psi)


def state_of(rho, M, N):
    return GaussianState(*fd.moments(rho, M, N))


# ------------------------------------------------------------------------------- channels vs oracle
class TestAgainstOracle:
    @pytest.mark.parametrize("eta", [0.0, 0.3, 0.92, 1.0])
    def test_loss_matches_kraus(self, eta):
        rho = single_mode_rho()
        out = fd.apply_kraus(rho, fd.loss_kraus(eta, N1), 1, N1)
        assert C.loss(eta).apply(state_of(rho, 1, N1)).isclose(state_of(out, 1, N1), 1e-10)

    def test_amplifier_matches_traced_two_mode_squeezer(self):
        M, N, G = 2, 30, 1.3
        sig = fd.displacement(0.2, 1, N) @ fd.squeezing(0.2, 0.3, 1, N) @ fd.vacuum(1, N)
        psi = fd.two_mode_squeezing(math.acosh(math.sqrt(G)), 0.0, M, N) @ np.kron(sig, fd.vacuum(1, N))
        red = np.einsum("iaja->ij", fd.dm(psi).reshape(N, N, N, N))
        assert C.amplifier(G).apply(state_of(fd.dm(sig), 1, N)).isclose(state_of(red, 1, N), 1e-10)

    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1.5])
    def test_phase_diffusion_has_the_exact_moments(self, sigma):
        rho = single_mode_rho()
        n = np.arange(N1)
        dephased = rho * np.exp(-(sigma**2) * (n[:, None] - n[None, :]) ** 2 / 2)
        with pytest.warns(ApproximationWarning):
            out = channels.phase_diffusion(state_of(rho, 1, N1), sigma)
        assert out.isclose(state_of(dephased, 1, N1), 1e-12)

    def test_phase_diffusion_cross_correlations(self):
        """Two-mode squeezed vacuum, phase noise on mode 1: E[e^{iφ}] = e^{−σ²/2} on the correlations."""
        M, N, sigma = 2, 22, 0.4
        rho = fd.dm(fd.two_mode_squeezing(0.3, 0.2, M, N) @ fd.vacuum(M, N))
        n = np.arange(N)
        phase = np.exp(-(sigma**2) * (n[:, None] - n[None, :]) ** 2 / 2)  # on mode 1
        r4 = rho.reshape(N, N, N, N) * phase[None, :, None, :]
        with pytest.warns(ApproximationWarning):
            out = channels.phase_diffusion(state_of(rho, M, N), sigma, mode=1)
        assert out.isclose(state_of(r4.reshape(N * N, N * N), M, N), 1e-10)

    def test_no_diffusion_is_the_identity_without_warning(self):
        s = GaussianState.squeezed(0.4, 0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert channels.phase_diffusion(s, 0.0).isclose(s, 1e-15)


# ------------------------------------------------------------------------------- identities
class TestIdentities:
    def test_ancilla_constructions(self):
        th, r, nbar = 0.5, 0.4, 0.7
        bs = C.from_ancilla(sy.beamsplitter(th), GaussianState.thermal(nbar))
        loss = C.loss(math.cos(th) ** 2, nbar)
        assert np.allclose(np.abs(bs.X), loss.X, atol=1e-14) and np.allclose(bs.Y, loss.Y, atol=1e-14)
        amp = C.from_ancilla(sy.two_mode_squeezing(r), GaussianState.vacuum(1))
        assert np.allclose(amp.Y, C.amplifier(math.cosh(r) ** 2).Y, atol=1e-14)
        shifted = C.from_ancilla(sy.beamsplitter(th), GaussianState.coherent(0.5))
        assert np.allclose(shifted.c, sy.beamsplitter(th)[np.ix_([0, 2], [1, 3])] @ GaussianState.coherent(0.5).mean)

    def test_compositions(self):
        assert np.allclose(C.loss(0.8).then(C.loss(0.5)).Y, C.loss(0.4).Y)
        G = 1.7  # loss 1/G then gain G: pure additive noise G − 1
        both = C.loss(1 / G).then(C.amplifier(G))
        assert np.allclose(both.X, np.eye(2)) and np.allclose(both.Y, (G - 1) * np.eye(2))
        a, b, c = C.loss(0.7, 0.2), C.from_symplectic(sy.squeezing(0.3, 0.4)), C.additive_noise(0.05)
        left, right = a.then(b).then(c), a.then(b.then(c))
        assert np.allclose(left.X, right.X) and np.allclose(left.Y, right.Y)

    def test_transfer_matrix(self, rng):
        etas = np.array([0.3, 0.9])
        T = np.diag(np.sqrt(etas))
        ch = C.from_transfer(T)
        assert np.allclose(ch.Y, np.diag(np.concatenate([1 - etas, 1 - etas]) / 2))
        U = sy.random_interferometer(rng, 3)
        assert C.from_transfer(U).is_unitary(1e-12)
        with pytest.raises(PhysicalityError):
            C.from_transfer(2 * U)

    @SETTINGS
    @given(seed=st.integers(0, 2**32 - 1))
    def test_apply_embed_and_tensor(self, seed):
        rng = np.random.default_rng(seed)
        s = GaussianState.from_symplectic(sy.random_symplectic(rng, 3), nu=[0.5, 0.8, 1.1], mean=rng.normal(size=6))
        eta, nbar = rng.uniform(0, 1), rng.uniform(0, 1)
        a, b = C.loss(eta, nbar), C.amplifier(1 + rng.uniform(0, 1))
        once = a.tensor(b).apply(s, [2, 0])
        twice = b.apply(a.apply(s, [2]), [0])
        assert once.isclose(twice, 1e-12)
        out = a.apply(s, [1])
        assert out.reduce([0, 2]).isclose(s.reduce([0, 2]), 1e-14)
        expected = eta * s.mean_photons()[1] + (1 - eta) * nbar  # beam splitter with a thermal port
        assert math.isclose(out.mean_photons()[1], expected, rel_tol=1e-10, abs_tol=1e-12)

    def test_unitary_channel_matches_ops(self, rng):
        s = GaussianState.from_symplectic(sy.random_symplectic(rng, 2), nu=0.7)
        S = sy.random_symplectic(rng, 2)
        assert C.from_symplectic(S).apply(s).isclose(ops.apply_symplectic(s, S), 1e-12)


# ------------------------------------------------------------------------------- validation
class TestValidation:
    @pytest.mark.parametrize(
        ("X", "Y"),
        [
            (2 * np.eye(2), np.zeros((2, 2))),  # gain without noise
            (np.diag([1.0, 0.0]), np.zeros((2, 2))),  # perfect projection
            (np.sqrt(1.5) * np.eye(2), 0.2 * np.eye(2)),  # amplifier with too little noise (needs 0.25)
        ],
    )
    def test_not_completely_positive(self, X, Y):
        with pytest.raises(PhysicalityError, match="completely positive"):
            C(X, Y)

    def test_minimal_amplifier_noise_is_accepted(self):
        C(np.sqrt(1.5) * np.eye(2), 0.25 * np.eye(2))

    @pytest.mark.parametrize(
        "call",
        [
            lambda: C(np.eye(2), np.eye(4)),
            lambda: C(np.eye(3), np.eye(3)),
            lambda: C(np.eye(2), np.zeros((2, 2)), c=np.zeros(3)),
            lambda: C(np.eye(2), np.array([[1.0, 0.5], [0.0, 1.0]])),
            lambda: C.loss(1.2),
            lambda: C.loss(0.5, -1.0),
            lambda: C.amplifier(0.5),
            lambda: C.additive_noise(-0.1),
            lambda: C.additive_noise(-np.eye(2)),
            lambda: C.from_symplectic(2 * np.eye(2)),
            lambda: C.from_ancilla(sy.beamsplitter(0.3), GaussianState.vacuum(2)),
            lambda: C.loss(0.5).then(C.identity(2)),
            lambda: C.loss(0.5).embed([0, 1], 3),
            lambda: channels.phase_diffusion(GaussianState.vacuum(1), -0.1),
        ],
    )
    def test_invalid(self, call):
        with pytest.raises((InputError, ShapeError)):
            call()

    def test_read_only(self):
        ch = C.loss(0.5)
        with pytest.raises(ValueError):
            ch.Y[0, 0] = 1.0
        assert repr(ch) == "GaussianChannel(modes=1, unitary=False)"


# ------------------------------------------------------------------------------- ledger
@st.composite
def chains(draw):
    """A random chain on 2–3 modes: gates, losses, amplifiers, additive noise, phase diffusion."""
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    M = int(rng.integers(2, 4))
    s = GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 1.0), nu=0.5 + rng.uniform(0, 0.5, M), mean=rng.normal(size=2 * M)
    )
    led = NoiseLedger.start(s)
    direct = s
    for k in range(int(rng.integers(1, 7))):
        choice = rng.integers(0, 5)
        mode = int(rng.integers(0, M))
        if choice == 0:
            S = sy.random_symplectic(rng, M, 0.5)
            led, direct = led.gate(S), ops.apply_symplectic(direct, S)
        elif choice == 1:
            ch = C.loss(rng.uniform(0.2, 1.0), rng.uniform(0, 0.5))
            led, direct = led.channel(ch, mode, f"loss{k}", "loss"), ch.apply(direct, [mode])
        elif choice == 2:
            ch = C.amplifier(1 + rng.uniform(0, 0.5))
            led, direct = led.channel(ch, mode, f"amp{k}", "gain"), ch.apply(direct, [mode])
        elif choice == 3:
            ch = C.additive_noise(rng.uniform(0, 0.1))
            led, direct = led.channel(ch, mode, f"elec{k}", "electronic"), ch.apply(direct, [mode])
        else:
            sigma = rng.uniform(0, 0.3)
            led = led.phase_diffusion(sigma, mode, f"phase{k}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ApproximationWarning)
                direct = channels.phase_diffusion(direct, sigma, mode)
    return led, direct


class TestLedger:
    @SETTINGS
    @given(chain=chains())
    def test_exact_decomposition_and_agreement_with_direct_application(self, chain):
        led, direct = chain
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ApproximationWarning)
            out = led.state
            scale = max(1.0, float(np.abs(out.cov).max()))
            assert out.isclose(direct, 1e-11 * scale)
            assert np.allclose(led.signal + led.added_noise(), out.cov, atol=1e-12 * scale)
            for c in led.contributions:  # every carried-forward noise is a covariance
                assert np.linalg.eigvalsh(c.cov).min() >= -1e-12 * scale
            rows = led.budget(0, 0.3)
            assert math.isclose(sum(r.fraction for r in rows), 1.0, rel_tol=1e-12)

    def test_chain_of_losses_closed_form(self):
        etas = [0.9, 0.95, 0.92]
        led = NoiseLedger.start(GaussianState.squeezed(1.0))
        for k, eta in enumerate(etas):
            led = led.channel(C.loss(eta), 0, f"L{k}", "loss")
        for k, c in enumerate(led.contributions):
            later = math.prod(etas[k + 1 :])
            assert np.allclose(c.cov, later * (1 - etas[k]) / 2 * np.eye(2), atol=1e-15)
        assert led.transmissivities() == pytest.approx({"L0": 0.9, "L1": 0.95, "L2": 0.92})
        assert np.allclose(led.signal, math.prod(etas) * GaussianState.squeezed(1.0).cov)

    def test_budget_units_grouping_and_what_if(self):
        r = 10 / cv.DB_PER_NEPER
        led = (
            NoiseLedger.start(GaussianState.squeezed(r))
            .channel(C.loss(0.90), 0, "OPO escape", "loss")
            .channel(C.loss(0.95), 0, "propagation", "loss")
            .channel(C.loss(0.92), 0, "detector QE", "loss")
            .channel(C.additive_noise(0.01), 0, "electronics", "electronic")
        )
        rows = {r.label: r for r in led.budget()}
        assert math.isclose(rows["electronics"].variance, 0.02, rel_tol=1e-12)  # SNU
        assert math.isclose(led.budget(snu=False)[-1].variance, 0.01, rel_tol=1e-12)
        assert set(led.by("kind")) == {"loss", "electronic"}
        total_db = -10 * math.log10(led.state.cov[0, 0] / 0.5)
        lost = 10 - total_db
        recovered = {c: led.db_recovered(c) for c in ["OPO escape", "propagation", "detector QE", "electronics"]}
        assert recovered["OPO escape"] == pytest.approx(1.26, abs=0.01)
        assert sum(recovered.values()) < lost  # the what-if view is not additive
        manual = (
            NoiseLedger.start(GaussianState.squeezed(r))
            .channel(C.loss(0.90), 0, "OPO escape", "loss")
            .channel(C.loss(0.95), 0, "propagation", "loss")
            .channel(C.loss(0.92), 0, "detector QE", "loss")
        )
        assert recovered["electronics"] == pytest.approx(10 * math.log10(led.state.cov[0, 0] / manual.state.cov[0, 0]))

    def test_phase_entries_are_flagged_and_warn(self):
        led = NoiseLedger.start(GaussianState.coherent(1.0)).phase_diffusion(0.2, 0, "LO lock")
        with pytest.warns(ApproximationWarning):
            contribs = led.contributions
        assert contribs[0].approximate and contribs[0].kind == "phase"
        assert contribs[0].cov[1, 1] > 0  # the coherent amplitude is smeared in p

    def test_gates_move_noise_between_modes(self):
        led = (
            NoiseLedger.start(GaussianState.vacuum(2))
            .channel(C.additive_noise(0.1), 0, "noise", "thermal")
            .gate(sy.beamsplitter(math.pi / 4), [0, 1], "50:50")
        )
        C0 = led.contributions[0].cov
        assert np.allclose(np.diag(C0), 0.05)  # half on each mode

    @pytest.mark.parametrize(
        "call",
        [
            lambda led: led.db_recovered("nobody"),
            lambda led: led.by("colour"),
            lambda led: led.channel(C.loss(0.5), [0, 1]),
            lambda led: led.channel("loss", 0),
            lambda led: led.phase_diffusion(-1.0),
            lambda led: led.gate(np.eye(2), 3),
        ],
    )
    def test_invalid(self, call):
        led = NoiseLedger.start(GaussianState.vacuum(2)).channel(C.loss(0.5), 0, "L", "loss")
        with pytest.raises(InputError):
            call(led)

    def test_start_needs_a_state(self):
        with pytest.raises(InputError):
            NoiseLedger.start(np.eye(2))
