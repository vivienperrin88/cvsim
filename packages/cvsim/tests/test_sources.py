"""SPOPO, single-pass waveguide and excess antisqueezing (S11).

Verification (roadmap S11):

* **OPO formula**: :meth:`SPOPO.spectrum` and the ledger state equal the spectra the oracle
  derives from quantum Langevin equations and input–output theory (``cvsim.reference.sources``),
  at every analysis frequency, to round-off (1e-13); threshold and CW (single-mode) limits.
* **Waveguide**: the per-supermode closed form equals the pixel-basis Lyapunov solution of the
  oracle (Van Loan exponential, no supermodes) to 1e-11, and the limit of N alternating
  squeeze / loss slices built from S06 channels: the error of the symmetric (Strang) splitting
  falls as 1/N² (checked between N = 50 and 100); ``κ → 0`` recovers the lossless state.
* **Excess noise**: phase-sensitive excess adds exactly the requested dB; gain fluctuations give
  the exact second moments of the mixture (Monte Carlo average of squeezed covariances, and
  Gauss–Hermite against direct quadrature); :func:`calibrate` inverts the models.
* **Ledger**: contributions add up to the state (exact), transmissivities are recorded,
  ``db_recovered`` of the escape loss gives back the ideal squeezing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.core import units as u
from cvsim.errors import ApproximationWarning, CalibrationError, InputError
from cvsim.gaussian import GaussianChannel, GaussianState
from cvsim.hardware import excess as ex
from cvsim.hardware import jsa, sources
from cvsim.reference import sources as ref

FAST = settings(max_examples=30, deadline=None)


@pytest.fixture(scope="module")
def sm() -> jsa.Supermodes:
    """Gaussian JSA on a small grid: K = 2.125, every supermode kept."""
    grid = jsa.FrequencyGrid.centered(24, 3.0)
    w = grid.detunings
    ws, wi = np.meshgrid(w, w, indexing="ij")
    f = np.exp(-((ws + wi) ** 2) / (2 * 0.7**2) - (ws - wi) ** 2 / (2 * 2.0**2) + 0.3j * (ws**2 + wi**2))
    return jsa.schmidt_decomposition(f, grid, tol=0, max_modes=24)


@pytest.fixture(scope="module")
def sm4(sm) -> jsa.Supermodes:
    return jsa.schmidt_decomposition(sm.jsa, sm.grid, max_modes=4)


def lossless(state: GaussianState) -> bool:
    return state.is_pure(atol=1e-9)


# ------------------------------------------------------------------------------- SPOPO
class TestSPOPO:
    @FAST
    @given(sigma=st.floats(0.0, 0.95), eta=st.floats(0.05, 1.0), x=st.floats(0.0, 20.0))
    def test_spectrum_matches_input_output_theory(self, sm4, sigma, eta, x):
        opo = sources.SPOPO(sm4, eta, linewidth=2.0, pump=sigma)
        sp = opo.spectrum([2.0 * x])
        sx, spp = ref.opo_spectra(sigma, eta, [x])
        assert sp.squeezed[0, 0] == pytest.approx(sx[0], abs=1e-13)
        assert sp.antisqueezed[0, 0] == pytest.approx(spp[0], abs=1e-12 * max(1.0, spp[0]))
        # every supermode: the same formula with σ_k = σ s_k/s₀
        for k, sk in enumerate(opo.pump_parameters):
            sxk, spk = ref.opo_spectra(sk, eta, [x])
            assert sp.squeezed[k, 0] == pytest.approx(sxk[0], abs=1e-13)
            assert sp.antisqueezed[k, 0] == pytest.approx(spk[0], abs=1e-12 * max(1.0, spk[0]))

    @FAST
    @given(sigma=st.floats(0.0, 0.95), eta=st.floats(0.05, 1.0), x=st.floats(0.0, 5.0))
    def test_ledger_state_equals_the_spectrum(self, sm4, sigma, eta, x):
        opo = sources.SPOPO(sm4, eta, linewidth=1.0, pump=sigma)
        out = opo.output(omega=x)
        sp = opo.spectrum([x])
        V = out.state.cov
        M = out.num_modes
        assert np.allclose(cv.to_snu(np.diag(V)[:M]), sp.squeezed[:, 0], atol=1e-13)
        assert np.allclose(cv.to_snu(np.diag(V)[M:]), sp.antisqueezed[:, 0], rtol=1e-12)
        assert np.allclose(V - np.diag(np.diag(V)), 0, atol=0)  # supermodes are independent, x squeezed

    def test_escape_what_if_gives_the_ideal_squeezing(self, sm4):
        out = sources.SPOPO(sm4, 0.8, 1.0, pump=0.6).output()
        sq, _ = out.quadrature_db()
        ideal = float(cv.db_from_r(-out.squeezing[0]))
        assert out.ledger.db_recovered("OPO escape") == pytest.approx(sq[0] - ideal, rel=1e-12)

    def test_pure_when_lossless_and_threshold_limit(self, sm4):
        assert lossless(sources.SPOPO(sm4, 1.0, 1.0, pump=0.8).output(omega=0.37).state)
        # σ → 1 at Ω = 0: S_− → 1 − η (the OPO formula's limit)
        eta = 0.9
        near = sources.SPOPO(sm4, eta, 1.0, pump=1 - 1e-9).spectrum([0.0])
        assert near.squeezed[0, 0] == pytest.approx(1 - eta, abs=1e-8)
        # far outside the cavity bandwidth: vacuum
        far = sources.SPOPO(sm4, eta, 1.0, pump=0.5).spectrum([1e6])
        assert np.allclose([far.squeezed, far.antisqueezed], 1.0, atol=1e-11)

    def test_cw_limit_single_mode(self):
        """One supermode (separable JSA): the CW OPO, whose squeezing at Ω = 0 is
        e^{−2r} = ((1 − σ)/(1 + σ))², i.e. r = artanh-like ln((1 + σ)/(1 − σ))."""
        grid = jsa.FrequencyGrid.centered(5, 1.0)
        f = np.outer([1e-3, 0.5, 1.0, 0.5, 1e-3], [1e-3, 0.5, 1.0, 0.5, 1e-3])
        one = jsa.schmidt_decomposition(f, grid)
        assert one.kept == 1 and one.schmidt_number == pytest.approx(1.0)
        opo = sources.SPOPO(one, 1.0, 1.0, pump=0.5)
        assert opo.squeezing_parameters(0.0)[0] == pytest.approx(math.log(3.0), rel=1e-14)

    def test_from_cavity(self, sm4):
        opo = sources.SPOPO.from_cavity(sm4)
        assert opo.escape_efficiency == pytest.approx(0.20 / 0.21)
        assert opo.linewidth == pytest.approx(76e6 * 0.21 / 2)
        assert u.hertz(opo.linewidth) / 1e6 == pytest.approx(1.270, abs=1e-3)
        assert opo.threshold_power_ratio() == pytest.approx(0.25)
        assert opo.output().ledger.transmissivities() == {"OPO escape": pytest.approx(0.20 / 0.21)}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"escape_efficiency": 0.0, "linewidth": 1.0},
            {"escape_efficiency": 1.1, "linewidth": 1.0},
            {"escape_efficiency": 0.9, "linewidth": 0.0},
            {"escape_efficiency": 0.9, "linewidth": 1.0, "pump": 1.0},
            {"escape_efficiency": 0.9, "linewidth": 1.0, "pump": -0.1},
        ],
    )
    def test_invalid(self, sm4, kwargs):
        with pytest.raises(InputError):
            sources.SPOPO(sm4, **kwargs)

    def test_invalid_cavity_and_frequencies(self, sm4):
        with pytest.raises(InputError, match="threshold"):
            sources.SPOPO(sm4, 0.9, 1.0, pump=1.2)
        with pytest.raises(InputError):
            sources.SPOPO("modes", 0.9, 1.0)  # type: ignore[arg-type]
        for kw in ({"transmission": 0.0}, {"loss": 1.0}, {"repetition_rate": -1.0}):
            with pytest.raises(InputError):
                sources.SPOPO.from_cavity(sm4, **kw)
        opo = sources.SPOPO(sm4, 0.9, 1.0)
        with pytest.raises(InputError):
            opo.squeezing_parameters(math.nan)
        with pytest.raises(InputError):
            opo.spectrum([[1.0, 2.0]])


# ------------------------------------------------------------------------------- waveguide
def sliced_waveguide(r: np.ndarray, kl: float, n: int) -> GaussianState:
    """Strang splitting with S06 channels: (loss ½ · squeeze · loss ½)^n on every supermode."""
    M = r.size
    half = GaussianChannel(
        math.sqrt(math.exp(-kl / (2 * n))) * np.eye(2 * M), 0.5 * -math.expm1(-kl / (2 * n)) * np.eye(2 * M)
    )
    sq = GaussianChannel.from_symplectic(np.diag(np.concatenate([np.exp(-r / n), np.exp(r / n)])))
    step = half.then(sq).then(half)
    state = GaussianState.vacuum(M)
    for _ in range(n):
        state = step.apply(state)
    return state


class TestWaveguide:
    @FAST
    @given(r0=st.floats(0.0, 2.0), db=st.floats(0.0, 3.0), length=st.floats(1e-3, 0.06))
    def test_closed_form_equals_pixel_lyapunov(self, sm, r0, db, length):
        wg = sources.Waveguide(sm, length=length, loss_db_per_cm=db, r0=r0, output_coupling=1.0)
        pix = wg.output().in_pixel_basis().cov
        V = ref.lossy_multimode_squeezer(sm.coupling(r0), wg.attenuation_length)
        assert np.allclose(pix, V, atol=1e-11 * math.exp(2 * r0))

    def test_slices_converge_quadratically(self, sm4):
        wg = sources.Waveguide(sm4, length=0.02, loss_db_per_cm=1.0, r0=1.5, output_coupling=1.0)
        exact = wg.output().state.cov
        errors = [
            np.abs(sliced_waveguide(wg.squeezing_parameters, wg.attenuation_length, n).cov - exact).max()
            for n in (50, 100)
        ]
        assert errors[1] < 1e-4
        assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.02)

    def test_lossless_limit_and_zero_squeezing(self, sm4):
        wg = sources.Waveguide(sm4, loss_db_per_cm=0.0, r0=1.0, output_coupling=1.0)
        out = wg.output()
        assert out.state.isclose(GaussianState.squeezed(wg.squeezing_parameters), atol=1e-14)
        assert lossless(out.state)
        assert np.allclose(wg.effective_transmissivity(), 1.0, atol=1e-14)
        flat = sources.Waveguide(sm4, loss_db_per_cm=1.0, r0=0.0).output().state
        assert flat.isclose(GaussianState.vacuum(4), atol=1e-15)
        assert np.all(np.isnan(sources.Waveguide(sm4, r0=0.0).effective_transmissivity()))

    def test_propagation_closed_form_at_2r_equal_kl(self):
        """The antisqueezed rate vanishes (2r = κℓ): Y_p → κℓ/2, continuous."""
        yx, yp = sources.waveguide_propagation([0.25, 0.25 + 1e-10], 0.5)
        assert yp[0] == pytest.approx(0.25, rel=1e-15)
        assert yp[1] == pytest.approx(yp[0], rel=1e-9)
        assert yx[0] == pytest.approx(0.25 * (1 - math.exp(-1.0)), rel=1e-14)

    def test_distributed_loss_costs_less_than_lumped(self, sm4):
        wg = sources.Waveguide(sm4, length=0.02, loss_db_per_cm=1.0, r0=1.0, output_coupling=1.0)
        eff = wg.effective_transmissivity()
        assert np.all(eff > wg.propagation_transmissivity)
        assert np.all(eff < 1)
        # and reproduces the squeezed variance by construction
        out = wg.output()
        r = wg.squeezing_parameters
        lumped = 0.5 * (eff * np.exp(-2 * r) + 1 - eff)
        assert np.allclose(np.diag(out.state.cov)[:4], lumped, atol=1e-14)

    def test_ledger(self, sm4):
        wg = sources.Waveguide(sm4)
        out = wg.output()
        assert out.ledger.transmissivities() == {
            "waveguide propagation loss": pytest.approx(10 ** (-0.2 * 1.0 / 10)),
            "waveguide output coupling": pytest.approx(0.9),
        }
        total = out.ledger.signal + out.ledger.added_noise()
        assert np.allclose(total, out.state.cov, atol=1e-15)
        assert [c.component for c in out.ledger.contributions] == [
            "waveguide propagation loss",
            "waveguide output coupling",
        ]
        # what-if: the output coupling made ideal recovers squeezing; distributed loss alone remains
        sq_db, _ = out.quadrature_db()
        alone = sources.Waveguide(sm4, output_coupling=1.0).output().quadrature_db()[0]
        assert out.ledger.db_recovered("waveguide output coupling") == pytest.approx(sq_db[0] - alone[0], rel=1e-12)

    @pytest.mark.parametrize(
        "kwargs",
        [{"length": 0.0}, {"loss_db_per_cm": -1.0}, {"r0": -0.1}, {"output_coupling": 0.0}, {"output_coupling": 1.5}],
    )
    def test_invalid(self, sm4, kwargs):
        with pytest.raises(InputError):
            sources.Waveguide(sm4, **kwargs)
        with pytest.raises(InputError):
            sources.Waveguide(None)  # type: ignore[arg-type]
        with pytest.raises(InputError):
            sources.waveguide_propagation([0.1], -1.0)


# ------------------------------------------------------------------------------- output object
class TestSourceOutput:
    def test_pixel_basis_round_trip_and_truncated_modes_in_vacuum(self, sm4):
        out = sources.SPOPO(sm4, 0.9, 1.0, pump=0.7).output()
        pix = out.in_pixel_basis()
        assert pix.num_modes == sm4.grid.size
        assert pix.purity() == pytest.approx(out.state.purity(), rel=1e-12)
        from cvsim.gaussian import symplectic as sy

        back = sy.change_basis(pix, sm4.unitary)
        assert back.reduce(range(4)).isclose(out.state, atol=1e-12)
        assert back.reduce(range(4, sm4.grid.size)).isclose(GaussianState.vacuum(sm4.grid.size - 4), atol=1e-12)

    def test_pixel_basis_when_every_mode_is_kept(self, sm):
        out = sources.Waveguide(sm, r0=0.5).output()
        assert out.in_pixel_basis().num_modes == sm.kept == sm.grid.size

    def test_quadrature_db(self, sm4):
        out = sources.SPOPO(sm4, 1.0, 1.0, pump=0.5).output(omega=0.0)
        sq, anti = out.quadrature_db()
        assert sq[0] == pytest.approx(float(cv.db_from_r(-math.log(3.0))), rel=1e-12)
        assert anti[0] == pytest.approx(-sq[0], rel=1e-12)


# ------------------------------------------------------------------------------- excess noise
class TestExcessAntisqueezing:
    @pytest.mark.parametrize("db", [0.3, 1.0, 3.0])
    def test_adds_the_requested_db_to_supermode_0(self, sm4, db):
        opo = sources.SPOPO(sm4, 0.85, 1.0, pump=0.6)
        base = opo.output()
        noisy = opo.output(excess=ex.ExcessNoise(antisqueezing_db=db))
        _, a0 = base.quadrature_db()
        s1, a1 = noisy.quadrature_db()
        assert a1[0] - a0[0] == pytest.approx(db, rel=1e-12)
        assert np.allclose(s1, base.quadrature_db()[0], atol=1e-14)  # squeezing untouched
        C = noisy.ledger.by("component")["excess antisqueezing"]
        eps = np.diag(C)[4:]
        r = noisy.squeezing
        assert np.allclose(eps / eps[0], r / r[0], rtol=1e-12)  # proportional scaling
        assert noisy.ledger.by("kind")["excess"].trace() == pytest.approx(eps.sum())

    def test_constant_scaling(self, sm4):
        noisy = sources.Waveguide(sm4).output(excess=ex.ExcessNoise(antisqueezing_db=1.0, scaling="constant"))
        eps = np.diag(noisy.ledger.by("component")["excess antisqueezing"])[4:]
        assert np.allclose(eps, eps[0], rtol=1e-14)

    def test_proportional_with_no_squeezing_adds_nothing(self):
        e = ex.antisqueezing_noise([0.5, 0.5], [0.0, 0.0], ex.ExcessNoise(antisqueezing_db=1.0))
        assert np.all(e == 0)
        with pytest.raises(InputError):
            ex.antisqueezing_noise([0.5], [0.0, 0.1], ex.ExcessNoise(antisqueezing_db=1.0))

    def test_settings_validation(self):
        assert not ex.ExcessNoise().active
        assert ex.ExcessNoise(gain_rms=0.1).active
        for kw in ({"antisqueezing_db": -1.0}, {"gain_rms": math.nan}, {"scaling": "linear"}):
            with pytest.raises(InputError):
                ex.ExcessNoise(**kw)


class TestGainFluctuations:
    def test_noise_is_the_exact_mixture_moment(self, rng):
        r = np.array([0.8, 0.3])
        g = 0.1
        Y = ex.gain_fluctuation_noise(r, g)
        samples = r[None, :] * (1 + g * rng.normal(size=(200_000, 1)))
        mc = 0.5 * np.concatenate([np.exp(-2 * samples).mean(0), np.exp(2 * samples).mean(0)])
        nominal = 0.5 * np.concatenate([np.exp(-2 * r), np.exp(2 * r)])
        assert np.allclose(nominal + Y, mc, rtol=5e-3)
        # the closed form ⟨e^{±2r}⟩ = e^{±2r̄ + 2σ²}
        sig = g * r
        assert np.allclose(nominal + Y, 0.5 * np.concatenate([np.exp(-2 * r + 2 * sig**2), np.exp(2 * r + 2 * sig**2)]))

    def test_gauss_hermite_average(self):
        r = np.array([0.5, 1.0])
        got = ex.gain_average(lambda q: np.exp(2 * q), r, 0.2)
        assert np.allclose(got, np.exp(2 * r + 2 * (0.2 * r) ** 2), rtol=1e-14)

    def test_spopo_with_gain_fluctuations(self, sm4):
        opo = sources.SPOPO(sm4, 0.9, 1.0, pump=0.6)
        with pytest.warns(ApproximationWarning, match="mixture"):
            noisy = opo.output(excess=ex.ExcessNoise(gain_rms=0.05))
        r = noisy.squeezing
        sig = 0.05 * r
        eta = 0.9
        vx = eta * 0.5 * np.exp(-2 * r + 2 * sig**2) + (1 - eta) / 2
        assert np.allclose(np.diag(noisy.state.cov)[:4], vx, rtol=1e-13)

    def test_waveguide_with_gain_fluctuations_is_the_average_output(self, sm4):
        """Exact second moments: E_r[V_out(r)] by brute-force quadrature over r of the closed form."""
        g = 0.08
        wg = sources.Waveguide(sm4, length=0.02, loss_db_per_cm=1.0, r0=1.2, output_coupling=1.0)
        with pytest.warns(ApproximationWarning):
            noisy = wg.output(excess=ex.ExcessNoise(gain_rms=g))
        r = wg.squeezing_parameters
        kl = wg.attenuation_length
        x = np.linspace(-8, 8, 4001)
        w = np.exp(-(x**2) / 2)
        w /= w.sum()
        vx = np.zeros(4)
        vp = np.zeros(4)
        for xi, wi in zip(x, w):
            q = r * (1 + g * xi)
            yx, yp = sources.waveguide_propagation(q, kl)
            vx += wi * (0.5 * math.exp(-kl) * np.exp(-2 * q) + yx)
            vp += wi * (0.5 * math.exp(-kl) * np.exp(2 * q) + yp)
        V = np.diag(noisy.state.cov)
        assert np.allclose(V[:4], vx, rtol=1e-10)
        assert np.allclose(V[4:], vp, rtol=1e-10)

    def test_invalid(self):
        with pytest.raises(InputError):
            ex.gain_fluctuation_noise([0.1], -0.1)
        with pytest.raises(InputError):
            ex.gain_fluctuation_noise([[0.1]], 0.1)


class TestCalibration:
    @FAST
    @given(r=st.floats(0.05, 2.0), eta=st.floats(0.1, 1.0))
    def test_pure_loss_round_trip(self, r, eta):
        vm = eta * math.exp(-2 * r) + 1 - eta
        vp = eta * math.exp(2 * r) + 1 - eta
        cal = ex.calibrate(10 * math.log10(vm), 10 * math.log10(vp))
        assert cal.r == pytest.approx(r, rel=1e-9)
        assert cal.efficiency == pytest.approx(eta, rel=1e-9)
        assert cal.excess == 0 and cal.excess_db == 0
        known = ex.calibrate(10 * math.log10(vm), 10 * math.log10(vp), efficiency=eta)
        assert known.r == pytest.approx(r, rel=1e-9) and known.excess_db == 0.0

    def test_recovers_the_excess_of_the_model(self, sm4):
        opo = sources.SPOPO(sm4, 0.8, 1.0, pump=0.6)
        out = opo.output(excess=ex.ExcessNoise(antisqueezing_db=1.5))
        sq, anti = out.quadrature_db()
        cal = ex.calibrate(sq[0], anti[0], efficiency=0.8)
        assert cal.r == pytest.approx(out.squeezing[0], rel=1e-12)
        assert cal.excess_db == pytest.approx(1.5, rel=1e-12)
        pred = 0.8 * math.exp(2 * cal.r) + 0.2
        assert cal.excess == pytest.approx(10 ** (anti[0] / 10) - pred, rel=1e-12)
        # without the efficiency, the excess is read as extra loss instead
        assert ex.calibrate(sq[0], anti[0]).efficiency < 0.8

    @pytest.mark.parametrize(
        ("r", "eta"), [(1e-8, 0.5), (1e-6, 0.7), (1e-4, 0.3), (1e-3, 1.0), (0.5, 0.5), (3.0, 0.999999)]
    )
    def test_weak_squeezing_keeps_precision(self, r, eta):
        """Review S11: ``(V₊ − 1)/(u − 1)`` cancelled for weak squeezing (relative error 5e-5 at
        ``r = 1e-6``, 10 % at ``1e-8``). dB values formed without cancellation (``log1p``/``expm1``)
        now give back ``r`` and ``η`` to ~1e-10: both need the second-order ``V₊ + V₋ − 2``, whose
        relative round-off from the dB inputs grows as ``ε/r`` (2e-10 at ``r = 1e-8``)."""
        ln10 = math.log(10.0)
        sq = 10 * math.log1p(eta * math.expm1(-2 * r)) / ln10
        anti = 10 * math.log1p(eta * math.expm1(2 * r)) / ln10
        cal = ex.calibrate(sq, anti)
        assert cal.r == pytest.approx(r, rel=1e-9)
        assert cal.efficiency == pytest.approx(eta, rel=1e-9)

    def test_nearly_pure_data_read_as_pure(self):
        """Review S11: inside the rtol band the fit gave r < 0 and η < 0, or divided by zero."""
        vm = 1 - 1e-5
        vp = 1 / vm - 1e-9  # V₋V₊ = 1 − 1e-9: within rtol
        cal = ex.calibrate(10 * math.log10(vm), 10 * math.log10(vp))
        assert cal.efficiency == 1.0
        assert cal.r == pytest.approx(0.25 * math.log(vp / vm), rel=1e-12) and cal.r > 0
        tiny = ex.calibrate(-1e-12, 1e-12)
        assert tiny.efficiency == 1.0
        assert tiny.r == pytest.approx(1e-12 * math.log(10) / 20, rel=1e-9)
        # 1 − V₋ underflows: no finite fit (was ZeroDivisionError, or r = inf)
        for sq in (-5e-324, -1e-320):
            with pytest.raises(CalibrationError, match="too weak"):
                ex.calibrate(sq, 1.0)

    @pytest.mark.parametrize(
        ("r", "eta", "extra"), [(1e-7, 0.5, 0.0), (1e-4, 0.8, 0.0), (1e-4, 0.8, 1e-6), (1.0, 0.6, 0.1)]
    )
    def test_known_efficiency_without_cancellation(self, r, eta, extra):
        """Review of the fix: the efficiency branch formed 1 − V₋ and V₊ − prediction by cancellation."""
        ln10 = math.log(10.0)
        sq = 10 * math.log1p(eta * math.expm1(-2 * r)) / ln10
        anti = 10 * math.log1p(eta * math.expm1(2 * r) + extra) / ln10
        cal = ex.calibrate(sq, anti, efficiency=eta, rtol=1e-12)
        assert cal.r == pytest.approx(r, rel=1e-12)
        assert cal.excess == pytest.approx(extra, rel=1e-6, abs=0)
        if extra:
            predicted = eta * math.expm1(2 * r) + 1
            assert cal.excess_db == pytest.approx(10 * math.log10((predicted + extra) / predicted), rel=1e-9)

    @FAST
    @given(r=st.floats(1e-9, 3.0), eta=st.floats(1e-3, 1.0))
    def test_fit_is_physical(self, r, eta):
        vm = eta * math.exp(-2 * r) + 1 - eta
        vp = eta * math.exp(2 * r) + 1 - eta
        if not vm < 1 < vp:
            return  # r·η below double precision: not representable as data
        cal = ex.calibrate(10 * math.log10(vm), 10 * math.log10(vp))
        assert cal.r > 0 and 0 < cal.efficiency <= 1

    def test_inconsistent_data(self):
        with pytest.raises(CalibrationError, match="uncertainty"):
            ex.calibrate(-6.0, 3.0)
        with pytest.raises(CalibrationError, match="deeper"):
            ex.calibrate(-10.0, 12.0, efficiency=0.8)
        with pytest.raises(CalibrationError, match="below the loss-only"):
            ex.calibrate(-3.0, 3.5, efficiency=0.6)
        for args in ((1.0, 3.0), (-3.0, -1.0), (math.nan, 3.0)):
            with pytest.raises(InputError):
                ex.calibrate(*args)
        with pytest.raises(InputError):
            ex.calibrate(-3.0, 6.0, efficiency=0.0)
        for kwargs in ({"rtol": -1.0}, {"rtol": math.nan}, {"efficiency": "high"}):
            with pytest.raises(InputError):
                ex.calibrate(-3.0, 6.0, **kwargs)
        with pytest.raises(InputError, match="squeezing_db must be a real number"):
            ex.calibrate("-3", 6.0)


class TestValidation:
    """Review S11: one validator for every scalar, raising InputError also for non-numbers."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: u.angular_frequency("1550 nm"),
            lambda: u.angular("abc"),
            lambda: u.hertz(math.inf),
            lambda: cv.rotation(np.complex64(1 + 2j)),
            lambda: cv.rotation(np.True_),
            lambda: u.attenuation_from_db_per_cm(None),
            lambda: cv.rotation("x"),
            lambda: jsa.PhaseMatching.sinc("0.5", 0.2),
            lambda: jsa.FrequencyGrid.centered(4, "1"),
            lambda: ex.ExcessNoise(gain_rms="0.1"),
            lambda: ex.ExcessNoise(antisqueezing_db=-1.0),
            lambda: ex.gain_fluctuation_noise([1.0], math.nan),
            lambda: ex.gain_average(np.exp, [1.0], -0.1),
            lambda: sources.waveguide_propagation([1.0], "1"),
            lambda: sources.waveguide_propagation([1.0], -1.0),
        ],
    )
    def test_invalid_scalars_raise_input_error(self, call):
        with pytest.raises(InputError, match="must be"):
            call()

    def test_source_parameters(self, sm4):
        for kwargs in ({"escape_efficiency": "0.9"}, {"linewidth": 0.0}, {"pump": "0.5"}, {"pump": 1.0}):
            args = {"escape_efficiency": 0.9, "linewidth": 1.0, "pump": 0.5} | kwargs
            with pytest.raises(InputError):
                sources.SPOPO(sm4, **args)
        for kwargs in ({"length": 0.0}, {"loss_db_per_cm": -1.0}, {"r0": "1"}, {"output_coupling": 1.5}):
            with pytest.raises(InputError):
                sources.Waveguide(sm4, **kwargs)
        with pytest.raises(InputError, match="omega"):
            sources.SPOPO(sm4, 0.9, 1.0).squeezing_parameters(math.inf)

    def test_gain_average_of_several_quantities(self):
        r = np.array([0.3, 1.0, 1.7])
        both = ex.gain_average(lambda q: np.stack(sources.waveguide_propagation(q, 0.4)), r, 0.1)
        assert both.shape == (2, 3)
        for k in (0, 1):
            one = ex.gain_average(lambda q, k=k: sources.waveguide_propagation(q, 0.4)[k], r, 0.1)
            assert np.array_equal(both[k], one)

    def test_numpy_integers_are_integers(self):
        """Review of the fix: 0-d NumPy arrays and NumPy floats of integral value are accepted sizes."""
        for n in (np.array(64), np.int64(64), np.float32(64.0), 64.0):
            assert jsa.FrequencyGrid.centered(n, 1.0).size == 64
        for n in (np.True_, 64.5, np.array([64]), "64"):
            with pytest.raises(InputError, match="integer"):
                jsa.FrequencyGrid.centered(n, 1.0)
