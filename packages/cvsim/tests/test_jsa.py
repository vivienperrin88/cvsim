"""Units of the hardware layer and the JSA → Schmidt supermodes pipeline (S11).

Verification (roadmap S11): analytic limits.

* Gaussian JSA ``exp(−Ω₊²/(2A²) − Ω₋²/(2B²))``: Mehler's formula gives the Schmidt coefficients
  ``s_n = √(1 − μ²) |μ|ⁿ``, ``μ = (A − B)/(A + B)``, the Schmidt number ``K = (A² + B²)/(2AB)``
  and Hermite–Gauss supermodes of width ``w² = AB/2``. On a grid that holds the JSA to ``e^{−40}``
  the discretisation error is spectral: tolerances 1e-12 (coefficients) and 1e-9 (mode shapes,
  sampled Hermite functions vs pixel vectors).
* CW limit: a pump narrower than the grid step makes ``J`` anti-diagonal; the supermodes come in
  degenerate sideband pairs ``(Ω, −Ω)`` with ``s ∝ |φ(Ω, −Ω)|``.
* Pixel-basis consistency: the Takagi route equals the multimode squeezer built from the pixel
  Hamiltonian by the oracle (``cvsim.reference.sources``), itself checked against the Fock oracle
  on two modes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import units as u
from cvsim.errors import ApproximationWarning, InputError
from cvsim.gaussian import GaussianState
from cvsim.gaussian import symplectic as sy
from cvsim.hardware import jsa, sources
from cvsim.reference import fock_dense as fd
from cvsim.reference import sources as ref

FAST = settings(max_examples=30, deadline=None)


def gaussian_jsa(A: float, B: float, n: int = 161, half: float | None = None, phase: float = 0.0):
    h = 9.0 * max(A, B) if half is None else half
    grid = jsa.FrequencyGrid.centered(n, h)
    w = grid.detunings
    ws, wi = np.meshgrid(w, w, indexing="ij")
    f = np.exp(-((ws + wi) ** 2) / (2 * A**2) - (ws - wi) ** 2 / (2 * B**2) + 1j * phase * (ws**2 + wi**2))
    return grid, f


def hermite_function(n: int, x: np.ndarray) -> np.ndarray:
    """Normalised Hermite function ψ_n(x) = H_n(x) e^{−x²/2} / √(2ⁿ n! √π)."""
    h = np.polynomial.hermite.hermval(x, [0] * n + [1])
    return h * np.exp(-(x**2) / 2) / math.sqrt(2.0**n * math.factorial(n) * math.sqrt(math.pi))


# ------------------------------------------------------------------------------- units
class TestUnits:
    def test_round_trips(self):
        lam = 795 * u.NM
        assert u.wavelength(u.angular_frequency(lam)) == pytest.approx(lam, rel=1e-15)
        dw = u.angular_bandwidth(2 * u.NM, lam)
        assert u.wavelength_bandwidth(dw, lam) == pytest.approx(2 * u.NM, rel=1e-15)
        assert u.hertz(u.angular(76 * u.MHZ)) == pytest.approx(76e6, rel=1e-15)
        assert u.db_per_cm_from_attenuation(u.attenuation_from_db_per_cm(0.3)) == pytest.approx(0.3, rel=1e-15)
        assert u.fwhm_from_amplitude_sigma(u.amplitude_sigma_from_fwhm(3.0)) == pytest.approx(3.0, rel=1e-15)

    def test_values(self):
        # 0.2 dB/cm over 10 cm is 2 dB: transmissivity 10^{-0.2}
        assert math.exp(-u.attenuation_from_db_per_cm(0.2) * 10 * u.CM) == pytest.approx(10**-0.2, rel=1e-14)
        # intensity exp(−x²/σ²) is ½ at x = FWHM/2
        sigma = u.amplitude_sigma_from_fwhm(2.0)
        assert math.exp(-(1.0**2) / sigma**2) == pytest.approx(0.5, rel=1e-14)
        # Δω/ω = Δλ/λ to first order
        lam = 1550 * u.NM
        assert u.angular_bandwidth(1 * u.NM, lam) / u.angular_frequency(lam) == pytest.approx(1 / 1550, rel=1e-14)

    @pytest.mark.parametrize(
        ("fn", "args"),
        [
            (u.angular_frequency, (0.0,)),
            (u.wavelength, (-1.0,)),
            (u.angular_bandwidth, (-1.0, 1e-6)),
            (u.wavelength_bandwidth, (1.0, 0.0)),
            (u.angular, (math.inf,)),
            (u.hertz, (math.nan,)),
            (u.attenuation_from_db_per_cm, (-0.1,)),
            (u.db_per_cm_from_attenuation, (-0.1,)),
            (u.amplitude_sigma_from_fwhm, (0.0,)),
            (u.fwhm_from_amplitude_sigma, (-1.0,)),
        ],
    )
    def test_invalid(self, fn, args):
        with pytest.raises(InputError):
            fn(*args)


# ------------------------------------------------------------------------------- grid, pump, phase matching
class TestGrid:
    def test_centered_is_symmetric(self):
        g = jsa.FrequencyGrid.centered(10, 4.5)
        assert np.allclose(g.detunings + g.detunings[::-1], 0, atol=1e-15)
        assert g.step == pytest.approx(1.0)
        assert g.size == 10
        assert "n=10" in repr(g)

    @pytest.mark.parametrize("bad", [[0.0], [0.0, 1.0, 3.0], [1.0, 0.0], [0.0, np.nan]])
    def test_invalid_detunings(self, bad):
        with pytest.raises(InputError):
            jsa.FrequencyGrid(bad)

    @pytest.mark.parametrize(("n", "h"), [(1, 1.0), (2.5, 1.0), (4, 0.0), (4, math.inf)])
    def test_invalid_centered(self, n, h):
        with pytest.raises(InputError):
            jsa.FrequencyGrid.centered(n, h)

    def test_auto_resolves_and_contains(self):
        pump = jsa.Pump.gaussian(1.0)
        pm = jsa.PhaseMatching.sinc(gvm=0.5, gvd_signal=0.2)
        g = jsa.FrequencyGrid.auto(pump, pm)
        w_plus = min(0.25, pm.half_width_sum)
        assert g.step <= w_plus / 2 * (1 + 1e-12)
        assert g.detunings[-1] == pytest.approx(4.0 * (w_plus + pm.half_width_difference))

    def test_auto_warns_when_coarse(self):
        with pytest.warns(ApproximationWarning, match="does not resolve"):
            jsa.FrequencyGrid.auto(jsa.Pump.gaussian(1.0), jsa.PhaseMatching.sinc(0.5, 0.2), n=16)

    def test_auto_needs_signal_gvd(self):
        with pytest.raises(InputError, match="gvd_signal"):
            jsa.FrequencyGrid.auto(jsa.Pump.gaussian(1.0), jsa.PhaseMatching.sinc(gvm=0.5, gvd_signal=0.0))
        with pytest.raises(InputError, match="extent"):
            jsa.FrequencyGrid.auto(jsa.Pump.gaussian(1.0), jsa.PhaseMatching.sinc(0.5, 0.2), extent=0)


class TestPumpAndPhaseMatching:
    def test_gaussian_pump_fwhm_and_chirp(self):
        p = jsa.Pump.gaussian(2.0, chirp=0.3)
        assert abs(p.amplitude(np.array([1.0]))[0]) ** 2 == pytest.approx(0.5, rel=1e-14)
        assert np.angle(p.amplitude(np.array([1.0]))[0]) == pytest.approx(0.15, rel=1e-14)
        assert p.half_width == 1.0

    def test_sampled_pump_interpolates_and_vanishes_outside(self):
        p = jsa.Pump.sampled([-1.0, 0.0, 1.0], [0.0, 1.0 + 1.0j, 0.0])
        out = p.amplitude(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
        assert np.allclose(out, [0, 0.5 + 0.5j, 1 + 1j, 0.5 + 0.5j, 0], atol=1e-15)
        assert p.half_width == 1.0

    @pytest.mark.parametrize(
        ("w", "a"), [([0.0], [1.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0, 1.0]), ([0.0, 1.0], [0.0, 0.0])]
    )
    def test_sampled_pump_invalid(self, w, a):
        with pytest.raises(InputError):
            jsa.Pump.sampled(w, a)

    def test_sinc_phase_matching(self):
        pm = jsa.PhaseMatching.sinc(gvm=0.4, gvd_signal=0.3, gvd_pump=0.1)
        ws, wi = np.array([0.7]), np.array([-0.2])
        dkl = 0.4 * 0.5 + 0.5 * 0.1 * 0.25 - 0.5 * 0.3 * (0.49 + 0.04)
        expected = math.sin(dkl / 2) / (dkl / 2) * np.exp(0.5j * dkl)
        assert pm.function(ws, wi)[0] == pytest.approx(expected, rel=1e-14)
        # first zeros along the anti-diagonal and the diagonal
        w0 = pm.half_width_difference
        assert abs(pm.function(np.array([w0]), np.array([-w0]))[0]) < 1e-15
        assert pm.half_width_difference == pytest.approx(math.sqrt(2 * math.pi / 0.3))
        assert pm.half_width_sum <= math.pi / 0.4

    def test_gaussian_phase_matching_matches_sinc_fwhm(self):
        g = jsa.PhaseMatching.gaussian(gvm=1.0, gvd_signal=0.0)
        s = jsa.PhaseMatching.sinc(gvm=1.0, gvd_signal=0.0)
        # sinc(x) = ½ at x ≈ 1.89549; the Gaussian approximation γ = 0.193 has the same amplitude FWHM
        x = 1.89549
        w = np.array([x])  # Ω_s = Ω_i = Ω gives ΔkL/2 = Ω for τ = 1
        assert abs(s.function(w, w)[0]) == pytest.approx(0.5, abs=1e-5)
        assert abs(g.function(w, w)[0]) == pytest.approx(0.5, rel=1e-3)
        assert g.half_width_difference == math.inf
        assert g.shape == "gaussian"

    def test_invalid_dispersion(self):
        with pytest.raises(InputError):
            jsa.PhaseMatching.sinc(gvm=math.nan, gvd_signal=1.0)

    def test_jsa_is_pump_times_phase_matching(self):
        pump = jsa.Pump.gaussian(1.0)
        pm = jsa.PhaseMatching.sinc(0.5, 0.2)
        g = jsa.FrequencyGrid.centered(7, 2.0)
        f = jsa.joint_spectral_amplitude(pump, pm, g)
        w = g.detunings
        assert f[2, 5] == pytest.approx(pump.amplitude(np.array([w[2] + w[5]]))[0] * pm.function(w[2:3], w[5:6])[0])
        assert np.allclose(f, f.T, atol=0)


# ------------------------------------------------------------------------------- Schmidt decomposition
class TestGaussianJSA:
    @pytest.mark.parametrize(("A", "B"), [(1.0, 0.25), (0.3, 1.0), (1.0, 1.0)])
    def test_schmidt_coefficients_and_number(self, A, B):
        grid, f = gaussian_jsa(A, B)
        sm = jsa.schmidt_decomposition(f, grid, tol=1e-14)
        mu = (A - B) / (A + B)
        n = np.arange(sm.kept)
        assert np.allclose(sm.coefficients[: sm.kept], math.sqrt(1 - mu**2) * abs(mu) ** n, atol=1e-12)
        assert sm.schmidt_number == pytest.approx((A**2 + B**2) / (2 * A * B), rel=1e-12)
        assert np.sum(sm.coefficients**2) == pytest.approx(1.0, rel=1e-14)

    def test_supermodes_are_hermite_gauss(self):
        A, B = 1.0, 0.3
        grid, f = gaussian_jsa(A, B, n=201)
        sm = jsa.schmidt_decomposition(f, grid, max_modes=6)
        w = math.sqrt(A * B / 2)
        for k in range(4):
            expected = hermite_function(k, grid.detunings / w) / math.sqrt(w)
            got = sm.functions[:, k]
            got = got * np.sign(np.real(got @ expected))  # the Takagi gauge fixes signs, not this convention
            assert np.allclose(got, expected, atol=1e-9 * np.abs(expected).max())

    def test_chirped_jsa_has_the_same_schmidt_coefficients(self):
        """A phase e^{iφ(Ω_s) + iφ(Ω_i)} is a local unitary: coefficients unchanged, modes multiplied."""
        grid, f0 = gaussian_jsa(1.0, 0.3)
        _, f1 = gaussian_jsa(1.0, 0.3, phase=0.7)
        s0 = jsa.schmidt_decomposition(f0, grid).coefficients
        s1 = jsa.schmidt_decomposition(f1, grid).coefficients
        assert np.allclose(s0[:10], s1[:10], atol=1e-13)

    def test_modes_orthonormal_and_reconstruct(self):
        grid, f = gaussian_jsa(1.0, 0.2)
        sm = jsa.schmidt_decomposition(f, grid)
        U, s = sm.unitary, sm.coefficients
        assert np.allclose(U.conj().T @ U, np.eye(grid.size), atol=1e-11)
        assert np.allclose(U @ np.diag(s) @ U.T, sm.jsa, atol=1e-11)
        assert np.allclose(np.sum(np.abs(sm.functions) ** 2, axis=0) * grid.step, 1.0, atol=1e-12)

    def test_truncation(self):
        grid, f = gaussian_jsa(1.0, 0.25)
        sm = jsa.schmidt_decomposition(f, grid, tol=1e-6)
        mu = 0.6
        # weights (1 − μ²) μ^{2n}: the tail beyond k modes is μ^{2k}
        k = math.ceil(math.log(1e-6) / math.log(mu**2))
        assert sm.kept == k
        assert sm.dropped_weight == pytest.approx(mu ** (2 * k), rel=1e-9)
        assert jsa.schmidt_decomposition(f, grid, max_modes=3).kept == 3
        assert jsa.schmidt_decomposition(f, grid, tol=0, max_modes=500).kept == grid.size

    def test_squeezing_and_coupling(self):
        grid, f = gaussian_jsa(1.0, 0.25)
        sm = jsa.schmidt_decomposition(f, grid, max_modes=5)
        r = sm.squeezing(1.2)
        assert r[0] == pytest.approx(1.2)
        assert np.allclose(r / r[0], sm.coefficients[:5] / sm.coefficients[0])
        Z = sm.coupling(1.2)
        assert np.abs(Z).sum() > 0
        assert np.allclose(Z, Z.T)
        for bad in (-0.1, math.nan):
            with pytest.raises(InputError):
                sm.squeezing(bad)
            with pytest.raises(InputError):
                sm.coupling(bad)


class TestSchmidtValidation:
    def test_type_ii_is_refused(self):
        grid = jsa.FrequencyGrid.centered(8, 2.0)
        w = grid.detunings
        ws, wi = np.meshgrid(w, w, indexing="ij")
        f = np.exp(-((ws + 2 * wi) ** 2))  # not symmetric
        with pytest.raises(InputError, match="type-II"):
            jsa.schmidt_decomposition(f, grid)

    def test_other_errors(self):
        grid = jsa.FrequencyGrid.centered(4, 1.0)
        with pytest.raises(InputError, match="match the grid"):
            jsa.schmidt_decomposition(np.eye(3), grid)
        with pytest.raises(InputError, match="zero"):
            jsa.schmidt_decomposition(np.zeros((4, 4)), grid)
        with pytest.raises(InputError, match="finite"):
            jsa.schmidt_decomposition(np.full((4, 4), np.nan), grid)
        with pytest.raises(InputError, match="tol"):
            jsa.schmidt_decomposition(np.eye(4), grid, tol=1.0)
        with pytest.raises(InputError, match="max_modes"):
            jsa.schmidt_decomposition(np.eye(4), grid, max_modes=0)

    def test_jsa_reaching_the_boundary_warns(self):
        grid, f = gaussian_jsa(1.0, 0.25, half=1.0)
        with pytest.warns(ApproximationWarning, match="boundary"):
            jsa.schmidt_decomposition(f, grid)


class TestCWLimit:
    def test_narrow_pump_gives_sideband_pairs(self):
        """A monochromatic pump couples Ω only with −Ω: J is anti-diagonal and the supermodes are
        degenerate pairs on (Ω, −Ω), with Schmidt coefficients ∝ |φ(Ω, −Ω)|."""
        grid = jsa.FrequencyGrid.centered(40, 3.0)
        pump = jsa.Pump.gaussian(1e-3 * grid.step)  # far narrower than the grid step
        pm = jsa.PhaseMatching.gaussian(gvm=0.3, gvd_signal=2.0)
        sm = jsa.schmidt_decomposition(jsa.joint_spectral_amplitude(pump, pm, grid), grid, tol=0, max_modes=40)
        w = grid.detunings
        phi = np.abs(pm.function(w, -w))
        expected = np.sort(phi)[::-1] / np.linalg.norm(phi)
        assert np.allclose(sm.coefficients, expected, atol=1e-14)
        assert all(len(g) % 2 == 0 for g in sm.groups)  # degenerate pairs (Ω, −Ω)
        # each kept pair lives on two sideband pixels ±Ω
        for k in range(0, 6, 2):
            support = np.flatnonzero(np.abs(sm.modes[:, k]) ** 2 + np.abs(sm.modes[:, k + 1]) ** 2 > 1e-20)
            assert len(support) == 2 and w[support[0]] == pytest.approx(-w[support[1]])
        # K = number of significant sideband pixels, weighted
        assert sm.schmidt_number == pytest.approx(np.sum(phi**2) ** 2 / np.sum(phi**4), rel=1e-12)

    def test_schmidt_number_grows_as_the_pump_narrows(self):
        pm = jsa.PhaseMatching.gaussian(gvm=0.0, gvd_signal=1.0)
        grid = jsa.FrequencyGrid.centered(161, 8.0)
        ks = [
            jsa.schmidt_decomposition(
                jsa.joint_spectral_amplitude(jsa.Pump.gaussian(fw), pm, grid), grid
            ).schmidt_number
            for fw in (2.0, 1.0, 0.5)
        ]
        assert ks[0] < ks[1] < ks[2]


# ------------------------------------------------------------------------------- pixel basis
def two_mode_generator(Z: np.ndarray, N: int) -> np.ndarray:
    a = fd.mode_operators(2, N)
    G = np.zeros_like(a[0])
    for j in range(2):
        for k in range(2):
            G += 0.5 * (np.conj(Z[j, k]) * a[j] @ a[k] - Z[j, k] * a[j].conj().T @ a[k].conj().T)
    return G


def test_oracle_squeezer_matches_fock_oracle():
    """The pixel Hamiltonian ↔ Z convention of the oracle, measured in a truncated Fock space."""
    Z = np.array([[0.25 + 0.1j, 0.15 - 0.05j], [0.15 - 0.05j, -0.1 + 0.2j]])
    S_fock, _ = fd.heisenberg_map_from_generator(two_mode_generator(Z, 30), 2, 30)
    assert np.allclose(S_fock, ref.multimode_squeezer(Z), atol=1e-9)


@FAST
@given(seed=st.integers(0, 2**32 - 1), r0=st.floats(0.0, 1.5))
def test_supermode_state_equals_pixel_squeezer(seed, r0):
    """Takagi route (supermodes squeezed in x, then the change of basis) = pixel-basis squeezer."""
    rng = np.random.default_rng(seed)
    n = 6
    grid = jsa.FrequencyGrid.centered(n, 1.0)
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    sm = jsa.schmidt_decomposition(X + X.T, grid, tol=0, edge_tol=1.0)  # a random matrix fills the grid
    out = sources.Waveguide(sm, loss_db_per_cm=0.0, r0=r0, output_coupling=1.0).output()
    S = ref.multimode_squeezer(sm.coupling(r0))
    assert np.allclose(out.in_pixel_basis().cov, S @ S.T / 2, atol=1e-11 * max(1.0, math.exp(2 * r0)))
    back = sy.change_basis(out.in_pixel_basis(), sm.unitary)
    assert back.reduce(range(sm.kept)).isclose(out.state, atol=1e-11 * math.exp(2 * r0))
    assert out.state.isclose(GaussianState.squeezed(sm.squeezing(r0)), atol=1e-12 * math.exp(2 * r0))


def test_supermodes_of_a_physical_spopo_crystal():
    """Default-like SPOPO crystal (2 nm pump at 397.5 nm, τ = 400 fs, β_s = 300 fs², β_p = 400 fs²):
    K is finite and converged under grid refinement, and the automatic grid agrees."""
    pump = jsa.Pump.gaussian(u.angular_bandwidth(2 * u.NM, 397.5 * u.NM))
    pm = jsa.PhaseMatching.sinc(gvm=400 * u.FS, gvd_signal=300 * u.FS**2, gvd_pump=400 * u.FS**2)
    ks = []
    for n in (240, 320):
        grid = jsa.FrequencyGrid.centered(n, 6.0e14)
        sm = jsa.schmidt_decomposition(jsa.joint_spectral_amplitude(pump, pm, grid), grid, max_modes=8)
        ks.append(sm.schmidt_number)
    assert 5 < ks[-1] < 100
    assert ks[0] == pytest.approx(ks[1], rel=1e-5)
    assert jsa.supermodes(pump, pm, max_modes=8).schmidt_number == pytest.approx(ks[1], rel=1e-3)


def test_supermodes_helper_uses_auto_grid():
    pump = jsa.Pump.gaussian(1.0)
    pm = jsa.PhaseMatching.gaussian(gvm=0.0, gvd_signal=2.0)
    sm = jsa.supermodes(pump, pm, max_modes=4)
    assert sm.kept == 4
    assert sm.grid.step <= 0.25 / 2 * (1 + 1e-12)
