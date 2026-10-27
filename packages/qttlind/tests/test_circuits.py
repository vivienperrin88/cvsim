"""CV modes and noise, circuits and executors, hardware models.

References: the Gaussian backend (exact for Gaussian channels), the dense Kraus oracle (same
scheme as the comb: implementation check), and analytic results (phase noise, Kerr, photon
subtraction of Dakna et al. 1997 / Ourjoumtsev et al. 2006).
"""

import math
import warnings

import numpy as np
import pytest
import scipy.linalg
from scipy import stats

from qttlind.bosons import states as bstates
from qttlind.circuits import (
    Circuit,
    CVMode,
    GaussianExecutor,
    HardwareModel,
    HybridExecutor,
    HybridState,
    QTTExecutor,
    SqueezingClippedWarning,
    expm_mpo,
    expm_terms,
    fit_loss,
    noise,
)
from qttlind.cv import gates as cvgates
from qttlind.errors import InputError
from qttlind.gaussian import GaussianState, ops
from qttlind.gaussian import hafnian as hf
from qttlind.gaussian.phase_space import omega
from qttlind.lindblad import LocalTerm, PurifiedComb
from qttlind.reference import dense_ref as dr


def comp(m):
    return m.compress(1e-24)[0]


def moments(comb: PurifiedComb, modes, positions=None):
    """Mean and covariance (xxpp) of the comb modes."""
    positions = list(range(len(modes))) if positions is None else positions
    M = len(modes)
    r = [(positions[j], modes[j].quadratures()[0]) for j in range(M)] + [
        (positions[j], modes[j].quadratures()[1]) for j in range(M)
    ]
    mean = np.array([comb.expect({j: o}).real for j, o in r])
    V = np.zeros((2 * M, 2 * M))
    for i, (ji, oi) in enumerate(r):
        for k, (jk, ok) in enumerate(r):
            ops_ = {ji: comp(oi @ ok)} if ji == jk else {ji: oi, jk: ok}
            V[i, k] = comb.expect(ops_).real - mean[i] * mean[k]
    return mean, (V + V.T) / 2


def dense_moments(rho, x, p):
    m = np.array([np.trace(rho @ x).real, np.trace(rho @ p).real])
    xp = np.trace(rho @ (x @ p + p @ x) / 2).real - m[0] * m[1]
    return m, np.array([[np.trace(rho @ x @ x).real - m[0] ** 2, xp], [xp, np.trace(rho @ p @ p).real - m[1] ** 2]])


SQ_COH = ops.displace(ops.squeeze(GaussianState.vacuum(1), 0.4), 0.6 - 0.3j)
# Position grid of 7 bits: its number operator reaches ~200, so loss kappa = 0.5 needs
# dt * kappa * n_max / 2 < 1 (the explicit-step stability limit, see qttlind.lindblad.kraus).
MODES_1 = {"fock": CVMode.fock("a", 7), "x": CVMode.position("a", 7)}
DT = 0.01


# =================================================================== modes and noise
@pytest.fixture(scope="module")
def loss_runs():
    """Loss kappa = 0.5 for t = 1.2 (dt = 0.01) in both bases, with the dense-oracle reference."""
    out = {}
    for basis, mode in MODES_1.items():
        psi = PurifiedComb.product_state([mode.state_from_gaussian(SQ_COH)])
        run = noise.evolve_lindblad(psi, [], noise.loss(mode, 0.5), 1.2, DT, tol=1e-14)
        A = mode.annihilation().to_dense()
        ref = dr.Purified.from_ket(psi.to_dense()[:, 0])
        Heff = dr.heff(np.zeros_like(A), [np.sqrt(0.5) * A])
        for _ in range(120):
            dr.kraus_step(ref, Heff, [np.sqrt(0.5) * A], DT, rtol=1e-14)
        out[basis] = (mode, psi, run, ref)
    return out


@pytest.mark.parametrize("basis", ["fock", "x"])
def test_loss_matches_oracle_and_gaussian_channel(loss_runs, basis):
    mode, _, run, ref = loss_runs[basis]
    m, V = moments(run.final_state, [mode])
    x, p = (q.to_dense() for q in mode.quadratures())
    md, Vd = dense_moments(ref.rho() / ref.trace(), x, p)
    np.testing.assert_allclose(m, md, atol=1e-7)  # implementation: same scheme as the oracle
    np.testing.assert_allclose(V, Vd, atol=1e-7)  # (both truncate at tol = 1e-14: 1e-7 in amplitude)
    assert run.track["trace_defect"].max() < 1e-6
    g = ops.loss(SQ_COH, math.exp(-0.6))
    np.testing.assert_allclose(m, g.mean, atol=5e-6)  # physics: exact channel, O(dt^2) scheme error
    np.testing.assert_allclose(V, g.cov, atol=5e-6)


def test_loss_converges_at_second_order(loss_runs):
    mode, psi, run, _ = loss_runs["fock"]
    g = ops.loss(SQ_COH, math.exp(-0.6))
    coarse = noise.evolve_lindblad(psi, [], noise.loss(mode, 0.5), 1.2, 2 * DT, tol=1e-14)
    e1 = np.abs(moments(coarse.final_state, [mode])[1] - g.cov).max()
    e2 = np.abs(moments(run.final_state, [mode])[1] - g.cov).max()
    assert 3.5 < e1 / e2 < 4.5


def test_unstable_step_in_position_space_is_reported():
    """8-bit grid: n reaches ~400, dt = 0.02 violates dt kappa n_max / 2 < 1 and the run blows up."""
    from qttlind.errors import TraceDefectWarning

    mode = CVMode.position("a", 8)
    psi = PurifiedComb.product_state([mode.state_from_gaussian(SQ_COH)])
    with pytest.warns(TraceDefectWarning):
        noise.evolve_lindblad(psi, [], noise.loss(mode, 0.5), 1.0, 0.02, tol=1e-14)


def test_cross_basis_agreement(loss_runs):
    """The position-space run, converted to Fock space, equals the Fock-space run."""
    mx, _, runx, _ = loss_runs["x"]
    _, _, runf, _ = loss_runs["fock"]
    rho_x = runx.final_state.reduced_dm(0)
    Phi = mx.fock_basis_matrix(40)
    rho_f_from_x = Phi.conj() @ rho_x @ Phi.T
    rho_f = runf.final_state.reduced_dm(0)[:40, :40]
    assert 0.5 * np.abs(np.linalg.eigvalsh(rho_f - rho_f_from_x)).sum() < 1e-6


def test_thermal_environment():
    """Loss into nbar = 0.3: equals the Gaussian thermal-loss channel, with an O(dt^2) scheme error."""
    mode = CVMode.fock("a", 7)
    psi = PurifiedComb.product_state([mode.state_from_gaussian(SQ_COH)])
    g = ops.loss(SQ_COH, math.exp(-0.6), nbar=0.3)
    errs = []
    for dt in (2 * DT, DT):
        run = noise.evolve_lindblad(psi, [], noise.thermal(mode, 0.5, 0.3), 1.2, dt, tol=1e-14)
        m, V = moments(run.final_state, [mode])
        errs.append(max(np.abs(m - g.mean).max(), np.abs(V - g.cov).max()))
    assert 3.5 < errs[0] / errs[1] < 4.5  # second order (2.1e-4 -> 5.1e-5 measured)
    assert errs[1] < 1e-4


def test_phase_noise_has_an_analytic_reference():
    """Gaussian channels cannot describe phase noise: the reference is <a>(t) = e^{-gamma t/2} <a>(0).

    Stability: gamma n_max^2 dt / 2 < 1 on 2^4 levels.
    """
    mode = CVMode.fock("a", 4)
    psi = PurifiedComb.product_state([bstates.coherent(1.5, 4)])
    a, n = mode.annihilation(), mode.number()
    errs = []
    for dt in (0.005, 0.0025):
        run = noise.evolve_lindblad(
            psi,
            [],
            noise.dephasing(mode, 0.4),
            1.0,
            dt,
            tol=1e-14,
            e_ops={"a": [LocalTerm(1.0, {0: a})], "n": [LocalTerm(1.0, {0: n})]},
        )
        errs.append(abs(run.expect["a"][-1] - run.expect["a"][0] * math.exp(-0.2)))
        n0 = run.expect["n"][0].real
        # every Kraus operator is diagonal: populations change only through the O(h^3) trace defect
        assert run.expect["n"][-1].real == pytest.approx(n0, abs=5e-3)
    assert n0 == pytest.approx(2.25, abs=1e-6)  # the truncated tail beyond n = 15 is ~1e-8
    assert 3.5 < errs[0] / errs[1] < 4.5  # second order (6.0e-4 -> 1.5e-4 measured)
    assert errs[1] < 3e-4


def test_kerr_hamiltonian_generates_the_kerr_gate():
    mode = CVMode.fock("a", 7)
    H = noise.kerr_hamiltonian(mode, 1.0)[0]
    U = expm_mpo(H.ops[0] * (-1j * math.pi / 2 * H.value(0)))
    psi = bstates.coherent(2.0, 7)
    out = U.apply(psi)
    ref = cvgates.kerr(psi, -math.pi / 2)
    assert abs(out.inner(ref)) ** 2 == pytest.approx(1.0, abs=1e-10)


# the grid check is conservative (it assumes n ~ 3 <n> + 10); the 1e-7 assertions below are the real test
@pytest.mark.filterwarnings("ignore::qttlind.errors.CutoffWarning")
def test_quadratic_hamiltonian_gives_the_symplectic_map():
    modes = [CVMode.position("a", 6), CVMode.position("b", 6)]
    bs = np.zeros((4, 4))
    bs[0, 3] = bs[3, 0] = 1.0  # x1 p2
    bs[1, 2] = bs[2, 1] = -1.0  # - x2 p1
    sq = np.zeros((4, 4))
    sq[0, 2] = sq[2, 0] = -1.0  # -(x1 p1 + p1 x1)/2
    Hm = 0.3 * bs + 0.2 * sq
    t = 1.0
    terms = noise.quadratic_hamiltonian(modes, Hm)
    s0 = GaussianState.product(GaussianState.coherent(0.5), GaussianState.vacuum(1))
    psi = PurifiedComb.product_state([modes[0].state_from_gaussian(s0.reduce([0])), modes[1].vacuum()])
    X = [LocalTerm(-1j * t * term.value(0), term.ops) for term in terms]
    out, _ = expm_terms(psi, X, tol=1e-20)
    want = ops.apply_symplectic(s0, scipy.linalg.expm(omega(2) @ Hm * t))
    m, V = moments(out, modes)
    np.testing.assert_allclose(m, want.mean, atol=1e-7)
    np.testing.assert_allclose(V, want.cov, atol=1e-7)


def test_mode_validation():
    with pytest.raises(InputError):
        CVMode("a", "x", 8)
    from qttlind.cv.grids import QuadratureGrid

    with pytest.raises(InputError):
        CVMode("a", "x", 8, QuadratureGrid(7))
    with pytest.raises(InputError):
        CVMode("a", "fock", 4, QuadratureGrid(4))
    with pytest.raises(InputError):
        CVMode("a", "momentum", 4)
    with pytest.raises(InputError):
        noise.loss(CVMode.fock("a", 3), -0.1)
    with pytest.raises(InputError):
        noise.quadratic_hamiltonian([CVMode.fock("a", 3)], np.array([[0, 1], [0, 0]]))
    with pytest.raises(InputError):
        CVMode.fock("a", 3).state_from_gaussian(GaussianState.thermal(0.5))


def test_states_from_gaussian_in_both_bases():
    for mode in MODES_1.values():
        psi = PurifiedComb.product_state([mode.state_from_gaussian(SQ_COH)])
        m, V = moments(psi, [mode])
        np.testing.assert_allclose(m, SQ_COH.mean, atol=1e-10)
        np.testing.assert_allclose(V, SQ_COH.cov, atol=1e-10)


# =================================================================== circuits and executors
def gaussian_circuit(q=None):
    c = Circuit(2).squeeze(0, 0.4).displace(0, 0.5).beamsplitter(0, 1, 0.3).loss(1, 0.8)
    if q is not None:
        c.measure_homodyne(1, postselect=q)
    return c


def test_circuit_is_data_and_validates():
    c = Circuit(2).squeeze(0, 0.5).kerr(1, 0.1)
    assert len(c) == 2 and not c.is_gaussian and c.instructions[0].is_gaussian
    assert "kerr" in repr(c)
    with pytest.raises(InputError):
        c.beamsplitter(0, 0, 0.1)
    with pytest.raises(InputError):
        c.squeeze(2, 0.1)
    with pytest.raises(InputError):
        c.loss(0, 1.5)
    with pytest.raises(InputError):
        c.measure_threshold(0, postselect=2)
    with pytest.raises(InputError):
        Circuit(0)
    with pytest.raises(TypeError):
        c.instructions[0].params["r"] = 1.0  # instructions are immutable


def test_gaussian_executor_refuses_non_gaussian():
    with pytest.raises(InputError, match="kerr"):
        GaussianExecutor().run(Circuit(1).squeeze(0, 0.2).kerr(0, 0.1))


def test_hybrid_stays_gaussian_on_gaussian_circuits():
    c = gaussian_circuit(q=0.2)
    g = GaussianExecutor().run(c)
    h = HybridExecutor([CVMode.fock("a", 5), CVMode.fock("b", 5)]).run(c)
    assert any("never left" in s for s in h.backend_log)
    np.testing.assert_array_equal(h.state.cov, g.state.cov)
    np.testing.assert_array_equal(h.state.mean, g.state.mean)
    assert h.probability == g.probability and h.outcomes == g.outcomes


@pytest.mark.parametrize("basis", ["fock", "x"])
def test_qtt_executor_matches_gaussian_moments(basis):
    modes = (
        [CVMode.fock("a", 6), CVMode.fock("b", 6)]
        if basis == "fock"
        else [CVMode.position("a", 6), CVMode.position("b", 6)]
    )
    q = 0.2 if basis == "fock" else 1 * modes[1].grid.dx  # a grid point
    c = gaussian_circuit(q)
    g = GaussianExecutor().run(c)
    r = QTTExecutor(modes).run(c)
    m, V = moments(r.state, modes)
    np.testing.assert_allclose(m, g.state.mean, atol=1e-8)
    np.testing.assert_allclose(V, g.state.cov, atol=1e-8)
    assert r.probability == pytest.approx(g.probability, rel=1e-8)


def test_two_mode_squeezing_on_the_comb():
    modes = [CVMode.fock("a", 5), CVMode.fock("b", 5)]
    c = Circuit(2).two_mode_squeeze(0, 1, 0.3, 0.4).rotate(1, 0.2)
    g = GaussianExecutor().run(c)
    r = QTTExecutor(modes).run(c)
    m, V = moments(r.state, modes)
    np.testing.assert_allclose(V, g.state.cov, atol=1e-8)
    np.testing.assert_allclose(m, g.state.mean, atol=1e-10)


@pytest.fixture(scope="module")
def kitten():
    r, th = 0.6, 0.3
    c = Circuit(2).squeeze(0, r).beamsplitter(0, 1, th).measure_fock(1, postselect=1)
    modes = [CVMode.fock("a", 6), CVMode.fock("b", 4)]
    return c, modes, r, th


def test_photon_subtraction_heralds_a_squeezed_single_photon(kitten):
    c, modes, r, th = kitten
    res = HybridExecutor(modes).run(c)
    g = ops.beamsplitter(ops.squeeze(GaussianState.vacuum(2), r), th)
    assert res.probability == pytest.approx(0.0292830694, abs=1e-8)
    assert res.probability == pytest.approx(hf.pattern_probability(g.reduce([1]), [1]), abs=1e-10)
    assert res.outcomes == {1: 1}
    assert any("moved to the comb" in s for s in res.backend_log)
    rp = math.atanh(math.cos(th) ** 2 * math.tanh(r))
    Nb = 256
    a = np.diag(np.sqrt(np.arange(1, Nb)), 1)
    target = scipy.linalg.expm((rp * a @ a - rp * a.T @ a.T) / 2)[:64, 1]
    rho = res.state.reduced_dm(0)
    assert np.vdot(target, rho @ target).real == pytest.approx(1.0, abs=1e-8)
    W0 = 2 / math.pi * res.state.expect({0: modes[0].parity()}).real
    assert pytest.approx(-2 / math.pi, abs=1e-6) == W0


def test_loss_after_heralding_fills_the_negativity(kitten):
    c, modes, _, _ = kitten
    res = HybridExecutor(modes).run(c.copy().loss(0, 0.8))
    W0 = 2 / math.pi * res.state.expect({0: modes[0].parity()}).real
    assert pytest.approx(-0.28971, abs=1e-4) == W0


def test_qtt_and_hybrid_agree_on_photon_subtraction(kitten):
    c, modes, _, _ = kitten
    h = HybridExecutor(modes).run(c)
    q = QTTExecutor(modes).run(c)
    assert q.probability == pytest.approx(h.probability, rel=1e-8)
    np.testing.assert_allclose(
        q.state.reduced_dm(0), h.state.reduced_dm(0), atol=1e-6
    )  # BS on a truncated 16-level tap vs exact Bargmann


def test_homodyne_sampling_ks():
    c = Circuit(1).squeeze(0, 0.5).rotate(0, 0.3).displace(0, 0.4).measure_homodyne(0, phi=0.7)
    a = GaussianExecutor().sample(c, 2000, np.random.default_rng(1))
    for mode in (CVMode.position("a", 8), CVMode.fock("a", 6)):
        b = QTTExecutor([mode]).sample(c, 2000, np.random.default_rng(2))
        assert stats.ks_2samp(a, b).pvalue > 1e-3
    h = HybridExecutor([CVMode.fock("a", 6)]).sample(c, 2000, np.random.default_rng(3))
    assert stats.ks_2samp(a, h).pvalue > 1e-3
    with pytest.raises(InputError):
        GaussianExecutor().sample(Circuit(1).squeeze(0, 0.1), 10)


def test_cubic_phase_shifts_momentum_by_gamma_x2():
    gamma = 0.1
    c = Circuit(1).squeeze(0, 0.3).displace(0, 0.4 + 0.2j)
    s0 = GaussianExecutor().run(c).state
    mode = CVMode.position("a", 8)
    res = HybridExecutor([mode]).run(c.copy().cubic_phase(0, gamma))
    x, p = mode.quadratures()
    x2 = s0.cov[0, 0] + s0.mean[0] ** 2
    assert res.state.expect({0: p}).real == pytest.approx(s0.mean[1] + gamma * x2, abs=1e-8)
    assert res.state.expect({0: x}).real == pytest.approx(s0.mean[0], abs=1e-10)


def test_hybrid_moves_only_correlated_modes():
    c = Circuit(3).squeeze(0, 0.3).beamsplitter(0, 1, 0.4).squeeze(2, 0.5).kerr(0, 0.05)
    res = HybridExecutor([CVMode.fock(n, 5) for n in "abc"]).run(c)
    assert isinstance(res.state, HybridState)
    assert res.state.gaussian_modes == (2,) and set(res.state.comb_modes) == {0, 1}
    np.testing.assert_allclose(res.state.gaussian.cov, ops.squeeze(GaussianState.vacuum(1), 0.5).cov)


@pytest.mark.parametrize("executor", ["gaussian", "qtt", "hybrid"])
def test_executors_share_one_interface(executor):
    c = Circuit(1).squeeze(0, 0.3, 0.2).rotate(0, 0.4).displace(0, 0.3 - 0.1j).loss(0, 0.7)
    ref = GaussianExecutor().run(c).state
    mode = CVMode.fock("a", 7)
    ex = {"gaussian": GaussianExecutor(), "qtt": QTTExecutor([mode]), "hybrid": HybridExecutor([mode])}[executor]
    res = ex.run(c)
    if isinstance(res.state, GaussianState):
        m, V = res.state.mean, res.state.cov
    else:
        m, V = moments(res.state, [mode])
    np.testing.assert_allclose(m, ref.mean, atol=1e-8)
    np.testing.assert_allclose(V, ref.cov, atol=1e-8)


def test_threshold_and_pnr_detection_against_hafnians():
    s = ops.squeeze(GaussianState.vacuum(1), 0.5)
    modes = [CVMode.fock("a", 6)]
    for n in (0, 2, 4):
        res = HybridExecutor(modes).run(Circuit(1).squeeze(0, 0.5).measure_fock(0, postselect=n))
        assert res.probability == pytest.approx(hf.pattern_probability(s, [n]), abs=1e-12)
    click = HybridExecutor(modes).run(Circuit(1).squeeze(0, 0.5).measure_threshold(0, postselect=1))
    assert click.probability == pytest.approx(hf.click_probability(s, [1]), abs=1e-12)


# =================================================================== hardware models
def test_compile_inserts_element_loss():
    c = Circuit(2).squeeze(0, 0.6).beamsplitter(0, 1, 0.4)
    model = HardwareModel(element_loss={"squeeze": 0.9, "beamsplitter": 0.95})
    cc = model.compile(c)
    assert [i.kind for i in cc] == ["squeeze", "loss", "beamsplitter", "loss", "loss"]
    hand = Circuit(2).squeeze(0, 0.6).loss(0, 0.9).beamsplitter(0, 1, 0.4).loss(0, 0.95).loss(1, 0.95)
    np.testing.assert_allclose(GaussianExecutor().run(cc).state.cov, GaussianExecutor().run(hand).state.cov, atol=1e-14)
    res = GaussianExecutor().run(model.compile(Circuit(1).squeeze(0, 1.0)))
    np.testing.assert_allclose(res.state.cov, ops.loss(GaussianState.squeezed_vacuum(1.0), 0.9).cov, atol=1e-12)


def test_detector_efficiency_is_loss_before_an_ideal_detector():
    eta_d = 0.7
    model = HardwareModel(detector_efficiency=eta_d)
    s = ops.loss(ops.squeeze(GaussianState.vacuum(1), 0.6), eta_d)
    for n in (0, 1, 2, 3):
        cc = model.compile(Circuit(1).squeeze(0, 0.6).measure_fock(0, postselect=n))
        assert cc.instructions[1].kind == "loss"
        res = HybridExecutor([CVMode.fock("a", 6)]).run(cc)
        assert res.probability == pytest.approx(hf.pattern_probability(s, [n]), abs=1e-10)


def test_dark_counts_make_vacuum_click():
    pd = 0.013
    cc = HardwareModel(dark_count_probability=pd).compile(Circuit(1).measure_threshold(0, postselect=1))
    res = HybridExecutor([CVMode.fock("a", 3)]).run(cc)
    assert res.probability == pytest.approx(pd, abs=1e-14)


def test_phase_noise_monte_carlo_and_dephasing_agree():
    sigma = 0.3
    rng = np.random.default_rng(11)
    thetas = rng.normal(0, sigma, 4000)
    samples = [np.exp(1j * th) for th in thetas]  # <a> of R(theta)|alpha=1> = e^{i theta}
    mc, se = np.mean(samples).real, np.std(np.real(samples)) / np.sqrt(len(samples))
    assert abs(mc - math.exp(-(sigma**2) / 2)) < 5 * se
    model = HardwareModel(phase_noise_std=sigma)
    cc = model.compile(Circuit(1).displace(0, 1.0))
    assert [i.kind for i in cc] == ["displace", "dephasing"]
    mode = CVMode.fock("a", 5)
    res = HybridExecutor([mode]).run(cc)
    a = res.state.expect({0: mode.annihilation()})
    # truncated coherent state on 32 levels: <a> = 1 up to 1e-12
    assert a.real == pytest.approx(math.exp(-(sigma**2) / 2), abs=1e-8)
    assert abs(a.imag) < 1e-12


def test_finite_squeezing_is_clipped():
    model = HardwareModel(max_squeezing_db=10.0)
    r15 = 15 / (20 * math.log10(math.e))
    with pytest.warns(SqueezingClippedWarning):
        cc = model.compile(Circuit(1).squeeze(0, r15))
    assert cc.instructions[0].params["r"] == pytest.approx(10 / (20 * math.log10(math.e)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SqueezingClippedWarning)
        model.compile(Circuit(1).squeeze(0, 0.5))


def test_fit_loss_recovers_eta():
    r, eta = 0.8, 0.83
    V = eta * math.exp(-2 * r) / 2 + (1 - eta) / 2
    assert fit_loss(HardwareModel(), r, V) == pytest.approx(eta, abs=1e-12)
    assert fit_loss(
        HardwareModel(detector_efficiency=0.9), r, 0.9 * eta * math.exp(-2 * r) / 2 + (1 - 0.9 * eta) / 2
    ) == pytest.approx(eta, abs=1e-12)
    with pytest.raises(InputError):
        fit_loss(None, r, 2.0)


def test_hardware_model_validation():
    with pytest.raises(InputError):
        HardwareModel(element_loss={"warp": 0.9})
    with pytest.raises(InputError):
        HardwareModel(detector_efficiency=1.2)
    with pytest.raises(InputError):
        HardwareModel(max_squeezing_db=-1)
