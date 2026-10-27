"""Purified comb and Kraus integrators, validated against the dense oracle (qttlind.reference.dense_ref)."""

import warnings

import numpy as np
import pytest

from qttlind.bosons import operators as bops
from qttlind.bosons import states as bstates
from qttlind.errors import InputError, ShapeError, TraceDefectWarning
from qttlind.lindblad import LocalTerm, PurifiedComb, evolve, heff_terms, kraus_step, product_of_terms
from qttlind.reference import dense_ref as dr


def comp(m):
    return m.compress(1e-28)[0]


def rho_of(c: PurifiedComb) -> np.ndarray:
    P = c.to_dense()
    return P @ P.conj().T


@pytest.fixture
def mixed(rng):
    bits = [2, 3]
    P = rng.normal(size=(32, 3)) + 1j * rng.normal(size=(32, 3))
    return bits, P, PurifiedComb.from_dense(P, bits)


# ------------------------------------------------------------------ comb
def test_from_dense_round_trip(mixed):
    _, P, c = mixed
    np.testing.assert_allclose(c.to_dense(), P, atol=1e-12)
    assert c.bond_dims() == {"chi_e": 4, "chi_q": 8, "chi_mu": 3}


def test_from_dense_accepts_ket_and_checks_shape(rng):
    v = rng.normal(size=8)
    c = PurifiedComb.from_dense(v, [3])
    np.testing.assert_allclose(c.to_dense()[:, 0], v, atol=1e-12)
    with pytest.raises(ShapeError):
        PurifiedComb.from_dense(v, [2])


def test_product_state_is_pure_and_rank_one():
    c = PurifiedComb.product_state([bstates.coherent(0.5, 2), bstates.fock(1, 3)])
    assert c.bond_dims()["chi_e"] == 1 and c.chi_mu == 1
    want = np.kron(bstates.coherent(0.5, 2).to_dense(), dr.basis(8, 1))
    np.testing.assert_allclose(c.to_dense()[:, 0], want, atol=1e-13)
    assert c.purity() == pytest.approx(1.0)


def test_apply_local_matches_kronecker_and_keeps_backbone(mixed):
    _, P, c = mixed
    out = c.apply_local({1: bops.annihilation(3), 0: bops.number(2)})
    op = np.kron(np.diag(np.arange(4.0)), dr.destroy(8))
    np.testing.assert_allclose(out.to_dense(), op @ P, atol=1e-12)
    assert out.chi_mu == c.chi_mu
    assert out.tree.bond_dim(0, 1) == c.tree.bond_dim(0, 1)  # backbone untouched


def test_apply_local_validates(mixed):
    _, _, c = mixed
    with pytest.raises(InputError):
        c.apply_local({2: bops.number(2)})
    with pytest.raises(ShapeError):
        c.apply_local({0: bops.number(3)})


def test_vector_and_matrix_addition(mixed):
    _, P, c = mixed
    d = c.apply_local({1: bops.number(3)})
    Q = d.to_dense()
    np.testing.assert_allclose(c.vector_add(d).to_dense(), P + Q, atol=1e-12)
    np.testing.assert_allclose(c.matrix_add(d).to_dense(), np.hstack([P, Q]), atol=1e-12)
    with pytest.raises(ShapeError):
        c.vector_add(c.matrix_add(d))


def test_compress_keeps_state_and_reports_weight(mixed):
    _, _, c = mixed
    s = c.vector_add(c)
    t, disc = s.compress(1e-14)
    np.testing.assert_allclose(t.to_dense(), s.to_dense(), atol=1e-11)
    assert t.bond_dims()["chi_e"] <= c.bond_dims()["chi_e"]
    assert disc <= 1e-12


@pytest.mark.parametrize("tol", [0.0, 1e-3, 1e-1])
def test_truncate_mu_matches_oracle(mixed, tol):
    _, _, c = mixed
    m = c.matrix_add(c.apply_local({0: bops.number(2)}))
    t, disc = m.truncate_mu(tol)
    ref = dr.Purified(m.to_dense())
    dref = ref.truncate(tol)
    assert t.chi_mu == ref.chi_mu
    assert disc == pytest.approx(dref, abs=1e-12)
    np.testing.assert_allclose(rho_of(t), ref.rho(), atol=1e-10 * np.abs(ref.rho()).max())


def test_observables_match_dense(mixed):
    _, P, c = mixed
    rho = P @ P.conj().T
    r = rho / np.trace(rho)
    assert c.trace() == pytest.approx(np.trace(rho).real)
    A1 = np.kron(np.eye(4), dr.destroy(8))
    assert c.expect({1: bops.annihilation(3)}) == pytest.approx(np.trace(A1 @ r), abs=1e-12)
    n = c.normalize()
    assert n.trace() == pytest.approx(1.0)
    assert n.purity() == pytest.approx(np.trace(r @ r).real)
    np.testing.assert_allclose(np.sort(n.spectrum())[::-1][:3], np.sort(np.linalg.eigvalsh(r))[::-1][:3], atol=1e-12)
    r4 = r.reshape(4, 8, 4, 8)
    np.testing.assert_allclose(n.reduced_dm(0), np.einsum("iaja->ij", r4), atol=1e-12)
    np.testing.assert_allclose(n.reduced_dm(1), np.einsum("aiaj->ij", r4), atol=1e-12)
    want_n = [
        np.trace(np.kron(np.diag(np.arange(4.0)), np.eye(8)) @ r).real,
        np.trace(np.kron(np.eye(4), np.diag(np.arange(8.0))) @ r).real,
    ]
    np.testing.assert_allclose(n.mean_photons(), want_n, atol=1e-12)
    with pytest.raises(InputError):
        n.reduced_dm(5)


# ------------------------------------------------------------------ Kraus
R = 4
N = 2**R


@pytest.fixture(scope="module")
def cat_model():
    """Driven Kerr oscillator with two-photon loss L = sqrt(kappa)(a^2 - alpha^2), QTT and dense."""
    a = bops.annihilation(R)
    ad = a.dagger()
    a2 = comp(a @ a)
    kerr, kappa, alpha, eps = -0.3, 0.5, 1.2, 0.4
    H = [LocalTerm(kerr / 2, {0: comp(comp(ad @ ad) @ a2)}), LocalTerm(eps, {0: a}), LocalTerm(eps, {0: ad})]
    L = [[LocalTerm(np.sqrt(kappa), {0: a2}), LocalTerm(-np.sqrt(kappa) * alpha**2, {})]]
    A = dr.destroy(N)
    Ad = A.conj().T
    Hd = kerr / 2 * Ad @ Ad @ A @ A + eps * (A + Ad)
    Ld = [np.sqrt(kappa) * (A @ A - alpha**2 * np.eye(N))]
    c0 = PurifiedComb.product_state([bstates.coherent(0.7, R)])
    return H, L, Hd, Ld, c0


def test_heff_terms_match_dense(cat_model):
    H, L, Hd, Ld, _ = cat_model
    terms = heff_terms(H, L)
    dense = sum(t.value(0.0) * (t.ops[0].to_dense() if t.ops else np.eye(N)) for t in terms)
    np.testing.assert_allclose(dense, dr.heff(Hd, Ld), atol=1e-12)


def test_product_of_terms_time_dependent_coefficients():
    a = bops.annihilation(3)
    x = LocalTerm(lambda t: 2 * t, {0: a})
    y = LocalTerm(3.0, {0: a.dagger(), 1: a})
    p = product_of_terms(x, y)
    assert p.is_time_dependent and p.value(0.5) == pytest.approx(3.0)
    np.testing.assert_allclose(p.ops[0].to_dense(), dr.destroy(8) @ dr.destroy(8).T, atol=1e-12)
    assert set(p.ops) == {0, 1}


@pytest.mark.parametrize("order", [1, 2])
def test_kraus_step_matches_oracle(cat_model, order):
    H, L, Hd, Ld, c = cat_model
    ref = dr.Purified.from_ket(c.to_dense()[:, 0])
    Heff, Hed = heff_terms(H, L), dr.heff(Hd, Ld)
    for i in range(10):
        c, info = kraus_step(c, Heff, L, i * 0.01, 0.01, order=order, tol=1e-26)
        dr.kraus_step(ref, Hed, Ld, 0.01, order=order, rtol=1e-26)
    np.testing.assert_allclose(rho_of(c), ref.rho(), atol=1e-11)
    assert info.trace_defect < (1e-4 if order == 1 else 1e-6)
    assert c.trace() == pytest.approx(1.0)


@pytest.mark.parametrize(("order", "slope"), [(1, 1.0), (2, 2.0)])
def test_global_order_against_exact_lindblad(cat_model, order, slope):
    H, L, Hd, Ld, c = cat_model
    T = 0.4
    exact = dr.exact_rho(rho_of(c), Hd, Ld, T)
    errs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TraceDefectWarning)
        for h in (0.04, 0.02, 0.01):
            run = evolve(c, H, L, T, h, order=order, tol=1e-16)
            errs.append(np.abs(rho_of(run.final_state) - exact).sum())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    np.testing.assert_allclose(rates, slope, atol=0.15)


def test_two_mode_beam_splitter_with_loss():
    """Two modes coupled by a hopping term, loss on mode 1: a genuinely entangled, mixed comb."""
    bits = 3
    a = bops.annihilation(bits)
    ad = a.dagger()
    g, kappa = 0.6, 0.4
    H = [LocalTerm(g, {0: ad, 1: a}), LocalTerm(g, {0: a, 1: ad}), LocalTerm(0.2, {1: bops.number(bits)})]
    L = [[LocalTerm(np.sqrt(kappa), {1: a})]]
    c = PurifiedComb.product_state([bstates.coherent(1.0, bits), bstates.fock(0, bits)])
    A = dr.destroy(8)
    I = np.eye(8)
    Hd = g * (np.kron(A.T, A) + np.kron(A, A.T)) + 0.2 * np.kron(I, A.T @ A)
    Ld = [np.sqrt(kappa) * np.kron(I, A)]
    run = evolve(
        c, H, L, 0.5, 0.01, order=2, tol=1e-20, e_ops={"n1": [LocalTerm(1.0, {1: bops.number(bits)})]}, every=10
    )
    exact = dr.exact_rho(rho_of(c), Hd, Ld, 0.5)
    np.testing.assert_allclose(rho_of(run.final_state), exact, atol=2e-5)
    assert run.final_state.bond_dims()["chi_e"] > 1
    assert run.final_state.chi_mu > 1
    assert len(run.times) == 6 and run.times[-1] == pytest.approx(0.5)
    n1 = np.trace(np.kron(I, A.T @ A) @ exact).real
    assert run.expect["n1"][-1].real == pytest.approx(n1, abs=1e-4)
    assert np.all(np.diff(run.track["purity"]) <= 1e-12)  # loss only lowers the purity here


def test_damped_oscillator_decays_exponentially():
    bits = 4
    a = bops.annihilation(bits)
    kappa = 1.0
    c = PurifiedComb.product_state([bstates.coherent(1.5, bits)])
    run = evolve(
        c,
        [],
        [[LocalTerm(np.sqrt(kappa), {0: a})]],
        1.0,
        0.01,
        tol=1e-14,
        e_ops={"n": [LocalTerm(1.0, {0: bops.number(bits)})]},
    )
    n0 = run.expect["n"][0].real
    np.testing.assert_allclose(run.expect["n"].real, n0 * np.exp(-kappa * run.times), rtol=2e-5)
    # a coherent state stays coherent under pure loss: purity 1, chi_mu 1 after truncation
    assert run.final_state.purity() == pytest.approx(1.0, abs=1e-6)


def test_cat_dark_state_is_stationary():
    """|C+_alpha> is a dark state of L = a^2 - alpha^2: the Kraus map leaves it invariant.

    The explicit scheme is stable when h * max n(n-1) / 2 < ~1 on every level that can be
    populated (compression noise included): h = 0.005 on 2^4 levels.
    """
    bits, alpha = 4, 1.0
    a = bops.annihilation(bits)
    L = [[LocalTerm(1.0, {0: comp(a @ a)}), LocalTerm(-(alpha**2), {})]]
    c = PurifiedComb.product_state([bstates.cat(alpha, bits, parity=+1)])
    run = evolve(c, [], L, 0.2, 0.005, tol=1e-16)
    np.testing.assert_allclose(rho_of(run.final_state), rho_of(c), atol=1e-7)  # dark up to 3e-6 at 2^4 levels
    assert np.all(run.track["trace_defect"] < 1e-8)


def test_large_step_warns_about_trace_defect(cat_model):
    H, L, _, _, c = cat_model
    with pytest.warns(TraceDefectWarning):
        kraus_step(c, heff_terms(H, L), L, 0.0, 0.5, order=1)


def test_input_validation(cat_model):
    H, L, _, _, c = cat_model
    with pytest.raises(InputError):
        kraus_step(c, heff_terms(H, L), L, 0.0, 0.01, order=3)
    with pytest.raises(InputError):
        evolve(c, H, L, 1.0, 0.0)


def test_caps_bound_bond_dimensions(cat_model):
    H, L, _, _, c = cat_model
    run = evolve(c, H, L, 0.2, 0.02, tol=1e-16, caps={"chi_q": 3, "chi_mu": 2})
    assert run.final_state.bond_dims()["chi_q"] <= 3
    assert run.final_state.chi_mu <= 2


def test_unstable_step_is_detected():
    """Too large a step for the cutoff energies: the trace defect grows and is reported."""
    bits = 5
    a = bops.annihilation(bits)
    L = [[LocalTerm(1.0, {0: comp(a @ a)}), LocalTerm(-1.0, {})]]
    c = PurifiedComb.product_state([bstates.cat(1.0, bits, parity=+1)])
    with pytest.warns(TraceDefectWarning):
        evolve(c, [], L, 0.4, 0.02, tol=1e-16)
