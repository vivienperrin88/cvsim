"""Self-tests of the dense reference oracle.

The oracle judges every later layer, so it must be obviously right first. Among the tests below:
a convergence-order test and a trace-defect scaling test.
"""

import numpy as np
import pytest

from qttlind.reference import dense_ref as dr

# ----------------------------------------------------------------------------- building blocks


def test_destroy_commutator_except_cutoff():
    N = 10
    a = dr.destroy(N)
    comm = a @ a.conj().T - a.conj().T @ a
    expected = np.eye(N)
    expected[-1, -1] = 1 - N  # truncation artefact at the last Fock state
    assert np.allclose(comm, expected)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5 + 0.5j])
def test_coherent_state_is_eigenvector(alpha):
    N = 60
    psi = dr.coherent(N, alpha)
    assert np.isclose(np.linalg.norm(psi), 1)
    assert np.allclose(dr.destroy(N) @ psi, alpha * psi, atol=1e-10)


def test_cat_parity():
    N = 40
    P = dr.parity_op(N)
    for parity in (+1, -1):
        c = dr.cat(N, 2.0, parity)
        assert np.isclose(np.vdot(c, P @ c).real, parity)


def test_embed_and_kron_kets():
    a = dr.destroy(3)
    A = dr.embed(a, 1, [2, 3])
    assert np.allclose(A, np.kron(np.eye(2), a))
    assert dr.kron(dr.basis(2, 1), dr.basis(3, 2)).shape == (6,)


# ----------------------------------------------------------------------------- purification


def _random_mixed(rng, D, p):
    Q, _ = np.linalg.qr(rng.normal(size=(D, len(p))) + 1j * rng.normal(size=(D, len(p))))
    return (Q * np.asarray(p)) @ Q.conj().T, Q * np.sqrt(np.asarray(p))


def test_purified_observables(rng):
    rho, Psi = _random_mixed(rng, 8, [0.6, 0.3, 0.1])
    st = dr.Purified(Psi)
    O = rng.normal(size=(8, 8))
    O = O + O.T
    assert np.allclose(st.rho(), rho)
    assert np.isclose(st.trace(), 1)
    assert np.isclose(st.purity(), np.trace(rho @ rho).real)
    assert np.isclose(st.expect(O), np.trace(rho @ O))


def test_vector_vs_matrix_addition(rng):
    P1 = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    P2 = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    r1, r2 = P1 @ P1.conj().T, P2 @ P2.conj().T
    mat = np.hstack([P1, P2])  # Eq. 45: concatenation on mu
    vec = P1 + P2  # Eq. 44: mu shared
    assert np.allclose(mat @ mat.conj().T, r1 + r2)
    assert not np.allclose(vec @ vec.conj().T, r1 + r2)


def test_truncation_discarded_weight_is_trace_norm_error(rng):
    rho, Psi = _random_mixed(rng, 10, [0.7, 0.2, 0.06, 0.03, 0.01])
    # random unitary gauge on mu, so the purification is not already diagonal
    G, _ = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
    st = dr.Purified(Psi @ G)
    disc = st.truncate(rtol=0.05)
    assert st.chi_mu == 3  # drops 0.03 + 0.01 = 0.04 <= 0.05
    assert np.isclose(disc, 0.04)
    err = np.abs(np.linalg.eigvalsh(rho - st.rho())).sum()
    assert np.isclose(err, disc)


def test_truncation_lossless_at_zero_tolerance(rng):
    rho, Psi = _random_mixed(rng, 6, [0.5, 0.3, 0.2])
    st = dr.Purified(np.hstack([Psi, np.zeros((6, 4))]))  # padded with exact zeros
    assert st.truncate(rtol=0.0) == 0.0
    assert st.chi_mu == 3
    assert np.allclose(st.rho(), rho)


def test_truncation_wide_purification_uses_qr_route(rng):
    D, chi = 4, 9
    Psi = rng.normal(size=(D, chi)) + 1j * rng.normal(size=(D, chi))
    st = dr.Purified(Psi.copy())
    rho = st.rho()
    st.truncate(rtol=0.0)
    assert st.chi_mu == D
    assert np.allclose(st.rho(), rho)


# ----------------------------------------------------------------------------- truncation rule


def _keep_brute_force(s, tol, chi_max):
    w = np.asarray(s) ** 2
    if w.size == 0:
        return 0
    if w.sum() == 0:
        return 1
    k = next(k for k in range(1, w.size + 1) if w[k:].sum() <= tol * w.sum())
    return min(k, max(1, chi_max)) if chi_max is not None else k


@pytest.mark.parametrize("trial", range(50))
def test_keep_count_matches_definition(rng, trial):
    n = int(rng.integers(1, 12))
    s = np.sort(rng.exponential(size=n) ** rng.uniform(1, 6))[::-1]
    tol = float(10 ** rng.uniform(-8, 0))
    cap = None if trial % 3 else int(rng.integers(0, n + 2))
    keep, disc = dr.keep_count(s, tol, cap)
    assert keep == _keep_brute_force(s, tol, cap)
    assert np.isclose(disc, (s[keep:] ** 2).sum() / (s**2).sum())


def test_keep_count_edge_cases():
    assert dr.keep_count([], 1e-8) == (0, 0.0)
    assert dr.keep_count([0.0, 0.0], 1e-8) == (1, 0.0)
    assert dr.keep_count([1.0, 1.0], 0.0, chi_max=0)[0] == 1


# ----------------------------------------------------------------------------- dynamics


def _kerr_model(N=12):
    a = dr.destroy(N)
    ad = a.conj().T
    H = 0.3 * ad @ a + 0.05 * ad @ ad @ a @ a + 0.4 * (a + ad)
    Ls = [np.sqrt(0.5) * a, np.sqrt(0.1) * ad @ a]
    return H, Ls, dr.coherent(N, 1.0)


def test_exact_rho_damped_oscillator():
    N, kappa, alpha, t = 30, 0.4, 2.0, 1.5
    a = dr.destroy(N)
    psi = dr.coherent(N, alpha)
    rho = dr.exact_rho(np.outer(psi, psi.conj()), a.conj().T @ a, [np.sqrt(kappa) * a], t)
    assert np.isclose(np.trace(rho).real, 1)
    assert np.isclose(np.trace(a.conj().T @ a @ rho).real, alpha**2 * np.exp(-kappa * t), rtol=1e-8)


@pytest.mark.parametrize("order,expected", [(1, 1.0), (2, 2.0)])
def test_kraus_convergence_order(order, expected):
    H, Ls, psi0 = _kerr_model()
    T = 1.0
    rex = dr.exact_rho(np.outer(psi0, psi0.conj()), H, Ls, T)
    He = dr.heff(H, Ls)
    hs = [0.02, 0.01, 0.005]
    errs = []
    for h in hs:
        st = dr.Purified.from_ket(psi0)
        for _ in range(round(T / h)):
            dr.kraus_step(st, He, Ls, h, order=order, rtol=1e-14)
        errs.append(np.linalg.norm(st.rho() - rex))
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert abs(slope - expected) < 0.15, (errs, slope)


def test_trace_defect_scaling():
    H, Ls, psi0 = _kerr_model()
    He = dr.heff(H, Ls)

    def defect(h):
        st = dr.Purified.from_ket(psi0)
        P = st.Psi
        P1 = He @ P
        X = P - 1j * h * P1
        blocks = [X - 0.5 * h * h * (He @ P1)]
        for L in Ls:
            Y = L @ P
            blocks += [np.sqrt(h / 2) * (Y - 1j * h * (He @ Y)), np.sqrt(h / 2) * (L @ X)]
        blocks += [h / np.sqrt(2) * (Lj @ (Lk @ P)) for Lj in Ls for Lk in Ls]
        return abs(sum(np.vdot(B, B).real for B in blocks) - 1)

    hs = [0.02, 0.01, 0.005]
    slope = np.polyfit(np.log(hs), np.log([defect(h) for h in hs]), 1)[0]
    assert slope > 2.7, slope


def test_kraus_step_keeps_a_density_matrix(rng):
    H, Ls, psi0 = _kerr_model()
    st = dr.Purified.from_ket(psi0)
    He = dr.heff(H, Ls)
    for _ in range(50):
        dr.kraus_step(st, He, Ls, 0.01)
    rho = st.rho()
    assert np.isclose(np.trace(rho).real, 1)
    assert np.allclose(rho, rho.conj().T)
    assert np.linalg.eigvalsh(rho).min() > -1e-12
    assert 1 < st.chi_mu < psi0.size


def test_evolve_time_dependent_matches_constant():
    H, Ls, psi0 = _kerr_model()
    r1 = dr.evolve(psi0, H, Ls, 0.5, 0.01, e_ops={"x": H}, every=50)
    r2 = dr.evolve(psi0, lambda t: H, Ls, 0.5, 0.01, e_ops={"x": H}, every=50)
    assert np.allclose(r1.expect["x"], r2.expect["x"])
    assert list(r1.times) == [0.0, 0.5]
