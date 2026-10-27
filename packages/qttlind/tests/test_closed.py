"""Closed-system integrators: time-dependent MPOs, RK4, Crank-Nicolson, TDVP, Magnus-TDVP."""

import numpy as np
import pytest
import scipy.linalg
from scipy.integrate import solve_ivp

from qttlind.bosons import operators as bops
from qttlind.bosons import states as bstates
from qttlind.closed import MagnusTDVP, TimeDependentMPO, evolve, tdvp_step
from qttlind.closed.effective import h_eff_bond, h_eff_one_site, h_eff_two_site
from qttlind.closed.linear_solve import als_solve
from qttlind.core.environments import EnvironmentCache
from qttlind.core.mpo import MPO
from qttlind.core.tt import TensorTrain
from qttlind.errors import InputError

K = 1 / 25


def comp(m):
    return m.compress(1e-28)[0]


def drive(t):
    return 0.5 * np.cos(1.3 * t)


@pytest.fixture(scope="module")
def kerr():
    """Driven Kerr oscillator (paper Eq. 32) on 2^5 levels and its dense DOP853 reference at t = 1."""
    R = 5
    a = bops.annihilation(R)
    ad = a.dagger()
    H0 = comp(bops.number(R) + comp(ad @ ad @ a @ a) * (K / 2))
    H1 = comp(a @ a + ad @ ad)
    Ht = TimeDependentMPO([(1.0, H0), (drive, H1)])
    psi0 = bstates.coherent(1.0, R)
    H0d, H1d = H0.to_dense(), H1.to_dense()
    sol = solve_ivp(
        lambda t, y: -1j * (H0d + drive(t) * H1d) @ y, (0, 1), psi0.to_dense(), method="DOP853", rtol=1e-13, atol=1e-13
    )
    return Ht, psi0, sol.y[:, -1]


def random_hermitian_mpo(rng, L=6, D=3):
    W = MPO([rng.normal(size=(1 if i == 0 else D, 2, 2, 1 if i == L - 1 else D)) for i in range(L)])
    H = comp(W + W.dagger())
    return H * (1 / np.linalg.norm(H.to_dense(), 2))


def full_rank_state(rng, L=6):
    v = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    return TensorTrain.from_dense(v / np.linalg.norm(v), [2] * L)


# ----------------------------------------------------------------------------- Hamiltonians


def test_time_dependent_mpo_sums_its_terms():
    a = bops.annihilation(4)
    n = bops.number(4)
    H = TimeDependentMPO([(2.0, n), (lambda t: np.sin(t), comp(a + a.dagger()))])
    t = 0.7
    expected = 2 * n.to_dense() + np.sin(t) * (a.to_dense() + a.dagger().to_dense())
    np.testing.assert_allclose(H.at(t).to_dense(), expected, atol=1e-10)
    assert H.coefficients(t)[1] == pytest.approx(np.sin(t))
    psi = bstates.coherent(0.8, 4)
    np.testing.assert_allclose(H.apply(t, psi).to_dense(), expected @ psi.to_dense(), atol=1e-10)
    with pytest.raises(InputError):
        TimeDependentMPO([(1.0, bops.number(3)), (1.0, bops.number(4))])
    with pytest.raises(InputError):
        TimeDependentMPO([])


# ----------------------------------------------------------------------------- effective Hamiltonians


def dense_projector(cores, k):
    """Isometry embedding core k (any physical dimension) into the full space, the other cores fixed."""
    left = np.ones((1, 1))
    for c in cores[:k]:
        left = np.einsum("al,lsr->asr", left, c).reshape(-1, c.shape[2])
    right = np.ones((1, 1))
    for c in reversed(cores[k + 1 :]):
        right = np.einsum("lsr,rb->lsb", c, right).reshape(c.shape[0], -1)
    shape = cores[k].shape
    cols = []
    for idx in np.ndindex(*shape):
        e = np.zeros(shape)
        e[idx] = 1
        cols.append(np.einsum("al,lsr,rb->asb", left, e, right).reshape(-1))
    return np.array(cols).T


def test_effective_hamiltonians_equal_the_projected_dense_operator(rng):
    H = random_hermitian_mpo(rng, L=5, D=2)
    psi = full_rank_state(rng, L=5).canonicalize(2)
    Hd = H.to_dense()
    cache = EnvironmentCache(psi, H, psi)
    k = 2
    P1 = dense_projector(psi.cores, k)
    x = rng.normal(size=psi.cores[k].shape) + 0j
    mv = h_eff_one_site(cache.left(k), H.cores[k], cache.right(k))
    np.testing.assert_allclose(mv(x).reshape(-1), (P1.conj().T @ Hd @ P1) @ x.reshape(-1), atol=1e-10)
    y = rng.normal(size=x.shape) + 0j
    assert np.vdot(y, mv(x)) == pytest.approx(np.conj(np.vdot(x, mv(y))))  # Hermitian
    psi2 = psi.canonicalize(1)
    c2 = EnvironmentCache(psi2, H, psi2)
    merged = np.einsum("lsr,rtq->lstq", psi2.cores[1], psi2.cores[2])
    P2 = dense_projector(psi2.cores[:1] + [merged.reshape(merged.shape[0], 4, merged.shape[3])] + psi2.cores[3:], 1)
    th = rng.normal(size=(psi2.cores[1].shape[0], 2, 2, psi2.cores[2].shape[2])) + 0j
    mv2 = h_eff_two_site(c2.left(1), H.cores[1], H.cores[2], c2.right(2))
    np.testing.assert_allclose(mv2(th).reshape(-1), (P2.conj().T @ Hd @ P2) @ th.reshape(-1), atol=1e-10)
    # bond matvec: H_eff on the bond between sites 1 and 2 = H2(theta) with theta = A C B for canonical A, B
    Lb, Rb = c2.left(2), c2.right(1)
    C = rng.normal(size=(Lb.shape[2], Rb.shape[2])) + 0j
    assert h_eff_bond(Lb, Rb)(C).shape == C.shape


# ----------------------------------------------------------------------------- TDVP


@pytest.mark.parametrize("h", [0.1, 0.5])
def test_tdvp_is_exact_when_the_manifold_is_the_whole_space(rng, h):
    """Lubich, Oseledets & Vandereycken (2015): full bond dimensions -> TDVP = exact evolution."""
    H = random_hermitian_mpo(rng)
    psi = full_rank_state(rng)
    out = tdvp_step(H, psi, h)
    ref = scipy.linalg.expm(-1j * h * H.to_dense()) @ psi.to_dense()
    np.testing.assert_allclose(out.to_dense(), ref, atol=1e-10)


def test_tdvp_error_decreases_with_the_bond_cap(rng):
    H = random_hermitian_mpo(rng)
    psi = full_rank_state(rng)
    ref = scipy.linalg.expm(-0.5j * H.to_dense()) @ psi.to_dense()
    errs = [np.linalg.norm(tdvp_step(H, psi, 0.5, chi_max=c).to_dense() - ref) for c in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_tdvp_conserves_norm_and_energy(rng):
    H = random_hermitian_mpo(rng)
    psi = full_rank_state(rng)
    E0 = psi.inner(H.apply(psi)).real
    for _ in range(5):
        psi = tdvp_step(H, psi, 0.2, tol=1e-20)
    assert psi.norm() == pytest.approx(1.0, abs=1e-9)
    assert psi.inner(H.apply(psi)).real == pytest.approx(E0, abs=1e-8)


def test_tdvp_single_site():
    H = MPO([np.diag([0.0, 1.0]).reshape(1, 2, 2, 1)])
    psi = TensorTrain([np.array([1.0, 1.0]).reshape(1, 2, 1) / np.sqrt(2)])
    out = tdvp_step(H, psi, 0.3)
    np.testing.assert_allclose(out.to_dense(), np.array([1, np.exp(-0.3j)]) / np.sqrt(2), atol=1e-12)


# ----------------------------------------------------------------------------- orders


@pytest.mark.parametrize("method,order", [("rk4", 4), ("crank_nicolson", 2), ("tdvp", 1), ("magnus_tdvp", 4)])
def test_global_convergence_orders(kerr, method, order):
    """Fig. 6b: RK4 and Magnus-TDVP are fourth order, Crank-Nicolson second, plain TDVP first."""
    Ht, psi0, ref = kerr
    errs = [
        np.linalg.norm(evolve(Ht, psi0, 1.0, h, method=method, tol=1e-28).final_state.to_dense() - ref)
        for h in (0.02, 0.01, 0.005)
    ]
    slopes = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(slopes - order) < 0.4), (method, errs, slopes)


def test_magnus_beats_rk4_and_builds_commutators_once(kerr):
    Ht, psi0, ref = kerr
    stepper = MagnusTDVP(Ht)
    assert stepper.commutator_count == 1  # only the (constant, driven) pair
    e_magnus = np.linalg.norm(evolve(Ht, psi0, 1.0, 0.02, step=stepper.step, tol=1e-28).final_state.to_dense() - ref)
    e_rk4 = np.linalg.norm(evolve(Ht, psi0, 1.0, 0.02, method="rk4", tol=1e-28).final_state.to_dense() - ref)
    assert e_magnus < e_rk4 / 100
    assert stepper.commutator_count == 1
    pair = MagnusTDVP.from_pair(Ht.terms[0][1], Ht.terms[1][1], drive)
    np.testing.assert_allclose(
        pair.magnus_hamiltonian(0.3, 0.01).to_dense(), stepper.magnus_hamiltonian(0.3, 0.01).to_dense(), atol=1e-12
    )


def test_evolve_records_observables():
    R = 5
    psi0 = bstates.coherent(1.5, R)
    n = bops.number(R)
    res = evolve(n, psi0, 1.0, 0.05, method="tdvp", e_ops={"n": n}, every=4)
    assert res.times.shape == res.expect["n"].shape == res.track["chi_q"].shape
    np.testing.assert_allclose(res.expect["n"].real, 2.25, atol=1e-8)  # <n> conserved by H = n
    np.testing.assert_allclose(res.track["norm"], 1.0, atol=1e-8)
    assert res.times[-1] == pytest.approx(1.0)
    with pytest.raises(InputError):
        evolve(n, psi0, 1.0, 0.1, method="euler")
    with pytest.raises(InputError):
        evolve(n, psi0, 1.0, -0.1)


def test_als_solve(rng):
    H = random_hermitian_mpo(rng)
    ident = MPO.identity([2] * 6)
    A = comp(ident + H * 0.3j)
    b = full_rank_state(rng)
    x = als_solve(A, b, b, tol=0.0)
    np.testing.assert_allclose(A.to_dense() @ x.to_dense(), b.to_dense(), atol=1e-9)
