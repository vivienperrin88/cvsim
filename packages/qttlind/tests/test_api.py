"""User API: operator algebra, System, state specifications, LindbladSolver, Result, evolve_closed."""

import math
import pickle
import warnings

import numpy as np
import pytest

import qttlind as ql
from qttlind.api.operators import Word, _wick
from qttlind.errors import CutoffWarning, InputError, ShapeError
from qttlind.reference import dense_ref as dr


def destroy(N):
    return dr.destroy(N)


# ------------------------------------------------------------------ operator algebra
@pytest.mark.parametrize(("p", "q", "r", "s"), [(0, 1, 1, 0), (1, 2, 2, 1), (2, 1, 0, 3), (1, 1, 1, 1), (0, 3, 2, 0)])
def test_wick_identity_against_dense_away_from_cutoff(p, q, r, s):
    N = 40
    A = destroy(N)
    Ad = A.T

    def word(p, q):
        return np.linalg.matrix_power(Ad, p) @ np.linalg.matrix_power(A, q)

    lhs = word(p, q) @ word(r, s)
    rhs = sum(c * word(w.p, w.q) for c, w in _wick(Word(p, q), Word(r, s)))
    k = N - 2 * max(p + r, q + s) - 2
    np.testing.assert_allclose(lhs[:k, :k], rhs[:k, :k], atol=1e-8)


def test_binomial_expansion_and_hashing():
    s = ql.System()
    a = s.add_mode("a", 4)
    lhs = (a + a.dag()) ** 2
    rhs = a.dag() ** 2 + 2 * a.dag() * a + a**2 + 1
    assert lhs == rhs and hash(lhs) == hash(rhs)
    assert {lhs: 1}[rhs] == 1
    assert a * a.dag() == a.dag() * a + 1  # normal ordering in the infinite space
    assert (a - a) == 0 and not (a - a)
    assert a.dag().dag() == a


def test_numpy_scalars_and_coefficient_types():
    s = ql.System()
    a = s.add_mode("a", 3)
    for x in (2, 2.0, 2.0 + 0j, np.float64(2.0), np.complex128(2.0), np.sqrt(4.0)):
        op = x * a
        assert isinstance(op, ql.Operator) and op == a + a
    assert (a * 3) / 3 == a
    with pytest.raises(TypeError):
        a * "x"
    with pytest.raises(InputError):
        a ** (-1)
    with pytest.raises(InputError):
        ql.coeff(2.0)


def test_time_dependent_coefficients():
    s = ql.System()
    a = s.add_mode("a", 3)

    def f(t):
        return np.exp(1j * t)

    H = ql.coeff(f) * a + (ql.coeff(f) * a).dag()
    assert H.is_time_dependent
    terms = H.to_local_terms(s)
    vals = sorted((t.value(0.3) for t in terms), key=lambda z: z.imag)
    np.testing.assert_allclose(vals, [np.exp(-0.3j), np.exp(0.3j)])
    # the conjugate wrapper is cached: H + H.dag() merges terms
    assert len((H + H.dag()).terms) == 2
    np.testing.assert_allclose(
        H.to_dense(s, 0.3), (np.exp(0.3j) * destroy(8) + np.exp(-0.3j) * destroy(8).T), atol=1e-12
    )


def test_two_mode_hamiltonian_matches_kronecker():
    """The cat-qubit stabilisation Hamiltonian on 2^3 x 2^2 levels."""
    s = ql.System()
    A = s.add_mode("memory", 3)
    B = s.add_mode("buffer", 2)
    g2, al, ez = 0.7, 1.3, 0.05
    H = g2 * A.dag() ** 2 * B + g2 * A**2 * B.dag() - g2 * al**2 * (B + B.dag()) + ez * (A + A.dag())
    a, b = destroy(8), destroy(4)
    Ia, Ib = np.eye(8), np.eye(4)
    ref = (
        g2 * np.kron(a.T @ a.T, b)
        + g2 * np.kron(a @ a, b.T)
        - g2 * al**2 * np.kron(Ia, b + b.T)
        + ez * np.kron(a + a.T, Ib)
    )
    np.testing.assert_allclose(H.to_dense(s), ref, atol=1e-12)


def test_parity_and_custom_basis():
    s = ql.System()
    a = s.add_mode("a", 3)
    E = np.array([0.0, 1.0, 2.5, 4.2, 6.0])
    n = np.diag(np.sqrt(np.arange(1, 5)), 1)
    nop = n + n.T
    basis = ql.CustomBasis(energies=E, ops={"n": nop})
    t = s.add_mode("t", 3, basis=basis)
    np.testing.assert_allclose(a.parity().to_dense(s), np.kron(np.diag((-1.0) ** np.arange(8)), np.eye(8)), atol=1e-12)
    H0 = np.zeros((8, 8))
    H0[:5, :5] = np.diag(E)
    np.testing.assert_allclose(t.H0().to_dense(s), np.kron(np.eye(8), H0), atol=1e-12)
    assert t.op_offsets("n") == [-1, 1]
    full = sum(t.op("n", offset=d) for d in t.op_offsets("n"))
    np.testing.assert_allclose(full.to_dense(s), t.op("n").to_dense(s), atol=1e-12)
    pad = np.zeros((8, 8))
    pad[:5, :5] = np.triu(nop, 1)
    np.testing.assert_allclose(t.op("n", offset=-1).to_dense(s), np.kron(np.eye(8), pad), atol=1e-12)
    with pytest.raises(InputError):
        a.H0()
    with pytest.raises(InputError):
        t.op("x")
    with pytest.raises(InputError):
        ql.CustomBasis(E, {"bad": np.eye(3)})


def test_mpo_cache_returns_same_object_and_clears():
    s = ql.System()
    a = s.add_mode("a", 4)
    t1 = (a.dag() ** 2 * a).to_local_terms(s)[0].ops[0]
    t2 = (a.dag() ** 2 * a).to_local_terms(s)[0].ops[0]
    assert t1 is t2
    ql.clear_mpo_cache()
    t3 = (a.dag() ** 2 * a).to_local_terms(s)[0].ops[0]
    assert t3 is not t1
    np.testing.assert_allclose(t3.to_dense(), t1.to_dense())


def test_repr_is_readable():
    s = ql.System()
    a = s.add_mode("a", 3)
    assert "a†^2" in repr(2 * a.dag() ** 2) and repr(a - a) == "0"
    assert "a: 2^3" in repr(s)


# ------------------------------------------------------------------ system and states
def test_system_validation():
    s = ql.System()
    s.add_mode("a", 3)
    with pytest.raises(InputError):
        s.add_mode("a", 3)
    with pytest.raises(InputError):
        s.add_mode("b", 0)
    with pytest.raises(InputError):
        s.add_mode("not valid", 2)
    with pytest.raises(InputError):
        s.product_state(z=ql.states.fock(0))
    with pytest.raises(InputError):
        s.product_state(a=ql.states.fock(8))
    with pytest.raises(InputError):
        ql.states.cat(1.0, parity=0)


def test_product_state_matches_dense():
    s = ql.System()
    s.add_mode("m", 4)
    s.add_mode("b", 2)
    st = s.product_state(m=ql.states.cat(1.2, parity=-1), b=ql.states.fock(1))
    want = np.kron(dr.cat(16, 1.2, -1), dr.basis(4, 1))
    want /= np.linalg.norm(want)
    np.testing.assert_allclose(st.to_dense()[:, 0], want, atol=1e-12)
    assert st.system is s
    st2 = s.product_state(m=ql.states.vector([1, 1j]))
    np.testing.assert_allclose(st2.to_dense()[[0, 4], 0], np.array([1, 1j]) / np.sqrt(2), atol=1e-12)  # m slow, b fast


def test_state_from_dense_pads_and_checks(rng):
    s = ql.System()
    s.add_mode("t", 2)
    s.add_mode("c", 3)
    v = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    st = s.state_from_dense(v, pad={"c": 8})
    full = np.zeros((4, 8), complex)
    full[:, :5] = v
    np.testing.assert_allclose(st.to_dense()[:, 0], full.reshape(-1) / np.linalg.norm(full), atol=1e-12)
    with pytest.raises(ShapeError):
        s.state_from_dense(np.ones((4, 9)))
    with pytest.raises(ShapeError):
        s.state_from_dense(np.ones((4, 5)), pad={"c": 16})
    with pytest.raises(InputError):
        s.state_from_dense(np.ones((4, 5)), pad={"x": 4})
    mixed = s.state_from_dense(rng.normal(size=(4, 8, 3)))
    assert mixed.chi_mu == 3


# ------------------------------------------------------------------ solver
@pytest.fixture
def damped():
    s = ql.System()
    a = s.add_mode("cav", 5)
    sol = ql.LindbladSolver(s, 1.0 * a.n(), [np.sqrt(0.2) * a], scheme="kraus2", dt=0.01, tol=1e-10)
    return s, a, sol


def test_damped_oscillator_through_solver(damped):
    s, a, sol = damped
    psi0 = s.product_state(cav=ql.states.coherent(2.0))
    r = sol.run(psi0, t_final=2.0, e_ops={"n": a.n(), "a": a}, every=10, track=["chi_mu", "purity", "reduced_dm:cav"])
    assert len(r.times) == 21 and r.times[-1] == pytest.approx(2.0)
    assert len(r.track["reduced_dm:cav"]) == 21 and r.track["reduced_dm:cav"][0].shape == (32, 32)
    np.testing.assert_allclose(r.expect["n"].real, 4 * np.exp(-0.2 * r.times), atol=1e-3)
    # agrees with the dense oracle running the same scheme much more tightly
    N = 32
    A = destroy(N)
    ref = dr.evolve(
        dr.coherent(N, 2.0),
        A.T @ A,
        [np.sqrt(0.2) * A],
        t_final=2.0,
        h=0.01,
        e_ops={"n": A.T @ A},
        every=10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(r.expect["n"].real, ref.expect["n"].real, atol=1e-6)
    np.testing.assert_allclose(np.abs(r.expect["a"]), 2 * np.exp(-0.1 * r.times), atol=1e-3)
    assert r.metadata["environment"]["numpy"] == np.__version__
    assert not r.interrupted


def test_cat_two_photon_loss_and_dephasing_matches_oracle():
    """Notebook 05: L1 = a^2 - alpha^2, L2 = sqrt(0.2) n on 2^5 levels."""
    alpha = 1.2
    s = ql.System()
    m = s.add_mode("cav", 5)
    sol = ql.LindbladSolver(s, 0 * m.n(), [m**2 - alpha**2, np.sqrt(0.2) * m.n()], dt=2e-3, tol=1e-12)
    r = sol.run(s.product_state(cav=ql.states.cat(alpha, parity=+1)), t_final=0.2, every=20, track=["purity"])
    N = 32
    A = destroy(N)
    ref = dr.evolve(
        dr.cat(N, alpha, +1),
        np.zeros((N, N)),
        [A @ A - alpha**2 * np.eye(N), np.sqrt(0.2) * A.T @ A],
        t_final=0.2,
        h=2e-3,
        every=20,
        rtol=1e-12,
    )
    np.testing.assert_allclose(r.track["purity"], ref.purity, atol=1e-8)


def test_keyboard_interrupt_returns_partial_result(damped):
    s, _, sol = damped

    def stop(step, t, info):
        if step == 7:
            raise KeyboardInterrupt

    r = sol.run(s.product_state(cav=ql.states.coherent(1.0)), t_final=1.0, every=2, progress=stop)
    assert r.interrupted and r.metadata["steps_done"] == 7
    assert r.times[-1] == pytest.approx(0.06)


def test_checkpoint_and_resume_bit_for_bit(damped, tmp_path):
    s, a, sol = damped
    psi0 = s.product_state(cav=ql.states.coherent(1.5))
    full = sol.run(psi0, t_final=0.5, e_ops={"n": a.n()}, every=5, track=["chi_mu"])
    path = tmp_path / "run.npz"

    def stop(step, t, info):
        if step == 23:
            raise KeyboardInterrupt

    part = sol.run(
        psi0,
        t_final=0.5,
        e_ops={"n": a.n()},
        every=5,
        track=["chi_mu"],
        checkpoint=str(path),
        checkpoint_every=10,
        progress=stop,
    )
    assert part.interrupted
    solver2, res = ql.LindbladSolver.resume(str(path))  # constant coefficients: the solver pickles
    assert isinstance(solver2, ql.LindbladSolver)
    np.testing.assert_array_equal(res.times, full.times)
    np.testing.assert_array_equal(res.expect["n"], full.expect["n"])
    np.testing.assert_array_equal(res.final_state.to_dense(), full.final_state.to_dense())


def test_resume_needs_solver_when_not_picklable(tmp_path):
    s = ql.System()
    a = s.add_mode("cav", 4)
    sol = ql.LindbladSolver(s, ql.coeff(lambda t: np.cos(t)) * (a + a.dag()), [0.3 * a], dt=0.01)
    psi0 = s.product_state()
    path = str(tmp_path / "cp.npz")
    short = sol.run(psi0, 0.1, e_ops={"n": a.n()}, every=4, checkpoint=path, checkpoint_every=5)
    with pytest.raises(InputError):
        ql.LindbladSolver.resume(path)
    with pytest.raises(InputError):
        ql.LindbladSolver.resume(path, sol, e_ops={})
    # the checkpoint holds t_final = 0.1: resuming a finished run returns it unchanged
    _, res = ql.LindbladSolver.resume(path, sol, e_ops={"n": a.n()})
    np.testing.assert_array_equal(res.expect["n"], short.expect["n"])
    np.testing.assert_array_equal(res.times, short.times)


def test_cutoff_warning():
    s = ql.System()
    a = s.add_mode("a", 5)
    sol = ql.LindbladSolver(s, 0 * a.n(), [0.1 * a], dt=0.01)
    with pytest.warns(CutoffWarning):
        sol.run(s.product_state(a=ql.states.coherent(np.sqrt(40))), t_final=0.01)
    s2 = ql.System()
    b = s2.add_mode("a", 8)
    sol2 = ql.LindbladSolver(s2, 0 * b.n(), [0.1 * b], dt=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CutoffWarning)
        sol2.run(s2.product_state(a=ql.states.coherent(np.sqrt(40))), t_final=0.01)


def test_result_save_load_and_pandas(damped, tmp_path):
    s, a, sol = damped
    r = sol.run(
        s.product_state(cav=ql.states.coherent(1.0)),
        t_final=0.1,
        e_ops={"n": a.n()},
        every=5,
        track=["chi_mu", "reduced_dm:cav"],
    )
    path = str(tmp_path / "res.npz")
    r.save(path)
    back = ql.Result.load(path)
    np.testing.assert_array_equal(back.times, r.times)
    np.testing.assert_array_equal(back.expect["n"], r.expect["n"])
    np.testing.assert_array_equal(back.track["chi_mu"], r.track["chi_mu"])
    np.testing.assert_array_equal(back.track["reduced_dm:cav"][1], r.track["reduced_dm:cav"][1])
    np.testing.assert_array_equal(back.final_state.to_dense(), r.final_state.to_dense())
    assert back.metadata["parameters"]["dt"] == 0.01 and "qttlind" in back.metadata["environment"]
    pd = pytest.importorskip("pandas")
    df = r.to_pandas()
    assert isinstance(df, pd.DataFrame) and list(df.columns) == ["n", "chi_mu"] and len(df) == 3


def test_solver_validation(damped):
    s, a, _ = damped
    other = ql.System().add_mode("x", 2)
    with pytest.raises(InputError):
        ql.LindbladSolver(s, a.n(), [a], dt=0.0)
    with pytest.raises(InputError):
        ql.LindbladSolver(s, a.n(), [a], dt=0.1, scheme="rk4")
    with pytest.raises(InputError):
        ql.LindbladSolver(s, other.n(), [a], dt=0.1)
    with pytest.raises(InputError):
        ql.LindbladSolver(s, a.n(), [a], dt=0.1, chi_max={"chi_x": 2})
    sol = ql.LindbladSolver(s, a.n(), [a], dt=0.1, chi_max=4)
    with pytest.raises(InputError):
        sol.run(s.product_state(), 1.0, track=["nonsense"])
    with pytest.raises(InputError):
        sol.run(s.product_state(), 1.0, track=["reduced_dm:nope"])
    with pytest.raises(InputError):
        sol.run(s.product_state().to_dense(), 1.0)


def test_kraus1_scheme_is_first_order():
    s = ql.System()
    a = s.add_mode("a", 4)
    errs = []
    for dt in (0.02, 0.01):
        sol = ql.LindbladSolver(s, 0.5 * (a + a.dag()), [np.sqrt(0.5) * a], scheme="kraus1", dt=dt, tol=1e-14)
        r = sol.run(s.product_state(a=ql.states.coherent(0.5)), t_final=0.4, e_ops={"n": a.n()})
        errs.append(r.final_state)
    A = destroy(16)
    rho0 = np.outer(dr.coherent(16, 0.5), dr.coherent(16, 0.5).conj())
    rho0 /= np.trace(rho0)
    ex = dr.exact_rho(rho0, 0.5 * (A + A.T), [np.sqrt(0.5) * A], 0.4)
    e = [np.abs(st.to_dense() @ st.to_dense().conj().T - ex).sum() for st in errs]
    assert math.log2(e[0] / e[1]) == pytest.approx(1.0, abs=0.15)


# ------------------------------------------------------------------ closed systems
def test_evolve_closed_matches_dense():
    s = ql.System()
    m = s.add_mode("osc", 5)

    def drive(t):
        return 0.5 * np.cos(1.3 * t)

    H = m.n() + 0.02 * m.dag() ** 2 * m**2 + ql.coeff(drive) * (m**2 + m.dag() ** 2)
    psi0 = s.product_state(osc=ql.states.coherent(1.0))
    from scipy.integrate import solve_ivp

    H0 = (m.n() + 0.02 * m.dag() ** 2 * m**2).to_dense(s)
    H1 = (m**2 + m.dag() ** 2).to_dense(s)
    sol = solve_ivp(
        lambda t, y: -1j * (H0 + drive(t) * H1) @ y,
        (0, 1),
        psi0.to_dense()[:, 0].astype(complex),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
    )
    ref = sol.y[:, -1]
    for method in ("rk4", "magnus_tdvp"):
        r = ql.evolve_closed(psi0, H, t_final=1.0, dt=0.01, method=method, tol=1e-14, every=10, e_ops={"n": m.n()})
        psi = r.final_state.to_dense()
        assert 1 - abs(np.vdot(ref, psi)) < 1e-6
        assert len(r.track["chi_q"]) == len(r.times) == 11
        assert r.expect["n"][-1].real == pytest.approx(np.vdot(ref, (m.n().to_dense(s)) @ ref).real, abs=1e-5)


def test_evolve_closed_two_modes_and_errors():
    s = ql.System()
    a = s.add_mode("a", 3)
    b = s.add_mode("b", 2)
    H = a.dag() * b + a * b.dag() + 0.3 * a.n()
    psi0 = s.product_state(a=ql.states.fock(1))
    r = ql.evolve_closed(psi0, H, t_final=0.5, dt=0.01, method="rk4", tol=1e-14)
    import scipy.linalg

    ref = scipy.linalg.expm(-0.5j * H.to_dense(s)) @ psi0.to_dense()[:, 0]
    np.testing.assert_allclose(r.final_state.to_dense(), ref, atol=1e-7)
    with pytest.raises(InputError):
        ql.evolve_closed(psi0.to_dense(), H, t_final=1, dt=0.1)
    mixed = s.state_from_dense(np.ones((8, 4, 2)))
    with pytest.raises(InputError):
        ql.evolve_closed(mixed, H, t_final=1, dt=0.1)
    with pytest.raises(InputError):
        ql.evolve_closed(psi0, 0 * a, t_final=1, dt=0.1)


def test_operators_pickle_without_lambdas():
    s = ql.System()
    a = s.add_mode("a", 3)
    op = 2 * a.dag() * a + a.parity()
    back = pickle.loads(pickle.dumps(op))
    assert back == op
