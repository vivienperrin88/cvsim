"""MPO application (naive, zip-up, fit), environments, Krylov exponential and TCI."""

import numpy as np
import pytest
import scipy.linalg

from qttlind.bosons import operators as bops
from qttlind.bosons import states as bstates
from qttlind.core.environments import EnvironmentCache, extend_left, extend_right
from qttlind.core.krylov import expm_krylov
from qttlind.core.mpo import MPO, apply_mpo
from qttlind.core.tci import quantics_tci
from qttlind.core.tt import TensorTrain
from qttlind.errors import ConvergenceError, InputError, QttlindWarning, ShapeError


def random_tt(rng, L, chi, d=2):
    shapes = [(1 if i == 0 else chi, d, 1 if i == L - 1 else chi) for i in range(L)]
    return TensorTrain([rng.normal(size=s) + 1j * rng.normal(size=s) for s in shapes])


def random_mpo(rng, L, D, d=2, hermitian=False):
    W = MPO([rng.normal(size=(1 if i == 0 else D, d, d, 1 if i == L - 1 else D)) for i in range(L)])
    if hermitian:
        W, _ = (W + W.dagger()).compress(1e-28)
    return W


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ----------------------------------------------------------------------------- apply_mpo


@pytest.mark.parametrize("method", ["naive", "zipup", "fit"])
@pytest.mark.parametrize("tol", [0.0, 1e-10])
def test_apply_mpo_matches_the_dense_product(rng, method, tol):
    W, psi = random_mpo(rng, 6, 3), random_tt(rng, 6, 4)
    out, disc = apply_mpo(W, psi, method=method, tol=tol)
    assert disc >= 0
    assert rel(out.to_dense(), W.to_dense() @ psi.to_dense()) <= 10 * np.sqrt(tol) + 1e-12


@pytest.mark.parametrize("method", ["naive", "zipup", "fit"])
def test_apply_mpo_on_a_physical_state(method):
    out, _ = apply_mpo(bops.annihilation(8), bstates.coherent(2.0, 8), method=method, tol=1e-20)
    np.testing.assert_allclose(out.to_dense(), 2.0 * bstates.coherent(2.0, 8).to_dense(), atol=1e-8)


@pytest.mark.parametrize("method", ["naive", "zipup", "fit"])
def test_apply_mpo_respects_chi_max(rng, method):
    out, _ = apply_mpo(random_mpo(rng, 6, 3), random_tt(rng, 6, 4), method=method, chi_max=3)
    assert out.max_rank <= 3


def test_fit_is_at_least_as_accurate_as_zipup_at_fixed_rank(rng):
    W, psi = random_mpo(rng, 7, 3), random_tt(rng, 7, 4)
    exact = W.to_dense() @ psi.to_dense()
    e_zip = rel(apply_mpo(W, psi, method="zipup", chi_max=4)[0].to_dense(), exact)
    e_fit = rel(apply_mpo(W, psi, method="fit", chi_max=4)[0].to_dense(), exact)
    assert e_fit <= e_zip * (1 + 1e-9)


def test_discarded_weight_estimates_the_error(rng):
    W, psi = random_mpo(rng, 6, 3), random_tt(rng, 6, 4)
    out, d = apply_mpo(W, psi, method="zipup", tol=1e-3)
    err2 = rel(out.to_dense(), W.to_dense() @ psi.to_dense()) ** 2
    assert err2 <= 100 * d + 1e-20


def test_apply_mpo_does_not_modify_its_input(rng):
    W, psi = random_mpo(rng, 5, 2), random_tt(rng, 5, 3)
    before = psi.to_dense().copy()
    apply_mpo(W, psi, method="fit", tol=1e-8)
    np.testing.assert_array_equal(psi.to_dense(), before)


def test_apply_mpo_errors(rng):
    with pytest.raises(ShapeError):
        apply_mpo(random_mpo(rng, 4, 2), random_tt(rng, 5, 2))
    with pytest.raises(InputError):
        apply_mpo(random_mpo(rng, 4, 2), random_tt(rng, 4, 2), method="bogus")


def test_fit_budget_overrun(rng):
    W, psi = random_mpo(rng, 6, 3), random_tt(rng, 6, 4)
    with pytest.raises(ConvergenceError):
        apply_mpo(W, psi, method="fit", chi_max=2, max_sweeps=1, conv_tol=0.0)
    with pytest.warns(QttlindWarning):
        out, _ = apply_mpo(W, psi, method="fit", chi_max=2, max_sweeps=1, conv_tol=0.0, strict=False)
    assert out.max_rank <= 2


# ----------------------------------------------------------------------------- environments


def test_extend_left_and_right_match_einsum(rng):
    env = rng.normal(size=(2, 3, 4)) + 0j
    bra, op, ket = rng.normal(size=(2, 2, 5)), rng.normal(size=(3, 2, 2, 6)), rng.normal(size=(4, 2, 7))
    np.testing.assert_allclose(
        extend_left(env, bra, op, ket), np.einsum("abc,aox,boiy,ciz->xyz", env, bra.conj(), op, ket)
    )
    envr = rng.normal(size=(5, 6, 7)) + 0j
    np.testing.assert_allclose(
        extend_right(envr, bra, op, ket), np.einsum("xyz,aox,boiy,ciz->abc", envr, bra.conj(), op, ket)
    )


def test_cache_contracts_to_the_expectation_value_at_every_cut(rng):
    phi, W, psi = random_tt(rng, 6, 3), random_mpo(rng, 6, 2), random_tt(rng, 6, 3)
    exact = phi.inner(W.apply(psi))
    cache = EnvironmentCache(phi, W, psi)
    assert cache.value() == pytest.approx(exact)
    for k in range(6):
        t = np.einsum(
            "abc,aox,boiy,ciz,xyz->", cache.left(k), phi.cores[k].conj(), W.cores[k], psi.cores[k], cache.right(k)
        )
        assert t == pytest.approx(exact)


def test_cache_invalidation_after_an_update(rng):
    phi, W, psi = random_tt(rng, 6, 3), random_mpo(rng, 6, 2), random_tt(rng, 6, 3)
    cache = EnvironmentCache(phi, W, psi)
    cache.left(5), cache.right(0)
    new = rng.normal(size=phi.cores[2].shape) + 0j
    cache.update_bra(2, new)
    phi2 = TensorTrain(phi.cores[:2] + [new] + phi.cores[3:])
    fresh = EnvironmentCache(phi2, W, psi)
    for k in range(6):
        np.testing.assert_allclose(cache.left(k), fresh.left(k))
        np.testing.assert_allclose(cache.right(k), fresh.right(k))


def test_cache_validates_shapes(rng):
    with pytest.raises(ShapeError):
        EnvironmentCache(random_tt(rng, 5, 2), random_mpo(rng, 6, 2), random_tt(rng, 6, 2))


# ----------------------------------------------------------------------------- Krylov


@pytest.mark.parametrize("n,tau", [(50, 0.1), (120, 0.5), (200, 1.0)])
def test_expm_krylov_matches_expm(rng, n, tau):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / (2 * np.sqrt(n))
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    w, info = expm_krylov(lambda x: H @ x, v, -1j * tau, tol=1e-12, krylov_dim=60)
    np.testing.assert_allclose(w, scipy.linalg.expm(-1j * tau * H) @ v, atol=1e-9 * np.linalg.norm(v))
    assert np.linalg.norm(w) == pytest.approx(np.linalg.norm(v), rel=1e-10)
    assert info.dim >= 2


def test_krylov_dimension_grows_with_tau(rng):
    A = rng.normal(size=(80, 80))
    H = (A + A.T) / 10
    v = rng.normal(size=80) + 0j
    dims = [expm_krylov(lambda x: H @ x, v, -1j * t, krylov_dim=80)[1].dim for t in (0.05, 0.5, 2.0)]
    assert dims[0] < dims[1] < dims[2]


def test_krylov_non_hermitian_arnoldi(rng):
    A = rng.normal(size=(40, 40)) / 6
    v = rng.normal(size=40) + 0j
    w, _ = expm_krylov(lambda x: A @ x, v, 0.7, hermitian=False, krylov_dim=40)
    np.testing.assert_allclose(w, scipy.linalg.expm(0.7 * A) @ v, atol=1e-9)


def test_krylov_exact_on_an_invariant_subspace():
    H = np.diag([1.0, 2.0, 3.0])
    v = np.array([1.0, 0.0, 0.0]) + 0j
    w, info = expm_krylov(lambda x: H @ x, v, -1j)
    assert info.happy_breakdown and info.dim == 1
    np.testing.assert_allclose(w, [np.exp(-1j), 0, 0])


def test_krylov_budget(rng):
    A = rng.normal(size=(100, 100))
    H = A + A.T
    with pytest.raises(ConvergenceError):
        expm_krylov(lambda x: H @ x, rng.normal(size=100) + 0j, -5j, krylov_dim=2)


def test_krylov_keeps_the_shape(rng):
    v = rng.normal(size=(2, 3, 4)) + 0j
    w, _ = expm_krylov(lambda x: 2 * x, v, 0.5)
    np.testing.assert_allclose(w, np.exp(1.0) * v)


# ----------------------------------------------------------------------------- TCI


@pytest.mark.parametrize(
    "name,f",
    [
        ("sqrt", lambda n: np.sqrt(n.astype(float))),
        ("exp", lambda n: np.exp(-n / 100.0)),
        ("coherent", lambda n: bstates.coherent_amplitudes(np.sqrt(30.0), n)),
    ],
)
def test_tci_matches_tt_svd_with_few_evaluations(name, f):
    R = 12
    tt, info = quantics_tci(f, R, tol=1e-20)
    ref = f(np.arange(2**R))
    assert np.abs(tt.to_dense() - ref).max() <= 1e-8 * np.abs(ref).max()
    assert info.converged and info.n_evals < 2**R


def test_tci_reproduces_polynomials_exactly():
    tt, info = quantics_tci(lambda n: n.astype(float) ** 3 - 2.0 * n, 10, tol=1e-24)
    np.testing.assert_allclose(tt.to_dense(), np.arange(1024.0) ** 3 - 2 * np.arange(1024.0), rtol=1e-10, atol=1e-6)
    assert max(info.ranks) <= 4


def test_tci_at_forty_bits():
    bits = 40
    tt, info = quantics_tci(lambda n: np.sqrt(n.astype(float)), bits, tol=1e-20)
    assert info.n_evals < 20_000
    rng = np.random.default_rng(1)
    for n in rng.integers(0, 2**bits, 5):
        v = np.ones(1)
        for c, b in zip(tt.cores, [(int(n) >> (bits - 1 - k)) & 1 for k in range(bits)]):
            v = v @ c[:, b, :]
        assert v[0].real == pytest.approx(np.sqrt(float(n)), rel=1e-8)


def test_tci_is_reproducible():
    f = lambda n: bstates.coherent_amplitudes(4.0, n)  # noqa: E731
    a, _ = quantics_tci(f, 10, seed=3)
    b, _ = quantics_tci(f, 10, seed=3)
    np.testing.assert_array_equal(a.to_dense(), b.to_dense())


def test_tci_zero_function_and_validation():
    tt, _ = quantics_tci(lambda n: np.zeros(n.shape), 6)
    assert np.all(tt.to_dense() == 0)
    with pytest.raises(InputError):
        quantics_tci(lambda n: n, 0)
    with pytest.raises(InputError):
        quantics_tci(lambda n: np.ones(3), 6)
