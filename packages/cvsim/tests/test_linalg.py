"""cvsim.core.linalg: the LAPACK seam and the symplectic decompositions.

Decompositions are checked by their defining identities (reconstruction, symplecticity,
orthogonality) on Hypothesis-generated inputs, including exactly degenerate and ill-conditioned
spectra, and by the gauge conventions of CONVENTIONS.md ("Decompositions"): a result must not
depend on the basis LAPACK returns inside a degenerate group, and a sweep with ``reference=``
must be continuous.

Tolerances: identities hold to round-off times the conditioning of the input, ``c = ‖S‖²``
(≈ e^{2 r_max}). A stress run over 3000 random and degenerate inputs (up to 6 modes, r up to 3)
gave worst relative errors of 2e-12 (Bloch–Messiah), 3e-12 (orthogonality of O1, O2), 8e-12
(Williamson), 5e-13 (Takagi); the tests allow 1e-11 · c or 1e-10 · c. Exact constructions
(identity references, already-diagonal inputs) are checked to 1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.core import linalg as la
from cvsim.errors import ConvergenceError, InputError, ShapeError

SETTINGS = settings(max_examples=40, deadline=None)


def random_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_symplectic(rng: np.random.Generator, M: int, r_max: float = 1.5) -> np.ndarray:
    """Passive · squeezing · passive with squeezing parameters in [0, r_max] (Bloch–Messiah form)."""
    r = rng.uniform(0, r_max, M)
    Z = np.diag(np.exp(np.concatenate([-r, r])))
    return cv.interferometer(random_unitary(rng, M)) @ Z @ cv.interferometer(random_unitary(rng, M))


def degenerate_symplectic(rng: np.random.Generator, rs: list[float]) -> np.ndarray:
    rs_ = np.asarray(rs, dtype=float)
    M = rs_.size
    Z = np.diag(np.exp(np.concatenate([-rs_, rs_])))
    return cv.interferometer(random_unitary(rng, M)) @ Z @ cv.interferometer(random_unitary(rng, M))


def bm_reconstruct(bm: la.BlochMessiah) -> np.ndarray:
    return bm.O1 @ np.diag(np.exp(np.concatenate([-bm.r, bm.r]))) @ bm.O2


def assert_orthosymplectic(O: np.ndarray, atol: float = 1e-12) -> None:
    assert np.allclose(O @ O.T, np.eye(O.shape[0]), atol=atol)
    assert cv.is_symplectic(O, atol)


seeds = st.integers(0, 2**32 - 1)


# ------------------------------------------------------------------------------- kernels
class TestKernels:
    def test_eigh(self, rng):
        A = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        A = A + A.conj().T
        w, U = la.eigh(A)
        assert np.all(np.diff(w) >= 0)
        assert np.allclose((U * w) @ U.conj().T, A, atol=1e-12)
        assert np.allclose(la.eigvalsh(A), w)

    def test_eigvals_general(self, rng):
        A = rng.normal(size=(5, 5))
        lam = la.eigvals(A)
        assert lam.dtype == complex and np.isclose(np.prod(lam), np.linalg.det(A))
        assert np.allclose(np.sort_complex(lam), np.sort_complex(np.linalg.eigvals(A)))
        with pytest.raises(ShapeError):
            la.eigvals(np.ones((2, 3)))

    @pytest.mark.parametrize(
        ("A", "err"),
        [
            (np.array([[1.0, 2.0], [0.0, 1.0]]), InputError),
            (np.ones((2, 3)), ShapeError),
            (np.ones(3), ShapeError),
            (np.array([[np.nan, 0], [0, 1]]), InputError),
        ],
    )
    def test_eigh_validates(self, A, err):
        with pytest.raises(err):
            la.eigh(A)

    def test_svd(self, rng):
        A = rng.normal(size=(4, 6))
        U, s, Vh = la.svd(A)
        assert np.allclose((U * s) @ Vh, A) and np.all(np.diff(s) <= 0)
        U, s, Vh = la.svd(A, full_matrices=True)
        assert U.shape == (4, 4) and Vh.shape == (6, 6)

    def test_svd_falls_back_to_gesvd(self, rng, monkeypatch):
        calls = []
        real = scipy.linalg.svd

        def flaky(a, full_matrices=True, lapack_driver="gesdd", check_finite=True):
            calls.append(lapack_driver)
            if lapack_driver == "gesdd":
                raise np.linalg.LinAlgError("SVD did not converge")
            return real(a, full_matrices=full_matrices, lapack_driver=lapack_driver, check_finite=check_finite)

        monkeypatch.setattr(la.scipy.linalg, "svd", flaky)
        A = rng.normal(size=(3, 3))
        U, s, Vh = la.svd(A)
        assert calls == ["gesdd", "gesvd"] and np.allclose((U * s) @ Vh, A)

    def test_svd_reports_double_failure(self, monkeypatch):
        def broken(*args, **kwargs):
            raise np.linalg.LinAlgError("no")

        monkeypatch.setattr(la.scipy.linalg, "svd", broken)
        with pytest.raises(ConvergenceError, match="gesdd nor gesvd"):
            la.svd(np.eye(2))

    @pytest.mark.parametrize(("A", "err"), [(np.ones(3), ShapeError), (np.array([[np.inf]]), InputError)])
    def test_svd_validates(self, A, err):
        with pytest.raises(err):
            la.svd(A)

    def test_solve_and_inv(self, rng):
        A = rng.normal(size=(4, 4)) + 4 * np.eye(4)
        b = rng.normal(size=4)
        assert np.allclose(A @ la.solve(A, b), b)
        assert np.allclose(la.inv(A) @ A, np.eye(4))
        with pytest.raises(InputError, match="singular"):
            la.solve(np.zeros((2, 2)), np.ones(2))
        with pytest.raises(InputError, match="singular"):
            la.inv(np.zeros((2, 2)))

    def test_cholesky_and_logdet(self, rng):
        B = rng.normal(size=(5, 5))
        A = B @ B.T + np.eye(5)
        L = la.cholesky(A)
        assert np.allclose(L @ L.T, A)
        assert la.logdet_pd(A) == pytest.approx(np.linalg.slogdet(A)[1], rel=1e-12)
        with pytest.raises(InputError, match="positive definite"):
            la.cholesky(-np.eye(2))

    def test_logdet_does_not_overflow(self):
        A = np.diag(np.full(400, 1e3))
        assert la.logdet_pd(A) == pytest.approx(400 * math.log(1e3))

    def test_sqrtm_psd(self, rng):
        B = rng.normal(size=(4, 2))
        A = B @ B.T  # rank 2: zero eigenvalues up to round-off
        R = la.sqrtm_psd(A)
        assert np.allclose(R @ R, A, atol=1e-12) and np.allclose(R, R.T)
        with pytest.raises(InputError, match="not positive semidefinite"):
            la.sqrtm_psd(np.diag([1.0, -0.1]))

    def test_expm(self):
        th = 0.4
        G = np.array([[0.0, -th], [th, 0.0]])
        assert np.allclose(la.expm(G), [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])

    def test_qr(self, rng):
        A = rng.normal(size=(5, 3))
        Q, R = la.qr(A)
        assert np.allclose(Q @ R, A) and np.allclose(Q.T @ Q, np.eye(3))
        with pytest.raises(ShapeError):
            la.qr(np.ones(3))

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_polar(self, rng, side):
        A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        U, P = la.polar(A, side)
        assert np.allclose(U @ U.conj().T, np.eye(4))
        assert np.all(la.eigvalsh(P) >= -1e-12)
        assert np.allclose(P @ U if side == "left" else U @ P, A)

    def test_polar_side_is_validated(self):
        with pytest.raises(InputError, match="side"):
            la.polar(np.eye(2), "up")

    def test_timed_counts_kernels_and_nests(self, rng):
        A = rng.normal(size=(3, 3)) + 3 * np.eye(3)
        with la.timed() as outer:
            la.inv(A)
            with la.timed() as inner:
                la.eigvalsh(A + A.T)
                la.svd(A)
        assert inner.calls == 2 and inner.by_kernel == {"eigvalsh": 1, "svd": 1}
        assert outer.calls == 3 and outer.by_kernel["inv"] == 1 and outer.seconds >= inner.seconds >= 0

    def test_degenerate_groups(self):
        assert [g.tolist() for g in la.degenerate_groups([3.0, 3.0, 1.0, 1e-9, 0.0])] == [[0, 1], [2], [3, 4]]
        assert [g.tolist() for g in la.degenerate_groups([1.0, 1.0001], rtol=1e-3)] == [[0, 1]]
        assert la.degenerate_groups([]) == []


# ------------------------------------------------------------------------------- symplectic eigenvalues and Williamson
def _random_covariance(rng, M, nu, r_max=1.5):
    S = random_symplectic(rng, M, r_max)
    return S @ np.diag(np.concatenate([nu, nu])) @ S.T, S


class TestWilliamson:
    @SETTINGS
    @given(seed=seeds, M=st.integers(1, 5), degenerate=st.booleans())
    def test_identities(self, seed, M, degenerate):
        rng = np.random.default_rng(seed)
        nu = np.sort(rng.uniform(0.5, 3.0, M))
        if degenerate:
            nu[: max(1, M // 2 + 1)] = 0.5  # several pure modes
        V, S0 = _random_covariance(rng, M, nu)
        w = la.williamson(V)
        c = np.abs(S0).max() ** 2
        assert np.allclose(w.nu, nu, atol=1e-10 * c)
        D = np.diag(np.concatenate([w.nu, w.nu]))
        assert np.allclose(w.S @ D @ w.S.T, V, atol=1e-10 * c)  # stress run: worst 7.7e-12·c
        assert cv.is_symplectic(w.S, 1e-10 * c)
        assert np.allclose(la.symplectic_eigenvalues(V), w.nu, atol=1e-10 * c)
        assert sum(len(g) for g in w.groups) == M

    def test_thermal_state_is_already_normal(self):
        """Identity reference: an equal-ν (degenerate) diagonal covariance gives S = I exactly."""
        w = la.williamson(1.7 * np.eye(6))
        assert np.allclose(w.nu, 1.7) and np.allclose(w.S, np.eye(6), atol=1e-12)
        assert [g.tolist() for g in w.groups] == [[0, 1, 2]]

    def test_distinct_thermal_modes_are_sorted(self):
        V = np.diag([2.0, 0.6, 1.0, 2.0, 0.6, 1.0])
        w = la.williamson(V)
        assert np.allclose(w.nu, [0.6, 1.0, 2.0])
        P = np.abs(w.S)  # a permutation of the modes, with signs fixed positive
        assert np.allclose(w.S, P) and np.allclose(P @ P.T, np.eye(6))

    def test_gauge_does_not_depend_on_lapack_basis(self, rng):
        V, _ = _random_covariance(rng, 4, np.array([0.5, 0.5, 0.5, 1.2]))
        tiny = 1e-13 * (lambda B: B + B.T)(rng.normal(size=(8, 8)))
        a, b = la.williamson(V), la.williamson(V + tiny)
        assert np.allclose(a.S, b.S, atol=1e-9)

    @pytest.mark.parametrize("nu", [[0.6, 1.1, 1.7], [0.5, 0.5, 1.3]], ids=["generic", "degenerate"])
    def test_factor_is_invariant_under_hbar_rescaling(self, rng, nu):
        """V → 2V (ħ = 1 → 2) doubles ν and leaves S unchanged (CONVENTIONS.md, comparison)."""
        V, _ = _random_covariance(rng, 3, np.array(nu))
        a, b = la.williamson(V), la.williamson(2 * V)
        assert np.allclose(b.nu, 2 * a.nu) and np.allclose(b.S, a.S, atol=1e-9)

    def test_factor_is_covariant_under_mode_relabelling(self, rng):
        """Non-degenerate ν: relabelling the modes permutes the rows of S, nothing else."""
        V, _ = _random_covariance(rng, 3, np.array([0.6, 1.1, 1.7]))
        p = np.array([2, 0, 1])
        P = np.eye(6)[np.concatenate([p, p + 3])]
        a, b = la.williamson(V), la.williamson(P @ V @ P.T)
        assert np.allclose(b.S, P @ a.S, atol=1e-9)

    def test_reference_tracks_a_previous_result(self, rng):
        V, _ = _random_covariance(rng, 3, np.array([0.8, 0.8, 0.8]))
        first = la.williamson(V)
        again = la.williamson(V, reference=first.S)
        assert np.allclose(again.S, first.S, atol=1e-10)

    @pytest.mark.parametrize(
        "V",
        [np.diag([0.5, -0.1, 0.5, 1.0]), np.zeros((2, 2)), np.ones((3, 3)), np.array([[1, 0.5j], [-0.5j, 1]])],
    )
    def test_rejects(self, V):
        with pytest.raises(InputError):
            la.williamson(V)
        with pytest.raises(InputError):
            la.symplectic_eigenvalues(V)

    def test_complex_input_with_zero_imaginary_part_is_accepted(self):
        w = la.williamson(np.eye(2, dtype=complex))
        assert np.allclose(w.nu, 1.0)

    def test_reference_shape_is_checked(self):
        with pytest.raises(ShapeError, match="reference"):
            la.williamson(np.eye(4), reference=np.eye(3))


# ------------------------------------------------------------------------------- Takagi
class TestTakagi:
    @SETTINGS
    @given(seed=seeds, n=st.integers(1, 6), pattern=st.sampled_from(["generic", "degenerate", "rank_deficient"]))
    def test_identities(self, seed, n, pattern):
        rng = np.random.default_rng(seed)
        if pattern == "generic":
            B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            A = B + B.T
        else:
            s = np.sort(rng.uniform(0.2, 3, n))[::-1]
            s[: (n + 1) // 2] = s[0]  # a degenerate group
            if pattern == "rank_deficient":
                s[-1] = 0.0
            U0 = random_unitary(rng, n)
            A = U0 @ np.diag(s) @ U0.T
        t = la.takagi(A)
        assert np.allclose(t.U @ np.diag(t.s) @ t.U.T, A, atol=1e-11 * max(1, np.abs(A).max()))
        assert np.allclose(t.U.conj().T @ t.U, np.eye(n), atol=1e-12)
        assert np.all(np.diff(t.s) <= 1e-12) and np.all(t.s >= 0)

    def test_identity_reference_on_a_diagonal_matrix(self):
        t = la.takagi(np.diag([3.0, 3.0, 1.0]))
        assert np.allclose(t.U, np.eye(3), atol=1e-12)

    def test_gauge_is_stable(self, rng):
        U0 = random_unitary(rng, 5)
        A = U0 @ np.diag([2, 2, 2, 1, 0]) @ U0.T
        noise = 1e-13 * (lambda B: B + B.T)(rng.normal(size=(5, 5)))
        assert np.allclose(la.takagi(A).U, la.takagi(A + noise).U, atol=1e-9)

    def test_rejects_non_symmetric(self):
        with pytest.raises(InputError, match="symmetric"):
            la.takagi(np.array([[1, 2], [3, 4]]))


# ------------------------------------------------------------------------------- Bloch–Messiah
class TestBlochMessiah:
    @SETTINGS
    @given(seed=seeds, M=st.integers(1, 5), r_max=st.floats(0.0, 3.0))
    def test_identities_generic(self, seed, M, r_max):
        rng = np.random.default_rng(seed)
        S = random_symplectic(rng, M, r_max)
        bm = la.bloch_messiah(S)
        c = np.abs(S).max() ** 2
        # squeezing below DEGENERACY_RTOL is reported as 0 (Hypothesis found r_max ~ 1e-7 cases)
        assert np.allclose(bm_reconstruct(bm), S, atol=1e-11 * c + 3 * la.DEGENERACY_RTOL)
        assert_orthosymplectic(bm.O1, 1e-12)
        assert_orthosymplectic(bm.O2, 1e-12)
        assert np.all(np.diff(bm.r) <= 1e-12) and np.all(bm.r >= 0)

    @pytest.mark.parametrize("r_max", [1e-12, 1e-9, 1e-8, 1.2e-7, 1e-6])
    def test_tiny_squeezing(self, r_max):
        """Regression (found by Hypothesis): tiny r once left O2 non-orthogonal by ~1e-9."""
        S = random_symplectic(np.random.default_rng(0), 3, r_max)
        bm = la.bloch_messiah(S)
        assert_orthosymplectic(bm.O1, 1e-12)
        assert_orthosymplectic(bm.O2, 1e-12)
        assert np.allclose(bm_reconstruct(bm), S, atol=3 * la.DEGENERACY_RTOL)

    @SETTINGS
    @given(seed=seeds, pattern=st.sampled_from([[0.7, 0.7, 0.7, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0] * 3, [2.0, 2.0]]))
    def test_identities_degenerate(self, seed, pattern):
        rng = np.random.default_rng(seed)
        S = degenerate_symplectic(rng, pattern)
        bm = la.bloch_messiah(S)
        c = np.abs(S).max() ** 2
        assert np.allclose(bm.r, sorted(pattern, reverse=True), atol=1e-10)
        assert np.allclose(bm_reconstruct(bm), S, atol=1e-11 * c)
        assert_orthosymplectic(bm.O1, 1e-12)
        assert_orthosymplectic(bm.O2, 1e-12)
        sizes = sorted(len(g) for g in bm.groups)
        assert sizes == sorted(pattern.count(v) for v in sorted(set(pattern), reverse=True))

    def test_passive_matrix_has_identity_supermodes(self, rng):
        """r = 0 everywhere: one U(M) group, aligned with the input modes, so O1 = I and O2 = S."""
        S = cv.interferometer(random_unitary(rng, 4))
        bm = la.bloch_messiah(S)
        assert np.allclose(bm.r, 0) and np.allclose(bm.O1, np.eye(8), atol=1e-12)
        assert np.allclose(bm.O2, S, atol=1e-12)

    def test_squeezing_one_mode(self):
        S = cv.embed(cv.squeezing(0.8), [2], 3)
        bm = la.bloch_messiah(S)
        assert np.allclose(bm.r, [0.8, 0, 0])
        assert np.allclose(bm.O1[:3, 0], [0, 0, 1], atol=1e-12)  # supermode 0 is mode 2

    def test_user_reference_basis(self, rng):
        """Equal squeezing of the modes of a reference basis R: the supermodes come back as R."""
        R = random_unitary(rng, 3)
        S = cv.interferometer(R) @ np.diag(np.exp([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]))
        bm = la.bloch_messiah(S, reference=R)
        U = bm.O1[:3, :3] + 1j * bm.O1[3:, :3]
        assert np.allclose(U, R, atol=1e-10)

    def test_gauge_does_not_depend_on_lapack_basis(self, rng):
        """The ε-split experiment of the S03 check-in: supermodes no longer jump."""
        Ua, Ub = random_unitary(rng, 4), random_unitary(rng, 4)

        def S_of(rs):
            return cv.interferometer(Ua) @ np.diag(np.exp(np.concatenate([-rs, rs]))) @ cv.interferometer(Ub)

        base = la.bloch_messiah(S_of(np.array([0.7, 0.7, 0.7, 0.0])))
        for eps in (1e-13, 1e-10):
            bm = la.bloch_messiah(S_of(np.array([0.7 + eps, 0.7, 0.7 - eps, 0.0])))
            assert np.allclose(bm.O1, base.O1, atol=1e-9)

    def test_sweep_through_near_degeneracy_with_reference(self, rng):
        Ua, Ub = random_unitary(rng, 4), random_unitary(rng, 4)

        def S_of(rs):
            return cv.interferometer(Ua) @ np.diag(np.exp(np.concatenate([-rs, rs]))) @ cv.interferometer(Ub)

        prev = la.bloch_messiah(S_of(np.array([0.7, 0.7, 0.7, 0.0])))
        for eps in (1e-6, 2e-6, 3e-6):
            S = S_of(np.array([0.7 + eps, 0.7, 0.7 - eps, 0.0]))
            bm = la.bloch_messiah(S, reference=prev.O1, rtol=1e-4)
            assert [len(g) for g in bm.groups] == [3, 1]
            assert np.allclose(bm.O1, prev.O1, atol=1e-9)
            # grouping values that differ by ~ε costs a reconstruction error ~ε (documented)
            assert np.allclose(bm_reconstruct(bm), S, atol=10 * eps)
            prev = bm

    @pytest.mark.parametrize(("S", "err"), [(2 * np.eye(2), InputError), (np.eye(3), ShapeError)])
    def test_rejects(self, S, err):
        with pytest.raises(err):
            la.bloch_messiah(S)


def test_lagrangian_basis_skips_dependent_candidates():
    """Candidates already spanned (here: the partner Ωᵀw of the first one) are skipped."""
    Om = cv.omega(2)
    e = np.eye(4)
    cands = np.column_stack([e[:, 0], Om.T @ e[:, 0], e[:, 1]])
    X = la._lagrangian_basis(cands, Om, 2)
    assert np.allclose(X, e[:, :2])


# ------------------------------------------------------------------------------- physical projection
class TestNearestPhysical:
    def test_physical_input_is_unchanged(self, rng):
        V, _ = _random_covariance(rng, 3, np.array([0.5, 0.9, 2.0]))
        assert np.array_equal(la.nearest_physical_covariance(V), (V + V.T) / 2)

    @SETTINGS
    @given(seed=seeds, M=st.integers(1, 4))
    def test_lifts_and_only_adds_noise(self, seed, M):
        rng = np.random.default_rng(seed)
        nu = rng.uniform(0.2, 1.5, M)  # some below 1/2: unphysical
        V, _ = _random_covariance(rng, M, nu, r_max=1.0)
        Vp = la.nearest_physical_covariance(V)
        assert la.symplectic_eigenvalues(Vp).min() >= 0.5 - 1e-9
        assert la.eigvalsh(Vp - V).min() >= -1e-9  # pure added noise
        kept = nu >= 0.5
        if kept.all():
            assert np.allclose(Vp, V)

    def test_not_even_positive_definite(self):
        V = np.diag([1.0, -0.2, 1.0, 0.3])
        Vp = la.nearest_physical_covariance(V)
        assert la.symplectic_eigenvalues(Vp).min() >= 0.5 - 1e-12
