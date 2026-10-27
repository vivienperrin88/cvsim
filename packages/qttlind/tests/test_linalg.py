"""The dense kernel (qttlind.core.linalg)."""

import numpy as np
import pytest

from qttlind.core import linalg
from qttlind.errors import InputError, ShapeError


def crandn(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


# ============================================================================ contract, fuse
class TestFuse:
    def test_shapes_and_values(self, rng):
        t = crandn(rng, 2, 3, 4, 5)
        m, left, right = linalg.fuse(t, 2)
        assert m.shape == (6, 20) and left == (2, 3) and right == (4, 5)
        assert m[1 * 3 + 2, 3 * 5 + 4] == t[1, 2, 3, 4]  # C order: last axis fastest

    @pytest.mark.parametrize("split", [0, 4])
    def test_extreme_splits(self, rng, split):
        t = crandn(rng, 2, 3, 4, 5)
        m, _, _ = linalg.fuse(t, split)
        assert m.size == t.size and 1 in m.shape

    @pytest.mark.parametrize("split", [-1, 5])
    def test_bad_split(self, rng, split):
        with pytest.raises(ShapeError):
            linalg.fuse(crandn(rng, 2, 3, 4, 5), split)


class TestContract:
    CASES = (
        ((3, 4, 5), [1, 2], (5, 4, 2), [1, 0]),  # two axes, out of order
        ((3, 4), [1], (4, 6), [0]),  # plain matrix product
        ((2, 3, 4), [], (5,), []),  # outer product
        ((2, 3, 4), [0, 1, 2], (4, 3, 2), [2, 1, 0]),  # full contraction -> scalar
        ((2, 3, 4), [-1], (4, 7), [0]),  # negative axis
    )

    @pytest.mark.parametrize("sa,ax_a,sb,ax_b", CASES)
    def test_matches_tensordot(self, rng, sa, ax_a, sb, ax_b):
        a, b = crandn(rng, *sa), crandn(rng, *sb)
        out = linalg.contract(a, ax_a, b, ax_b)
        ref = np.tensordot(a, b, (ax_a, ax_b))
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_matches_einsum_on_an_mps_step(self, rng):
        env, core = crandn(rng, 3, 3), crandn(rng, 3, 2, 4)
        out = linalg.contract(env, [1], core, [0])
        np.testing.assert_allclose(out, np.einsum("ab,bsc->asc", env, core), atol=1e-12)

    @pytest.mark.parametrize(
        "sa,ax_a,sb,ax_b,match",
        [
            ((3, 4), [1], (5, 6), [0], "dim"),  # dimensions differ
            ((3, 4), [1, 0], (4, 3), [0], "paired"),  # different numbers of axes
            ((3, 4), [2], (4, 3), [0], "range"),  # axis out of range
            ((3, 4), [1, 1], (4, 4), [0, 1], "repeated"),
        ],
    )
    def test_errors(self, rng, sa, ax_a, sb, ax_b, match):
        with pytest.raises(ShapeError, match=match):
            linalg.contract(crandn(rng, *sa), ax_a, crandn(rng, *sb), ax_b)


# ============================================================================ qr, svd, timing
class TestQR:
    @pytest.mark.parametrize("shape,split", [((3, 2, 5), 2), ((8, 2, 2), 1), ((2, 2, 10), 2)])
    def test_reconstruction_and_isometry(self, rng, shape, split):
        t = crandn(rng, *shape)
        q, r = linalg.qr(t, split)
        np.testing.assert_allclose(linalg.contract(q, [-1], r, [0]), t, atol=1e-12)
        qm, _, _ = linalg.fuse(q, split)
        np.testing.assert_allclose(qm.conj().T @ qm, np.eye(qm.shape[1]), atol=1e-12)
        k = min(int(np.prod(shape[:split])), int(np.prod(shape[split:])))
        assert q.shape == shape[:split] + (k,) and r.shape == (k,) + shape[split:]


class TestSVD:
    def test_exact_reconstruction(self, rng):
        t = crandn(rng, 4, 2, 6)
        res = linalg.svd(t, split=2)
        assert res.discarded == 0.0
        np.testing.assert_allclose(linalg.contract(res.U * res.S, [-1], res.Vh, [0]), t, atol=1e-12)
        assert np.all(np.diff(res.S) <= 0)

    def test_discarded_weight_equals_squared_error(self, rng):
        """The key identity behind the tolerance convention."""
        u, _ = np.linalg.qr(crandn(rng, 20, 20))
        v, _ = np.linalg.qr(crandn(rng, 20, 20))
        s = 2.0 ** -np.arange(20)
        m = (u * s) @ v.conj().T
        res = linalg.svd(m, split=1, tol=1e-6)
        approx = (res.U * res.S) @ res.Vh
        rel_err2 = np.linalg.norm(m - approx) ** 2 / np.linalg.norm(m) ** 2
        assert res.discarded == pytest.approx(rel_err2, rel=1e-8)
        assert res.discarded <= 1e-6

    def test_chi_max_caps(self, rng):
        res = linalg.svd(crandn(rng, 6, 6), split=1, chi_max=2)
        assert res.S.shape == (2,) and res.U.shape == (6, 2) and res.Vh.shape == (2, 6)
        assert res.discarded > 0

    def test_bad_chi_max(self, rng):
        with pytest.raises(InputError):
            linalg.svd(crandn(rng, 3, 3), split=1, chi_max=0)

    def test_falls_back_to_gesvd(self, rng, monkeypatch):
        """Simulate gesdd failing to converge; the result must still be correct."""

        def broken_svd(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(np.linalg, "svd", broken_svd)
        t = crandn(rng, 5, 4)
        res = linalg.svd(t, split=1)
        np.testing.assert_allclose((res.U * res.S) @ res.Vh, t, atol=1e-12)


class TestTimed:
    def test_counts_calls(self, rng):
        t = crandn(rng, 4, 4)
        with linalg.timed() as outer:
            linalg.svd(t, 1)
            with linalg.timed() as inner:
                linalg.qr(t, 1)
                linalg.contract(t, [1], t, [0])
        assert inner.calls == 2 and outer.calls == 3
        assert outer.seconds >= inner.seconds > 0

    def test_block_is_closed_even_after_an_exception(self, rng):
        with pytest.raises(ShapeError), linalg.timed():
            linalg.fuse(crandn(rng, 2, 2), 5)
        assert linalg._ACTIVE == []
