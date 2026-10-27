"""TensorTrain."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qttlind.core.tt import TensorTrain
from qttlind.errors import InputError, ShapeError


def crandn(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def random_tt(rng, dims, chi):
    """A random tensor train with bonds min(chi, max possible)."""
    L = len(dims)
    bonds = [1]
    for i in range(1, L):
        bonds.append(min(chi, int(np.prod(dims[:i])), int(np.prod(dims[i:]))))
    bonds.append(1)
    return TensorTrain([crandn(rng, bonds[i], dims[i], bonds[i + 1]) for i in range(L)])


# ============================================================================
class TestConstruction:
    def test_valid_train(self, rng):
        tt = TensorTrain([crandn(rng, 1, 2, 3), crandn(rng, 3, 4, 5), crandn(rng, 5, 2, 1)])
        assert tt.length == 3 and tt.dims == (2, 4, 2) and tt.ranks == (3, 5)
        assert tt.max_rank == 5 and tt.size == 6 + 60 + 10 and tt.center is None
        assert all(c.dtype == np.complex128 for c in tt.cores)
        assert "ranks=(3, 5)" in repr(tt)

    def test_real_input_is_promoted(self):
        tt = TensorTrain([np.ones((1, 2, 1))])
        assert tt.cores[0].dtype == np.complex128

    def test_copy_isolates(self, rng):
        core = crandn(rng, 1, 2, 1)
        tt = TensorTrain([core])
        core[0, 0, 0] = 99
        assert tt.cores[0][0, 0, 0] != 99

    @pytest.mark.parametrize(
        "shapes,match",
        [
            ([], "at least one"),
            ([(1, 2)], "3 axes"),
            ([(2, 2, 1)], "first core"),
            ([(1, 2, 2)], "last core"),
            ([(1, 2, 3), (2, 2, 1)], "bond 0"),
            ([(1, 0, 1)], "empty"),
        ],
    )
    def test_invalid_shapes(self, shapes, match):
        with pytest.raises(ShapeError, match=match):
            TensorTrain([np.zeros(s) for s in shapes])

    @pytest.mark.parametrize("center", [-1, 3])
    def test_invalid_center(self, rng, center):
        with pytest.raises(InputError):
            TensorTrain([crandn(rng, 1, 2, 1)] * 3, center=center)


class TestDense:
    @pytest.mark.parametrize("dims", [(2,), (2, 2, 2, 2, 2), (3, 4, 2), (2, 5)])
    def test_tt_svd_roundtrip(self, rng, dims):
        v = crandn(rng, int(np.prod(dims)))
        tt = TensorTrain.from_dense(v, dims)
        assert tt.dims == tuple(dims) and tt.center == len(dims) - 1
        np.testing.assert_allclose(tt.to_dense(), v, atol=1e-12)

    def test_tt_svd_maximal_ranks(self, rng):
        tt = TensorTrain.from_dense(crandn(rng, 2**6), [2] * 6)
        assert tt.ranks == (2, 4, 8, 4, 2)  # min(2^i, 2^(L-i))

    def test_tt_svd_is_left_canonical(self, rng):
        tt = TensorTrain.from_dense(crandn(rng, 2**5), [2] * 5)
        assert all(tt.is_left_isometry(i) for i in range(4))

    def test_low_rank_vector_is_detected(self, rng):
        """A product of local vectors has rank 1 at any tolerance."""
        locs = [crandn(rng, 2) for _ in range(6)]
        v = locs[0]
        for x in locs[1:]:
            v = np.kron(v, x)
        tt = TensorTrain.from_dense(v, [2] * 6, tol=1e-24)
        assert tt.max_rank == 1

    def test_truncation_error_is_controlled(self, rng):
        x = np.linspace(0, 1, 2**10)
        v = np.exp(-(((x - 0.3) / 0.05) ** 2)) + 0j
        tt = TensorTrain.from_dense(v, [2] * 10, tol=1e-12)
        err2 = np.linalg.norm(tt.to_dense() - v) ** 2 / np.linalg.norm(v) ** 2
        assert err2 < 10 * 1e-12 and tt.max_rank < 32

    def test_size_mismatch(self, rng):
        with pytest.raises(ShapeError, match="does not match"):
            TensorTrain.from_dense(crandn(rng, 10), [2, 2, 2])

    def test_product(self, rng):
        locs = [crandn(rng, d) for d in (2, 3, 2)]
        tt = TensorTrain.product(locs)
        np.testing.assert_allclose(tt.to_dense(), np.kron(np.kron(locs[0], locs[1]), locs[2]))
        assert tt.ranks == (1, 1)

    def test_product_errors(self):
        with pytest.raises(InputError):
            TensorTrain.product([])
        with pytest.raises(ShapeError):
            TensorTrain.product([np.ones((2, 2))])


# ============================================================================
class TestCanonical:
    @pytest.mark.parametrize("center", [0, 2, 5])
    def test_canonicalize(self, rng, center):
        tt = random_tt(rng, [2] * 6, chi=4)
        c = tt.canonicalize(center)
        np.testing.assert_allclose(c.to_dense(), tt.to_dense(), atol=1e-10)
        assert c.center == center
        assert all(c.is_left_isometry(i) for i in range(center))
        assert all(c.is_right_isometry(i) for i in range(center + 1, 6))
        assert tt.center is None  # the original is untouched

    def test_bad_center(self, rng):
        with pytest.raises(InputError):
            random_tt(rng, [2] * 3, 2).canonicalize(3)

    def test_inner_and_norm(self, rng):
        a, b = random_tt(rng, [2, 3, 2, 2], 3), random_tt(rng, [2, 3, 2, 2], 2)
        assert a.inner(b) == pytest.approx(np.vdot(a.to_dense(), b.to_dense()), rel=1e-10)
        assert a.norm() == pytest.approx(np.linalg.norm(a.to_dense()), rel=1e-12)
        assert a.canonicalize(2).norm() == pytest.approx(a.norm(), rel=1e-12)

    def test_inner_dims_mismatch(self, rng):
        with pytest.raises(ShapeError):
            random_tt(rng, [2, 2], 2).inner(random_tt(rng, [2, 3], 2))


class TestAlgebra:
    def test_add_and_sub(self, rng):
        a, b = random_tt(rng, [2] * 5, 3), random_tt(rng, [2] * 5, 2)
        s = a + b
        assert s.ranks == tuple(x + y for x, y in zip(a.ranks, b.ranks))
        np.testing.assert_allclose(s.to_dense(), a.to_dense() + b.to_dense(), atol=1e-12)
        np.testing.assert_allclose((a - b).to_dense(), a.to_dense() - b.to_dense(), atol=1e-12)

    def test_add_single_core(self, rng):
        a, b = random_tt(rng, [5], 1), random_tt(rng, [5], 1)
        np.testing.assert_allclose((a + b).to_dense(), a.to_dense() + b.to_dense())

    def test_add_dims_mismatch(self, rng):
        with pytest.raises(ShapeError):
            random_tt(rng, [2, 2], 2) + random_tt(rng, [2, 3], 2)

    def test_scalar_multiplication(self, rng):
        a = random_tt(rng, [2] * 4, 3)
        np.testing.assert_allclose((2.5j * a).to_dense(), 2.5j * a.to_dense())
        np.testing.assert_allclose((a * 3).to_dense(), 3 * a.to_dense())


class TestCompress:
    def test_lossless_compression_finds_true_rank(self, rng):
        a = random_tt(rng, [2] * 8, 3)
        doubled = a + a  # ranks 6, true rank 3
        c, disc = doubled.compress(tol=1e-24)
        assert c.max_rank <= 3 and disc < 1e-20 and c.center == 0
        np.testing.assert_allclose(c.to_dense(), 2 * a.to_dense(), atol=1e-10)

    def test_discarded_weight_bounds_the_error(self, rng):
        a = random_tt(rng, [2] * 8, 8)
        c, disc = a.compress(tol=1e-3)
        err2 = np.linalg.norm(c.to_dense() - a.to_dense()) ** 2 / a.norm() ** 2
        assert err2 <= disc * (1 + 1e-8) + 1e-14

    def test_chi_max(self, rng):
        c, _ = random_tt(rng, [2] * 8, 8).compress(chi_max=3)
        assert c.max_rank == 3

    @settings(max_examples=25, deadline=None)
    @given(seed=st.integers(0, 2**32 - 1), chi=st.integers(1, 6), L=st.integers(2, 7))
    def test_compress_property(self, seed, chi, L):
        """Property: compressing at tol=0 never changes the vector."""
        rng = np.random.default_rng(seed)
        a = random_tt(rng, [2] * L, chi)
        c, _ = a.compress(tol=0.0)
        np.testing.assert_allclose(c.to_dense(), a.to_dense(), atol=1e-9 * max(1.0, a.norm()))


# ============================================================================
class TestHadamard:
    def test_elementwise_product(self, rng):
        a, b = random_tt(rng, [2] * 6, 3), random_tt(rng, [2] * 6, 2)
        h = a.hadamard(b)
        assert h.ranks == tuple(x * y for x, y in zip(a.ranks, b.ranks))
        np.testing.assert_allclose(h.to_dense(), a.to_dense() * b.to_dense(), atol=1e-10)

    def test_dims_mismatch(self, rng):
        with pytest.raises(ShapeError):
            random_tt(rng, [2, 2], 2).hadamard(random_tt(rng, [2, 3], 2))
