"""The MPO class (qttlind.core.mpo)."""

import numpy as np
import pytest

from qttlind.core.mpo import MPO
from qttlind.core.tt import TensorTrain
from qttlind.errors import ShapeError


def crandn(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def random_mpo(rng, dims_out, dims_in, chi):
    L = len(dims_out)
    bonds = [1] + [chi] * (L - 1) + [1]
    return MPO([crandn(rng, bonds[k], dims_out[k], dims_in[k], bonds[k + 1]) for k in range(L)])


def random_state(rng, dims):
    return TensorTrain.from_dense(crandn(rng, int(np.prod(dims))), dims)


class TestConstruction:
    def test_properties(self, rng):
        w = random_mpo(rng, (2, 3), (4, 2), 5)
        assert w.length == 2 and w.dims_out == (2, 3) and w.dims_in == (4, 2) and w.ranks == (5,)
        assert "ranks=(5,)" in repr(w)

    @pytest.mark.parametrize("shapes", [[], [(1, 2, 2)], [(2, 2, 2, 1)], [(1, 2, 2, 3), (2, 2, 2, 1)]])
    def test_invalid(self, shapes):
        with pytest.raises(ShapeError):
            MPO([np.zeros(s) for s in shapes])

    def test_identity(self):
        np.testing.assert_array_equal(MPO.identity([2, 3, 2]).to_dense(), np.eye(12))


class TestDense:
    def test_to_dense_index_order(self):
        """Row index = outputs with site 1 most significant; column index = inputs."""
        w0 = np.zeros((1, 2, 2, 1))
        w0[0, 1, 0, 0] = 1  # |1><0| on site 1
        w1 = np.zeros((1, 2, 2, 1))
        w1[0, 0, 1, 0] = 1  # |0><1| on site 2
        dense = MPO([w0, w1]).to_dense()
        expected = np.zeros((4, 4))
        expected[2, 1] = 1  # out (1,0) = 2, in (0,1) = 1
        np.testing.assert_array_equal(dense, expected)

    @pytest.mark.parametrize("dims_out,dims_in", [((2, 2, 2), None), ((2, 3), (3, 2))])
    def test_from_dense_roundtrip(self, rng, dims_out, dims_in):
        din = dims_out if dims_in is None else dims_in
        a = crandn(rng, int(np.prod(dims_out)), int(np.prod(din)))
        w = MPO.from_dense(a, dims_out, dims_in)
        np.testing.assert_allclose(w.to_dense(), a, atol=1e-11)

    def test_from_dense_shape_check(self, rng):
        with pytest.raises(ShapeError):
            MPO.from_dense(crandn(rng, 4, 4), [2, 2, 2])

    def test_tensor_train_view_roundtrip(self, rng):
        w = random_mpo(rng, (2, 2), (2, 2), 3)
        back = MPO.from_tensor_train(w.as_tensor_train(), w.dims_out, w.dims_in)
        np.testing.assert_allclose(back.to_dense(), w.to_dense())


class TestAlgebra:
    def test_apply(self, rng):
        w = random_mpo(rng, (2, 3, 2), (2, 2, 2), 3)
        psi = random_state(rng, (2, 2, 2))
        out = w.apply(psi)
        assert out.ranks == tuple(a * b for a, b in zip(w.ranks, psi.ranks))
        np.testing.assert_allclose(out.to_dense(), w.to_dense() @ psi.to_dense(), atol=1e-10)

    def test_apply_dims_mismatch(self, rng):
        with pytest.raises(ShapeError):
            random_mpo(rng, (2, 2), (2, 2), 2).apply(random_state(rng, (2, 3)))

    def test_matmul_add_scale(self, rng):
        a = random_mpo(rng, (2, 2, 2), (2, 2, 2), 2)
        b = random_mpo(rng, (2, 2, 2), (2, 2, 2), 3)
        A, B = a.to_dense(), b.to_dense()
        np.testing.assert_allclose((a @ b).to_dense(), A @ B, atol=1e-10)
        np.testing.assert_allclose((a + b).to_dense(), A + B, atol=1e-10)
        np.testing.assert_allclose((a - 2j * b).to_dense(), A - 2j * B, atol=1e-10)
        assert (a @ b).ranks == (6, 6) and (a + b).ranks == (5, 5)

    def test_matmul_dims_mismatch(self, rng):
        with pytest.raises(ShapeError):
            random_mpo(rng, (2, 2), (2, 3), 2) @ random_mpo(rng, (2, 2), (2, 2), 2)

    def test_dagger(self, rng):
        w = random_mpo(rng, (2, 3), (3, 2), 2)
        np.testing.assert_allclose(w.dagger().to_dense(), w.to_dense().conj().T)

    def test_compress(self, rng):
        a = random_mpo(rng, (2, 2, 2, 2), (2, 2, 2, 2), 2)
        c, disc = (a + a).compress(tol=1e-24)
        assert c.max_rank <= 2 and disc < 1e-20
        np.testing.assert_allclose(c.to_dense(), 2 * a.to_dense(), atol=1e-10)
