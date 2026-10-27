"""Bosonic quantics operators (qttlind.bosons.operators)."""

import numpy as np
import pytest

from qttlind.bosons import operators as ops
from qttlind.bosons import states
from qttlind.errors import InputError
from qttlind.reference import dense_ref as dr


# ============================================================================ the shift
class TestShift:
    @pytest.mark.parametrize("bits,m", [(1, 0), (1, 1), (4, 1), (5, 3), (6, 7), (8, 77), (8, 255)])
    def test_dense_matrix(self, bits, m):
        T = ops.shift(bits, m)
        np.testing.assert_array_equal(T.to_dense(), np.eye(2**bits, k=m))

    def test_rank_two(self):
        assert ops.shift(10, 1).ranks == (2,) * 9

    def test_action_on_a_state(self):
        """(T psi)_n = psi_{n+1}: shifting moves amplitudes DOWN by one."""
        psi = states.fock(5, 4)
        out = ops.shift(4, 1).apply(psi).to_dense()
        np.testing.assert_array_equal(out, np.eye(16)[4])

    def test_overflow_is_dropped(self):
        out = ops.shift(3, 1).apply(states.fock(0, 3)).to_dense()
        np.testing.assert_array_equal(out, np.zeros(8))  # nothing maps to n = -1

    def test_composition(self):
        np.testing.assert_array_equal((ops.shift(5, 2) @ ops.shift(5, 3)).to_dense(), ops.shift(5, 5).to_dense())

    @pytest.mark.parametrize("m", [-1, 16])
    def test_bad_amount(self, m):
        with pytest.raises(InputError):
            ops.shift(4, m)


# ============================================================================ diagonals & ladders
class TestDiagonals:
    def test_diag(self, rng):
        f = states.from_function(lambda n: np.cos(0.3 * n) + 1j * n, 6)
        np.testing.assert_allclose(ops.diag(f).to_dense(), np.diag(f.to_dense()), atol=1e-12)
        assert ops.diag(f).ranks == f.ranks

    @pytest.mark.parametrize("bits", [1, 4, 9])
    def test_number_is_exact_rank_two(self, bits):
        n = ops.number(bits)
        np.testing.assert_array_equal(n.to_dense(), np.diag(np.arange(2.0**bits)))
        assert all(r == 2 for r in n.ranks)

    @pytest.mark.parametrize("m", [0, 1, 2, 3, 4])
    def test_falling_factorial_exact_rank(self, m):
        n = np.arange(2.0**10)
        ref = np.prod([n - p for p in range(m)], axis=0) if m else np.ones_like(n)
        ff = ops.falling_factorial(m, 10)
        # values reach 1e12 for m = 4: compare relative to the largest entry (round-off scale)
        np.testing.assert_allclose(ff.to_dense(), ref, rtol=0, atol=1e-12 * np.abs(ref).max())
        assert ff.max_rank == m + 1  # Table I: (a^dag)^m a^m has rank m+1

    def test_normal_power_matches_dense(self):
        a = dr.destroy(64)
        np.testing.assert_allclose(ops.normal_power(2, 6).to_dense(), a.conj().T @ a.conj().T @ a @ a, atol=1e-9)

    def test_parity_rank_one(self):
        p = ops.parity(6)
        assert p.max_rank == 1
        np.testing.assert_array_equal(p.to_dense(), dr.parity_op(64))


class TestLadder:
    @pytest.mark.parametrize("bits", [4, 7, 10])
    def test_annihilation(self, bits):
        a = ops.annihilation(bits)
        ref = dr.destroy(2**bits)
        assert np.linalg.norm(a.to_dense() - ref) / np.linalg.norm(ref) < 1e-11
        assert a.max_rank <= 10  # Table I: rank 8-9 at 1e-12

    def test_creation_is_the_adjoint(self):
        np.testing.assert_allclose(ops.creation(6).to_dense(), dr.destroy(64).T, atol=1e-10)

    @pytest.mark.parametrize("m", [2, 3])
    def test_annihilation_power(self, m):
        ref = np.linalg.matrix_power(dr.destroy(128), m)
        out = ops.annihilation_power(m, 7).to_dense()
        assert np.linalg.norm(out - ref) / np.linalg.norm(ref) < 1e-10

    def test_large_coherent_state_eigenvalue(self):
        """Fig. 3's second metric: <alpha|a|alpha>/alpha -> 1 for a big coherent state."""
        bits = 12
        alpha = 2 ** ((bits - 1) / 2)
        psi = states.coherent(alpha, bits, tol=1e-20)
        a_psi = ops.annihilation(bits).apply(psi)
        assert psi.inner(a_psi) / alpha == pytest.approx(1.0, abs=1e-8)

    def test_commutator_away_from_cutoff(self):
        bits = 6
        a, ad = ops.annihilation(bits), ops.creation(bits)
        comm = (a @ ad - ad @ a).to_dense()
        np.testing.assert_allclose(comm[:-1, :-1], np.eye(63), atol=1e-9)

    @pytest.mark.parametrize("m", [0, 16])
    def test_bad_power(self, m):
        with pytest.raises(InputError):
            ops.annihilation_power(m, 4)
