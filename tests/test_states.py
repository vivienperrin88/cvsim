"""Quantics bosonic states (qttlind.bosons.states)."""

import numpy as np
import pytest

from qttlind.bosons import states
from qttlind.errors import InputError
from qttlind.reference import dense_ref as dr


@pytest.mark.parametrize("k,bits", [(0, 1), (5, 4), (255, 8), (1000, 12)])
def test_fock(k, bits):
    tt = states.fock(k, bits)
    assert tt.max_rank == 1
    np.testing.assert_array_equal(tt.to_dense(), np.eye(2**bits)[k])


def test_fock_out_of_range():
    with pytest.raises(InputError):
        states.fock(16, 4)


def test_from_function_polynomial_has_low_rank():
    tt = states.from_function(lambda n: n**2 + 1.0, 10)
    np.testing.assert_allclose(tt.to_dense(), np.arange(1024.0) ** 2 + 1)
    assert tt.max_rank == 3  # degree-2 polynomial -> rank 3


@pytest.mark.parametrize("bad", [0, 63])
def test_from_function_bits_guard(bad):
    with pytest.raises(InputError):
        states.from_function(lambda n: n, bad)


def test_from_function_shape_guard():
    with pytest.raises(InputError, match="shape"):
        states.from_function(lambda n: np.ones(3), 4)


def test_coherent_amplitudes_are_stable_for_large_n():
    n = np.arange(5000)
    c = states.coherent_amplitudes(40.0, n)  # naive alpha^n / sqrt(n!) overflows here
    assert np.all(np.isfinite(c))
    assert np.sum(np.abs(c) ** 2) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5 + 0.5j, -2.0j])
def test_coherent_matches_oracle(alpha):
    tt = states.coherent(alpha, 7)
    np.testing.assert_allclose(tt.to_dense(), dr.coherent(128, alpha), atol=1e-9)
    assert tt.norm() == pytest.approx(1.0, abs=1e-12)


def test_coherent_rank_stays_small_as_the_space_grows():
    """The paper's point (Fig. 4/5): |alpha|^2 = N/3 keeps a small rank for every N."""
    ranks = [states.coherent(np.sqrt(2**R / 3), R, tol=1e-16).max_rank for R in (8, 12, 16)]
    assert max(ranks) <= 12
    assert ranks[-1] <= ranks[0] + 4


@pytest.mark.parametrize("parity", [1, -1])
def test_cat_matches_oracle_and_has_parity(parity):
    tt = states.cat(2.0, 7, parity)
    psi = tt.to_dense()
    np.testing.assert_allclose(psi, dr.cat(128, 2.0, parity), atol=1e-9)
    assert np.vdot(psi, dr.parity_op(128) @ psi).real == pytest.approx(parity, abs=1e-10)


@pytest.mark.parametrize("args", [(2.0, 5, 0), (0.0, 5, -1)])
def test_cat_errors(args):
    with pytest.raises(InputError):
        states.cat(*args)


def test_from_function_beyond_dense_sampling_uses_tci():
    bits = states.MAX_DENSE_BITS + 4  # 2^30 points: never sampled densely
    tt = states.from_function(lambda n: np.sqrt(n.astype(float)), bits, tol=1e-20)
    assert tt.length == bits and tt.max_rank <= 12
    for n in (0, 1, 12345, 2**29 + 7):
        v = np.ones(1)
        for c, b in zip(tt.cores, [(n >> (bits - 1 - k)) & 1 for k in range(bits)]):
            v = v @ c[:, b, :]
        assert v[0].real == pytest.approx(np.sqrt(n), abs=1e-7 * np.sqrt(2.0**bits))  # max-norm accuracy
