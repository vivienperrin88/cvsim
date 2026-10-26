"""The shared conventions.

The production code (qttlind.core.conventions) and the independent oracle (qttlind.reference)
must implement the same rules. Three kinds of tests appear here:

* example tests   -- a few hand-checked cases (the paper's example, edge cases);
* comparison tests -- production vs oracle vs the mathematical definition, on random inputs;
* property tests  -- Hypothesis generates inputs and checks an invariant (round trip, bounds).
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qttlind.core import conventions as conv
from qttlind.errors import InputError, QttlindError
from qttlind.reference import dense_ref as dr
from qttlind.reference import quantics as qref

# ----------------------------------------------------------------------------- quantics map


def test_bit_order_paper_example():
    assert conv.bits_of(5, 4) == [0, 1, 0, 1] == qref.bits_of(5, 4)
    assert conv.index_of([1, 0, 1, 0]) == 10 == qref.index_of([1, 0, 1, 0])


@pytest.mark.parametrize("bits", [1, 2, 7, 12, 62])
def test_bits_agree_with_oracle(rng, bits):
    values = {0, 2**bits - 1} | {int(v) for v in rng.integers(0, 2 ** min(bits, 62), size=20)}
    for n in values:
        b = conv.bits_of(n, bits)
        assert b == qref.bits_of(n, bits)
        assert conv.index_of(b) == n


@given(bits=st.integers(1, 62), data=st.data())
def test_roundtrip_property(bits, data):
    n = data.draw(st.integers(0, 2**bits - 1))
    digits = conv.bits_of(n, bits)
    assert len(digits) == bits and set(digits) <= {0, 1}
    assert conv.index_of(digits) == n


def test_bits_accept_numpy_integers():
    assert conv.bits_of(np.int64(6), 3) == [1, 1, 0]
    assert conv.index_of(np.array([1, 1, 0])) == 6


@pytest.mark.parametrize("bad", [(16, 4), (0, 0), (0, 63), (-1, 4)])
def test_bit_errors(bad):
    for impl in (conv, qref):
        with pytest.raises(ValueError):
            impl.bits_of(*bad)
    with pytest.raises(InputError):  # the library raises its own type...
        conv.bits_of(*bad)
    with pytest.raises(QttlindError):  # ...which a single except can catch
        conv.bits_of(*bad)


@pytest.mark.parametrize("sigma", [[], [0, 2]])
def test_index_errors(sigma):
    for impl in (conv, qref):
        with pytest.raises(ValueError):
            impl.index_of(sigma)


def test_bits_array_matches_scalar_version(rng):
    n = rng.integers(0, 2**10, size=(3, 5))
    out = conv.bits_array(n, 10)
    assert out.shape == (3, 5, 10) and out.dtype == np.int8
    for idx in np.ndindex(3, 5):
        assert out[idx].tolist() == conv.bits_of(int(n[idx]), 10)


@pytest.mark.parametrize("bad", [[-1], [16], [0.5]])
def test_bits_array_errors(bad):
    with pytest.raises(InputError):
        conv.bits_array(np.array(bad), 4)


# ----------------------------------------------------------------------------- truncation rule


def _keep_by_definition(s, tol, chi_max):
    """Literal transcription of the rule in CONVENTIONS.md (slow, obviously right)."""
    w = np.asarray(s, float) ** 2
    if w.size == 0:
        return 0
    if w.sum() == 0:
        return 1
    k = next(k for k in range(1, w.size + 1) if w[k:].sum() <= tol * w.sum())
    return min(k, max(1, chi_max)) if chi_max is not None else k


@pytest.mark.parametrize("trial", range(200))
def test_keep_count_agrees(rng, trial):
    n = int(rng.integers(1, 40))
    s = np.sort(rng.exponential(size=n) ** rng.uniform(1, 8))[::-1]
    if trial % 7 == 0:
        s[n // 2 :] = 0.0  # exact zeros, as from rank-deficient matrices
    tol = 0.0 if trial % 11 == 0 else float(10 ** rng.uniform(-14, 0))
    cap = None if trial % 3 else int(rng.integers(0, n + 2))
    keep, disc = conv.keep_count(s, tol, cap)
    keep_ref, disc_ref = dr.keep_count(s, tol, cap)
    assert keep == keep_ref == _keep_by_definition(s, tol, cap)
    assert disc == pytest.approx(disc_ref, rel=1e-12, abs=1e-300)


@given(
    s=st.lists(st.floats(0, 1e6, allow_nan=False), min_size=1, max_size=30),
    tol=st.floats(0, 1),
)
def test_keep_count_bounds_property(s, tol):
    s = np.sort(np.asarray(s))[::-1]
    keep, disc = conv.keep_count(s, tol)
    assert 1 <= keep <= s.size
    assert 0.0 <= disc <= tol + 1e-12  # never discard more than allowed


@pytest.mark.parametrize("s,tol,cap", [([], 1e-8, None), ([0.0, 0.0], 1e-8, None), ([1.0, 1.0], 0.0, 0)])
def test_keep_count_edge_cases(s, tol, cap):
    assert conv.keep_count(s, tol, cap) == dr.keep_count(s, tol, cap)


@pytest.mark.parametrize(
    "s,tol,match",
    [
        ([1.0, 2.0], 0.1, "sorted"),
        ([1.0], -1.0, "tol"),
        ([float("nan")], 0.1, "finite"),
        ([1.0, -0.5], 0.1, "finite"),
        ([[1.0]], 0.1, "1-D"),
    ],
)
def test_keep_count_rejects_bad_input(s, tol, match):
    with pytest.raises(InputError, match=match):
        conv.keep_count(s, tol)
