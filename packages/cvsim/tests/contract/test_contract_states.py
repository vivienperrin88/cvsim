"""Contract: cvsim.gaussian.GaussianState against qttlind.gaussian.states / .measure (decision D2).

Same constructors give the same moments to a few ulps; every figure of merit both libraries
compute agrees to round-off on Hypothesis-generated multimode states. qttlind's ``fidelity``
exists only when one state is pure; cvsim's general mixed-state fidelity is checked against it
there and against closed forms in ``tests/test_gaussian_states.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from harness import EXACT, ROUNDOFF, assert_same, qttlind_module, random_unitary
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.gaussian import GaussianState

pytestmark = pytest.mark.qttlind
FAST = settings(max_examples=60, deadline=None)


@pytest.fixture(scope="module")
def qs():
    return qttlind_module("gaussian.states")


@pytest.fixture(scope="module")
def qm():
    return qttlind_module("gaussian.measure")


@st.composite
def states(draw, M=None, pure=None):
    M = draw(st.integers(1, 4)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    r = rng.uniform(0, 1.0, M)
    S = cv.interferometer(random_unitary(rng, M)) @ np.diag(np.exp(np.concatenate([-r, r])))
    S = S @ cv.interferometer(random_unitary(rng, M))
    pure = draw(st.booleans()) if pure is None else pure
    nu = 0.5 + (0.0 if pure else rng.uniform(0.0, 1.5, M))
    return GaussianState.from_symplectic(S, nu=nu, mean=rng.normal(size=2 * M))


def as_qttlind(qs, s: GaussianState):
    return qs.GaussianState(s.mean, s.cov)


# ------------------------------------------------------------------------------- constructors
def test_constructors(qs):
    pairs = [
        (GaussianState.vacuum(3), qs.GaussianState.vacuum(3)),
        (GaussianState.coherent([0.3 - 0.1j, 1.2j]), qs.GaussianState.coherent([0.3 - 0.1j, 1.2j])),
        (GaussianState.thermal([0.2, 3.0]), qs.GaussianState.thermal([0.2, 3.0])),
        (GaussianState.squeezed(0.7, 1.3), qs.GaussianState.squeezed_vacuum(0.7, 1.3)),
        (
            GaussianState.squeezed([0.4, 0.2], [0.1, -0.5]),
            qs.GaussianState.product(
                qs.GaussianState.squeezed_vacuum(0.4, 0.1), qs.GaussianState.squeezed_vacuum(0.2, -0.5)
            ),
        ),
    ]
    for ours, theirs in pairs:
        assert_same(ours.mean, theirs.mean, EXACT, "mean")
        assert_same(ours.cov, theirs.cov, EXACT, "cov")


@FAST
@given(a=states(), b=states())
def test_product_and_reduce(qs, a, b):
    ours = a.tensor(b)
    theirs = qs.GaussianState.product(as_qttlind(qs, a), as_qttlind(qs, b))
    assert_same(ours.cov, theirs.cov, EXACT, "product cov")
    keep = list(range(ours.num_modes))[::-2]
    assert_same(ours.reduce(keep).cov, theirs.reduce(keep).cov, EXACT, "reduced cov")
    assert_same(ours.reduce(keep).mean, theirs.reduce(keep).mean, EXACT, "reduced mean")


# ------------------------------------------------------------------------------- figures of merit
@FAST
@given(s=states())
def test_purity_photons_entropy(qs, s):
    q = as_qttlind(qs, s)
    assert math.isclose(s.purity(), q.purity(), rel_tol=ROUNDOFF)
    assert_same(s.mean_photons(), q.mean_photons(), ROUNDOFF * max(1.0, float(s.mean_photons().max())), "photons")
    assert math.isclose(s.entropy(), q.entropy(), rel_tol=1e-10, abs_tol=1e-10)
    assert s.is_pure() == q.is_pure()


@FAST
@given(s=states(), data=st.data())
def test_overlap_and_vacuum_probability(qs, qm, s, data):
    t = data.draw(states(M=s.num_modes))
    q, r = as_qttlind(qs, s), as_qttlind(qs, t)
    assert math.isclose(s.overlap(t), qm.overlap(q, r), rel_tol=1e-11, abs_tol=1e-300)
    assert math.isclose(s.vacuum_probability(), qm.vacuum_probability(q), rel_tol=1e-11, abs_tol=1e-300)
    assert math.isclose(s.vacuum_probability([0]), qm.vacuum_probability(q, [0]), rel_tol=1e-11, abs_tol=1e-300)


@FAST
@given(s=states(pure=True), data=st.data())
def test_fidelity_with_a_pure_state(qs, qm, s, data):
    t = data.draw(states(M=s.num_modes))
    expected = qm.fidelity(as_qttlind(qs, s), as_qttlind(qs, t))
    assert math.isclose(s.fidelity(t), expected, rel_tol=1e-11, abs_tol=1e-300)
    assert math.isclose(t.fidelity(s), expected, rel_tol=1e-11, abs_tol=1e-300)


@FAST
@given(s=states(M=2))
def test_log_negativity(qs, qm, s):
    q = as_qttlind(qs, s)
    assert math.isclose(s.log_negativity([0]), qm.log_negativity(q, [0]), rel_tol=1e-10, abs_tol=1e-12)


@FAST
@given(s=states())
def test_wigner(qs, qm, s):
    x, p = np.linspace(-2, 2, 7), np.linspace(-1.5, 2.5, 5)
    mode = s.num_modes - 1
    expected = qm.wigner(as_qttlind(qs, s), x, p, mode=mode)
    assert_same(s.wigner_grid(x, p, mode=mode), expected, ROUNDOFF * max(1.0, float(np.abs(expected).max())), "Wigner")
