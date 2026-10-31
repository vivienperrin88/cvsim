"""Contract: cvsim.gaussian.measure against qttlind.gaussian.measure (decision D2).

Ideal homodyne (any angle) and heterodyne with imposed outcomes, on Hypothesis-generated
multimode states: outcome densities and conditional states agree to round-off. qttlind has no
detector model or general-dyne; those are checked against explicit optical models in
``tests/test_measure.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from harness import ROUNDOFF, assert_same, qttlind_module
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.gaussian import GaussianState, measure
from cvsim.gaussian import symplectic as sy

pytestmark = pytest.mark.qttlind
FAST = settings(max_examples=40, deadline=None)


@st.composite
def states(draw):
    M = draw(st.integers(1, 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 1.0), nu=0.5 + rng.uniform(0, 0.8, M), mean=rng.normal(size=2 * M)
    )


@pytest.fixture(scope="module")
def q():
    return qttlind_module("gaussian.states"), qttlind_module("gaussian.measure")


def same_state(ours, theirs):
    if ours is None:
        assert theirs is None
        return
    scale = max(1.0, float(np.abs(ours.cov).max()))
    assert_same(ours.mean, theirs.mean, ROUNDOFF * scale, "conditional mean")
    assert_same(ours.cov, theirs.cov, ROUNDOFF * scale, "conditional cov")


@FAST
@given(s=states(), phi=st.floats(-3.2, 3.2), out=st.floats(-2, 2), data=st.data())
def test_homodyne(q, s, phi, out, data):
    qs, qm = q
    mode = data.draw(st.integers(0, s.num_modes - 1))
    ours = measure.homodyne(s, mode, phi, outcome=out)
    theirs = qm.homodyne(qs.GaussianState(s.mean, s.cov), mode=mode, phi=phi, outcome=out)
    assert math.isclose(ours.density, theirs.density, rel_tol=1e-11, abs_tol=1e-300)
    same_state(ours.state, theirs.state)
    grid = np.linspace(-3, 3, 7)
    assert_same(
        measure.homodyne_density(s, grid, mode, phi),
        qm.homodyne_density(qs.GaussianState(s.mean, s.cov), grid, mode, phi),
        1e-12,
        "density",
    )


@FAST
@given(s=states(), re=st.floats(-1.5, 1.5), im=st.floats(-1.5, 1.5), data=st.data())
def test_heterodyne(q, s, re, im, data):
    qs, qm = q
    mode = data.draw(st.integers(0, s.num_modes - 1))
    alpha = complex(re, im)
    ours = measure.heterodyne(s, mode, outcome=alpha)
    theirs = qm.heterodyne(qs.GaussianState(s.mean, s.cov), mode=mode, outcome=alpha)
    assert ours.outcome == pytest.approx(theirs.outcome)
    assert math.isclose(ours.density, theirs.density, rel_tol=1e-11, abs_tol=1e-300)
    same_state(ours.state, theirs.state)
