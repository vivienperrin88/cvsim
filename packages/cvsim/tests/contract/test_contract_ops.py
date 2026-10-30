"""Contract: cvsim.gaussian.ops against qttlind.gaussian.ops (decision D2).

Both libraries apply the same gate to the same state; the moments must agree to round-off on
Hypothesis-generated multimode states. Gates only qttlind lacks (CZ networks, CX, shear, Fourier)
are checked against the matrices of ``core.conventions`` in ``tests/test_symplectic.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from harness import ROUNDOFF, assert_same, qttlind_module, random_unitary
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.gaussian import GaussianState, ops
from cvsim.gaussian import symplectic as sy

pytestmark = pytest.mark.qttlind
FAST = settings(max_examples=40, deadline=None)
angles = st.floats(-3.2, 3.2)
small = st.floats(-1.0, 1.0)


@pytest.fixture(scope="module")
def qs():
    return qttlind_module("gaussian.states")


@pytest.fixture(scope="module")
def qo():
    return qttlind_module("gaussian.ops")


@st.composite
def states(draw):
    M = draw(st.integers(2, 4))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    nu = 0.5 + rng.uniform(0, 1.0, M) * draw(st.sampled_from([0.0, 1.0]))
    return GaussianState.from_symplectic(sy.random_symplectic(rng, M, 1.0), nu=nu, mean=rng.normal(size=2 * M))


def same(ours: GaussianState, theirs) -> None:
    scale = max(1.0, float(np.abs(ours.cov).max()))
    assert_same(ours.mean, theirs.mean, ROUNDOFF * scale, "mean")
    assert_same(ours.cov, theirs.cov, ROUNDOFF * scale, "cov")


@FAST
@given(s=states(), r=small, a=angles, b=angles, z=st.complex_numbers(max_magnitude=2.0))
def test_single_and_two_mode_gates(qs, qo, s, r, a, b, z):
    q = qs.GaussianState(s.mean, s.cov)
    last = s.num_modes - 1
    same(ops.displace(s, z, modes=last), qo.displace(q, z, mode=last))
    same(ops.rotate(s, a, mode=1), qo.rotate(q, a, mode=1))
    same(ops.squeeze(s, r, b, mode=0), qo.squeeze(q, r, b, mode=0))
    same(ops.beamsplitter(s, a, b, modes=(last, 0)), qo.beamsplitter(q, a, b, modes=(last, 0)))
    same(ops.two_mode_squeeze(s, r, a, modes=(0, 1)), qo.two_mode_squeeze(q, r, a, modes=(0, 1)))


@FAST
@given(s=states(), seed=st.integers(0, 2**32 - 1))
def test_interferometer_and_symplectic(qs, qo, s, seed):
    rng = np.random.default_rng(seed)
    q = qs.GaussianState(s.mean, s.cov)
    U = random_unitary(rng, s.num_modes)
    same(ops.interferometer(s, U), qo.interferometer(q, U))
    sub = [s.num_modes - 1, 0]
    S = sy.random_symplectic(rng, 2, 0.8)
    same(ops.apply_symplectic(s, S, sub), qo.apply_symplectic(q, S, sub))


@FAST
@given(seed=st.integers(0, 2**32 - 1), N=st.integers(2, 6))
def test_clements_mesh_runs_on_qttlind(qs, qo, seed, N):
    """cvsim's mesh, executed gate by gate with qttlind's rotate/beamsplitter, equals the interferometer."""
    rng = np.random.default_rng(seed)
    U = random_unitary(rng, N)
    s = GaussianState.from_symplectic(sy.random_symplectic(rng, N, 0.8), mean=rng.normal(size=2 * N))
    q = qs.GaussianState(s.mean, s.cov)
    for name, par, modes in sy.clements(U).gates():
        q = qo.rotate(q, par[0], mode=modes[0]) if name == "R" else qo.beamsplitter(q, *par, modes=modes)
    same(ops.interferometer(s, U), q)
