"""Contract: cvsim.gaussian.channels against qttlind.gaussian.ops (marker ``qttlind``) and The Walrus
(marker ``thewalrus``, ħ = 2), decision D2 and CONVENTIONS.md "Comparison with other libraries"."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from harness import ROUNDOFF, assert_same, qttlind_module, random_unitary
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.errors import PhysicalityError
from cvsim.gaussian import GaussianChannel as C
from cvsim.gaussian import GaussianState
from cvsim.gaussian import symplectic as sy

FAST = settings(max_examples=40, deadline=None)
unit = st.floats(0.0, 1.0)


@st.composite
def states(draw):
    M = draw(st.integers(1, 3))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 1.0), nu=0.5 + rng.uniform(0, 1, M), mean=rng.normal(size=2 * M)
    )


def same(ours: GaussianState, mean, cov) -> None:
    scale = max(1.0, float(np.abs(ours.cov).max()))
    assert_same(ours.mean, mean, ROUNDOFF * scale, "mean")
    assert_same(ours.cov, cov, ROUNDOFF * scale, "cov")


# ------------------------------------------------------------------------------- qttlind
@pytest.mark.qttlind
@FAST
@given(s=states(), eta=unit, nbar=st.floats(0, 2), g=st.floats(1, 3), sigma2=st.floats(0, 1))
def test_against_qttlind(s, eta, nbar, g, sigma2):
    qs, qo = qttlind_module("gaussian.states"), qttlind_module("gaussian.ops")
    q = qs.GaussianState(s.mean, s.cov)
    m = s.num_modes - 1
    out = qo.loss(q, eta, mode=m, nbar=nbar)
    same(C.loss(eta, nbar).apply(s, [m]), out.mean, out.cov)
    out = qo.amplifier(q, g, mode=0, nbar=nbar)
    same(C.amplifier(g, nbar).apply(s, [0]), out.mean, out.cov)
    out = qo.thermal_noise(q, sigma2, mode=m)
    same(C.additive_noise(sigma2).apply(s, [m]), out.mean, out.cov)


@pytest.mark.qttlind
def test_cp_check_agrees_with_qttlind():
    qo = qttlind_module("gaussian.ops")
    from qttlind.errors import InputError as QInputError

    for X, Y, ok in [
        (np.sqrt(1.5) * np.eye(2), 0.25 * np.eye(2), True),
        (np.sqrt(1.5) * np.eye(2), 0.2 * np.eye(2), False),
        (2 * np.eye(2), np.zeros((2, 2)), False),
        (np.sqrt(0.6) * np.eye(2), 0.2 * np.eye(2), True),  # loss η = 0.6 into vacuum
    ]:
        if ok:
            C(X, Y)
            qo.check_channel(X, Y)
        else:
            with pytest.raises(PhysicalityError):
                C(X, Y)
            with pytest.raises(QInputError):
                qo.check_channel(X, Y)


# ------------------------------------------------------------------------------- The Walrus (ħ = 2)
@pytest.mark.thewalrus
@FAST
@given(s=states(), eta=unit, nbar=st.floats(0, 2), seed=st.integers(0, 2**32 - 1))
def test_against_the_walrus(s, eta, nbar, seed):
    pytest.importorskip("thewalrus")
    tws = importlib.import_module("thewalrus.symplectic")
    mu2, cov2 = s.to_hbar(2.0)
    m = s.num_modes - 1
    mu_tw, cov_tw = tws.loss(mu2, cov2, eta, m, nbar=nbar, hbar=2)
    same(C.loss(eta, nbar).apply(s, [m]), *GaussianState.from_hbar(mu_tw, cov_tw, hbar=2.0).to_hbar(1.0))
    rng = np.random.default_rng(seed)
    T = random_unitary(rng, s.num_modes) @ np.diag(np.sqrt(rng.uniform(0, 1, s.num_modes)))
    mu_tw, cov_tw = tws.passive_transformation(mu2, cov2, T, hbar=2)
    same(C.from_transfer(T).apply(s), *GaussianState.from_hbar(mu_tw, cov_tw, hbar=2.0).to_hbar(1.0))
