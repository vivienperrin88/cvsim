"""Reference baseline: The Walrus 0.22 for the Gaussian core (marker ``thewalrus``, ħ = 2).

States and figures of merit (``thewalrus.quantum``), gate matrices (``thewalrus.symplectic``), loss
and photon numbers. The decompositions are compared in ``test_contract_thewalrus.py``. The
two-mode squeezing phase follows the Strawberry Fields family: ``φ_TW = φ_cvsim + π``.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from harness import HBAR_REF, assert_same, random_unitary, to_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.errors import PhysicalityError
from cvsim.gaussian import GaussianChannel as C
from cvsim.gaussian import GaussianState
from cvsim.gaussian import symplectic as sy

pytestmark = pytest.mark.thewalrus
FAST = settings(max_examples=40, deadline=None)


@pytest.fixture(scope="module")
def tq():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus.quantum")


@pytest.fixture(scope="module")
def tws():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus.symplectic")


@st.composite
def states(draw, M=None):
    M = draw(st.integers(1, 4)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    pure = draw(st.booleans())
    nu = 0.5 + (0.0 if pure else rng.uniform(0.05, 1.0, M))
    return GaussianState.from_symplectic(sy.random_symplectic(rng, M, 1.0), nu=nu, mean=rng.normal(size=2 * M))


# ------------------------------------------------------------------------------- gates
def test_gate_matrices(tws):
    assert_same(tws.squeezing(0.3, 0.5), cv.squeezing(0.3, 0.5), 1e-15, "S")
    assert_same(tws.rotation(0.7), cv.rotation(0.7), 1e-15, "R")
    assert_same(tws.beam_splitter(0.4, 0.9), cv.beamsplitter(0.4, 0.9), 1e-15, "BS")
    assert_same(tws.two_mode_squeezing(0.3, 0.7 + math.pi), cv.two_mode_squeezing(0.3, 0.7), 1e-15, "S2 (φ + π)")
    assert not np.allclose(tws.two_mode_squeezing(0.3, 0.7), cv.two_mode_squeezing(0.3, 0.7))
    assert_same(
        tws.squeezing([0.2, 0.4], [0.1, -0.3]),
        sy.embed(cv.squeezing(0.2, 0.1), [0], 2) @ sy.embed(cv.squeezing(0.4, -0.3), [1], 2),
        1e-15,
        "S ⊗ S",
    )


@FAST
@given(seed=st.integers(0, 2**32 - 1), N=st.integers(1, 5))
def test_interferometer_and_expand(tws, seed, N):
    rng = np.random.default_rng(seed)
    U = random_unitary(rng, N)
    assert_same(tws.interferometer(U), cv.interferometer(U), 1e-14, "interferometer")
    if N >= 2:
        assert_same(
            tws.expand(cv.beamsplitter(0.3, 0.2), [N - 1, 0], N),
            sy.embed(cv.beamsplitter(0.3, 0.2), [N - 1, 0], N),
            1e-15,
            "expand",
        )


def test_vacuum_and_displacement(tws):
    mu, cov = tws.vacuum_state(3, hbar=HBAR_REF)
    assert_same(cov / HBAR_REF, GaussianState.vacuum(3).cov, 1e-15, "vacuum")
    assert_same(mu, np.zeros(6), 0.0, "vacuum mean")
    assert_same(
        tws.expand_vector(0.3 - 0.2j, 1, 2, hbar=HBAR_REF) / math.sqrt(HBAR_REF),
        GaussianState.coherent([0, 0.3 - 0.2j]).mean,
        1e-15,
        "α",
    )


# ------------------------------------------------------------------------------- figures of merit
@FAST
@given(s=states(), data=st.data())
def test_figures_of_merit(tq, s, data):
    t = data.draw(states(M=s.num_modes))
    m, c = to_ref(s.mean, s.cov)
    m2, c2 = to_ref(t.mean, t.cov)
    assert math.isclose(tq.fidelity(m, c, m2, c2, hbar=HBAR_REF), s.fidelity(t), rel_tol=1e-8, abs_tol=1e-13)
    assert math.isclose(tq.vonneumann_entropy(c, hbar=HBAR_REF), s.entropy(), rel_tol=1e-9, abs_tol=1e-10)
    scale = max(1.0, float(s.mean_photons().max())) ** 2
    assert_same(tq.photon_number_mean_vector(m, c, hbar=HBAR_REF), s.mean_photons(), 1e-11 * scale, "⟨n⟩")
    assert_same(tq.photon_number_covmat(m, c, hbar=HBAR_REF), s.photon_covariance(), 1e-10 * scale, "Cov(n)")
    assert tq.is_pure_cov(c, hbar=HBAR_REF) == s.is_pure()
    assert math.isclose(
        float(np.real(tq.density_matrix_element(m, c, [0] * s.num_modes, [0] * s.num_modes, hbar=HBAR_REF))),
        s.vacuum_probability(),
        rel_tol=1e-9,
        abs_tol=1e-15,
    )
    keep = [s.num_modes - 1, 0] if s.num_modes >= 3 else [s.num_modes - 1]  # a strict subset (see below)
    mr, cr = tq.reduced_gaussian(m, c, keep)
    red = s.reduce(keep)
    assert_same(cr / HBAR_REF, red.cov, 1e-14, "reduced") and assert_same(
        mr / math.sqrt(HBAR_REF), red.mean, 1e-14, "reduced"
    )


@FAST
@given(s=states(M=2))
def test_log_negativity_in_nats(tq, s):
    c = to_ref(s.mean, s.cov)[1]
    assert math.isclose(
        tq.log_negativity(c, [0], hbar=HBAR_REF), s.log_negativity([0], base=math.e), rel_tol=1e-9, abs_tol=1e-12
    )


def test_reduced_gaussian_ignores_the_order_of_a_full_mode_list(tq):
    """Documented difference: The Walrus returns the state unchanged when every mode is listed, so
    ``reduced_gaussian(…, [1, 0])`` does not swap the modes; cvsim's ``reduce([1, 0])`` does."""
    s = GaussianState.coherent([0.1, 0.5j])
    m, c = to_ref(s.mean, s.cov)
    mr, _ = tq.reduced_gaussian(m, c, [1, 0])
    assert_same(mr / math.sqrt(HBAR_REF), s.mean, 1e-15, "unchanged")
    assert not np.allclose(s.reduce([1, 0]).mean, s.mean)


def test_validity_checks_agree(tq):
    good = GaussianState.squeezed(0.4).cov
    bad = np.diag([0.2, 0.5])
    assert tq.is_valid_cov(good * HBAR_REF, hbar=HBAR_REF)
    assert not tq.is_valid_cov(bad * HBAR_REF, hbar=HBAR_REF)
    with pytest.raises(PhysicalityError):
        GaussianState(np.zeros(2), bad)


# ------------------------------------------------------------------------------- channels
@FAST
@given(s=states(M=3), eta=st.floats(0.0, 1.0))
def test_photon_number_after_loss(tws, s, eta):
    out = C.loss(eta).apply(s, [1])
    m, c = to_ref(out.mean, out.cov)
    mean_n, var_n = tws.mean_photon_number(*tws.reduced_state(m, c, 1), hbar=HBAR_REF)
    assert math.isclose(mean_n, out.mean_photons()[1], rel_tol=1e-10, abs_tol=1e-12)
    assert math.isclose(var_n, out.photon_covariance()[1, 1], rel_tol=1e-9, abs_tol=1e-12)
