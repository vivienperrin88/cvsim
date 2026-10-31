"""Contract and reference baselines for photon counting (S10).

* qttlind (marker ``qttlind``): ``gaussian.hafnian`` (hafnian, loop hafnian, torontonian, PNR
  pattern and click probabilities) and ``gaussian.fock.fock_density_matrix``, round-off agreement.
* The Walrus (marker ``thewalrus``): ``hafnian`` (with and without loops), ``perm``, ``tor``,
  ``threshold_detection_prob``, ``quantum.probabilities`` and ``quantum.density_matrix``
  (ħ = 2 moments; its density tensor is ``(n₁, m₁, n₂, m₂, …)``).
* MrMustard (marker ``mrmustard``): ``PNRDetector`` (binomial efficiency, Poissonian dark counts)
  and ``ThresholdDetector`` applied to Gaussian states — joint distributions and heralded states
  (a second bridge process on MrMustard's TensorFlow backend: its dark-count convolution exists
  only there).
  Its threshold detector with dark clicks uses ``P(no click | n) = (1 − η)ⁿ − p_dark`` instead of
  ``(1 − η)ⁿ (1 − p_dark)`` (wrong for ``n ≥ 1``): pinned below as a strict xfail, with a test that
  reproduces its numbers from cvsim's exact photon statistics.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from harness import assert_same, mrmustard, qttlind_module, random_unitary, to_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import hafnian as hf
from cvsim.gaussian import GaussianState, ops, photon
from cvsim.gaussian import symplectic as sy

FAST = settings(max_examples=25, deadline=None)


@st.composite
def states(draw, M=None, r_max=0.6, mean_scale=0.5):
    M = draw(st.integers(1, 3)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    pure = draw(st.booleans())
    nu = 0.5 + (0.0 if pure else rng.uniform(0.0, 0.4, M))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, r_max), nu=nu, mean=mean_scale * rng.normal(size=2 * M)
    )


def sym(rng, n, cplx=True):
    A = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if cplx else 0)
    return (A + A.T) / 2


def scale(A: np.ndarray) -> float:
    """``(n−1)!! max|A|^{n/2}`` bounds every (loop) hafnian of an ``n x n`` matrix: tolerances are
    relative to it (the power-trace formula sums ``2^{n/2}`` terms of that size)."""
    n = A.shape[0] + A.shape[0] % 2
    return max(1.0, math.prod(range(n - 1, 0, -2)) * float(np.abs(A).max(initial=1.0)) ** (n // 2))


def walrus_dm(s: GaussianState, cutoff: int, tq) -> np.ndarray:
    """The Walrus density matrix reshaped to cvsim's ``(n₁ n₂ …) x (m₁ m₂ …)`` layout."""
    mu, V = to_ref(s.mean, s.cov)
    M = s.num_modes
    t = np.asarray(tq.density_matrix(mu, V, cutoff=cutoff, hbar=2.0))
    t = t.transpose([*range(0, 2 * M, 2), *range(1, 2 * M, 2)])
    return t.reshape(cutoff**M, cutoff**M)


# =============================================================================== qttlind
@pytest.fixture(scope="module")
def qh():
    return qttlind_module("gaussian.hafnian")


@pytest.fixture(scope="module")
def qf():
    return qttlind_module("gaussian.fock")


@pytest.fixture(scope="module")
def qs():
    return qttlind_module("gaussian.states")


@pytest.mark.qttlind
@FAST
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(0, 6))
def test_qttlind_matrix_functions(qh, seed, n):
    rng = np.random.default_rng(seed)
    A = sym(rng, 2 * n)
    assert_same(hf.hafnian(A), qh.hafnian(A), 1e-13 * scale(A), "hafnian")
    B = sym(rng, n + 3)
    assert_same(hf.loop_hafnian(B), qh.loop_hafnian(B), 1e-13 * scale(B), "loop hafnian (any size)")
    O = 0.3 * sym(rng, 2 * max(n, 1))
    assert_same(hf.torontonian(O), qh.torontonian(O), 1e-12 * 2 ** O.shape[0], "torontonian")


@pytest.mark.qttlind
@FAST
@given(s=states())
def test_qttlind_statistics(qh, qs, s):
    q = qs.GaussianState(s.mean, s.cov)
    M = s.num_modes
    for pattern in [(0,) * M, (1,) * M, tuple(range(M)), (2,) + (0,) * (M - 1)]:
        assert_same(photon.pattern_probability(s, pattern), qh.pattern_probability(q, pattern), 1e-12, f"p{pattern}")
    for clicks in np.ndindex(*(2,) * M):
        assert_same(photon.click_probability(s, clicks), qh.click_probability(q, clicks), 1e-12, f"click {clicks}")


@pytest.mark.qttlind
@FAST
@given(s=states(M=2))
def test_qttlind_fock_density_matrix(qf, qs, s):
    q = qs.GaussianState(s.mean, s.cov)
    assert_same(photon.fock_density_matrix(s, 6), qf.fock_density_matrix(q, 6), 1e-12, "dm")


# =============================================================================== The Walrus
@pytest.fixture(scope="module")
def tw():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus")


@pytest.fixture(scope="module")
def tq():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus.quantum")


@pytest.mark.thewalrus
@FAST
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 6))
def test_walrus_matrix_functions(tw, seed, n):
    rng = np.random.default_rng(seed)
    A = sym(rng, 2 * n)
    tol = 1e-13 * scale(A)
    assert_same(hf.hafnian(A), tw.hafnian(A), tol, "hafnian")
    assert_same(hf.hafnian(A, method="recursive"), tw.hafnian(A, method="recursive"), tol, "hafnian (recursive)")
    assert_same(hf.loop_hafnian(A), tw.hafnian(A, loop=True), tol, "loop hafnian")
    R = sym(rng, 2 * n, cplx=False)
    assert_same(hf.hafnian(R), tw.hafnian(R), 1e-13 * scale(R), "real hafnian")
    P = rng.normal(size=(n + 1, n + 1)) + 1j * rng.normal(size=(n + 1, n + 1))
    bound = math.factorial(n + 1) * float(np.abs(P).max()) ** (n + 1)
    assert_same(hf.permanent(P), tw.perm(P), 1e-13 * bound, "permanent")
    # The Walrus's recursive torontonian (Cholesky inside) needs a physical O = I − Q⁻¹
    s = GaussianState.from_symplectic(sy.random_symplectic(rng, n, 0.8), nu=0.5 + rng.uniform(0, 0.5, n))
    Q = importlib.import_module("thewalrus.quantum").Qmat(2.0 * s.cov, hbar=2.0)
    O = np.eye(2 * n) - np.linalg.inv(Q)
    assert_same(hf.torontonian(O), tw.tor(O), 1e-12 * 2**n, "torontonian")


@pytest.mark.thewalrus
@FAST
@given(s=states())
def test_walrus_pnr_probabilities(tq, s):
    mu, V = to_ref(s.mean, s.cov)
    cutoff = 5
    theirs = np.real(tq.probabilities(mu, V, cutoff, hbar=2.0))
    ours = photon.pnr_distribution(s, list(range(s.num_modes)), cutoff)
    assert_same(ours, theirs, 1e-12, "p(n)")
    for pattern in [(1,) * s.num_modes, tuple(range(s.num_modes))]:
        assert_same(photon.pattern_probability(s, pattern), theirs[pattern], 1e-12, f"p{pattern}")


@pytest.mark.thewalrus
@FAST
@given(s=states(M=2))
def test_walrus_density_matrix(tq, s):
    assert_same(photon.fock_density_matrix(s, 6), walrus_dm(s, 6, tq), 1e-12, "dm")


@pytest.mark.thewalrus
@FAST
@given(s=states())
def test_walrus_threshold(tw, s):
    mu, V = to_ref(s.mean, s.cov)
    for clicks in np.ndindex(*(2,) * s.num_modes):
        theirs = float(np.real(tw.threshold_detection_prob(mu, V, np.array(clicks), hbar=2.0)))
        assert_same(photon.click_probability(s, clicks), theirs, 1e-12, f"click {clicks}")


@pytest.mark.thewalrus
def test_walrus_heralded_state(tq):
    """Heralding = slicing The Walrus density matrix on the herald's diagonal and renormalising."""
    s = ops.displace(ops.squeeze(GaussianState.two_mode_squeezed(0.5), 0.2, 0.4, 1), 0.2 - 0.1j, 0)
    cutoff = 12  # both sides are exact matrix elements below the cutoff
    t = walrus_dm(s, cutoff, tq).reshape((cutoff,) * 4)
    for n in range(4):
        h = photon.herald(s, [0], [n], cutoff)
        block = t[n, :, n, :]
        assert_same(h.probability, np.real(np.trace(block)), 1e-12, f"p({n})")
        assert_same(h.dm, block / np.trace(block), 1e-12, f"heralded state n={n}")


# =============================================================================== MrMustard
MM_CUTOFF = 28  # MrMustard's Fock conversion of the same states, truncation error < 1e-9


def mm_state(seed: int) -> GaussianState:
    rng = np.random.default_rng(seed)
    s = GaussianState.two_mode_squeezed(0.45)
    s = ops.interferometer(s, random_unitary(rng, 2))
    s = ops.displace(s, complex(*rng.normal(0, 0.25, 2)), 0)
    return ops.squeeze(s, rng.uniform(0, 0.2), rng.uniform(-3, 3), 1)


def mm_count(kind: str, s: GaussianState, modes, **p):
    mu, V = to_ref(s.mean, s.cov)
    return mrmustard("tensorflow")(
        "count",
        kind=kind,
        cov=V,
        means=mu,
        modes=list(modes),
        cutoffs=[MM_CUTOFF] * s.num_modes,
        internal_cutoff=MM_CUTOFF,
        **p,
    )["tensor"]


@pytest.mark.mrmustard
@pytest.mark.parametrize(("seed", "eta", "dark"), [(0, 1.0, 0.0), (1, 0.7, 0.0), (2, 0.85, 0.1)])
def test_mrmustard_pnr_distribution(seed, eta, dark):
    s = mm_state(seed)
    theirs = mm_count("pnr", s, [0, 1], efficiency=eta, dark_counts=dark)
    ours = photon.pnr_distribution(s, [0, 1], MM_CUTOFF, eta, dark)
    assert_same(ours[:10, :10], theirs[:10, :10], 1e-9, f"p(n, m), η={eta}, dark={dark}")


@pytest.mark.mrmustard
@pytest.mark.parametrize(("seed", "eta", "dark"), [(3, 1.0, 0.0), (4, 0.6, 0.05)])
def test_mrmustard_pnr_herald(seed, eta, dark):
    """MrMustard returns ``(outcome, kets, bras)`` unnormalised conditional states."""
    s = mm_state(seed)
    theirs = mm_count("pnr", s, [0], efficiency=eta, dark_counts=dark)
    for n in range(3):
        h = photon.herald(s, [0], [n], MM_CUTOFF, eta, dark)
        block = theirs[n]
        assert_same(h.probability, np.real(np.trace(block)), 1e-9, f"p({n})")
        assert_same(h.dm[:12, :12], (block / np.trace(block))[:12, :12], 1e-8, f"heralded state n={n}")


@pytest.mark.mrmustard
@pytest.mark.parametrize(("seed", "eta"), [(5, 1.0), (6, 0.55)])
def test_mrmustard_threshold(seed, eta):
    s = mm_state(seed)
    theirs = mm_count("threshold", s, [0, 1], efficiency=eta, dark_prob=0.0)
    assert_same(photon.click_distribution(s, [0, 1], eta), theirs, 1e-9, f"clicks, η={eta}")


@pytest.mark.mrmustard
def test_mrmustard_threshold_dark_formula_reproduced():
    """MrMustard's dark-click kernel, ``P(no click | n) = (1 − η)ⁿ − p``, applied to cvsim's exact
    photon statistics reproduces its numbers: the difference below is that kernel, nothing else."""
    s, eta, p = mm_state(7), 0.8, 0.05
    theirs = mm_count("threshold", s, [0, 1], efficiency=eta, dark_prob=p)
    n = np.arange(MM_CUTOFF)
    quiet = (1 - eta) ** n - p
    K = np.stack([quiet, 1 - quiet])
    P = photon.pnr_distribution(s, [0, 1], MM_CUTOFF)
    assert_same(K @ P @ K.T, theirs, 1e-9, "MrMustard kernel")


@pytest.mark.mrmustard
@pytest.mark.xfail(
    strict=True, reason="MrMustard 0.7.3 ThresholdDetector: P(no click|n) = (1-η)^n - p_dark, not (1-η)^n (1-p_dark)"
)
def test_mrmustard_threshold_dark_counts():
    s, eta, p = mm_state(7), 0.8, 0.05
    theirs = mm_count("threshold", s, [0, 1], efficiency=eta, dark_prob=p)
    assert_same(photon.click_distribution(s, [0, 1], eta, p), theirs, 1e-6, "clicks with dark counts")


def test_dark_click_model_is_independent_noise():
    """cvsim's model: a dark click is an independent Bernoulli(p) event OR-ed with the photon click."""
    s, eta, p = mm_state(8), 0.8, 0.05
    P = photon.pnr_distribution(s, [0, 1], 24, eta)  # tail beyond 24 photons < 1e-12
    quiet = (1 - p) * np.eye(24)[0]
    K = np.stack([quiet, 1 - quiet])
    assert_same(photon.click_distribution(s, [0, 1], eta, p), K @ P @ K.T, 1e-10, "independent dark clicks")
    assert math.isclose(photon.click_distribution(GaussianState.vacuum(2), [0, 1], 1.0, p)[1, 1], p * p)
