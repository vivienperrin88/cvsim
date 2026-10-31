"""Reference baseline: MrMustard 0.7.3 (marker ``mrmustard``), through the bridge of ``harness``.

MrMustard runs in its own environment (it pins NumPy < 2); every quantity is converted from its
ħ = 2 to cvsim's ħ = 1. Conventions that differ by design are mapped explicitly (two-mode
squeezing phase ``φ_MM = φ_cvsim + π``; Coherent/Dgate ``x + iy = α``; homodyne outcomes in units of
``√ħ``). Where MrMustard disagrees with the Fock oracle, the comparison is an ``xfail(strict=True)``
test naming the upstream problem, next to a test showing cvsim agrees with the oracle: if MrMustard
is fixed, the strict xfail fails and the entry in ``docs/design/equivalence.md`` must be updated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from harness import HBAR_REF, ROUNDOFF, assert_same, from_ref, mrmustard, random_unitary, to_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.core import conventions as cv
from cvsim.gaussian import GaussianChannel as C
from cvsim.gaussian import GaussianState, measure
from cvsim.gaussian import symplectic as sy
from cvsim.reference import fock_dense as fd

pytestmark = pytest.mark.mrmustard
FAST = settings(max_examples=25, deadline=None)


@st.composite
def states(draw, M=None):
    M = draw(st.integers(1, 3)) if M is None else M
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return GaussianState.from_symplectic(
        sy.random_symplectic(rng, M, 1.0), nu=0.5 + rng.uniform(0, 0.8, M), mean=rng.normal(size=2 * M)
    )


def mm_state(s: GaussianState) -> dict:
    m, c = to_ref(s.mean, s.cov)
    return {"cov": c, "means": m}


def same_moments(mean, cov, s: GaussianState, atol: float = ROUNDOFF) -> None:
    scale = max(1.0, float(np.abs(s.cov).max()))
    assert_same(mean, s.mean, atol * scale, "mean")
    assert_same(cov, s.cov, atol * scale, "cov")


# ------------------------------------------------------------------------------- gates
@pytest.mark.parametrize(
    ("name", "params", "ours"),
    [
        ("rotation", {"theta": 0.7}, cv.rotation(0.7)),
        ("squeezing", {"r": 0.3, "phi": 0.5}, cv.squeezing(0.3, 0.5)),
        ("beam_splitter", {"theta": 0.4, "phi": 0.9}, cv.beamsplitter(0.4, 0.9)),
        ("two_mode_squeezing", {"r": 0.3, "phi": 0.7}, cv.two_mode_squeezing(0.3, 0.7 + math.pi)),
        ("quadratic_phase", {"s": 0.4}, cv.shear(0.4)),
        ("controlled_z", {"g": 0.4}, cv.cz(np.array([[0.0, 0.4], [0.4, 0.0]]))),
        ("controlled_x", {"g": 0.4}, cv.cx(0.4)),
    ],
)
def test_gate_matrices(name, params, ours):
    assert_same(mrmustard()("gate", name=name, **params), ours, 1e-14, name)


def test_two_mode_squeezing_phase_convention():
    """S2_cvsim(r, φ) = S2_MrMustard(r, φ + π): the generators differ by a sign (CONVENTIONS.md)."""
    mm = mrmustard()
    assert not np.allclose(mm("gate", name="two_mode_squeezing", r=0.3, phi=0.0), cv.two_mode_squeezing(0.3, 0.0))
    assert_same(mm("gate", name="two_mode_squeezing", r=0.3, phi=math.pi), cv.two_mode_squeezing(0.3, 0.0), 1e-14, "S2")


@FAST
@given(seed=st.integers(0, 2**32 - 1), N=st.integers(1, 5))
def test_interferometer(seed, N):
    U = random_unitary(np.random.default_rng(seed), N)
    assert_same(mrmustard()("interferometer", U=U), cv.interferometer(U), 1e-14, "interferometer")


def test_displacement_units():
    """Dgate(x, y) is α = x + iy: d = √(2ħ)(x, y)."""
    assert_same(
        mrmustard()("gate", name="displacement", x=0.3, y=-0.2) / math.sqrt(HBAR_REF),
        cv.displacement_vector(0.3 - 0.2j),
        1e-15,
        "D",
    )


# ------------------------------------------------------------------------------- states
@pytest.mark.parametrize(
    ("name", "params", "ours"),
    [
        ("vacuum", {"num_modes": 2}, GaussianState.vacuum(2)),
        ("coherent", {"x": 0.3, "y": -0.2}, GaussianState.coherent(0.3 - 0.2j)),
        ("squeezed", {"r": 0.4, "phi": 0.6}, GaussianState.squeezed(0.4, 0.6)),
        ("tmsv", {"r": 0.4, "phi": 0.6 + math.pi}, GaussianState.two_mode_squeezed(0.4, 0.6)),
        ("thermal", {"nbar": 0.7}, GaussianState.thermal(0.7)),
        ("displaced_squeezed", {"r": 0.3, "phi": 0.2, "x": 0.1, "y": 0.4}, None),
    ],
)
def test_states(name, params, ours):
    st_ = mrmustard()("state", name=name, **params)
    mean, cov = from_ref(st_["means"], st_["cov"])
    if ours is None:  # D(α) S(r, φ)|0⟩
        ours = GaussianState.from_symplectic(cv.squeezing(0.3, 0.2), mean=cv.displacement_vector(0.1 + 0.4j))
    same_moments(mean, cov, ours, 1e-14)


# ------------------------------------------------------------------------------- figures of merit
@FAST
@given(s=states(), data=st.data())
def test_purity_fidelity_photon_means(s, data):
    t = data.draw(states(M=s.num_modes))
    f = mrmustard()("figures", **mm_state(s), cov2=to_ref(t.mean, t.cov)[1], means2=to_ref(t.mean, t.cov)[0])
    assert math.isclose(f["purity"], s.purity(), rel_tol=1e-10)
    assert math.isclose(f["fidelity"], s.fidelity(t), rel_tol=1e-8, abs_tol=1e-14)  # MrMustard: matrix sqrtm
    assert_same(f["number_means"], s.mean_photons(), 1e-11 * max(1.0, float(s.mean_photons().max())), "photons")


@FAST
@given(s=states(M=1))
def test_single_mode_entropy_and_photon_variance(s):
    f = mrmustard()("figures", **mm_state(s))
    # MrMustard's g(ν) has a regulariser: xlogy(ν−, ν− + 1e-9), a bias of ~1e-9 per mode
    assert math.isclose(f["entropy"], s.entropy(), rel_tol=1e-9, abs_tol=2e-9)
    assert math.isclose(float(f["number_cov"][0, 0]), float(s.photon_covariance()[0, 0]), rel_tol=1e-10)


def _two_mode(seed: int) -> GaussianState:
    rng = np.random.default_rng(seed)
    return GaussianState.from_symplectic(sy.random_symplectic(rng, 2, 1.0), nu=0.5 + rng.uniform(0, 0.8, 2))


@pytest.mark.parametrize("seed", [0, 1, 4, 5, 7, 9, 10])
def test_log_negativity_two_modes(seed):
    s = _two_mode(seed)
    f = mrmustard()("figures", **mm_state(s), transpose_modes=[0])
    assert math.isclose(f["log_negativity"], s.log_negativity([0]), rel_tol=1e-9, abs_tol=1e-12)


def test_log_negativity_of_lossy_tmsv():
    s = C.loss(0.7).apply(GaussianState.two_mode_squeezed(0.6), [1])
    f = mrmustard()("figures", **mm_state(s), transpose_modes=[0])
    assert math.isclose(f["log_negativity"], s.log_negativity([0]), rel_tol=1e-10)


# --- known upstream discrepancies (MrMustard 0.7.3), each arbitrated by the Fock oracle ---------------
@pytest.mark.xfail(
    strict=True, reason="MrMustard symplectic_eigenvals pairs eig(iΩV)[::2], which LAPACK does not order in ± pairs"
)
def test_mrmustard_multimode_symplectic_eigenvalues():
    s = _two_mode(0)  # MrMustard returns (0.838, 0.838) for (0.740, 0.838)
    f = mrmustard()("figures", **mm_state(s))
    assert_same(np.sort(f["symplectic_eigenvals"]) / HBAR_REF, s.symplectic_eigenvalues(), 1e-9, "ν")


@pytest.mark.xfail(strict=True, reason="MrMustard von_neumann_entropy inherits the eigenvalue pairing problem")
def test_mrmustard_multimode_entropy():
    s = _two_mode(0)
    assert math.isclose(mrmustard()("figures", **mm_state(s))["entropy"], s.entropy(), rel_tol=1e-6)


def test_cvsim_log_negativity_matches_the_walrus_where_mrmustard_does_not():
    tq = pytest.importorskip("thewalrus.quantum")
    s = _two_mode(32)
    expected = tq.log_negativity(to_ref(s.mean, s.cov)[1], [0], hbar=HBAR_REF) / math.log(2)  # nats -> bits
    assert math.isclose(s.log_negativity([0]), expected, rel_tol=1e-10) and expected > 0.4


@pytest.mark.xfail(strict=True, reason="MrMustard log_negativity: same eigenvalue pairing problem (returns 0 here)")
def test_mrmustard_log_negativity_seed_32():
    s = _two_mode(32)
    f = mrmustard()("figures", **mm_state(s), transpose_modes=[0])
    assert math.isclose(f["log_negativity"], s.log_negativity([0]), rel_tol=1e-9)


def _displaced_tms():
    N = 22
    psi = (
        fd.displacement(0.3, 2, N, mode=1)
        @ fd.displacement(0.2j, 2, N, mode=0)
        @ fd.two_mode_squeezing(0.3, 0.4, 2, N)
        @ fd.vacuum(2, N)
    )
    rho = fd.dm(psi)
    a = fd.mode_operators(2, N)
    n = [x.conj().T @ x for x in a]
    mean = [np.trace(rho @ k).real for k in n]
    cov = np.array([[np.trace(rho @ x @ y).real - mean[i] * mean[j] for j, y in enumerate(n)] for i, x in enumerate(n)])
    return GaussianState(*fd.moments(rho, 2, N)), cov


def test_cvsim_photon_covariance_matches_the_oracle_on_displaced_correlated_modes():
    s, oracle = _displaced_tms()
    assert_same(s.photon_covariance(), oracle, 1e-10, "Cov(n_j, n_k)")


@pytest.mark.xfail(strict=True, reason="MrMustard number_cov: off-diagonal displacement term is half of d_J V_JK d_K")
def test_mrmustard_photon_covariance_on_displaced_correlated_modes():
    s, oracle = _displaced_tms()
    assert_same(mrmustard()("figures", **mm_state(s))["number_cov"], oracle, 1e-8, "Cov(n_j, n_k)")


# ------------------------------------------------------------------------------- channels
@pytest.mark.parametrize(
    ("name", "params", "ours"),
    [
        ("loss", {"eta": 0.7, "nbar": 0.3}, C.loss(0.7, 0.3)),
        ("amplifier", {"gain": 1.4, "nbar": 0.3}, C.amplifier(1.4, 0.3)),
        ("noise", {"noise": 0.2}, C.additive_noise(0.1)),  # MrMustard: Y = noise · ħ/2
    ],
)
def test_channels(name, params, ours):
    ch = mrmustard()("channel", name=name, **params)
    assert_same(ch["X"], ours.X, 1e-15, f"{name} X")
    assert_same(ch["Y"] / HBAR_REF, ours.Y, 1e-15, f"{name} Y")


@FAST
@given(s=states(M=3), eta=st.floats(0.0, 1.0), nbar=st.floats(0.0, 1.0))
def test_channel_application(s, eta, nbar):
    ch = C.loss(eta, nbar)
    X, Y = ch.X, ch.Y * HBAR_REF
    r = mrmustard()("cptp", **mm_state(s), X=X, Y=Y, state_modes=[0, 1, 2], transf_modes=[1])
    same_moments(*from_ref(r["means"], r["cov"]), ch.apply(s, [1]))


def test_composition_with_a_diagonal_second_channel():
    a, b = C.loss(0.7, 0.2), C.amplifier(1.3)
    r = mrmustard()("compose", X1=a.X, Y1=a.Y, X2=b.X, Y2=b.Y)
    c = a.then(b)
    assert_same(r["X"], c.X, 1e-15, "X") and assert_same(r["Y"], c.Y, 1e-15, "Y")


@pytest.mark.xfail(strict=True, reason="MrMustard compose_channels_XYd uses X2 Y1 X2 instead of X2 Y1 X2ᵀ")
def test_mrmustard_composition_with_a_rotation():
    a, b = C.additive_noise(np.diag([0.1, 0.3])), C.from_symplectic(cv.rotation(0.6))
    r = mrmustard()("compose", X1=a.X, Y1=a.Y, X2=b.X, Y2=b.Y)
    assert_same(r["Y"], a.then(b).Y, 1e-12, "Y")


# ------------------------------------------------------------------------------- measurements
@FAST
@given(s=states(M=3), re=st.floats(-1, 1), im=st.floats(-1, 1))
def test_heterodyne(s, re, im):
    alpha = complex(re, im)
    ours = measure.heterodyne(s, 2, outcome=alpha)
    scale = math.sqrt(2 * HBAR_REF)  # Heterodyne(x, y) takes the means of |α⟩: √(2ħ)(Re α, Im α)
    r = mrmustard()("lab_measure", kind="heterodyne", **mm_state(s), modes=[2], x=scale * re, y=scale * im)
    same_moments(*from_ref(r["means"], r["cov"]), ours.state)
    # MrMustard reports ⟨α|ρ|α⟩, cvsim the density per d²α: a factor π
    assert math.isclose(r["norm"], math.pi * ours.density, rel_tol=1e-10)


@FAST
@given(s=states(M=3), phi=st.floats(-3, 3), q=st.floats(-1.5, 1.5))
def test_homodyne(s, phi, q):
    """MrMustard projects on a displaced squeezed state with r = settings.HOMODYNE_SQUEEZING (10):
    conditional states agree to O(e^{−2r}); the reported number is that overlap, ``p(q) √(4π) e^{−r}``."""
    mm = mrmustard()
    r_hom = mm.versions["homodyne_squeezing"]
    ours = measure.homodyne(s, 1, phi, outcome=q)
    r = mm("lab_measure", kind="homodyne", **mm_state(s), modes=[1], angle=phi, result=q * math.sqrt(HBAR_REF))
    tol = 50 * math.exp(-2 * r_hom) * max(1.0, float(np.abs(s.cov).max())) ** 2
    same_moments(*from_ref(r["means"], r["cov"]), ours.state, atol=tol)
    assert math.isclose(r["norm"], ours.density * math.sqrt(4 * math.pi) * math.exp(-r_hom), rel_tol=1e-6)
