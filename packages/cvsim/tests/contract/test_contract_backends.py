"""Contract and reference baselines for the backends (S09).

* qttlind (marker ``qttlind``): ``GaussianBackend("qttlind")`` against ``GaussianBackend("cvsim")``
  on Hypothesis-generated programs (gates, channels, post-selected measurements with detector
  imperfections), round-off agreement.
* MrMustard (marker ``mrmustard``): the same circuits translated to MrMustard's lab gates
  (Gaussian moments, round-off), its Gaussian-to-Fock conversion and its own Fock simulation
  against the Fock backend.
* The Walrus (marker ``thewalrus``): ``thewalrus.quantum.density_matrix`` of the output Gaussian
  state against the Fock backend. Its tensor is ordered ``(n₁, m₁, n₂, m₂, …)``; cvsim's density
  matrix is ``(n₁ n₂ …) x (m₁ m₂ …)``.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from harness import HBAR_REF, assert_same, from_ref, mrmustard, random_unitary
from hypothesis import given, settings
from hypothesis import strategies as st

from cvsim.backends import FockBackend, GaussianBackend
from cvsim.gaussian import symplectic as sy
from cvsim.ir import Program, op

FAST = settings(max_examples=30, deadline=None)
#: Truncation error of the Fock backend on ``small_program`` against the exact matrix elements:
#: 6.8e-7 (cutoff 24), 8.2e-8 (28), 1.0e-8 (32), 1.2e-9 (36): geometric, as expected.
CUTOFF = 36


# ------------------------------------------------------------------------------- random programs
@st.composite
def programs(draw, measure: bool = True):
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    M = 3
    p = Program(M)
    for m in range(M):
        kind = draw(st.sampled_from(["Vacuum", "Coherent", "Squeezed", "Thermal"]))
        params = {
            "Vacuum": {},
            "Coherent": {"alpha": complex(*rng.normal(0, 0.3, 2))},
            "Squeezed": {"r": rng.uniform(0, 0.4), "phi": rng.uniform(-3, 3)},
            "Thermal": {"nbar": rng.uniform(0, 0.3)},
        }[kind]
        p = p >> op(kind, m, **params)
    for _ in range(draw(st.integers(1, 8))):
        a, b = (int(x) for x in rng.choice(M, 2, replace=False))
        name = draw(
            st.sampled_from(
                [
                    "D",
                    "R",
                    "S",
                    "P",
                    "F",
                    "BS",
                    "S2",
                    "CZ",
                    "CX",
                    "Interferometer",
                    "Symplectic",
                    "Loss",
                    "Amplifier",
                    "AdditiveNoise",
                ]
            )
        )
        if name == "D":
            o = op("D", a, alpha=complex(*rng.normal(0, 0.3, 2)))
        elif name in ("R", "F"):
            o = op(name, a, **({"theta": rng.uniform(-3, 3)} if name == "R" else {}))
        elif name == "S":
            o = op("S", a, r=rng.uniform(0, 0.4), phi=rng.uniform(-3, 3))
        elif name == "P":
            o = op("P", a, s=rng.uniform(-0.5, 0.5))
        elif name in ("BS", "S2"):
            o = op(
                name,
                (a, b),
                **({"theta": rng.uniform(-3, 3)} if name == "BS" else {"r": rng.uniform(0, 0.3)}),
                phi=rng.uniform(-3, 3),
            )
        elif name in ("CZ", "CX"):
            o = op(name, (a, b), g=rng.uniform(-0.5, 0.5))
        elif name == "Interferometer":
            o = op(name, (a, b), U=random_unitary(rng, 2))
        elif name == "Symplectic":
            o = op(name, (a, b), S=sy.random_symplectic(rng, 2, 0.3))
        elif name == "Loss":
            o = op(name, a, eta=rng.uniform(0.3, 1), nbar=rng.uniform(0, 0.3))
        elif name == "Amplifier":
            o = op(name, a, gain=rng.uniform(1, 1.3), nbar=rng.uniform(0, 0.3))
        else:
            o = op(name, a, sigma2=rng.uniform(0, 0.1))
        p = p >> o
    if measure and draw(st.booleans()):
        p = p >> op(
            "Homodyne",
            2,
            phi=rng.uniform(-3, 3),
            efficiency=rng.uniform(0.5, 1),
            electronic=rng.uniform(0, 0.1),
            outcome=rng.normal(),
        )
    if measure and draw(st.booleans()):
        p = p >> op(
            "Heterodyne",
            1,
            efficiency=rng.uniform(0.5, 1),
            electronic=rng.uniform(0, 0.1),
            outcome=complex(*rng.normal(0, 0.5, 2)),
        )
    return p


@pytest.mark.qttlind
@FAST
@given(p=programs())
def test_qttlind_engine_matches_cvsim_engine(p):
    a, b = GaussianBackend("cvsim").run(p), GaussianBackend("qttlind").run(p)
    assert a.modes == b.modes
    scale = max(1.0, float(np.abs(a.state.cov).max()))
    assert_same(a.state.cov, b.state.cov, 1e-11 * scale, "cov")
    assert_same(a.state.mean, b.state.mean, 1e-11 * scale, "mean")
    for oa, ob in zip(a.outcomes, b.outcomes):
        assert math.isclose(oa.density, ob.density, rel_tol=1e-10)


# ------------------------------------------------------------------------------- MrMustard translation
def to_mrmustard(p: Program) -> tuple[list, list]:
    """cvsim IR -> MrMustard lab objects, with the conventions of docs/design/equivalence.md."""
    init: list = [None] * p.num_modes
    gates: list = []
    for o in p.ops:
        q = o.params
        if o.kind == "prepare":
            assert init[o.modes[0]] is None, "re-preparation is not translated"
            init[o.modes[0]] = {
                "Vacuum": ["Vacuum", {"num_modes": 1}],
                "Coherent": ["Coherent", {"x": complex(q.get("alpha", 0)).real, "y": complex(q.get("alpha", 0)).imag}],
                "Squeezed": ["SqueezedVacuum", {"r": q.get("r"), "phi": q.get("phi")}],
                "Thermal": ["Thermal", {"nbar": q.get("nbar")}],
            }[o.name]
            continue
        m = list(o.modes)
        name = o.name
        if name == "D":
            gates.append(["Dgate", {"x": complex(q["alpha"]).real, "y": complex(q["alpha"]).imag}, m])
        elif name in ("R", "F"):
            gates.append(["Rgate", {"angle": q["theta"] if name == "R" else math.pi / 2}, m])
        elif name == "S":
            gates.append(["Sgate", {"r": q["r"], "phi": q["phi"]}, m])
        elif name == "P":
            gates.append(["Pgate", {"shearing": q["s"]}, m])
        elif name == "BS":
            gates.append(["BSgate", {"theta": q["theta"], "phi": q["phi"]}, m])
        elif name == "S2":
            gates.append(["S2gate", {"r": q["r"], "phi": q["phi"] + math.pi}, m])  # S2 phase convention
        elif name in ("CZ", "CX"):
            gates.append([f"{name}gate", {"s": q["g"]}, m])
        elif name == "Interferometer":
            U = np.asarray(q["U"])
            gates.append(
                ["Interferometer", {"num_modes": len(m), "unitary": {"re": U.real.tolist(), "im": U.imag.tolist()}}, m]
            )
        elif name == "Symplectic":
            gates.append(["Ggate", {"num_modes": len(m), "symplectic": np.asarray(q["S"]).tolist()}, m])
        elif name == "Loss":
            gates.append(["Attenuator", {"transmissivity": q["eta"], "nbar": q["nbar"]}, m])
        elif name == "Amplifier":
            gates.append(["Amplifier", {"gain": q["gain"], "nbar": q["nbar"]}, m])
        elif name == "AdditiveNoise":
            gates.append(["AdditiveNoise", {"noise": 2 * q["sigma2"]}, m])  # Y = noise · ħ/2 (ħ = 2)
        elif name == "PhaseNoise":
            gates.append(["PhaseNoise", {"phase_stdev": q["sigma"]}, m])
        else:  # pragma: no cover
            raise AssertionError(name)
    init = [i if i is not None else ["Vacuum", {"num_modes": 1}] for i in init]
    return init, gates


@pytest.mark.mrmustard
@FAST
@given(p=programs(measure=False))
def test_gaussian_backend_matches_mrmustard_circuits(p):
    init, gates = to_mrmustard(p)
    r = mrmustard()("run_circuit", initial=init, ops=gates)
    mean, cov = from_ref(r["means"], r["cov"])
    s = GaussianBackend().run(p).state
    scale = max(1.0, float(np.abs(s.cov).max()))
    assert_same(cov, s.cov, 1e-11 * scale, "cov")
    assert_same(mean, s.mean, 1e-11 * scale, "mean")


def small_program() -> Program:
    U = random_unitary(np.random.default_rng(3), 2)
    return (
        Program(2)
        >> op("Squeezed", 0, r=0.3)
        >> op("Coherent", 1, alpha=0.2 - 0.1j)
        >> op("BS", (0, 1), theta=0.6, phi=0.3)
        >> op("Loss", 0, eta=0.85, nbar=0.1)
        >> op("S2", (0, 1), r=0.2, phi=0.5)
        >> op("CZ", (0, 1), g=0.3)
        >> op("Interferometer", (0, 1), U=U)
        >> op("AdditiveNoise", 1, sigma2=0.02)
        >> op("Amplifier", 0, gain=1.1)
    )


@functools.cache
def fock_dm() -> np.ndarray:
    """The Fock backend's output on ``small_program`` (computed once: ~30 s at cutoff 36)."""
    return FockBackend(CUTOFF).run(small_program()).state.dm


@pytest.mark.mrmustard
def test_fock_backend_matches_mrmustard_fock_conversion():
    """MrMustard converts the exact Gaussian output to the Fock basis; the Fock backend propagates
    a truncated density matrix: they agree to the truncation error (1.2e-9 at cutoff 36)."""
    p = small_program()
    init, gates = to_mrmustard(p)
    dm_mm = mrmustard()("run_circuit", initial=init, ops=gates, cutoffs=[CUTOFF, CUTOFF])["dm"]
    assert_same(np.asarray(dm_mm).reshape(CUTOFF**2, CUTOFF**2), fock_dm(), 5e-9, "density matrix")


@pytest.mark.mrmustard
def test_fock_backend_matches_mrmustard_fock_simulation_with_phase_noise():
    """Both simulate in the Fock basis, including exact dephasing (MrMustard's PhaseNoise acts on
    Fock states only)."""
    N = 30
    p = (
        Program(1)
        >> op("Coherent", 0, alpha=0.7)
        >> op("S", 0, r=0.2, phi=0.4)
        >> op("PhaseNoise", 0, sigma=0.4)
        >> op("R", 0, theta=0.3)
    )
    init, gates = to_mrmustard(p)
    dm_mm = np.asarray(mrmustard()("run_circuit", initial=init, ops=gates, cutoffs=[N], fock=True)["dm"])
    dm = FockBackend(N).run(p).state.dm
    assert_same(dm_mm, dm, 1e-9, "density matrix")


@pytest.mark.thewalrus
def test_fock_backend_matches_the_walrus_density_matrix():
    tq = pytest.importorskip("thewalrus.quantum")
    p = small_program()
    gauss = GaussianBackend().run(p).state
    mu, cov = gauss.to_hbar(HBAR_REF)
    exact = tq.density_matrix(mu, cov, cutoff=CUTOFF, hbar=HBAR_REF).transpose(0, 2, 1, 3).reshape(CUTOFF**2, CUTOFF**2)
    assert_same(fock_dm(), exact, 5e-9, "density matrix")


@pytest.mark.thewalrus
@pytest.mark.mrmustard
def test_the_walrus_and_mrmustard_agree_on_the_fock_conversion():
    tq = pytest.importorskip("thewalrus.quantum")
    p = small_program()
    init, gates = to_mrmustard(p)
    N = 12
    dm_mm = np.asarray(mrmustard()("run_circuit", initial=init, ops=gates, cutoffs=[N, N])["dm"]).reshape(N * N, N * N)
    mu, cov = GaussianBackend().run(p).state.to_hbar(HBAR_REF)
    exact = tq.density_matrix(mu, cov, cutoff=N, hbar=HBAR_REF).transpose(0, 2, 1, 3).reshape(N * N, N * N)
    assert_same(dm_mm, exact, 1e-13, "MrMustard vs The Walrus")


def test_programs_strategy_covers_every_gaussian_operation():
    names = {
        "D",
        "R",
        "S",
        "P",
        "F",
        "BS",
        "S2",
        "CZ",
        "CX",
        "Interferometer",
        "Symplectic",
        "Loss",
        "Amplifier",
        "AdditiveNoise",
    }
    assert names <= set(GaussianBackend().supported)
