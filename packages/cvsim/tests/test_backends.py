"""cvsim.backends: the same programs on three engines, Fock-only physics, results, refusals.

Agreement: the Gaussian engines (cvsim, qttlind) to round-off; the Fock backend's quadrature
moments to 1e-8 at cutoff 24 for two modes with at most ~1 photon per mode (the truncation error,
measured at 3e-9 on the densest program here). Non-Gaussian checks use closed forms (Hong–Ou–Mandel,
the Kerr-evolved coherent amplitude).
"""

from __future__ import annotations

import cmath
import importlib.util
import math
import warnings

import numpy as np
import pytest

from cvsim.backends import FockBackend, FockState, GaussianBackend, Result, run, select_backend
from cvsim.errors import ApproximationWarning, CutoffWarning, InputError
from cvsim.gaussian import symplectic as sy
from cvsim.ir import Parameter, Program, op

N = 24
HAVE_QTTLIND = importlib.util.find_spec("qttlind") is not None
needs_qttlind = pytest.mark.skipif(not HAVE_QTTLIND, reason="qttlind is not installed")


def gaussian_backends() -> list[GaussianBackend]:
    return [GaussianBackend("cvsim")] + ([GaussianBackend("qttlind")] if HAVE_QTTLIND else [])


def chain() -> Program:
    return (
        Program(["a", "b"])
        >> op("Squeezed", 0, r=0.3)
        >> op("Coherent", 1, alpha=0.2 - 0.1j)
        >> op("BS", (0, 1), theta=0.6, phi=0.3)
        >> op("R", 1, theta=0.4)
        >> op("Loss", 0, eta=0.85, nbar=0.1, component="source", noise_kind="loss")
        >> op("S2", (0, 1), r=0.2, phi=0.5)
        >> op("CZ", (0, 1), g=0.3)
        >> op("CX", (1, 0), g=-0.2)
        >> op("P", 1, s=0.2)
        >> op("F", 0)
        >> op("D", 0, alpha=0.1j)
        >> op("AdditiveNoise", 1, sigma2=0.02, component="electronics", noise_kind="electronic")
        >> op("Amplifier", 0, gain=1.1, nbar=0.2)
    )


def gaussian_programs() -> list[Program]:
    rng = np.random.default_rng(4)
    U = sy.random_interferometer(rng, 2)
    S = sy.random_symplectic(rng, 2, 0.4)
    base = chain()
    return [
        base,
        Program(2)
        >> op("Thermal", 0, nbar=0.3)
        >> op("Squeezed", 1, r=0.2, phi=1.0)
        >> op("Interferometer", (0, 1), U=U),
        Program(2) >> op("Symplectic", (0, 1), S=S) >> op("Loss", 1, eta=0.6),
        base >> op("Homodyne", 0, phi=0.4, efficiency=0.9, electronic=0.02, outcome=0.3),
        base >> op("Heterodyne", 1, efficiency=0.8, electronic=0.01, outcome=0.1 + 0.2j),
        base >> op("Homodyne", 0, outcome=-0.2) >> op("Vacuum", 0) >> op("BS", (0, 1), theta=0.3),
    ]


def assert_states_agree(a: Result, b: Result, atol: float) -> None:
    assert a.modes == b.modes
    sa, sb = a.state, b.state
    ma, Va = (sa.mean, sa.cov) if not isinstance(sa, FockState) else sa.moments()
    mb, Vb = (sb.mean, sb.cov) if not isinstance(sb, FockState) else sb.moments()
    assert np.allclose(ma, mb, atol=atol) and np.allclose(Va, Vb, atol=atol)
    for oa, ob in zip(a.outcomes, b.outcomes):
        assert oa.op_index == ob.op_index and oa.name == ob.name
        assert abs(complex(oa.value) - complex(ob.value)) < 1e-12
        assert math.isclose(oa.density, ob.density, rel_tol=max(atol, 1e-12))


# ------------------------------------------------------------------------------- three engines
@needs_qttlind
@pytest.mark.parametrize("k", range(6))
def test_same_program_on_three_engines(k):
    p = gaussian_programs()[k]
    a = GaussianBackend("cvsim").run(p)
    b = GaussianBackend("qttlind").run(p)
    c = FockBackend(N).run(p)
    assert_states_agree(a, b, 1e-12)
    assert_states_agree(a, c, 1e-8)


def test_phase_noise_moments_agree_with_exact_dephasing():
    p = (
        Program(2)
        >> op("Coherent", 0, alpha=0.6)
        >> op("Squeezed", 1, r=0.3)
        >> op("BS", (0, 1), theta=0.5)
        >> op("PhaseNoise", 1, sigma=0.3)
    )
    with pytest.warns(ApproximationWarning):
        a = GaussianBackend().run(p)
    c = FockBackend(N).run(p)
    assert_states_agree(a, c, 1e-9)


def test_dispatcher():
    g = Program(1) >> op("S", 0, r=0.1)
    ng = Program(1) >> op("Fock", 0, n=1) >> op("Kerr", 0, kappa=0.2)
    assert isinstance(select_backend(g), GaussianBackend) and isinstance(select_backend(ng), FockBackend)
    assert run(g).backend == "gaussian[cvsim]" and run(ng, cutoff=8).backend == "fock[cutoff=8]"
    assert run(g, "fock", cutoff=10).backend == "fock[cutoff=10]"
    if HAVE_QTTLIND:
        assert run(g, "qttlind").backend == "gaussian[qttlind]"
    assert run(g, FockBackend(6)).backend == "fock[cutoff=6]"


# ------------------------------------------------------------------------------- Fock-only physics
def test_hong_ou_mandel():
    p = Program(2) >> op("Fock", 0, n=1) >> op("Fock", 1, n=1) >> op("BS", (0, 1), theta=math.pi / 4)
    dm = FockBackend(6).run(p).state.dm
    idx = lambda i, j: i * 6 + j  # noqa: E731
    assert abs(dm[idx(1, 1), idx(1, 1)]) < 1e-14
    assert math.isclose(dm[idx(2, 0), idx(2, 0)].real, 0.5, abs_tol=1e-13) and math.isclose(
        dm[idx(0, 2), idx(0, 2)].real, 0.5, abs_tol=1e-13
    )


def test_kerr_on_a_coherent_state():
    """U = exp(iκn²): U†aU = e^{iκ(2n+1)} a, so ⟨a⟩ = α e^{iκ} exp(|α|²(e^{2iκ} − 1))."""
    alpha, kappa = 0.8, 0.15
    st = FockBackend(40).run(Program(1) >> op("Coherent", 0, alpha=alpha) >> op("Kerr", 0, kappa=kappa)).state
    mean, _ = st.moments()
    a = alpha * cmath.exp(1j * kappa) * cmath.exp(abs(alpha) ** 2 * (cmath.exp(2j * kappa) - 1))
    assert np.allclose(mean, math.sqrt(2) * np.array([a.real, a.imag]), atol=1e-12)
    assert np.allclose(st.mean_photons(), alpha**2, atol=1e-12)  # Kerr keeps the photon distribution


def test_fock_state_helpers():
    st = FockBackend(8).run(Program(2) >> op("Fock", 0, n=1) >> op("Thermal", 1, nbar=0.05)).state
    assert repr(st).startswith("FockState(modes=2, cutoff=8")
    assert math.isclose(st.reduce([0]).purity(), 1.0) and st.reduce([1, 0]).num_modes == 2
    assert np.allclose(st.reduce([1, 0]).mean_photons()[::-1], st.mean_photons())
    with pytest.raises(InputError):
        st.reduce([0, 0])
    with pytest.raises(InputError):
        FockState(np.eye(3), 2)


def test_cutoff_warning():
    with pytest.warns(CutoffWarning, match="raise the cutoff"):
        FockBackend(6).run(Program(1) >> op("Coherent", 0, alpha=1.5))


# ------------------------------------------------------------------------------- results, sampling, ledger
def test_sampling_is_reproducible_and_distributed_correctly():
    p = chain() >> op("Homodyne", 0, phi=0.7)
    for backend in (*gaussian_backends(), FockBackend(16)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CutoffWarning)
            a = backend.run(p, rng=np.random.default_rng(7))
            b = backend.run(p, rng=np.random.default_rng(7))
        assert a.outcome_values() == b.outcome_values()
    exact = GaussianBackend().run(chain() >> op("Homodyne", 0, phi=0.7, outcome=0.0))
    rng = np.random.default_rng(1)
    bare = Program(1) >> op("Squeezed", 0, r=0.4) >> op("Homodyne", 0, phi=0.3)
    draws = np.array([FockBackend(30).run(bare, rng=rng).outcomes[0].value for _ in range(3000)])
    var = (math.cos(0.3) ** 2 * math.exp(-0.8) + math.sin(0.3) ** 2 * math.exp(0.8)) / 2
    assert abs(draws.mean()) < 0.05 and abs(draws.var() / var - 1) < 0.08
    assert exact.log_density == pytest.approx(math.log(exact.outcomes[0].density))


def test_ledger_and_modes():
    res = GaussianBackend().run(chain())
    assert res.ledger is not None and res.modes == (0, 1)
    comps = [c.component for c in res.ledger.contributions]
    assert comps == ["source", "electronics", "Amplifier"]
    assert res.ledger.state.isclose(res.state, 1e-12)
    assert GaussianBackend().run(chain() >> op("Homodyne", 0, outcome=0.0)).ledger is None
    if HAVE_QTTLIND:
        assert GaussianBackend("qttlind").run(chain()).ledger is None


def test_measuring_everything_and_preparing_again():
    p = Program(2) >> op("Squeezed", 0, r=0.2) >> op("Homodyne", 0, outcome=0.1) >> op("Heterodyne", 1, outcome=0j)
    for backend in (*gaussian_backends(), FockBackend(10)):
        res = backend.run(p)
        assert res.state is None and res.modes == () and len(res.outcomes) == 2
    q = p >> op("Coherent", 1, alpha=0.3)
    res = GaussianBackend().run(q)
    assert res.modes == (1,) and np.allclose(res.state.complex_moments()[0], [0.3])


# ------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("backend", "program", "match"),
    [
        (lambda: GaussianBackend(), Program(1) >> op("Kerr", 0, kappa=0.1), r"operation 0 \(Kerr\)"),
        (lambda: GaussianBackend(), Program(1) >> op("Fock", 0, n=1), r"operation 0 \(Fock\)"),
        pytest.param(
            lambda: GaussianBackend("qttlind"),
            Program(1) >> op("PhaseNoise", 0, sigma=0.1),
            r"PhaseNoise",
            marks=needs_qttlind,
        ),
        (lambda: FockBackend(5), Program(1) >> op("Heterodyne", 0), r"sample operation 0 \(Heterodyne\)"),
        (lambda: FockBackend(3), Program(1) >> op("Fock", 0, n=3), r"cutoff > 3"),
        (lambda: GaussianBackend(), Program(1) >> op("S", 0, r=Parameter("r")), r"free parameters"),
        (lambda: GaussianBackend(), Program(1) >> op("Homodyne", 0), r"rng"),
    ],
)
def test_refusals(backend, program, match):
    with pytest.raises(InputError, match=match):
        backend().run(program)


def test_invalid_construction_and_options():
    with pytest.raises(InputError):
        GaussianBackend("strawberry")
    with pytest.raises(InputError):
        FockBackend(1)
    with pytest.raises(InputError, match="unknown backend"):
        run(Program(1), "mrmustard")
    with pytest.raises(InputError, match="unknown option"):
        run(Program(1), shots=3)
    with pytest.raises(InputError):
        select_backend("program")  # type: ignore[arg-type]
    with pytest.raises(InputError, match="Program"):
        GaussianBackend().run("program")  # type: ignore[arg-type]
    GaussianBackend().check(Program(1) >> op("Homodyne", 0))  # check() does not need the rng
