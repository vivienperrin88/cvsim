"""Contract: the S11 sources against qttlind, The Walrus 0.22 and MrMustard 0.7.3.

None of the three libraries models a JSA, an OPO cavity or a waveguide with distributed loss (read
from their sources, October 2026; ``docs/design/equivalence.md``, "S11"). What they do implement is
the Gaussian state a source emits: squeezed vacua in the supermodes, losses, added noise, and the
interferometer ``S(U)`` that writes the supermodes on the frequency pixels. The contract is that
state, in the **pixel basis** (``SourceOutput.in_pixel_basis``), rebuilt gate by gate in each
library from the source's own parameters ``r_k``, ``η``, ``U``:

* SPOPO at an analysis frequency: ``S(r_k(Ω))`` on each supermode, loss ``η``, then ``S(U)``;
* waveguide: the reference libraries have no distributed loss, so the guide is built there as a
  symmetric (Strang) product of ``N`` slices ``loss(e^{−κℓ/2N}) · S(r_k/N) · loss(e^{−κℓ/2N})``;
  the splitting error is ``O(1/N²)``, and both its size and its rate are checked against cvsim's
  exact closed form (between ``N = 20`` and ``N = 40``);
* phase-sensitive excess noise ``diag(0, ε_k)``: added to the covariance directly in The Walrus
  (MrMustard's ``AdditiveNoise`` is phase-insensitive only).

Convention map: ħ = 2 there (``harness.to_ref``), cvsim's interferometer matrix is The Walrus's
and MrMustard's ``Interferometer(unitary=U)`` (asserted by the S07 and S09 contract tests).
Tolerances: ``ROUNDOFF`` relative to the largest covariance entry (short chains of products).
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from harness import HBAR_REF, ROUNDOFF, assert_same, from_ref, mrmustard, qttlind_module, to_ref

from cvsim.hardware import excess as ex
from cvsim.hardware import jsa, sources


@pytest.fixture(scope="module")
def sm() -> jsa.Supermodes:
    """A chirped Gaussian JSA on 8 pixels, 4 supermodes kept (the other 4 in vacuum)."""
    grid = jsa.FrequencyGrid.centered(8, 2.5)
    w = grid.detunings
    ws, wi = np.meshgrid(w, w, indexing="ij")
    f = np.exp(-((ws + wi) ** 2) / (2 * 0.8**2) - (ws - wi) ** 2 / (2 * 2.2**2) + 0.2j * (ws**2 + wi**2))
    return jsa.schmidt_decomposition(f, grid, max_modes=4, edge_tol=1.0)


@pytest.fixture(scope="module")
def opo(sm) -> sources.SPOPO:
    return sources.SPOPO(sm, escape_efficiency=0.85, linewidth=1.0, pump=0.7)


OMEGA = 0.4  # analysis frequency, in units of the cavity half width


def scale(V: np.ndarray) -> float:
    return ROUNDOFF * max(1.0, float(np.abs(V).max()))


# ------------------------------------------------------------------------------- The Walrus
@pytest.fixture(scope="module")
def tw():
    pytest.importorskip("thewalrus")
    return importlib.import_module("thewalrus.symplectic"), importlib.import_module("thewalrus.quantum")


def tw_pixel_state(tws, r, eta, U, n, eps=None):
    """Squeeze supermodes 0..M-1 of n vacuum modes, lose η on each, add ε to p, apply S(U) (ħ = 2)."""
    M = len(r)
    cov = HBAR_REF / 2 * np.eye(2 * n)
    mu = np.zeros(2 * n)
    S = tws.squeezing(np.concatenate([r, np.zeros(n - M)]), np.zeros(n))
    cov = S @ cov @ S.T
    for k in range(M):
        mu, cov = tws.loss(mu, cov, eta, k, hbar=HBAR_REF)
    if eps is not None:
        cov = cov + np.diag(np.concatenate([np.zeros(n), HBAR_REF * np.asarray(eps), np.zeros(n - M)]))
    Ss = tws.interferometer(U)
    return mu, Ss @ cov @ Ss.T


@pytest.mark.thewalrus
def test_spopo_pixel_state_matches_thewalrus(tw, sm, opo):
    tws, tq = tw
    out = opo.output(omega=OMEGA)
    mu, cov = tw_pixel_state(tws, out.squeezing, opo.escape_efficiency, sm.unitary, sm.grid.size)
    ours = out.in_pixel_basis()
    m_ref, c_ref = to_ref(ours.mean, ours.cov)
    assert_same(c_ref, cov, scale(cov), "pixel covariance")
    assert_same(m_ref, mu, ROUNDOFF, "mean")
    # photon numbers per pixel: the spectrum of the emitted light
    n_tw = [tq.photon_number_mean(mu, cov, j, hbar=HBAR_REF) for j in range(sm.grid.size)]
    assert_same(ours.mean_photons(), n_tw, 1e-12, "⟨n⟩ per pixel")


@pytest.mark.thewalrus
def test_excess_antisqueezing_matches_thewalrus(tw, sm, opo):
    tws, _ = tw
    out = opo.output(omega=OMEGA, excess=ex.ExcessNoise(antisqueezing_db=1.2))
    eps = np.diag(out.ledger.by("component")["excess antisqueezing"])[out.num_modes :]
    _, cov = tw_pixel_state(tws, out.squeezing, opo.escape_efficiency, sm.unitary, sm.grid.size, eps=eps)
    _, c_ref = to_ref(np.zeros(2 * sm.grid.size), out.in_pixel_basis().cov)
    assert_same(c_ref, cov, scale(cov), "pixel covariance with excess antisqueezing")


@pytest.mark.thewalrus
def test_waveguide_converges_to_thewalrus_slices(tw, sm):
    tws, _ = tw
    wg = sources.Waveguide(sm, length=0.02, loss_db_per_cm=1.5, r0=1.0, output_coupling=0.9)
    exact = to_ref(np.zeros(2 * sm.grid.size), wg.output().in_pixel_basis().cov)[1]
    n, M = sm.grid.size, sm.kept
    r = wg.squeezing_parameters
    half = math.exp(-wg.attenuation_length / 2)

    def sliced(N: int) -> np.ndarray:
        mu, cov = np.zeros(2 * n), HBAR_REF / 2 * np.eye(2 * n)
        S = tws.squeezing(np.concatenate([r / N, np.zeros(n - M)]), np.zeros(n))
        for _ in range(N):
            for k in range(M):
                mu, cov = tws.loss(mu, cov, half ** (1 / N), k, hbar=HBAR_REF)
            cov = S @ cov @ S.T
            for k in range(M):
                mu, cov = tws.loss(mu, cov, half ** (1 / N), k, hbar=HBAR_REF)
        for k in range(M):
            mu, cov = tws.loss(mu, cov, 0.9, k, hbar=HBAR_REF)
        Ss = tws.interferometer(sm.unitary)
        return Ss @ cov @ Ss.T

    e20, e40 = (float(np.abs(sliced(N) - exact).max()) for N in (20, 40))
    assert e40 < 1e-3
    assert e20 / e40 == pytest.approx(4.0, rel=0.05)


# ------------------------------------------------------------------------------- MrMustard
def _complex(U: np.ndarray) -> dict:
    return {"re": U.real.tolist(), "im": U.imag.tolist()}


@pytest.mark.mrmustard
def test_spopo_pixel_state_matches_mrmustard(sm, opo):
    out = opo.output(omega=OMEGA)
    n, M = sm.grid.size, out.num_modes
    initial = [["SqueezedVacuum", {"r": float(out.squeezing[k]), "phi": 0.0}] for k in range(M)]
    initial += [["Vacuum", {"num_modes": 1}]] * (n - M)
    ops = [["Attenuator", {"transmissivity": opo.escape_efficiency, "nbar": 0.0}, [k]] for k in range(M)]
    ops.append(["Interferometer", {"num_modes": n, "unitary": _complex(sm.unitary)}, list(range(n))])
    res = mrmustard()("run_circuit", initial=initial, ops=ops)
    mean, cov = from_ref(res["means"], res["cov"])
    ours = out.in_pixel_basis()
    assert_same(cov, ours.cov, 1e-11 * max(1.0, float(np.abs(ours.cov).max())), "pixel covariance")
    assert_same(mean, ours.mean, 1e-11, "mean")


@pytest.mark.mrmustard
def test_waveguide_converges_to_mrmustard_slices(sm):
    wg = sources.Waveguide(sm, length=0.02, loss_db_per_cm=1.5, r0=1.0, output_coupling=1.0)
    out = wg.output()
    M = out.num_modes
    r = wg.squeezing_parameters
    half = math.exp(-wg.attenuation_length / 2)
    exact = out.state.cov

    def sliced(N: int) -> np.ndarray:
        initial = [["Vacuum", {"num_modes": 1}]] * M
        ops = []
        for _ in range(N):
            ops += [["Attenuator", {"transmissivity": half ** (1 / N), "nbar": 0.0}, [k]] for k in range(M)]
            ops += [["Sgate", {"r": float(r[k] / N), "phi": 0.0}, [k]] for k in range(M)]
            ops += [["Attenuator", {"transmissivity": half ** (1 / N), "nbar": 0.0}, [k]] for k in range(M)]
        res = mrmustard()("run_circuit", initial=initial, ops=ops)
        return from_ref(res["means"], res["cov"])[1]

    e20, e40 = (float(np.abs(sliced(N) - exact).max()) for N in (20, 40))
    assert e40 < 1e-3
    assert e20 / e40 == pytest.approx(4.0, rel=0.05)


# ------------------------------------------------------------------------------- qttlind
@pytest.mark.qttlind
def test_spopo_pixel_state_matches_qttlind(sm, opo):
    qs, qo = qttlind_module("gaussian.states"), qttlind_module("gaussian.ops")
    out = opo.output(omega=OMEGA)
    n, M = sm.grid.size, out.num_modes
    q = qs.GaussianState(np.zeros(2 * n), 0.5 * np.eye(2 * n))
    for k in range(M):
        q = qo.squeeze(q, float(out.squeezing[k]), 0.0, mode=k)
        q = qo.loss(q, opo.escape_efficiency, mode=k)
    q = qo.interferometer(q, sm.unitary)
    ours = out.in_pixel_basis()
    assert_same(ours.cov, q.cov, ROUNDOFF * max(1.0, float(np.abs(ours.cov).max())), "pixel covariance")
    assert_same(ours.mean, q.mean, ROUNDOFF, "mean")
