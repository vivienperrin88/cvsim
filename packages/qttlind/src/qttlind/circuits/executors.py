"""Three executors behind one interface: Gaussian, quantics comb, and hybrid.

* :class:`GaussianExecutor` -- exact, on :mod:`qttlind.gaussian`; refuses non-Gaussian
  instructions, naming them.
* :class:`QTTExecutor` -- every mode on the purified comb (:class:`~.modes.CVMode` per mode).
* :class:`HybridExecutor` -- runs the Gaussian prefix on the covariance matrix. At the first
  instruction it cannot run there, it moves to the comb **the modes correlated with the ones it
  touches** (the connected component of the cross-covariance graph, threshold
  ``correlation_tol``), through the Bargmann recursion, and keeps the others Gaussian (D12).

Conventions shared by all executors (identical to :mod:`qttlind.gaussian.ops`):

* gates ``exp(X)``: rotation ``X = i theta n``; squeezing ``(conj(z) a^2 - z a^dag^2)/2``;
  displacement ``alpha a^dag - conj(alpha) a``; beam splitter ``theta (e^{i phi} a b^dag - h.c.)``;
  two-mode squeezing ``conj(z) a b - z a^dag b^dag``; Kerr ``i kappa n^2``; cubic phase
  ``i gamma x^3 / 3``;
* a **measured mode is reset to vacuum** (its outcome is in ``Result.outcomes``);
* ``Result.probability`` multiplies the probabilities (densities, for homodyne) of the
  post-selected outcomes; sampled outcomes do not contribute.

On the comb, single-mode gates are MPOs (exact diagonal phases where they exist, otherwise
``expm`` of the generator: dense up to 2^10 levels, computed in a padded Fock space to avoid
cutoff artefacts), two-mode gates :func:`~.comb_ops.expm_terms`, pure loss its exact Kraus
operators ``K_k = sqrt((1-eta)^k / k!) eta^{n/2} a^k``, phase noise a Gauss-Hermite average of
rotations, thermal loss the Lindblad equation (second order in its step).
"""

from __future__ import annotations

import functools
import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import scipy.linalg

from ..bosons import operators as bops
from ..bosons import states as bstates
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..cv import grids as cvgrids
from ..errors import CutoffWarning, InputError
from ..gaussian import measure as gmeasure
from ..gaussian import ops as gops
from ..gaussian.fock import fock_density_matrix, fock_ket
from ..gaussian.phase_space import mode_indices
from ..gaussian.states import GaussianState
from ..lindblad.comb import PurifiedComb
from ..lindblad.kraus import LocalTerm
from . import noise
from .circuit import Circuit, Instruction
from .comb_ops import apply_kraus, expm_mpo, expm_terms, rank_one
from .modes import CVMode

__all__ = ["GaussianExecutor", "HybridExecutor", "HybridState", "QTTExecutor", "Result"]

log = logging.getLogger(__name__)

_TOL = 1e-24
#: Gauss-Hermite nodes for phase noise.
GH_NODES = 24
#: Populations below this are ignored when choosing Kraus sets and cut-offs.
POP_TOL = 1e-15


class Result(NamedTuple):
    state: Any  # GaussianState | PurifiedComb | HybridState
    outcomes: dict[int, float | int]
    probability: float
    backend_log: list[str]


@dataclass
class HybridState:
    """Some modes Gaussian, the others on the comb (both parts uncorrelated).

    ``comb_modes[k]`` is the circuit mode at comb position ``k`` (hand-off order).
    """

    gaussian: GaussianState | None
    gaussian_modes: tuple[int, ...]
    comb: PurifiedComb | None
    comb_modes: tuple[int, ...]


# ---------------------------------------------------------------------------------- Gaussian
def _insert_vacuum(rest: GaussianState | None, mode: int, M: int) -> GaussianState:
    """The state of ``M`` modes: ``rest`` on the other modes, vacuum on ``mode``."""
    full_mean = np.zeros(2 * M)
    full_cov = 0.5 * np.eye(2 * M)
    if rest is not None:
        others = [k for k in range(M) if k != mode]
        idx = mode_indices(others, M)
        full_mean[idx] = rest.mean
        full_cov[np.ix_(idx, idx)] = rest.cov
    return GaussianState(full_mean, full_cov, validate=False)


class GaussianExecutor:
    """Exact Gaussian simulation (mean and covariance)."""

    name = "gaussian"

    def sample(
        self, circuit: Circuit, shots: int, rng: np.random.Generator | None = None, state: GaussianState | None = None
    ) -> np.ndarray:
        """``shots`` outcomes of the final (not post-selected) homodyne measurement."""
        rng = rng or np.random.default_rng()
        prefix, last = _split_last_measurement(circuit)
        return _sample_gaussian(self.run(prefix, state=state, rng=rng).state, last, shots, rng)

    def supports(self, instr: Instruction) -> bool:
        return instr.is_gaussian

    def run(
        self, circuit: Circuit, state: GaussianState | None = None, rng: np.random.Generator | None = None
    ) -> Result:
        st = GaussianState.vacuum(circuit.num_modes) if state is None else state
        if st.num_modes != circuit.num_modes:
            raise InputError(f"state has {st.num_modes} modes, the circuit {circuit.num_modes}")
        outcomes: dict[int, float | int] = {}
        prob = 1.0
        for instr in circuit:
            if not self.supports(instr):
                raise InputError(f"GaussianExecutor cannot run the non-Gaussian instruction {instr!r}")
            st, out, p = self.apply(st, instr, rng)
            if out is not None:
                outcomes[instr.modes[0]] = out
            prob *= p
        return Result(st, outcomes, prob, [f"gaussian: {len(circuit)} instructions"])

    @staticmethod
    def apply(st: GaussianState, instr: Instruction, rng: np.random.Generator | None):
        k, m, p = instr.kind, instr.modes, instr.params
        if k == "squeeze":
            return gops.squeeze(st, p["r"], p["phi"], m[0]), None, 1.0
        if k == "displace":
            return gops.displace(st, p["alpha"], m[0]), None, 1.0
        if k == "rotate":
            return gops.rotate(st, p["theta"], m[0]), None, 1.0
        if k == "beamsplitter":
            return gops.beamsplitter(st, p["theta"], p["phi"], m), None, 1.0
        if k == "two_mode_squeeze":
            return gops.two_mode_squeeze(st, p["r"], p["phi"], m), None, 1.0
        if k == "loss":
            return gops.loss(st, p["eta"], m[0], p["nbar"]), None, 1.0
        if k == "measure_homodyne":
            q = p["postselect"]
            if q is None and rng is None:
                rng = np.random.default_rng()
            res = gmeasure.homodyne(st, m[0], p["phi"], outcome=q, rng=None if q is not None else rng)
            new = _insert_vacuum(res.state, m[0], st.num_modes)
            return new, res.outcome, (res.density if q is not None else 1.0)
        raise InputError(f"GaussianExecutor cannot run {instr!r}")


# ---------------------------------------------------------------------------------- comb backend
@functools.lru_cache(maxsize=256)
def _single_mode_gate(mode: CVMode, kind: str, params: tuple) -> MPO:
    """Unitary MPO of a single-mode gate in the mode's basis."""
    p = dict(params)
    N = mode.dim
    if mode.basis == "fock":
        n = np.arange(N, dtype=float)
        if kind == "rotate":
            return _diag(np.exp(1j * p["theta"] * n), mode.bits)
        if kind == "kerr":
            return _diag(np.exp(1j * p["kappa"] * n**2), mode.bits)
        if N > 1024:
            X = _generator_mpo(mode, kind, p)
            return expm_mpo(X, tol=_TOL)
        Nb = 2 * N  # padded space: exact on the levels the state occupies
        a = np.diag(np.sqrt(np.arange(1, Nb)), 1)
        Xd = _generator_dense(kind, p, a)
        Ud = scipy.linalg.expm(Xd)[:N, :N]
        return MPO.from_dense(Ud, [2] * mode.bits, tol=_TOL)
    # position basis
    if kind == "cubic_phase":
        x = mode.grid.x()  # type: ignore[union-attr]
        return _diag(np.exp(1j * p["gamma"] * x**3 / 3), mode.bits)
    return expm_mpo(_generator_mpo(mode, kind, p), tol=_TOL)


def _diag(values: np.ndarray, bits: int) -> MPO:
    return bops.diag(TensorTrain.from_dense(values.astype(np.complex128), [2] * bits, tol=_TOL))


def _generator_dense(kind: str, p: dict, a: np.ndarray) -> np.ndarray:
    ad = a.conj().T
    if kind == "squeeze":
        z = p["r"] * np.exp(1j * p["phi"])
        return (np.conj(z) * a @ a - z * ad @ ad) / 2
    if kind == "displace":
        return p["alpha"] * ad - np.conj(p["alpha"]) * a
    if kind == "cubic_phase":
        x = (a + ad) / np.sqrt(2)
        return 1j * p["gamma"] * x @ x @ x / 3
    raise InputError(f"no single-mode generator for {kind!r}")


def _generator_mpo(mode: CVMode, kind: str, p: dict) -> MPO:
    a = mode.annihilation()
    ad = a.dagger()

    def c(m: MPO) -> MPO:
        return m.compress(_TOL)[0]

    if kind == "squeeze":
        z = p["r"] * np.exp(1j * p["phi"])
        return c(c(a @ a) * (np.conj(z) / 2) - c(ad @ ad) * (z / 2))
    if kind == "displace":
        return c(ad * p["alpha"] - a * np.conj(p["alpha"]))
    if kind == "rotate":
        return mode.number() * (1j * p["theta"])
    if kind == "kerr":
        n = mode.number()
        return c(n @ n) * (1j * p["kappa"])
    if kind == "cubic_phase":
        x, _ = mode.quadratures()
        return c(c(x @ x) @ x) * (1j * p["gamma"] / 3)
    raise InputError(f"no single-mode generator for {kind!r}")


@functools.lru_cache(maxsize=64)
def _norm_number(mode: CVMode) -> float:
    from .comb_ops import op_norm

    return op_norm(mode.number())


def _log_binom(n: np.ndarray, k: int) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


@functools.lru_cache(maxsize=64)
def _damping(mode: CVMode, eta: float) -> MPO:
    """``eta^{n/2}``."""
    if mode.basis == "fock":
        return _diag(eta ** (np.arange(mode.dim) / 2), mode.bits)
    return expm_mpo(mode.number() * (0.5 * math.log(eta)), tol=_TOL)


class _CombBackend:
    """The comb part of an executor: a :class:`PurifiedComb` whose positions map to circuit modes."""

    def __init__(self, cvmodes: Sequence[CVMode], tol: float, caps: dict | None, log_: list[str]):
        self.cv = list(cvmodes)
        self.tol = tol
        self.caps = dict(caps or {})
        self.comb: PurifiedComb | None = None
        self.order: list[int] = []
        self.log = log_

    # ------------------------------------------------------------- building
    def pos(self, mode: int) -> int:
        return self.order.index(mode)

    def add_product(self, modes: Sequence[int], legs: Sequence[TensorTrain]) -> None:
        sub = PurifiedComb.product_state(list(legs))
        self._attach(sub, modes)

    def add_gaussian(self, state: GaussianState, modes: Sequence[int]) -> None:
        """Move a Gaussian state of ``modes`` (uncorrelated with everything else) to the comb."""
        cvm = [self.cv[m] for m in modes]
        if len(modes) == 1 and state.is_pure():
            self.add_product(modes, [cvm[0].state_from_gaussian(state, tol=1e-28)])
            self.log.append(f"hand-off: mode {modes[0]} ({cvm[0].basis} basis), pure, single mode")
            return
        caps = [m.fock_cutoff() for m in cvm]
        nbar = state.mean_photons()
        cut = [int(min(c, max(16, math.ceil(n + 10 * math.sqrt(n * (n + 1)) + 16)))) for c, n in zip(caps, nbar)]
        pure = state.is_pure()
        limit = 2**22 if pure else 4096
        while True:
            D = int(np.prod(cut))
            if limit < D:
                raise InputError(
                    f"hand-off of modes {list(modes)} needs {D} Fock states (limit {limit} for a "
                    f"{'pure' if pure else 'mixed'} state): reduce squeezing or move fewer modes"
                )
            if pure:
                v = fock_ket(state, cut).reshape(-1)
                weight = float(np.vdot(v, v).real)
            else:
                rho = fock_density_matrix(state, cut)
                weight = float(np.trace(rho).real)
            if weight >= 1 - 1e-12 or cut == caps:
                break
            cut = [min(c, 2 * k) for c, k in zip(caps, cut)]
        if weight < 1 - 1e-10:
            warnings.warn(
                f"hand-off of modes {list(modes)}: {1 - weight:.1e} of the norm lies above the Fock cut-offs {cut}",
                CutoffWarning,
                stacklevel=3,
            )
        if pure:
            Psi = v[:, None]
        else:
            lam, V = np.linalg.eigh((rho + rho.conj().T) / 2)
            keep = lam > POP_TOL * lam.max()
            Psi = V[:, keep] * np.sqrt(lam[keep])
        chi = Psi.shape[1]
        A = Psi.reshape(*cut, chi)
        full = [m.dim for m in cvm]
        out = np.zeros((*full, chi), dtype=np.complex128)
        out[tuple(slice(0, k) for k in cut) + (slice(None),)] = A
        sub = PurifiedComb.from_dense(out.reshape(-1, chi), [m.bits for m in cvm], tol=1e-28)
        x_ops: dict[int, MPO] = {}
        for j, (m, k) in enumerate(zip(cvm, cut)):
            op = m.fock_to_basis_mpo(max(1, math.ceil(math.log2(k))))
            if op is not None:
                x_ops[j] = op
        if x_ops:
            sub = sub.apply_local(x_ops, tol=self.tol)
        sub = sub.normalize()
        self._attach(sub, modes)
        self.log.append(
            f"hand-off: modes {list(modes)} via Bargmann, Fock cut-offs {cut}, chi_mu={chi}, bonds {sub.bond_dims()}"
        )

    def _attach(self, sub: PurifiedComb, modes: Sequence[int]) -> None:
        self.comb = sub if self.comb is None else self.comb.tensor(sub).compress(self.tol, self.caps)[0]
        self.order += list(modes)

    # ------------------------------------------------------------- helpers
    def _populations(self, mode: int) -> np.ndarray:
        """Fock populations ``p_n`` of a comb mode (dense reduced density matrix)."""
        assert self.comb is not None
        cvm = self.cv[mode]
        rho = self.comb.reduced_dm(self.pos(mode))
        nmax = cvm.fock_cutoff()
        Phi = cvm.fock_basis_matrix(nmax)
        p = np.einsum("nj,jk,nk->n", Phi.conj(), rho, Phi).real
        return np.clip(p, 0, None)

    def _vacuum_bra(self, mode: int, bra: TensorTrain) -> MPO:
        return rank_one(self.cv[mode].vacuum(), bra)

    def _project(self, kraus, norm_before: float) -> float:
        assert self.comb is not None
        new = apply_kraus(self.comb, kraus, tol=self.tol, caps=self.caps)
        tr = new.trace()
        if tr <= 1e-300:
            raise InputError("the post-selected outcome has probability zero")
        self.comb = new.normalize()
        return tr / norm_before

    # ------------------------------------------------------------- instructions
    def apply(self, instr: Instruction, rng: np.random.Generator | None) -> tuple[float | int | None, float]:
        assert self.comb is not None
        k, p = instr.kind, dict(instr.params)
        pos = [self.pos(m) for m in instr.modes]
        cvm = [self.cv[m] for m in instr.modes]
        if k in ("squeeze", "displace", "rotate", "kerr", "cubic_phase"):
            U = _single_mode_gate(cvm[0], k, tuple(sorted(p.items())))
            self.comb = self.comb.apply_local({pos[0]: U}, tol=self.tol, caps=self.caps)
            return None, 1.0
        if k in ("beamsplitter", "two_mode_squeeze"):
            a0, a1 = cvm[0].annihilation(), cvm[1].annihilation()
            if k == "beamsplitter":
                e = p["theta"] * np.exp(1j * p["phi"])
                X = [
                    LocalTerm(e, {pos[0]: a0, pos[1]: a1.dagger()}),
                    LocalTerm(-np.conj(e), {pos[0]: a0.dagger(), pos[1]: a1}),
                ]
            else:
                z = p["r"] * np.exp(1j * p["phi"])
                X = [
                    LocalTerm(np.conj(z), {pos[0]: a0, pos[1]: a1}),
                    LocalTerm(-z, {pos[0]: a0.dagger(), pos[1]: a1.dagger()}),
                ]
            self.comb, info = expm_terms(self.comb, X, tol=self.tol, caps=self.caps)
            self.comb = self.comb.normalize()
            self.log.append(
                f"{instr!r} on comb: {info['substeps']} sub-steps, order <= {info['max_order']}, "
                f"bonds {self.comb.bond_dims()}"
            )
            return None, 1.0
        if k == "loss":
            self._loss(instr.modes[0], p["eta"], p["nbar"])
            return None, 1.0
        if k == "dephasing":
            self._dephase(instr.modes[0], p["sigma"])
            return None, 1.0
        if k == "measure_fock":
            return self._measure_fock(instr.modes[0], p["postselect"], rng)
        if k == "measure_threshold":
            return self._measure_threshold(instr.modes[0], p["postselect"], p["dark_count"], rng)
        if k == "measure_homodyne":
            return self._measure_homodyne(instr.modes[0], p["phi"], p["postselect"], rng)
        raise InputError(f"the comb backend cannot run {instr!r}")

    def _loss(self, mode: int, eta: float, nbar: float) -> None:
        assert self.comb is not None
        if eta == 1.0 and nbar == 0.0:
            return
        cvm, j = self.cv[mode], self.pos(mode)
        if nbar > 0:
            t = -math.log(max(eta, 1e-300))
            # explicit-step stability: dt * max decay rate / 2 < ~1 over the whole truncated space
            gamma_max = (2 * nbar + 1) * _norm_number(cvm) + nbar
            dt = min(0.01, t / 20, 1.0 / gamma_max)
            run = noise.evolve_lindblad(
                self.comb, [], noise.thermal(cvm, 1.0, nbar, index=j), t, dt, tol=self.tol, caps=self.caps
            )
            self.comb = run.final_state
            self.log.append(f"thermal loss on mode {mode}: Lindblad, {len(run.steps)} steps of {dt:.3g}")
            return
        pops = self._populations(mode)
        pops = pops / max(pops.sum(), 1e-300)
        kraus: list[Any]
        if eta == 0.0:
            kraus = [{j: self._vacuum_bra(mode, cvm.fock_state(n))} for n in range(len(pops)) if pops[n] > POP_TOL]
        else:
            # weight of K_k: Tr(K_k rho K_k^dag) = sum_n p_n C(n, k) (1 - eta)^k eta^(n - k)  (binomial)
            n = np.arange(len(pops))
            weights = [
                float(
                    np.sum(pops[k:] * np.exp(_log_binom(n[k:], k) + k * math.log1p(-eta) + (n[k:] - k) * math.log(eta)))
                )
                if eta < 1
                else float(k == 0)
                for k in range(len(pops))
            ]
            a = cvm.annihilation()
            damp = _damping(cvm, float(eta))
            kraus = []
            ak: MPO | None = None
            for kk, w in enumerate(weights):
                ak = MPO.identity([2] * cvm.bits) if ak is None else (a @ ak).compress(_TOL)[0]
                if w < POP_TOL:
                    if kk > 0 and all(x < POP_TOL for x in weights[kk:]):
                        break
                    continue
                c = math.sqrt((1 - eta) ** kk / math.factorial(kk))
                kraus.append((c, {j: (damp @ ak).compress(_TOL)[0]}))
        self.comb = apply_kraus(self.comb, kraus, tol=self.tol, caps=self.caps).normalize()
        self.log.append(f"loss eta={eta} on mode {mode}: {len(kraus)} Kraus operators, chi_mu={self.comb.chi_mu}")

    def _dephase(self, mode: int, sigma: float) -> None:
        assert self.comb is not None
        if sigma == 0:
            return
        x, w = np.polynomial.hermite.hermgauss(GH_NODES)
        cvm, j = self.cv[mode], self.pos(mode)
        kraus = []
        for xi, wi in zip(x, w):
            U = _single_mode_gate(cvm, "rotate", (("theta", float(np.sqrt(2) * sigma * xi)),))
            kraus.append((math.sqrt(wi / math.sqrt(math.pi)), {j: U}))
        self.comb = apply_kraus(self.comb, kraus, tol=self.tol, caps=self.caps).normalize()

    def _measure_fock(self, mode: int, n: int | None, rng) -> tuple[int, float]:
        assert self.comb is not None
        if n is None:
            pops = self._populations(mode)
            rng = rng or np.random.default_rng()
            n = int(rng.choice(len(pops), p=pops / pops.sum()))
            post = False
        else:
            post = True
        bra = self.cv[mode].fock_state(n)
        prob = self._project([{self.pos(mode): self._vacuum_bra(mode, bra)}], self.comb.trace())
        return n, (prob if post else 1.0)

    def _measure_threshold(self, mode: int, click: int | None, dark: float, rng) -> tuple[int, float]:
        assert self.comb is not None
        pops = self._populations(mode)
        if click is None:
            p_click = 1 - (1 - dark) * pops[0] / max(pops.sum(), 1e-300)
            rng = rng or np.random.default_rng()
            click = int(rng.random() < p_click)
            post = False
        else:
            post = True
        cvm, j = self.cv[mode], self.pos(mode)
        if click == 0:
            kraus = [(math.sqrt(1 - dark), {j: self._vacuum_bra(mode, cvm.vacuum())})]
        else:
            kraus = [
                (1.0, {j: self._vacuum_bra(mode, cvm.fock_state(n))}) for n in range(1, len(pops)) if pops[n] > POP_TOL
            ]
            if dark > 0:
                kraus.append((math.sqrt(dark), {j: self._vacuum_bra(mode, cvm.vacuum())}))
            if not kraus:
                raise InputError("a click has probability zero")
        prob = self._project(kraus, self.comb.trace())
        return click, (prob if post else 1.0)

    def _measure_homodyne(self, mode: int, phi: float, q: float | None, rng) -> tuple[float, float]:
        assert self.comb is not None
        cvm, j = self.cv[mode], self.pos(mode)
        if phi != 0.0:  # x_phi eigenvectors are e^{i phi n}|q>: rotate by -phi, then measure x
            U = _single_mode_gate(cvm, "rotate", (("theta", -phi),))
            self.comb = self.comb.apply_local({j: U}, tol=self.tol, caps=self.caps)
        post = q is not None
        if cvm.basis == "x":
            g = cvm.grid
            assert g is not None
            if q is None:
                probs = np.clip(np.diag(self.comb.reduced_dm(j)).real, 0, None)
                rng = rng or np.random.default_rng()
                idx = int(rng.choice(g.size, p=probs / probs.sum()))
                outcome = (idx - g.size / 2) * g.dx + g.dx * (rng.random() - 0.5)  # continuous within the cell
            else:
                idx = int(np.clip(np.rint(q / g.dx + g.size / 2), 0, g.size - 1))
                outcome = (idx - g.size / 2) * g.dx
            bra = bstates.fock(idx, cvm.bits) * (1 / math.sqrt(g.dx))
        else:
            if q is None:
                outcome = self._sample_quadrature(mode, rng or np.random.default_rng())
            else:
                outcome = float(q)
            amp = cvgrids.hermite_functions(cvm.dim, np.array([outcome]))[:, 0].astype(np.complex128)
            bra = TensorTrain.from_dense(amp, [2] * cvm.bits, tol=1e-28)
        density = self._project([{j: self._vacuum_bra(mode, bra)}], self.comb.trace())
        return float(outcome), (density if post else 1.0)

    def _quadrature_cdf(self, mode: int) -> tuple[np.ndarray, np.ndarray]:
        """Grid and cumulative distribution of ``x`` for a Fock-basis comb mode."""
        assert self.comb is not None
        rho = self.comb.reduced_dm(self.pos(mode))
        pops = np.clip(np.diag(rho).real, 0, None)
        nmax = int(np.nonzero(pops > POP_TOL)[0].max()) + 1 if np.any(pops > POP_TOL) else 1
        L = math.sqrt(2 * nmax + 1) + 8
        q = np.linspace(-L, L, 8193)
        Phi = cvgrids.hermite_functions(nmax, q)  # (nmax, len(q))
        dens = np.clip(np.einsum("nq,nm,mq->q", Phi, rho[:nmax, :nmax], Phi).real, 0, None)
        cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(q))])
        return q, cdf / cdf[-1]

    def _sample_quadrature(self, mode: int, rng: np.random.Generator, shots: int | None = None):
        q, cdf = self._quadrature_cdf(mode)
        u = rng.random() if shots is None else rng.random(shots)
        out = np.interp(u, cdf, q)
        return float(out) if shots is None else out


# ---------------------------------------------------------------------------------- sampling
def _split_last_measurement(circuit: Circuit) -> tuple[Circuit, Instruction]:
    if not len(circuit) or not circuit.instructions[-1].is_measurement:
        raise InputError("sample() needs a circuit ending with a measurement")
    last = circuit.instructions[-1]
    if last.params.get("postselect") is not None:
        raise InputError("sample() needs the last measurement without post-selection")
    prefix = circuit.copy()
    prefix.instructions.pop()
    return prefix, last


def _sample_gaussian(st: GaussianState, instr: Instruction, shots: int, rng: np.random.Generator) -> np.ndarray:
    if instr.kind != "measure_homodyne":
        raise InputError(f"the Gaussian backend cannot sample {instr!r}")
    idx = mode_indices([instr.modes[0]], st.num_modes)
    u = np.array([np.cos(instr.params["phi"]), np.sin(instr.params["phi"])])
    mu = float(u @ st.mean[idx])
    var = float(u @ st.cov[np.ix_(idx, idx)] @ u)
    return rng.normal(mu, np.sqrt(var), size=shots)


def _sample_comb(be: _CombBackend, instr: Instruction, shots: int, rng: np.random.Generator) -> np.ndarray:
    mode = instr.modes[0]
    assert be.comb is not None
    if instr.kind == "measure_fock":
        pops = be._populations(mode)
        return rng.choice(len(pops), size=shots, p=pops / pops.sum())
    if instr.kind == "measure_threshold":
        pops = be._populations(mode)
        p_click = 1 - (1 - instr.params["dark_count"]) * pops[0] / pops.sum()
        return (rng.random(shots) < p_click).astype(int)
    cvm, j = be.cv[mode], be.pos(mode)
    phi = instr.params["phi"]
    if phi != 0.0:
        U = _single_mode_gate(cvm, "rotate", (("theta", -phi),))
        be.comb = be.comb.apply_local({j: U}, tol=be.tol, caps=be.caps)
    if cvm.basis == "x":
        g = cvm.grid
        assert g is not None
        probs = np.clip(np.diag(be.comb.reduced_dm(j)).real, 0, None)
        idx = rng.choice(g.size, size=shots, p=probs / probs.sum())
        return (idx - g.size / 2) * g.dx + g.dx * (rng.random(shots) - 0.5)
    return be._sample_quadrature(mode, rng, shots)


class _Sampler:
    def sample(
        self, circuit: Circuit, shots: int, rng: np.random.Generator | None = None, state: GaussianState | None = None
    ) -> np.ndarray:
        """``shots`` outcomes of the final (not post-selected) measurement; the prefix runs once."""
        rng = rng or np.random.default_rng()
        prefix, last = _split_last_measurement(circuit)
        res = self.run(prefix, state=state, rng=rng)  # type: ignore[attr-defined]
        st = res.state
        if isinstance(st, GaussianState):
            return _sample_gaussian(st, last, shots, rng)
        be = _CombBackend(self.modes, self.tol, self.caps, [])  # type: ignore[attr-defined]
        if isinstance(st, HybridState):
            if last.modes[0] in st.gaussian_modes:
                local = Instruction(last.kind, (st.gaussian_modes.index(last.modes[0]),), dict(last.params))
                return _sample_gaussian(st.gaussian, local, shots, rng)  # type: ignore[arg-type]
            be.comb, be.order = st.comb, list(st.comb_modes)
        else:  # a plain comb is always in circuit order
            be.comb, be.order = st, list(range(circuit.num_modes))
        return _sample_comb(be, last, shots, rng)


# ---------------------------------------------------------------------------------- executors
class QTTExecutor(_Sampler):
    """Every mode on the purified comb, from the start."""

    name = "qtt"

    def __init__(self, modes: Sequence[CVMode], *, tol: float = 1e-14, chi_max: dict | None = None):
        self.modes = list(modes)
        self.tol = tol
        self.caps = chi_max

    def supports(self, instr: Instruction) -> bool:
        return True

    def run(
        self, circuit: Circuit, state: GaussianState | None = None, rng: np.random.Generator | None = None
    ) -> Result:
        if len(self.modes) != circuit.num_modes:
            raise InputError(f"{len(self.modes)} CV modes for a circuit of {circuit.num_modes}")
        logl: list[str] = []
        be = _CombBackend(self.modes, self.tol, self.caps, logl)
        M = circuit.num_modes
        if state is None:
            be.add_product(range(M), [m.vacuum() for m in self.modes])
            logl.append(f"qtt: vacuum on {M} modes")
        else:
            if state.num_modes != M:
                raise InputError(f"state has {state.num_modes} modes, the circuit {M}")
            be.add_gaussian(state, list(range(M)))
        outcomes: dict[int, float | int] = {}
        prob = 1.0
        for instr in circuit:
            out, pr = be.apply(instr, rng)
            if out is not None:
                outcomes[instr.modes[0]] = out
            prob *= pr
        comb = be.comb
        assert comb is not None
        if be.order != list(range(M)):
            logl.append(f"comb mode order: {be.order}")
        return Result(comb, outcomes, prob, logl)


def _components(
    state: GaussianState, modes: list[int], tol: float, seeds: Sequence[int] | None = None
) -> list[list[int]]:
    """Connected components (over ``modes``, positions in ``state``) of the cross-covariance graph."""
    M = state.num_modes
    V = state.cov

    def corr(i: int, j: int) -> bool:
        ii, jj = mode_indices([i], M), mode_indices([j], M)
        return float(np.abs(V[np.ix_(ii, jj)]).max()) > tol

    remaining = list(range(M))
    comps = []
    starts = list(range(M)) if seeds is None else list(seeds)
    seen: set[int] = set()
    for s in starts:
        if s in seen:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in remaining:
                if w not in seen and corr(u, w):
                    seen.add(w)
                    stack.append(w)
        comps.append(sorted(comp))
    return [[modes[i] for i in c] for c in comps]


class HybridExecutor(_Sampler):
    """Gaussian as long as possible; correlated blocks move to the comb when needed."""

    name = "hybrid"

    def __init__(
        self,
        modes: Sequence[CVMode],
        correlation_tol: float = 1e-12,
        *,
        tol: float = 1e-14,
        chi_max: dict | None = None,
    ):
        self.modes = list(modes)
        self.correlation_tol = correlation_tol
        self.tol = tol
        self.caps = chi_max

    def supports(self, instr: Instruction) -> bool:
        return True

    def run(
        self, circuit: Circuit, state: GaussianState | None = None, rng: np.random.Generator | None = None
    ) -> Result:
        M = circuit.num_modes
        if len(self.modes) != M:
            raise InputError(f"{len(self.modes)} CV modes for a circuit of {M}")
        logl: list[str] = []
        g: GaussianState | None = GaussianState.vacuum(M) if state is None else state
        g_modes = list(range(M))  # circuit modes still Gaussian, in the order of g
        be = _CombBackend(self.modes, self.tol, self.caps, logl)
        outcomes: dict[int, float | int] = {}
        prob = 1.0
        for step, instr in enumerate(circuit):
            on_comb = [m for m in instr.modes if m in be.order]
            if instr.is_gaussian and not on_comb:
                idx = [g_modes.index(m) for m in instr.modes]
                local = Instruction(instr.kind, tuple(idx), dict(instr.params), True)
                assert g is not None
                g, out, pr = GaussianExecutor.apply(g, local, rng)
            else:
                need = [m for m in instr.modes if m not in be.order]
                if need:
                    assert g is not None
                    seeds = [g_modes.index(m) for m in need]
                    comps = _components(g, g_modes, self.correlation_tol, seeds)
                    move = sorted({m for c in comps for m in c})
                    pos = [g_modes.index(m) for m in move]
                    be.add_gaussian(g.reduce(pos), move)
                    logl.append(f"step {step} ({instr.kind}): modes {move} moved to the comb")
                    rest = [i for i in range(len(g_modes)) if i not in pos]
                    g = g.reduce(rest) if rest else None
                    g_modes = [g_modes[i] for i in rest]
                out, pr = be.apply(instr, rng)
            if out is not None:
                outcomes[instr.modes[0]] = out
            prob *= pr
        if be.comb is None:
            logl.append("hybrid: the circuit never left the Gaussian backend")
            final: Any = g
        elif not g_modes and be.order == list(range(M)):
            final = be.comb
        else:
            final = HybridState(g, tuple(g_modes), be.comb, tuple(be.order))
        return Result(final, outcomes, prob, logl)
