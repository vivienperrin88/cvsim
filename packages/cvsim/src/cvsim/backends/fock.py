"""Dense Fock-space backend: density matrices truncated at ``cutoff`` photons per mode.

Runs every operation of the IR, Gaussian or not, so it is the reference for non-Gaussian programs
and a third engine for Gaussian ones. Gates are the exponentials of their generators
(CONVENTIONS.md, "Gates") on the modes they touch; ``Interferometer`` is run as its Clements mesh
and ``Symplectic`` as Bloch–Messiah (interferometer, squeezers, interferometer). Loss and
amplification with a thermal environment are dilations (beam splitter or two-mode squeezer with a
thermal ancilla, traced out); additive noise of variance ``σ²`` is loss ``1/G`` then gain ``G``
with ``G = 1 + σ²`` (an exact identity of Gaussian channels); phase noise is exact dephasing
``ρ_nm → ρ_nm e^{−σ²(n−m)²/2}``. Measurements follow the detector model of CONVENTIONS.md; their
outcomes are post-selected or, for homodyne, sampled from the marginal on a fine grid (sampled
heterodyne is not supported here). A :class:`~cvsim.errors.CutoffWarning` is issued when any
mode's population at the highest kept level exceeds ``cutoff_tol``: the truncation then matters.

Cost: a state of ``m`` modes holds ``cutoff^{2m}`` numbers; meant for a few modes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np

from ..core import linalg as la
from ..errors import CutoffWarning, InputError
from ..gaussian import symplectic as sy
from ..ir import Op, Program
from .base import Outcome, Result, check_runnable

__all__ = ["FockBackend", "FockState"]

_SUPPORTED = (
    "Vacuum", "Coherent", "Squeezed", "Thermal", "Fock",
    "D", "R", "S", "P", "F", "BS", "S2", "CZ", "CX", "Interferometer", "Symplectic", "Kerr",
    "Loss", "Amplifier", "AdditiveNoise", "PhaseNoise",
    "Homodyne", "Heterodyne", "PNR", "Threshold",
)  # fmt: skip


# ------------------------------------------------------------------------------- single-mode algebra
def _destroy(N: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1).astype(complex)


def _hermite_functions(N: int, q: np.ndarray) -> np.ndarray:
    """``ψ_n(q) = ⟨q|n⟩`` (ħ = 1), shape ``(N, len(q))``."""
    psi = np.zeros((N, q.size))
    psi[0] = math.pi**-0.25 * np.exp(-(q**2) / 2)
    if N > 1:
        psi[1] = math.sqrt(2.0) * q * psi[0]
    for n in range(2, N):
        psi[n] = math.sqrt(2.0 / n) * q * psi[n - 1] - math.sqrt((n - 1) / n) * psi[n - 2]
    return psi


class FockState:
    """Density matrix of ``m`` modes, each truncated at ``cutoff`` levels (mode 0 most significant).

    Parameters
    ----------
    dm
        ``cutoff^m x cutoff^m`` density matrix.
    cutoff
        Levels per mode.
    """

    def __init__(self, dm: np.ndarray, cutoff: int):
        d = np.asarray(dm, dtype=complex)
        m = round(math.log(d.shape[0], cutoff)) if d.shape[0] > 1 else 0
        if d.ndim != 2 or d.shape[0] != d.shape[1] or cutoff**m != d.shape[0]:
            raise InputError(f"dm must be cutoff^m x cutoff^m, got shape {d.shape} for cutoff {cutoff}")
        self.dm = d
        self.cutoff = int(cutoff)
        self.num_modes = int(m)

    def __repr__(self) -> str:
        return f"FockState(modes={self.num_modes}, cutoff={self.cutoff}, trace={self.trace():.6g})"

    def trace(self) -> float:
        """``Tr ρ`` (below 1 if the truncation lost population)."""
        return float(np.trace(self.dm).real)

    def purity(self) -> float:
        """``Tr ρ²``."""
        return float(np.real(np.vdot(self.dm.conj().T, self.dm)))

    def _ops(self) -> list[np.ndarray]:
        N, m = self.cutoff, self.num_modes
        a = _destroy(N)
        out = []
        for k in range(m):
            out.append(np.kron(np.kron(np.eye(N**k), a), np.eye(N ** (m - k - 1))))
        return out

    def mean_photons(self) -> np.ndarray:
        """``⟨n_k⟩`` per mode."""
        return np.array([float(np.trace(self.dm @ a.conj().T @ a).real) for a in self._ops()])

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance (xxpp, ħ = 1) of the quadratures."""
        a = self._ops()
        r = [(x + x.conj().T) / math.sqrt(2) for x in a] + [(x - x.conj().T) / (1j * math.sqrt(2)) for x in a]
        mean = np.array([np.trace(self.dm @ ri).real for ri in r])
        cov = np.array(
            [
                [0.5 * np.trace(self.dm @ (ri @ rj + rj @ ri)).real - mean[i] * mean[j] for j, rj in enumerate(r)]
                for i, ri in enumerate(r)
            ]
        )
        return mean, cov

    def reduce(self, modes: Sequence[int]) -> FockState:
        """Partial trace, keeping ``modes`` in the order listed.

        Parameters
        ----------
        modes
            Distinct mode indices.
        """
        keep = [int(k) for k in modes]
        if len(set(keep)) != len(keep) or any(not 0 <= k < self.num_modes for k in keep):
            raise InputError(f"modes must be distinct in 0..{self.num_modes - 1}, got {modes}")
        t = _tensor(self.dm, self.num_modes, self.cutoff)
        m = self.num_modes
        for k in sorted(set(range(m)) - set(keep), reverse=True):
            t = np.trace(t, axis1=k, axis2=k + t.ndim // 2)
        remaining = sorted(keep)
        order = [remaining.index(k) for k in keep]
        n = len(keep)
        t = t.transpose([*order, *(n + i for i in order)])
        return FockState(t.reshape(self.cutoff**n, self.cutoff**n), self.cutoff)


def _tensor(dm: np.ndarray, m: int, N: int) -> np.ndarray:
    return dm.reshape((N,) * (2 * m))


# ------------------------------------------------------------------------------- the backend
class FockBackend:
    """Dense density-matrix simulator.

    Parameters
    ----------
    cutoff
        Fock levels kept per mode (photon numbers ``0..cutoff−1``).
    cutoff_tol
        Population at the highest level above which a ``CutoffWarning`` is issued.
    """

    def __init__(self, cutoff: int = 20, cutoff_tol: float = 1e-8):
        if not isinstance(cutoff, (int, np.integer)) or cutoff < 2:
            raise InputError(f"cutoff must be an integer >= 2, got {cutoff!r}")
        self.cutoff = int(cutoff)
        self.cutoff_tol = float(cutoff_tol)
        self.name = f"fock[cutoff={self.cutoff}]"
        N = self.cutoff
        self._a = _destroy(N)
        self._ad = self._a.conj().T
        self._n = np.diag(np.arange(N, dtype=float)).astype(complex)
        self._x = (self._a + self._ad) / math.sqrt(2)
        self._p = (self._a - self._ad) / (1j * math.sqrt(2))
        self._I = np.eye(N, dtype=complex)

    @property
    def supported(self) -> tuple[str, ...]:
        """Operation names this backend runs."""
        return _SUPPORTED

    def check(self, program: Program, rng: np.random.Generator | None = None) -> None:
        """Raise ``InputError`` naming the first operation this backend cannot run.

        Parameters
        ----------
        program
            Bound program.
        rng
            The generator the run will get.
        """
        check_runnable(program, _SUPPORTED, self.name, rng if rng is not None else np.random.default_rng(0))
        for k, o in enumerate(program.ops):
            if o.name == "Heterodyne" and o.params["outcome"] is None:
                raise InputError(f"{self.name} cannot sample operation {k} (Heterodyne): give its outcome")

    # ------------------------------------------------------------------------- tensors
    def _apply(self, t: np.ndarray, U: np.ndarray, axes: Sequence[int]) -> np.ndarray:
        """``U ρ U†`` with ``U`` acting on ``axes`` (ket axes; the bra axes follow)."""
        N, k = self.cutoff, len(axes)
        m = t.ndim // 2
        Ut = U.reshape((N,) * (2 * k))
        t = np.tensordot(Ut, t, axes=(list(range(k, 2 * k)), list(axes)))
        t = np.moveaxis(t, list(range(k)), list(axes))
        bra = [m + a for a in axes]
        t = np.tensordot(t, Ut.conj(), axes=(bra, list(range(k, 2 * k))))
        return np.moveaxis(t, list(range(t.ndim - k, t.ndim)), bra)

    def _kraus(self, t: np.ndarray, Ks: Sequence[np.ndarray], axis: int) -> np.ndarray:
        return sum((self._apply(t, K, [axis]) for K in Ks), np.zeros_like(t))

    def _gate(self, G: np.ndarray) -> np.ndarray:
        return la.expm(G)

    def _two(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.kron(A, B)

    def _unitary(self, o: Op) -> np.ndarray:
        p, a, ad, x, pp, I = o.params, self._a, self._ad, self._x, self._p, self._I
        name = o.name
        if name == "D":
            al = complex(p["alpha"])
            return self._gate(al * ad - np.conj(al) * a)
        if name == "R":
            return np.diag(np.exp(1j * p["theta"] * np.arange(self.cutoff)))
        if name == "F":
            return np.diag(np.exp(1j * (math.pi / 2) * np.arange(self.cutoff)))
        if name == "Kerr":
            return np.diag(np.exp(1j * p["kappa"] * np.arange(self.cutoff) ** 2))
        if name == "S":
            z = p["r"] * complex(math.cos(p["phi"]), math.sin(p["phi"]))
            return self._gate((np.conj(z) * a @ a - z * ad @ ad) / 2)
        if name == "P":
            return self._gate(1j * p["s"] * x @ x / 2)
        A, Ad, B, Bd = self._two(a, I), self._two(ad, I), self._two(I, a), self._two(I, ad)
        if name == "BS":
            e = complex(math.cos(p["phi"]), math.sin(p["phi"]))
            return self._gate(p["theta"] * (e * A @ Bd - np.conj(e) * Ad @ B))
        if name == "S2":
            z = p["r"] * complex(math.cos(p["phi"]), math.sin(p["phi"]))
            return self._gate(np.conj(z) * A @ B - z * Ad @ Bd)
        if name == "CZ":
            return self._gate(1j * p["g"] * self._two(x, I) @ self._two(I, x))
        if name == "CX":
            return self._gate(-1j * p["g"] * self._two(x, I) @ self._two(I, pp))
        raise AssertionError(name)  # pragma: no cover - guarded by check_runnable

    # ------------------------------------------------------------------------- mode bookkeeping
    def _single(self, o: Op) -> np.ndarray:
        """Density matrix of a one-mode preparation."""
        N, p = self.cutoff, o.params
        if o.name == "Thermal":
            nb = p["nbar"]
            w = (nb / (1 + nb)) ** np.arange(N) / (1 + nb)
            return np.diag(w).astype(complex)
        ket = np.zeros(N, dtype=complex)
        if o.name == "Fock":
            if p["n"] >= N:
                raise InputError(f"Fock state |{p['n']}⟩ needs cutoff > {p['n']} (cutoff is {N})")
            ket[p["n"]] = 1
            return np.outer(ket, ket.conj())
        ket[0] = 1
        if o.name == "Coherent":
            ket = self._unitary(Op("D", {"alpha": p["alpha"]}, (0,))) @ ket
        elif o.name == "Squeezed":
            ket = self._unitary(Op("S", {"r": p["r"], "phi": p["phi"]}, (0,))) @ ket
        return np.outer(ket, ket.conj())

    def _add_mode(
        self, t: np.ndarray | None, live: list[int], rho1: np.ndarray, k: int
    ) -> tuple[np.ndarray, list[int]]:
        """Tensor ``rho1`` in as register mode ``k``, keeping modes sorted."""
        if t is None:
            return rho1, [k]
        m = t.ndim // 2
        big = np.multiply.outer(t, rho1)  # axes: kets(m), bras(m), new ket, new bra
        big = np.moveaxis(big, 2 * m, m)  # kets(m), new ket, bras(m), new bra
        modes = [*live, k]
        order = sorted(range(m + 1), key=lambda i: modes[i])
        big = big.transpose([*order, *(m + 1 + i for i in order)])
        return big, sorted(modes)

    def _trace_axis(self, t: np.ndarray, axis: int) -> np.ndarray | None:
        m = t.ndim // 2
        if m == 1:
            return None
        return np.asarray(np.trace(t, axis1=axis, axis2=m + axis))

    def _channel(self, t: np.ndarray, o: Op, axis: int) -> np.ndarray:
        p = o.params
        if o.name == "PhaseNoise":
            n = np.arange(self.cutoff)
            damp = np.exp(-(p["sigma"] ** 2) * (n[:, None] - n[None, :]) ** 2 / 2)
            m = t.ndim // 2
            d = damp.reshape([self.cutoff if i in (axis, m + axis) else 1 for i in range(t.ndim)])
            return np.asarray(t * d)
        if o.name == "Loss":
            return self._thermal_loss(t, axis, p["eta"], p["nbar"])
        if o.name == "Amplifier":
            return self._thermal_amplifier(t, axis, p["gain"], p["nbar"])
        G = 1.0 + p["sigma2"]  # AdditiveNoise: loss 1/G then gain G
        return self._amplify(self._loss(t, axis, 1.0 / G), axis, G)

    # Every phase-insensitive Gaussian channel is pure loss then a quantum-limited amplifier
    # (exact identities of (X, Y), checked in the tests): thermal loss (η, n̄) is loss η/G' then
    # gain G' = 1 + (1 − η) n̄; a thermal amplifier (G, n̄) is loss G/G' then gain G' = G + (G − 1) n̄.
    def _thermal_loss(self, t: np.ndarray, axis: int, eta: float, nbar: float) -> np.ndarray:
        g = 1.0 + (1.0 - eta) * nbar
        return self._amplify(self._loss(t, axis, eta / g), axis, g) if g > 1.0 else self._loss(t, axis, eta)

    def _thermal_amplifier(self, t: np.ndarray, axis: int, gain: float, nbar: float) -> np.ndarray:
        g = gain + (gain - 1.0) * nbar
        return self._amplify(self._loss(t, axis, gain / g), axis, g) if g > gain else self._amplify(t, axis, gain)

    def _loss(self, t: np.ndarray, axis: int, eta: float) -> np.ndarray:
        """Pure loss, Kraus ``K_k = Σ_n √(C(n, k) (1−η)^k η^(n−k)) |n−k⟩⟨n|``."""
        if eta >= 1.0:
            return t
        N = self.cutoff
        Ks = []
        for k in range(N):
            K = np.zeros((N, N), dtype=complex)
            for nn in range(k, N):
                K[nn - k, nn] = math.sqrt(math.comb(nn, k) * (1 - eta) ** k * eta ** (nn - k))
            Ks.append(K)
        return self._kraus(t, Ks, axis)

    def _amplify(self, t: np.ndarray, axis: int, gain: float) -> np.ndarray:
        """Quantum-limited amplifier, Kraus ``A_k = Σ_n √C(n+k, k) tanh^k r / cosh^{n+1} r |n+k⟩⟨n|``
        (a two-mode squeezer ``cosh² r = G`` with a vacuum idler measured in ``|k⟩``)."""
        if gain <= 1.0:
            return t
        N = self.cutoff
        c, th = math.sqrt(gain), math.sqrt((gain - 1.0) / gain)
        Ks = []
        for k in range(N):
            K = np.zeros((N, N), dtype=complex)
            for nn in range(N - k):
                K[nn + k, nn] = math.sqrt(math.comb(nn + k, k)) * th**k / c ** (nn + 1)
            Ks.append(K)
        return self._kraus(t, Ks, axis)

    def _gate_sequence(self, o: Op) -> list[Op]:
        """Interferometer → Clements gates; Symplectic → Bloch–Messiah (U2, squeezers, U1)."""
        if o.name == "Interferometer":
            mesh = sy.clements(np.asarray(o.params["U"]))
            return [_gate_op(name, par, tuple(o.modes[i] for i in loc)) for name, par, loc in mesh.gates()]
        S = np.asarray(o.params["S"], dtype=float)
        bm = la.bloch_messiah(S)
        U1, U2 = sy.unitary_from_passive(bm.O1, atol=1e-8), sy.unitary_from_passive(bm.O2, atol=1e-8)
        seq = [_gate_op(name, par, tuple(o.modes[i] for i in loc)) for name, par, loc in sy.clements(U2).gates()]
        seq += [Op("S", {"r": float(r), "phi": 0.0}, (o.modes[i],)) for i, r in enumerate(bm.r) if r > 0]
        seq += [_gate_op(name, par, tuple(o.modes[i] for i in loc)) for name, par, loc in sy.clements(U1).gates()]
        return seq

    # ------------------------------------------------------------------------- run
    def run(self, program: Program, rng: np.random.Generator | None = None) -> Result:
        """Execute ``program`` from the vacuum.

        Parameters
        ----------
        program
            Bound program.
        rng
            Generator for sampled homodyne outcomes.
        """
        self.check(program, rng)
        check_runnable(program, _SUPPORTED, self.name, rng)
        N, M = self.cutoff, program.num_modes
        vac = np.zeros((N, N), dtype=complex)
        vac[0, 0] = 1
        t: np.ndarray | None = None
        live: list[int] = []
        for m in range(M):
            t, live = self._add_mode(t, live, vac, m)
        outcomes: list[Outcome] = []
        warned = False
        for k, o in enumerate(program.ops):
            if o.kind == "prepare":
                m0 = o.modes[0]
                if m0 in live:
                    assert t is not None
                    t = self._trace_axis(t, live.index(m0))
                    live = [x for x in live if x != m0]
                t, live = self._add_mode(t, live, self._single(o), m0)
            elif o.kind == "measure":
                assert t is not None
                t, value, density = self._measure(t, o, live.index(o.modes[0]), rng)
                outcomes.append(Outcome(k, o.name, o.modes, value, density, o.params["outcome"] is not None))
                live = [x for x in live if x != o.modes[0]]
            else:
                assert t is not None
                for step in self._gate_sequence(o) if o.name in ("Interferometer", "Symplectic") else [o]:
                    axes = [live.index(m) for m in step.modes]
                    t = (
                        self._channel(t, step, axes[0])
                        if step.kind == "channel"
                        else self._apply(t, self._unitary(step), axes)
                    )
            if t is not None and not warned:
                warned = self._check_cutoff(t, k)
        if t is None:
            return Result(None, (), tuple(outcomes), self.name)
        m = t.ndim // 2
        state = FockState(t.reshape(N**m, N**m), N)
        return Result(state, tuple(live), tuple(outcomes), self.name)

    def _check_cutoff(self, t: np.ndarray, k: int) -> bool:
        m = t.ndim // 2
        flat = t.reshape(self.cutoff**m, self.cutoff**m)
        diag = np.real(np.diag(flat)).reshape((self.cutoff,) * m)
        for axis in range(m):
            top = float(np.take(diag, self.cutoff - 1, axis=axis).sum())
            if top > self.cutoff_tol:
                warnings.warn(
                    f"{self.name}: after operation {k}, population {top:.2e} at the cutoff "
                    f"(mode slot {axis}); raise the cutoff",
                    CutoffWarning,
                    stacklevel=3,
                )
                return True
        return False

    def _count(
        self, t: np.ndarray, o: Op, axis: int, rng: np.random.Generator | None
    ) -> tuple[np.ndarray | None, int, float]:
        """PNR / click detection: loss η, then the POVM of the dark-count model (CONVENTIONS.md)."""
        p, N = o.params, self.cutoff
        t = self._loss(t, axis, p["efficiency"])
        m = t.ndim // 2
        red = t
        for ax in sorted(set(range(m)) - {axis}, reverse=True):
            red = np.trace(red, axis1=ax, axis2=ax + red.ndim // 2)
        diag = np.clip(np.real(np.diag(red)), 0.0, None)

        def weights(n: int) -> np.ndarray:
            """POVM element of outcome ``n`` on the Fock diagonal."""
            w = np.zeros(N)
            if o.name == "PNR":
                nu = p["dark_counts"]
                for k in range(min(n, N - 1) + 1):
                    w[k] = math.exp(-nu) * nu ** (n - k) / math.factorial(n - k) if nu > 0 else float(k == n)
                return w
            pd = p["dark_prob"]
            if n == 0:
                w[0] = 1.0 - pd
            else:
                w[:] = 1.0
                w[0] = pd
            return w

        n_outcomes = N if o.name == "PNR" else 2
        if p["outcome"] is None:
            assert rng is not None
            probs = np.array([weights(n) @ diag for n in range(n_outcomes)])
            n_out = int(rng.choice(n_outcomes, p=probs / probs.sum()))
        else:
            n_out = int(p["outcome"])
            if n_out >= n_outcomes:
                raise InputError(f"{o.name} outcome {n_out} is out of range (cutoff {N})")
        w = weights(n_out)
        if m == 1:
            return None, n_out, float(w @ diag)
        rest = np.zeros((N,) * (2 * (m - 1)), dtype=complex)
        for k in range(N):
            if w[k] != 0.0:
                rest += w[k] * np.take(np.take(t, k, axis=axis), k, axis=m - 1 + axis)
        prob = float(np.real(np.trace(rest.reshape(N ** (m - 1), N ** (m - 1)))))
        return (rest / prob if prob > 0 else rest), n_out, prob

    def _measure(
        self, t: np.ndarray, o: Op, axis: int, rng: np.random.Generator | None
    ) -> tuple[np.ndarray | None, float | complex, float]:
        p = o.params
        if o.name in ("PNR", "Threshold"):
            return self._count(t, o, axis, rng)
        eta, vel = p["efficiency"], p["electronic"]
        het = o.name == "Heterodyne"
        if eta < 1:
            t = self._loss(t, axis, eta)
        if vel > 0:
            G = 1.0 + (vel if het else vel / 2)
            t = self._amplify(self._loss(t, axis, 1.0 / G), axis, G)
        scale = math.sqrt(eta)
        if het:
            alpha = complex(p["outcome"]) * scale
            ket = np.zeros(self.cutoff, dtype=complex)
            ket[0] = 1
            ket = self._unitary(Op("D", {"alpha": alpha}, (0,))) @ ket
            vec = ket.conj()
            rest, prob = self._project(t, axis, vec)
            return rest, alpha / scale, prob / math.pi * eta
        t = self._apply(t, np.diag(np.exp(-1j * p["phi"] * np.arange(self.cutoff))), [axis])  # R(−φ): x_φ → x
        if p["outcome"] is None:
            assert rng is not None
            q = self._sample_homodyne(t, axis, rng)
        else:
            q = float(p["outcome"]) * scale
        vec = _hermite_functions(self.cutoff, np.array([q]))[:, 0].astype(complex)
        rest, density = self._project(t, axis, vec)
        return rest, q / scale, density * scale

    def _project(self, t: np.ndarray, axis: int, bra: np.ndarray) -> tuple[np.ndarray | None, float]:
        """``⟨v|ρ|v⟩`` on ``axis`` (``bra`` = conj(v)): the unnormalised rest and its trace."""
        m = t.ndim // 2
        r = np.tensordot(bra, t, axes=([0], [axis]))  # ket axis removed
        r = np.tensordot(r, bra.conj(), axes=([m - 1 + axis], [0]))  # bra axis of that mode
        if m == 1:
            return None, float(np.real(r))
        k = m - 1
        prob = float(np.real(np.trace(r.reshape(self.cutoff**k, self.cutoff**k))))
        return (r / prob if prob > 0 else r), prob

    def _sample_homodyne(self, t: np.ndarray, axis: int, rng: np.random.Generator) -> float:
        m = t.ndim // 2
        red = t
        for ax in sorted(set(range(m)) - {axis}, reverse=True):
            red = np.trace(red, axis1=ax, axis2=ax + red.ndim // 2)
        L = math.sqrt(2 * self.cutoff + 1) + 4.0
        q = np.linspace(-L, L, 8001)
        psi = _hermite_functions(self.cutoff, q)
        p = np.maximum(np.real(np.sum(psi * (red @ psi), axis=0)), 0)  # ⟨q|ρ|q⟩
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        return float(np.interp(rng.uniform(), cdf, q))


def _gate_op(name: str, par: tuple[float, ...], modes: tuple[int, ...]) -> Op:
    if name == "R":
        return Op("R", {"theta": par[0]}, modes)
    return Op("BS", {"theta": par[0], "phi": par[1]}, modes)
