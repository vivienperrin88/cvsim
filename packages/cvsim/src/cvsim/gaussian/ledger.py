"""Noise accounting (decision D7): which component added how much noise, exactly.

A :class:`NoiseLedger` follows a state through gates, Gaussian channels and phase diffusion and
keeps, besides the state itself, the decomposition (CONVENTIONS.md, "Noise ledger")

    V_out = signal + Σ_k C_k,     signal = 𝓛_n ⋯ 𝓛_1 (V_in),     C_k = 𝓛_n ⋯ 𝓛_{k+1} (Y_k),

where ``𝓛_j`` is the linear map step ``j`` applies to covariance matrices (``V → X V Xᵀ`` for a gate
or channel, the dephasing map for phase diffusion) and ``Y_k`` the noise step ``k`` adds there. Each
contribution is the noise of one component *carried forward to the output*; they add up to
``V_out − signal`` exactly, and are tagged with the component name and the kind of imperfection.
Loss also attenuates the signal; the ledger records the transmissivity of each loss-like step.

Variances do add, decibels do not: :meth:`NoiseLedger.db_recovered` answers "how many dB would
this component alone cost", by replaying the chain with that component made ideal. Those numbers
are a what-if view and do not sum to the total.

Phase diffusion is not Gaussian; its step uses the moment-matched map (see
:func:`~cvsim.gaussian.channels.phase_diffusion`): the contribution it books is the variance of the
randomly rotated mean, and its entries are flagged ``approximate``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..core import conventions as cv
from ..core._types import FloatArray
from ..errors import ApproximationWarning, InputError
from .channels import GaussianChannel, dephase_mean, dephase_moments
from .states import GaussianState

__all__ = ["KINDS", "BudgetRow", "Contribution", "NoiseLedger"]

#: Suggested kinds of imperfection (any string is accepted; these group naturally).
KINDS = ("loss", "gain", "thermal", "electronic", "phase", "excess", "mismatch", "other")

_Map = Callable[[FloatArray], FloatArray]


class Contribution(NamedTuple):
    """Noise added by one step, carried forward to the output (``cov``, ``2M x 2M``)."""

    component: str
    kind: str
    step: int
    cov: FloatArray
    approximate: bool
    transmissivity: float | None


class BudgetRow(NamedTuple):
    """One line of a quadrature budget: variance and its share of the output variance."""

    label: str
    kind: str
    variance: float
    fraction: float


@dataclass(frozen=True)
class _Step:
    kind: str  # "gate" | "channel" | "phase"
    component: str
    noise_kind: str
    modes: tuple[int, ...]
    channel: GaussianChannel | None = None
    sigma: float = 0.0
    transmissivity: float | None = None


@dataclass(frozen=True)
class NoiseLedger:
    """A state followed through an optical chain, with its exact noise decomposition.

    Build it with :meth:`start`, then chain :meth:`gate`, :meth:`channel` and
    :meth:`phase_diffusion`; each returns a new ledger.
    """

    initial: GaussianState
    steps: tuple[_Step, ...] = ()
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------------- building
    @classmethod
    def start(cls, state: GaussianState) -> NoiseLedger:
        """A ledger whose input is ``state`` (its covariance is the signal).

        Parameters
        ----------
        state
            Input state.
        """
        if not isinstance(state, GaussianState):
            raise InputError(f"expected a GaussianState, got {type(state).__name__}")
        return cls(state)

    def _modes(self, modes: Sequence[int] | int | None, m: int) -> tuple[int, ...]:
        M = self.initial.num_modes
        if modes is None:
            ms = tuple(range(M))
        elif isinstance(modes, (int, np.integer)):
            ms = (int(modes),)
        else:
            ms = tuple(int(k) for k in modes)
        cv.mode_indices(ms, M)
        if len(ms) != m:
            raise InputError(f"operation acts on {m} mode(s), got modes {ms}")
        return ms

    def gate(self, S: npt.ArrayLike, modes: Sequence[int] | int | None = None, component: str = "gate") -> NoiseLedger:
        """Append a gate (no noise; it transforms the signal and every earlier contribution).

        Parameters
        ----------
        S
            Symplectic matrix on ``modes``.
        modes
            Modes it acts on (all by default).
        component
            Name, for the record.
        """
        ch = GaussianChannel.from_symplectic(S)
        step = _Step("gate", component, "", self._modes(modes, ch.num_modes), ch)
        return NoiseLedger(self.initial, (*self.steps, step))

    def channel(
        self,
        channel: GaussianChannel,
        modes: Sequence[int] | int | None = None,
        component: str = "channel",
        kind: str = "other",
    ) -> NoiseLedger:
        """Append a Gaussian channel; its ``Y`` is booked under ``component`` and ``kind``.

        Parameters
        ----------
        channel
            The channel.
        modes
            Modes it acts on (all by default).
        component
            Hardware component it models (e.g. ``"OPO escape"``, ``"detector 2"``).
        kind
            Kind of imperfection, e.g. one of :data:`KINDS`.
        """
        if not isinstance(channel, GaussianChannel):
            raise InputError(f"expected a GaussianChannel, got {type(channel).__name__}")
        X = channel.X
        g = float(X[0, 0]) ** 2
        eta = g if g < 1.0 and np.allclose(X, math.sqrt(g) * np.eye(X.shape[0]), atol=1e-14) else None
        step = _Step("channel", component, kind, self._modes(modes, channel.num_modes), channel, transmissivity=eta)
        return NoiseLedger(self.initial, (*self.steps, step))

    def phase_diffusion(self, sigma: float, mode: int = 0, component: str = "phase noise") -> NoiseLedger:
        """Append moment-matched phase diffusion ``φ ~ N(0, σ²)`` on ``mode`` (kind ``"phase"``).

        Parameters
        ----------
        sigma
            Phase standard deviation (radians).
        mode
            Mode index.
        component
            Name of the source (e.g. ``"LO lock"``).
        """
        if not sigma >= 0 or not math.isfinite(sigma):
            raise InputError(f"sigma must be finite and >= 0, got {sigma}")
        step = _Step("phase", component, "phase", self._modes([mode], 1), sigma=float(sigma))
        return NoiseLedger(self.initial, (*self.steps, step))

    # ------------------------------------------------------------------------- evaluation
    def _maps(self, step: _Step) -> tuple[_Map, Callable[[FloatArray], FloatArray]]:
        """(map on covariance-like matrices, map on mean vectors) of one step on all modes."""
        M = self.initial.num_modes
        if step.kind == "phase":
            mode = step.modes[0]
            return (lambda A: dephase_moments(A, step.sigma, mode)), (lambda d: dephase_mean(d, step.sigma, mode))
        assert step.channel is not None
        ch = step.channel.embed(step.modes, M)
        return (lambda A: ch.X @ A @ ch.X.T), (lambda d: ch.X @ d + ch.c)

    def _run(self, skip: str | None = None) -> tuple[FloatArray, FloatArray, list[Contribution]]:
        d = self.initial.mean.copy()
        signal = self.initial.cov.copy()
        contribs: list[tuple[_Step, int, FloatArray]] = []
        for k, step in enumerate(self.steps):
            if skip is not None and step.kind != "gate" and step.component == skip:
                continue
            on_cov, on_mean = self._maps(step)
            new_d = on_mean(d)
            signal = on_cov(signal)
            contribs = [(s, j, on_cov(C)) for s, j, C in contribs]
            if step.kind == "channel":
                assert step.channel is not None
                added = step.channel.embed(step.modes, self.initial.num_modes).Y
            elif step.kind == "phase":
                # variance of the randomly rotated mean: E[R d dᵀ Rᵀ] − E[R d] E[R d]ᵀ
                added = dephase_moments(np.outer(d, d), step.sigma, step.modes[0]) - np.outer(new_d, new_d)
            else:
                added = None
            if added is not None:
                contribs.append((step, k, (added + added.T) / 2))
            d = new_d
        out = [
            Contribution(s.component, s.noise_kind, j, C, s.kind == "phase", s.transmissivity) for s, j, C in contribs
        ]
        return d, signal, out

    def _evaluated(self) -> tuple[FloatArray, FloatArray, list[Contribution]]:
        if "run" not in self._cache:
            if any(s.kind == "phase" and s.sigma > 0 for s in self.steps):
                warnings.warn(
                    "NoiseLedger: phase diffusion uses a moment-matched Gaussian approximation",
                    ApproximationWarning,
                    stacklevel=3,
                )
            self._cache["run"] = self._run()
        return self._cache["run"]  # type: ignore[no-any-return]

    @property
    def state(self) -> GaussianState:
        """The output state (exact for gates and Gaussian channels)."""
        d, signal, contribs = self._evaluated()
        V = signal + sum((c.cov for c in contribs), np.zeros_like(signal))
        return GaussianState(d, (V + V.T) / 2, validate=False)

    @property
    def signal(self) -> FloatArray:
        """The input covariance carried through the whole chain (no added noise)."""
        return self._evaluated()[1]

    @property
    def contributions(self) -> list[Contribution]:
        """One entry per noisy step, in chain order; their ``cov`` add up to ``state.cov − signal``."""
        return self._evaluated()[2]

    def added_noise(self) -> FloatArray:
        """``Σ_k C_k``."""
        return sum((c.cov for c in self.contributions), np.zeros_like(self.signal))

    def by(self, key: str = "component") -> dict[str, FloatArray]:
        """Contributions summed per ``"component"`` or per ``"kind"``, in order of first appearance.

        Parameters
        ----------
        key
            ``"component"`` or ``"kind"``.
        """
        if key not in ("component", "kind"):
            raise InputError(f"key must be 'component' or 'kind', got {key!r}")
        out: dict[str, FloatArray] = {}
        for c in self.contributions:
            name = c.component if key == "component" else c.kind
            out[name] = out.get(name, np.zeros_like(c.cov)) + c.cov
        return out

    def transmissivities(self) -> dict[str, float]:
        """Product of the transmissivities of each component's loss-like steps (``X = √η I``)."""
        out: dict[str, float] = {}
        for c in self.contributions:
            if c.transmissivity is not None:
                out[c.component] = out.get(c.component, 1.0) * c.transmissivity
        return out

    # ------------------------------------------------------------------------- quadrature reports
    def _u(self, mode: int, phi: float) -> FloatArray:
        M = self.initial.num_modes
        u = np.zeros(2 * M)
        i = cv.mode_indices([mode], M)
        u[i] = cv.homodyne_vector(phi)
        return u

    def budget(self, mode: int = 0, phi: float = 0.0, key: str = "component", snu: bool = True) -> list[BudgetRow]:
        """Variance of ``x_φ`` of ``mode`` split into signal and contributions (exact, additive).

        Parameters
        ----------
        mode
            Mode measured.
        phi
            Quadrature angle (``x_φ = cos φ x + sin φ p``).
        key
            Group contributions by ``"component"`` or ``"kind"``.
        snu
            Report in shot-noise units (vacuum = 1) rather than ħ = 1 units (vacuum = ½).
        """
        u = self._u(mode, phi)
        unit = 1.0 / cv.VACUUM_VARIANCE if snu else 1.0
        total = float(u @ self.state.cov @ u)
        kinds = {c.component: c.kind for c in self.contributions} if key == "component" else {}
        rows = [BudgetRow("signal", "", unit * float(u @ self.signal @ u), float(u @ self.signal @ u) / total)]
        for name, C in self.by(key).items():
            v = float(u @ C @ u)
            rows.append(BudgetRow(name, kinds.get(name, name), unit * v, v / total))
        return rows

    def db_recovered(self, component: str, mode: int = 0, phi: float = 0.0) -> float:
        """What-if (not additive): dB by which the ``x_φ`` noise of ``mode`` would fall if every step of
        ``component`` were ideal (removed), all else unchanged.

        Parameters
        ----------
        component
            Component name.
        mode
            Mode measured.
        phi
            Quadrature angle.
        """
        if component not in {s.component for s in self.steps if s.kind != "gate"}:
            raise InputError(f"no noisy step from component {component!r}")
        u = self._u(mode, phi)
        _, signal, contribs = self._run(skip=component)
        ideal = signal + sum((c.cov for c in contribs), np.zeros_like(signal))
        return 10.0 * math.log10(float(u @ self.state.cov @ u) / float(u @ ideal @ u))
