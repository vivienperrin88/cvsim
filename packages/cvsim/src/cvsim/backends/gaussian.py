"""Gaussian backend, on cvsim's own engine or on qttlind's (decision D2: the switch).

Both engines run the same programs from the same IR and return a cvsim
:class:`~cvsim.gaussian.GaussianState`, so their results can be compared number for number.

* ``engine="cvsim"`` (default): ``cvsim.gaussian`` (ops, channels, measure). When the program
  only prepares states and then applies gates and channels, the result carries a
  :class:`~cvsim.gaussian.NoiseLedger` booking each channel under its op's ``component`` and
  ``noise_kind``.
* ``engine="qttlind"``: ``qttlind.gaussian``. Its native functions run D, R, S, BS, S2,
  Interferometer, Loss, Amplifier, AdditiveNoise, ideal homodyne and heterodyne; F is
  ``rotate(π/2)``; P, CZ, CX and Symplectic go through qttlind's ``apply_symplectic`` with the
  matrices of ``cvsim.core.conventions``; detector efficiency and electronic noise are applied as
  loss and noise before qttlind's ideal measurement (the model of CONVENTIONS.md). PhaseNoise is
  refused (qttlind has no moment-matched dephasing).

A measured mode is consumed; preparing it again puts a new state there. The final state holds the
modes not measured at the end, in register order (``Result.modes``).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core import conventions as cv
from ..errors import InputError
from ..gaussian import GaussianChannel, GaussianState, NoiseLedger, channels, measure, ops
from ..ir import Op, Program
from .base import Outcome, Result, check_runnable

__all__ = ["GaussianBackend", "op_symplectic"]

_GATES = ("D", "R", "S", "P", "F", "BS", "S2", "CZ", "CX", "Interferometer", "Symplectic")
_PREPARE = ("Vacuum", "Coherent", "Squeezed", "Thermal")
_CHANNELS = ("Loss", "Amplifier", "AdditiveNoise", "PhaseNoise")
_MEASURE = ("Homodyne", "Heterodyne")


def op_symplectic(o: Op) -> np.ndarray:
    """Symplectic matrix of a Gaussian gate (on the op's own modes), from ``core.conventions``.

    Parameters
    ----------
    o
        A gate op other than ``D``.
    """
    p = o.params
    return {
        "R": lambda: cv.rotation(p.get("theta", 0.0)),
        "S": lambda: cv.squeezing(p.get("r", 0.0), p.get("phi", 0.0)),
        "P": lambda: cv.shear(p.get("s", 0.0)),
        "F": cv.fourier,
        "BS": lambda: cv.beamsplitter(p.get("theta", 0.0), p.get("phi", 0.0)),
        "S2": lambda: cv.two_mode_squeezing(p.get("r", 0.0), p.get("phi", 0.0)),
        "CZ": lambda: cv.cz(np.array([[0.0, p.get("g", 0.0)], [p.get("g", 0.0), 0.0]])),
        "CX": lambda: cv.cx(p.get("g", 0.0)),
        "Interferometer": lambda: cv.interferometer(np.asarray(p.get("U"), dtype=complex)),
        "Symplectic": lambda: np.asarray(p.get("S"), dtype=float),
    }[o.name]()


def op_channel(o: Op) -> GaussianChannel:
    """The :class:`GaussianChannel` of a channel op (PhaseNoise excluded).

    Parameters
    ----------
    o
        Loss, Amplifier or AdditiveNoise op.
    """
    p = o.params
    if o.name == "Loss":
        return GaussianChannel.loss(p["eta"], p["nbar"])
    if o.name == "Amplifier":
        return GaussianChannel.amplifier(p["gain"], p["nbar"])
    return GaussianChannel.additive_noise(p["sigma2"])


def _prepared(o: Op) -> GaussianState:
    p = o.params
    if o.name == "Vacuum":
        return GaussianState.vacuum(1)
    if o.name == "Coherent":
        return GaussianState.coherent(complex(p["alpha"]))
    if o.name == "Squeezed":
        return GaussianState.squeezed(p["r"], p["phi"])
    return GaussianState.thermal(p["nbar"])


def _insert(
    rest: GaussianState | None, rest_modes: Sequence[int], new: GaussianState, k: int, order: list[int]
) -> tuple[GaussianState, list[int]]:
    """Put single-mode ``new`` on register mode ``k`` next to ``rest`` (on ``rest_modes``)."""
    modes = sorted([*rest_modes, k])
    state = new if rest is None else GaussianState.product(rest, new)
    current = [*rest_modes, k]
    perm = [current.index(m) for m in modes]
    return state.reduce(perm), modes


class GaussianBackend:
    """Run Gaussian programs with cvsim's engine or qttlind's.

    Parameters
    ----------
    engine
        ``"cvsim"`` or ``"qttlind"`` (needs the optional qttlind package).
    """

    def __init__(self, engine: str = "cvsim"):
        if engine not in ("cvsim", "qttlind"):
            raise InputError(f"engine must be 'cvsim' or 'qttlind', got {engine!r}")
        self.engine = engine
        self.name = f"gaussian[{engine}]"
        if engine == "qttlind":
            try:
                self._qs = importlib.import_module("qttlind.gaussian.states")
                self._qo = importlib.import_module("qttlind.gaussian.ops")
                self._qm = importlib.import_module("qttlind.gaussian.measure")
            except ImportError as err:  # pragma: no cover - qttlind is a workspace member
                raise ImportError("engine='qttlind' needs the optional qttlind package (cvsim[qttlind])") from err

    @property
    def supported(self) -> tuple[str, ...]:
        """Operation names this backend runs."""
        names = _PREPARE + _GATES + _CHANNELS + _MEASURE
        return tuple(n for n in names if not (self.engine == "qttlind" and n == "PhaseNoise"))

    def check(self, program: Program, rng: np.random.Generator | None = None) -> None:
        """Raise ``InputError`` naming the first operation this backend cannot run.

        Parameters
        ----------
        program
            Bound program.
        rng
            The generator the run will get.
        """
        check_runnable(program, self.supported, self.name, rng if rng is not None else np.random.default_rng(0))

    def run(self, program: Program, rng: np.random.Generator | None = None) -> Result:
        """Execute ``program`` from the vacuum.

        Parameters
        ----------
        program
            Bound program.
        rng
            Generator for sampled measurement outcomes.
        """
        check_runnable(program, self.supported, self.name, rng)
        M = program.num_modes
        state: GaussianState | None = GaussianState.vacuum(M)
        live = list(range(M))  # register modes held by `state`, in its order (always sorted)
        outcomes: list[Outcome] = []
        ledger_ok = all(o.kind != "measure" for o in program.ops)
        seen_other = False
        for k, o in enumerate(program.ops):
            if o.kind == "prepare":
                if seen_other:
                    ledger_ok = False
                m = o.modes[0]
                if m in live:
                    rest = [x for x in live if x != m]
                    state_rest = None if not rest else state.reduce([live.index(x) for x in rest])  # type: ignore[union-attr]
                else:
                    rest, state_rest = live, state
                state, live = _insert(state_rest, rest, _prepared(o), m, live)
                continue
            seen_other = True
            assert state is not None  # a measured mode is only reused after a preparation (IR check)
            local = [live.index(m) for m in o.modes]
            if o.kind == "measure":
                state, value, density = self._measure(state, o, local[0], rng)
                outcomes.append(Outcome(k, o.name, o.modes, value, density, o.params.get("outcome") is not None))
                live = [x for x in live if x != o.modes[0]]
                continue
            state = self._cvsim_step(state, o, local) if self.engine == "cvsim" else self._qttlind_step(state, o, local)
        ledger = self._ledger(program) if self.engine == "cvsim" and ledger_ok else None
        if state is None:
            return Result(None, (), tuple(outcomes), self.name, ledger)
        return Result(state, tuple(live), tuple(outcomes), self.name, ledger)

    # ------------------------------------------------------------------------- cvsim engine
    def _cvsim_step(self, state: GaussianState, o: Op, local: list[int]) -> GaussianState:
        if o.name == "D":
            return ops.displace(state, complex(o.params["alpha"]), local)
        if o.kind == "gate":
            return ops.apply_symplectic(state, op_symplectic(o), local)
        if o.name == "PhaseNoise":  # moment-matched: warns with ApproximationWarning
            return channels.phase_diffusion(state, o.params["sigma"], local[0])
        return op_channel(o).apply(state, local)

    def _ledger(self, program: Program) -> NoiseLedger | None:
        prep = [o for o in program.ops if o.kind == "prepare"]
        rest = [o for o in program.ops if o.kind != "prepare"]
        M = program.num_modes
        singles = {o.modes[0]: _prepared(o) for o in prep}
        initial = GaussianState.product(*(singles.get(m, GaussianState.vacuum(1)) for m in range(M)))
        led = NoiseLedger.start(initial)
        for o in rest:
            if o.name == "D":
                shift = cv.displacement_vector(complex(o.params["alpha"]))
                led = led.gate(np.eye(2), list(o.modes), o.component or "D", shift=shift)
            elif o.kind == "gate":
                led = led.gate(op_symplectic(o), list(o.modes), o.component or o.name)
            elif o.name == "PhaseNoise":
                led = led.phase_diffusion(o.params["sigma"], o.modes[0], o.component or o.name)
            else:
                led = led.channel(op_channel(o), list(o.modes), o.component or o.name, o.noise_kind or _KIND[o.name])
        return led

    def _measure(
        self, state: GaussianState, o: Op, local: int, rng: np.random.Generator | None
    ) -> tuple[GaussianState | None, float | complex, float]:
        p = o.params
        if self.engine == "cvsim":
            if o.name == "Homodyne":
                res = measure.homodyne(
                    state,
                    local,
                    p["phi"],
                    efficiency=p["efficiency"],
                    electronic=p["electronic"],
                    outcome=p["outcome"],
                    rng=None if p["outcome"] is not None else rng,
                )
            else:
                res = measure.heterodyne(
                    state,
                    local,
                    efficiency=p["efficiency"],
                    electronic=p["electronic"],
                    outcome=None if p["outcome"] is None else complex(p["outcome"]),
                    rng=None if p["outcome"] is not None else rng,
                )
            return res.state, res.outcome, res.density  # type: ignore[return-value]
        return self._qttlind_measure(state, o, local, rng)

    # ------------------------------------------------------------------------- qttlind engine
    def _q(self, s: GaussianState) -> Any:
        return self._qs.GaussianState(s.mean, s.cov)

    def _back(self, q: Any) -> GaussianState:
        return GaussianState(q.mean, q.cov, validate=False)

    def _qttlind_step(self, state: GaussianState, o: Op, local: list[int]) -> GaussianState:
        qo, q, p = self._qo, self._q(state), o.params
        if o.name == "D":
            out = qo.displace(q, complex(p["alpha"]), mode=local[0])
        elif o.name == "R":
            out = qo.rotate(q, p["theta"], mode=local[0])
        elif o.name == "F":
            out = qo.rotate(q, math.pi / 2, mode=local[0])
        elif o.name == "S":
            out = qo.squeeze(q, p["r"], p["phi"], mode=local[0])
        elif o.name == "BS":
            out = qo.beamsplitter(q, p["theta"], p["phi"], modes=local)
        elif o.name == "S2":
            out = qo.two_mode_squeeze(q, p["r"], p["phi"], modes=local)
        elif o.name == "Interferometer":
            out = qo.interferometer(q, np.asarray(p["U"]), modes=local)
        elif o.kind == "gate":  # P, CZ, CX, Symplectic: cvsim's matrix, qttlind's application
            out = qo.apply_symplectic(q, op_symplectic(o), modes=local)
        elif o.name == "Loss":
            out = qo.loss(q, p["eta"], mode=local[0], nbar=p["nbar"])
        elif o.name == "Amplifier":
            out = qo.amplifier(q, p["gain"], mode=local[0], nbar=p["nbar"])
        else:
            out = qo.thermal_noise(q, p["sigma2"], mode=local[0])
        return self._back(out)

    def _qttlind_measure(
        self, state: GaussianState, o: Op, local: int, rng: np.random.Generator | None
    ) -> tuple[GaussianState | None, float | complex, float]:
        p, qo, qm = o.params, self._qo, self._qm
        eta, vel = p["efficiency"], p["electronic"]
        q = self._q(state)
        het = o.name == "Heterodyne"
        if eta < 1 or vel > 0:  # CONVENTIONS.md: loss and noise before an ideal detector, outcome / √η
            q = qo.loss(q, eta, mode=local)
            q = qo.thermal_noise(q, vel if het else vel / 2, mode=local)
        scale = math.sqrt(eta)
        given = p["outcome"]
        if het:
            res = qm.heterodyne(
                q,
                mode=local,
                outcome=None if given is None else complex(given) * scale,
                rng=None if given is not None else rng,
            )
            value, density = complex(res.outcome) / scale, float(res.density) * eta
        else:
            res = qm.homodyne(
                q,
                mode=local,
                phi=p["phi"],
                outcome=None if given is None else float(given) * scale,
                rng=None if given is not None else rng,
            )
            value, density = float(res.outcome) / scale, float(res.density) * scale
        return (None if res.state is None else self._back(res.state)), value, density


_KIND = {"Loss": "loss", "Amplifier": "gain", "AdditiveNoise": "thermal"}
