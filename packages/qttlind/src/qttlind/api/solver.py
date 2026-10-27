"""User-facing solvers: :class:`LindbladSolver`, :func:`evolve_closed` and :class:`Result`.

A run alternates Kraus steps (:mod:`qttlind.lindblad.kraus`) with measurements every ``every``
steps. Long runs are made safe by:

* **interruption** -- ``KeyboardInterrupt`` (Ctrl-C, or raised by the ``progress`` callback)
  returns the partial result with ``interrupted=True``;
* **checkpoints** -- every ``checkpoint_every`` steps the state and the partial result are written
  atomically to ``checkpoint`` (``.npz`` plus JSON metadata: versions, parameters, git commit);
  :meth:`LindbladSolver.resume` continues bit-for-bit;
* **cutoff checks** -- a :class:`~qttlind.errors.CutoffWarning` once per mode when
  ``<n> + 4 sqrt(<n>) >= levels`` (results past that point are truncation artefacts).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import platform
import subprocess
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy

from ..closed import TimeDependentMPO
from ..closed import evolve as closed_evolve
from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import CutoffWarning, InputError
from ..lindblad.comb import PurifiedComb
from ..lindblad.kraus import LocalTerm, StepInfo, heff_terms, kraus_step
from .operators import Operator
from .system import System, chain_mpo

__all__ = ["LindbladSolver", "Result", "evolve_closed"]

log = logging.getLogger(__name__)

SCHEMES = {"kraus1": 1, "kraus2": 2}
SCALAR_TRACKS = ("purity", "chi_mu", "chi_e", "chi_q", "trace_defect", "discarded", "trace")


def _version() -> str:
    from .. import __version__

    return __version__


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, cwd=os.path.dirname(__file__)
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict[str, Any]:
    return {
        "qttlind": _version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
    }


# ---------------------------------------------------------------------------------- Result
@dataclass
class Result:
    """Times, expectation values and diagnostics of a run.

    ``expect[name]`` and scalar ``track`` entries are arrays aligned with ``times``;
    ``track["reduced_dm:<mode>"]`` is a list of dense matrices.
    """

    times: np.ndarray
    expect: dict[str, np.ndarray]
    track: dict[str, Any]
    final_state: Any
    interrupted: bool = False
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Result({len(self.times)} times up to t={self.times[-1] if len(self.times) else 0:.4g}, "
            f"expect={sorted(self.expect)}, track={sorted(self.track)}, interrupted={self.interrupted})"
        )

    def to_pandas(self):
        """A DataFrame indexed by time with the expectation values and scalar diagnostics."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Result.to_pandas needs pandas: `uv add pandas` or `pip install qttlind[pandas]`"
            ) from exc
        cols: dict[str, np.ndarray] = {}
        for k, v in self.expect.items():
            v = np.asarray(v)
            cols[k] = v.real if np.allclose(v.imag, 0, atol=1e-12) else v
        for k, v in self.track.items():
            arr = np.asarray(v) if not k.startswith("reduced_dm:") else None
            if arr is not None and arr.ndim == 1 and len(arr) == len(self.times):
                cols[k] = arr
        return pd.DataFrame(cols, index=pd.Index(self.times, name="t"))

    # ------------------------------------------------------------- persistence
    def _arrays(self) -> dict[str, np.ndarray]:
        data: dict[str, np.ndarray] = {"times": np.asarray(self.times)}
        for k, v in self.expect.items():
            data[f"expect:{k}"] = np.asarray(v)
        for k, v in self.track.items():
            data[f"track:{k}"] = np.asarray(v)
        return data

    def save(self, path: str) -> None:
        """Write ``path`` (``.npz``): arrays, final state (comb or tensor train) and JSON metadata."""
        data = self._arrays()
        if isinstance(self.final_state, PurifiedComb):
            data.update(self.final_state.to_arrays())
        elif isinstance(self.final_state, TensorTrain):
            for i, c in enumerate(self.final_state.cores):
                data[f"tt_core_{i}"] = c
        meta = dict(self.metadata, interrupted=self.interrupted)
        data["metadata_json"] = np.array(json.dumps(meta, default=str))
        _atomic_savez(path, data)

    @classmethod
    def load(cls, path: str) -> Result:
        with np.load(path, allow_pickle=False) as z:
            files = {k: z[k] for k in z.files}
        return cls._from_arrays(files)

    @classmethod
    def _from_arrays(cls, files: Mapping[str, np.ndarray]) -> Result:
        meta = json.loads(str(files["metadata_json"]))
        expect = {k.split(":", 1)[1]: v for k, v in files.items() if k.startswith("expect:")}
        track: dict[str, Any] = {}
        for k, v in files.items():
            if k.startswith("track:"):
                name = k.split(":", 1)[1]
                track[name] = list(v) if name.startswith("reduced_dm:") else v
        final: Any = None
        if "comb_bits" in files:
            final = PurifiedComb.from_arrays(files)
        elif "tt_core_0" in files:
            n = sum(1 for k in files if k.startswith("tt_core_"))
            final = TensorTrain([files[f"tt_core_{i}"] for i in range(n)])
        interrupted = bool(meta.pop("interrupted", False))
        return cls(np.asarray(files["times"]), expect, track, final, interrupted, meta)


def _atomic_savez(path: str, data: Mapping[str, Any]) -> None:
    path = os.fspath(path)
    if not path.endswith(".npz"):
        path += ".npz"
    tmp = path + ".tmp.npz"
    np.savez(tmp, **dict(data))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------- recorder
class _Recorder:
    def __init__(self, solver: LindbladSolver, e_ops: Mapping[str, Operator], track: Sequence[str]):
        self.solver = solver
        self.e_terms = {k: solver._lower(op) for k, op in e_ops.items()}
        for k in track:
            if k not in SCALAR_TRACKS and not k.startswith("reduced_dm:") and not k.startswith("n:"):
                raise InputError(f"unknown track {k!r}: use {SCALAR_TRACKS}, 'n:<mode>' or 'reduced_dm:<mode>'")
            if ":" in k:
                solver.system.index(k.split(":", 1)[1])
        self.track_keys = list(track)
        self.times: list[float] = []
        self.expect: dict[str, list[complex]] = {k: [] for k in e_ops}
        self.track: dict[str, list[Any]] = {k: [] for k in track}
        self.warned: set[str] = set()

    def record(self, t: float, psi: PurifiedComb, info: StepInfo | None, disc: float) -> None:
        self.times.append(t)
        for k, terms in self.e_terms.items():
            self.expect[k].append(psi.expect([(term.value(t), term.ops) for term in terms]))
        bd = None
        for k in self.track_keys:
            if k in ("chi_mu", "chi_e", "chi_q"):
                bd = bd or psi.bond_dims()
                self.track[k].append(bd[k])
            elif k == "purity":
                self.track[k].append(psi.purity())
            elif k == "trace_defect":
                self.track[k].append(0.0 if info is None else info.trace_defect)
            elif k == "discarded":
                self.track[k].append(disc)
            elif k == "trace":
                self.track[k].append(psi.trace())
            elif k.startswith("reduced_dm:"):
                self.track[k].append(psi.reduced_dm(self.solver.system.index(k.split(":", 1)[1])))
            elif k.startswith("n:"):
                idx = self.solver.system.index(k.split(":", 1)[1])
                self.track[k].append(psi.expect({idx: self.solver._number(idx)}).real)
        self.check_cutoff(psi, t)

    def check_cutoff(self, psi: PurifiedComb, t: float) -> None:
        for i, m in enumerate(self.solver.system.modes):
            if m.name in self.warned:
                continue
            n = psi.expect({i: self.solver._number(i)}).real
            levels = m.info.basis.dim if m.info.basis is not None else m.dim
            if n + 4 * np.sqrt(max(n, 0.0)) >= levels:
                self.warned.add(m.name)
                warnings.warn(
                    f"mode {m.name!r}: <n> = {n:.3g} at t = {t:.4g} is close to its cutoff of {levels} levels"
                    " (<n> + 4 sqrt(<n>) >= levels); increase its bits",
                    CutoffWarning,
                    stacklevel=4,
                )

    def result(self, psi: PurifiedComb, interrupted: bool, metadata: dict) -> Result:
        expect = {k: np.array(v, dtype=np.complex128) for k, v in self.expect.items()}
        track: dict[str, Any] = {}
        for k, v in self.track.items():
            track[k] = list(v) if k.startswith("reduced_dm:") else np.array(v)
        return Result(np.array(self.times), expect, track, psi, interrupted, metadata)

    def state_arrays(self) -> dict[str, np.ndarray]:
        out = {"rec_times": np.array(self.times, dtype=float)}
        for k, v in self.expect.items():
            out[f"rec_expect:{k}"] = np.array(v, dtype=np.complex128)
        for k, v in self.track.items():
            out[f"rec_track:{k}"] = np.array(v)
        return out

    def restore(self, files: Mapping[str, np.ndarray], warned: Sequence[str]) -> None:
        self.times = list(files["rec_times"])
        for k in self.expect:
            self.expect[k] = list(files[f"rec_expect:{k}"])
        for k in self.track:
            self.track[k] = list(files[f"rec_track:{k}"])
        self.warned = set(warned)


# ---------------------------------------------------------------------------------- solver
class LindbladSolver:
    """``d rho/dt = -i[H, rho] + sum_k D[L_k] rho`` on the purified comb with Kraus steps.

    Parameters
    ----------
    system
        The :class:`~qttlind.api.system.System` the operators refer to.
    H, jump_ops
        :class:`~qttlind.api.operators.Operator` expressions (``0`` for no Hamiltonian).
    scheme
        ``"kraus2"`` (second order, default) or ``"kraus1"``.
    dt
        Time step. The scheme is explicit: ``dt * max decay rate / 2`` must stay below ~1 over the
        truncated space (see :mod:`qttlind.lindblad.kraus`).
    tol
        Discarded weight per bond and on ``mu`` at each truncation.
    chi_max
        Caps per bond kind (``{"chi_e": .., "chi_q": .., "chi_mu": ..}``) or one int for all.
    """

    def __init__(
        self,
        system: System,
        H: Operator | complex,
        jump_ops: Sequence[Operator],
        *,
        scheme: str = "kraus2",
        dt: float,
        tol: float = 1e-8,
        chi_max: Mapping[str, int] | int | None = None,
    ):
        if scheme not in SCHEMES:
            raise InputError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
        if not dt > 0:
            raise InputError(f"dt must be positive, got {dt}")
        if not 0 <= tol < 1:
            raise InputError(f"tol must be in [0, 1), got {tol}")
        self.system = system
        self.H = Operator._coerce(H) if not isinstance(H, Operator) else H
        if self.H is None:
            raise InputError(f"H must be an Operator or a number, got {type(H).__name__}")
        self.jump_ops = [Operator._coerce(L) or L for L in jump_ops]
        if not all(isinstance(L, Operator) for L in self.jump_ops):
            raise InputError("jump operators must be Operator expressions")
        for op in [self.H, *self.jump_ops]:
            for m in op.modes:
                system.index(m)
        self.scheme = scheme
        self.order = SCHEMES[scheme]
        self.dt = float(dt)
        self.tol = float(tol)
        if isinstance(chi_max, (int, np.integer)):
            chi_max = {"chi_e": int(chi_max), "chi_q": int(chi_max), "chi_mu": int(chi_max)}
        self.caps = dict(chi_max or {})
        bad = set(self.caps) - {"chi_e", "chi_q", "chi_mu"}
        if bad:
            raise InputError(f"unknown bond kinds {sorted(bad)} in chi_max; use chi_e, chi_q, chi_mu")
        self._H_terms = self._lower(self.H)
        self._L_terms = [self._lower(L) for L in self.jump_ops]
        self._Heff = heff_terms(self._H_terms, self._L_terms)
        self._numbers: dict[int, MPO] = {}

    # ------------------------------------------------------------- helpers
    def _lower(self, op: Operator) -> list[LocalTerm]:
        return op.to_local_terms(self.system)

    def _number(self, i: int) -> MPO:
        if i not in self._numbers:
            from ..bosons.operators import number

            self._numbers[i] = number(self.system.bits[i])
        return self._numbers[i]

    def parameters(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "dt": self.dt,
            "tol": self.tol,
            "chi_max": self.caps,
            "system": self.system.describe(),
            "H": repr(self.H),
            "jump_ops": [repr(L) for L in self.jump_ops],
        }

    def __repr__(self) -> str:
        return f"LindbladSolver({self.system!r}, scheme={self.scheme!r}, dt={self.dt}, tol={self.tol})"

    def step(self, psi: PurifiedComb, t: float) -> tuple[PurifiedComb, StepInfo]:
        """One Kraus step from ``t`` (exposed for custom loops)."""
        return kraus_step(psi, self._Heff, self._L_terms, t, self.dt, order=self.order, tol=self.tol, caps=self.caps)

    # ------------------------------------------------------------- run
    def run(
        self,
        psi0: PurifiedComb,
        t_final: float,
        *,
        e_ops: Mapping[str, Operator] | None = None,
        every: int = 1,
        track: Sequence[str] = (),
        progress: Callable[[int, float, StepInfo], None] | None = None,
        checkpoint: str | None = None,
        checkpoint_every: int = 1000,
        log_every: int | None = None,
    ) -> Result:
        """Integrate from ``t = 0`` to ``t_final``; measure every ``every`` steps (and at the end)."""
        if not isinstance(psi0, PurifiedComb):
            raise InputError(f"psi0 must be a PurifiedComb (System.product_state, ...), got {type(psi0).__name__}")
        if psi0.bits != self.system.bits:
            raise InputError(f"psi0 has bits {psi0.bits} but the system has {self.system.bits}")
        if t_final < 0 or every < 1 or checkpoint_every < 1:
            raise InputError("need t_final >= 0, every >= 1 and checkpoint_every >= 1")
        nsteps = round(t_final / self.dt)
        if abs(nsteps * self.dt - t_final) > 1e-9 * max(1.0, t_final):
            log.warning(
                "t_final = %g is not a multiple of dt = %g: running %d steps to t = %g",
                t_final,
                self.dt,
                nsteps,
                nsteps * self.dt,
            )
        spec = {
            "t_final": float(t_final),
            "every": int(every),
            "track": list(track),
            "checkpoint_every": int(checkpoint_every),
            "nsteps": nsteps,
        }
        e_ops = dict(e_ops or {})
        for op in e_ops.values():
            if not isinstance(op, Operator):
                raise InputError("e_ops values must be Operator expressions")
        rec = _Recorder(self, e_ops, track)
        psi = psi0.normalize()
        rec.record(0.0, psi, None, 0.0)
        return self._loop(psi, 0, rec, spec, e_ops, progress, checkpoint, log_every, 0.0)

    def _loop(self, psi, start, rec, spec, e_ops, progress, checkpoint, log_every, disc_total) -> Result:
        nsteps, every, cp_every = spec["nsteps"], spec["every"], spec["checkpoint_every"]
        log_every = log_every or max(1, nsteps // 10)
        interrupted = False
        t0 = time.perf_counter()
        done = start
        try:
            for i in range(start, nsteps):
                psi, info = self.step(psi, i * self.dt)
                disc_total += info.discarded_mu + info.discarded_bonds
                done = i + 1
                if done % every == 0 or done == nsteps:
                    rec.record(done * self.dt, psi, info, disc_total)
                if progress is not None:
                    progress(done, done * self.dt, info)
                if done % log_every == 0:
                    log.info(
                        "step %d/%d  t=%.4g  %s  (%.1fs)",
                        done,
                        nsteps,
                        done * self.dt,
                        psi.bond_dims(),
                        time.perf_counter() - t0,
                    )
                if checkpoint is not None and (done % cp_every == 0 or done == nsteps):
                    self._write_checkpoint(checkpoint, psi, done, rec, spec, e_ops, disc_total)
        except KeyboardInterrupt:
            interrupted = True
            log.warning("interrupted at step %d/%d: returning the partial result", done, nsteps)
        meta = {
            "parameters": self.parameters(),
            "run": spec,
            "environment": _environment(),
            "steps_done": done,
            "wall_time": time.perf_counter() - t0,
            "discarded_total": disc_total,
            "cutoff_warnings": sorted(rec.warned),
        }
        return rec.result(psi, interrupted, meta)

    # ------------------------------------------------------------- checkpoints
    def _write_checkpoint(self, path, psi, done, rec, spec, e_ops, disc_total) -> None:
        data = dict(psi.to_arrays())
        data.update(rec.state_arrays())
        try:
            blob = pickle.dumps({"solver": self, "e_ops": e_ops})
            data["pickled"] = np.frombuffer(blob, dtype=np.uint8)
            picklable = True
        except (pickle.PicklingError, AttributeError, TypeError):
            picklable = False
        meta = {
            "step": done,
            "spec": spec,
            "e_ops": sorted(e_ops),
            "warned": sorted(rec.warned),
            "disc_total": disc_total,
            "picklable": picklable,
            "parameters": self.parameters(),
            "environment": _environment(),
        }
        data["checkpoint_json"] = np.array(json.dumps(meta, default=str))
        _atomic_savez(path, data)
        log.debug("checkpoint written at step %d to %s", done, path)

    @classmethod
    def resume(
        cls,
        path: str,
        solver: LindbladSolver | None = None,
        *,
        e_ops: Mapping[str, Operator] | None = None,
        progress: Callable[[int, float, StepInfo], None] | None = None,
        checkpoint: str | None = None,
    ) -> tuple[LindbladSolver, Result]:
        """Continue the run saved in ``path`` to its ``t_final``; returns the solver and the full result.

        The solver and ``e_ops`` are restored from the checkpoint when they could be pickled;
        otherwise (lambda coefficients, typically) pass the same ``solver`` and ``e_ops`` again.
        """
        path = os.fspath(path)
        if not path.endswith(".npz") and not os.path.exists(path):
            path += ".npz"
        with np.load(path, allow_pickle=False) as z:
            files = {k: z[k] for k in z.files}
        meta = json.loads(str(files["checkpoint_json"]))
        if solver is None:
            if "pickled" not in files:
                raise InputError(
                    "this checkpoint has no pickled solver (lambda coefficients?): pass solver= and e_ops="
                )
            obj = pickle.loads(files["pickled"].tobytes())
            solver = obj["solver"]
            e_ops = obj["e_ops"] if e_ops is None else e_ops
        e_ops = dict(e_ops or {})
        if sorted(e_ops) != meta["e_ops"]:
            raise InputError(f"the checkpoint measured e_ops {meta['e_ops']}, got {sorted(e_ops)}")
        spec = meta["spec"]
        rec = _Recorder(solver, e_ops, spec["track"])
        rec.restore(files, meta["warned"])
        psi = PurifiedComb.from_arrays(files)
        res = solver._loop(psi, meta["step"], rec, spec, e_ops, progress, checkpoint or path, None, meta["disc_total"])
        return solver, res


# ---------------------------------------------------------------------------------- closed systems
def _closed_hamiltonian(system: System, H: Operator) -> TimeDependentMPO:
    terms: list[tuple[Any, MPO]] = []
    for term in H.to_local_terms(system):
        terms.append((term.coeff, chain_mpo(dict(term.ops), system.bits)))
    if not terms:
        raise InputError("the Hamiltonian is zero")
    return TimeDependentMPO(terms)


def _chain_operator(system: System, op: Operator) -> MPO:
    if op.is_time_dependent:
        raise InputError("e_ops of evolve_closed must be time-independent")
    acc: MPO | None = None
    for term in op.to_local_terms(system):
        m = chain_mpo(dict(term.ops), system.bits) * term.value(0.0)
        acc = m if acc is None else (acc + m).compress(1e-24)[0]
    if acc is None:
        acc = MPO.identity([2] * sum(system.bits)) * 0.0
    return acc


def evolve_closed(
    state: PurifiedComb | TensorTrain,
    H: Operator,
    *,
    t_final: float,
    dt: float,
    method: str = "magnus_tdvp",
    tol: float = 1e-8,
    every: int = 1,
    e_ops: Mapping[str, Operator] | None = None,
    chi_max: int | None = None,
    system: System | None = None,
) -> Result:
    """Schrodinger evolution of a pure state with the closed integrators (RK4, CN, TDVP, Magnus-TDVP).

    ``state`` is a pure comb from :meth:`System.product_state` (or a tensor train over all bits);
    ``final_state`` is a :class:`~qttlind.core.tt.TensorTrain` over all bits (modes in order, MSB
    first) and ``track["chi_q"]`` its largest bond dimension.
    """
    if system is None:
        system = getattr(state, "system", None)
    if system is None:
        raise InputError("pass system=: the state does not know which System it belongs to")
    psi = system.comb_to_chain(state)
    Ht = _closed_hamiltonian(system, H)
    ops = {k: _chain_operator(system, v) for k, v in (e_ops or {}).items()}
    t0 = time.perf_counter()
    r = closed_evolve(Ht, psi, t_final, dt, method=method, tol=tol, chi_max=chi_max, e_ops=ops, every=every)
    meta = {
        "parameters": {
            "method": method,
            "dt": dt,
            "tol": tol,
            "chi_max": chi_max,
            "H": repr(H),
            "system": system.describe(),
        },
        "environment": _environment(),
        "wall_time": time.perf_counter() - t0,
        **r.metadata,
    }
    return Result(r.times, dict(r.expect), dict(r.track), r.final_state, False, meta)
