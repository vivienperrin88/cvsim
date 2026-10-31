"""Operations of the intermediate representation: what to do, not how (no physics here).

An :class:`Op` is a frozen value: a registered ``name``, its ``params`` (numbers, complex numbers,
arrays or free :class:`Parameter` placeholders), the ``modes`` it acts on, and provenance used by
the noise ledger (``component``, ``noise_kind``). The registry :data:`SPECS` fixes, for every name,
its kind, number of modes, parameter names, defaults and checks; constructing an :class:`Op`
validates against it. Backends (``cvsim.backends``) decide what each name *means*; parameter
conventions are those of CONVENTIONS.md (the gate table, channels, measurements).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import InputError, ShapeError

__all__ = ["KINDS", "SPECS", "Op", "OpSpec", "Parameter", "op"]

#: Kinds of operation, in the order a backend usually meets them.
KINDS = ("prepare", "gate", "channel", "measure")


@dataclass(frozen=True)
class Parameter:
    """A free, named parameter, bound later with :meth:`cvsim.ir.Program.bind`.

    Parameters
    ----------
    name
        Identifier (letters, digits, ``_``).
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.replace("_", "a").isalnum() or self.name[0].isdigit():
            raise InputError(f"parameter name must be an identifier, got {self.name!r}")

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"


Check = Callable[[Any], str | None]  # returns an error message or None


def _real(lo: float = -math.inf, hi: float = math.inf, lo_open: bool = False) -> Check:
    def check(v: Any) -> str | None:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return f"must be a real number, got {type(v).__name__}"
        x = float(v)
        if not math.isfinite(x) or x < lo or x > hi or (lo_open and x == lo):
            bound = f"({lo}, {hi}]" if lo_open else f"[{lo}, {hi}]"
            return f"must be finite and in {bound}, got {x}"
        return None

    return check


def _complex(v: Any) -> str | None:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, complex, np.number)):
        return f"must be a number, got {type(v).__name__}"
    return None if math.isfinite(abs(complex(v))) else "must be finite"


def _matrix(kind: str) -> Check:
    def check(v: Any) -> str | None:
        a = np.asarray(v)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0:
            return f"must be a square matrix, got shape {a.shape}"
        if not np.all(np.isfinite(a)):
            return "has non-finite entries"
        if kind == "real" and np.iscomplexobj(a) and np.abs(a.imag).max() > 0:
            return "must be real"
        return None

    return check


def _optional(check: Check) -> Check:
    return lambda v: None if v is None else check(v)


def _int(lo: int) -> Check:
    def check(v: Any) -> str | None:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < lo:
            return f"must be an integer >= {lo}, got {v!r}"
        return None

    return check


@dataclass(frozen=True)
class OpSpec:
    """Registry entry: ``kind``, number of modes (``None``: any, fixed by an array parameter),
    parameter checks and defaults, and whether the operation keeps states Gaussian."""

    kind: str
    num_modes: int | None
    checks: Mapping[str, Check]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    gaussian: bool = True
    doc: str = ""


_ANGLE = _real()
_R = _real()
_ETA = _real(0.0, 1.0)
_NBAR = _real(0.0)
_EFF = _real(0.0, 1.0, lo_open=True)

#: Every operation name the IR knows (CONVENTIONS.md gives the meaning of each parameter).
SPECS: Mapping[str, OpSpec] = {
    # preparations (reset the listed modes)
    "Vacuum": OpSpec("prepare", 1, {}, doc="|0⟩"),
    "Coherent": OpSpec("prepare", 1, {"alpha": _complex}, doc="|α⟩"),
    "Squeezed": OpSpec("prepare", 1, {"r": _R, "phi": _ANGLE}, {"phi": 0.0}, doc="S(r, φ)|0⟩"),
    "Thermal": OpSpec("prepare", 1, {"nbar": _NBAR}, doc="thermal state, n̄ photons"),
    "Fock": OpSpec("prepare", 1, {"n": _int(0)}, gaussian=False, doc="|n⟩"),
    # gates
    "D": OpSpec("gate", 1, {"alpha": _complex}, doc="displacement D(α)"),
    "R": OpSpec("gate", 1, {"theta": _ANGLE}, doc="rotation exp(iθn)"),
    "S": OpSpec("gate", 1, {"r": _R, "phi": _ANGLE}, {"phi": 0.0}, doc="squeezing S(r, φ)"),
    "P": OpSpec("gate", 1, {"s": _R}, doc="shear exp(i s x²/2)"),
    "F": OpSpec("gate", 1, {}, doc="Fourier R(π/2)"),
    "BS": OpSpec("gate", 2, {"theta": _ANGLE, "phi": _ANGLE}, {"phi": 0.0}, doc="beam splitter BS(θ, φ)"),
    "S2": OpSpec("gate", 2, {"r": _R, "phi": _ANGLE}, {"phi": 0.0}, doc="two-mode squeezing S2(r, φ)"),
    "CZ": OpSpec("gate", 2, {"g": _R}, doc="exp(i g x_a x_b)"),
    "CX": OpSpec("gate", 2, {"g": _R}, doc="exp(−i g x_a p_b)"),
    "Interferometer": OpSpec("gate", None, {"U": _matrix("complex")}, doc="a_j → Σ U_jk a_k"),
    "Symplectic": OpSpec("gate", None, {"S": _matrix("real")}, doc="general Gaussian unitary"),
    "Kerr": OpSpec("gate", 1, {"kappa": _R}, gaussian=False, doc="exp(iκ n²)"),
    # channels
    "Loss": OpSpec("channel", 1, {"eta": _ETA, "nbar": _NBAR}, {"nbar": 0.0}, doc="transmissivity η"),
    "Amplifier": OpSpec("channel", 1, {"gain": _real(1.0), "nbar": _NBAR}, {"nbar": 0.0}, doc="gain G"),
    "AdditiveNoise": OpSpec("channel", 1, {"sigma2": _real(0.0)}, doc="random displacements, variance σ²"),
    "PhaseNoise": OpSpec("channel", 1, {"sigma": _real(0.0)}, doc="random phase φ ~ N(0, σ²)"),
    # measurements (consume their modes)
    "Homodyne": OpSpec(
        "measure",
        1,
        {"phi": _ANGLE, "efficiency": _EFF, "electronic": _real(0.0), "outcome": _optional(_real())},
        {"phi": 0.0, "efficiency": 1.0, "electronic": 0.0, "outcome": None},
        doc="x_φ",
    ),
    "Heterodyne": OpSpec(
        "measure",
        1,
        {"efficiency": _EFF, "electronic": _real(0.0), "outcome": _optional(_complex)},
        {"efficiency": 1.0, "electronic": 0.0, "outcome": None},
        doc="coherent-state projection",
    ),
}

_ARRAY_MODES = {"Interferometer": lambda a: a.shape[0], "Symplectic": lambda a: a.shape[0] // 2}


def _freeze(v: Any) -> Any:
    if isinstance(v, Parameter) or v is None:
        return v
    if isinstance(v, (np.ndarray, list, tuple)):
        a = np.array(v)
        a.setflags(write=False)
        return a
    if isinstance(v, (bool, np.bool_)):
        return v
    if isinstance(v, (complex, np.complexfloating)):
        return complex(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


@dataclass(frozen=True)
class Op:
    """One operation: ``name`` from :data:`SPECS`, parameters, modes and provenance.

    Parameters
    ----------
    name
        Registered operation name.
    params
        Parameter values (missing ones take the spec's defaults); a :class:`Parameter` leaves one
        free.
    modes
        Modes it acts on, in the order of the operation.
    component
        Hardware component it models (noise ledger), e.g. ``"OPO escape"``.
    noise_kind
        Kind of imperfection for the ledger (``"loss"``, ``"electronic"``, …).
    """

    name: str
    params: Mapping[str, Any]
    modes: tuple[int, ...]
    component: str = ""
    noise_kind: str = ""

    def __post_init__(self) -> None:
        spec = SPECS.get(self.name)
        if spec is None:
            raise InputError(f"unknown operation {self.name!r}; known: {sorted(SPECS)}")
        given = dict(self.params)
        unknown = set(given) - set(spec.checks)
        if unknown:
            raise InputError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; expected {sorted(spec.checks)}")
        values = {**spec.defaults, **given}
        missing = set(spec.checks) - set(values)
        if missing:
            raise InputError(f"{self.name}: missing parameter(s) {sorted(missing)}")
        frozen = {}
        for key in spec.checks:  # registry order
            v = _freeze(values[key])
            if not isinstance(v, Parameter):
                msg = spec.checks[key](v)
                if msg:
                    raise InputError(f"{self.name}: parameter {key!r} {msg}")
            frozen[key] = v
        modes = tuple(int(m) for m in self.modes)
        if any(m < 0 for m in modes) or len(set(modes)) != len(modes) or not modes:
            raise InputError(f"{self.name}: modes must be distinct non-negative integers, got {self.modes}")
        expected = spec.num_modes
        if expected is None:
            arr = next(v for v in frozen.values() if isinstance(v, np.ndarray))
            expected = _ARRAY_MODES[self.name](arr)
            if self.name == "Symplectic" and arr.shape[0] % 2:
                raise ShapeError("Symplectic: S must be 2m x 2m")
        if len(modes) != expected:
            raise ShapeError(f"{self.name} acts on {expected} mode(s), got modes {modes}")
        object.__setattr__(self, "params", frozen)
        object.__setattr__(self, "modes", modes)

    @property
    def spec(self) -> OpSpec:
        """Registry entry of this operation."""
        return SPECS[self.name]

    @property
    def kind(self) -> str:
        """``"prepare"``, ``"gate"``, ``"channel"`` or ``"measure"``."""
        return self.spec.kind

    @property
    def free_parameters(self) -> frozenset[str]:
        """Names of the :class:`Parameter` placeholders still unbound."""
        return frozenset(v.name for v in self.params.values() if isinstance(v, Parameter))

    def bind(self, values: Mapping[str, Any]) -> Op:
        """A copy with the placeholders listed in ``values`` replaced (and validated).

        Parameters
        ----------
        values
            ``{parameter name: value}``; names this op does not use are ignored.
        """
        params = {
            k: values[v.name] if isinstance(v, Parameter) and v.name in values else v for k, v in self.params.items()
        }
        return Op(self.name, params, self.modes, self.component, self.noise_kind)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Op):
            return NotImplemented
        if (self.name, self.modes, self.component, self.noise_kind) != (
            other.name,
            other.modes,
            other.component,
            other.noise_kind,
        ):
            return False
        for k, v in self.params.items():
            w = other.params[k]
            if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
                if not (
                    isinstance(v, np.ndarray)
                    and isinstance(w, np.ndarray)
                    and v.shape == w.shape
                    and np.array_equal(v, w)
                ):
                    return False
            elif v != w:
                return False
        return True

    __hash__ = None  # type: ignore[assignment]  # contains arrays

    def __repr__(self) -> str:
        parts = []
        for k, v in self.params.items():
            parts.append(f"{k}=<{v.shape[0]}x{v.shape[1]}>" if isinstance(v, np.ndarray) else f"{k}={v!r}")
        tag = f", component={self.component!r}" if self.component else ""
        return f"{self.name}({', '.join(parts)}) on {list(self.modes)}{tag}"


def op(name: str, modes: int | Sequence[int], component: str = "", noise_kind: str = "", **params: Any) -> Op:
    """Build an :class:`Op`: ``op("S", 0, r=0.5)``, ``op("BS", (0, 1), theta=math.pi / 4)``.

    Parameters
    ----------
    name
        Registered operation name.
    modes
        Mode index or indices.
    component, noise_kind
        Provenance for the noise ledger.
    **params
        Parameter values.
    """
    ms = (int(modes),) if isinstance(modes, (int, np.integer)) else tuple(modes)
    return Op(name, params, ms, component, noise_kind)
