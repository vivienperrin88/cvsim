"""Immutable programs: a mode register and a sequence of :class:`~cvsim.ir.ops.Op`.

A :class:`Program` never changes: ``program >> op`` (or ``>> other_program``) returns a new one,
so the same program can be reused, bound to several parameter values, or run on several backends.
Composition reads left to right in time, as in MrMustard (``a >> b``: ``a`` first), without
MrMustard's in-place mode assignment. Validation at construction: modes in range, and no gate,
channel or measurement on a mode that has been measured and not prepared again.

:meth:`Program.to_json` / :meth:`Program.from_json` give a versioned, self-describing text form
(complex numbers and arrays as ``{"re": …, "im": …}``, free parameters as ``{"param": name}``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import InputError
from .ops import SPECS, Op, Parameter

__all__ = ["SCHEMA", "Mode", "Program"]

#: Version of the JSON form written by :meth:`Program.to_json`.
SCHEMA = "cvsim.ir/1"


@dataclass(frozen=True)
class Mode:
    """A named mode of the register.

    Parameters
    ----------
    name
        Label (unique in a program), e.g. ``"signal"`` or ``"supermode 0"``.
    description
        Free text; the spectral/temporal shape arrives with mode bases (S12).
    """

    name: str
    description: str = ""


def _modes(spec: int | Sequence[str | Mode]) -> tuple[Mode, ...]:
    if isinstance(spec, (int, np.integer)):
        if int(spec) < 1:
            raise InputError(f"a program needs at least one mode, got {spec}")
        return tuple(Mode(f"m{k}") for k in range(int(spec)))
    modes = tuple(m if isinstance(m, Mode) else Mode(str(m)) for m in spec)
    if not modes:
        raise InputError("a program needs at least one mode")
    names = [m.name for m in modes]
    if len(set(names)) != len(names):
        raise InputError(f"mode names must be unique, got {names}")
    return modes


class Program:
    """A mode register and a time-ordered tuple of operations.

    Parameters
    ----------
    modes
        Number of modes, or their names / :class:`Mode` objects.
    ops
        Operations in time order.
    name
        Optional label.
    """

    __slots__ = ("_modes", "_name", "_ops")

    def __init__(self, modes: int | Sequence[str | Mode], ops: Iterable[Op] = (), name: str = ""):
        self._modes = _modes(modes)
        self._ops = tuple(ops)
        self._name = str(name)
        self._validate()

    def _validate(self) -> None:
        M = len(self._modes)
        consumed: set[int] = set()
        for k, o in enumerate(self._ops):
            if not isinstance(o, Op):
                raise InputError(f"operation {k} is not an Op: {o!r}")
            bad = [m for m in o.modes if m >= M]
            if bad:
                raise InputError(f"operation {k} ({o.name}) acts on mode(s) {bad}, the program has {M}")
            if o.kind == "prepare":
                consumed.difference_update(o.modes)
                continue
            used = consumed.intersection(o.modes)
            if used:
                raise InputError(
                    f"operation {k} ({o.name}) acts on measured mode(s) {sorted(used)}; prepare them again first"
                )
            if o.kind == "measure":
                consumed.update(o.modes)

    # ------------------------------------------------------------------------- access
    @property
    def modes(self) -> tuple[Mode, ...]:
        """The register."""
        return self._modes

    @property
    def num_modes(self) -> int:
        """Number of modes."""
        return len(self._modes)

    @property
    def ops(self) -> tuple[Op, ...]:
        """The operations, in time order."""
        return self._ops

    @property
    def name(self) -> str:
        """Label."""
        return self._name

    def mode_index(self, name: str) -> int:
        """Index of the mode called ``name``.

        Parameters
        ----------
        name
            Mode name.
        """
        for k, m in enumerate(self._modes):
            if m.name == name:
                return k
        raise InputError(f"no mode named {name!r}; modes: {[m.name for m in self._modes]}")

    @property
    def parameters(self) -> frozenset[str]:
        """Names of the free parameters."""
        return frozenset().union(*(o.free_parameters for o in self._ops))

    @property
    def is_bound(self) -> bool:
        """No free parameter left."""
        return not self.parameters

    @property
    def is_gaussian(self) -> bool:
        """Every operation keeps Gaussian states Gaussian."""
        return all(o.spec.gaussian for o in self._ops)

    @property
    def measured_modes(self) -> tuple[int, ...]:
        """Modes measured at the end of the program (not prepared again afterwards)."""
        consumed: list[int] = []
        for o in self._ops:
            if o.kind == "prepare":
                consumed = [m for m in consumed if m not in o.modes]
            elif o.kind == "measure":
                consumed.extend(o.modes)
        return tuple(consumed)

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self) -> Iterator[Op]:
        return iter(self._ops)

    def __repr__(self) -> str:
        label = f" {self._name!r}" if self._name else ""
        return f"Program{label}(modes={self.num_modes}, ops={len(self._ops)}, free={sorted(self.parameters)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Program):
            return NotImplemented
        return self._modes == other._modes and self._name == other._name and self._ops == other._ops

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------------- building
    def __rshift__(self, other: Op | Program | Iterable[Op]) -> Program:
        """``self`` then ``other`` (an op, ops, or a program on the same register)."""
        if isinstance(other, Op):
            new: tuple[Op, ...] = (other,)
        elif isinstance(other, Program):
            if other.num_modes != self.num_modes:
                raise InputError(f"programs have {self.num_modes} and {other.num_modes} modes")
            new = other.ops
        else:
            new = tuple(other)
        return Program(self._modes, self._ops + new, self._name)

    def bind(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> Program:
        """Replace free parameters by values (validated against each operation's spec).

        Parameters
        ----------
        values
            ``{name: value}``.
        **kwargs
            The same, as keywords.
        """
        vals = {**(values or {}), **kwargs}
        unknown = set(vals) - self.parameters
        if unknown:
            raise InputError(f"no free parameter(s) {sorted(unknown)}; free: {sorted(self.parameters)}")
        return Program(self._modes, (o.bind(vals) for o in self._ops), self._name)

    # ------------------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dictionary (see :meth:`to_json`)."""
        return {
            "schema": SCHEMA,
            "name": self._name,
            "modes": [{"name": m.name, "description": m.description} for m in self._modes],
            "ops": [
                {
                    "name": o.name,
                    "modes": list(o.modes),
                    "params": {k: _encode(v) for k, v in o.params.items()},
                    **({"component": o.component} if o.component else {}),
                    **({"noise_kind": o.noise_kind} if o.noise_kind else {}),
                }
                for o in self._ops
            ],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Text form, versioned by :data:`SCHEMA`.

        Parameters
        ----------
        indent
            JSON indentation (``None`` for one line).
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Program:
        """Inverse of :meth:`to_dict`.

        Parameters
        ----------
        data
            Dictionary in the form written by :meth:`to_dict`.
        """
        if data.get("schema") != SCHEMA:
            raise InputError(f"unsupported program schema {data.get('schema')!r}; expected {SCHEMA!r}")
        modes = [Mode(m["name"], m.get("description", "")) for m in data["modes"]]
        ops = []
        for k, o in enumerate(data["ops"]):
            if o.get("name") not in SPECS:
                raise InputError(f"operation {k}: unknown name {o.get('name')!r}")
            params = {p: _decode(v) for p, v in o.get("params", {}).items()}
            ops.append(Op(o["name"], params, tuple(o["modes"]), o.get("component", ""), o.get("noise_kind", "")))
        return cls(modes, ops, data.get("name", ""))

    @classmethod
    def from_json(cls, text: str) -> Program:
        """Inverse of :meth:`to_json`.

        Parameters
        ----------
        text
            JSON text.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            raise InputError(f"not valid JSON: {err}") from err
        return cls.from_dict(data)


def _encode(v: Any) -> Any:
    if isinstance(v, Parameter):
        return {"param": v.name}
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            return {"re": v.real.tolist(), "im": v.imag.tolist()}
        return {"re": v.tolist()}
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    return v


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v) == {"param"}:
            return Parameter(v["param"])
        if "re" in v:
            re = v["re"]
            if isinstance(re, list):
                a = np.asarray(re, dtype=float)
                return a + 1j * np.asarray(v["im"], dtype=float) if "im" in v else a
            return complex(re, v.get("im", 0.0))
        raise InputError(f"cannot decode parameter value {v!r}")
    return v
