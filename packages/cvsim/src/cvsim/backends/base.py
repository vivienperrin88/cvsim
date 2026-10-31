"""What every backend promises: :class:`Backend` (``check`` and ``run``) and :class:`Result`.

A backend executes a bound :class:`~cvsim.ir.Program` from the vacuum. Measurements whose outcome
is given in the program are post-selected; the others are sampled from ``rng`` (required then).
The result holds the state of the modes that are not measured at the end (``modes`` lists them,
in register order), the outcomes and their probability densities, and, when the backend keeps
one, the noise ledger. A backend refuses a program it cannot run *before* running it, with an
``InputError`` naming the first unsupported operation and its index.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..errors import InputError
from ..ir import Op, Program

__all__ = ["Backend", "Outcome", "Result", "check_runnable"]


@dataclass(frozen=True)
class Outcome:
    """One measurement: the index of its op in the program, its modes, value and density."""

    op_index: int
    name: str
    modes: tuple[int, ...]
    value: float | complex
    density: float
    postselected: bool


@dataclass(frozen=True)
class Result:
    """What a run produces.

    ``state`` is a :class:`~cvsim.gaussian.GaussianState` (Gaussian backends) or a
    :class:`~cvsim.backends.fock.FockState`, on the program modes listed in ``modes``.
    ``log_density`` is the sum of the log densities of all outcomes (``0`` without measurements).
    """

    state: Any
    modes: tuple[int, ...]
    outcomes: tuple[Outcome, ...] = ()
    backend: str = ""
    ledger: Any = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def log_density(self) -> float:
        """``Σ log p`` over the outcomes."""
        return float(sum(math.log(o.density) if o.density > 0 else -math.inf for o in self.outcomes))

    def outcome_values(self) -> list[float | complex]:
        """The measured values, in program order."""
        return [o.value for o in self.outcomes]


@runtime_checkable
class Backend(Protocol):
    """A simulator for :class:`~cvsim.ir.Program`."""

    name: str

    def check(self, program: Program) -> None:
        """Raise ``InputError`` naming the first operation this backend cannot run."""
        ...

    def run(self, program: Program, rng: np.random.Generator | None = None) -> Result:
        """Execute ``program`` from the vacuum."""
        ...


def check_runnable(program: Program, supported: Sequence[str], backend: str, rng: np.random.Generator | None) -> None:
    """Common checks: a :class:`Program`, bound, only ``supported`` operations, ``rng`` when sampling.

    Parameters
    ----------
    program
        The program.
    supported
        Operation names the backend runs.
    backend
        Its name, for messages.
    rng
        The generator the run will get (``None`` if none).
    """
    if not isinstance(program, Program):
        raise InputError(f"expected a cvsim.ir.Program, got {type(program).__name__}")
    if not program.is_bound:
        raise InputError(f"program has free parameters {sorted(program.parameters)}: bind them first")
    for k, o in enumerate(program.ops):
        if o.name not in supported:
            raise InputError(f"{backend} cannot run operation {k} ({o.name}); it supports {sorted(supported)}")
    if rng is None and any(_sampled(o) for o in program.ops):
        raise InputError("program samples a measurement outcome: pass rng=np.random.default_rng(seed)")


def _sampled(o: Op) -> bool:
    return o.kind == "measure" and o.params.get("outcome") is None
