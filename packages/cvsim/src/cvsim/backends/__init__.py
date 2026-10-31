"""Backends (layer 4): run :class:`~cvsim.ir.Program` objects.

* :class:`GaussianBackend` — ``engine="cvsim"`` (default) or ``"qttlind"``: the backend switch of
  decision D2, same programs and same :class:`Result` on either engine;
* :class:`FockBackend` — dense density matrices with a photon cutoff: every operation, Gaussian or
  not;
* :func:`select_backend` / :func:`run` — the dispatcher: Gaussian programs go to the Gaussian
  backend, the others to the Fock backend.

This is the only subpackage that may import qttlind (``cvsim._layers.QTTLIND_ALLOWED``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..errors import InputError
from ..ir import Program
from .base import Backend, Outcome, Result, check_runnable
from .fock import FockBackend, FockState
from .gaussian import GaussianBackend

__all__ = [
    "Backend",
    "FockBackend",
    "FockState",
    "GaussianBackend",
    "Outcome",
    "Result",
    "check_runnable",
    "run",
    "select_backend",
]


def select_backend(program: Program, cutoff: int = 20, engine: str = "cvsim") -> Backend:
    """The backend :func:`run` uses: Gaussian for Gaussian programs, Fock otherwise.

    Parameters
    ----------
    program
        The program.
    cutoff
        Fock cutoff if the Fock backend is chosen.
    engine
        Gaussian engine (``"cvsim"`` or ``"qttlind"``).
    """
    if not isinstance(program, Program):
        raise InputError(f"expected a cvsim.ir.Program, got {type(program).__name__}")
    return GaussianBackend(engine) if program.is_gaussian else FockBackend(cutoff)


def run(
    program: Program, backend: Backend | str | None = None, rng: np.random.Generator | None = None, **options: object
) -> Result:
    """Run ``program`` on ``backend`` (an instance, ``"gaussian"``, ``"qttlind"``, ``"fock"``, or
    ``None`` to choose automatically).

    Parameters
    ----------
    program
        Bound program.
    backend
        Backend or its short name.
    rng
        Generator for sampled outcomes.
    **options
        ``cutoff`` for the Fock backend.
    """
    cutoff = int(options.pop("cutoff", 20))  # type: ignore[call-overload]
    if options:
        raise InputError(f"unknown option(s) {sorted(options)}")
    if backend is None:
        chosen: Backend = select_backend(program, cutoff)
    elif isinstance(backend, str):
        names: dict[str, Callable[[], Backend]] = {
            "gaussian": lambda: GaussianBackend("cvsim"),
            "qttlind": lambda: GaussianBackend("qttlind"),
            "fock": lambda: FockBackend(cutoff),
        }
        if backend not in names:
            raise InputError(f"unknown backend {backend!r}; use one of {sorted(names)} or a Backend instance")
        chosen = names[backend]()
    else:
        chosen = backend
    return chosen.run(program, rng=rng)
