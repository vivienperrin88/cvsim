"""The closed-system driver: integrate ``i dpsi/dt = H(t) psi`` with any stepper.

A stepper follows the protocol ``step(H, psi, t, h, *, tol, chi_max) -> TensorTrain``
(:class:`Stepper`); :func:`make_stepper` returns the built-in ones by name.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..core.mpo import MPO, apply_mpo
from ..core.tt import TensorTrain
from ..errors import InputError
from .hamiltonian import TimeDependentMPO
from .integrators import crank_nicolson_step, rk4_step
from .tdvp import MagnusTDVP, tdvp_frozen_step

__all__ = ["METHODS", "ClosedResult", "Stepper", "evolve", "make_stepper"]

log = logging.getLogger(__name__)

METHODS = ("rk4", "crank_nicolson", "tdvp", "magnus_tdvp")


class Stepper(Protocol):
    def __call__(
        self, H: TimeDependentMPO, psi: TensorTrain, t: float, h: float, *, tol: float, chi_max: int | None
    ) -> TensorTrain: ...


@dataclass
class ClosedResult:
    times: np.ndarray
    expect: dict[str, np.ndarray]
    track: dict[str, np.ndarray]
    final_state: TensorTrain
    metadata: dict = field(default_factory=dict)


def make_stepper(method: str, H: TimeDependentMPO) -> Stepper:
    """The built-in stepper ``method`` (one of :data:`METHODS`) for the Hamiltonian ``H``."""
    if method == "rk4":
        return rk4_step
    if method == "crank_nicolson":
        return crank_nicolson_step
    if method == "tdvp":
        return tdvp_frozen_step
    if method == "magnus_tdvp":
        return MagnusTDVP(H).step
    raise InputError(f"unknown method {method!r}; choose one of {METHODS}")


def _expect(op: MPO, psi: TensorTrain) -> complex:
    out, _ = apply_mpo(op, psi, method="zipup", tol=0.0)
    return psi.inner(out) / psi.inner(psi)


def evolve(
    H: TimeDependentMPO | MPO,
    psi0: TensorTrain,
    t_final: float,
    h: float,
    *,
    method: str = "magnus_tdvp",
    step: Stepper | None = None,
    tol: float = 1e-16,
    chi_max: int | None = None,
    e_ops: Mapping[str, MPO] | None = None,
    every: int = 1,
) -> ClosedResult:
    """Integrate from ``t = 0`` to ``t_final`` in steps ``h``; record every ``every`` steps.

    ``tol`` is the discarded weight per bond after every operation (``1e-16`` = relative error 1e-8).
    ``track`` always contains ``chi_q`` (largest bond) and ``norm``.
    """
    if not t_final >= 0 or not h > 0:
        raise InputError(f"need t_final >= 0 and h > 0, got {t_final}, {h}")
    if every < 1:
        raise InputError(f"every must be >= 1, got {every}")
    Ht = H if isinstance(H, TimeDependentMPO) else TimeDependentMPO.constant(H)
    stepper = step if step is not None else make_stepper(method, Ht)
    e_ops = dict(e_ops or {})
    nsteps = round(t_final / h)
    times, chi, norms = [], [], []
    ex: dict[str, list[complex]] = {k: [] for k in e_ops}
    psi = psi0

    def record(t: float) -> None:
        times.append(t)
        chi.append(psi.max_rank)
        norms.append(psi.norm())
        for k, op in e_ops.items():
            ex[k].append(_expect(op, psi))

    record(0.0)
    for i in range(nsteps):
        psi = stepper(Ht, psi, i * h, h, tol=tol, chi_max=chi_max)
        if (i + 1) % every == 0 or i + 1 == nsteps:
            record((i + 1) * h)
            log.debug("step %d/%d: chi_q = %d", i + 1, nsteps, psi.max_rank)
    return ClosedResult(
        np.array(times),
        {k: np.array(v) for k, v in ex.items()},
        {"chi_q": np.array(chi), "norm": np.array(norms)},
        psi,
        {"method": method if step is None else getattr(step, "__name__", "custom"), "h": h, "tol": tol},
    )
