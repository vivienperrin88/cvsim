"""A CV circuit is data: a list of instructions, independent of how it is executed."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import InputError

__all__ = ["GAUSSIAN_KINDS", "MEASUREMENT_KINDS", "NON_GAUSSIAN_KINDS", "Circuit", "Instruction"]

GAUSSIAN_KINDS = frozenset(
    {"squeeze", "displace", "rotate", "beamsplitter", "two_mode_squeeze", "loss", "measure_homodyne"}
)
NON_GAUSSIAN_KINDS = frozenset({"kerr", "cubic_phase", "dephasing", "measure_fock", "measure_threshold"})
MEASUREMENT_KINDS = frozenset({"measure_homodyne", "measure_fock", "measure_threshold"})


@dataclass(frozen=True)
class Instruction:
    """One operation: ``kind`` on ``modes`` with ``params``; ``is_gaussian`` declares the capability needed."""

    kind: str
    modes: tuple[int, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    is_gaussian: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def is_measurement(self) -> bool:
        return self.kind in MEASUREMENT_KINDS

    def with_params(self, **changes: Any) -> Instruction:
        return Instruction(self.kind, self.modes, {**self.params, **changes}, self.is_gaussian)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.kind}({', '.join(map(str, self.modes))}{', ' if args else ''}{args})"


class Circuit:
    """An ordered list of :class:`Instruction` on ``num_modes`` modes; builder methods return ``self``."""

    def __init__(self, num_modes: int):
        if not isinstance(num_modes, (int, np.integer)) or num_modes < 1:
            raise InputError(f"a circuit needs >= 1 mode, got {num_modes!r}")
        self.num_modes = int(num_modes)
        self.instructions: list[Instruction] = []

    # ------------------------------------------------------------- generic
    def append(self, instr: Instruction) -> Circuit:
        for m in instr.modes:
            if not 0 <= m < self.num_modes:
                raise InputError(f"{instr.kind}: mode {m} out of range 0..{self.num_modes - 1}")
        if len(set(instr.modes)) != len(instr.modes):
            raise InputError(f"{instr.kind}: repeated modes {instr.modes}")
        self.instructions.append(instr)
        return self

    def _add(self, kind: str, modes: tuple[int, ...], gaussian: bool, **params: Any) -> Circuit:
        return self.append(Instruction(kind, tuple(int(m) for m in modes), params, gaussian))

    def copy(self) -> Circuit:
        c = Circuit(self.num_modes)
        c.instructions = list(self.instructions)
        return c

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        body = "\n".join(f"  {i!r}" for i in self.instructions)
        return f"Circuit({self.num_modes} modes)\n{body}" if body else f"Circuit({self.num_modes} modes)"

    @property
    def is_gaussian(self) -> bool:
        return all(i.is_gaussian for i in self.instructions)

    # ------------------------------------------------------------- Gaussian
    def squeeze(self, mode: int, r: float, phi: float = 0.0) -> Circuit:
        """``S(r, phi) = exp((conj(z) a^2 - z a^dag^2)/2)``, ``z = r e^{i phi}``."""
        return self._add("squeeze", (mode,), True, r=float(r), phi=float(phi))

    def displace(self, mode: int, alpha: complex) -> Circuit:
        """``D(alpha) = exp(alpha a^dag - conj(alpha) a)``."""
        return self._add("displace", (mode,), True, alpha=complex(alpha))

    def rotate(self, mode: int, theta: float) -> Circuit:
        """``R(theta) = exp(i theta n)``."""
        return self._add("rotate", (mode,), True, theta=float(theta))

    def beamsplitter(self, a: int, b: int, theta: float, phi: float = 0.0) -> Circuit:
        """``BS(theta, phi) = exp(theta (e^{i phi} a b^dag - e^{-i phi} a^dag b))``; transmissivity ``cos^2 theta``."""
        return self._add("beamsplitter", (a, b), True, theta=float(theta), phi=float(phi))

    def two_mode_squeeze(self, a: int, b: int, r: float, phi: float = 0.0) -> Circuit:
        """``S2(r, phi) = exp(conj(z) a b - z a^dag b^dag)``."""
        return self._add("two_mode_squeeze", (a, b), True, r=float(r), phi=float(phi))

    def loss(self, mode: int, eta: float, nbar: float = 0.0) -> Circuit:
        """Beam splitter of transmissivity ``eta`` with a thermal environment of ``nbar`` photons."""
        if not 0 <= eta <= 1 or nbar < 0:
            raise InputError(f"loss needs 0 <= eta <= 1 and nbar >= 0, got eta={eta}, nbar={nbar}")
        return self._add("loss", (mode,), True, eta=float(eta), nbar=float(nbar))

    def measure_homodyne(self, mode: int, phi: float = 0.0, postselect: float | None = None) -> Circuit:
        """Measure ``x_phi = cos(phi) x + sin(phi) p``; the mode is then reset to vacuum."""
        return self._add(
            "measure_homodyne",
            (mode,),
            True,
            phi=float(phi),
            postselect=None if postselect is None else float(postselect),
        )

    # ------------------------------------------------------------- non-Gaussian
    def kerr(self, mode: int, kappa: float) -> Circuit:
        """``K(kappa) = exp(i kappa n^2)``."""
        return self._add("kerr", (mode,), False, kappa=float(kappa))

    def cubic_phase(self, mode: int, gamma: float) -> Circuit:
        """``V(gamma) = exp(i gamma x^3 / 3)``."""
        return self._add("cubic_phase", (mode,), False, gamma=float(gamma))

    def dephasing(self, mode: int, sigma: float) -> Circuit:
        """Random rotation of standard deviation ``sigma`` (phase noise, ``gamma t = sigma^2``)."""
        if sigma < 0:
            raise InputError(f"sigma must be >= 0, got {sigma}")
        return self._add("dephasing", (mode,), False, sigma=float(sigma))

    def measure_fock(self, mode: int, postselect: int | None = None) -> Circuit:
        """Photon-number-resolving detection; the mode is then reset to vacuum."""
        return self._add("measure_fock", (mode,), False, postselect=None if postselect is None else int(postselect))

    def measure_threshold(self, mode: int, postselect: int | None = None, dark_count: float = 0.0) -> Circuit:
        """Click (1) / no-click (0) detection with dark-count probability ``dark_count``; mode reset to vacuum."""
        if postselect not in (None, 0, 1):
            raise InputError(f"threshold outcomes are 0 or 1, got {postselect}")
        if not 0 <= dark_count <= 1:
            raise InputError(f"dark_count must be in [0, 1], got {dark_count}")
        return self._add("measure_threshold", (mode,), False, postselect=postselect, dark_count=float(dark_count))
