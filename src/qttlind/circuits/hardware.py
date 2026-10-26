"""Hardware models: compile an ideal circuit into the noisy one a device runs.

A :class:`HardwareModel` is a circuit transformation; every executor then runs the compiled
circuit unchanged -- the digital twin of the device.

* insertion loss of each element kind -- ``loss(mode, eta, nbar)`` after the element, on each of
  its modes;
* finite squeezing (``max_squeezing_db``) -- ``r`` clipped to ``db / (20 log10 e)``, with a warning;
* detector efficiency ``eta_d`` -- ``loss(mode, eta_d, nbar)`` before every detector;
* dark counts ``p_d`` -- the threshold detector's ``dark_count`` (a click with probability ``p_d``);
* phase noise ``sigma`` per element -- ``dephasing(mode, sigma)`` after the element
  (``gamma t = sigma^2``);
* thermal background ``nbar`` -- the ``nbar`` of every inserted loss.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..errors import InputError, QttlindWarning
from .circuit import GAUSSIAN_KINDS, NON_GAUSSIAN_KINDS, Circuit

__all__ = ["HardwareModel", "SqueezingClippedWarning", "fit_loss", "squeezing_db"]

_ELEMENT_KINDS = (GAUSSIAN_KINDS | NON_GAUSSIAN_KINDS) - {"loss", "dephasing"}


class SqueezingClippedWarning(QttlindWarning):
    """A requested squeezing exceeds what the modelled source produces and was clipped."""


def squeezing_db(r: float) -> float:
    """Squeezing in dB: ``20 log10(e) r``."""
    return 20 * math.log10(math.e) * r


@dataclass(frozen=True)
class HardwareModel:
    """Calibrated imperfections of a CV photonic device."""

    element_loss: Mapping[str, float] = field(default_factory=dict)
    detector_efficiency: float = 1.0
    dark_count_probability: float = 0.0
    phase_noise_std: float = 0.0
    max_squeezing_db: float | None = None
    thermal_nbar: float = 0.0

    def __post_init__(self) -> None:
        loss = dict(self.element_loss)
        for kind, eta in loss.items():
            if kind not in _ELEMENT_KINDS:
                raise InputError(f"unknown element kind {kind!r}; known: {sorted(_ELEMENT_KINDS)}")
            if not 0 <= eta <= 1:
                raise InputError(f"transmissivity of {kind!r} must be in [0, 1], got {eta}")
        object.__setattr__(self, "element_loss", MappingProxyType(loss))
        if not 0 <= self.detector_efficiency <= 1:
            raise InputError(f"detector_efficiency must be in [0, 1], got {self.detector_efficiency}")
        if not 0 <= self.dark_count_probability <= 1:
            raise InputError(f"dark_count_probability must be in [0, 1], got {self.dark_count_probability}")
        if self.phase_noise_std < 0 or self.thermal_nbar < 0:
            raise InputError("phase_noise_std and thermal_nbar must be >= 0")
        if self.max_squeezing_db is not None and self.max_squeezing_db <= 0:
            raise InputError(f"max_squeezing_db must be > 0, got {self.max_squeezing_db}")

    @property
    def r_max(self) -> float | None:
        return None if self.max_squeezing_db is None else self.max_squeezing_db / (20 * math.log10(math.e))

    def compile(self, circuit: Circuit) -> Circuit:
        """The noisy circuit this device runs for the ideal ``circuit``."""
        out = Circuit(circuit.num_modes)
        nbar = self.thermal_nbar
        for instr in circuit:
            if instr.is_measurement:
                m = instr.modes[0]
                if self.detector_efficiency < 1:
                    out.loss(m, self.detector_efficiency, nbar)
                if instr.kind == "measure_threshold" and self.dark_count_probability > 0:
                    d = 1 - (1 - instr.params["dark_count"]) * (1 - self.dark_count_probability)
                    instr = instr.with_params(dark_count=d)
                out.append(instr)
                continue
            if instr.kind == "squeeze" and self.r_max is not None and abs(instr.params["r"]) > self.r_max:
                r = instr.params["r"]
                warnings.warn(
                    f"squeezing of {squeezing_db(abs(r)):.2f} dB clipped to {self.max_squeezing_db} dB",
                    SqueezingClippedWarning,
                    stacklevel=2,
                )
                instr = instr.with_params(r=math.copysign(self.r_max, r))
            out.append(instr)
            if instr.kind in ("loss", "dephasing"):
                continue
            eta = self.element_loss.get(instr.kind)
            for m in instr.modes:
                if eta is not None and eta < 1:
                    out.loss(m, eta, nbar)
                if self.phase_noise_std > 0:
                    out.dephasing(m, self.phase_noise_std)
        return out


def fit_loss(model: HardwareModel | None, r: float, measured_variance: float) -> float:
    """Element transmissivity from a measured squeezed-quadrature variance.

    The squeezed quadrature of ``S(r)|0>`` has variance ``V0 = e^{-2r}/2``; after a total
    transmissivity ``eta_tot`` it is ``V = eta_tot V0 + (1 - eta_tot)/2``. The detector efficiency
    of ``model`` (if any) is divided out: ``eta = eta_tot / eta_d``.
    """
    V0 = math.exp(-2 * r) / 2
    if r == 0:
        raise InputError("a calibration needs r != 0")
    eta_tot = (measured_variance - 0.5) / (V0 - 0.5)
    if not 0 <= eta_tot <= 1 + 1e-12:
        raise InputError(f"variance {measured_variance} is not reachable from S({r})|0> by loss (eta = {eta_tot:.4g})")
    eta_d = 1.0 if model is None else model.detector_efficiency
    if eta_d == 0:
        raise InputError("detector efficiency 0: nothing to calibrate")
    return min(1.0, eta_tot / eta_d)
