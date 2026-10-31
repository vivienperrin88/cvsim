"""Physical units of the hardware layer: SI floats, and the conversions laboratories speak in.

cvsim's hardware models take and return **SI values** (seconds, metres, hertz, angular
frequencies in rad/s); this module names the scale factors and converts the quantities that data
sheets and papers quote in other forms (CONVENTIONS.md, "Units and spectra"):

* wavelengths ↔ angular frequencies, ``ω = 2πc/λ``;
* a bandwidth quoted in wavelength around ``λ`` ↔ angular bandwidth, ``Δω = 2πc Δλ/λ²`` (first
  order, the convention of every spectrometer reading);
* frequencies ``f`` (Hz) ↔ angular frequencies ``Ω = 2πf``;
* propagation loss in dB per centimetre ↔ the power attenuation coefficient ``κ`` (1/m) of
  ``P(z) = P(0) e^{−κz}``: ``κ = α_dB · ln(10)/10 · 100``;
* a Gaussian **intensity** FWHM ``Δ`` ↔ the standard deviation ``σ`` of the **amplitude**
  ``exp(−Ω²/(2σ²))``: ``σ = Δ/(2√ln 2)``.

Example
-------
>>> from cvsim.core import units as u
>>> round(u.angular_frequency(795 * u.NM) / 1e15, 6)       # rad/s, /1e15
2.369373
>>> round(u.angular_bandwidth(2 * u.NM, 397.5 * u.NM) / 1e12, 4)   # 2 nm at 397.5 nm, rad/s /1e12
23.8427
>>> round(u.attenuation_from_db_per_cm(0.2), 6)            # 0.2 dB/cm -> 1/m
4.60517
"""

from __future__ import annotations

import math

from ..errors import InputError

__all__ = [
    "CM",
    "FS",
    "GHZ",
    "MHZ",
    "MM",
    "NM",
    "PS",
    "SPEED_OF_LIGHT",
    "UM",
    "amplitude_sigma_from_fwhm",
    "angular",
    "angular_bandwidth",
    "angular_frequency",
    "attenuation_from_db_per_cm",
    "db_per_cm_from_attenuation",
    "fwhm_from_amplitude_sigma",
    "hertz",
    "wavelength",
    "wavelength_bandwidth",
]

#: Speed of light in vacuum (m/s, exact).
SPEED_OF_LIGHT = 299_792_458.0
#: Nanometre (m).
NM = 1e-9
#: Micrometre (m).
UM = 1e-6
#: Millimetre (m).
MM = 1e-3
#: Centimetre (m).
CM = 1e-2
#: Femtosecond (s).
FS = 1e-15
#: Picosecond (s).
PS = 1e-12
#: Megahertz (Hz).
MHZ = 1e6
#: Gigahertz (Hz).
GHZ = 1e9

_TWO_PI_C = 2.0 * math.pi * SPEED_OF_LIGHT
_DB_PER_CM = math.log(10.0) / 10.0 / CM  # (1/m) per (dB/cm)
_FWHM_PER_SIGMA = 2.0 * math.sqrt(math.log(2.0))


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0):
        raise InputError(f"{name} must be finite and > 0, got {value}")
    return v


def _non_negative(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0):
        raise InputError(f"{name} must be finite and >= 0, got {value}")
    return v


def angular_frequency(wavelength: float) -> float:
    """``ω = 2πc/λ`` (rad/s) of a vacuum wavelength ``λ`` (m).

    Parameters
    ----------
    wavelength
        Vacuum wavelength (m), > 0.
    """
    return _TWO_PI_C / _positive("wavelength", wavelength)


def wavelength(omega: float) -> float:
    """Vacuum wavelength ``λ = 2πc/ω`` (m) of an angular frequency ``ω`` (rad/s).

    Parameters
    ----------
    omega
        Angular frequency (rad/s), > 0.
    """
    return _TWO_PI_C / _positive("omega", omega)


def angular_bandwidth(delta_wavelength: float, center_wavelength: float) -> float:
    """Angular bandwidth ``Δω = 2πc Δλ/λ²`` (rad/s) of a wavelength bandwidth ``Δλ`` around ``λ``.

    Parameters
    ----------
    delta_wavelength
        Bandwidth in wavelength (m), >= 0 (e.g. a FWHM read on a spectrometer).
    center_wavelength
        Centre wavelength (m), > 0.
    """
    lam = _positive("center_wavelength", center_wavelength)
    return _TWO_PI_C * _non_negative("delta_wavelength", delta_wavelength) / lam**2


def wavelength_bandwidth(delta_omega: float, center_wavelength: float) -> float:
    """Inverse of :func:`angular_bandwidth`: ``Δλ = λ² Δω/(2πc)`` (m).

    Parameters
    ----------
    delta_omega
        Angular bandwidth (rad/s), >= 0.
    center_wavelength
        Centre wavelength (m), > 0.
    """
    lam = _positive("center_wavelength", center_wavelength)
    return lam**2 * _non_negative("delta_omega", delta_omega) / _TWO_PI_C


def angular(frequency: float) -> float:
    """``Ω = 2πf`` (rad/s) of a frequency ``f`` (Hz).

    Parameters
    ----------
    frequency
        Frequency (Hz).
    """
    f = float(frequency)
    if not math.isfinite(f):
        raise InputError(f"frequency must be finite, got {frequency}")
    return 2.0 * math.pi * f


def hertz(omega: float) -> float:
    """``f = Ω/(2π)`` (Hz) of an angular frequency ``Ω`` (rad/s).

    Parameters
    ----------
    omega
        Angular frequency (rad/s).
    """
    w = float(omega)
    if not math.isfinite(w):
        raise InputError(f"omega must be finite, got {omega}")
    return w / (2.0 * math.pi)


def attenuation_from_db_per_cm(db_per_cm: float) -> float:
    """Power attenuation coefficient ``κ`` (1/m), ``P(z) = P(0) e^{−κz}``, of a loss in dB/cm.

    Parameters
    ----------
    db_per_cm
        Propagation loss (dB/cm), >= 0.
    """
    return _non_negative("db_per_cm", db_per_cm) * _DB_PER_CM


def db_per_cm_from_attenuation(kappa: float) -> float:
    """Inverse of :func:`attenuation_from_db_per_cm`.

    Parameters
    ----------
    kappa
        Power attenuation coefficient (1/m), >= 0.
    """
    return _non_negative("kappa", kappa) / _DB_PER_CM


def amplitude_sigma_from_fwhm(fwhm: float) -> float:
    """``σ`` of the Gaussian amplitude ``exp(−x²/(2σ²))`` whose intensity has FWHM ``Δ``: ``Δ/(2√ln 2)``.

    Parameters
    ----------
    fwhm
        Full width at half maximum of the intensity ``exp(−x²/σ²)``, > 0.
    """
    return _positive("fwhm", fwhm) / _FWHM_PER_SIGMA


def fwhm_from_amplitude_sigma(sigma: float) -> float:
    """Inverse of :func:`amplitude_sigma_from_fwhm`: ``Δ = 2√ln 2 σ``.

    Parameters
    ----------
    sigma
        Standard deviation of the Gaussian amplitude, > 0.
    """
    return _positive("sigma", sigma) * _FWHM_PER_SIGMA
