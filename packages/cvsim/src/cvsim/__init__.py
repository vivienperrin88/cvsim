"""cvsim -- hardware-aware simulation of multimode continuous-variable quantum optics.

cvsim models the experiments of multimode CV photonics -- frequency-comb and waveguide squeezers,
supermode bases, mode-selective homodyne and photon subtraction, reconfigurable cluster states,
Gaussian reservoir computing -- with every imperfection booked in a noise ledger. Pure Python on
NumPy and SciPy.

It complements :mod:`qttlind` (same repository): cvsim owns the multimode, spectral and hardware
layers; qttlind's quantics comb is an optional backend for non-Gaussian and Lindblad dynamics, and
its Gaussian backend can be selected in place of cvsim's own (they are cross-validated).

Subpackages arrive session by session (``docs/design/roadmap.md``); the layer each belongs to is
fixed in :mod:`cvsim._layers`. ``FEATURES`` lists what is implemented; callers test it rather than
individual modules::

    import cvsim
    cvsim.require("gaussian")   # ImportError naming what is missing
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import errors

try:
    __version__ = version("cvsim")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    __version__ = "0+unknown"

#: Every capability the roadmap plans (the comment names the session that delivers it).
KNOWN_FEATURES: frozenset[str] = frozenset(
    {
        "conventions",  # S02
        "linalg",  # S03
        "gaussian",  # S04-S05
        "channels",  # S06 (with the noise ledger)
        "gaussian_measure",  # S07
        "ir",  # S08
        "backends",  # S09
        "photon_counting",  # S10
        "hardware",  # S11-S14
        "networks",  # S16
        "mbqc",  # S17
        "routing",  # S18
        "fock",  # S20
        "lcg",  # S20
        "mps",  # S21
        "subtraction",  # S22
        "nongaussian",  # S23
        "gbs",  # S25
        "reservoir",  # S26
        "qml",  # S27
        "lindblad",  # S29-S30
        "qttlind_backend",  # S31 (also needs the optional qttlind package)
        "characterise",  # S33-S34
    }
)

# A feature name is listed only when its API is complete and its tests pass.
_implemented: set[str] = {
    "conventions",
    "linalg",
    "gaussian",
    "channels",
    "gaussian_measure",
    "ir",
    "backends",
    "photon_counting",
}

FEATURES: frozenset[str] = frozenset(_implemented)

__all__ = ["FEATURES", "KNOWN_FEATURES", "__version__", "errors", "require"]


def require(*features: str) -> None:
    """Raise ImportError unless every named feature is implemented in this version.

    Parameters
    ----------
    *features
        Names from :data:`KNOWN_FEATURES`.

    Raises
    ------
    InputError
        For a name that is not a known feature (a typo must not pass silently).
    ImportError
        If a known feature is not implemented yet.
    """
    unknown = set(features) - KNOWN_FEATURES
    if unknown:
        raise errors.InputError(f"unknown feature(s) {sorted(unknown)}; known: {sorted(KNOWN_FEATURES)}")
    missing = set(features) - FEATURES
    if missing:
        raise ImportError(
            f"cvsim {__version__} does not implement {sorted(missing)} yet; available: {sorted(FEATURES)}"
        )
