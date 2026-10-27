"""cvsim -- hardware-aware simulation of multimode continuous-variable quantum optics.

cvsim models the experiments of multimode CV photonics -- frequency-comb and waveguide squeezers,
supermode bases, mode-selective homodyne and photon subtraction, reconfigurable cluster states,
Gaussian reservoir computing -- with every imperfection booked in a noise ledger. Pure Python on
NumPy and SciPy.

It complements :mod:`qttlind` (same repository): cvsim owns the multimode, spectral and hardware
layers; qttlind's quantics comb is an optional backend for non-Gaussian and Lindblad dynamics, and
its Gaussian backend can be selected in place of cvsim's own (they are cross-validated).

Subpackages arrive session by session (see ``PROGRESS.md``); the layer they belong to is fixed in
``docs/design/architecture.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cvsim")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
