"""Layer 1: physics-agnostic foundations.

* :mod:`cvsim.core.conventions` -- phase-space conventions as code (ordering, units, gate
  symplectic matrices from their Bogoliubov forms); each rule is implemented a second time,
  independently, by the oracle :mod:`cvsim.reference.fock_dense` and compared by the tests;
* :mod:`cvsim.core.linalg` -- the only door to LAPACK: kernels with timing and fallbacks, and the
  symplectic decompositions (Williamson, Bloch–Messiah, Takagi) with a reproducible gauge;
* :mod:`cvsim.core.hafnian` -- permanent, hafnian, loop hafnian and torontonian (S10);
* :mod:`cvsim.core.units` -- SI scale factors and the conversions of laboratory units (wavelength,
  bandwidth, dB/cm, FWHM) used by the hardware layer (S11).
"""

from __future__ import annotations

from . import conventions, hafnian, linalg, units

__all__ = ["conventions", "hafnian", "linalg", "units"]
