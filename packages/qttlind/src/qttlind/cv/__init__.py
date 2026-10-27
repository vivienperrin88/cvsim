"""Continuous-variable optics in the quantics representation.

A mode is a quantics tensor train in the **position** representation (or the momentum one, a
Fourier transform away), or in the Fock representation of :mod:`qttlind.bosons`. Ladder
operators in position space are MPOs of rank ~15 (through the quantics Fourier transform), so
Gaussian and non-Gaussian dynamics both run without leaving position space.

* :mod:`.grids`     -- grids, Hermite functions, position-space states (Gaussian, cat, GKP)
* :mod:`.fourier`   -- the quantics Fourier transform as an MPO
* :mod:`.operators` -- x, p, a, a^dag, n as MPOs in position space; Fock <-> x transform
* :mod:`.gates`     -- phase-space gates: phases, displacement, shears, symplectic, cubic
* :mod:`.measure`   -- probabilities, homodyne, photon counting, sampling
"""

from . import fourier, gates, grids, measure, operators
from .grids import QuadratureGrid

__all__ = ["QuadratureGrid", "fourier", "gates", "grids", "measure", "operators"]
