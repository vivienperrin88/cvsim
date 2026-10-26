"""Exact Gaussian backend for continuous-variable optics.

A Gaussian state of ``M`` modes is a mean vector and a covariance matrix; Gaussian gates are
symplectic matrices, Gaussian channels are ``(X, Y)`` pairs, Gaussian measurements are Schur
complements. Everything costs O(M^3): this backend handles thousands of modes and is exact.

* :mod:`.phase_space` -- conventions (hbar = 1, xxpp), symplectic algebra
* :mod:`.states`      -- :class:`GaussianState`
* :mod:`.ops`         -- gates and channels (loss, amplification, thermal noise)
* :mod:`.measure`     -- homodyne, heterodyne, overlaps, Wigner, log-negativity
* :mod:`.fock`        -- Fock amplitudes by the Bargmann recursion
* :mod:`.hafnian`     -- photon-number and click statistics

This subpackage imports only NumPy, SciPy and :mod:`qttlind.errors`: it is usable on its own,
and it is the "Gaussian part" of the hybrid executor (:mod:`qttlind.circuits`).
"""

from . import fock, hafnian, measure, ops, phase_space
from .states import GaussianState

__all__ = ["GaussianState", "fock", "hafnian", "measure", "ops", "phase_space"]
