"""Bosonic modes in the quantics (binary Fock) representation.

* :mod:`~qttlind.bosons.states`    -- Fock, coherent and cat states, sampled functions
* :mod:`~qttlind.bosons.operators` -- shift, ladder operators, exact polynomials, parity
"""

from . import operators, states

__all__ = ["operators", "states"]
