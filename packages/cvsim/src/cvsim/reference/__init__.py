"""Independent oracles used by the test suite (aside from the layers).

:mod:`cvsim.reference.fock_dense` computes in a truncated Fock space with no phase-space shortcut;
:mod:`cvsim.reference.sources` recomputes the source models from their equations of motion
(input–output theory, pixel-basis Hamiltonians, Lyapunov equations).
Modules here import NumPy and SciPy only -- never cvsim -- and no cvsim module imports them, so a
convention or formula error in the library cannot be reproduced by its own check.
"""

from __future__ import annotations

from . import fock_dense, sources

__all__ = ["fock_dense", "sources"]
