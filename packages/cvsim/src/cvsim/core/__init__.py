"""Layer 1: physics-agnostic foundations.

* :mod:`cvsim.core.conventions` -- phase-space conventions as code (ordering, units, gate
  symplectic matrices from their Bogoliubov forms); each rule is implemented a second time,
  independently, by the oracle :mod:`cvsim.reference.fock_dense` and compared by the tests.
"""

from __future__ import annotations

from . import conventions

__all__ = ["conventions"]
