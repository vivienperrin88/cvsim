"""Dense NumPy reference implementations ("oracles") used to validate the QTT solver.

They need only NumPy and SciPy and never import the rest of qttlind, so they stay an independent check.
"""

from . import dense_ref, quantics

__all__ = ["dense_ref", "quantics"]
