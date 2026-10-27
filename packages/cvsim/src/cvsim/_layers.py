"""The layer of every subpackage (``docs/design/architecture.md``, section 3), as data.

A module may import from its own layer or from lower layers, never from a higher one. Two packages
stand aside: ``errors`` (imported by everyone, imports nothing from cvsim) and ``reference`` (the
dense oracles: import NumPy and SciPy only, never cvsim, so that they stay independent of what
they check).

The test suite enforces the rule by parsing every module (``tests/test_guards.py``). A new
subpackage must be registered here before it can be added: an unknown name fails the suite.
"""

from __future__ import annotations

from types import MappingProxyType

__all__ = ["ASIDE", "LAYERS", "QTTLIND_ALLOWED", "SAME_LAYER_FORBIDDEN"]

#: Layer number of each subpackage of ``cvsim``.
LAYERS = MappingProxyType(
    {
        # layer 1 -- physics-agnostic foundations
        "core": 1,
        "ir": 1,
        # layer 2 -- representations of states and their elementary maps
        "gaussian": 2,
        "fock": 2,
        "lcg": 2,
        "mps": 2,
        "lindblad": 2,
        # layer 3 -- hardware models and data analysis
        "hardware": 3,
        "characterise": 3,
        # layer 4 -- execution and applications
        "backends": 4,
        "experiments": 4,
        "protocols": 4,
    }
)

#: Packages outside the layers. ``errors`` may be imported by anyone; ``reference`` by tests only.
ASIDE = frozenset({"errors", "reference"})

#: Same-layer pairs ``(importer, imported)`` that are nevertheless forbidden, to keep siblings
#: decoupled (as qttlind keeps ``closed`` and ``lindblad`` unaware of each other).
#: Characterisation may use hardware models (detector calibration); hardware never depends on data
#: analysis.
SAME_LAYER_FORBIDDEN = frozenset({("hardware", "characterise")})

#: The only subpackage that may import the optional dependency ``qttlind`` (decision D2).
QTTLIND_ALLOWED = frozenset({"backends"})
