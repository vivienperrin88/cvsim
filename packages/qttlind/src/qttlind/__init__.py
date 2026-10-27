"""qttlind -- multilevel tensor-network compression for Lindblad dynamics of bosonic circuits.

Implements Moulinas & Waintal, arXiv:2605.27515: a purified density matrix rho = Psi Psi^dagger,
stored as a comb tensor network (inter-mode backbone + one quantics leg per mode), advanced with
Kraus-map integrators. Pure Python on NumPy/SciPy.

Subpackages:

* :mod:`qttlind.core`      -- conventions, dense kernel (linalg), TensorTrain, MPO
* :mod:`qttlind.bosons`    -- quantics bosonic states and operators
* :mod:`qttlind.closed`    -- closed-system integrators
* :mod:`qttlind.lindblad`  -- purified comb and Kraus integrators
* :mod:`qttlind.api`       -- System, LindbladSolver, Result, plotting
* :mod:`qttlind.gaussian`  -- exact Gaussian backend for CV optics (no tensor networks)
* :mod:`qttlind.cv`        -- CV states, gates and measurements in quantics form
* :mod:`qttlind.circuits`  -- CV circuits, hybrid Gaussian/QTT executor, digital twin
* :mod:`qttlind.reference` -- dense NumPy oracles used to validate everything above

``FEATURES`` lists what is implemented. Callers test it rather than individual modules::

    import qttlind as ql
    ql.require("lindblad", "api")   # ImportError naming what is missing
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import api, bosons, circuits, closed, core, cv, errors, gaussian, lindblad, reference
from .api import (
    CustomBasis,
    LindbladSolver,
    Mode,
    Operator,
    Result,
    System,
    clear_mpo_cache,
    coeff,
    evolve_closed,
    states,
)
from .core import conventions
from .core.mpo import MPO, apply_mpo
from .core.tci import quantics_tci
from .core.tree import TreeTN
from .core.tt import TensorTrain
from .lindblad import PurifiedComb

try:
    __version__ = version("qttlind")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0+unknown"

KNOWN_FEATURES: frozenset[str] = frozenset(
    {"conventions", "qtt", "mpo", "tci", "closed", "comb", "lindblad", "api", "gaussian", "cv", "cv_circuits"}
)

# A feature name is listed only when its API is complete and its tests pass (see CONVENTIONS.md).
_implemented: set[str] = {
    "conventions",
    "qtt",
    "mpo",
    "tci",
    "closed",
    "comb",
    "lindblad",
    "api",
    "gaussian",
    "cv",
    "cv_circuits",
}

FEATURES: frozenset[str] = frozenset(_implemented)

__all__ = [
    "FEATURES",
    "KNOWN_FEATURES",
    "MPO",
    "CustomBasis",
    "LindbladSolver",
    "Mode",
    "Operator",
    "PurifiedComb",
    "Result",
    "System",
    "TensorTrain",
    "TreeTN",
    "__version__",
    "api",
    "apply_mpo",
    "bosons",
    "circuits",
    "clear_mpo_cache",
    "closed",
    "coeff",
    "conventions",
    "core",
    "cv",
    "errors",
    "evolve_closed",
    "gaussian",
    "lindblad",
    "quantics_tci",
    "reference",
    "require",
    "states",
]


def require(*features: str) -> None:
    """Raise ImportError unless every named feature is implemented in this version.

    Raises
    ------
    ValueError
        For a name that is not a known feature (a typo should not pass silently).
    ImportError
        If a known feature is not implemented yet.
    """
    unknown = set(features) - KNOWN_FEATURES
    if unknown:
        raise ValueError(f"unknown feature(s) {sorted(unknown)}; known: {sorted(KNOWN_FEATURES)}")
    missing = set(features) - FEATURES
    if missing:
        raise ImportError(
            f"qttlind {__version__} does not implement {sorted(missing)} yet; available: {sorted(FEATURES)}"
        )
