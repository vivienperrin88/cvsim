"""Purified comb state and Kraus-map Lindblad integrators.

``rho = Psi Psi^dagger`` stored as a comb: backbone bonds ``chi_e``, quantics legs ``chi_q`` and a
purification leaf ``chi_mu``, advanced with the first/second-order Kraus maps of the paper's
Sec. V. Operators are sums of :class:`LocalTerm` (a coefficient times single-mode MPOs).
"""

from ..errors import TraceDefectWarning
from .comb import PurifiedComb
from .kraus import (
    LindbladRun,
    LocalTerm,
    StepInfo,
    apply_terms,
    evolve,
    heff_terms,
    kraus_step,
    product_of_terms,
)

__all__ = [
    "LindbladRun",
    "LocalTerm",
    "PurifiedComb",
    "StepInfo",
    "TraceDefectWarning",
    "apply_terms",
    "evolve",
    "heff_terms",
    "kraus_step",
    "product_of_terms",
]
