"""Intermediate representation (layer 1): programs as data, independent of any backend.

* :class:`Op` — one operation from the registry :data:`SPECS` (preparations, gates, channels,
  measurements), with parameters, modes and noise-ledger provenance; :func:`op` builds one;
* :class:`Parameter` — a free parameter, bound later;
* :class:`Program` — an immutable mode register and operation sequence, composed with ``>>``,
  bound with :meth:`Program.bind`, serialised to versioned JSON.

What the operations *do* is decided by the backends (``cvsim.backends``, S09).
"""

from __future__ import annotations

from .ops import KINDS, SPECS, Op, OpSpec, Parameter, op
from .program import SCHEMA, Mode, Program

__all__ = ["KINDS", "SCHEMA", "SPECS", "Mode", "Op", "OpSpec", "Parameter", "Program", "op"]
