"""Exception and warning hierarchy of qttlind.

Every error raised on purpose by the library derives from :class:`QttlindError`, so callers can
catch "anything qttlind complained about" in one ``except``. Each concrete class *also* derives
from the matching built-in (``ValueError``, ``RuntimeError``), so code written against the
standard exceptions keeps working.

Policy (see the architecture page, decision D7):

* Validate arguments at **public** entry points and raise :class:`InputError` / :class:`ShapeError`
  with a message that says what was expected and what was received.
* Inside hot loops, do not re-validate: rely on the invariants established at the boundary.
* Use ``assert`` only for internal invariants that can never be violated by user input.
* Use warnings (:class:`QttlindWarning` subclasses) for results that are valid but suspicious.
"""

from __future__ import annotations

__all__ = [
    "ConvergenceError",
    "CutoffWarning",
    "InputError",
    "QttlindError",
    "QttlindWarning",
    "ShapeError",
    "TraceDefectWarning",
]


class QttlindError(Exception):
    """Base class of every exception raised on purpose by qttlind."""


class InputError(QttlindError, ValueError):
    """An argument has an invalid value (out of range, wrong sign, not sorted, ...)."""


class ShapeError(InputError):
    """Array or tensor-network shapes are inconsistent (bond mismatch, wrong length, ...)."""


class ConvergenceError(QttlindError, RuntimeError):
    """An iterative numerical method did not converge within its budget."""


class QttlindWarning(UserWarning):
    """Base class of qttlind warnings."""


class CutoffWarning(QttlindWarning):
    """A truncated Hilbert space is close to being exhausted (population near the cutoff)."""


class TraceDefectWarning(QttlindWarning):
    """A Kraus step lost (or gained) a large fraction of the trace: the time step is too big."""
