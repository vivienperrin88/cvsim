"""Exception and warning hierarchy of cvsim.

Every error raised on purpose by the library derives from :class:`CvsimError`, so callers can
catch "anything cvsim complained about" in one ``except``. Each concrete class *also* derives from
the matching built-in (``ValueError``, ``RuntimeError``), so code written against the standard
exceptions keeps working. The hierarchy mirrors qttlind's (``qttlind.errors``) class for class,
with two additions for physics and calibration data.

Policy (decision D3, inherited from qttlind's D7):

* Validate arguments at **public** entry points; raise :class:`InputError` (or a subclass) with a
  message that says what was expected and what was received.
* Inside hot loops, do not re-validate: rely on the invariants established at the boundary.
* Use ``assert`` only for internal invariants that user input can never violate.
* Use warnings (:class:`CvsimWarning` subclasses) for results that are valid but suspicious or
  approximate; the test suite turns them into errors unless a test expects them.

=====================  ======================  ==================================================
class                  also a                  raised when
=====================  ======================  ==================================================
``InputError``         ``ValueError``          an argument has an invalid value
``ShapeError``         ``InputError``          array shapes or mode counts are inconsistent
``PhysicalityError``   ``InputError``          a state or channel violates quantum mechanics
                                               (``V + iΩ/2 ≱ 0``, a channel not completely
                                               positive, a density matrix not positive)
``CalibrationError``   ``InputError``          calibration data cannot come from the model
                                               (e.g. a measured variance unreachable by loss)
``ConvergenceError``   ``RuntimeError``        an iterative method did not converge
=====================  ======================  ==================================================
"""

from __future__ import annotations

__all__ = [
    "ApproximationWarning",
    "CalibrationError",
    "ConvergenceError",
    "CutoffWarning",
    "CvsimError",
    "CvsimWarning",
    "InputError",
    "PhysicalityError",
    "ShapeError",
]


class CvsimError(Exception):
    """Base class of every exception raised on purpose by cvsim."""


class InputError(CvsimError, ValueError):
    """An argument has an invalid value (out of range, wrong sign, unknown name, ...)."""


class ShapeError(InputError):
    """Array shapes or mode counts are inconsistent (a ``2M x 2M`` matrix for ``M`` modes, ...)."""


class PhysicalityError(InputError):
    """A state or channel violates quantum mechanics beyond the numerical tolerance.

    Examples: a covariance matrix with ``V + iΩ/2`` not positive semidefinite (the uncertainty
    principle), a Gaussian channel ``(X, Y)`` that is not completely positive, a density matrix
    with a negative eigenvalue.
    """


class CalibrationError(InputError):
    """Calibration data are inconsistent with the hardware model they are meant to calibrate."""


class ConvergenceError(CvsimError, RuntimeError):
    """An iterative numerical method did not converge within its budget."""


class CvsimWarning(UserWarning):
    """Base class of cvsim warnings."""


class ApproximationWarning(CvsimWarning):
    """A result relies on an approximation whose validity the caller should check.

    Example: phase noise applied to a Gaussian state by moment matching (the exact channel maps
    Gaussian states to non-Gaussian mixtures).
    """


class CutoffWarning(CvsimWarning):
    """A truncated Fock space is close to being exhausted (population near the cutoff)."""
