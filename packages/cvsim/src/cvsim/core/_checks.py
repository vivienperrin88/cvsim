"""Validation of scalar parameters, shared by every module (one message format, one behaviour).

Each helper converts its argument to ``float`` (or ``int``) and raises
:class:`~cvsim.errors.InputError` naming the parameter, the accepted range and the value received,
also when the value is not a number at all (``float("abc")`` would raise a bare ``ValueError``).
"""

from __future__ import annotations

import math
import operator

import numpy as np

from ..errors import InputError

__all__ = ["finite", "in_range", "integer", "non_negative", "positive", "real"]


def real(name: str, value: object) -> float:
    """``value`` as a ``float`` (may be infinite or nan), or :class:`InputError` if it is not a real number.

    Booleans, complex numbers (Python or NumPy, which ``float`` would truncate) and strings are not
    real numbers; NumPy integer and floating scalars and 0-d arrays are.
    """
    if isinstance(value, (bool, str, bytes)) or np.asarray(value).dtype.kind in "bcSUV":
        raise InputError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a real number, got {value!r}") from None


def finite(name: str, value: object) -> float:
    """``value`` as a finite ``float``."""
    v = real(name, value)
    if not math.isfinite(v):
        raise InputError(f"{name} must be a finite real number, got {value!r}")
    return v


def in_range(
    name: str,
    value: object,
    lo: float = -math.inf,
    hi: float = math.inf,
    *,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    """``value`` as a finite ``float`` in the interval from ``lo`` to ``hi`` (closed unless ``*_open``)."""
    v = real(name, value)
    ok = math.isfinite(v) and (v > lo if lo_open else v >= lo) and (v < hi if hi_open else v <= hi)
    if not ok:
        if math.isinf(hi) and not math.isinf(lo):
            expected = f"finite and {'>' if lo_open else '>='} {lo:g}"
        else:
            expected = f"in {'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}"
        raise InputError(f"{name} must be {expected}, got {value!r}")
    return v


def positive(name: str, value: object) -> float:
    """``value`` as a finite ``float`` > 0."""
    return in_range(name, value, 0.0, lo_open=True)


def non_negative(name: str, value: object) -> float:
    """``value`` as a finite ``float`` >= 0."""
    return in_range(name, value, 0.0)


def integer(name: str, value: object, minimum: int) -> int:
    """``value`` as an ``int`` >= ``minimum``: integers (Python, NumPy, 0-d integer arrays) and
    integral reals such as ``4.0``."""
    n: int | None = None
    if not isinstance(value, (bool, np.bool_)):
        try:
            n = operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            try:
                v = real(name, value)
            except InputError:
                v = math.nan
            if math.isfinite(v) and v.is_integer():
                n = int(v)
    if n is None or n < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return n
