"""Quantics tensor trains of single-mode bosonic states.

A state of one mode truncated to ``N = 2**bits`` Fock states is a :class:`TensorTrain` of
length ``bits`` with physical dimension 2 per core (quantics encoding, MSB first).

Numerical trick: coherent-state amplitudes ``e^{-|a|^2/2} a^n / sqrt(n!)`` overflow for large
``n`` if computed naively; build their logarithm with ``gammaln`` and exponentiate at the end.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import gammaln

from ..core.conventions import bits_of
from ..core.tt import TensorTrain
from ..errors import InputError

__all__ = ["MAX_DENSE_BITS", "cat", "coherent", "coherent_amplitudes", "fock", "from_function"]

#: Above this many bits, dense construction (2**bits complex numbers) is refused.
MAX_DENSE_BITS = 26

#: Default discarded weight for state construction (relative error ~1e-10).
DEFAULT_TOL = 1e-20


def fock(k: int, bits: int) -> TensorTrain:
    """Exact rank-1 quantics tensor train of the Fock state ``|k>``."""
    digits = bits_of(k, bits)
    return TensorTrain.product([np.eye(2)[s] for s in digits])


def from_function(
    f: Callable[[np.ndarray], np.ndarray], bits: int, tol: float = DEFAULT_TOL, chi_max: int | None = None
) -> TensorTrain:
    """Tensor train of ``f(n)`` for every ``n`` in ``range(2**bits)``.

    Up to ``MAX_DENSE_BITS`` bits, ``f`` is sampled densely and compressed by TT-SVD (exact up to
    ``tol``). Beyond, the tensor train is built by tensor cross interpolation
    (:func:`qttlind.core.tci.quantics_tci`) from ``O(bits chi^2)`` evaluations, then compressed.

    Raises
    ------
    InputError
        If ``bits`` is not in ``1..62`` or if ``f`` does not return one value per input.
    """
    if not 1 <= bits <= 62:
        raise InputError(f"bits must be in 1..62, got {bits}")
    if bits > MAX_DENSE_BITS:
        from ..core.tci import quantics_tci

        tt, _info = quantics_tci(f, bits, tol=tol, chi_max=chi_max)
        out, _ = tt.compress(tol, chi_max)
        return out
    n = np.arange(2**bits)
    values = np.asarray(f(n), dtype=np.complex128)
    if values.shape != n.shape:
        raise InputError(f"f must return an array of shape {n.shape}, got {values.shape}")
    return TensorTrain.from_dense(values, [2] * bits, tol=tol, chi_max=chi_max)


def coherent_amplitudes(alpha: complex, n: np.ndarray) -> np.ndarray:
    """``<n|alpha> = exp(-|alpha|^2/2) alpha^n / sqrt(n!)``, computed in log space."""
    n = np.asarray(n)
    if alpha == 0:
        return (n == 0).astype(np.complex128)
    log_mod = -(abs(alpha) ** 2) / 2 + n * np.log(abs(alpha)) - 0.5 * gammaln(n + 1)
    return np.exp(log_mod + 1j * n * np.angle(alpha))


def coherent(alpha: complex, bits: int, tol: float = DEFAULT_TOL, chi_max: int | None = None) -> TensorTrain:
    """Normalised coherent state ``|alpha>`` truncated to ``2**bits`` Fock states."""
    tt = from_function(lambda n: coherent_amplitudes(alpha, n), bits, tol, chi_max)
    return tt * (1.0 / tt.norm())


def cat(
    alpha: complex, bits: int, parity: int = 1, tol: float = DEFAULT_TOL, chi_max: int | None = None
) -> TensorTrain:
    """Normalised cat state ``|alpha> + parity |-alpha>`` (``parity`` = +1 even, -1 odd)."""
    if parity not in (1, -1):
        raise InputError(f"parity must be +1 or -1, got {parity}")
    if alpha == 0 and parity == -1:
        raise InputError("the odd cat state is undefined for alpha = 0")
    tt = from_function(lambda n: coherent_amplitudes(alpha, n) * (1 + parity * (-1.0) ** n), bits, tol, chi_max)
    return tt * (1.0 / tt.norm())
