"""The quantics Fourier transform: position <-> momentum.

On the grids of :mod:`.grids`, the momentum amplitudes are the centred unitary DFT of the position
amplitudes (``p_k x_j = 2 pi (k - N/2)(j - N/2) / N`` for every box length)::

    c~_k = N^{-1/2} sum_j exp(-2 pi i (k - N/2)(j - N/2) / N) c_j.

**Bits.** With MSB-first bits ``j = sum_a j_a 2^{R-a}`` and the *output written LSB first*
(site ``c`` carries ``k^_c = k_{R+1-c}``), the plain DFT factorises into pairwise phases::

    exp(-2 pi i k j / N) = prod_{c <= a} exp(-2 pi i j_a k^_c 2^{c-a-1})

(terms with ``c > a`` are integer multiples of ``2 pi``). Each input bit ``j_a`` contributes a
factor of MPO rank 2 (``j_a = 0``: nothing; ``j_a = 1``: a product of phases on sites ``1..a``),
and the element-wise (Hadamard) product of the ``R`` factors, compressed after each step,
**stays at rank ~ 10** for any ``R`` (Chen, Stoudenmire, White, PRX Quantum 4, 040318, 2023).
The bit reversal of the output is *not* low rank -- but reversing the order of the sites of a
tensor train is free, so we never build it.

The centring factors are rank 1: ``exp(-2 pi i (k - N/2)(j - N/2)/N) = exp(-2 pi i kj/N) (-1)^j
(-1)^k exp(-i pi N/2)``, and ``(-1)^j`` only depends on the last bit of ``j``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..core.mpo import MPO
from ..core.tt import TensorTrain
from ..errors import InputError

__all__ = ["FOURIER_TOL", "dft_mpo", "reverse_sites", "to_momentum", "to_position"]

#: Discarded weight used when building and applying the Fourier MPO (relative error ~1e-12).
FOURIER_TOL = 1e-24


def _bit_factor(a: int, bits: int) -> TensorTrain:
    """Rank-2 factor of input bit ``a`` (0-based site) in the fused (out, in) tensor-train view.

    Fused physical index ``o * 2 + i`` (out bit ``o = k^_c``, in bit ``i = j_c``), as in
    :meth:`MPO.as_tensor_train`. Bond channel 0 carries the branch ``j_a = 0`` (factor 1),
    channel 1 the branch ``j_a = 1``, which picks up ``exp(-2 pi i k^_c 2^{c-a-1})`` on sites ``c <= a``.
    """
    cores = []
    for c in range(bits):
        if c < a:  # phase on the output bit, any input bit
            w = np.zeros((2, 4, 2), dtype=complex)
            for o in (0, 1):
                for i in (0, 1):
                    w[0, 2 * o + i, 0] = 1.0
                    w[1, 2 * o + i, 1] = np.exp(-2j * np.pi * o * 2.0 ** (c - a - 1))
            cores.append(w)
        elif c == a:  # the input bit selects the branch; its own output phase is (-1)^o
            w = np.zeros((2, 4, 1), dtype=complex)
            for o in (0, 1):
                w[0, 2 * o + 0, 0] = 1.0
                w[1, 2 * o + 1, 0] = (-1.0) ** o
            cores.append(w)
        else:  # no constraint
            cores.append(np.ones((1, 4, 1), dtype=complex))
    cores[0] = cores[0].sum(axis=0, keepdims=True)  # enter both branches
    return TensorTrain(cores, copy=False)


@lru_cache(maxsize=32)
def dft_mpo(bits: int, tol: float = FOURIER_TOL) -> MPO:
    """Centred unitary DFT as an MPO whose **output** bits are in reversed (LSB-first) order.

    Built as the Hadamard product of the ``bits`` factors of :func:`_bit_factor`, compressed after
    every product, times the rank-1 centring factors and ``N^{-1/2}``. Cached: it depends only on
    ``(bits, tol)``. Treat the returned MPO as read-only.
    """
    if not 1 <= bits <= 62:
        raise InputError(f"bits must be in 1..62, got {bits}")
    tt = _bit_factor(0, bits)
    for a in range(1, bits):
        tt, _ = tt.hadamard(_bit_factor(a, bits)).compress(tol)
    # centring: (-1)^j on the input LSB (site bits-1), (-1)^k on the output LSB (site 0)
    # and the constant exp(-i pi N/2) N^{-1/2}
    sign_in = np.array([1, -1, 1, -1], dtype=complex)  # (o, i) -> (-1)^i
    sign_out = np.array([1, 1, -1, -1], dtype=complex)  # (o, i) -> (-1)^o
    cores = [c.copy() for c in tt.cores]
    cores[-1] = cores[-1] * sign_in[None, :, None]
    cores[0] = cores[0] * sign_out[None, :, None]
    N = 2**bits
    phase = np.exp(-1j * np.pi * (N // 2)) / np.sqrt(N) if bits < 63 else 1.0
    cores[0] = cores[0] * phase
    return MPO.from_tensor_train(TensorTrain(cores, copy=False), [2] * bits, [2] * bits)


def reverse_sites(tt: TensorTrain) -> TensorTrain:
    """The same tensor with its sites in the opposite order (exact, no arithmetic)."""
    return TensorTrain([c.transpose(2, 1, 0) for c in reversed(tt.cores)])


def to_momentum(psi: TensorTrain, tol: float = FOURIER_TOL, chi_max: int | None = None) -> TensorTrain:
    """Momentum-space amplitudes ``c~_k`` (MSB first) of position-space amplitudes ``c_j``."""
    out, _ = dft_mpo(psi.length, FOURIER_TOL).apply(psi).compress(tol, chi_max)
    return reverse_sites(out)


def to_position(phi: TensorTrain, tol: float = FOURIER_TOL, chi_max: int | None = None) -> TensorTrain:
    """Inverse of :func:`to_momentum`: reverse, then apply the adjoint MPO."""
    out, _ = dft_mpo(phi.length, FOURIER_TOL).dagger().apply(reverse_sites(phi)).compress(tol, chi_max)
    return out
