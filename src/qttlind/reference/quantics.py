"""Reference quantics index map (paper Eq. 4), independent of qttlind.core.conventions.

n = sum_{i=1..R} 2^(R-i) sigma_i, most significant bit first (sigma_1 sits next to the backbone).
"""

from __future__ import annotations

MAX_BITS = 62


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"number of bits must be in 1..={MAX_BITS}, got {bits}")


def bits_of(n: int, bits: int) -> list[int]:
    """Binary digits of n on `bits` bits, MSB first: bits_of(5, 4) == [0, 1, 0, 1]."""
    _check_bits(bits)
    if not 0 <= n < 2**bits:
        raise ValueError(f"{n} does not fit in {bits} bits")
    return [(n >> (bits - 1 - i)) & 1 for i in range(bits)]


def index_of(sigma) -> int:
    """Inverse of bits_of."""
    sigma = list(sigma)
    _check_bits(len(sigma))
    n = 0
    for s in sigma:
        if s not in (0, 1):
            raise ValueError(f"quantics digits must be 0 or 1, got {s}")
        n = 2 * n + int(s)
    return n
