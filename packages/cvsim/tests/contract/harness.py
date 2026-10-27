"""Shared tools for the contract tests: cvsim versus qttlind (decision D2).

The two libraries implement the same conventions independently (no shared code). A contract test
feeds both the same inputs and requires the same outputs to a tolerance justified by the
computation, never "whatever makes it pass":

* ``EXACT``    -- closed-form matrices built from the same elementary functions: a few ulps;
* ``ROUNDOFF`` -- results of short chains of floating-point operations (products of a few
  matrices, Schur complements of well-conditioned blocks): 1e-12 on O(1) numbers.

Every test module here carries ``pytestmark = pytest.mark.qttlind`` and gets qttlind through
:func:`qttlind_module`, so the suite skips cleanly when the optional package is absent.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest

EXACT = 1e-14
ROUNDOFF = 1e-12


def qttlind_module(name: str) -> ModuleType:
    """Import ``qttlind.<name>``, skipping the calling test if qttlind is not installed."""
    pytest.importorskip("qttlind")
    return importlib.import_module(f"qttlind.{name}")


def assert_same(cvsim_value, qttlind_value, atol: float, what: str = "") -> None:
    """Element-wise agreement with an informative message (largest deviation and where)."""
    a = np.asarray(cvsim_value, dtype=complex)
    b = np.asarray(qttlind_value, dtype=complex)
    assert a.shape == b.shape, f"{what}: shapes differ, cvsim {a.shape} vs qttlind {b.shape}"
    if a.size == 0:
        return
    dev = np.abs(a - b)
    worst = np.unravel_index(int(np.argmax(dev)), dev.shape)
    assert dev.max() <= atol, f"{what}: max |cvsim - qttlind| = {dev.max():.2e} at {worst} (atol {atol:.0e})"


def random_unitary(rng: np.random.Generator, n: int) -> np.ndarray:
    """Haar-random unitary (QR of a complex Ginibre matrix with the phase fix)."""
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))
