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


# ------------------------------------------------------------------------------- MrMustard bridge
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HBAR_REF = 2.0  # The Walrus and MrMustard
_BRIDGE = Path(__file__).resolve().parents[1] / "reference" / "mrmustard_bridge.py"
_REPO = Path(__file__).resolve().parents[4]


def mrmustard_python() -> Path | None:
    """Interpreter of the MrMustard reference environment: ``$CVSIM_MRMUSTARD_PYTHON``, else
    ``<repo>/.reference-envs/mrmustard/bin/python`` (built by ``tools/setup_reference_envs.sh``)."""
    env = os.environ.get("CVSIM_MRMUSTARD_PYTHON")
    path = Path(env) if env else _REPO / ".reference-envs" / "mrmustard" / "bin" / "python"
    return path if path.exists() else None


class MrMustard:
    """A persistent MrMustard process (TensorFlow is imported once per test session)."""

    def __init__(self, python: Path, backend: str = "numpy"):
        self._proc = subprocess.Popen(
            [str(python), str(_BRIDGE), backend],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.versions = self("versions")

    def __call__(self, fn: str, **args):
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps({"fn": fn, "args": args}, default=_encode) + "\n")
        self._proc.stdin.flush()
        reply = json.loads(self._proc.stdout.readline())
        if "error" in reply:
            raise RuntimeError(f"MrMustard bridge: {reply['error']}")
        return _decode(reply["ok"])

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait(timeout=30)


def _encode(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return {"re": a.real.tolist(), "im": a.imag.tolist()}
    return a.tolist()


def _decode(x):
    if isinstance(x, dict):
        if set(x) == {"re", "im"}:
            return np.asarray(x["re"]) + 1j * np.asarray(x["im"])
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return np.asarray(x, dtype=float)
    return x


_mm: dict[str, MrMustard] = {}


def mrmustard(backend: str = "numpy") -> MrMustard:
    """The shared bridge running MrMustard's ``backend`` (``"numpy"`` or ``"tensorflow"``, one
    process each); skips the calling test if the reference environment is missing."""
    if backend not in _mm:
        python = mrmustard_python()
        if python is None:
            pytest.skip("MrMustard reference environment not found (tools/setup_reference_envs.sh)")
        _mm[backend] = MrMustard(python, backend)
    return _mm[backend]


def to_ref(mean, cov):
    """cvsim moments (ħ = 1) in the ħ = 2 convention of The Walrus and MrMustard."""
    return np.asarray(mean) * np.sqrt(HBAR_REF), np.asarray(cov) * HBAR_REF


def from_ref(mean, cov):
    """ħ = 2 moments back to cvsim's ħ = 1."""
    return np.asarray(mean) / np.sqrt(HBAR_REF), np.asarray(cov) / HBAR_REF
