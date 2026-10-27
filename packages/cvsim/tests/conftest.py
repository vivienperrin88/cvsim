"""Shared pytest configuration for cvsim.

* Falls back to the in-tree ``src/`` sources when cvsim is not installed.
* ``rng``: a seeded NumPy generator, so stochastic tests are reproducible.
* Tests marked ``slow`` are skipped unless ``--runslow`` is given (CI runs them weekly).
* cvsim's own warnings (``CvsimWarning``) fail a test unless it expects them (``pytest.warns``).
* Tests marked ``qttlind`` are skipped when the optional qttlind package is not importable, so the
  suite also passes for a bare ``pip install cvsim``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from collections.abc import Iterator

import numpy as np
import pytest

if importlib.util.find_spec("cvsim") is None:  # not installed: use the source tree
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAVE_QTTLIND = importlib.util.find_spec("qttlind") is not None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", help="also run tests marked slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_slow = pytest.mark.skip(reason="slow test: use --runslow")
    skip_qttlind = pytest.mark.skip(reason="qttlind is not installed")
    for item in items:
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        if "qttlind" in item.keywords and not HAVE_QTTLIND:
            item.add_marker(skip_qttlind)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator: every test run sees the same random numbers."""
    return np.random.default_rng(20261027)


@pytest.fixture(autouse=True)
def _cvsim_warnings_are_errors() -> Iterator[None]:
    """Turn every :class:`cvsim.errors.CvsimWarning` into an error inside each test.

    Imported lazily (not in ``pyproject.toml``) so that coverage measures cvsim from its first import.
    """
    from cvsim.errors import CvsimWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", CvsimWarning)
        yield
