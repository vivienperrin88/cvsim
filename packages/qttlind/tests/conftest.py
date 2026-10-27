"""Shared pytest configuration for qttlind.

* Falls back to the in-tree ``src/`` sources when qttlind is not installed.
* ``rng``: a seeded NumPy generator, so stochastic tests are reproducible.
* Tests marked ``slow`` are skipped unless ``--runslow`` is given (CI runs them nightly).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

if importlib.util.find_spec("qttlind") is None:  # not installed: use the source tree
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", help="also run tests marked slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test: use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator: every test run sees the same random numbers."""
    return np.random.default_rng(20261013)
