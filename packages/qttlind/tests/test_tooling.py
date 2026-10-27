"""The project's own tooling: the version is defined once, in pyproject.toml."""

import os
import tomllib

from conftest import REPO_ROOT

import qttlind


def test_version_matches_pyproject():
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert qttlind.__version__ == declared
