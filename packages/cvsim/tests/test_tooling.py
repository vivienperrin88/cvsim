"""The project's own tooling: version, licence, workspace wiring."""

import os
import tomllib

from conftest import REPO_ROOT

import cvsim


def _pyproject() -> dict:
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def test_version_matches_pyproject():
    assert cvsim.__version__ == _pyproject()["project"]["version"]


def test_package_is_private():
    """PyPI refuses any upload carrying this classifier: the code cannot be published by accident."""
    project = _pyproject()["project"]
    assert "Private :: Do Not Upload" in project["classifiers"]
    assert project["license"] == "LicenseRef-Proprietary"
    with open(os.path.join(REPO_ROOT, "LICENSE")) as fh:
        assert "All rights reserved" in fh.read()


def test_runtime_dependencies_are_numpy_and_scipy_only():
    deps = {d.split(">")[0].split("=")[0].split("<")[0].strip() for d in _pyproject()["project"]["dependencies"]}
    assert deps == {"numpy", "scipy"}


def test_qttlind_is_an_optional_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any(req.startswith("qttlind") for req in extras["qttlind"])


def test_python_floor_matches_qttlind():
    assert _pyproject()["project"]["requires-python"] == ">=3.12"
