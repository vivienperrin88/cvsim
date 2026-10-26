"""Package surface: metadata, feature flags, errors, doctests."""

import doctest
import importlib

import pytest

import qttlind
from qttlind import errors


def test_version_is_a_string():
    assert isinstance(qttlind.__version__, str) and qttlind.__version__


def test_features_are_known():
    assert isinstance(qttlind.FEATURES, frozenset)
    assert qttlind.FEATURES <= qttlind.KNOWN_FEATURES


def test_every_known_feature_is_implemented():
    assert qttlind.FEATURES == qttlind.KNOWN_FEATURES
    qttlind.require(*sorted(qttlind.KNOWN_FEATURES))


def test_require_rejects_unknown_features():
    with pytest.raises(ValueError, match="unknown feature"):
        qttlind.require("no-such-feature")


@pytest.mark.parametrize(
    "name", ["core", "bosons", "closed", "lindblad", "api", "gaussian", "cv", "circuits", "reference", "errors"]
)
def test_subpackages_import_and_are_documented(name):
    assert importlib.import_module(f"qttlind.{name}").__doc__


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(errors.InputError, errors.QttlindError)
        assert issubclass(errors.InputError, ValueError)
        assert issubclass(errors.ShapeError, errors.InputError)
        assert issubclass(errors.ConvergenceError, RuntimeError)
        assert issubclass(errors.CutoffWarning, UserWarning)

    def test_one_except_catches_all(self):
        for exc in (errors.InputError, errors.ShapeError, errors.ConvergenceError):
            with pytest.raises(errors.QttlindError):
                raise exc("boom")


DOCTEST_MODULES = [
    "qttlind.core.conventions",
    "qttlind.core.linalg",
]


@pytest.mark.parametrize("module", DOCTEST_MODULES)
def test_doctests(module):
    """Run a module's docstring examples. Examples that reach an unimplemented exercise skip."""
    mod = importlib.import_module(module)
    finder = doctest.DocTestFinder()
    runner = doctest.DebugRunner(verbose=False)  # raises on the first problem
    n = 0
    for test in finder.find(mod):
        try:
            runner.run(test)
        except doctest.UnexpectedException as exc:
            if isinstance(exc.exc_info[1], NotImplementedError):
                raise exc.exc_info[1] from None  # -> reported as TODO by conftest
            raise
        n += len(test.examples)
    assert n > 0
