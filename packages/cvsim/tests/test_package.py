"""Package surface: metadata, feature flags, errors and warnings, doctests."""

import doctest
import importlib
import warnings

import pytest

import cvsim
from cvsim import errors


def test_version_is_a_string():
    assert isinstance(cvsim.__version__, str) and cvsim.__version__


def test_package_is_documented():
    assert cvsim.__doc__ and "qttlind" in cvsim.__doc__


# ------------------------------------------------------------------------------- feature flags
class TestFeatures:
    def test_features_are_known(self):
        assert isinstance(cvsim.FEATURES, frozenset)
        assert cvsim.FEATURES <= cvsim.KNOWN_FEATURES

    def test_require_nothing_passes(self):
        cvsim.require()

    def test_require_implemented_features(self):
        cvsim.require(*sorted(cvsim.FEATURES))

    def test_unknown_feature_is_an_input_error(self):
        with pytest.raises(errors.InputError, match=r"unknown feature\(s\) \['gausian'\]"):
            cvsim.require("gausian")

    def test_unknown_feature_is_still_a_value_error(self):
        with pytest.raises(ValueError):
            cvsim.require("no-such-feature")

    @pytest.mark.parametrize("name", sorted(cvsim.KNOWN_FEATURES - cvsim.FEATURES))
    def test_planned_feature_raises_import_error_naming_it(self, name):
        with pytest.raises(ImportError, match=name):
            cvsim.require(name)


# ------------------------------------------------------------------------------- errors
class TestErrors:
    @pytest.mark.parametrize(
        ("cls", "bases"),
        [
            (errors.InputError, (errors.CvsimError, ValueError)),
            (errors.ShapeError, (errors.InputError, ValueError)),
            (errors.PhysicalityError, (errors.InputError, ValueError)),
            (errors.CalibrationError, (errors.InputError, ValueError)),
            (errors.ConvergenceError, (errors.CvsimError, RuntimeError)),
        ],
    )
    def test_hierarchy(self, cls, bases):
        for base in bases:
            assert issubclass(cls, base)

    def test_one_except_catches_all(self):
        for exc in (
            errors.InputError,
            errors.ShapeError,
            errors.PhysicalityError,
            errors.CalibrationError,
            errors.ConvergenceError,
        ):
            with pytest.raises(errors.CvsimError):
                raise exc("boom")

    def test_builtin_errors_are_not_cvsim_errors(self):
        """A plain ValueError raised by NumPy must not be mistaken for a deliberate cvsim error."""
        assert not issubclass(ValueError, errors.CvsimError)

    @pytest.mark.parametrize("cls", [errors.ApproximationWarning, errors.CutoffWarning])
    def test_warnings(self, cls):
        assert issubclass(cls, errors.CvsimWarning)
        assert issubclass(cls, UserWarning)

    def test_cvsim_warnings_fail_tests_unless_expected(self):
        """conftest turns cvsim warnings into errors; a test that expects one uses pytest.warns."""
        with pytest.raises(errors.ApproximationWarning):
            warnings.warn("approximate", errors.ApproximationWarning, stacklevel=1)
        with pytest.warns(errors.ApproximationWarning):
            warnings.warn("approximate", errors.ApproximationWarning, stacklevel=1)

    def test_other_warnings_are_untouched(self):
        with pytest.warns(UserWarning):
            warnings.warn("not ours", UserWarning, stacklevel=1)

    def test_every_class_is_exported(self):
        public = {n for n, v in vars(errors).items() if isinstance(v, type) and issubclass(v, Exception)}
        assert public == set(errors.__all__)


# ------------------------------------------------------------------------------- doctests
DOCTEST_MODULES = ["cvsim.core.conventions", "cvsim.core.linalg"]


@pytest.mark.parametrize("module", DOCTEST_MODULES)
def test_doctests(module):
    """Run a module's docstring examples (each module listed here must have some)."""
    mod = importlib.import_module(module)
    runner = doctest.DebugRunner(verbose=False)  # raises on the first failure
    n = 0
    for test in doctest.DocTestFinder().find(mod):
        runner.run(test)
        n += len(test.examples)
    assert n > 0
