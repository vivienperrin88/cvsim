"""Package surface: metadata and documentation. (Errors and feature flags arrive in S01.)"""

import cvsim


def test_version_is_a_string():
    assert isinstance(cvsim.__version__, str) and cvsim.__version__


def test_package_is_documented():
    assert cvsim.__doc__ and "qttlind" in cvsim.__doc__
