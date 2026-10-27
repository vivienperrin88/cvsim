"""Architecture guards: each check is first shown to catch synthetic violations, then run on cvsim.

A guard that never fires is worthless, so every rule has a "must flag" and a "must accept" case.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import REPO_ROOT
from guards import (
    ALL_CHECKS,
    Module,
    doc_violations,
    imports,
    iter_modules,
    lapack_violations,
    layer_violations,
    qttlind_violations,
    rng_violations,
)

from cvsim._layers import ASIDE, LAYERS, QTTLIND_ALLOWED, SAME_LAYER_FORBIDDEN

SRC = Path(REPO_ROOT) / "src"


def mod(name: str, source: str, is_package: bool = False) -> Module:
    return Module(name, textwrap.dedent(source), is_package)


DOC = '"""Docstring."""\n'


# ------------------------------------------------------------------------------- import resolution
class TestImports:
    def test_relative_imports_resolve_from_a_module(self):
        m = mod("cvsim.gaussian.ops", "from ..core import linalg\nfrom .states import X\nfrom . import y\n")
        assert [t for _, t in imports(m)] == ["cvsim.core.linalg", "cvsim.gaussian.states.X", "cvsim.gaussian.y"]

    def test_relative_imports_resolve_from_a_package(self):
        m = mod("cvsim.gaussian", "from .ops import squeeze\nfrom ..errors import InputError\n", is_package=True)
        assert [t for _, t in imports(m)] == ["cvsim.gaussian.ops.squeeze", "cvsim.errors.InputError"]

    def test_absolute_imports(self):
        m = mod("cvsim.core.x", "import numpy as np\nfrom cvsim.gaussian import ops\n")
        assert [t for _, t in imports(m)] == ["numpy", "cvsim.gaussian.ops"]


# ------------------------------------------------------------------------------- layer rule
class TestLayerRule:
    @pytest.mark.parametrize(
        ("importer", "source"),
        [
            ("cvsim.core.units", "from ..gaussian import ops\n"),  # layer 1 -> 2
            ("cvsim.gaussian.ops", "from cvsim.hardware import sources\n"),  # 2 -> 3
            ("cvsim.hardware.sources", "from ..backends import base\n"),  # 3 -> 4
            ("cvsim.gaussian.ops", "from ..reference import fock_dense\n"),  # oracle is for tests
            ("cvsim.gaussian.ops", "from cvsim import FEATURES\n"),  # package root
            ("cvsim.gaussian.ops", "from .. import _layers\n"),  # private top-level module
            ("cvsim.gaussian.ops", "from ..nosuch import x\n"),  # unregistered target
            ("cvsim.reference.fock_dense", "from ..core import conventions\n"),  # oracle independence
            ("cvsim.errors", "from .core import linalg\n"),
            ("cvsim.hardware.detectors", "from ..characterise import estimators\n"),  # forbidden sibling
            ("cvsim.newpkg.mod", "import numpy\n"),  # unregistered importer
        ],
    )
    def test_flags(self, importer, source):
        assert layer_violations(mod(importer, source))

    @pytest.mark.parametrize(
        ("importer", "source"),
        [
            ("cvsim.gaussian.ops", "from ..core import linalg\nfrom ..errors import InputError\n"),
            ("cvsim.gaussian.ops", "from .states import GaussianState\n"),  # same package
            ("cvsim.fock.states", "from ..lcg import states\n"),  # same layer
            ("cvsim.characterise.estimators", "from ..hardware import detectors\n"),  # allowed sibling
            ("cvsim.protocols.gbs", "from ..backends import base\nfrom ..hardware import sources\n"),
            ("cvsim", "from . import errors\nfrom .gaussian import ops\n"),  # the root sees everything
            ("cvsim.reference.fock_dense", "import numpy as np\nimport scipy.linalg\n"),
            ("cvsim.reference.extra", "from .fock_dense import destroy\n"),  # an aside package's own modules
        ],
    )
    def test_accepts(self, importer, source):
        assert layer_violations(mod(importer, source)) == []

    def test_layer_map_is_consistent(self):
        assert set(LAYERS.values()) == {1, 2, 3, 4}
        assert not set(LAYERS) & ASIDE
        for a, b in SAME_LAYER_FORBIDDEN:
            assert LAYERS[a] == LAYERS[b], (a, b)
        assert set(LAYERS) >= QTTLIND_ALLOWED


class TestQttlindConfinement:
    def test_flags_outside_backends(self):
        assert qttlind_violations(mod("cvsim.gaussian.ops", "import qttlind\n"))
        assert qttlind_violations(mod("cvsim.core.x", "from qttlind.gaussian import ops\n"))
        assert qttlind_violations(mod("cvsim", "import qttlind\n"))

    def test_accepts_in_backends(self):
        assert qttlind_violations(mod("cvsim.backends.qttlind_gaussian", "import qttlind.gaussian\n")) == []

    def test_cvsim_imports_without_qttlind(self):
        """Every module except the qttlind adapters imports in an interpreter where qttlind is absent."""
        code = textwrap.dedent(
            """
            import importlib, pkgutil, sys
            sys.modules["qttlind"] = None  # any `import qttlind` now raises ImportError
            import cvsim
            for info in pkgutil.walk_packages(cvsim.__path__, "cvsim."):
                if "qttlind" not in info.name:
                    importlib.import_module(info.name)
            print("ok")
            """
        )
        env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
        assert res.returncode == 0 and res.stdout.strip() == "ok", res.stderr


# ------------------------------------------------------------------------------- numerics
class TestLapackSeam:
    @pytest.mark.parametrize(
        "source",
        [
            "import numpy as np\nnp.linalg.eigh(a)\n",
            "import numpy\nnumpy.linalg.svd(a)\n",
            "from numpy import linalg\n",
            "from numpy.linalg import eigh\n",
            "import scipy.linalg\n",
            "from scipy import linalg as sla\n",
            "from scipy.linalg import expm\n",
            "import scipy.sparse as sps\nsps.linalg.eigs(a)\n",
            "import scipy\nscipy.linalg.expm(a)\n",
        ],
    )
    def test_flags(self, source):
        assert lapack_violations(mod("cvsim.gaussian.ops", source))

    @pytest.mark.parametrize("name", ["cvsim.core.linalg", "cvsim.reference.fock_dense"])
    def test_allowed_homes(self, name):
        assert lapack_violations(mod(name, "import numpy as np\nimport scipy.linalg\nnp.linalg.eigh(a)\n")) == []

    def test_accepts_plain_numpy(self):
        src = "import numpy as np\nfrom ..core import linalg\nnp.einsum('ij->i', a)\nlinalg.svd(a)\n"
        assert lapack_violations(mod("cvsim.gaussian.ops", src)) == []


class TestRng:
    @pytest.mark.parametrize(
        "source",
        [
            "import numpy as np\nnp.random.seed(0)\n",
            "import numpy as np\nx = np.random.normal(size=3)\n",
            "import numpy\nnumpy.random.rand(3)\n",
            "import numpy.random as npr\nnpr.rand(3)\n",
            "from numpy import random\nrandom.randn(3)\n",
            "from numpy.random import rand\n",
        ],
    )
    def test_flags_legacy_global_state(self, source):
        assert rng_violations(mod("cvsim.gaussian.measure", source))

    def test_accepts_generators(self):
        src = (
            "import numpy as np\nfrom numpy.random import Generator\n"
            "rng = np.random.default_rng(1)\nx = rng.normal(size=3)\ndef f(rng: np.random.Generator): ...\n"
        )
        assert rng_violations(mod("cvsim.gaussian.measure", src)) == []


# ------------------------------------------------------------------------------- documentation
class TestDocs:
    def test_flags_missing_module_docstring(self):
        assert doc_violations(mod("cvsim.core.x", "__all__ = []\n"))

    def test_flags_missing_all(self):
        assert doc_violations(mod("cvsim.core.x", DOC))

    def test_flags_all_listing_unknown_names(self):
        assert doc_violations(mod("cvsim.core.x", DOC + "__all__ = ['ghost']\n"))

    def test_flags_function_without_parameters_section(self):
        src = DOC + "__all__ = ['f']\ndef f(x):\n    '''Do something.'''\n"
        assert doc_violations(mod("cvsim.core.x", src))

    def test_flags_undocumented_public_class_and_function(self):
        assert doc_violations(mod("cvsim.core.x", DOC + "__all__ = ['C']\nclass C:\n    pass\n"))
        assert doc_violations(mod("cvsim.core.x", DOC + "__all__ = ['g']\ndef g():\n    pass\n"))

    def test_accepts_complete_documentation(self):
        src = DOC + textwrap.dedent(
            '''
            from numpy import pi
            __all__ = ["C", "f", "g", "pi"]
            def f(x):
                """Do something.

                Parameters
                ----------
                x
                    A number.
                """
            def g():
                """No arguments, no Parameters section needed."""
            def _private(y):
                pass
            class C:
                """A class."""
            '''
        )
        assert doc_violations(mod("cvsim.core.x", src)) == []

    def test_private_modules_need_only_a_docstring(self):
        assert doc_violations(mod("cvsim.core._helpers", DOC + "def f(x):\n    pass\n")) == []


# ------------------------------------------------------------------------------- the package itself
MODULES = list(iter_modules(SRC))


def test_the_scan_sees_the_package():
    names = {m.name for m in MODULES}
    assert {"cvsim", "cvsim.errors", "cvsim._layers"} <= names


@pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__)
def test_cvsim_passes(check):
    problems = [p for m in MODULES for p in check(m)]
    assert problems == [], "\n".join(problems)
