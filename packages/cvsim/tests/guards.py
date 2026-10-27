"""Static checks of the code base, by parsing (never importing) its modules.

Each check takes a module's dotted name and source text and returns a list of violations, so it
can be tested on synthetic snippets (``tests/test_guards.py``) as well as run on the package.

* :func:`layer_violations` -- the dependency rule of ``cvsim._layers``;
* :func:`qttlind_violations` -- only ``cvsim.backends`` may import the optional qttlind;
* :func:`lapack_violations` -- dense linear algebra goes through ``cvsim.core.linalg`` (D3);
* :func:`rng_violations` -- randomness through ``numpy.random.Generator`` only, never the legacy
  global state (seeded, reproducible runs);
* :func:`doc_violations` -- module docstrings, ``__all__`` that lists real names, docstrings with
  a numpydoc *Parameters* section for public functions that take arguments.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from cvsim._layers import ASIDE, LAYERS, QTTLIND_ALLOWED, SAME_LAYER_FORBIDDEN

PACKAGE = "cvsim"
LINALG_HOME = "cvsim.core.linalg"
LINALG_MODULES = frozenset({"numpy.linalg", "scipy.linalg", "scipy.sparse.linalg"})
RNG_ALLOWED = frozenset({"default_rng", "Generator", "SeedSequence", "BitGenerator", "PCG64", "PCG64DXSM"})
_PARAMETERS = re.compile(r"^\s*Parameters\s*\n\s*-{3,}\s*$", re.M)


@dataclass(frozen=True)
class Module:
    name: str  # dotted, e.g. "cvsim.gaussian.ops"
    source: str
    is_package: bool  # an __init__.py

    @property
    def tree(self) -> ast.Module:
        return ast.parse(self.source, filename=self.name)


def iter_modules(src_root: Path) -> Iterator[Module]:
    """Every module under ``src_root/cvsim``."""
    for path in sorted((src_root / PACKAGE).rglob("*.py")):
        parts = path.relative_to(src_root).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        yield Module(".".join(parts), path.read_text(encoding="utf-8"), is_package)


def subpackage(name: str) -> str | None:
    """``"cvsim.gaussian.ops"`` -> ``"gaussian"``; ``None`` for the top-level package itself."""
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else None


# ------------------------------------------------------------------------------- imports
def imports(module: Module) -> Iterator[tuple[int, str]]:
    """``(line, absolute dotted name)`` of everything ``module`` imports.

    ``from X import a`` yields ``X.a`` (``a`` may be a submodule; for the rules below only the
    first two components matter, and ``X.a`` has the same ones as ``X`` unless ``X`` is the
    top-level package).
    """
    package = module.name if module.is_package else module.name.rpartition(".")[0]
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts) + (f".{node.module}" if node.module else "")
            else:
                base = node.module or ""
            for alias in node.names:
                yield node.lineno, f"{base}.{alias.name}" if alias.name != "*" else base


def layer_violations(module: Module) -> list[str]:
    """Imports that break the layer rule of :mod:`cvsim._layers`."""
    out: list[str] = []
    own = subpackage(module.name)
    if own is not None and own not in LAYERS and own not in ASIDE and not own.startswith("_"):
        out.append(f"{module.name}: subpackage {own!r} is not registered in cvsim._layers")
        return out
    for line, target in imports(module):
        if not (target == PACKAGE or target.startswith(PACKAGE + ".")):
            continue
        dep = subpackage(target)
        where = f"{module.name}:{line} imports {target}"
        if own is None or own.startswith("_"):  # top-level package and private top-level modules
            continue
        if own in ASIDE:
            if dep != own:  # its own modules are fine
                out.append(f"{where}: {own!r} must not import anything from cvsim")
        elif dep is None or dep.startswith("_"):
            out.append(f"{where}: subpackages import siblings or errors, never the package root or its private modules")
        elif dep == "errors":
            continue
        elif dep == "reference":
            out.append(f"{where}: the reference oracles are for tests only")
        elif dep not in LAYERS:
            out.append(f"{where}: {dep!r} is not a registered subpackage")
        elif LAYERS[dep] > LAYERS[own]:
            out.append(f"{where}: layer {LAYERS[own]} ({own}) must not import layer {LAYERS[dep]} ({dep})")
        elif (own, dep) in SAME_LAYER_FORBIDDEN:
            out.append(f"{where}: {own!r} must not import its sibling {dep!r}")
    return out


def qttlind_violations(module: Module) -> list[str]:
    """Imports of qttlind outside the subpackages allowed to use the optional backend."""
    own = subpackage(module.name)
    if own in QTTLIND_ALLOWED:
        return []
    return [
        f"{module.name}:{line} imports {target}: only {sorted(QTTLIND_ALLOWED)} may import qttlind"
        for line, target in imports(module)
        if target == "qttlind" or target.startswith("qttlind.")
    ]


# ------------------------------------------------------------------------------- numerics
def _aliases(tree: ast.Module, modules: set[str]) -> set[str]:
    """Local names bound to any of ``modules`` (``import numpy as np`` -> ``np``)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    names.add(alias.asname or alias.name.split(".")[0])
    return names


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def lapack_violations(module: Module) -> list[str]:
    """Uses of ``numpy.linalg`` / ``scipy.linalg`` outside ``cvsim.core.linalg`` and the oracles."""
    if module.name == LINALG_HOME or subpackage(module.name) == "reference":
        return []
    tree = module.tree
    out: list[str] = []
    for line, target in imports(module):
        if any(target == m or target.startswith(m + ".") for m in LINALG_MODULES):
            out.append(f"{module.name}:{line} imports {target}: use cvsim.core.linalg")
    np_names = _aliases(tree, {"numpy", "scipy", "scipy.sparse"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "linalg":
            dotted = _dotted(node)
            if dotted is not None and dotted.split(".")[0] in np_names:
                out.append(f"{module.name}:{node.lineno} uses {dotted}: use cvsim.core.linalg")
    return out


def rng_violations(module: Module) -> list[str]:
    """Uses of NumPy's legacy global random state (``np.random.rand``, ``np.random.seed``, ...)."""
    if subpackage(module.name) == "reference":
        return []
    out: list[str] = []
    np_names = _aliases(module.tree, {"numpy", "numpy.random"})
    # `import numpy.random as npr` binds the submodule itself
    random_names = {
        alias.asname
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "numpy.random" and alias.asname
    }
    random_names |= {
        alias.asname or alias.name
        for node in ast.walk(module.tree)
        if isinstance(node, ast.ImportFrom) and node.module == "numpy"
        for alias in node.names
        if alias.name == "random"
    }
    np_names -= random_names
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted is None:
                continue
            parts = dotted.split(".")
            if (len(parts) == 2 and parts[0] in random_names and parts[1] not in RNG_ALLOWED) or (
                len(parts) == 3 and parts[0] in np_names and parts[1] == "random" and parts[2] not in RNG_ALLOWED
            ):
                out.append(f"{module.name}:{node.lineno} uses {dotted}: pass a numpy.random.Generator instead")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy.random":
            for alias in node.names:
                if alias.name not in RNG_ALLOWED:
                    out.append(f"{module.name}:{node.lineno} imports numpy.random.{alias.name}: use a Generator")
    return out


# ------------------------------------------------------------------------------- documentation
def _defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _declared_all(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "__all__":
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple)) and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts
                ):
                    return [e.value for e in value.elts]  # type: ignore[attr-defined]
                return []
    return None


def _takes_arguments(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    a = fn.args
    return bool(a.posonlyargs or a.args or a.kwonlyargs or a.vararg or a.kwarg)


def doc_violations(module: Module) -> list[str]:
    """Missing module docstrings, ``__all__`` problems, undocumented public functions and classes."""
    tree = module.tree
    out: list[str] = []
    if not ast.get_docstring(tree):
        out.append(f"{module.name}: module docstring missing")
    if any(part.startswith("_") for part in module.name.split(".")[1:]):
        return out  # private module: docstring only
    declared = _declared_all(tree)
    if declared is None:
        out.append(f"{module.name}: public module without __all__")
        return out
    defined = _defined_names(tree)
    for name in declared:
        if name not in defined and name != "__version__":
            out.append(f"{module.name}: __all__ lists {name!r}, which the module does not define or import")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in declared:
            doc = ast.get_docstring(node)
            if not doc:
                out.append(f"{module.name}.{node.name}: public function without a docstring")
            elif _takes_arguments(node) and not _PARAMETERS.search(doc):
                out.append(f"{module.name}.{node.name}: docstring has no numpydoc 'Parameters' section")
        elif isinstance(node, ast.ClassDef) and node.name in declared and not ast.get_docstring(node):
            out.append(f"{module.name}.{node.name}: public class without a docstring")
    return out


ALL_CHECKS = (layer_violations, qttlind_violations, lapack_violations, rng_violations, doc_violations)
