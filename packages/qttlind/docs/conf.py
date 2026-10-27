"""Sphinx configuration for qttlind (build: ``uv run --group docs sphinx-build -W -b html docs docs/_build/html``)."""

import os
import shutil
import sys
from importlib.metadata import version as _version

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

project = "qttlind"
author = "Vivien Perrin"
copyright = "2026, Vivien Perrin"
try:
    release = _version("qttlind")
except Exception:  # pragma: no cover - docs built from a bare checkout
    release = "2.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# -- tutorials: the notebooks of ../notebooks are copied in at build time (single source)
_NB_SRC = os.path.join(ROOT, "notebooks")
_NB_DST = os.path.join(HERE, "tutorials", "notebooks")


def _copy_notebooks() -> None:
    if os.path.isdir(_NB_DST):
        shutil.rmtree(_NB_DST)
    os.makedirs(_NB_DST)
    for name in sorted(os.listdir(_NB_SRC)):
        if name.endswith(".ipynb"):
            shutil.copy2(os.path.join(_NB_SRC, name), _NB_DST)
    figs = os.path.join(_NB_SRC, "figures")
    if os.path.isdir(figs):
        shutil.copytree(figs, os.path.join(_NB_DST, "figures"))


_copy_notebooks()

nb_execution_mode = os.environ.get("QTTLIND_DOCS_EXECUTE", "off")  # notebooks ship with their outputs
nb_execution_timeout = 3600
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "ignore-module-all": True,  # document objects where they are defined, not where re-exported
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# QTTLIND_DOCS_OFFLINE=1 skips the intersphinx inventories (builds without network access)
if os.environ.get("QTTLIND_DOCS_OFFLINE") == "1":
    extensions.remove("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["mystnb.unknown_mime_type", "myst.header"]
nitpicky = False

html_theme = "furo"
html_title = f"qttlind {release}"
html_static_path = ["_static"]
