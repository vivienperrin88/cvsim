# Installation

cvsim is pure Python on NumPy and SciPy (Python 3.12 or newer); nothing is compiled. It lives in
`packages/cvsim` of a uv workspace that also contains qttlind.

## Development (uv, from the repository root)

```bash
uv sync                          # both packages (editable) + dev and notebook tools, locked
cd packages/cvsim
uv run pytest                    # the test suite
uv run pytest --runslow          # plus the long-running tests
uv run mypy && uv run ruff check
```

| group | for | install |
| --- | --- | --- |
| `dev` (default) | pytest, Hypothesis, ruff, mypy, pre-commit, pandas, qttlind | `uv sync` |
| `notebooks` (default) | JupyterLab, matplotlib, nbclient | `uv sync` |
| `docs` | Sphinx, furo, MyST-NB | `uv sync --group docs` |
| `bench` | threadpoolctl | `uv sync --group bench` |

## Library only

```bash
uv build --package cvsim                 # writes dist/cvsim-<version>-py3-none-any.whl
pip install "dist/cvsim-<version>-py3-none-any.whl"            # NumPy + SciPy
pip install "cvsim[qttlind] @ file://$PWD/dist/cvsim-<version>-py3-none-any.whl"  # with the backend
```

cvsim is private: its metadata carries `Private :: Do Not Upload`, so PyPI rejects it.

## Building this documentation

```bash
uv sync --group docs
cd packages/cvsim && uv run sphinx-build -W -b html docs docs/_build/html
```
