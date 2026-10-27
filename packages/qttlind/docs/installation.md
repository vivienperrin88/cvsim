# Installation

qttlind is pure Python on NumPy and SciPy (Python 3.12 or newer); nothing is compiled.

## With pip

From a source checkout or a built wheel (`uv build` writes it to `dist/`):

```bash
pip install .                                   # the library, from the repository root
pip install ".[plot,pandas]"                    # plus matplotlib and pandas
pip install dist/qttlind-2.0.0-py3-none-any.whl # or the wheel
```

## From source, with uv (development)

[uv](https://docs.astral.sh/uv/) installs the exact, locked environment — Python included:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
git clone <repository-url> qttlind && cd qttlind
uv sync                                             # .venv with dev + notebook tools
uv run pytest                                       # the test suite (~5 min)
uv run pytest --runslow                             # plus the long-running tests
uv run jupyter lab notebooks/                       # the tutorials
```

Dependency groups keep the library lean: users get NumPy and SciPy only.

| group | for | install |
| --- | --- | --- |
| `dev` (default) | pytest, Hypothesis, ruff, mypy, pre-commit, pandas | `uv sync` |
| `notebooks` (default) | JupyterLab, matplotlib, nbclient | `uv sync` |
| `docs` | Sphinx, furo, MyST-NB | `uv sync --group docs` |
| `bench` | threadpoolctl (BLAS thread control) | `uv sync --group bench` |

## Building this documentation

```bash
uv sync --group docs
uv run sphinx-build -W -b html docs docs/_build/html
```

Notebooks are rendered with their stored outputs. To re-execute them during the build, set
`QTTLIND_DOCS_EXECUTE=force` (about 20 minutes at the default sizes).

## Threads

Dense kernels are LAPACK calls on small matrices: one BLAS thread per process is usually fastest.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```
