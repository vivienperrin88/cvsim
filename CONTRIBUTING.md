# Contributing

`main` is always green: every change arrives through a short-lived branch, with its tests. The
rules are qttlind's (`packages/qttlind/CONTRIBUTING.md`), extended to two packages.

## Workflow

```bash
git switch main && git pull
git switch -c feat/gaussian-channels          # <type>/<topic>; one session = one branch
# ... implement, test, commit often ...
uv run ruff check && uv run ruff format --check            # from the root: both packages
(cd packages/cvsim   && uv run mypy && uv run pytest)
(cd packages/qttlind && uv run mypy && uv run pytest)      # when qttlind or the adapter changed
git switch main && git merge --no-ff feat/gaussian-channels
```

## Commits

Small and focused, tests included, [Conventional Commits](https://www.conventionalcommits.org/),
with the package in the scope when it is not obvious:

```
feat(cvsim/gaussian): thermal-loss channel with ledger entry
test(cvsim/contract): squeezing agrees with qttlind.gaussian on random inputs
fix(qttlind/circuits): ...
build: bump numpy
```

Pre-commit hooks (ruff check and format, whitespace, large files) run on every commit; install
them once with `uv run pre-commit install`. Notebooks are committed **with** their outputs.

## Review checklist

- [ ] `uv run pytest` passes in every package touched (and `--runslow` for numerical changes).
- [ ] New public functions have a docstring with Parameters / Returns / Raises.
- [ ] Every invalid input raises the package's `errors` type with a message saying what was
      expected and what was received; there is a test for it.
- [ ] New numerics are checked against an independent reference (dense oracle, closed form, the
      other Gaussian engine, a measured convergence order) — values, not just "it runs".
- [ ] No `numpy.linalg` / `scipy.linalg` call outside the package's `core/linalg.py` on a hot path.
- [ ] A module imports only from its own layer or lower ones (`docs/design/architecture.md`).
- [ ] `ruff`, `ruff format --check` and `mypy` are clean; the docs build with `-W`.
- [ ] The package's `CHANGELOG.md` has an entry under "Unreleased"; `PROGRESS.md` is updated.

## Dependencies

```bash
uv add --package cvsim <package>       # runtime dependency of one package (keep cvsim at NumPy/SciPy)
uv add --dev <package>                 # development tool (root dev group)
uv lock --upgrade-package numpy        # bump one package deliberately
```

Commit `pyproject.toml` and `uv.lock` together; CI installs with `uv sync --locked`.

## Releases

Each package is versioned on its own (semver). Bump `version` in its `pyproject.toml`, move the
"Unreleased" entries of its `CHANGELOG.md`, `uv build --package <name>`, test the wheel in a fresh
environment, then tag `<name>-vX.Y.Z`.
