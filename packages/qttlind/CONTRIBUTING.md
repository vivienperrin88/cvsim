# Contributing

`main` is always green: every change arrives through a short-lived branch, with its tests.

## Workflow

```bash
git switch main && git pull
git switch -c feat/thermal-kraus            # <type>/<topic>
# ... implement, test, commit often ...
uv run pytest && uv run ruff check && uv run ruff format --check && uv run mypy
git switch main && git merge --no-ff feat/thermal-kraus
```

## Commits

Small and focused: one logical change, tests included, [Conventional
Commits](https://www.conventionalcommits.org/):

```
feat(circuits): exact Kraus operators for thermal loss
fix(linalg): fall back to gesvd when gesdd does not converge
test(api): checkpoint/resume with time-dependent coefficients
docs(guide): stability of the Kraus steps
perf(kraus): one compression per operator application
```

Pre-commit hooks (ruff check and format, whitespace, large files) run on every commit; install
them once with `uv run pre-commit install`. Notebooks are committed **with** their outputs (the
documentation renders them); re-execute them before committing a change that affects them.

## Review checklist

- [ ] `uv run pytest` passes (and `uv run pytest --runslow` for numerical changes).
- [ ] New public functions have a docstring with Parameters / Returns / Raises.
- [ ] Every invalid input raises a `qttlind.errors` type with a message saying what was expected
      and what was received; there is a test for it.
- [ ] New numerics are checked against an independent reference (dense oracle, exact solution,
      Gaussian backend, or a measured order of convergence) — values, not just "it runs".
- [ ] No `numpy.linalg` call outside `qttlind/core/linalg.py` on a hot path.
- [ ] `ruff`, `ruff format --check` and `mypy` are clean; the docs build with `-W`.
- [ ] `CHANGELOG.md` has an entry under "Unreleased".

## Dependencies

```bash
uv add <package>                   # runtime dependency -> [project.dependencies]
uv add --group dev <package>       # development tool
uv lock --upgrade-package numpy    # bump one package deliberately
```

Commit `pyproject.toml` and `uv.lock` together; CI installs with `uv sync --locked`.

## Releases

Semantic versioning. Bump `version` in `pyproject.toml`, move the "Unreleased" entries of
`CHANGELOG.md` under the new version, `uv build`, install the wheel in a fresh environment and run
the tests against it, then `git tag -a vX.Y.Z -m "qttlind X.Y.Z"`.
