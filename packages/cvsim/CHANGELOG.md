# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/),
[Semantic Versioning](https://semver.org/)).

## [Unreleased]

### Added

- **S01** — `cvsim.errors` (`CvsimError`; `InputError`, `ShapeError`, `PhysicalityError`,
  `CalibrationError`, `ConvergenceError`; `CvsimWarning`, `ApproximationWarning`, `CutoffWarning`),
  feature flags `FEATURES` / `KNOWN_FEATURES` / `require`, the layer map `cvsim._layers`, and
  guard tests that parse every module: layer rule, qttlind confined to `backends` (and cvsim
  importable without it), LAPACK seam, no legacy global RNG, docstrings / `__all__` / numpydoc
  *Parameters*. cvsim warnings fail the tests unless expected.
- **S00** — package scaffold in the monorepo: `pyproject.toml` (setuptools, Python ≥ 3.12,
  NumPy/SciPy only, optional `qttlind` extra, private classifiers), proprietary licence, test
  configuration (`slow`, `qttlind`, `repro` markers; seeded `rng`), Sphinx documentation skeleton,
  architecture decision records D1–D8, notebook and doc-example runners.
