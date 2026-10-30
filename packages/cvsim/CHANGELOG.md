# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/),
[Semantic Versioning](https://semver.org/)).

## [Unreleased]

### Added

- **S05** — `cvsim.gaussian.symplectic`: the gate matrices of the conventions (re-exported),
  `from_hamiltonian` (`exp(−itH)` → `expm(tΩH)`), `is_passive`, `unitary_from_passive`,
  `random_interferometer` (Haar), `random_symplectic`; **Clements mesh** `clements(U)` (rectangular,
  `N(N−1)/2` cells of `BS(θ, 0)·R(φ)`, depth `N`, output phases; `gates()` in time order);
  **state preparation** `preparation(state)` (thermal → `I(U₂)` → squeezers → `I(U₁)` →
  displacement); **mode-basis change** `change_basis(state, U)` (supermodes, pulse shapes).
  `cvsim.gaussian.ops`: `apply_symplectic`, `displace`, `rotate`, `squeeze`, `beamsplitter`,
  `two_mode_squeeze`, `interferometer`, `cz`, `cx`, `shear`, `fourier`. Tested against the Fock
  oracle (Heisenberg map of `exp(−itĤ)`, gate sequences on density matrices), on Haar-random and
  degenerate unitaries (identity, permutations, phases, DFT), and against `qttlind.gaussian.ops`
  (3 contract tests, including cvsim's Clements mesh executed with qttlind's gates).
- **S04** — `cvsim.gaussian.GaussianState`: an immutable mean and covariance, validated against
  the uncertainty principle (`PhysicalityError`). Constructors `vacuum`, `coherent`, `thermal`,
  `squeezed` (multimode), `two_mode_squeezed`, `from_symplectic(S, nu, mean)`, `product`,
  `from_hbar`, `from_xpxp`; `reduce`, `tensor`, `to_hbar`, `to_xpxp`, `isclose`; symplectic
  eigenvalues, purity, entropy, mean photons, photon covariance, complex moments; overlap and
  general **mixed-state fidelity** (pure Williamson modes projected out exactly, then Banchi,
  Braunstein, Pirandola 2015 on eigenvalues, no matrix square root: 1e-13–1e-11 relative, also
  with pure modes entangled with mixed ones); log-negativity; multimode Wigner function and single-mode
  grids; vacuum probability. `core.linalg.eigvals` (general eigenvalues). Tested against the
  Fock oracle (density matrices built from generators and Kraus maps), against closed forms
  (thermal and Scutaru fidelities, products) on Hypothesis states, and against
  `qttlind.gaussian.states/measure` (7 contract tests).
- **S03b** — comparison with The Walrus, Strawberry Fields and MrMustard (CONVENTIONS.md,
  "Comparison with other libraries"): result layout, ordering, block order and gauge of Williamson,
  Bloch–Messiah, Takagi and the symplectic eigenvalues, side by side. Optional extra `compare`
  (The Walrus 0.22) and marker `thewalrus`: 10 contract tests compute the matrix relating each
  cvsim factor to The Walrus's and require exactly the expected form (ordering, the π/2 block swap
  of Bloch–Messiah, a gauge rotation inside degenerate groups). Two new tests of cvsim's gauge:
  invariance under ħ rescaling and covariance under mode relabelling.
- **S03** — `cvsim.core.linalg` (feature `linalg`), the only module calling LAPACK: `eigh`,
  `eigvalsh`, `svd` (gesdd → gesvd fallback), `solve`, `inv`, `cholesky`, `logdet_pd`, `sqrtm_psd`,
  `expm`, `qr`, `polar`, `timed()` statistics per kernel; `symplectic_eigenvalues`, `williamson`,
  `bloch_messiah` (supermodes; orthogonal symplectic factors to round-off), `takagi`,
  `degenerate_groups`, `nearest_physical_covariance`. Degenerate groups are reported and their gauge
  is fixed by orthogonal Procrustes against a reference basis (`reference=`), so results are
  reproducible and sweeps continuous. Tested on Hypothesis inputs incl. degenerate, rank-deficient
  and ill-conditioned spectra; contract test of ν against qttlind.
- **S02** — `cvsim.core.conventions` (feature `conventions`): Ω, mode indices, xxpp↔xpxp,
  complex basis, ħ / shot-noise / dB conversions, `symplectic_from_bogoliubov`, `is_symplectic`,
  `embed`, the gates D, R, S, BS, S2, interferometer, CZ(A), CX, P, F (= R(π/2)), the homodyne
  vector and the loss channel `(X, Y)`. `cvsim.reference.fock_dense`: an independent
  truncated-Fock oracle that measures symplectic matrices from the generators (dense, or with
  `expm_multiply` for three modes) and rejects non-Gaussian gates. Every rule is tested twice
  (conventions vs oracle, Hypothesis) and against qttlind (`tests/contract`, 19 contract tests).
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
