# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/),
[Semantic Versioning](https://semver.org/)).

## [Unreleased]

### Changed

- Repository: qttlind now lives in `packages/qttlind` of the cvsim monorepo (uv workspace, one
  root `uv.lock`); CI, pre-commit and `.python-version` moved to the repository root. The
  package itself is unchanged.

## [2.0.0] — 2026-10-26

First production release: the complete solver of Moulinas & Waintal (arXiv:2605.27515) and the
continuous-variable extension.

### Added

- **core** — conventions (bit order, shared truncation rule), the `linalg` seam to LAPACK,
  `TensorTrain` (TT-SVD, canonical forms, compression, arithmetic), `MPO` and `apply_mpo`
  (naive, zip-up, variational fitting), environments, Krylov exponentials (Lanczos/Arnoldi with
  adaptive sub-stepping), quantics TCI, general tree tensor networks with per-kind bond caps.
- **bosons** — quantics Fock states (Fock, coherent, cat, TCI-sampled functions up to 62 bits),
  shift, ladder, number, parity and polynomial operators.
- **closed** — time-dependent MPO Hamiltonians, RK4, Crank–Nicolson (two-site ALS + GMRES),
  two-site TDVP, Magnus-TDVP (order 4), `evolve`.
- **lindblad** — the purified comb (`PurifiedComb`: local operators, vector/matrix sums, tensor
  products, μ truncation, observables, reduced density matrices, serialisation) and the first-
  and second-order Kraus integrators with trace-defect monitoring.
- **api** — operator algebra with normal ordering, time-dependent coefficients and custom bases;
  `System`, state specifications, `LindbladSolver` (interruption, atomic checkpoints, bit-for-bit
  resume, cutoff warnings), `Result` (save/load, pandas), `evolve_closed`.
- **gaussian** — exact Gaussian backend: states, gates, channels, homodyne/heterodyne, fidelities,
  log-negativity, Bargmann/Fock recursion, hafnians, loop hafnians, torontonians.
- **cv** — quadrature grids, quantics Fourier transform, position-space operators, phase-space
  gates by shear decomposition, measurements, Fock ↔ position conversion.
- **circuits** — `CVMode` (Fock or position basis), Lindblad noise models, `Circuit`,
  `GaussianExecutor`, `QTTExecutor`, `HybridExecutor` (correlated-block hand-off), many-shot
  sampling, `HardwareModel` (digital twin) and `fit_loss`.
- Fourteen tutorial notebooks (quick mode by default, published sizes with
  `QTTLIND_PAPER_SIZE=1`), Sphinx documentation, CI.
