# Roadmap: sessions S00–S38

Approved plan. Each session is one branch with its tests, a changelog entry and an update of
`PROGRESS.md`; it is merged when every gate is green and reviewed by the project owner before the
next one starts. **⏸** marks a planned mid-session check-in. Contract tests compare cvsim with
qttlind (marker `qttlind`); the reference oracle is `cvsim.reference`. **From S07 on, every session
also compares its Gaussian and CV features with The Walrus and MrMustard, the reference baselines
(markers `thewalrus`, `mrmustard`), and updates `docs/design/equivalence.md`.**

## M0 — Foundations

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S00 | monorepo (qttlind with history), cvsim scaffold, CI, docs, decision records D1–D8 | all gates; clean-wheel test without qttlind | end |
| S01 | `errors` (`CvsimError`, `InputError`, `ShapeError`, `PhysicalityError`, `CalibrationError`, `ConvergenceError`; warnings), `FEATURES` / `require`, layer map | hierarchy tests; **guard tests**: layer rule (AST), LAPACK seam, qttlind confined to `backends`, `cvsim` importable without qttlind, module docstrings and `__all__`, numpydoc *Parameters* sections; each guard tested on synthetic violations | end |
| S02 | `core.conventions` (Ω, index maps xxpp↔xpxp, ħ=1↔2, SNU, dB↔r, gate symplectics from generators); `reference.fock_dense` oracle; contract-test harness | [tested] rules twice; Hypothesis round trips; first contract tests (R, S, BS, S2 symplectics vs qttlind) | ⏸ sign-convention table |

## M1 — Gaussian core

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S03 | `core.linalg`: Williamson, Takagi (degenerate-safe), Bloch–Messiah, symplectic eigenvalues, `slogdet`, PSD `sqrtm`, polar, nearest physical covariance, `timed()` | reconstruction to 1e-12; random symplectics `expm(ΩH)`; degenerate spectra; ill-conditioning | ⏸ gauge choice for degenerate subspaces |
| S04 | `GaussianState`: constructors, reduce, purity, entropy, photon moments, ν_k, mixed fidelity (Banchi 2015), overlap, log-negativity, Wigner, moments | oracle; contract vs `qttlind.gaussian.states/measure` on Hypothesis states | end |
| S05 | symplectic library (R, S, D, BS, S2, CZ, CX, P, F, `interferometer`, `from_hamiltonian`); Clements; Bloch–Messiah preparation; supermode basis change | symplecticity; Clements reconstruction; `expm` oracle; contract vs `qttlind.gaussian.ops` | end |
| S06 | `GaussianChannel(X, Y)` (CP check, composition); loss, amplifier, additive noise, random displacement, ancilla mixing, moment-matched phase diffusion; `NoiseLedger` | ledger total = exact Y; CP violations caught; Kraus oracle; contract | ⏸ ledger attribution convention |
| S07 | multimode homodyne (any angle, any LO mode), heterodyne, general-dyne, conditioning, sampling, efficiency, electronic noise; **full Gaussian contract suite** | Schur complements vs oracle; equivalence report cvsim ↔ qttlind | end of M1 |

## M2 — IR and backends

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S08 | `Mode` registry, `Op` kinds, immutable `Program`, `Parameter` binding, JSON | round trips; validation errors | end |
| S09 | `Backend` protocol, `Result`; `GaussianBackend(engine="cvsim"\|"qttlind")`; dense `FockBackend`; dispatcher | same Program on three engines; seeds; refusals | end |
| S10 | permanent (Glynn), hafnian, loop hafnian, torontonian (independent); PNR/click; detector models; heralded Fock states | brute-force Fock; contract vs `qttlind.gaussian.hafnian` | end |

## M3 — Hardware

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S11 | sources: JSA (pump × phase matching) → Schmidt supermodes; SPOPO (cavity, threshold, escape efficiency, squeezing spectrum); single-pass waveguide with distributed loss; excess antisqueezing | analytic limits (Gaussian-JSA Schmidt number, OPO formula, CW limit) | ⏸ source models and parameter ranges |
| S12 | `ModeBasis` (HG, pixel/band, supermode, time-bin), overlaps, unitary completion, pulse shaper (resolution, pixelation, errors) → LO mode and mismatch | orthonormality, unitarity, visibility closed form | end |
| S13 | homodyne detector model (QE, visibility, clearance, CMRR, bandwidth crosstalk, ADC), lock-noise spectrum → σ_φ | closed forms; crosstalk map | end |
| S14 | `Setup` dataclasses, TOML I/O, `compile(setup, protocol)` with provenance, budget reports | round trips; example setups; budget sums | end |
| S15 | R1 (Roslund 2014), R3 (Kouadou 2023) | values within the papers' uncertainties | ⏸ reference values |

## M4 — Networks, MBQC, routing

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S16 | graph models, clusters from supermodes, nullifiers, van Loock–Furusawa witnesses | ideal limit; U-vs-CZ equivalence | end |
| S17 | teleportation gates, SL(2) compiler, multimode gates, feed-forward latency and EOM errors | ideal limit; noise per step | end |
| S18 | entanglement routing (measurement-basis optimisation) | small analytic networks | end |
| S19 | R2, R6, R10 | values vs papers | ⏸ reference values |

## M5 — Non-Gaussian

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S20 | Fock states/DMs with cutoff policy; Gaussian→Fock recursion; LCoG (log-weights); exact phase noise | oracle; contract vs `qttlind.gaussian.fock` | end |
| S21 | native MPS (canonical forms, gates, qttlind truncation rule) | vs dense | end |
| S22 | mode-selective subtraction/addition hardware → heralded state and rate | ideal limit; Ra 2020 | ⏸ selectivity model |
| S23 | cat breeding, GKP (LCoG), stellar rank, negativity, witnesses | analytic cases | end |
| S24 | R5, R11 | | ⏸ reference values |

## M6 — GBS, reservoir computing, QML

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S25 | noisy GBS probabilities, chain-rule sampler, validation statistics | brute force; contract | end |
| S26 | Gaussian reservoir (encodings, memory, finite shots), ridge, capacities, comb reservoir | capacity bounds; seeds | end |
| S27 | cluster/teleportation QML layers, moment gradients | finite differences | end |
| S28 | R7, R8 | | ⏸ reference values |

## M7 — Lindblad and the qttlind comb

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S29 | `LindbladSpec`, Gaussian Lyapunov, OPO input-output, dense solver | OPO spectrum | end |
| S30 | MPS quantum trajectories | statistical convergence | end |
| S31 | qttlind adapter (Program → circuits with Clements workaround; LindbladSpec → `LindbladSolver`) | 4 engines agree | ⏸ adapter API |
| S32 | R12 | | end |

## M8 — Characterisation

| S | Deliverables | Verification | Check-in |
| --- | --- | --- | --- |
| S33 | covariance from homodyne records, shot-noise calibration, bootstrap, projection, supermode retrieval | unbiased; coverage | end |
| S34 | Gaussian MLE, witnesses with error bars, Fock MaxLik, negativity witness | synthetic recovery | end |
| S35 | R4, R9 | | ⏸ reference values |

## M9 — Release

| S | Deliverables |
| --- | --- |
| S36 | user guide and tutorials |
| S37 | benchmarks (1000 temporal modes, banded covariances), profiling |
| S38 | API freeze, v1.0.0, wheel tests, tag |
