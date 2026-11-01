# Changelog

All notable changes are documented here ([Keep a Changelog](https://keepachangelog.com/),
[Semantic Versioning](https://semver.org/)).

## [Unreleased]

### Added

- **S11** — sources (feature `sources`; subpackage `cvsim.hardware`, layer 3). `cvsim.core.units`:
  SI scale factors and laboratory conversions (wavelength ↔ ω, Δλ ↔ Δω, Hz ↔ rad/s, dB/cm ↔ κ,
  intensity FWHM ↔ amplitude σ). `cvsim.hardware.jsa`: `FrequencyGrid` (`centered`, `auto`),
  `Pump` (`gaussian` with chirp, `sampled`), `PhaseMatching` (`sinc`, `gaussian`; GVM and GVD
  parameters, no Sellmeier data), `joint_spectral_amplitude`, `schmidt_decomposition` →
  `Supermodes` (Schmidt coefficients and number, mode functions, truncation, pixel coupling `Z`);
  degenerate type-0/I only, type-II refused. `cvsim.hardware.sources`: `SPOPO` (`from_cavity`:
  repetition rate, output coupler, loss; threshold, escape efficiency, linewidth; `spectrum`,
  `output` at an analysis frequency), `Waveguide` (distributed loss solved exactly per supermode,
  output coupling, `effective_transmissivity`), `SourceOutput` (state, `NoiseLedger`,
  `quadrature_db`, `in_pixel_basis`). `cvsim.hardware.excess`: `ExcessNoise` (phase-sensitive
  excess antisqueezing, gain fluctuations with exact second moments), `calibrate` (squeezing /
  antisqueezing in dB → r, η, excess). Oracle `cvsim.reference.sources` (input–output theory of
  the OPO, pixel-basis squeezer, Van Loan solution of the lossy Lyapunov equation). Contract tests
  of the emitted states against qttlind, The Walrus and MrMustard (waveguide as Strang slices).
- **S10** — photon counting (feature `photon_counting`). `cvsim.core.hafnian`: `permanent`
  (Glynn, Gray code), `hafnian` (power trace, or subset recursion), `loop_hafnian`, `torontonian`.
  `cvsim.gaussian.photon`: `husimi_parameters`, `pattern_probability`, `fock_density_matrix`
  (loop-hafnian recurrence), `pnr_distribution` (efficiency, Poissonian dark counts),
  `click_probability` / `click_distribution` (efficiency, dark clicks), `herald` (heralded
  non-Gaussian states with realistic detectors). IR measurements `PNR` and `Threshold`, run by the
  Fock backend (sampled or post-selected). Brute-force and closed-form tests; contract tests against
  `qttlind.gaussian.hafnian` / `.fock`, The Walrus (`hafnian`, `perm`, `tor`, `probabilities`,
  `density_matrix`, `threshold_detection_prob`) and MrMustard's `PNRDetector` and
  `ThresholdDetector` (bridge on MrMustard's TensorFlow backend). One MrMustard error pinned as a
  strict xfail: its threshold dark-click kernel.
- **S09** — `cvsim.backends` (feature `backends`): `Backend` protocol, `Result` (state, register
  modes, outcomes with densities, noise ledger), `GaussianBackend(engine="cvsim" | "qttlind")`
  (the D2 switch; cvsim's engine books every channel in a `NoiseLedger`), dense `FockBackend`
  (`FockState`; every IR operation including Fock states, Kerr and exact phase noise; Clements and
  Bloch–Messiah for interferometers and general Gaussian unitaries; Kraus forms of every
  phase-insensitive channel; homodyne sampling; `CutoffWarning`), dispatcher `select_backend` /
  `run`. Same programs agree on the three engines; against MrMustard (lab circuits, its Fock
  conversion and Fock simulation) and The Walrus (`density_matrix`). `NoiseLedger.gate` accepts a
  displacement (`shift`).
- **S08** — `cvsim.ir` (feature `ir`): `Op` validated against the registry `SPECS` (preparations
  Vacuum, Coherent, Squeezed, Thermal, Fock; gates D, R, S, P, F, BS, S2, CZ, CX, Interferometer,
  Symplectic, Kerr; channels Loss, Amplifier, AdditiveNoise, PhaseNoise; measurements Homodyne,
  Heterodyne), with noise-ledger provenance; `Parameter` placeholders; immutable `Program`
  (named `Mode` register, `>>` composition in time order, `bind`, measured-mode tracking and
  reuse checks, Gaussianity) and its versioned JSON form (`cvsim.ir/1`). Hypothesis round trips.
- **S07** — `cvsim.gaussian.measure`: `general_dyne` (any Gaussian measurement `y = L r_A + ξ`),
  `homodyne` (any angle, one or several modes jointly), `heterodyne` (outcome `α`, density per
  `d²α`), `homodyne_lo` (local oscillator in any mode function), `homodyne_density`,
  `sample_homodyne`; detector efficiency and electronic noise (`homodyne_noise`). Tested against the
  Fock oracle (conditional states, Husimi densities), against explicit optical models of the
  detector (loss + noise + ideal homodyne; the eight-port heterodyne), and against
  `qttlind.gaussian.measure`. Features `gaussian`, `channels` and `gaussian_measure` are now listed
  in `cvsim.FEATURES`.
- **Reference baselines (S07)** — The Walrus and MrMustard are compared with every Gaussian / CV
  feature from now on. MrMustard 0.7.3 runs in its own environment
  (`tools/setup_reference_envs.sh`, git-ignored `.reference-envs/`) behind a JSON-lines bridge
  (`tests/reference/mrmustard_bridge.py`), marker `mrmustard`; new contract files
  `test_contract_mrmustard.py` (gates, states, purity, mixed fidelity, photon numbers, entropy,
  log-negativity, channels, CPTP application, composition, homodyne and heterodyne conditioning)
  and `test_contract_thewalrus_gaussian.py` (gates, `expand`, states, fidelity, entropy, photon
  moments, vacuum element, reduction, validity, log-negativity, loss). Four MrMustard errors were
  found and pinned as strict xfails (multimode symplectic eigenvalues and hence entropy and
  log-negativity, the photon-number covariance of displaced correlated modes, channel
  composition), plus documented differences (two-mode squeezing phase, units, finite-squeezing
  homodyne, The Walrus' `reduced_gaussian` ordering). `docs/design/equivalence.md` is now the
  four-way report (qttlind, The Walrus, MrMustard, oracle); CI job `reference-baselines` runs
  these comparisons with no skips allowed.
- **S06** — `cvsim.gaussian.GaussianChannel` `(X, Y, c)`: complete-positivity check, `then`,
  `tensor`, `embed`, `apply`; `loss`, `amplifier`, `additive_noise`, `from_transfer` (lossy linear
  optics), `from_ancilla` (gate with an ancilla, traced out), `from_symplectic`, `identity`.
  `channels.phase_diffusion` (moment-matched, `ApproximationWarning`). **`NoiseLedger`**: exact
  decomposition of the output covariance into the signal and each component's noise carried
  forward to the output, by component or kind; transmissivities; quadrature budgets in shot-noise
  units; `db_recovered` what-if view (non-additive, labelled). Tested against the Fock oracle (loss
  Kraus map, amplifier as a traced two-mode squeezer, exact dephasing), closed forms, Hypothesis
  chains (ledger sum and direct application agree to 1e-11), qttlind (`loss`, `amplifier`,
  `thermal_noise`, `check_channel`) and The Walrus (`loss`, `passive_transformation`).
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

### Fixed

- `core.linalg.nearest_physical_covariance` returned a projected matrix for a physical input whose
  pure mode came out of Williamson as `ν = ½ − 3e-14` (LAPACK round-off; seen with the locked NumPy
  2.5 / SciPy 1.18). Eigenvalues within `atol · max(1, max|V|)` of the floor (new argument, default
  `1e-12`) now count as physical.
  A negative or non-finite `atol` now raises `InputError`.
- S11 review (`fix/S11-review`):
  - `hardware.jsa.Pump.sampled` set `half_width` to the extent of the samples instead of the HWHM
    of `|α|²`, so `FrequencyGrid.auto` gave a narrow sampled pump a grid that did not resolve it,
    silently. The half-maximum crossings are now solved on the linear interpolation; `half_width`
    and the new `center` span every peak above half maximum (the grid's span is shifted by
    `|center|/2`, so an off-centre pump stays on it).
  - `Pump` has a `resolution_width` (the narrowest peak's HWHM, shortened by a turning phase:
    `1/max|dφ/dΩ₊|` over the peaks and their crossings; `min(h, 1/(|c| h))` for a chirp `c`), and
    `FrequencyGrid.auto` sizes its step from the main-lobe scale of the JSA (pump, phase matching
    along both diagonals).
  - `PhaseMatching.half_width_sum` took `min(π/|τ|, √(2π/|b|))`, which overestimates the first
    zero when GVM and the quadratic term are both present; it is now the exact closest zero
    `2π/(|τ| + √(τ² + 2π|b|))`. Its docstring stated the wrong level for the Gaussian shape
    (`exp(−γπ²) ≈ 0.149`, not `1/e`).
  - `hardware.excess.calibrate(efficiency=None)` returned a negative `r` and efficiency for data in
    the `rtol` band of a pure state, divided by zero for very weak squeezing, and lost precision
    for weak squeezing (relative error 5e-5 at `r = 1e-6`). Both branches now form the differences
    from the dB values without cancellation; the band is read as a pure state (`η = 1`,
    `r = ¼ ln(V₊/V₋)`), and a squeezing too weak to fit in double precision raises
    `CalibrationError`. `rtol` is validated.
  - `Supermodes` and `SourceOutput` compare and hash by identity: the generated `__eq__` compared
    arrays and raised, which also broke `==` and `hash` on `SPOPO` and `Waveguide`.
  - `FrequencyGrid.auto` validates `points_per_width` and `max_points`; every scalar validation of
    `core.conventions`, `core.units`, `core.linalg` and `hardware` goes through `core._checks`,
    so non-numeric input (also NumPy booleans and complex numbers) raises `InputError` instead of a
    bare `ValueError` or a silent truncation; sizes accept NumPy integers and 0-d arrays. `Waveguide.output`
    averages both propagation-noise components in one Gauss–Hermite pass (`gain_average` accepts
    an output with a leading axis), and the per-mode loss is built from `conventions.loss_xy`.
