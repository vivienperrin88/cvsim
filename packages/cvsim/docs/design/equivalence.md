# Equivalence report: cvsim against qttlind, The Walrus and MrMustard

Updated every session (CONTRIBUTING.md, "Reference baselines"); state after S11.

cvsim is compared with three independent implementations:

* **qttlind** (same repository, marker `qttlind`): the second Gaussian engine of decision D2,
  same conventions, number for number.
* **The Walrus 0.22** (marker `thewalrus`, extra `compare`, ħ = 2): the reference for Gaussian
  quantum optics; also the basis of Strawberry Fields.
* **MrMustard 0.7.3** (marker `mrmustard`, own environment built by
  `tools/setup_reference_envs.sh`, ħ = 2): reached through `tests/reference/mrmustard_bridge.py`,
  because it pins NumPy < 2.

When two of them disagree, the Fock oracle (`cvsim.reference`) or a closed form decides. A
confirmed upstream error is kept as an `xfail(strict=True)` test naming it, so that a fix upstream
is noticed. Tolerances: `EXACT` = 1e-14 (closed-form matrices); `ROUNDOFF` = 1e-12 relative to the
largest covariance entry; densities and figures of merit 1e-8 to 1e-11 relative, as stated in
each test. Inputs are hand-picked cases plus Hypothesis-generated states on 1–4 modes, pure and
mixed, squeezed up to `r = 1` per mode.

## Convention map

| Quantity | cvsim (ħ = 1) | The Walrus, MrMustard (ħ = 2) |
| --- | --- | --- |
| Moments | `d`, `V`; vacuum `V = I/2` | `√2 d`, `2V`; vacuum `V = I` |
| Ordering, `Ω` | xxpp, `[[0, I], [−I, 0]]` | same |
| `R(θ)`, `S(r, φ)`, `BS(θ, φ)`, `P(s)`, `CZ`, `CX`, interferometer | as in CONVENTIONS.md | identical matrices |
| Two-mode squeezing `S2(r, φ)` | `exp(z̄ab − z a†b†)` | `S2(r, φ + π)` in cvsim's convention |
| Displacement | `α`, `d = √2 (Re α, Im α)` | MrMustard `Dgate(x, y)`, `Coherent(x, y)`: `α = x + iy` |
| Additive noise | `Y` | MrMustard `noise_Y(n)`: `Y = n ħ/2` (so `n = 2Y` in cvsim units) |
| Log-negativity | bits | The Walrus: nats; MrMustard: bits |
| Fidelity | squared Uhlmann | same in both |
| Homodyne outcome | `x_φ` (ħ = 1) | MrMustard `Homodyne(result)`: `√ħ x_φ` |
| Heterodyne | `α`, density per `d²α` | MrMustard reports `⟨α|ρ|α⟩` = π × density |
| Homodyne in MrMustard | projective | projection on a displaced squeezed state, `r = 10`: conditioning exact to `O(e^{−20})`; the reported number is that overlap, `p(q) √(4π) e^{−r}` |

## What is compared

| Area | cvsim | qttlind | The Walrus | MrMustard |
| --- | --- | --- | --- | --- |
| Gate matrices | `core.conventions`, `gaussian.symplectic` | ✔ R, S, BS, S2, interferometer, F | ✔ R, S, BS, S2, interferometer, `expand` | ✔ R, S, BS, S2, P, CZ, CX, interferometer, D |
| Decompositions (Williamson, Bloch–Messiah, Takagi, ν) | `core.linalg` | ✔ ν | ✔ up to ordering and gauge | — (none of its own) |
| States | `GaussianState` constructors, `reduce`, `product` | ✔ | ✔ vacuum, displacement, reduction | ✔ Vacuum, Coherent, SqueezedVacuum, TMSV, Thermal, DisplacedSqueezed |
| Purity, entropy | `purity`, `entropy` | ✔ | ✔ | ✔ purity; entropy single-mode (see issues) |
| Fidelity, mixed states | `fidelity` | pure partner only | ✔ | ✔ |
| Photon numbers | `mean_photons`, `photon_covariance` | ✔ means | ✔ means and covariance | ✔ means; covariance (see issues) |
| Vacuum probability, Wigner | `vacuum_probability`, `wigner` | ✔ | ✔ vacuum element | — |
| Log-negativity | `log_negativity` | ✔ | ✔ | ✔ (see issues) |
| Gates on states | `gaussian.ops` | ✔ | via matrices | via matrices |
| Channels | `GaussianChannel` | ✔ loss, amplifier, noise, CP check | ✔ loss, lossy linear optics, photon number after loss | ✔ loss, amplifier, noise, CPTP application, composition |
| Measurements | `gaussian.measure` | ✔ homodyne, heterodyne (ideal) | — | ✔ homodyne, heterodyne, general-dyne |
| Source output states (S11) | `hardware.sources` in the pixel basis | ✔ rebuilt with its gates | ✔ rebuilt with its gates; waveguide as slices | ✔ rebuilt as a lab circuit; waveguide as slices |

Test files: `test_contract_{conventions,states,ops,channels,measure}.py` (qttlind),
`test_contract_thewalrus.py` and `test_contract_thewalrus_gaussian.py` (The Walrus),
`test_contract_mrmustard.py` (MrMustard).

## Upstream issues found

Each is reproduced by a test; cvsim's value is confirmed by the Fock oracle or by The Walrus.

| Library | Issue | Effect | Test |
| --- | --- | --- | --- |
| MrMustard 0.7.3 | `symplectic_eigenvals` takes `|eig(iΩV)|[::2]`, assuming LAPACK returns ± pairs adjacent; it does not | wrong ν (e.g. (0.838, 0.838) for (0.740, 0.838)) for many multimode states, hence wrong `von_neumann_entropy` and sometimes `log_negativity` (0 instead of 0.463 bits; The Walrus agrees with cvsim) | `test_mrmustard_multimode_symplectic_eigenvalues`, `…_entropy`, `…_log_negativity_seed_32` (xfail) |
| MrMustard 0.7.3 | `number_cov`: the off-diagonal displacement term is half of `d_J V_JK d_K` | wrong `Cov(n_j, n_k)` for displaced, correlated modes (0.0939 vs 0.0865 in the Fock oracle) | `test_mrmustard_photon_covariance_on_displaced_correlated_modes` (xfail) |
| MrMustard 0.7.3 | `compose_channels_XYd` computes `X₂ Y₁ X₂` instead of `X₂ Y₁ X₂ᵀ` | wrong (even non-symmetric) `Y` when the second channel is not symmetric, e.g. a rotation | `test_mrmustard_composition_with_a_rotation` (xfail) |
| MrMustard 0.7.3 | `von_neumann_entropy` regularises `xlogy(ν−, ν− + 1e-9)` | bias of about 1e-9 nats per mode | tolerance in `test_single_mode_entropy_and_photon_variance` |
| MrMustard 0.7.3 | `Measurement.__lshift__` never returns its result and then fails building its error message | `detector << state` raises `AttributeError`; `detector.primal(state)` works | read in the source and reproduced by hand; the bridge uses `primal` |
| MrMustard 0.7.3 | `ThresholdDetector` with dark clicks builds `P(no click | n) = (1 − η)ⁿ − p_dark` instead of `(1 − η)ⁿ (1 − p_dark)` | right for `n = 0`, wrong for `n ≥ 1`: click probabilities off by up to 0.02 for `p_dark = 0.05`; its numbers are reproduced exactly by that kernel on cvsim's photon statistics | `test_mrmustard_threshold_dark_counts` (xfail), `test_mrmustard_threshold_dark_formula_reproduced` |
| MrMustard 0.7.3 | `PNRDetector` always convolves with the dark-count prior, implemented only in the TensorFlow math backend | `PNRDetector` fails on the default NumPy backend even with `dark_counts = 0` | the bridge runs a second process on the TensorFlow backend |
| The Walrus 0.22 | `reduced_gaussian` returns the state unchanged when every mode is listed | `reduced_gaussian(…, [1, 0])` does not reorder; cvsim's `reduce([1, 0])` does | `test_reduced_gaussian_ignores_the_order_of_a_full_mode_list` |
| The Walrus 0.22, Strawberry Fields 0.23 | decomposition ordering and gauge left to LAPACK | see CONVENTIONS.md, "Comparison with other libraries" | `test_contract_thewalrus.py` |

## S08: the intermediate representation

`cvsim.ir` is data only and has no counterpart to compare numbers with. Its design follows
MrMustard's left-to-right `>>` composition (`a >> b`: `a` first) but keeps programs immutable;
the semantics of every operation are compared with The Walrus and MrMustard when programs are
executed by the backends (S09).

## S09: backends

| Comparison | Test | Agreement |
| --- | --- | --- |
| `GaussianBackend("qttlind")` vs `GaussianBackend("cvsim")`, random 3-mode programs with every Gaussian operation and post-selected imperfect measurements | `test_contract_backends.py` (`qttlind`) | 1e-11 relative |
| `GaussianBackend` vs the same circuits in MrMustard's lab gates (`Dgate`, `Rgate`, `Sgate`, `Pgate`, `BSgate`, `S2gate` with `φ + π`, `CZgate`, `CXgate`, `Interferometer`, `Ggate`, `Attenuator`, `Amplifier`, `AdditiveNoise` with `noise = 2σ²`) | `test_contract_backends.py` (`mrmustard`) | 1e-11 relative |
| `FockBackend` vs MrMustard's Gaussian-to-Fock conversion and vs `thewalrus.quantum.density_matrix` (exact matrix elements) | `test_contract_backends.py` | truncation error: 6.8e-7, 8.2e-8, 1.0e-8, 1.2e-9 at cutoffs 24, 28, 32, 36 |
| `FockBackend` vs MrMustard's own Fock simulation, with exact phase noise | `test_contract_backends.py` (`mrmustard`) | 1e-9 |
| The Walrus vs MrMustard Fock conversion | `test_contract_backends.py` | 1e-13 |
| Three engines (cvsim Gaussian, qttlind Gaussian, Fock) on the same programs | `tests/test_backends.py` | round-off; Fock moments 1e-8 at cutoff 24 |

The Walrus' `density_matrix` tensor is ordered `(n₁, m₁, n₂, m₂, …)`; cvsim's density matrices
are `(n₁ n₂ …) × (m₁ m₂ …)` (mode 0 most significant).

## S10: photon counting

| Comparison | Test | Agreement |
| --- | --- | --- |
| `core.hafnian`: hafnian (power trace and subset recursion), loop hafnian, torontonian vs `qttlind.gaussian.hafnian`; plus `perm`, `tor`, `hafnian(method="recursive")`, `hafnian(loop=True)` of The Walrus | `test_contract_photon.py` | `1e-13 · (n−1)!! max|A|^{n/2}` (the bound on any hafnian), `1e-12 · 2^m` for torontonians |
| `pattern_probability`, `click_probability` vs qttlind; `pnr_distribution` vs `thewalrus.quantum.probabilities`; clicks vs `thewalrus.threshold_detection_prob` (displaced, mixed, 1–3 modes) | `test_contract_photon.py` | 1e-12 |
| `fock_density_matrix` vs `qttlind.gaussian.fock` and `thewalrus.quantum.density_matrix` | `test_contract_photon.py` | 1e-12 |
| `herald` vs slices of The Walrus' density matrix | `test_contract_photon.py` | 1e-12 |
| `pnr_distribution` (efficiency, Poisson dark counts) and `herald` vs MrMustard's `PNRDetector`; `click_distribution` vs `ThresholdDetector` (no dark clicks) | `test_contract_photon.py` (`mrmustard`) | 1e-9 (MrMustard truncates at its internal cutoff 28) |
| PNR and threshold measurements of the Fock backend vs `photon.herald` / `click_distribution` | `tests/test_photon.py` | 1e-10 at cutoff 32 |
| Brute force: permanent and hafnian by enumeration, loop hafnian by matchings, torontonian vs inclusion–exclusion, closed forms (thermal, coherent, squeezed, two-mode squeezed heralding, Poisson detector model) | `tests/test_photon.py` | round-off |

Detector models are those of MrMustard: efficiency is a loss channel before an ideal detector
(binomial thinning of the count); PNR dark counts are Poissonian and add to the count; a
threshold detector clicks with probability `p_dark` with no light, independently of the photons.

## S11: sources

None of the three libraries models a source: no JSA or Schmidt decomposition, no OPO cavity
(threshold, escape efficiency, squeezing spectrum), no distributed loss, no excess-noise model
(read from the sources of The Walrus 0.22, MrMustard 0.7.3 and qttlind, October 2026; qttlind's
`circuits.hardware` only clips squeezing at a maximum in dB). They do implement the state a source
emits, so the contract is that state in the frequency-pixel basis (`SourceOutput.in_pixel_basis`),
rebuilt gate by gate from the source's own `r_k`, `η` and supermode unitary `U`:

| Comparison | Test | Agreement |
| --- | --- | --- |
| SPOPO output at an analysis frequency: `S(r_k(Ω))`, loss `η`, interferometer `S(U)` on 8 pixels (4 supermodes, 4 in vacuum) — qttlind `gaussian.ops`, The Walrus `symplectic` (+ `photon_number_mean` per pixel), MrMustard `SqueezedVacuum`, `Attenuator`, `Interferometer` | `test_contract_sources.py` | 1e-12 relative (qttlind, The Walrus), 1e-11 (MrMustard) |
| Phase-sensitive excess antisqueezing `diag(0, ε_k)` added to the covariance in The Walrus (MrMustard's `AdditiveNoise` is phase-insensitive only) | `test_contract_sources.py` (`thewalrus`) | 1e-12 relative |
| Waveguide with distributed loss vs a Strang product of `N` slices `loss · S(r/N) · loss` in The Walrus and in MrMustard: the splitting error is below 1e-3 at `N = 40` and falls by 4.0 (±5 %) from `N = 20` to `N = 40` | `test_contract_sources.py` | `O(1/N²)` as expected |

What neither library offers is checked against closed forms and the independent oracle
`cvsim.reference.sources` (NumPy/SciPy only, no cvsim import), in `tests/test_jsa.py` and
`tests/test_sources.py`:

| Feature | Check | Agreement |
| --- | --- | --- |
| Schmidt decomposition of a Gaussian JSA | Mehler's formula: `s_n = √(1 − μ²)|μ|ⁿ`, `K = (A² + B²)/(2AB)`, Hermite–Gauss modes of width `√(AB/2)` | 1e-12 (coefficients, K), 1e-9 (sampled mode shapes) |
| CW limit (pump narrower than the grid step) | anti-diagonal JSA: degenerate sideband pairs on `±Ω`, `s ∝ |φ(Ω, −Ω)|` | 1e-14 |
| Supermodes → pixel state | Takagi route vs the oracle's pixel-Hamiltonian squeezer `expm(ΩH)`, itself measured in the Fock oracle on two modes | 1e-11; Fock 1e-9 |
| SPOPO spectra and state, every supermode | oracle: Langevin equations and input–output theory (transfer functions, vacuum inputs at both ports); limits `σ → 1`, `Ω → ∞`, η = 1 (pure) | 1e-13 |
| Waveguide closed form | oracle: pixel-basis Lyapunov equation solved by Van Loan's exponential; Strang slices of S06 channels (`O(1/N²)`) | 1e-11 relative |
| Gain fluctuations | Monte Carlo of the mixture; `⟨e^{±2r}⟩` closed form; waveguide average by direct quadrature over `r` | 5e-3 (MC), 1e-14, 1e-10 |
| Excess antisqueezing and calibration | requested dB reproduced; `calibrate` inverts the pure-loss model and recovers the excess of the model | 1e-12 |
| Physical default crystal (2 nm pump at 397.5 nm, τ = 400 fs, β_s = 300 fs², β_p = 400 fs²) | Schmidt number converged under grid refinement (K ≈ 21.7) | 1e-5 relative |

No upstream issue was found in S11.

## Only in cvsim, and how it is checked

| Feature | Check |
| --- | --- |
| Exact fidelity with pure modes (pure modes projected out before the closed form) | Fock oracle; thermal and Scutaru closed forms; products and entangled pure modes, to 1e-11 |
| Decompositions with a fixed gauge | identities, invariance under ħ rescaling and relabelling |
| `from_hamiltonian`, Clements mesh, state preparation, mode-basis change | Fock oracle, reconstruction, round trips; the mesh is also run with qttlind's gates |
| `from_transfer`, `from_ancilla`, phase diffusion | The Walrus `passive_transformation`; ancilla closed forms; exact dephasing in the Fock oracle |
| `NoiseLedger` | exact sum identity and direct application on Hypothesis chains |
| Detector efficiency and electronic noise, local oscillator in any mode, joint sampling | explicit optical models (loss + noise + ideal homodyne; the eight-port heterodyne), Fock oracle |
| JSA, Schmidt supermodes, SPOPO, waveguide, excess noise, calibration (S11) | closed forms and `cvsim.reference.sources`, see "S11: sources" |

## Only in the references (for now)

MrMustard's Fock-space and Bargmann representations and The Walrus' approximate hafnians and
GBS samplers come with the non-Gaussian and GBS sessions (S20–S28). The non-Gaussian comb and the Lindblad
engine stay in qttlind (decision D1).

## Running the comparisons

```bash
packages/cvsim/tools/setup_reference_envs.sh   # once: the MrMustard environment
cd packages/cvsim
uv run pytest tests/contract -m qttlind -v
uv run pytest tests/contract -m "thewalrus or mrmustard" -v -rs
```
