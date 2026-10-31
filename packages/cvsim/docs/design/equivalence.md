# Equivalence report: cvsim against qttlind, The Walrus and MrMustard

Updated every session (CONTRIBUTING.md, "Reference baselines"); state after S07, the end of the
Gaussian core (milestone M1).

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
| The Walrus 0.22 | `reduced_gaussian` returns the state unchanged when every mode is listed | `reduced_gaussian(…, [1, 0])` does not reorder; cvsim's `reduce([1, 0])` does | `test_reduced_gaussian_ignores_the_order_of_a_full_mode_list` |
| The Walrus 0.22, Strawberry Fields 0.23 | decomposition ordering and gauge left to LAPACK | see CONVENTIONS.md, "Comparison with other libraries" | `test_contract_thewalrus.py` |

## Only in cvsim, and how it is checked

| Feature | Check |
| --- | --- |
| Exact fidelity with pure modes (pure modes projected out before the closed form) | Fock oracle; thermal and Scutaru closed forms; products and entangled pure modes, to 1e-11 |
| Decompositions with a fixed gauge | identities, invariance under ħ rescaling and relabelling |
| `from_hamiltonian`, Clements mesh, state preparation, mode-basis change | Fock oracle, reconstruction, round trips; the mesh is also run with qttlind's gates |
| `from_transfer`, `from_ancilla`, phase diffusion | The Walrus `passive_transformation`; ancilla closed forms; exact dephasing in the Fock oracle |
| `NoiseLedger` | exact sum identity and direct application on Hypothesis chains |
| Detector efficiency and electronic noise, local oscillator in any mode, joint sampling | explicit optical models (loss + noise + ideal homodyne; the eight-port heterodyne), Fock oracle |

## Only in the references (for now)

Fock amplitudes, hafnians and photon-number distributions (qttlind, The Walrus, MrMustard) arrive
with photon counting in S10 and will be compared there; MrMustard's Fock-space and Bargmann
representations with the non-Gaussian sessions (S20–S24). The non-Gaussian comb and the Lindblad
engine stay in qttlind (decision D1).

## Running the comparisons

```bash
packages/cvsim/tools/setup_reference_envs.sh   # once: the MrMustard environment
cd packages/cvsim
uv run pytest tests/contract -m qttlind -v
uv run pytest tests/contract -m "thewalrus or mrmustard" -v -rs
```
