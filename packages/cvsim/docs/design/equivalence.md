# Equivalence report: cvsim ↔ qttlind (Gaussian core)

End of milestone M1 (S07). Decision D2 asks for two independently written Gaussian engines that
agree number for number. This page lists every Gaussian operation both libraries implement, the
contract test that compares them (marker `qttlind`, directory `tests/contract`), and the tolerance.
It also lists what only one of them has, and how that part is checked instead.

Tolerances: `EXACT` = 1e-14 (closed-form matrices), `ROUNDOFF` = 1e-12 relative to the largest
covariance entry (short computations); densities and figures of merit 1e-10 to 1e-11 relative.
Inputs are hand-picked cases plus Hypothesis-generated states on 1–4 modes, mixed or pure,
squeezed up to `r = 1` per mode.

## Operations both libraries implement

| Area | cvsim | qttlind | Contract test | Tolerance |
| --- | --- | --- | --- | --- |
| Conventions | `core.conventions.omega`, `mode_indices`, `complex_basis`, `symplectic_from_bogoliubov` | `gaussian.phase_space.*` | `test_contract_conventions.py` | EXACT |
| Gate matrices | `rotation`, `squeezing`, `beamsplitter`, `two_mode_squeezing`, `interferometer`, `fourier` (= qttlind `F†`), `displacement_vector` | `gaussian.ops.*_matrix`, `phase_space` | `test_contract_conventions.py` | EXACT |
| Symplectic eigenvalues | `core.linalg.symplectic_eigenvalues` | `phase_space.symplectic_eigenvalues` | `test_contract_conventions.py` | ROUNDOFF |
| States | `GaussianState.vacuum`, `coherent`, `thermal`, `squeezed`, `product`, `reduce` | `GaussianState.*` (`squeezed_vacuum`) | `test_contract_states.py` | EXACT |
| Figures of merit | `purity`, `mean_photons`, `entropy`, `is_pure` | same names | `test_contract_states.py` | 1e-10 |
| Distances | `overlap`, `fidelity` (pure partner), `vacuum_probability` | `measure.overlap`, `fidelity`, `vacuum_probability` | `test_contract_states.py` | 1e-11 |
| Entanglement, phase space | `log_negativity`, `wigner_grid` | `measure.log_negativity`, `wigner` | `test_contract_states.py` | 1e-10 / ROUNDOFF |
| Gates on states | `ops.apply_symplectic`, `displace`, `rotate`, `squeeze`, `beamsplitter`, `two_mode_squeeze`, `interferometer` | `ops.*` | `test_contract_ops.py` | ROUNDOFF |
| Clements mesh | `symplectic.clements(U).gates()` | executed with `ops.rotate` / `ops.beamsplitter` | `test_contract_ops.py` | ROUNDOFF |
| Channels | `GaussianChannel.loss`, `amplifier`, `additive_noise` | `ops.loss`, `amplifier`, `thermal_noise` | `test_contract_channels.py` | ROUNDOFF |
| Complete positivity | `GaussianChannel(X, Y)` check | `ops.check_channel` | `test_contract_channels.py` | same accept / reject |
| Homodyne | `measure.homodyne` (ideal), `homodyne_density` | `measure.homodyne`, `homodyne_density` | `test_contract_measure.py` | 1e-11 / ROUNDOFF |
| Heterodyne | `measure.heterodyne` (ideal) | `measure.heterodyne` | `test_contract_measure.py` | 1e-11 / ROUNDOFF |

Conventions that differ by design are checked as such: cvsim's `fourier()` is `R(π/2)` and
qttlind's `F` is its inverse; both libraries give log-negativity in bits and the squared fidelity.

## Only in cvsim, and how it is checked

| Feature | Check |
| --- | --- |
| Mixed-state fidelity (pure modes projected out, then Banchi–Braunstein–Pirandola) | Fock oracle, thermal and Scutaru closed forms, products and entangled pure modes, to 1e-11 |
| Photon covariance, complex moments, multimode Wigner function | Fock oracle |
| Decompositions with a fixed gauge (Williamson, Bloch–Messiah, Takagi) | identities; The Walrus up to the gauge (`test_contract_thewalrus.py`) |
| `from_hamiltonian`, Clements, state preparation, mode-basis change | Fock oracle, reconstruction, round trips |
| CZ networks, CX, shear gates on states | matrices of `core.conventions`, Fock oracle (S02) |
| `from_transfer`, `from_ancilla`, phase diffusion | The Walrus `passive_transformation`; ancilla closed forms; exact dephasing in the Fock oracle |
| `NoiseLedger` | exact sum identity and direct application on Hypothesis chains |
| Detector efficiency and electronic noise, general-dyne, LO in any mode, joint sampling | explicit optical models (loss + noise + ideal homodyne; the eight-port heterodyne), Fock oracle |

## Only in qttlind

Fock amplitudes of Gaussian states (`gaussian.fock`) and hafnians (`gaussian.hafnian`) arrive in
cvsim with photon counting in S10, which adds their contract tests. The non-Gaussian comb and the
Lindblad engine stay in qttlind (decision D1), reached through the backend adapter in S31.

## Running the report

```bash
cd packages/cvsim
uv run pytest tests/contract -m qttlind -v      # every comparison listed above
uv run pytest tests/contract -m thewalrus -v    # The Walrus comparisons
```
