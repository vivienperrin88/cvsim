# Continuous-variable extension

The Gaussian backend, position-space quantics and the circuit emulator (digital twin) extend the paper's solver to noisy continuous-variable optical circuits. Decisions D12–D19 continue the records of the architecture page.

## 1. The problem

A CV photonic processor prepares squeezed vacua, mixes them in an interferometer, applies a few
non-Gaussian operations, and measures with homodyne or photon-number-resolving detectors. Its
imperfections are well described by Lindblad terms: propagation and coupling loss
($L=\sqrt\kappa\,a$), thermal noise ($a$ and $a^\dagger$ jumps), phase noise ($L\propto a^\dagger a$),
plus detector inefficiency and dark counts.

Three regimes coexist:

| Regime | Example | Best representation | Cost |
| --- | --- | --- | --- |
| Gaussian states, Gaussian noise | squeezed light through a lossy interferometer, homodyne | mean vector + covariance matrix | $O(M^3)$, exact |
| Photon statistics of Gaussian states | Gaussian boson sampling | hafnians of sub-matrices | exponential in the photon number, exact |
| Non-Gaussian few-mode states | cat and GKP states, cubic phase gate, heralding | quantics tensor trains (this course) | polynomial in $\log N$ when ranks stay small |

No single representation covers all three: the covariance matrix cannot describe a cat state,
and a tensor network wastes exponential effort on Gaussian correlations that the covariance
matrix stores exactly. The design therefore combines them.

## 2. Measurements behind the design

All numbers below come from dense TT-SVDs at discarded weight $10^{-16}$–$10^{-24}$ per bond (the
truncation rule of CONVENTIONS.md). They are reproduced by the test suite and by notebooks
11–12.

**States (quantics rank).**

| State | Fock basis: bits / rank | Position basis: rank (12–16 bits) |
| --- | --- | --- |
| squeezed vacuum, 6 → 20 dB | 6–11 bits / 4–6 | 8 |
| GKP, Δ = 0.4 → 0.1 (≈5 → 17 dB) | 7–11 bits / 7 → 27 | 7–8 on the default self-dual grid; 10–18 in a tight box |
| cubic-phase or sheared squeezed state | — | 8–20 |

Fock is better for squeezed light (the paper's argument, notebook 03). Position wins for GKP
beyond ~10 dB, and the gap widens with squeezing: the Fock rank keeps growing while the position
rank does not, because a lattice of peaks is regularity *across scales* — exactly what quantics
captures (notebook 11 reproduces the table).

**Operators (MPO rank, position basis).**

| Operator | Rank | Depends on $R$? |
| --- | --- | --- |
| $x$ | 2 (exact) | no |
| Fourier transform, output bits reversed | 11–12 | no (tested to $R=24$) |
| Fourier transform with bit reversal | $2^R$ | yes — never built |
| $p = F^\dagger\,\mathrm{diag}(p)\,F$, $a$, $a^\dagger$, $n$ | 13–19 | barely |
| $e^{isx^2/2}$, $e^{i\gamma x^3/3}$ over the whole box | up to $2^{R/2}$ | yes |
| Hermite transform Fock ↔ position | ≈ number of Fock states kept | yes |

Two consequences shape everything that follows:

* **Ladder operators are cheap in position space.** The whole Lindblad machinery of the solver can
  run with modes in position space, with no change of basis.
* **Changing basis is expensive** ($O(N)$, not $O(\log N)$). Do it rarely: to import a state given
  in Fock space, or to read out photon statistics.

And a subtler one: a diagonal phase *operator* over the whole box is high-rank, but the *state*
it produces from a localised wavefunction is not (rank 8–20). So gates multiply the state
element-wise and compress at once, instead of building the operator as an MPO.

## 3. Architecture

```
                ┌──────────────────────────────────────────────────────────┐
   layer 4      │ api  System, solvers   circuits                         │
                │                        Circuit, executors, HardwareModel │
                └──────────────┬────────────────────┬──────────────┬──────┘
                               │                    │              │
                ┌──────────────▼──────┐  ┌──────────▼───────┐      │
   layer 3      │ closed   lindblad   │  │                  │      │
                └──────────────┬──────┘  │                  │      │
                               │         │                  │      │
                ┌──────────────▼─────────▼──┐  ┌────────────▼──────▼──┐
   layer 2      │ bosons        cv          │  │ gaussian             │
                │ Fock basis    position     │  │ covariance matrices, │
                │               basis        │  │ Bargmann, hafnians   │
                └──────────────┬────────────┘  └──────────┬───────────┘
                               │                          │
                ┌──────────────▼──────────────┐           │
   layer 1      │ core  TensorTrain, MPO, ... │           │
                └──────────────┬──────────────┘           │
                               └───────── NumPy / SciPy ──┘

   beside them: reference  — dense_ref and cv_dense: independent oracles
```

* **`gaussian` imports only NumPy, SciPy and `errors`.** It is a complete, standalone Gaussian
  simulator, usable without any tensor network, and testable in isolation. You can build it
  right after the core and bosons layers.
* **`cv` sits beside `bosons`** in layer 2: the position-basis counterpart of the Fock-basis
  package, on the same `core`. It may use `gaussian` (same layer) to build states from
  covariance matrices.
* **`circuits` is a layer-4 package**, like `api`: it is the only place that sees both backends
  and the comb, and the only place that decides when a mode leaves the Gaussian world.

| Package | Owns | Public objects (when complete) |
| --- | --- | --- |
| `gaussian.phase_space` | conventions, symplectic algebra | `omega`, `symplectic_from_bogoliubov`, `symplectic_eigenvalues` |
| `gaussian.states` | Gaussian states | `GaussianState` |
| `gaussian.ops` | gates and channels | `squeeze`, `beamsplitter`, `loss`, `apply_channel`, … |
| `gaussian.measure` | measurements, overlaps, Wigner | `homodyne`, `heterodyne`, `wigner`, `log_negativity` |
| `gaussian.fock` | Gaussian → Fock | `bargmann`, `fock_ket`, `fock_density_matrix` |
| `gaussian.hafnian` | photon statistics | `hafnian`, `loop_hafnian`, `pattern_probability`, `click_probability` |
| `cv.grids` | grids and position-space states | `QuadratureGrid`, `coherent`, `from_gaussian`, `cat`, `gkp` |
| `cv.fourier` | x ↔ p | `dft_mpo`, `to_momentum`, `to_position` |
| `cv.operators` | x, p, a, n as MPOs; Fock ↔ x | `momentum`, `annihilation`, `number`, `fock_to_position` |
| `cv.gates` | phase-space gates | `displace`, `symplectic`, `cubic_phase`, `kerr` |
| `cv.measure` | readout | `sample`, `homodyne`, `fock_probabilities`, `quadrature_moments` |
| `circuits` | circuits, executors, digital twin | `Circuit`, `GaussianExecutor`, `HybridExecutor`, `HardwareModel` |
| `reference.cv_dense` | dense Fock-space CV oracle | gates as `expm`, Kraus loss, moments, Wigner |

## 4. Design decisions

### D12. Exact Gaussian backend first, tensor networks only where needed

*Context.* Gaussian states of $M$ modes are $2M + 4M^2$ real numbers; Gaussian gates, channels
and measurements map them in closed form. A tensor network of the same state costs far more and
is only approximate.
*Decision.* A standalone Gaussian backend (Part VIII), and a hybrid executor that keeps
every mode Gaussian **until the first non-Gaussian element touches it**.
*Consequences.* Gaussian-dominated circuits run at the scale of real hardware (thousands of
modes, notebook 13). The tensor network only ever holds the non-Gaussian part.

### D13. One set of conventions, borrowed from the community

*Context.* CV optics has competing conventions ($\hbar\in\{1/2,1,2\}$, xxpp versus xpxp, signs of
squeezing). Silent mismatches are the classic bug.
*Decision.* $\hbar=1$, $x=(a+a^\dagger)/\sqrt2$ (as in the paper's Hermite functions),
xxpp ordering, vacuum covariance $I/2$, and the **gate generators of Strawberry Fields**, so that
published circuits translate. Each gate is implemented through its Heisenberg form
$U^\dagger aU = Aa + Ba^\dagger$, the one formula you can check by hand.
*Consequences.* CONVENTIONS.md gains a CV section; the oracle `cv_dense` implements every gate as
$\exp$ of its generator, and the tests compare the two.

### D14. Position space is a first-class basis; self-dual grids by default

*Context.* §2: ladder operators are rank ~15 in position space, GKP states are cheaper there,
homodyne detection and $x$-polynomial gates are diagonal there.
*Decision.* `cv` represents a mode by amplitudes $c_j=\sqrt{dx}\,\psi(x_j)$ on $2^R$ points of a
centred grid; the default box $L=\sqrt{2\pi N}$ makes the x and p grids identical.
*Consequences.* The Fourier transform maps the grid onto itself; the grid resolves photon
numbers up to $\approx\pi N/4$; one `QuadratureGrid` object carries the units.

### D15. Never build the bit reversal

*Context.* The quantum Fourier transform with reversed output bits has rank ~11; with the
reversal, $2^R$. Reversing the **order of the sites** of a tensor train is free.
*Decision.* `dft_mpo` returns the bit-reversed transform; `to_momentum` applies it and reverses
the sites.
*Consequences.* x ↔ p costs one MPO application at rank ~11, for any $R$.

### D16. Change basis rarely

*Context.* §2: Fock ↔ position costs $O(N)$.
*Decision.* Dynamics (gates, Lindblad terms) run in one basis per mode; the Hermite transform is
an interface operation, and the executor chooses the basis once per mode.
*Consequences.* Photon-number readout of a position-space mode uses overlaps with Hermite-function
tensor trains instead of a full change of basis.

### D17. Diagonal gates: multiply, then compress

*Context.* §2: the operator $e^{if(x)}$ can be high-rank while its product with the state is not.
*Decision.* `cv.gates.phase` samples the phase as a tensor train, multiplies element-wise
(`hadamard`), compresses. Linear phases are built exactly at rank 1.
*Consequences.* The intermediate rank is the product of the two ranks; this is fine for one mode
at $R\lesssim 20$. `qttlind.core.tci` can replace it by TCI on the product function when that is not enough.

### D18. Gaussian → Fock by recursion, statistics by hafnians

*Context.* Fock amplitudes of Gaussian states are loop hafnians, but each amplitude also follows
from its neighbours by a two-term recursion (the Bargmann representation).
*Decision.* `gaussian.fock` builds amplitudes and density matrices by the recursion (cheap, no
combinatorics); `gaussian.hafnian` computes individual pattern probabilities (Gaussian boson
sampling) and click probabilities.
*Consequences.* The hand-off from the Gaussian backend to the comb is a recursion; the two
routes cross-check each other in the tests.

### D19. Two oracles and differential testing

*Context.* Requirement R1 of the architecture: every number must be checkable. CV adds new formulas with new sign traps.
*Decision.* A second dense oracle, `reference.cv_dense`, computes everything in a truncated
Fock space without sharing a formula with the backends. On top of it, **differential tests**
compare the two backends against each other (Gaussian gates applied by shears in position space
against the covariance update; homodyne conditioning on a tensor train against the Schur
complement).
*Consequences.* Most CV tests compare `cv` with `gaussian`: a bug in either shows up as a
disagreement, and the oracle tells which one is wrong.

## 5. Scope and limits

* **Entanglement between modes** is the real limit, not the quantics legs. Deep interferometers
  fed with squeezed light are hard to simulate by any method (that is the point of Gaussian
  boson sampling). Tensor networks win when **loss** is significant (Oh, Lim, Fefferman, Jiang,
  *Nature Physics* 2024, simulate experimental GBS this way) and in **time-multiplexed**
  architectures, where neighbouring temporal modes interact along a line — the comb's backbone.
* **Large non-Gaussian circuits on many modes** are out of scope; a few non-Gaussian modes in a
  Gaussian bath are in scope.
* **Mode mismatch and spectral structure** are modelled as extra modes or as loss; the library does
  not cover multimode spectral decompositions.

## Glossary (additions)

| Term | Meaning |
| --- | --- |
| quadratures | $x=(a+a^\dagger)/\sqrt2$, $p=(a-a^\dagger)/(i\sqrt2)$ |
| covariance matrix | $V_{ij}=\langle\{\Delta r_i,\Delta r_j\}\rangle/2$; vacuum $I/2$ |
| symplectic matrix | $S\Omega S^T=\Omega$; the action of a Gaussian unitary on quadratures |
| Bogoliubov form | $U^\dagger aU=Aa+Ba^\dagger$ |
| Gaussian channel | $V\mapsto XVX^T+Y$, $d\mapsto Xd$ |
| homodyne / heterodyne | measuring one quadrature / projecting on coherent states |
| Bargmann function | generating function of Fock amplitudes; Gaussian for Gaussian states |
| (loop) hafnian | sum over perfect matchings (with self-loops) of a symmetric matrix |
| self-dual grid | $L=\sqrt{2\pi N}$, so that $dx=dp$ |
| GKP state | grid state: peaks at multiples of $\sqrt\pi$, width $\Delta$ |
| digital twin | an emulator driven by a calibrated model of a specific device |
