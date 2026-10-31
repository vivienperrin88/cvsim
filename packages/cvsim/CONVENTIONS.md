# Conventions

These choices are fixed once for the whole code base. They are **identical to qttlind's**
(`packages/qttlind/CONVENTIONS.md`, section *Continuous variables*) so that the two Gaussian
engines can be compared number for number (decision D2). Rules marked **[tested]** are implemented
twice, independently, in `cvsim.core.conventions` and in the oracle `cvsim.reference`, and
cross-checked by `tests/test_conventions.py`; when qttlind is installed, the contract tests also
compare them with qttlind.

## Numbers

* Scalars are double precision: covariances, means and symplectic matrices are `float64`,
  complex amplitudes and Fock-space objects `complex128`.
* ħ = 1 internally. `core.conventions.convert_hbar` converts to other conventions (Strawberry
  Fields and The Walrus use ħ = 2, vacuum `V = I`); `to_snu` gives shot-noise units (vacuum = 1),
  `db_from_variance` the noise level relative to the vacuum in dB (negative = squeezing),
  `db_from_r(r) = 20 log10(e) r` ≈ 8.686 r dB. **[tested]**

## Continuous variables

* **Quadratures** **[tested]**: `x = (a + a†)/√2`, `p = (a − a†)/(i√2)`, `[x, p] = i`; vacuum
  `V = I/2`.
* **Ordering** **[tested]**: `r = (x_1, …, x_M, p_1, …, p_M)` ("xxpp"); covariance
  `V_ij = ⟨{Δr_i, Δr_j}⟩/2`; `[r_i, r_j] = iΩ_ij`, `Ω = [[0, I], [−I, 0]]`. `to_xpxp` /
  `from_xpxp` reorder to `(x1, p1, x2, p2, …)`. Complex amplitudes: `(a, a†) = W r`,
  `W = [[I, iI], [I, −iI]]/√2`.
* **Fock spaces**: mode 0 is the most significant factor of the Kronecker product.
* **Gates** **[tested]**: a gate `U` with `U† a U = A a + B a†` acts as `μ → S μ`, `V → S V Sᵀ`,
  `S = [[Re(A+B), −Im(A−B)], [Im(A+B), Re(A−B)]]`.

  | gate | generator | `U† a U` | `S` |
  | --- | --- | --- | --- |
  | `D(α)` | `exp(α a† − ᾱ a)` | `a + α` | `μ → μ + √2 (Re α, Im α)` |
  | `R(θ)` | `exp(iθ n)` | `e^{iθ} a` | counter-clockwise rotation |
  | `S(r, φ)` | `exp((z̄ a² − z a†²)/2)`, `z = r e^{iφ}` | `cosh r a − e^{iφ} sinh r a†` | `φ = 0`: `diag(e^{−r}, e^{r})`; `x_{φ/2}` squeezed |
  | `BS(θ, φ)` | `exp(θ(e^{iφ} a b† − e^{−iφ} a† b))` | `cos θ a − e^{−iφ} sin θ b` | transmissivity `cos²θ` |
  | `S2(r, φ)` | `exp(z̄ ab − z a† b†)` | `cosh r a − e^{iφ} sinh r b†` | |
  | interferometer | `exp(i a† H a)`, `U = e^{iH}` | `Σ_k U_jk a_k` | `[[Re U, −Im U], [Im U, Re U]]` |
  | `CZ(A)` | `exp(i/2 Σ A_jk x_j x_k)` | `p → p + A x` | `[[I, 0], [A, I]]`; nullifiers `p − A x` |
  | `CX(g)` | `exp(−i g x_a p_b)` | `x_b → x_b + g x_a`, `p_a → p_a − g p_b` | |
  | `P(s)` | `exp(i s x²/2)` | `p → p + s x` | `[[1, 0], [s, 1]]` |
  | `F` | `R(π/2)` | `x → −p`, `p → x` | qttlind's `F` (unitary DFT) is `F†` |

* **Channels** **[tested]**: `μ → X μ`, `V → X V Xᵀ + Y`. Loss of transmissivity `η` into a
  thermal environment of `n̄` photons: `X = √η I`, `Y = (1 − η)(n̄ + ½) I`; the Lindblad equation
  with `L = √κ a` during `t` is this channel with `η = e^{−κt}`.
* **Homodyne** **[tested]**: measures `x_φ = cos φ x + sin φ p`, equivalently `R(−φ)` then `x`.
* **Oracle**: `cvsim.reference.fock_dense` *measures* each gate's symplectic matrix in a truncated
  Fock space, `S = C (iΩ)^{-1}` with `C_ik = ⟨0|[U† r_i U, r_k]|0⟩`, and checks that `U† r U` is
  linear on probe states (a Kerr gate is rejected). It imports nothing from cvsim.
* **Contract** (`tests/contract`, marker `qttlind`): every rule above agrees with
  `qttlind.gaussian` to a few ulps (closed forms) or 1e-12 (short computations).

## Symplectic library **[tested]**

`cvsim.gaussian.symplectic` (matrices) and `cvsim.gaussian.ops` (gates applied to states):

* **Composition.** Gates `g₁, g₂, …` applied in that order have `S = ⋯ S₂ S₁`; on states
  `d → S d`, `V → S V Sᵀ`.
* **Quadratic Hamiltonians.** `from_hamiltonian(H, t)` is the gate `exp(−i t H)` with
  `H = ½ rᵀ H r`: `S = expm(t Ω H)`. `H = −I` gives `R(t)`; `H = −[[0, 1], [1, 0]]` gives `S(t, 0)`.
* **Clements mesh.** A cell on modes `(m, m+1)` is `T(θ, φ) = BS(θ, 0) · R_m(φ)`, unitary
  `[[e^{iφ} cos θ, −sin θ], [e^{iφ} sin θ, cos θ]]`; `U = diag(phases) · T_K ⋯ T_1` with
  `N(N−1)/2` cells, depth `N`, `θ ∈ [0, π/2]`, `φ ∈ (−π, π]`. `Clements.gates()` lists the gates
  in time order in the conventions of the gate table.
* **State preparation.** `state = D(d) · I(U₁) · Sq(r) · I(U₂) · ⊗ thermal(ν)`, `r` descending,
  `ν` ascending (Williamson, then Bloch–Messiah, with their gauge); `U₂ = I` for a pure state.
* **Mode bases.** `change_basis(state, U)`: new mode `b_k = Σ_j conj(U_jk) a_j`, so column `k`
  of `U` is the mode function of `b_k` in the old basis; `d' = S(U)ᵀ d`, `V' = S(U)ᵀ V S(U)`. The
  supermodes of a state are the columns of the `U` of its Bloch–Messiah `O1`.

## Channels and the noise ledger **[tested]**

* **Channels** `GaussianChannel(X, Y, c)`: `d → X d + c`, `V → X V Xᵀ + Y`; completely positive
  iff `Y + (i/2)(Ω − X Ω Xᵀ) >= 0` (checked at construction, `PhysicalityError` otherwise).
  `a.then(b)` is `a` first: `X = X_b X_a`, `Y = X_b Y_a X_bᵀ + Y_b`. Loss `η` into `n̄`:
  `X = √η I`, `Y = (1 − η)(n̄ + ½) I`; amplifier `G`: `X = √G I`, `Y = (G − 1)(n̄ + ½) I`; lossy
  linear optics `T` (`‖T‖ <= 1`): `X = S(T)`, `Y = ½(I − XXᵀ)`.
* **Phase diffusion** `φ ~ N(0, σ²)` is not Gaussian: cvsim returns the Gaussian state with the
  exact first and second moments (mean × `e^{−σ²/2}`, anisotropic part of the mode's second moments
  × `e^{−2σ²}`, its correlations × `e^{−σ²/2}`) and warns with `ApproximationWarning`.
* **Noise ledger** (decision D7, approved convention): `V_out = signal + Σ_k C_k` with
  `signal = 𝓛_n ⋯ 𝓛_1(V_in)` and `C_k = 𝓛_n ⋯ 𝓛_{k+1}(Y_k)` — each component's added noise
  *carried forward to the output* through every later gate, channel and dephasing map. The
  contributions add up to `V_out − signal` exactly and are tagged by component and kind; a
  loss-like step (`X = √η I`, `η < 1`) also records its transmissivity. Quadrature budgets are
  `uᵀ C_k u` for `x_φ` of a mode, in shot-noise units by default. **dB are not additive**:
  `db_recovered(component)` replays the chain with that component ideal and is reported as a
  separate what-if view; such numbers do not sum to the total loss of squeezing. Phase-diffusion
  entries (the variance of the randomly rotated mean) are flagged `approximate`.

## Measurements **[tested]**

`cvsim.gaussian.measure`. One model: a measurement on modes `A` records `y = L r_A + ξ`,
`ξ ~ N(0, N)`; the outcome has mean `L d_A` and covariance `Σ = L V_A Lᵀ + N`; the other modes
are conditioned by the Schur complement `V_B' = V_B − K Σ Kᵀ`, `d_B' = d_B + K(y − μ)`,
`K = V_{BA} Lᵀ Σ⁻¹`; the measured modes are consumed.

* **Homodyne** of `x_φ = cos φ x + sin φ p` (`L = (cos φ, sin φ)`); several modes can be measured
  jointly. **Heterodyne**: `L = I`, ideal `N = I/2`; outcome `α = (y_x + i y_p)/√2`, density per
  `d²α` (the Husimi function `⟨α|ρ|α⟩/π` for an ideal detector).
* **Detector**: efficiency `η` and electronic noise `v_el` (shot-noise units, per photodiode pair)
  are loss and added noise before an ideal detector, the outcome **rescaled by `1/√η`** so that it
  estimates the quadrature at the detector input. Added variance: `(1 − η + v_el)/(2η)` for
  homodyne, `½ + (1 − η + v_el)/η` per quadrature for heterodyne (each arm sees half the signal).
* **Local oscillator in any mode**: the measured mode is `b = Σ_j conj(f_j) a_j` (unit `f`); the
  remaining modes are returned in a basis completing `f`, whose mode functions come with the
  result. Mode mismatch (a non-normalised overlap) is a hardware model (S12), not accepted here.

## Backends **[tested]**

* A program runs from the vacuum; a measured mode is consumed until prepared again;
  `Result.modes` lists the program modes of the final state, in register order.
* Measurement outcomes given in the program are post-selected, the others sampled from `rng`.
* `FockBackend`: density matrices, mode 0 most significant, cutoff `N` per mode. Gates are the
  exponentials of their generators on the touched modes; `Interferometer` runs as its Clements
  mesh and `Symplectic` as Bloch–Messiah. Every phase-insensitive channel is pure loss then a
  quantum-limited amplifier: thermal loss `(η, n̄)` = loss `η/G'`, gain `G' = 1 + (1 − η) n̄`;
  thermal amplifier `(G, n̄)` = loss `G/G'`, gain `G' = G + (G − 1) n̄`; additive noise `σ²` = loss
  `1/G`, gain `G = 1 + σ²`. Phase noise is exact dephasing `ρ_nm e^{−σ²(n−m)²/2}`.

## Photon counting **[tested]**

`cvsim.core.hafnian` and `cvsim.gaussian.photon`.

* **Fock elements** of a Gaussian state: with `Q = σ_c + I/2` (`σ_c` the covariance of
  `(a, a†)`), `A = X(I − Q⁻¹)`, `γ = X Q⁻¹ β` and `T = exp(−½ β† Q⁻¹ β)/√det Q`,
  `⟨n|ρ|m⟩ = T lhaf(A_{(m⊕n)}; γ)/√(n! m!)`, computed by the recurrence
  `√(k_i + 1) g_{k+e_i} = γ_i g_k + Σ_j A_ij √k_j g_{k−e_j}`. `fock_density_matrix` has the layout
  of the Fock backend: `(n₁ n₂ …) × (m₁ m₂ …)`, mode 0 most significant.
* **Threshold detectors**: `P(clicks on S, none elsewhere) = Σ_{Z ⊆ S} (−1)^{|Z|} p₀(rest ∪ Z)`
  from vacuum probabilities of reduced states (the torontonian for zero mean).
* **Detector models** (as MrMustard): efficiency `η` is a loss channel before an ideal detector;
  PNR dark counts are Poissonian with mean `ν` and add to the count; a threshold detector clicks
  with probability `p_dark` with no light, independently of the photons.
* **Heralding**: `herald` returns the normalised state of the other modes and the probability of
  the pattern; the IR measurements `PNR` and `Threshold` do the same in the Fock backend
  (outcomes sampled or post-selected).

## Figures of merit **[tested]**

`cvsim.gaussian.GaussianState`, checked against the Fock oracle and, where it has them, qttlind:

* **Purity** `Tr ρ² = 1/(2^M √det V)`. **Entropy** von Neumann, in **nats** by default
  (`entropy(base=2)` for bits).
* **Fidelity** is the *squared* Uhlmann fidelity `F = (Tr √(√ρ σ √ρ))²` (= `|⟨ψ|φ⟩|²` for pure
  states), as in qttlind. Pure Williamson modes (`ν − ½ < pure_tol · max(1, ‖S‖²)`, default
  `PURE_TOL = 1e-12`) are projected out exactly, then mixed states use Banchi, Braunstein &
  Pirandola (2015); this avoids the `√ε` amplification of round-off near `ν = ½`.
* **Logarithmic negativity** `E_N = log‖ρ^{T_A}‖₁` in **bits** by default (`base=math.e` for
  nats), as in qttlind: the two-mode squeezed vacuum has `E_N = 2r / ln 2`.
* **Photon moments**: `⟨n_k⟩`, `Cov(n_j, n_k)`, and `(⟨a⟩, ⟨a†a⟩, ⟨aa⟩)` with
  `N_jk = ⟨a_j† a_k⟩`, `A_jk = ⟨a_j a_k⟩`.
* **Wigner function** normalised to `∫ W d^{2M}r = 1` over `(x, p)` (not over `α`).

## Decompositions **[tested]**

`cvsim.core.linalg` is the only module that calls LAPACK (**[guarded]**). Its symplectic
decompositions follow these rules:

* **Ordering.** Williamson `ν` ascending; Bloch–Messiah `r >= 0` descending (most squeezed
  supermode first); Takagi `s` descending.
* **Factors.** `williamson`: `V = S diag(ν, ν) Sᵀ`, `S` symplectic. `bloch_messiah`:
  `S = O1 diag(e^{−r}, e^{r}) O2`, `O1`, `O2` orthogonal symplectic to round-off; the supermodes are
  `U[:, k]` with `O1 = [[Re U, −Im U], [Im U, Re U]]`. `takagi`: `A = U diag(s) Uᵀ`, `U` unitary.
* **Degeneracy.** Values within a relative `rtol` (default `1e-8`, `DEGENERACY_RTOL`) form a group;
  every decomposition returns its `groups`. Squeezing below `rtol` is reported as `r = 0`
  (reconstruction error at most about `rtol · ‖S‖`). Grouping values that differ by `ε` (a larger
  `rtol`) costs a reconstruction error of order `ε`.
* **Gauge.** Inside a group the factors are defined only up to a rotation: `O(k)` for equal
  `r > 0` and equal `s > 0`, the passive `U(k)` for `r = 0`, `s = 0` and every Williamson group. It
  is fixed by **orthogonal Procrustes against a reference basis**: the `k` reference columns with
  the largest weight in the group's subspace are selected (in reference order) and the group is
  rotated to be closest to them. The default reference is the input modes; `reference=` accepts a
  previous result (to keep a parameter sweep continuous) or a user basis (e.g. Hermite–Gauss mode
  functions). For a one-dimensional group this fixes the sign (largest component of positive real
  part) or the phase (largest component real positive). Results therefore do not depend on the
  basis LAPACK happens to return.
* **Physical projection.** `nearest_physical_covariance` raises every symplectic eigenvalue below
  `1/2` to `1/2` in the Williamson basis: the correction is pure added noise and vanishes for a
  physical input (eigenvalues within `atol · max(1, max|V|)` below `1/2`, default `atol = 1e-12`,
  are Williamson round-off and count as physical: the input is returned unchanged). It is not the
  Frobenius-nearest physical matrix (a semidefinite program).

## Comparison with other libraries

How the same decompositions come out of The Walrus 0.22, Strawberry Fields 0.23 and MrMustard
0.7.3 (read from their sources and checked numerically, October 2026). All three use ħ = 2
(vacuum `V = I`) and the xxpp ordering with `Ω = [[0, I], [−I, 0]]`; apart from ħ this is cvsim's
convention. Decomposition *factors* do not depend on ħ; symplectic eigenvalues scale with it
(vacuum `ν = 1` there, `1/2` here). Entries marked **[contract]** are asserted by
`tests/contract/test_contract_thewalrus.py` (marker `thewalrus`, extra `compare`): the matrix
relating the two results is computed and must have exactly the stated form, acting inside the
degeneracy groups only. Strawberry Fields and MrMustard are not installed for the tests (heavy
dependencies); their entries were checked by running their decomposition code by hand.

| | cvsim | The Walrus (`thewalrus.decompositions`) | Strawberry Fields (`sf.decompositions`) |
| --- | --- | --- | --- |
| Williamson: result | `(nu, S, groups)`, `V = S diag(ν, ν) Sᵀ` | `(Db, S)`, `V = S Db Sᵀ` | `(Db, S)`, `V = S Db Sᵀ` (the docstring says `Sᵀ Db S`; the code returns the former) |
| Williamson: order of ν | ascending | LAPACK's Schur order, unsorted **[contract]** | Schur order, unsorted |
| Williamson: gauge | Procrustes (sign/phase per mode, `U(k)` per group) | LAPACK's per-mode rotation **[contract]** | LAPACK's; can differ from The Walrus's |
| Bloch–Messiah: result | `S = O1 diag(e^{−r}, e^{r}) O2` | `S = O D Q`, `D = diag(e^{r}, e^{−r})` | `S = O1 D O2`, `D = diag(e^{r}, e^{−r})` |
| Bloch–Messiah: blocks | squeezed quadrature first | anti-squeezed first: `O = O1 Ω K`, `K` a gauge rotation **[contract]** | anti-squeezed first; per-mode signs can differ from The Walrus |
| Bloch–Messiah: order | `r` descending | `r` descending **[contract]** | `r` descending |
| Bloch–Messiah: no squeezing | `O1 = I`, `O2 = S` | `O = I`, `Q = S` **[contract]** | `O1 = S`, `O2 = I` |
| Bloch–Messiah: degeneracy | relative `rtol` (1e-8), reported in `groups` | not detected; LAPACK's basis **[contract]** | values rounded to 9 decimals, then an SVD per group |
| Takagi: result | `(s, U, groups)`, `s` descending | `(s, U)`, `s` descending | `(s, U)`, `s` descending and rounded to 13 decimals |
| Takagi: gauge | sign / `O(k)` fixed by Procrustes | LAPACK's signs **[contract]** | LAPACK's signs; groups found by rounding |
| Symplectic eigenvalues | ascending, vacuum 1/2 | unsorted, in the input's scale (vacuum 1 for ħ = 2) **[contract]** | — |

**MrMustard 0.7.3** has no Williamson, Bloch–Messiah or Takagi of its own and does not call The
Walrus's (it imports The Walrus only to report its version). Its
`physics.gaussian.symplectic_eigenvals` returns `|eig(iΩ V · 2/ħ)|[::2]`: unsorted, in shot-noise
units (vacuum 1).

**What the gauge buys.** In these libraries the gauge is whatever LAPACK returns: repeatable on one
machine, but with nothing else guaranteed. For example, The Walrus's Williamson factor changes when
the same covariance is rescaled from ħ = 1 to ħ = 2, or when the modes are relabelled and then
relabelled back. The Walrus and Strawberry Fields can also return different supermode signs for the
same matrix. cvsim's factors are invariant under the ħ rescaling and follow a relabelling of the
modes (`S(PVPᵀ) = P S(V)` for non-degenerate ν; inside a degenerate group the columns may be
reordered) **[tested]**. With `reference=` they also stay continuous along a parameter sweep. To
compare a cvsim result with one of these libraries, compare gauge-invariant quantities (ν, r, s,
`O1 D O2`, projectors on degenerate groups) or compute the relating rotation as the contract tests
do.

## Errors and warnings

* Every deliberate error derives from `cvsim.errors.CvsimError`: `InputError` (also a
  `ValueError`) for invalid arguments, with the subclasses `ShapeError` (inconsistent shapes or
  mode counts), `PhysicalityError` (a state or channel that violates quantum mechanics beyond the
  numerical tolerance) and `CalibrationError` (calibration data the model cannot produce);
  `ConvergenceError` (also a `RuntimeError`) for iterative methods that fail.
* Validate at public entry points; messages state what was expected and what was received.
* Warnings derive from `CvsimWarning`: `ApproximationWarning` when a result relies on an
  approximation (for instance moment-matched phase noise), `CutoffWarning` when a Fock cutoff is
  nearly exhausted. The test suite turns them into errors unless a test expects them.

## Randomness

* Every stochastic function takes a `numpy.random.Generator` (argument `rng`); the legacy global
  state (`np.random.seed`, `np.random.normal`, …) is never used. Tests use the seeded `rng`
  fixture. **[guarded]** by `tests/test_guards.py`.

## Architecture rules **[guarded]**

Checked by parsing every module (`tests/test_guards.py`), each guard being itself tested on
synthetic violations:

* the layer rule of `cvsim._layers` (a module imports its own layer or lower ones; `errors`
  imports nothing from cvsim; `reference` imports nothing from cvsim and is imported by tests only;
  subpackages never import the package root);
* `qttlind` is imported only in `cvsim.backends`, and every other module imports in an interpreter
  where qttlind is absent;
* `numpy.linalg` and `scipy.linalg` are used only in `cvsim.core.linalg` and the oracles;
* every module has a docstring; every public module an `__all__` that lists real names; every
  public function a docstring, with a numpydoc *Parameters* section when it takes arguments.

## Feature flags

`cvsim.FEATURES` lists the implemented capabilities, `cvsim.KNOWN_FEATURES` every planned one (with
the session that delivers it). A name enters `FEATURES` only when its API is complete and tested.
Callers check features (`cvsim.require("gaussian")`), never a bare import.
