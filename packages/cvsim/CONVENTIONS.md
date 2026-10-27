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
