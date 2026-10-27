# Conventions

These choices are fixed once for the whole code base: the solver (`qttlind.core`, `bosons`,
`closed`, `lindblad`), the dense reference oracle (`qttlind.reference`) and the tutorials. Rules
marked **[tested]** are implemented twice, independently, in `qttlind.core.conventions` and in the
oracle, and cross-checked by `tests/test_conventions.py`; changing one without the other fails
the suite.

## Numbers

* Scalars are complex double precision everywhere (`complex128`). Real-valued quantities
  (singular values, weights, purities) are `float64`.
* ħ = 1. The transmon examples use time in **ns** and angular frequencies in **rad/ns**, i.e. a
  frequency f in GHz enters as 2πf. Cat-qubit examples are dimensionless (g₂ = 1).

## Quantics index map **[tested]**

A Fock index `n` in `0..2^R` is written on `R` bits, **most significant bit first** (paper Eq. 4):

```
n = Σ_{i=1..R} 2^(R−i) σ_i        bits_of(5, 4) == [0, 1, 0, 1]
```

* σ₁ (the MSB) is the leg tensor attached to the backbone; σ_R (the LSB) is the free end.
* MPO site tensors carry `(σ_i, σ'_i)` in the same order. Bit-interleaved dense layouts list
  `(σ₁, σ'₁, σ₂, σ'₂, …)`.
* `R` is between 1 and 62, so `2^R` fits in a NumPy `int64`.
* Solver: `qttlind.core.conventions.{bits_of, index_of, bits_array}`. Oracle: `qttlind.reference.quantics`.

## Truncation tolerance **[tested]**

A tolerance `tol` is the **discarded relative weight per bond**. For singular values
`s₀ ≥ s₁ ≥ … ≥ 0`, keep the smallest `k ≥ 1` such that

```
Σ_{j≥k} s_j²  /  Σ_j s_j²   ≤   tol
```

then cap `k` at `max(1, chi_max)` when a cap is given; the returned discarded weight is what was
actually dropped. An empty spectrum keeps 0, an all-zero spectrum keeps 1.

* `tol` is the **square** of the relative Frobenius error of that bond: `tol = 1e-16` means a
  relative error of `1e-8`. The paper quotes tolerances without defining them; we read its
  "tolerance 1e−8" as `tol = 1e-8` on this weight (see the implementation guide, Section 6).
* On the purification index μ, the discarded weight equals the trace-norm error made on
  ρ = ΨΨ†, so `tol` bounds the error of every bounded observable relative to its norm.
* Solver: `qttlind.core.conventions.keep_count`. Oracle: `qttlind.reference.dense_ref.keep_count`.

## Tensor layouts

| Object | Core axes | Notes |
| --- | --- | --- |
| `TensorTrain` core | `(left, phys, right)` | outer bonds 1; site 0 most significant |
| `MPO` core | `(left, out, in, right)` | `apply` contracts `in` with the state's `phys` |
| fused bonds after a product | `(operator bond, state bond)` | C order: operator bond slowest |

Public methods never modify their arguments or `self`; they return new objects.

## Errors and warnings

* Every deliberate error derives from `qttlind.errors.QttlindError`: `InputError` (also a
  `ValueError`) for invalid arguments, `ShapeError` for inconsistent shapes, `ConvergenceError`
  (also a `RuntimeError`) for iterative methods that fail.
* Validate at public entry points; messages state what was expected and what was received.
* Warnings derive from `QttlindWarning` (`CutoffWarning` when a mode's population approaches its
  Fock cutoff).
* Exercises not done yet raise `NotImplementedError("Sxx · name")`; the test suite reports them as
  TODO.

## Purified density matrices

* ρ = ΨΨ† with the purification index μ last in dense form (`Psi.shape == (D, chi_mu)`) and as a
  leaf node hanging off the last backbone tensor in the comb.
* The trace is restored after every Kraus step (ρ → ρ/Tr ρ); the trace defect before
  renormalisation is logged as a stability diagnostic.
* Dropping `V†` after an SVD on μ is a gauge choice: it never changes ρ.

## Multi-mode ordering

* Modes are ordered as they are added to a `System`, left to right along the backbone.
* Dense multi-mode vectors use the Kronecker order of that list, mode 0 most significant
  (`numpy.kron(op_mode0, op_mode1)`).

## Time-dependent Hamiltonians

* The Kraus integrators evaluate time-dependent coefficients at the step midpoint `t + h/2`.
* Closed-system Magnus-TDVP uses the two Gauss points `t + (3 ∓ √3) h / 6` (paper Eq. 28).

## Continuous variables

* **Quadratures**: ħ = 1, `x = (a + a†)/√2`, `p = (a − a†)/(i√2)`, `[x, p] = i`. Vacuum: `V = I/2`.
* **Ordering**: quadrature vectors are `r = (x_1, …, x_M, p_1, …, p_M)` ("xxpp"); covariance
  `V_ij = ⟨{Δr_i, Δr_j}⟩/2`; symplectic form `Ω = [[0, I], [−I, 0]]`.
* **Gates** use the Strawberry Fields generators (`qttlind.reference.cv_dense` lists them):
  `R(θ) = exp(iθn)`, `S(r, φ) = exp((z̄a² − za†²)/2)`, `BS(θ, φ) = exp(θ(e^{iφ}ab† − e^{−iφ}a†b))`,
  `S2(r, φ) = exp(z̄ab − za†b†)`, `D(α) = exp(αa† − ᾱa)`, `K(κ) = exp(iκn²)`, `V(γ) = exp(iγx³/3)`.
  A gate's symplectic matrix comes from its Heisenberg form `U†aU = Aa + Ba†`; states transform as
  `d → S d`, `V → S V Sᵀ`.
* **Channels**: `d → X d`, `V → X V Xᵀ + Y`; loss of transmissivity η is `X = √η I`,
  `Y = (1 − η)(n̄ + ½) I`, and equals the Lindblad loss `L = √κ a` for `η = exp(−κt)`.
* **Position grids**: `x_j = (j − N/2) dx`, `dx = L/N`, `p_k = (k − N/2) dp`, `dp = 2π/L`; the
  default box `L = √(2πN)` is self-dual (`dx = dp`). Tensor trains store `c_j = √dx ψ(x_j)`
  (unit norm), most significant bit first, like Fock indices.
* **Fourier transform**: centred and unitary, `c̃_k = N^{−1/2} Σ_j exp(−2πi(k − N/2)(j − N/2)/N) c_j`;
  `F = R(−π/2)`, `F φ_n = (−i)^n φ_n`. `cv.fourier.dft_mpo` has its **output bits reversed**;
  `to_momentum` reverses the sites afterwards.
* **Bargmann**: `F(z) = Σ ρ_mn ᾱ^m β^n / √(m! n!)` with `z = (ᾱ, β)`: ket indices first, then
  bra indices; kets of pure states are normalised with `⟨0|ψ⟩ > 0`.
* **Gaussian oracle and backend are independent**: `reference.cv_dense` computes in a truncated
  Fock space (gates by `expm`, moments by traces) and shares no formula with `qttlind.gaussian`.

## Feature flags

`qttlind.FEATURES` lists the implemented capabilities. A name is listed in
`src/qttlind/__init__.py` (the `_implemented` set) only when its API is complete and tested;
release 2.0 implements all of them:

| Feature | Meaning |
| --- | --- |
| `conventions` | this file's tested rules, `qttlind.conventions` |
| `qtt` | tensor trains and single-mode quantics states |
| `mpo` | MPOs and bosonic quantics operators |
| `tci` | tensor cross interpolation |
| `closed` | closed-system integrators |
| `comb` | purified comb state |
| `lindblad` | Kraus-map Lindblad integrators |
| `api` | the `System` / `LindbladSolver` user API |
| `gaussian` | the exact Gaussian backend `qttlind.gaussian` |
| `cv` | CV states, gates and measurements in quantics form, `qttlind.cv` |
| `cv_circuits` | circuits, hybrid executor and hardware models, `qttlind.circuits` |

Callers check features (`qttlind.require("lindblad")`), never a bare `import qttlind`.

## Dense linear algebra

Every contraction, QR and truncated SVD on a hot path goes through
`qttlind.core.linalg`; no other module calls `numpy.linalg` directly there. That module is the one
place a compiled kernel (for instance C++ through nanobind) may replace a NumPy routine, keeping
the same signatures, if profiling ever shows interpreter overhead to matter.
