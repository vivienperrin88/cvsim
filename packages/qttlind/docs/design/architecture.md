# Architecture and design decisions

This page records **what** qttlind computes, **how** the library is organised, and **why**. When something in the code looks arbitrary, the reason is here.

## 1. The problem

Superconducting circuits are a handful of bosonic modes (cavities, transmons, buffers) coupled
to each other and to a lossy environment. Their dynamics is the Lindblad equation

$$
\frac{d\rho}{dt} = -i[H,\rho] + \sum_k \Big(L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\Big).
$$

With $M$ modes truncated to $N$ Fock states each, $\rho$ has $N^{2M}$ complex entries. Two numbers
from the paper set the scale:

* two cat qubits (memory cavity with $N=40$, buffer with $N=8$, two modes per qubit):
  $16\,\text{bytes}\times(40\times8)^4 \approx 170$ GB;
* the transmon ionisation run of Sec. VII ($2^5$ transmon levels × $2^{10}$ cavity photons):
  $(2^{15})^2 = 10^9$ entries, which the original study ran on Google TPUs.

Brute force (QuTiP, Dynamiqs) is exact but its cost grows like $N^{2M}$. We need compression.

## 2. The method in one page

The paper stacks three independent compressions, each controlled by one bond dimension.

1. **Purity** ($\chi_\mu$). Write $\rho = \Psi\Psi^\dagger$ with $\Psi$ a $D\times\chi_\mu$ matrix
   ($D=N^M$). States of interest stay close to pure, so $\chi_\mu \sim 1/\mathrm{Tr}\rho^2$ is
   small. Positivity of $\rho$ is then guaranteed by construction. Memory: $D\chi_\mu$.
2. **Entanglement between modes** ($\chi_e$). Write $\Psi$ as a matrix product state over the
   modes: modes that behave like qubits exchange little information. Memory: $\sim\chi_e^2 N M$.
3. **Quantics** ($\chi_q$). Write each Fock index in binary, $n=\sum_i 2^{R-i}\sigma_i$, and
   treat the $R=\log_2 N$ bits as separate indices of a tensor train. Smooth amplitudes
   (coherent states, $\sqrt n$, polynomials) have small rank in this representation.
   Memory: $\sim\chi_q^2\log_2 N$ per mode.

Together they give a **comb** tensor network: a backbone of one tensor per mode, one quantics leg
of $R$ binary tensors hanging from each, and one leaf carrying the purification index $\mu$:

```
          χe            χe
   M(a,0) ───── M(b,0) ───── Ψ(0) ── μ
     │ χq         │ χq
     σ1           σ1
     │            │
     σ2           σ2
     ⋮            ⋮
     σR           σR
```

Time evolution is not done by an ODE solver on $\rho$ but by **Kraus maps**,
$\rho \mapsto \sum_i K_i\rho K_i^\dagger$, which map a purification to a purification:
$\Psi \mapsto [K_1\Psi,\ K_2\Psi,\ \dots]$ (concatenation along $\mu$). Truncating $\Psi$ can never
make $\rho$ non-positive.

## 3. From the method to software requirements

| # | Requirement | Consequence for the design |
| --- | --- | --- |
| R1 | Every number the library produces must be checkable. | An independent dense **oracle**; tests at every layer. |
| R2 | The three compressions are independent knobs. | Truncation rule shared by all bonds; caps per bond *kind*. |
| R3 | Large $N$ must cost $\log N$, not $N$. | Nothing in the solver may build a $2^R$ vector (except in tests and guarded helpers). |
| R4 | Physicists use it from notebooks. | A small symbolic API (`a.dag()**2 * b`) on top; plotting helpers. |
| R5 | It will be extended: other bases, integrators, TCI. | Layers with narrow interfaces; physics-agnostic core. |
| R6 | It must be fast where it matters. | Measure first; one place where dense kernels live. |

## 4. Architecture

### 4.1 Layers

```
                ┌──────────────────────────────────────────────┐
   layer 4      │ api        System, Mode, operator algebra,  │
                │            LindbladSolver, Result, plots     │
                └───────────────┬──────────────────┬───────────┘
                                │                  │
                ┌───────────────▼──────┐  ┌────────▼──────────────┐
   layer 3      │ closed               │  │ lindblad              │
                │ RK4, Krylov, TDVP    │  │ comb, Kraus steps     │
                └───────────────┬──────┘  └────────┬──────────────┘
                                │                  │
                ┌───────────────▼──────────────────▼───────────┐
   layer 2      │ bosons     quantics states, shift, ladder,   │
                │            polynomials, custom bases          │
                └───────────────────────┬──────────────────────┘
                                        │
                ┌───────────────────────▼──────────────────────┐
   layer 1      │ core       conventions, linalg, TensorTrain, │
                │            MPO, compression, (tree, TCI)      │
                └───────────────────────┬──────────────────────┘
                                        │
                                   NumPy / SciPy (LAPACK)

   beside them: reference  — dense oracle, imports only NumPy/SciPy (never the layers above)
                errors     — exception types, imported by everyone
```

**The dependency rule.** A module may import from its own layer or from lower layers, never from
a higher one. `core` knows nothing about bosons; `bosons` knows nothing about time evolution;
`closed` and `lindblad` do not know each other; only `api` sees everything.

Why it matters:

* **Testability.** `core` can be tested with random tensors, without any physics. A bug in a
  tensor-train routine shows up in `core`'s tests, not three layers up in a cat-qubit run.
* **Reuse.** The same `TensorTrain` represents a quantum state, a sampled function and (with a
  fused index) an operator. TCI and the comb are added to `core` without touching
  `bosons`.
* **Replaceability.** Swapping the integrator (TDVP for RK4, a new Kraus scheme) touches one
  layer-3 module.

### 4.2 Responsibilities

| Package | Owns | Public objects (when complete) |
| --- | --- | --- |
| `qttlind.errors` | exception and warning types | `QttlindError`, `InputError`, `ShapeError`, `ConvergenceError`, `CutoffWarning` |
| `qttlind.core.conventions` | bit order, truncation rule | `bits_of`, `index_of`, `bits_array`, `keep_count` |
| `qttlind.core.linalg` | every dense contraction / QR / SVD | `contract`, `fuse`, `qr`, `svd`, `timed` |
| `qttlind.core.tt` | chains of 3-index tensors | `TensorTrain` |
| `qttlind.core.mpo` | chains of 4-index tensors, MPO–state products | `MPO`, `apply_mpo` |
| `qttlind.bosons` | everything specific to a quantised oscillator | `states.coherent`, `operators.annihilation`, … |
| `qttlind.closed` | Schrödinger-equation integrators | `rk4_step`, `tdvp`, `evolve` |
| `qttlind.lindblad` | purified comb, Kraus integrators | `PurifiedComb`, `kraus_step` |
| `qttlind.api` | the user-facing layer | `System`, `LindbladSolver`, `Result` |
| `qttlind.reference` | dense, obviously-right implementations | `dense_ref`, `quantics` |

## 5. Design decisions

Each decision is recorded as *context → decision → consequences*, the format of an
*architecture decision record*. New decisions are added as new records.

### D1. Pure Python on NumPy/SciPy

**Context.** A Lindblad step is about a hundred applications of "MPO on a quantics leg, then
recompress". We timed that kernel in NumPy:

| Bits R | Bond χ | Kernel (ms) | Inside LAPACK/BLAS (ms) | Python overhead |
| --- | --- | --- | --- | --- |
| 5 | 4 | 0.49 | 0.36 | 26% |
| 10 | 16 | 7.0 | 6.0 | 15% |
| 14 | 60 | 176 | 165 | 6% |
| 14 | 120 | 214 | 198 | 7.5% |

**Decision.** Write everything in Python; keep dense kernels in one module (D4).

**Consequences.** At the paper's bond dimensions a compiled core would gain at most ~1.1×;
algorithmic choices gain ~10×. Development and debugging stay in one language. JAX
was rejected for the core (shapes change at every step → recompilation; its CPU SVD is LAPACK
anyway). C++ via nanobind remains the escape hatch for one kernel if profiling ever demands it.

### D2. Plain arrays with fixed axis orders, thin classes around them

**Context.** Tensor-network libraries either name indices (ITensor: contraction by matching
names) or fix axis orders by convention (most MPS codes).

**Decision.** Cores are plain `complex128` arrays with documented axis orders: TT cores
`(left, phys, right)`, MPO cores `(left, out, in, right)`. Classes (`TensorTrain`, `MPO`) add
validation and algorithms, nothing else.

**Consequences.** Zero overhead and full NumPy interoperability; the price is discipline about
axis orders, which we pay with (i) validation in constructors, (ii) small hand-checked tests of
index order, (iii) comments stating the shape after every non-trivial contraction:

```python
c = linalg.contract(w, [2], a, [1])  # (lw, o, rw, la, ra)
```

### D3. Methods return new objects

**Decision.** `tt.canonicalize(3)` returns a new `TensorTrain`; `tt` is unchanged.

**Consequences.** No aliasing bugs ("why did my initial state change?"), easy tests (compare
before and after). Cost: copies, negligible next to SVDs (D1's table). If profiling later shows
a copy matters, add a private in-place variant, never change the public semantics.

### D4. One seam to LAPACK: `core.linalg`

**Decision.** Every contraction, QR and SVD on a hot path goes through `qttlind.core.linalg`.

**Consequences.** One place to (i) time LAPACK (`with linalg.timed() as stats:`), (ii) handle
failures (gesdd → gesvd fallback), (iii) apply the truncation rule, (iv) swap in a compiled
kernel. A `grep numpy.linalg` outside that module is a code-review finding.

### D5. Conventions are code, tested twice

**Decision.** The bit order and the truncation rule live in `core.conventions`, are documented in
`CONVENTIONS.md`, and are implemented a second time, independently, in the oracle; tests compare
the two.

**Consequences.** A "tolerance" means the same thing everywhere (we found and fixed one mismatch
between helper functions before the library started — exactly the bug this prevents).

### D6. Feature flags

**Decision.** `qttlind.FEATURES` lists implemented capabilities; code that needs one calls
`qttlind.require("lindblad")`.

**Consequences.** Notebooks and downstream code switch from the oracle to the solver per
capability, during development, without `try: import` guesswork.

### D7. Errors: fail early, loudly and specifically

**Decision.** Validate at public boundaries; raise `qttlind.errors` types whose message states
what was expected and what was received; never use `assert` for user input; warn (do not fail)
for valid-but-suspicious results.

**Consequences.** A shape bug surfaces at the call that caused it, with a sentence that explains
it, instead of as a NumPy broadcasting error three functions deeper.

### D8. Oracle-driven development

**Decision.** Every algorithm is tested against a slower, simpler implementation of the same
mathematics: `np.tensordot` for `contract`, dense matrices for MPOs, `scipy.linalg.expm` for
integrators, the dense purified Kraus oracle for the Lindblad solver.

**Consequences.** Tests check *values*, not just that code runs; the oracle must be small enough
to be obviously right (`qttlind.reference` imports nothing else from the package).

### D9. Hamiltonians are lists of local terms, never global MPOs

**Decision.** $H = \sum_t c_t(t)\prod_\alpha O^{(t)}_\alpha$; each $O_\alpha$ is a small cached MPO
acting on one quantics leg. Applying $H$ = applying each term leg by leg, then summing.

**Consequences.** No multi-mode MPO construction (the hardest part of MPO libraries); the comb's
backbone is never touched by an operator. This is the paper's own prescription (Sec. II B).

### D10. Kraus maps, not an ODE solver on ρ

**Decision.** Integrate with the first/second-order Kraus schemes of Sec. V.

**Consequences.** Positivity by construction; the purification index absorbs mixing. Price: the
Taylor propagators are only conditionally stable, which we monitor with the trace
defect.

### D11. A general tree, configured as a comb

**Decision.** Implement a tree tensor network once; the comb is one configuration of it, a
single-mode QTT another.

**Consequences.** One canonicalisation and compression code path for every geometry; the
single-mode code of Parts III–V is the special case tested first.

## 6. Numerical policies

* **Precision.** `complex128` everywhere. Real inputs are promoted at construction.
* **Tolerances.** A compression tolerance `tol` is a discarded relative weight per bond: the
  squared relative Frobenius error. Comparisons in tests use
  `np.testing.assert_allclose(actual, desired, rtol, atol)` with tolerances justified by the
  computation (round-off ~ $10^{-15}$ × condition, truncation ~ $\sqrt{\text{tol}}$), never
  "whatever makes it pass".
* **Determinism.** Random tests use a seeded `numpy.random.Generator` (the `rng` fixture);
  Hypothesis tests are reproducible from their seed.
* **Overflow and cancellation.** Compute in log space when factorials or exponentials appear; sum small terms first.
* **Stability before speed.** Know the stability limit of every time stepper and
  monitor a cheap diagnostic (trace defect, norm drift) at run time.

## 7. Tooling, and why each tool

| Tool | Job | Why this one |
| --- | --- | --- |
| **uv** | Python versions, virtual env, dependency resolution, lockfile | One fast tool; `uv sync` reproduces the exact environment from `uv.lock` |
| **src layout** | `src/qttlind/` instead of `qttlind/` at the root | Tests import the *installed* package, so packaging mistakes show up immediately |
| **pyproject.toml** | Metadata, dependencies, tool configuration | One file, standard (PEP 517/621/735) |
| **pytest** | Tests, fixtures, parametrisation, markers | De-facto standard; the library plugs into its hooks |
| **Hypothesis** | Property-based tests | Finds edge cases you did not think of (empty arrays, `bits=62`) |
| **ruff** | Lint + format | One fast tool replacing flake8, isort, black |
| **mypy** | Static type checking | Catches wrong argument types before the run |
| **pre-commit** | Runs ruff and friends on every commit | Keeps the history clean without thinking about it |
| **GitHub Actions** | CI: lint, type-check, test on every push | `main` stays green |

Dependency groups (`[dependency-groups]` in `pyproject.toml`) separate what the *library* needs
(NumPy, SciPy) from what *development* needs (`dev`: pytest, ruff, …) and what *notebooks* need
(`notebooks`: JupyterLab, matplotlib). Users of the library get only the first.

## Glossary

| Term | Meaning |
| --- | --- |
| core | One tensor of a tensor train / MPO / tree |
| bond (dimension) | Index shared by two neighbouring cores (its size, χ) |
| physical index | Index of the represented vector (here a bit σ ∈ {0,1}) |
| quantics | Binary encoding of a large index into many 2-valued indices |
| canonical form | Gauge where all cores but one (the centre) are isometries |
| purification | Ψ with ρ = ΨΨ†; its extra index is μ |
| Kraus map | ρ ↦ Σ K_i ρ K_i†; completely positive |
| oracle | Slow, simple reference implementation used by tests |
