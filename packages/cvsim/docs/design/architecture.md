# Architecture and design decisions

This page records **what** cvsim computes, **how** the library is organised, and **why**. Decisions
are *architecture decision records* (context → decision → consequences), numbered D1, D2, … and
never rewritten: a changed decision gets a new record that supersedes the old one.

## 1. The problem

Multimode CV experiments, typified by frequency-comb and pulse-by-pulse platforms, produce tens to
thousands of squeezed modes from one source (a synchronously pumped OPO or a single-pass
waveguide), entangle them through a choice of **measurement basis** (pulse shaping of the local
oscillator) rather than through gates, and add non-Gaussianity by **mode-selective** photon
subtraction. What limits them is a long list of small imperfections: escape efficiency, spectral
impurity, local-oscillator mode mismatch, detector efficiency and electronic noise, phase-lock
jitter, cross-talk between pulses, feed-forward latency.

A simulator for this platform must therefore (i) be exact and fast for multimode Gaussian states,
(ii) describe modes by their **spectral/temporal shape**, not just an index, (iii) account for every
imperfection separately, and (iv) reach non-Gaussian and open-system dynamics when needed.

qttlind already covers (iv) with its quantics comb and covers (i) for mode-indexed circuits; its CV
design record states explicitly that it does not cover multimode spectral decompositions or mode
mismatch beyond extra modes and loss (`packages/qttlind/docs/design/cv_design.md`, §5). That gap is
cvsim's purpose.

## 2. Requirements

| # | Requirement | Consequence |
| --- | --- | --- |
| R1 | Every number must be checkable. | Independent dense oracle; **two** independent Gaussian engines compared (D2). |
| R2 | Every imperfection must be attributable. | Channels carry provenance; a `NoiseLedger` books their contributions. |
| R3 | Modes are spectral objects. | Mode bases (Hermite–Gauss, pixel, supermode) are first-class; overlaps define mismatch. |
| R4 | Hardware is data. | Setups are serialisable component models (TOML), not code. |
| R5 | Same rigour and tooling as qttlind. | D3. |
| R6 | Non-Gaussian and Lindblad dynamics when needed. | Backend protocol; qttlind as an optional backend (D2). |

## 3. Layers

```
layer 4  protocols    networks · mbqc · routing · gbs · nongaussian · reservoir · qml
         experiments  Setup → Program, noise-ledger reports, synthetic data
         backends     protocol + gaussian · fock · lcg · mps · qttlind (optional)
layer 3  hardware     sources · mode bases · detectors · subtraction · feed-forward · phase · setup
         characterise estimation, physical projection, MLE, supermode retrieval, witnesses
layer 2  gaussian     states, symplectic library, channels (X, Y), multimode measurements
         fock · lcg · mps      non-Gaussian representations
         lindblad     backend-agnostic LindbladSpec, Gaussian Lyapunov, dense reference
layer 1  core         conventions, units, linalg (the LAPACK seam), validation, RNG policy, ir
aside    reference    dense oracles: import NumPy/SciPy only, never cvsim
         errors       exception and warning types, imported by everyone
```

**Dependency rule.** A module imports from its own layer or lower ones, never higher; `reference`
imports nothing from cvsim. From S01 the rule is enforced by a test that parses every import.

## 4. Design decisions

### D1. cvsim complements qttlind; it does not duplicate its tensor networks

*Context.* qttlind provides quantics tensor trains, the purified comb, Kraus-map Lindblad
integration and a hybrid Gaussian/comb circuit executor. Re-implementing them would double the
maintenance for no gain.
*Decision.* cvsim never re-implements the comb. Non-Gaussian, many-level and Lindblad problems that
exceed cvsim's own small backends (dense Fock, LCG, native MPS) are delegated to qttlind through
an adapter in `cvsim.backends.qttlind`.
*Consequences.* qttlind is an optional extra (`cvsim[qttlind]`); everything else works with NumPy
and SciPy alone. Features qttlind lacks (an `interferometer` or general `(X, Y)` channel
instruction, multimode homodyne) are **worked around in the adapter** — Clements decomposition into
beam splitters and rotations, explicit refusal with a precise error otherwise — and listed in
`docs/design/qttlind_interface.md` for a possible qttlind 2.1, decided later.

### D2. An independent Gaussian engine, cross-validated against qttlind's, and switchable

*Context.* Both libraries need an exact Gaussian engine. Re-using `qttlind.gaussian` would save
code; writing a second one, from the same conventions but sharing no code, gives a differential
test of both (the reasoning of qttlind's D19).
*Decision.* `cvsim.gaussian` is written independently. A **contract test suite** (marker
`qttlind`) checks that the two engines agree on every gate, channel, measurement and figure of
merit, on hand-checked cases, on random multimode inputs and on Hypothesis-generated ones, to
round-off tolerances. A Gaussian **backend switch** (`backend="cvsim"` default, or `"qttlind"`)
runs the same program on either engine.
*Consequences.* A convention drift or a bug in either library fails the contract suite, and the
dense oracle tells which side is wrong. The switch is only exposed for operations both engines
support; the others raise an `InputError` naming the missing operation.

### D3. qttlind's engineering rules apply unchanged

*Decision.* cvsim adopts qttlind's decisions D1–D8 as they stand: pure Python on NumPy/SciPy;
plain arrays with documented axis orders and thin classes; methods return new objects; one seam to
LAPACK (`cvsim.core.linalg`); conventions as code, tested twice; feature flags
(`cvsim.FEATURES`, `cvsim.require`); errors that fail early, loudly and specifically;
oracle-driven development. Same tools: uv, `src/` layout, setuptools, pytest with strict markers,
Hypothesis, ruff, mypy, pre-commit, Sphinx (furo, MyST-NB), GitHub Actions.
*Consequences.* A contributor of one package can review the other. cvsim adds two **automated**
guards that qttlind enforces by review: the layer rule and the LAPACK seam are tests (S01).
mypy is stricter (`disallow_untyped_defs`), coverage has a floor (90 % of branches).

### D4. Same CV conventions as qttlind

*Decision.* ħ = 1, `x = (a + a†)/√2`, xxpp ordering, `V = ⟨{Δr, Δr}⟩/2`, vacuum `V = I/2`,
Strawberry-Fields gate generators, channels `d → X d`, `V → X V Xᵀ + Y`, homodyne of
`x_φ = cos φ x + sin φ p`.
*Consequences.* The contract tests of D2 are possible at all. The interview course package `cvq`
uses other squeezing and beam-splitter parametrisations; it stays a separate teaching code and is
not absorbed.

### D5. One repository, two packages

*Context.* qttlind is a backend of cvsim and is developed by the same person.
*Decision.* A uv workspace: `packages/qttlind` (imported with its full history) and
`packages/cvsim`, one `uv.lock`, CI and pre-commit at the root, each package checked from its own
directory with its own configuration.
*Consequences.* A change to qttlind and the cvsim code that needs it can land in one branch. The
packages stay independently buildable: the CI wheel job installs each one alone.

### D6. Private by default

*Decision.* cvsim is proprietary ("All rights reserved"); its metadata carries
`Private :: Do Not Upload`, which PyPI rejects, so it cannot be published by accident. qttlind
keeps its own MIT licence.

### D7. Noise is booked, not just applied

*Decision.* Every channel applied by cvsim carries its provenance (component, imperfection kind);
a `NoiseLedger` records each one's contribution to the added noise and loss of every mode.
*Consequences.* A squeezing or fidelity budget can be decomposed term by term — exactly for
Gaussian channels, to first order otherwise, and labelled as such.

### D8. Development in sessions, with check-ins

*Decision.* The work is split into sessions S00–S38 (`PROGRESS.md`), each one branch with its tests
and changelog entry, merged with `--no-ff` only when every gate is green, and reviewed by the
project owner before the next session starts.
