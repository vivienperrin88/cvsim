# Continuous-variable optics

## The Gaussian backend

{mod}`qttlind.gaussian` is a complete, exact simulator of Gaussian states (mean and covariance,
xxpp ordering, $\hbar=1$, vacuum covariance $I/2$, Strawberry Fields generators):

```python
from qttlind.gaussian import GaussianState, ops, measure, fock, hafnian

s = ops.beamsplitter(ops.squeeze(GaussianState.vacuum(2), 0.6), 0.3)
s = ops.loss(s, 0.9, mode=1, nbar=0.01)
hafnian.pattern_probability(s, [1, 2])  # photon-number statistics (loop hafnians)
fock.fock_density_matrix(s, [20, 20])  # Bargmann recursion
measure.homodyne(s, mode=1, outcome=0.3)  # conditional state
```

## Position-space quantics

{mod}`qttlind.cv` represents a mode on a quadrature grid of $2^R$ points (self-dual box
$L=\sqrt{2\pi2^R}$ by default): squeezed, cat and GKP states have small quantics ranks there.
Gates are diagonal phases and quantics Fourier transforms; ladder operators are MPOs of rank ~15,
so the whole Lindblad machinery applies unchanged.

## Circuits and executors

```python
from qttlind.circuits import Circuit, CVMode, GaussianExecutor, QTTExecutor, HybridExecutor

c = Circuit(2).squeeze(0, 0.6).beamsplitter(0, 1, 0.3).measure_fock(1, postselect=1).loss(0, 0.8)
modes = [CVMode.fock("out", 6), CVMode.position("tap", 8)]
res = HybridExecutor(modes).run(c)
res.state, res.outcomes, res.probability, res.backend_log
```

* {class}`~qttlind.circuits.GaussianExecutor` — exact; refuses non-Gaussian instructions.
* {class}`~qttlind.circuits.QTTExecutor` — every mode on the comb, each in its basis.
* {class}`~qttlind.circuits.HybridExecutor` — Gaussian until an instruction needs more; then the
  **connected block of correlated modes** moves to the comb (Bargmann recursion, cut-offs chosen
  adaptively), the others stay Gaussian.
* `executor.sample(circuit, shots, rng)` draws many outcomes of a final measurement from one run.

Instructions: `squeeze`, `displace`, `rotate`, `beamsplitter`, `two_mode_squeeze`, `loss`
(with `nbar`), `measure_homodyne` (Gaussian); `kerr`, `cubic_phase`, `dephasing`,
`measure_fock`, `measure_threshold` (with dark counts). A measured mode is reset to vacuum.

On the comb, single-mode gates are MPOs (exact diagonal phases where possible, otherwise the
exponential of the generator), two-mode gates a Taylor series in sub-steps chosen from a norm
bound (never amplifying compression noise), pure loss its exact Kraus operators
$K_k=\sqrt{(1-\eta)^k/k!}\,\eta^{n/2}a^k$, phase noise a Gauss–Hermite average of rotations.

## Hardware models (digital twin)

```python
from qttlind.circuits import HardwareModel, fit_loss

model = HardwareModel(
    element_loss={"beamsplitter": 0.98, "squeeze": 0.95},
    detector_efficiency=0.85,
    dark_count_probability=1e-4,
    phase_noise_std=0.02,
    max_squeezing_db=12.0,
    thermal_nbar=0.0,
)
noisy = model.compile(ideal_circuit)  # a Circuit: every executor runs it
eta = fit_loss(model, r=0.8, measured_variance=0.12)
```

## Noise as Lindblad terms

{mod}`qttlind.circuits.noise` gives jump operators and Hamiltonians for loss, thermal noise,
dephasing, Kerr and arbitrary quadratic Hamiltonians on {class}`~qttlind.circuits.CVMode` objects,
and {func}`~qttlind.circuits.noise.evolve_lindblad` integrates them on the comb.
