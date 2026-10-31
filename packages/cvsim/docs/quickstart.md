# Quick start

The Gaussian core (milestone M1): states, gates, the symplectic library, channels with a noise
ledger, and measurements.

```python
import math

import numpy as np

from cvsim.gaussian import GaussianState, ops
from cvsim.gaussian import symplectic as sy

# a two-mode squeezed vacuum, r = 0.5
tms = GaussianState.two_mode_squeezed(0.5)
print(tms.mean_photons())  # sinh² r per mode
print(tms.log_negativity([0]))  # 2r / ln 2 ebits

thermal = GaussianState.thermal([0.2, 1.0])
print(thermal.entropy(base=2))  # bits
print(GaussianState.coherent(0.5).fidelity(GaussianState.thermal(0.3)))

# states from Strawberry Fields / The Walrus (ħ = 2) are converted on the way in
mean, cov = tms.to_hbar(2.0)
assert GaussianState.from_hbar(mean, cov, hbar=2.0).isclose(tms)
assert math.isclose(tms.purity(), 1.0)

# gates: two orthogonally squeezed modes on a 50:50 beam splitter are entangled
s = ops.beamsplitter(GaussianState.squeezed([0.6, 0.6], [0.0, math.pi]), math.pi / 4)
print(s.log_negativity([0]))

# any interferometer as a Clements mesh: 6 Mach–Zehnder cells of depth 4 for 4 modes
mesh = sy.clements(sy.random_interferometer(np.random.default_rng(1), 4))
print(len(mesh.cells), mesh.depth())

# and any state as squeezers followed by an interferometer
recipe = sy.preparation(s)
print(recipe.r)  # [0.6, 0.6]

# a squeezing budget: 10 dB at the source, three losses and electronic noise
from cvsim.gaussian import GaussianChannel, NoiseLedger

budget = (
    NoiseLedger.start(GaussianState.squeezed(10 / 8.686))
    .channel(GaussianChannel.loss(0.90), 0, "OPO escape", "loss")
    .channel(GaussianChannel.loss(0.95), 0, "propagation", "loss")
    .channel(GaussianChannel.loss(0.92), 0, "detector QE", "loss")
    .channel(GaussianChannel.additive_noise(0.01), 0, "electronics", "electronic")
)
for row in budget.budget(mode=0, phi=0.0):  # shot-noise units; the fractions add up to 1
    print(f"{row.label:12s} {row.variance:.4f}  {row.fraction:.0%}")
print(budget.db_recovered("OPO escape"))  # what-if, not additive

# measurement: homodyne on one half of a two-mode squeezed state steers the other half
from cvsim.gaussian import measure

res = measure.homodyne(tms, mode=0, phi=0.0, efficiency=0.9, electronic=0.01, outcome=0.8)
print(res.density, res.state.mean)  # conditional state of mode 1

# programs (cvsim.ir) run on any backend (cvsim.backends)
from cvsim.backends import run
from cvsim.ir import Parameter, Program, op

prog = (
    Program(["signal", "idler"])
    >> op("Squeezed", 0, r=Parameter("r"))
    >> op("BS", (0, 1), theta=math.pi / 4)
    >> op("Loss", 1, eta=0.9, component="fibre", noise_kind="loss")
)
gaussian = run(prog.bind(r=0.5))  # Gaussian program: Gaussian backend, with a noise ledger
fock = run(prog.bind(r=0.5) >> op("Kerr", 0, kappa=0.1), cutoff=12)  # non-Gaussian: Fock backend
print(gaussian.backend, fock.backend, fock.state.mean_photons())
print(Program.from_json(prog.to_json()) == prog)
```

## Sources (S11)

A synchronously pumped OPO: the JSA of its crystal gives the supermodes, the cavity their
squeezing, and every imperfection is booked in the noise ledger. All inputs are SI values.

```python
from cvsim.core import units as u
from cvsim.hardware import excess, jsa, sources

pump = jsa.Pump.gaussian(u.angular_bandwidth(2 * u.NM, 397.5 * u.NM))  # 2 nm FWHM at 397.5 nm
crystal = jsa.PhaseMatching.sinc(gvm=400 * u.FS, gvd_signal=300 * u.FS**2, gvd_pump=400 * u.FS**2)
grid = jsa.FrequencyGrid.centered(240, 6.0e14)  # detunings from ω₀, rad/s
modes = jsa.schmidt_decomposition(jsa.joint_spectral_amplitude(pump, crystal, grid), grid, max_modes=8)
print(round(modes.schmidt_number, 1), modes.coefficients[:3])  # K ≈ 21.7

opo = sources.SPOPO.from_cavity(modes, repetition_rate=76 * u.MHZ, transmission=0.2, loss=0.01, pump=0.5)
out = opo.output(omega=u.angular(1 * u.MHZ), excess=excess.ExcessNoise(antisqueezing_db=0.5))
squeezed, antisqueezed = out.quadrature_db()
print(squeezed[:3], antisqueezed[:3])  # dB per supermode
for row in out.ledger.budget(mode=0, phi=0.0):
    print(f"{row.label:22s} {row.variance:.4f}")

# a single-pass waveguide with 0.2 dB/cm over 10 mm, 8.7 dB of lossless squeezing
guide = sources.Waveguide(modes, length=10 * u.MM, loss_db_per_cm=0.2, r0=1.0, output_coupling=0.9)
print(guide.effective_transmissivity()[:3], guide.output().quadrature_db()[0][:3])

# from a measurement back to the model: −4.5 dB / +7.0 dB with a known efficiency of 0.85
print(excess.calibrate(-4.5, 7.0, efficiency=0.85))
```
