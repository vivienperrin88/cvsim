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
```
