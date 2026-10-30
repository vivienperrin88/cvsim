# Quick start

The Gaussian core arrives in milestone M1 (sessions S04–S07). States, gates and the symplectic
library are available now:

```python
import math

import numpy as np

from cvsim.gaussian import GaussianState, ops
from cvsim.gaussian import symplectic as sy

# a two-mode squeezed vacuum, r = 0.5 (loss channels arrive in S06)
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
```
