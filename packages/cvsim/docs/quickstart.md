# Quick start

The Gaussian core arrives in milestone M1 (sessions S04–S07). States are available now:

```python
import math

from cvsim.gaussian import GaussianState

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
```
