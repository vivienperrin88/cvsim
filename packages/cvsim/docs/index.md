# cvsim

**Hardware-aware simulation of multimode continuous-variable quantum optics, with noise accounting.**

cvsim models multimode CV photonic experiments — frequency-comb and waveguide squeezers, supermode
bases, mode-selective homodyne detection and photon subtraction, reconfigurable cluster states,
Gaussian reservoir computing — and books every imperfection in a **noise ledger**.

It is developed next to qttlind, in the same repository: qttlind's quantics comb is an optional
backend for non-Gaussian and Lindblad dynamics, and its Gaussian engine can be selected in place
of cvsim's own; the two are cross-validated.

```python
import cvsim

assert cvsim.__version__
```

```{toctree}
:hidden:
:maxdepth: 2

quickstart
installation
tutorials/index
design/architecture
design/roadmap
design/equivalence
conventions
api/index
changelog
```
