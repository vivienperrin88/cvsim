# qttlind

**Multilevel tensor-network simulation of open bosonic and continuous-variable quantum systems.**

qttlind implements the method of *Moulinas & Waintal, arXiv:2605.27515*: a purified density matrix
$\rho = \Psi\Psi^\dagger$ is stored as a **comb** tensor network — a backbone of one tensor per
mode, a **quantics** (binary) leg per mode, a purification leaf — and advanced with positivity-
preserving **Kraus maps**. Three independent bond dimensions control the three compressions:
purity ($\chi_\mu$), inter-mode entanglement ($\chi_e$) and the quantics rank of each mode ($\chi_q$).
A mode with $2^{20}$ Fock levels costs $\log_2 2^{20} = 20$ small tensors.

On top of the paper's solver, qttlind ships an **exact Gaussian backend** and a **hybrid
Gaussian / quantics circuit emulator** for noisy continuous-variable optics, with calibrated
hardware models (a *digital twin*).

::::{grid} 2
:gutter: 3

:::{grid-item-card} Get started
:link: quickstart
:link-type: doc
Install with `uv` or `pip`, simulate a damped cat qubit in ten lines.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Fourteen notebooks reproducing the paper's figures and textbook CV benchmarks.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
Operators, solvers, tolerances, circuits and hardware models.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public module, class and function.
:::
::::

```python
import numpy as np
import qttlind as ql

system = ql.System()
a = system.add_mode("cavity", bits=5)  # 32 Fock levels, 5 quantics bits
# two-photon loss L = a^2 - alpha^2 pumps the vacuum into an even cat state
solver = ql.LindbladSolver(system, 0 * a.n(), [a**2 - 2.0], dt=1e-3, tol=1e-10)
result = solver.run(
    system.product_state(cavity=ql.states.fock(0)), t_final=1.0, e_ops={"n": a.n()}, track=["chi_q", "purity"]
)
result.to_pandas().plot()
```

```{toctree}
:hidden:
:maxdepth: 2

quickstart
installation
guide/index
tutorials/index
design/architecture
design/cv_design
conventions
api/index
changelog
contributing
citing
```
