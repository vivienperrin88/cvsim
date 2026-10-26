# qttlind

**Multilevel tensor-network simulation of open bosonic and continuous-variable quantum systems.**

qttlind implements the method of *Moulinas & Waintal, arXiv:2605.27515*: a purified density matrix
ρ = ΨΨ† stored as a **comb** tensor network — one backbone tensor per mode, one **quantics**
(binary) leg per mode, a purification leaf — advanced with positivity-preserving **Kraus maps**.
Three bond dimensions control three independent compressions: purity (χμ), inter-mode
entanglement (χe) and the quantics rank of each mode (χq). A mode with 2²⁰ Fock levels costs 20
small tensors.

On top of it: an **exact Gaussian backend**, **position-space quantics** for CV optics, and a
**hybrid Gaussian / quantics circuit emulator** with calibrated hardware models (a digital twin
of a noisy photonic processor). Pure Python on NumPy and SciPy.

```python
import numpy as np
import qttlind as ql

system = ql.System()
A = system.add_mode("memory", bits=5)
B = system.add_mode("buffer", bits=3)
H = A.dag() ** 2 * B + A**2 * B.dag() - 4.0 * (B + B.dag()) + 0.04 * (A + A.dag())
solver = ql.LindbladSolver(system, H, [np.sqrt(10.0) * B], dt=2e-3, tol=1e-8)
res = solver.run(
    system.product_state(memory=ql.states.cat(2.0), buffer=ql.states.fock(0)),
    t_final=1.0,
    e_ops={"parity": A.parity()},
    every=50,
    track=["chi_e", "chi_q", "chi_mu"],
)
res.to_pandas()
```

## Install

```bash
uv sync                      # development: locked environment with tests, notebooks, tools
pip install .                # or just the library (NumPy, SciPy)
```

```bash
uv run pytest                                  # test suite
uv run pytest --runslow                        # plus long-running tests
uv run jupyter lab notebooks/                  # tutorials (quick mode)
QTTLIND_PAPER_SIZE=1 uv run jupyter lab notebooks/   # published sizes (hours)
uv sync --group docs && uv run sphinx-build -W -b html docs docs/_build/html
```

## What is inside

| package | contents |
| --- | --- |
| `qttlind.core` | conventions, LAPACK seam, `TensorTrain`, `MPO`/`apply_mpo` (zip-up, fitting), Krylov, TCI, tree networks |
| `qttlind.bosons` | quantics Fock states and bosonic operators |
| `qttlind.closed` | RK4, Crank–Nicolson, TDVP, Magnus-TDVP |
| `qttlind.lindblad` | `PurifiedComb`, first/second-order Kraus integrators |
| `qttlind.api` | operator algebra, `System`, `LindbladSolver`, `Result`, `evolve_closed` |
| `qttlind.gaussian` | exact Gaussian backend: gates, channels, measurements, Bargmann, hafnians |
| `qttlind.cv` | quadrature grids, quantics Fourier transform, phase-space gates, measurements |
| `qttlind.circuits` | `CVMode`, noise models, `Circuit`, Gaussian / QTT / hybrid executors, `HardwareModel` |
| `qttlind.reference` | independent dense oracles used by the tests |

Tutorials `notebooks/00-13` reproduce the paper's figures (quantics ranks, MPO ranks, integrator
orders, cat-qubit Z gate, transmon ionisation) and CV benchmarks (GBS statistics, photon
subtraction, GKP states, Kerr and cubic-phase gates, cluster states with a hardware twin).

Documentation: `docs/` (Sphinx) — quick start, user guide, the architecture and design records,
conventions, tutorials and the full API reference. See `CONTRIBUTING.md` for the workflow and
`CITATION.cff` to cite.

License: MIT.
