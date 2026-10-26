# Quick start

## A damped oscillator

```python
import numpy as np
import qttlind as ql

system = ql.System()
a = system.add_mode("cav", bits=5)  # 2^5 = 32 Fock levels

solver = ql.LindbladSolver(
    system,
    H=1.0 * a.n(),  # omega a^dag a
    jump_ops=[np.sqrt(0.2) * a],  # loss at rate kappa = 0.2
    scheme="kraus2",
    dt=0.01,
    tol=1e-8,
)
psi0 = system.product_state(cav=ql.states.coherent(2.0))
res = solver.run(psi0, t_final=10.0, e_ops={"n": a.n()}, every=10, track=["chi_mu", "purity"])

print(res.expect["n"][-1].real, 4 * np.exp(-0.2 * 10))  # <n>(t) = |alpha|^2 e^{-kappa t}
df = res.to_pandas()  # time-indexed DataFrame
res.save("damped.npz")  # arrays, final state, metadata
```

* Operators are written as expressions of modes: `a.dag()`, `a.n()`, `a.parity()`, sums,
  products and powers, normal-ordered automatically.
* `ql.coeff(f) * op` makes a coefficient time dependent.
* `tol` is the discarded weight at every truncation; `chi_max={"chi_q": 16}` caps bonds.

## A cat qubit (two modes)

```python
G2, KB, EZ, alpha = 1.0, 10.0, 0.04, 2.0
system = ql.System()
A = system.add_mode("memory", bits=5)
B = system.add_mode("buffer", bits=3)
H = G2 * A.dag() ** 2 * B + G2 * A**2 * B.dag() - G2 * alpha**2 * (B + B.dag()) + EZ * (A + A.dag())
solver = ql.LindbladSolver(system, H, [np.sqrt(KB) * B], dt=2e-3, tol=1e-8)
res = solver.run(
    system.product_state(memory=ql.states.cat(alpha), buffer=ql.states.fock(0)),
    t_final=1.0,
    e_ops={"parity": A.parity()},
    every=50,
    track=["chi_e", "chi_q", "chi_mu"],
)
```

## Closed systems

```python
system = ql.System()
m = system.add_mode("osc", bits=6)
H = m.n() + 0.02 * m.dag() ** 2 * m**2 + ql.coeff(lambda t: 0.5 * np.cos(t)) * (m**2 + m.dag() ** 2)
res = ql.evolve_closed(
    system.product_state(osc=ql.states.coherent(1.0)),
    H,
    t_final=2.0,
    dt=0.01,
    method="magnus_tdvp",
    tol=1e-10,
    every=10,
)
psi = res.final_state  # a TensorTrain over all bits
```

Methods: `rk4` (order 4), `crank_nicolson` (2), `tdvp` (1, frozen Hamiltonian),
`magnus_tdvp` (4; commutators computed once).

## A photonic circuit

```python
from qttlind.circuits import Circuit, CVMode, HybridExecutor, HardwareModel

c = Circuit(2).squeeze(0, 0.6).beamsplitter(0, 1, 0.3).measure_fock(1, postselect=1)
model = HardwareModel(element_loss={"beamsplitter": 0.98}, detector_efficiency=0.9)
res = HybridExecutor([CVMode.fock("out", 6), CVMode.fock("tap", 4)]).run(model.compile(c))
print(res.probability, res.backend_log)  # heralding probability; when modes left the Gaussian backend
```

The Gaussian part runs exactly on covariance matrices; at the photon detection the two
correlated modes move to the comb.

## Where next

* the {doc}`tutorials <tutorials/index>` reproduce the paper's figures;
* the {doc}`user guide <guide/index>` explains tolerances, stability and performance;
* the {doc}`architecture <design/architecture>` page records why the library is built this way.
