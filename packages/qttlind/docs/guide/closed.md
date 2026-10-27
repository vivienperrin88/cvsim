# Closed systems

{func}`qttlind.evolve_closed` integrates the Schrödinger equation of a pure state with one of four
integrators on a tensor train over all bits (modes in order, MSB first):

| method | order | cost per step | notes |
| --- | --- | --- | --- |
| `rk4` | 4 | 4 MPO applications + compressions | explicit; $h\|H\|\lesssim2.8$ |
| `crank_nicolson` | 2 | one linear solve (two-site ALS + GMRES) | unconditionally stable |
| `tdvp` | 1 (time-dependent $H$) | two-site sweeps, Krylov exponentials | $H$ frozen at the start of the step |
| `magnus_tdvp` | 4 | TDVP with the 4th-order Magnus Hamiltonian | commutators of the time-dependent terms computed once |

```python
res = ql.evolve_closed(
    psi0, H, t_final=30.0, dt=0.01, method="magnus_tdvp", tol=1e-10, every=10, e_ops={"n": m.n()}, chi_max=64
)
res.track["chi_q"]  # largest bond of the state at each record
```

For time-independent $H$ TDVP is exact up to the projection on the current bond dimensions (the
two-site variant grows them as needed). The Krylov exponential splits its step automatically
when $\|hH\|$ is too large for its budget.

The building blocks are public: {class}`~qttlind.closed.TimeDependentMPO`,
{func}`~qttlind.closed.tdvp_step`, {class}`~qttlind.closed.MagnusTDVP`,
{func}`~qttlind.closed.rk4_step`, {func}`~qttlind.closed.crank_nicolson_step`, and in the core
{func}`~qttlind.core.mpo.apply_mpo` (naive, zip-up and variational fitting),
{func}`~qttlind.core.krylov.expm_krylov` and {func}`~qttlind.core.tci.quantics_tci`.
