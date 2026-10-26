# Systems, operators and states

## Modes

```python
system = ql.System()
a = system.add_mode("memory", bits=8)  # Fock levels 0 .. 255
b = system.add_mode("buffer", bits=4)
```

Modes are ordered by insertion: this is the order of the comb backbone and of every dense
Kronecker product (mode 0 slowest). A mode *is* its annihilation operator; `a.dag()`, `a.n()`,
`a.parity()` build the others.

## Operator algebra

Expressions are finite sums of products of single-mode factors:

```python
H = g2 * a.dag() ** 2 * b + g2 * a**2 * b.dag() + eps * (a + a.dag())
L = a**2 - alpha**2  # numbers are multiples of the identity
```

* **Normal ordering.** Single-mode monomials are kept as $a^{\dagger p}a^q$; products follow
  Wick's rule, $(a^{\dagger p}a^q)(a^{\dagger r}a^s)=\sum_k\binom qk\binom rk k!\,a^{\dagger p+r-k}a^{q+s-k}$.
  On a truncated space this differs from multiplying truncated matrices on the last level only:
  `a * a.dag()` is $a^\dagger a+1$, exact in the infinite space.
* **Equality and hashing.** Canonical forms make `(a + a.dag())**2 == a.dag()**2 + 2*a.dag()*a + a**2 + 1`.
* **Time dependence.** `ql.coeff(f) * op` multiplies by $f(t)$; `op.dag()` conjugates $f$ too
  (the conjugate wrapper is cached, so `H + H.dag()` merges its terms).
* **NumPy scalars** (`np.sqrt(kappa) * a`) work as expected.
* `op.to_dense(system)` returns the dense matrix (small systems, tests); `op.to_local_terms(system)`
  the list of {class}`~qttlind.lindblad.LocalTerm` the solvers use — single-mode MPOs, cached
  per `(bits, p, q)` (`ql.clear_mpo_cache()` empties the cache).

## Custom bases

A mode can be described in its own eigenbasis — a transmon, a fluxonium:

```python
basis = ql.CustomBasis(energies=E, ops={"n": n_matrix})  # dense, in that basis
T = system.add_mode("transmon", bits=5, basis=basis)
H = T.H0() + g * T.op("n") * (c + c.dag())
for d in T.op_offsets("n"):  # split by diagonal offset k - l
    H += ql.coeff(lambda t, d=d: np.exp(1j * d * w * t)) * T.op("n", offset=d) * c
```

Levels beyond `len(energies)` are padded with zeros; `T.level()` is the level index.

## Initial states

```python
psi0 = system.product_state(memory=ql.states.cat(2.0, parity=+1), buffer=ql.states.fock(0))
psi0 = system.state_from_dense(amplitudes, pad={"cavity": 1024})  # (d_0, d_1[, chi_mu]) array
```

State specifications (`fock`, `coherent`, `cat`, `vector`) are resolved when the number of bits
is known; unspecified modes start in $|0\rangle$. A dense array of shape `(d_0, d_1, ...)` is
zero-padded to $2^{\text{bits}}$ levels per mode; a last extra axis is a purification index.
