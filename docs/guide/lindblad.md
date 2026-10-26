# Open systems: the Lindblad solver

$$
\dot\rho = -i[H,\rho] + \sum_k \Big(L_k\rho L_k^\dagger - \tfrac12\{L_k^\dagger L_k,\rho\}\Big).
$$

## The purified comb

$\rho=\Psi\Psi^\dagger$ is stored as a {class}`~qttlind.lindblad.PurifiedComb`: one backbone tensor
per mode (bonds $\chi_e$), one quantics leg per mode (bonds $\chi_q$, bits most significant
first) and a leaf carrying the purification index of dimension $\chi_\mu$. Single-mode operators
act on their leg only and never touch the backbone.

```python
comb = res.final_state
comb.bond_dims()  # {"chi_e": .., "chi_q": .., "chi_mu": ..}
comb.reduced_dm(0)  # dense reduced density matrix of mode 0
comb.purity(), comb.spectrum(), comb.mean_photons()
comb.expect({0: mpo})  # Tr(rho O) / Tr(rho)
```

## Kraus integrators

With $H_{\rm eff}=H-\tfrac i2\sum_kL_k^\dagger L_k$ and $U=1-ihH_{\rm eff}$,
$U'=U-\tfrac{h^2}2H_{\rm eff}^2$:

* `kraus1`: $K=\{U,\ \sqrt h L_k\}$ — first order;
* `kraus2`: $K=\{U',\ \sqrt{h/2}(UL_k+L_kU),\ (h/\sqrt2)L_jL_k\}$ — second order, $H_{\rm eff}$
  applied $2+l$ times per step.

$\rho\mapsto\sum_iK_i\rho K_i^\dagger$ is a concatenation of the blocks $K_i\Psi$ along $\mu$:
**positivity is preserved by construction**, whatever the truncation. Coefficients are evaluated
at the midpoint $t+h/2$.

## Stability: choosing `dt`

The Taylor propagators are explicit. A level with decay rate $\Gamma$ (eigenvalue of
$\sum_kL_k^\dagger L_k$) is damped only when $h\Gamma/2\lesssim1$. Compression noise reaches every
level of the truncated space, so $h$ must resolve the largest $\Gamma$ of the *truncated space*,
not only the populated levels: for two-photon loss $\kappa n(n-1)$ at $n=2^R-1$.

The solver records the **trace defect** $|{\rm Tr}\,K(\rho)-1|$ before renormalising: $O(h^3)$
when the step is adequate. A {class}`~qttlind.errors.TraceDefectWarning` (default threshold
$10^{-3}$) means the step is too large.

## Runs, results, checkpoints

```python
res = solver.run(
    psi0,
    t_final,
    e_ops={"P": A.parity()},
    every=50,
    track=["purity", "chi_mu", "chi_q", "chi_e", "trace_defect", "n:memory", "reduced_dm:buffer"],
    progress=lambda step, t, info: None,
    checkpoint="run.npz",
    checkpoint_every=1000,
)
```

* `Result.times`, `.expect[name]`, `.track[name]`, `.final_state`, `.metadata` (parameters,
  versions of qttlind/NumPy/SciPy/Python, git commit, wall time, discarded weight).
* `Result.save(path)` / `Result.load(path)`; `Result.to_pandas()` (needs pandas).
* **Interruption**: Ctrl-C (or `KeyboardInterrupt` from `progress`) returns the partial result
  with `interrupted=True`.
* **Checkpoints** are written atomically; `LindbladSolver.resume(path)` continues **bit for
  bit**. Solvers whose coefficients are lambdas cannot be pickled: pass them again,
  `LindbladSolver.resume(path, solver, e_ops=...)`.
* **Cutoff check**: a {class}`~qttlind.errors.CutoffWarning` once per mode when
  $\langle n\rangle+4\sqrt{\langle n\rangle}$ reaches the number of levels — results past that
  point are truncation artefacts.

## Lower-level access

{func}`qttlind.lindblad.kraus_step` and {func}`qttlind.lindblad.evolve` take lists of
{class}`~qttlind.lindblad.LocalTerm` directly (a coefficient times single-mode MPOs keyed by
mode index); {mod}`qttlind.circuits.noise` builds such lists for CV modes in the Fock or the
position basis.
