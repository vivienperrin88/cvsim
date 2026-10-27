# Accuracy, validation and performance

## One tolerance rule everywhere

Every `tol` is a **discarded relative weight** per truncation: singular values are dropped while
$\sum_{\text{dropped}}s_j^2\le\text{tol}\sum_js_j^2$. The relative Frobenius error of one
truncation is therefore $\sqrt{\text{tol}}$: `tol=1e-8` means $10^{-4}$ in amplitude. On the
purification index the discarded weight is exactly the trace-norm error on $\rho$.

Caps per bond kind (`chi_e`, `chi_q`, `chi_mu`) bound the cost when the tolerance alone would not.

## How results are validated

The test suite checks **values**, layer by layer, against independent references:

* dense oracles in {mod}`qttlind.reference` (`dense_ref`, `cv_dense`), which import nothing else
  from the package: every MPO, the comb, the Kraus steps (agreement to $10^{-11}$), the operator
  algebra;
* exact solutions: `scipy.linalg.expm` and `solve_ivp` (DOP853) for integrators, measured orders
  of convergence (RK4: 4, Crank–Nicolson: 2, TDVP: 1, Magnus-TDVP: 4; Kraus: 1 and 2);
* the Gaussian backend for every Gaussian channel on the comb, in both bases;
* analytic physics: exponential decay, the cat dark state, phase-noise decay, photon subtraction
  (heralding probability, $W(0)=-2/\pi$), the cubic-phase momentum shift.

## Performance

qttlind is pure Python on NumPy/SciPy. At small bond dimensions a Kraus step costs ~10-100 ms,
dominated by Python overhead; at the paper's sizes ($\chi\sim60$-$120$) LAPACK dominates and a
compiled core would gain little (see the architecture page, D1). Practical advice:

* one BLAS thread per process (`OMP_NUM_THREADS=1`), several processes for parameter sweeps;
* the largest stable `dt` (see the stability section of the Lindblad guide);
* the loosest `tol` the observable tolerates (check by halving it);
* `every` large enough that measurements do not dominate (each `e_op` is a contraction).

Paper-size runs of the tutorials (`QTTLIND_PAPER_SIZE=1`) take hours; the default quick mode runs
every notebook in minutes.
