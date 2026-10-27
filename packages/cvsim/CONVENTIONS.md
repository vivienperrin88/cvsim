# Conventions

These choices are fixed once for the whole code base. They are **identical to qttlind's**
(`packages/qttlind/CONVENTIONS.md`, section *Continuous variables*) so that the two Gaussian
engines can be compared number for number (decision D2). Rules marked **[tested]** are implemented
twice, independently, in `cvsim.core.conventions` and in the oracle `cvsim.reference`, and
cross-checked by `tests/test_conventions.py`; when qttlind is installed, the contract tests also
compare them with qttlind.

*The continuous-variable rules are written in session S02.*

## Errors and warnings

* Every deliberate error derives from `cvsim.errors.CvsimError`: `InputError` (also a
  `ValueError`) for invalid arguments, with the subclasses `ShapeError` (inconsistent shapes or
  mode counts), `PhysicalityError` (a state or channel that violates quantum mechanics beyond the
  numerical tolerance) and `CalibrationError` (calibration data the model cannot produce);
  `ConvergenceError` (also a `RuntimeError`) for iterative methods that fail.
* Validate at public entry points; messages state what was expected and what was received.
* Warnings derive from `CvsimWarning`: `ApproximationWarning` when a result relies on an
  approximation (for instance moment-matched phase noise), `CutoffWarning` when a Fock cutoff is
  nearly exhausted. The test suite turns them into errors unless a test expects them.

## Randomness

* Every stochastic function takes a `numpy.random.Generator` (argument `rng`); the legacy global
  state (`np.random.seed`, `np.random.normal`, …) is never used. Tests use the seeded `rng`
  fixture. **[guarded]** by `tests/test_guards.py`.

## Architecture rules **[guarded]**

Checked by parsing every module (`tests/test_guards.py`), each guard being itself tested on
synthetic violations:

* the layer rule of `cvsim._layers` (a module imports its own layer or lower ones; `errors`
  imports nothing from cvsim; `reference` imports nothing from cvsim and is imported by tests only;
  subpackages never import the package root);
* `qttlind` is imported only in `cvsim.backends`, and every other module imports in an interpreter
  where qttlind is absent;
* `numpy.linalg` and `scipy.linalg` are used only in `cvsim.core.linalg` and the oracles;
* every module has a docstring; every public module an `__all__` that lists real names; every
  public function a docstring, with a numpydoc *Parameters* section when it takes arguments.

## Feature flags

`cvsim.FEATURES` lists the implemented capabilities, `cvsim.KNOWN_FEATURES` every planned one (with
the session that delivers it). A name enters `FEATURES` only when its API is complete and tested.
Callers check features (`cvsim.require("gaussian")`), never a bare import.
