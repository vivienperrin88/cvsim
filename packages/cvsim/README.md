# cvsim

**Hardware-aware simulation of multimode continuous-variable quantum optics, with noise accounting.**

cvsim models multimode CV photonic experiments — frequency-comb and waveguide squeezers, supermode
bases, mode-selective homodyne detection and photon subtraction, reconfigurable cluster states,
Gaussian reservoir computing — with every imperfection booked in a **noise ledger**: where did the
squeezing go? Pure Python on NumPy and SciPy.

It lives next to [qttlind](../qttlind) in the same repository: cvsim owns the multimode, spectral
and hardware layers; qttlind's quantics comb is an optional backend for non-Gaussian and Lindblad
dynamics, and its Gaussian backend can be selected instead of cvsim's own (the two are
cross-validated).

> Status: pre-alpha, built session by session (`PROGRESS.md` at the repository root).
> Proprietary — see `LICENSE`.

## Install

```bash
uv sync                      # from the repository root: both packages + dev and notebook tools
cd packages/cvsim
uv run pytest                # this package's tests
uv run pytest --runslow      # plus long-running tests
```

```python
import cvsim

print(cvsim.__version__)
```
