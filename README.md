# cvsim monorepo

Two Python packages, one uv workspace, one lockfile.

| package | what | licence |
| --- | --- | --- |
| [`packages/cvsim`](packages/cvsim) | hardware-aware simulation of multimode continuous-variable quantum optics, with noise accounting | proprietary |
| [`packages/qttlind`](packages/qttlind) | multilevel tensor-network (quantics comb) Lindblad solver, Gaussian backend, CV circuits — v2.0.0 | MIT |

cvsim uses qttlind as an optional backend (non-Gaussian and Lindblad dynamics) and cross-validates
its own Gaussian engine against qttlind's.

```bash
uv sync                                   # both packages, editable, + dev and notebook tools
cd packages/cvsim && uv run pytest        # one package's tests (each has its own config)
cd packages/qttlind && uv run pytest
uv run ruff check && uv run ruff format --check     # from the root: both packages
```

Workflow, commit and review rules: [`CONTRIBUTING.md`](CONTRIBUTING.md). Development plan and
status: [`PROGRESS.md`](PROGRESS.md).
