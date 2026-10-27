# Development plan and status

Each session is one branch, merged when every gate is green and reviewed by the owner before the
next one starts. The detailed content of every session (deliverables, verification, planned
check-ins) is in [`packages/cvsim/docs/design/roadmap.md`](packages/cvsim/docs/design/roadmap.md).

| M | session | content | status |
| --- | --- | --- | --- |
| M0 | S00 | monorepo (qttlind with history), cvsim scaffold, CI, docs, decision records | **done** |
| M0 | S01 | errors, feature flags, layer-rule and LAPACK-seam guard tests | **done** |
| M0 | S02 | conventions + independent oracle, [tested] rules, contract-test harness vs qttlind | **done** |
| M1 | S03 | `core.linalg`: Williamson, Takagi, Bloch–Messiah, stable logdet, physical projection | **done** |
| M1 | S04 | `GaussianState` | next |
| M1 | S05 | symplectic library (incl. Clements) | |
| M1 | S06 | channels `(X, Y)` + `NoiseLedger` | |
| M1 | S07 | multimode homodyne / heterodyne; full Gaussian contract suite vs `qttlind.gaussian` | |
| M2 | S08 | IR: `Program`, `Op` | |
| M2 | S09 | backend protocol; cvsim and qttlind Gaussian backends (switch); dense Fock backend | |
| M2 | S10 | PNR / click detectors | |
| M3 | S11–S15 | sources (JSA → supermodes), mode bases and pulse shaper, detectors and phase noise, TOML setups, R1/R3 | |
| M4 | S16–S19 | networks, MBQC and feed-forward, routing, R2/R6/R10 | |
| M5 | S20–S24 | Fock/LCG, MPS, subtraction hardware, cats/GKP, R5/R11 | |
| M6 | S25–S28 | noisy GBS, reservoir computing, QML, R7/R8 | |
| M7 | S29–S32 | `LindbladSpec`, Lyapunov, MPS trajectories, qttlind comb adapter, R12 | |
| M8 | S33–S35 | characterisation, R4/R9 | |
| M9 | S36–S38 | docs and notebooks, benchmarks, v1.0.0 | |
