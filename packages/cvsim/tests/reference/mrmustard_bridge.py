"""JSON-lines bridge to MrMustard, run *inside the MrMustard reference environment*.

MrMustard 0.7.3 (the current PyPI release) pins NumPy < 2 and The Walrus 0.19, so it cannot share
cvsim's environment; ``tools/setup_reference_envs.sh`` builds a separate one and the contract tests
(marker ``mrmustard``) start this script there as a subprocess. Each request is one JSON line
``{"fn": name, "args": {...}}``; the answer is ``{"ok": result}`` or ``{"error": message}``.
Arrays travel as nested lists; complex matrices as ``{"re": ..., "im": ...}``.

This file imports only NumPy and MrMustard: it must not import cvsim (it runs elsewhere).
Every quantity is in MrMustard's own units (ħ = 2 by default); the conversion is the test's job.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from mrmustard import settings
from mrmustard.lab import gates as G
from mrmustard.lab import states as S
from mrmustard.lab.abstract import State
from mrmustard.lab.detectors import Heterodyne, Homodyne
from mrmustard.physics import gaussian as g


def arr(x):
    if isinstance(x, dict) and "re" in x:
        return np.asarray(x["re"], dtype=float) + 1j * np.asarray(x["im"], dtype=float)
    return np.asarray(x, dtype=float)


def out(x):
    if x is None:
        return None
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return {"re": a.real.tolist(), "im": a.imag.tolist()}
    return a.tolist()


def fn_versions():
    import mrmustard
    import thewalrus

    return {
        "mrmustard": mrmustard.__version__,
        "thewalrus": thewalrus.__version__,
        "numpy": np.__version__,
        "hbar": float(settings.HBAR),
        "homodyne_squeezing": float(settings.HOMODYNE_SQUEEZING),
    }


GATES = {
    "rotation": lambda p: g.rotation_symplectic(p["theta"]),
    "squeezing": lambda p: g.squeezing_symplectic(p["r"], p["phi"]),
    "beam_splitter": lambda p: g.beam_splitter_symplectic(p["theta"], p["phi"]),
    "two_mode_squeezing": lambda p: g.two_mode_squeezing_symplectic(p["r"], p["phi"]),
    "quadratic_phase": lambda p: g.quadratic_phase(p["s"]),
    "controlled_z": lambda p: g.controlled_Z(p["g"]),
    "controlled_x": lambda p: g.controlled_X(p["g"]),
    "displacement": lambda p: g.displacement(p["x"], p["y"]),
}


def fn_gate(name, **p):
    return out(GATES[name]({k: np.float64(v) for k, v in p.items()}))


def fn_interferometer(U):
    U = arr(U)
    return out(G.Interferometer(num_modes=U.shape[0], unitary=U).X_matrix)


STATES = {
    "vacuum": lambda p: S.Vacuum(p["num_modes"]),
    "coherent": lambda p: S.Coherent(x=p["x"], y=p["y"]),
    "squeezed": lambda p: S.SqueezedVacuum(r=p["r"], phi=p["phi"]),
    "tmsv": lambda p: S.TMSV(r=p["r"], phi=p["phi"]),
    "thermal": lambda p: S.Thermal(nbar=p["nbar"]),
    "displaced_squeezed": lambda p: S.DisplacedSqueezed(r=p["r"], phi=p["phi"], x=p["x"], y=p["y"]),
}


def fn_state(name, **p):
    st = STATES[name](p)
    return {"cov": out(st.cov), "means": out(st.means)}


def fn_figures(cov, means, cov2=None, means2=None, transpose_modes=None):
    cov, means = arr(cov), arr(means)
    res = {
        "purity": float(g.purity(cov)),
        "entropy": float(g.von_neumann_entropy(cov)),
        "number_means": out(g.number_means(cov, means)),
        "number_cov": out(g.number_cov(cov, means)),
        "symplectic_eigenvals": out(np.sort(np.asarray(g.symplectic_eigenvals(cov)))),
    }
    if cov2 is not None:
        res["fidelity"] = float(g.fidelity(means, cov, arr(means2), arr(cov2)))
    if transpose_modes is not None:
        res["log_negativity"] = float(g.log_negativity(g.physical_partial_transpose(cov, transpose_modes)))
    return res


def fn_channel(name, **p):
    if name == "loss":
        X, Y, _ = g.loss_XYd(np.atleast_1d(p["eta"]), np.atleast_1d(p["nbar"]))
    elif name == "amplifier":
        X, Y, _ = g.amp_XYd(np.atleast_1d(p["gain"]), np.atleast_1d(p["nbar"]))
    elif name == "noise":
        X, Y = np.eye(2), g.noise_Y(np.atleast_1d(p["noise"]))
    else:
        raise KeyError(name)
    return {"X": out(X), "Y": out(Y)}


def fn_compose(X1, Y1, X2, Y2):
    X, Y, _ = g.compose_channels_XYd(arr(X1), arr(Y1), None, arr(X2), arr(Y2), None)
    return {"X": out(X), "Y": out(Y)}


def fn_cptp(cov, means, X, Y, state_modes, transf_modes):
    c, m = g.CPTP(arr(cov), arr(means), arr(X), arr(Y), None, list(state_modes), list(transf_modes))
    return {"cov": out(c), "means": out(m)}


def fn_general_dyne(cov, means, proj_cov, proj_means, modes):
    o, prob, c, m = g.general_dyne(arr(cov), arr(means), arr(proj_cov), arr(proj_means), list(modes))
    return {"outcome": out(o), "prob": float(np.real(prob)), "cov": out(c), "means": out(m)}


def fn_lab_measure(kind, cov, means, modes, **p):
    st = State(cov=arr(cov), means=arr(means))
    if kind == "homodyne":
        det = Homodyne(quadrature_angle=p["angle"], result=p["result"], modes=list(modes))
    else:
        det = Heterodyne(x=p["x"], y=p["y"], modes=list(modes))
    res = det.primal(st)
    if isinstance(res, State):
        return {"cov": out(res.cov), "means": out(res.means), "norm": float(np.real(res._norm))}
    return {"prob": float(np.real(res))}


FUNCTIONS = {name[3:]: f for name, f in globals().items() if name.startswith("fn_")}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            result = FUNCTIONS[req["fn"]](**req.get("args", {}))
            sys.stdout.write(json.dumps({"ok": result}) + "\n")
        except Exception as err:  # report, keep serving
            sys.stdout.write(json.dumps({"error": f"{type(err).__name__}: {err}"}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
