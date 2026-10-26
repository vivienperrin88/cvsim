"""Hardware noise as Lindblad terms on :class:`CVMode` objects.

* **loss** -- ``L = sqrt(kappa) a``; exact check ``gaussian.ops.loss`` with ``eta = e^{-kappa t}``;
* **thermal environment** -- ``sqrt(kappa (nbar + 1)) a`` and ``sqrt(kappa nbar) a^dag``; check
  ``loss(..., nbar=nbar)``;
* **phase noise** -- ``sqrt(gamma) n``; check ``<a>(t) = e^{-gamma t/2} <a>(0)``;
* **self-Kerr** -- ``H = K n^2``; check ``cv.gates.kerr`` with ``kappa = -K t``;
* **quadratic Hamiltonian** -- ``H = r^T H r / 2`` (xxpp); check the symplectic map
  ``S(t) = expm(Omega H t)``.

Phase noise is *not* Gaussian: the Gaussian backend cannot be its reference, the analytic decay of
``<a>`` is. Every function returns :class:`~qttlind.lindblad.kraus.LocalTerm` lists whose keys are
positions in the ``modes`` sequence of the comb, ready for :func:`evolve_lindblad`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..core.mpo import MPO
from ..errors import InputError
from ..lindblad.comb import PurifiedComb
from ..lindblad.kraus import LindbladRun, LocalTerm, evolve
from .modes import CVMode

__all__ = [
    "dephasing",
    "evolve_lindblad",
    "kerr_hamiltonian",
    "loss",
    "quadratic_hamiltonian",
    "thermal",
]

_TOL = 1e-24


def _check_rate(name: str, value: float) -> None:
    if not value >= 0:
        raise InputError(f"{name} must be >= 0, got {value}")


def loss(mode: CVMode, kappa: float, index: int = 0) -> list[list[LocalTerm]]:
    """Jump operators of pure loss at rate ``kappa`` on the mode at position ``index``."""
    _check_rate("kappa", kappa)
    return [[LocalTerm(np.sqrt(kappa), {index: mode.annihilation()})]]


def thermal(mode: CVMode, kappa: float, nbar: float, index: int = 0) -> list[list[LocalTerm]]:
    """Loss into a thermal environment of ``nbar`` photons."""
    _check_rate("kappa", kappa)
    _check_rate("nbar", nbar)
    out = [[LocalTerm(np.sqrt(kappa * (nbar + 1)), {index: mode.annihilation()})]]
    if nbar > 0:
        out.append([LocalTerm(np.sqrt(kappa * nbar), {index: mode.creation()})])
    return out


def dephasing(mode: CVMode, gamma: float, index: int = 0) -> list[list[LocalTerm]]:
    """Phase noise ``L = sqrt(gamma) n``: coherences ``rho_mn`` decay as ``e^{-gamma t (m-n)^2 / 2}``."""
    _check_rate("gamma", gamma)
    return [[LocalTerm(np.sqrt(gamma), {index: mode.number()})]]


def kerr_hamiltonian(mode: CVMode, K: float, index: int = 0) -> list[LocalTerm]:
    """``H = K n^2``."""
    n = mode.number()
    return [LocalTerm(K, {index: (n @ n).compress(_TOL)[0]})]


def quadratic_hamiltonian(modes: Sequence[CVMode], H: np.ndarray) -> list[LocalTerm]:
    """Local terms of ``H = (1/2) r^T H r`` with ``r = (x_1..x_M, p_1..p_M)`` (xxpp, ``H`` real symmetric)."""
    H = np.asarray(H, dtype=float)
    M = len(modes)
    if H.shape != (2 * M, 2 * M):
        raise InputError(f"H must be {2 * M} x {2 * M} for {M} modes, got {H.shape}")
    if not np.allclose(H, H.T, atol=1e-12):
        raise InputError("H must be symmetric")
    quads = [m.quadratures() for m in modes]

    def r(i: int) -> tuple[int, MPO]:
        return (i % M, quads[i % M][i // M])

    terms: list[LocalTerm] = []
    for i in range(2 * M):
        for j in range(2 * M):
            if H[i, j] == 0:
                continue
            (mi, ri), (mj, rj) = r(i), r(j)
            if mi == mj:
                ops = {mi: (ri @ rj).compress(_TOL)[0]}
            else:
                ops = {mi: ri, mj: rj}
            terms.append(LocalTerm(0.5 * H[i, j], ops))
    return terms


def evolve_lindblad(
    state: PurifiedComb,
    H: Sequence[LocalTerm],
    jump_ops: Sequence[Sequence[LocalTerm]],
    t_final: float,
    dt: float,
    *,
    order: int = 2,
    tol: float = 1e-12,
    caps: Mapping[str, int] | None = None,
    e_ops: Mapping[str, Sequence[LocalTerm]] | None = None,
    every: int = 1,
) -> LindbladRun:
    """Kraus integration of ``H`` and ``jump_ops`` (lists of local terms, e.g. from this module)."""
    return evolve(
        state,
        list(H),
        [list(L) for L in jump_ops],
        t_final,
        dt,
        order=order,
        tol=tol,
        caps=caps,
        e_ops=e_ops,
        every=every,
    )
