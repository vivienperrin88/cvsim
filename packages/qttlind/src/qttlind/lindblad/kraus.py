"""Kraus-map integrators of the Lindblad equation on the purified comb (paper Sec. V B).

With ``H_eff = H - (i/2) sum_k L_k^dag L_k``, ``U = 1 - i h H_eff`` and ``U' = U - (h^2/2) H_eff^2``:

    K_1 = U + h L,        K_2 = U' + (h/2)(U L + L U) + (h^2/2) L^2,

each a Kraus map ``rho -> sum_i K_i rho K_i^dag``, i.e. ``Psi -> [K_1 Psi | K_2 Psi | ...]`` on the
purification. The second-order step applies ``H_eff`` only ``2 + l`` times:

1. ``Psi1 = H_eff Psi``, ``Psi2 = H_eff Psi1``;
2. ``A = Psi - i h Psi1 - (h^2/2) Psi2`` (``= U' Psi``) and ``X = Psi - i h Psi1`` (``= U Psi``);
3. for each ``k``: ``Y_k = L_k Psi``, ``B_k = Y_k - i h H_eff Y_k`` (``= U L_k Psi``), ``C_k = L_k X``;
4. for each ``j, k``: ``D_jk = L_j Y_k``;
5. ``Psi <- [A, sqrt(h/2) B_k, sqrt(h/2) C_k, (h / sqrt 2) D_jk]`` by pairwise matrix additions,
   truncating ``mu`` after each; renormalise.

The trace defect ``|Tr K(rho) - 1|`` is recorded before renormalising: ``O(h^3)`` per step when the
step is small enough for the *populated* energies, growing step after step otherwise (then a
:class:`~qttlind.errors.TraceDefectWarning` is emitted).

Stability: the scheme is explicit. A level with decay rate ``Gamma`` (eigenvalue of
``sum_k L_k^dag L_k``) is damped only when ``h Gamma / 2 <~ 1``; compression noise reaches every
level of the cutoff, so ``h`` must resolve the largest ``Gamma`` of the truncated space
(``kappa n(n-1)`` for two-photon loss), not only the populated ones.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core.mpo import MPO
from ..errors import InputError, TraceDefectWarning
from .comb import PurifiedComb

__all__ = [
    "LindbladRun",
    "LocalTerm",
    "StepInfo",
    "TraceDefectWarning",
    "apply_terms",
    "evolve",
    "heff_terms",
    "kraus_step",
    "product_of_terms",
]

log = logging.getLogger(__name__)

#: Discarded weight when multiplying single-mode MPOs (relative error 1e-12).
MPO_TOL = 1e-24


@dataclass(frozen=True)
class LocalTerm:
    """``coeff(t) x prod_modes ops[mode]`` with single-mode MPOs; ``ops = {}`` is the identity."""

    coeff: complex | Callable[[float], complex]
    ops: Mapping[int, MPO] = field(default_factory=dict)

    def value(self, t: float) -> complex:
        return complex(self.coeff(t)) if callable(self.coeff) else complex(self.coeff)

    @property
    def is_time_dependent(self) -> bool:
        return callable(self.coeff)


Operator = Sequence[LocalTerm]  # a sum of local terms


def _mul_coeff(a, b):
    if not callable(a) and not callable(b):
        return complex(a) * complex(b)
    fa = a if callable(a) else (lambda t, a=a: a)
    fb = b if callable(b) else (lambda t, b=b: b)
    return lambda t: complex(fa(t)) * complex(fb(t))


def _conj_coeff(a):
    if callable(a):
        return lambda t: np.conj(complex(a(t)))
    return np.conj(complex(a))


def product_of_terms(x: LocalTerm, y: LocalTerm, *, tol: float = MPO_TOL) -> LocalTerm:
    """``x * y`` (``x`` applied last), mode by mode."""
    ops: dict[int, MPO] = {}
    for m in sorted(set(x.ops) | set(y.ops)):
        if m in x.ops and m in y.ops:
            ops[m], _ = (x.ops[m] @ y.ops[m]).compress(tol)
        else:
            ops[m] = x.ops.get(m) or y.ops[m]
    return LocalTerm(_mul_coeff(x.coeff, y.coeff), ops)


def _dagger(term: LocalTerm) -> LocalTerm:
    return LocalTerm(_conj_coeff(term.coeff), {m: op.dagger() for m, op in term.ops.items()})


def heff_terms(H: Operator, L: Sequence[Operator], *, tol: float = MPO_TOL) -> list[LocalTerm]:
    """The local terms of ``H_eff = H - (i/2) sum_k L_k^dag L_k`` (each ``L_k`` a sum of local terms)."""
    out = list(H)
    for Lk in L:
        for x in Lk:
            xd = _dagger(x)
            for y in Lk:
                p = product_of_terms(xd, y, tol=tol)
                out.append(LocalTerm(_mul_coeff(-0.5j, p.coeff), p.ops))
    return out


def apply_terms(
    state: PurifiedComb, terms: Operator, t: float, *, tol: float, caps: Mapping[str, int] | None = None
) -> PurifiedComb | None:
    """``sum_terms coeff(t) ops |Psi>`` (``None`` if every coefficient vanishes).

    Each term is applied with a compression of the legs it changed (cheap, local); the
    block-diagonal sum of all of them is then compressed once.
    """
    parts = []
    for term in terms:
        c = term.value(t)
        if c == 0:
            continue
        parts.append(state.apply_local(term.ops, tol=tol, caps=caps).scale(c))
    if not parts:
        return None
    return PurifiedComb.vector_sum(parts).compress(tol, caps)[0]


def _axpy(*pairs: tuple[complex, PurifiedComb | None], tol: float, caps) -> PurifiedComb:
    parts = [v.scale(c) for c, v in pairs if v is not None and c != 0]
    assert parts, "empty linear combination"
    if len(parts) == 1:
        return parts[0]
    return PurifiedComb.vector_sum(parts).compress(tol, caps)[0]


@dataclass(frozen=True)
class StepInfo:
    trace_defect: float
    discarded_mu: float
    discarded_bonds: float
    chi_e: int
    chi_q: int
    chi_mu: int


def kraus_step(
    state: PurifiedComb,
    Heff: Operator,
    L: Sequence[Operator],
    t: float,
    h: float,
    *,
    order: int = 2,
    tol: float = 1e-8,
    caps: Mapping[str, int] | None = None,
    warn_defect: float = 1e-3,
) -> tuple[PurifiedComb, StepInfo]:
    """One Kraus step from ``t`` to ``t + h`` (coefficients evaluated at the midpoint ``t + h/2``).

    ``Heff`` are the terms of the effective Hamiltonian (:func:`heff_terms`), ``L`` the jump
    operators (each a sum of local terms). ``tol`` is the discarded weight on every bond and on
    ``mu`` (a bound on the trace-norm error per step); ``caps`` per bond kind
    (``chi_e``, ``chi_q``, ``chi_mu``).
    """
    if order not in (1, 2):
        raise InputError(f"order must be 1 or 2, got {order}")
    caps = dict(caps or {})
    chi_mu_cap = caps.pop("chi_mu", None)
    tm = t + h / 2
    P = state
    P1 = apply_terms(P, Heff, tm, tol=tol, caps=caps)
    X = _axpy((1.0, P), (-1j * h, P1), tol=tol, caps=caps)
    blocks: list[PurifiedComb] = []
    if order == 1:
        blocks.append(X)
        for Lk in L:
            Y = apply_terms(P, Lk, tm, tol=tol, caps=caps)
            if Y is not None:
                blocks.append(Y.scale(np.sqrt(h)))
    else:
        P2 = apply_terms(P1, Heff, tm, tol=tol, caps=caps) if P1 is not None else None
        blocks.append(_axpy((1.0, X), (-0.5 * h**2, P2), tol=tol, caps=caps))
        c = np.sqrt(h / 2)
        Ys = [apply_terms(P, Lk, tm, tol=tol, caps=caps) for Lk in L]
        for Lk, Y in zip(L, Ys):
            if Y is None:
                continue
            HY = apply_terms(Y, Heff, tm, tol=tol, caps=caps)
            blocks.append(_axpy((c, Y), (-1j * h * c, HY), tol=tol, caps=caps))
            LX = apply_terms(X, Lk, tm, tol=tol, caps=caps)
            if LX is not None:
                blocks.append(LX.scale(c))
        c2 = h / np.sqrt(2)
        for Lj in L:
            for Y in Ys:
                if Y is None:
                    continue
                D = apply_terms(Y, Lj, tm, tol=tol, caps=caps)
                if D is not None:
                    blocks.append(D.scale(c2))
    # pairwise matrix additions, truncating mu after each: chi_mu never exceeds twice its final value
    acc = blocks[0]
    disc_bonds = 0.0
    for B in blocks[1:]:
        acc, d = acc.matrix_add(B).compress(tol, caps)
        disc_bonds += d
        acc, _ = acc.truncate_mu(tol, chi_mu_cap)
    acc, disc_mu = acc.truncate_mu(tol, chi_mu_cap)
    acc, d = acc.compress(tol, caps)
    disc_bonds += d
    tr = acc.trace()
    defect = abs(tr - 1.0) if abs(state.trace() - 1.0) < 1e-6 else abs(tr / state.trace() - 1.0)
    acc = acc.normalize()
    if defect > warn_defect:
        warnings.warn(
            f"trace defect {defect:.2e} at t = {t:.4g}: reduce the time step", TraceDefectWarning, stacklevel=2
        )
    bd = acc.bond_dims()
    return acc, StepInfo(defect, disc_mu, disc_bonds, bd["chi_e"], bd["chi_q"], bd["chi_mu"])


@dataclass
class LindbladRun:
    times: np.ndarray
    expect: dict[str, np.ndarray]
    track: dict[str, np.ndarray]
    final_state: PurifiedComb
    steps: list[StepInfo] = field(default_factory=list)


def evolve(
    state: PurifiedComb,
    H: Operator,
    L: Sequence[Operator],
    t_final: float,
    h: float,
    *,
    order: int = 2,
    tol: float = 1e-8,
    caps: Mapping[str, int] | None = None,
    e_ops: Mapping[str, Operator] | None = None,
    every: int = 1,
    warn_defect: float = 1e-3,
) -> LindbladRun:
    """Integrate from 0 to ``t_final`` with Kraus steps of size ``h``; record every ``every`` steps."""
    if not h > 0 or t_final < 0:
        raise InputError(f"need h > 0 and t_final >= 0, got {h}, {t_final}")
    Heff = heff_terms(H, L)
    e_ops = dict(e_ops or {})
    nsteps = round(t_final / h)
    times: list[float] = []
    ex: dict[str, list[complex]] = {k: [] for k in e_ops}
    track: dict[str, list[float]] = {"trace_defect": [], "chi_e": [], "chi_q": [], "chi_mu": [], "purity": []}
    steps: list[StepInfo] = []
    psi = state.normalize()

    def record(tt: float, info: StepInfo | None) -> None:
        times.append(tt)
        for k, op in e_ops.items():
            ex[k].append(psi.expect([(term.value(tt), term.ops) for term in op]))
        bd = psi.bond_dims()
        track["chi_e"].append(bd["chi_e"])
        track["chi_q"].append(bd["chi_q"])
        track["chi_mu"].append(bd["chi_mu"])
        track["purity"].append(psi.purity())
        track["trace_defect"].append(0.0 if info is None else info.trace_defect)

    record(0.0, None)
    for i in range(nsteps):
        psi, info = kraus_step(psi, Heff, L, i * h, h, order=order, tol=tol, caps=caps, warn_defect=warn_defect)
        steps.append(info)
        if (i + 1) % every == 0 or i + 1 == nsteps:
            record((i + 1) * h, info)
            log.debug("Kraus step %d/%d: %s", i + 1, nsteps, info)
    return LindbladRun(
        np.array(times), {k: np.array(v) for k, v in ex.items()}, {k: np.array(v) for k, v in track.items()}, psi, steps
    )
