"""
qttlind.reference.dense_ref -- dense NumPy/SciPy reference ("oracle") for the multilevel QTT
Lindblad solver.

It implements ONLY the purification axis of Moulinas & Waintal (arXiv:2605.27515):
rho = Psi Psi^dagger with Psi a dense (D x chi_mu) matrix, advanced with the first/second
order Kraus maps of Sec. V and truncated on the purification index by SVD.

Use it to:
  * validate the QTT implementation (qttlind.lindblad) at small sizes: same scheme, so the two
    must agree to ~tol,
  * run the small-size versions of the tutorial notebooks before the QTT solver exists.

No tensor networks here on purpose: this file must stay obviously correct.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

# ----------------------------------------------------------------------------- operators


def destroy(N: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1).astype(complex)


def eye(N: int) -> np.ndarray:
    return np.eye(N, dtype=complex)


def kron(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of operators (2-D) or of kets (1-D)."""
    out = np.array([1.0 + 0j]) if ops[0].ndim == 1 else np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def embed(op: np.ndarray, which: int, dims: Sequence[int]) -> np.ndarray:
    """Place a single-mode operator on mode `which` of a multi-mode space (mode 0 = leftmost)."""
    return kron(*[op if i == which else eye(d) for i, d in enumerate(dims)])


def coherent(N: int, alpha: complex) -> np.ndarray:
    n = np.arange(N)
    logfact = np.array([0.0] + list(np.cumsum(np.log(np.arange(1, N)))))
    if alpha == 0:
        amp = (n == 0).astype(complex)
    else:
        amp = np.exp(-(abs(alpha) ** 2) / 2 + n * np.log(alpha + 0j) - 0.5 * logfact)
    return amp / np.linalg.norm(amp)


def cat(N: int, alpha: complex, parity: int = +1) -> np.ndarray:
    v = coherent(N, alpha) + parity * coherent(N, -alpha)
    return v / np.linalg.norm(v)


def basis(N: int, k: int) -> np.ndarray:
    v = np.zeros(N, complex)
    v[k] = 1
    return v


def parity_op(N: int) -> np.ndarray:
    return np.diag((-1.0) ** np.arange(N)).astype(complex)


# ----------------------------------------------------------------------------- purified state


@dataclass
class Purified:
    """rho = Psi Psi^dagger,  Psi has shape (D, chi_mu)."""

    Psi: np.ndarray

    @classmethod
    def from_ket(cls, psi: np.ndarray) -> Purified:
        return cls(psi.reshape(-1, 1).astype(complex))

    @property
    def chi_mu(self) -> int:
        return self.Psi.shape[1]

    def rho(self) -> np.ndarray:
        return self.Psi @ self.Psi.conj().T

    def trace(self) -> float:
        return float(np.real(np.vdot(self.Psi, self.Psi)))

    def purity(self) -> float:
        G = self.Psi.conj().T @ self.Psi  # chi x chi Gram matrix: Tr rho^2 = ||G||_F^2
        return float(np.real(np.sum(np.abs(G) ** 2)) / self.trace() ** 2)

    def expect(self, O: np.ndarray) -> complex:
        return np.trace(self.Psi.conj().T @ (O @ self.Psi)) / self.trace()

    def normalize(self) -> None:
        self.Psi /= np.sqrt(self.trace())

    def truncate(self, rtol: float = 1e-8, chi_max: int | None = None) -> float:
        """Truncate the purification index.  Psi = U S V^dag -> U S (V^dag is a gauge on mu).
        Discards the smallest eigenvalues s^2 of rho while their cumulative weight is <= rtol * Tr rho.
        The discarded weight is exactly the trace-norm error made on rho, so rtol is a direct
        bound on the error of every expectation value of a bounded observable (relative to ||O||).
        Returns the discarded weight."""
        P = self.Psi
        if P.shape[1] > P.shape[0]:  # cheaper route when chi > D: thin QR of P^dag
            _, Rm = np.linalg.qr(P.conj().T)  # P = Rm^dag Q^dag ; Q^dag is a gauge on mu
            P = Rm.conj().T
        U, s, _ = np.linalg.svd(P, full_matrices=False)
        keep, discarded = keep_count(s, rtol, chi_max)
        self.Psi = U[:, :keep] * s[:keep]
        return discarded


def keep_count(s: np.ndarray, tol: float, chi_max: int | None = None) -> tuple[int, float]:
    """The shared truncation rule (CONVENTIONS.md), written independently of qttlind.core.conventions.keep_count.

    For singular values s sorted non-increasingly, keep the smallest k >= 1 whose discarded
    relative weight sum_{j>=k} s_j^2 / sum_j s_j^2 is <= tol; then cap k at max(1, chi_max).
    Returns (k, discarded weight actually dropped). Empty input -> (0, 0); all zeros -> (1, 0)."""
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0, 0.0
    w = s**2
    tot = w.sum()
    if tot == 0.0:
        return 1, 0.0
    tail = np.cumsum(w[::-1])[::-1]  # tail[k] = sum_{j>=k} w_j
    keep = max(1, int(np.sum(tail > tol * tot)))  # smallest k with tail[k] <= tol * tot
    if chi_max is not None:
        keep = min(keep, max(1, chi_max))
    return keep, float(w[keep:].sum() / tot)


# ----------------------------------------------------------------------------- Kraus maps


def kraus_step(
    state: Purified,
    Heff: np.ndarray,
    Ls: Sequence[np.ndarray],
    h: float,
    order: int = 2,
    rtol: float = 1e-8,
    chi_max: int | None = None,
) -> float:
    """One step of the Kraus-map integrator of Sec. V B (Cao & Lu scheme).

    order=1:  K = U + h L                         (Eq. 55)
    order=2:  K = U' + h/2 (U L + L U) + h^2/2 L^2  (Eq. 56)
    Each Kraus operator K_i maps Psi -> K_i Psi; the outer sum is a *matrix addition*
    (horizontal stacking of the mu-blocks).  Then truncate mu and renormalize trace.
    """
    # Organised exactly like Phase 7 of the guide: H_eff is applied 2 + l times, never squared.
    # Works with dense ndarrays or scipy.sparse matrices.
    P = state.Psi
    P1 = Heff @ P  # H_eff Psi
    X = P - 1j * h * P1  # U Psi
    blocks = []
    if order == 1:
        blocks.append(X)
        blocks += [np.sqrt(h) * (L @ P) for L in Ls]
    elif order == 2:
        P2 = Heff @ P1
        blocks.append(X - 0.5 * h**2 * P2)  # U' Psi
        c = np.sqrt(h / 2)
        Ys = [L @ P for L in Ls]
        for L, Y in zip(Ls, Ys):
            blocks.append(c * (Y - 1j * h * (Heff @ Y)))  # U L_k Psi
            blocks.append(c * (L @ X))  # L_k U Psi
        c2 = h / np.sqrt(2)
        for Lj in Ls:
            for Y in Ys:
                blocks.append(c2 * (Lj @ Y))  # L_j L_k Psi
    else:
        raise ValueError("order must be 1 or 2")
    blocks = [np.asarray(B) for B in blocks]
    state.Psi = np.hstack(blocks)
    disc = state.truncate(rtol=rtol, chi_max=chi_max)
    state.normalize()
    return disc


def heff(H, Ls):
    """H_eff = H - i/2 sum_k L_k^dag L_k  (Eq. 51); dense or scipy.sparse."""
    out = H * (1.0 + 0j)
    for L in Ls:
        out = out - 0.5j * (L.conj().T @ L)
    return out


@dataclass
class Result:
    times: np.ndarray
    expect: dict
    purity: np.ndarray
    chi_mu: np.ndarray
    states: list = field(default_factory=list)


def evolve(
    psi0: np.ndarray | Purified,
    H: Callable[[float], np.ndarray] | np.ndarray,
    Ls: Sequence[np.ndarray],
    t_final: float,
    h: float,
    order: int = 2,
    e_ops: dict | None = None,
    every: int = 1,
    rtol: float = 1e-8,
    chi_max: int | None = None,
    keep_states: bool = False,
) -> Result:
    """Integrate the Lindblad equation with the purified Kraus scheme.
    H may be a constant matrix or a callable H(t); for time dependence the step uses H at the
    midpoint t + h/2 (2nd-order accurate for the Kraus scheme, our choice -- the paper is silent)."""
    st = psi0 if isinstance(psi0, Purified) else Purified.from_ket(psi0)
    st = Purified(st.Psi.copy())
    st.normalize()
    e_ops = e_ops or {}
    Hc = None if callable(H) else heff(H, Ls)
    nsteps = round(t_final / h)
    ts, ex, pur, chis, states = [], {k: [] for k in e_ops}, [], [], []

    def record(t):
        ts.append(t)
        for k, O in e_ops.items():
            ex[k].append(st.expect(O))
        pur.append(st.purity())
        chis.append(st.chi_mu)
        if keep_states:
            states.append(Purified(st.Psi.copy()))

    record(0.0)
    for i in range(nsteps):
        t = i * h
        He = Hc if Hc is not None else heff(H(t + h / 2), Ls)
        kraus_step(st, He, Ls, h, order=order, rtol=rtol, chi_max=chi_max)
        if (i + 1) % every == 0:
            record((i + 1) * h)
    return Result(np.array(ts), {k: np.array(v) for k, v in ex.items()}, np.array(pur), np.array(chis), states)


# ----------------------------------------------------------------------------- exact reference


def liouvillian(H: np.ndarray, Ls: Sequence[np.ndarray]) -> np.ndarray:
    """Column-stacking vectorization: vec(A X B) = (B^T kron A) vec(X)."""
    D = H.shape[0]
    I = np.eye(D)
    Lv = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        LdL = L.conj().T @ L
        Lv += np.kron(L.conj(), L) - 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    return Lv


def exact_rho(rho0: np.ndarray, H: np.ndarray, Ls: Sequence[np.ndarray], t: float) -> np.ndarray:
    """``exp(L t) rho0``; for ``D > 24`` the action of the exponential (no ``D^2 x D^2`` expm)."""
    D = rho0.shape[0]
    Lv = liouvillian(H, Ls)
    r0 = rho0.reshape(-1, order="F").astype(np.complex128)
    if D <= 24:
        v = sla.expm(Lv * t) @ r0
    else:
        import scipy.sparse
        import scipy.sparse.linalg

        v = scipy.sparse.linalg.expm_multiply(scipy.sparse.csr_matrix(Lv) * t, r0)
    return v.reshape(D, D, order="F")


# ----------------------------------------------------------------------------- transmon helpers


def transmon_charge_basis(EC: float, EJ: float, ncut: int = 60, ng: float = 0.0):
    """``H_t = 4 EC (n - ng)^2 - EJ cos(phi)`` in the charge basis ``|n| <= ncut``. Returns ``(H, n_op)``."""
    n = np.arange(-ncut, ncut + 1)
    H = np.diag(4 * EC * (n - ng) ** 2).astype(complex)
    off = -EJ / 2 * np.ones(2 * ncut)
    H += np.diag(off, 1) + np.diag(off, -1)
    return H, np.diag(n).astype(complex)


def transmon_eigbasis(EC: float, EJ: float, nlev: int, ncut: int = 60):
    """Energies (relative to ground) and n_t matrix in the transmon eigenbasis (Eq. 71)."""
    H, nop = transmon_charge_basis(EC, EJ, ncut)
    E, V = np.linalg.eigh(H)
    V = V[:, :nlev]
    # fix gauge so that <k|n_t|k+1> is purely imaginary-positive -> real symmetric after i*
    nt = V.conj().T @ nop @ V
    return E[:nlev] - E[0], nt


def branch_analysis(
    Hfull: np.ndarray,
    a_op: np.ndarray,
    Nt_op: np.ndarray,
    Nc_op: np.ndarray,
    bare_index: Callable[[int, int], int],
    n_t_branches: int,
    n_c_max: int,
):
    """Appendix A: label dressed eigenstates ``|i_t, i_c>`` by maximal overlap.

    Returns ``dict (i_t, i_c) -> (energy, Nt, Nc)``.
    """
    E, V = np.linalg.eigh(Hfull)
    labeled = np.zeros(len(E), bool)
    out = {}
    for it in range(n_t_branches):
        bare = np.zeros(Hfull.shape[0], complex)
        bare[bare_index(it, 0)] = 1
        ov = np.abs(V.conj().T @ bare) ** 2
        ov[labeled] = -1
        k = int(np.argmax(ov))
        labeled[k] = True
        vec = V[:, k]
        out[(it, 0)] = (E[k], np.real(vec.conj() @ Nt_op @ vec), np.real(vec.conj() @ Nc_op @ vec))
        for ic in range(1, n_c_max):
            w = a_op.conj().T @ vec
            w /= np.linalg.norm(w)
            ov = np.abs(V.conj().T @ w) ** 2
            ov[labeled] = -1
            k = int(np.argmax(ov))
            labeled[k] = True
            vec = V[:, k]
            out[(it, ic)] = (E[k], np.real(vec.conj() @ Nt_op @ vec), np.real(vec.conj() @ Nc_op @ vec))
    return out
