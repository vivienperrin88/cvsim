"""
tutorial_utils.py -- shared helpers for the qttlind tutorial notebooks.

* Backend detection and run size: notebooks use the QTT solver (``qttlind.FEATURES``) in a quick
  mode by default -- reduced sizes, QTT runs cross-checked against the dense NumPy oracle
  ``qttlind.reference.dense_ref`` -- and the published sizes when ``QTTLIND_PAPER_SIZE=1``.
* Small pedagogical NumPy implementations of the quantics building blocks (TT-SVD, shift MPO,
  quantics ranks of operators, tree "comb" ranks of a dense purification) so the notebooks can
  show the *ideas* before the QTT solver exists.
* Wigner function from a (reduced) density matrix via the position representation.

Conventions (CONVENTIONS.md): bits most significant first; every ``tol`` is a discarded
relative weight per bond (the square of the relative Frobenius error).
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import qttlind as ql
except ImportError:  # source checkout, package not installed: use the in-tree sources
    sys.path.insert(0, os.path.join(_HERE, "..", "src"))
    import qttlind as ql  # noqa: E402

from qttlind.reference import dense_ref as dr  # noqa: E402
from qttlind.reference.quantics import bits_of, index_of  # noqa: E402,F401

#: Capabilities the tutorials call into (all present in qttlind >= 2.0).
TUTORIAL_FEATURES = frozenset({"qtt", "mpo", "closed", "comb", "lindblad", "api"})
HAVE_QTTLIND = TUTORIAL_FEATURES <= ql.FEATURES

FIGDIR = os.path.join(_HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


PAPER_SIZE_REQUESTED = os.environ.get("QTTLIND_PAPER_SIZE") == "1"


def backend_banner() -> str:
    if HAVE_QTTLIND:
        mode = ("published sizes (QTTLIND_PAPER_SIZE=1: hours)" if PAPER_SIZE_REQUESTED
                else "quick mode (set QTTLIND_PAPER_SIZE=1 for the published sizes)")
        return f"qttlind {ql.__version__}, QTT solver: {mode}."
    missing = sorted(TUTORIAL_FEATURES - ql.FEATURES)
    return (f"qttlind {ql.__version__} implements {sorted(ql.FEATURES)}; the tutorials also need {missing}. "
            "Using the dense NumPy oracle at reduced sizes.")


@contextmanager
def timer(label: str = "", store: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = dt
    print(f"[{label}] {dt:.2f} s")


def savefig(fig, name: str):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

# ----------------------------------------------------------------------------- quantics basics
# bits_of / index_of come from qttlind.reference.quantics (MSB first, Eq. 4).

def _keep(s: np.ndarray, tol: float, chi_max: int | None) -> int:
    """The shared truncation rule: tol = discarded relative weight (CONVENTIONS.md)."""
    return dr.keep_count(s, tol, chi_max)[0]


def tt_svd(v: np.ndarray, R: int, tol: float = 1e-24, chi_max: int | None = None, d: int = 2):
    """Dense vector of length d^R -> list of cores (chi_l, d, chi_r), left-to-right (MSB first).

    tol = discarded relative weight per bond (1e-24 <-> relative error 1e-12 per bond)."""
    cores, r = [], 1
    M = np.asarray(v, dtype=complex).reshape(1, -1)
    for _ in range(R - 1):
        M = M.reshape(r * d, -1)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        k = _keep(s, tol, chi_max)
        cores.append(U[:, :k].reshape(r, d, k))
        M = s[:k, None] * Vh[:k]
        r = k
    cores.append(M.reshape(r, d, 1))
    return cores


def tt_to_dense(cores) -> np.ndarray:
    out = cores[0]
    for c in cores[1:]:
        out = np.tensordot(out, c, axes=([-1], [0]))
    return out.reshape(-1)


def tt_ranks(cores) -> list[int]:
    return [c.shape[2] for c in cores[:-1]]


def matrix_to_qtt_mpo(A: np.ndarray, R: int, tol: float = 1e-24, chi_max: int | None = None):
    """2^R x 2^R matrix -> MPO cores (chi_l, 4, chi_r) with bit-interleaved (sigma_i, sigma_i')."""
    T = np.asarray(A).reshape([2] * (2 * R))
    T = T.transpose([x for i in range(R) for x in (i, R + i)])
    return tt_svd(T.reshape(-1), R, tol=tol, chi_max=chi_max, d=4)


def mpo_to_dense(cores, R: int) -> np.ndarray:
    T = tt_to_dense(cores).reshape([2] * (2 * R))
    T = T.transpose(list(range(0, 2 * R, 2)) + list(range(1, 2 * R, 2)))
    return T.reshape(2 ** R, 2 ** R)


def shift_mpo(R: int, m: int = 1):
    """Exact rank-2 MPO of T_{n,n'} = delta_{n+m, n'} (binary adder with carry, MSB first).
    Returns cores W[c_left, c_right, sigma, sigma'] and the boundary carries."""
    mb = bits_of(m, R)
    Ws = []
    for i in range(R):
        W = np.zeros((2, 2, 2, 2))
        for cr in (0, 1):
            for s in (0, 1):
                tot = s + cr + mb[i]
                W[tot // 2, cr, s, tot % 2] = 1.0
        Ws.append(W)
    return Ws  # left boundary: carry-out of MSB must be 0; right boundary: carry-in of LSB = 0


def shift_mpo_dense(R: int, m: int = 1) -> np.ndarray:
    Ws = shift_mpo(R, m)
    full = np.einsum("a,abst->bst", np.array([1.0, 0.0]), Ws[0])
    for W in Ws[1:]:
        full = np.einsum("c...,cdst->d...st", full, W)
    full = np.einsum("d...,d->...", full, np.array([1.0, 0.0]))
    full = full.transpose(list(range(0, 2 * R, 2)) + list(range(1, 2 * R, 2)))
    return full.reshape(2 ** R, 2 ** R)

# ----------------------------------------------------------------------------- comb ranks of a purification

def _rank(M: np.ndarray, tol: float) -> int:
    """Numerical rank with the solver's rule: drop the smallest s^2 while their sum <= tol * sum(s^2)."""
    s = np.linalg.svd(M, compute_uv=False)
    return _keep(s, tol, None) if s.size else 1


def comb_ranks(Psi: np.ndarray, bits: list[int], tol: float = 1e-8) -> dict:
    """Bond dimensions of the paper's comb (Fig. 8) for a dense purification Psi (D x chi_mu).

    Modes are ordered left to right, the purification node hangs after the last mode.
    Every tree edge splits the indices in two; its optimal bond dimension is the numerical
    rank of the corresponding unfolding. `tol` = discarded relative weight per cut (the solver's rule)."""
    M = len(bits)
    R_tot = sum(bits)
    chi = Psi.shape[1]
    T = Psi.reshape([2] * R_tot + [chi])
    offs = np.cumsum([0] + bits)

    def unfold(S):
        rest = [i for i in range(R_tot + 1) if i not in S]
        A = T.transpose(list(S) + rest)
        return A.reshape(int(np.prod([T.shape[i] for i in S])), -1)

    out = {"chi_mu": _rank(Psi, tol), "chi_e": [], "chi_q": []}
    for a in range(M - 1):  # backbone edges between mode a and a+1 (mu sits at the far end)
        out["chi_e"].append(_rank(unfold(list(range(0, offs[a + 1]))), tol))
    for a in range(M):
        leg = []
        for i in range(bits[a]):  # i=0: leg<->backbone edge (all bits), then internal edges
            S = list(range(offs[a] + i, offs[a + 1]))
            leg.append(_rank(unfold(S), tol))
        out["chi_q"].append(leg)
    return out


def compression_stages(Psi: np.ndarray, bits: list[int], tol: float = 1e-8) -> dict:
    """Number of complex entries needed at each compression stage (Fig. 1)."""
    D = Psi.shape[0]
    r = comb_ranks(Psi, bits, tol)
    Ns = [2 ** b for b in bits]
    e = [1] + r["chi_e"] + [None]
    # stage 2: MPS over modes with full local dimension, mu node at the end
    e_last = _rank(Psi, tol)  # bond into the mu node = chi_mu (mu node is a leaf)
    e[-1] = e_last
    mps = sum(e[a] * Ns[a] * e[a + 1] for a in range(len(bits))) + e_last * r["chi_mu"]
    # stage 3: comb = backbone nodes (e_a, e_a+1, q_a) + legs + mu node
    comb = e_last * r["chi_mu"]
    for a, b in enumerate(bits):
        q = r["chi_q"][a] + [1]
        comb += e[a] * e[a + 1] * q[0]
        comb += sum(q[i] * 2 * q[i + 1] for i in range(b))
    return {"full rho": D ** 2, "+ purity": D * r["chi_mu"], "+ entanglement": mps,
            "+ quantics": comb, "ranks": r}

# ----------------------------------------------------------------------------- Hermite functions, Wigner

def hermite_functions(nmax: int, x: np.ndarray) -> np.ndarray:
    """phi_n(x), n < nmax, Eq. 13, by the stable normalized recurrence. Shape (nmax, len(x))."""
    x = np.asarray(x, float)
    phi = np.zeros((nmax, x.size))
    phi[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if nmax > 1:
        phi[1] = np.sqrt(2.0) * x * phi[0]
    for n in range(1, nmax - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * x * phi[n] - np.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def wigner(rho: np.ndarray, re_alpha: np.ndarray, im_alpha: np.ndarray, oversample: int = 2) -> np.ndarray:
    """Wigner function W(alpha) on a grid of Re(alpha) x Im(alpha), alpha = (x + i p)/sqrt(2).

    W(x, p) = (1/pi) Int dy <x+y|rho|x-y> exp(-2 i p y).  Normalised so that Int W d^2alpha = 1.
    Returns array of shape (len(im_alpha), len(re_alpha))."""
    N = rho.shape[0]
    xs = np.sqrt(2) * np.asarray(re_alpha, float)
    ps = np.sqrt(2) * np.asarray(im_alpha, float)
    xmax = max(np.abs(xs).max(), np.sqrt(2 * N + 1)) + 4.0
    dx = min(xs[1] - xs[0] if xs.size > 1 else 0.1, 0.5 / np.sqrt(2 * N + 1)) / oversample
    grid = np.arange(-2 * xmax, 2 * xmax + dx / 2, dx)
    phi = hermite_functions(N, grid)
    rx = phi.T @ rho @ phi                   # <x|rho|x'> on the grid
    W = np.zeros((ps.size, xs.size))
    G = grid.size
    for j, x0 in enumerate(xs):
        c = int(round((x0 - grid[0]) / dx))
        smax = min(c, G - 1 - c)
        s = np.arange(-smax, smax + 1)
        f = rx[c + s, c - s]                 # rho(x+y, x-y), y = s dx
        W[:, j] = np.real(np.exp(-2j * np.outer(ps, s * dx)) @ f) * dx / np.pi
    return W * 2.0                            # d^2 alpha = dx dp / 2


def reduced_dm(Psi: np.ndarray, dims: list[int], keep: int) -> np.ndarray:
    """Reduced density matrix of mode `keep` from a dense purification Psi (D x chi)."""
    chi = Psi.shape[1]
    T = Psi.reshape(list(dims) + [chi])
    T = np.moveaxis(T, keep, 0).reshape(dims[keep], -1)
    rho = T @ T.conj().T
    return rho / np.trace(rho).real

# ----------------------------------------------------------------------------- transmon + cavity (Sec. VII)

TRANSMON_PARAMS = dict(EC=0.280, EJ_over_EC=50.0, fr=7.5, g=0.250, kappa=0.020, eps_d=0.280)
"""Paper values in GHz (E/h and omega/2pi). Convert to rad/ns with 2*pi."""


def transmon_cavity_model(n_t: int, n_c: int, params: dict | None = None, ncut: int = 80):
    """Sparse operators of Eq. (70) in the transmon eigenbasis x cavity Fock basis.

    Angular units: energies in rad/ns, time in ns.  Returns a dict with
    H0 (undriven), V = -i (a - a^dag) (multiplies eps_d sin(w_d t)), L (cavity loss),
    Nt, Nc, a, transmon energies Et (GHz) and the n_t matrix."""
    import scipy.sparse as sp
    p = dict(TRANSMON_PARAMS, **(params or {}))
    tw = 2 * np.pi
    Et, nt = dr.transmon_eigbasis(p["EC"], p["EJ_over_EC"] * p["EC"], n_t, ncut=ncut)
    It, Ic = sp.identity(n_t, format="csr"), sp.identity(n_c, format="csr")
    a1 = sp.csr_matrix(dr.destroy(n_c))
    a = sp.kron(It, a1, format="csr")
    Ht = sp.kron(sp.diags(tw * Et), Ic, format="csr")
    n = sp.kron(sp.csr_matrix(nt), Ic, format="csr")
    adag = a.conj().T.tocsr()
    H0 = Ht + tw * p["fr"] * (adag @ a) - 1j * tw * p["g"] * (n @ (a - adag))
    V = -1j * (a - adag)
    Nt = sp.kron(sp.diags(np.arange(n_t, dtype=float)), Ic, format="csr")
    Nc = (adag @ a).tocsr()
    L = np.sqrt(tw * p["kappa"]) * a
    return dict(H0=H0.tocsr(), V=V.tocsr(), L=L.tocsr(), Nt=Nt, Nc=Nc, a=a, Et=Et, nt=nt,
                eps_d=tw * p["eps_d"], dims=[n_t, n_c], params=p)


def dressed_states(model: dict, n_branches: int = 4, n_c_max: int = 4):
    """Branch-analysis labels (Appendix A) of the undriven H0. Returns (labels, eigvecs dict)."""
    H = model["H0"].toarray()
    n_c = model["dims"][1]
    E, V = np.linalg.eigh(H)
    a = model["a"].toarray()
    labeled = np.zeros(len(E), bool)
    lab, vecs = {}, {}
    for it in range(n_branches):
        bare = np.zeros(H.shape[0], complex)
        bare[it * n_c] = 1
        ov = np.abs(V.conj().T @ bare) ** 2
        ov[labeled] = -1
        k = int(np.argmax(ov))
        labeled[k] = True
        vec = V[:, k]
        for ic in range(n_c_max):
            if ic > 0:
                w = a.conj().T @ vec
                ov = np.abs(V.conj().T @ (w / np.linalg.norm(w))) ** 2
                ov[labeled] = -1
                k = int(np.argmax(ov))
                labeled[k] = True
                vec = V[:, k]
            lab[(it, ic)] = (E[k],
                             float(np.real(vec.conj() @ (model["Nt"] @ vec))),
                             float(np.real(vec.conj() @ (model["Nc"] @ vec))))
            vecs[(it, ic)] = vec.copy()
    return lab, vecs

# ----------------------------------------------------------------------------- driven Kerr oscillator (Sec. III B)

KERR = dict(omega0=1.0, K=1 / 25, alpha0=4.0)


def kerr_drive(t):
    """f(t) of Eq. (34)."""
    return 0.02 * np.cos(np.sqrt(2) * t) + 2 * np.pi / 100 * np.cos(np.sqrt(3) * t)


def driven_kerr(R: int, omega0: float = KERR["omega0"], K: float = KERR["K"]):
    """H(t) = H0 + f(t) H1 with H0 = w0 n + K/2 a^dag2 a^2, H1 = a^2 + a^dag2 (Eq. 32). Sparse."""
    import scipy.sparse as sp
    N = 2 ** R
    n = np.arange(N, dtype=float)
    H0 = sp.diags(omega0 * n + K / 2 * n * (n - 1)).astype(complex).tocsr()
    a = sp.csr_matrix(dr.destroy(N))
    H1 = (a @ a + (a @ a).conj().T).tocsr()
    return H0, H1


def magnus4_evolve(psi0, H0, H1, f, t_final, h, t_out=None):
    """Dense reference: 4th-order Magnus exponential integrator (Eqs. 27-31) with expm_multiply.
    Returns (times, states) at the requested output times (multiples of h)."""
    from scipy.sparse.linalg import expm_multiply
    C = (H0 @ H1 - H1 @ H0).tocsr()
    nsteps = int(round(t_final / h))
    out_steps = set(int(round(t / h)) for t in (t_out if t_out is not None else [t_final]))
    psi = np.asarray(psi0, complex).copy()
    times, states = [], []
    if 0 in out_steps:
        times.append(0.0); states.append(psi.copy())
    c1, c2 = (3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6
    for k in range(nsteps):
        t = k * h
        f1, f2 = f(t + c1 * h), f(t + c2 * h)
        HO = H0 + 0.5 * (f1 + f2) * H1 + 1j * np.sqrt(3) * h / 12 * (f2 - f1) * C
        psi = expm_multiply(-1j * h * HO, psi)
        if k + 1 in out_steps:
            times.append((k + 1) * h); states.append(psi.copy())
    return np.array(times), states


class FastSum:
    """H(t) = sum_i c_i(t) M_i on a shared sparsity pattern: one vector op per time step."""

    def __init__(self, mats):
        import scipy.sparse as sp
        mats = [sp.csr_matrix(M) for M in mats]
        pattern = sum(abs(M) for M in mats).tocsr()
        pattern.sort_indices()
        self.shape = pattern.shape
        self.indptr, self.indices = pattern.indptr.copy(), pattern.indices.copy()
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        keys = rows.astype(np.int64) * self.shape[1] + self.indices
        self.data = np.zeros((len(mats), keys.size), complex)
        for i, M in enumerate(mats):
            C = M.tocoo()
            pos = np.searchsorted(keys, C.row.astype(np.int64) * self.shape[1] + C.col)
            np.add.at(self.data[i], pos, C.data)

    def at(self, coeffs):
        import scipy.sparse as sp
        return sp.csr_matrix((np.asarray(coeffs) @ self.data, self.indices, self.indptr), shape=self.shape)


def transmon_cavity_drive_frame(model: dict, omega_d: float):
    """Exact frame change U = exp(i w_d t (K + a^dag a)), K = transmon level index (no RWA).

    |k><l| -> e^{i w_d t (k-l)} |k><l|,  a -> a e^{-i w_d t};  H' = U H U^dag - w_d (K + a^dag a).
    Returns heff(t): sparse H'_eff(t) including -i kappa/2 a^dag a, and the jump operator list."""
    import scipy.sparse as sp
    p = model["params"]; tw = 2 * np.pi
    n_t, n_c = model["dims"]
    Et, nt = model["Et"], model["nt"]
    a1 = sp.csr_matrix(dr.destroy(n_c)); Ic = sp.identity(n_c, format="csr")
    a = sp.kron(sp.identity(n_t), a1, format="csr"); ad = a.conj().T.tocsr()
    K = np.arange(n_t)
    Hd = sp.kron(sp.diags(tw * Et - omega_d * K), Ic) + (tw * p["fr"] - omega_d) * (ad @ a) \
        - 0.5j * tw * p["kappa"] * (ad @ a)
    mats, phases = [Hd.tocsr()], [None]
    g = tw * p["g"]
    for d in range(-(n_t - 1), n_t):
        Nd = np.diag(np.diag(nt, -d), -d) if d else np.diag(np.diag(nt))   # entries with k - l = d
        if np.abs(Nd).max() < 1e-12:
            continue
        Ndk = sp.kron(sp.csr_matrix(Nd), Ic, format="csr")
        mats += [(-1j * g) * (Ndk @ a), (1j * g) * (Ndk @ ad)]
        phases += [d - 1, d + 1]
    eps = model["eps_d"]
    mats += [-1j * a, 1j * ad]                       # times eps sin(w_d t) e^{-+ i w_d t}
    fs = FastSum(mats)

    def heff(t):
        c = [1.0] + [np.exp(1j * ph * omega_d * t) for ph in phases[1:]]
        s = eps * np.sin(omega_d * t)
        c += [s * np.exp(-1j * omega_d * t), s * np.exp(1j * omega_d * t)]
        return fs.at(c)

    return heff, [np.sqrt(tw * p["kappa"]) * a]


# ============================================================================= CV extension
# Notebooks 09-13 compare the package with analytic results and with the dense CV oracle
# ``qttlind.reference.cv_dense`` (an independent reference: it shares no code with the package).

from qttlind.reference import cv_dense as cd  # noqa: E402

HAVE_GAUSSIAN = "gaussian" in ql.FEATURES
HAVE_CV = "cv" in ql.FEATURES
HAVE_CIRCUITS = "cv_circuits" in ql.FEATURES


def cv_banner() -> str:
    parts = [f"gaussian backend: {'yes' if HAVE_GAUSSIAN else 'no'}",
             f"quantics CV: {'yes' if HAVE_CV else 'no'}",
             f"circuits: {'yes' if HAVE_CIRCUITS else 'no'}"]
    return f"qttlind {ql.__version__} -- " + "; ".join(parts) + "."


class DenseModes:
    """A few modes in a truncated Fock space: ladder operators and gates by expm (the oracle's way).

    ``gate(name, modes, *params)`` applies a Strawberry-Fields-convention gate to the ket ``psi``.
    """

    def __init__(self, cutoffs):
        self.dims = list(cutoffs)
        self.a = [dr.embed(dr.destroy(n), k, self.dims) for k, n in enumerate(self.dims)]
        self.D = int(np.prod(self.dims))
        self.psi = np.zeros(self.D, complex)
        self.psi[0] = 1.0

    def product_state(self, *kets):
        """Start from a product of single-mode kets (each of length = that mode's cutoff)."""
        self.psi = dr.kron(*[np.asarray(k, complex) for k in kets])
        return self

    def _expm(self, G):
        from scipy.linalg import expm
        return expm(G)

    def gate(self, name, modes, *p):
        a = self.a
        if name == "S":
            (k,), (r, phi) = modes, (p + (0.0,))[:2]
            z = r * np.exp(1j * phi)
            G = (np.conj(z) * a[k] @ a[k] - z * a[k].conj().T @ a[k].conj().T) / 2
        elif name == "D":
            (k,), (al,) = modes, p
            G = al * a[k].conj().T - np.conj(al) * a[k]
        elif name == "R":
            (k,), (th,) = modes, p
            G = 1j * th * a[k].conj().T @ a[k]
        elif name == "BS":
            (i, j), (th, phi) = modes, (p + (0.0,))[:2]
            G = th * (np.exp(1j * phi) * a[i] @ a[j].conj().T - np.exp(-1j * phi) * a[i].conj().T @ a[j])
        elif name == "S2":
            (i, j), (r, phi) = modes, (p + (0.0,))[:2]
            z = r * np.exp(1j * phi)
            G = np.conj(z) * a[i] @ a[j] - z * a[i].conj().T @ a[j].conj().T
        else:
            raise ValueError(f"unknown gate {name!r}")
        self.psi = self._expm(G) @ self.psi
        return self

    def interferometer(self, U):
        """a -> U a, as exp of i log(U) in the one-particle space (U unitary)."""
        from scipy.linalg import logm
        Hm = -1j * logm(U)  # U = exp(i Hm), Hm Hermitian
        G = sum(1j * Hm[i, j] * self.a[i].conj().T @ self.a[j] for i in range(len(self.a)) for j in range(len(self.a)))
        self.psi = self._expm(G) @ self.psi
        return self

    def probabilities(self):
        return (np.abs(self.psi) ** 2).reshape(self.dims)


def wigner_from_wavefunction(psi: np.ndarray, x: np.ndarray, p: np.ndarray, x_out: np.ndarray) -> np.ndarray:
    """W(x, p) = (1/pi) int dy conj(psi(x+y)) psi(x-y) e^{2 i p y}, psi sampled on a uniform grid x.

    Returns shape (len(p), len(x_out)); x_out must be grid points (it is snapped to the grid)."""
    dx = x[1] - x[0]
    W = np.zeros((len(p), len(x_out)))
    for col, x0 in enumerate(x_out):
        c = int(round((x0 - x[0]) / dx))
        smax = min(c, len(x) - 1 - c)
        s = np.arange(-smax, smax + 1)
        f = np.conj(psi[c + s]) * psi[c - s]
        W[:, col] = np.real(np.exp(2j * np.outer(p, s * dx)) @ f) * dx / np.pi
    return W


def gkp_wavefunction(x: np.ndarray, delta: float, logical: int = 0) -> np.ndarray:
    """Finite-energy square GKP state (Gaussian envelope), normalised on the grid x (dense helper)."""
    xs = (2 * np.arange(-200, 201) + logical) * np.sqrt(np.pi)
    xs = xs[np.abs(xs) < np.abs(x).max() + 10 * delta]
    psi = sum(np.exp(-delta**2 * c**2 / 2) * np.exp(-(x - c) ** 2 / (2 * delta**2)) for c in xs)
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0]))
