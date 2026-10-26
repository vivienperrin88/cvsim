"""
qttlind.reference.cv_dense -- dense Fock-space oracle for continuous-variable (CV) optics.

Everything here is computed the slow, obvious way, in a truncated Fock space: gates are matrix
exponentials of their generators, channels are explicit Kraus sums, moments are traces against
dense quadrature matrices. It validates the Gaussian backend (``qttlind.gaussian``) and the
quantics CV layer (``qttlind.cv``) without sharing a single formula with them.

Conventions (CONVENTIONS.md, "Continuous variables"): hbar = 1, x = (a + a^dag)/sqrt(2),
p = (a - a^dag)/(i sqrt(2)), quadrature vector r = (x_1..x_M, p_1..p_M) ("xxpp"), vacuum
covariance I/2. Multi-mode Kronecker order: mode 0 most significant.

Gate generators (the same as Strawberry Fields, so published circuits translate directly):

    D(alpha)      = exp(alpha a^dag - conj(alpha) a)
    R(theta)      = exp(i theta n)
    S(r, phi)     = exp((conj(z) a^2 - z a^dag^2)/2),           z = r e^{i phi}
    BS(theta,phi) = exp(theta (e^{i phi} a b^dag - e^{-i phi} a^dag b))
    S2(r, phi)    = exp(conj(z) a b - z a^dag b^dag),           z = r e^{i phi}
    K(kappa)      = exp(i kappa n^2)
    V(gamma)      = exp(i gamma x^3 / 3)

No tensor networks and no phase-space shortcuts here on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np
import scipy.linalg as sla
from scipy.special import gammaln

from .dense_ref import destroy, embed, eye

__all__ = [
    "beamsplitter",
    "cubic_phase",
    "displacement",
    "fock_ket",
    "hafnian_recursive",
    "hermite_functions",
    "kerr",
    "loop_hafnian_recursive",
    "loss_channel",
    "moments",
    "position_wavefunction",
    "quadratures",
    "rotation",
    "squeezed_vacuum_amplitudes",
    "squeezing",
    "thermal_dm",
    "two_mode_squeezing",
    "wigner_parity",
]

# ----------------------------------------------------------------------------- operators


def quadratures(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``x`` and ``p`` on ``N`` Fock states (exact matrix elements; truncated space)."""
    a = destroy(N)
    return (a + a.conj().T) / np.sqrt(2), (a - a.conj().T) / (1j * np.sqrt(2))


def fock_ket(N: int, n: int) -> np.ndarray:
    v = np.zeros(N, complex)
    v[n] = 1.0
    return v


def displacement(N: int, alpha: complex) -> np.ndarray:
    a = destroy(N)
    return sla.expm(alpha * a.conj().T - np.conj(alpha) * a)


def rotation(N: int, theta: float) -> np.ndarray:
    return np.diag(np.exp(1j * theta * np.arange(N)))


def squeezing(N: int, r: float, phi: float = 0.0) -> np.ndarray:
    a = destroy(N)
    z = r * np.exp(1j * phi)
    return sla.expm((np.conj(z) * a @ a - z * a.conj().T @ a.conj().T) / 2)


def beamsplitter(N1: int, N2: int, theta: float, phi: float = 0.0) -> np.ndarray:
    a, b = np.kron(destroy(N1), eye(N2)), np.kron(eye(N1), destroy(N2))
    return sla.expm(theta * (np.exp(1j * phi) * a @ b.conj().T - np.exp(-1j * phi) * a.conj().T @ b))


def two_mode_squeezing(N1: int, N2: int, r: float, phi: float = 0.0) -> np.ndarray:
    a, b = np.kron(destroy(N1), eye(N2)), np.kron(eye(N1), destroy(N2))
    z = r * np.exp(1j * phi)
    return sla.expm(np.conj(z) * a @ b - z * a.conj().T @ b.conj().T)


def kerr(N: int, kappa: float) -> np.ndarray:
    n = np.arange(N)
    return np.diag(np.exp(1j * kappa * n**2))


def cubic_phase(N: int, gamma: float) -> np.ndarray:
    """exp(i gamma x^3/3) in a truncated Fock space -- only accurate far below the cutoff."""
    x, _ = quadratures(N)
    return sla.expm(1j * gamma * x @ x @ x / 3)


def thermal_dm(N: int, nbar: float) -> np.ndarray:
    n = np.arange(N)
    p = (nbar / (1 + nbar)) ** n / (1 + nbar) if nbar > 0 else (n == 0).astype(float)
    return np.diag(p).astype(complex)


def squeezed_vacuum_amplitudes(N: int, r: float, phi: float = 0.0) -> np.ndarray:
    """Closed form <2m|S(r,phi)|0> = (-e^{i phi} tanh r)^m sqrt((2m)!)/(2^m m!) / sqrt(cosh r)."""
    c = np.zeros(N, complex)
    m = np.arange(0, (N + 1) // 2)
    logmag = 0.5 * gammaln(2 * m + 1) - gammaln(m + 1) - m * np.log(2) + m * np.log(np.tanh(r) if r > 0 else 1.0)
    if r == 0:
        c[0] = 1.0
        return c
    c[2 * m] = np.exp(logmag) * (-np.exp(1j * phi)) ** m / np.sqrt(np.cosh(r))
    return c


# ----------------------------------------------------------------------------- channels


def loss_channel(rho: np.ndarray, eta: float, which: int, dims: Sequence[int]) -> np.ndarray:
    """Pure loss (beam splitter of transmissivity eta with vacuum) on mode `which`, by Kraus sum.

    ``K_k = sum_n sqrt(C(n, k)) eta^((n-k)/2) (1-eta)^(k/2) |n-k><n|``.
    """
    N = dims[which]
    out = np.zeros_like(rho)
    for k in range(N):
        K = np.zeros((N, N), complex)
        for n in range(k, N):
            K[n - k, n] = np.sqrt(comb(n, k) * eta ** (n - k) * (1 - eta) ** k)
        Kf = embed(K, which, dims)
        out += Kf @ rho @ Kf.conj().T
    return out


# ----------------------------------------------------------------------------- moments


def moments(rho: np.ndarray, dims: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance (xxpp order, V_ij = <{dr_i, dr_j}>/2) of a dense state."""
    M = len(dims)
    ops = []
    for k in range(M):
        x, _ = quadratures(dims[k])
        ops.append(embed(x, k, dims))
    for k in range(M):
        _, p = quadratures(dims[k])
        ops.append(embed(p, k, dims))
    mean = np.array([np.trace(rho @ o).real for o in ops])
    V = np.zeros((2 * M, 2 * M))
    for i in range(2 * M):
        for j in range(2 * M):
            sym = (ops[i] @ ops[j] + ops[j] @ ops[i]) / 2
            V[i, j] = np.trace(rho @ sym).real - mean[i] * mean[j]
    return mean, V


# ----------------------------------------------------------------------------- position space


def hermite_functions(nmax: int, x: np.ndarray) -> np.ndarray:
    """phi_n(x) = <x|n>, n < nmax, by the normalised three-term recurrence. Shape (nmax, len(x))."""
    x = np.asarray(x, float)
    phi = np.zeros((nmax, x.size))
    phi[0] = np.pi**-0.25 * np.exp(-(x**2) / 2)
    if nmax > 1:
        phi[1] = np.sqrt(2.0) * x * phi[0]
    for n in range(1, nmax - 1):
        phi[n + 1] = np.sqrt(2.0 / (n + 1)) * x * phi[n] - np.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def position_wavefunction(ket: np.ndarray, x: np.ndarray) -> np.ndarray:
    """psi(x) = sum_n c_n phi_n(x) for a single-mode Fock-space ket."""
    return hermite_functions(ket.size, x).T @ ket


# ----------------------------------------------------------------------------- Wigner function


def wigner_parity(rho: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """W(alpha) = (2/pi) Tr[D(-alpha) rho D(alpha) P]  (normalised: int W d^2 alpha = 1).

    W(x, p) in quadrature units is W(alpha)/2 with alpha = (x + i p)/sqrt(2).
    """
    N = rho.shape[0]
    P = np.diag((-1.0) ** np.arange(N))
    out = np.empty(np.shape(alphas), float)
    for idx, al in np.ndenumerate(np.asarray(alphas)):
        Dm = displacement(N, -al)
        out[idx] = (2 / np.pi) * np.trace(Dm @ rho @ Dm.conj().T @ P).real
    return out


# ----------------------------------------------------------------------------- hafnians (brute force)


def hafnian_recursive(A: np.ndarray) -> complex:
    """Sum over perfect matchings, by expansion along the first row. Exponential; n <= 12."""
    A = np.asarray(A)
    n = A.shape[0]
    if n == 0:
        return 1.0 + 0j
    if n % 2:
        return 0.0 + 0j
    total = 0.0 + 0j
    rest = list(range(1, n))
    for j in rest:
        keep = [k for k in rest if k != j]
        total += A[0, j] * hafnian_recursive(A[np.ix_(keep, keep)])
    return total


def loop_hafnian_recursive(A: np.ndarray) -> complex:
    """Sum over matchings with loops (diagonal entries are the loop weights)."""
    A = np.asarray(A)
    n = A.shape[0]
    if n == 0:
        return 1.0 + 0j
    rest = list(range(1, n))
    total = A[0, 0] * loop_hafnian_recursive(A[np.ix_(rest, rest)])
    for j in rest:
        keep = [k for k in rest if k != j]
        total += A[0, j] * loop_hafnian_recursive(A[np.ix_(keep, keep)])
    return total
