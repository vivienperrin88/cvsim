"""Dense truncated-Fock oracle for continuous-variable optics.

Everything is computed the slow, obvious way: operators are explicit matrices on ``N`` Fock levels
per mode (``N**M`` in total, mode 0 most significant in the Kronecker product), gates are matrix
exponentials of their generators, channels are Kraus sums or the exponential of the Lindblad
superoperator, moments are traces. **This module imports nothing from cvsim**, so that it can check
:mod:`cvsim.core.conventions` (and everything built on it) without sharing a single formula.

In particular the symplectic matrix of a gate is *measured* here, not written down: for a gate
``U`` that is linear in the quadratures, ``[U† r_i U, r_k]`` is a c-number, and

    S = C (iΩ)^{-1},   C_ik = ⟨0|[U† r_i U, r_k]|0⟩,   iΩ_jk = ⟨0|[r_j, r_k]|0⟩,

with Ω itself read from the commutators of the truncated quadrature matrices. Truncation errors
vanish as the cutoff grows; tests choose cutoffs where they are below round-off.

Generators (Strawberry Fields, and ``F = R(π/2)``), ħ = 1, ``x = (a + a†)/√2``:

    D(α) = exp(α a† − ᾱ a)              R(θ) = exp(iθ n)
    S(r, φ) = exp((z̄ a² − z a†²)/2)     BS(θ, φ) = exp(θ(e^{iφ} a b† − e^{−iφ} a† b))
    S2(r, φ) = exp(z̄ ab − z a† b†)      CZ(A) = exp(i/2 Σ A_jk x_j x_k)
    CX(g) = exp(−i g x_a p_b)           P(s) = exp(i s x²/2)
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "apply_kraus",
    "beamsplitter",
    "coherent",
    "cx",
    "cz",
    "cz_generator",
    "destroy",
    "displacement",
    "dm",
    "fock",
    "heisenberg_map",
    "heisenberg_map_from_generator",
    "hermite_functions",
    "homodyne_density",
    "lindblad_loss",
    "loss_kraus",
    "mode_operators",
    "moments",
    "omega_from_commutators",
    "quadratures",
    "rotation",
    "shear",
    "squeezing",
    "two_mode_squeezing",
    "vacuum",
    "wigner",
]

Array = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


def _expm(G: Array) -> Array:
    """Dense matrix exponential, typed."""
    return np.asarray(sla.expm(G), dtype=np.complex128)


# ------------------------------------------------------------------------------- operators
def destroy(N: int) -> Array:
    """Annihilation operator on ``N`` Fock levels.

    Parameters
    ----------
    N
        Number of Fock levels.
    """
    return np.diag(np.sqrt(np.arange(1, N, dtype=float)), 1).astype(complex)


def mode_operators(M: int, N: int) -> list[Array]:
    """Annihilation operators of ``M`` modes on ``N**M`` levels (mode 0 most significant).

    Parameters
    ----------
    M
        Number of modes.
    N
        Fock levels per mode.
    """
    a, eye = destroy(N), np.eye(N, dtype=complex)
    ops = []
    for k in range(M):
        op = np.ones((1, 1), dtype=complex)
        for j in range(M):
            op = np.kron(op, a if j == k else eye)
        ops.append(op)
    return ops


def quadratures(M: int, N: int) -> list[Array]:
    """The quadratures ``x_k = (a_k + a_k†)/√2`` then ``p_k = (a_k − a_k†)/(i√2)``, xxpp order.

    Parameters
    ----------
    M
        Number of modes.
    N
        Fock levels per mode.
    """
    ops = mode_operators(M, N)
    xs = [(a + a.conj().T) / np.sqrt(2) for a in ops]
    ps = [(a - a.conj().T) / (1j * np.sqrt(2)) for a in ops]
    return xs + ps


def _sparse_quadratures(M: int, N: int) -> list[sp.csr_matrix]:
    """Sparse versions of :func:`quadratures` (for large ``N**M``)."""
    a1 = sp.diags(np.sqrt(np.arange(1, N, dtype=float)), 1, format="csr").astype(complex)
    eye = sp.identity(N, format="csr", dtype=complex)
    xs, ps = [], []
    for k in range(M):
        a = sp.csr_matrix(np.ones((1, 1), dtype=complex))
        for j in range(M):
            a = sp.kron(a, a1 if j == k else eye, format="csr")
        xs.append((a + a.conj().T) / np.sqrt(2))
        ps.append((a - a.conj().T) / (1j * np.sqrt(2)))
    return xs + ps


def _number(M: int, N: int, mode: int) -> Array:
    a = mode_operators(M, N)[mode]
    return a.conj().T @ a


# ------------------------------------------------------------------------------- states
def vacuum(M: int, N: int) -> Array:
    """The vacuum of ``M`` modes as a vector of length ``N**M``.

    Parameters
    ----------
    M
        Number of modes.
    N
        Fock levels per mode.
    """
    v = np.zeros(N**M, dtype=complex)
    v[0] = 1.0
    return v


def fock(ns: Sequence[int], N: int) -> Array:
    """The Fock state with ``ns[k]`` photons in mode ``k``.

    Parameters
    ----------
    ns
        Photon number of each mode.
    N
        Fock levels per mode.
    """
    index = 0
    for n in ns:
        index = index * N + int(n)
    v = np.zeros(N ** len(ns), dtype=complex)
    v[index] = 1.0
    return v


def coherent(alphas: Sequence[complex], N: int) -> Array:
    """Product of coherent states, each from its Fock expansion ``e^{−|α|²/2} Σ αⁿ/√n! |n⟩`` (renormalised).

    Parameters
    ----------
    alphas
        Amplitude of each mode.
    N
        Fock levels per mode.
    """
    out = np.ones(1, dtype=complex)
    for alpha in alphas:
        n = np.arange(N)
        logfact = np.array([np.sum(np.log(np.arange(1, k + 1))) for k in n])
        amp = np.exp(n * np.log(complex(alpha)) - logfact / 2) if alpha != 0 else (n == 0).astype(complex)
        amp = amp / np.linalg.norm(amp)
        out = np.kron(out, amp)
    return out


def dm(ket: Array) -> Array:
    """``|ψ⟩⟨ψ|``.

    Parameters
    ----------
    ket
        State vector.
    """
    return np.outer(ket, ket.conj())


# ------------------------------------------------------------------------------- gates (exp of generators)
def displacement(alpha: complex, M: int, N: int, mode: int = 0) -> Array:
    """``exp(α a† − ᾱ a)`` on ``mode``.

    Parameters
    ----------
    alpha
        Complex amplitude.
    M, N
        Number of modes and Fock levels per mode.
    mode
        The displaced mode.
    """
    a = mode_operators(M, N)[mode]
    return _expm(alpha * a.conj().T - np.conj(alpha) * a)


def rotation(theta: float, M: int, N: int, mode: int = 0) -> Array:
    """``exp(iθ n)`` on ``mode``.

    Parameters
    ----------
    theta
        Angle.
    M, N
        Number of modes and Fock levels per mode.
    mode
        The rotated mode.
    """
    return _expm(1j * theta * _number(M, N, mode))


def squeezing(r: float, phi: float, M: int, N: int, mode: int = 0) -> Array:
    """``exp((z̄ a² − z a†²)/2)``, ``z = r e^{iφ}``, on ``mode``.

    Parameters
    ----------
    r, phi
        Squeezing parameter and phase.
    M, N
        Number of modes and Fock levels per mode.
    mode
        The squeezed mode.
    """
    a = mode_operators(M, N)[mode]
    z = r * np.exp(1j * phi)
    return _expm((np.conj(z) * a @ a - z * a.conj().T @ a.conj().T) / 2)


def beamsplitter(theta: float, phi: float, M: int, N: int, modes: tuple[int, int] = (0, 1)) -> Array:
    """``exp(θ(e^{iφ} a b† − e^{−iφ} a† b))`` on ``modes = (a, b)``.

    Parameters
    ----------
    theta, phi
        Mixing angle and phase.
    M, N
        Number of modes and Fock levels per mode.
    modes
        The two modes ``(a, b)``.
    """
    ops = mode_operators(M, N)
    a, b = ops[modes[0]], ops[modes[1]]
    return _expm(theta * (np.exp(1j * phi) * a @ b.conj().T - np.exp(-1j * phi) * a.conj().T @ b))


def two_mode_squeezing(r: float, phi: float, M: int, N: int, modes: tuple[int, int] = (0, 1)) -> Array:
    """``exp(z̄ ab − z a†b†)``, ``z = r e^{iφ}``, on ``modes = (a, b)``.

    Parameters
    ----------
    r, phi
        Squeezing parameter and phase.
    M, N
        Number of modes and Fock levels per mode.
    modes
        The two modes ``(a, b)``.
    """
    ops = mode_operators(M, N)
    a, b = ops[modes[0]], ops[modes[1]]
    z = r * np.exp(1j * phi)
    return _expm(np.conj(z) * a @ b - z * a.conj().T @ b.conj().T)


def cz_generator(A: npt.ArrayLike, N: int) -> sp.csr_matrix:
    """Anti-Hermitian generator ``i/2 Σ_jk A_jk x_j x_k`` of :func:`cz`, as a sparse matrix.

    Parameters
    ----------
    A
        Real symmetric weight matrix.
    N
        Fock levels per mode.
    """
    A_ = np.asarray(A, dtype=float)
    M = A_.shape[0]
    xs = _sparse_quadratures(M, N)[:M]
    G = sp.csr_matrix((N**M, N**M), dtype=complex)
    for j in range(M):
        for k in range(M):
            G = G + A_[j, k] * (xs[j] @ xs[k])
    return (0.5j * G).tocsr()


def cz(A: npt.ArrayLike, N: int) -> Array:
    """``exp(i/2 Σ_jk A_jk x_j x_k)`` on ``M = len(A)`` modes.

    Parameters
    ----------
    A
        Real symmetric weight matrix.
    N
        Fock levels per mode.
    """
    return _expm(cz_generator(A, N).toarray())


def cx(g: float, M: int, N: int, modes: tuple[int, int] = (0, 1)) -> Array:
    """``exp(−i g x_a p_b)`` on ``modes = (a, b)``.

    Parameters
    ----------
    g
        Gain.
    M, N
        Number of modes and Fock levels per mode.
    modes
        Control and target ``(a, b)``.
    """
    r = quadratures(M, N)
    return _expm(-1j * g * r[modes[0]] @ r[M + modes[1]])


def shear(s: float, M: int, N: int, mode: int = 0) -> Array:
    """``exp(i s x²/2)`` on ``mode``.

    Parameters
    ----------
    s
        Shear strength.
    M, N
        Number of modes and Fock levels per mode.
    mode
        The sheared mode.
    """
    x = quadratures(M, N)[mode]
    return _expm(0.5j * s * x @ x)


# ------------------------------------------------------------------------------- phase-space maps
def omega_from_commutators(M: int, N: int) -> Real:
    """``Ω_jk = −i ⟨0|[r_j, r_k]|0⟩`` read from the truncated quadrature matrices.

    Parameters
    ----------
    M, N
        Number of modes and Fock levels per mode.
    """
    r = quadratures(M, N)
    v = vacuum(M, N)
    W = np.array([[v.conj() @ (rj @ rk - rk @ rj) @ v for rk in r] for rj in r])
    return np.real(-1j * W)


def heisenberg_map(U: Array, M: int, N: int) -> tuple[Real, Real]:
    """``(S, c)`` with ``U† r_i U = Σ_j S_ij r_j + c_i`` for a gate linear in the quadratures.

    ``S = C (iΩ)^{-1}`` with ``C_ik = ⟨0|[U† r_i U, r_k]|0⟩``; ``c_i = ⟨0|U† r_i U|0⟩`` (the vacuum
    has zero mean). Only the low-lying states enter, so the cutoff ``N`` only needs to hold
    ``U|0⟩`` and ``U|1_k⟩``.

    Parameters
    ----------
    U
        Unitary on ``N**M`` levels.
    M, N
        Number of modes and Fock levels per mode.

    Raises
    ------
    ValueError
        ``U† r U`` is not a real linear function of ``r`` on the vacuum: ``U`` is not a Gaussian
        gate, or ``N`` is too small for it.
    """
    r = quadratures(M, N)
    v = vacuum(M, N)
    heis = [U.conj().T @ ri @ U for ri in r]
    C = np.array([[v.conj() @ (h @ rk - rk @ h) @ v for rk in r] for h in heis])
    probes = np.column_stack([v] + [rk @ v for rk in r])
    return _linear_map(C, [h @ probes for h in heis], r, probes, M)


def _linear_map(
    C: Array, heis_on_probes: Sequence[Array], r: Sequence[Array], probes: Array, M: int
) -> tuple[Real, Real]:
    """``S = C (iΩ)^{-1}``, ``c_i = ⟨0|U† r_i U|0⟩``, then check that the map *is* linear.

    A Gaussian gate satisfies ``U† r_i U |w⟩ = (Σ_j S_ij r_j + c_i)|w⟩`` for every ``w``; this is
    checked on the probes ``|0⟩`` and ``r_k|0⟩`` (columns of ``probes``; ``heis_on_probes[i]`` holds
    ``U† r_i U`` applied to them). The vacuum alone would not do: a Kerr gate acts on it like a
    rotation. Any non-Gaussian gate, or a cutoff too small for this one, leaves a residual.

    The residual is a *gross* check (threshold 1e-3; a Kerr gate of strength 0.3 leaves 0.4): it
    involves ``r_i`` acting on ``U|w⟩`` at the edge of the truncated space, where truncation errors
    are largest. The accuracy of ``S`` itself is what the tests compare, to 1e-9.
    """
    S = C @ np.linalg.inv(1j * omega_from_commutators(M, 3))  # Ω does not depend on the cutoff
    v = probes[:, 0]
    c = np.array([v.conj() @ hp[:, 0] for hp in heis_on_probes])
    r_probes = [ri @ probes for ri in r]
    residual = max(
        float(np.abs(hp - sum(S[i, j] * r_probes[j] for j in range(len(r))) - c[i] * probes).max())
        for i, hp in enumerate(heis_on_probes)
    )
    if residual > 1e-3 or max(np.abs(S.imag).max(), np.abs(c.imag).max()) > 1e-8:
        raise ValueError(
            "the gate is not a real linear map of the quadratures, or the cutoff is too small "
            f"(residual {residual:.1e})"
        )
    return S.real, c.real


def heisenberg_map_from_generator(G: sp.spmatrix | Array, M: int, N: int) -> tuple[Real, Real]:
    """:func:`heisenberg_map` of ``U = exp(G)`` without forming ``U``.

    Only ``U|0⟩``, ``U r_k|0⟩`` and ``U† r_i U`` on those probes are needed:
    ``C_ik = ⟨U0|r_i|U r_k 0⟩ − ⟨U r_k 0|r_i|U0⟩``, each ``U v`` computed by
    ``scipy.sparse.linalg.expm_multiply``. This makes large cutoffs on
    three or more modes affordable.

    Parameters
    ----------
    G
        Anti-Hermitian generator on ``N**M`` levels (sparse or dense).
    M, N
        Number of modes and Fock levels per mode.

    Raises
    ------
    ValueError
        As :func:`heisenberg_map`.
    """
    r = _sparse_quadratures(M, N)
    v = vacuum(M, N)
    vecs = np.column_stack([v] + [rk @ v for rk in r])
    Gs = sp.csr_matrix(G)
    out = spla.expm_multiply(Gs, vecs)
    u0, uk = out[:, 0], out[:, 1:]
    C = np.array([[u0.conj() @ (ri @ uk[:, k]) - uk[:, k].conj() @ (ri @ u0) for k in range(len(r))] for ri in r])
    n = vecs.shape[1]
    back = spla.expm_multiply(-Gs, np.column_stack([ri @ out for ri in r]))  # U† r_i U on each probe
    return _linear_map(C, [back[:, i * n : (i + 1) * n] for i in range(len(r))], r, vecs, M)


def moments(rho: Array, M: int, N: int) -> tuple[Real, Real]:
    """Mean ``Tr(ρ r_i)`` and covariance ``Re Tr(ρ {Δr_i, Δr_j})/2`` of a density matrix.

    Parameters
    ----------
    rho
        Density matrix on ``N**M`` levels.
    M, N
        Number of modes and Fock levels per mode.
    """
    r = quadratures(M, N)
    mean = np.array([np.trace(rho @ ri).real for ri in r])
    eye = np.eye(N**M)
    d = [ri - mi * eye for ri, mi in zip(r, mean)]
    cov = np.array([[np.trace(rho @ (di @ dj + dj @ di)).real / 2 for dj in d] for di in d])
    return mean, cov


# ------------------------------------------------------------------------------- channels
def loss_kraus(eta: float, N: int) -> list[Array]:
    """Kraus operators of pure loss: ``K_k = Σ_n √C(n,k) η^{(n−k)/2} (1−η)^{k/2} |n−k⟩⟨n|``.

    Parameters
    ----------
    eta
        Transmissivity.
    N
        Fock levels.
    """
    out = []
    for k in range(N):
        K = np.zeros((N, N), dtype=complex)
        for n in range(k, N):
            K[n - k, n] = np.sqrt(comb(n, k) * eta ** (n - k) * (1 - eta) ** k)
        out.append(K)
    return out


def apply_kraus(rho: Array, kraus: Sequence[Array], M: int, N: int, mode: int = 0) -> Array:
    """``Σ_k K_k ρ K_k†`` with single-mode Kraus operators acting on ``mode``.

    Parameters
    ----------
    rho
        Density matrix on ``N**M`` levels.
    kraus
        Single-mode Kraus operators (``N x N``).
    M, N
        Number of modes and Fock levels per mode.
    mode
        The mode they act on.
    """
    out = np.zeros_like(rho)
    for K in kraus:
        full = np.ones((1, 1), dtype=complex)
        for j in range(M):
            full = np.kron(full, K if j == mode else np.eye(N))
        out += full @ rho @ full.conj().T
    return out


def lindblad_loss(rho: Array, kappa: float, t: float, N: int) -> Array:
    """Single-mode ``dρ/dt = κ(a ρ a† − {a†a, ρ}/2)`` integrated exactly (exponential of the superoperator).

    Parameters
    ----------
    rho
        Density matrix on ``N`` levels.
    kappa
        Decay rate.
    t
        Duration.
    N
        Fock levels.
    """
    a = destroy(N)
    eye = np.eye(N)
    n = a.conj().T @ a
    # row-major vectorisation: vec(A ρ B) = (A ⊗ Bᵀ) vec(ρ)
    L = kappa * (np.kron(a, a.conj()) - 0.5 * np.kron(n, eye) - 0.5 * np.kron(eye, n.T))
    return (_expm(L * t) @ rho.reshape(-1)).reshape(N, N)


# ------------------------------------------------------------------------------- phase-space functions
def hermite_functions(N: int, q: npt.ArrayLike) -> Real:
    """``ψ_n(q) = ⟨q|n⟩`` for ``n < N`` (rows), normalised Hermite functions for ħ = 1.

    Parameters
    ----------
    N
        Number of functions.
    q
        Points.
    """
    q_ = np.atleast_1d(np.asarray(q, dtype=float))
    psi = np.zeros((N, q_.size))
    psi[0] = np.pi**-0.25 * np.exp(-(q_**2) / 2)
    if N > 1:
        psi[1] = np.sqrt(2.0) * q_ * psi[0]
    for n in range(2, N):
        psi[n] = np.sqrt(2.0 / n) * q_ * psi[n - 1] - np.sqrt((n - 1) / n) * psi[n - 2]
    return psi


def homodyne_density(rho: Array, phi: float, q: npt.ArrayLike, N: int) -> Real:
    """Single-mode density of ``x_φ = cos φ x + sin φ p``: ``⟨q|R(−φ) ρ R(φ)|q⟩`` with ``R(θ) = e^{iθn}``.

    ``R(−φ)`` maps the observable ``x`` back to ``x_φ`` (``R(φ)† x R(φ) = cos φ x − sin φ p``, so
    ``R(−φ)† x R(−φ) = x_φ``).

    Parameters
    ----------
    rho
        Density matrix on ``N`` levels.
    phi
        Local-oscillator phase.
    q
        Points.
    N
        Fock levels.
    """
    R = np.diag(np.exp(-1j * phi * np.arange(N)))
    rot = R @ rho @ R.conj().T
    psi = hermite_functions(N, q)
    return np.real(np.einsum("mq,mn,nq->q", psi, rot, psi))


def wigner(rho: Array, x: float, p: float, N: int) -> float:
    """Single-mode ``W(x, p) = Tr[ρ D(α) Π D(α)†]/π``, ``α = (x + ip)/√2``, ``Π = (−1)^n``.

    Parameters
    ----------
    rho
        Density matrix on ``N`` levels.
    x, p
        Phase-space point.
    N
        Fock levels.
    """
    D = displacement((x + 1j * p) / np.sqrt(2), 1, N)
    parity = np.diag((-1.0) ** np.arange(N))
    return float(np.real(np.trace(rho @ D @ parity @ D.conj().T)) / np.pi)
