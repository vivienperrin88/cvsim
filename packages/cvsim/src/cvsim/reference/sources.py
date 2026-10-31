"""Independent oracles for the source models of :mod:`cvsim.hardware.sources`.

Each source is recomputed from its equations of motion, by a different route from the library's
closed forms, with NumPy and SciPy only (this module imports nothing from cvsim):

* :func:`opo_spectra` — a degenerate OPO below threshold from quantum Langevin equations and
  input–output theory: drift matrix, transfer functions ``(−iΩ − A)⁻¹``, output spectral matrix
  from vacuum inputs at the output coupler and at the loss port. No formula for ``S_∓`` is written
  down; Collett and Gardiner's comes out.
* :func:`multimode_squeezer` — the symplectic matrix of
  ``exp(½ Σ_jk (conj(Z_jk) a_j a_k − Z_jk a_j† a_k†))`` from its quadratic Hamiltonian in the pixel
  basis, ``S = expm(Ω H)``; no Takagi factorisation, no supermodes.
* :func:`lossy_multimode_squeezer` — the covariance after squeezing that builds up uniformly over
  a length while a frequency-flat loss acts: the Lyapunov equation
  ``dV/dz = A V + V Aᵀ + D`` integrated exactly with Van Loan's block exponential, in the pixel
  basis.

Conventions (ħ = 1, ``x = (a + a†)/√2``, xxpp order, vacuum ``V = I/2``) are restated here, not
imported.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

__all__ = ["lossy_multimode_squeezer", "multimode_squeezer", "opo_spectra", "squeezer_hamiltonian"]

Array = npt.NDArray[np.complex128]
Real = npt.NDArray[np.float64]


def _omega(M: int) -> Real:
    eye = np.eye(M)
    return np.block([[np.zeros((M, M)), eye], [-eye, np.zeros((M, M))]])


def opo_spectra(pump: float, escape_efficiency: float, omega_over_gamma: npt.ArrayLike) -> tuple[Real, Real]:
    """Output ``x`` and ``p`` noise spectra (shot-noise units) of a degenerate OPO below threshold.

    Cavity field ``a`` with total amplitude decay rate ``γ = γ_T + γ_L`` (output coupler and loss
    port, ``γ_T = η γ``) and parametric gain ``ε = σ γ`` (threshold at ``σ = 1``) from
    ``Ĥ = (iε/2)(a² − a†²)``, the pump phase that squeezes ``x`` (``ȧ ∋ −i[a, Ĥ] = −ε a†``):

        ȧ = −γ a − ε a† + √(2γ_T) a_in + √(2γ_L) b_in,    a_out = √(2γ_T) a − a_in.

    In quadratures ``ṙ = A r + B_T r_in + B_L r_b``; in Fourier space (``e^{−iΩt}``)
    ``r(Ω) = (−iΩ − A)⁻¹ (…)``. The output spectral matrix for vacuum inputs (unit spectra in
    shot-noise units) is ``H_T H_T† + H_L H_L†``; its diagonal is returned. Time is in units of
    ``1/γ`` (``γ = 1``).

    Parameters
    ----------
    pump
        ``σ = ε/γ`` in ``[0, 1)``.
    escape_efficiency
        ``η = γ_T/γ`` in ``(0, 1]``.
    omega_over_gamma
        Analysis frequencies ``Ω/γ``.
    """
    s, eta = float(pump), float(escape_efficiency)
    if not (0 <= s < 1 and 0 < eta <= 1):
        raise ValueError(f"need 0 <= pump < 1 and 0 < escape_efficiency <= 1, got {pump}, {escape_efficiency}")
    gT, gL = eta, 1.0 - eta
    # ȧ = −a − s a† → ẋ = (−1 − s) x, ṗ = (−1 + s) p
    A = np.diag([-1.0 - s, -1.0 + s])
    BT, BL = np.sqrt(2 * gT) * np.eye(2), np.sqrt(2 * gL) * np.eye(2)
    C, D = np.sqrt(2 * gT) * np.eye(2), -np.eye(2)
    w = np.atleast_1d(np.asarray(omega_over_gamma, dtype=float))
    sx, sp = np.empty_like(w), np.empty_like(w)
    for k, om in enumerate(w):
        G = np.linalg.inv(-1j * om * np.eye(2) - A)
        HT = C @ G @ BT + D
        HL = C @ G @ BL
        Sm = HT @ HT.conj().T + HL @ HL.conj().T
        sx[k], sp[k] = Sm[0, 0].real, Sm[1, 1].real
    return sx, sp


def squeezer_hamiltonian(Z: npt.ArrayLike) -> Real:
    """Matrix ``H`` (xxpp) of ``Ĥ = ½ rᵀ H r`` with ``exp(−iĤ) = exp(½ Σ (conj(Z) a a − Z a† a†))``.

    ``Ĥ = (i/2) Σ (conj(Z_jk) a_j a_k − Z_jk a_j† a_k†)``; with ``Z = R + iQ`` and
    ``a = (x + ip)/√2``: ``Ĥ = ½ (xᵀQx − pᵀQp) − xᵀRp`` (symmetric ``R``, ``Q``), so
    ``H = [[Q, −R], [−R, −Q]]``.

    Parameters
    ----------
    Z
        Complex symmetric ``M x M`` matrix.
    """
    Z_ = np.asarray(Z, dtype=complex)
    R, Q = Z_.real, Z_.imag
    return np.block([[Q, -R], [-R, -Q]])


def multimode_squeezer(Z: npt.ArrayLike) -> Real:
    """Symplectic matrix of ``exp(½ Σ (conj(Z) a a − Z a† a†))``: ``ṙ = i[Ĥ, r] = Ω H r`` with
    ``H`` from :func:`squeezer_hamiltonian`, so ``S = expm(−i·i Ω H) = expm(Ω H)`` for unit time.

    Parameters
    ----------
    Z
        Complex symmetric ``M x M`` matrix.
    """
    H = squeezer_hamiltonian(Z)
    return np.asarray(sla.expm(_omega(H.shape[0] // 2) @ H).real, dtype=float)


def lossy_multimode_squeezer(Z: npt.ArrayLike, attenuation_length: float) -> Real:
    """Covariance after squeezing ``Z`` built up uniformly over a length with power loss ``κℓ``.

    In the coordinate ``u = z/ℓ`` the quadratures obey ``dr/du = A r + noise`` with
    ``A = Ω H − (κℓ/2) I`` and the vacuum noise of the loss ``D = (κℓ/2) I``; the covariance obeys
    ``dV/du = A V + V Aᵀ + D``, ``V(0) = I/2``. Van Loan: with
    ``E = expm([[−A, D], [0, Aᵀ]])``, ``e^{A} = E₂₂ᵀ`` and ``∫₀¹ e^{As} D e^{Aᵀs} ds = E₂₂ᵀ E₁₂``.

    Parameters
    ----------
    Z
        Complex symmetric ``M x M`` matrix (total squeezing over the length).
    attenuation_length
        ``κℓ`` >= 0.
    """
    H = squeezer_hamiltonian(Z)
    n = H.shape[0]
    a = float(attenuation_length)
    A = _omega(n // 2) @ H - 0.5 * a * np.eye(n)
    D = 0.5 * a * np.eye(n)
    E = sla.expm(np.block([[-A, D], [np.zeros((n, n)), A.T]]))
    eA = E[n:, n:].T
    Q = eA @ E[:n, n:]
    V = eA @ (0.5 * np.eye(n)) @ eA.T + Q
    return np.asarray((V + V.T) / 2, dtype=float)
