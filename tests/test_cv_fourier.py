"""The quantics Fourier transform (qttlind.cv.fourier)."""

import numpy as np
import pytest

from qttlind.core.tt import TensorTrain
from qttlind.cv import fourier as fo
from qttlind.cv import grids as gr
from qttlind.errors import InputError


def centred_dft(N):
    j = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(j - N / 2, j - N / 2) / N) / np.sqrt(N)


def bit_reversal(R):
    return np.array([int(format(k, f"0{R}b")[::-1], 2) for k in range(2**R)])


def test_bit_factor_matches_its_definition():
    R, a = 4, 2
    f = fo._bit_factor(a, R).to_dense().reshape([4] * R)
    for idx in np.ndindex(*([4] * R)):
        o = [i // 2 for i in idx]
        j_a = idx[a] % 2
        expected = np.prod([np.exp(-2j * np.pi * o[c] * 2.0 ** (c - a - 1)) for c in range(a + 1)]) if j_a else 1.0
        assert f[idx] == pytest.approx(expected)


@pytest.mark.parametrize("R", [1, 2, 3, 5, 8])
def test_dft_mpo_is_the_bit_reversed_centred_dft(R):
    np.testing.assert_allclose(fo.dft_mpo(R).to_dense(), centred_dft(2**R)[bit_reversal(R)], atol=1e-12)


def test_dft_mpo_rank_is_small_and_flat():
    """Chen, Stoudenmire & White (2023): the QFT without the bit reversal has small rank."""
    ranks = [fo.dft_mpo(R).max_rank for R in (10, 16, 24)]
    assert max(ranks) <= 16
    assert ranks[-1] <= ranks[0] + 2


def test_reverse_sites():
    rng = np.random.default_rng(0)
    tt = TensorTrain([rng.normal(size=s) for s in [(1, 2, 3), (3, 2, 2), (2, 2, 1)]])
    T = tt.to_dense().reshape(2, 2, 2)
    np.testing.assert_allclose(fo.reverse_sites(tt).to_dense().reshape(2, 2, 2), T.transpose(2, 1, 0))


@pytest.mark.parametrize("alpha", [0.0, 1.2 - 0.4j])
def test_to_momentum_matches_numpy_fft(alpha):
    g = gr.QuadratureGrid(10)
    c = gr.coherent(alpha, g).to_dense()
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c))) / np.sqrt(c.size)
    np.testing.assert_allclose(fo.to_momentum(gr.coherent(alpha, g)).to_dense(), ref, atol=1e-10)


def test_coherent_state_in_momentum_space():
    """<p|alpha> is a Gaussian centred at p0 = sqrt(2) Im(alpha): the self-dual grid is also the p grid."""
    g = gr.QuadratureGrid(10)
    alpha = 0.9 + 0.6j
    phi = fo.to_momentum(gr.coherent(alpha, g))
    prob = np.abs(phi.to_dense()) ** 2
    p0 = np.sum(g.p() * prob)
    assert p0 == pytest.approx(np.sqrt(2) * alpha.imag, abs=1e-9)


def test_hermite_functions_are_eigenvectors():
    """F phi_n = (-i)^n phi_n: the Fourier transform is the rotation R(-pi/2) = exp(-i pi n / 2)."""
    g = gr.QuadratureGrid(10)
    for n in (0, 1, 2, 7):
        phi = gr.fock(n, g)
        out = fo.to_momentum(phi)
        np.testing.assert_allclose(out.to_dense(), (-1j) ** n * phi.to_dense(), atol=1e-9)


def test_round_trip_and_unitarity():
    g = gr.QuadratureGrid(12)
    psi = gr.cat(2.0, g)
    phi = fo.to_momentum(psi)
    assert phi.norm() == pytest.approx(1.0, abs=1e-12)
    back = fo.to_position(phi)
    assert (back - psi).compress(0.0)[0].norm() < 1e-10


def test_large_grids_stay_cheap():
    g = gr.QuadratureGrid(20)
    psi = gr.squeezed_vacuum(0.8, g)
    phi = fo.to_momentum(psi)
    assert phi.max_rank <= 24
    assert phi.norm() == pytest.approx(1.0, abs=1e-10)


def test_bits_validation():
    with pytest.raises(InputError):
        fo.dft_mpo(0)
