"""surface_spectral_function: batched (multi-core) backend == serial backend."""

import numpy as np
import pytest
import torch

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.surface import surface_spectral_function


def _random_hr(nw=2, seed=0):
    """Small Hermitian H(R) on a simple cubic lattice: on-site + nn hops."""
    rng = np.random.default_rng(seed)
    Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
          (0, 0, 1), (0, 0, -1)]
    R = np.array(Rs, dtype=np.int64)
    H = np.zeros((len(Rs), nw, nw), dtype=np.complex128)
    for i, r in enumerate(Rs):
        A = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
        H[i] = 0.3 * A
    # enforce H(-R) = H(R)^dagger (and Hermitian on-site)
    idx = {tuple(r): i for i, r in enumerate(Rs)}
    for i, r in enumerate(Rs):
        j = idx[tuple(-np.array(r))]
        H[i] = 0.5 * (H[i] + H[j].conj().T)
        H[j] = H[i].conj().T
    return HamiltonianR(H_R=torch.from_numpy(H), R_vectors=R,
                        degen=np.ones(len(Rs), np.int64), nw=nw)


def test_batched_matches_serial():
    hr = _random_hr(nw=2, seed=1)
    real_lattice = np.eye(3) * 5.0
    kpath = np.array([[0.0, 0.0], [0.13, 0.0], [0.27, 0.19], [0.5, 0.5]])
    energies = np.linspace(-0.4, 0.4, 11)
    eta = 0.03

    ser = surface_spectral_function(hr, real_lattice, (0, 0, 1), kpath, energies,
                                    eta=eta, backend="serial")
    bat = surface_spectral_function(hr, real_lattice, (0, 0, 1), kpath, energies,
                                    eta=eta, backend="batched")
    assert np.allclose(ser.A_surface, bat.A_surface, rtol=1e-8, atol=1e-10)
    assert np.allclose(ser.A_bulk, bat.A_bulk, rtol=1e-8, atol=1e-10)


def test_batched_matches_serial_with_spin_and_matrix_element():
    hr = _random_hr(nw=2, seed=2)
    real_lattice = np.eye(3) * 5.0
    kpath = np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.4]])
    energies = np.linspace(-0.3, 0.3, 7)

    # a Hermitian "spin" operator S(R) on the same R grid (use S_z-like on-site)
    nR, nw = hr.H_R.shape[0], hr.nw
    ss = np.zeros((nR, nw, nw, 3), dtype=np.complex128)
    ss[0, :, :, 2] = np.diag([1.0, -1.0])          # S_z on-site
    M = (np.random.default_rng(3).normal(size=(kpath.shape[0], nw))
         + 1j * np.random.default_rng(4).normal(size=(kpath.shape[0], nw)))

    kw = dict(eta=0.03, spin_op_r=ss, matrix_element=M)
    ser = surface_spectral_function(hr, real_lattice, (0, 0, 1), kpath, energies,
                                    backend="serial", **kw)
    bat = surface_spectral_function(hr, real_lattice, (0, 0, 1), kpath, energies,
                                    backend="batched", **kw)
    for attr in ("A_surface", "A_bulk", "A_up", "A_dn", "A_arpes"):
        assert np.allclose(getattr(ser, attr), getattr(bat, attr),
                           rtol=1e-8, atol=1e-10), attr
    # spin channels partition the total
    assert np.allclose(bat.A_up + bat.A_dn, bat.A_surface, rtol=1e-7, atol=1e-9)


# --------------------------------------------------------------------------
# Finite slabs cut out of the bulk H(R)
# --------------------------------------------------------------------------

def _cubic_tb(t=0.05, a=5.0):
    """One s orbital per simple-cubic site, nearest-neighbour hopping."""
    Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    H = np.zeros((len(Rs), 1, 1), dtype=np.complex128)
    H[1:, 0, 0] = -t
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=np.array(Rs, dtype=np.int64),
                        degen=np.ones(len(Rs), dtype=np.int64), nw=1,
                        centres=np.zeros((1, 3)), real_lattice=a * np.eye(3),
                        mp_grid=(6, 6, 6)), a, t


class TestBuildSlab:
    def test_open_chain_eigenvalues_are_exact(self):
        """A simple-cubic (001) slab is an open chain along z at every k_par, so
        its eigenvalues are known in closed form:

            E = -2t(cos kx + cos ky) - 2t cos(pi n / (N+1)),  n = 1..N

        the standing waves of a chain with hard walls. Nothing about that is
        approximate, so this pins the layerisation, the truncation of hoppings
        that would leave the slab, and the R bookkeeping simultaneously."""
        from waw.analysis.surface import build_slab
        from waw.analysis.elph import band_eigensystem
        hr, a, t = _cubic_tb()
        N = 6
        slab = build_slab(hr, a * np.eye(3), (0, 0, 1), N)
        assert slab.nw == N
        k1, k2 = 0.13, 0.27
        e, _ = band_eigensystem(slab, np.array([[k1, k2, 0.0]]))
        exact = np.sort([-2 * t * (np.cos(2 * np.pi * k1) + np.cos(2 * np.pi * k2))
                         - 2 * t * np.cos(np.pi * n / (N + 1)) for n in range(1, N + 1)])
        assert np.abs(e[0] - exact).max() < 1e-12

    def test_centres_are_stacked_along_the_surface_normal(self):
        from waw.analysis.surface import build_slab
        hr, a, _ = _cubic_tb()
        slab = build_slab(hr, a * np.eye(3), (0, 0, 1), 4)
        assert np.allclose(slab.centres[:, 2], a * np.arange(4))
        assert np.allclose(slab.centres[:, :2], 0.0)

    def test_slab_hamiltonian_is_hermitian(self):
        from waw.analysis.surface import build_slab
        from waw.core.hamiltonian import operator_k
        hr, a, _ = _cubic_tb()
        slab = build_slab(hr, a * np.eye(3), (0, 0, 1), 5)
        Hk = operator_k(slab.H_R, slab.R_vectors, slab.degen,
                        np.array([[0.2, 0.35, 0.0]])).numpy()
        assert np.abs(Hk - np.conj(np.swapaxes(Hk, -1, -2))).max() < 1e-14

    def test_thick_slab_reproduces_the_semi_infinite_surface_state(self):
        """A slab is only a stand-in for the semi-infinite crystal if it is
        thick enough. Rather than assert a thickness, check the convergence: the
        slab's density of states at a fixed (k_par, E) must approach the
        decimated `surface_spectral_function` result as layers are added."""
        from waw.analysis.surface import build_slab, surface_spectral_function
        from waw.analysis.elph import band_eigensystem
        hr, a, t = _cubic_tb()
        kpar = np.array([[0.2, 0.1]])
        E = np.array([-0.13])
        eta = 5e-3
        sf = surface_spectral_function(hr, a * np.eye(3), (0, 0, 1), kpar, E, eta=eta)
        target = float(sf.A_bulk[0, 0])
        errs = []
        for N in (8, 24, 72):
            slab = build_slab(hr, a * np.eye(3), (0, 0, 1), N)
            e, _ = band_eigensystem(slab, np.array([[0.2, 0.1, 0.0]]))
            dos = float(((eta / np.pi) / ((E[0] - e[0]) ** 2 + eta ** 2)).sum()) / N
            errs.append(abs(dos - target))
        assert errs[-1] < errs[0], f"slab DOS not converging to the bulk: {errs}"
        assert errs[-1] < 0.15 * target

    def test_too_thin_a_slab_warns(self):
        from waw.analysis.surface import build_slab
        hr, a, _ = _cubic_tb()
        with pytest.warns(UserWarning, match="hybridise"):
            build_slab(hr, a * np.eye(3), (0, 0, 1), 1)

    def test_layer_weights_select_one_face(self):
        from waw.analysis.surface import build_slab, slab_layer_weights
        hr, a, _ = _cubic_tb()
        slab = build_slab(hr, a * np.eye(3), (0, 0, 1), 5)
        bot = slab_layer_weights(slab, 2, "bottom")
        top = slab_layer_weights(slab, 2, "top")
        assert bot.sum() == 2 and top.sum() == 2
        assert bot[0] == 1 and bot[-1] == 0
        assert top[-1] == 1 and top[0] == 0
