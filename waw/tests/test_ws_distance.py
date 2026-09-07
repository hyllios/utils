"""
Tests for waw.core.ws_distance (use_ws_distance: Wannier-centre-aware
minimal-image interpolation, wannier90's default). Validated via:

  1. exactness at on-mesh k-points (Fourier series exact there regardless
     of the (i,j)-dependent phase),
  2. reduction to plain interpolation when all WF centres coincide,
  3. agreement of the vectorized precompute against an independent
     brute-force minimal-image reference, on a case where the shift
     actually triggers (large WF offset vs a small supercell).
"""

import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR, interpolate_bands, operator_k
from waw.core.ws_distance import build_ws_distance
from waw.analysis.dos import _uniform_mesh


def _two_band(offdiag=0.02):
    a = 5.0
    real_lattice = np.diag([a, 20.0, 20.0])
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    H_R = np.zeros((3, 2, 2), dtype=complex)
    H_R[0] = [[0.1, offdiag], [offdiag, -0.1]]
    H_R[1] = np.array([[0.03, offdiag], [offdiag, -0.05]])
    H_R[2] = H_R[1].conj().T
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(3), nw=2)
    return hr, real_lattice


def _brute_ws_phase(R_vectors, centres, mp_grid, real_lattice, kpt, tol=1e-5, w=3):
    """Independent reference: per (i,j,R) minimal-image phase, no precompute."""
    N = np.asarray(mp_grid)
    nR, nw = R_vectors.shape[0], centres.shape[0]
    R_cart = R_vectors @ real_lattice
    ph = np.zeros((nR, nw, nw), dtype=complex)
    shifts = np.array([[a, b, c] for a in range(-w, w + 1)
                       for b in range(-w, w + 1) for c in range(-w, w + 1)])
    shift_int = shifts * N[None, :]
    shift_cart = shift_int @ real_lattice
    for r in range(nR):
        for i in range(nw):
            for j in range(nw):
                T = R_cart[r] + centres[j] - centres[i]
                d = np.linalg.norm(T[None, :] + shift_cart, axis=1)
                dmin = d.min()
                deg = np.abs(d - dmin) < tol
                Rp = R_vectors[r][None, :] + shift_int[deg]
                ph[r, i, j] = np.exp(2j * np.pi * (Rp @ kpt)).sum() / deg.sum()
    return ph


def test_ws_exact_at_mesh_points():
    hr, real_lattice = _two_band()
    centres = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    ws = build_ws_distance(hr.R_vectors, centres, (4, 1, 1), real_lattice)
    kmesh = _uniform_mesh((4, 1, 1))
    b_plain = interpolate_bands(hr, kmesh, ws=None)
    b_ws = interpolate_bands(hr, kmesh, ws=ws)
    assert np.abs(b_plain - b_ws).max() < 1e-12


def test_ws_reduces_to_plain_when_centres_coincide():
    hr, real_lattice = _two_band()
    ws0 = build_ws_distance(hr.R_vectors, np.zeros((2, 3)), (4, 1, 1), real_lattice)
    kpath = np.stack([np.linspace(0, 0.5, 25), np.zeros(25), np.zeros(25)], axis=1)
    assert np.abs(interpolate_bands(hr, kpath, ws=None)
                  - interpolate_bands(hr, kpath, ws=ws0)).max() < 1e-12


def test_ws_phase_matches_bruteforce_when_shift_triggers():
    """Small supercell (3a=15 Bohr) + large WF offset (6 Bohr) forces the
    off-diagonal (0,1) block to be remapped to a farther cell -- verify the
    vectorized precompute against the brute-force minimal-image reference."""
    hr, real_lattice = _two_band()
    centres = np.array([[0.0, 0, 0], [6.0, 0, 0]])
    mp_grid = (3, 1, 1)
    ws = build_ws_distance(hr.R_vectors, centres, mp_grid, real_lattice)
    # the shift must actually have triggered somewhere (offdiag block relabeled)
    assert not np.array_equal(ws.shiftedR[:, :, :, 0, :], np.broadcast_to(
        hr.R_vectors[:, None, None, :], ws.shiftedR[:, :, :, 0, :].shape))
    for kpt in [np.array([0.1, 0, 0]), np.array([0.3, 0, 0]), np.array([0.5, 0, 0])]:
        ref = _brute_ws_phase(hr.R_vectors, centres, mp_grid, real_lattice, kpt)
        got = ws.phase(kpt)
        assert np.abs(ref - got).max() < 1e-10


def test_ws_changes_offmesh_operator_when_offset_large():
    """With a triggering offset, ws-distance must actually change the
    off-mesh interpolated H(k) operator (an (i,j)-dependent phase on the
    off-diagonal blocks) -- else it is a silent no-op. NB: for a 2-band
    model this changes the eigenVECTORS (hence spin) but not the
    eigenVALUES, since 2x2 eigenvalues depend only on |H_01|; see the
    3-band test below for an energy change."""
    hr, real_lattice = _two_band()
    centres = np.array([[0.0, 0, 0], [6.0, 0, 0]])
    ws = build_ws_distance(hr.R_vectors, centres, (3, 1, 1), real_lattice)
    kpath = np.stack([np.linspace(0.05, 0.45, 15), np.zeros(15), np.zeros(15)], axis=1)
    Ok_plain = operator_k(hr.H_R, hr.R_vectors, hr.degen, kpath).detach().cpu().numpy()
    Ok_ws = operator_k(hr.H_R, hr.R_vectors, hr.degen, kpath, ws=ws).detach().cpu().numpy()
    assert np.abs(Ok_plain - Ok_ws).max() > 1e-3


def _three_band(offdiag=0.03):
    a = 5.0
    real_lattice = np.diag([a, 20.0, 20.0])
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    H_R = np.zeros((3, 3, 3), dtype=complex)
    H_R[0] = np.diag([0.2, 0.0, -0.2]).astype(complex)
    H_R[0, 0, 1] = H_R[0, 1, 0] = offdiag
    H_R[0, 1, 2] = H_R[0, 2, 1] = offdiag
    H_R[1] = np.diag([0.03, 0.0, -0.04]).astype(complex)
    H_R[1, 0, 2] = offdiag            # long-range 0<->2 coupling
    H_R[2] = H_R[1].conj().T
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(3), nw=3)
    return hr, real_lattice


def test_ws_changes_bands_for_three_band_model():
    """For >=3 bands the off-diagonal phases genuinely shift eigenvalues."""
    hr, real_lattice = _three_band()
    centres = np.array([[0.0, 0, 0], [0.0, 0, 0], [6.0, 0, 0]])
    ws = build_ws_distance(hr.R_vectors, centres, (3, 1, 1), real_lattice)
    kpath = np.stack([np.linspace(0.05, 0.45, 15), np.zeros(15), np.zeros(15)], axis=1)
    diff = np.abs(interpolate_bands(hr, kpath, ws=None)
                  - interpolate_bands(hr, kpath, ws=ws)).max()
    assert diff > 1e-4
    # still exact at mesh points
    kmesh = _uniform_mesh((3, 1, 1))
    assert np.abs(interpolate_bands(hr, kmesh, ws=None)
                  - interpolate_bands(hr, kmesh, ws=ws)).max() < 1e-12
