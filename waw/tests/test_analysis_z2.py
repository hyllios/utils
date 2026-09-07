"""
Tests for waw/analysis/z2.py.

Validates the Wilson-loop Z2 machinery against synthetic models with
provably/independently known answers (no real material/DFT involved, and
no real wannier90/postw90 reference exists for this capability at all --
it isn't a Wannier90 feature):

1. A time-reversal-doubled Qi-Wu-Zhang (QWZ) model, H_D(k) = diag(H_QWZ(k, u),
   conj(H_QWZ(-k, u))), is a standard textbook construction of a genuine 2D
   quantum-spin-Hall insulator with the provable identity Z2 = C_QWZ mod 2
   (reusing the same QWZ model already validated in test_analysis_topology.py).

2. A 4-band Clifford-algebra Dirac lattice model (H(k) = sin(kx)*Gamma1 +
   sin(ky)*Gamma2 + sin(kz)*Gamma3 + [u - t*(cos kx+cos ky+cos kz)]*Gamma4,
   the standard minimal lattice model for a strong 3D topological
   insulator) is cross-validated per-plane against Z2Pack (Gresch et al.,
   Comput. Phys. Commun. 224, 165 (2018)) in test_analysis_z2_vs_z2pack.py
   -- exact agreement across the trivial/weak/strong-TI regimes, including
   the physically-required x/y/z permutation symmetry of this cubic model.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.topology import chern_number
from waw.analysis.z2 import z2_invariant_plane, z2_invariants_3d

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_blocks(u: float) -> dict:
    """H(R) blocks (2x2) of the QWZ model, keyed by R = (r1, r2, 0)."""
    return {
        (1, 0, 0):  (-1j / 2) * SX + 0.5 * SZ,
        (-1, 0, 0): (1j / 2) * SX + 0.5 * SZ,
        (0, 1, 0):  (-1j / 2) * SY + 0.5 * SZ,
        (0, -1, 0): (1j / 2) * SY + 0.5 * SZ,
        (0, 0, 0):  u * SZ,
    }


def _doubled_qwz_hr(u: float) -> HamiltonianR:
    """
    4-band time-reversal-doubled QWZ model: block_diag(H_QWZ(k,u),
    conj(H_QWZ(-k,u))). In real space, H_QWZ(-k) at R is H_QWZ(k) at -R
    (Fourier transform flips sign of R under k -> -k), so the second
    block's R-space blocks are the complex conjugate of the first block's
    R -> -R.
    """
    blocks = _qwz_blocks(u)
    R_list = list(blocks.keys())
    H_R = np.zeros((len(R_list), 4, 4), dtype=np.complex128)
    for i, R in enumerate(R_list):
        H_R[i, :2, :2] = blocks[R]
        H_R[i, 2:, 2:] = blocks[tuple(-r for r in R)].conj()

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                        R_vectors=np.array(R_list, dtype=np.int64),
                        degen=np.ones(len(R_list), dtype=np.int64), nw=4)


@pytest.mark.parametrize("u", [1.0, -1.0, 1.5])
def test_z2_plane_matches_chern_mod_2_topological(u):
    hr = _doubled_qwz_hr(u)
    c = chern_number(hr, plane=(0, 1), mesh=(30, 30), groups=((0,),)).chern[0]
    expected = int(round(abs(c))) % 2
    z2 = z2_invariant_plane(hr, band_group=(0, 1), plane=(0, 1),
                            fixed_index=2, fixed_value=0.0, mesh=41, n_loop=21)
    assert z2 == expected


@pytest.mark.parametrize("u", [3.0, -3.0])
def test_z2_plane_trivial_regime(u):
    hr = _doubled_qwz_hr(u)
    z2 = z2_invariant_plane(hr, band_group=(0, 1), plane=(0, 1),
                            fixed_index=2, fixed_value=0.0, mesh=41, n_loop=21)
    assert z2 == 0


def test_z2_plane_independent_of_fixed_value_for_this_kz_independent_model():
    # the doubled-QWZ model has no kz dependence at all (R_vectors' 3rd
    # component is always 0), so the "kz=0" and "kz=0.5" planes must give
    # the identical Z2 -- a strong, model-specific consistency check.
    hr = _doubled_qwz_hr(1.0)
    z2_0 = z2_invariant_plane(hr, band_group=(0, 1), plane=(0, 1),
                              fixed_index=2, fixed_value=0.0, mesh=41, n_loop=21)
    z2_half = z2_invariant_plane(hr, band_group=(0, 1), plane=(0, 1),
                                 fixed_index=2, fixed_value=0.5, mesh=41, n_loop=21)
    assert z2_0 == z2_half


def test_z2_invariants_3d_wiring_matches_direct_plane_calls():
    """
    Plumbing check (not new physics): z2_invariants_3d must reproduce
    exactly the same six per-plane values as calling z2_invariant_plane
    directly with the same (plane, fixed_index, fixed_value) arguments.
    """
    hr = _doubled_qwz_hr(1.0)
    result = z2_invariants_3d(hr, band_group=(0, 1), mesh=41, n_loop=21)
    expected = {
        "x0": z2_invariant_plane(hr, (0, 1), (1, 2), 0, 0.0, mesh=41, n_loop=21),
        "x1": z2_invariant_plane(hr, (0, 1), (1, 2), 0, 0.5, mesh=41, n_loop=21),
        "y0": z2_invariant_plane(hr, (0, 1), (2, 0), 1, 0.0, mesh=41, n_loop=21),
        "y1": z2_invariant_plane(hr, (0, 1), (2, 0), 1, 0.5, mesh=41, n_loop=21),
        "z0": z2_invariant_plane(hr, (0, 1), (0, 1), 2, 0.0, mesh=41, n_loop=21),
        "z1": z2_invariant_plane(hr, (0, 1), (0, 1), 2, 0.5, mesh=41, n_loop=21),
    }
    assert result.z2_planes == expected


def test_z2_invariants_3d_trivial_atomic_limit():
    """A large-gap (deep trivial, |u|>>2) 3D system must give nu0=0
    consistently on all three normal directions."""
    hr = _doubled_qwz_hr(5.0)
    result = z2_invariants_3d(hr, band_group=(0, 1), mesh=41, n_loop=21)
    assert result.consistent
    assert result.nu0 == 0
