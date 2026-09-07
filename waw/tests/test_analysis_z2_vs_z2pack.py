"""
Cross-validates waw/analysis/z2.py against Z2Pack (Gresch et al., Comput.
Phys. Commun. 224, 165 (2018)) -- an independent, published Wilson-loop
Z2 implementation -- on a 4-band Clifford-algebra Dirac lattice model, the
standard minimal lattice model for a 3D strong topological insulator:

    H(k) = sin(kx)*G1 + sin(ky)*G2 + sin(kz)*G3
           + [u - t*(cos kx + cos ky + cos kz)]*G4

with Gamma matrices G1=tau_x{kron}sigma_x, G2=tau_x{kron}sigma_y,
G3=tau_x{kron}sigma_z, G4=tau_z{kron}I (inversion operator P=G4). Every
band is exactly doubly (Kramers) degenerate everywhere in the BZ, the
same generic structure as a real centrosymmetric+time-reversal-symmetric
material (e.g. the beta-PdBi2 target of this project's Stage B/C work) --
unlike the 2-band QWZ model in test_analysis_z2.py, so this is the
relevant validation for real multi-band systems.

This test caught a real bug: an earlier version of `wilson_loop_wcc` used
an overlap-matrix dagger convention that does not telescope to a gauge-
invariant loop product for exactly-degenerate multi-band subspaces
(confirmed by direct regauging experiments -- see z2.py's git history) --
it silently gave wrong, x/y/z-symmetry-violating per-plane Z2 values that
did not converge with mesh refinement. Fixed by daggering the destination
(not source) eigenvectors in the link variable.
"""

from pathlib import Path
import logging
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

z2pack = pytest.importorskip("z2pack")

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.z2 import z2_invariant_plane, z2_invariants_3d

logging.disable(logging.CRITICAL)   # z2pack logs verbosely by default

_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_TX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_TZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_G1 = np.kron(_TX, _SX)
_G2 = np.kron(_TX, _SY)
_G3 = np.kron(_TX, _SZ)
_G4 = np.kron(_TZ, _I2)


def _dirac_hr(u: float, t: float = 1.0) -> HamiltonianR:
    """Nearest-neighbour real-space form of the Clifford-Dirac model above."""
    blocks = {
        (1, 0, 0):  -1j * _G1 / 2 - t * _G4 / 2, (-1, 0, 0): 1j * _G1 / 2 - t * _G4 / 2,
        (0, 1, 0):  -1j * _G2 / 2 - t * _G4 / 2, (0, -1, 0): 1j * _G2 / 2 - t * _G4 / 2,
        (0, 0, 1):  -1j * _G3 / 2 - t * _G4 / 2, (0, 0, -1): 1j * _G3 / 2 - t * _G4 / 2,
        (0, 0, 0):  u * _G4,
    }
    R_list = list(blocks.keys())
    H_R = np.stack([blocks[R] for R in R_list])
    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                        R_vectors=np.array(R_list, dtype=np.int64),
                        degen=np.ones(len(R_list), dtype=np.int64), nw=4)


def _dirac_h_direct(k, u: float, t: float = 1.0) -> np.ndarray:
    kx, ky, kz = 2 * np.pi * k[0], 2 * np.pi * k[1], 2 * np.pi * k[2]
    return (np.sin(kx) * _G1 + np.sin(ky) * _G2 + np.sin(kz) * _G3
            + (u - t * (np.cos(kx) + np.cos(ky) + np.cos(kz))) * _G4)


def _z2pack_plane(u: float, fixed_index: int, fixed_value: float) -> int:
    """z2pack's own Z2 of the TRIM plane `fixed_index = fixed_value`."""
    system = z2pack.hm.System(lambda k: _dirac_h_direct(k, u), dim=3, bands=2)
    free = [i for i in range(3) if i != fixed_index]

    def surface(t1, t2):
        k = [None, None, None]
        k[fixed_index] = fixed_value
        k[free[0]] = t1 / 2
        k[free[1]] = t2
        return k

    result = z2pack.surface.run(system=system, surface=surface, num_lines=21,
                                iterator=range(21, 42, 4), pos_tol=1e-3,
                                gap_tol=0.1, move_tol=0.2)
    return z2pack.invariant.z2(result)


@pytest.mark.parametrize("u,fixed_index,fixed_value", [
    (2.0, 2, 0.0), (2.0, 2, 0.5),
    (-2.0, 2, 0.0), (-2.0, 2, 0.5),
    (0.5, 2, 0.0), (4.0, 2, 0.0),
])
def test_z2_plane_matches_z2pack(u, fixed_index, fixed_value):
    hr = _dirac_hr(u)
    waw_z2 = z2_invariant_plane(hr, band_group=(0, 1), plane=(0, 1),
                                fixed_index=fixed_index, fixed_value=fixed_value,
                                mesh=41, n_loop=21)
    ref_z2 = _z2pack_plane(u, fixed_index, fixed_value)
    assert waw_z2 == ref_z2


@pytest.mark.parametrize("u,expected_nu0", [(2.0, 1), (-2.0, 1), (0.5, 0), (4.0, 0)])
def test_z2_invariants_3d_strong_ti_matches_z2pack_phase_diagram(u, expected_nu0):
    """
    The full 3D wiring on a genuine (Z2Pack-cross-validated) strong-TI
    model: nu0=1 with all three normal directions agreeing, matching the
    x/y/z permutation symmetry the cubic Dirac model requires.
    """
    hr = _dirac_hr(u)
    result = z2_invariants_3d(hr, band_group=(0, 1), mesh=41, n_loop=21)
    assert result.consistent, f"nu0 disagreement across planes: {result.z2_planes}"
    assert result.nu0 == expected_nu0
    assert result.z2_planes["x0"] == result.z2_planes["y0"] == result.z2_planes["z0"]
    assert result.z2_planes["x1"] == result.z2_planes["y1"] == result.z2_planes["z1"]
