"""
Tests for waw.analysis.specific_heat (electronic heat capacity at
fixed chemical potential, following Phoebe's specific_heat.cpp). No
external reference exists (wannier90/BoltzWann has no such observable),
so these check physical invariants only: positivity (a provable,
model-independent property -- see module docstring), vanishing deep in
a gap, and linear scaling with num_elec_per_state.
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis import electronic_specific_heat
from waw.units import to_si_units

K_B_HARTREE = 3.166811563e-6
AU_TIME_PER_FS = 1.0 / 2.4188843265857e-2


def _cubic_tb_hr(t: float = -0.01, a: float = 5.0) -> HamiltonianR:
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    H_R = np.zeros((len(R), 1, 1), dtype=complex)
    for i in range(1, 7):
        H_R[i, 0, 0] = t
    return HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(len(R)), nw=1)


def _gapped_hr(gap: float = 0.5, t: float = 0.01, a: float = 5.0) -> HamiltonianR:
    """Two flat-ish bands separated by `gap`, well isolated (insulator toy)."""
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    H_R = np.zeros((3, 2, 2), dtype=complex)
    H_R[0] = np.diag([gap / 2, -gap / 2])
    H_R[1] = np.diag([t, -t])
    H_R[2] = np.diag([t, -t])
    return HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(3), nw=2)


def test_specific_heat_is_nonnegative():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    for mu in [-0.05, 0.0, 0.03]:
        for kT in [50 * K_B_HARTREE, 500 * K_B_HARTREE]:
            c_v = electronic_specific_heat(hr, real_lattice, recip_lattice, (12, 12, 12), mu, kT)
            assert c_v >= 0.0


def test_specific_heat_vanishes_deep_in_gap():
    hr = _gapped_hr(gap=0.5)
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    c_v = electronic_specific_heat(hr, real_lattice, recip_lattice, (10, 1, 1),
                                    mu=0.0, kT=10 * K_B_HARTREE)
    assert c_v < 1e-10


def test_specific_heat_peaks_near_a_band():
    """mu tracking into the gapped model's lower band should raise C_v well
    above the deep-gap value, since states become thermally accessible."""
    hr = _gapped_hr(gap=0.5)
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kT = 300 * K_B_HARTREE   # comparable to the toy band's 0.02 Ha width
    c_v_gap = electronic_specific_heat(hr, real_lattice, recip_lattice, (30, 1, 1), mu=0.0, kT=kT)
    c_v_band = electronic_specific_heat(hr, real_lattice, recip_lattice, (30, 1, 1), mu=-0.25, kT=kT)
    assert c_v_band > 100 * max(c_v_gap, 1e-300)


def test_specific_heat_scales_linearly_with_num_elec_per_state():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    c1 = electronic_specific_heat(hr, real_lattice, recip_lattice, (10, 10, 10),
                                  mu=0.0, kT=100 * K_B_HARTREE, num_elec_per_state=1)
    c2 = electronic_specific_heat(hr, real_lattice, recip_lattice, (10, 10, 10),
                                  mu=0.0, kT=100 * K_B_HARTREE, num_elec_per_state=2)
    assert c2 == pytest.approx(2 * c1)


def test_to_si_units_are_positive_and_finite():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kT = 300 * K_B_HARTREE
    c_v = electronic_specific_heat(hr, real_lattice, recip_lattice, (10, 10, 10),
                                    mu=0.0, kT=kT)
    c_v_si = to_si_units(c_v, "specific_heat", kT=kT)
    assert np.isfinite(c_v_si) and c_v_si >= 0.0
