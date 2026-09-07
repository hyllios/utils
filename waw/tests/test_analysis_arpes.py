"""ARPES photoemission matrix elements (analysis.arpes)."""

import numpy as np
import pytest

from waw.analysis.arpes import (
    photoemission_matrix_element as ME,
    wannier_lm_from_projections,
)
from waw.units import EV_TO_HARTREE

HV, WF = 50.0, 4.5
PZ, PX, PY = (1, 1), (1, 2), (1, 3)
DZ2, DXZ, DYZ, DX2Y2, DXY = (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)


def _mag(lm, eps, kpar=(0.0, 0.0)):
    k = np.array([kpar], float)
    c = np.array([[0.0, 0.0, 0.0]])
    return abs(ME(k, c, [lm], eps, HV, work_function=WF)[0, 0])


def test_dipole_selection_rules_normal_emission():
    """At normal emission only M=0 final harmonics survive: p_z (and d_z2) are
    bright for out-of-plane polarization and dark for in-plane, p_x/p_y the
    reverse, and d_xy (no m=0 component) is dark for any polarization."""
    assert _mag(PZ, [0, 0, 1]) > 1.0
    assert _mag(PZ, [1, 0, 0]) < 1e-6 * _mag(PZ, [0, 0, 1])
    assert _mag(PX, [0, 0, 1]) < 1e-6 * _mag(PX, [1, 0, 0])
    assert _mag(PY, [0, 0, 1]) < 1e-6 * _mag(PY, [0, 1, 0])
    assert _mag(DZ2, [0, 0, 1]) > 0.1
    assert _mag(DXY, [0, 0, 1]) < 1e-6 * _mag(DZ2, [0, 0, 1])
    assert _mag(DXY, [1, 0, 0]) < 1e-6 * _mag(DZ2, [0, 0, 1])


def test_structure_factor_phase():
    """Two identical orbitals differ only by the structure phase
    e^{i k_f . (r2 - r1)}."""
    kpar = np.array([[0.3, 0.1]])
    r1, r2 = np.array([0.0, 0.0, 0.0]), np.array([1.2, 0.4, 0.7])
    M = ME(kpar, np.array([r1, r2]), [PZ, PZ], [0, 0, 1], HV, work_function=WF)
    kf_mag = np.sqrt(2 * (HV - WF) * EV_TO_HARTREE)
    kz = np.sqrt(kf_mag**2 - 0.3**2 - 0.1**2)
    pred = np.exp(1j * np.array([0.3, 0.1, kz]) @ (r2 - r1))
    assert abs(M[0, 1] / M[0, 0] - pred) < 1e-9


def test_evanescent_beyond_final_momentum():
    """k_par larger than |k_f| cannot be emitted -> zero intensity."""
    kf_mag = np.sqrt(2 * (HV - WF) * EV_TO_HARTREE)
    kpar = np.array([[kf_mag + 1.0, 0.0]])
    M = ME(kpar, np.array([[0.0, 0.0, 0.0]]), [PZ], [0, 0, 1], HV, work_function=WF)
    assert np.all(M == 0.0)


def test_circular_polarization_dichroism():
    """Left vs right circular polarization give different intensity on a
    chiral orbital arrangement (nonzero circular dichroism)."""
    kpar = np.array([[0.25, 0.15]])
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    lcp = [1, 1j, 0]; rcp = [1, -1j, 0]
    Il = np.abs(ME(kpar, centres, [PX, PY], lcp, HV, work_function=WF).sum()) ** 2
    Ir = np.abs(ME(kpar, centres, [PX, PY], rcp, HV, work_function=WF).sum()) ** 2
    assert abs(Il - Ir) > 1e-6 * (Il + Ir)


def test_wannier_lm_from_projections_spinor_ordering():
    """Each projection spec yields two consecutive spinor WFs sharing (l, mr)."""
    projs = [((0, 0, 0), 2, 1, 1, (0, 0, 1), (1, 0, 0), 1.0),   # d_z2
             ((0, 0, 0), 1, 2, 1, (0, 0, 1), (1, 0, 0), 1.0)]   # p_x
    lm = wannier_lm_from_projections(projs, spinor=True)
    assert lm == [(2, 1), (2, 1), (1, 2), (1, 2)]
    assert wannier_lm_from_projections(projs, spinor=False) == [(2, 1), (1, 2)]


def test_matrix_element_shape_and_finiteness():
    kpar = np.random.default_rng(0).uniform(-0.4, 0.4, (20, 2))
    centres = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.2]])
    M = ME(kpar, centres, [DXZ, PY], [1, 0, 1], HV, work_function=WF)
    assert M.shape == (20, 2)
    assert np.all(np.isfinite(M))


def test_surface_matrix_element_contraction_single_orbital():
    """For a 1-orbital surface, surface's matrix-element intensity A_arpes must
    equal |M|^2 * A_surface (M† A_top M = |M|^2 A_top when nw=1)."""
    import torch
    from waw.core.hamiltonian import HamiltonianR
    from waw.analysis.surface import surface_spectral_function
    from waw.units import EV_TO_HARTREE

    # 1-orbital chain along z: on-site 0, nn hop -1 eV
    R = np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)
    t = -1.0 * EV_TO_HARTREE
    H_R = torch.tensor([[[0.0]], [[t]], [[t]]], dtype=torch.complex128)
    hr = HamiltonianR(H_R=H_R, R_vectors=R, degen=np.ones(3, np.int64), nw=1)
    real_lattice = np.eye(3) * 5.0     # Bohr

    kpath = np.array([[0.0, 0.0], [0.1, 0.0]])
    energies = np.linspace(-0.2, 0.2, 15) * EV_TO_HARTREE
    M = np.array([[2.0 + 1.0j], [0.5 - 0.3j]])   # (nk, nw=1)

    sf = surface_spectral_function(hr, real_lattice, (0, 0, 1), kpath, energies,
                                   eta=0.02 * EV_TO_HARTREE, matrix_element=M)
    assert sf.A_arpes is not None and sf.A_arpes.shape == (2, len(energies))
    expected = (np.abs(M[:, 0])**2)[:, None] * sf.A_surface
    assert np.allclose(sf.A_arpes, expected, rtol=1e-9, atol=1e-12)
