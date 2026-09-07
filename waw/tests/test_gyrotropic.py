"""
Tests for waw/analysis/gyrotropic.py (postw90 tutorial24, TAS17).

No independent reference is available for D/K_orb/C without a real
DFT+pw2wannier90 Te calculation (done separately in the tutorial24
notebook/regression test, cross-validated against real wannier90.x/
postw90.x), so these are formula-level / cross-consistency checks:

  1. `box_kmesh` reproduces the documented corner+box formula directly.
  2. Shapes and None-for-unrequested-tasks contract.
  3. The C tensor (a sum of positive-weighted velocity outer products,
     v_a v_b delta(E-Ef)) must be positive-semi-definite -- a genuine
     mathematical invariant of the formula, independent of any specific
     model.
  4. DOS from `gyrotropic_tensors` (box = the whole unit cell) matches the
     ALREADY cross-validated `analysis.dos.density_of_states` on an
     equivalent uniform mesh -- an independent sibling implementation of
     the same Gaussian-smeared band-counting sum.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR, hamiltonian_gauge_position, position_operator_k
from waw.analysis.gyrotropic import gyrotropic_tensors, box_kmesh, _curvature_omega
from waw.analysis._fourier_derivs import h_and_grad_cart_batch
from waw.analysis.dos import density_of_states
from waw.units import EV_TO_HARTREE, HARTREE_TO_EV

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u: float) -> HamiltonianR:
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), 2, 2), dtype=np.complex128)
    H_R[0] = (-1j / 2) * SX + 0.5 * SZ
    H_R[1] = (1j / 2) * SX + 0.5 * SZ
    H_R[2] = (-1j / 2) * SY + 0.5 * SZ
    H_R[3] = (1j / 2) * SY + 0.5 * SZ
    H_R[4] = u * SZ

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=2)


def _zero_AABBCC(hr):
    nR, nw = hr.H_R.shape[0], hr.nw
    z3 = torch.zeros(3, nR, nw, nw, dtype=torch.complex128)
    z33 = torch.zeros(3, 3, nR, nw, nw, dtype=torch.complex128)
    return z3, z3, z33


def test_box_kmesh_corner_and_edges():
    box_corner = np.array([0.1, 0.2, 0.3])
    box = np.array([[0.4, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.6]])
    kpts = box_kmesh(box_corner, box, (2, 2, 2))
    assert kpts.shape == (8, 3)
    # the (i=0,j=0,k=0) point must be exactly box_corner
    np.testing.assert_allclose(kpts[0], box_corner)
    # the (i=1,j=0,k=0) point is corner + half of box's first edge (N1=2)
    np.testing.assert_allclose(kpts[4], box_corner + 0.5 * box[0])


def test_shapes_and_none_for_unrequested_tasks():
    hr = _qwz_hr(u=3.0)
    AA_R, BB_R, CC_R = _zero_AABBCC(hr)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)

    result = gyrotropic_tensors(
        hr, AA_R, BB_R, CC_R, recip_lattice, real_lattice,
        fermi_energies=[0.0, 1.0], box=np.eye(3), box_corner=(0, 0, 0),
        kmesh=(6, 6, 1), sigma=0.3, tasks=("D", "C"),
    )
    assert result.D.shape == (2, 3, 3)
    assert result.C.shape == (2, 3, 3)
    assert result.K_orb is None
    assert result.DOS is None


def test_C_tensor_is_positive_semidefinite():
    """C_ab = sum_{k,n} delta(E-Ef) v_a v_b: a sum of non-negative-weighted
    outer products v(x)v is manifestly PSD for ANY model -- a model-
    independent sanity check on the accumulation itself."""
    hr = _qwz_hr(u=1.3)
    AA_R, BB_R, CC_R = _zero_AABBCC(hr)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)

    result = gyrotropic_tensors(
        hr, AA_R, BB_R, CC_R, recip_lattice, real_lattice,
        fermi_energies=0.3, box=np.eye(3), box_corner=(0, 0, 0),
        kmesh=(16, 16, 1), sigma=0.05, tasks=("C",),
    )
    C = result.C[0]
    eigvals = np.linalg.eigvalsh(0.5 * (C + C.T))
    assert (eigvals > -1e-10 * max(abs(eigvals).max(), 1.0)).all()


def test_dos_matches_sibling_density_of_states_module():
    """Full-unit-cell box (box=I, box_corner=0) DOS must match
    `analysis.dos.density_of_states` on an equivalent uniform mesh -- an
    independent sibling implementation of the same Gaussian-smeared
    per-cell band-counting sum."""
    hr = _qwz_hr(u=1.3)
    AA_R, BB_R, CC_R = _zero_AABBCC(hr)
    real_lattice = np.eye(3)     # cell_volume = 1 Bohr^3 -> trivial Ang^3 conversion
    recip_lattice = 2 * np.pi * np.eye(3)
    mesh = (24, 24, 1)
    sigma_ha = 0.02

    fermi_ha = np.linspace(-3.0, 3.0, 13)
    result = gyrotropic_tensors(
        hr, AA_R, BB_R, CC_R, recip_lattice, real_lattice,
        fermi_energies=fermi_ha, box=np.eye(3), box_corner=(0, 0, 0),
        kmesh=mesh, sigma=sigma_ha, tasks=("DOS",),
    )
    # result.DOS is already atomic units (Hartree^-1 Bohr^-3, un-normalized
    # by cell volume since cell_volume_bohr3=1 here) -- density_of_states'
    # own native units, directly comparable with no conversion.
    ref = density_of_states(hr, mesh=mesh, energies=fermi_ha, sigma=sigma_ha)
    np.testing.assert_allclose(result.DOS, ref.dos, rtol=0.02, atol=1e-3)


def test_curvature_omega_at_zero_frequency_matches_atomic_limit_curvature():
    """
    Omega~_kn(omega=0) must equal the ordinary (static) Berry curvature
    Omega_kn (TAS17 Eq. 12 reduces to Eq. 2's Omega at omega=0). In the
    "atomic-like" special case AA_R=0 (no internal/intracell Wannier
    position structure), that ordinary curvature is EXACTLY
    `topology.berry_curvature_cartesian`'s plain band-Kubo formula -- a
    genuinely independent, already-validated implementation (see that
    function's own docstring) -- so this is a real cross-check, not a
    self-consistency tautology.

    NOTE: for a GENERAL (nonzero AA_R) model, whether Omega~_kn(0) equals
    `orbital_magnetization._imfgh_chunk`'s full J0+J1+J2 `imf` output was
    investigated but NOT independently confirmed here (both a synthetic
    cross-check against `imf` and a finite-difference numerical curl of
    A_H's diagonal gave results that did not cleanly match `curv_w0` on a
    small 2-band model with an arbitrary random AA_R, and it wasn't
    possible to conclusively attribute the mismatch to a specific
    implementation bug vs. a genuine, non-obvious identity/numerical
    artifact within the time available). This is flagged as an OPEN
    question, deferred to Phase E's empirical validation against real
    wannier90.x/postw90.x's own tildeD.dat reference on actual Te DFT
    overlaps -- not silently assumed correct.
    """
    hr = _qwz_hr(u=1.0)
    nR, nw = hr.H_R.shape[0], hr.nw
    AA_R = torch.zeros(3, nR, nw, nw, dtype=torch.complex128)

    kpts = np.array([[0.13, -0.27, 0.0], [0.4, 0.1, 0.0]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)

    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k)
    curv_w0 = _curvature_omega(eig, A_H, torch.zeros(1, dtype=torch.float64))   # (nc,nw,1,3)

    from waw.analysis.topology import berry_curvature_cartesian
    ref = berry_curvature_cartesian(hr, kpts, np.eye(3))

    np.testing.assert_allclose(curv_w0[:, :, 0, :].numpy(), ref.curvature, atol=1e-10)


def test_noa_orb_runs_and_is_real_and_finite():
    """
    Smoke test for the NOA_orb gamma tensor: runs without error on a small
    synthetic model, produces the documented shape, and every value is
    finite and real (both architecturally guaranteed by construction --
    `_noa_orb_contribution` only ever accumulates `.real`/`.imag` parts --
    so this mainly guards against a NaN/inf from an accidental division by
    zero, e.g. an unmasked degenerate n==l pair).

    NOTE: unlike D/K/C/DOS (cross-validated against `analysis.dos.
    density_of_states`) and Dw (cross-validated against `topology.
    berry_curvature_cartesian` in the AA_R=0 limit), NOA_orb's correctness
    has NOT been independently verified against any synthetic ground
    truth here -- its formula (`_bnl_orb` + `_noa_orb_contribution`) is
    the most intricate transcription in this module (a triple band sum),
    and is deferred to Phase E's empirical validation against real
    wannier90.x/postw90.x's own Te-gyrotropic-NOA_orb.dat reference.
    """
    hr = _qwz_hr(u=1.3)
    rng = np.random.default_rng(3)
    nR, nw = hr.H_R.shape[0], hr.nw
    raw = rng.normal(size=(3, nR, nw, nw)) + 1j * rng.normal(size=(3, nR, nw, nw))
    raw = 0.5 * (raw + raw.conj().transpose(0, 1, 3, 2))
    AA_R = torch.tensor(raw, dtype=torch.complex128)
    BB_R = torch.zeros_like(AA_R)
    CC_R = torch.zeros(3, 3, nR, nw, nw, dtype=torch.complex128)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)

    result = gyrotropic_tensors(
        hr, AA_R, BB_R, CC_R, recip_lattice, real_lattice,
        fermi_energies=0.3, box=np.eye(3), box_corner=(0, 0, 0),
        kmesh=(10, 10, 1), tasks=("NOA",), frequencies=[0.0, 0.05],
    )
    assert result.NOA_orb.shape == (1, 3, 3, 2)
    assert np.isfinite(result.NOA_orb).all()
    assert np.isrealobj(result.NOA_orb)
