"""
Tests for waw/analysis/orbital_magnetization.py — tutorial19.

Strategy: cross-check `_imfgh_chunk`'s -2Im[f] term against the ALREADY
cross-validated (real wannier90.x, tutorial18) `topology.
wannier_interpolated_curvature`, on identical synthetic (hr, AA_R) input --
both functions build H0/UU/JJp/JJm/f_proj/A_k/omega_bar_k the same way, so
an exact match is a strong, free correctness check on the new module's
shared machinery, independent of BB_R/CC_R (already unit-tested directly
in test_hamiltonian.py::TestBBR/TestCCR against an explicit-loop reference).
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR, position_operator_k
from waw.analysis._fourier_derivs import h_and_grad_frac_batch
from waw.analysis.topology import wannier_interpolated_curvature
from waw.analysis.orbital_magnetization import _imfgh_chunk, orbital_magnetization

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


def test_imf_matches_wannier_interpolated_curvature():
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)

    rng = np.random.default_rng(0)
    nR = len(hr.R_vectors)
    AA_R = torch.tensor(rng.normal(size=(3, nR, 2, 2)) + 1j * rng.normal(size=(3, nR, 2, 2)))
    AA_R = 0.5 * (AA_R + AA_R.conj().transpose(-1, -2))   # Hermitize, matching compute_position_r

    kpts = rng.uniform(0, 1, size=(50, 3))
    kpts[:, 2] = 0.0
    fermi_e = np.array([0.0, 1.5])

    ref = wannier_interpolated_curvature(hr, AA_R, recip_lattice, real_lattice, kpts, fermi_e)

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.complex128)
    H0, grad = h_and_grad_frac_batch(hr, kpts)
    grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad)
    A_k, omega_bar_k = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kpts)
    BB_k = torch.zeros_like(A_k)
    CC_k = torch.zeros(A_k.shape[0], 3, 3, 2, 2, dtype=torch.complex128)

    imf, img, imh = _imfgh_chunk(H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k, fermi_e)

    np.testing.assert_allclose(imf, ref, atol=1e-12, rtol=0)


def test_imgh_J2_only_survives_when_AA_BB_CC_are_zero():
    """
    With AA_R = BB_R = CC_R = 0 identically, every J0/J1 trace in
    -2Im[f]/-2Im[g]/-2Im[h] contracts at least one of A, Omega_bar, BB, or
    Lambda (all zero) -- EXCEPT each quantity's own J2 term
    (-2Im[Tr(JJm.JJp)]-type, imh's additionally sandwiching HH), which
    depends only on JJm/JJp/HH and is generally nonzero (the same reason
    `test_analysis_topology.py::test_ahc_matches_closed_form_prediction_
    from_chern_number` gets a nonzero AHC from AA_R=0 -- the pure-J2/Chern
    contribution). So imf/img/imh here must equal their OWN J2 pieces
    exactly, computed independently without going through `_imfgh_chunk`.
    """
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    kpts = np.random.default_rng(1).uniform(0, 1, size=(20, 3))
    fermi_e = np.array([0.3])

    inv_recip = torch.as_tensor(np.linalg.inv(2 * np.pi * np.eye(3)), dtype=torch.complex128)
    H0, grad = h_and_grad_frac_batch(hr, kpts)
    grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad)
    A_k = torch.zeros(len(kpts), 3, 2, 2, dtype=torch.complex128)
    omega_bar_k = torch.zeros_like(A_k)
    BB_k = torch.zeros_like(A_k)
    CC_k = torch.zeros(len(kpts), 3, 3, 2, 2, dtype=torch.complex128)

    imf, img, imh = _imfgh_chunk(H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k, fermi_e)

    from waw.analysis.topology import _jjp_jjm_batch, _AXIAL_PAIRS

    eig, UU = torch.linalg.eigh(H0)
    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)
    JJp_e, JJm_e = _jjp_jjm_batch(dH_eig, eig, float(fermi_e[0]))
    JJp = torch.einsum('kin,kanm,kjm->kaij', UU, JJp_e, UU.conj())
    JJm = torch.einsum('kin,kanm,kjm->kaij', UU, JJm_e, UU.conj())

    imf_j2 = np.empty((len(kpts), 1, 3))
    img_j2 = np.empty((len(kpts), 1, 3))
    imh_j2 = np.empty((len(kpts), 1, 3))
    for comp, (alpha, beta) in enumerate(_AXIAL_PAIRS):
        JJma, JJpb = JJm[:, alpha], JJp[:, beta]
        imf_j2[:, 0, comp] = (-2.0 * torch.einsum('kij,kji->k', JJma, JJpb).imag).cpu().numpy()
        img_j2[:, 0, comp] = (-2.0 * torch.einsum(
            'kij,kji->k', torch.matmul(JJma, H0), JJpb).imag).cpu().numpy()
        imh_j2[:, 0, comp] = (-2.0 * torch.einsum(
            'kij,kji->k', torch.matmul(H0, JJma), JJpb).imag).cpu().numpy()

    np.testing.assert_allclose(imf, imf_j2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(img, img_j2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(imh, imh_j2, atol=1e-10, rtol=0)


def test_orbital_magnetization_zero_for_zero_position_operators():
    """End-to-end `orbital_magnetization` on a small mesh must also give
    exactly zero when AA_R/BB_R/CC_R are all zero."""
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)
    nR = len(hr.R_vectors)
    AA_R = torch.zeros(3, nR, 2, 2, dtype=torch.complex128)
    BB_R = torch.zeros(3, nR, 2, 2, dtype=torch.complex128)
    CC_R = torch.zeros(3, 3, nR, 2, 2, dtype=torch.complex128)

    result = orbital_magnetization(hr, AA_R, BB_R, CC_R, recip_lattice, real_lattice,
                                   fermi_energies=0.0, mesh=(10, 10, 1))
    assert result.m_orb.shape == (1, 3)
    np.testing.assert_allclose(result.m_orb, 0.0, atol=1e-10)
