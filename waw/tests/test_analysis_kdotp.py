"""
Tests for waw/analysis/kdotp.py.

Uses a hand-built 3-band HamiltonianR (A, B, C) with an analytically known
structure: A and C, B and C couple linearly in kx/ky (same construction as
test_analysis_effective_mass.py's `_kane_model_hr`, already validated
there); A and B ALSO couple directly to each other and carry distinct
onsite energies, so at a generic (low-symmetry) k-point they are
NON-degenerate -- a general test of `kdotp_coefficients`'s per-band-pair
(not group-averaged) energy denominators, unlike `degenerate_effective_
mass`'s exact-degeneracy assumption.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR, interpolate_bands
from waw.analysis.kdotp import kdotp_coefficients
from waw.units import to_eVA_units, HARTREE_TO_EV, BOHR_TO_ANG

A_BOHR = 5.0


def _abc_model_hr(p=2.0, Ec=6.0, onsite_ab=(0.0, 0.3), t_ab=0.05, a=A_BOHR):
    """
    3-band model: A (0), B (1) are the k.p subset of interest; C (2) is the
    remote band. A-C/B-C couple linearly in kx/ky (WYSV-style p.k coupling,
    same pattern as `_kane_model_hr`); A-B couple directly via an ordinary
    (even-in-R) hopping `t_ab` and have distinct onsite energies, so A/B
    are non-degenerate away from special k-points.
    """
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), 3, 3), dtype=np.complex128)
    # A-C / B-C: linear-in-k coupling (odd under R -> -R)
    H_R[0, 0, 2], H_R[0, 2, 0] = p, -p         # R = (+1,0,0)
    H_R[1, 0, 2], H_R[1, 2, 0] = -p, p         # R = (-1,0,0)
    H_R[2, 1, 2], H_R[2, 2, 1] = p, -p         # R = (0,+1,0)
    H_R[3, 1, 2], H_R[3, 2, 1] = -p, p         # R = (0,-1,0)
    # A-B: ordinary hopping (even under R -> -R), same value at +-x
    H_R[0, 0, 1] = H_R[0, 1, 0] = t_ab
    H_R[1, 0, 1] = H_R[1, 1, 0] = t_ab
    # onsite
    H_R[4, 0, 0], H_R[4, 1, 1] = onsite_ab
    H_R[4, 2, 2] = Ec

    real_lattice = a * np.eye(3)
    hr = HamiltonianR(
        H_R=torch.tensor(H_R, dtype=torch.complex128),
        R_vectors=R_vectors, degen=degen, nw=3,
    )
    return hr, 2 * np.pi * np.linalg.inv(real_lattice).T


K0 = np.array([0.13, -0.07, 0.05])   # generic, low-symmetry test point


def test_zeroth_order_matches_direct_diagonalization():
    hr, recip = _abc_model_hr()
    result = kdotp_coefficients(hr, recip, K0, bands=(0, 1))
    eig_direct = interpolate_bands(hr, K0.reshape(1, 3))[0]
    np.testing.assert_allclose(result.eig0, eig_direct[[0, 1]], atol=1e-10)


def test_all_bands_reduces_to_plain_hessian_rotation():
    """With every band in the subset, there's no remote state left for the
    virtual-state correction -- second_order must reduce EXACTLY to
    0.5 * (H-gauge-rotated Hessian), independently recomputed here."""
    hr, recip = _abc_model_hr()
    result = kdotp_coefficients(hr, recip, K0, bands=(0, 1, 2))

    from waw.analysis._fourier_derivs import h_and_hess_cart_batch
    H0, grad_cart, hess_cart = h_and_hess_cart_batch(hr, K0.reshape(1, 3), recip)
    eig, UU = torch.linalg.eigh(H0[0])
    HH_dadb_bar = torch.einsum('ni,abnm,mj->abij', UU.conj(), hess_cart[0], UU)
    expected = 0.5 * HH_dadb_bar.permute(2, 3, 0, 1).detach().cpu().numpy()

    np.testing.assert_allclose(result.second_order, expected, atol=1e-10)


def test_first_order_is_hermitian_and_matches_finite_difference_velocity():
    hr, recip = _abc_model_hr()
    result = kdotp_coefficients(hr, recip, K0, bands=(0, 1))

    # Hermiticity: first_order[n,m,a] == conj(first_order[m,n,a])
    np.testing.assert_allclose(
        result.first_order, result.first_order.transpose(1, 0, 2).conj(), atol=1e-10,
    )

    # diagonal element == finite-difference dE_0/dk_a (central difference,
    # Cartesian) -- express a small Cartesian step back in fractional
    # coordinates via dk_frac = dk_cart . inv(recip_lattice).
    h = 1e-5
    for a in range(3):
        cart_step = np.zeros(3); cart_step[a] = h
        frac_step = cart_step @ np.linalg.inv(recip)   # dk_frac s.t. dk_cart = dk_frac @ recip... see below
        e_plus = interpolate_bands(hr, (K0 + frac_step).reshape(1, 3))[0]
        e_minus = interpolate_bands(hr, (K0 - frac_step).reshape(1, 3))[0]
        dE0_dka = (e_plus[0] - e_minus[0]) / (2 * h)
        np.testing.assert_allclose(result.first_order[0, 0, a].real, dE0_dka, atol=1e-6)


def test_second_order_virtual_correction_improves_kp_reconstruction():
    """
    The physically meaningful check: reconstructing H_eff(q) = eig0 + q.first_order
    + q q second_order and diagonalizing it should approximate the TRUE
    interpolated bands near k0 to O(q^3) -- i.e. adding the second-order term
    (which includes the remote-band C virtual-state correction) must reduce
    the reconstruction error compared to using only 0th+1st order, and that
    error must shrink faster than linearly as q -> 0.
    """
    hr, recip = _abc_model_hr()
    bands = (0, 1)
    result = kdotp_coefficients(hr, recip, K0, bands=bands)

    def reconstruct(q_cart, order2):
        H_eff = np.diag(result.eig0.astype(np.complex128))
        H_eff = H_eff + np.einsum('a,nma->nm', q_cart, result.first_order)
        if order2:
            H_eff = H_eff + np.einsum('a,b,nmab->nm', q_cart, q_cart, result.second_order)
        H_eff = (H_eff + H_eff.conj().T) / 2
        return np.linalg.eigvalsh(H_eff)

    errors_1st, errors_2nd = [], []
    for h in (0.02, 0.01, 0.005):
        q_cart = np.array([1.0, 0.3, -0.6]) * h
        q_frac = q_cart @ np.linalg.inv(recip)
        true_eig = interpolate_bands(hr, (K0 + q_frac).reshape(1, 3))[0][list(bands)]
        true_eig = np.sort(true_eig)

        e1 = np.sort(reconstruct(q_cart, order2=False))
        e2 = np.sort(reconstruct(q_cart, order2=True))

        errors_1st.append(np.max(np.abs(e1 - true_eig)))
        errors_2nd.append(np.max(np.abs(e2 - true_eig)))

    # 2nd order strictly more accurate than 1st order alone at every step
    for e1, e2 in zip(errors_1st, errors_2nd):
        assert e2 < e1

    # 2nd-order error should shrink faster than linearly as h -> 0 (O(h^3)
    # vs 1st-order's O(h^2)): halving h should shrink the 2nd-order error
    # by well more than a factor of 2.
    assert errors_2nd[0] / errors_2nd[1] > 3.0
    assert errors_2nd[1] / errors_2nd[2] > 3.0


def test_eva_unit_conversion():
    hr, recip = _abc_model_hr()
    result = kdotp_coefficients(hr, recip, K0, bands=(0, 1))
    fo_eva = to_eVA_units(result.first_order, "kdotp_first_order")
    so_eva = to_eVA_units(result.second_order, "kdotp_second_order")
    np.testing.assert_allclose(fo_eva, result.first_order * HARTREE_TO_EV * BOHR_TO_ANG)
    np.testing.assert_allclose(so_eva, result.second_order * HARTREE_TO_EV * BOHR_TO_ANG ** 2)
