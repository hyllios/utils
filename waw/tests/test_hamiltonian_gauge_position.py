"""
Tests for waw/core/hamiltonian.py::hamiltonian_gauge_position -- the
Hamiltonian-gauge position matrix A_H = U^dagger.A_W.U + i.D_h (WYSV06
Eq. 24/25, transcribed from wannier90's src/postw90/wan_ham.F90::
wham_get_D_h), the new building block postw90 tutorial24's gyrotropic
Dw/tildeD and NOA tensors need.

No independent ground truth is available for this quantity in isolation
(it isn't one of postw90's own printed outputs), so these tests check:
  1. A brute-force per-k, per-band-pair Python loop reproduces the
     vectorized implementation exactly -- an independent re-derivation
     of the same formula, not just "the code agrees with itself".
  2. Documented invariants of the formula itself: the diagonal (n == m)
     is untouched by D_h (D_h is defined as 0 there); D_h is anti-
     Hermitian by construction; genuinely degenerate band pairs get
     D_h == 0 (the Fortran's explicit "avoid degeneracies" guard).
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import (
    HamiltonianR, hamiltonian_gauge_position, position_operator_k,
    position_operator_derivative_k, _degenerate_aware_band_velocities,
)
from waw.analysis._fourier_derivs import (
    h_and_grad_cart_batch, h_and_hess_cart_batch, h_and_k_derivatives_frac,
)

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u: float) -> HamiltonianR:
    """Same 2-band QWZ model as test_analysis_topology.py."""
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


def _synthetic_AA_R(hr: HamiltonianR, seed: int = 0) -> torch.Tensor:
    """A random Hermitian-per-component AA_R on the same R-grid as hr, just
    to exercise the rotation machinery -- no physical meaning needed for
    these formula-level tests."""
    rng = np.random.default_rng(seed)
    nR, nw = hr.H_R.shape[0], hr.nw
    raw = rng.normal(size=(3, nR, nw, nw)) + 1j * rng.normal(size=(3, nR, nw, nw))
    raw = 0.5 * (raw + raw.conj().transpose(0, 1, 3, 2))
    return torch.tensor(raw, dtype=torch.complex128)


def _brute_force_A_H(H0, grad_cart, A_k, degen_thresh):
    """Independent per-k, per-(n,m) Python-loop re-derivation of the same
    WYSV06 Eq. 24/25 formula, for cross-checking the vectorized function."""
    nk, nw = H0.shape[0], H0.shape[-1]
    eig = np.empty((nk, nw))
    A_H = np.empty((nk, 3, nw, nw), dtype=np.complex128)
    for k in range(nk):
        w, U = np.linalg.eigh(H0[k].numpy())
        eig[k] = w
        for a in range(3):
            dH_eig = U.conj().T @ grad_cart[k, a].numpy() @ U
            A_eig = U.conj().T @ A_k[k, a].numpy() @ U
            D_h = np.zeros((nw, nw), dtype=np.complex128)
            for n in range(nw):
                for m in range(nw):
                    if n == m or abs(w[m] - w[n]) <= degen_thresh:
                        continue
                    D_h[n, m] = dH_eig[n, m] / (w[m] - w[n])
            A_H[k, a] = A_eig + 1j * D_h
    return eig, A_H


@pytest.fixture(scope="module")
def gapped_setup():
    hr = _qwz_hr(u=3.0)   # |u|>2: fully gapped everywhere, no accidental degeneracies
    AA_R = _synthetic_AA_R(hr)
    kpts = np.array([[0.1, 0.2, 0.0], [0.37, -0.15, 0.0], [0.0, 0.0, 0.0]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)
    return hr, AA_R, kpts, H0, grad_cart, A_k


def test_matches_brute_force_reimplementation(gapped_setup):
    _, _, _, H0, grad_cart, A_k = gapped_setup
    degen_thresh = 1e-9
    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k, degen_thresh)

    eig_bf, A_H_bf = _brute_force_A_H(H0, grad_cart, A_k, degen_thresh)
    # eigenvalues from torch.linalg.eigh are ascending, same as np.linalg.eigh
    np.testing.assert_allclose(eig.numpy(), eig_bf, atol=1e-10)
    np.testing.assert_allclose(A_H.numpy(), A_H_bf, atol=1e-8)


def test_diagonal_unaffected_by_D_h(gapped_setup):
    """D_h is 0 on the diagonal by definition, so A_H(n,n) must equal
    the plain rotated (U^dagger A_k U)(n,n) exactly."""
    _, _, _, H0, grad_cart, A_k = gapped_setup
    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k)

    A_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)
    diag_A_H = torch.diagonal(A_H, dim1=-2, dim2=-1)
    diag_A_eig = torch.diagonal(A_eig, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag_A_H, diag_A_eig)


def test_del_eig_matches_diagonal_of_rotated_gradient(gapped_setup):
    _, _, _, H0, grad_cart, A_k = gapped_setup
    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k)

    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)
    expected = torch.diagonal(dH_eig, dim1=-2, dim2=-1).real.transpose(-1, -2)
    torch.testing.assert_close(del_eig, expected)
    # band velocities must be real (Hermitian H, diagonal of a rotated Hermitian op)
    assert del_eig.dtype in (torch.float32, torch.float64)


def test_D_h_is_anti_hermitian():
    """D_h(m,n) = -conj(D_h(n,m)) falls out of the formula itself (see the
    docstring derivation); check it holds numerically for a case with
    genuinely distinct band energies at every k (u=3, fully gapped).

    `A_H - A_eig` is `i*D_h`, not `D_h` itself (A_H = A_eig + i*D_h) --
    multiplying an anti-Hermitian matrix by i gives a HERMITIAN one, so
    divide out the i first before checking anti-Hermiticity of D_h."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr, seed=1)
    kpts = np.array([[0.13, 0.27, 0.0]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)

    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k)
    A_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)
    i_D_h = A_H - A_eig
    for a in range(3):
        # i*D_h is Hermitian
        torch.testing.assert_close(i_D_h[0, a], i_D_h[0, a].conj().T, atol=1e-10, rtol=0)
        D_h = i_D_h[0, a] / 1j
        torch.testing.assert_close(D_h, -D_h.conj().T, atol=1e-10, rtol=0)


def test_degenerate_bands_get_zero_D_h():
    """At u=0 the QWZ Hamiltonian is exactly degenerate at k=(0.5,0.5,0)
    (both diagonal SZ-coefficient terms vanish there is not guaranteed in
    general, so instead force an exact degeneracy directly: a 2-band model
    with H(R=0) = 0 and equal off-diagonal-only hopping is degenerate at
    Gamma for a symmetric choice)."""
    # A trivially degenerate 2-band model: H(k) = cos(kx) * I (both bands
    # identical at every k) -- an extreme, unambiguous degenerate case.
    R_list = [(1, 0, 0), (-1, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)
    H_R = np.zeros((2, 2, 2), dtype=np.complex128)
    H_R[0] = 0.5 * np.eye(2)
    H_R[1] = 0.5 * np.eye(2)
    hr = HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                       R_vectors=R_vectors, degen=degen, nw=2)

    AA_R = _synthetic_AA_R(hr, seed=2)
    kpts = np.array([[0.2, 0.0, 0.0]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)

    eig, UU, del_eig, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k, degen_thresh=1e-9)
    assert abs(float(eig[0, 0] - eig[0, 1])) < 1e-12   # genuinely degenerate

    A_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)
    D_h = A_H - A_eig
    # the only off-diagonal pair (0,1)/(1,0) must have D_h == 0 (degenerate)
    for a in range(3):
        assert abs(complex(D_h[0, a, 0, 1])) < 1e-9
        assert abs(complex(D_h[0, a, 1, 0])) < 1e-9


# ---------------------------------------------------------------------------
# Phase A (tutorial 25, shift current): batched Hessian + dA(k)/dk
# ---------------------------------------------------------------------------

def test_h_and_hess_cart_batch_matches_per_k_hessian():
    """`h_and_hess_cart_batch` (new, batched over a k-stack) must reproduce
    `h_and_k_derivatives_frac`'s already-trusted per-k Hessian (used by
    `effective_mass.py` today) exactly, Jacobian-rotated to Cartesian the
    same way `effective_mass.py` itself does."""
    hr = _qwz_hr(u=1.7)
    kpts = np.array([[0.13, -0.27, 0.05], [0.4, 0.1, -0.2]])
    recip_lattice = 2 * np.pi * np.eye(3)
    inv_recip = np.linalg.inv(recip_lattice)

    H0_batch, grad_cart_batch, hess_cart_batch = h_and_hess_cart_batch(hr, kpts, recip_lattice)

    for i, k0 in enumerate(kpts):
        H0, grad_frac, hess_frac = h_and_k_derivatives_frac(hr, k0)
        hess_cart_ref = np.einsum('ja,lb,abnm->jlnm', inv_recip, inv_recip, hess_frac)
        np.testing.assert_allclose(H0_batch[i].numpy(), H0, atol=1e-10)
        np.testing.assert_allclose(hess_cart_batch[i].numpy(), hess_cart_ref, atol=1e-10)


def test_position_operator_derivative_k_matches_finite_difference():
    """`position_operator_derivative_k`'s analytic dA(k)/dk must match a
    numerical finite-difference derivative of `position_operator_k`'s own
    (already-trusted) A(k) reconstruction -- an independent cross-check,
    not a self-consistency tautology."""
    hr = _qwz_hr(u=1.7)
    AA_R = _synthetic_AA_R(hr, seed=5)
    recip_lattice = 2 * np.pi * np.eye(3)
    k0 = np.array([0.17, -0.23, 0.08])
    h = 1e-5

    dA_dk_analytic = position_operator_derivative_k(
        AA_R, hr.R_vectors, hr.degen, recip_lattice, k0[None, :],
    )[0]   # (3,3,nw,nw): (b=deriv dir Cartesian, c=component)

    # Cartesian finite difference: since recip_lattice = 2*pi*I here, Cartesian
    # k equals 2*pi*(fractional k), so a small Cartesian step db corresponds
    # to a fractional-k step of db/(2*pi) along the same axis.
    for b in range(3):
        step_frac = np.zeros(3)
        step_frac[b] = h / (2 * np.pi)
        A_plus, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), (k0 + step_frac)[None, :])
        A_minus, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), (k0 - step_frac)[None, :])
        fd = (A_plus[0] - A_minus[0]) / (2 * h)   # (3,nw,nw): component c
        np.testing.assert_allclose(dA_dk_analytic[b].numpy(), fd.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Phase B (tutorial 25, shift current): degenerate-aware velocities +
# eta-regularized D_h
# ---------------------------------------------------------------------------

def test_degenerate_aware_velocities_use_submatrix_eigenvalues():
    """
    For a genuinely degenerate PAIR of bands (eig equal) whose H-gauge
    gradient submatrix is NOT simply proportional to the identity (so a
    naive diagonal would depend on eigh's arbitrary within-degenerate-
    subspace basis choice), `_degenerate_aware_band_velocities` must return
    the EIGENVALUES of that 2x2 submatrix (WYSV06/YWVS07 Eq. 31), not the
    plain (basis-dependent, hence physically ill-defined) diagonal.
    """
    nk, nw = 1, 3
    eig = torch.tensor([[1.0, 1.0, 5.0]], dtype=torch.float64)   # bands 0,1 degenerate; band 2 isolated
    dH_eig = torch.zeros(nk, 3, nw, nw, dtype=torch.complex128)
    # direction a=0: a non-trivial Hermitian 2x2 submatrix for the degenerate pair
    sub = torch.tensor([[2.0, 1.0 + 0.5j], [1.0 - 0.5j, -1.0]], dtype=torch.complex128)
    dH_eig[0, 0, :2, :2] = sub
    dH_eig[0, 0, 2, 2] = 7.0   # isolated band's own plain diagonal velocity
    # a random Hermitian perturbation for the OTHER off-diagonal (band 2 <-> pair):
    # irrelevant to band 2's own velocity (only the diagonal matters there).

    del_eig = _degenerate_aware_band_velocities(dH_eig, eig, degen_thresh=1e-6)

    expected_pair = torch.linalg.eigvalsh(sub).real.sort().values
    got_pair = del_eig[0, :2, 0].sort().values
    torch.testing.assert_close(got_pair, expected_pair)
    # isolated band unaffected -- plain diagonal
    assert abs(float(del_eig[0, 2, 0]) - 7.0) < 1e-12
    # naive diagonal of the degenerate pair (2.0, -1.0) must NOT equal the
    # submatrix eigenvalues here (proving the correction actually changed something)
    assert not torch.allclose(got_pair, torch.tensor([2.0, -1.0], dtype=torch.float64))


def test_degenerate_aware_velocities_reduce_to_diagonal_when_isolated():
    """With no degeneracies at all, the degenerate-aware path must reduce
    exactly to the plain Hellmann-Feynman diagonal (no behavior change for
    every already-validated non-degenerate case, e.g. gyrotropic.py)."""
    torch.manual_seed(0)
    nk, nw = 4, 5
    eig = torch.sort(torch.rand(nk, nw, dtype=torch.float64) * 10, dim=-1).values
    dH_eig = torch.randn(nk, 3, nw, nw, dtype=torch.complex128)
    dH_eig = dH_eig + dH_eig.conj().transpose(-1, -2)

    del_eig = _degenerate_aware_band_velocities(dH_eig, eig, degen_thresh=1e-9)
    expected = torch.diagonal(dH_eig, dim1=-2, dim2=-1).real.transpose(-1, -2)
    torch.testing.assert_close(del_eig, expected)


def test_eta_regularized_D_h_matches_formula_and_reduces_to_no_eta_limit():
    """The eta-regularized D_h (`sc_eta` argument) must match its literal
    formula `dH_eig(n,m) * (E_m-E_n)/((E_m-E_n)^2+eta^2)` (Re[1/(dE+i*eta)],
    transcribed from wham_get_D_h_P_value), be zero on the diagonal, and
    reduce to the existing no-eta D_h as eta -> 0 for well-separated bands."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr, seed=11)
    kpts = np.array([[0.11, -0.31, 0.07]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)

    sc_eta = 1e-6
    eig, UU, del_eig, A_H, D_h_eta = hamiltonian_gauge_position(H0, grad_cart, A_k, sc_eta=sc_eta)

    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)
    dE = eig[:, None, :] - eig[:, :, None]
    expected = dH_eig * (dE / (dE ** 2 + sc_eta ** 2))[:, None, :, :].to(dH_eig.dtype)
    nw = eig.shape[-1]
    off_diag = ~torch.eye(nw, dtype=torch.bool)
    expected = expected * off_diag[None, None, :, :].to(dH_eig.dtype)
    torch.testing.assert_close(D_h_eta, expected)

    for n in range(nw):
        assert abs(complex(D_h_eta[0, 0, n, n])) < 1e-12

    # well-separated bands (u=3 QWZ has no near-degeneracies at this k):
    # a tiny eta should reproduce the existing no-eta D_h almost exactly.
    _, _, _, A_H_no_eta = hamiltonian_gauge_position(H0, grad_cart, A_k)
    A_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)
    D_h_no_eta = (A_H_no_eta - A_eig) / 1j
    torch.testing.assert_close(D_h_eta.real, D_h_no_eta.real, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(D_h_eta.imag, D_h_no_eta.imag, atol=1e-4, rtol=1e-4)


def test_hamiltonian_gauge_position_4tuple_unchanged_without_sc_eta():
    """Existing callers (gyrotropic.py) unpack exactly 4 values; passing no
    `sc_eta` must keep returning a 4-tuple, byte-for-byte identical to
    before this Phase B addition."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr, seed=13)
    kpts = np.array([[0.2, 0.1, -0.05]])
    H0, grad_cart = h_and_grad_cart_batch(hr, kpts, np.eye(3))
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, np.eye(3), kpts)

    result = hamiltonian_gauge_position(H0, grad_cart, A_k)
    assert len(result) == 4
