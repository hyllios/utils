"""
Tests for waw.core.hamiltonian's `transl_inv_full` translational-invariance
correction (Wannier90 tutorial37, `compute_position_r`/`compute_bb_r`/
`compute_cc_r`'s `centres=`/`H_R=`/`BB_R=` arguments), transcribed from
`postw90/get_oper.F90`'s `get_AA_R`/`get_BB_R`/`get_CC_R`.

The decisive, unambiguous test here is AA(R)'s own operator identity: under
a uniform shift of the coordinate origin by delta (equivalently, every
Wannier centre in the system shifts by the SAME delta -- exactly bcc Fe's
tutorial37 scenario, a single atom per cell), <0n|r|Rm> -> <0n|(r+delta)|Rm>
= AA(R)(m,n) + delta * delta_{mn} * delta_{R,0} (Wannier-function
orthonormality, <0n|Rm> = delta_{mn}delta_{R0}) -- i.e. ONLY the R=0
diagonal should shift (by delta, matching the centre shift itself); every
off-diagonal or R!=0 element must stay EXACTLY the same. This holds
regardless of any BB/CC sign-convention subtlety, so it's used as the
primary correctness check.

A corresponding shift in the input overlap data is built from the known
gauge-transformation rule: U(k) -> exp(i*k.delta)*U(k) transforms
M_tilde(k,b,m,n) -> M_tilde(k,b,m,n)*exp(i*b.delta) (b-dependent only, NOT
m,n-dependent, since a uniform origin shift applies the SAME phase to
every Wannier function).
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import (
    compute_position_r, compute_bb_r, compute_cc_r, compute_hr, HamiltonianR,
    _negate_R_index, _wigner_seitz,
)
from waw.core.spread import compute_wannier_centres


def _synthetic_system(nk=8, nnb=6, nw=3, seed=0):
    """
    Small synthetic dataset: random M_tilde (Hermitian-consistent overlaps
    are not required for these tests -- only the ALGEBRAIC transformation
    properties of compute_position_r/compute_bb_r are being checked, not
    a physically-converged Wannier model), simple-cubic-like bvecs shared
    at every k (no per-k permutation needed for this test).
    """
    torch.manual_seed(seed)
    M_tilde = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
    M_tilde = M_tilde + M_tilde.conj().transpose(-1, -2)   # Hermitian-ish, harmless

    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])[:nnb]
    wb = torch.full((nnb,), 1.0 / (2 * np.pi ** 2), dtype=torch.float64)
    bvecs = torch.tensor(np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64)

    kpts = torch.rand(nk, 3, dtype=torch.float64)
    mp_grid = (2, 2, 2)
    real_lattice = 5.0 * np.eye(3)

    centres = torch.randn(nw, 3, dtype=torch.float64)
    return M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres


def test_negate_R_index_self_consistent():
    R_arr, _ = _wigner_seitz((2, 2, 2), 5.0 * np.eye(3))
    idx = _negate_R_index(R_arr)
    np.testing.assert_allclose(R_arr[idx], -R_arr, atol=1e-10)
    # involution: negating twice returns the original index
    np.testing.assert_array_equal(idx[idx], np.arange(len(R_arr)))


def test_aa_r_transl_inv_full_shifts_only_the_r0_diagonal():
    """
    `centres` must always be the ACTUAL Wannier centres of the overlap
    data it's paired with (`compute_wannier_centres(M_tilde, ...)`), never
    an independently-chosen value -- so the shifted dataset's centres are
    RECOMPUTED from the shifted M_tilde here, not manually offset. This
    also doubles as a check that the assumed gauge transformation
    (U(k) -> exp(i k.delta) U(k) <=> M_tilde(k,b) -> M_tilde(k,b)*exp(i
    b.delta)) really does shift the reported centres by exactly delta, as
    claimed -- PROVIDED `angle()`'s principal branch never wraps, which is
    why the diagonal phase and `delta` are both kept deliberately small
    below (a genuinely delocalized/runaway-centre M_tilde would need the
    same `guiding_centres`/branch-consistent machinery `_guided_phase`
    exists for -- irrelevant to what THIS correction is testing).
    """
    nk, nnb, nw = 8, 6, 3
    torch.manual_seed(0)
    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    wb = torch.full((nnb,), 1.0 / (2 * np.pi ** 2), dtype=torch.float64)
    bvecs = torch.tensor(np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64)
    kpts = torch.rand(nk, 3, dtype=torch.float64)
    mp_grid = (2, 2, 2)
    real_lattice = 5.0 * np.eye(3)

    # Well-localized-like synthetic M_tilde: SMALL diagonal phase (bounded
    # away from the +-pi branch cut) plus modest off-diagonal noise.
    small_phase = 0.05 * torch.randn(nk, nnb, nw, dtype=torch.float64)
    diag_mag = 0.8 + 0.1 * torch.rand(nk, nnb, nw, dtype=torch.float64)
    M_tilde = torch.diag_embed(diag_mag.to(torch.complex128) * torch.exp(1j * small_phase.to(torch.complex128)))
    off_diag_noise = 0.05 * (torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
                             + 1j * torch.randn(nk, nnb, nw, nw, dtype=torch.complex128))
    off_diag_noise = off_diag_noise - torch.diag_embed(torch.diagonal(off_diag_noise, dim1=-2, dim2=-1))
    M_tilde = M_tilde + off_diag_noise

    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    r0_idx = int(np.nonzero((R_arr == 0).all(axis=1))[0][0])

    centres = compute_wannier_centres(M_tilde, wb, bvecs)
    AA_R = compute_position_r(M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres=centres)

    delta = torch.tensor([0.02, -0.03, 0.01], dtype=torch.float64)
    # M_tilde(k,b) -> M_tilde(k,b) * exp(i b.delta) shifts centres by -delta
    # (not +delta): centres = -(1/Nk) sum_k,b wb*b*angle(M_diag), and
    # angle(M_diag) -> angle(M_diag) + b.delta shifts it by
    # -(1/Nk) sum_k,b wb*b_a*(b.delta) = -sum_c delta_c * sum_b wb*b_a*b_c
    # = -delta_a (Marzari-Vanderbilt completeness, sum_b wb*b_a*b_c = delta_ac)
    # -- confirmed numerically before fixing this assertion's sign.
    b_dot_delta = torch.einsum('kba,a->kb', bvecs, delta)
    phase = torch.polar(torch.ones_like(b_dot_delta), b_dot_delta).to(M_tilde.dtype)
    M_tilde_shifted = M_tilde * phase[:, :, None, None]

    centres_shifted = compute_wannier_centres(M_tilde_shifted, wb, bvecs)
    np.testing.assert_allclose(centres_shifted.numpy(), (centres - delta).numpy(), atol=1e-9)
    delta = -delta   # the ACTUAL centre shift this M_tilde transformation produces

    AA_R_shifted = compute_position_r(
        M_tilde_shifted, wb, bvecs, kpts, mp_grid, real_lattice, centres=centres_shifted,
    )

    diff = (AA_R_shifted - AA_R).detach().cpu().numpy()   # (3, nR, nw, nw)

    # Off-diagonal / R!=0: must be EXACTLY unchanged.
    mask = np.ones(diff.shape[1:], dtype=bool)   # (nR, nw, nw)
    nw = centres.shape[0]
    mask[r0_idx] = ~np.eye(nw, dtype=bool)
    np.testing.assert_allclose(diff[:, mask], 0.0, atol=1e-9)

    # R=0 diagonal: must shift by EXACTLY delta.
    diag_diff = np.stack([diff[a, r0_idx].diagonal() for a in range(3)], axis=-1)   # (nw, 3)
    np.testing.assert_allclose(diag_diff, np.tile(delta.numpy(), (nw, 1)), atol=1e-9)


def test_aa_r_default_path_unaffected_by_centres_none():
    """Sanity: omitting `centres` reproduces the pre-existing plain formula
    (no accidental behavior change for every other existing caller)."""
    M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, _ = _synthetic_system()
    AA_R_a = compute_position_r(M_tilde, wb, bvecs, kpts, mp_grid, real_lattice)
    AA_R_b = compute_position_r(M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres=None)
    torch.testing.assert_close(AA_R_a, AA_R_b)


def test_bb_r_requires_h_r_when_centres_given():
    M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres = _synthetic_system()
    with pytest.raises(ValueError, match="H_R"):
        compute_bb_r(M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres=centres)


def test_bb_r_transl_inv_full_is_finite_and_changes_the_result():
    """
    Sanity check (not a full independent derivation -- see module
    docstring): the correction actually does something (isn't a silent
    no-op) and produces a finite, well-shaped result. The decisive
    quantitative check is AA(R)'s own test above (unambiguous operator
    identity) plus the real-wannier90.x cross-validation in the tutorial37
    notebook itself.
    """
    nk, nnb, nw = 8, 6, 3
    torch.manual_seed(1)
    M_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres = _synthetic_system(nk, nnb, nw, seed=1)
    H_tilde = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)

    R_arr, degen = _wigner_seitz(mp_grid, real_lattice)
    H_R = torch.randn(len(R_arr), nw, nw, dtype=torch.complex128)
    H_R = H_R + H_R.conj().transpose(-1, -2)   # Hermitian-ish

    BB_R_plain = compute_bb_r(H_tilde, wb, bvecs, kpts, mp_grid, real_lattice)
    BB_R_corrected = compute_bb_r(H_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres=centres, H_R=H_R)

    assert torch.isfinite(BB_R_corrected.abs()).all()
    assert BB_R_corrected.shape == BB_R_plain.shape
    assert not torch.allclose(BB_R_corrected, BB_R_plain)


def test_cc_r_requires_bb_r_and_h_r_when_centres_given():
    nk, nnb, nw = 4, 6, 2
    torch.manual_seed(2)
    uHu = torch.randn(nk, nnb, nnb, nw, nw, dtype=torch.complex128)
    W = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(nk, -1, -1).contiguous()
    kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
    wb = torch.full((nnb,), 0.1, dtype=torch.float64)
    bvecs = torch.zeros(nk, nnb, 3, dtype=torch.float64)
    kpts = torch.rand(nk, 3, dtype=torch.float64)
    centres = torch.zeros(nw, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="BB_R and H_R"):
        compute_cc_r(uHu, W, kb_idx, wb, bvecs, kpts, (2, 2, 1), 5.0 * np.eye(3), centres=centres)


def test_cc_r_transl_inv_full_is_finite_and_changes_the_result():
    nk, nnb, nw = 4, 6, 2
    torch.manual_seed(3)
    uHu = torch.randn(nk, nnb, nnb, nw, nw, dtype=torch.complex128)
    W = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(nk, -1, -1).contiguous()
    kb_idx = torch.randint(0, nk, (nk, nnb))

    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    wb = torch.full((nnb,), 1.0 / (2 * np.pi ** 2), dtype=torch.float64)
    bvecs = torch.tensor(np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64)
    kpts = torch.rand(nk, 3, dtype=torch.float64)
    mp_grid = (2, 2, 1)
    real_lattice = 5.0 * np.eye(3)
    centres = torch.randn(nw, 3, dtype=torch.float64)

    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    H_R = torch.randn(len(R_arr), nw, nw, dtype=torch.complex128)
    H_R = H_R + H_R.conj().transpose(-1, -2)
    H_tilde = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
    BB_R = compute_bb_r(H_tilde, wb, bvecs, kpts, mp_grid, real_lattice, centres=centres, H_R=H_R)

    CC_R_plain = compute_cc_r(uHu, W, kb_idx, wb, bvecs, kpts, mp_grid, real_lattice)
    CC_R_corrected = compute_cc_r(
        uHu, W, kb_idx, wb, bvecs, kpts, mp_grid, real_lattice,
        centres=centres, BB_R=BB_R, H_R=H_R,
    )

    assert torch.isfinite(CC_R_corrected.abs()).all()
    assert CC_R_corrected.shape == CC_R_plain.shape
    assert not torch.allclose(CC_R_corrected, CC_R_plain)
