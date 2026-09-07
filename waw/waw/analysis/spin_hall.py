"""
Spin Hall conductivity (SHC) and spin Nernst conductivity (SNC),
postw90 `berry_task=eval_shc`. Two SHC methods are implemented, selected
via `method='qiao'|'ryoo'` throughout this module:

  - `shc_method=qiao` (Wannier90 tutorials 29/30, Platinum). Qiao, Zhou,
    Yuan & Zhao, PRB 98, 214402 (2018) ("QZYZ18") -- an APPROXIMATION
    reconstructing the needed spin-decorated overlaps from the already-
    available `.mmn`+`.spn` (no new ab-initio quantities).
  - `shc_method=ryoo` (Ryoo, Park & Souza, PRB 99, 235113 (2019) "RPS19")
    -- uses genuine ab-initio `.sIu`/`.sHu` matrix elements one
    neighbour-shell away (`build_shc_ryoo_operators`,
    `interfaces.wannier90.io.read_sIu`/`read_sHu`, pw2wannier90's own
    `write_sIu`/`write_sHu`), avoiding Qiao's completeness-insertion
    approximation. Expected to agree closely but not exactly with Qiao on
    a real material (different approximations of the same physical
    quantity) -- see e.g. the WannierBerri spin Hall tutorial's own
    comparison on bcc Pt.

`spin_nernst_conductivity` is the thermoelectric companion (Xiao-Yao-Niu
2006 Mott relation applied to the energy-resolved SHC), reusing
`topology._nernst_mott_integral` verbatim -- see that function's own
docstring, it is agnostic to what "sigma(E)" physically is.

SHC is structurally AHC with one velocity-operator leg replaced by a
spin-current operator: for a fixed (alpha, beta, gamma) Cartesian/spin
triple (`shc_alpha`/`shc_beta`/`shc_gamma`, 0-based here vs. wannier90's
1-based x/y/z),

    sigma_shc(E_F) = fac * (1/Nk) sum_k sum_{n: E_n<E_F} sum_{m!=n}
                     -2 Im[ js_k(n,m) * i*(E_m-E_n) * A_H(m,n,beta) ]
                     / ((E_m-E_n)^2 + eta^2)

with `A_H` the degenerate-corrected Hamiltonian-gauge position matrix
(`core.hamiltonian.hamiltonian_gauge_position`, WYSV06 Eq. 24/25), and
`js_k(n,m)` the Qiao spin-current matrix element (QZYZ18 Eq. 23-27),
built per k from four H(k)-eigenbasis matrices:

    S_k  = <u|sigma_gamma|u>                        (rotated SS_R)
    K_k  = SR_alpha_k + S_k @ D_h(alpha)             (rotated SR_R + D_h)
    L_k  = SHR_alpha_k + SH_k @ D_h(alpha)           (rotated SHR_R + D_h)
    B_k[n,m] = del_eig(alpha)[m]*S_k[n,m] + eig[m]*K_k[n,m] - L_k[n,m]
    js_k = (B_k + B_k^dagger) / 2

`D_h(alpha)` is the same degenerate-perturbation matrix used internally
by `hamiltonian_gauge_position`: `D_h = -i*(A_H - AA_bar)`, `AA_bar`
being the plain (un-degenerate-corrected) rotation of the position
operator. `SR_alpha_k`/`SHR_alpha_k` carry an extra `-i` factor that
cancels the `+i` wannier90's `get_SHC_R` bakes into `SR_R`/`SHR_R` at
construction time.

Adaptive smearing (`kubo_adpt_smr`, on by default) reuses
`shift_current._kmesh_spacing`/`_adaptive_eta` (YWVS07).

New R-space operators (`build_shc_operators`): `SS_R`, `SR_R`, `SHR_R`,
`SH_R`, built from `.spn` + `.mmn` -- no new ab-initio file types.
`SS_R`/`SH_R` use the plain per-k operator pattern
(`spin_texture.spin_operator_r`/`core.hamiltonian.compute_operator_r`);
`SR_R`/`SHR_R` use the finite-difference b-vector pattern
(`core.hamiltonian.compute_bb_r`) fed a spin- (or spin-times-Hamiltonian-)
weighted overlap via `core.spread.weight_overlaps_by_operator` (a
matrix-product generalization of `weight_overlaps_by_eigenvalues`, needed
because the spin operator mixes bands). A single `rotate_overlaps(W,
...)` call with the full converged gauge `W = V@U_final` reproduces the
two-step V-then-U_final composition exactly, so no separate bookkeeping
is needed.

Units: `sigma^{alpha,beta}_{gamma}` in (hbar/e)*S/cm (postw90's own
unit), `fac = 1e8*e^2/(hbar*V)/2` -- note the extra `/2` and missing
minus sign relative to `topology.anomalous_hall_conductivity`'s
`fac = -1e8*e^2/(hbar*V)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..core.hamiltonian import (
    HamiltonianR, hamiltonian_gauge_position, position_operator_k,
    compute_bb_r, compute_operator_r, operator_k,
)
from ..core.spread import rotate_overlaps, weight_overlaps_by_operator
from ..units import (
    BOHR_TO_ANG, EV_TO_HARTREE, E_CHARGE, HBAR_SI, HARTREE_TO_EV, K_B_HARTREE,
    register_si_unit, register_from_si_unit,
)
from ._fourier_derivs import h_and_grad_cart_batch, KCHUNK
from .spin_texture import spin_operator_r
from .shift_current import (
    _kmesh_spacing, _adaptive_eta,
    KUBO_ADPT_SMR_FAC_DEFAULT, KUBO_ADPT_SMR_MAX_DEFAULT,
)
from .topology import _nernst_mott_integral


@dataclass
class SHCResult:
    """
    Spin Hall conductivity vs. Fermi energy, atomic units (Bohr^2, bare --
    no cell-volume normalization or e^2/hbar prefactor applied). Convert
    with ``waw.units.to_si_units(result.sigma, "spin_hall_conductivity",
    cell_volume_bohr3=...)`` for (hbar/e)*S/cm.
    """
    fermi_energies: np.ndarray   # (nf,) Hartree
    sigma:          np.ndarray   # (nf,) Bohr^2 (raw, atomic)
    alpha:          int
    beta:           int
    gamma:          int
    mesh:           tuple


@dataclass
class SHCACResult:
    """
    AC (frequency-dependent) spin Hall conductivity at one fixed Fermi
    energy, atomic units (complex Bohr^2, bare). Convert with
    ``waw.units.to_si_units(result.sigma, "spin_hall_conductivity",
    cell_volume_bohr3=...)`` for complex (hbar/e)*S/cm (matching postw90's
    own `<seed>-shc-freqscan.dat` Re/Im columns).
    """
    fermi_energy: float          # Hartree
    omega:        np.ndarray     # (nfreq,) Hartree
    sigma:        np.ndarray     # (nfreq,) complex Bohr^2 (raw, atomic)
    alpha:        int
    beta:         int
    gamma:        int
    mesh:         tuple


def build_shc_operators(
    W:            Tensor,
    Mmn:          Tensor,
    kb_idx:       Tensor,
    spn_bloch:    np.ndarray,
    eig_bloch:    np.ndarray,
    wb:           Tensor,
    bvecs:        Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build the four Wannier-gauge R-space operators the spin Hall
    conductivity (Qiao method) needs, all from `.spn` + `.mmn`.
    wannier90: `get_oper.F90::get_SHC_R`.

    Args:
      W        : (nk, nb, nw) complex, FULL converged gauge (V@U_final for
                 disentangled bands, U_final alone for isolated bands)
      Mmn      : (nk, nnb, nb, nb) complex, RAW overlaps (pre-rotation)
      kb_idx   : (nk, nnb) long, neighbour-index table
      spn_bloch: (nk, nb, nb, 3) complex, `.spn` Pauli matrices in the
                 ab-initio band basis (`read_spn`'s own array)
      eig_bloch: (nk, nb) real, Hartree, ab-initio eigenvalues, SAME band
                 window/ordering as Mmn/spn_bloch's bra/ket index
      wb, bvecs, kpts, mp_grid, real_lattice : as `compute_position_r`

    Returns (SS_R, SR_R, SHR_R, SH_R):
      SS_R : (nR, nw, nw, 3) complex          -- <0n|sigma_c|Rm>, axis c LAST
      SR_R : (3, 3, nR, nw, nw) complex        -- <0n|sigma_c.(r-R)_a|Rm>, axes (c, a) FIRST
      SHR_R: (3, 3, nR, nw, nw) complex        -- <0n|sigma_c.H.(r-R)_a|Rm>, axes (c, a) FIRST
      SH_R : (nR, nw, nw, 3) complex           -- <0n|sigma_c.H|Rm>, axis c LAST
    """
    SS_R = spin_operator_r(W, spn_bloch, kpts, mp_grid, real_lattice)   # (nR, nw, nw, 3)

    spn_t = torch.as_tensor(spn_bloch, dtype=W.dtype, device=W.device)
    eig_t = torch.as_tensor(eig_bloch, dtype=torch.float64, device=W.device)

    SR_list, SHR_list, SH_list = [], [], []
    for c in range(3):
        S_c = spn_t[..., c]                                   # (nk, nb, nb)
        SH_c = S_c * eig_t[:, None, :].to(S_c.dtype)           # (sigma_c @ H): weight the KET index

        SR_tilde = rotate_overlaps(W, weight_overlaps_by_operator(Mmn, S_c), kb_idx)
        SHR_tilde = rotate_overlaps(W, weight_overlaps_by_operator(Mmn, SH_c), kb_idx)

        SR_list.append(compute_bb_r(SR_tilde, wb, bvecs, kpts, mp_grid, real_lattice))    # (3, nR, nw, nw)
        SHR_list.append(compute_bb_r(SHR_tilde, wb, bvecs, kpts, mp_grid, real_lattice))
        SH_list.append(compute_operator_r(W, SH_c, kpts, mp_grid, real_lattice))          # (nR, nw, nw)

    SR_R = torch.stack(SR_list, dim=0)      # (3(c), 3(a), nR, nw, nw)
    SHR_R = torch.stack(SHR_list, dim=0)    # (3(c), 3(a), nR, nw, nw)
    SH_R = torch.stack(SH_list, dim=-1)     # (nR, nw, nw, 3(c))

    return SS_R, SR_R, SHR_R, SH_R


def build_shc_ryoo_operators(
    W:            Tensor,
    sIu:          Tensor,
    sHu:          Tensor,
    kb_idx:       Tensor,
    wb:           Tensor,
    bvecs:        Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> tuple[Tensor, Tensor]:
    """
    Build the two Wannier-gauge R-space operators the spin Hall
    conductivity Ryoo method (Ryoo, Park & Souza, PRB 99, 235113 (2019))
    needs, from the genuine ab-initio `.sIu`/`.sHu` matrix elements
    (`interfaces.wannier90.io.read_sIu`/`read_sHu`) -- NOT an approximation
    reconstructed from `.mmn`+`.spn` (that's what the Qiao method above
    already is; Ryoo avoids that approximation by using real ab-initio
    data one neighbour-shell away). wannier90 develop branch:
    `get_oper.F90::get_SAA_R`/`get_SBB_R`.

    Unlike Qiao's `SR_R`/`SHR_R` (which need an extra
    `weight_overlaps_by_operator` step to insert a spin operator into a
    plain `Mmn`), `sIu`/`sHu` are ALREADY the full spin-sandwiched (and,
    for sHu, spin+Hamiltonian-sandwiched) ab-initio overlaps -- so building
    SAA_R/SBB_R is structurally IDENTICAL to how `compute_position_r`/
    `compute_bb_r` build AA_R/BB_R from a plain `Mmn`: rotate with the SAME
    two-k (bra at k, ket at k+b) `W`-gauge rotation, then the standard
    b-vector-weighted Wigner-Seitz transform. No new low-level R-space
    machinery needed, just feeding `compute_bb_r` a spin-decorated input.

    Args:
      W        : (nk, nb, nw) complex, FULL converged gauge (V@U_final)
      sIu, sHu : (nk, nnb, nb, nb, 3) complex, `read_sIu(...)["data"]`/
                 `read_sHu(...)["data"]`, SAME band window/ordering as
                 `Mmn`'s bra/ket index. `sHu` must already be in Hartree
                 (the raw file is in eV -- convert with `EV_TO_HARTREE`
                 before calling, matching how `.uHu` is handled elsewhere).
      kb_idx, wb, bvecs, kpts, mp_grid, real_lattice : as `compute_bb_r`

    Returns (SAA_R, SBB_R):
      SAA_R: (3(gamma), 3(alpha), nR, nw, nw) complex -- <0n|sigma_gamma.(r-R)_alpha|Rm>
      SBB_R: (3(gamma), 3(alpha), nR, nw, nw) complex -- <0n|sigma_gamma.H.(r-R)_alpha|Rm>
    """
    # SAA_R is exactly `spin_texture.spin_position_r`'s operator (same sIu,
    # same two-k gauge rotation, same non-Hermitized b-weighted transform) --
    # shared with `analysis.spin_accumulation`'s s^i_na rather than duplicated.
    from .spin_texture import spin_position_r
    SAA_R = spin_position_r(W, sIu, kb_idx, wb, bvecs, kpts, mp_grid, real_lattice)

    SBB_R = torch.stack([
        compute_bb_r(rotate_overlaps(W, sHu[..., c], kb_idx),
                     wb, bvecs, kpts, mp_grid, real_lattice)
        for c in range(3)
    ], dim=0)   # (3(gamma), 3(alpha), nR, nw, nw)
    return SAA_R, SBB_R


def _js_k_batch_ryoo(
    eig: Tensor, dH_eig_alpha: Tensor, UU: Tensor,
    S_k_w: Tensor, SAA_k_w: Tensor, SBB_k_w: Tensor,
) -> Tensor:
    """
    Ryoo's per-k spin-current matrix js_k(n,m), in the H(k) eigenbasis, for
    a fixed (alpha, gamma) direction pair. wannier90 develop branch:
    `berry.F90::berry_get_js_k`'s Ryoo (else) branch, RPS19 Eq. (21)/(26):

        VV0(n,m)       = (U^dagger . dH/dk_alpha . U)(n,m)    -- full matrix,
                          NOT just the diagonal velocity `del_eig` Qiao uses
        S_k(n,m)       = <u_n|sigma_gamma|u_m>                 (rotated SS_R)
        spinVel0       = VV0 @ S_k + S_k @ VV0                 (matrix product,
                          the symmetrized {v_alpha, sigma_gamma} anticommutator)
        js_k(n,m) = 0.5 * [ spinVel0(n,m)
                            - i*(eig(m)*SAA(n,m) - SBB(n,m))
                            + i*(eig(n)*conj(SAA(m,n)) - conj(SBB(m,n))) ]

    Args:
      eig         : (nk, nw) real, Hartree (H(k) eigenvalues)
      dH_eig_alpha: (nk, nw, nw) complex, Hartree*Bohr, the FULL
                    H(k)-eigenbasis-rotated Cartesian gradient (alpha
                    component) -- `einsum('kni,knm,kmj->kij', UU.conj(),
                    grad_cart[:,alpha], UU)`, not yet exposed by
                    `hamiltonian_gauge_position` (which only returns the
                    diagonal `del_eig` and the degenerate-corrected `A_H`)
      UU          : (nk, nw, nw) complex, H0's eigenvectors
      S_k_w       : (nk, nw, nw) complex, Wannier-gauge SS_R[...,gamma]
                    interpolated to k (NOT yet rotated)
      SAA_k_w, SBB_k_w: (nk, nw, nw) complex, Wannier-gauge
                    SAA_R[gamma,alpha]/SBB_R[gamma,alpha] interpolated to k
                    (NOT yet rotated)

    Returns js_k: (nk, nw, nw) complex, H(k) eigenbasis, axes (n, m).
    """
    def rot(O):
        return torch.einsum('kni,knm,kmj->kij', UU.conj(), O.to(UU.dtype), UU)

    S_k = rot(S_k_w)
    SAA_k = rot(SAA_k_w)
    SBB_k = rot(SBB_k_w)

    spinVel0 = torch.matmul(dH_eig_alpha, S_k) + torch.matmul(S_k, dH_eig_alpha)

    eig_col = eig[:, None, :].to(S_k.dtype)   # eig(m), broadcast over columns
    eig_row = eig[:, :, None].to(S_k.dtype)   # eig(n), broadcast over rows

    term_nm = -1j * (eig_col * SAA_k - SBB_k)
    SAA_mn = SAA_k.conj().transpose(-1, -2)   # [n,m] entry = conj(SAA(m,n))
    SBB_mn = SBB_k.conj().transpose(-1, -2)
    term_mn = 1j * (eig_row * SAA_mn - SBB_mn)

    return 0.5 * (spinVel0 + term_nm + term_mn)


def _js_k_batch(
    eig: Tensor, del_eig_alpha: Tensor, UU: Tensor, D_h_alpha: Tensor,
    S_k_w: Tensor, SR_k_w: Tensor, SHR_k_w: Tensor, SH_k_w: Tensor,
) -> Tensor:
    """
    Qiao's per-k spin-current matrix js_k(n,m), in the H(k) eigenbasis,
    for a fixed (alpha, gamma) direction pair (gamma already selected by
    the caller when interpolating `S_k_w`/`SR_k_w`/`SHR_k_w`/`SH_k_w`).
    wannier90: `berry.F90::berry_get_js_k` (Qiao branch).

    Args:
      eig          : (nk, nw) real, Hartree (H(k) eigenvalues)
      del_eig_alpha: (nk, nw) real, Hartree*Bohr, band velocity in the
                     alpha direction (`hamiltonian_gauge_position`'s
                     `del_eig[:, :, alpha]`)
      UU           : (nk, nw, nw) complex, H0's eigenvectors
      D_h_alpha    : (nk, nw, nw) complex, Bohr, the alpha-component of
                     the (no-eta) degenerate-perturbation matrix
      S_k_w, SH_k_w  : (nk, nw, nw) complex, Wannier-gauge SS_R[...,gamma]/
                       SH_R[...,gamma] interpolated to k (NOT yet rotated)
      SR_k_w, SHR_k_w: (nk, nw, nw) complex, Wannier-gauge SR_R[gamma,alpha]/
                       SHR_R[gamma,alpha] interpolated to k (NOT yet rotated)

    Returns js_k: (nk, nw, nw) complex, H(k) eigenbasis, axes (n, m).
    """
    def rot(O):
        return torch.einsum('kni,knm,kmj->kij', UU.conj(), O.to(UU.dtype), UU)

    S_k = rot(S_k_w)
    SH_k = rot(SH_k_w)
    # extra -i cancels the +i wannier90's get_SHC_R bakes into SR_R/SHR_R
    SR_alpha_k = -1j * rot(SR_k_w)
    SHR_alpha_k = -1j * rot(SHR_k_w)

    K_k = SR_alpha_k + torch.matmul(S_k, D_h_alpha)
    L_k = SHR_alpha_k + torch.matmul(SH_k, D_h_alpha)

    # weighted by the column (m) index, matching berry_get_js_k's broadcast
    B_k = (del_eig_alpha[:, None, :].to(S_k.dtype) * S_k
           + eig[:, None, :].to(S_k.dtype) * K_k
           - L_k)
    return 0.5 * (B_k + B_k.conj().transpose(-1, -2))


def _shc_per_k_ingredients(
    hr: HamiltonianR, AA_R: Tensor, SS_R: Tensor,
    SR_R: Tensor | None, SHR_R: Tensor | None, SH_R: Tensor | None,
    recip_lattice: np.ndarray, real_lattice: np.ndarray, kc: np.ndarray,
    alpha: int, beta: int, gamma: int, degen_thresh_ha: float,
    kubo_eigval_max_ha: float | None,
    method: str = "qiao",
    SAA_R: Tensor | None = None, SBB_R: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Shared per-k-chunk setup for `spin_hall_conductivity` (Fermi-energy
    scan) and `spin_hall_conductivity_ac` (frequency scan): H(k)
    eigenbasis, band velocities, `js_k`, and the `prod`/mask common to
    both accumulation formulas. `method='qiao'` (default, needs SR_R/
    SHR_R/SH_R) or `'ryoo'` (needs SAA_R/SBB_R instead) selects which
    per-k spin-current construction feeds the SAME downstream dE/prod/
    mask accumulation -- everything past `js_k` is method-independent.

    Returns (eig, del_eig, dE, prod, mask):
      eig    : (nk, nw) real, Hartree
      del_eig: (nk, nw, 3) real, Hartree*Bohr
      dE     : (nk, nw, nw) real, Hartree -- eig(m)-eig(n), axes (n, m)
      prod   : (nk, nw, nw) complex, Hartree^2*Bohr^2 -- js_k(n,m)*i*dE*AA(m,n,beta)
      mask   : (nk, nw, nw) bool -- n!=m, and (if requested) both bands <= kubo_eigval_max
    """
    H0, grad_cart = h_and_grad_cart_batch(hr, kc, recip_lattice)
    A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)
    eig, UU, del_eig, A_H = hamiltonian_gauge_position(
        H0, grad_cart, A_k, degen_thresh=degen_thresh_ha,
    )

    AA_bar = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)
    D_h_alpha = (-1j * (A_H - AA_bar))[:, alpha]   # (nk, nw, nw), Bohr

    SS_gamma = SS_R[..., gamma]
    S_k_w = operator_k(SS_gamma, hr.R_vectors, hr.degen, kc)

    if method == "qiao":
        SH_gamma = SH_R[..., gamma]
        SR_ga = SR_R[gamma, alpha]
        SHR_ga = SHR_R[gamma, alpha]

        SH_k_w = operator_k(SH_gamma, hr.R_vectors, hr.degen, kc)
        SR_k_w = operator_k(SR_ga, hr.R_vectors, hr.degen, kc)
        SHR_k_w = operator_k(SHR_ga, hr.R_vectors, hr.degen, kc)

        js_k = _js_k_batch(eig, del_eig[:, :, alpha], UU, D_h_alpha,
                            S_k_w, SR_k_w, SHR_k_w, SH_k_w)
    elif method == "ryoo":
        SAA_ga = SAA_R[gamma, alpha]
        SBB_ga = SBB_R[gamma, alpha]
        SAA_k_w = operator_k(SAA_ga, hr.R_vectors, hr.degen, kc)
        SBB_k_w = operator_k(SBB_ga, hr.R_vectors, hr.degen, kc)

        dH_eig_alpha = torch.einsum('kni,knm,kmj->kij', UU.conj(), grad_cart[:, alpha], UU)
        js_k = _js_k_batch_ryoo(eig, dH_eig_alpha, UU, S_k_w, SAA_k_w, SBB_k_w)
    else:
        raise ValueError(f"method must be 'qiao' or 'ryoo', got {method!r}")

    dE = eig[:, None, :] - eig[:, :, None]                    # (nk,nw,nw): eig(m)-eig(n), axes (n,m)
    AA_beta_nm = A_H[:, beta].transpose(-1, -2)               # AA_beta_nm[n,m] = A_H[m,n] = Fortran's AA(m,n,beta)
    prod = js_k * 1j * dE.to(js_k.dtype) * AA_beta_nm

    nw = eig.shape[-1]
    eye = torch.eye(nw, dtype=torch.bool, device=eig.device)
    mask = ~eye[None, :, :]
    if kubo_eigval_max_ha is not None:
        below = eig <= kubo_eigval_max_ha
        mask = mask & below[:, :, None] & below[:, None, :]

    return eig, del_eig, dE, prod, mask


def spin_berry_curvature_kpath(
    hr:            HamiltonianR,
    AA_R:          Tensor,
    SS_R:          Tensor,
    SR_R:          Tensor | None = None,
    SHR_R:         Tensor | None = None,
    SH_R:          Tensor | None = None,
    recip_lattice: np.ndarray = None,
    real_lattice:  np.ndarray = None,
    kpath:         np.ndarray = None,
    fermi_energy:  float = None,
    alpha:         int = 0,
    beta:          int = 1,
    gamma:         int = 2,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    kubo_eigval_max: float | None = None,
    eta:           float = 0.04 * EV_TO_HARTREE,
    method:        str = "qiao",
    SAA_R:         Tensor | None = None,
    SBB_R:         Tensor | None = None,
) -> np.ndarray:
    """
    Occupied-band-summed spin Berry curvature along an explicit k-path
    (postw90 `kpath_task = shc`), i.e. the SAME per-k integrand
    `spin_hall_conductivity` sums over a uniform mesh, evaluated instead at
    a hand-picked list of k-points and returned WITHOUT the final k-average
    -- useful for visualizing where in the BZ the SHC comes from (e.g.
    comparing how "smooth" the Qiao vs Ryoo integrands are along the same
    path, WannierBerri's own spin-Hall tutorial's Fig. comparing the two
    methods this way).

    Fixed smearing only (`kubo_adpt_smr` doesn't apply -- `_kmesh_spacing`
    needs a uniform mesh, not meaningful for an arbitrary path).

    Args mirror `spin_hall_conductivity`, except:
      kpath        : (nk, 3) fractional k-points (NOT a uniform mesh)
      fermi_energy : scalar, Hartree (ONE fixed Fermi energy)

    Returns curvature: (nk,) real, Bohr^2 (bare atomic units, same
    convention as `SHCResult.sigma` before `to_si_units`).
    """
    kpath = np.asarray(kpath, dtype=np.float64)

    eig, del_eig, dE, prod, mask = _shc_per_k_ingredients(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice, kpath,
        alpha, beta, gamma, degen_thresh, kubo_eigval_max,
        method=method, SAA_R=SAA_R, SBB_R=SBB_R,
    )

    denom = dE ** 2 + eta ** 2
    weight = torch.where(mask, -2.0 / denom, torch.zeros_like(denom))
    omega_per_nm = weight * prod.imag                          # (nk, nw, nw)
    omega_per_n = omega_per_nm.sum(dim=-1)                      # sum over m -> (nk, nw)

    occ = (eig < fermi_energy).to(omega_per_n.dtype)
    curvature = (omega_per_n * occ).sum(dim=-1)                 # (nk,), Bohr^2

    return curvature.cpu().numpy()


def spin_hall_conductivity(
    hr:            HamiltonianR,
    AA_R:          Tensor,
    SS_R:          Tensor,
    SR_R:          Tensor | None = None,
    SHR_R:         Tensor | None = None,
    SH_R:          Tensor | None = None,
    recip_lattice: np.ndarray = None,
    real_lattice:  np.ndarray = None,
    fermi_energies = None,
    mesh:          tuple = (20, 20, 20),
    alpha:         int = 0,
    beta:          int = 1,
    gamma:         int = 2,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    kubo_eigval_max: float | None = None,
    kubo_adpt_smr: bool = True,
    eta:           float = 0.04 * EV_TO_HARTREE,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
    method:        str = "qiao",
    SAA_R:         Tensor | None = None,
    SBB_R:         Tensor | None = None,
) -> SHCResult:
    """
    Spin Hall conductivity vs. Fermi energy (postw90 `berry_task=eval_shc`).
    wannier90: `berry_get_shc_klist`'s Fermi-scan branch. `kubo_adpt_smr=True`
    (default) reuses `shift_current._adaptive_eta`/`_kmesh_spacing` (YWVS07).

    `method='qiao'` (default, needs `SR_R`/`SHR_R`/`SH_R` from
    `build_shc_operators`) or `'ryoo'` (needs `SAA_R`/`SBB_R` from
    `build_shc_ryoo_operators` instead, built from genuine ab-initio
    `.sIu`/`.sHu` rather than Qiao's `.mmn`+`.spn` approximation) --
    QZYZ18 vs. RPS19, expected to agree closely but not exactly (different
    approximations of the same physical quantity; see module docstring).

    Args:
      hr, AA_R           : Hartree/Bohr, HamiltonianR + position-operator R-matrix
      SS_R               : from `build_shc_operators` (needed by both methods)
      SR_R, SHR_R, SH_R  : from `build_shc_operators`, required iff method='qiao'
      SAA_R, SBB_R       : from `build_shc_ryoo_operators`, required iff method='ryoo'
      recip_lattice, real_lattice : Bohr^-1 / Bohr
      fermi_energies     : scalar or (nf,) array, Hartree
      mesh               : uniform full-BZ mesh (postw90's `berry_kmesh`)
      alpha, beta, gamma : 0-based Cartesian/spin-polarization directions
                           (wannier90's `shc_alpha`/`shc_beta`/`shc_gamma`
                           default to 1-based x,y,z = alpha=0,beta=1,gamma=2)
      kubo_eigval_max    : optional band-energy cutoff (postw90's
                           `kubo_eigval_max`), Hartree, None = no cutoff
      eta                : fixed smearing width, Hartree, used only if
                           `kubo_adpt_smr=False`

    Returns SHCResult (sigma in bare Bohr^2, atomic units).
    """
    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    nf = len(fermi_energies)
    Na, Nb, Nc = mesh
    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)
    nk_total = len(kpts)

    delta_k = _kmesh_spacing(mesh, recip_lattice) if kubo_adpt_smr else None

    sigma_sum = np.zeros(nf)

    for lo in range(0, nk_total, KCHUNK):
        kc = kpts[lo:lo + KCHUNK]

        eig, del_eig, dE, prod, mask = _shc_per_k_ingredients(
            hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice, kc,
            alpha, beta, gamma, degen_thresh, kubo_eigval_max,
            method=method, SAA_R=SAA_R, SBB_R=SBB_R,
        )

        if kubo_adpt_smr:
            eta_use = _adaptive_eta(del_eig, delta_k, prefactor=kubo_adpt_smr_fac, max_width=kubo_adpt_smr_max)
        else:
            eta_use = eta

        denom = dE ** 2 + eta_use ** 2
        weight = torch.where(mask, -2.0 / denom, torch.zeros_like(denom))
        omega_per_nm = weight * prod.imag                          # (nk, nw, nw)
        omega_per_n = omega_per_nm.sum(dim=-1)                      # sum over m -> (nk, nw)

        occ = (eig[:, :, None] < torch.as_tensor(fermi_energies, dtype=eig.dtype)[None, None, :]).to(omega_per_n.dtype)
        sigma_k = torch.einsum('kn,knf->kf', omega_per_n, occ)     # (nk, nf), Bohr^2

        sigma_sum += sigma_k.sum(dim=0).cpu().numpy()

    sigma_bohr2 = sigma_sum / nk_total

    return SHCResult(fermi_energies=fermi_energies, sigma=sigma_bohr2,
                      alpha=alpha, beta=beta, gamma=gamma, mesh=mesh)


def spin_hall_conductivity_ac(
    hr:            HamiltonianR,
    AA_R:          Tensor,
    SS_R:          Tensor,
    SR_R:          Tensor | None = None,
    SHR_R:         Tensor | None = None,
    SH_R:          Tensor | None = None,
    recip_lattice: np.ndarray = None,
    real_lattice:  np.ndarray = None,
    fermi_energy: float = None,
    omega:        np.ndarray = None,
    mesh:          tuple = (20, 20, 20),
    alpha:         int = 0,
    beta:          int = 1,
    gamma:         int = 2,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    kubo_eigval_max: float | None = None,
    kubo_adpt_smr: bool = True,
    eta:           float = 0.04 * EV_TO_HARTREE,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
    method:        str = "qiao",
    SAA_R:         Tensor | None = None,
    SBB_R:         Tensor | None = None,
) -> SHCACResult:
    """
    AC (frequency-dependent) spin Hall conductivity at one fixed Fermi
    energy (postw90 `berry_task=shc`, `shc_freq_scan=true`). wannier90:
    `berry_get_shc_klist`'s `lfreq` branch, sharing every ingredient with
    the Fermi-scan branch (`_shc_per_k_ingredients`) but with a complex
    resonance denominator at each optical frequency instead of a real
    Lorentzian:

        cdum = omega(ifreq) + i*eta_smr
        cfac = -2 / (rfac^2 - cdum^2)              (complex)
        omega_list(ifreq) += cfac * Im(prod)
        shc_k_freq += occ_freq(n) * omega_list     (T=0 step function at
                                                     this one fermi_energy)

    Same final unit conversion as the Fermi-scan case, applied to the
    complex result. `method='qiao'`/`'ryoo'` as in `spin_hall_conductivity`.

    Args mirror `spin_hall_conductivity`, except:
      fermi_energy : scalar, Hartree (ONE fixed Fermi energy, not a scan)
      omega        : (nfreq,) real, Hartree -- the optical frequency grid
                     (postw90's `kubo_freq_min/max/step`)

    Returns SHCACResult (sigma complex, bare Bohr^2, atomic units).
    """
    omega = np.asarray(omega, dtype=np.float64)
    nfreq = len(omega)
    Na, Nb, Nc = mesh
    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)
    nk_total = len(kpts)

    delta_k = _kmesh_spacing(mesh, recip_lattice) if kubo_adpt_smr else None

    sigma_sum = np.zeros(nfreq, dtype=np.complex128)

    for lo in range(0, nk_total, KCHUNK):
        kc = kpts[lo:lo + KCHUNK]

        eig, del_eig, dE, prod, mask = _shc_per_k_ingredients(
            hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice, kc,
            alpha, beta, gamma, degen_thresh, kubo_eigval_max,
            method=method, SAA_R=SAA_R, SBB_R=SBB_R,
        )

        if kubo_adpt_smr:
            eta_use = _adaptive_eta(del_eig, delta_k, prefactor=kubo_adpt_smr_fac, max_width=kubo_adpt_smr_max)
        else:
            eta_use = eta

        omega_t = torch.as_tensor(omega, dtype=torch.float64, device=eig.device)
        if isinstance(eta_use, torch.Tensor):
            eta_c = eta_use[..., None].to(torch.complex128)
        else:
            eta_c = torch.tensor(eta_use, dtype=torch.complex128, device=eig.device)
        cdum = omega_t[None, None, None, :].to(torch.complex128) + 1j * eta_c     # (nk,nw,nw,nfreq)
        cfac = -2.0 / (dE[..., None].to(torch.complex128) ** 2 - cdum ** 2)

        contrib = cfac * prod.imag[..., None].to(torch.complex128)                          # (nk,nw,nw,nfreq)
        contrib = torch.where(mask[..., None], contrib, torch.zeros_like(contrib))
        omega_per_n = contrib.sum(dim=-2)                                                    # sum over m -> (nk,nw,nfreq)

        occ_n = (eig < fermi_energy).to(torch.complex128)                                    # (nk,nw)
        sigma_k = torch.einsum('kn,knf->kf', occ_n, omega_per_n)                             # (nk,nfreq)

        sigma_sum += sigma_k.sum(dim=0).cpu().numpy()

    sigma_bohr2 = sigma_sum / nk_total

    return SHCACResult(fermi_energy=fermi_energy, omega=omega, sigma=sigma_bohr2,
                        alpha=alpha, beta=beta, gamma=gamma, mesh=mesh)


@register_si_unit("spin_hall_conductivity")
def _spin_hall_conductivity_to_si(sigma_bohr2, *, cell_volume_bohr3: float):
    """
    Bohr^2 -> (hbar/e)*S/cm, shared by `spin_hall_conductivity` (real) and
    `spin_hall_conductivity_ac` (complex). `berry.F90`'s
    `fac = 1e8*e^2/(hbar*V)/2` (see module docstring for the /2 and sign
    vs. `topology.anomalous_hall_conductivity`).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    sigma_ang2 = np.asarray(sigma_bohr2) * BOHR_TO_ANG ** 2
    fac = 1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3) / 2.0
    return sigma_ang2 * fac


@register_from_si_unit("spin_hall_conductivity")
def _spin_hall_conductivity_from_si(sigma_si, *, cell_volume_bohr3: float):
    """Inverse of `_spin_hall_conductivity_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    fac = 1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3) / 2.0
    sigma_ang2 = np.asarray(sigma_si) / fac
    return sigma_ang2 / BOHR_TO_ANG ** 2


@dataclass
class SpinNernstResult:
    """
    Intrinsic spin Nernst conductivity vs. temperature, atomic units.
    Convert with ``waw.units.to_si_units(result.alpha, "spin_nernst_conductivity",
    cell_volume_bohr3=..., kT_values=result.kT_values)`` for (hbar/e) A/(m K).
    The Kelvin temperature axis is recovered from ``kT_values`` via
    ``waw.units.K_B_HARTREE`` (kT/k_B).
    """
    kT_values:  np.ndarray     # (nT,) Hartree
    mu:         float          # Hartree, chemical potential the scan is taken at
    alpha:      np.ndarray     # (nT,) atomic units (Hartree*Bohr^2)
    energies:   np.ndarray     # (nE,) Hartree, the sigma(E) grid the Mott integral used
    sigma_of_E: np.ndarray     # (nE,) Bohr^2, the SHC-vs-energy curve the transform used
    alpha_dir:  int
    beta_dir:   int
    gamma_dir:  int
    mesh:       tuple


def spin_nernst_conductivity(
    hr:            HamiltonianR,
    AA_R:          Tensor,
    SS_R:          Tensor,
    SR_R:          Tensor | None = None,
    SHR_R:         Tensor | None = None,
    SH_R:          Tensor | None = None,
    recip_lattice: np.ndarray = None,
    real_lattice:  np.ndarray = None,
    *,
    mu:            float,
    kT_values,
    mesh:          tuple = (20, 20, 20),
    alpha:         int = 0,
    beta:          int = 1,
    gamma:         int = 2,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    kubo_eigval_max: float | None = None,
    kubo_adpt_smr: bool = True,
    eta:           float = 0.04 * EV_TO_HARTREE,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
    energies:      np.ndarray | None = None,
    energy_halfwidth: float = 0.4 * EV_TO_HARTREE,
    n_energies:    int = 81,
    method:        str = "qiao",
    SAA_R:         Tensor | None = None,
    SBB_R:         Tensor | None = None,
) -> SpinNernstResult:
    """
    Intrinsic spin Nernst conductivity alpha^s_ij(mu, T) (Hsieh, Prasad &
    Guo, PRB 106, 165102 (2022) Eq. 3/4 -- the thermoelectric companion of
    the SHC): the generalized Mott relation (Xiao-Yao-Niu 2006) applied to
    the energy-resolved SHC, reusing `topology._nernst_mott_integral`
    VERBATIM -- that function is a generic Sommerfeld-kernel convolution
    agnostic to what "sigma(E)" physically is (it already only takes an
    energy grid + a sigma(E) array + mu/kT), so no new low-level math is
    needed here at all, only orchestration: `spin_hall_conductivity`'s own
    Fermi-energy-scan capability directly gives the sigma(E) curve this
    needs.

    In the low-T limit this is the Mott formula alpha^s_ij = -(pi^2/3)
    (k_B^2 T/e) dsigma^s_ij/dE, Eq. (4) of the paper above.

    Convert via `waw.units.to_si_units(result.alpha, "spin_nernst_conductivity",
    cell_volume_bohr3=..., kT_values=result.kT_values)` for (hbar/e) A/(m K).

    Args mirror `spin_hall_conductivity`, plus:
      mu        : chemical potential (Hartree) to take the Nernst scan at (E_F)
      kT_values : scalar or array of kT (Hartree), the atomic-units
                  temperature axis; converted to Kelvin at SI-conversion
                  time via `waw.units.K_B_HARTREE`
      energies, energy_halfwidth, n_energies : the underlying SHC(E) energy
                  grid, as in `topology.anomalous_nernst_conductivity`
                  (defaults to `mu +/- max(energy_halfwidth, 8*kT_max)`)

    Returns SpinNernstResult (alpha in atomic units), also carrying the
    sigma(E) curve it used.
    """
    kT_values = np.atleast_1d(np.asarray(kT_values, dtype=np.float64))
    if energies is None:
        half = max(energy_halfwidth, 8.0 * float(kT_values.max()))
        energies = np.linspace(mu - half, mu + half, n_energies)
    energies = np.asarray(energies, dtype=np.float64)

    shc = spin_hall_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energies=energies, mesh=mesh, alpha=alpha, beta=beta, gamma=gamma,
        degen_thresh=degen_thresh, kubo_eigval_max=kubo_eigval_max,
        kubo_adpt_smr=kubo_adpt_smr, eta=eta,
        kubo_adpt_smr_fac=kubo_adpt_smr_fac, kubo_adpt_smr_max=kubo_adpt_smr_max,
        method=method, SAA_R=SAA_R, SBB_R=SBB_R,
    )
    sigma_of_E = shc.sigma   # (nE,) Bohr^2

    # `_nernst_mott_integral` broadcasts its (nE,) kernel against a
    # trailing-COMPONENT axis (AHC's own sigma_of_E is (nE,3)); without an
    # explicit singleton axis here, (nE,1)*(nE,) broadcasts into an
    # (nE,nE) outer product instead of collapsing the energy axis.
    alpha_out = np.array([_nernst_mott_integral(energies, sigma_of_E[:, None], mu, float(kt))[0]
                          for kt in kT_values])   # (nT,), Hartree*Bohr^2

    return SpinNernstResult(kT_values=kT_values, mu=float(mu), alpha=alpha_out,
                            energies=energies, sigma_of_E=sigma_of_E,
                            alpha_dir=alpha, beta_dir=beta, gamma_dir=gamma, mesh=mesh)


@register_si_unit("spin_nernst_conductivity")
def _spin_nernst_conductivity_to_si(alpha_atomic, *, cell_volume_bohr3: float, kT_values):
    """
    Atomic units -> (hbar/e) A/(m K), same Sommerfeld-integral bookkeeping
    as `topology._anomalous_nernst_to_si` but using `spin_hall_conductivity`'s
    own SI factor (`e^2/hbar/V /2`, not AHC's bare `e^2/hbar/V`).
    """
    alpha_atomic = np.asarray(alpha_atomic, dtype=np.float64)
    T_kelvin = np.asarray(kT_values, dtype=np.float64) / K_B_HARTREE
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    K_scm = BOHR_TO_ANG ** 2 * (1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3) / 2.0)
    prefac = -100.0 * HARTREE_TO_EV / T_kelvin                     # (nT,)
    prefac = prefac.reshape(prefac.shape + (1,) * (alpha_atomic.ndim - 1))
    return prefac * K_scm * alpha_atomic


@register_from_si_unit("spin_nernst_conductivity")
def _spin_nernst_conductivity_from_si(alpha_si, *, cell_volume_bohr3: float, kT_values):
    """Inverse of `_spin_nernst_conductivity_to_si` -- same `cell_volume_bohr3`/`kT_values` kwargs."""
    alpha_si = np.asarray(alpha_si, dtype=np.float64)
    T_kelvin = np.asarray(kT_values, dtype=np.float64) / K_B_HARTREE
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    K_scm = BOHR_TO_ANG ** 2 * (1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3) / 2.0)
    prefac = -100.0 * HARTREE_TO_EV / T_kelvin
    prefac = prefac.reshape(prefac.shape + (1,) * (alpha_si.ndim - 1))
    return alpha_si / (prefac * K_scm)
