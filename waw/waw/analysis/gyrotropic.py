"""
Gyrotropic effects (postw90 `gyrotropic` module): the D, K_orb, C, DOS,
Dw/tildeD, and NOA_orb tensors of Tsirkin, Aguado Puente & Souza, PRB 97,
035158 (2017) ("TAS17").

D/K_orb/C/DOS are "Fermi-surface" tensors: a single band n is picked out
at a time via the same "fake occupation" trick wannier90 uses (only band
n occupied, via `orbital_magnetization._imfgh_chunk`'s `occ_override` /
`topology._jjp_jjm_from_occ`), weighted by a Gaussian delta(E_n - E_F)
and averaged over a k-mesh restricted to an arbitrary parallelepiped box
(`gyrotropic_box`/`gyrotropic_kmesh`), since the Fermi-surface physics
these tensors probe is often localized near specific k-points.

Internally in Hartree/Bohr atomic units; the unit conversions to
Ampere/Ampere-cm^-1/eV^-1 Ang^-3 use wannier90's own `fac = ...`
prefactors (documented at each conversion function below).

D/K/C/Dw/NOA_orb have 1/(E_n-E_m) (or steeper) energy denominators near
band touchings/Weyl points, so results near such a feature are
mesh-sensitive and need a targeted, locally dense `gyrotropic_box`/
`gyrotropic_kmesh`, not a naive uniform full-BZ mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, position_operator_k, operator_k, hamiltonian_gauge_position
from ..units import (
    HARTREE_TO_EV, EV_TO_HARTREE, BOHR_TO_ANG, ANG_TO_BOHR, E_CHARGE, HBAR_SI, EPS0_SI,
    register_si_unit, register_from_si_unit,
)
from ..core.distributions import gaussian_smearing
from ._fourier_derivs import h_and_grad_frac_batch, KCHUNK
from .orbital_magnetization import _imfgh_chunk
from .topology import _AXIAL_PAIRS

__all__ = ["GyrotropicResult", "box_kmesh", "gyrotropic_tensors"]


def box_kmesh(box_corner, box, mesh: tuple[int, int, int]) -> np.ndarray:
    """
    A uniform mesh over an arbitrary parallelepiped sub-region of the BZ,
    in fractional (crystal) coordinates:

        k = box_corner + (i/N1, j/N2, k/N3) . box     i in [0,N1), etc.

    Args:
      box_corner: (3,) fractional coordinates of the box's origin corner
      box       : (3, 3) rows = the box's edge vectors, fractional coordinates
      mesh      : (N1, N2, N3) mesh density along each box edge

    Returns kpts: (N1*N2*N3, 3) fractional coordinates.
    """
    N1, N2, N3 = mesh
    i, j, k = np.meshgrid(np.arange(N1), np.arange(N2), np.arange(N3), indexing="ij")
    frac = np.stack([i.ravel() / N1, j.ravel() / N2, k.ravel() / N3], axis=-1)
    box = np.asarray(box, dtype=np.float64)
    box_corner = np.asarray(box_corner, dtype=np.float64)
    return box_corner[None, :] + frac @ box


def _curvature_omega(eig: torch.Tensor, A_H: torch.Tensor, freq_ha: torch.Tensor) -> torch.Tensor:
    """
    Band-resolved, frequency-dependent Berry curvature tildeOmega_kn(omega)
    (TAS17 Eq. 12):

        tildeOmega_kn,i(omega) = -2 sum_{m != n} Im[A_H(n,m,alpha_i).A_H(m,n,beta_i)]
                                  . omega_mn^2 / (omega_mn^2 - omega^2)

    (alpha_i, beta_i) = `topology._AXIAL_PAIRS`'s cyclic mapping, omega_mn =
    E_m - E_n. Real poles at omega = omega_mn (no smearing) -- unlike D/K/C,
    which use a Gaussian delta(E-E_F), Dw's only lineshape is this
    resonance denominator itself.

    At freq=0 this reduces to the ordinary Berry curvature for AA_R=0
    (verified against `topology.berry_curvature_cartesian`); the general
    (nonzero AA_R) freq=0 limit against the full J0+J1+J2 curvature is not
    independently cross-checked here.

    Args:
      eig : (nc, nw) real, Hartree
      A_H : (nc, 3, nw, nw) complex, Bohr -- Hamiltonian-gauge position
            matrix (`core.hamiltonian.hamiltonian_gauge_position`)
      freq_ha: (nfreq,) real, Hartree

    Returns curv_w: (nc, nw, nfreq, 3) real, Bohr^2.
    """
    nc, nw = eig.shape
    nfreq = freq_ha.shape[0]

    wmn = eig[:, None, :] - eig[:, :, None]           # (nc,nw,nw): E_m - E_n (row n, col m)
    wmn2 = (wmn ** 2)[..., None]                       # (nc,nw,nw,1)
    freq2 = (freq_ha ** 2).to(eig.dtype)[None, None, None, :]   # (1,1,1,nfreq)
    denom = wmn2 - freq2                               # (nc,nw,nw,nfreq)
    eye = torch.eye(nw, dtype=torch.bool, device=eig.device)
    ratio = torch.where(eye[None, :, :, None], torch.zeros_like(denom), wmn2 / denom)

    curv_w = torch.zeros(nc, nw, nfreq, 3, dtype=eig.dtype, device=eig.device)
    for i, (alpha, beta) in enumerate(_AXIAL_PAIRS):
        # prod[k,n,m] = A_H(n,m,alpha) . A_H(m,n,beta)
        prod = A_H[:, alpha] * A_H[:, beta].transpose(-1, -2)
        curv_w[..., i] = -2.0 * torch.einsum('knm,knmf->knf', prod.imag.to(eig.dtype), ratio)
    return curv_w


def _bnl_orb(eig: torch.Tensor, A_H: torch.Tensor, del_eig: torch.Tensor) -> torch.Tensor:
    """
    B_{n,l,a,c} for all (n,l) band pairs (masked to occ x unocc later),
    wannier90's `gyrotropic_get_NOA_Bnl_orb`:

        B(n,l,a,c) = -i(dE_n/dk_a + dE_l/dk_a) . A_H(n,l,c)
                    + sum_m [ (E_n-E_m).A_H(n,m,a).A_H(m,l,c)
                             - (E_l-E_m).A_H(n,m,c).A_H(m,l,a) ]

    (sum over all bands m; the occ/unocc restriction only applies at the
    (n,l) outer pair in `_noa_orb_contribution`).

    Args:
      eig: (nc, nw) real, Hartree
      A_H: (nc, 3, nw, nw) complex, Bohr
      del_eig: (nc, nw, 3) real, Hartree*Bohr

    Returns Bnl: (nc, 3, 3, nw, nw) complex, Hartree*Bohr^2 -- axes (a, c, n, l).
    """
    nc, nw = eig.shape
    eig_c = eig.to(A_H.dtype)   # real -> complex, for matmul with A_H

    Bnl = torch.zeros(nc, 3, 3, nw, nw, dtype=A_H.dtype, device=A_H.device)
    for a in range(3):
        dea = del_eig[:, :, a]
        sum_del_a = dea[:, :, None] + dea[:, None, :]      # (nc,nw,nw): dE_n/dk_a + dE_l/dk_a
        for c in range(3):
            term1 = -1j * sum_del_a.to(A_H.dtype) * A_H[:, c]   # [n,l] = ... . A_H(n,l,c)

            AaAc = torch.matmul(A_H[:, a], A_H[:, c])                          # sum_m A_H(n,m,a)A_H(m,l,c)
            Aa_Em_Ac = torch.matmul(A_H[:, a] * eig_c[:, None, :], A_H[:, c])  # sum_m E_m.A_H(n,m,a)A_H(m,l,c)
            part1 = eig_c[:, :, None] * AaAc - Aa_Em_Ac                        # (E_n - E_m) sum, E_n along rows

            AcAa = torch.matmul(A_H[:, c], A_H[:, a])                          # sum_m A_H(n,m,c)A_H(m,l,a)
            Ac_Em_Aa = torch.matmul(A_H[:, c] * eig_c[:, None, :], A_H[:, a])  # sum_m E_m.A_H(n,m,c)A_H(m,l,a)
            part2 = eig_c[:, None, :] * AcAa - Ac_Em_Aa                        # (E_l - E_m) sum, E_l along cols

            Bnl[:, a, c] = term1 + part1 - part2
    return Bnl


def _noa_orb_contribution(
    eig: torch.Tensor, A_H: torch.Tensor, del_eig: torch.Tensor,
    occ_mask: torch.Tensor, freq_ha: torch.Tensor,
) -> torch.Tensor:
    """
    The NOA_orb gamma tensor's per-k contribution for one Fermi energy
    (`occ_mask`), summed over occ x unocc band pairs and evaluated on a
    frequency grid (wannier90's `gyrotropic_get_NOA_k`, orbital-only branch):

        gamma_{ab,c}(w) = sum_{n in occ, l in unocc}
            Re(multW1).Re[A_H(l,n,b).B(n,l,a,c) - A_H(l,n,a).B(n,l,b,c)]
          + Re(-multW1.(2.w_ln^2.multW1 + 1)).(dE_n/dk_c + dE_l/dk_c)
              .Im[A_H(n,l,a).A_H(l,n,b)]

        multW1 = 1 / (w_ln^2 - w^2),  w_ln = E_l - E_n

    (a, b) = `topology._AXIAL_PAIRS`'s cyclic mapping for the `ab` axial
    index; `c` runs over the 3 Cartesian directions independently. No
    smearing -- occ/unocc is a hard cut at the given Fermi energy (matching
    the Fortran's own `eig(n) < fermi_energy_list(ifermi)` partition).

    Args:
      eig: (nc, nw) real, Hartree
      A_H: (nc, 3, nw, nw) complex, Bohr
      del_eig: (nc, nw, 3) real, Hartree*Bohr
      occ_mask: (nc, nw) bool -- True where eig < this Fermi energy
      freq_ha: (nfreq,) real, Hartree

    Returns (nc, 3, 3, nfreq) real, Hartree^-1 * Bohr^3 (raw, un-kweighted).
    """
    nc, nw = eig.shape
    nfreq = freq_ha.shape[0]

    Bnl = _bnl_orb(eig, A_H, del_eig)   # (nc,3,3,nw,nw), axes (a,c,n,l)

    wln = eig[:, None, :] - eig[:, :, None]                 # (nc,nw,nw): E_l - E_n (row n, col l)
    wln2 = (wln ** 2)[..., None]                             # (nc,nw,nw,1)
    freq2 = (freq_ha ** 2).to(eig.dtype)[None, None, None, :]
    denom = wln2 - freq2                                     # (nc,nw,nw,nfreq)

    unocc_mask = ~occ_mask
    pair_mask_b = occ_mask[:, :, None] & unocc_mask[:, None, :]           # (nc,nw,nw) bool: n occ, l unocc
    valid = pair_mask_b[..., None] & (denom != 0)                          # (nc,nw,nw,nfreq)

    # safe division: n==l (or any exactly-degenerate occ/unocc pair at
    # freq=0) gives denom=0 -- `pair_mask` alone can't zero this out
    # afterward since 0*NaN=NaN, so replace the denominator before
    # dividing (same pattern as core.hamiltonian.hamiltonian_gauge_position's
    # own degenerate-pair guard).
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    multW1 = torch.where(valid, 1.0 / safe_denom, torch.zeros_like(denom))   # (nc,nw,nw,nfreq) real
    multWm = multW1
    multWe = -multW1 * (2.0 * wln2 * multW1 + 1.0)

    pair_mask = pair_mask_b.to(eig.dtype)   # (nc,nw,nw): n occ, l unocc

    out = torch.zeros(nc, 3, 3, nfreq, dtype=eig.dtype, device=eig.device)
    for ab, (a, b) in enumerate(_AXIAL_PAIRS):
        A_ln_b = A_H[:, b].transpose(-1, -2)   # [n,l] = A_H(l,n,b)
        A_ln_a = A_H[:, a].transpose(-1, -2)   # [n,l] = A_H(l,n,a)
        A_nl_a = A_H[:, a]                     # [n,l] = A_H(n,l,a)
        Im_AnlA_AlnB = (A_nl_a * A_ln_b).imag  # (nc,nw,nw)

        for c in range(3):
            term_m = (A_ln_b * Bnl[:, a, c] - A_ln_a * Bnl[:, b, c]).real   # (nc,nw,nw)
            sum_del_c = del_eig[:, :, c][:, :, None] + del_eig[:, :, c][:, None, :]   # (nc,nw,nw)

            weighted = (
                multWm * (term_m * pair_mask)[..., None]
                + multWe * (sum_del_c * Im_AnlA_AlnB * pair_mask)[..., None]
            )   # (nc,nw,nw,nfreq)
            out[:, ab, c, :] = weighted.sum(dim=(1, 2))
    return out


@dataclass
class GyrotropicResult:
    """
    Gyrotropic Fermi-surface tensors vs. Fermi energy (task-dependent
    fields are None if not requested). D/Dw are already dimensionless
    (a pure Bohr^3/Bohr^3 ratio -- no unit-system conversion needed, see
    `gyrotropic_tensors`); K_orb/C/DOS/NOA_orb are atomic-units raw sums,
    convert with ``waw.units.to_si_units(result.K_orb, "gyrotropic_K",
    cell_volume_bohr3=...)`` (similarly ``"gyrotropic_C"``/
    ``"gyrotropic_dos"``/``"gyrotropic_noa"``) for Ampere/Ampere-cm^-1/
    eV^-1 Ang^-3/the NOA practical unit.
    """
    fermi_energies: np.ndarray          # (nf,) Hartree
    D:              np.ndarray | None   # (nf, 3, 3) dimensionless
    K_orb:          np.ndarray | None   # (nf, 3, 3) Hartree*Bohr^3 (raw, atomic)
    C:              np.ndarray | None   # (nf, 3, 3) Hartree*Bohr^2 (raw, atomic)
    DOS:            np.ndarray | None   # (nf,) Hartree^-1*Bohr^-3 (raw, atomic)
    Dw:             np.ndarray | None   # (nf, nfreq, 3, 3) dimensionless
    NOA_orb:        np.ndarray | None   # (nf, 3, 3, nfreq) Hartree^-1*Bohr^3 (raw, atomic) -- axes (ab-axial, c, freq)
    frequencies:    np.ndarray | None   # (nfreq,) Hartree, or None if neither Dw nor NOA requested
    box_kmesh:      tuple               # (N1, N2, N3) mesh density used


def gyrotropic_tensors(
    hr: HamiltonianR, AA_R: torch.Tensor, BB_R: torch.Tensor, CC_R: torch.Tensor,
    recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energies, box, box_corner, kmesh: tuple[int, int, int],
    sigma: float = 0.01 * EV_TO_HARTREE, degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    tasks: tuple[str, ...] = ("D", "K", "C", "DOS"),
    frequencies=None,
) -> GyrotropicResult:
    """
    The D, K_orb, C, and DOS Fermi-surface tensors (TAS17 Eq. 2, Eq. 3,
    Eq. B6, and the density of states):

        D_ab   = (1/Nk) sum_{k,n} delta(E_kn-E_F) . dE_kn/dk_a . Omega^b_kn
        K_orb,ab = (1/Nk) sum_{k,n} delta(E_kn-E_F) . dE_kn/dk_a . m^orb_kn,b
        C_ab   = (1/Nk) sum_{k,n} delta(E_kn-E_F) . dE_kn/dk_a . dE_kn/dk_b
        DOS    = (1/Nk) sum_{k,n} delta(E_kn-E_F)

    where Omega_kn (curvature) and m^orb_kn (orbital moment) are extracted
    per band via the same "fake occupation" trick as wannier90 (only band n
    occupied, `orbital_magnetization._imfgh_chunk`'s `occ_override`), not a
    Fermi-sea sum over all occupied bands. Adjacent-band degeneracies are
    skipped (`degen_thresh`) since Omega/m^orb are only meaningful for a
    non-degenerate band.

    K_orb/C need `BB_R`/`CC_R` regardless of whether `D`/`DOS`-only is
    requested; D/DOS alone could skip the extra Fourier interpolation, but
    these tensors are typically requested together.

    Args:
      hr, AA_R, BB_R, CC_R: Hartree/Bohr, same R-grid (as `orbital_magnetization`)
      recip_lattice, real_lattice: Bohr^-1/Bohr
      fermi_energies: scalar or (nf,) array, Hartree
      box, box_corner, kmesh: see `box_kmesh`
      sigma: Gaussian smearing width for delta(E-E_F), Hartree
                (`gyrotropic_smr_fixed_en_width`)
      degen_thresh: adjacent-band skip threshold, Hartree
                (`gyrotropic_degen_thresh`)
      tasks: subset of ("D", "K", "C", "DOS", "Dw", "NOA") to compute
      frequencies: scalar or (nfreq,) array, Hartree -- required if "Dw" or
                "NOA" in tasks (TAS17 Eq. 12's frequency-dependent tildeD
                tensor / Eq. C10-15's frequency-dependent NOA_orb gamma
                tensor; both have real poles at band-pair transition
                energies, no smearing, and share the same frequency grid
                here)

    Returns GyrotropicResult (unrequested fields are None).
    """
    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    nf = len(fermi_energies)

    want_D = "D" in tasks
    want_K = "K" in tasks
    want_C = "C" in tasks
    want_DOS = "DOS" in tasks
    want_Dw = "Dw" in tasks
    want_NOA = "NOA" in tasks
    need_BB_CC = want_K
    need_freq = want_Dw or want_NOA

    if need_freq:
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        freq_ha_t = torch.as_tensor(frequencies, dtype=torch.float64)
        nfreq = len(frequencies)
    else:
        frequencies = None
        nfreq = 0

    kpts = box_kmesh(box_corner, box, kmesh)
    nk = len(kpts)
    N1, N2, N3 = kmesh
    kweight = abs(np.linalg.det(np.asarray(box, dtype=np.float64))) / (N1 * N2 * N3)

    D_sum = np.zeros((nf, 3, 3)) if want_D else None
    K_sum = np.zeros((nf, 3, 3)) if want_K else None
    C_sum = np.zeros((nf, 3, 3)) if want_C else None
    DOS_sum = np.zeros(nf) if want_DOS else None
    Dw_sum = np.zeros((nf, nfreq, 3, 3)) if want_Dw else None
    NOA_sum = np.zeros((nf, 3, 3, nfreq)) if want_NOA else None

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.complex128)

    for lo in range(0, nk, KCHUNK):
        kc = kpts[lo:lo + KCHUNK]
        H0, grad_frac = h_and_grad_frac_batch(hr, kc)
        grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad_frac)
        grad_cart = 0.5 * (grad_cart + grad_cart.conj().transpose(-1, -2))

        A_k, omega_bar_k = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)
        if need_BB_CC:
            BB_k, _ = position_operator_k(BB_R, hr.R_vectors, hr.degen, real_lattice, kc)
            CC_k = torch.stack([torch.stack([
                operator_k(CC_R[a, b], hr.R_vectors, hr.degen, kc) for b in range(3)
            ], dim=1) for a in range(3)], dim=1)   # (nc, 3, 3, nw, nw)
        else:
            nw = hr.nw
            nc = len(kc)
            BB_k = torch.zeros(nc, 3, nw, nw, dtype=A_k.dtype)
            CC_k = torch.zeros(nc, 3, 3, nw, nw, dtype=A_k.dtype)

        # hamiltonian_gauge_position always diagonalizes H0 and returns
        # del_eig (needed by D/K/C/Dw alike) plus A_H (needed only by Dw,
        # but cheap enough to always compute here -- same simplification
        # as always requiring BB_R/CC_R for K, see this function's docstring).
        eig, UU, del_eig_t, A_H = hamiltonian_gauge_position(H0, grad_cart, A_k)
        eig_np = eig.cpu().numpy()
        nc, nw = eig_np.shape
        del_eig = del_eig_t.cpu().numpy()              # (nc,nw,3) Hartree*Bohr

        if want_Dw:
            curv_w_all = _curvature_omega(eig, A_H, freq_ha_t)   # (nc,nw,nfreq,3) Bohr^2
            curv_w_all_np = curv_w_all.cpu().numpy()

        # adjacent-band degeneracy skip (ascending eig within this k-chunk),
        # transcribed from gyrotropic.F90's "avoid degeneracies" guard
        skip = np.zeros((nc, nw), dtype=bool)
        if nw > 1:
            close = (eig_np[:, 1:] - eig_np[:, :-1]) <= degen_thresh
            skip[:, 1:] |= close
            skip[:, :-1] |= close

        for n in range(nw):
            if skip[:, n].all():
                continue
            onehot = torch.zeros(nc, nw, dtype=torch.float64)
            onehot[:, n] = 1.0

            curv_nk = orb_nk = None
            if want_K:
                imf_n, img_n, imh_n = _imfgh_chunk(
                    H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k,
                    np.zeros(1), occ_override=onehot[None],
                )
                curv_nk = imf_n[:, 0, :]                 # (nc,3) Bohr^2
                orb_nk = imh_n[:, 0, :] - img_n[:, 0, :]  # (nc,3) Hartree*Bohr^2
            elif want_D:
                imf_n, _, _ = _imfgh_chunk(
                    H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k,
                    np.zeros(1), occ_override=onehot[None],
                )
                curv_nk = imf_n[:, 0, :]

            dvec = del_eig[:, n, :]                       # (nc,3) Hartree*Bohr
            if want_Dw:
                curv_w_n = curv_w_all_np[:, n, :, :]       # (nc,nfreq,3) Bohr^2

            for ife in range(nf):
                arg = eig_np[:, n] - fermi_energies[ife]
                delta = gaussian_smearing(arg, sigma)          # (nc,) 1/Hartree
                delta = np.where(skip[:, n], 0.0, delta) * kweight

                if want_D:
                    D_sum[ife] += np.einsum('ka,kb->ab', dvec, curv_nk * delta[:, None])
                if want_K:
                    K_sum[ife] += np.einsum('ka,kb->ab', dvec, orb_nk * delta[:, None])
                if want_C:
                    C_sum[ife] += np.einsum('ka,kb->ab', dvec, dvec * delta[:, None])
                if want_DOS:
                    DOS_sum[ife] += delta.sum()
                if want_Dw:
                    Dw_sum[ife] += np.einsum('ka,kfb->fab', dvec * delta[:, None], curv_w_n)

        if want_NOA:
            # NOA is a genuine occ x unocc sum with a hard Fermi-energy cut
            # (no per-band isolation, no smearing), computed once per
            # k-chunk per Fermi energy, outside the D/K/C/Dw per-band loop.
            for ife in range(nf):
                occ_mask = torch.as_tensor(eig_np < fermi_energies[ife])
                noa_k = _noa_orb_contribution(eig, A_H, del_eig_t, occ_mask, freq_ha_t)   # (nc,3,3,nfreq)
                NOA_sum[ife] += noa_k.sum(dim=0).cpu().numpy() * kweight

    # `kweight` (= det(box)/(N1*N2*N3)) already provides the full
    # per-k-point normalization; no further division by nk here.

    cell_volume_bohr3 = abs(np.linalg.det(real_lattice))

    # D/Dw are a bare Bohr^3/Bohr^3 ratio, already dimensionless (no
    # to_si_units/to_eVA_units conversion needed or registered).
    D = D_sum / cell_volume_bohr3 if want_D else None
    Dw = Dw_sum / cell_volume_bohr3 if want_Dw else None

    # K_orb/C/DOS/NOA_orb are returned as the raw atomic sum (Hartree*Bohr^3,
    # Hartree*Bohr^2, Hartree^-1*Bohr^-3, Hartree^-1*Bohr^3 respectively);
    # convert with `waw.units.to_si_units(result.K_orb, "gyrotropic_K",
    # cell_volume_bohr3=...)` (similarly "gyrotropic_C"/"gyrotropic_dos"/
    # "gyrotropic_noa") -- see the registered converters below for the `fac`
    # formulas.
    K_orb = K_sum if want_K else None
    C = C_sum if want_C else None
    DOS = DOS_sum if want_DOS else None
    NOA_orb = NOA_sum if want_NOA else None

    return GyrotropicResult(
        fermi_energies=fermi_energies, D=D, K_orb=K_orb, C=C, DOS=DOS,
        Dw=Dw, NOA_orb=NOA_orb, frequencies=frequencies, box_kmesh=kmesh,
    )


@register_si_unit("gyrotropic_K")
def _gyrotropic_K_to_si(K_atomic, *, cell_volume_bohr3: float):
    """
    Hartree*Bohr^3 -> Ampere. raw = delta(1/Ha)*dvec(Ha*Bohr)*orb(Ha*Bohr^2)
    -> Ha*Bohr^3, i.e. eV*Ang^3 after conversion. wannier90's K_orb `fac`:
    fac = e_SI^2/(2 hbar_SI cell_volume_ang3) (cell_volume left in Ang^3,
    not m^3, since it cancels the Ang^3 in the raw eV*Ang^3 value).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    K_raw_eVAng3 = np.asarray(K_atomic) * HARTREE_TO_EV * BOHR_TO_ANG ** 3
    fac_K = E_CHARGE ** 2 / (2.0 * HBAR_SI * cell_volume_ang3)
    return K_raw_eVAng3 * fac_K


@register_from_si_unit("gyrotropic_K")
def _gyrotropic_K_from_si(K_si, *, cell_volume_bohr3: float):
    """Inverse of `_gyrotropic_K_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    fac_K = E_CHARGE ** 2 / (2.0 * HBAR_SI * cell_volume_ang3)
    K_raw_eVAng3 = np.asarray(K_si) / fac_K
    return K_raw_eVAng3 * EV_TO_HARTREE * ANG_TO_BOHR ** 3


@register_si_unit("gyrotropic_C")
def _gyrotropic_C_to_si(C_atomic, *, cell_volume_bohr3: float):
    """
    Hartree*Bohr^2 -> Ampere/cm. raw = delta(1/Ha)*dvec(Ha*Bohr)*dvec(Ha*Bohr)
    -> Ha*Bohr^2, i.e. eV*Ang^2 after conversion. wannier90's C `fac`:
    fac = 1e8*e_SI^2/(2*pi*hbar_SI*cell_volume_ang3) (1e8 converts the
    implicit Ang^-1 in eV*Ang^2/cell_volume(Ang^3) = eV/Ang into cm^-1).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    C_raw_eVAng2 = np.asarray(C_atomic) * HARTREE_TO_EV * BOHR_TO_ANG ** 2
    fac_C = 1.0e8 * E_CHARGE ** 2 / (2.0 * np.pi * HBAR_SI * cell_volume_ang3)
    return C_raw_eVAng2 * fac_C


@register_from_si_unit("gyrotropic_C")
def _gyrotropic_C_from_si(C_si, *, cell_volume_bohr3: float):
    """Inverse of `_gyrotropic_C_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    fac_C = 1.0e8 * E_CHARGE ** 2 / (2.0 * np.pi * HBAR_SI * cell_volume_ang3)
    C_raw_eVAng2 = np.asarray(C_si) / fac_C
    return C_raw_eVAng2 * EV_TO_HARTREE * ANG_TO_BOHR ** 2


@register_si_unit("gyrotropic_dos")
def _gyrotropic_dos_to_si(DOS_atomic, *, cell_volume_bohr3: float):
    """
    Hartree^-1*Bohr^-3 -> eV^-1 Ang^-3. raw = delta(1/Ha) summed -> convert
    1/Ha to 1/eV (divide by HARTREE_TO_EV, since a delta function's value
    scales inversely with its argument's unit), then divide by
    cell_volume(Ang^3).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    DOS_raw_invEv = np.asarray(DOS_atomic) / HARTREE_TO_EV
    return DOS_raw_invEv / cell_volume_ang3


@register_from_si_unit("gyrotropic_dos")
def _gyrotropic_dos_from_si(DOS_si, *, cell_volume_bohr3: float):
    """Inverse of `_gyrotropic_dos_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    DOS_raw_invEv = np.asarray(DOS_si) * cell_volume_ang3
    return DOS_raw_invEv * HARTREE_TO_EV


@register_si_unit("gyrotropic_noa")
def _gyrotropic_noa_to_si(NOA_atomic, *, cell_volume_bohr3: float):
    """
    Hartree^-1*Bohr^3 -> the NOA_orb practical unit. raw =
    multW1(1/Ha^2)*A_H(Bohr)*Bnl(Ha*Bohr^2) [or the multWe/Im[AA] term, same
    units] -> Ha^-1*Bohr^3 -> eV^-1*Ang^3 (divide by HARTREE_TO_EV, multiply
    by BOHR_TO_ANG^3). wannier90's NOA_orb `fac`:
    fac = 1e10*e_SI/(cell_volume_ang3*eps0_SI).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    NOA_raw_eVAng3 = (np.asarray(NOA_atomic) * BOHR_TO_ANG ** 3) / HARTREE_TO_EV
    fac_NOA = 1.0e10 * E_CHARGE / (cell_volume_ang3 * EPS0_SI)
    return NOA_raw_eVAng3 * fac_NOA


@register_from_si_unit("gyrotropic_noa")
def _gyrotropic_noa_from_si(NOA_si, *, cell_volume_bohr3: float):
    """Inverse of `_gyrotropic_noa_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    fac_NOA = 1.0e10 * E_CHARGE / (cell_volume_ang3 * EPS0_SI)
    NOA_raw_eVAng3 = np.asarray(NOA_si) / fac_NOA
    return (NOA_raw_eVAng3 * HARTREE_TO_EV) * ANG_TO_BOHR ** 3
