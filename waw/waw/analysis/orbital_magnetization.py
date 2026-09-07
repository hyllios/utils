"""
Orbital magnetization via Wannier interpolation (postw90 `berry_task = morb`).

**RESOLVED 2026-07-29: there was no bug here. `img`/`imh`/CC are correct;
the postw90 "reference" they were compared against was corrupted.**

`.uHu` (and `.uIu`) carry NO per-block labels. pw2wannier90 writes their
(b1, b2) blocks in the neighbour order of the `.nnkp` it was handed, and
postw90 reads them SEQUENTIALLY into its own internally regenerated order
(`kmesh_get`). waw writes its own `.nnkp` (header: "Written by waw") with a
different -- equally valid -- permutation of the same b-vector set, e.g. at
Gamma for bcc Fe 4x4x4 the neighbour k-indices are
w90 [1,4,16,3,7,19,12,13,28,48,49,52] vs waw [48,49,52,12,13,3,1,7,4,28,19,16].
postw90 therefore pairs every uHu block with the WRONG b-vector and the
WRONG neighbour gauge, corrupting CC and everything built on it (K_orb,
NOA_orb, M_orb), while leaving `.mmn`-derived quantities (which ARE labelled
by (ik, ik2, G) and matched on read) and `imf`-only quantities (AHC, ANC,
gyrotropic C and D) correct.

Proof, at machine precision: substituting w90's own (wb, bk, neighbour-k)
order into waw's `compute_cc_r` construction reproduces postw90's dumped
CC(Gamma) to 3e-10 -- so the two implementations are algorithmically
identical (`tests/test_gyrotropic_vs_postw90.py::
test_postw90_CC_is_waw_CC_with_w90s_neighbour_order`). The dumps come from a
patched wannier90 v3.1.0 built here, verified to reproduce the shipped
binary's M_orb bit-for-bit before being trusted.

Corollaries:
  * waw's own pipeline is self-consistent (it writes the `.nnkp`, reads back
    `.mmn`/`.uHu` in that same order), so waw's M_orb/K_orb/SAC numbers do
    not inherit this.
  * **postw90 must NOT be used as a reference for uHu/uIu-derived quantities
    unless the `.nnkp` was written by wannier90 itself** (`wannier90.x -pp`)
    and the overlaps regenerated against it. Retracted on this basis: the
    2026-07-28 claims that img was "17% high" and gyrotropic K_orb "42-74%
    off", and the suspicion that postw90's nonzero M_orb for
    time-reversal-symmetric Te came from its Wigner-Seitz set -- it came
    from this mispairing (a scrambled CC breaks the symmetry).
  * **The sign question is closed too (2026-07-29).** Permuting the `.uHu`
    blocks from waw's neighbour order into wannier90's and re-running the
    patched postw90 gives M_orb = [0.0004, -0.0056, -0.0658] mu_B/cell --
    waw's answer, sign included (the same binary on the unpermuted file gives
    [0.0895, 0.0521, 0.0512]). So this module is certified end to end. The
    apparent disagreement with wannier90's published tutorial-19 value
    (+0.0658) is the DFT setup, not the code: their example19 magnetizes Fe
    along -z (`starting_magnetization = -1`), ours along +z (`+0.4`), and
    M_orb is a pseudovector that flips with the magnetization; the magnitude
    0.0658 agrees to three figures.

TESTING HAZARD worth knowing: postw90 APPENDS to `seedname.wpout`. Grepping
the file for a result returns the OLDEST run unless you take the last match --
which is how the mispaired value initially looked reproducible across changes.

Ceresoli, Thonhauser, Vanderbilt & Resta, PRB 74, 024408 (2006) ("CTVR06")
and Lopez, Vanderbilt, Souza & Tsirkin, PRB 85, 014435 (2012) ("LVTS12")
give the intrinsic orbital magnetization of a Bloch band as a Fermi-surface
average of three gauge-invariant traces, -2Im[f], -2Im[g], -2Im[h]
(CTVR06 Eq. 33-35 / LVTS12 Eq. 6-8), each a J0+J1+J2 sum in the WYSV06
style used by `topology.wannier_interpolated_curvature` for the anomalous
Hall conductivity (imf is exactly that curvature trace):

    M_orb,a(E_F) = -( <img_a> + <imh_a> - 2 E_F <imf_a> )   [Bohr magneton/cell]

with <.> the Brillouin-zone average, everything in atomic units (Hartree,
Bohr). In atomic units mu_B = 1/2 and -(e/2*hbar) = -1/2, so these cancel
and no extra unit-conversion factor is needed beyond the overall sign
(wannier90's own eV/Angstrom bookkeeping requires an explicit factor,
`fac = -eV_au/bohr**2` in berry.F90, for the same reason).

-2Im[f] needs only AA_R (`core.hamiltonian.compute_position_r`), the same
machinery `topology.wannier_interpolated_curvature`/
`anomalous_hall_conductivity` use. -2Im[g] and -2Im[h] need two additional
real-space quantities, BB_R and CC_R (`core.hamiltonian.compute_bb_r`/
`compute_cc_r`); CC_R needs the `.uHu` file (`interfaces.wannier90.io.
read_uHu`, `pw2wannier90 write_uHu=.true.`), the ab-initio matrix element
<u_{k+b1}|H_k|u_{k+b2}> that BB_R alone (.mmn + eigenvalues) cannot supply.

`.uHu` matrix elements follow the same eV convention as `.eig`; convert
with `EV_TO_HARTREE` before calling `compute_cc_r`, as `interfaces.
wannier90.loader.load` does for `.eig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, position_operator_k, operator_k, _wigner_seitz
from ._fourier_derivs import h_and_grad_frac_batch
from .topology import _jjp_jjm_batch, _jjp_jjm_from_occ, _AXIAL_PAIRS, _KCHUNK

__all__ = ["OrbitalMagnetizationResult", "orbital_magnetization"]


@dataclass
class OrbitalMagnetizationResult:
    """Orbital magnetization vs. Fermi energy, in Bohr magneton/cell."""
    fermi_energies: np.ndarray   # (nf,) Hartree
    m_orb:          np.ndarray   # (nf, 3) Bohr magneton/cell, Cartesian (x, y, z)
    mesh:           tuple        # (N_a, N_b, N_c) k-mesh density used


def _imfgh_chunk(
    H0: torch.Tensor, grad_cart: torch.Tensor,
    A_k: torch.Tensor, omega_bar_k: torch.Tensor,
    BB_k: torch.Tensor, CC_k: torch.Tensor,
    fermi_energies: np.ndarray,
    occ_override: torch.Tensor | None = None,
) -> tuple:
    """
    -2Im[f], -2Im[g], -2Im[h] (each J0+J1+J2-summed) for one k-chunk and a
    list of Fermi energies. Transcribed from wannier90's
    `berry.F90::berry_get_imfgh_klist`.

    H0, A_k, omega_bar_k, BB_k, CC_k are in the Wannier gauge (H0 = Wannier
    H(k), not diagonalized); JJp/JJm/f/g are built in the H(k) eigenbasis
    then rotated back to the Wannier gauge, matching
    `topology.wannier_interpolated_curvature`'s imf construction.

    `occ_override` (optional): (n_scen, nc, nw) real 0/1 occupation weight
    per scenario, bypassing the `eig < fermi_energies[i]` threshold
    (`fermi_energies` then used only for its length). This is
    `postw90.gyrotropic`'s "fake occupation" trick for extracting a single
    band's curvature/orbital moment rather than a Fermi-sea sum
    (`wham_get_eig_UU_HH_JJlist`/`wham_get_occ_mat_list`'s `occ=` branches;
    see `topology._jjp_jjm_from_occ`). `occ_override=None` reproduces the
    plain Fermi-threshold behavior.

    Returns (imf, img, imh), each (nk_chunk, nf, 3) real, Hartree*Bohr^2
    (axial vector: yz, zx, xy).
    """
    nc, nw = H0.shape[0], H0.shape[-1]
    nf = len(fermi_energies)
    if occ_override is not None:
        assert occ_override.shape[0] == nf

    eig, UU = torch.linalg.eigh(H0)                 # (nc,nw),(nc,nw,nw), Hartree
    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)

    imf = np.empty((nc, nf, 3))
    img = np.empty((nc, nf, 3))
    imh = np.empty((nc, nf, 3))

    for ife, fe in enumerate(fermi_energies):
        if occ_override is not None:
            occ = occ_override[ife].to(UU.dtype)             # (nc,nw)
            JJp_e, JJm_e = _jjp_jjm_from_occ(dH_eig, eig, occ_override[ife])
        else:
            occ = (eig < fe).to(UU.dtype)                     # (nc,nw)
            JJp_e, JJm_e = _jjp_jjm_batch(dH_eig, eig, float(fe))
        U_occ = UU * occ[:, None, :]
        f_proj = U_occ @ U_occ.conj().transpose(-1, -2)      # (nc,nw,nw), Wannier gauge
        eye = torch.eye(nw, dtype=UU.dtype, device=UU.device)
        g_proj = eye - f_proj

        JJp = torch.einsum('kin,kanm,kjm->kaij', UU, JJp_e, UU.conj())   # -> Wannier gauge
        JJm = torch.einsum('kin,kanm,kjm->kaij', UU, JJm_e, UU.conj())

        for comp, (alpha, beta) in enumerate(_AXIAL_PAIRS):
            Aa, Ab = A_k[:, alpha], A_k[:, beta]
            JJpb, JJma = JJp[:, beta], JJm[:, alpha]
            JJmb = JJm[:, beta]

            # ---- -2Im[f] : J0+J1+J2 (identical to wannier_interpolated_curvature) ----
            J0f = torch.einsum('kij,kji->k', f_proj, omega_bar_k[:, comp]).real
            J1f = -2.0 * (
                torch.einsum('kij,kji->k', Aa, JJpb)
                + torch.einsum('kij,kji->k', JJma, Ab)
            ).imag
            J2f = -2.0 * torch.einsum('kij,kji->k', JJma, JJpb).imag
            imf[:, ife, comp] = (J0f + J1f + J2f).cpu().numpy()

            # ---- shared J0 pieces for -2Im[g]/-2Im[h] ----
            HH_Aa = torch.matmul(H0, Aa)
            HH_Ob = torch.matmul(H0, omega_bar_k[:, comp])
            Lambda_ij = 1j * (CC_k[:, alpha, beta] - CC_k[:, alpha, beta].conj().transpose(-1, -2))

            tmp5 = torch.matmul(torch.matmul(HH_Aa, f_proj), Ab)
            s = 2.0 * torch.einsum('kij,kji->k', f_proj, tmp5).imag
            J0g = torch.einsum('kij,kji->k', f_proj, Lambda_ij).real - s
            J0h = torch.einsum('kij,kji->k', f_proj, HH_Ob).real + s

            # ---- J1 pieces ----
            HH_JJma = torch.matmul(H0, JJma)
            J1g = -2.0 * (
                torch.einsum('kij,kji->k', JJma, BB_k[:, beta])
                - torch.einsum('kij,kji->k', JJmb, BB_k[:, alpha])
            ).imag
            J1h = -2.0 * (
                torch.einsum('kij,kji->k', HH_Aa, JJpb)
                + torch.einsum('kij,kji->k', HH_JJma, Ab)
            ).imag

            # ---- J2 pieces ----
            JJma_HH = torch.matmul(JJma, H0)
            J2g = -2.0 * torch.einsum('kij,kji->k', JJma_HH, JJpb).imag
            J2h = -2.0 * torch.einsum('kij,kji->k', HH_JJma, JJpb).imag

            img[:, ife, comp] = (J0g + J1g + J2g).cpu().numpy()
            imh[:, ife, comp] = (J0h + J1h + J2h).cpu().numpy()

    return imf, img, imh


def orbital_magnetization(
    hr: HamiltonianR, AA_R: torch.Tensor, BB_R: torch.Tensor, CC_R: torch.Tensor,
    recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energies, mesh: tuple = (20, 20, 20),
) -> OrbitalMagnetizationResult:
    """
    Intrinsic orbital magnetization vs. Fermi energy, averaged over a
    uniform k-mesh (postw90 `berry_task = morb`; CTVR06/LVTS12).

        M_orb,a(E_F) = -( <img_a> + <imh_a> - 2 E_F <imf_a> )

    with <.> the mesh average and E_F in Hartree; see this module's
    docstring for why no further unit-conversion factor is needed when
    everything (hr, AA_R, BB_R, CC_R) is in Hartree/Bohr, waw's core
    convention.

    Args:
      hr                 : Hartree, core convention
      AA_R               : (3, nR, nw, nw), from `core.hamiltonian.compute_position_r`
      BB_R               : (3, nR, nw, nw), from `core.hamiltonian.compute_bb_r`
      CC_R               : (3, 3, nR, nw, nw), from `core.hamiltonian.compute_cc_r`
                           (all on the SAME R_vectors/degen grid as `hr`)
      recip_lattice, real_lattice : Bohr^-1/Bohr (core convention)
      fermi_energies     : scalar or (nf,) array, Hartree
      mesh               : (N_a, N_b, N_c) uniform k-mesh density

    Returns OrbitalMagnetizationResult (m_orb in Bohr magneton/cell).
    """
    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    nf = len(fermi_energies)
    Na, Nb, Nc = mesh

    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)
    nk = len(kpts)

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.complex128)

    imf_sum = np.zeros((nf, 3))
    img_sum = np.zeros((nf, 3))
    imh_sum = np.zeros((nf, 3))

    for lo in range(0, nk, _KCHUNK):
        kc = kpts[lo:lo + _KCHUNK]
        H0, grad = h_and_grad_frac_batch(hr, kc)
        grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad)

        A_k, omega_bar_k = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)
        BB_k, _ = position_operator_k(BB_R, hr.R_vectors, hr.degen, real_lattice, kc)

        nw = hr.nw
        CC_k = torch.stack([
            torch.stack([
                operator_k(CC_R[a, b], hr.R_vectors, hr.degen, kc) for b in range(3)
            ], dim=1) for a in range(3)
        ], dim=1)   # (nc, 3, 3, nw, nw)

        imf_c, img_c, imh_c = _imfgh_chunk(H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k,
                                          fermi_energies)
        imf_sum += imf_c.sum(axis=0)
        img_sum += img_c.sum(axis=0)
        imh_sum += imh_c.sum(axis=0)

    imf_avg = imf_sum / nk
    img_avg = img_sum / nk
    imh_avg = imh_sum / nk

    m_orb = -(img_avg + imh_avg - 2.0 * fermi_energies[:, None] * imf_avg)

    return OrbitalMagnetizationResult(fermi_energies=fermi_energies, m_orb=m_orb, mesh=mesh)
