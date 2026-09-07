"""
pyscf (PBC, Gaussian basis) -> waw `HamiltonianR` via Löwdin.

The third localized-basis door into waw after Wannierization and SIESTA
NAOs: a periodic pyscf mean-field (KUKS/KUHF/KRKS) provides the Fock
matrix F(k) and overlap S(k) on its k-mesh -- Gaussian AOs are localized
but non-orthogonal, exactly the SIESTA situation, so the same Löwdin
orthogonalization H_orth(k) = S^{-1/2} F S^{-1/2} + Fourier transform to
the mesh's Wigner-Seitz R-set yields a standard `HamiltonianR`.

pyscf works in Hartree/Bohr natively -- no unit conversion anywhere, and
unlike SIESTA's HSX there is NO Fermi-level shift: the model shares the
absolute energy reference of `mo_energy` (take E_F from the smearing
chemical potential).

Same capability map as the SIESTA route: everything H-only (bands, DOS,
Fermi surfaces, surface GF, `analysis.exchange`, `analysis.magnon`)
works unchanged, with per-spin channels sharing the AO basis by
construction; Berry-curvature-class quantities need position matrix
elements (pyscf CAN provide `int1e_r` dipoles -- a future extension,
deliberately not wired up yet).

Caveat mirrored from the SIESTA loader: Löwdin tails make interpolation
between mesh points approximate (exact ON the mesh), and Gaussian
virtual states well above E_F are basis artifacts -- keep analyses in
the physical window.
"""
from __future__ import annotations

import numpy as np
import torch

from ...core.hamiltonian import HamiltonianR, _wigner_seitz


def lowdin_hamiltonian_pyscf(
    fock_k: np.ndarray,
    ovlp_k: np.ndarray,
    kpts_reduced: np.ndarray,
    mp_grid: tuple[int, int, int],
    real_lattice_bohr: np.ndarray,
) -> HamiltonianR:
    """
    Löwdin-orthogonalized `HamiltonianR` from Fock/overlap arrays.

    Args:
      fock_k            : (nk, nao, nao) complex, Hartree -- ONE spin
                          channel of the converged Fock matrix
      ovlp_k            : (nk, nao, nao) complex overlap
      kpts_reduced      : (nk, 3) k-points in crystal (reduced) coords;
                          must be the full uniform `mp_grid` mesh
      mp_grid           : the k-mesh dimensions
      real_lattice_bohr : (3, 3) lattice vectors as rows, Bohr

    Returns HamiltonianR (Hartree, absolute reference of mo_energy).
    """
    fock_k = np.asarray(fock_k)
    ovlp_k = np.asarray(ovlp_k)
    nk, nao = fock_k.shape[0], fock_k.shape[-1]
    assert nk == int(np.prod(mp_grid)), "need the full uniform mesh"

    H_orth = np.zeros((nk, nao, nao), dtype=complex)
    for ik in range(nk):
        w, v = np.linalg.eigh(ovlp_k[ik])
        if w.min() < 1e-10:
            raise ValueError(
                f"overlap (nearly) singular at k index {ik} "
                f"(min eigenvalue {w.min():.2e}): use a less diffuse basis"
            )
        S_inv_sqrt = (v / np.sqrt(w)) @ v.conj().T
        Ho = S_inv_sqrt @ fock_k[ik] @ S_inv_sqrt
        H_orth[ik] = 0.5 * (Ho + Ho.conj().T)

    R_arr, degen = _wigner_seitz(mp_grid, np.asarray(real_lattice_bohr))
    phase = np.exp(-2j * np.pi * (np.asarray(kpts_reduced) @ R_arr.T))
    H_R = np.einsum("kr,kmn->rmn", phase, H_orth) / nk

    return HamiltonianR(
        H_R=torch.tensor(H_R, dtype=torch.complex128),
        R_vectors=R_arr, degen=degen, nw=nao,
    )


def from_kuks(kmf, mp_grid: tuple[int, int, int]):
    """
    Extract per-spin `HamiltonianR` models + E_F from a converged
    spin-polarized periodic pyscf mean-field (KUKS/KUHF).

    Returns (hr_up, hr_dn, efermi_hartree, kpts_reduced).
    """
    cell = kmf.cell
    a = cell.lattice_vectors()                     # Bohr, rows
    kpts_red = cell.get_scaled_kpts(kmf.kpts)
    kpts_red -= np.round(kpts_red)

    dm = kmf.make_rdm1()
    fock = kmf.get_fock(dm=dm)                     # (2, nk, nao, nao)
    ovlp = np.asarray(kmf.get_ovlp())              # (nk, nao, nao)

    mu = None
    for attr in ("mu", "mu0"):
        v = getattr(kmf, attr, None)
        if v is not None:
            mu = float(v)
            break
    if mu is None:
        occ = np.asarray(kmf.mo_occ)               # (2, nk, nao)
        en = np.asarray(kmf.mo_energy)
        mu = 0.5 * (en[occ > 0.5].max() + en[occ < 0.5].min())

    hrs = [lowdin_hamiltonian_pyscf(np.asarray(fock[s]), ovlp, kpts_red,
                                    mp_grid, a) for s in (0, 1)]
    return hrs[0], hrs[1], float(mu), kpts_red
