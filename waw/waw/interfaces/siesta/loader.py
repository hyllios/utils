"""
SIESTA NAO Hamiltonian -> waw `HamiltonianR` via Löwdin orthogonalization.

SIESTA's numerical atomic orbitals are localized but NON-orthogonal: the
band problem is the generalized H(k) psi = eps S(k) psi. waw's core works
in an orthonormal localized basis, so this loader orthogonalizes on a
uniform k-mesh,

    H_orth(k) = S(k)^{-1/2} H(k) S(k)^{-1/2},

and Fourier-transforms back to R on the mesh's Wigner-Seitz cell (same
R-set/degeneracy convention as `core.hamiltonian.compute_hr`). The
resulting `HamiltonianR` reproduces SIESTA's eigenvalues EXACTLY on the
mesh and interpolates between mesh points like any Wannier model -- the
Löwdin orbitals keep the NAO locality (with slightly longer tails), so
mesh convergence behaves like a same-size Wannier basis.

What this enables with NO Wannierization step: band interpolation, DOS,
Fermi surfaces, surface Green functions/ARPES geometry, and -- because a
collinear run gives one H(R) per spin channel in the SAME orbital basis
by construction (the atomic orbitals know nothing about spin) -- the
LKAG exchange (`analysis.exchange`) and magnon (`analysis.magnon`)
machinery with none of the projection-gauge care Wannier channels need.

What it does NOT give exactly: position matrix elements (AA_R etc.).
NAO codes do not export <phi|r|phi'>, so Berry-curvature-class
quantities (AHC/SHC/orbital moments/SAC) are unavailable; the
atomic-centre diagonal approximation could be added but is deliberately
NOT silently substituted here.

Spin operators ARE exact for spinor (SOC) runs:
<phi_i sigma|s_a|phi_j sigma'> = (sigma_a)_{sigma sigma'} S_ij / 2 --
see `spin_operator_r_nao`.

All returns in Hartree/Bohr atomic units (SIESTA/sisl work in eV/Ang;
converted here at the boundary).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...core.hamiltonian import HamiltonianR, _wigner_seitz
from ...units import EV_TO_HARTREE, ANG_TO_BOHR


def load_hamiltonian(fdf_path):
    """sisl Hamiltonian (with overlap) from a completed SIESTA run's fdf."""
    import sisl

    return sisl.get_sile(str(fdf_path)).read_hamiltonian()


def _full_mesh(mp_grid):
    g = np.meshgrid(*[np.arange(n) for n in mp_grid], indexing="ij")
    k = np.stack([x.ravel() for x in g], axis=1) / np.array(mp_grid, dtype=float)
    return k - np.round(k)


def lowdin_hamiltonian(
    H,
    mp_grid: tuple[int, int, int],
    spin: int | None = None,
    centres: str | None = None,
) -> HamiltonianR:
    """
    Löwdin-orthogonalized `HamiltonianR` from a sisl Hamiltonian.

    Args:
      H       : sisl.Hamiltonian (from `load_hamiltonian`)
      mp_grid : uniform k-mesh for the orthogonalization + FT. Must be
                dense enough that the LÖWDIN H(R) (longer-ranged than the
                bare NAO H) has decayed at the BvK boundary. Convergence
                is energy-resolved: on bcc-Fe/DZP the bands up to
                E_F+2 eV interpolate to 155/35/13 meV at 8/12/16 points
                per axis (exact ON the mesh at any size), while the high
                NAO virtual states (> E_F+10 eV, basis-tail artifacts)
                never interpolate well -- restrict any analysis to the
                physically meaningful energy range, as with any NAO code.
      spin    : 0/1 for the channels of a collinear run, None for
                unpolarized (or a spinor run, where sisl returns the
                full 2x2 spinor matrix and nw = 2*n_orb).
      centres : None (default) leaves `hr.centres` unset, so `ws="auto"`
                interpolation falls back to the plain Wigner-Seitz sum and
                anything needing orbital positions (`analysis.surface`
                termination selection, `analysis.floquet`) has to be given
                them explicitly. "atomic" fills them from
                `orbital_centres` -- each NAO sits on its own atom, which is
                exact for this basis (see there), unlike the atomic-centre
                approximation to the off-diagonal <phi|r|phi'> that this
                loader still refuses to fabricate. Opt-in rather than the
                default only because it changes what `ws="auto"` does.

    Returns HamiltonianR (Hartree), nw = number of NAOs (x2 for spinor),
    carrying `real_lattice` and `mp_grid` (so `hopping_range`,
    `analysis.surface` and `ws="auto"` all work without being handed the cell
    again).

    ENERGY REFERENCE: SIESTA's saved H (HSX/TSHS, both read by sisl) stores H - E_F S
    (the TranSIESTA convention), so the returned model has its Fermi level
    at EXACTLY 0 -- do NOT combine it with `read_fermi_level` (that is
    the .EIG file's own reference, shifted by the same E_F).
    """
    if centres not in (None, "atomic"):
        raise ValueError(f"centres must be None or 'atomic', got {centres!r}")
    kpts = _full_mesh(mp_grid)
    nk = len(kpts)
    kw = {} if spin is None else {"spin": spin}

    Hk0 = np.asarray(H.Hk(kpts[0], format="array", **kw))
    nw = Hk0.shape[0]
    H_orth = np.zeros((nk, nw, nw), dtype=complex)
    for ik, k in enumerate(kpts):
        Hk = np.asarray(H.Hk(k, format="array", **kw))
        Sk = np.asarray(H.Sk(k, format="array"))
        if Sk.shape[0] != nw:      # spinor H with spin-diagonal S
            Sk = np.kron(Sk, np.eye(2))
        w, v = np.linalg.eigh(Sk)
        if w.min() < 1e-10:
            raise ValueError(
                f"overlap matrix (nearly) singular at k={k} "
                f"(min eigenvalue {w.min():.2e}): basis linearly dependent"
            )
        S_inv_sqrt = (v / np.sqrt(w)) @ v.conj().T
        Ho = S_inv_sqrt @ Hk @ S_inv_sqrt
        H_orth[ik] = 0.5 * (Ho + Ho.conj().T)

    real_lattice = np.asarray(H.geometry.lattice.cell) * ANG_TO_BOHR
    R_arr, degen = _wigner_seitz(mp_grid, real_lattice)
    phase = np.exp(-2j * np.pi * (kpts @ R_arr.T))          # (nk, nR)
    H_R = np.einsum("kr,kmn->rmn", phase, H_orth) / nk * EV_TO_HARTREE

    return HamiltonianR(
        H_R=torch.tensor(H_R, dtype=torch.complex128),
        R_vectors=R_arr, degen=degen, nw=nw,
        centres=orbital_centres(H, nw) if centres == "atomic" else None,
        real_lattice=real_lattice, mp_grid=tuple(mp_grid),
    )


def orbital_centres(H, nw=None) -> np.ndarray:
    """(nw, 3) Cartesian centres in Bohr, one per NAO: its own atom's position.

    Exact for the NAOs and for the Löwdin orbitals built from them -- symmetric
    orthogonalization mixes in neighbour tails symmetrically, so the centre of
    mass does not move off the site. It is NOT a substitute for <phi|r|phi'>:
    the off-diagonal position matrix elements a Berry-curvature calculation
    needs are still unavailable from a NAO code.
    """
    xyz = np.asarray(H.geometry.xyz) * ANG_TO_BOHR
    cen = np.array([xyz[H.geometry.o2a(i)] for i in range(H.geometry.no)])
    if nw is not None and nw == 2 * len(cen):        # spinor: two per orbital
        cen = np.repeat(cen, 2, axis=0)
    if nw is not None and len(cen) != nw:
        raise ValueError(f"{len(cen)} orbital centres for nw={nw}")
    return cen


def spin_operator_r_nao(H, mp_grid: tuple[int, int, int]) -> torch.Tensor:
    """
    Exact spin operator SS_R for a SIESTA spinor (SOC) run, in the SAME
    Löwdin gauge as `lowdin_hamiltonian`: s_a(k) = sigma_a (x) S(k)/2
    rotated by S^{-1/2} on both sides (which makes it simply
    sigma_a (x) I/2 -- the Löwdin basis is orthonormal, so spin is purely
    the spinor index), then FT to the same WS R-set.

    Returns (nR, nw, nw, 3) complex, Hartree-free (spin in hbar=1 units),
    the layout `analysis.spin_texture` consumers expect.
    """
    kpts = _full_mesh(mp_grid)
    n_orb = np.asarray(H.Sk(kpts[0], format="array")).shape[0]
    nw = 2 * n_orb
    real_lattice = np.asarray(H.geometry.lattice.cell) * ANG_TO_BOHR
    R_arr, degen = _wigner_seitz(mp_grid, real_lattice)

    pauli = np.array([[[0, 1], [1, 0]],
                      [[0, -1j], [1j, 0]],
                      [[1, 0], [0, -1]]])
    SS = np.zeros((len(R_arr), nw, nw, 3), dtype=complex)
    iR0 = int(np.where((R_arr == 0).all(axis=1))[0][0])
    for a in range(3):
        SS[iR0, :, :, a] = 0.5 * np.kron(np.eye(n_orb), pauli[a])
    return torch.tensor(SS, dtype=torch.complex128)
