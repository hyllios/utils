"""
Heisenberg exchange couplings from the magnetic force theorem (LKAG).

Collinear two-spin-channel formulation of Liechtenstein, Katsnelson,
Antropov and Gubanov, J. Magn. Magn. Mater. 67, 65 (1987):

    J_ij(R) = 1/(4 pi) Im Int^{E_F} dE
              Tr_orb [ Delta_i G^up_ij(R, E) Delta_j G^dn_ji(-R, E) ]

where Delta_i = H^up_ii(R=0) - H^dn_ii(R=0) is the on-site spin splitting
of magnetic atom i and G^sigma is the lattice Green's function of each
spin channel.  The energy integral runs over the Ozaki continued-fraction
contour (Ozaki, PRB 75, 035123 (2007)): poles of the Fermi function
resummation at z_p = +- i kT / x_p with Gauss-quadrature-like residues.

Sign convention matches TB2J: Heisenberg energy
E = - sum_{i != j} J_ij e_i . e_j (unit spin vectors, both (i,j) and
(j,i) counted), so J > 0 is ferromagnetic.  Certified against TB2J's
wann2J.py on identical hr.dat input.

All energies in Hartree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waw.core.hamiltonian import HamiltonianR
from waw.core.distributions import fermi_dirac
from waw.units import K_B_HARTREE

KT_600K = 600.0 * K_B_HARTREE


def cfr_contour(nz: int, kT: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Ozaki continued-fraction poles and residue weights of the Fermi
    function (PRB 75, 035123).

    Returns
    -------
    z : (2*npos,) complex — pole positions relative to the Fermi level
        (purely imaginary, +-i kT / x_p pairs)
    w : (2*npos,) complex — quadrature weights such that
        Int^{E_F} dE f(E) g(E)  ~  Im[ -pi/2 * sum_p w_p g(z_p + E_F) ]
        (the -pi/2 is applied by the caller, as in TB2J)
    """
    j = np.arange(nz - 1)
    b = 1.0 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)))
    B = np.diag(b, -1) + np.diag(b, 1)
    poles, vecs = np.linalg.eig(B)
    residues = 0.25 * np.abs(vecs[0, :]) ** 2 / poles**2
    z, w = [], []
    for p, r in zip(poles, residues):
        if p > 0:
            z += [1j * kT / p, -1j * kT / p]
            w += [2j * kT * r, 2j * kT * r]
    return np.array(z), np.array(w)


@dataclass
class ExchangeResult:
    J: dict            # (R, i, j) -> J_iso (Hartree); R a 3-tuple of ints
    J_orb: dict | None  # (R, i, j) -> (ni, nj) orbital-resolved matrix
    magmoms: np.ndarray  # (n_atoms,) z magnetic moment per magnetic atom
    delta: list        # per-atom on-site spin splitting blocks (Hartree)


def _eigensystem(hr: HamiltonianR, kpts: np.ndarray):
    from waw.analysis.elph import band_eigensystem
    return band_eigensystem(hr, kpts)


def heisenberg_exchange(
    hr_up: HamiltonianR,
    hr_dn: HamiltonianR,
    kpts: np.ndarray,
    efermi: float,
    orbital_groups: list[list[int]],
    R_list: np.ndarray,
    nz: int = 100,
    kT: float = KT_600K,
    state_window: tuple[float, float] | None = None,
    orbital_resolved: bool = False,
) -> ExchangeResult:
    """
    Isotropic Heisenberg J_ij(R) between magnetic atoms via LKAG.

    Args:
      hr_up, hr_dn   : real-space Hamiltonians of the two spin channels
                       (same Wannier basis ordering in both — use a
                       projection gauge so up/down orbitals correspond)
      kpts           : (nk, 3) full Monkhorst-Pack mesh, crystal coords
      efermi         : Fermi level (Hartree), common to both channels
      orbital_groups : orbital indices of each magnetic atom, e.g.
                       [[4,5,6,7,8]] for one atom's d block
      R_list         : (nR, 3) int lattice vectors at which to evaluate J
      nz             : number of Ozaki CFR pole pairs
      kT             : contour temperature (Hartree); default 600 K
      state_window   : optional (lo, hi) in Hartree — TB2J-style band
                       truncation of the Green's function: a band survives
                       if ANY of its eigenvalues falls in [lo, hi), all
                       bands above the last survivor are dropped whole,
                       bands below are always kept (TB2J cuts at
                       E_F + 5.1 eV; None = exact, no truncation).

                   *** THIS CAN CHANGE THE SIGN OF J. *** It is the only
                   difference between this and TB2J's default, and on
                   Ag2NdCd it decides the answer. Same H, same E_F, same
                   moment to four digits:

                       group          exact    E_F+5.1    TB2J itself
                       Nd 4f only    -0.392     -0.385         --
                       all Nd NAOs   -0.795     +0.141      +0.179

                   The codes agree once the truncation matches. But an
                   all-orbital group whose sign moves with a technical
                   cutoff is not converged, and `siesta2J --elements X`
                   uses all orbitals, so a ferromagnetic J from TB2J may
                   rest on it. A group restricted to the shell carrying
                   the moment (4f here) was insensitive. Quote both, or
                   state which truncation was used
      orbital_resolved : also return per-orbital J matrices

    Returns ExchangeResult; J values follow E = -sum_{i!=j} J e_i.e_j.
    """
    kpts = np.asarray(kpts, dtype=float)
    nk = len(kpts)
    R_list = np.asarray(R_list, dtype=int)
    z_poles, w_poles = cfr_contour(nz, kT)

    m_flat = [o for g in orbital_groups for o in g]
    slices, off = [], 0
    for g in orbital_groups:
        slices.append(slice(off, off + len(g)))
        off += len(g)

    eig_up, U_up = _eigensystem(hr_up, kpts)
    eig_dn, U_dn = _eigensystem(hr_dn, kpts)

    # on-site spin splitting from the untruncated H(R=0) = <H(k)>_BZ
    def _h0(eig, U):
        return np.einsum("kib,kb,kjb->ij", U, eig, U.conj()) / nk
    H0_up, H0_dn = _h0(eig_up, U_up), _h0(eig_dn, U_dn)
    Delta_full = H0_up - H0_dn
    delta = [Delta_full[np.ix_(g, g)] for g in orbital_groups]

    # z magnetic moments (for the sign convention and reporting)
    occ_up = fermi_dirac(eig_up, efermi, kT)
    occ_dn = fermi_dirac(eig_dn, efermi, kT)
    magmoms = np.array([
        (np.einsum("kb,kib->", occ_up, np.abs(U_up[:, g, :]) ** 2)
         - np.einsum("kb,kib->", occ_dn, np.abs(U_dn[:, g, :]) ** 2)) / nk
        for g in orbital_groups
    ])

    if state_window is not None:
        lo, hi = state_window

        def _band_keep(eig):
            inside = np.where(np.any((eig >= lo) & (eig < hi), axis=0))[0]
            keep = np.zeros(eig.shape, dtype=bool)
            keep[:, : inside[-1] + 1] = True
            return keep
        keep_up, keep_dn = _band_keep(eig_up), _band_keep(eig_dn)
    else:
        keep_up = keep_dn = None

    # all R needed: requested list plus negatives (for G_ji(-R))
    R_all = np.unique(np.vstack([R_list, -R_list]), axis=0)
    R_index = {tuple(R): i for i, R in enumerate(R_all)}
    phase = np.exp(-2j * np.pi * (R_all @ kpts.T)) / nk  # (nR_all, nk)

    U_up_m = np.ascontiguousarray(U_up[:, m_flat, :])
    U_dn_m = np.ascontiguousarray(U_dn[:, m_flat, :])

    n_at = len(orbital_groups)
    acc = {(tuple(R), i, j): 0.0j
           for R in R_list for i in range(n_at) for j in range(n_at)}
    acc_orb = ({k: 0.0j for k in acc} if orbital_resolved else None)

    for z, w in zip(z_poles, w_poles):
        d_up = 1.0 / (z + efermi - eig_up)
        d_dn = 1.0 / (z + efermi - eig_dn)
        if keep_up is not None:
            d_up = np.where(keep_up, d_up, 0.0)
            d_dn = np.where(keep_dn, d_dn, 0.0)
        Gk_up = np.einsum("kib,kb,kjb->kij", U_up_m, d_up, U_up_m.conj())
        Gk_dn = np.einsum("kib,kb,kjb->kij", U_dn_m, d_dn, U_dn_m.conj())
        GR_up = np.einsum("rk,kij->rij", phase, Gk_up)
        GR_dn = np.einsum("rk,kij->rij", phase, Gk_dn)
        for R in R_list:
            iR, iRm = R_index[tuple(R)], R_index[tuple(-R)]
            for i in range(n_at):
                for j in range(n_at):
                    M1 = delta[i] @ GR_up[iR][slices[i], slices[j]]
                    M2 = delta[j] @ GR_dn[iRm][slices[j], slices[i]]
                    t = (M1 * M2.T) * (w / (4.0 * np.pi))
                    key = (tuple(R), i, j)
                    acc[key] += t.sum()
                    if orbital_resolved:
                        acc_orb[key] += t

    J, J_orb = {}, ({} if orbital_resolved else None)
    for key, val in acc.items():
        _, i, j = key
        sgn = np.sign(magmoms[i] * magmoms[j])
        J[key] = float(np.imag(-np.pi / 2.0 * val) / sgn)
        if orbital_resolved:
            J_orb[key] = np.imag(-np.pi / 2.0 * acc_orb[key]) / sgn
    return ExchangeResult(J=J, J_orb=J_orb, magmoms=magmoms, delta=delta)


def heisenberg_exchange_nonortho(
    H_up_k: np.ndarray,
    H_dn_k: np.ndarray,
    S_k: np.ndarray,
    kpts: np.ndarray,
    efermi: float,
    orbital_groups: list[list[int]],
    R_list: np.ndarray,
    nz: int = 100,
    kT: float = KT_600K,
    state_window: tuple[float, float] | None = None,
) -> ExchangeResult:
    """
    LKAG J_ij(R) in a NON-orthogonal localized basis (SIESTA NAOs,
    Gaussian AOs, ...), without Löwdin orthogonalization: the lattice
    Green's function is G(z) = (z S(k) - H(k))^-1, evaluated via the
    generalized eigenproblem H C = S C eps with C^dag S C = 1 (TB2J's
    non-orthogonal route -- `eigen_to_G` with generalized eigenvectors).
    Delta_i is the on-site block of H^up(R=0) - H^dn(R=0) in the RAW
    basis, H(R=0) = <H(k)>_BZ. Magnetic moments are Mulliken
    (rho = C f C^dag S).

    Args mirror `heisenberg_exchange`, but take k-space arrays directly
    (Hartree): H_up_k/H_dn_k/S_k of shape (nk, nb, nb) on the FULL
    uniform mesh `kpts` (nk, 3) crystal coords.

    Certified against TB2J's siesta2J on identical SIESTA data
    (tests/test_analysis_exchange.py::test_siesta_nonortho_parity).
    """
    kpts = np.asarray(kpts, dtype=float)
    nk = len(kpts)
    R_list = np.asarray(R_list, dtype=int)
    z_poles, w_poles = cfr_contour(nz, kT)

    m_flat = [o for g in orbital_groups for o in g]
    slices, off = [], 0
    for g in orbital_groups:
        slices.append(slice(off, off + len(g)))
        off += len(g)

    def _geig(Hk):
        from scipy.linalg import eigh as geigh

        es, cs = [], []
        for ik in range(nk):
            e, c = geigh(Hk[ik], S_k[ik])
            es.append(e); cs.append(c)
        return np.array(es), np.array(cs)

    eig_up, C_up = _geig(np.asarray(H_up_k))
    eig_dn, C_dn = _geig(np.asarray(H_dn_k))

    H0_up = np.asarray(H_up_k).mean(axis=0)
    H0_dn = np.asarray(H_dn_k).mean(axis=0)
    Delta_full = H0_up - H0_dn
    delta = [Delta_full[np.ix_(g, g)] for g in orbital_groups]

    occ_up = fermi_dirac(eig_up, efermi, kT)
    occ_dn = fermi_dirac(eig_dn, efermi, kT)

    def _mulliken(C, occ):
        rho = np.einsum("kpn,kn,kqn->kpq", C, occ, C.conj())
        rho = np.einsum("kpq,kqr->kpr", rho, np.asarray(S_k))
        return rho.mean(axis=0)
    rho_up, rho_dn = _mulliken(C_up, occ_up), _mulliken(C_dn, occ_dn)
    magmoms = np.array([
        float(np.trace(rho_up[np.ix_(g, g)]).real
              - np.trace(rho_dn[np.ix_(g, g)]).real)
        for g in orbital_groups
    ])

    if state_window is not None:
        lo, hi = state_window

        def _band_keep(eig):
            inside = np.where(np.any((eig >= lo) & (eig < hi), axis=0))[0]
            keep = np.zeros(eig.shape, dtype=bool)
            keep[:, : inside[-1] + 1] = True
            return keep
        keep_up, keep_dn = _band_keep(eig_up), _band_keep(eig_dn)
    else:
        keep_up = keep_dn = None

    R_all = np.unique(np.vstack([R_list, -R_list]), axis=0)
    R_index = {tuple(R): i for i, R in enumerate(R_all)}
    phase = np.exp(-2j * np.pi * (R_all @ kpts.T)) / nk

    C_up_m = np.ascontiguousarray(C_up[:, m_flat, :])
    C_dn_m = np.ascontiguousarray(C_dn[:, m_flat, :])

    n_at = len(orbital_groups)
    acc = {(tuple(R), i, j): 0.0j
           for R in R_list for i in range(n_at) for j in range(n_at)}

    for z, w in zip(z_poles, w_poles):
        d_up = 1.0 / (z + efermi - eig_up)
        d_dn = 1.0 / (z + efermi - eig_dn)
        if keep_up is not None:
            d_up = np.where(keep_up, d_up, 0.0)
            d_dn = np.where(keep_dn, d_dn, 0.0)
        Gk_up = np.einsum("kib,kb,kjb->kij", C_up_m, d_up, C_up_m.conj())
        Gk_dn = np.einsum("kib,kb,kjb->kij", C_dn_m, d_dn, C_dn_m.conj())
        GR_up = np.einsum("rk,kij->rij", phase, Gk_up)
        GR_dn = np.einsum("rk,kij->rij", phase, Gk_dn)
        for R in R_list:
            iR, iRm = R_index[tuple(R)], R_index[tuple(-R)]
            for i in range(n_at):
                for j in range(n_at):
                    M1 = delta[i] @ GR_up[iR][slices[i], slices[j]]
                    M2 = delta[j] @ GR_dn[iRm][slices[j], slices[i]]
                    acc[(tuple(R), i, j)] += (M1 * M2.T).sum() * (w / (4.0 * np.pi))

    J = {}
    for key, val in acc.items():
        _, i, j = key
        sgn = np.sign(magmoms[i] * magmoms[j])
        J[key] = float(np.imag(-np.pi / 2.0 * val) / sgn)
    return ExchangeResult(J=J, J_orb=None, magmoms=magmoms, delta=delta)
