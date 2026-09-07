"""
Ballistic quantum transport (Wannier90's "bulk" transport mode,
`transport=true`): Landauer conductance and density of states for a
periodic Wannier tight-binding chain, via the Lopez-Sancho/Lopez-Sancho/
Rubio (1984, 1985) iterative surface Green's-function decimation.

Atomic units throughout (Hartree, Bohr); convert at the caller when
comparing against Wannier90's eV-based `_qc.dat`/`_dos.dat` output.

"Bulk" mode treats the whole periodic material as its own two
semi-infinite leads: a principal layer (num_pl unit cells, chosen so
hopping beyond it is negligible) is embedded between self-energies built
from that same principal-layer Hamiltonian, and the Landauer/Caroli
formula T(E) = Tr[Gamma_L G^r Gamma_R G^a] gives the transmission (in
units of the conductance quantum; this returns the raw trace, matching
`_qc.dat`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR

_ETA = 1.8374661e-5   # Hartree == Wannier90's eta = 5e-4 eV broadening
_NTERX = 50           # max Sancho-Rubio decimation iterations
_EPS7 = 1e-7          # convergence threshold on the decimation


@dataclass
class TransportResult:
    """Bulk transport output on a uniform energy grid (E - E_F, Hartree)."""
    energies:     np.ndarray   # (n_e,)  Hartree, relative to fermi_energy
    transmission: np.ndarray   # (n_e,)  dimensionless (units of 2e^2/h)
    dos:          np.ndarray   # (n_e,)  states/Hartree (per principal layer)
    num_pl:       int          # unit cells per principal layer
    num_bb:       int          # = num_pl * nw, the block Hamiltonian size


def _one_dim_lattice_index(real_lattice: np.ndarray, axis: int, atol: float = 1e-8) -> int:
    """
    Which lattice-vector row (0,1,2) of ``real_lattice`` is parallel to
    the given cartesian axis (0=x,1=y,2=z) -- i.e. its full length is
    entirely along that one component.
    """
    for i in range(3):
        length = np.linalg.norm(real_lattice[i])
        if abs(abs(real_lattice[i, axis]) - length) < atol:
            return i
    raise ValueError(f"No lattice vector found parallel to cartesian axis {axis}")


def _extend_hr_one_dim(H_R: np.ndarray, R_1d: np.ndarray, mp_grid_1d: int) -> tuple[np.ndarray, int]:
    """
    Build the 1D-chain H(R) with one periodic-image "buffer" shell on each
    side (`irvec_max = max(|R|) + 1`): H(R) is periodic with period
    `mp_grid_1d`, so the buffer shell is the alias of the opposite edge of
    the Wigner-Seitz range, needed so the later per-pair distance cutoff
    can pick whichever alias gives the smaller separation in the
    semi-infinite (non-periodic) chain.

    Args:
      H_R   : (nR, nw, nw) complex H(R) (Hartree); R_1d gives each slice's
              transport-axis R-component (other two components must be 0).
              Kept fully complex rather than real-truncated: transmission/
              DOS are gauge-invariant, so the true complex H(R) works
              regardless of which gauge the optimizer lands on.
      R_1d:   (nR,) int, R-vector's transport-axis component
      mp_grid_1d: number of k-points along the transport axis

    Returns:
      hr_ext: (2*irvec_max+1, nw, nw) complex, indexed by n1+irvec_max
      irvec_max: int
    """
    r_max = int(np.abs(R_1d).max())
    irvec_max = r_max + 1
    lookup = {int(r): i for i, r in enumerate(R_1d)}
    nw = H_R.shape[-1]
    hr_ext = np.zeros((2 * irvec_max + 1, nw, nw), dtype=np.complex128)
    for n1 in range(-irvec_max, irvec_max + 1):
        r = ((n1 + r_max) % mp_grid_1d) - r_max
        hr_ext[n1 + irvec_max] = H_R[lookup[r]]
    return hr_ext, irvec_max


def _apply_one_dim_cutoff(
    hr_ext: np.ndarray, irvec_max: int, centres_1d: np.ndarray,
    lattice_length: float, dist_cutoff: float,
) -> np.ndarray:
    """
    Zero out hr_ext[n1][j, i] whenever the transport-axis separation
    |centres_1d[i] - centres_1d[j] + n1*lattice_length| between WF i at
    cell n1 and WF j at cell 0 exceeds dist_cutoff (Wannier90's
    dist_cutoff_mode='one_dim').
    """
    hr_ext = hr_ext.copy()
    nw = hr_ext.shape[-1]
    dist_ij = centres_1d[:, None] - centres_1d[None, :]           # (nw_i, nw_j)
    for n1 in range(-irvec_max, irvec_max + 1):
        dist = np.abs(dist_ij + n1 * lattice_length)              # (nw_i, nw_j)
        mask = dist.T > dist_cutoff                                # -> [j, i] layout
        hr_ext[n1 + irvec_max][mask] = 0.0
    return hr_ext


def _wrap_home_cell(
    centres: np.ndarray, real_lattice: np.ndarray,
    translation_centre_frac: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    Wrap Wannier centres into the home-cell window centred at
    ``translation_centre_frac`` (fractional coords), mirroring Wannier90's
    ``translate_home_cell``.

    A centre reported outside the home cell (a benign gauge artifact) can
    make the one-dim distance cutoff pick the wrong periodic alias,
    changing which R-shells survive and hence the principal-layer size
    num_pl (transmission/DOS still come out right either way; an
    unwrapped principal layer is just less minimal).
    """
    inv_lattice = np.linalg.inv(real_lattice)
    r_frac = centres @ inv_lattice
    r_frac_min = np.asarray(translation_centre_frac) - 0.5
    shift = -np.floor(r_frac - r_frac_min)
    return (r_frac + shift) @ real_lattice


def _principal_layer_size(hr_ext: np.ndarray, irvec_max: int, hr_cutoff: float) -> int:
    """
    Largest |n1| for which max_ij |hr_ext[n1]| still exceeds hr_cutoff
    (after the distance cutoff); shells beyond that are zeroed.
    """
    num_pl = 0
    for n1 in range(-irvec_max, irvec_max + 1):
        if np.abs(hr_ext[n1 + irvec_max]).max() > hr_cutoff:
            num_pl = max(num_pl, abs(n1))
        else:
            hr_ext[n1 + irvec_max] = 0.0
    return num_pl


def _build_block_hamiltonians(hr_ext: np.ndarray, irvec_max: int, num_pl: int, nw: int):
    """
    Stack num_pl unit cells into the principal-layer on-site block hB0 and
    the hopping-to-the-next-principal-layer block hB1 (both
    (num_pl*nw, num_pl*nw)):
    hB0[block j, block i] = H(R = i - j); hB1[block j-1, block i]
    = H(R = i - (j-1) + num_pl).
    """

    def _h(n1):
        return hr_ext[n1 + irvec_max]

    nb = num_pl * nw
    hB0 = np.zeros((nb, nb), dtype=np.complex128)
    hB1 = np.zeros((nb, nb), dtype=np.complex128)
    for j in range(num_pl):
        for i in range(num_pl):
            hB0[j * nw:(j + 1) * nw, i * nw:(i + 1) * nw] = _h(i - j)
    for j in range(1, num_pl + 1):
        for i in range(j):
            hB1[(j - 1) * nw:j * nw, i * nw:(i + 1) * nw] = _h(i - (j - 1) + num_pl)
    return hB0, hB1


def _sancho_rubio(H00: np.ndarray, H01: np.ndarray, z: complex,
                   nterx: int = _NTERX, eps: float = _EPS7) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterative transfer-matrix decimation (Lopez Sancho, Lopez Sancho &
    Rubio, J. Phys. F 14, 1205 (1984); ibid. 15, 851 (1985)).

    Returns (T, Ttilde) such that the semi-infinite-lead self-energies are
    Sigma_R = H01 @ T, Sigma_L = H01^H @ Ttilde.
    """
    n = H00.shape[0]
    I = np.eye(n, dtype=np.complex128)
    g0 = np.linalg.inv(z * I - H00)
    tau = g0 @ H01.conj().T
    taut = g0 @ H01

    T = tau.copy()
    Tt = taut.copy()
    tsum = taut.copy()
    tsumt = tau.copy()

    for _ in range(nterx):
        s1 = I - tau @ taut - taut @ tau
        s2 = np.linalg.inv(s1)
        tau2 = s2 @ (tau @ tau)
        taut2 = s2 @ (taut @ taut)

        T = T + tsum @ tau2
        tsum = tsum @ taut2

        Tt = Tt + tsumt @ taut2
        tsumt = tsumt @ tau2

        tau, taut = tau2, taut2

        if (np.abs(tau2).sum() < eps) and (np.abs(taut2).sum() < eps):
            break

    return T, Tt


def transport_bulk(
    hr:              HamiltonianR,
    real_lattice:    np.ndarray,
    centres:         np.ndarray,
    mp_grid:         tuple[int, int, int],
    one_dim_axis:    str,
    dist_cutoff:     float,
    fermi_energy:    float,
    energy_window:   tuple[float, float],
    energy_step:     float,
    hr_cutoff:       float = 0.0,
    translate_home_cell: bool = False,
    translation_centre_frac: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> TransportResult:
    """
    Bulk-mode ballistic transport: Landauer transmission T(E) and DOS(E)
    for the periodic Wannier tight-binding chain along ``one_dim_axis``.

    Args:
      hr            : HamiltonianR (H(R) in Hartree; only the direction with
                       mp_grid[axis] > 1 may carry nonzero R-components --
                       the other two mp_grid entries must be 1).
      real_lattice  : (3, 3) rows, Bohr.
      centres       : (nw, 3) Wannier centres, Bohr.  Used as-is unless
                       ``translate_home_cell=True``.
      mp_grid       : Monkhorst-Pack grid used to build ``hr``.
      one_dim_axis  : 'x', 'y', or 'z' -- the cartesian transport direction.
      dist_cutoff   : Bohr. Hopping between WFs separated by more than
                       this along the transport axis is discarded
                       (Wannier90's dist_cutoff_mode='one_dim'; note W90's
                       own keyword is in Angstrom -- convert at the caller).
      fermi_energy  : Hartree. hB0's diagonal is shifted by -fermi_energy,
                       so the returned ``energies`` are E - fermi_energy.
      energy_window : (E_min, E_max) Hartree, relative to fermi_energy
                       (matching .win's tran_win_min/tran_win_max, which
                       are eV -- convert at the caller).
      energy_step   : Hartree.
      hr_cutoff     : Hartree. Matrix elements below this (after the
                       distance cutoff) are treated as zero; also used to
                       trim the principal-layer size. 0.0 (W90's default)
                       disables it.
      translate_home_cell : Wannier90's `translate_home_cell` (default
                       False). When True, wrap centres into the home-cell
                       window centred at `translation_centre_frac` before
                       the distance cutoff (see `_wrap_home_cell`) --
                       needed whenever centres can legitimately sit
                       outside a single cell in the reported gauge (e.g.
                       after `guiding_centres`), so the cutoff picks the
                       same periodic alias Wannier90 does.
      translation_centre_frac : fractional coords the home-cell window is
                       centred on when `translate_home_cell=True` (default
                       (0.5,0.5,0.5), matching Wannier90's own default).

    Returns:
      TransportResult
    """
    axis = {"x": 0, "y": 1, "z": 2}[one_dim_axis.lower()]
    lat_idx = _one_dim_lattice_index(real_lattice, axis)
    mp_grid_1d = mp_grid[lat_idx]
    lattice_length = np.linalg.norm(real_lattice[lat_idx])

    H_R = hr.H_R.detach().cpu().numpy()   # Hartree, like everything here
    R_1d = hr.R_vectors[:, lat_idx]
    nw = hr.nw

    if translate_home_cell:
        centres = _wrap_home_cell(centres, real_lattice, translation_centre_frac)

    hr_ext, irvec_max = _extend_hr_one_dim(H_R, R_1d, mp_grid_1d)

    hr_ext = _apply_one_dim_cutoff(
        hr_ext, irvec_max, centres[:, axis], lattice_length, dist_cutoff,
    )
    num_pl = _principal_layer_size(hr_ext, irvec_max, hr_cutoff)

    hB0, hB1 = _build_block_hamiltonians(hr_ext, irvec_max, num_pl, nw)
    num_bb = hB0.shape[0]

    for i in range(num_bb):
        hB0[i, i] -= fermi_energy

    e_min, e_max = energy_window
    # The small epsilon keeps the point count stable when the caller's
    # window/step were converted from eV (floor((6.5-(-6.5))/0.01) can land
    # on 1299.9999... after a unit conversion and silently drop a point).
    n_e = int(np.floor((e_max - e_min) / energy_step + 1e-8)) + 1
    energies = e_min + np.arange(n_e) * energy_step

    transmission = np.empty(n_e)
    dos = np.empty(n_e)
    I = np.eye(num_bb, dtype=np.complex128)

    for k, E in enumerate(energies):
        z = E + 1j * _ETA
        T, Tt = _sancho_rubio(hB0, hB1, z)

        SigmaL = hB1.conj().T @ Tt
        SigmaR = hB1 @ T
        GammaL = 1j * (SigmaL - SigmaL.conj().T)
        GammaR = 1j * (SigmaR - SigmaR.conj().T)

        g_B = np.linalg.inv(E * I - hB0 - SigmaL - SigmaR)

        transmission[k] = np.trace(GammaL @ g_B @ GammaR @ g_B.conj().T).real
        dos[k] = -np.trace(g_B).imag / np.pi

    return TransportResult(
        energies=energies, transmission=transmission, dos=dos,
        num_pl=num_pl, num_bb=num_bb,
    )


# ---------------------------------------------------------------------------
# Lead-conductor-lead (LCR) transport
# ---------------------------------------------------------------------------

@dataclass
class LCRTransportResult:
    """Lead-conductor-lead transport output on a uniform energy grid (E - E_F, Hartree)."""
    energies:     np.ndarray   # (n_e,)  Hartree, relative to fermi_energy
    transmission: np.ndarray   # (n_e,)  dimensionless (units of 2e^2/h)
    dos:          np.ndarray   # (n_e,)  states/Hartree (conductor region only)
    num_cc:       int          # number of conductor WFs


def _build_lead_left(H0: np.ndarray, sorted_idx: np.ndarray, num_wann: int,
                      num_ll: int, num_cell_ll: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Idealized (symmetrized) left-lead on-site/hopping blocks hL0/hL1, built
    from the num_ll leftmost WFs (in position-sorted order) by copying the
    couplings measured between the num_cell_ll sub-cells inside that
    reference block (Wannier90's `tran_lcr_2c2_build_ham`).
    """
    ncell = num_ll // num_cell_ll
    hL0 = np.zeros((num_ll, num_ll), dtype=np.complex128)
    hL1 = np.zeros((num_ll, num_ll), dtype=np.complex128)

    for i in range(1, num_cell_ll + 1):
        sub_block = np.zeros((ncell, ncell), dtype=np.complex128)
        for j in range(ncell):
            for k in range(ncell):
                row = sorted_idx[j]
                col = sorted_idx[(i - 1) * ncell + k]
                sub_block[j, k] = H0[row, col]

        for j in range(1, num_cell_ll - i + 2):
            r0 = (j - 1) * ncell
            c0 = (j - 1) * ncell + (i - 1) * ncell
            hL0[r0:r0 + ncell, c0:c0 + ncell] = sub_block
            if i > 1:
                hL0[c0:c0 + ncell, r0:r0 + ncell] = sub_block.conj().T

        if i > 1:
            for j in range(1, i):
                r0 = (num_cell_ll - (i - j)) * ncell
                c0 = (j - 1) * ncell
                hL1[r0:r0 + ncell, c0:c0 + ncell] = sub_block

        if i == 1:
            sub_block2 = np.zeros((ncell, ncell), dtype=np.complex128)
            for j in range(ncell):
                for k in range(ncell):
                    row = sorted_idx[num_wann - num_ll + j]
                    col = sorted_idx[(i - 1) * ncell + k]
                    sub_block2[j, k] = H0[row, col]
            for j in range(1, num_cell_ll - i + 2):
                r0 = (j - 1) * ncell
                c0 = (j - 1) * ncell + (i - 1) * ncell
                hL1[r0:r0 + ncell, c0:c0 + ncell] = sub_block2

    if num_cell_ll == 1:
        for j in range(num_wann - ncell, num_wann):
            for k in range(ncell):
                hL1[j - num_wann + ncell, k] = H0[sorted_idx[j], sorted_idx[k]]

    return hL0, hL1


def _build_lead_right(H0: np.ndarray, sorted_idx: np.ndarray, num_wann: int,
                       num_ll: int, num_cell_ll: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-lead counterpart of `_build_lead_left` (see there)."""
    ncell = num_ll // num_cell_ll
    hR0 = np.zeros((num_ll, num_ll), dtype=np.complex128)
    hR1 = np.zeros((num_ll, num_ll), dtype=np.complex128)

    for i in range(1, num_cell_ll + 1):
        sub_block = np.zeros((ncell, ncell), dtype=np.complex128)
        for j in range(ncell):
            for k in range(ncell):
                row = sorted_idx[num_wann - i * ncell + j]
                col = sorted_idx[num_wann - ncell + k]
                sub_block[j, k] = H0[row, col]

        for j in range(1, num_cell_ll - i + 2):
            r0 = (j - 1) * ncell
            c0 = (j - 1) * ncell + (i - 1) * ncell
            hR0[r0:r0 + ncell, c0:c0 + ncell] = sub_block
            if i > 1:
                hR0[c0:c0 + ncell, r0:r0 + ncell] = sub_block.conj().T

        if i > 1:
            for j in range(1, i):
                r0 = (num_cell_ll - (i - j)) * ncell
                c0 = (j - 1) * ncell
                hR1[r0:r0 + ncell, c0:c0 + ncell] = sub_block

        if i == 1:
            sub_block2 = np.zeros((ncell, ncell), dtype=np.complex128)
            for j in range(ncell):
                for k in range(ncell):
                    row = sorted_idx[(i - 1) * ncell + k]
                    col = sorted_idx[num_wann - num_ll + j]
                    sub_block2[j, k] = H0[row, col]
            for j in range(1, num_cell_ll - i + 2):
                r0 = (j - 1) * ncell
                c0 = (j - 1) * ncell + (i - 1) * ncell
                hR1[r0:r0 + ncell, c0:c0 + ncell] = sub_block2

    if num_cell_ll == 1:
        for j in range(ncell):
            for k in range(num_wann - ncell, num_wann):
                hR1[k - num_wann + ncell, j] = H0[sorted_idx[j], sorted_idx[k]]

    return hR0, hR1


def _smooth_chain_gauge(H0: np.ndarray, sorted_idx: np.ndarray) -> np.ndarray:
    """
    Fix the arbitrary per-WF phases so nearest-neighbour hoppings along the
    position-sorted chain are real and negative (all-positive-s-orbital
    convention), via a diagonal gauge H -> D^dagger H D with |D_nn| = 1
    (the role Wannier90's `tran_parity_enforce` plays for real WFs).

    Needed because the idealized-lead construction copies cell-to-cell
    sub-blocks along block diagonals, which only represents the next cell
    correctly if the WF phases repeat from cell to cell; enforcing
    real-negative nearest-neighbour hoppings gives that periodicity for a
    chain with one WF per site.
    """
    nw = H0.shape[0]
    d = np.ones(nw, dtype=np.complex128)
    for j in range(nw - 1):
        a, b = sorted_idx[j], sorted_idx[j + 1]
        h_ab = np.conj(d[a]) * H0[a, b]          # current gauge-rotated hopping
        if abs(h_ab) < 1e-10:
            d[b] = d[a]                           # broken chain: keep phase
        else:
            d[b] = -abs(h_ab) / h_ab              # make conj(d_a) H_ab d_b = -|H_ab|
    return np.conj(d[:, None]) * H0 * d[None, :]


def transport_lcr(
    hr:              HamiltonianR,
    real_lattice:    np.ndarray,
    centres:         np.ndarray,
    one_dim_axis:    str,
    dist_cutoff:     float,
    num_ll:          int,
    num_cell_ll:     int,
    fermi_energy:    float,
    energy_window:   tuple[float, float],
    energy_step:     float,
    use_same_lead:   bool = True,
    translation_centre_frac: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> LCRTransportResult:
    """
    Lead-conductor-lead (LCR, '2c2') ballistic transport: Landauer
    transmission T(E) and conductor DOS(E), for a finite (typically
    Gamma-only) Wannierized structure whose two ends are identical (or a
    matched pair of) periodic leads around a central conductor region
    (e.g. a defect embedded in an otherwise-periodic chain).

    Reproduces Wannier90's `tran_lcr_2c2_build_ham` / `tran_lcr`.
    Wannier90 identifies which WFs belong to the left lead / conductor /
    right lead via a multipole "signature" match against a `.unkg` file,
    needed when a cell holds several inequivalent orbitals per atom in
    arbitrary order; this implementation instead sorts WFs directly by
    position along `one_dim_axis`, exact and simpler for one Wannier
    function per atom in a simple chain, but not a substitute for the
    signature method on more complex multi-orbital geometries.

    Args:
      hr           : HamiltonianR. Only H(R=0) is used (LCR mode assumes a
                     single, typically Gamma-only k-point -- the finite
                     structure's own supercell periodicity plays no role).
      centres      : (nw, 3) Wannier centres, Bohr.
      one_dim_axis : 'x', 'y', or 'z' -- the cartesian transport direction.
      dist_cutoff  : Bohr. Hopping between WFs separated by more than this
                     along the transport axis is discarded (with the
                     PBC-aliased second chance below; W90's own keyword is
                     Angstrom -- convert at the caller).
      num_ll       : number of WFs in the left-lead reference block
                     (Wannier90's `tran_num_ll`).
      num_cell_ll  : number of repeat unit cells inside that reference
                     block (Wannier90's `tran_num_cell_ll`); num_ll must be
                     a multiple of this.
      fermi_energy : Hartree. On-site blocks are shifted by -fermi_energy,
                     so the returned `energies` are E - fermi_energy.
      energy_window: (E_min, E_max) Hartree, relative to fermi_energy.
      energy_step  : Hartree.
      use_same_lead: if True (default, matching a symmetric structure),
                     build the right lead's surface Green's function from
                     the same transfer matrices as the left lead instead
                     of recomputing them from a separate hR0/hR1 (only
                     valid when both leads are physically identical).

    Returns:
      LCRTransportResult
    """
    axis = {"x": 0, "y": 1, "z": 2}[one_dim_axis.lower()]
    nw = hr.nw

    H0 = hr.H_R[hr.R_vectors.tolist().index([0, 0, 0])].detach().cpu().numpy()   # Hartree

    lat_idx = _one_dim_lattice_index(real_lattice, axis)
    lattice_length = np.linalg.norm(real_lattice[lat_idx])

    # LCR always sorts home-cell-translated centres: the supercell is a
    # periodic ring, and where the coordinate window cuts it decides which
    # cells the idealized leads are built from. Untranslated centres could
    # cut the ring at the defect, building the "ideal lead" out of defect cells.
    centres = _wrap_home_cell(centres, real_lattice, translation_centre_frac)

    centres_1d = centres[:, axis]
    dx = np.abs(centres_1d[:, None] - centres_1d[None, :])
    # An element also survives if within dist_cutoff under periodic
    # boundary conditions (+- one supercell lattice vector): these
    # wrap-around couplings fill the leads' hL1/hR1 inter-principal-layer
    # blocks; a plain |dx| cutoff would zero them all.
    dist = np.minimum(dx, np.abs(dx - lattice_length))
    H0 = H0.copy()
    H0[dist > dist_cutoff] = 0.0

    sorted_idx = np.argsort(centres_1d)

    # Phase-align the WFs along the chain before any cell-to-cell block
    # copying (Wannier90's tran_parity_enforce; see _smooth_chain_gauge).
    H0 = _smooth_chain_gauge(H0, sorted_idx)

    hL0, hL1 = _build_lead_left(H0, sorted_idx, nw, num_ll, num_cell_ll)
    hLC = np.zeros((num_ll, num_ll), dtype=np.complex128)
    for i in range(num_ll):
        for j in range(num_ll):
            hLC[i, j] = H0[sorted_idx[i], sorted_idx[num_ll + j]]

    if use_same_lead:
        hR0 = hR1 = None
    else:
        hR0, hR1 = _build_lead_right(H0, sorted_idx, nw, num_ll, num_cell_ll)

    hCR = np.zeros((num_ll, num_ll), dtype=np.complex128)
    for i in range(num_ll):
        for j in range(num_ll):
            row = sorted_idx[nw - 2 * num_ll + i]
            col = sorted_idx[nw - num_ll + j]
            hCR[i, j] = H0[row, col]

    num_cc = nw - 2 * num_ll
    hC = np.zeros((num_cc, num_cc), dtype=np.complex128)
    for i in range(num_cc):
        for j in range(num_cc):
            row = sorted_idx[num_ll + i]
            col = sorted_idx[num_ll + j]
            hC[i, j] = H0[row, col]
    # W90 zeroes conductor elements below 1e-4 eV (10*eps5 in
    # tran_lcr_2c2_build_ham) to band the conductor matrix; replicated for
    # numerical fidelity even though we solve dense (3.675e-6 Ha == 1e-4 eV).
    hC[np.abs(hC) < 3.675e-6] = 0.0

    for i in range(num_ll):
        hL0[i, i] -= fermi_energy
        if hR0 is not None:
            hR0[i, i] -= fermi_energy
    for i in range(num_cc):
        hC[i, i] -= fermi_energy

    e_min, e_max = energy_window
    # epsilon guard: see transport_bulk's energy-grid comment
    n_e = int(np.floor((e_max - e_min) / energy_step + 1e-8)) + 1
    energies = e_min + np.arange(n_e) * energy_step

    Icc = np.eye(num_cc, dtype=np.complex128)

    transmission = np.empty(n_e)
    dos = np.empty(n_e)

    for k, E in enumerate(energies):
        z = E + 1j * _ETA
        totL, tottL = _sancho_rubio(hL0, hL1, z)
        g_surf_L = np.linalg.inv(E * np.eye(num_ll) - hL0 - hL1.conj().T @ tottL)

        sLr = hLC.conj().T @ g_surf_L @ hLC

        if use_same_lead:
            g_surf_R = np.linalg.inv(E * np.eye(num_ll) - hL0 - hL1 @ totL)
        else:
            totR, tottR = _sancho_rubio(hR0, hR1, z)
            g_surf_R = np.linalg.inv(E * np.eye(num_ll) - hR0 - hR1 @ totR)

        sRr = hCR @ g_surf_R @ hCR.conj().T

        Sigma = np.zeros((num_cc, num_cc), dtype=np.complex128)
        Sigma[:num_ll, :num_ll] += sLr
        Sigma[num_cc - num_ll:, num_cc - num_ll:] += sRr

        g_C = np.linalg.inv(E * Icc - hC - Sigma)

        GammaL = 1j * (sLr - sLr.conj().T)
        GammaR = 1j * (sRr - sRr.conj().T)

        g_LR = g_C[:num_ll, num_cc - num_ll:]
        transmission[k] = np.trace(GammaL @ g_LR @ GammaR @ g_LR.conj().T).real
        dos[k] = -np.trace(g_C).imag / np.pi

    return LCRTransportResult(
        energies=energies, transmission=transmission, dos=dos, num_cc=num_cc,
    )
