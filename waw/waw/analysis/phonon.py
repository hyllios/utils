"""
Phonon band structure via Fourier interpolation of real-space interatomic
force constants (`ph.x` DFPT + `q2r.x`, `interfaces.quantum_espresso.
phonon_io.read_force_constants`) -- the lattice-dynamics analogue of this
project's own electronic Wannier-Hamiltonian interpolation.

The analogy is exact and is exploited directly: a force-constant matrix
C(R)_{ipol,na;jpol,nb} (3 Cartesian components x nat atoms, i.e. 3*nat
"orbitals") Fourier-interpolates to an arbitrary q-point via precisely the
same real-space-sum machinery `core.hamiltonian` already implements for
H(R) -> H(k):

  - `core.hamiltonian._wigner_seitz` reduces the q2r.x coarse-grid R-vectors
    to a Wigner-Seitz-minimal set with plain degeneracies (same call, same
    `mp_grid` = the q2r.x `(nr1,nr2,nr3)` real-space grid).
  - `core.ws_distance.build_ws_distance` then does the SAME atom-position-
    aware refinement wannier90's `use_ws_distance` does for Wannier
    centres, here with the crystal's own atomic positions playing that
    role -- confirmed to be the exact same algorithm as QE's own
    `PHonon/PH/matdyn.f90::frc_blk`/`wsweight` (wide supercell search +
    minimum-image degeneracy averaging using R + tau_na - tau_nb, phase
    using the bare lattice R) by reading that source directly.

Combined index convention throughout: `I = na*3 + ipol` (atom-major, then
Cartesian x/y/z) for the 3*nat "orbital" dimension -- so a mode's
atom-projected weight is a plain reshape + sum, no separate bookkeeping.

Units: force constants enter in Hartree/Bohr^2 (already converted from
the file's Ry/Bohr^2 by the reader) and masses in physical amu; frequencies
come out directly in cm^-1 (with the standard phonon-code sign convention:
negative cm^-1 for an unstable/imaginary mode, omega = sign(omega^2)*
sqrt(|omega^2|)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch

from ..core.hamiltonian import _wigner_seitz, HamiltonianR
from ..core.ws_distance import build_ws_distance
from ..core.distributions import gaussian_smearing
from ..units import HARTREE_TO_J, BOHR_RADIUS_M, AMU_TO_KG

from ..units import C_LIGHT_CM_PER_S as _C_CM_PER_S


@dataclass
class PhononBands:
    """
    Phonon band structure vs. an explicit q-point list.

    freq_cm1 : (nq, 3*nat) real, cm^-1, ascending per q (negative = unstable/
               imaginary mode, standard phonon-code sign convention)
    eigvecs  : (nq, 3*nat, 3*nat) complex, columns are the mass-weighted
               eigenvectors (combined index I = na*3+ipol), already
               orthonormal (sum_I |e_I|^2 = 1 per mode) -- feed directly to
               `atom_projected_weights`
    qpts     : (nq, 3) fractional q-points, as given
    nat      : int
    """
    freq_cm1: np.ndarray
    eigvecs:  np.ndarray
    qpts:     np.ndarray
    nat:      int


def apply_acoustic_sum_rule(fc_data: dict) -> dict:
    """
    Enforce the acoustic sum rule (ASR) on `read_force_constants`'s raw
    force constants -- matdyn.f90's `asr='simple'` option (`set_asr`),
    the "previous implementation ... correction of the diagonal elements
    of the force constants matrix": for each Cartesian pair (i,j) and
    atom `na`, the on-site R=0 block is corrected so the force constants
    sum to zero over all (R, nb) -- translational invariance (rigidly
    shifting the whole crystal costs zero energy).

        C(0)_{ij}(na,na) -= sum_{nb,R} C(R)_{ij}(na,nb)

    Real DFPT force constants are NOT exactly translationally invariant
    (finite k-mesh/q-mesh, finite SCF thresholds) -- for a well-converged
    insulator (e.g. this project's own diamond fixture) the residual is a
    few cm^-1 at most, small enough that `interpolate_phonons` is usually
    called directly on the raw `read_force_constants` output. For a METAL
    (e.g. MgB2), the acoustic-mode residual at Gamma can be much larger
    (tens of cm^-1) on a modest electronic k-mesh -- phonons in metals are
    intrinsically more sensitive to electronic-structure convergence than
    in insulators, since the long-wavelength lattice response couples to
    the Fermi surface. Applying this correction is standard practice
    (`matdyn.x`'s own default is `asr='crystal'`, an optimized variant of
    the same idea), not a cover-up: it does not change the OPTICAL modes
    materially and forces the acoustic branch to its physically-required
    zero at Gamma.

    Returns a new dict (shallow copy of `fc_data` with `fc` replaced) --
    does not mutate the input.
    """
    fc = fc_data["fc"].copy()   # (3,3,nat,nat,nr1,nr2,nr3)
    nat = fc_data["nat"]
    total = fc.sum(axis=(3, 4, 5, 6))   # (3,3,nat) -- sum over nb and every R
    for na in range(nat):
        fc[:, :, na, na, 0, 0, 0] -= total[:, :, na]
    out = dict(fc_data)
    out["fc"] = fc
    return out


_OMEGA2_CONV = (HARTREE_TO_J / BOHR_RADIUS_M ** 2) / AMU_TO_KG   # Hartree/(Bohr^2*amu) -> rad^2/s^2


def cm1_to_omega2_au(nu_cm1) -> np.ndarray:
    """
    cm^-1 -> omega^2 in the SAME raw atomic units (Hartree/(Bohr^2*amu))
    `phonon_hamiltonian`'s eigenvalues come out in -- the inverse of
    `interpolate_phonons`'s own `nu_cm1 = sign(omega2)*sqrt(|omega2*conv|)/
    (2*pi)/c_cm_per_s` conversion, needed to build an energy/frequency grid
    in these units for `analysis.surface.surface_spectral_function` (which
    is unit-agnostic and just needs `energies` in the SAME units as
    `hr.H_R`'s own eigenvalues). Preserves sign (a negative cm^-1 input,
    the standard imaginary-mode convention, maps to a negative omega^2).
    """
    nu_cm1 = np.asarray(nu_cm1, dtype=np.float64)
    omega_si = nu_cm1 * _C_CM_PER_S * 2.0 * np.pi
    return np.sign(omega_si) * omega_si ** 2 / _OMEGA2_CONV


def omega2_au_to_cm1(omega2_au) -> np.ndarray:
    """Inverse of `cm1_to_omega2_au`."""
    omega2_au = np.asarray(omega2_au, dtype=np.float64)
    omega2_si = omega2_au * _OMEGA2_CONV
    omega_si = np.sign(omega2_si) * np.sqrt(np.abs(omega2_si))
    return (omega_si / (2.0 * np.pi)) / _C_CM_PER_S


def phonon_hamiltonian(fc_data: dict, real_lattice: np.ndarray) -> HamiltonianR:
    """
    Package the mass-weighted dynamical matrix D(R) = C(R)/sqrt(M_na*M_nb)
    as a plain `core.hamiltonian.HamiltonianR` -- so ANY bulk-H(R)-only
    analysis already written for the electronic case runs UNCHANGED on
    phonons too, most usefully `analysis.surface.surface_spectral_function`
    (Lopez-Sancho/Rubio decimation), whose `build_surface_layers` only
    ever touches `hr.H_R`/`hr.R_vectors`/`hr.degen`/`hr.nw` -- entirely
    physics-agnostic. Mass-weighting is a per-(I,J) outer-product rescaling
    that does NOT depend on R, so (unlike `interpolate_phonons`, which
    mass-weights per-q after the Fourier sum) it commutes with the R-space
    transform and can be applied once here.

    `hr.H_R`'s eigenvalues come out as omega^2 in raw atomic units
    (Hartree/(Bohr^2*amu)) -- convert a desired cm^-1 energy/broadening
    grid with `cm1_to_omega2_au` before calling `surface_spectral_function`,
    and convert its returned `.energies` back with `omega2_au_to_cm1`.

    SIMPLIFICATION vs. `interpolate_phonons`: uses the PLAIN Wigner-Seitz
    reduction (`core.hamiltonian._wigner_seitz`) without `interpolate_
    phonons`'s atom-position-aware `build_ws_distance` refinement --
    `surface.build_surface_layers` re-indexes each matrix element by a
    SINGLE R (the shared `HamiltonianR` convention, same as the electronic
    case), not the per-(I,J)-pair multi-image phase `build_ws_distance`
    provides, so that finer correction has nowhere to plug into this
    particular code path. A reasonable simplification for a qualitative
    surface calculation, not a bug -- this project's own electronic
    surface notebooks don't universally need it either.
    """
    nat = fc_data["nat"]
    nr1, nr2, nr3 = (int(x) for x in fc_data["grid"])
    fc = fc_data["fc"]              # (3,3,nat,nat,nr1,nr2,nr3) Hartree/Bohr^2
    masses_amu = np.asarray(fc_data["masses_amu"], dtype=np.float64)
    types = np.asarray(fc_data["types"], dtype=np.int64)
    real_lattice = np.asarray(real_lattice, dtype=np.float64)

    mp_grid = (nr1, nr2, nr3)
    R_vectors, degen = _wigner_seitz(mp_grid, real_lattice)
    nR = len(R_vectors)

    m1 = np.mod(R_vectors[:, 0], nr1)
    m2 = np.mod(R_vectors[:, 1], nr2)
    m3 = np.mod(R_vectors[:, 2], nr3)
    fc_gathered = fc[:, :, :, :, m1, m2, m3]                  # (3,3,nat,nat,nR)
    fc_gathered = np.moveaxis(fc_gathered, -1, 0)             # (nR,3,3,nat,nat)

    nw = 3 * nat
    C_R = fc_gathered.transpose(0, 3, 1, 4, 2).reshape(nR, nw, nw)   # I=na*3+ipol

    mass_combined = np.repeat(masses_amu[types], 3)            # (3*nat,) amu
    inv_mass_sqrt_outer = 1.0 / np.sqrt(np.outer(mass_combined, mass_combined))
    D_R = C_R * inv_mass_sqrt_outer[None, :, :]
    # NOT Hermitized per-R here (that would wrongly force each individual
    # R-block symmetric instead of the correct global H(-R) = H(R)^dagger
    # pairing) -- H(k) built from this via a Fourier sum is Hermitian for
    # real k regardless, exactly as the plain electronic `operator_k` path
    # relies on without any per-R correction either.

    return HamiltonianR(H_R=torch.as_tensor(D_R, dtype=torch.complex128),
                        R_vectors=R_vectors, degen=degen, nw=nw)


def interpolate_phonons(
    fc_data: dict,
    real_lattice: np.ndarray,
    atom_positions_frac: np.ndarray,
    qpts_frac: np.ndarray,
) -> PhononBands:
    """
    Fourier-interpolate `read_force_constants`'s real-space force constants
    to an arbitrary list of q-points and diagonalize the (mass-weighted)
    dynamical matrix at each one.

    Args:
      fc_data            : dict from `interfaces.quantum_espresso.phonon_io.
                            read_force_constants` -- uses `nat`, `grid`,
                            `fc` (Hartree/Bohr^2), `masses_amu`, `types`
                            (`ntyp`/`ibrav`/`celldm`/`tau_alat` are NOT used
                            here; this project always runs QE with
                            `ibrav=0` and supplies the lattice/positions
                            explicitly, same convention as `core.hamiltonian`)
      real_lattice       : (3, 3) float, Bohr, rows = lattice vectors
                           (SAME cell used for the `ph.x`/`q2r.x` run)
      atom_positions_frac: (nat, 3) float, fractional coordinates, in the
                           SAME atom order as `fc_data`'s `types` (i.e. the
                           `ATOMIC_POSITIONS` order the SCF/ph.x run used)
      qpts_frac          : (nq, 3) fractional q-points

    Returns a `PhononBands`.
    """
    nat = fc_data["nat"]
    nr1, nr2, nr3 = (int(x) for x in fc_data["grid"])
    fc = fc_data["fc"]              # (3,3,nat,nat,nr1,nr2,nr3) Hartree/Bohr^2
    masses_amu = np.asarray(fc_data["masses_amu"], dtype=np.float64)
    types = np.asarray(fc_data["types"], dtype=np.int64)
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    atom_positions_frac = np.asarray(atom_positions_frac, dtype=np.float64)
    qpts_frac = np.atleast_2d(np.asarray(qpts_frac, dtype=np.float64))

    mp_grid = (nr1, nr2, nr3)
    R_vectors, degen = _wigner_seitz(mp_grid, real_lattice)   # (nR,3) int, (nR,) int
    nR = len(R_vectors)

    # The force-constant array is periodic with period (nr1,nr2,nr3) in
    # real space (it came from a Fourier transform of a coarse q-mesh) --
    # any WS-reduced R (which can land outside the original [0,nr) box)
    # maps back to the SAME stored value via a plain modulo, exactly
    # matching matdyn.f90's own `m1 = mod(n1+1,nr1); if(m1<=0) m1=m1+nr1`.
    m1 = np.mod(R_vectors[:, 0], nr1)
    m2 = np.mod(R_vectors[:, 1], nr2)
    m3 = np.mod(R_vectors[:, 2], nr3)
    fc_gathered = fc[:, :, :, :, m1, m2, m3]                  # (3,3,nat,nat,nR)
    fc_gathered = np.moveaxis(fc_gathered, -1, 0)             # (nR,3,3,nat,nat)

    nw = 3 * nat
    # combined index I = na*3+ipol: reshape (r,na,ipol,nb,jpol) -> (r,I,J)
    C_R = fc_gathered.transpose(0, 3, 1, 4, 2).reshape(nR, nw, nw)

    atom_positions_cart = atom_positions_frac @ real_lattice   # (nat,3) Bohr
    centres_cart = np.repeat(atom_positions_cart, 3, axis=0)   # (3*nat,3), matches I=na*3+ipol

    ws = build_ws_distance(R_vectors, centres_cart, mp_grid, real_lattice)

    mass_combined = np.repeat(masses_amu[types], 3)            # (3*nat,) amu
    inv_mass_sqrt_outer = 1.0 / np.sqrt(np.outer(mass_combined, mass_combined))

    freq_cm1 = np.empty((len(qpts_frac), nw))
    eigvecs = np.empty((len(qpts_frac), nw, nw), dtype=np.complex128)

    conv = (HARTREE_TO_J / BOHR_RADIUS_M ** 2) / AMU_TO_KG     # -> rad^2/s^2

    for iq, q in enumerate(qpts_frac):
        wsph = ws.phase(q)                                     # (nR, nw, nw) complex
        C_q = np.einsum("r,rij,rij->ij", 1.0 / degen, wsph, C_R)   # (nw,nw) complex
        D_q = C_q * inv_mass_sqrt_outer
        D_q = 0.5 * (D_q + D_q.conj().T)                       # Hermitize defensively

        eigval, eigvec = np.linalg.eigh(D_q)                   # ascending, real eigval
        omega2_si = eigval * conv                              # rad^2/s^2
        omega_si = np.sign(omega2_si) * np.sqrt(np.abs(omega2_si))
        nu_cm1 = (omega_si / (2.0 * np.pi)) / _C_CM_PER_S

        freq_cm1[iq] = nu_cm1
        eigvecs[iq] = eigvec

    return PhononBands(freq_cm1=freq_cm1, eigvecs=eigvecs, qpts=qpts_frac, nat=nat)


def atom_projected_weights(bands: PhononBands, types: np.ndarray, ntyp: int) -> np.ndarray:
    """
    Per-species character of each phonon mode: for species `t`, the sum
    over that species' atoms and all 3 Cartesian components of
    |eigenvector|^2 -- already normalized to 1 total per mode since the
    (mass-weighted) eigenvectors from `interpolate_phonons` are orthonormal
    by construction (`eigh` on a Hermitian matrix), exactly analogous to
    an electronic-structure fatband's |c_i|^2 orbital projection.

    Args:
      bands : from `interpolate_phonons`
      types : (nat,) int, 0-based species index per atom (`fc_data["types"]`)
      ntyp  : number of species

    Returns weights: (nq, 3*nat, ntyp) real, in [0, 1], summing to 1 over
    the last axis for every (q, mode).
    """
    nq, nw, _ = bands.eigvecs.shape
    nat = bands.nat
    prob = np.abs(bands.eigvecs) ** 2                # (nq, I, mode) -- I = na*3+ipol
    prob = prob.reshape(nq, nat, 3, nw).sum(axis=2)   # (nq, nat, mode) -- sum over Cartesian

    weights = np.zeros((nq, nw, ntyp))
    for t in range(ntyp):
        mask = (types == t)
        weights[:, :, t] = prob[:, mask, :].sum(axis=1)   # (nq, mode)
    return weights


@dataclass
class PhononDOS:
    """
    Gaussian-broadened phonon density of states, with a per-species
    (atom-projected) decomposition -- the lattice-dynamics analogue of
    `analysis.dos.density_of_states`, reusing the SAME `gaussian_smearing`
    broadening kernel (unit-agnostic: fed cm^-1 here instead of Hartree).

    freq_cm1    : (n_freq,) cm^-1 grid
    dos_total   : (n_freq,) states/cm^-1, per cell (sum over all 3*nat modes)
    dos_species : (n_freq, ntyp) states/cm^-1 -- summing over the last axis
                  at any frequency reproduces `dos_total` there, since
                  `atom_projected_weights` already sums to 1 per mode
    """
    freq_cm1:    np.ndarray
    dos_total:   np.ndarray
    dos_species: np.ndarray


def phonon_density_of_states(
    fc_data: dict,
    real_lattice: np.ndarray,
    atom_positions_frac: np.ndarray,
    mesh: tuple[int, int, int],
    freq_cm1: np.ndarray | None = None,
    n_freq: int = 500,
    sigma_cm1: float = 5.0,
    pad_cm1: float = 20.0,
) -> PhononDOS:
    """
    Phonon density of states (total + per-species) by Gaussian-broadened
    Fourier interpolation on a dense uniform q-mesh -- structurally
    identical to `analysis.dos.density_of_states`, with `interpolate_phonons`
    playing the role of `interpolate_bands` and `atom_projected_weights`
    supplying the per-mode species decomposition (the phonon analogue of
    an electronic PDOS).

    Args:
      fc_data, real_lattice, atom_positions_frac : as `interpolate_phonons`
      mesh      : (N1, N2, N3) dense q-mesh (e.g. (20, 20, 20)) -- NOT the
                  coarse `ph.x`/`q2r.x` grid `fc_data["grid"]` itself
      freq_cm1  : explicit frequency grid, cm^-1; if None, built
                  automatically from [min(freq) - pad, max(freq) + pad]
      n_freq    : number of grid points when `freq_cm1` is None
      sigma_cm1 : Gaussian broadening width, cm^-1
      pad_cm1   : padding, cm^-1, added to the automatic frequency range

    Returns a `PhononDOS`.
    """
    N1, N2, N3 = mesh
    i, j, k = np.meshgrid(np.arange(N1), np.arange(N2), np.arange(N3), indexing="ij")
    qpts = np.stack([i.ravel() / N1, j.ravel() / N2, k.ravel() / N3], axis=-1)

    bands = interpolate_phonons(fc_data, real_lattice, atom_positions_frac, qpts_frac=qpts)
    ntyp = int(fc_data["ntyp"]) if "ntyp" in fc_data else int(np.max(fc_data["types"]) + 1)
    weights = atom_projected_weights(bands, fc_data["types"], ntyp)   # (nq, nmode, ntyp)

    if freq_cm1 is None:
        freq_cm1 = np.linspace(bands.freq_cm1.min() - pad_cm1, bands.freq_cm1.max() + pad_cm1, n_freq)
    freq_cm1 = np.asarray(freq_cm1, dtype=np.float64)

    nq = bands.freq_cm1.shape[0]
    diff = freq_cm1[:, None, None] - bands.freq_cm1[None, :, :]   # (n_freq, nq, nmode)
    g = gaussian_smearing(diff, sigma_cm1)                        # (n_freq, nq, nmode)

    dos_total = g.sum(axis=(1, 2)) / nq
    dos_species = np.einsum("fqm,qmt->ft", g, weights) / nq       # (n_freq, ntyp)

    return PhononDOS(freq_cm1=freq_cm1, dos_total=dos_total, dos_species=dos_species)
