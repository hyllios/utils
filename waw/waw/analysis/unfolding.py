"""
Band-structure unfolding: supercell eigenstates projected onto the primitive
Brillouin zone.

A supercell calculation folds the primitive bands onto a smaller zone, mixing
them; a defect or reconstruction then makes the folded bands hybridise. Unfolding
undoes the folding, so a defective supercell can be compared with the pristine
primitive band structure on the same axes -- the perturbed bands acquire a
spectral broadening instead of appearing as a forest of folded replicas.

THE PROJECTION. A supercell state |K m> expanded in plane waves,
|K m> = sum_G C_Km(G) |K + G>, contributes to the primitive-cell k-point k the
weight

    P_Km(k) = sum_{G : K + G == k modulo a PRIMITIVE reciprocal vector} |C_Km(G)|^2

(Popescu and Zunger, PRB 85, 085201 (2012), Eq. 5). With the supercell built as
A_sc = M A_pc the reciprocal bases satisfy B_sc = M^-T B_pc, so a k-point with
fractional coordinates kappa_pc in the primitive basis sits at

    kappa_sc = kappa_pc M^T

in the supercell basis, and the plane wave with integer supercell index G lands
on the primitive k-point iff (kappa_sc + G) M^-T - kappa_pc is a triplet of
integers.

NORMALISATION. The coefficients are PAW PSEUDO-wavefunctions, so sum_G |C|^2 is
not 1 -- it ranges from ~0.3 for a deep semicore state to ~1 for a
free-electron-like one, the remainder living in the augmentation spheres. The
weight is therefore divided by that sum, making P a FRACTION of the pseudo
wavefunction: P = 1 means the state is entirely a single primitive Bloch state,
P = 1/4 that it is spread over the four folded images of a 2x2 cell. Dividing by
1 instead would print the PAW deficiency as if it were unfolding physics.

TIME REVERSAL. kappa_sc is reduced into the first zone modulo 1, and if the
result is not in the calculation's k-set its negative is tried, which is legal
whenever the system is time-reversal symmetric (no magnetic field; spin-orbit
coupling alone does not break it). `spectral_weights` reports which route each
k-point took so this is auditable rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UnfoldedSpectrum:
    """Unfolded weights along a primitive-cell k-path."""
    kpoints_pc: np.ndarray      # (nk_pc, 3) fractional, primitive basis
    energies: np.ndarray        # (nk_pc, nbands) eV, as VASP reported them
    weights: np.ndarray         # (nk_pc, nbands) in [0, 1]
    sc_index: np.ndarray        # (nk_pc,) which supercell k-point was used
    time_reversed: np.ndarray   # (nk_pc,) bool: -k was needed
    band_range: tuple           # (first, last) band index actually read


def match_supercell_kpoint(kappa_pc, kpts_sc, M, tol=1e-5):
    """
    Locate the supercell k-point holding a given primitive k-point.

    Returns (index, time_reversed). Raises if neither +k nor -k is present:
    silently falling back to the nearest k-point would unfold the wrong state.
    """
    kappa_pc = np.asarray(kappa_pc, dtype=float)
    target = kappa_pc @ np.asarray(M, dtype=float).T
    for reversed_ in (False, True):
        t = -target if reversed_ else target
        d = np.asarray(kpts_sc) - t
        d = d - np.round(d)                       # both live modulo 1
        i = int(np.argmin(np.linalg.norm(d, axis=1)))
        if np.linalg.norm(d[i]) < tol:
            return i, reversed_
    raise ValueError(
        f"primitive k-point {kappa_pc} maps to supercell {target % 1.0}, which is "
        f"in the calculation's k-set neither directly nor by time reversal. The "
        f"supercell matrix, the path, or the k-point list disagree.")


def primitive_mask(gvectors, kappa_sc, kappa_pc, M, tol=1e-6):
    """
    Which plane waves of a supercell state belong to a given primitive k-point.

    True where (kappa_sc + G) M^-T - kappa_pc is integral, i.e. where the plane
    wave differs from the primitive k-point by a primitive reciprocal vector.
    """
    Minv_T = np.linalg.inv(np.asarray(M, dtype=float)).T
    resid = (np.asarray(gvectors) + np.asarray(kappa_sc)) @ Minv_T \
        - np.asarray(kappa_pc)
    return np.all(np.abs(resid - np.round(resid)) < tol, axis=1)


def spectral_weights(wavecar, kpoints_pc, M, bands=None, spin=0,
                     energy_window=None, fermi=0.0, tol=1e-5):
    """
    Unfold a supercell WAVECAR onto a primitive-cell k-path.

    Args:
      wavecar       : `interfaces.vasp.wavecar.Wavecar`
      kpoints_pc    : (nk, 3) primitive-basis fractional k-points
      M             : (3, 3) integer supercell matrix, A_sc = M A_pc
      bands         : explicit (first, last) band index range, or None
      energy_window : (lo, hi) eV relative to `fermi`; bands outside are skipped.
                      These files are hundreds of GiB and a band record is ~1 MiB,
                      so a window is usually the difference between minutes and
                      hours. The window is applied per k-point over the union of
                      bands that fall inside it at ANY k, so the returned array
                      is rectangular.
      fermi         : eV, the reference for `energy_window`

    Returns `UnfoldedSpectrum`.
    """
    kpoints_pc = np.atleast_2d(np.asarray(kpoints_pc, dtype=float))
    M = np.asarray(M)
    nk = len(kpoints_pc)
    ksc = wavecar.kpoints[spin]

    idx, rev = np.zeros(nk, dtype=int), np.zeros(nk, dtype=bool)
    for i, kp in enumerate(kpoints_pc):
        idx[i], rev[i] = match_supercell_kpoint(kp, ksc, M, tol=tol)

    eig_all = wavecar.eigenvalues[spin]
    if bands is not None:
        b0, b1 = bands
    elif energy_window is not None:
        lo, hi = energy_window
        inside = ((eig_all[idx] - fermi) >= lo) & ((eig_all[idx] - fermi) <= hi)
        cols = np.where(inside.any(axis=0))[0]
        if len(cols) == 0:
            raise ValueError(f"no band falls inside {energy_window} eV of "
                             f"{fermi}; widen the window")
        b0, b1 = int(cols.min()), int(cols.max()) + 1
    else:
        b0, b1 = 0, wavecar.header.nbands

    energies = np.zeros((nk, b1 - b0))
    weights = np.zeros((nk, b1 - b0))
    for i, kp in enumerate(kpoints_pc):
        ik = int(idx[i])
        G = wavecar.gvectors(ik, spin=spin)
        k_sc = ksc[ik]
        # under time reversal the state at -k is the conjugate at +k, so the
        # plane-wave indices flip sign along with the k-point
        mask = primitive_mask(-G if rev[i] else G,
                              -k_sc if rev[i] else k_sc, kp, M)
        for j, ib in enumerate(range(b0, b1)):
            c = wavecar.coefficients(ik, ib, spin=spin)
            p = (np.abs(c) ** 2)
            if p.ndim == 2:                        # spinor: sum both components
                p = p.sum(axis=0)
            total = p.sum()
            energies[i, j] = eig_all[ik, ib]
            weights[i, j] = p[mask].sum() / total if total > 0 else 0.0
    return UnfoldedSpectrum(kpoints_pc=kpoints_pc, energies=energies,
                            weights=weights, sc_index=idx, time_reversed=rev,
                            band_range=(b0, b1))


def spectral_function(spectrum: UnfoldedSpectrum, energy_grid, sigma=0.05):
    """
    Gaussian-broaden the discrete weights into A(k, E) for plotting.

    `energy_grid` in eV on the same reference as `spectrum.energies`; `sigma` in
    eV is a plotting width, not a physical linewidth.
    """
    e = np.asarray(energy_grid, dtype=float)
    d = e[None, :, None] - spectrum.energies[:, None, :]
    g = np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return (g * spectrum.weights[:, None, :]).sum(axis=2)
