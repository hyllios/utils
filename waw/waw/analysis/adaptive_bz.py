"""
Adaptive Brillouin-zone integration for quantities -- like the anomalous
Hall conductivity -- that converge slowly on a uniform k-mesh because
their integrand is concentrated in small regions of the BZ (near band
touchings straddling the Fermi energy), rather than smoothly spread
over it.

This is deterministic adaptive quadrature (Richardson/AMR-style; the
classic condensed-matter analog is the improved tetrahedron method of
Blochl, Jepsen & Andersen, PRB 49, 16223 (1994)), not Monte Carlo
importance sampling: the BZ integral is low-dimensional (d=3) and the
"danger zones" are analytically locatable (small band gaps near E_F),
which favors deterministic refinement -- exact convergence as you
refine further, no residual stochastic noise floor -- over a stochastic
proposal distribution.

Algorithm: recursively subdivide k-space cells (octree over the
fractional BZ cube) wherever a cheap band-gap criterion (evaluated at
each cell's 8 corners) signals a nearby occupied/unoccupied band
touching; the (comparatively expensive) real integrand --
`topology.wannier_interpolated_curvature` -- is evaluated only once,
in one batched call, at the centres of the resulting leaf cells, each
weighted by its own actual k-space volume.

Prototype status: validated against `topology.py`'s own sharp-step/
adaptive-smearing machinery and empirically against Fe3RuN's AHC (this
project's own worked example of the classic slow-AHC-convergence
problem) -- not yet cross-validated against a real wannier90/postw90
adaptive-refinement reference (postw90 doesn't have one; its own
`kubo_adpt_smr` is smearing, not mesh refinement).

The Richardson-refinement engine (`_richardson_refine`) is physics-
agnostic -- `adaptive_ahc_richardson` and `adaptive_shc_richardson`
(the spin Hall conductivity analogue, reusing `spin_hall.
spin_berry_curvature_kpath` as the per-k evaluator) are both thin
wrappers around the SAME octree bookkeeping, differing only in which
per-k integrand function is plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, interpolate_bands
from ..units import HARTREE_TO_EV, EV_TO_HARTREE
from .shift_current import _kmesh_spacing, KUBO_ADPT_SMR_FAC_DEFAULT, KUBO_ADPT_SMR_MAX_DEFAULT
from .topology import wannier_interpolated_curvature
from .spin_hall import spin_berry_curvature_kpath


@dataclass
class AdaptiveBZResult:
    """Result of `adaptive_ahc`."""
    sigma:            np.ndarray   # (3,) Bohr^2, axial vector (yz, zx, xy)
    n_kpoints:        int          # total k-points evaluated (gap-check corners + leaf centres)
    n_leaves:         int          # number of leaf cells the final integral summed over
    max_depth_reached: int
    truncated:        bool         # True if max_kpoints was hit before every flagged cell finished refining


def _cell_corners(origins: np.ndarray, size: np.ndarray) -> np.ndarray:
    """
    8 corners of each fractional-k-space box in a batch of cells.

    origins: (ncell, 3) lower corner of each box
    size:    (3,) box edge lengths (same for every cell in this batch --
             true for every level of an octree refined breadth-first)

    Returns (ncell, 8, 3).
    """
    offsets = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64)
    return origins[:, None, :] + offsets[None, :, :] * size[None, None, :]


def _min_occ_unocc_gap(bands: np.ndarray, fermi_energy: float) -> np.ndarray:
    """
    Per-k-point minimum |E_occ - E_unocc| over all occupied/unoccupied
    band pairs at that k -- the quantity whose smallness drives the
    AHC's 1/(E_n-E_m) divergence for pairs straddling E_F
    (`topology._jjp_jjm_batch`'s literal denominator).

    bands: (nk, nw), Hartree. Returns (nk,) -- +inf where every band is on
    the same side of E_F (nothing dangerous at that k).
    """
    nk, nw = bands.shape
    occ = bands < fermi_energy
    out = np.full(nk, np.inf)
    for k in range(nk):
        e_occ = bands[k, occ[k]]
        e_unocc = bands[k, ~occ[k]]
        if len(e_occ) and len(e_unocc):
            out[k] = np.min(np.abs(e_occ[:, None] - e_unocc[None, :]))
    return out


def adaptive_ahc(
    hr: HamiltonianR, AA_R: "torch.Tensor", recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energy: float, *,
    base_mesh: tuple[int, int, int] = (6, 6, 6),
    max_depth: int = 5,
    gap_threshold: float = 0.05 * EV_TO_HARTREE,
    max_kpoints: int = 200_000,
    kubo_adpt_smr: bool = True,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> AdaptiveBZResult:
    """
    Anomalous Hall conductivity at a single Fermi energy, on an
    adaptively-refined k-mesh: start from `base_mesh`, and recursively
    octree-subdivide any cell whose 8 corners show an occupied/
    unoccupied band pair closer than `gap_threshold` (a cheap,
    bands-only check -- `core.hamiltonian.interpolate_bands`, not the
    full curvature machinery), down to `max_depth` extra halvings. The
    real (expensive) integrand -- `topology.wannier_interpolated_curvature`
    -- is evaluated exactly once, in one batched call, at the resulting
    leaf cells' centres, weighted by each leaf's own k-space volume
    (NOT a plain unweighted mean -- refined leaves represent much
    smaller BZ volume than untouched ones).

    Complements (not replaces) `topology.anomalous_hall_conductivity`'s
    own `kubo_adpt_smr` adaptive-*smearing* option: refinement fixes
    *where* you sample, smearing tempers whatever residual divergence
    survives within the finest cells actually reached. Both default to
    the same `delta_k`, computed from `base_mesh` (a deliberate
    simplification for this prototype -- a finer per-leaf-size adaptive
    width would sharpen this further, not yet implemented).

    A corner-only gap check can in principle miss a feature entirely
    contained within a cell without touching any of its 8 corners; in
    practice this is the same accepted limitation every adaptive mesh
    method has, and is mitigated by not letting `base_mesh` itself be
    too coarse to begin with.

    Args:
      hr, AA_R, recip_lattice, real_lattice: as `anomalous_hall_conductivity`
      fermi_energy    : Hartree, a single value (not a scan -- the
                        refined mesh is specific to where the danger
                        zones sit relative to this E_F)
      base_mesh       : starting uniform mesh, refined from here
      max_depth       : maximum extra octree halvings beyond `base_mesh`
      gap_threshold: HARTREE (the default is written as 0.05 eV in Hartree
                     so the physical scale stays legible). Refine while the
                     minimum occ/unocc gap in a
                        cell is below this (eV)
      max_kpoints     : safety cap on total k-points evaluated; if hit,
                        remaining unresolved cells are used as leaves
                        as-is (reported via `.truncated`, not silently)
      kubo_adpt_smr, kubo_adpt_smr_fac, kubo_adpt_smr_max: passed to
                        `wannier_interpolated_curvature` for the final
                        leaf-centre evaluation (see there)

    Returns AdaptiveBZResult (bare Bohr^2, no cell-volume/e^2-hbar
    prefactor -- convert via `waw.units.to_si_units(..., "hall_conductivity", ...)`
    exactly as for `AHCResult.sigma`).
    """
    fermi_energy = float(fermi_energy)
    Na, Nb, Nc = base_mesh

    origins = np.stack(np.meshgrid(np.arange(Na) / Na, np.arange(Nb) / Nb, np.arange(Nc) / Nc,
                                    indexing='ij'), axis=-1).reshape(-1, 3)
    size = np.array([1.0 / Na, 1.0 / Nb, 1.0 / Nc])

    leaves_center = []
    leaves_volume = []
    n_gapcheck_kpts = 0
    max_depth_reached = 0
    truncated = False

    depth = 0
    while len(origins) and depth <= max_depth:
        n_cells = len(origins)
        if n_gapcheck_kpts + 8 * n_cells + n_cells > max_kpoints:
            # budget exhausted: stop refining, use every remaining cell as a leaf as-is
            leaves_center.append(origins + size / 2)
            leaves_volume.append(np.full(n_cells, np.prod(size)))
            truncated = True
            break

        corners = _cell_corners(origins, size).reshape(-1, 3)          # (n_cells*8, 3)
        n_gapcheck_kpts += len(corners)
        bands = np.asarray(interpolate_bands(hr, corners))
        gaps = _min_occ_unocc_gap(bands, fermi_energy).reshape(n_cells, 8).min(axis=1)

        refine = gaps < gap_threshold
        if depth == max_depth:
            refine[:] = False   # depth cap: everything left becomes a leaf this round

        # non-refined cells at this depth become leaves
        if np.any(~refine):
            leaves_center.append(origins[~refine] + size / 2)
            leaves_volume.append(np.full(np.count_nonzero(~refine), np.prod(size)))

        n_refine = int(np.count_nonzero(refine))
        if n_refine == 0:
            break
        max_depth_reached = depth + 1

        # octree-subdivide every refined cell into 8 children for the next level
        half = size / 2
        child_offsets = np.array([[i, j, k] for i in (0, half[0]) for j in (0, half[1]) for k in (0, half[2])])
        parents = origins[refine]                                       # (n_refine, 3)
        origins = (parents[:, None, :] + child_offsets[None, :, :]).reshape(-1, 3)   # (n_refine*8, 3)
        size = half
        depth += 1

    centers = np.concatenate(leaves_center, axis=0)
    volumes = np.concatenate(leaves_volume, axis=0)
    n_leaves = len(centers)

    delta_k = _kmesh_spacing(base_mesh, recip_lattice) if kubo_adpt_smr else None
    curvature = wannier_interpolated_curvature(
        hr, AA_R, recip_lattice, real_lattice, centers, fermi_energy,
        delta_k=delta_k, adpt_smr_fac=kubo_adpt_smr_fac, adpt_smr_max=kubo_adpt_smr_max,
    )   # (n_leaves, 1, 3)

    sigma = np.average(curvature[:, 0, :], axis=0, weights=volumes)

    return AdaptiveBZResult(
        sigma=sigma, n_kpoints=n_gapcheck_kpts + n_leaves, n_leaves=n_leaves,
        max_depth_reached=max_depth_reached, truncated=truncated,
    )


def _richardson_refine(
    eval_fn, ncomp: int, base_mesh: tuple[int, int, int],
    max_depth: int, rtol: float, atol: float, max_kpoints: int,
) -> AdaptiveBZResult:
    """
    Generic two-level Richardson/adaptive-quadrature octree engine,
    factored out of `adaptive_ahc_richardson` so it can drive ANY
    per-k integrand, not just the AHC's Berry curvature -- the octree
    bookkeeping (cell bisection, leaf-volume weighting, budget/depth
    caps) has no AHC-specific physics in it at all.

    `eval_fn(kpts) -> (n, ncomp) real ndarray` is the only thing that
    changes between quantities; `adaptive_ahc_richardson` wraps
    `topology.wannier_interpolated_curvature` (ncomp=3, an axial
    vector), `adaptive_shc_richardson` wraps
    `spin_hall.spin_berry_curvature_kpath` (ncomp=1, a scalar per
    fixed alpha/beta/gamma) -- same engine, same convergence criterion,
    different physics plugged in.

    Each cell's own midpoint-rule estimate (`eval_fn` at its centre) is
    compared to the *refined* estimate from averaging its 8 children's
    own midpoint estimates. A cell is accepted as a leaf (using the
    more accurate children-average as its value) once

        |value_children_mean - value_parent| < atol + rtol * |value_children_mean|

    (a standard combined absolute+relative tolerance, as in
    `numpy.allclose`); otherwise the children become the next level's
    cells and the same check recurses on *them*. Self-calibrating by
    construction -- no material-specific energy scale to guess (see
    `adaptive_ahc_richardson`'s own docstring for why `adaptive_ahc`'s
    gap-threshold alternative proved awkward to tune on a real system).

    `max_depth=0` is a well-defined degenerate case (no refinement
    rounds at all -- every base-mesh cell's own single-point midpoint
    value is used as-is, the plain non-adaptive midpoint rule) rather
    than an error; handled as an explicit special case since the
    refine/accept loop below only ever records a cell as a leaf via a
    parent-vs-children comparison, which needs at least one refinement
    round to happen at all.

    Returns `AdaptiveBZResult` with `sigma` shape `(ncomp,)`.
    """
    Na, Nb, Nc = base_mesh
    origins = np.stack(np.meshgrid(np.arange(Na) / Na, np.arange(Nb) / Nb, np.arange(Nc) / Nc,
                                    indexing='ij'), axis=-1).reshape(-1, 3)
    size = np.array([1.0 / Na, 1.0 / Nb, 1.0 / Nc])

    centers = origins + size / 2
    values = eval_fn(centers)
    n_kpoints = len(centers)

    if max_depth == 0:
        sigma = values.mean(axis=0)   # every base cell has equal volume
        return AdaptiveBZResult(sigma=sigma, n_kpoints=n_kpoints, n_leaves=n_kpoints,
                                max_depth_reached=0, truncated=False)

    leaves_volume = [np.array([])]
    leaves_value = [np.zeros((0, ncomp))]
    max_depth_reached = 0
    truncated = False

    depth = 0
    while len(origins) and depth < max_depth:
        n_cells = len(origins)
        if n_kpoints + 8 * n_cells > max_kpoints:
            leaves_volume.append(np.full(n_cells, np.prod(size)))
            leaves_value.append(values)
            truncated = True
            origins = np.empty((0, 3))
            break

        half = size / 2
        child_offsets = np.array([[i, j, k] for i in (0, half[0]) for j in (0, half[1]) for k in (0, half[2])])
        child_origins = (origins[:, None, :] + child_offsets[None, :, :]).reshape(-1, 3)   # (n_cells*8,3)
        child_centers = child_origins + half / 2
        child_values = eval_fn(child_centers)                       # (n_cells*8,ncomp)
        n_kpoints += len(child_centers)

        child_values_grouped = child_values.reshape(n_cells, 8, ncomp)
        child_mean = child_values_grouped.mean(axis=1)              # (n_cells,ncomp)

        diff = np.linalg.norm(child_mean - values, axis=1)
        scale = np.linalg.norm(child_mean, axis=1)
        converged = diff < (atol + rtol * scale)
        if depth == max_depth - 1:
            converged[:] = True   # depth cap: accept every remaining cell as-is this round

        if np.any(converged):
            leaves_volume.append(np.full(np.count_nonzero(converged), np.prod(size)))
            leaves_value.append(child_mean[converged])

        keep = ~converged
        if not np.any(keep):
            origins = np.empty((0, 3))
            break
        max_depth_reached = depth + 1

        origins = child_origins.reshape(n_cells, 8, 3)[keep].reshape(-1, 3)
        values = child_values_grouped[keep].reshape(-1, ncomp)
        size = half
        depth += 1

    all_volume = np.concatenate(leaves_volume, axis=0)
    all_value = np.concatenate(leaves_value, axis=0)

    sigma = np.average(all_value, axis=0, weights=all_volume)
    return AdaptiveBZResult(
        sigma=sigma, n_kpoints=n_kpoints, n_leaves=len(all_value),
        max_depth_reached=max_depth_reached, truncated=truncated,
    )


def adaptive_ahc_richardson(
    hr: HamiltonianR, AA_R: "torch.Tensor", recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energy: float, *,
    base_mesh: tuple[int, int, int] = (6, 6, 6),
    max_depth: int = 5,
    rtol: float = 0.15,
    atol: float = 1e-3,
    max_kpoints: int = 200_000,
    kubo_adpt_smr: bool = True,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> AdaptiveBZResult:
    """
    Like `adaptive_ahc`, but the refinement trigger is pure numerical
    convergence of the real integrand (see `_richardson_refine` for the
    generic engine this wraps) -- no physical gap threshold to
    calibrate per material.

    Tolerance calibration matters more than it might look: even on a
    perfectly smooth, genuinely gapped model (no near-degeneracy
    anywhere), a too-tight `rtol` (e.g. 0.05, this function's first
    prototype default) keeps refining far past what the physics needs,
    simply chasing the ordinary O(h^2) midpoint-rule discretization
    error of ANY curved-but-benign integrand -- confirmed directly: on
    the gapped QWZ test model, rtol=0.05 used 570k points for the same
    answer rtol=0.2 got in under 13k. `rtol=0.15` is a more reasonable
    default; still tightenable for a final, publication-grade number.

    The tradeoff vs. `adaptive_ahc`'s gap check: every candidate
    cell needs the *real* (comparatively expensive) curvature evaluated
    at 1+8 points to make the refine/accept decision, vs. `adaptive_ahc`'s
    cheap bands-only gap check at 8 points.

    Args, returns: as `adaptive_ahc`, except `rtol`/`atol` replace
    `gap_threshold`.
    """
    fermi_energy = float(fermi_energy)
    delta_k = _kmesh_spacing(base_mesh, recip_lattice) if kubo_adpt_smr else None

    def eval_curv(kpts: np.ndarray) -> np.ndarray:
        c = wannier_interpolated_curvature(
            hr, AA_R, recip_lattice, real_lattice, kpts, fermi_energy,
            delta_k=delta_k, adpt_smr_fac=kubo_adpt_smr_fac, adpt_smr_max=kubo_adpt_smr_max,
        )
        return c[:, 0, :]   # (n, 3) -- single fermi_energy

    return _richardson_refine(eval_curv, ncomp=3, base_mesh=base_mesh, max_depth=max_depth,
                              rtol=rtol, atol=atol, max_kpoints=max_kpoints)


def adaptive_shc_richardson(
    hr: HamiltonianR, AA_R: "torch.Tensor", SS_R: "torch.Tensor",
    SR_R: "torch.Tensor | None" = None, SHR_R: "torch.Tensor | None" = None,
    SH_R: "torch.Tensor | None" = None,
    recip_lattice: np.ndarray = None, real_lattice: np.ndarray = None,
    fermi_energy: float = None, *,
    alpha: int = 0, beta: int = 1, gamma: int = 2,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    eta: float = 0.04 * EV_TO_HARTREE,
    method: str = "qiao",
    SAA_R: "torch.Tensor | None" = None, SBB_R: "torch.Tensor | None" = None,
    base_mesh: tuple[int, int, int] = (6, 6, 6),
    max_depth: int = 5,
    rtol: float = 0.15,
    atol: float = 1e-3,
    max_kpoints: int = 200_000,
) -> AdaptiveBZResult:
    """
    Spin Hall conductivity at a single Fermi energy, on an adaptively
    Richardson-refined k-mesh -- the SHC analogue of
    `adaptive_ahc_richardson`, reusing the SAME generic octree engine
    (`_richardson_refine`) with `spin_hall.spin_berry_curvature_kpath`
    (already validated: averaging it over a uniform mesh reproduces
    `spin_hall_conductivity`'s own fixed-eta result exactly, see
    `test_spin_hall.py`) as the per-k evaluator in place of AHC's
    `wannier_interpolated_curvature`. No new low-level physics needed --
    this function is pure orchestration.

    Only FIXED smearing (`eta`) applies here, unlike `adaptive_ahc_
    richardson`'s optional `kubo_adpt_smr`: `spin_berry_curvature_kpath`
    itself only supports fixed eta (`_kmesh_spacing`'s adaptive-width
    trick needs a uniform mesh, not meaningful for an irregular,
    adaptively-refined k-point set) -- see that function's own
    docstring for the same limitation.

    Args mirror `spin_hall_conductivity`/`adaptive_ahc_richardson`:
      hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
      alpha, beta, gamma, degen_thresh, method, SAA_R, SBB_R : as
                          `spin_hall_conductivity` (SR_R/SHR_R/SH_R
                          required iff method='qiao'; SAA_R/SBB_R
                          required iff method='ryoo')
      fermi_energy      : Hartree, a single value (not a scan)
      eta               : fixed Green's-function broadening, Hartree
      base_mesh, max_depth, rtol, atol, max_kpoints : as
                          `adaptive_ahc_richardson`

    Returns `AdaptiveBZResult` (`sigma` shape `(1,)`, bare Bohr^2 --
    convert with `waw.units.to_si_units(res.sigma[0], "spin_hall_
    conductivity", cell_volume_bohr3=...)`, same convention as
    `SHCResult.sigma`).
    """
    fermi_energy = float(fermi_energy)

    def eval_curv(kpts: np.ndarray) -> np.ndarray:
        c = spin_berry_curvature_kpath(
            hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
            kpath=kpts, fermi_energy=fermi_energy, alpha=alpha, beta=beta, gamma=gamma,
            degen_thresh=degen_thresh, eta=eta, method=method, SAA_R=SAA_R, SBB_R=SBB_R,
        )
        return c[:, None]   # (n, 1)

    return _richardson_refine(eval_curv, ncomp=1, base_mesh=base_mesh, max_depth=max_depth,
                              rtol=rtol, atol=atol, max_kpoints=max_kpoints)
