"""
Fermi surface isosurface computation (physics/geometry only).

`fermi_surface_sheets` builds an in-Python triangle-mesh isosurface (for
`waw.vis.fermi_surface.plot_fermi_surface`), on the *half-open* `[0,1)^3`
periodic mesh convention (`analysis.dos._uniform_mesh`, the convention
the rest of `analysis/` uses) -- marching_cubes needs a periodic volume
it can `np.pad(..., mode="wrap")` itself.

Units: atomic units throughout (Hartree fermi_energy, Bohr^-1 reciprocal
lattice), like the rest of `analysis`.

The XCrySDen `.bxsf` file-writer path (`fermi_surface`/`write_bxsf`/
`fermi_surface_kgrid`, a closed (N+1)^3 grid convention reproducing
Wannier90's `plot_fermi_surface`) lives in
`waw.interfaces.wannier90.bxsf` instead -- it is a file-format interface
boundary (eV/Angstrom^-1 by XCrySDen/Wannier90 convention), not analysis
physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.hamiltonian import HamiltonianR, interpolate_bands
from .dos import _uniform_mesh


@dataclass
class FermiSheet:
    """One band's Fermi-surface isosurface: a triangle mesh with vertices
    welded (deduplicated) across adjacent triangles (`_weld_vertices`),
    needed for `waw.vis.fermi_surface.plot_fermi_surface` to smooth-shade
    it (Plotly's Mesh3d only averages normals where triangles share a
    vertex index; an unwelded mesh renders as flat facets)."""
    band_index: int
    vertices:   np.ndarray   # (nv, 3) float64, Cartesian Bohr^-1
    faces:      np.ndarray   # (nf, 3) int, indices into vertices


def _bz_facets(recip_lattice: np.ndarray):
    """(vertices, outward_normal) pairs for the first-BZ Voronoi facets of
    any reciprocal lattice, Cartesian, same units as `recip_lattice` --
    thin wrapper around `ase.dft.bz.bz_vertices`."""
    from ase.dft.bz import bz_vertices
    return bz_vertices(np.asarray(recip_lattice, dtype=np.float64))


def _in_bz_mask(points: np.ndarray, facets, tol: float = 1e-9) -> np.ndarray:
    """True where `points` (n, 3) Cartesian lie inside/on every Voronoi
    half-space defined by `facets` (`_bz_facets`'s return value), i.e.
    inside the first Brillouin zone."""
    mask = np.ones(len(points), dtype=bool)
    for verts, normal in facets:
        centroid = verts.mean(axis=0)
        mask &= (points - centroid) @ normal <= tol
    return mask


_REPLICA_SHIFTS = np.array(
    [[a, b, c] for a in (-1, 0) for b in (-1, 0) for c in (-1, 0)], dtype=np.float64
)


def _bz_planes(facets):
    """(centroid, unit outward normal) per BZ facet, from `_bz_facets`."""
    return [(v.mean(axis=0), n / np.linalg.norm(n)) for v, n in facets]


def _clip_polygon_to_halfspace(poly, centroid, nhat, tol):
    """
    Sutherland-Hodgman: clip a convex polygon (m, 3) to the half-space
    `(p - centroid) . nhat <= 0`, inserting exact edge/plane intersection
    points. Returns (m', 3); empty if the polygon is entirely outside.
    """
    d = (poly - centroid) @ nhat
    out = []
    m = len(poly)
    for i in range(m):
        j = (i + 1) % m
        di, dj = d[i], d[j]
        if di <= tol:
            out.append(poly[i])
        if (di < -tol and dj > tol) or (di > tol and dj < -tol):
            out.append(poly[i] + (di / (di - dj)) * (poly[j] - poly[i]))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def _clip_triangles_to_bz(tri, facets, tol: float = 1e-12):
    """
    Clip a triangle soup `tri` (ntri, 3, 3) exactly to the first Brillouin
    zone, rather than keeping or dropping whole triangles by centroid.

    Triangles fully inside are passed through untouched; those fully
    outside any single facet plane are dropped; only the few that straddle
    the boundary are polygon-clipped (against every facet) and
    fan-triangulated. That keeps the cost proportional to the boundary,
    not to the surface.

    The point is the neck openings: keeping whole straddling triangles
    lets the surface poke through the BZ face by up to a triangle's width
    (Cu overshot |Gamma-L| by 5%) and leaves the open ends visibly ragged.
    Exact clipping puts the rim precisely in the facet plane, which is
    what FermiSurfer-style renderings show.
    """
    tri = np.asarray(tri, dtype=np.float64)
    if len(tri) == 0:
        return tri
    planes = _bz_planes(facets)
    # signed distance of every vertex to every facet plane: (nplane, ntri, 3)
    d = np.stack([(tri.reshape(-1, 3) - c) @ nh for c, nh in planes])
    d = d.reshape(len(planes), len(tri), 3)
    inside = (d <= tol).all(axis=0).all(axis=1)          # all verts inside all planes
    dropped = (d > tol).all(axis=2).any(axis=0)          # some plane has all 3 verts outside
    straddle = ~inside & ~dropped

    pieces = [tri[inside]]
    for t in tri[straddle]:
        poly = t
        for c, nh in planes:
            poly = _clip_polygon_to_halfspace(poly, c, nh, tol)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            fan = np.stack([[poly[0], poly[j], poly[j + 1]]
                            for j in range(1, len(poly) - 1)])
            pieces.append(fan)
    return np.concatenate([p for p in pieces if len(p)], axis=0) if any(
        len(p) for p in pieces) else tri[:0]


def _weld_vertices(vertices: np.ndarray, faces: np.ndarray, decimals: int = 7):
    """
    Merge vertices that coincide to within `decimals` decimal places so
    adjacent triangles share a vertex index, not just an equal coordinate
    value. Re-expanding `marching_cubes`'s welded mesh into a per-triangle
    `(ntri, 3, 3)` array for the BZ-containment filter destroys that
    sharing; this re-welds it. A rounding tolerance is needed since
    vertices from different periodic replicas reach the same boundary
    point via different floating-point paths (`(v + shift_a) @ recip` vs.
    `(v + shift_b) @ recip`), equal only up to rounding.
    """
    rounded = np.round(vertices, decimals)
    uniq, inverse = np.unique(rounded, axis=0, return_inverse=True)
    new_faces = inverse[faces.reshape(-1)].reshape(faces.shape)
    return uniq, new_faces


def fermi_surface_sheets(
    hr:            HamiltonianR,
    recip_lattice: np.ndarray,
    fermi_energy:  float,
    mesh:          tuple[int, int, int] = (40, 40, 40),
    bands:         Sequence[int] | None = None,
    clip_to_bz:    bool = True,
) -> list[FermiSheet]:
    """
    Fermi-surface isosurface per band, as triangle meshes clipped to the
    first Brillouin zone -- the physics/geometry half of
    `waw.vis.fermi_surface.plot_fermi_surface` (that module only
    renders; this function does the interpolation + marching cubes +
    BZ folding).

    Interpolates `hr` on the half-open `[0,1)^3` periodic mesh
    (`analysis.dos._uniform_mesh`, distinct from `fermi_surface_kgrid`'s
    closed `(N+1)^3` .bxsf convention), periodically pads it
    (`np.pad(..., mode="wrap")`) so `skimage.measure.marching_cubes` sees
    a continuous field across the cell boundary, then replicates each
    band's raw marching-cubes patch over one shell of periodic images
    (which tile fractional [-1, 1]^3, covering the BZ) and clips the
    result to the true BZ (`_clip_triangles_to_bz`).

    **Open sheets stay open.** A Fermi surface that crosses a BZ facet --
    Cu's necks through the hexagonal L faces being the textbook case --
    is genuinely open there: it continues into the neighbouring zone, and
    nothing is added to cap it. The rim of such an opening is a real
    boundary edge of the triangle mesh, which is exactly what makes the
    necks read as tubes rather than bumps.

    A band is skipped entirely if `fermi_energy` doesn't lie strictly
    between its minimum and maximum interpolated eigenvalue anywhere on
    the mesh (no Fermi surface for that band) -- so a `fermi_energy`
    outside every band's range returns an empty list, not spurious
    sheets.

    Args:
      hr           : HamiltonianR from waw.core.hamiltonian.compute_hr
      recip_lattice: (3, 3) rows = b1,b2,b3, Bohr^-1 (waw's usual
                     convention)
      fermi_energy : Hartree (atomic units, like the rest of `analysis`)
      mesh         : (N1, N2, N3) interpolation grid density
      bands        : optional subset of band indices to consider (0-based,
                     into the sorted-ascending `interpolate_bands` output);
                     None = every band
      clip_to_bz   : clip triangles exactly at the BZ facets (default).
                     False falls back to keeping whole triangles whose
                     centroid is inside, which is marginally cheaper but
                     lets the surface overshoot the facet by up to a
                     triangle width and leaves open rims ragged.

    Note
    ----
    For a Fermi surface to be correct at all, the Wannier model must
    reproduce the ab-initio bands *at* E_F -- so the disentanglement
    frozen window has to extend through E_F, not stop below it. A window
    ending just under E_F silently moves states across it and can change
    the surface's topology: Cu's neck state at L (0.99 eV below E_F in
    DFT) came out 0.71 eV *above* it, closing all eight necks and turning
    the surface into a sphere.

    Returns a list of `FermiSheet`, one per band that has a nonempty
    surface inside the BZ (skipped bands are simply absent, not
    included as empty sheets).
    """
    from skimage.measure import marching_cubes

    recip_lattice = np.asarray(recip_lattice, dtype=np.float64)
    N1, N2, N3 = mesh
    kpts = _uniform_mesh(mesh)
    eig = interpolate_bands(hr, kpts).reshape(N1, N2, N3, -1)
    nw = eig.shape[-1]
    band_indices = range(nw) if bands is None else bands

    facets = _bz_facets(recip_lattice)

    sheets = []
    for ib in band_indices:
        vol = eig[:, :, :, ib]
        if not (vol.min() < fermi_energy < vol.max()):
            continue

        volp = np.pad(vol, ((0, 1), (0, 1), (0, 1)), mode="wrap")
        verts, faces, _, _ = marching_cubes(volp, level=fermi_energy)
        vfrac = verts / np.array([N1, N2, N3])

        kept_vertices, kept_faces, offset = [], [], 0
        for shift in _REPLICA_SHIFTS:
            vcart = (vfrac + shift) @ recip_lattice
            tri = vcart[faces]                                  # (ntri, 3, 3)
            if clip_to_bz:
                tri = _clip_triangles_to_bz(tri, facets)
                keep = np.ones(len(tri), dtype=bool)
            else:
                keep = _in_bz_mask(tri.mean(axis=1), facets)
            if not keep.any():
                continue
            tri_kept = tri[keep].reshape(-1, 3)                  # (3*nkeep, 3)
            n_new = tri_kept.shape[0]
            kept_vertices.append(tri_kept)
            kept_faces.append(np.arange(n_new).reshape(-1, 3) + offset)
            offset += n_new

        if not kept_vertices:
            continue
        v_all, f_all = _weld_vertices(
            np.concatenate(kept_vertices, axis=0), np.concatenate(kept_faces, axis=0),
        )
        sheets.append(FermiSheet(band_index=ib, vertices=v_all, faces=f_all))
    return sheets
