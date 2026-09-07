"""
Fermi-surface isosurface geometry, on an analytic one-band model placed on
an fcc lattice -- so the BZ is a truncated octahedron whose facets are
oblique to the interpolation mesh, as in the real Cu case.

    E(k) = -2t [cos 2*pi*k1 + cos 2*pi*k2 + cos 2*pi*k3]     (k fractional)

giving E(Gamma) = -6t and a band maximum of +6t. With t = 1 a level of -3
leaves a CLOSED pocket well inside the zone, while +3 produces a surface
that crosses the hexagonal facets and must stay OPEN there.

That open case is the one that matters: Cu's eight necks pass through the
hexagonal L faces, and a renderer that caps them reports a featureless
sphere instead of the textbook surface.
"""

import collections

import numpy as np
import torch

from waw.analysis.fermi_surface import (
    _bz_facets, _bz_planes, fermi_surface_sheets,
)
from waw.core.hamiltonian import HamiltonianR

_MESH = (24, 24, 24)


def _fcc_1band(t: float = 1.0):
    """One-band nearest-neighbour H(R) on an fcc lattice (a = 1 Bohr)."""
    R = np.array([[0, 0, 0],
                  [1, 0, 0], [-1, 0, 0],
                  [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1]], dtype=np.int64)
    H_R = np.zeros((len(R), 1, 1), dtype=np.complex128)
    H_R[1:, 0, 0] = -t
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R,
                      degen=np.ones(len(R), dtype=np.int64), nw=1)
    real = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    recip = 2.0 * np.pi * np.linalg.inv(real).T
    return hr, recip


def _boundary_edges(sheet):
    """Edges used by exactly one triangle = the open rim of the mesh."""
    counts = collections.Counter()
    for a, b, c in sheet.faces:
        for e in ((a, b), (b, c), (c, a)):
            counts[(min(e), max(e))] += 1
    return np.array([k for k, n in counts.items() if n == 1], dtype=np.int64)


def _max_overshoot(vertices, planes):
    """Largest signed distance outside any BZ facet plane (<0 = all inside)."""
    return max(float(((vertices - c) @ nh).max()) for c, nh in planes)


def test_interior_pocket_is_a_closed_manifold():
    hr, recip = _fcc_1band()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=-3.0, mesh=_MESH)
    assert len(sheets) == 1
    assert len(_boundary_edges(sheets[0])) == 0
    planes = _bz_planes(_bz_facets(recip))
    assert _max_overshoot(sheets[0].vertices, planes) < 0.0    # strictly inside


def test_surface_crossing_a_facet_stays_open():
    """The Cu-neck case: the surface continues into the neighbouring zone,
    so its rim must remain an open boundary lying in the facet plane."""
    hr, recip = _fcc_1band()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=3.0, mesh=_MESH)
    assert len(sheets) == 1
    rim = _boundary_edges(sheets[0])
    assert len(rim) > 0, "the neck openings were capped shut"

    facets = _bz_facets(recip)
    planes = _bz_planes(facets)
    v = sheets[0].vertices
    mid = 0.5 * (v[rim[:, 0]] + v[rim[:, 1]])
    dist = np.min([np.abs((mid - c) @ nh) for c, nh in planes], axis=0)
    assert dist.max() < 1e-6                       # rim lies IN the facet planes

    # and on the hexagonal facets specifically, as Cu's necks do
    nearest = np.argmin([np.abs((mid - c) @ nh) for c, nh in planes], axis=0)
    n_hex = sum(1 for j in nearest if len(facets[j][0]) == 6)
    assert n_hex == len(rim)


def test_clipping_is_exact_where_the_centroid_test_overshoots():
    """clip_to_bz holds the mesh inside the BZ to machine precision. The
    centroid fallback keeps whole straddling triangles, so the surface pokes
    out by a fraction of a triangle -- on real Cu it passed |Gamma-L| by 5%
    and left the neck rims visibly ragged."""
    hr, recip = _fcc_1band()
    planes = _bz_planes(_bz_facets(recip))

    clipped = fermi_surface_sheets(hr, recip, 3.0, mesh=_MESH, clip_to_bz=True)[0]
    whole = fermi_surface_sheets(hr, recip, 3.0, mesh=_MESH, clip_to_bz=False)[0]

    assert _max_overshoot(clipped.vertices, planes) < 1e-6
    assert _max_overshoot(whole.vertices, planes) > 1e-2


def test_clipping_leaves_a_fully_interior_surface_untouched():
    hr, recip = _fcc_1band()
    a = fermi_surface_sheets(hr, recip, -3.0, mesh=(20, 20, 20), clip_to_bz=True)[0]
    b = fermi_surface_sheets(hr, recip, -3.0, mesh=(20, 20, 20), clip_to_bz=False)[0]
    assert len(a.faces) == len(b.faces)
    assert np.allclose(np.sort(np.linalg.norm(a.vertices, axis=1)),
                       np.sort(np.linalg.norm(b.vertices, axis=1)), atol=1e-9)


def test_no_sheet_when_the_level_misses_every_band():
    hr, recip = _fcc_1band()
    assert fermi_surface_sheets(hr, recip, -99.0, mesh=(12, 12, 12)) == []
    assert fermi_surface_sheets(hr, recip, +99.0, mesh=(12, 12, 12)) == []
