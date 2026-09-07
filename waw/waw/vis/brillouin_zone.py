"""
Generic first-Brillouin-zone wireframe geometry, for any crystal.

Thin wrapper around `ase.dft.bz.bz_vertices` (the same Voronoi/Wigner-Seitz
decomposition `ase.dft.kpoints.BandPath.plot()` uses internally). Purely
geometric: whatever units `recip_lattice` is given in (Bohr^-1 throughout
the rest of this project) are the units the returned edges come back in.
"""

from __future__ import annotations

import numpy as np


def bz_edges(recip_lattice: np.ndarray) -> list[np.ndarray]:
    """
    Edges of the first Brillouin zone (Wigner-Seitz cell of the
    reciprocal lattice) for ANY crystal -- one closed polygon loop per
    BZ facet.

    Args:
      recip_lattice: (3, 3) rows = b1, b2, b3, any consistent unit
                     (Bohr^-1 for waw's own convention).

    Returns a list of (n_i, 3) arrays, each a closed loop (first vertex
    repeated as the last) tracing one polyhedron facet -- ready to hand
    straight to a line-plotting call (`Scatter3d`, `Line3DCollection`,
    ...), Cartesian, same units as `recip_lattice`.
    """
    from ase.dft.bz import bz_vertices

    facets = bz_vertices(np.asarray(recip_lattice, dtype=np.float64))
    return [np.concatenate([verts, verts[:1]], axis=0) for verts, _normal in facets]
