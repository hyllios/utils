"""
waw.interfaces.ase — the ASE-based interface.

Structure/k-mesh helpers (``structure``) and a numpy/.npz driver (``driver``)
around the atomic-unit core.  Uses ASE for structures and Brillouin-zone paths
and numpy for on-disk data, keeping all eV/Angstrom <-> Hartree/Bohr conversions
at this boundary.

Typical usage::

    from waw.interfaces.ase import structure, wannierize
    result = wannierize(atoms, mp_grid, kpts,
                        mmn=mmn, amn=amn, eig=eig,
                        nnkpts=nnkpts, g_vectors=g_vectors,
                        outer_window=(-1e3, 38.0))
"""

from . import structure
from .structure import real_lattice, recip_lattice, monkhorst_pack, band_path, band_path_segments
from .driver import (
    wannierize,
    build_wannier_data,
    wannier_data_from_atoms,
    nnkp_from_kpb_map,
    save_npz,
    load_npz,
)

__all__ = [
    "structure",
    "real_lattice",
    "recip_lattice",
    "monkhorst_pack",
    "band_path",
    "band_path_segments",
    "wannierize",
    "build_wannier_data",
    "wannier_data_from_atoms",
    "nnkp_from_kpb_map",
    "save_npz",
    "load_npz",
]
