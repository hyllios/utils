"""
waw.interfaces.wannier90 — the legacy Wannier90 file interface.

Reads the Wannier90 input files (.win/.eig/.mmn/.amn/.nnkp/UNK) and writes the
usual outputs (_hr.dat/_centres.xyz/.chk.fmt/.xsf/.bxsf).  This layer is the
eV/Angstrom <-> atomic-unit boundary for the Wannier90 workflow.
"""

from .loader   import load, parse_real_lattice, parse_recip_lattice, parse_atoms
from .pipeline import wannierize, WannierResult
from .realspace import (
    build_wannier_functions,
    write_xsf,
    plot_wannier_functions,
    RealSpaceWF,
)
from .bxsf import (
    fermi_surface_kgrid,
    write_bxsf,
    fermi_surface,
)

__all__ = [
    "load",
    "parse_real_lattice",
    "parse_recip_lattice",
    "parse_atoms",
    "wannierize",
    "WannierResult",
    "build_wannier_functions",
    "write_xsf",
    "plot_wannier_functions",
    "RealSpaceWF",
    "fermi_surface_kgrid",
    "write_bxsf",
    "fermi_surface",
]
