"""
waw — Wannier Analysis Workstation.

A PyTorch-based Wannier function optimization engine and analysis toolkit:
a drop-in replacement for the Wannier90 minimization engine (reads the same
.win/.eig/.mmn/.amn/.nnkp inputs, writes the same _hr.dat/_centres.xyz
outputs), plus an analysis layer for band structure, DOS, effective mass,
Berry curvature / anomalous Hall conductivity, spin texture, Boltzmann and
ballistic transport, surface spectral functions, and alloy CPA.

Typical usage::

    from waw import wannierize
    result = wannierize("path/to/silicon", nw=4)
    bands  = result.hr.interpolate(kpath)
"""

from . import analysis
from .parallel          import set_num_threads, get_num_threads, default_num_threads
from .core.types        import WannierData
from .core.optim        import minimize_spread, SpreadResult
from .core.global_optim import global_minimize_spread
from .core.disentangle  import disentangle, DisentangleResult
from .core.hamiltonian  import compute_hr, interpolate_bands, HamiltonianR
from .core.init         import svd_init, random_unitary
from .core.spread       import compute_spread, compute_spread_from_M_tilde, rotate_overlaps
from .core.window       import auto_window, WindowResult
from .interfaces.wannier90.pipeline  import wannierize, WannierResult
from .core.pipeline                  import save_wannier_result, load_wannier_result
from .interfaces.wannier90.loader    import load, parse_real_lattice, parse_recip_lattice, parse_atoms
from .interfaces.wannier90.realspace import (
    build_wannier_functions, write_xsf, plot_wannier_functions, RealSpaceWF,
)
from .vis.bands import plot_bands, BandSeries

__all__ = [
    # analysis subpackage
    "analysis",
    # CPU thread configuration
    "set_num_threads",
    "get_num_threads",
    "default_num_threads",
    # pipeline
    "wannierize",
    "WannierResult",
    "save_wannier_result",
    "load_wannier_result",
    # window selection
    "auto_window",
    "WindowResult",
    # data
    "WannierData",
    "load",
    "parse_real_lattice",
    "parse_recip_lattice",
    "parse_atoms",
    # optimization
    "minimize_spread",
    "global_minimize_spread",
    "SpreadResult",
    # disentanglement
    "disentangle",
    "DisentangleResult",
    # Hamiltonian
    "compute_hr",
    "interpolate_bands",
    "HamiltonianR",
    # initialization
    "svd_init",
    "random_unitary",
    # spread functional
    "compute_spread",
    "compute_spread_from_M_tilde",
    "rotate_overlaps",
    # real-space Wannier functions from UNK files
    "build_wannier_functions",
    "write_xsf",
    "plot_wannier_functions",
    "RealSpaceWF",
    # band-structure plotting
    "plot_bands",
    "BandSeries",
]
