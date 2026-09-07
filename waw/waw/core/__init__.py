"""
waw.core — the numerical engine, in atomic units only.

Disentanglement, wannierisation (spread minimization), the k-mesh geometry,
and the real-space Hamiltonian H(R).  Everything here speaks atomic units
(Bohr, Bohr^-1, Bohr^2; energies in Hartree) and plain tensors/arrays;
file formats and physical-unit conversions live in the interface layers.
"""

from .types       import WannierData
from .kmesh       import _compute_bvecs_and_weights, _nnkp_from_mmn, generate_nnkp
from .spread      import (
    compute_spread,
    compute_spread_from_M_tilde,
    compute_wannier_centres,
    rotate_overlaps,
)
from .init        import svd_init, random_unitary
from .optim       import minimize_spread, SpreadResult
from .global_optim import global_minimize_spread
from .disentangle import disentangle, DisentangleResult
from .hamiltonian import (
    compute_hr, interpolate_bands, operator_k, HamiltonianR, compute_operator_r,
    compute_position_r, position_operator_k,
)
from .ws_distance import WsDistance, build_ws_distance
from .window      import auto_window, WindowResult
from .pipeline    import wannierize, WannierResult
from .distributions import (
    fermi_dirac,
    minus_fermi_deriv,
    dfermi_dE,
    bose_einstein,
    gaussian_smearing,
)

__all__ = [
    "WannierData",
    "generate_nnkp",
    "wannierize",
    "WannierResult",
    "compute_spread",
    "compute_spread_from_M_tilde",
    "compute_wannier_centres",
    "rotate_overlaps",
    "svd_init",
    "random_unitary",
    "minimize_spread",
    "SpreadResult",
    "global_minimize_spread",
    "disentangle",
    "DisentangleResult",
    "compute_hr",
    "interpolate_bands",
    "operator_k",
    "HamiltonianR",
    "compute_operator_r",
    "compute_position_r",
    "position_operator_k",
    "WsDistance",
    "build_ws_distance",
    "auto_window",
    "WindowResult",
    "fermi_dirac",
    "minus_fermi_deriv",
    "dfermi_dE",
    "bose_einstein",
    "gaussian_smearing",
]
