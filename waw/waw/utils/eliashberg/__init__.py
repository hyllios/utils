"""
Band-resolved isotropic Eliashberg solver.

Starts from alpha^2F -- a single spectrum or a band-resolved matrix
alpha^2F_ij(omega) -- plus a Coulomb pseudopotential mu* (scalar or matrix),
and returns Tc from the LINEARIZED equations, where the vanishing of the gap
turns the problem into an eigenvalue condition rho(Tc) = 1.

    from waw.utils.eliashberg import tc_linearized

    res = tc_linearized(a2f, omega, mu_star=0.11, omega_c=omega_c)
    print(res.tc, 'K')

``a2f`` is ``(n_omega,)`` or ``(nb, nb, n_omega)`` and ``omega`` is in HARTREE,
this project's atomic-unit convention; temperatures come back in Kelvin. That
is the same footing `waw.analysis.elph.alpha2f` and `alpha2f_matrix` produce,
so their output feeds straight in.

Cross-checked against an independent Fortran implementation of the FULL
nonlinear equations on two systems: CaC6 (single band, its 14.149 K vs 14.099 K
here) and MgB2 (2x2 sigma/pi matrix, 32.011 K vs 31.907 K), with lambda
agreeing to 4e-16 under matched quadrature. The digitised alpha^2F for both is
archived in tests/data/eliashberg/ and the comparison is pinned in
tests/test_utils_eliashberg.py.

`nonlinear.py` solves the full equations at finite T: `solve_gap` for one
temperature, `gap_vs_temperature` for Delta(T). Both MgB2 gaps come out
separately and match the reference solver to better than 0.1%.
"""

from .kernels import (
    coulomb_weights,
    isotropic_average,
    lambda_kernel,
    lambda_plus_minus,
    mass_renormalization_normal_state,
    matsubara_frequencies,
    rescale_mu_star,
)
from .nonlinear import (
    GapResult,
    GapVsTemperature,
    gap_vs_temperature,
    solve_gap,
)
from .linearized import (
    LinearizedResult,
    TcResult,
    leading_eigenvalue,
    linearized_kernel,
    tc_linearized,
)

__all__ = [
    "matsubara_frequencies", "lambda_kernel", "lambda_plus_minus",
    "mass_renormalization_normal_state", "coulomb_weights", "isotropic_average", "rescale_mu_star",
    "LinearizedResult", "TcResult", "linearized_kernel", "leading_eigenvalue",
    "tc_linearized",
    "GapResult", "GapVsTemperature", "solve_gap", "gap_vs_temperature",
]
