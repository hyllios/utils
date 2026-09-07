"""
Density-functional theory for superconductors (SCDFT), band-resolved isotropic,
with the Sanna-Pellegrini-Gross exchange-correlation functional
(Phys. Rev. Lett. 125, 057001 (2020)).

`tc_scdft` gives Tc from the equation linearized in the pairing potential;
`solve_delta_s` solves it self-consistently for Delta_s(xi).

Pass ``functional="lm2005"`` to any of them for the older Luders-Marques
functional (Phys. Rev. B 72, 024545 (2005)) instead of the default "spg", or
``functional="akashi"`` for Akashi and Arita's particle-hole asymmetric
renormalisation kernel (Phys. Rev. B 88, 014514 (2013)) -- the new Z of their
Eq. (40) with LM2005's pairing kernel, which is the combination their own model
calculation uses. It needs a `dos_ratio` callable to do anything: its extra term
is odd in xi' and vanishes for a symmetric DOS. The
two differ in BOTH kernels: LM2005 repairs its renormalisation kernel with the
J of its Eqs. (80)-(81), giving Z(0) ~ lambda, where SPG keeps the unrepaired
symmetrised form with Z(0) = 2*lambda; and LM2005's pairing kernel is its
Eq. (74), built from I with no fitted constants, where SPG's is its Eq. (11)
with gamma1, gamma2, gamma3.

`tc_unexpanded` is a third, EXPERIMENTAL construction living in
`unexpanded.py`: the same Sham-Schlueter condition solved without replacing
the interacting propagators at all, closed by a small Krylov profile basis
projected in the metric the SS condition supplies. It is multiband, and at
its default settings it reproduces band-limited Migdal-Eliashberg Tc to 1e-4
on every model tested (Einstein and shaped spectra, a two-band MgB2-like
lambda matrix, bare static mu with no mu*). It is a separate entry point
rather than another `functional=` value because it is built in Matsubara
space instead of from the I and J functions, and because it has been
validated against Eliashberg on models plus one real material (Al) -- not
against experiment. Read its module docstring before using it.

`tc_analytic` (in `analytic.py`) is the STANDALONE counterpart: explicit
xi-space kernels in the LM2005/SPG class, built with Z-dressed lines (which
removes the 2-lambda defect at the root) and an arbitrary statically screened
Coulomb W(xi,xi') instead of a scalar mu -- exact in the Coulomb channel for
any static W. No Eliashberg kernel is applied anywhere, nothing is fitted,
and no Matsubara kernel matrix is formed, so it has no low-temperature floor
and handles multi-eV band widths. Measured accuracy: Tc within 0.97-1.03 of
band-limited Migdal-Eliashberg on the Einstein grid, similar on structured-W,
shaped-spectra and multiband tests. Also EXPERIMENTAL; read its docstring.

Inputs match the Eliashberg module: alpha^2F as a single spectrum or a band
matrix, plus a dimensionless Coulomb parameter (scalar or matrix) standing in
for the paper's static-RPA screened interaction. Hartree in, Kelvin out.

Delta_s is the Kohn-Sham PAIRING POTENTIAL and not an excitation gap -- the
paper's own exact result is that it vanishes at E_F as T -> 0.
"""

from .analytic import (AnalyticKernels, build_analytic,
                       linearized_eigenvalue_analytic, tc_analytic)
from .functions import (bose, fermi, i_akashi, i_function, j_akashi,
                        j_function, j_lueders, p_smooth)
from .unexpanded import (
    UnexpandedKernels,
    build_kernels,
    linearized_eigenvalue_unexpanded,
    n_half_for,
    tc_unexpanded,
)
from .solver import (
    SCDFT_GAMMA4_GAP,
    SCDFT_GAMMAS_KERNEL,
    SCDFT_GAMMAS_SCTK,
    SCDFT_GAMMAS_TC,
    ScdftResult,
    energy_grid,
    gap_operator,
    linearized_eigenvalue,
    solve_delta_s,
    tc_scdft,
    z_kernel,
)

__all__ = ["fermi", "bose", "i_function", "j_function", "j_lueders", "energy_grid",
           "z_kernel", "gap_operator", "linearized_eigenvalue", "tc_scdft",
           "solve_delta_s", "ScdftResult", "SCDFT_GAMMAS_KERNEL", "SCDFT_GAMMAS_TC",
           "SCDFT_GAMMAS_SCTK", "SCDFT_GAMMA4_GAP",
           "tc_unexpanded", "build_kernels", "UnexpandedKernels",
           "linearized_eigenvalue_unexpanded", "n_half_for",
           "tc_analytic", "build_analytic", "AnalyticKernels",
           "linearized_eigenvalue_analytic"]
