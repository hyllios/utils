"""
Electronic specific heat (heat capacity at fixed chemical potential),
following Phoebe's `src/observable/specific_heat.cpp` (Cepellotti et al.,
J. Phys. Mater. 5 (2022) 035003):

    C_v(mu, kT) = (gs / V) * (1/Nk) * sum_{k,n} (eps_kn - mu)^2 *
                  (-df/dE)(eps_kn; mu, kT)

the standard grand-canonical (fixed-mu, not fixed-N) electronic heat
capacity -- a direct k-mesh sum weighted by -df/dE (no energy-grid/TDF
machinery needed, since there is no velocity factor). Uses the same
`core.distributions.minus_fermi_deriv` as boltzmann.py.

UNITS. Atomic units: mu, kT in Hartree; `electronic_specific_heat`
returns Hartree/Bohr^3 (energy density per Kelvin of "kT", not yet per
Kelvin of real temperature). C_v needs a real temperature (Kelvin) to
become a physical heat capacity (dU/dT needs literal T), so
`waw.units.to_si_units(c_v_reduced, "specific_heat", kT=...)` takes `kT`
(same Hartree value passed to `electronic_specific_heat`) and recovers
Kelvin via `K_B_HARTREE`.
"""

from __future__ import annotations

import numpy as np

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import minus_fermi_deriv
from ..units import (
    HARTREE_TO_J, BOHR_RADIUS_M, K_B_HARTREE,
    register_si_unit, register_from_si_unit,
)
from .boltzmann import band_velocities
from .dos import _uniform_mesh


def electronic_specific_heat(
    hr:                 HamiltonianR,
    real_lattice:        np.ndarray,
    recip_lattice:        np.ndarray,
    mesh:                tuple[int, int, int],
    mu:                  float,
    kT:                  float,
    num_elec_per_state:  int = 2,
) -> float:
    """
    C_v_reduced(mu, kT), Hartree/Bohr^3 (see module docstring). Provably
    >= 0 for any Hamiltonian: (eps-mu)^2 >= 0 and -df/dE >= 0 term by
    term, so the sum is too -- asserted internally as a regression guard.
    """
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    V = abs(np.linalg.det(real_lattice))
    kpts = _uniform_mesh(mesh)
    nk = kpts.shape[0]
    eig, _ = band_velocities(hr, kpts, recip_lattice)

    weight = minus_fermi_deriv(eig, mu, kT)           # (nk, nw), >= 0
    c_v = num_elec_per_state / (V * nk) * np.sum((eig - mu) ** 2 * weight)

    assert c_v >= -1e-8 * max(abs(c_v), 1e-30), \
        "electronic_specific_heat came out negative -- see module docstring's positivity proof"
    return float(c_v)


@register_si_unit("specific_heat")
def _specific_heat_to_si(c_v_reduced, *, kT: float):
    """
    Convert `electronic_specific_heat`'s result to SI (J / K / m^3).

    c_v_reduced * HARTREE_TO_J / BOHR_RADIUS_M^3 gives an energy density
    [J/m^3]; dividing by the real temperature in Kelvin (`kT/K_B_HARTREE`)
    gives [J/K/m^3].
    """
    temperature_kelvin = kT / K_B_HARTREE
    return c_v_reduced * HARTREE_TO_J / (BOHR_RADIUS_M ** 3 * temperature_kelvin)


@register_from_si_unit("specific_heat")
def _specific_heat_from_si(c_v_si, *, kT: float):
    """Inverse of `_specific_heat_to_si` -- same `kT` (Hartree) kwarg."""
    temperature_kelvin = kT / K_B_HARTREE
    return c_v_si * (BOHR_RADIUS_M ** 3 * temperature_kelvin) / HARTREE_TO_J
