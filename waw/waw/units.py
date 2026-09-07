"""
Unit conversions — the single source of truth for waw.

`waw.core` and `waw.analysis` work exclusively in atomic units (Bohr,
Hartree, Bohr^-1, Bohr^2, ...). Conversion to/from physical units
(Angstrom, eV, SI) happens only at the interface boundary, via
`to_si_units`/`to_eVA_units` (atomic -> physical) and
`from_si_units`/`from_eVA_units` (physical -> atomic).

Bohr/Hartree factors come from ``ase.units`` (CODATA), with a numeric
fallback so the core remains importable without ASE. The SI base
constants below (`E_CHARGE`, `HBAR_SI`, ...) are CODATA-2018-exact rather
than routed through ``ase.units``, whose `_e`/`_hbar` are older CODATA-2014
values.
"""

from __future__ import annotations

import math
from typing import Callable

try:
    from ase.units import Bohr as _BOHR_IN_ANG, Hartree as _HARTREE_IN_EV
except ImportError:  # numeric fallback (CODATA), keeps core importable sans ASE
    _BOHR_IN_ANG = 0.5291772105638411
    _HARTREE_IN_EV = 27.211386024367243

# Length: 1 Bohr in Angstrom, and its inverse.
BOHR_TO_ANG = _BOHR_IN_ANG
ANG_TO_BOHR = 1.0 / _BOHR_IN_ANG

# Energy: 1 Hartree in eV, and its inverse.
HARTREE_TO_EV = _HARTREE_IN_EV
EV_TO_HARTREE = 1.0 / _HARTREE_IN_EV

# SI base constants, CODATA-2018/SI-exact.
E_CHARGE      = 1.602176634e-19      # C, elementary charge (exact, SI)
HBAR_SI       = 1.054571817e-34      # J s (CODATA 2018)
BOHR_RADIUS_M = 5.29177210903e-11    # m, Bohr radius (CODATA 2018)
HARTREE_TO_J  = 4.3597447222071e-18  # J, Hartree energy (CODATA 2018)
K_B_SI        = 1.380649e-23         # J/K, Boltzmann constant (exact, SI)
EPS0_SI       = 8.8541878128e-12     # F/m, vacuum permittivity (CODATA 2018)
K_B_HARTREE   = K_B_SI / HARTREE_TO_J   # Boltzmann constant, Hartree/Kelvin
AMU_TO_KG     = 1.66053906660e-27    # kg, unified atomic mass unit (CODATA 2018)
PLANCK_H_SI   = 6.62607015e-34       # J s, Planck constant (exact, 2019 SI redefinition)
C_LIGHT_CM_PER_S = 2.99792458e10     # cm/s, speed of light (exact, SI)
ELECTRON_MASS_KG = 9.1093837015e-31  # kg, electron mass (CODATA 2018)
VACUUM_PERMITTIVITY_SI = 8.8541878128e-12  # F/m, electric constant (CODATA 2018)

# Phonon wavenumber (cm^-1, the standard spectroscopic/lattice-dynamics
# practical unit) <-> Hartree, via E = h*c*nu_cm1 (E in Hartree when hbar=1).
HARTREE_TO_CM1 = HARTREE_TO_J / (PLANCK_H_SI * C_LIGHT_CM_PER_S)
CM1_TO_HARTREE = 1.0 / HARTREE_TO_CM1

# Mass: amu <-> atomic units (electron mass = 1) -- needed wherever a mass
# entering in physical amu (e.g. `interfaces.quantum_espresso.phonon_io`'s
# `masses_amu`) must combine with genuinely atomic-units quantities
# (Hartree, Bohr, hbar=1) in the SAME formula, e.g. `analysis.elph`'s
# phonon-normal-mode electron-phonon coupling.
#: Atomic unit of electric field, E_h / (e a_0) = 5.14220675e11 V/m. Derived
#: rather than quoted so it cannot drift from the constants above.
AU_FIELD_V_PER_M = HARTREE_TO_J / (E_CHARGE * BOHR_RADIUS_M)

#: Atomic unit of magnetic flux density, hbar / (e a_0^2) = 2.350517e5 T.
#: Derived, so it cannot drift from the constants above. The atomic unit of
#: magnetic FLUX is then hbar/e, making the superconducting flux quantum
#: Phi_0 = h/2e = pi in atomic units -- which is why `analysis.critical_fields`
#: can write Hc2 = 1/(2 xi^2).
AU_B_FIELD_TESLA = HBAR_SI / (E_CHARGE * BOHR_RADIUS_M ** 2)

#: Fine-structure constant, derived. In atomic units the speed of light is
#: 1/alpha and the vacuum permeability is mu_0 = 4 pi alpha^2 (with
#: eps_0 = 1/4pi), which is how magnetostatics enters `critical_fields`.
FINE_STRUCTURE = (E_CHARGE ** 2
                  / (4.0 * math.pi * VACUUM_PERMITTIVITY_SI * HBAR_SI
                     * C_LIGHT_CM_PER_S * 1e-2))

AMU_TO_ME = AMU_TO_KG / ELECTRON_MASS_KG
ME_TO_AMU = 1.0 / AMU_TO_ME


def bohr_to_ang(x):
    """Convert a length (or array of lengths) from Bohr to Angstrom."""
    return x * BOHR_TO_ANG


def ang_to_bohr(x):
    """Convert a length (or array of lengths) from Angstrom to Bohr."""
    return x * ANG_TO_BOHR


def hartree_to_ev(x):
    """Convert an energy (or array of energies) from Hartree to eV."""
    return x * HARTREE_TO_EV


def ev_to_hartree(x):
    """Convert an energy (or array of energies) from eV to Hartree."""
    return x * EV_TO_HARTREE


def hartree_to_cm1(x):
    """Convert an energy (or array of energies) from Hartree to cm^-1."""
    return x * HARTREE_TO_CM1


def cm1_to_hartree(x):
    """Convert an energy (or array of energies) from cm^-1 to Hartree."""
    return x * CM1_TO_HARTREE


# ---------------------------------------------------------------------------
# Named-quantity dispatchers.
#
# `analysis/` functions return raw atomic-units quantities (e.g. a bare
# Berry-curvature average in Bohr^2), with renormalization by cell volume
# or temperature and the SI/CGS unit-system prefactor applied here, given
# as explicit keyword context (`cell_volume_bohr3`, `temperature_kelvin`,
# ...).
#
# Each quantity's converter is registered by the module owning the physics
# (`analysis/topology.py`, `analysis/spin_hall.py`, ...) via
# `register_si_unit`/`register_eva_unit`; this module only owns the
# dispatch mechanism and the constants above. Most quantities also
# register the inverse conversion via `register_from_si_unit`/
# `register_from_eva_unit`.
# ---------------------------------------------------------------------------

SI_CONVERTERS:  dict[str, Callable] = {}
EVA_CONVERTERS: dict[str, Callable] = {}
FROM_SI_CONVERTERS:  dict[str, Callable] = {}
FROM_EVA_CONVERTERS: dict[str, Callable] = {}


def register_si_unit(quantity: str):
    """Decorator: register `fn` as the atomic-units -> SI converter for `quantity`."""
    def _decorator(fn):
        SI_CONVERTERS[quantity] = fn
        return fn
    return _decorator


def register_eva_unit(quantity: str):
    """Decorator: register `fn` as the atomic-units -> eV/Angstrom converter for `quantity`."""
    def _decorator(fn):
        EVA_CONVERTERS[quantity] = fn
        return fn
    return _decorator


def register_from_si_unit(quantity: str):
    """Decorator: register `fn` as the SI -> atomic-units converter for `quantity`
    (the inverse of `register_si_unit`'s converter)."""
    def _decorator(fn):
        FROM_SI_CONVERTERS[quantity] = fn
        return fn
    return _decorator


def register_from_eva_unit(quantity: str):
    """Decorator: register `fn` as the eV/Angstrom -> atomic-units converter
    for `quantity` (the inverse of `register_eva_unit`'s converter)."""
    def _decorator(fn):
        FROM_EVA_CONVERTERS[quantity] = fn
        return fn
    return _decorator


def to_si_units(value, quantity: str, **kwargs):
    """
    Convert an atomic-units `value` to full SI units for the named
    `quantity` (e.g. ``"hall_conductivity"``, ``"specific_heat"``, ...).
    See the quantity's owning `analysis/` module for which extra keyword
    context it needs (typically `cell_volume_bohr3` and/or
    `temperature_kelvin`).
    """
    try:
        fn = SI_CONVERTERS[quantity]
    except KeyError:
        raise ValueError(
            f"unknown SI quantity {quantity!r}; known quantities: "
            f"{sorted(SI_CONVERTERS)}"
        ) from None
    return fn(value, **kwargs)


def to_eVA_units(value, quantity: str, **kwargs):
    """
    Convert an atomic-units `value` to eV/Angstrom-based practical units
    for the named `quantity` (e.g. ``"energy"``, ``"length"``, ``"area"``,
    ``"volume"``).
    """
    try:
        fn = EVA_CONVERTERS[quantity]
    except KeyError:
        raise ValueError(
            f"unknown eV/Angstrom quantity {quantity!r}; known quantities: "
            f"{sorted(EVA_CONVERTERS)}"
        ) from None
    return fn(value, **kwargs)


def from_si_units(value, quantity: str, **kwargs):
    """
    Convert a full-SI-units `value` back to atomic units for the named
    `quantity` -- the inverse of `to_si_units`. Needs the same extra
    keyword context the forward conversion does (e.g. `cell_volume_bohr3`,
    `kT_values`), so `to_si_units(from_si_units(x, q, **kw), q, **kw) == x`
    for every registered quantity except "thermoelectric_conductivity"
    (see `analysis.boltzmann`'s inverse converter).
    """
    try:
        fn = FROM_SI_CONVERTERS[quantity]
    except KeyError:
        raise ValueError(
            f"unknown SI quantity {quantity!r}; known quantities: "
            f"{sorted(FROM_SI_CONVERTERS)}"
        ) from None
    return fn(value, **kwargs)


def from_eVA_units(value, quantity: str, **kwargs):
    """
    Convert an eV/Angstrom-based `value` back to atomic units for the
    named `quantity` -- the inverse of `to_eVA_units`.
    """
    try:
        fn = FROM_EVA_CONVERTERS[quantity]
    except KeyError:
        raise ValueError(
            f"unknown eV/Angstrom quantity {quantity!r}; known quantities: "
            f"{sorted(FROM_EVA_CONVERTERS)}"
        ) from None
    return fn(value, **kwargs)


@register_eva_unit("energy")
def _energy_to_eva(value):
    return value * HARTREE_TO_EV


@register_eva_unit("length")
def _length_to_eva(value):
    return value * BOHR_TO_ANG


@register_eva_unit("area")
def _area_to_eva(value):
    return value * BOHR_TO_ANG ** 2


@register_eva_unit("volume")
def _volume_to_eva(value):
    return value * BOHR_TO_ANG ** 3


# Berry curvature is conventionally reported as an area (Bohr^2 -> Ang^2),
# same conversion as "area" -- an alias so callers can use the physically
# meaningful name.
EVA_CONVERTERS["berry_curvature"] = EVA_CONVERTERS["area"]


@register_from_eva_unit("energy")
def _energy_from_eva(value):
    return value * EV_TO_HARTREE


@register_from_eva_unit("length")
def _length_from_eva(value):
    return value * ANG_TO_BOHR


@register_from_eva_unit("area")
def _area_from_eva(value):
    return value * ANG_TO_BOHR ** 2


@register_from_eva_unit("volume")
def _volume_from_eva(value):
    return value * ANG_TO_BOHR ** 3


FROM_EVA_CONVERTERS["berry_curvature"] = FROM_EVA_CONVERTERS["area"]
