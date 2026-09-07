"""
waw/units.py: the atomic-units <-> eV/Angstrom/SI conversion boundary.
"""

import numpy as np
import pytest

from waw import units

# Import every analysis module that registers a quantity via
# @register_si_unit/@register_eva_unit (and now @register_from_si_unit/
# @register_from_eva_unit) -- these are lazily registered on import (not
# all pulled in by `waw.analysis`'s own __init__), so the round-trip test
# below needs them all imported at least once first.
from waw.analysis import (   # noqa: F401
    kdotp, viscosity, superfluid, boltzmann, spin_hall, specific_heat,
    shift_current, gyrotropic, topology,
)


def test_bohr_ang_hartree_ev_are_inverses():
    assert units.BOHR_TO_ANG * units.ANG_TO_BOHR == pytest.approx(1.0)
    assert units.HARTREE_TO_EV * units.EV_TO_HARTREE == pytest.approx(1.0)
    assert units.bohr_to_ang(units.ang_to_bohr(3.7)) == pytest.approx(3.7)
    assert units.hartree_to_ev(units.ev_to_hartree(3.7)) == pytest.approx(3.7)


def test_si_constants_are_self_consistent():
    # K_B_HARTREE is derived from K_B_SI/HARTREE_TO_J, not an independent literal.
    assert units.K_B_HARTREE == pytest.approx(units.K_B_SI / units.HARTREE_TO_J)
    # sanity: all positive, roughly the expected order of magnitude (CODATA-2018).
    for name in ("E_CHARGE", "HBAR_SI", "BOHR_RADIUS_M", "HARTREE_TO_J",
                 "K_B_SI", "EPS0_SI"):
        assert getattr(units, name) > 0


def test_to_eva_units_energy_length_area_volume():
    assert units.to_eVA_units(2.0, "energy") == pytest.approx(2.0 * units.HARTREE_TO_EV)
    assert units.to_eVA_units(2.0, "length") == pytest.approx(2.0 * units.BOHR_TO_ANG)
    assert units.to_eVA_units(2.0, "area") == pytest.approx(2.0 * units.BOHR_TO_ANG ** 2)
    assert units.to_eVA_units(2.0, "volume") == pytest.approx(2.0 * units.BOHR_TO_ANG ** 3)


def test_berry_curvature_is_an_area_alias():
    assert units.to_eVA_units(5.0, "berry_curvature") == units.to_eVA_units(5.0, "area")


def test_to_eva_units_unknown_quantity_raises_with_known_list():
    with pytest.raises(ValueError, match="unknown eV/Angstrom quantity"):
        units.to_eVA_units(1.0, "not_a_real_quantity")


def test_to_si_units_unknown_quantity_raises_with_known_list():
    with pytest.raises(ValueError, match="unknown SI quantity"):
        units.to_si_units(1.0, "not_a_real_quantity")


def test_register_si_unit_decorator_adds_to_registry():
    @units.register_si_unit("_test_only_quantity")
    def _converter(value, scale=1.0):
        return value * scale

    try:
        assert units.SI_CONVERTERS["_test_only_quantity"] is _converter
        assert units.to_si_units(3.0, "_test_only_quantity", scale=2.0) == pytest.approx(6.0)
    finally:
        del units.SI_CONVERTERS["_test_only_quantity"]


def test_register_eva_unit_decorator_adds_to_registry():
    @units.register_eva_unit("_test_only_quantity")
    def _converter(value):
        return value * 10.0

    try:
        assert units.EVA_CONVERTERS["_test_only_quantity"] is _converter
        assert units.to_eVA_units(3.0, "_test_only_quantity") == pytest.approx(30.0)
    finally:
        del units.EVA_CONVERTERS["_test_only_quantity"]


# ---------------------------------------------------------------------------
# from_si_units / from_eVA_units -- the inverse direction
# ---------------------------------------------------------------------------

def test_from_eva_units_energy_length_area_volume():
    assert units.from_eVA_units(2.0, "energy") == pytest.approx(2.0 * units.EV_TO_HARTREE)
    assert units.from_eVA_units(2.0, "length") == pytest.approx(2.0 * units.ANG_TO_BOHR)
    assert units.from_eVA_units(2.0, "area") == pytest.approx(2.0 * units.ANG_TO_BOHR ** 2)
    assert units.from_eVA_units(2.0, "volume") == pytest.approx(2.0 * units.ANG_TO_BOHR ** 3)


def test_from_eva_units_berry_curvature_is_an_area_alias():
    assert units.from_eVA_units(5.0, "berry_curvature") == units.from_eVA_units(5.0, "area")


def test_from_eva_units_unknown_quantity_raises_with_known_list():
    with pytest.raises(ValueError, match="unknown eV/Angstrom quantity"):
        units.from_eVA_units(1.0, "not_a_real_quantity")


def test_from_si_units_unknown_quantity_raises_with_known_list():
    with pytest.raises(ValueError, match="unknown SI quantity"):
        units.from_si_units(1.0, "not_a_real_quantity")


def test_register_from_si_unit_decorator_adds_to_registry():
    @units.register_from_si_unit("_test_only_quantity")
    def _converter(value, scale=1.0):
        return value / scale

    try:
        assert units.FROM_SI_CONVERTERS["_test_only_quantity"] is _converter
        assert units.from_si_units(6.0, "_test_only_quantity", scale=2.0) == pytest.approx(3.0)
    finally:
        del units.FROM_SI_CONVERTERS["_test_only_quantity"]


def test_register_from_eva_unit_decorator_adds_to_registry():
    @units.register_from_eva_unit("_test_only_quantity")
    def _converter(value):
        return value / 10.0

    try:
        assert units.FROM_EVA_CONVERTERS["_test_only_quantity"] is _converter
        assert units.from_eVA_units(30.0, "_test_only_quantity") == pytest.approx(3.0)
    finally:
        del units.FROM_EVA_CONVERTERS["_test_only_quantity"]


# Representative kwargs for every quantity that needs extra context, so a
# single round-trip loop can exercise all of them uniformly. Values are
# arbitrary but physically sane orders of magnitude.
_CELL_VOLUME = 500.0   # Bohr^3
_KT_VALUES = np.array([300.0, 500.0]) * 3.166811563e-6   # Hartree, ~300/500 K
_SEEBECK_REDUCED = np.tile(0.01 * np.eye(3), (1, len(_KT_VALUES), 1, 1))   # (1, nT, 3, 3)

_EVA_QUANTITIES = {
    "energy": {}, "length": {}, "area": {}, "volume": {}, "berry_curvature": {},
    "kdotp_first_order": {}, "kdotp_second_order": {},
}
_SI_QUANTITIES = {
    "viscosity": {},
    "superfluid_weight": {},
    "electrical_conductivity": {},
    "seebeck": {},
    "thermal_conductivity": {"kT_values": _KT_VALUES},
    "spin_hall_conductivity": {"cell_volume_bohr3": _CELL_VOLUME},
    "specific_heat": {"kT": _KT_VALUES[0]},
    "shift_current": {"cell_volume_bohr3": _CELL_VOLUME},
    "gyrotropic_K": {"cell_volume_bohr3": _CELL_VOLUME},
    "gyrotropic_C": {"cell_volume_bohr3": _CELL_VOLUME},
    "gyrotropic_dos": {"cell_volume_bohr3": _CELL_VOLUME},
    "gyrotropic_noa": {"cell_volume_bohr3": _CELL_VOLUME},
    "hall_conductivity": {"cell_volume_bohr3": _CELL_VOLUME},
    "anomalous_nernst": {"cell_volume_bohr3": _CELL_VOLUME, "kT_values": _KT_VALUES},
    # "thermoelectric_conductivity" deliberately excluded -- not a plain
    # rescaling (see test_thermoelectric_conductivity_from_si_solves_linear_system).
}


@pytest.mark.parametrize("quantity,kwargs", sorted(_EVA_QUANTITIES.items()))
def test_eva_round_trip(quantity, kwargs):
    x = 2.3456
    si_or_eva = units.to_eVA_units(x, quantity, **kwargs)
    assert units.from_eVA_units(si_or_eva, quantity, **kwargs) == pytest.approx(x)


@pytest.mark.parametrize("quantity,kwargs", sorted(_SI_QUANTITIES.items()))
def test_si_round_trip(quantity, kwargs):
    rng = np.random.default_rng(0)
    if quantity == "thermal_conductivity":
        x = rng.uniform(0.1, 2.0, size=(1, len(_KT_VALUES), 3, 3))
    elif quantity == "anomalous_nernst":
        x = rng.uniform(0.1, 2.0, size=(len(_KT_VALUES), 3))
    else:
        x = 1.7
    si = units.to_si_units(x, quantity, **kwargs)
    back = units.from_si_units(si, quantity, **kwargs)
    np.testing.assert_allclose(back, x, rtol=1e-10)


def test_thermoelectric_conductivity_from_si_solves_linear_system():
    """
    The one documented exception: `thermoelectric_conductivity`'s forward
    conversion combines `elcond_atomic` WITH a separately-supplied
    `seebeck_reduced` into a different physical quantity, so its inverse
    needs to solve a linear system (not divide by a constant) to recover
    `elcond_atomic` -- verified directly against a random invertible
    `seebeck_reduced` here, rather than via the generic round-trip loop
    above.
    """
    rng = np.random.default_rng(1)
    elcond_atomic = rng.uniform(0.1, 2.0, size=(1, 2, 3, 3))
    # random but well-conditioned (diagonally dominant) seebeck_reduced
    seebeck_reduced = 0.01 * (np.eye(3) + 0.05 * rng.standard_normal((1, 2, 3, 3)))

    si = units.to_si_units(
        elcond_atomic, "thermoelectric_conductivity", seebeck_reduced=seebeck_reduced,
    )
    back = units.from_si_units(
        si, "thermoelectric_conductivity", seebeck_reduced=seebeck_reduced,
    )
    np.testing.assert_allclose(back, elcond_atomic, rtol=1e-8)


def test_si_to_reduced_is_directly_callable_and_matches_dispatcher():
    """`superfluid.si_to_reduced` mirrors `reduced_to_si`'s own
    directly-callable + registered-dispatcher dual usage."""
    from waw.analysis.superfluid import si_to_reduced, reduced_to_si

    D_reduced = 0.42
    D_si = reduced_to_si(D_reduced)
    assert si_to_reduced(D_si) == pytest.approx(D_reduced)
    assert units.from_si_units(D_si, "superfluid_weight") == pytest.approx(D_reduced)
