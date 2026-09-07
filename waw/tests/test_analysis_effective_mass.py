"""
Tests for waw/analysis/effective_mass.py.

Uses hand-built HamiltonianR models with an analytically known dispersion
(simple-cubic single/decoupled-two-band tight binding) so the computed
effective masses can be checked against a closed-form reference instead
of just shape/sanity checks.

For a single band E(k) = 2 s t (cos 2*pi*kx + cos 2*pi*ky + cos 2*pi*kz)
(s = +-1, k in crystal coordinates, cubic real-space lattice constant a):
  - s=-1: minimum at Gamma, maximum at R=(0.5,0.5,0.5)
  - s=+1: maximum at Gamma, minimum at R=(0.5,0.5,0.5)
  - isotropic effective mass at either extremum has magnitude
        m* = 1 / (2 t a^2)     (m_e units; t in Hartree, a in Bohr --
        in atomic units hbar = m_e = 1, so no conversion factor)
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.effective_mass import (
    find_band_extrema, analyze_effective_mass, semiconductor_band_edges,
    degenerate_effective_mass, masses_along, BandExtremum,
)

A_BOHR = 5.0
CUBIC_REAL_LATTICE  = A_BOHR * np.eye(3)
CUBIC_RECIP_LATTICE = 2 * np.pi * np.linalg.inv(CUBIC_REAL_LATTICE).T


def _cubic_hr(hoppings, onsite=None):
    """
    Decoupled-band simple-cubic tight-binding model.

    hoppings: (nw,) per-band hopping amplitude t_n; band n disperses as
              E_n(k) = 2 * t_n * (cos 2pi kx + cos 2pi ky + cos 2pi kz) + onsite_n
    A positive t_n gives a maximum at Gamma; negative gives a minimum at Gamma.
    """
    nw = len(hoppings)
    onsite = np.zeros(nw) if onsite is None else np.asarray(onsite)

    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
              (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), nw, nw), dtype=np.complex128)
    for i in range(6):
        for n in range(nw):
            H_R[i, n, n] = hoppings[n]
    for n in range(nw):
        H_R[6, n, n] = onsite[n]

    return HamiltonianR(
        H_R=torch.tensor(H_R, dtype=torch.complex128),   # model directly in Hartree
        R_vectors=R_vectors, degen=degen, nw=nw,
    )


def _expected_mass(t, a=A_BOHR):
    return 1.0 / (2.0 * abs(t) * a ** 2)


# ===========================================================================
# Single isotropic band
# ===========================================================================

def test_min_at_gamma_isotropic_mass():
    t = 1.0
    hr = _cubic_hr([-t])   # minimum at Gamma
    extrema = find_band_extrema(hr, 0, "min", CUBIC_RECIP_LATTICE)

    assert len(extrema) >= 1
    e0 = extrema[0]
    np.testing.assert_allclose(e0.kpt, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(e0.energy, -6 * t, atol=1e-6)

    result = analyze_effective_mass(hr, e0, CUBIC_RECIP_LATTICE)
    expected = _expected_mass(t)
    np.testing.assert_allclose(result.principal_masses, [expected] * 3, rtol=1e-4)
    np.testing.assert_allclose(result.anisotropy, 1.0, rtol=1e-4)
    np.testing.assert_allclose(result.dos_mass, expected, rtol=1e-4)
    np.testing.assert_allclose(result.conductivity_mass, expected, rtol=1e-4)


def test_max_at_r_point_isotropic_hole_mass():
    t = 1.0
    hr = _cubic_hr([-t])   # maximum at R = (0.5, 0.5, 0.5)
    extrema = find_band_extrema(hr, 0, "max", CUBIC_RECIP_LATTICE)

    assert len(extrema) >= 1
    e0 = extrema[0]
    np.testing.assert_allclose(e0.kpt, [0.5, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(e0.energy, 6 * t, atol=1e-6)

    result = analyze_effective_mass(hr, e0, CUBIC_RECIP_LATTICE)
    expected = _expected_mass(t)
    # Hole mass is reported positive despite the negative curvature.
    np.testing.assert_allclose(result.principal_masses, [expected] * 3, rtol=1e-4)


# ===========================================================================
# Anisotropic single band
# ===========================================================================

def test_anisotropic_principal_masses_and_axes():
    tx, ty, tz = -0.5, -1.0, -2.0   # all minima at Gamma, different masses

    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)
    t_per_R = [tx, tx, ty, ty, tz, tz]

    H_R = np.zeros((len(R_list), 1, 1), dtype=np.complex128)
    for i, t in enumerate(t_per_R):
        H_R[i, 0, 0] = t
    hr = HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                       R_vectors=R_vectors, degen=degen, nw=1)

    extrema = find_band_extrema(hr, 0, "min", CUBIC_RECIP_LATTICE)
    e0 = extrema[0]
    np.testing.assert_allclose(e0.kpt, [0.0, 0.0, 0.0], atol=1e-6)

    result = analyze_effective_mass(hr, e0, CUBIC_RECIP_LATTICE)
    expected = sorted(_expected_mass(t) for t in (tx, ty, tz))
    np.testing.assert_allclose(sorted(result.principal_masses), expected, rtol=1e-4)

    # Each principal axis should be (close to) a single Cartesian direction.
    for col in range(3):
        axis = np.abs(result.principal_axes[:, col])
        assert np.isclose(axis.max(), 1.0, atol=1e-3)
        assert np.sum(axis > 1e-3) == 1


# ===========================================================================
# Two-band semiconductor model (direct gap at Gamma)
# ===========================================================================

def test_semiconductor_band_edges_direct_gap():
    t_v, t_c = 0.5, 1.0
    delta_e = 10.0   # onsite separation; keeps bands from ever crossing

    hr = _cubic_hr([t_v, -t_c], onsite=[0.0, delta_e])

    result = semiconductor_band_edges(hr, n_valence=1, recip_lattice=CUBIC_RECIP_LATTICE)

    assert result.direct
    np.testing.assert_allclose(result.gap, 1.0, atol=1e-4)

    vbm = result.vbm[0]
    cbm = result.cbm[0]
    np.testing.assert_allclose(vbm.extremum.kpt, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(cbm.extremum.kpt, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vbm.extremum.energy, 3.0, atol=1e-6)
    np.testing.assert_allclose(cbm.extremum.energy, 4.0, atol=1e-6)

    np.testing.assert_allclose(vbm.principal_masses, [_expected_mass(t_v)] * 3, rtol=1e-4)
    np.testing.assert_allclose(cbm.principal_masses, [_expected_mass(t_c)] * 3, rtol=1e-4)


def test_semiconductor_band_edges_requires_conduction_band():
    hr = _cubic_hr([-1.0])   # nw=1, no conduction band available
    with pytest.raises(ValueError):
        semiconductor_band_edges(hr, n_valence=1, recip_lattice=CUBIC_RECIP_LATTICE)


# ===========================================================================
# Degeneracy flagging
# ===========================================================================

def test_degenerate_band_is_flagged():
    t = 1.0
    hr = _cubic_hr([-t, -t])   # two identical, fully degenerate bands

    extrema = find_band_extrema(hr, 0, "min", CUBIC_RECIP_LATTICE)
    e0 = extrema[0]
    assert 1 in e0.degenerate_with

    # Mass tensor still computes without raising, even though physically
    # this needs a degenerate k.p treatment to be meaningful.
    result = analyze_effective_mass(hr, e0, CUBIC_RECIP_LATTICE)
    assert np.all(np.isfinite(result.principal_masses))


def test_triple_degeneracy_transitive_closure():
    """
    band_index-1/+1 neighbour-only checking would only catch band 1 from
    band 2 (missing band 0 entirely, transitively degenerate with band 1).
    """
    t = 1.0
    hr = _cubic_hr([-t, -t, -t])   # three identical, fully degenerate bands

    extrema = find_band_extrema(hr, 1, "min", CUBIC_RECIP_LATTICE)
    e0 = extrema[0]
    assert set(e0.degenerate_with) == {0, 2}


# ===========================================================================
# Degenerate k.p effective mass (Loewdin partitioning) — Kane-like model
#
# 3 orbitals A, B, C on a simple-cubic lattice. A, B are degenerate at
# Gamma (onsite energy 0); C is a remote band at onsite energy Ec. A
# couples to C only along x (odd/antisymmetric hopping -> pure sine,
# vanishing at Gamma to zeroth order but linear to first order); B
# couples to C only along y, by the same construction. No direct A-B
# coupling anywhere. This gives closed-form Loewdin masses:
#
#   Q_xx[A,A] = Q_yy[B,B] = -4 p^2 a^2 / Ec      (all other Q_ab entries 0)
#
# mass = sign / (2*Q) (Q already carries the Taylor-series 1/2, see
# `masses_along`'s docstring), so along [100] band A has finite mass
# Ec/(8 p^2 a^2) (atomic units) and band B is exactly flat (infinite
# mass); along [110] the two branches become degenerate again, each with
# half that curvature (i.e. the same finite mass, since mass ~ 1/curvature).
# ===========================================================================

def _kane_model_hr(p, Ec, a=A_BOHR):
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), 3, 3), dtype=np.complex128)
    # A=0, B=1, C=2
    H_R[0, 0, 2], H_R[0, 2, 0] = p, -p       # R = (+1,0,0)
    H_R[1, 0, 2], H_R[1, 2, 0] = -p, p       # R = (-1,0,0)
    H_R[2, 1, 2], H_R[2, 2, 1] = p, -p       # R = (0,+1,0)
    H_R[3, 1, 2], H_R[3, 2, 1] = -p, p       # R = (0,-1,0)
    H_R[4, 2, 2] = Ec                        # R = (0,0,0), onsite

    real_lattice = a * np.eye(3)
    return HamiltonianR(
        H_R=torch.tensor(H_R, dtype=torch.complex128),   # model directly in Hartree
        R_vectors=R_vectors, degen=degen, nw=3,
    ), 2 * np.pi * np.linalg.inv(real_lattice).T


def test_degenerate_kp_mass_along_100():
    p, Ec = 2.0, 5.0
    hr, recip_lattice = _kane_model_hr(p, Ec)

    extremum = BandExtremum(kpt=np.zeros(3), energy=0.0, band_index=0,
                             kind="max", degenerate_with=(1,))
    result = degenerate_effective_mass(hr, extremum, recip_lattice)

    assert result.band_group == (0, 1)
    np.testing.assert_allclose(result.linear_term, 0.0, atol=1e-10)

    masses = masses_along(result, [1.0, 0.0, 0.0])
    expected_finite = Ec / (8 * p ** 2 * A_BOHR ** 2)
    # One branch flat (infinite mass) along pure [100]; the other finite.
    assert np.isinf(masses).sum() == 1
    finite = masses[np.isfinite(masses)]
    np.testing.assert_allclose(finite, [expected_finite], rtol=1e-6)


def test_degenerate_kp_mass_along_110():
    """
    Along [110], both A and C, B and C are simultaneously "probed", which
    induces a remote-band-mediated A-B cross term (Q_xy != 0) even though
    there's no direct A-B coupling anywhere in the model. The resulting
    2x2 inverse-mass matrix is exactly rank-1: one branch stays exactly
    flat (infinite mass, same as along pure [100]/[010]), the other has
    exactly the *same* finite mass as along [100] (not half of it, as a
    naive "no cross term" guess would predict).
    """
    p, Ec = 2.0, 5.0
    hr, recip_lattice = _kane_model_hr(p, Ec)

    extremum = BandExtremum(kpt=np.zeros(3), energy=0.0, band_index=0,
                             kind="max", degenerate_with=(1,))
    result = degenerate_effective_mass(hr, extremum, recip_lattice)

    masses = masses_along(result, [1.0, 1.0, 0.0])
    expected_finite = Ec / (8 * p ** 2 * A_BOHR ** 2)
    assert np.isinf(masses).sum() == 1
    finite = masses[np.isfinite(masses)]
    np.testing.assert_allclose(finite, [expected_finite], rtol=1e-6)
