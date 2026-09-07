"""
Tests for waw/analysis/superfluid.py.

Validates against:
  1. A single-band (no interband coupling at all) simple-cubic model:
     - the geometric term must be *exactly* zero (there are no m != n
       pairs to sum over), a structural check.
     - the conventional term is cross-checked against an independent
       brute-force reference computed directly in the test (hand-derived
       analytic band velocity, not calling any of the module's internals)
       to validate the formula, prefactors, and BZ/volume normalization
       all at once.
  2. A 2-band model with a k-dependent off-diagonal coupling, where the
     geometric term becomes genuinely nonzero away from special points,
     and vanishes continuously as the coupling strength -> 0.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.superfluid import (
    superfluid_weight, superfluid_weight_at_k, superfluid_weight_small_gap,
    penetration_depth, reduced_to_si,
)
from waw.units import EV_TO_HARTREE

A_BOHR = 5.0
REAL_LATTICE = A_BOHR * np.eye(3)


def _single_band_hr(t: float) -> HamiltonianR:
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)
    H_R = np.full((len(R_list), 1, 1), t, dtype=np.complex128)
    # Model built directly in atomic units (t in Hartree); most tests here
    # are scale-invariant, so the numerical values are just model units.
    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=1)


def test_geometric_vanishes_for_single_band():
    hr = _single_band_hr(-1.0)
    result = superfluid_weight(hr, REAL_LATTICE, delta=0.05, mu=0.0, mesh=(6, 6, 6))
    np.testing.assert_allclose(result.geometric, np.zeros((3, 3)), atol=1e-14)


def test_conventional_matches_independent_brute_force():
    t = 1.0
    delta = 0.05
    mu = 0.0
    mesh = (8, 8, 8)
    hr = _single_band_hr(-t)

    result = superfluid_weight(hr, REAL_LATTICE, delta=delta, mu=mu, mesh=mesh, kT=0.0)

    # Independent reference: hand-derived analytic band velocity for
    # eps(k) = -2t(cos 2pi kx + cos 2pi ky + cos 2pi kz) - mu, on a cubic
    # lattice (recip = (2pi/a) I), computed without calling any of the
    # module's own machinery.
    N1, N2, N3 = mesh
    V = A_BOHR ** 3
    D_conv_ref = np.zeros((3, 3))
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                kx, ky, kz = i / N1, j / N2, k / N3
                eps = -2 * t * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky)
                                 + np.cos(2 * np.pi * kz)) - mu
                E = np.sqrt(eps ** 2 + delta ** 2)
                v = 2 * t * A_BOHR * np.array([
                    np.sin(2 * np.pi * kx), np.sin(2 * np.pi * ky), np.sin(2 * np.pi * kz),
                ])
                D_conv_ref += (delta ** 2 / E ** 3) * np.outer(v, v)
    D_conv_ref /= (N1 * N2 * N3 * V)

    np.testing.assert_allclose(result.conventional, D_conv_ref, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result.total, D_conv_ref, rtol=1e-10, atol=1e-15)


def test_conventional_tensor_is_symmetric():
    hr = _single_band_hr(-1.0)
    result = superfluid_weight(hr, REAL_LATTICE, delta=0.1, mu=0.3, mesh=(6, 6, 6))
    np.testing.assert_allclose(result.conventional, result.conventional.T, atol=1e-12)


# ===========================================================================
# Coupled 2-band model: geometric term nonzero away from special points,
# vanishing continuously as the coupling -> 0.
# ===========================================================================

def _coupled_two_band_hr(t1: float, t2: float, g: float, onsite2: float) -> HamiltonianR:
    """H(k) = 2 cos(2 pi kx) [[t1, g], [g, t2]] + diag(0, onsite2)."""
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    block = np.array([[t1, g], [g, t2]], dtype=np.complex128)
    H_R = np.zeros((3, 2, 2), dtype=np.complex128)
    H_R[0] = block
    H_R[1] = block
    H_R[2] = np.array([[0.0, 0.0], [0.0, onsite2]], dtype=np.complex128)

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=2)


def test_geometric_nonzero_with_interband_coupling():
    hr = _coupled_two_band_hr(t1=-1.0, t2=1.0, g=0.3, onsite2=2.0)
    k_frac = np.array([0.15, 0.0, 0.0])
    D_conv, D_geom = superfluid_weight_at_k(hr, k_frac, REAL_LATTICE,
                                             delta=0.2, mu=0.0, kT=0.0)
    assert np.any(np.abs(D_geom) > 1e-8)
    np.testing.assert_allclose(D_geom, D_geom.T, atol=1e-12)


def test_geometric_vanishes_as_coupling_goes_to_zero():
    k_frac = np.array([0.15, 0.0, 0.0])
    couplings = [0.3, 0.1, 0.03, 0.01, 0.0]
    norms = []
    for g in couplings:
        hr = _coupled_two_band_hr(t1=-1.0, t2=1.0, g=g, onsite2=2.0)
        _, D_geom = superfluid_weight_at_k(hr, k_frac, REAL_LATTICE,
                                            delta=0.2, mu=0.0, kT=0.0)
        norms.append(np.linalg.norm(D_geom))

    assert norms[-1] == 0.0   # g = 0: no coupling at all
    assert all(n2 < n1 for n1, n2 in zip(norms, norms[1:]))   # monotonically shrinking


# ===========================================================================
# Penetration depth (Eq. 17): a pure formula check, Ds already in SI units.
# ===========================================================================

def test_penetration_depth_formula():
    mu0 = 4 * np.pi * 1e-7
    Ds = 1e35   # arbitrary SI-unit value for a formula-only check
    lam = penetration_depth(Ds, mu0=mu0)
    np.testing.assert_allclose(lam, 1.0 / np.sqrt(mu0 * Ds))


def test_derived_prefactor_gives_physically_sane_penetration_depth():
    """
    Order-of-magnitude sanity check on the derived e^2/hbar^2 prefactor
    (reduced_to_si): metallic-scale toy parameters (t ~ 1 eV bandwidth,
    Delta ~ 0.05 eV gap, a ~ 5 Bohr lattice constant) should give a
    penetration depth in the same ballpark as real conventional
    superconductors (nm to a few hundred nm; see Table I of Hiorth et
    al., e.g. Al=13nm, Pb=11nm, Nb=15nm), not off by many orders of
    magnitude as a wrong power of e or hbar in the prefactor would cause.
    """
    # This test needs genuinely PHYSICAL scales (it checks SI output), so
    # the eV-scale toy parameters are converted to Hartree here, at the
    # test level -- the analysis API itself is atomic-units only.
    hr = _single_band_hr(-1.0 * EV_TO_HARTREE)
    real_lattice = A_BOHR * np.eye(3)
    result = superfluid_weight(hr, real_lattice, delta=0.05 * EV_TO_HARTREE,
                               mu=0.0, mesh=(10, 10, 10))

    Ds_si = reduced_to_si(result.total)
    lam_nm = penetration_depth(np.diag(Ds_si)) * 1e9

    assert np.all(np.isfinite(lam_nm))
    assert np.all((lam_nm > 1.0) & (lam_nm < 1000.0))


# ===========================================================================
# Small-gap (Delta -> 0) limit: Delta^2/(eps^2+Delta^2)^{3/2} -> 2 delta(eps)
# ===========================================================================

def test_conv_prefactor_integrates_to_exactly_two():
    """
    int Delta^2/(eps^2+Delta^2)^{3/2} d(eps) = 2 for *any* Delta > 0 (via
    eps = Delta tan(theta)) -- the exact identity underlying the claim
    that this prefactor -> 2 delta(eps) as Delta -> 0. Checked numerically
    for several Delta, independent of any BZ/mesh machinery.
    """
    for delta in (1.0, 0.1, 0.01, 0.0001):
        eps = np.linspace(-200 * delta, 200 * delta, 2_000_001)
        prefactor = delta ** 2 / (eps ** 2 + delta ** 2) ** 1.5
        integral = np.trapezoid(prefactor, eps)
        np.testing.assert_allclose(integral, 2.0, rtol=1e-4)


def test_small_gap_conventional_isotropic_for_cubic_single_band():
    """The Delta -> 0 Fermi-surface integral must still respect the cubic
    lattice's symmetry: isotropic D_conv for a single isotropic-hopping band."""
    hr = _single_band_hr(-1.0)
    result = superfluid_weight_small_gap(hr, REAL_LATTICE, delta=0.01, mu=0.5,
                                          sigma=0.3, mesh=(24, 24, 24))
    eigvals = np.linalg.eigvalsh(result.conventional)
    np.testing.assert_allclose(eigvals, eigvals[0], rtol=0.05)
    off_diag = result.conventional - np.diag(np.diag(result.conventional))
    assert np.all(np.abs(off_diag) < 0.05 * np.abs(np.diag(result.conventional)).max())


def test_small_gap_geometric_unaffected():
    """D_geom should be identical (still exactly 0) regardless of conv_sigma,
    since only D_conv is reformulated -- structural, not a numeric approximation."""
    hr = _single_band_hr(-1.0)
    result = superfluid_weight_small_gap(hr, REAL_LATTICE, delta=0.05, mu=0.0,
                                          sigma=0.3, mesh=(10, 10, 10))
    np.testing.assert_allclose(result.geometric, np.zeros((3, 3)), atol=1e-14)


def test_small_gap_roughly_consistent_with_exact_formula():
    """
    For a genuinely small Delta (resolvable, if barely, by the exact
    formula on a moderately fine mesh), the small-gap approximation
    (coarser mesh, decoupled numerical sigma) should land in the same
    ballpark as the exact calculation -- not a tight match (different
    methods), but same order of magnitude and sign.
    """
    hr = _single_band_hr(-1.0)
    delta = 0.3
    exact = superfluid_weight(hr, REAL_LATTICE, delta=delta, mu=0.5, mesh=(30, 30, 30))
    approx = superfluid_weight_small_gap(hr, REAL_LATTICE, delta=delta, mu=0.5,
                                          sigma=0.3, mesh=(20, 20, 20))

    exact_trace = np.trace(exact.conventional)
    approx_trace = np.trace(approx.conventional)
    assert exact_trace > 0 and approx_trace > 0
    ratio = approx_trace / exact_trace
    assert 0.3 < ratio < 3.0
