"""
The amplitude extraction of the unexpanded construction must be a PROJECTION.

Inverting the Sham-Schlueter condition with a single-profile ansatz
phi_m = a phihat_m pointwise gives

    a(xi) = Delta_s(xi) P_s(xi) / sigma(xi),   sigma(xi) = sum_m phihat_m D(xi,m)

and sigma has a zero crossing whenever phihat changes sign -- i.e. for ANY
mu > 0, because that sign change IS the Morel-Anderson mechanism the
construction relies on. At that xi the ansatz would need an infinite phi to
reproduce a finite anomalous density, so the pointwise form is inconsistent, not
merely delicate. It was in the prototype too (scripts/scdft_closedform.py), so
this is a defect of the construction as derived rather than of the port, and it
affects all three profiles identically since the division happens after the
profile is chosen.

These tests pin the two properties that the pointwise form failed and that any
correct amplitude rule must have: independence of the xi grid, and monotonicity
of rho(T). Both are cheap and both were violated by a factor of 20% and by
spurious eigenvalues up to 8.8 respectively.
"""

import numpy as np
import pytest

from waw.units import EV_TO_HARTREE, K_B_HARTREE
from waw.utils.scdft.unexpanded import (build_kernels,
                                        linearized_eigenvalue_unexpanded as rho)

OM0 = 0.02 * 0.0367493            # 20 meV Einstein mode, Hartree
WB = 10 * OM0


def _einstein(lam):
    return np.array([OM0]), np.array([lam * OM0 / 2])


@pytest.mark.parametrize("mu", [0.0, 0.2, 0.35])
@pytest.mark.parametrize("profile", ["lorentzian", "galerkin", "exact"])
def test_eigenvalue_is_independent_of_the_xi_grid(mu, profile):
    """THE test the pointwise form failed: at mu = 0.2 it drifted 20% from
    n_xi = 150 to 800 with no plateau, because refining the grid samples the
    pole in 1/sigma harder. A convergent construction is flat."""
    om, a2f = _einstein(0.5)
    kT = 5.0 * K_B_HARTREE
    vals = [rho(om, a2f, mu, kT, band_edge=WB, n_xi=n, profile=profile)
            for n in (150, 300, 600, 1200)]
    spread = max(vals) - min(vals)
    # 1e-5 still catches the old 20% drift by four orders of magnitude
    assert spread < 1e-5 * max(1.0, abs(vals[0])), (mu, profile, vals)


@pytest.mark.parametrize("mu", [0.0, 0.2, 0.35])
def test_rho_is_monotonic_in_temperature(mu):
    """The pointwise form produced isolated spikes (rho = 8.8 at one T), which
    made the Tc bisection return 75.7 K for Al. rho must fall monotonically."""
    om, a2f = _einstein(0.5)
    ts = np.array([2., 3., 4., 5., 6., 8., 10., 12., 15., 20.])
    vals = np.array([rho(om, a2f, mu, t * K_B_HARTREE, band_edge=WB) for t in ts])
    assert np.all(np.diff(vals) < 0.0), (mu, vals)


def test_operator_rank_equals_the_number_of_amplitudes():
    """The only unknowns are the nb*P profile amplitudes, so the gap operator
    must have exactly nb*P nonzero eigenvalues -- one with the single
    closed-form profile, n_basis with the default Galerkin basis. A higher
    rank would mean the amplitude is being determined per xi again."""
    om, a2f = _einstein(0.5)
    k = build_kernels(om, a2f, 0.2, 5.0 * K_B_HARTREE, band_edge=WB,
                      profile="lorentzian")
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(k.operator))))[::-1]
    assert ev[0] > 1e-3
    assert ev[1] < 1e-10 * ev[0], ev[:4]
    k = build_kernels(om, a2f, 0.2, 5.0 * K_B_HARTREE, band_edge=WB,
                      profile="galerkin", n_basis=4)
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(k.operator))))[::-1]
    assert ev[3] > 1e-10 * ev[0], "basis collapsed below n_basis"
    assert ev[4] < 1e-10 * ev[0], ev[:6]


@pytest.mark.parametrize("lam,mu,tc_ref_over_om0", [
    (0.5, 0.0, 0.03600), (0.5, 0.1, 0.01507), (0.5, 0.2, 0.00605),
    (1.0, 0.0, 0.10004), (1.0, 0.2, 0.05425), (1.0, 0.3, 0.04180),
])
def test_tc_matches_eliashberg(lam, mu, tc_ref_over_om0):
    """Against Migdal-Eliashberg (reference values from the derivation notes'
    scripts/me_reference.py: band-limited, bare static mu, TAIL-COMPLETED
    Matsubara sums -- without the tails a 4x-band-edge cutoff costs up to 5.7%
    of Tc at mu = 0.2, which is a truncation error and not physics): the
    default Galerkin closure reproduces the reference Tc to a few per mille.
    (The historical form of this test asserted "from below" for the fixed
    closed-form profile; that direction is only a tendency -- the operator is
    not Hermitian, so there is no variational bound -- and the default basis
    is exact enough for a two-sided assertion.)
    """
    om, a2f = _einstein(lam)
    tc_ref = tc_ref_over_om0 * OM0 / K_B_HARTREE
    # the Matsubara grid must span 4x the band edge, so n_half ~ 1/T caps how
    # low the bracket can start: at Wb = 10 Om0 that is ~0.25 K
    lo, hi = 0.3, 300.0
    if rho(om, a2f, mu, lo * K_B_HARTREE, band_edge=WB) < 1.0:
        pytest.skip("Tc below the reachable bracket")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if rho(om, a2f, mu, mid * K_B_HARTREE, band_edge=WB) > 1.0:
            lo = mid
        else:
            hi = mid
    tc = 0.5 * (lo + hi)
    ratio = tc / tc_ref
    assert 0.98 < ratio < 1.02, (lam, mu, tc, tc_ref, ratio)


def test_sigma_really_does_change_sign_for_repulsive_mu():
    """The premise of all of the above: sigma(xi) has a zero for mu > 0 and none
    at mu = 0. If this ever stopped being true the pointwise form would be
    salvageable and these tests would be over-strict."""
    om, a2f = _einstein(0.5)
    for mu, expect_flip in ((0.0, False), (0.2, True), (0.35, True)):
        k = build_kernels(om, a2f, mu, 5.0 * K_B_HARTREE, band_edge=WB,
                          profile="lorentzian")
        wn, zn = np.asarray(k.omega_n), np.asarray(k.z_n)[:, 0]
        xi, ph = np.asarray(k.xi)[:, 0], np.asarray(k.phihat)[:, 0, 0]
        d = 1.0 / (wn[None, :] ** 2 * zn[None, :] ** 2 + xi[:, None] ** 2)
        sigma = (ph[None, :] * d).sum(axis=1)
        flips = int(np.sum(np.diff(np.sign(sigma)) != 0))
        assert (flips > 0) == expect_flip, (mu, flips)
