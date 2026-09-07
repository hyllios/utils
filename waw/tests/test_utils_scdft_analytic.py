"""
The standalone analytic SCDFT functional (waw.utils.scdft.analytic).

What is pinned here and why:

* the master double sum S(x,y,W) against a brute-force Matsubara sum -- every
  kernel is built from it, so an algebra slip there poisons everything;
* the degenerate-band identity, which catches any multiband bookkeeping error;
* h_Z -> g at zero coupling -- the true-Z flat-channel weight must reduce to
  the bare pair weight, tail included;
* one fixed-temperature eigenvalue to six digits as a pure regression guard;
* Tc against the tail-completed Migdal-Eliashberg reference inside the
  DOCUMENTED accuracy band (0.97-1.03 on the Einstein grid); this functional
  is a genuine approximation, unlike the Galerkin closure of `unexpanded`
  which reproduces the reference to 1e-4 by construction;
* that a structured statically screened W(xi,xi') is accepted and the result
  is xi-grid stable;
* that there is no low-temperature Matsubara floor: the operator builds at
  50 mK with a 1 eV band in seconds, where the `unexpanded` dense grid would
  need n_half ~ 1e5 and refuses.
"""

import numpy as np
import pytest

from waw.units import EV_TO_HARTREE, K_B_HARTREE
from waw.utils.scdft.analytic import (build_analytic,
                                      linearized_eigenvalue_analytic as rho,
                                      tc_analytic)

OM0 = 0.020 * EV_TO_HARTREE
WB = 10.0 * OM0


def _einstein(lam):
    return np.array([OM0]), np.array([lam * OM0 / 2.0])


def test_master_sum_matches_brute_force():
    from waw.utils.scdft.analytic import _s_master
    beta = 11.0
    kT = 1.0 / beta
    n = 3000
    wn = (2.0 * np.arange(-n, n) + 1.0) * np.pi * kT
    for x, y, w in ((0.3, 0.7, 1.1), (1.4, 0.2, 0.5), (0.9, 0.9 - 0.4, 0.4)):
        kb = 2.0 * w / (w ** 2 + (wn[:, None] - wn[None, :]) ** 2)
        ref = kT ** 2 * (kb / ((wn[:, None] ** 2 + x ** 2)
                               * (wn[None, :] ** 2 + y ** 2))).sum()
        got = float(_s_master(np.array([x]), np.array([y]), w, beta)[0, 0])
        assert got == pytest.approx(ref, rel=1e-6), (x, y, w)


def test_two_degenerate_bands_reduce_to_one():
    om, a1 = _einstein(1.0)
    a2 = np.full((2, 2, 1), 1.0 * OM0 / 4.0)
    kT = 10.0 * K_B_HARTREE
    r1 = rho(om, a1, 0.2, kT, band_edge=WB)
    r2 = rho(om, a2, np.full((2, 2), 0.1), kT, band_edge=WB)
    assert abs(r1 - r2) < 1e-12 * abs(r1)


def test_h_z_reduces_to_bare_at_zero_coupling():
    from waw.utils.scdft.analytic import _gfun, _h_z_true
    beta = 1.0 / (5.0 * K_B_HARTREE)
    xi = np.geomspace(1e-6, WB, 50)
    a = np.full((1, 1, 1), 1e-12)
    hz = _h_z_true(xi, beta, np.array([OM0]), a, 4.0 * WB, 0)
    g = _gfun(xi, beta)
    assert np.max(np.abs(hz / g - 1.0)) < 1e-5


def test_fixed_temperature_eigenvalue_regression():
    """Six-digit regression guard at lambda = 1, mu = 0.2, T = 10 K. If a
    refactor moves this, either it fixed a real bug (re-measure and update
    with a note) or it broke the kernels."""
    om, a2f = _einstein(1.0)
    r = rho(om, a2f, 0.2, 10.0 * K_B_HARTREE, band_edge=WB)
    assert r == pytest.approx(1.1255024, rel=2e-6)


@pytest.mark.parametrize("lam,mu,tc_ref_over_om0", [
    (0.5, 0.2, 0.00605), (1.0, 0.0, 0.10004), (1.0, 0.3, 0.04180),
])
def test_tc_within_the_documented_band(lam, mu, tc_ref_over_om0):
    """Reference values from scripts/me_reference.py of the derivation notes
    (tail-completed). The documented accuracy of this functional on the
    Einstein grid is 0.97-1.03; assert with a small margin."""
    om, a2f = _einstein(lam)
    tc_ref = tc_ref_over_om0 * OM0 / K_B_HARTREE
    tc = tc_analytic(om, a2f, mu, band_edge=WB, t_min=0.5)
    assert 0.95 < tc / tc_ref < 1.05, (tc, tc_ref, tc / tc_ref)


def test_structured_w_is_accepted_and_grid_stable():
    om, a2f = _einstein(1.0)
    e_scr = WB / 3.0

    def w(i, j, xi, xip):
        return 0.2 * e_scr ** 2 / np.sqrt((e_scr ** 2 + xi ** 2)
                                          * (e_scr ** 2 + xip ** 2))
    kT = 10.0 * K_B_HARTREE
    r1 = rho(om, a2f, w, kT, band_edge=WB, n_xi=200)
    r2 = rho(om, a2f, w, kT, band_edge=WB, n_xi=400)
    assert abs(r1 - r2) < 2e-3 * abs(r1), (r1, r2)
    # the structured repulsion is weaker than the flat one at high energy,
    # so it must suppress the eigenvalue less than flat mu = 0.2
    r_flat = rho(om, a2f, 0.2, kT, band_edge=WB, n_xi=200)
    assert r1 > r_flat


def test_no_low_temperature_floor():
    """50 mK with a 1 eV band: the unexpanded dense Matsubara grid would need
    n_half ~ 4 Wb / (2 pi kT) ~ 5e6 and raises; here only the 1-D h_Z sum
    grows, and the operator must build and be finite."""
    om, a2f = _einstein(0.5)
    k = build_analytic(om, a2f, 0.1, 0.05 * K_B_HARTREE,
                       band_edge=1.0 * EV_TO_HARTREE, n_xi=80)
    assert np.isfinite(k.operator).all()
    assert (k.sigma_s > 0).all(), "sigma_s must be positive definite"


def test_multiband_tc_against_the_block_reference_value():
    """MgB2-like Einstein lambda matrix (Golubov et al.) at uniform mu = 0.1:
    the tail-completed block ME reference gives 47.31 K
    (scripts/analytic_bench.py); documented accuracy band as above."""
    omb = 0.065 * EV_TO_HARTREE
    lam = np.array([[1.017, 0.213], [0.155, 0.448]])
    om = np.array([omb])
    a2f = (lam * omb / 2.0)[:, :, None]
    tc = tc_analytic(om, a2f, np.full((2, 2), 0.1),
                     band_edge=np.full(2, 10.0 * omb), t_min=2.0)
    assert 0.95 < tc / 47.31 < 1.05, tc
