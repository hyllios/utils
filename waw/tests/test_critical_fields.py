"""
Critical fields (arXiv:2601.21044 + Brandt PRB 68, 054506).

The load-bearing tests are the two Brandt states as exact constraints on
alpha(kappa), and a free-electron gas where n(E_F), <v^2>, lambda_L and xi are
all analytic -- so the whole chain is checked against closed form rather than
against itself.
"""
import numpy as np
import pytest

from waw.analysis.critical_fields import (
    alpha_brandt, critical_fields, coherence_length, london_depth_dft,
    fermi_surface_averages, KAPPA_TYPE_BOUNDARY, PHI0_AU, BRANDT56)
from waw.units import (to_si_units, from_si_units, FINE_STRUCTURE,
                       AU_B_FIELD_TESLA, HARTREE_TO_J, E_CHARGE, HBAR_SI,
                       BOHR_RADIUS_M)

KC = 1.0 / np.sqrt(2.0)


# --------------------------------------------------------------- alpha(kappa) --
def test_brandt_eq56_gives_hc1_one_at_type_boundary():
    """Brandt: Eq. 56 "yields at kappa = 1/sqrt(2) the correct value hc1 = 1"."""
    hc1 = (np.log(KC) + alpha_brandt(KC)) / (2 * KC ** 2)
    assert hc1 == pytest.approx(1.0, abs=2e-3)     # within his quoted eps


def test_brandt_eq57_is_exact_at_the_type_boundary():
    """Eq. 57 is stated to have "the correct limits at kappa = 1/sqrt2"."""
    hc1 = (np.log(KC) + alpha_brandt(KC, form="eq57")) / (2 * KC ** 2)
    assert hc1 == pytest.approx(1.0, abs=1e-9)


def test_brandt_eq56_large_kappa_limit():
    """Brandt: "for kappa >> 1 it has the limit alpha = 0.49693"."""
    assert alpha_brandt(1e8) == pytest.approx(BRANDT56["alpha_inf"], abs=1e-9)


def test_the_two_brandt_forms_agree_to_about_one_percent():
    """Brandt calls Eq. 57 "an hc1 with error still less than 1%"."""
    k = np.geomspace(KC, 1000.0, 60)
    h56 = (np.log(k) + alpha_brandt(k)) / (2 * k ** 2)
    h57 = (np.log(k) + alpha_brandt(k, form="eq57")) / (2 * k ** 2)
    assert np.max(np.abs(h57 / h56 - 1.0)) < 0.015


def test_eq57_denominator_does_not_diverge_at_the_boundary():
    """The `+ 2` in (2 kappa - sqrt2 + 2) matters: without it Eq. 57 blows up
    exactly at kappa = 1/sqrt(2). Guards against 'fixing' it back."""
    assert np.isfinite(alpha_brandt(KC, form="eq57"))
    assert alpha_brandt(KC, form="eq57") == pytest.approx(
        0.5 + (1 + np.log(2)) / 2.0, rel=1e-12)


def test_alpha_rejects_nonpositive_and_unknown_form():
    with pytest.raises(ValueError):
        alpha_brandt(0.0)
    with pytest.raises(ValueError):
        alpha_brandt(2.0, form="eq99")


# ------------------------------------------------------------- field algebra --
def test_field_ratios_match_the_papers_identities():
    """Eqs. 10-11 are also written as ratios to Hc2: Hc = Hc2/(sqrt2 kappa) and
    Hc1 = Hc2 [ln k + alpha]/(2 kappa^2)."""
    cf = critical_fields(lambda_L=500.0, xi=20.0)
    assert cf.kappa == pytest.approx(25.0)
    assert cf.Hc == pytest.approx(cf.Hc2 / (np.sqrt(2) * cf.kappa), rel=1e-12)
    assert cf.Hc1 == pytest.approx(
        cf.Hc2 * (np.log(cf.kappa) + cf.alpha_brandt) / (2 * cf.kappa ** 2),
        rel=1e-12)


def test_hc2_is_flux_quantum_over_two_pi_xi_squared():
    xi = 37.0
    cf = critical_fields(lambda_L=1000.0, xi=xi)
    assert cf.Hc2 == pytest.approx(PHI0_AU / (2 * np.pi * xi ** 2), rel=1e-14)
    assert cf.Hc2 == pytest.approx(1.0 / (2 * xi ** 2), rel=1e-14)


def test_type_i_gives_no_hc1():
    """Below 1/sqrt2 there is no vortex lattice; ln(kappa) < 0 would otherwise
    return a plausible-looking negative field."""
    cf = critical_fields(lambda_L=10.0, xi=100.0)
    assert not cf.type_ii and cf.kappa < KAPPA_TYPE_BOUNDARY
    assert np.isnan(cf.Hc1)


def test_type_classification_at_the_boundary():
    assert critical_fields(KC * 100.0 * 1.001, 100.0).type_ii
    assert not critical_fields(KC * 100.0 * 0.999, 100.0).type_ii


def test_critical_fields_rejects_nonpositive_lengths():
    for bad in ((0.0, 10.0), (10.0, 0.0), (-1.0, 10.0)):
        with pytest.raises(ValueError):
            critical_fields(*bad)


# -------------------------------------------------------------------- units ---
def test_tesla_round_trip_and_scale():
    B = np.array([1.0, 3.5e-6])
    assert np.allclose(from_si_units(to_si_units(B, "magnetic_flux_density"),
                                     "magnetic_flux_density"), B)
    # a.u. of B = hbar/(e a0^2); the accepted value is 2.350517e5 T
    assert AU_B_FIELD_TESLA == pytest.approx(2.350517e5, rel=1e-6)


def test_hc2_of_niobium_is_a_sane_number_in_tesla():
    """Nb: xi ~ 38 nm, so Hc2 ~ 0.2-0.4 T -- a unit-slip of 4 pi or of the flux
    quantum would show up as orders of magnitude here."""
    xi_bohr = 38e-9 / BOHR_RADIUS_M
    cf = critical_fields(lambda_L=39e-9 / BOHR_RADIUS_M, xi=xi_bohr)
    hc2_t = to_si_units(cf.Hc2, "magnetic_flux_density")
    assert 0.1 < hc2_t < 0.6
    # cross-check against Phi0/(2 pi xi^2) evaluated wholly in SI
    phi0_si = 2.067833848e-15
    assert hc2_t == pytest.approx(phi0_si / (2 * np.pi * (38e-9) ** 2), rel=1e-5)


# ------------------------------------------------- free electrons, analytic ---
def _free_electron_bands(kf_bohr, n_per_axis, box):
    """eps = k^2/2 on a cubic k-box of half-width `box`, with velocities k."""
    g = (np.arange(n_per_axis) + 0.5) / n_per_axis - 0.5
    kx, ky, kz = np.meshgrid(2 * box * g, 2 * box * g, 2 * box * g, indexing="ij")
    k = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=-1)
    eps = 0.5 * (k ** 2).sum(axis=-1)
    return eps[:, None], k[:, None, :], (2 * box) ** 3


@pytest.mark.parametrize("sigma", [0.04, 0.02])
def test_free_electron_gas_matches_analytic_dos_v2_and_smearing_bias(sigma):
    """For eps = k^2/2: n(E_F) = k_F/pi^2 per unit volume (both spins) and
    <v^2> -> k_F^2 as sigma -> 0.

    At finite smearing both averages are biased upward, by amounts that are
    themselves analytic, and asserting THOSE pins the weighting logic far more
    tightly than a loose tolerance on k_F^2 would. With v^2 = 2 eps and
    |v| = sqrt(2 eps) on a free-electron band whose DOS goes as sqrt(eps):

      area weight  ~ delta_sigma(eps-mu) * sqrt(eps) * sqrt(eps) = ...* eps
        => <v^2>_area = 2 (mu^2 + sigma^2)/mu = k_F^2 (1 + sigma^2/mu^2)
      dos weight   ~ delta_sigma(eps-mu) * sqrt(eps)
        => <v^2>_dos  = k_F^2 (1 + sigma^2/(2 mu^2)) to O(sigma^2)

    i.e. the area weighting is biased exactly twice as hard, because it favours
    the fast outer side of the shell. Both are reproduced to ~1e-5.
    """
    kf = 0.8
    mu = 0.5 * kf ** 2
    eps, vel, kvol = _free_electron_bands(kf, 120, 1.6)
    # the k-box is the "cell" in reciprocal space; the real-space volume that
    # makes the DOS per unit volume is V = (2 pi)^3 / kvol
    volume = (2 * np.pi) ** 3 / kvol
    fs = fermi_surface_averages(eps, vel, mu, sigma=sigma, volume=volume)
    r = (sigma / mu) ** 2
    assert fs["v2_area"] == pytest.approx(kf ** 2 * (1 + r), rel=1e-5)
    assert fs["v2_dos"] == pytest.approx(kf ** 2 * (1 + 0.5 * r), rel=1e-4)
    assert fs["dos"] == pytest.approx(kf / np.pi ** 2, rel=5e-3)


def test_free_electron_v2_converges_to_kf_squared_as_smearing_shrinks():
    """Both weightings coincide only where |v| is constant on the Fermi surface,
    which the free-electron gas satisfies in the sigma -> 0 limit."""
    kf = 0.8
    mu = 0.5 * kf ** 2
    eps, vel, kvol = _free_electron_bands(kf, 120, 1.6)
    volume = (2 * np.pi) ** 3 / kvol
    prev = None
    for sigma in (0.04, 0.02, 0.01):
        fs = fermi_surface_averages(eps, vel, mu, sigma=sigma, volume=volume)
        err = abs(fs["v2_area"] / kf ** 2 - 1.0)
        if prev is not None:
            assert err < prev
        prev = err
    assert err < 2e-3


def test_free_electron_london_depth_matches_the_textbook_formula():
    """Eq. 4 must reduce to lambda^-2 = mu_0 n e^2/m for a free-electron gas,
    since n(E_F)<v^2>/3 = n/m there. In atomic units mu_0 = 4 pi alpha^2 and
    n = k_F^3/(3 pi^2)."""
    kf = 0.8
    n_elec = kf ** 3 / (3 * np.pi ** 2)
    mu0 = 4 * np.pi * FINE_STRUCTURE ** 2
    lam_textbook = np.sqrt(1.0 / (mu0 * n_elec))         # m = e = 1
    lam_eq4 = london_depth_dft(dos=kf / np.pi ** 2, v2=kf ** 2)
    assert lam_eq4 == pytest.approx(lam_textbook, rel=1e-12)


def test_electron_phonon_renormalisation_directions():
    """Eq. 6: lambda grows as sqrt(1+lambda_ep), xi shrinks as 1/(1+lambda_ep),
    so kappa grows as (1+lambda_ep)^{3/2} and a material can only become MORE
    type-II from electron-phonon coupling."""
    lam_ep = 1.2
    lam0 = london_depth_dft(dos=0.1, v2=1.0)
    lam1 = london_depth_dft(dos=0.1, v2=1.0, lambda_ep=lam_ep)
    xi0 = coherence_length(v2=1.0, delta=1e-3)
    xi1 = coherence_length(v2=1.0, delta=1e-3, lambda_ep=lam_ep)
    assert lam1 / lam0 == pytest.approx(np.sqrt(1 + lam_ep), rel=1e-12)
    assert xi1 / xi0 == pytest.approx(1.0 / (1 + lam_ep), rel=1e-12)
    assert ((lam1 / xi1) / (lam0 / xi0)
            == pytest.approx((1 + lam_ep) ** 1.5, rel=1e-12))


def test_coherence_length_scales_inversely_with_the_gap():
    assert (coherence_length(1.0, 2e-3) / coherence_length(1.0, 4e-3)
            == pytest.approx(2.0, rel=1e-12))
    with pytest.raises(ValueError):
        coherence_length(1.0, 0.0)


def test_area_and_dos_weightings_differ_on_an_anisotropic_surface():
    """The two averages coincide only for constant |v|. A deliberately
    anisotropic velocity field must separate them -- otherwise the distinction
    the module makes would be untestable and could silently be dropped."""
    rng = np.random.default_rng(0)
    nk = 20000
    eps = rng.normal(0.0, 0.01, (nk, 1))
    vmag = rng.uniform(0.2, 2.0, (nk, 1))
    direc = rng.normal(size=(nk, 1, 3))
    direc /= np.linalg.norm(direc, axis=-1, keepdims=True)
    vel = vmag[..., None] * direc
    fs = fermi_surface_averages(eps, vel, 0.0, sigma=0.02, volume=1.0)
    assert fs["v2_area"] > fs["v2_dos"] * 1.05      # area weighting favours fast v
