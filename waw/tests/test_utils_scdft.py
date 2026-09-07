"""
SCDFT with the Sanna-Pellegrini-Gross functional (`waw/utils/scdft`),
Phys. Rev. Lett. 125, 057001 (2020).

WHAT IS PINNED HERE, and what is not, because the two differ in strength:

* `functions.py` is verified RIGOROUSLY. The paper's I (Eq. 12) and J (Eq. 13)
  are reproduced to 1e-12 or better wherever the literal formulas can be
  evaluated, including at the removable singularities -- where the literal
  formulas are 0/0 and ours give the correct limits, confirmed by approaching
  them. The rewritings are also overflow-free at realistic beta, which the
  literal forms are not (they need exp(387) for xi = 1 eV at 30 K).

* The solver is pinned on internal consistency and on the paper's qualitative
  results: the linearized Tc tracks Eliashberg Tc at lambda >= 1, and Delta_s
  has a local minimum at E_F. The paper's Fig. 2 is a bitmap, so its
  weak-coupling Tc cannot be checked numerically -- and that is exactly where
  ours deviates (see the module docstring).
"""

import numpy as np
import pytest

from waw.units import EV_TO_HARTREE, HARTREE_TO_EV, K_B_HARTREE
from waw.utils.eliashberg import tc_linearized
from waw.utils.scdft import (
    SCDFT_GAMMA4_GAP,
    SCDFT_GAMMAS_KERNEL,
    energy_grid,
    i_function,
    j_function,
    solve_delta_s,
    tc_scdft,
    z_kernel,
)

WPH = 60e-3 / HARTREE_TO_EV          # the paper's model Einstein frequency
BETA = 40.0                          # small enough that the literal forms work


def _f(x, beta=BETA):
    return 1.0 / (1.0 + np.exp(beta * x))


def _i_literal(xi, xip, w, beta=BETA):
    """The paper's Eq. (12) exactly as printed."""
    return (_f(xi, beta) * _f(xip, beta) / (np.exp(beta * w) - 1.0)) * (
        (np.exp(beta * xi) - np.exp(beta * (xip + w))) / (xi - xip - w)
        - (np.exp(beta * xip) - np.exp(beta * (xi + w))) / (xi - xip + w))


def _j_literal(xi, e, w, g, beta=BETA):
    """The paper's Eq. (13) exactly as printed."""
    a = xi - w
    n = 1.0 / (np.exp(beta * w) - 1.0)
    return (_f(xi, beta) + n) * (_f(a, beta) / ((a - e) * (a - g))
                                 + _f(e, beta) / ((g - e) * (a - e))
                                 - _f(g, beta) / ((g - e) * (a - g)))


# ------------------------------------------------------- the special functions

@pytest.mark.parametrize("args", [(0.05, 0.02, 0.01), (0.2, -0.1, 0.06),
                                  (0.01, 0.011, 0.03), (0.3, 0.3, 0.02),
                                  (0.0, 0.0, 0.01), (-0.05, 0.02, 0.04)])
def test_i_function_matches_the_paper_formula(args):
    assert i_function(*args, BETA) == pytest.approx(_i_literal(*args), rel=1e-11)


@pytest.mark.parametrize("args", [(0.3, 0.2, 0.05, 0.19), (-0.4, 0.7, 0.1, 0.38),
                                  (0.11, 0.02, 0.03, 0.114)])
def test_j_function_matches_the_paper_formula(args):
    assert j_function(*args, BETA) == pytest.approx(_j_literal(*args), rel=1e-11)


@pytest.mark.parametrize("args,which", [((0.02, 0.05, 0.03), "d2"),
                                        ((0.031, 0.001, 0.03), "d1")])
def test_i_function_gives_the_limit_where_the_paper_formula_is_0_over_0(args, which):
    """One denominator vanishes identically, so the printed formula is 0/0 there
    (numpy quietly returns 0 or nan). Ours must give the limit -- checked by
    approaching it."""
    ours = float(i_function(*args, BETA))
    approach = [_i_literal(args[0] + eps, args[1], args[2])
                for eps in (1e-6, 1e-8, 1e-10)]
    assert approach[-1] == pytest.approx(ours, rel=1e-6)
    assert abs(approach[-1] - ours) < abs(approach[0] - ours)   # converging


def test_special_functions_survive_realistic_temperatures():
    """The literal formulas need exp(beta*xi) with beta*xi ~ 400 at 30 K and
    xi = 1 eV. Ours must stay finite over the whole window."""
    beta = 1.0 / (30.0 * K_B_HARTREE)
    xi = np.linspace(-0.04, 0.04, 101)
    i = i_function(xi, 1e-4, 2.2e-3, beta)
    j = j_function(xi, 1e-4, 2.2e-3, 8.4e-3, beta)
    assert np.all(np.isfinite(i)) and np.all(np.isfinite(j))
    assert np.any(np.abs(i) > 0.0)


def test_bose_handles_the_negative_branch():
    """The s2 = -1 term of their Eq. (11) evaluates n(-w) = -(1 + n(w))."""
    from waw.utils.scdft import bose

    w, beta = 2.2e-3, 400.0
    assert bose(-w, beta) == pytest.approx(-(1.0 + bose(w, beta)))


# ------------------------------------------------------------------- the grid

def test_energy_grid_is_symmetric_and_excludes_the_fermi_level():
    kT = 20.0 * K_B_HARTREE
    xi, w = energy_grid(kT, cutoff=30.0 * WPH, n_points=200)
    assert np.all(np.abs(xi) > 0.0)                    # 1/tanh diverges at 0
    assert xi == pytest.approx(-xi[::-1])              # symmetric
    assert np.all(np.diff(xi) > 0.0)
    assert w.sum() == pytest.approx(xi[-1] - xi[0], rel=1e-6)
    assert np.abs(xi).min() < kT                       # resolves the tanh scale


# ------------------------------------------------------------------ the solver

def test_z_kernel_is_positive_and_scales_with_lambda():
    """Z is the mass-renormalization kernel: positive, peaked at E_F, falling
    off away from it, and linear in the coupling strength."""
    kT = 20.0 * K_B_HARTREE
    xi, w = energy_grid(kT, cutoff=30.0 * WPH, n_points=160)
    om = np.array([WPH])
    z1 = z_kernel(xi, w, om, np.array([[[0.5 * WPH / 2]]]), 1.0 / kT)[:, 0]
    z2 = z_kernel(xi, w, om, np.array([[[1.0 * WPH / 2]]]), 1.0 / kT)[:, 0]
    mid = len(xi) // 2
    assert np.all(z1 > 0.0)
    assert z2 == pytest.approx(2.0 * z1, rel=1e-9)     # linear in alpha^2F
    assert z1[mid] > z1[-1]                            # peaked at E_F
    assert z1[-1] < 0.05 * z1[mid]                     # decays away from it


def test_tc_tracks_eliashberg_at_moderate_and_strong_coupling():
    """The paper's Fig. 2 claim: with gamma1 = 1.33, gamma2 = 3.8 the functional
    reproduces Eliashberg Tc for the Einstein model. It does so at lambda >= 1
    (0.4-2% here); at lambda = 0.3 ours is 65% high, which the paper's bitmap
    figure cannot settle."""
    from waw.utils.eliashberg import tc_linearized

    for lam, tol in ((1.0, 0.06), (1.5, 0.06)):
        w = np.linspace(0.2 * WPH, 2.0 * WPH, 400)
        sig = 0.01 * WPH
        a2f = (lam * WPH / 2) * np.exp(-0.5 * ((w - WPH) / sig) ** 2) / (
            sig * np.sqrt(2 * np.pi))
        tc_el = tc_linearized(a2f, w, 0.0, omega_c=10 * WPH, n_matsubara=400).tc
        tc_sc = tc_scdft(np.array([WPH]), np.array([[[lam * WPH / 2]]]), 0.0,
                         n_points=120)
        assert tc_sc == pytest.approx(tc_el, rel=tol), f"lambda = {lam}"


def test_tc_falls_with_the_coulomb_parameter():
    om, a2f = np.array([WPH]), np.array([[[1.0 * WPH / 2]]])
    tcs = [tc_scdft(om, a2f, mu, n_points=100) for mu in (0.0, 0.1)]
    assert tcs[0] > tcs[1] > 0.0


def test_delta_s_has_a_local_minimum_at_the_fermi_level():
    """The paper's Fig. 1: Delta_s does NOT peak at E_F -- it has a local
    minimum there with vanishing derivative, which is why it must not be read
    as an excitation gap."""
    om, a2f = np.array([WPH]), np.array([[[1.0 * WPH / 2]]])
    tc = tc_scdft(om, a2f, 0.0, n_points=120)
    res = solve_delta_s(om, a2f, 0.0, 0.9 * tc * K_B_HARTREE, n_points=120,
                        max_iter=600)
    d = np.abs(res.delta_s[:, 0])
    mid = len(d) // 2
    assert d[mid] < d[mid + 10]                        # rises away from E_F
    assert d[mid - 1] < d[mid - 11]
    assert np.abs(res.delta_s_at_fermi()[0]) < res.delta_s_max[0]
    assert d[0] < 0.5 * d.max() and d[-1] < 0.5 * d.max()   # decays at the edges


def test_the_gamma_parameters_are_the_published_values():
    """gamma1 = gamma3 = 1.33 and gamma2 = 3.8 are the KERNEL parameters
    (Nature Rev. Phys. 6, 570 (2024) supplementary Eqs. 4-7). gamma4 = 1.95 is a
    different parameter belonging to the physical-gap functional, not an
    alternative gamma3."""
    assert SCDFT_GAMMAS_KERNEL == (1.33, 3.8, 1.33)
    assert SCDFT_GAMMA4_GAP == 1.95


def test_band_resolved_input_reduces_to_the_single_band_case():
    """A 2x2 alpha^2F whose blocks each carry half the coupling must give the
    same Tc as the single-band spectrum with the full coupling."""
    lam = 1.0
    single = tc_scdft(np.array([WPH]), np.array([[[lam * WPH / 2]]]), 0.0,
                      n_points=100)
    two = tc_scdft(np.array([WPH]),
                   np.full((2, 2, 1), 0.5 * lam * WPH / 2), 0.0, n_points=100)
    assert two == pytest.approx(single, rel=0.02)


def test_closed_form_derivative_matches_a_finite_difference():
    """Eq. (4) needs d/dxi of the symmetrised I. `di_dxi` gives it in closed
    form; a central difference is the independent check, over a range of
    temperatures and including the removable singularities of I itself."""
    from waw.utils.scdft.functions import di_dxi, i_function

    cases = [(0.05, 0.02, 0.01), (0.2, -0.1, 0.06), (0.01, 0.011, 0.03),
             (1e-4, 5e-4, 2.2e-3), (3e-3, 1e-3, 2.2e-3), (0.03, 0.001, 0.03),
             (0.021, 0.02, 0.001), (2.2e-3, 0.0, 2.2e-3)]
    for beta in (40.0, 400.0, 4000.0):
        h = 1e-4 / beta
        for xi, xip, w in cases:
            num = float((i_function(xi + h, xip, w, beta)
                         - i_function(xi - h, xip, w, beta)) / (2 * h))
            ana = float(di_dxi(xi, xip, w, beta))
            if abs(num) > 1e-20:
                assert ana == pytest.approx(num, rel=1e-5), (beta, xi, xip, w)


def test_z_kernel_is_unchanged_by_using_the_closed_form():
    """Switching Eq. (4)'s derivative from a difference to the closed form must
    not move Z: the difference was already accurate, so this is a cleanliness
    change, not a correction."""
    from waw.utils.scdft import energy_grid, z_kernel

    kT = 30.0 * K_B_HARTREE
    xi, w = energy_grid(kT, cutoff=30.0 * WPH, n_points=160)
    z = z_kernel(xi, w, np.array([WPH]), np.array([[[1.0 * WPH / 2]]]), 1.0 / kT)
    mid = len(xi) // 2
    assert np.all(z > 0.0) and z[mid, 0] > z[-1, 0]
    assert z[mid, 0] == pytest.approx(1.8623, rel=2e-3)   # pinned value


def _einstein(wlog_meV, lam):
    w0 = wlog_meV * 1e-3 * EV_TO_HARTREE
    return np.array([w0]), np.array([lam * w0 / 2])


def test_coulomb_cutoff_is_independent_of_the_grid():
    """
    W = mu Theta(L-|xi|) Theta(L-|xi'|): Tc must depend on L, never on the grid.

    Before the Theta factors went in, the Coulomb cutoff WAS the grid extent
    (30 x the phonon maximum), so mu silently meant something different for
    every phonon energy.
    """
    om, a2f = _einstein(40.0, 1.0)
    L = 5.0 * EV_TO_HARTREE
    tcs = [tc_scdft(om, a2f, 0.164, cutoff=f * om[0], mu_cutoff=L, n_points=200)
           for f in (10, 30, 120)]
    assert max(tcs) - min(tcs) < 1e-3 * max(tcs)


def test_coulomb_window_removes_the_phonon_energy_bend():
    """
    Tc/Tc_Eliashberg must not depend on omega_ph once mu carries its own cutoff.

    With the cutoff tied to the grid this ratio bent with omega_ph -- 9 points
    apart at lambda = 0.7 between 20 and 80 meV, with different lambda shapes.
    """
    EF = 5.0 * EV_TO_HARTREE
    mu = 0.164
    for lam in (1.0, 2.0):
        ratios = []
        for wlog in (20.0, 80.0):
            om, a2f = _einstein(wlog, lam)
            ng = 2001
            s = 0.02 * om[0]
            omg = np.linspace(1e-9, 6 * om[0], ng)
            a2g = (a2f[0] * np.exp(-0.5 * ((omg - om[0]) / s) ** 2)
                   / (s * np.sqrt(2 * np.pi)))
            wc = 10 * om[0]
            mus = mu / (1.0 + mu * np.log(EF / wc))
            tc_el = tc_linearized(a2g, omg, mu_star=mus, omega_c=wc).tc
            ratios.append(tc_scdft(om, a2f, mu, mu_cutoff=EF, n_points=200) / tc_el)
        assert abs(ratios[0] / ratios[1] - 1.0) < 0.02


def test_coulomb_window_kills_repulsion_outside_it():
    """
    Both Heaviside factors, read straight off the operator.

    The mu-dependent part of M must vanish identically for |xi| > L (the factor
    on the output index) and for |xi'| > L (the one that cuts the log integral).
    """
    from waw.utils.scdft.solver import gap_operator, z_kernel
    from waw.utils.scdft.solver import _omega_nodes

    om, a2f = _einstein(40.0, 1.0)
    om, a2f_w = _omega_nodes(om, a2f)
    kT = 20.0 * K_B_HARTREE
    beta = 1.0 / kT
    L = 20.0 * om[0]
    xi, w_xi = energy_grid(kT, 60.0 * om[0], 200)
    zero = np.zeros((1, 1))
    m0 = gap_operator(xi, w_xi, om, a2f_w, zero, beta, None, mu_cutoff=L)
    m1 = gap_operator(xi, w_xi, om, a2f_w, zero + 0.3, beta, None, mu_cutoff=L)
    coul = (m1 - m0)[:, 0, :, 0]
    out = np.abs(xi) > L
    assert np.allclose(coul[out, :], 0.0, atol=1e-300)   # Theta(L - |xi|)
    assert np.allclose(coul[:, out], 0.0, atol=1e-300)   # Theta(L - |xi'|)
    assert np.abs(coul[~out][:, ~out]).max() > 0.0


# --------------------------------------------------------------------------
# Certification against a direct Matsubara evaluation of Eqs. (5)-(9).
#
# The kernels below use NO closed form: they sum over Matsubara frequencies
# straight from the Letter's Eqs. (5)-(9) for the Einstein model, with
# sum_k' -> N_F integral dxi' and |g|^2 = lambda*omega/(2 N_F). That makes them
# an independent reference for Eqs. (10)-(11) -- prefactors, the eight s1s2s3
# signs, the J arguments, the isotropic normalisation and the omega weighting
# all at once. Both bugs this module has had would have failed these.
#
#   Sigma12(w_n) = (lam g1 g2^2 w^4/beta) sum_n' A(n') /
#                    [(w_n'^2 + g2^2 w^2)((w_n - w_n')^2 + w^2)]
#   A(n')  = integral dxi' Delta'/(w_n'^2 + xi'^2 + g3 Delta'^2)
#   Sigma11(w_n) = i a(w_n),
#   a(w_n) = (lam w^2/beta) sum_n' w_n' B(n')/((w_n - w_n')^2 + w^2)
#   B(n')  = integral dxi'/(w_n'^2 + xi'^2 + Delta'^2)
#
# Partially linearized as the paper is (the xi_k side at Delta_k -> 0), so
# tau^I -> -tanh(beta xi/2)/(2 xi) and tau^K -> -(1/beta) sum_n Sigma12/(w_n^2 +
# xi^2). Then Delta_s = (tau^K - tau^L)/tau^I gives
#
#   K-part = (2 xi/(beta tanh(beta xi/2))) sum_n Sigma12(w_n)/(w_n^2 + xi^2)
#   Z      = (4 xi/(beta tanh(beta xi/2))) sum_n w_n a(w_n)/(w_n^2 + xi^2)^2
# --------------------------------------------------------------------------

def _matsubara_odd(beta, n_mats):
    return (2 * np.arange(-n_mats, n_mats) + 1) * np.pi / beta


def _direct_kph(xi, w_xi, omega, lam, beta, delta=None, n_mats=6000,
                gammas=SCDFT_GAMMAS_KERNEL):
    """The phonon part of Eq. (1), summed directly. Returns the vector when
    ``delta`` is given, the (n_xi, n_xi') operator when it is None."""
    g1, g2, g3 = gammas
    wn = _matsubara_odd(beta, n_mats)
    e_g3sq = xi ** 2 + (0.0 if delta is None else g3 * delta ** 2)
    amp = w_xi if delta is None else w_xi * delta
    A = amp[None, :] / (wn[:, None] ** 2 + e_g3sq[None, :])       # (n', y)
    A *= (1.0 / (wn ** 2 + (g2 * omega) ** 2))[:, None]
    lorentz = 1.0 / ((wn[:, None] - wn[None, :]) ** 2 + omega ** 2)
    sig12 = (lam * g1 * g2 ** 2 * omega ** 4 / beta) * (lorentz @ A)
    core = 1.0 / (wn[:, None] ** 2 + xi[None, :] ** 2)            # (n, x)
    pref = 2.0 * xi / (beta * np.tanh(0.5 * beta * xi))
    out = pref[:, None] * (core.T @ sig12)
    return out.sum(axis=1) if delta is not None else out


def _direct_z(xi, w_xi, omega, lam, beta, n_mats=6000):
    wn = _matsubara_odd(beta, n_mats)
    B = (w_xi[None, :] / (wn[:, None] ** 2 + xi[None, :] ** 2)).sum(axis=1)
    lorentz = 1.0 / ((wn[:, None] - wn[None, :]) ** 2 + omega ** 2)
    a = (lam * omega ** 2 / beta) * (lorentz @ (wn * B))
    core = 1.0 / (wn[:, None] ** 2 + xi[None, :] ** 2) ** 2
    return (4.0 * xi / (beta * np.tanh(0.5 * beta * xi))) * (core.T @ (wn * a))


def _model(lam, T, npts=100, cut=30.0, wph=WPH):
    from waw.utils.scdft.solver import _omega_nodes
    kT = T * K_B_HARTREE
    xi, w_xi = energy_grid(kT, cut * wph, npts)
    om, a2f_w = _omega_nodes(np.array([wph]), np.array([lam * wph / 2]))
    return xi, w_xi, om, a2f_w, kT, 1.0 / kT


def test_linearized_kph_against_direct_matsubara():
    """Eq. (11) at Delta = 0, elementwise, against the direct sum."""
    from waw.utils.scdft.solver import gap_operator

    xi, w_xi, om, a2f_w, kT, beta = _model(1.0, 80.0)
    code = gap_operator(xi, w_xi, om, a2f_w, np.zeros((1, 1)), beta,
                        None)[:, 0, :, 0]
    direct = _direct_kph(xi, w_xi, WPH, 1.0, beta)
    assert np.abs(code / direct - 1.0).max() < 1e-8


def test_z_against_direct_matsubara():
    """Eq. (10) against the direct sum. LM2005 Eq. (78) writes this kernel with
    the opposite sign of 1/tanh to the Letter's Eq. (10) -- the magnitudes agree
    exactly and 1 + Z > 0 fixes the sign, so compare magnitudes."""
    xi, w_xi, om, a2f_w, kT, beta = _model(1.0, 80.0)
    code = z_kernel(xi, w_xi, om, a2f_w, beta)[:, 0]
    direct = _direct_z(xi, w_xi, WPH, 1.0, beta)
    assert np.all(code > 0.0)
    # 1e-6, not 1e-8: Z's frequency sum has a 1/w_n^3 tail, so the DIRECT side
    # is truncation-limited here (it tightens with n_mats), unlike K^ph's 1/w_n^4
    assert np.abs(code / direct - 1.0).max() < 1e-6


@pytest.mark.parametrize("wph_meV,lam,T", [(30.0, 0.5, 20.0), (60.0, 1.0, 20.0),
                                           (90.0, 2.5, 80.0)])
@pytest.mark.parametrize("g3", [1.33, 1.95])
def test_nonlinear_kph_1_over_e_is_the_gamma3_energy(wph_meV, lam, T, g3):
    """
    The surviving 1/E' in the phonon kernel is sqrt(xi'^2 + gamma3 Delta'^2).

    Eq. (11)'s tanh[(beta/2)E_k'] and Eq. (1)'s tanh(beta E'/2)/E' cancel
    whatever they are, so only 1/E' survives and the printed equations do not
    say which E' it is. The direct route has no such ambiguity. Using the bare
    sqrt(xi'^2 + Delta'^2) instead is a 5.6% error at gamma3 = 1.33 and 14.5% at
    gamma3 = 1.95 -- and it is invisible at Delta = 0, where both reduce to
    |xi'|, so no Tc test can catch it.
    """
    from waw.utils.scdft.solver import gap_operator

    wph = wph_meV * 1e-3 * EV_TO_HARTREE
    gammas = (1.33, 3.8, g3)
    xi, w_xi, om, a2f_w, kT, beta = _model(lam, T, wph=wph)
    delta = 0.35 * wph * np.exp(-(np.abs(xi) - wph) ** 2 / (2 * (0.7 * wph) ** 2))

    M = gap_operator(xi, w_xi, om, a2f_w, np.zeros((1, 1)), beta,
                     delta[:, None], gammas)
    code = np.einsum("xiyj,yj->xi", M, delta[:, None])[:, 0]
    direct = _direct_kph(xi, w_xi, wph, lam, beta, delta=delta, gammas=gammas)
    assert np.abs(code / direct - 1.0).max() < 2e-5

    # and the bare energy is wrong by the ratio of the two energies
    bare = np.hypot(xi, delta)
    eg3 = np.sqrt(xi ** 2 + g3 * delta ** 2)
    assert (eg3 / bare).max() > 1.05                # the two really do differ


def test_tc_converges_as_one_over_the_energy_window():
    """
    Both kernels have power-law xi' tails, so Tc converges only as 1/L.

    This is what made Tc drift against the published Fig. 2: at the old default
    L = 30 omega_max the error is lambda-dependent, +3% at lambda = 0.4 and
    +6% at lambda = 2.8. The 1/L law is what lets 2*Tc(2L) - Tc(L) remove it.
    """
    om, a2f = _einstein(60.0, 1.0)
    tcs = {f: tc_scdft(om, a2f, 0.0, cutoff=f * om[0], n_points=240,
                       t_max=900.0, tol=1e-6) for f in (120, 240, 480, 960)}
    d1, d2, d3 = (tcs[240] - tcs[120], tcs[480] - tcs[240], tcs[960] - tcs[480])
    assert d1 < d2 < d3 < 0.0                          # monotone from above
    assert d2 / d1 == pytest.approx(0.5, abs=0.06)     # 1/L, not log
    assert d3 / d2 == pytest.approx(0.5, abs=0.06)
    assert tcs[120] / (2 * tcs[960] - tcs[480]) - 1.0 > 0.009   # 30 was worse


def test_default_energy_window_is_converged():
    """The default must sit close to the L -> infinity limit, not 4% above it."""
    om, a2f = _einstein(60.0, 1.0)
    default = tc_scdft(om, a2f, 0.0, n_points=240, t_max=900.0, tol=1e-6)
    far = [tc_scdft(om, a2f, 0.0, cutoff=f * om[0], n_points=240, t_max=900.0,
                    tol=1e-6) for f in (480, 960)]
    limit = 2 * far[1] - far[0]
    assert abs(default / limit - 1.0) < 0.006


def test_fig2_ratio_has_the_published_shape():
    """
    Against the digitised Fig. 2: Tc_SCDFT/Tc_Eliashberg must overshoot at weak
    coupling, cross below 1 near lambda = 1, dip, and return toward 1.

    The internal ratio is the digitisation-robust comparison -- it cancels the
    axis calibration, which both published curves share. Before the energy
    window was converged this ratio never crossed below 1 at all.
    """
    from waw.utils.eliashberg import tc_linearized

    def tc_el(lam):
        s = 0.02 * WPH
        w = np.linspace(1e-9, 6 * WPH, 4001)
        a2f = (lam * WPH / 2) * np.exp(-0.5 * ((w - WPH) / s) ** 2) / (
            s * np.sqrt(2 * np.pi))
        return tc_linearized(a2f, w, 0.0, omega_c=10 * WPH).tc

    lams = (0.4, 1.0, 1.6, 2.0, 2.8)
    r = {}
    for lam in lams:
        om, a2f = _einstein(60.0, lam)
        r[lam] = tc_scdft(om, a2f, 0.0, n_points=240, t_max=900.0,
                          tol=1e-6) / tc_el(lam)
    assert r[0.4] == pytest.approx(1.31, abs=0.03)      # paper 1.332
    assert r[1.0] < 1.0 < r[0.4]                        # crosses below 1
    assert min(r.values()) == pytest.approx(0.96, abs=0.02)   # paper 0.976
    assert r[2.8] > r[2.0]                              # returns toward 1


# --------------------------------------------------------------------------
# Cross-check against SCTK (github.com/mitsuaki1987/sctk), a THIRD independent
# implementation of the same functional: Kawamura evaluated the Matsubara sums
# of Eqs. (10)-(11) symbolically and ships them as closed forms in
# src/sctk_kernel_weight.f90 (scdft_kernel = 2, in the tree since 2023 at
# commit 772e8d6). The forms below are its generic branch (all confluence
# special cases avoided by the choice of test points) plus its T = 0 limits.
# The conventions: K^ph_kk' = sum_nu |g|^2 Kweight(beta, |xi|, |xi'|, w) and
# Z_k = -sum_k'nu |g|^2 Zweight(beta, |xi|, |xi'|, w), both per k' state.
#
# NOTE SCTK's production constants are g1 = 1.33, g2 = 3.88 -- NOT the 3.8 the
# Letter prints. The comparison here uses one set in both codes on purpose:
# it pins the analytic FORM, not the fit constants.
# --------------------------------------------------------------------------

def _sctk_zweight(beta, x0, y0, z0):
    x, y, z = beta * x0, beta * y0, beta * z0
    tx, ty, tz = np.tanh(0.5 * x), np.tanh(0.5 * y), np.tanh(0.5 * z)
    mp, pm = 1.0 / (-x + y + z), 1.0 / (x - y + z)
    mm, pp = 1.0 / (-x - y + z), 1.0 / (x + y + z)
    wz = (-(((1 + tx * ty) * tz + tx + ty) * pp - 0.5 * (1 - tx**2) * (1 + ty * tz)) * pp
          - (((1 - tx * ty) * tz + tx - ty) * pm - 0.5 * (1 - tx**2) * (1 - ty * tz)) * pm
          + (((1 - tx * ty) * tz - tx + ty) * mp - 0.5 * (1 - tx**2) * (1 + ty * tz)) * mp
          + (((1 + tx * ty) * tz - tx - ty) * mm - 0.5 * (1 - tx**2) * (1 - ty * tz)) * mm
          ) / (4.0 * tz * tx)
    return wz * beta**2


def _sctk_kweight(beta, x0, y0, z0, g1, g2):
    x, y, z = beta * x0, beta * y0, beta * z0
    tx, ty, tz = np.tanh(0.5 * x), np.tanh(0.5 * y), np.tanh(0.5 * z)
    tg = np.tanh(0.5 * g2 * z)
    wk = (
        -((x - z) * (-tg + ty) * (1 - tx * tz) + g2 * z * (tx + ty - tz - tx * ty * tz)
          - y * (tx - tz + tg - tx * tz * tg))
        / ((x + y - z) * (-x - (g2 - 1) * z) * (y - g2 * z))
        + ((x - z) * (-tg - ty) * (1 - tx * tz) + g2 * z * (tx - ty - tz + tx * ty * tz)
           + y * (tx - tz + tg - tx * tz * tg))
        / ((x - y - z) * (-x - (g2 - 1) * z) * (-y - g2 * z))
        + ((-x - z) * (-tg + ty) * (1 + tx * tz) + g2 * z * (-tx + ty - tz + tx * ty * tz)
           - y * (-tx - tz + tg + tx * tz * tg))
        / ((-x + y - z) * (x - (g2 - 1) * z) * (y - g2 * z))
        - ((-x - z) * (-tg - ty) * (1 + tx * tz) + g2 * z * (-tx - ty - tz - tx * ty * tz)
           + y * (-tx - tz + tg + tx * tz * tg))
        / ((-x - y - z) * (x - (g2 - 1) * z) * (-y - g2 * z))
        + ((x - z) * (tg + ty) * (1 - tx * tz) - g2 * z * (tx + ty - tz - tx * ty * tz)
           - y * (tx - tz - tg + tx * tz * tg))
        / ((x + y - z) * (-x + (g2 + 1) * z) * (y + g2 * z))
        - ((x - z) * (tg - ty) * (1 - tx * tz) - g2 * z * (tx - ty - tz + tx * ty * tz)
           + y * (tx - tz - tg + tx * tz * tg))
        / ((x - y - z) * (-x + (g2 + 1) * z) * (-y + g2 * z))
        - ((-x - z) * (tg + ty) * (1 + tx * tz) - g2 * z * (-tx + ty - tz + tx * ty * tz)
           - y * (-tx - tz - tg - tx * tz * tg))
        / ((-x + y - z) * (x + (g2 + 1) * z) * (y + g2 * z))
        + ((-x - z) * (tg - ty) * (1 + tx * tz) - g2 * z * (-tx - ty - tz - tx * ty * tz)
           + y * (-tx - tz - tg - tx * tz * tg))
        / ((-x - y - z) * (x + (g2 + 1) * z) * (-y + g2 * z))
    ) * g1 * g2 * z / (4.0 * tx * ty * tz)
    return wk * beta


_SCTK_POINTS = [(0.31, 0.83), (1.57, 0.91), (5.13, 2.31), (0.11, 12.3),
                (25.7, 0.37), (2.71, 2.93)]


def test_kph_matches_sctk_closed_form():
    """Eq. (11), linearized, against SCTK's symbolic Matsubara summation."""
    from waw.utils.scdft.functions import j_function

    g1, g2, _ = SCDFT_GAMMAS_KERNEL
    beta = 2000.0
    for a, b in _SCTK_POINTS:
        xi, xip = a * WPH, b * WPH
        s = sum(s1 * s2 * s3 * float(j_function(xi, s1 * xip, s2 * WPH,
                                                s3 * g2 * WPH, beta))
                for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1))
        ours = g1 * g2 * WPH * s / (np.tanh(0.5 * beta * xi)
                                    * np.tanh(0.5 * beta * xip))
        sctk = _sctk_kweight(beta, xi, xip, WPH, g1, g2)
        assert ours == pytest.approx(sctk, rel=1e-9)
        # and the T = 0 closed form
        cold = sum(s1 * s2 * s3 * float(j_function(xi, s1 * xip, s2 * WPH,
                                                   s3 * g2 * WPH, 2e5))
                   for s1 in (1, -1) for s2 in (1, -1) for s3 in (1, -1))
        cold *= g1 * g2 * WPH / (np.tanh(1e5 * xi) * np.tanh(1e5 * xip))
        t0 = -2 * g1 * g2 * WPH * (xi + xip + WPH + g2 * WPH) / (
            (xi + xip + WPH) * (xip + g2 * WPH) * (xi + WPH + g2 * WPH))
        assert cold == pytest.approx(t0, rel=1e-9)


def test_z_integrand_matches_sctk_closed_form():
    """Eq. (10)'s integrand against SCTK's, incl. the -1/(x+y+z)^2 limit."""
    from waw.utils.scdft.functions import di_dxi

    beta = 2000.0
    for a, b in _SCTK_POINTS:
        xi, xip = a * WPH, b * WPH
        d = float(di_dxi(xi, xip, WPH, beta)) + float(di_dxi(xi, -xip, WPH, beta))
        ours = -d / np.tanh(0.5 * beta * xi)
        assert ours == pytest.approx(-_sctk_zweight(beta, xi, xip, WPH), rel=1e-9)
        d0 = float(di_dxi(xi, xip, WPH, 2e5)) + float(di_dxi(xi, -xip, WPH, 2e5))
        cold = -d0 / np.tanh(1e5 * xi)
        assert cold == pytest.approx(1.0 / (xi + xip + WPH)**2, rel=1e-9)


# ---------------------------------------------------------------------------
# The LM2005 functional (Luders et al., Phys. Rev. B 72, 024545 (2005)).
# ---------------------------------------------------------------------------

def _jt_literal(xi, xip, w, b):
    """Their Eq. (81) exactly as printed, valid away from xi' = xi - w."""
    from waw.utils.scdft.functions import bose, fermi
    d = xi - xip - w
    return -(fermi(xi, b) + bose(w, b)) / d * (
        (fermi(xip, b) - fermi(xi - w, b)) / d
        - b * fermi(xi - w, b) * fermi(-xi + w, b))


def test_j_lueders_matches_the_literal_equation_81():
    """Away from its removable pole the divided-difference form must reproduce
    Eqs. (80)-(81) term for term."""
    from waw.utils.scdft.functions import j_lueders
    b = 1.0 / 0.002
    for xi, xip, w in [(0.01, 0.02, 0.005), (-0.03, 0.011, 0.004),
                       (0.05, -0.02, 0.009), (0.002, 0.05, 0.003)]:
        got = float(j_lueders(xi, xip, w, b))
        ref = _jt_literal(xi, xip, w, b) - _jt_literal(xi, xip, -w, b)
        assert abs(got - ref) <= 1e-10 * abs(ref)


def test_j_lueders_is_finite_at_its_removable_pole():
    """The literal form diverges at xi' = xi - w; the divided difference must
    stay finite and continuous there."""
    from waw.utils.scdft.functions import j_lueders
    b, xi, w = 1.0 / 0.002, 0.02, 0.005
    at = float(j_lueders(xi, xi - w, w, b))
    near = float(j_lueders(xi, xi - w + 1e-7, w, b))
    assert np.isfinite(at)
    assert abs(at - near) < 1e-3 * abs(at)


def test_lm2005_z_is_lambda_where_spg_is_two_lambda():
    """The whole point of Luders Eq. (79): their own text says the symmetrised
    Eq. (78) gives Z(0) ~ 2*lambda, "twice the value expected", and Eq. (79)
    repairs it to ~lambda. SPG kept Eq. (78), so the two functionals must differ
    by very close to a factor two at the Fermi surface."""
    from waw.utils.scdft.solver import _omega_nodes, energy_grid, z_kernel
    lam = 1.0
    om, a2f_w = _omega_nodes(np.array([WPH]), np.array([lam * WPH / 2]))
    kT = 0.02 * WPH
    xi, w_xi = energy_grid(kT, 400.0 * WPH, 600)
    i0 = int(np.argmin(np.abs(xi)))
    z_spg = z_kernel(xi, w_xi, om, a2f_w, 1.0 / kT, "spg")[i0, 0]
    z_lm = z_kernel(xi, w_xi, om, a2f_w, 1.0 / kT, "lm2005")[i0, 0]
    assert 1.85 * lam < z_spg < 2.05 * lam
    assert 0.90 * lam < z_lm < 1.05 * lam


def test_lm2005_pairing_kernel_has_even_parity_in_xi_prime():
    """[I(xi,xi') - I(xi,-xi')] is ODD in xi', and so is Eq. (74)'s
    1/tanh(beta xi'/2). Cancelling the latter against Eq. (1)'s
    tanh(beta E'/2)/E' -- which is legitimate for SPG -- leaves an operator odd
    in xi' here, and an odd operator annihilates every even gap, giving rho = 0.
    Guard the parity directly."""
    from waw.utils.scdft.solver import (_omega_nodes, energy_grid, gap_operator,
                                        z_kernel)
    om, a2f_w = _omega_nodes(np.array([WPH]), np.array([1.0 * WPH / 2]))
    kT = 0.05 * WPH
    xi, w_xi = energy_grid(kT, 100.0 * WPH, 120)
    beta = 1.0 / kT
    M = gap_operator(xi, w_xi, om, a2f_w, np.zeros((1, 1)), beta, None,
                     functional="lm2005")[:, 0, :, 0]
    # xi grid is symmetric, so reversing it maps xi' -> -xi'
    assert np.allclose(M, M[:, ::-1], rtol=0, atol=1e-8 * np.max(np.abs(M)))
    z = z_kernel(xi, w_xi, om, a2f_w, beta, "lm2005")
    rho = np.linalg.eigvals(
        (M / (1.0 + z[:, 0])[:, None])).real.max()
    assert rho > 0.1, "an odd operator would give rho = 0"


def test_lm2005_tc_is_below_spg():
    """LM2005 is documented to underestimate Tc -- that is what SPG was built to
    fix (their Table: LM2005 is below experiment for all eleven materials)."""
    om, a2f = np.array([WPH]), np.array([1.0 * WPH / 2])
    tc_spg = tc_scdft(om, a2f, 0.0, cutoff=400 * WPH, n_points=140,
                      functional="spg")
    tc_lm = tc_scdft(om, a2f, 0.0, cutoff=400 * WPH, n_points=140,
                     functional="lm2005")
    assert 0.0 < tc_lm < tc_spg


def test_unknown_functional_is_rejected():
    om, a2f = np.array([WPH]), np.array([1.0 * WPH / 2])
    with pytest.raises(ValueError, match="functional"):
        tc_scdft(om, a2f, 0.0, n_points=60, functional="nope")


# ---------------------------------------------------------------------------
# The unexpanded (Matsubara) construction. EXPERIMENTAL -- see its docstring.
# ---------------------------------------------------------------------------

def test_unexpanded_reproduces_the_normal_state_z():
    """Z_n is the ordinary Migdal mass renormalisation, so Z(w -> 0) = 1 + lambda.
    Everything else in the module is built on it."""
    from waw.utils.scdft.unexpanded import build_kernels
    for lam in (0.5, 1.0, 1.5):
        k = build_kernels(np.array([WPH]), np.array([lam * WPH / 2]), 0.0,
                          20.0 * K_B_HARTREE, band_edge=10 * WPH, n_xi=120)
        z0 = k.z_n[int(np.argmin(np.abs(k.omega_n))), 0]
        assert abs(z0 - (1.0 + lam)) < 0.01 * (1.0 + lam)


def test_unexpanded_einstein_lambda_is_special_cased():
    """`lambda_kernel` integrates with the trapezoid rule and returns 0 on a
    one-point grid, which would silently give lambda = 0 and Tc = 0. The single
    Einstein mode must be handled analytically instead."""
    from waw.utils.scdft.unexpanded import build_kernels
    k = build_kernels(np.array([WPH]), np.array([1.0 * WPH / 2]), 0.0,
                      20.0 * K_B_HARTREE, band_edge=10 * WPH, n_xi=80)
    assert np.max(np.abs(k.z_n - 1.0)) > 0.5, "lambda collapsed to zero"
    assert np.isfinite(k.operator).all()


def test_unexpanded_matsubara_reach_is_enforced():
    """A grid that stops at the band edge cannot build the Morel-Anderson sign
    change and would return Tc = 0 -- a failure that looks like physics. It must
    raise instead."""
    from waw.utils.scdft.unexpanded import build_kernels
    with pytest.raises(ValueError, match="Morel-Anderson|band edge"):
        build_kernels(np.array([WPH]), np.array([1.0 * WPH / 2]), 0.0,
                      20.0 * K_B_HARTREE, band_edge=10 * WPH, n_half=20,
                      n_xi=80)


def test_unexpanded_tc_approaches_eliashberg_for_a_wide_band():
    """At mu = 0 the band edge also truncates the phonon xi' integral, so Tc
    rises toward the Eliashberg value as the band widens. The Lorentzian closure
    is documented to overshoot by a few percent once the band is wide enough."""
    from waw.utils.scdft.unexpanded import tc_unexpanded
    from waw.utils.eliashberg import tc_linearized
    lam = 1.0
    sig = 0.02 * WPH
    w = np.linspace(WPH - 8 * sig, WPH + 8 * sig, 401)
    a2g = (lam * WPH / 2) * np.exp(-0.5 * ((w - WPH) / sig) ** 2) / (
        sig * np.sqrt(2 * np.pi))
    tc_el = tc_linearized(a2g, w, 0.0, omega_c=10 * WPH, n_matsubara=800).tc
    args = dict(band_edge=None, n_xi=250, profile="lorentzian", t_min=5.0)
    r = []
    for f in (10, 40):
        args["band_edge"] = f * WPH
        r.append(tc_unexpanded(np.array([WPH]), np.array([lam * WPH / 2]), 0.0,
                               **args) / tc_el)
    assert r[0] < r[1], "Tc must rise as the band widens"
    assert 0.95 < r[1] < 1.15


def test_unexpanded_rejects_bad_inputs():
    from waw.utils.scdft.unexpanded import build_kernels
    om, a2f = np.array([WPH]), np.array([WPH / 2])
    kw = dict(band_edge=10 * WPH)
    with pytest.raises(ValueError, match="profile"):
        build_kernels(om, a2f, 0.0, 20.0 * K_B_HARTREE, profile="nope", **kw)
    with pytest.raises(ValueError, match="n_basis"):
        build_kernels(om, a2f, 0.0, 20.0 * K_B_HARTREE, n_basis=0, **kw)
    with pytest.raises(ValueError, match="mu"):
        build_kernels(om, a2f, np.zeros((3, 3)), 20.0 * K_B_HARTREE, **kw)
    two = np.zeros((2, 2, 1)) + WPH / 2
    with pytest.raises(ValueError, match="band_edge"):
        build_kernels(om, two, 0.0, 20.0 * K_B_HARTREE,
                      band_edge=np.array([1.0, 2.0, 3.0]) * WPH)


def test_unexpanded_two_degenerate_bands_reduce_to_one():
    """A single band split into two identical halves (each row of a2f and of mu
    summing to the single-band values) must give the same leading eigenvalue,
    for every profile. This pins the multiband bookkeeping: any mistake in the
    band sums, the per-band Z, or the block projection breaks the identity."""
    from waw.utils.scdft.unexpanded import linearized_eigenvalue_unexpanded
    lam, mu, kT = 1.0, 0.2, 20.0 * K_B_HARTREE
    a1 = np.array([lam * WPH / 2])
    a2 = np.full((2, 2, 1), lam * WPH / 4)
    kw = dict(band_edge=10 * WPH, n_xi=150)
    for profile in ("lorentzian", "galerkin", "exact"):
        r1 = linearized_eigenvalue_unexpanded(np.array([WPH]), a1, mu, kT,
                                              profile=profile, **kw)
        r2 = linearized_eigenvalue_unexpanded(np.array([WPH]), a2,
                                              np.full((2, 2), mu / 2), kT,
                                              profile=profile, **kw)
        assert abs(r1 - r2) < 1e-10 * abs(r1), (profile, r1, r2)


def test_unexpanded_galerkin_converges_where_the_hierarchy_diverges():
    """The Krylov-basis Galerkin closure must converge in P to profile="exact".
    The single-vector hierarchy (profile="iterate") does NOT converge in k once
    the Coulomb repulsion is strong: the kernel's negative Coulomb eigenvalue
    exceeds the physical one in modulus, so repeated application amplifies the
    wrong branch (measured here: 250%+ error by k = 12, where galerkin P = 4
    is at 1e-5). This is the reason "galerkin" is the default and "iterate" is
    kept only for the record.

    Compared at fixed T through the leading eigenvalue rather than through Tc:
    a Tc bisection would evaluate at low temperatures, where the Matsubara grid
    that has to span the band makes the eigensolve very expensive.
    """
    from waw.utils.scdft.unexpanded import linearized_eigenvalue_unexpanded
    om, a2f = np.array([WPH]), np.array([0.5 * WPH / 2])
    kw = dict(band_edge=10 * WPH, n_xi=150)
    kT = 10.0 * K_B_HARTREE
    mu = 0.35
    rho_exact = linearized_eigenvalue_unexpanded(om, a2f, mu, kT,
                                                 profile="exact", **kw)
    rho_g4 = linearized_eigenvalue_unexpanded(om, a2f, mu, kT,
                                              profile="galerkin", n_basis=4, **kw)
    assert abs(rho_g4 - rho_exact) < 1e-4 * rho_exact, (rho_g4, rho_exact)
    rho_it = linearized_eigenvalue_unexpanded(om, a2f, mu, kT,
                                              profile="iterate", n_iter=12, **kw)
    assert abs(rho_it - rho_exact) > 0.5 * rho_exact, \
        "the bare hierarchy unexpectedly converged; revisit the default choice"


def test_unexpanded_galerkin_matches_exact_on_real_mgb2():
    """The archived two-band alpha^2F_ij matrix of MgB2 (the same file the
    Eliashberg cross-code tests use): the default P = 4 Galerkin basis must
    reproduce the exact-profile eigenvalue on a real material's coupling
    matrix, interband channels and full frequency structure included. Compared
    at fixed T through the eigenvalue; the Tc-level result (ratio 1.0000 at
    mu = 0 and 0.1) is in the derivation notes' scripts/unexpanded_mgb2.py."""
    import pathlib
    from waw.utils.scdft.unexpanded import linearized_eigenvalue_unexpanded
    z = np.load(pathlib.Path(__file__).parent / "data" / "eliashberg"
                / "mgb2_a2f.npz")
    om, a2f = z["omega"], z["a2f"]
    kw = dict(band_edge=0.023, n_xi=150)         # ~10 x w_ln of this spectrum
    kT = 40.0 * K_B_HARTREE
    mu = np.full((2, 2), 0.1)
    rho_g = linearized_eigenvalue_unexpanded(om, a2f, mu, kT,
                                             profile="galerkin", **kw)
    rho_x = linearized_eigenvalue_unexpanded(om, a2f, mu, kT,
                                             profile="exact", **kw)
    assert abs(rho_g - rho_x) < 5e-4 * rho_x, (rho_g, rho_x)


def test_unexpanded_lorentzian_is_galerkin_with_one_vector():
    """profile="lorentzian" is by construction the P = 1 special case."""
    from waw.utils.scdft.unexpanded import linearized_eigenvalue_unexpanded
    om, a2f = np.array([WPH]), np.array([1.0 * WPH / 2])
    kw = dict(band_edge=10 * WPH, n_xi=120)
    kT = 20.0 * K_B_HARTREE
    r_lor = linearized_eigenvalue_unexpanded(om, a2f, 0.2, kT,
                                             profile="lorentzian", **kw)
    r_g1 = linearized_eigenvalue_unexpanded(om, a2f, 0.2, kT,
                                            profile="galerkin", n_basis=1, **kw)
    assert r_lor == pytest.approx(r_g1, rel=1e-12)


def test_unexpanded_exact_profile_solves_the_eliashberg_eigenproblem():
    """profile="exact" must take the leading REAL eigenvector. With a repulsive
    mu the Matsubara kernel carries a large negative eigenvalue whose modulus
    exceeds the physical one, so a largest-modulus power iteration would pick
    the wrong branch and the profile would come out sign-alternating."""
    from waw.utils.scdft.unexpanded import build_kernels
    k = build_kernels(np.array([WPH]), np.array([1.0 * WPH / 2]), 0.3,
                      60.0 * K_B_HARTREE, band_edge=20 * WPH, n_xi=120,
                      profile="exact")
    ph = k.phihat[:, 0, 0]
    n0 = int(np.argmin(np.abs(k.omega_n)))
    assert ph[n0] > 0.0, "profile must be positive at low frequency"
    # Morel-Anderson: with mu > 0 it must change sign once at high frequency,
    # not oscillate the way a wrong-branch eigenvector would.
    sign_changes = int(np.sum(np.diff(np.sign(ph[n0:])) != 0))
    assert sign_changes == 1, f"expected one sign change, got {sign_changes}"
