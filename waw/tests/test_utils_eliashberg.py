"""
Band-resolved linearized Eliashberg solver (`waw/utils/eliashberg`).

The headline tests are cross-code. An independent Fortran implementation
solves the FULL nonlinear equations and locates Tc by fitting a model curve to
Delta(T); run on two systems with mu* = 0.11 and omega_c = 500 meV it reports

    CaC6 (single band)         :  Tc = 14.149 K
    MgB2 (2x2 sigma/pi matrix) :  Tc = 32.011 K

Its alpha^2F for both, converted to atomic units, is archived in
tests/data/eliashberg/*.npz together with those settings and reference Tc. The
linearized solver should land just below them, because the gap-curve
extrapolation slightly overshoots; a consistent -0.3% on both systems is the
expected signature, whereas a discrepancy differing in sign or size between a
single-band and a two-band case would indicate a formulation error.

The rest pins the pieces that a wrong sign or a misfolded kernel would break
while still leaving Tc plausible.
"""

import pathlib

import numpy as np
import pytest

from waw.utils.eliashberg import (
    coulomb_weights,
    isotropic_average,
    lambda_kernel,
    lambda_plus_minus,
    leading_eigenvalue,
    matsubara_frequencies,
    tc_linearized,
)
from waw.utils.eliashberg.cli import load_a2f
from waw.units import HARTREE_TO_EV, K_B_HARTREE

DATA = pathlib.Path(__file__).parent / "data" / "eliashberg"


class _Case:
    """One archived alpha^2F plus the settings and reference Tc that go with it."""

    def __init__(self, stem):
        z = np.load(DATA / f"{stem}_a2f.npz")
        self.omega = z["omega"]
        self.a2f = z["a2f"]
        self.mu_star = float(z["mu_star"])
        self.omega_c = float(z["omega_c"])
        self.n_matsubara = int(z["n_matsubara"])
        self.reference_tc = float(z["reference_tc"])

    def lambda_matrix(self):
        return np.trapezoid(2.0 * self.a2f / self.omega, self.omega, axis=-1)


def _load(stem):
    c = _Case(stem)
    return c, c


# ----------------------------------------------------------------- cross-code

@pytest.mark.parametrize("stem", ["cac6", "mgb2"])
def test_tc_matches_the_reference_fortran_solver(stem):
    data, settings = _load(stem)
    res = tc_linearized(data.a2f, data.omega, settings.mu_star,
                        omega_c=settings.omega_c,
                        n_matsubara=settings.n_matsubara)
    ref = data.reference_tc
    assert res.tc == pytest.approx(ref, rel=0.01), (
        f"{stem}: linearized Tc {res.tc:.3f} K vs reference {ref:.3f} K"
    )
    assert res.tc < ref                    # below it, for the reason in the docstring
    assert res.rho_at_tc == pytest.approx(1.0, abs=1e-3)


def test_lambda_reproduces_the_reference_value_for_cac6():
    """The reference prints lambda = 0.77760233847502902 for CaC6.

    It integrates with a rectangle rule at the grid spacing; we use the
    trapezoid rule, so the two differ by 3.4e-6 relative -- quadrature, not
    physics. Reproducing its rule reproduces its digits to 4e-16, which is the
    real statement that the integrand matches.
    """
    data, _ = _load("cac6")
    ref = 0.7776023384750290
    assert data.lambda_matrix()[0, 0] == pytest.approx(ref, rel=1e-5)

    omega, a2f = data.omega, data.a2f[0, 0]
    rectangle = float((2.0 * a2f / omega).sum() * (omega[1] - omega[0]))
    assert rectangle == pytest.approx(ref, rel=1e-14)


def test_mgb2_is_two_gap_s_plus_plus_with_dominant_sigma_coupling():
    """MgB2's physics, which a transposed band matrix would scramble: the
    sigma band carries the strong intraband coupling and both gaps share a
    sign."""
    data, settings = _load("mgb2")
    lam = data.lambda_matrix()
    assert lam[0, 0] > lam[1, 1] > 0            # sigma-sigma is the strongest
    assert lam[0, 1] > 0 and lam[1, 0] > 0      # interband present
    res = tc_linearized(data.a2f, data.omega, settings.mu_star,
                        omega_c=settings.omega_c, n_matsubara=256)
    assert res.gap_symmetry is not None
    assert np.all(res.gap_symmetry > 0)         # s++, not s+-


# ------------------------------------------------------------------- numerics

def test_power_iteration_agrees_with_dense_diagonalisation():
    """The kernel is not symmetric (the 1/(Z_n omega_m) prefactor breaks it),
    so the dominant eigenvalue is not variational and power iteration needs
    checking against a full diagonalisation."""
    data, settings = _load("cac6")
    for t in (10.0, 14.1, 20.0):
        kT = t * K_B_HARTREE
        kw = dict(n_matsubara=150, omega_c=settings.omega_c)
        a = leading_eigenvalue(data.a2f, data.omega, settings.mu_star, kT,
                               method="power", **kw).rho
        b = leading_eigenvalue(data.a2f, data.omega, settings.mu_star, kT,
                               method="dense", **kw).rho
        assert a == pytest.approx(b, rel=1e-8)


def test_rho_decreases_through_one_at_tc():
    data, settings = _load("cac6")
    res = tc_linearized(data.a2f, data.omega, settings.mu_star,
                        omega_c=settings.omega_c, n_matsubara=200)
    kw = dict(n_matsubara=200, omega_c=settings.omega_c)
    below = leading_eigenvalue(data.a2f, data.omega, settings.mu_star,
                               0.8 * res.tc * K_B_HARTREE, **kw).rho
    above = leading_eigenvalue(data.a2f, data.omega, settings.mu_star,
                               1.2 * res.tc * K_B_HARTREE, **kw).rho
    assert below > 1.0 > above


def test_tc_falls_as_mu_star_rises_and_vanishes_when_it_dominates():
    data, _ = _load("cac6")
    tcs = [tc_linearized(data.a2f, data.omega, mu, n_matsubara=200).tc
           for mu in (0.0, 0.08, 0.11, 0.16)]
    assert tcs[0] > tcs[1] > tcs[2] > tcs[3] > 0.0
    assert tc_linearized(data.a2f, data.omega, 5.0, n_matsubara=200).tc == 0.0


def test_tc_is_converged_in_the_matsubara_count():
    """Truncating the Matsubara sum drops pair-breaking high-frequency weight
    and so overestimates Tc; the default settings must be past that."""
    data, settings = _load("cac6")
    coarse = tc_linearized(data.a2f, data.omega, settings.mu_star,
                           omega_c=settings.omega_c, n_matsubara=64).tc
    fine = tc_linearized(data.a2f, data.omega, settings.mu_star,
                         omega_c=settings.omega_c, n_matsubara=1000).tc
    default = tc_linearized(data.a2f, data.omega, settings.mu_star,
                            omega_c=settings.omega_c).tc
    assert coarse > fine                       # truncation raises Tc
    assert default == pytest.approx(fine, rel=2e-3)


# -------------------------------------------------------------------- kernels

def test_lambda_at_zero_index_difference_is_the_static_lambda():
    omega = np.linspace(1e-4, 4e-3, 400)
    a2f = 0.3 * np.exp(-((omega - 2e-3) / 3e-4) ** 2)
    static = np.trapezoid(2.0 * a2f / omega, omega)
    lam = lambda_kernel(a2f, omega, kT=1e-4, n_max=3)
    assert lam[0, 0, 0] == pytest.approx(static, rel=1e-12)
    assert np.all(np.diff(lam[:, 0, 0]) < 0)          # decays with |n-m|


def test_lambda_kernel_rejects_a_zero_frequency():
    omega = np.linspace(0.0, 4e-3, 50)
    with pytest.raises(ValueError, match="strictly positive"):
        lambda_kernel(np.ones_like(omega), omega, kT=1e-4, n_max=1)


def test_folded_kernels_have_the_documented_structure():
    lam = np.arange(20, dtype=np.float64).reshape(20, 1, 1)
    lp, lm = lambda_plus_minus(lam, 4)
    for n in range(4):
        for m in range(4):
            assert lp[n, m, 0, 0] == abs(n - m) + (n + m + 1)
            assert lm[n, m, 0, 0] == abs(n - m) - (n + m + 1)


def test_matsubara_frequencies_are_the_odd_multiples():
    kT = 1e-4
    w = matsubara_frequencies(kT, 5)
    assert w == pytest.approx(np.pi * kT * np.array([1, 3, 5, 7, 9]))


def test_coulomb_weights_place_the_cutoff_continuously():
    """A sharp integer cutoff would make Tc(omega_c) a sawtooth; the last
    Matsubara point carries a fractional weight instead."""
    kT, n = 1e-4, 40
    w = matsubara_frequencies(kT, n)
    for omega_c in (w[5], 0.5 * (w[5] + w[6]), w[6]):
        c = coulomb_weights(kT, n, omega_c)
        assert np.all(c >= 0.0) and np.all(c <= 1.0)
        # sum(c) = x + 1 where omega_c = pi kT (2x + 1), so inverting the
        # weighted count must return omega_c exactly -- including between grid
        # points, which is the whole point of the fractional last weight.
        effective = np.pi * kT * (2.0 * c.sum() - 1.0)
        assert effective == pytest.approx(omega_c, rel=1e-12)
    assert coulomb_weights(kT, n, w[3])[:4].sum() == pytest.approx(4.0)


def test_isotropic_average_weights_on_the_row_index():
    """a2F_ij carries the DOS of band j, so the average weights by N_i."""
    a2f = np.zeros((2, 2, 3))
    a2f[0, :, :] = 1.0
    a2f[1, :, :] = 3.0
    avg = isotropic_average(a2f, dos=np.array([1.0, 0.0]))
    assert avg == pytest.approx(np.full(3, 2.0))       # only band 0's row: 1+1
    avg2 = isotropic_average(a2f, dos=np.array([0.0, 1.0]))
    assert avg2 == pytest.approx(np.full(3, 6.0))      # only band 1's row: 3+3


def test_single_band_input_is_accepted_as_a_bare_spectrum():
    data, settings = _load("cac6")
    flat = data.a2f[0, 0]
    a = tc_linearized(flat, data.omega, settings.mu_star,
                      omega_c=settings.omega_c, n_matsubara=200).tc
    b = tc_linearized(data.a2f, data.omega, settings.mu_star,
                      omega_c=settings.omega_c, n_matsubara=200).tc
    assert a == pytest.approx(b, rel=1e-12)


def test_mu_star_matrix_reduces_to_the_scalar_when_uniform():
    data, settings = _load("mgb2")
    scalar = tc_linearized(data.a2f, data.omega, 0.11,
                           omega_c=settings.omega_c, n_matsubara=128).tc
    matrix = tc_linearized(data.a2f, data.omega, np.full((2, 2), 0.11),
                           omega_c=settings.omega_c, n_matsubara=128).tc
    assert scalar == pytest.approx(matrix, rel=1e-12)


def test_mu_star_shape_is_checked():
    data, _ = _load("mgb2")
    with pytest.raises(ValueError, match=r"scalar or \(2, 2\)"):
        tc_linearized(data.a2f, data.omega, np.zeros((3, 3)), n_matsubara=32)


# ------------------------------------------------------------------- loading

def test_archived_data_has_the_expected_shapes_and_settings():
    cac6, mgb2 = _Case("cac6"), _Case("mgb2")
    assert cac6.a2f.shape == (1, 1, 2500) and mgb2.a2f.shape == (2, 2, 500)
    assert np.all(cac6.omega > 0.0) and np.all(mgb2.omega > 0.0)
    assert mgb2.omega.max() * HARTREE_TO_EV * 1e3 == pytest.approx(102.4, abs=0.1)
    for c in (cac6, mgb2):
        assert c.mu_star == 0.11
        assert c.omega_c * HARTREE_TO_EV * 1e3 == pytest.approx(500.0, rel=1e-6)


def test_cli_loader_round_trips_npz_and_text(tmp_path):
    """The CLI reads a native .npz or a columnar text file; the text layout puts
    the alpha^2F band blocks in row-major (i, j) order after the omega column."""
    data, _ = _load("mgb2")
    npz = tmp_path / "a2f.npz"
    np.savez(npz, omega=data.omega, a2f=data.a2f)
    w, a = load_a2f(npz)
    assert w == pytest.approx(data.omega) and a == pytest.approx(data.a2f)

    txt = tmp_path / "a2f.dat"
    cols = np.column_stack([data.omega, data.a2f.reshape(4, -1).T])
    np.savetxt(txt, cols)
    w2, a2 = load_a2f(txt)
    assert w2 == pytest.approx(data.omega, rel=1e-10)
    assert a2 == pytest.approx(data.a2f, rel=1e-10, abs=1e-14)


def test_cli_loader_rejects_an_ambiguous_column_count(tmp_path):
    txt = tmp_path / "bad.dat"
    np.savetxt(txt, np.ones((10, 4)))          # 3 alpha^2F columns: not a square
    with pytest.raises(ValueError, match="perfect"):
        load_a2f(txt)


def test_cli_loader_converts_units(tmp_path):
    from waw.units import CM1_TO_HARTREE
    npz = tmp_path / "a2f.npz"
    np.savez(npz, omega=np.array([100.0, 200.0]), a2f=np.array([0.1, 0.2]))
    w, _ = load_a2f(npz, unit="cm-1")
    assert w == pytest.approx(np.array([100.0, 200.0]) * CM1_TO_HARTREE)


# ------------------------------------------------------------- Coulomb cutoff

def test_default_coulomb_cutoff_is_ten_times_the_phonon_maximum():
    """EPW's documentation calls 10x the maximum phonon frequency the common
    choice, and its own Pb example is exactly that (phonons to ~10 meV,
    wscut = 100 meV)."""
    data, _ = _load("mgb2")
    res = tc_linearized(data.a2f, data.omega, 0.11, n_matsubara=128)
    assert res.omega_c == pytest.approx(10.0 * data.omega.max())


def test_matsubara_extent_stays_above_the_coulomb_cutoff():
    """EPW uses one variable for both, so converging its phonon sum silently
    redefines mu*. Here they are separate, and the sum must never be truncated
    inside the Coulomb window."""
    data, _ = _load("cac6")
    omega_c = 3.0 * data.omega.max()
    res = tc_linearized(data.a2f, data.omega, 0.11, omega_c=omega_c)
    kT = res.tc * K_B_HARTREE
    reached = np.pi * kT * (2 * res.n_matsubara - 1)
    assert reached > omega_c


def test_rescale_mu_star_follows_the_morel_anderson_law():
    from waw.utils.eliashberg import rescale_mu_star

    mu1, w1, w2 = 0.11, 0.5, 1.0
    mu2 = rescale_mu_star(mu1, w1, w2)
    assert 1.0 / mu2 == pytest.approx(1.0 / mu1 + np.log(w1 / w2))
    assert mu2 > mu1                              # raising the cutoff raises mu*
    assert rescale_mu_star(mu2, w2, w1) == pytest.approx(mu1)   # invertible
    assert rescale_mu_star(mu1, w1, w1) == pytest.approx(mu1)   # identity

    matrix = rescale_mu_star(np.full((2, 2), mu1), w1, w2)
    assert matrix.shape == (2, 2) and matrix == pytest.approx(mu2)

    with pytest.raises(ValueError, match="positive"):
        rescale_mu_star(mu1, -1.0, 1.0)
    # the pole is hit RAISING the cutoff far: mu* -> mu as omega_c -> E_F, and
    # the logarithmic form diverges before it gets there
    with pytest.raises(ValueError, match="pole"):
        rescale_mu_star(0.11, 1e-6, 1.0)


def test_rescaling_mu_star_reduces_the_drift_of_tc_with_the_cutoff():
    """The point of the law: Tc at fixed mu* moves a lot with omega_c (MgB2
    30.4 -> 34.1 K from 3x to 10x), and rescaling mu* consistently removes most
    of it. Not all -- the Morel-Anderson form is approximate, more so for a
    multiband system carrying one scalar mu* in every channel."""
    from waw.utils.eliashberg import rescale_mu_star

    for stem in ("cac6", "mgb2"):
        data, _ = _load(stem)
        w0, mu0 = data.omega_c, data.mu_star
        base = tc_linearized(data.a2f, data.omega, mu0, omega_c=w0,
                             n_matsubara=512).tc
        wide = 10.0 * data.omega.max()
        fixed = tc_linearized(data.a2f, data.omega, mu0, omega_c=wide,
                              n_matsubara=512).tc
        moved = tc_linearized(data.a2f, data.omega,
                              rescale_mu_star(mu0, w0, wide), omega_c=wide,
                              n_matsubara=512).tc
        assert abs(moved - base) < abs(fixed - base), (
            f"{stem}: rescaling should shrink the drift, got "
            f"{moved:.3f} vs {fixed:.3f} against {base:.3f} K"
        )


# ------------------------------------------------- full (nonlinear) equations

def _ref_gap_curve(stem):
    """The reference solver's own Delta(T), digitised from its gap0_of_T."""
    return {
        "cac6": np.array([
            [2.77776134, 2.31991457], [6.94440336, 2.21993842],
            [8.74063730, 2.07448837], [10.80093036, 1.76424512],
            [12.08618060, 1.44353397], [12.90922726, 1.14148159],
            [13.30632475, 0.94422551], [13.58840737, 0.76499525],
        ]),
        "mgb2": np.array([
            [4.47779139, 5.25222100, 0.53014434],
            [11.19447849, 5.11172048, 0.43231701],
            [14.64819081, 4.98679640, 0.39844160],
            [22.59771436, 4.22324146, 0.30570532],
            [25.86182848, 3.58618632, 0.25067012],
            [28.10140921, 2.94376485, 0.20104979],
        ]),
    }[stem]


@pytest.mark.parametrize("stem", ["cac6", "mgb2"])
def test_nonlinear_gap_matches_the_reference_solver(stem):
    """Delta(n=0) against the reference's own full-equation output, over the
    range where BOTH are converged. MgB2 exercises the two-gap case: sigma and
    pi come out separately and both must match."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load(stem)
    curve = _ref_gap_curve(stem)
    mev = HARTREE_TO_EV * 1e3
    worst = 0.0
    # skip the lowest temperature: at a fixed n_matsubara = 500 the sum reaches
    # only ~4x the phonon maximum there, so BOTH codes are truncated and by
    # slightly different amounts (see the dedicated truncation test below)
    for row in curve[1:]:
        t, d_ref = row[0], row[1:]
        res = solve_gap(data.a2f, data.omega, data.mu_star, t * K_B_HARTREE,
                        n_matsubara=data.n_matsubara, omega_c=data.omega_c,
                        tol=1e-10)
        assert res.converged
        got = res.delta_0 * mev
        assert len(got) == len(d_ref)
        worst = max(worst, float(np.max(np.abs(got / d_ref - 1.0))))
    assert worst < 2.5e-3, f"{stem}: worst gap deviation {100*worst:.3f}%"


def test_two_gap_structure_of_mgb2_is_resolved():
    """The sigma gap must come out several times the pi gap -- the physics that
    a single isotropic spectrum cannot represent."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("mgb2")
    res = solve_gap(data.a2f, data.omega, data.mu_star, 5.0 * K_B_HARTREE,
                    n_matsubara=data.n_matsubara, omega_c=data.omega_c)
    sigma, pi = res.delta_0
    assert sigma > 5.0 * pi > 0.0
    assert np.all(res.delta_0 > 0.0)                  # s++, both signs equal


def test_gap_closes_where_the_linearized_equations_say_it_should():
    """The two solvers share `gap_kernel`, so the temperature at which the full
    equations lose their nontrivial solution must be the linearized Tc. This is
    the strongest internal check on both."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    tc = tc_linearized(data.a2f, data.omega, data.mu_star,
                       omega_c=data.omega_c, n_matsubara=data.n_matsubara).tc
    kw = dict(n_matsubara=data.n_matsubara, omega_c=data.omega_c, tol=1e-11)
    below = solve_gap(data.a2f, data.omega, data.mu_star,
                      0.98 * tc * K_B_HARTREE, **kw)
    above = solve_gap(data.a2f, data.omega, data.mu_star,
                      1.02 * tc * K_B_HARTREE, **kw)
    assert below.is_superconducting and below.delta_0[0] > 0.0
    assert not above.is_superconducting


def test_gap_follows_the_mean_field_square_root_law_near_tc():
    """Delta = A sqrt(1 - T/Tc) with CONSTANT A. This is what identifies the
    right Tc: with ours the prefactor holds to ~1% over a 9x range of
    (1 - T/Tc), while pairing the reference's near-Tc gaps with its own
    extrapolated Tc spreads A by 13% -- its iteration had not reached the fixed
    point, and its Tc estimate inherited that."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    tc = tc_linearized(data.a2f, data.omega, data.mu_star,
                       omega_c=data.omega_c, n_matsubara=data.n_matsubara).tc
    mev = HARTREE_TO_EV * 1e3
    prefactors = []
    for t in (13.90836, 13.98917, 14.04444, 14.07842):
        d = solve_gap(data.a2f, data.omega, data.mu_star, t * K_B_HARTREE,
                      n_matsubara=data.n_matsubara, omega_c=data.omega_c,
                      tol=1e-11).delta_0[0] * mev
        prefactors.append(d / np.sqrt(1.0 - t / tc))
    spread = (max(prefactors) - min(prefactors)) / np.mean(prefactors)
    assert spread < 0.03, f"prefactor spread {100*spread:.2f}%"


def test_anderson_and_linear_mixing_reach_the_same_fixed_point():
    """Acceleration must not move the answer, only the cost of getting there."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    kw = dict(n_matsubara=200, omega_c=data.omega_c, tol=1e-10)
    fast = solve_gap(data.a2f, data.omega, data.mu_star,
                     8.0 * K_B_HARTREE, acceleration="anderson", **kw)
    slow = solve_gap(data.a2f, data.omega, data.mu_star,
                     8.0 * K_B_HARTREE, acceleration="linear",
                     max_iter=20000, **kw)
    assert fast.converged and slow.converged
    assert fast.delta_0 == pytest.approx(slow.delta_0, rel=1e-5)
    assert fast.n_iterations < slow.n_iterations / 5


def test_mass_renormalization_obeys_its_low_temperature_sum_rule():
    """Z at the lowest Matsubara frequency approaches 1 + sum_j lambda_ij."""
    from waw.utils.eliashberg import solve_gap

    for stem in ("cac6", "mgb2"):
        data, _ = _load(stem)
        res = solve_gap(data.a2f, data.omega, data.mu_star, 2.0 * K_B_HARTREE,
                        n_matsubara=2000, omega_c=data.omega_c, tol=1e-11)
        expected = 1.0 + data.lambda_matrix().sum(axis=1)
        assert res.z[0] == pytest.approx(expected, rel=0.03)
        assert np.all(res.z >= 1.0)
        assert np.all(np.diff(res.z, axis=0) < 0.0)       # decreasing in n


def test_gap_vs_temperature_is_monotonic_and_consistent_with_tc():
    from waw.utils.eliashberg import gap_vs_temperature

    data, _ = _load("mgb2")
    # deliberately NOT pinning n_matsubara: the count has to grow as 1/T, and
    # fixing it truncates the low-T points enough to make the small pi gap
    # come out non-monotonic
    sweep = gap_vs_temperature(data.a2f, data.omega, data.mu_star,
                               omega_c=data.omega_c, n_temperatures=6,
                               t_min_fraction=0.3, n_matsubara_min=256,
                               n_matsubara_max=1024)
    assert sweep.delta_0.shape == (6, 2)
    for band in range(2):
        d = sweep.delta_0[:, band]
        assert np.all(np.diff(d) < 0.0), "gap must fall with temperature"
        assert d[0] > 0.0
    assert sweep.tc_from_gap() <= sweep.tc_linearized


def test_solver_returns_the_normal_state_above_tc():
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    res = solve_gap(data.a2f, data.omega, data.mu_star, 40.0 * K_B_HARTREE,
                    n_matsubara=256, omega_c=data.omega_c)
    assert not res.is_superconducting
    assert res.z[0, 0] > 1.0            # Z is still renormalised in the normal state


def test_s_plus_minus_seed_relaxes_to_the_true_solution():
    """Seeding opposite signs must not force an s+- answer: MgB2's solution is
    s++, so the sign is not re-imposed during iteration and must relax back."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("mgb2")
    res = solve_gap(data.a2f, data.omega, data.mu_star, 5.0 * K_B_HARTREE,
                    n_matsubara=256, omega_c=data.omega_c,
                    signs=np.array([1.0, -1.0]), tol=1e-10)
    assert res.converged
    assert np.all(res.delta_0 > 0.0), "an s+- seed should relax to s++ here"


def test_low_temperature_needs_more_matsubara_points():
    """A fixed Matsubara count reaches lower in frequency as T falls, and the
    truncation inflates the gap. This is why the reference comparison skips its
    lowest temperature: at n = 500 and T = 2.78 K the sum spans only 4x the
    phonon maximum, and the converged answer is 0.9% below what either code
    reports there."""
    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    mev = HARTREE_TO_EV * 1e3
    kT = 2.7778 * K_B_HARTREE
    gaps = [solve_gap(data.a2f, data.omega, data.mu_star, kT, n_matsubara=n,
                      omega_c=data.omega_c, tol=1e-11).delta_0[0] * mev
            for n in (500, 1000, 2000, 4000)]
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3]      # truncation inflates it
    assert gaps[-1] == pytest.approx(gaps[-2], rel=2e-3)   # and it converges


def test_truncating_inside_the_coulomb_cutoff_warns():
    import warnings

    from waw.utils.eliashberg import solve_gap

    data, _ = _load("cac6")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_gap(data.a2f, data.omega, data.mu_star, 2.0 * K_B_HARTREE,
                  n_matsubara=32, omega_c=data.omega_c, max_iter=5)
    assert any("Coulomb" in str(w.message) for w in caught)
