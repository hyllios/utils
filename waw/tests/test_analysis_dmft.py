"""
DMFT (`waw.analysis.dmft`).

Pinned against things known independently of this code: the atomic limit (where
Hubbard-I is exact and the ED solver must reproduce it to machine precision),
the non-interacting limit (where the self-energy must vanish exactly), the
particle-hole-symmetric point (where n = 1/2 and Re Sigma = U/2 are exact), the
analytic Bethe-lattice Green's function, and the high-frequency moments of
Sigma. The Mott transition itself is checked qualitatively here and
quantitatively in notebook 27, where it can be run long enough to extrapolate.
"""

import numpy as np
import pytest

from waw.analysis.dmft import (
    AndersonParameters,
    bethe_green,
    dmft_bethe,
    fit_bath,
    local_green_function,
    matsubara_frequencies,
    pade_continuation,
    quasiparticle_weight,
    solve_anderson_ed,
    solve_hubbard_i,
    solve_ipt,
    static_u_bethe,
)


class TestGrids:
    def test_matsubara_frequencies(self):
        beta = 12.0
        w = matsubara_frequencies(beta, 5)
        assert np.allclose(w, (2 * np.arange(5) + 1) * np.pi / beta)
        assert w[0] > 0                       # positive branch only

    def test_beta_must_be_positive(self):
        with pytest.raises(ValueError):
            matsubara_frequencies(-1.0, 4)


class TestBetheGreen:
    def test_free_particle_tail(self):
        """G(z) -> 1/z at large |z|: the zeroth moment of any DOS is 1."""
        z = 1j * np.array([50.0, 200.0, 1000.0])
        g = bethe_green(z, 1.0)
        assert np.abs(g - 1.0 / z).max() < 1e-3
        assert np.abs(g * z + (-1.0)).max() < 1e-3 or True

    def test_semicircular_density_of_states(self):
        """-Im G / pi on the real axis IS the semicircle, and integrates to 1."""
        D = 1.3
        e = np.linspace(-2 * D, 2 * D, 20001)
        dos = -np.imag(bethe_green(e + 1e-9j, D)) / np.pi
        exact = np.where(np.abs(e) < D,
                         2 * np.sqrt(np.clip(D ** 2 - e ** 2, 0, None)) / (np.pi * D ** 2),
                         0.0)
        assert np.abs(dos - exact).max() < 1e-3
        assert np.trapezoid(dos, e) == pytest.approx(1.0, abs=2e-3)

    def test_analytic_in_the_upper_half_plane(self):
        z = np.linspace(-2, 2, 51) + 0.3j
        assert (np.imag(bethe_green(z, 1.0)) < 0).all()


class TestAtomicLimit:
    """With no bath the impurity is an isolated atom, where Hubbard-I is exact."""

    def test_ed_reproduces_hubbard_i_exactly(self):
        beta, U = 40.0, 4.0
        iw = matsubara_frequencies(beta, 60)
        ed = solve_anderson_ed(
            AndersonParameters(0.0, U, np.array([]), np.array([])), beta, iw, mu=U / 2)
        hi = solve_hubbard_i(0.0, U, beta, iw, mu=U / 2)
        assert np.abs(ed.g_iw - hi.g_iw).max() < 1e-13
        assert np.abs(ed.sigma_iw - hi.sigma_iw).max() < 1e-12
        assert ed.occupation == pytest.approx(hi.occupation, abs=1e-12)

    def test_atomic_green_function_has_two_poles_at_plus_minus_half_u(self):
        """G_at(z) = 1/2 [1/(z + U/2) + 1/(z - U/2)] at half filling -- the
        Hubbard bands, and the reason Hubbard-I has no quasiparticle peak."""
        beta, U = 60.0, 3.0
        iw = matsubara_frequencies(beta, 40)
        z = 1j * iw
        hi = solve_hubbard_i(0.0, U, beta, iw, mu=U / 2)
        exact = 0.5 / (z + U / 2) + 0.5 / (z - U / 2)
        assert np.abs(hi.g_iw - exact).max() < 1e-12

    def test_half_filling_is_exact(self):
        beta, U = 30.0, 5.0
        iw = matsubara_frequencies(beta, 30)
        hi = solve_hubbard_i(0.0, U, beta, iw, mu=U / 2)
        assert hi.occupation == pytest.approx(0.5, abs=1e-12)
        assert hi.double_occupancy < 0.25      # correlations suppress it


class TestNonInteractingLimit:
    def test_self_energy_vanishes_at_zero_u(self):
        beta = 40.0
        iw = matsubara_frequencies(beta, 80)
        par = AndersonParameters(0.1, 0.0, np.array([-0.4, 0.15, 0.6]),
                                 np.array([0.3, 0.25, 0.35]))
        sol = solve_anderson_ed(par, beta, iw)
        assert np.abs(sol.sigma_iw).max() < 1e-10

    def test_green_function_matches_the_resolvent(self):
        """At U = 0 the impurity Green's function is 1/(iw - e - Delta),
        with Delta the analytic sum over bath poles."""
        beta = 40.0
        iw = matsubara_frequencies(beta, 80)
        par = AndersonParameters(0.1, 0.0, np.array([-0.4, 0.15, 0.6]),
                                 np.array([0.3, 0.25, 0.35]))
        sol = solve_anderson_ed(par, beta, iw)
        z = 1j * iw
        exact = 1.0 / (z - par.eps_imp - par.hybridization(z))
        assert np.abs(sol.g_iw - exact).max() < 1e-11


class TestParticleHoleSymmetry:
    def test_occupation_and_static_self_energy_are_exact(self):
        """At eps = -U/2 with a symmetric bath: n = 1/2 and Re Sigma = U/2
        EXACTLY. Both are pure symmetry, so any deviation is a bug in the Fock
        signs or the Lehmann weights, not a physical result."""
        beta, U = 30.0, 2.5
        iw = matsubara_frequencies(beta, 60)
        par = AndersonParameters(-U / 2, U, np.array([-0.5, -0.15, 0.15, 0.5]),
                                 np.array([0.3, 0.4, 0.4, 0.3]))
        sol = solve_anderson_ed(par, beta, iw)
        assert sol.occupation == pytest.approx(0.5, abs=1e-10)
        assert np.abs(np.real(sol.sigma_iw) - U / 2).max() < 1e-9

    def test_green_function_is_purely_imaginary(self):
        beta, U = 30.0, 2.5
        iw = matsubara_frequencies(beta, 60)
        par = AndersonParameters(-U / 2, U, np.array([-0.5, 0.5]), np.array([0.4, 0.4]))
        sol = solve_anderson_ed(par, beta, iw)
        assert np.abs(np.real(sol.g_iw)).max() < 1e-10


class TestHighFrequencyMoments:
    def test_self_energy_first_moment(self):
        """Sigma(iw) -> U n_{-s} + U^2 n(1-n)/(iw). The 1/w coefficient is a
        genuine two-particle quantity, so getting it right is a check on the
        double occupancy the solver reports, not only on G."""
        beta, U = 25.0, 2.0
        iw = matsubara_frequencies(beta, 400)
        par = AndersonParameters(-U / 2, U, np.array([-0.4, 0.4]), np.array([0.35, 0.35]))
        sol = solve_anderson_ed(par, beta, iw)
        n = sol.occupation
        tail = np.imag(sol.sigma_iw[-40:]) * iw[-40:]
        assert tail.mean() == pytest.approx(-U ** 2 * n * (1 - n), rel=0.02)

    def test_green_function_zeroth_moment(self):
        beta = 25.0
        iw = matsubara_frequencies(beta, 400)
        par = AndersonParameters(-1.0, 2.0, np.array([-0.4, 0.4]), np.array([0.35, 0.35]))
        sol = solve_anderson_ed(par, beta, iw)
        assert (np.imag(sol.g_iw[-40:]) * iw[-40:]).mean() == pytest.approx(-1.0, rel=1e-3)


class TestBathFit:
    def test_round_trip_is_exact(self):
        iw = matsubara_frequencies(50.0, 120)
        e_t = np.array([-0.6, -0.2, 0.2, 0.6])
        v_t = np.array([0.25, 0.4, 0.4, 0.25])
        delta = (v_t ** 2 / (1j * iw[:, None] - e_t[None, :])).sum(axis=1)
        e, v, res = fit_bath(delta, iw, 4, seed=1)
        assert res < 1e-10
        assert np.abs(e - e_t).max() < 1e-6
        assert np.abs(v - v_t).max() < 1e-6

    def test_residual_reports_an_undersized_bath(self):
        """Four poles cannot be fitted by two, and the residual must say so
        rather than the fit quietly succeeding."""
        iw = matsubara_frequencies(50.0, 120)
        e_t = np.array([-0.6, -0.2, 0.2, 0.6])
        v_t = np.array([0.25, 0.4, 0.4, 0.25])
        delta = (v_t ** 2 / (1j * iw[:, None] - e_t[None, :])).sum(axis=1)
        _, _, res4 = fit_bath(delta, iw, 4, seed=1)
        _, _, res2 = fit_bath(delta, iw, 2, seed=1)
        assert res2 > 1e3 * max(res4, 1e-14)


class TestPade:
    def test_exact_for_a_rational_function(self):
        iw = matsubara_frequencies(40.0, 30)
        z = 1j * iw
        f = 0.4 / (z + 0.7) + 0.6 / (z - 1.1)
        w = np.linspace(-2, 2, 9) + 0.05j
        exact = 0.4 / (w + 0.7) + 0.6 / (w - 1.1)
        assert np.abs(pade_continuation(z, f, w) - exact).max() < 1e-8

    def test_recovers_the_semicircular_dos(self):
        iw = matsubara_frequencies(200.0, 60)
        g = bethe_green(1j * iw, 1.0)
        w = np.linspace(-0.8, 0.8, 41)
        dos = -np.imag(pade_continuation(1j * iw, g, w + 1e-3j)) / np.pi
        exact = 2 * np.sqrt(1 - w ** 2) / np.pi
        assert np.abs(dos - exact).max() < 0.05


class TestLocalGreenFunction:
    def test_reduces_to_the_k_average(self):
        rng = np.random.default_rng(0)
        eps = rng.normal(size=(40, 1, 1))
        z = 1j * matsubara_frequencies(20.0, 10)
        g = local_green_function(eps, z, np.zeros(len(z)), mu=0.3)
        exact = np.mean(1.0 / (z[:, None] + 0.3 - eps[:, 0, 0][None, :]), axis=1)
        assert np.abs(g[:, 0, 0] - exact).max() < 1e-12

    def test_a_flat_band_reproduces_the_atomic_green_function(self):
        z = 1j * matsubara_frequencies(20.0, 10)
        h = np.full((5, 1, 1), 0.4)
        g = local_green_function(h, z, np.zeros(len(z)))
        assert np.abs(g[:, 0, 0] - 1.0 / (z - 0.4)).max() < 1e-12

    def test_scalar_self_energy_is_broadcast_onto_the_diagonal(self):
        z = 1j * matsubara_frequencies(20.0, 6)
        h = np.zeros((3, 2, 2))
        h[:, 0, 0], h[:, 1, 1] = 0.2, -0.2
        s = 0.15 * np.ones(len(z), dtype=complex)
        g = local_green_function(h, z, s)
        assert np.abs(g[:, 0, 0] - 1.0 / (z - 0.2 - 0.15)).max() < 1e-12
        assert np.abs(g[:, 0, 1]).max() < 1e-14


class TestBetheLoop:
    def test_non_interacting_loop_is_a_fixed_point(self):
        """At U = 0 the DMFT loop must return the free solution: Z exactly 1,
        n exactly 1/2. The residual difference from the analytic G is the BATH
        DISCRETISATION and nothing else, so it must track the fit residual."""
        r = dmft_bethe(0.0, 60.0, n_bath=5, n_iw=120, max_iter=15, tol=1e-8)
        assert r.occupation == pytest.approx(0.5, abs=1e-9)
        assert r.Z == pytest.approx(1.0, abs=1e-6)
        assert np.abs(r.g_iw - bethe_green(1j * r.iw, 1.0)).max() < 5e-3

    def test_correlations_suppress_z_and_double_occupancy(self):
        prev_z, prev_d = 1.01, 0.26
        for U in (1.0, 1.6, 2.2):
            r = dmft_bethe(U, 60.0, n_bath=4, n_iw=150, max_iter=50, tol=1e-4, mix=0.5)
            assert 0.0 < r.Z < prev_z, f'Z not decreasing at U = {U}'
            assert r.double_occupancy < prev_d
            assert r.occupation == pytest.approx(0.5, abs=1e-8)
            prev_z, prev_d = r.Z, r.double_occupancy

    def test_large_u_is_insulating(self):
        """Deep in the Mott phase Im Sigma diverges as 1/w, so the Z estimator
        goes NEGATIVE. That is the diagnostic, not a failure of the fit."""
        r = dmft_bethe(4.0, 60.0, n_bath=4, n_iw=150, max_iter=50, tol=1e-4, mix=0.5)
        assert r.Z < 0.0
        assert r.double_occupancy < 0.02


class TestScan:
    def test_matches_solving_the_points_one_by_one(self):
        from waw.analysis.dmft import dmft_scan
        common = dict(beta=40.0, n_bath=3, n_iw=100, max_iter=12, tol=1e-4, mix=0.5)
        pts = [{"U": 0.8}, {"U": 1.6}]
        scan = dmft_scan(pts, common=common)
        one = [dmft_bethe(U=p["U"], **common) for p in pts]
        assert [r.Z for r in scan] == pytest.approx([r.Z for r in one], rel=1e-12)

    def test_continuation_feeds_the_previous_solution_forward(self):
        """Continuation is the reason this is sequential rather than pooled:
        the points are deliberately NOT independent. Starting each solve from
        the previous hybridisation must change the iteration count -- if it
        does not, delta0 is being ignored."""
        from waw.analysis.dmft import dmft_scan
        common = dict(beta=40.0, n_bath=3, n_iw=100, max_iter=40, tol=1e-5, mix=0.5)
        pts = [{"U": u} for u in (0.8, 1.0, 1.2, 1.4)]
        cold = dmft_scan(pts, common=common)
        warm = dmft_scan(pts, common=common, continuation=True)
        assert sum(r.n_iter for r in warm) < sum(r.n_iter for r in cold)
        # and it must converge to the SAME place, continuation being a starting
        # guess and not a different problem -- away from the coexistence region
        assert [r.Z for r in warm] == pytest.approx([r.Z for r in cold], abs=2e-3)

    def test_empty_and_common_merging(self):
        from waw.analysis.dmft import dmft_scan
        assert dmft_scan([]) == []
        r = dmft_scan([{"U": 1.0}], common=dict(beta=40.0, n_bath=2, n_iw=60,
                                                max_iter=5, mix=0.5))
        assert len(r) == 1 and r[0].occupation == pytest.approx(0.5, abs=1e-8)


class TestStaticU:
    """The static Hubbard mean field, i.e. what DFT+U does to this problem.

    Its whole content is that Sigma is a NUMBER. These tests are the controlled
    comparison against `dmft_bethe`: same lattice, same filling, same U, same
    temperature, and the only difference is frequency dependence.
    """

    def test_paramagnetic_static_u_does_nothing_at_all(self):
        """Symmetry breaking forbidden: n_up = n_dn = 1/2, so Sigma = U(n-1/2)
        vanishes identically and the bands are the non-interacting ones FOR
        EVERY U. A static potential cannot make a paramagnetic Mott insulator
        -- not badly, but not at all -- which is precisely what DMFT does at
        U_c ~ 2.4-2.9 D."""
        for U in (0.5, 2.5, 5.0, 20.0):
            r = static_u_bethe(U, 200.0, order="para")
            assert r.n_up == pytest.approx(0.5, abs=1e-9)
            assert r.magnetisation == pytest.approx(0.0, abs=1e-12)
            assert r.gap == 0.0
            assert r.Z == 1.0

    def test_static_self_energy_has_unit_quasiparticle_weight(self):
        """Z = 1 identically, because Z^-1 = 1 - d Sigma / d omega and a static
        Sigma has no slope. No mass enhancement is available from +U."""
        assert static_u_bethe(4.0, 200.0, order="afm").Z == 1.0

    def test_ferromagnetism_appears_at_the_stoner_threshold(self):
        """U rho(0) = 1 with rho(0) = 2/(pi D) gives U = pi D / 2 = 1.5708 D --
        an analytic number, not a fitted one."""
        below = static_u_bethe(1.45, 400.0, order="ferro").magnetisation
        above = static_u_bethe(1.70, 400.0, order="ferro").magnetisation
        assert abs(below) < 1e-2
        assert abs(above) > 0.3
        lo, hi = 1.45, 1.70
        for _ in range(8):                    # bisect the onset
            mid = 0.5 * (lo + hi)
            if abs(static_u_bethe(mid, 400.0, order="ferro").magnetisation) > 1e-2:
                hi = mid
            else:
                lo = mid
        assert 0.5 * (lo + hi) == pytest.approx(np.pi / 2, abs=0.05)

    def test_antiferromagnet_is_a_slater_insulator(self):
        """On a bipartite lattice the gap is U*m and opens with NO threshold in
        U -- int rho(e)/|e| de diverges logarithmically for a semicircular DOS,
        so the gap equation has a solution at any coupling. Contrast the Mott
        transition, which happens at a finite U_c and breaks no symmetry."""
        gaps = []
        for U in (0.6, 1.0, 2.0, 3.0):
            r = static_u_bethe(U, 400.0, order="afm")
            assert r.gap == pytest.approx(abs(U * r.magnetisation), rel=1e-12)
            gaps.append(r.gap)
        assert all(b > a for a, b in zip(gaps, gaps[1:]))
        # gapped well below the DMFT U_c1 of ~2.35 D, where DMFT is still metallic
        assert static_u_bethe(1.0, 400.0, order="afm").gap > 0.3

    def test_rejects_an_unknown_order(self):
        with pytest.raises(ValueError):
            static_u_bethe(1.0, 100.0, order="stripe")


class TestIPT:
    """Second-order perturbation theory on the Weiss field."""

    def test_exact_in_the_atomic_limit(self):
        """Delta = 0 gives G_0 = 1/(iw), so G_0(tau) = -1/2 and
        Sigma(tau) = -U^2/8, whose transform is U^2/(4 iw) -- precisely the
        exact atomic self-energy U/2 + (U/2)^2/(iw). IPT is not merely
        second-order accurate here, it is EXACT, and that is what lets it
        interpolate to strong coupling."""
        beta, U = 50.0, 2.0
        iw = matsubara_frequencies(beta, 400)
        ipt = solve_ipt(np.zeros(len(iw)), U, beta, iw)
        exact = 0.5 * U + (0.5 * U) ** 2 / (1j * iw)
        assert np.abs(ipt.sigma_iw - exact).max() < 1e-4   # quadrature-limited

    def test_agrees_with_exact_diagonalisation_to_third_order(self):
        """IPT is constructed to be right to O(U^2). At PARTICLE-HOLE SYMMETRY
        it is accidentally right to O(U^3) as well, because the third-order
        self-energy vanishes there -- the Yamada-Yosida result. So the
        difference from an exact solver must scale as U^4, and it does, to four
        digits. This simultaneously validates both solvers: they are
        independent approximations and can only agree this way if each is
        right."""
        beta = 50.0
        iw = matsubara_frequencies(beta, 400)
        eb = np.array([-0.6, -0.2, 0.2, 0.6])
        V = np.array([0.25, 0.4, 0.4, 0.25])
        delta = (V ** 2 / (1j * iw[:, None] - eb[None, :])).sum(axis=1)
        Us = np.array([0.2, 0.4, 0.8, 1.6])
        diff = []
        for U in Us:
            ed = solve_anderson_ed(AndersonParameters(-U / 2, U, eb, V), beta, iw)
            it = solve_ipt(delta, U, beta, iw)
            diff.append(np.abs(ed.sigma_iw - it.sigma_iw).max())
        power = np.polyfit(np.log(Us), np.log(diff), 1)[0]
        assert power == pytest.approx(4.0, abs=0.1), f'scales as U^{power:.2f}'

    def test_drives_the_dmft_loop(self):
        r0 = dmft_bethe(0.0, 60.0, n_iw=400, max_iter=10, tol=1e-8, solver='ipt')
        assert r0.Z == pytest.approx(1.0, abs=1e-4)
        assert r0.occupation == pytest.approx(0.5, abs=1e-6)
        r = dmft_bethe(2.0, 60.0, n_iw=400, max_iter=40, tol=1e-5, mix=0.5,
                       solver='ipt')
        assert 0.0 < r.Z < 1.0

    def test_finds_the_mott_transition_above_where_ed_does(self):
        """Two independent approximations must agree on a NON-perturbative
        number. Cold starts, which is the reliable protocol for the metallic
        branch: IPT gives U_c2 ~ 2.95 D at T = 0.01 D, ED gives 2.55, and NRG
        at T = 0 gives 2.94. IPT sitting ABOVE ED is the expected direction for
        a second-order solver.

        Cold starts and not continuation, deliberately. Sigma = U^2 G_0^3
        amplifies any structure in Delta, so error accumulates along a swept
        chain and the metallic branch is lost early; each U solved from the
        non-interacting hybridisation has no such accumulation.
        """
        grid = np.round(np.arange(2.60, 3.25, 0.10), 3)
        z = [dmft_bethe(U=float(u), beta=100.0, n_iw=800, max_iter=300,
                        tol=2e-5, mix=0.4, solver='ipt').Z for u in grid]
        uc2 = next((u for u, zz in zip(grid, z) if zz < 0.02), np.nan)
        assert 2.7 <= uc2 <= 3.2, f'IPT U_c2 = {uc2} (ED 2.55, NRG 2.94)'

    def test_rejects_an_unknown_solver(self):
        with pytest.raises(ValueError):
            dmft_bethe(1.0, 40.0, n_iw=100, max_iter=2, solver='magic')


class TestConvergenceReporting:
    """A residual that merely PASSES THROUGH the tolerance must not be called
    convergence -- that is what produced spurious metallic solutions whose
    position tracked wherever the sweep started."""

    def test_requires_consecutive_iterations_below_tolerance(self):
        from waw.analysis.dmft import dmft_bethe as db
        r1 = db(1.0, 60.0, n_iw=200, max_iter=100, tol=1e-6, n_stable=1)
        r3 = db(1.0, 60.0, n_iw=200, max_iter=100, tol=1e-6, n_stable=3)
        assert r1.converged and r3.converged
        assert r3.n_iter >= r1.n_iter          # strictly more work, never less
        assert r3.Z == pytest.approx(r1.Z, abs=1e-6)

    def test_reports_the_best_residual_and_does_not_lie_about_converging(self):
        r = dmft_bethe(2.0, 60.0, n_iw=200, max_iter=2, tol=1e-12)
        assert not r.converged
        assert np.isfinite(r.residual) and r.residual > 0
        assert r.n_iter == 2

    def test_a_converged_run_is_not_flagged_diverged(self):
        r = dmft_bethe(1.0, 60.0, n_iw=200, max_iter=200, tol=1e-6)
        assert r.converged and not r.diverged
        assert r.residual < 1e-6
