"""
Band-resolved LOVA Boltzmann conductivity (`waw.analysis.elph_boltzmann`).

The solver is pinned against the analytic limits of Allen's transport theory
rather than against itself: the high-temperature slope (which fixes the kernel
normalisation), the Bloch-Gruneisen T^5 law (which is the whole reason to use a
velocity-weighted alpha2F_tr rather than a plain one), Drude for a single sheet,
additivity of decoupled sheets, and the current-conserving singular case. The
synthetic spectral functions here need no DFT input, so these run in
milliseconds and cannot drift with the el-ph machinery.
"""

import numpy as np
import pytest

from waw.analysis.elph import TransportSpectralMatrix
from waw.analysis.elph_boltzmann import (
    lambda_tr,
    lova_conductivity,
    lump_sheets,
    transport_kernel,
)
from waw.units import K_B_HARTREE

WPH = 0.002        # Hartree, ~54 meV Einstein mode
VOL = 100.0        # Bohr^3


def _einstein(lam_out, lam_in, drude=1.0, n_sheets=1, width=2.0e-5, wmax=0.004):
    """
    A sheet-resolved spectral matrix for an Einstein mode, built so that the
    single-sheet transport coupling is exactly ``2*(lam_out - lam_in)*...``:
    each entry is a narrow Gaussian of weight ``lam * drude * WPH / 2`` at WPH,
    which integrates to give alpha2F_tr = (F_out - F_in)/D with
    lambda_tr = lam_out - lam_in.
    """
    w = np.linspace(0.0, wmax, 4001)
    g = np.exp(-0.5 * ((w - WPH) / width) ** 2) / (width * np.sqrt(2 * np.pi))
    lam_out = np.atleast_2d(np.asarray(lam_out, dtype=np.float64))
    lam_in = np.atleast_2d(np.asarray(lam_in, dtype=np.float64))
    d = np.broadcast_to(np.atleast_1d(drude), (n_sheets,)).astype(np.float64)
    # weight of each (i,j) entry: lambda_ij * D_i * WPH / 2
    f_out = (lam_out * d[:, None])[None, :, :, None] * (WPH / 2) * g
    f_in = (lam_in * d[:, None])[None, :, :, None] * (WPH / 2) * g
    f_out = np.repeat(f_out, 3, axis=0)
    f_in = np.repeat(f_in, 3, axis=0)
    return TransportSpectralMatrix(
        omega=w, f_out=f_out, f_in=f_in,
        drude=np.tile(d, (3, 1)), dos=np.ones(n_sheets))


class TestKernel:
    def test_high_temperature_limit_is_four_pi_kt_over_omega(self):
        """[y/sinh y]^2 -> 1, so K -> 4 pi kT/w. This normalisation is what
        makes 1/tau_tr -> 2 pi lambda_tr kT, the standard high-T rate."""
        w = np.linspace(1e-6, 1e-4, 50)
        kT = 1.0                      # kT >> w
        assert np.allclose(transport_kernel(w, kT), 4 * np.pi * kT / w, rtol=1e-8)

    def test_kernel_is_exponentially_small_for_omega_far_above_kt(self):
        k = transport_kernel(np.array([1.0]), 1e-3)      # y = 500
        assert 0.0 <= k[0] < 1e-300

    def test_zero_and_negative_frequencies_are_zero_not_nan(self):
        k = transport_kernel(np.array([-1e-3, 0.0, 1e-3]), 1e-3)
        assert k[0] == 0.0 and k[1] == 0.0 and np.isfinite(k[2])


class TestSingleSheetIsDrude:
    def test_high_temperature_rate_is_two_pi_lambda_tr_kt(self):
        """
        THE normalisation test. For one sheet, sigma = e^2 D tau_tr / Omega, and
        at kT >> w_ph the transport rate must be 1/tau_tr = 2 pi lambda_tr kT
        (hbar = 1). Checked through the solver, so it validates the kernel, the
        F_out - F_in bookkeeping and the Drude assembly together.
        """
        s = _einstein(lam_out=1.3, lam_in=0.0, drude=2.5)
        # 1e-3, not machine precision: lambda_tr = 2 int a2F_tr/w picks up the
        # second moment of the Gaussian standing in for the Einstein delta,
        # (width/WPH)^2 = 1e-4 here. A fixture artefact, not the solver's.
        assert lambda_tr(s) == pytest.approx(1.3, rel=1e-3)
        T = 3000.0                                     # kT = 9.5 mHa >> 2 mHa
        r = lova_conductivity(s, [T], VOL)
        tau = r.tau_transport()[0, 0]
        expected = 1.0 / (2 * np.pi * 1.3 * T * K_B_HARTREE)
        assert tau == pytest.approx(expected, rel=0.02)
        # and sigma is Drude with that lifetime
        # note the spin factor: sigma = num_elec_per_state * D * tau / Omega
        assert r.sigma[0, 0] == pytest.approx(2.0 * 2.5 * expected / VOL, rel=0.02)

    def test_resistivity_is_linear_in_t_at_high_temperature(self):
        s = _einstein(lam_out=1.0, lam_in=0.0)
        T = np.array([2000.0, 3000.0, 4000.0, 5000.0])
        rho = 1.0 / lova_conductivity(s, T, VOL).sigma[:, 0]
        slope = np.diff(rho) / np.diff(T)
        assert np.allclose(slope, slope[0], rtol=0.02)      # constant slope
        assert rho[0] / T[0] == pytest.approx(slope[0], rel=0.03)  # through 0

    def test_only_the_difference_of_out_and_in_matters_for_one_sheet(self):
        """alpha2F_tr = (F_out - F_in)/D: for a single sheet, adding the same
        amount to both weights must leave sigma untouched -- the cancellation
        that a per-pair alpha2F_tr encodes and that band space breaks."""
        a = lova_conductivity(_einstein(1.0, 0.0), [500.0], VOL).sigma[0, 0]
        b = lova_conductivity(_einstein(3.0, 2.0), [500.0], VOL).sigma[0, 0]
        assert a == pytest.approx(b, rel=1e-10)


class TestBlochGruneisen:
    def test_debye_spectrum_gives_t_to_the_fifth(self):
        """
        With alpha2F_tr ~ w^4 (acoustic modes, Debye, once the transport weight
        has added its q^2), rho ~ T^5 well below the Debye temperature. This is
        the exponent a self-energy relaxation time gets wrong (it gives T^3),
        so it is the sharpest test that the vertex weight is being used.
        """
        wd = 0.001                                   # Debye energy, Hartree
        w = np.linspace(0.0, wd, 3000)
        shape = (w / wd) ** 4                        # alpha2F_tr ~ w^4
        f_out = np.repeat((shape * (WPH / 2))[None, None, None, :], 3, axis=0)
        s = TransportSpectralMatrix(
            omega=w, f_out=f_out, f_in=np.zeros_like(f_out),
            drude=np.ones((3, 1)), dos=np.ones(1))
        T = np.array([2.0, 3.0, 4.0, 6.0])           # kT << wd (wd ~ 316 K)
        rho = 1.0 / lova_conductivity(s, T, VOL).sigma[:, 0]
        p = np.polyfit(np.log(T), np.log(rho), 1)[0]
        assert p == pytest.approx(5.0, abs=0.1)


class TestBandResolution:
    def test_decoupled_sheets_conduct_in_parallel(self):
        """Two sheets that never scatter into each other must give exactly
        sigma_1 + sigma_2 -- the collision matrix is block diagonal."""
        one = _einstein([[1.0]], [[0.0]], drude=2.0)
        two = _einstein([[1.5]], [[0.0]], drude=3.0)
        both = _einstein([[1.0, 0.0], [0.0, 1.5]], [[0.0, 0.0], [0.0, 0.0]],
                         drude=[2.0, 3.0], n_sheets=2)
        T = [400.0]
        s1 = lova_conductivity(one, T, VOL).sigma[0, 0]
        s2 = lova_conductivity(two, T, VOL).sigma[0, 0]
        s12 = lova_conductivity(both, T, VOL).sigma[0, 0]
        assert s12 == pytest.approx(s1 + s2, rel=1e-10)

    def test_band_resolution_can_only_raise_sigma_above_the_isotropic_answer(self):
        """
        Variational: solving for one Phi per sheet searches a larger space than
        a single Phi for the whole Fermi surface, so it can only lower rho. The
        gap is largest when the sheets have very different couplings -- the
        MgB2 sigma/pi situation, and the reason to do this at all.
        """
        for lam_a, lam_b in [(0.3, 3.0), (0.5, 1.0), (2.0, 2.0)]:
            s = _einstein([[lam_a, 0.2], [0.2, lam_b]],
                          [[0.0, 0.05], [0.05, 0.0]],
                          drude=[1.0, 1.0], n_sheets=2)
            band = lova_conductivity(s, [300.0], VOL).sigma[0, 0]
            iso = lova_conductivity(lump_sheets(s), [300.0], VOL).sigma[0, 0]
            assert band >= iso * (1 - 1e-12)
        # and for strongly contrasted sheets the difference is real, not 0
        s = _einstein([[0.3, 0.2], [0.2, 3.0]], [[0.0, 0.05], [0.05, 0.0]],
                      drude=[1.0, 1.0], n_sheets=2)
        band = lova_conductivity(s, [300.0], VOL).sigma[0, 0]
        iso = lova_conductivity(lump_sheets(s), [300.0], VOL).sigma[0, 0]
        assert band > 1.05 * iso

    def test_interband_in_scattering_reduces_the_resistivity(self):
        """
        The physical content of keeping F_in off-diagonal: interband scattering
        that preserves the current direction (v_i . v_j > 0) relaxes the current
        less than its out-scattering rate implies. Same out-scattering, more
        in-scattering -> higher sigma.
        """
        out = [[1.0, 0.5], [0.5, 1.0]]
        weak = _einstein(out, [[0.0, 0.0], [0.0, 0.0]], drude=[1.0, 1.0], n_sheets=2)
        strong = _einstein(out, [[0.0, 0.4], [0.4, 0.0]], drude=[1.0, 1.0], n_sheets=2)
        T = [300.0]
        assert (lova_conductivity(strong, T, VOL).sigma[0, 0]
                > lova_conductivity(weak, T, VOL).sigma[0, 0])


class TestCurrentConservingCases:
    def test_pure_forward_scattering_has_no_momentum_sink(self):
        """F_in = F_out means every event preserves the current: M = 0 and
        sigma diverges. The solver must say so rather than return a number."""
        s = _einstein([[1.0]], [[1.0]], drude=1.0)
        with pytest.raises(RuntimeError, match="no channel relaxes the current"):
            lova_conductivity(s, [300.0], VOL)

    def test_impurities_regularise_it_and_give_the_clean_limit(self):
        """With a momentum sink restored, sigma = e^2 D tau_imp / Omega exactly:
        the el-ph term drops out of a current-conserving collision matrix."""
        s = _einstein([[1.0]], [[1.0]], drude=2.0)
        rate = 1e-5
        r = lova_conductivity(s, [300.0], VOL, impurity_rate=rate)
        assert r.sigma[0, 0] == pytest.approx(2.0 * 2.0 / rate / VOL, rel=1e-10)

    def test_two_sheets_trading_equal_carriers_also_conserve_current(self):
        """Interband-only scattering between sheets of equal velocity relaxes
        nothing: out and in cancel sheet by sheet."""
        s = _einstein([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]],
                      drude=[1.0, 1.0], n_sheets=2)
        with pytest.raises(RuntimeError, match="no channel relaxes the current"):
            lova_conductivity(s, [300.0], VOL)


class TestApiAndUnits:
    def test_si_conversion_matches_the_boltzmann_module_convention(self):
        """sigma is in the same atomic convention as `analysis.boltzmann`, so
        the registered conversion must apply unchanged."""
        from waw.units import to_si_units
        s = _einstein(1.0, 0.0)
        r = lova_conductivity(s, [300.0], VOL)
        assert np.allclose(r.sigma_si, to_si_units(r.sigma, "electrical_conductivity"))
        # a metal-like resistivity comes out in a plausible range, not 1e30
        assert 1e-3 < r.resistivity_microohm_cm[0, 0] < 1e6

    def test_rejects_zero_or_negative_temperature(self):
        with pytest.raises(ValueError, match="must be > 0"):
            lova_conductivity(_einstein(1.0, 0.0), [0.0], VOL)

    def test_rejects_mismatched_shapes(self):
        s = _einstein(1.0, 0.0)
        bad = s._replace(drude=np.ones((3, 2)))
        with pytest.raises(ValueError, match="shape mismatch"):
            lova_conductivity(bad, [300.0], VOL)

    def test_cubic_symmetry_gives_three_identical_components(self):
        r = lova_conductivity(_einstein(1.0, 0.0), [300.0], VOL)
        assert np.allclose(r.sigma[0, 0], r.sigma[0, 1:])


class TestSpinDegeneracy:
    """
    The spin factor does NOT cancel here, unlike in lambda. El-ph scattering is
    spin-diagonal, so the collision matrix and tau_tr are per-spin and carry no
    factor, while the current sums over both spin channels. Omitting it halves
    sigma -- caught on Al via the plasma frequency.
    """

    def test_sigma_is_linear_in_the_occupancy_factor(self):
        s = _einstein(1.0, 0.0, drude=2.0)
        one = lova_conductivity(s, [300.0], VOL, num_elec_per_state=1.0)
        two = lova_conductivity(s, [300.0], VOL, num_elec_per_state=2.0)
        assert two.sigma[0, 0] == pytest.approx(2.0 * one.sigma[0, 0], rel=1e-12)
        # tau_tr must be UNCHANGED: it is a per-spin quantity
        assert (one.tau_transport()[0, 0]
                == pytest.approx(two.tau_transport()[0, 0], rel=1e-12))

    def test_plasma_frequency_matches_the_free_electron_value(self):
        """
        Omega_p^2 = 4 pi e^2 n_eff D / Omega. For a free-electron gas the
        identity N(eF) <v_a^2> = n/2 per spin makes this sqrt(4 pi n) exactly,
        which is the check that caught the missing factor of 2 on Al.
        """
        n_elec = 3.0                        # electrons per cell
        n = n_elec / VOL                    # density, 1/Bohr^3
        # D is a PER-CELL sum, so the free-electron identity N(eF)<v_a^2> = n
        # (per volume, both spins) reads D = n_elec/2 per spin channel.
        s = _einstein(1.0, 0.0, drude=n_elec / 2.0)
        r = lova_conductivity(s, [300.0], VOL)
        assert r.plasma_frequency()[0] == pytest.approx(np.sqrt(4 * np.pi * n), rel=1e-12)


class TestDrudeWeightAgainstTheOtherTransportModule:
    def test_matches_the_constant_relaxation_time_tdf_at_the_fermi_level(self):
        """
        Cross-check between the two transport routes, which share no code:
        `band_sheets.sheet_drude_weight` summed over sheets must equal the
        diagonal of `analysis.boltzmann`'s transport distribution function at
        eF (which is (1/Omega) sum_k v_a v_b tau delta(E - eps), i.e. D_a/Omega
        at tau = 1), including the spin factor and the volume. Catches a unit
        or convention slip in either module.
        """
        from waw.analysis.band_sheets import band_character_weights, sheet_drude_weight
        from waw.analysis.boltzmann import band_velocities, transport_distribution_function
        from waw.analysis.elph import band_eigensystem
        import torch

        from waw.core.hamiltonian import HamiltonianR
        from waw.interfaces.ase.structure import monkhorst_pack

        # simple-cubic single-band tight binding: E(k) = -2t sum cos(k a)
        a = 5.0
        real_lat = a * np.eye(3)
        recip = 2 * np.pi * np.linalg.inv(real_lat).T
        R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                      [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)
        t = 0.05
        h = np.zeros((len(R), 1, 1), dtype=np.complex128)
        h[1:, 0, 0] = -t
        hr = HamiltonianR(H_R=torch.tensor(h), R_vectors=R,
                          degen=np.ones(len(R), dtype=np.int64), nw=1)

        mesh = (24, 24, 24)
        kpts = monkhorst_pack(mesh)
        eig, U = band_eigensystem(hr, kpts)
        _, v = band_velocities(hr, kpts, recip)
        eF = -1.0 * t                     # inside the band
        sigma_e = 0.02
        w = band_character_weights(U, [[0]])
        d = sheet_drude_weight(eig, v, eF, w, sigma_e)          # (3, 1)

        vol = float(np.abs(np.linalg.det(real_lat)))
        # needs >= 2 energies for its bin width; take the one nearest eF
        egrid = eF + np.linspace(-0.01, 0.01, 41)
        tdf = transport_distribution_function(
            hr, real_lat, recip, mesh, egrid, relax_time=1.0,
            num_elec_per_state=2.0, smearing=sigma_e)
        j = int(np.argmin(np.abs(egrid - eF)))
        assert tdf.tdf[j, 0, 0] == pytest.approx(2.0 * d[0, 0] / vol, rel=0.02)
        assert tdf.tdf[j, 1, 1] == pytest.approx(2.0 * d[1, 0] / vol, rel=0.02)


class TestSpinResolvedTransport:
    """
    The two-current model for a collinear magnet (Fert & Campbell, J. Phys. F 6,
    849 (1976)): without spin-orbit coupling the electron-phonon interaction is
    spin-diagonal, so the channels never scatter into each other and conduct in
    parallel. Stacking them as sheets with zero inter-channel blocks makes the
    collision matrix block diagonal, and the ordinary solve gives sigma_up +
    sigma_down with no special-casing.

    The occupancy is where this goes wrong in practice: each channel's bands hold
    ONE electron, so a spin-resolved solve needs num_elec_per_state = 1, not the
    2 that is right for a non-magnetic band. Since the spin factor does not
    cancel in transport, a machinery written for the non-magnetic case that keeps
    a hard-wired 2 silently doubles a magnet's conductivity -- the shape of bug
    Miguel warned about in EPW.
    """

    def test_spin_degenerate_ferromagnet_reproduces_the_nonmagnetic_answer(self):
        """
        THE bookkeeping test. Take a "magnet" whose two channels are identical
        copies of one non-magnetic sheet. Solved spin-resolved at occupancy 1 it
        must equal the single sheet solved at occupancy 2 -- exactly. Any stray
        factor of 2 in the spin path shows up here and nowhere else.
        """
        from waw.analysis.elph_boltzmann import spin_resolved_conductivity

        one = _einstein(1.2, 0.1, drude=2.0)
        T = [77.0, 300.0]
        nonmag = lova_conductivity(one, T, VOL, num_elec_per_state=2.0)
        total, up, dn = spin_resolved_conductivity(one, one, T, VOL)
        assert np.allclose(total.sigma, nonmag.sigma, rtol=1e-12)
        # and the two channels each carry exactly half
        assert np.allclose(up.sigma, dn.sigma, rtol=1e-12)
        assert np.allclose(up.sigma + dn.sigma, total.sigma, rtol=1e-12)

    def test_channels_conduct_in_parallel_and_never_mix(self):
        """Different couplings and Drude weights per channel: the total must be
        the sum of the independently solved channels, and the collision matrix
        must have zero inter-channel blocks."""
        from waw.analysis.elph_boltzmann import (spin_resolved_conductivity,
                                                 stack_spin_channels)

        maj = _einstein(0.4, 0.0, drude=3.0)      # weakly coupled, heavy Drude
        mino = _einstein(2.5, 0.0, drude=0.7)     # strongly coupled, light
        T = [300.0]
        total, up, dn = spin_resolved_conductivity(maj, mino, T, VOL)
        assert total.sigma[0, 0] == pytest.approx(up.sigma[0, 0] + dn.sigma[0, 0],
                                                 rel=1e-12)
        M = total.collision[0, 0]
        assert M.shape == (2, 2)
        assert M[0, 1] == 0.0 and M[1, 0] == 0.0          # spin is conserved
        # the weakly coupled channel carries most of the current: a "short
        # circuit" by the majority channel, which is the two-current model's
        # signature in a real ferromagnet
        assert up.sigma[0, 0] > 5.0 * dn.sigma[0, 0]

    def test_spin_polarisation_of_the_current(self):
        from waw.analysis.elph_boltzmann import spin_resolved_conductivity

        maj = _einstein(0.5, 0.0, drude=2.0)
        mino = _einstein(2.0, 0.0, drude=1.0)
        total, up, dn = spin_resolved_conductivity(maj, mino, [300.0], VOL)
        pol = (up.sigma[0, 0] - dn.sigma[0, 0]) / total.sigma[0, 0]
        assert 0.0 < pol < 1.0
        # majority-dominated: 4x the coupling ratio and 2x the Drude weight
        assert pol > 0.7

    def test_the_wrapper_refuses_to_let_the_spin_factor_be_overridden(self):
        from waw.analysis.elph_boltzmann import spin_resolved_conductivity

        one = _einstein(1.0, 0.0)
        with pytest.raises(ValueError, match="num_elec_per_state"):
            spin_resolved_conductivity(one, one, [300.0], VOL,
                                       num_elec_per_state=2.0)

    def test_stacking_requires_a_common_frequency_grid(self):
        from waw.analysis.elph_boltzmann import stack_spin_channels

        a = _einstein(1.0, 0.0)
        b = _einstein(1.0, 0.0, wmax=0.005)          # different grid
        with pytest.raises(ValueError, match="share one omega grid"):
            stack_spin_channels(a, b)
        with pytest.raises(ValueError, match="at least two"):
            stack_spin_channels(a)

    def test_sheet_resolution_survives_inside_each_spin_channel(self):
        """A magnet whose channels are themselves two-sheeted: stacking must
        give 4 sheets, block diagonal 2+2, and still equal the sum."""
        from waw.analysis.elph_boltzmann import (spin_resolved_conductivity,
                                                 stack_spin_channels)

        maj = _einstein([[0.5, 0.1], [0.1, 1.0]], [[0.0, 0.02], [0.02, 0.0]],
                        drude=[1.0, 2.0], n_sheets=2)
        mino = _einstein([[1.5, 0.2], [0.2, 2.0]], [[0.0, 0.05], [0.05, 0.0]],
                         drude=[0.5, 0.8], n_sheets=2)
        stacked = stack_spin_channels(maj, mino)
        assert stacked.f_out.shape[1] == 4
        assert np.all(stacked.f_out[:, :2, 2:, :] == 0.0)
        assert np.all(stacked.f_in[:, 2:, :2, :] == 0.0)
        total, up, dn = spin_resolved_conductivity(maj, mino, [300.0], VOL)
        assert total.sigma[0, 0] == pytest.approx(up.sigma[0, 0] + dn.sigma[0, 0],
                                                 rel=1e-12)
