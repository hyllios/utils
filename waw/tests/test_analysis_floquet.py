"""
Floquet engineering of a Wannier Hamiltonian (`waw.analysis.floquet`).

The construction is pinned against things known independently of it: the
undriven limit, hermiticity, the analytic high-frequency (Magnus) result for a
Dirac cone under circular light, dynamical band narrowing through the first zero
of J_0, and photon-truncation convergence. A two-band graphene-like model is
used so the analytic comparisons are exact rather than approximate.
"""

import numpy as np
import pytest
import torch

from waw.analysis.floquet import (
    drive_amplitude_from_field,
    floquet_blocks,
    floquet_convergence,
    floquet_hamiltonian,
    floquet_quasi_energies,
    floquet_spectral_function,
)
from waw.core.hamiltonian import HamiltonianR
from waw.units import HARTREE_TO_EV

A_GR = 4.6511          # graphene lattice constant in Bohr (2.461 Ang)
T_HOP = 0.1            # Hartree, nearest-neighbour hopping (~2.7 eV)


def _graphene():
    """Nearest-neighbour honeycomb model with REAL graphene geometry.

    a1 = a(1,0), a2 = a(1/2, sqrt3/2); A at the origin, B at (a1+a2)/3 so all
    three bonds have the true length a/sqrt3. The centres matter: they set the
    Peierls bond vectors, and with them the intra-cell bond is driven like the
    other two (see `_bond_vectors`).

    H_AB(k) = -t (1 + e^{-2pi i k1} + e^{-2pi i k2}), which vanishes at
    k = (1/3, -1/3) -- that is the Dirac point in this convention.
    """
    a1 = A_GR * np.array([1.0, 0.0, 0.0])
    a2 = A_GR * np.array([0.5, np.sqrt(3) / 2, 0.0])
    real_lattice = np.array([a1, a2, [0.0, 0.0, 20.0]])
    tau_B = (a1 + a2) / 3.0
    Rs = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)]
    H = np.zeros((len(Rs), 2, 2), dtype=np.complex128)
    for i in (0, 1, 2):
        H[i, 0, 1] = -T_HOP                    # A(0) <- B(R)
    for i in (0, 3, 4):
        H[i, 1, 0] = -T_HOP                    # the hermitian partners
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=np.array(Rs, dtype=np.int64),
                        degen=np.ones(len(Rs), dtype=np.int64), nw=2,
                        centres=np.array([[0.0, 0.0, 0.0], tau_B]),
                        real_lattice=real_lattice, mp_grid=(12, 12, 1))


BOND = A_GR / np.sqrt(3.0)          # true NN bond length, Bohr


K_POINT = np.array([[1 / 3, -1 / 3, 0.0]])      # Dirac point of this convention


class TestUndrivenLimit:
    def test_zero_amplitude_gives_exact_photon_replicas(self):
        """At A0 = 0 the sectors decouple and the quasi-energies must be the
        undriven bands shifted by m*omega, exactly."""
        hr = _graphene()
        k = np.array([[0.2, 0.1, 0.0], [1 / 3, -1 / 3, 0.0]])
        omega = 0.05
        eps, _ = floquet_quasi_energies(hr, k, amplitude=0.0, omega=omega,
                                        n_photon=2)
        from waw.analysis.elph import band_eigensystem
        e0, _ = band_eigensystem(hr, k)
        expect = np.sort(np.concatenate(
            [e0 + m * omega for m in range(-2, 3)], axis=1), axis=1)
        assert np.allclose(eps, expect, atol=1e-12)

    def test_only_the_p_zero_block_survives_at_zero_amplitude(self):
        hr = _graphene()
        # a generic k, not K: nearest-neighbour graphene has H(K) = 0 exactly,
        # so the p = 0 block there is null and would pass the test vacuously
        b = floquet_blocks(hr, np.array([[0.2, 0.1, 0.0]]), amplitude=0.0,
                           n_photon=2)
        for idx, p in enumerate(range(-4, 5)):
            if p == 0:
                assert np.abs(b[idx]).max() > 1e-3
            else:
                assert np.abs(b[idx]).max() < 1e-14, f"p={p} should vanish"

    def test_hamiltonian_is_hermitian(self):
        hr = _graphene()
        HF = floquet_hamiltonian(hr, np.array([[0.21, 0.13, 0.0]]),
                                 amplitude=0.4, omega=0.05, n_photon=3)
        assert np.abs(HF - np.conj(np.swapaxes(HF, -1, -2))).max() < 1e-13


class TestLightInducedGap:
    """
    The headline: circular light opens a gap at the Dirac point.

    In the off-resonant limit the Magnus expansion gives an effective Haldane
    mass from the commutator of the +-1 harmonics,

        H_eff = H_0 + [H_{+1}, H_{-1}]/omega,   [H_{+1},H_{-1}] = -(v A0)^2 sigma_z

    so the gap is 2 (v_F A0)^2 / omega. That is a genuinely independent
    prediction -- it involves only the Fermi velocity, the drive and the
    frequency -- and it is the sharpest available check that the Bessel-weighted
    blocks carry the right phases, since a sign error in the helicity or in
    i^p J_p e^{ip phi} destroys the commutator and with it the gap.
    """

    @staticmethod
    def _fermi_velocity(hr):
        """dE/dk at the Dirac point, from a small finite difference."""
        from waw.analysis.elph import band_eigensystem
        rec = 2 * np.pi * np.linalg.inv(hr.real_lattice).T
        dk = 1e-4
        kk = np.array([[1 / 3 + dk, -1 / 3, 0.0], [1 / 3, -1 / 3, 0.0]])
        e, _ = band_eigensystem(hr, kk)
        dE = e[0, 1] - e[1, 1]
        dkc = np.linalg.norm(dk * rec[0])
        return abs(dE / dkc)

    @pytest.mark.parametrize("amplitude", [0.10, 0.15])
    def test_gap_matches_the_magnus_prediction(self, amplitude):
        hr = _graphene()
        v = self._fermi_velocity(hr)
        omega = 0.6                      # >> bandwidth 6t = 0.6 Ha -> off-resonant
        eps, _ = floquet_quasi_energies(hr, K_POINT, amplitude=amplitude,
                                        omega=omega, n_photon=4)
        mid = eps.shape[1] // 2
        gap = float(eps[0, mid] - eps[0, mid - 1])
        predicted = 2.0 * (v * amplitude) ** 2 / omega
        assert gap == pytest.approx(predicted, rel=0.25), (
            f"gap {gap*HARTREE_TO_EV*1e3:.2f} meV vs Magnus "
            f"{predicted*HARTREE_TO_EV*1e3:.2f} meV")

    def test_gap_grows_as_the_square_of_the_drive(self):
        hr = _graphene()
        omega = 0.6
        amps = np.array([0.06, 0.09, 0.12])
        gaps = []
        for a in amps:
            eps, _ = floquet_quasi_energies(hr, K_POINT, amplitude=a,
                                            omega=omega, n_photon=4)
            mid = eps.shape[1] // 2
            gaps.append(eps[0, mid] - eps[0, mid - 1])
        p = np.polyfit(np.log(amps), np.log(gaps), 1)[0]
        assert p == pytest.approx(2.0, abs=0.15)

    def test_linear_polarisation_opens_no_gap(self):
        """Only circular light breaks time reversal; under linear light the
        spectrum must stay GAPLESS. Catches a helicity/phase bug that would
        manufacture a mass from nothing.

        The Dirac point is not pinned, though: linear light along x renormalises
        the three bonds unequally (each by J_0(A0 d_x) with its own d_x), which
        MOVES the degeneracy in k rather than lifting it. So the test is a
        refinement one -- on a patch around K the smallest splitting must fall
        in proportion to the grid spacing, the signature of a cone being sampled
        ever closer to its apex, while a real gap would sit still. Just reading
        the splitting at K would report a spurious "gap" of tens of meV.
        """
        hr = _graphene()

        def min_split(pol, half, n):
            g = np.linspace(-half, half, n)
            kx, ky = np.meshgrid(g, g, indexing="ij")
            patch = np.stack([1 / 3 + kx.ravel(), -1 / 3 + ky.ravel(),
                              np.zeros(kx.size)], axis=1)
            eps, _ = floquet_quasi_energies(hr, patch, amplitude=0.12, omega=0.6,
                                            n_photon=4, polarization=pol)
            mid = eps.shape[1] // 2
            return float((eps[:, mid] - eps[:, mid - 1]).min()) * HARTREE_TO_EV * 1e3

        coarse, fine = min_split("linear", 0.02, 41), min_split("linear", 0.02, 401)
        assert fine < 0.15 * coarse, f"linear splitting {fine:.2f} meV does not refine away"
        gap_c = min_split("circular", 0.02, 401)
        assert gap_c > 100.0 and gap_c > 20 * fine

    def test_helicity_reverses_nothing_in_the_gap_but_flips_the_blocks(self):
        """Both helicities gap equally (the gap is |mass|), while the p = +-1
        blocks are exchanged -- the signature of opposite induced Chern number."""
        hr = _graphene()
        kw = dict(amplitude=0.12, omega=0.6, n_photon=4)
        e_p, _ = floquet_quasi_energies(hr, K_POINT, helicity=+1, **kw)
        e_m, _ = floquet_quasi_energies(hr, K_POINT, helicity=-1, **kw)
        mid = e_p.shape[1] // 2
        assert (e_p[0, mid] - e_p[0, mid - 1]) == pytest.approx(
            e_m[0, mid] - e_m[0, mid - 1], rel=1e-9)
        bp = floquet_blocks(hr, K_POINT, amplitude=0.12, n_photon=1, helicity=+1)
        bm = floquet_blocks(hr, K_POINT, amplitude=0.12, n_photon=1, helicity=-1)
        assert np.abs(bp[2 + 1] - bm[2 - 1]).max() < 1e-12   # p=+1 <-> p=-1


class TestDynamicalNarrowing:
    def test_bandwidth_collapses_at_the_first_zero_of_J0(self):
        """The p = 0 block carries J_0(A0 rho), so at A0 rho = 2.405 the
        nearest-neighbour hopping is switched off and the band flattens --
        coherent destruction of tunnelling, and a check that the Bessel argument
        is the physical A0*|R_perp| and not something off by a lattice factor."""
        hr = _graphene()
        k = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1 / 3, -1 / 3, 0.0]])
        from waw.analysis.elph import band_eigensystem
        w0 = np.ptp(band_eigensystem(hr, k)[0])
        # all three bonds have length BOND, so one amplitude kills them together
        b = floquet_blocks(hr, k, amplitude=2.40483 / BOND, n_photon=0)[0]
        w_driven = np.ptp(np.linalg.eigvalsh(0.5 * (b + np.conj(np.swapaxes(b, -1, -2)))))
        assert w_driven < 0.06 * w0


class TestSpectralFunction:
    def test_undriven_spectrum_has_all_weight_in_the_central_replica(self):
        hr = _graphene()
        k = np.array([[0.2, 0.1, 0.0]])
        omega = 0.05
        E = np.linspace(-0.45, 0.45, 1200)
        A = floquet_spectral_function(hr, k, E, amplitude=0.0, omega=omega,
                                      n_photon=2, broadening=2e-3)
        from waw.analysis.elph import band_eigensystem
        e0 = band_eigensystem(hr, k)[0][0]
        peaks = E[np.r_[False, (A[0][1:-1] > A[0][:-2]) & (A[0][1:-1] > A[0][2:]), False]]
        # every undriven band appears; nothing at +-omega
        for e in e0:
            assert np.abs(peaks - e).min() < 5e-3
        for e in e0:
            for m in (-1, 1):
                near = np.abs(E - (e + m * omega)) < 3e-3
                assert A[0][near].max() < 0.05 * A[0].max()

    def test_driving_transfers_weight_into_sidebands(self):
        hr = _graphene()
        k = np.array([[0.2, 0.1, 0.0]])
        omega = 0.05
        E = np.linspace(-0.45, 0.45, 1200)
        common = dict(omega=omega, n_photon=3, broadening=2e-3)
        A0 = floquet_spectral_function(hr, k, E, amplitude=0.0, **common)
        Ad = floquet_spectral_function(hr, k, E, amplitude=0.25, **common)
        from waw.analysis.elph import band_eigensystem
        e0 = band_eigensystem(hr, k)[0][0]
        # the sideband sits one photon above the DRIVEN main band, and the drive
        # has already narrowed that band by J_0(A0*bond) = 0.89 -- looking for it
        # at the undriven energy + omega misses it by several broadenings
        eps, _ = floquet_quasi_energies(hr, k, amplitude=0.25, omega=omega,
                                        n_photon=3)
        main = eps[0][np.abs(eps[0] - e0[0]).argmin()]
        side = np.abs(E - (main + omega)) < 4e-3
        base = np.abs(E - (e0[0] + omega)) < 4e-3
        assert Ad[0][side].max() > 20 * A0[0][base].max()
        # spectral weight is conserved: the drive redistributes, not creates
        assert np.trapezoid(Ad[0], E) == pytest.approx(np.trapezoid(A0[0], E), rel=0.02)

    def test_orbital_weights_select_a_sublattice(self):
        hr = _graphene()
        k = np.array([[0.2, 0.1, 0.0]])
        E = np.linspace(-0.4, 0.4, 400)
        kw = dict(amplitude=0.2, omega=0.05, n_photon=2, broadening=3e-3)
        full = floquet_spectral_function(hr, k, E, **kw)
        one = floquet_spectral_function(hr, k, E, orbital_weights=[1.0, 0.0], **kw)
        assert np.trapezoid(one[0], E) == pytest.approx(
            0.5 * np.trapezoid(full[0], E), rel=0.05)


class TestTruncation:
    def test_convergence_helper_reports_a_shrinking_shift(self):
        hr = _graphene()
        k = np.array([[1 / 3, -1 / 3, 0.0], [0.25, 0.1, 0.0]])
        c = floquet_convergence(hr, k, amplitude=0.3, omega=0.3,
                                photon_range=(1, 2, 3, 4, 5))
        sh = [s for s in c["max_shift_meV"] if not np.isnan(s)]
        assert sh[-1] < sh[0]
        assert sh[-1] < 1.0

    def test_a_stronger_drive_needs_more_photons(self):
        hr = _graphene()
        k = np.array([[1 / 3, -1 / 3, 0.0]])
        weak = floquet_convergence(hr, k, amplitude=0.1, omega=0.3,
                                   photon_range=(1, 2))["max_shift_meV"][1]
        strong = floquet_convergence(hr, k, amplitude=0.8, omega=0.3,
                                     photon_range=(1, 2))["max_shift_meV"][1]
        assert strong > weak


class TestUnits:
    def test_field_to_amplitude_conversion(self):
        """A0 = E/omega in atomic units. Worth stating the scale plainly: at
        1e9 V/m and 120 meV (the Bi2Se3 experiment's photon energy) A0 = 0.44
        /Bohr, so A0 * (a few Bohr) is of order 1 -- the mid-infrared drive is
        NOT perturbative, which is exactly why sidebands are visible there and
        why the photon truncation has to be converged rather than assumed."""
        a0 = drive_amplitude_from_field(1e9, 0.120)
        assert a0 == pytest.approx(0.4410, rel=1e-3)
        # linear in field, inverse in frequency
        assert drive_amplitude_from_field(2e9, 0.120) == pytest.approx(2 * a0, rel=1e-12)
        assert drive_amplitude_from_field(1e9, 0.240) == pytest.approx(a0 / 2, rel=1e-12)

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="omega must be > 0"):
            drive_amplitude_from_field(1e9, 0.0)
        hr = _graphene()
        with pytest.raises(ValueError, match="broadening"):
            floquet_spectral_function(hr, K_POINT, np.linspace(-1, 1, 10),
                                      amplitude=0.1, omega=0.05, n_photon=1,
                                      broadening=0.0)
        with pytest.raises(ValueError, match="polarization"):
            floquet_blocks(hr, K_POINT, amplitude=0.1, n_photon=1,
                           polarization="elliptical")


class TestDrivePlane:
    """The field plane is a property of the experiment, not of the axes the
    Hamiltonian happens to be written in."""

    @staticmethod
    def _rotate(hr, Rm):
        from waw.core.hamiltonian import HamiltonianR
        return HamiltonianR(H_R=hr.H_R, R_vectors=hr.R_vectors, degen=hr.degen,
                            nw=hr.nw, centres=hr.centres @ Rm.T,
                            real_lattice=hr.real_lattice @ Rm.T,
                            mp_grid=hr.mp_grid)

    def test_rotating_crystal_and_field_together_changes_nothing(self):
        """Rotate the lattice, the centres and the drive plane by the same
        orthogonal matrix: every quasi-energy must be untouched. This is the
        check that matters for a slab, whose surface normal is whatever the
        surface transformation makes it and is essentially never Cartesian z."""
        hr = _graphene()
        c, sn = np.cos(0.7), np.sin(0.7)
        # tilt the whole crystal out of the xy plane, about the x axis
        Rm = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]], dtype=float)
        kw = dict(amplitude=0.12, omega=0.6, n_photon=3)
        ref, _ = floquet_quasi_energies(hr, K_POINT, **kw)
        rot, _ = floquet_quasi_energies(self._rotate(hr, Rm), K_POINT,
                                        drive_plane=np.array([[1.0, 0, 0],
                                                              [0, c, sn]]), **kw)
        assert np.abs(rot - ref).max() < 1e-12

    def test_a_tilted_crystal_with_the_default_plane_gets_it_wrong(self):
        """The converse, so the parameter is shown to be load-bearing: leave the
        drive in the xy plane while the crystal is tilted out of it and the
        induced gap is wrong by tens of meV."""
        hr = _graphene()
        c, sn = np.cos(0.7), np.sin(0.7)
        Rm = np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]], dtype=float)
        kw = dict(amplitude=0.12, omega=0.6, n_photon=3)
        ref, _ = floquet_quasi_energies(hr, K_POINT, **kw)
        bad, _ = floquet_quasi_energies(self._rotate(hr, Rm), K_POINT, **kw)
        mid = ref.shape[1] // 2
        g_ref = (ref[0, mid] - ref[0, mid - 1]) * HARTREE_TO_EV * 1e3
        g_bad = (bad[0, mid] - bad[0, mid - 1]) * HARTREE_TO_EV * 1e3
        assert abs(g_bad - g_ref) > 20.0

    def test_a_field_normal_to_a_two_dimensional_crystal_does_nothing(self):
        """Light polarised along the surface normal cannot drive in-plane
        hopping: every bond has zero projection, so J_0 = 1 and J_{p!=0} = 0."""
        hr = _graphene()
        b = floquet_blocks(hr, K_POINT, amplitude=5.0, n_photon=2,
                           drive_plane=np.array([[0.0, 0, 1.0], [1.0, 0, 0]]),
                           polarization="linear")
        for idx, p in enumerate(range(-4, 5)):
            if p != 0:
                assert np.abs(b[idx]).max() < 1e-14


class TestDrivenDiracCone:
    """
    The Bi2Se3 tr-ARPES experiment, reduced to its two numbers.

    Wang, Steinberg, Jarillo-Herrero and Gedik (Science 342, 453 (2013)) drove
    the Bi2Se3(111) surface Dirac cone with a 120 meV mid-infrared pulse of peak
    field 2.5e7 V/m and measured, at t = 0:

      * a dynamical gap 2*Delta = 62 meV at the n = 0 / n = +-1 avoided crossing,
        seen for k PERPENDICULAR to a linear field and absent for k parallel;
      * no resolvable gap where n = +1 crosses n = -1;
      * a gap 2*kappa = 53 meV at the Dirac point under circular polarisation,
        their evidence for broken time-reversal symmetry.

    For an ideal cone the first of these is 2*Delta = hbar v A0 exactly (the
    +-1 harmonic of v sigma.A(t) has amplitude v A0 / 2, and an avoided crossing
    opens twice its coupling), which with hbar v = 3 eV Ang is 62.5 meV. That
    the experiment landed on 62 meV is what makes this a usable test: it fixes
    the amplitude convention, the harmonic weights and the polarisation
    geometry all at once, against a measurement.

    The cone is regularised on a square lattice with a Wilson mass. The lattice
    is not free: it leaves a residual splitting at the Dirac point growing as
    a^2 (0.8 meV at a = 1 Ang, 19 meV at 5 Ang), so a is kept small and that
    residual is the tolerance on the "gapless" assertions rather than zero.
    """

    A_LAT = 1.0                    # Ang; small enough that the lattice artefact is < 1 meV
    HBAR_V = 3.0                   # eV Ang, the measured Bi2Se3 surface velocity
    OMEGA_EV = 0.120
    FIELD = 2.5e7                  # V/m

    @classmethod
    def _cone(cls):
        from waw.units import BOHR_TO_ANG
        a = cls.A_LAT / BOHR_TO_ANG
        v = cls.HBAR_V / (HARTREE_TO_EV * BOHR_TO_ANG)
        M = 0.15
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.diag([1.0, -1.0]).astype(complex)
        Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
        H = np.zeros((5, 2, 2), dtype=complex)
        H[0] = 2 * M * sz
        H[1] = -1j * v / (2 * a) * sx - 0.5 * M * sz
        H[2] = +1j * v / (2 * a) * sx - 0.5 * M * sz
        H[3] = -1j * v / (2 * a) * sy - 0.5 * M * sz
        H[4] = +1j * v / (2 * a) * sy - 0.5 * M * sz
        hr = HamiltonianR(H_R=torch.tensor(H), R_vectors=np.array(Rs, dtype=np.int64),
                          degen=np.ones(5, dtype=np.int64), nw=2,
                          centres=np.zeros((2, 3)),
                          real_lattice=np.diag([a, a, 30.0]), mp_grid=(1, 1, 1))
        return hr, a, v

    @classmethod
    def _crossing_k(cls):
        """omega / 2v, where the n = 0 and n = +1 cones meet. 1/Bohr."""
        from waw.units import EV_TO_HARTREE
        _, _, v = cls._cone()
        return cls.OMEGA_EV * EV_TO_HARTREE / (2 * v)

    @classmethod
    def _scan(cls, axis, pol, k_lo, k_hi, n=241):
        """Quasi-energies along one axis, k in 1/Bohr."""
        hr, a, v = cls._cone()
        from waw.units import EV_TO_HARTREE
        w = cls.OMEGA_EV * EV_TO_HARTREE
        A0 = drive_amplitude_from_field(cls.FIELD, cls.OMEGA_EV)
        t = np.linspace(k_lo, k_hi, n) * a / (2 * np.pi)
        k = np.zeros((n, 3))
        k[:, axis] = t
        eps, _ = floquet_quasi_energies(hr, k, amplitude=A0, omega=w, n_photon=4,
                                        polarization=pol)
        return eps, eps.shape[1] // 2, v, A0, w

    def test_dynamical_gap_matches_the_measured_62_meV(self):
        """k perpendicular to a linear field: the n = 0 / n = +1 avoided crossing."""
        kc = self._crossing_k()
        eps, mid, v, A0, w = self._scan(1, "linear", 0.6 * kc, 1.6 * kc)
        gap = float((eps[:, mid + 1] - eps[:, mid]).min()) * HARTREE_TO_EV * 1e3
        analytic = v * A0 * HARTREE_TO_EV * 1e3
        assert gap == pytest.approx(analytic, rel=0.03), (
            f"{gap:.1f} meV vs hbar v A0 = {analytic:.1f} meV")
        assert gap == pytest.approx(62.0, abs=6.0)      # the measurement

    def test_no_gap_for_momentum_along_a_linear_field(self):
        """The polarisation-dependent asymmetry the paper reports: gapped along
        k_y, gapless along k_x."""
        kc = self._crossing_k()
        eps, mid, v, A0, w = self._scan(0, "linear", 0.6 * kc, 1.6 * kc)
        gap = float((eps[:, mid + 1] - eps[:, mid]).min()) * HARTREE_TO_EV * 1e3
        assert gap < 5.0, f"linear light gapped the parallel direction by {gap:.1f} meV"

    def test_circular_light_gaps_the_dirac_point(self):
        """2*kappa, the time-reversal-breaking signature. The Magnus estimate
        2 (v A0)^2 / omega = 65 meV overshoots because 120 meV is not large
        against v A0; the exact Sambe diagonalisation is what to compare."""
        eps, mid, v, A0, w = self._scan(0, "circular", -1e-9, 1e-9, n=3)
        gap = float(eps[1, mid] - eps[1, mid - 1]) * HARTREE_TO_EV * 1e3
        magnus = 2 * (v * A0) ** 2 / w * HARTREE_TO_EV * 1e3
        assert gap < magnus
        assert gap == pytest.approx(53.0, abs=12.0), f"Dirac gap {gap:.1f} meV vs measured 53"

    def test_linear_light_leaves_the_dirac_point_ungapped(self):
        eps, mid, *_ = self._scan(0, "linear", -1e-9, 1e-9, n=3)
        gap = float(eps[1, mid] - eps[1, mid - 1]) * HARTREE_TO_EV * 1e3
        assert gap < 3.0

    def test_the_crossing_sits_where_the_cone_puts_it(self):
        """omega / 2v = 0.020 1/Ang for these parameters. The paper measures
        0.03 and attributes the excess to the cone's own non-linearity, so this
        checks the ideal-cone value, not theirs."""
        from waw.units import BOHR_TO_ANG
        kc = self._crossing_k()
        eps, mid, v, A0, w = self._scan(1, "linear", 0.4 * kc, 1.8 * kc, n=401)
        t = np.linspace(0.4 * kc, 1.8 * kc, 401)
        k_at_gap = t[(eps[:, mid + 1] - eps[:, mid]).argmin()] / BOHR_TO_ANG
        assert k_at_gap == pytest.approx(0.020, abs=0.004)


class TestConvergenceHelper:
    def test_tracks_the_same_physical_states_across_truncations(self):
        """The Sambe spectrum is unbounded and gets DENSER as photon sectors are
        added, so any tracker that indexes into the raw sorted eigenvalues -- or
        picks the eigenvalues nearest an energy -- follows a different physical
        state at every truncation and reports drift on a system that cannot have
        moved. `floquet_convergence` restricts to the m = 0-dominant states
        first, which is what makes the number it prints mean something.

        Test: a 24-orbital slab (dense spectrum) with a drive so weak that
        nothing can shift, so any reported drift is the tracker's own."""
        from waw.analysis.surface import build_slab
        # a genuinely coupled 12-layer simple-cubic slab: 12 non-degenerate
        # standing waves per k_par, so the spectrum is dense without being
        # degenerate (12 decoupled sheets would be, and degenerate eigenvectors
        # can mix photon sectors arbitrarily, which is a different problem)
        a, t = 5.0, 0.05
        Rs = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (0, 0, 1), (0, 0, -1)]
        H = np.zeros((len(Rs), 1, 1), dtype=np.complex128)
        H[1:, 0, 0] = -t
        cubic = HamiltonianR(H_R=torch.tensor(H), R_vectors=np.array(Rs, dtype=np.int64),
                             degen=np.ones(len(Rs), dtype=np.int64), nw=1,
                             centres=np.zeros((1, 3)), real_lattice=a * np.eye(3),
                             mp_grid=(6, 6, 6))
        slab = build_slab(cubic, a * np.eye(3), (0, 0, 1), 12)
        k = np.array([[0.2, 0.13, 0.0]])
        kw = dict(amplitude=1e-4, omega=0.05, photon_range=(1, 2, 3, 4))
        for label, extra in (("default", {}), ("e_ref", dict(e_ref=0.0))):
            shifts = floquet_convergence(slab, k, **kw, **extra)["max_shift_meV"]
            assert max(shifts[1:]) < 1e-6, f"{label}: {shifts}"

        # the naive alternative, for contrast: index into the full spectrum
        naive = []
        for n in (1, 2, 3, 4):
            eps, _ = floquet_quasi_energies(slab, k, amplitude=1e-4, omega=0.05,
                                            n_photon=n)
            mid = eps.shape[1] // 2
            naive.append(eps[0, mid - 2:mid + 2])
        drift = max(np.abs(naive[i] - naive[i - 1]).max() for i in (1, 2, 3))
        assert drift * HARTREE_TO_EV * 1e3 > 10.0
