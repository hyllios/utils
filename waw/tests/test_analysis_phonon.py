"""
Unit tests for waw.analysis.phonon + interfaces.quantum_espresso.phonon_io
(new capability: Fourier interpolation of q2r.x real-space force constants,
the lattice-dynamics analogue of this project's own electronic Wannier-
Hamiltonian interpolation -- reuses core.hamiltonian._wigner_seitz and
core.ws_distance.build_ws_distance directly, since the atom-position-aware
folding needed here is mathematically identical to wannier90's own
use_ws_distance, confirmed by reading QE's matdyn.f90::frc_blk/wsweight
source directly).

Primary validation: a REAL (not synthetic) force-constant file for
diamond, bundled unmodified from Quantum ESPRESSO's own example suite
(tests/data/diamond_phonon/, see that directory's README for provenance).
Diamond's zone-center optical phonon is a very well known experimental
quantity (~1332-1333 cm^-1) -- reproducing it is a decisive, independent
end-to-end check of the reader + WS-folding + mass-weighting +
diagonalization + unit-conversion pipeline all at once, not requiring any
of waw's own DFT.
"""

from pathlib import Path

import numpy as np
import pytest

from waw.interfaces.quantum_espresso.phonon_io import read_force_constants, read_ph_frequencies
from waw.analysis.phonon import (
    interpolate_phonons, atom_projected_weights, apply_acoustic_sum_rule,
    phonon_density_of_states, phonon_hamiltonian, cm1_to_omega2_au, omega2_au_to_cm1,
    PhononBands, PhononDOS,
)
from waw.core.hamiltonian import interpolate_bands, HamiltonianR

DATA_DIR = Path(__file__).parent / "data" / "diamond_phonon"
MGB2_DATA_DIR = Path(__file__).parent / "data" / "mgb2_phonon"


def _mgb2_fc():
    d = read_force_constants(MGB2_DATA_DIR / "mgb2.fc")
    a_ang, c_ang = 3.086, 3.523
    from waw.units import ANG_TO_BOHR
    a, c = a_ang * ANG_TO_BOHR, c_ang * ANG_TO_BOHR
    real_lattice = np.array([[a, 0.0, 0.0],
                              [-a / 2, a * np.sqrt(3) / 2, 0.0],
                              [0.0, 0.0, c]])
    positions_frac = np.array([[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])
    return d, real_lattice, positions_frac


def _diamond_fc():
    d = read_force_constants(DATA_DIR / "diam.ifc")
    a = d["celldm"][0]
    # ibrav=2 (FCC), standard QE convention
    real_lattice = a * np.array([[-0.5, 0.0, 0.5],
                                  [0.0, 0.5, 0.5],
                                  [-0.5, 0.5, 0.0]])
    tau_cart = d["tau_alat"] * a   # alat units -> Bohr
    atom_positions_frac = tau_cart @ np.linalg.inv(real_lattice)
    return d, real_lattice, atom_positions_frac


class TestReadForceConstants:
    def test_diamond_header_fields(self):
        d = read_force_constants(DATA_DIR / "diam.ifc")
        assert d["ntyp"] == 1
        assert d["nat"] == 2
        assert d["species"] == ["C"]
        np.testing.assert_allclose(d["masses_amu"], [12.01078057], rtol=1e-6)
        assert tuple(d["types"]) == (0, 0)
        assert d["has_zstar"] is True
        assert d["epsilon"].shape == (3, 3)
        assert d["born_charges"].shape == (2, 3, 3)
        assert tuple(d["grid"]) == (3, 3, 3)
        assert d["fc"].shape == (3, 3, 2, 2, 3, 3, 3)

    def test_diamond_fc_units_are_hartree_not_rydberg(self):
        """The reader must divide the file's raw Ry/Bohr^2 values by 2 --
        cross-check against the file's own first raw value (0.975215616296
        Ry, per the fixture's line 19)."""
        d = read_force_constants(DATA_DIR / "diam.ifc")
        np.testing.assert_allclose(d["fc"][0, 0, 0, 0, 0, 0, 0], 0.975215616296 / 2.0, rtol=1e-10)

    def test_diamond_ibrav2_has_no_at_alat_block(self):
        d = read_force_constants(DATA_DIR / "diam.ifc")
        assert d["ibrav"] == 2
        assert d["at_alat"] is None


class TestReadForceConstantsIbrav0:
    """MgB2 (see data/mgb2_phonon/README.md): this project's own QE
    convention is ALWAYS ibrav=0, which q2r.x/matdyn.f90 writes 3 extra
    lattice-vector lines for right after the header -- the diamond
    fixture (ibrav=2) never exercises this. Also exercises a genuine QE
    cosmetic quirk (an all-blank species symbol field for the 2nd
    species) that a naive parser silently misaligns."""

    def test_ibrav0_lattice_vectors_parsed(self):
        d = read_force_constants(MGB2_DATA_DIR / "mgb2.fc")
        assert d["ibrav"] == 0
        assert d["at_alat"] is not None
        assert d["at_alat"].shape == (3, 3)
        # hexagonal: a1=(1,0,0), a2=(-1/2,sqrt(3)/2,0), a3=(0,0,c/a)
        np.testing.assert_allclose(d["at_alat"][0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(d["at_alat"][1], [-0.5, np.sqrt(3) / 2, 0.0], atol=1e-6)
        assert d["at_alat"][2, 2] > 1.0   # c/a > 1 for this cell

    def test_blank_species_symbol_does_not_corrupt_mass_or_type_mapping(self):
        """The 2nd species (Mg) prints as an all-blank quoted field in the
        real file -- must not misparse the mass value as the symbol or
        vice versa (a bug this exact fixture caught: a naive
        quote-strip-then-split() silently collapses the blank field and
        shifts every subsequent token by one)."""
        d = read_force_constants(MGB2_DATA_DIR / "mgb2.fc")
        assert d["ntyp"] == 2
        assert d["species"][0] == "B"
        assert d["species"][1] == ""   # genuinely blank in the file, not corrupted
        np.testing.assert_allclose(sorted(d["masses_amu"]), [10.81000051, 24.30500115], rtol=1e-6)
        assert set(d["types"]) == {0, 1}


class TestReadPhFrequencies:
    """Transcribed verbatim from a real single-q ph.x run on fcc Al."""

    TEXT = (
        "     Representation #   3 mode #   3\n"
        "\n"
        "     freq (    1) =       2.393647 [THz] =      79.843469 [cm-1]\n"
        "     freq (    2) =       2.625675 [THz] =      87.583100 [cm-1]\n"
        "     freq (    3) =       5.461610 [THz] =     182.179706 [cm-1]\n"
        "\n"
        "     Mode symmetry, C_1 (1)     point group:\n"
    )

    def test_single_q_three_modes(self, tmp_path):
        out = tmp_path / "al.ph.out"
        out.write_text(self.TEXT)
        freqs = read_ph_frequencies(out, n_modes=3)
        assert freqs.shape == (1, 3)
        np.testing.assert_allclose(freqs[0], [79.843469, 87.583100, 182.179706])

    def test_multiple_q_points_concatenated(self, tmp_path):
        out = tmp_path / "al.ph.out"
        out.write_text(self.TEXT + self.TEXT)
        freqs = read_ph_frequencies(out, n_modes=3)
        assert freqs.shape == (2, 3)
        np.testing.assert_allclose(freqs[0], freqs[1])

    def test_no_matches_raises(self, tmp_path):
        out = tmp_path / "empty.out"
        out.write_text("nothing here\n")
        with pytest.raises(ValueError):
            read_ph_frequencies(out, n_modes=3)

    def test_bad_mode_count_raises(self, tmp_path):
        out = tmp_path / "al.ph.out"
        out.write_text(self.TEXT)
        with pytest.raises(ValueError):
            read_ph_frequencies(out, n_modes=2)


class TestInterpolatePhonons:
    def test_diamond_gamma_optical_mode_matches_experiment(self):
        """Diamond's triply-degenerate zone-center optical phonon is a very
        well known experimental quantity (~1332-1333 cm^-1, Raman-active
        T2g mode) -- typical PBE-DFPT slightly overestimates it; demand
        agreement to a few cm^-1/a fraction of a percent, not exact."""
        d, real_lattice, positions = _diamond_fc()
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        optical = np.sort(bands.freq_cm1[0])[-3:]
        np.testing.assert_allclose(optical, 1337.9, atol=1.0)   # triply degenerate
        assert abs(optical.mean() - 1332.5) < 10.0   # within ~1% of the real experimental value

    def test_diamond_gamma_acoustic_modes_near_zero(self):
        """The 3 acoustic modes at Gamma must vanish exactly by translational
        invariance -- a small residual (a few cm^-1) is expected here since
        no acoustic-sum-rule correction is applied to the raw force
        constants, but it must be MUCH smaller than the optical branch."""
        d, real_lattice, positions = _diamond_fc()
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        acoustic = np.sort(bands.freq_cm1[0])[:3]
        assert np.all(np.abs(acoustic) < 10.0)

    def test_diamond_bands_are_real_and_finite_off_gamma(self):
        d, real_lattice, positions = _diamond_fc()
        qpts = np.array([[0.5, 0.0, 0.0], [0.25, 0.25, 0.25], [0.1, 0.2, -0.05]])
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=qpts)
        assert bands.freq_cm1.shape == (3, 6)
        assert np.all(np.isfinite(bands.freq_cm1))
        # Hermitian dynamical matrix -> real eigenvalues; no big unstable
        # (very negative) modes expected for a well-converged bulk diamond
        assert bands.freq_cm1.min() > -50.0

    def test_eigenvectors_are_orthonormal(self):
        """The mass-weighted eigenvectors from `eigh` on a Hermitian matrix
        must be exactly orthonormal -- needed for atom_projected_weights'
        own "weights sum to 1" guarantee."""
        d, real_lattice, positions = _diamond_fc()
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.13, 0.27, -0.05]])
        V = bands.eigvecs[0]
        np.testing.assert_allclose(V.conj().T @ V, np.eye(6), atol=1e-10)


class TestPhononHamiltonian:
    """`phonon_hamiltonian` packages the mass-weighted dynamical matrix as
    a plain `HamiltonianR` -- so `analysis.surface.surface_spectral_function`
    (written for the electronic case, touches only `hr.H_R`/`R_vectors`/
    `degen`/`nw`) runs unchanged on phonons. It deliberately skips
    `interpolate_phonons`'s atom-position (`build_ws_distance`) refinement,
    so the two are NOT expected to agree away from Gamma -- but MUST agree
    EXACTLY at Gamma, where every phase factor is 1 regardless of which
    convention is used, a decisive convention-independent check."""

    def test_returns_hamiltonian_r(self):
        d, real_lattice, _ = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        hr = phonon_hamiltonian(d, real_lattice)
        assert isinstance(hr, HamiltonianR)
        assert hr.nw == 3 * d["nat"]
        assert hr.H_R.shape == (len(hr.R_vectors), hr.nw, hr.nw)

    def test_matches_interpolate_phonons_exactly_at_gamma(self):
        d, real_lattice, positions = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        hr = phonon_hamiltonian(d, real_lattice)
        omega2 = interpolate_bands(hr, np.array([[0.0, 0.0, 0.0]]))
        freq_from_hr = omega2_au_to_cm1(omega2)
        np.testing.assert_allclose(np.sort(freq_from_hr[0]), np.sort(bands.freq_cm1[0]), atol=1e-3)

    def test_off_gamma_agrees_only_qualitatively(self):
        """Away from Gamma the two conventions genuinely differ (one has
        the atom-position refinement, one doesn't) -- same ballpark, not
        an exact match; this documents that difference rather than
        asserting a false exact equivalence."""
        d, real_lattice, positions = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        q = [[0.1, 0.2, 0.05]]
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=q)
        hr = phonon_hamiltonian(d, real_lattice)
        freq_from_hr = omega2_au_to_cm1(interpolate_bands(hr, np.array(q)))
        assert np.all(np.isfinite(freq_from_hr))
        # same overall scale, not a tight match
        np.testing.assert_allclose(np.sort(freq_from_hr[0]), np.sort(bands.freq_cm1[0]), atol=30.0)


class TestOmega2Conversion:
    def test_roundtrip(self):
        nu = np.array([-50.0, 0.0, 328.3, 702.5, 1337.9])
        np.testing.assert_allclose(omega2_au_to_cm1(cm1_to_omega2_au(nu)), nu, atol=1e-8)

    def test_negative_cm1_maps_to_negative_omega2(self):
        """Preserves the standard imaginary-mode sign convention."""
        assert cm1_to_omega2_au(-100.0) < 0.0
        assert cm1_to_omega2_au(100.0) > 0.0


class TestApplyAcousticSumRule:
    """Real MgB2 case: matches matdyn.f90's own `asr='simple'` exactly
    (verified by transcribing that subroutine's loop directly)."""

    def test_mgb2_acoustic_modes_are_large_before_asr(self):
        """Establishes the baseline this fix addresses: MgB2 is a metal on
        a modest (12,12,8) electronic k-mesh, so the RAW force constants
        violate the acoustic sum rule by tens of cm^-1 at Gamma -- much
        larger than diamond's (an insulator) few-cm^-1 residual."""
        d, real_lattice, positions = _mgb2_fc()
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        acoustic = np.sort(bands.freq_cm1[0])[:3]
        assert np.all(np.abs(acoustic) > 50.0)   # nowhere near zero without ASR

    def test_mgb2_acoustic_modes_vanish_after_asr(self):
        d, real_lattice, positions = _mgb2_fc()
        d_corrected = apply_acoustic_sum_rule(d)
        bands = interpolate_phonons(d_corrected, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        acoustic = np.sort(bands.freq_cm1[0])[:3]
        np.testing.assert_allclose(acoustic, 0.0, atol=1e-4)

    def test_mgb2_optical_modes_barely_shift(self):
        """The correction should fix translational invariance without
        materially changing the (already physical) optical branches."""
        d, real_lattice, positions = _mgb2_fc()
        bands_raw = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        bands_asr = interpolate_phonons(apply_acoustic_sum_rule(d), real_lattice, positions,
                                         qpts_frac=[[0.0, 0.0, 0.0]])
        optical_raw = np.sort(bands_raw.freq_cm1[0])[3:]
        optical_asr = np.sort(bands_asr.freq_cm1[0])[3:]
        np.testing.assert_allclose(optical_asr, optical_raw, atol=15.0)   # cm^-1, small shift only

    def test_does_not_mutate_input(self):
        d, _, _ = _mgb2_fc()
        fc_before = d["fc"].copy()
        apply_acoustic_sum_rule(d)
        np.testing.assert_array_equal(d["fc"], fc_before)

    def test_synthetic_translational_invariance_exact(self):
        """A hand-built 2-atom fc with a deliberately nonzero net on-site
        term: after ASR, summing any atom's force constants over (nb,R)
        must vanish EXACTLY (the defining property, checked directly
        rather than via the eigenvalue side effect)."""
        nat, grid = 2, (1, 1, 1)
        rng = np.random.default_rng(0)
        fc = rng.normal(size=(3, 3, nat, nat, 1, 1, 1))
        fc_data = {"nat": nat, "fc": fc}
        corrected = apply_acoustic_sum_rule(fc_data)
        total = corrected["fc"].sum(axis=(3, 4, 5, 6))
        np.testing.assert_allclose(total, 0.0, atol=1e-12)


class TestAtomProjectedWeights:
    def test_diamond_single_species_all_weight_on_carbon(self):
        d, real_lattice, positions = _diamond_fc()
        bands = interpolate_phonons(d, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0], [0.3, 0.1, 0.0]])
        w = atom_projected_weights(bands, d["types"], d["ntyp"])
        assert w.shape == (2, 6, 1)
        np.testing.assert_allclose(w[..., 0], 1.0, atol=1e-10)

    def test_synthetic_two_species_weights_sum_to_one_and_localize_correctly(self):
        """A synthetic 2-atom, 2-species system with force constants built
        so each atom is dynamically ISOLATED (no coupling between them) --
        every mode must localize 100% on its own atom, an unambiguous,
        independently-checkable answer for the atom-weighting sum."""
        nat = 2
        grid = (1, 1, 1)
        fc = np.zeros((3, 3, nat, nat, 1, 1, 1))
        # on-site "spring" for each atom, no na-nb coupling at all
        fc[:, :, 0, 0, 0, 0, 0] = np.diag([1.0, 1.0, 1.0]) * 0.3
        fc[:, :, 1, 1, 0, 0, 0] = np.diag([1.0, 1.0, 1.0]) * 0.5
        fc_data = {
            "nat": nat, "grid": np.array(grid), "fc": fc,
            "masses_amu": np.array([12.0, 28.0855]), "types": np.array([0, 1]),
        }
        real_lattice = np.eye(3) * 20.0
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        bands = interpolate_phonons(fc_data, real_lattice, positions, qpts_frac=[[0.0, 0.0, 0.0]])
        w = atom_projected_weights(bands, fc_data["types"], ntyp=2)[0]   # (6, 2)

        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-10)
        # each mode should be (numerically) pure: weight ~1 on exactly one species
        assert np.all((np.abs(w[:, 0] - 1.0) < 1e-8) | (np.abs(w[:, 1] - 1.0) < 1e-8))
        # 3 modes on species 0 (lighter/softer atom), 3 on species 1
        assert (np.abs(w[:, 0] - 1.0) < 1e-8).sum() == 3
        assert (np.abs(w[:, 1] - 1.0) < 1e-8).sum() == 3


class TestPhononDensityOfStates:
    def test_diamond_single_species_dos_species_equals_dos_total(self):
        """Single-species diamond -- the PDOS decomposition has nowhere
        else to put weight, so dos_species[:, 0] must equal dos_total
        exactly (not just sum to it, since there's only one species)."""
        d, real_lattice, positions = _diamond_fc()
        result = phonon_density_of_states(d, real_lattice, positions, mesh=(4, 4, 4), n_freq=200)
        assert isinstance(result, PhononDOS)
        np.testing.assert_allclose(result.dos_species[:, 0], result.dos_total, atol=1e-10)

    def test_mgb2_species_dos_sums_to_total(self):
        """Real 2-species case: summing the per-species PDOS over species
        must reproduce the total DOS at every frequency -- the decisive
        check that the decomposition doesn't double-count or drop weight."""
        d, real_lattice, positions = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        result = phonon_density_of_states(d, real_lattice, positions, mesh=(6, 6, 6), n_freq=300)
        np.testing.assert_allclose(result.dos_species.sum(axis=-1), result.dos_total, atol=1e-8)

    def test_dos_is_nonnegative_and_finite(self):
        d, real_lattice, positions = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        result = phonon_density_of_states(d, real_lattice, positions, mesh=(6, 6, 6), n_freq=300)
        assert np.all(np.isfinite(result.dos_total))
        assert np.all(result.dos_total >= 0.0)
        assert np.all(result.dos_species >= 0.0)

    def test_integrates_to_the_right_mode_count(self):
        """Integrating the total DOS over frequency must recover 3*nat
        (the total number of phonon branches) -- the phonon analogue of
        the electronic DOS integrating to the number of bands, a decisive
        normalization check independent of the broadening width."""
        d, real_lattice, positions = _mgb2_fc()
        d = apply_acoustic_sum_rule(d)
        result = phonon_density_of_states(d, real_lattice, positions, mesh=(8, 8, 8),
                                          n_freq=2000, sigma_cm1=3.0, pad_cm1=50.0)
        integral = np.trapezoid(result.dos_total, result.freq_cm1)
        np.testing.assert_allclose(integral, 3 * d["nat"], rtol=0.02)

    def test_explicit_freq_grid_is_used_verbatim(self):
        d, real_lattice, positions = _diamond_fc()
        freq_cm1 = np.linspace(0.0, 1500.0, 50)
        result = phonon_density_of_states(d, real_lattice, positions, mesh=(4, 4, 4),
                                          freq_cm1=freq_cm1)
        np.testing.assert_array_equal(result.freq_cm1, freq_cm1)
