"""
Tests for `analysis.band_sheets` -- physical-band / Fermi-sheet identification
(soft Wannier-character weights + eigenvector-overlap band tracking).
"""

import numpy as np
import pytest

from waw.analysis import band_sheets as bs


class TestCharacterWeights:
    def test_partition_of_unity(self):
        """Groups covering all orbitals -> weights sum to 1 over sheets (U's
        columns are normalised)."""
        rng = np.random.default_rng(0)
        nk, nw = 5, 6
        M = rng.normal(size=(nk, nw, nw)) + 1j * rng.normal(size=(nk, nw, nw))
        U, _ = np.linalg.qr(M)                      # unitary per k (normalised columns)
        w = bs.band_character_weights(U, [[0, 1, 2], [3, 4, 5]])
        assert w.shape == (nk, 2, nw)
        assert np.allclose(w.sum(axis=1), 1.0)      # sum over sheets == 1 for every band
        assert np.all(w >= -1e-12)

    def test_dropped_orbitals_sum_below_one(self):
        rng = np.random.default_rng(1)
        M = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        U, _ = np.linalg.qr(M)
        w = bs.band_character_weights(U, [[0], [1]])   # orbitals 2,3 dropped
        assert np.all(w.sum(axis=1) <= 1.0 + 1e-12)

    def test_overlapping_groups_rejected(self):
        U = np.eye(3)[None]
        with pytest.raises(ValueError, match="overlap"):
            bs.band_character_weights(U, [[0, 1], [1, 2]])

    def test_pure_character_assignment(self):
        """A band that is purely orbital 0 gets weight 1 on the sheet containing
        orbital 0, 0 on the other."""
        U = np.zeros((1, 2, 2), complex)
        U[0, :, 0] = [1, 0]          # band 0 = orbital 0
        U[0, :, 1] = [0, 1]          # band 1 = orbital 1
        w = bs.band_character_weights(U, [[0], [1]])
        assert np.allclose(w[0, :, 0], [1, 0])
        assert np.allclose(w[0, :, 1], [0, 1])


class TestFollowBands:
    def test_tracks_character_through_real_crossing(self):
        """Two non-mixing bands (orbital 0 rising, orbital 1 falling) that CROSS:
        the eigenvalue-sorted index swaps at the crossing, but follow_bands must
        give the orbital-0 band a single consistent label on both sides."""
        ks = np.linspace(0.05, 0.95, 10)            # skips the exact degeneracy at 0.5
        a, b = ks, 1 - ks
        U = np.zeros((len(ks), 2, 2), complex)
        for i in range(len(ks)):
            U[i] = np.eye(2) if a[i] < b[i] else np.array([[0.0, 1.0], [1.0, 0.0]])
        label = bs.follow_bands(U, seed=0)
        # which sorted-band carries orbital-0 character at each k
        orb0_band = np.argmax(np.abs(U[:, 0, :]) ** 2, axis=1)
        orb0_label = label[np.arange(len(ks)), orb0_band]
        assert len(np.unique(orb0_label)) == 1      # one physical band throughout
        # ... and the index genuinely swapped (so this is a nontrivial test)
        assert orb0_band[0] != orb0_band[-1]


class TestSheetDOS:
    def test_sheet_dos_sums_to_total(self):
        rng = np.random.default_rng(2)
        nk, nw = 40, 4
        eig = rng.normal(size=(nk, nw))
        M = rng.normal(size=(nk, nw, nw)) + 1j * rng.normal(size=(nk, nw, nw))
        U, _ = np.linalg.qr(M)
        w = bs.band_character_weights(U, [[0, 1], [2, 3]])
        N_i = bs.sheet_dos(eig, 0.0, w, sigma=0.2)
        from waw.core.distributions import gaussian_smearing
        total = gaussian_smearing(eig, 0.2).sum() / nk
        assert np.isclose(N_i.sum(), total)
