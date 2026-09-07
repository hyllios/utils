"""
Tests for waw.interfaces.quantum_espresso.upf (Wannier90 tutorial 35's
`atom_proj_ext` file plumbing) -- round-trips against the already-committed
real Si.upf pseudopotential, no DFT run needed.
"""

from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.interfaces.quantum_espresso.upf import read_pswfc, write_atom_proj_ext
import pathlib
from waw.interfaces.quantum_espresso.upf import read_norm_conserving

REPO = Path(__file__).parent.parent
SI_UPF = REPO / "workflows" / "pseudos" / "Si.upf"


def test_read_pswfc_finds_both_channels_with_matching_grid():
    radial = read_pswfc(SI_UPF)
    assert sorted(radial) == [0, 1]   # Si: 3S (l=0), 3P (l=1)
    r0, chi0 = radial[0]
    r1, chi1 = radial[1]
    assert r0.shape == chi0.shape == r1.shape == chi1.shape
    np.testing.assert_array_equal(r0, r1)   # shared PP_MESH radial grid
    assert r0[0] == pytest.approx(0.0)
    assert np.isfinite(chi0).all() and np.isfinite(chi1).all()


def test_write_atom_proj_ext_round_trip(tmp_path):
    radial = read_pswfc(SI_UPF)
    write_atom_proj_ext(tmp_path, {"Si": radial})

    dat = tmp_path / "Si.dat"
    assert dat.exists()
    lines = dat.read_text().splitlines()

    n_pts, n_l = (int(x) for x in lines[0].split())
    # Si.upf's PP_MESH is *linear* (r[0] == 0.0 exactly) -- ln(r) is
    # undefined there, so that one point is dropped (chi(0)=0 anyway).
    assert n_pts == radial[0][0].shape[0] - 1
    assert n_l == 2
    assert [int(x) for x in lines[1].split()] == [0, 1]
    assert len(lines) == 2 + n_pts

    data = np.array([[float(x) for x in ln.split()] for ln in lines[2:]])
    r_expected = radial[0][0][1:]
    np.testing.assert_allclose(data[:, 0], np.log(r_expected))   # xgrid = ln(r)
    np.testing.assert_allclose(data[:, 1], r_expected)           # rgrid = r
    np.testing.assert_allclose(data[:, 2], radial[0][1][1:])
    np.testing.assert_allclose(data[:, 3], radial[1][1][1:])


def test_write_atom_proj_ext_rejects_mismatched_grids(tmp_path):
    r = np.linspace(0, 1, 10)
    bad = np.linspace(0, 1, 12)
    radial = {0: (r, np.zeros(10)), 1: (bad, np.zeros(12))}
    with pytest.raises(ValueError):
        write_atom_proj_ext(tmp_path, {"X": radial})


# --------------------------------------------------------------------------
# j-averaging of fully relativistic pseudopotentials (QE's average_pp)
# --------------------------------------------------------------------------

class TestJAveraging:
    """A has_so=T pseudopotential carries two KB projectors per l > 0. QE
    averages them for any collinear run, so anything rebuilt from the raw file
    -- a bare electron-phonon vertex, above all -- must average them too or it
    counts every l > 0 channel twice."""

    PSEUDOS = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "pseudos"

    def _skip_without(self, name):
        p = self.PSEUDOS / name
        if not p.exists():
            pytest.skip(f"{name} not available")
        return p

    def test_scalar_pseudopotential_is_untouched(self):
        p = self._skip_without("Nb.upf")
        a = read_norm_conserving(p)
        r = read_norm_conserving(p, average_j=False)
        assert a["ells"] == r["ells"] == [0, 0, 1, 1, 2, 2]
        assert np.abs(a["dij"] - r["dij"]).max() == 0.0
        for x, y in zip(a["betas"], r["betas"]):
            assert np.abs(x - y).max() == 0.0

    def test_relativistic_pairs_collapse(self):
        p = self._skip_without("Co-rel.upf")
        raw = read_norm_conserving(p, average_j=False)
        avg = read_norm_conserving(p)
        assert raw["ells"] == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        assert avg["ells"] == [0, 0, 1, 1, 2, 2]

    def test_averaged_couplings_are_the_multiplicity_weighted_mean(self):
        """D = [(l+1) D_{j=l+1/2} + l D_{j=l-1/2}] / (2l+1) -- each j weighted
        by 2j+1. Checked against the raw file's own numbers, so it cannot drift
        with the pseudopotential."""
        p = self._skip_without("Co-rel.upf")
        raw = read_norm_conserving(p, average_j=False)
        avg = read_norm_conserving(p)
        d_raw, d_avg = np.diag(raw["dij"]), np.diag(avg["dij"])
        assert d_avg[0] == d_raw[0] and d_avg[1] == d_raw[1]     # l = 0 untouched
        # Co orders each pair as (j = l-1/2, j = l+1/2)
        for k, (lo, hi, l) in enumerate([(2, 3, 1), (4, 5, 1), (6, 7, 2), (8, 9, 2)]):
            expect = ((l + 1) * d_raw[hi] + l * d_raw[lo]) / (2 * l + 1)
            assert d_avg[2 + k] == pytest.approx(expect, rel=1e-12)

    def test_averaged_projector_matches_the_qe_formula(self):
        p = self._skip_without("Co-rel.upf")
        raw = read_norm_conserving(p, average_j=False)
        avg = read_norm_conserving(p)
        d = np.diag(raw["dij"])
        l, lo, hi = 1, 2, 3
        d_avg = ((l + 1) * d[hi] + l * d[lo]) / (2 * l + 1)
        expect = ((l + 1) * np.sqrt(d[hi] / d_avg) * raw["betas"][hi]
                  + l * np.sqrt(d[lo] / d_avg) * raw["betas"][lo]) / (2 * l + 1)
        assert np.abs(avg["betas"][2] - expect).max() < 1e-12

    def test_rejects_a_non_diagonal_kb_matrix(self):
        from waw.interfaces.quantum_espresso.upf import average_j_channels
        b = [np.ones(5), np.ones(5)]
        dij = np.array([[1.0, 0.3], [0.3, 1.0]])
        with pytest.raises(ValueError, match="not diagonal"):
            average_j_channels(b, [1, 1], dij, np.array([0.5, 1.5]))
