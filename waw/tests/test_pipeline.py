"""
Tests for waw/pipeline.py — unified Wannierization pipeline.

  1. Returns WannierResult.
  2. nw inferred from .win num_wann when not supplied.
  3. Omega matches W90 reference on Si (isolated bands).
  4. H(R) round-trip: interpolated bands at original k-points reproduce DFT eig.
  5. per-WF spreads sum to total Omega.
  6. Output files created when write_outputs=True.
  7. No output files when write_outputs=False.
  8. WannierResult fields have correct shapes.
  9. dis is None for isolated bands.
 10. Entangled path: dis is not None when nb > nw (synthetic data).
 11. Top-level import: `from waw import wannierize` works.
 12. parse_real_lattice returns (3,3) array in Bohr.
 13. .chk.fmt content: isolated-bands (Si) and entangled (synthetic) cases.
"""

from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import waw
from waw import wannierize, WannierResult
from waw.interfaces.wannier90.loader import parse_real_lattice
from waw.interfaces.wannier90.io   import read_win, read_chk_fmt
from waw.units import BOHR_TO_ANG

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.mmn").exists()

W90_OMEGA    = 6.468598306   # Ang^2


# ===========================================================================
# Helpers
# ===========================================================================

@pytest.fixture(scope="module")
def si_result(tmp_path_factory):
    """Run wannierize on Si once; reuse across tests in this module."""
    if not HAS_SI:
        pytest.skip("Silicon reference data not found")
    tmp = tmp_path_factory.mktemp("si_out")
    # Copy input files to a temp directory so output files land there
    for suffix in (".win", ".nnkp", ".eig", ".mmn", ".amn"):
        shutil.copy(SI_DIR / f"silicon{suffix}", tmp / f"silicon{suffix}")
    result = wannierize(
        tmp / "silicon",
        n_iter=1000,
        write_outputs=True,
        verbose=False,
    )
    return result, tmp


# ===========================================================================
# 1. Return type
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_returns_wannier_result(si_result):
    result, _ = si_result
    assert isinstance(result, WannierResult)


# ===========================================================================
# 2. nw inferred from .win
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_nw_from_params(si_result):
    result, _ = si_result
    assert result.spread.U_final.shape[-1] == 4   # num_wann=4 from silicon.win


# ===========================================================================
# 3. Omega matches W90
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_si_omega_matches_w90(si_result):
    result, _ = si_result
    omega_ang2 = result.spread.Omega * BOHR_TO_ANG**2
    assert abs(omega_ang2 - W90_OMEGA) < 1e-3, (
        f"Omega={omega_ang2:.6f} Ang², expected {W90_OMEGA:.6f}"
    )


# ===========================================================================
# 4. H(R) round-trip
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_hr_round_trip(si_result):
    """Interpolated bands at original k-points reproduce DFT eigenvalues."""
    result, _ = si_result
    from waw import interpolate_bands
    kpts_np = result.wdata.kpts.numpy()
    bands   = interpolate_bands(result.hr, kpts_np)         # (nk, nw)
    eig_ref = np.sort(result.wdata.eig.numpy()[:, :4], axis=1)
    max_err = np.abs(bands - eig_ref).max()
    assert max_err < 0.01, f"Max band interpolation error: {max_err*1000:.2f} meV"


# ===========================================================================
# 5. Per-WF spreads sum to total Omega
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_per_wf_spreads_sum_to_omega(si_result):
    result, _ = si_result
    total_from_wf = result.spreads_bohr2.sum()
    assert abs(total_from_wf - result.omega_final) < 1e-10, (
        f"Σ σ_n² = {total_from_wf:.8f}, omega_final = {result.omega_final:.8f}"
    )


# ===========================================================================
# 6. Output files created
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_output_files_created(si_result):
    result, tmp = si_result
    assert (tmp / "silicon_hr.dat").exists(),      "_hr.dat not written"
    assert (tmp / "silicon_centres.xyz").exists(), "_centres.xyz not written"
    assert (tmp / "silicon.chk.fmt").exists(),     ".chk.fmt not written"


# ===========================================================================
# 7. No output when write_outputs=False
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_no_output_when_disabled(tmp_path):
    for suffix in (".win", ".nnkp", ".eig", ".mmn", ".amn"):
        shutil.copy(SI_DIR / f"silicon{suffix}", tmp_path / f"silicon{suffix}")
    wannierize(tmp_path / "silicon", n_iter=10, write_outputs=False, verbose=False)
    assert not (tmp_path / "silicon_hr.dat").exists()
    assert not (tmp_path / "silicon_centres.xyz").exists()
    assert not (tmp_path / "silicon.chk.fmt").exists()


# ===========================================================================
# 8. Shapes
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_shapes(si_result):
    result, _ = si_result
    nk, nw = 64, 4
    assert result.spread.U_final.shape == (nk, nw, nw)
    assert result.spread.centres.shape == (nw, 3)
    assert result.spreads_bohr2.shape  == (nw,)
    assert result.hr.H_R.shape[1:]    == (nw, nw)
    assert result.hr.R_vectors.shape[1] == 3


# ===========================================================================
# 9. dis is None for isolated bands
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_dis_none_for_isolated(si_result):
    result, _ = si_result
    assert result.dis is None


# ===========================================================================
# 10. Entangled path: dis is not None (synthetic)
# ===========================================================================

class TestEntangledPath:
    """
    Synthetic test: create fake input files with nb > nw and verify that
    wannierize runs the disentanglement branch (dis is not None).
    Uses a 2×2×2 cubic grid, nb=4 bands, nw=2 Wannier functions.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        self._write_synthetic_inputs(tmp_path)

    def _write_synthetic_inputs(self, d: Path):
        nk, nb, nw, nnb = 8, 4, 2, 6
        rng = np.random.default_rng(0)

        # k-points: 2×2×2 grid in [0,1)
        kpts = np.array([[i/2, j/2, k/2]
                         for i in range(2) for j in range(2) for k in range(2)])

        # Build nnkpts + G-vectors with proper ±½ steps in each crystal direction.
        # For N=2, +x and -x land on the SAME k-point but with different G-vectors,
        # which produces distinct b-vectors: b = (½,0,0) and (-½,0,0).
        nnkpts    = np.zeros((nk, nnb), dtype=np.int64)
        g_vectors = np.zeros((nk, nnb, 3), dtype=np.int64)
        steps = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        for ik in range(nk):
            ki = kpts[ik]
            for ib, step_dir in enumerate(steps):
                step = np.array(step_dir, dtype=np.float64) / 2   # ±½ in crystal
                kj_raw = ki + step
                kj     = kj_raw % 1.0                             # fold to [0,1)
                G      = np.round(kj_raw - kj).astype(int)
                # find ik2 by closest match
                diffs  = np.linalg.norm(kpts - kj, axis=1)
                ik2    = int(np.argmin(diffs))
                nnkpts[ik, ib]    = ik2
                g_vectors[ik, ib] = G

        # .win
        (d / "test.win").write_text(
            "num_wann = 2\nnum_bands = 4\nmp_grid = 2 2 2\n"
            "dis_win_max = 10.0\n"
            "begin unit_cell_cart\nBohr\n"
            " 5.0  0.0  0.0\n 0.0  5.0  0.0\n 0.0  0.0  5.0\n"
            "end unit_cell_cart\n"
            "begin atoms_frac\nSi 0.0 0.0 0.0\nend atoms_frac\n"
        )

        # .nnkp
        lines = ["written by test\n", f"begin kpoints\n{nk}\n"]
        for k in kpts:
            lines.append(f" {k[0]:.6f} {k[1]:.6f} {k[2]:.6f}\n")
        lines.append(f"end kpoints\nbegin nnkpts\n{nnb}\n")
        for ik in range(nk):
            for ib in range(nnb):
                ik2 = int(nnkpts[ik, ib])
                lines.append(f" {ik+1} {ik2+1}  0  0  0\n")
        lines.append("end nnkpts\n")
        (d / "test.nnkp").write_text("".join(lines))

        # .eig
        eig_lines = []
        for ib in range(nb):
            for ik in range(nk):
                e = -3.0 + ib * 2.0 + rng.uniform(-0.5, 0.5)
                eig_lines.append(f" {ib+1}  {ik+1}  {e:.8f}\n")
        (d / "test.eig").write_text("".join(eig_lines))

        # .mmn — random complex overlaps close to identity
        mmn_lines = [" written by test\n", f" {nb} {nk} {nnb}\n"]
        for ik in range(nk):
            for ib in range(nnb):
                ik2 = int(nnkpts[ik, ib])
                mmn_lines.append(f" {ik+1} {ik2+1}  0  0  0\n")
                M = np.eye(nb, dtype=np.complex128) + 0.05 * (
                    rng.standard_normal((nb, nb))
                    + 1j * rng.standard_normal((nb, nb))
                )
                for n in range(nb):
                    for m in range(nb):
                        mmn_lines.append(
                            f"  {M[m, n].real:.10f}  {M[m, n].imag:.10f}\n"
                        )
        (d / "test.mmn").write_text("".join(mmn_lines))

        # .amn — random projections
        amn_lines = [" written by test\n", f" {nb} {nk} {nw}\n"]
        for m in range(nb):
            for n in range(nw):
                for ik in range(nk):
                    v = rng.standard_normal() + 1j * rng.standard_normal()
                    amn_lines.append(
                        f" {m+1}  {n+1}  {ik+1}  {v.real:.10f}  {v.imag:.10f}\n"
                    )
        (d / "test.amn").write_text("".join(amn_lines))

    def test_entangled_dis_not_none(self):
        result = wannierize(
            self.tmp / "test", nw=2,
            n_iter=20, write_outputs=False, verbose=False,
        )
        assert result.dis is not None, "dis should be set for the entangled case"

    def test_entangled_u_shape(self):
        result = wannierize(
            self.tmp / "test", nw=2,
            n_iter=20, write_outputs=False, verbose=False,
        )
        assert result.spread.U_final.shape == (8, 2, 2)

    def test_entangled_spreads_sum(self):
        result = wannierize(
            self.tmp / "test", nw=2,
            n_iter=20, write_outputs=False, verbose=False,
        )
        total_from_wf = result.spreads_bohr2.sum()
        assert abs(total_from_wf - result.omega_final) < 1e-10, (
            f"Σ σ_n² = {total_from_wf}, omega_final = {result.omega_final}"
        )


# ===========================================================================
# 11. Top-level import
# ===========================================================================

def test_toplevel_import():
    assert hasattr(waw, "wannierize")
    assert hasattr(waw, "WannierResult")
    assert hasattr(waw, "interpolate_bands")
    assert hasattr(waw, "disentangle")


# ===========================================================================
# 12. parse_real_lattice
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_parse_real_lattice_shape():
    params = read_win(SI_DIR / "silicon.win")
    lat = parse_real_lattice(params)
    assert lat.shape == (3, 3)
    assert lat.dtype == np.float64
    # lattice vectors should have magnitude on the order of Bohr (not Angstrom)
    norms = np.linalg.norm(lat, axis=1)
    assert all(3.0 < n < 30.0 for n in norms), f"Unexpected lattice vector norms: {norms}"


# ===========================================================================
# 13. .chk.fmt content
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_chk_isolated_bands(si_result):
    """Isolated-bands case (Si): no disentanglement, u_matrix_opt/lwindow absent."""
    result, tmp = si_result
    chk = read_chk_fmt(tmp / "silicon.chk.fmt")

    nk, nb, nw = 64, 4, 4
    assert chk["num_bands"] == nb
    assert chk["num_wann"]  == nw
    assert chk["have_disentangled"] is False
    assert chk["checkpoint"] == "postwann"
    assert chk["u_matrix_opt"] is None
    assert chk["lwindow"] is None
    assert chk["u_matrix"].shape == (nk, nw, nw)
    np.testing.assert_allclose(
        chk["u_matrix"], result.spread.U_final.numpy(), rtol=1e-10,
    )
    # .chk.fmt is in Angstrom/Angstrom^2 throughout (confirmed against a
    # real wannier90.x + w90chk2chk.x -export run), not the Bohr/Bohr^2
    # used internally by result.spread/spreads_bohr2.
    np.testing.assert_allclose(
        chk["wannier_centres"], result.spread.centres.numpy() * BOHR_TO_ANG, rtol=1e-10,
    )
    np.testing.assert_allclose(
        chk["wannier_spreads"], result.spreads_bohr2 * BOHR_TO_ANG**2, rtol=1e-10,
    )
    np.testing.assert_allclose(
        chk["real_lattice"],
        parse_real_lattice(read_win(tmp / "silicon.win")) * BOHR_TO_ANG,
        rtol=1e-10,
    )


class TestChkEntangled:
    """Entangled case (synthetic, reusing TestEntangledPath's fixtures)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp = tmp_path
        TestEntangledPath._write_synthetic_inputs(self, tmp_path)

    def test_chk_entangled_content(self):
        result = wannierize(
            self.tmp / "test", nw=2,
            n_iter=20, write_outputs=True, verbose=False,
        )
        chk = read_chk_fmt(self.tmp / "test.chk.fmt")

        nk, nb, nw = 8, 4, 2
        assert chk["have_disentangled"] is True
        assert chk["u_matrix_opt"].shape == (nk, nb, nw)
        assert chk["lwindow"].shape == (nk, nb)
        assert chk["ndimwin"].shape == (nk,)
        # No outer/frozen window was set -> every band is in the outer window.
        assert chk["lwindow"].all()
        np.testing.assert_array_equal(chk["ndimwin"], nb)
        np.testing.assert_allclose(
            chk["u_matrix_opt"], result.dis.V.numpy(), rtol=1e-10,
        )
        np.testing.assert_allclose(
            chk["omega_invariant"], result.dis.omega_i * BOHR_TO_ANG**2, rtol=1e-10,
        )
        np.testing.assert_allclose(
            chk["u_matrix"], result.spread.U_final.numpy(), rtol=1e-10,
        )
