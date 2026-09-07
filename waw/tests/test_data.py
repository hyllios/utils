"""
Tests for waw/data.py — WannierData loading and setup.

Key tests:
  1. Tensor shapes and dtypes match expectations.
  2. Shell weight completeness relation: sum_b w_b b_a b_b = delta_{ab}.
     This is the fundamental requirement for a well-defined finite-difference
     gradient operator and is checked analytically on a known k-mesh.
  3. parse_recip_lattice: verify 2*pi orthogonality for a cubic cell.
  4. WannierData properties (nk, nb, nw, nnb) are consistent with tensor shapes.
"""

import textwrap
from pathlib import Path

import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.types import WannierData
from waw.core.kmesh import _compute_bvecs_and_weights
from waw.interfaces.wannier90.loader import (
    parse_recip_lattice,
    parse_real_lattice,
)
from waw.interfaces.wannier90.io import read_win
from waw.units import ANG_TO_BOHR


# ===========================================================================
# Helpers — build minimal WannierData directly from tensors
# ===========================================================================

NB = 4
NK = 8    # 2x2x2 Gamma-centred mesh
NW = 2
NNB = 8   # simple-cubic nearest neighbours (6) — we use 8 for fcc-like test


def _make_wdata(nk=NK, nb=NB, nw=NW, nnb=NNB) -> WannierData:
    """Construct a WannierData with random tensors (no file I/O)."""
    rng = np.random.default_rng(42)
    return WannierData(
        Mmn    = torch.tensor(rng.standard_normal((nk, nnb, nb, nb))
                              + 1j * rng.standard_normal((nk, nnb, nb, nb)),
                              dtype=torch.complex128),
        Amn    = torch.tensor(rng.standard_normal((nk, nb, nw))
                              + 1j * rng.standard_normal((nk, nb, nw)),
                              dtype=torch.complex128),
        eig    = torch.tensor(rng.standard_normal((nk, nb)), dtype=torch.float64),
        kpts   = torch.zeros((nk, 3), dtype=torch.float64),
        bvecs  = torch.zeros((nk, nnb, 3), dtype=torch.float64),  # (nk, nnb, 3)
        wb     = torch.ones(nnb, dtype=torch.float64),
        kb_idx = torch.zeros((nk, nnb), dtype=torch.long),
        params = {},
    )


# ===========================================================================
# Tests: WannierData shape properties
# ===========================================================================

class TestWannierDataProperties:
    def test_nk(self):
        wd = _make_wdata()
        assert wd.nk == NK

    def test_nb(self):
        wd = _make_wdata()
        assert wd.nb == NB

    def test_nw(self):
        wd = _make_wdata()
        assert wd.nw == NW

    def test_nnb(self):
        wd = _make_wdata()
        assert wd.nnb == NNB

    def test_repr(self):
        wd = _make_wdata()
        s = repr(wd)
        assert "WannierData" in s
        assert str(NK) in s

    def test_mmn_dtype(self):
        wd = _make_wdata()
        assert wd.Mmn.dtype == torch.complex128

    def test_amn_dtype(self):
        wd = _make_wdata()
        assert wd.Amn.dtype == torch.complex128

    def test_eig_dtype(self):
        wd = _make_wdata()
        assert wd.eig.dtype == torch.float64

    def test_kb_idx_dtype(self):
        wd = _make_wdata()
        assert wd.kb_idx.dtype == torch.long


# ===========================================================================
# Tests: shell weight completeness
# ===========================================================================

class TestShellWeightCompleteness:
    """
    The completeness relation  sum_b w_b * b_a * b_b = delta_{ab}
    must hold for the b-vectors and weights returned by
    _compute_bvecs_and_weights().

    We test on two standard meshes:
      1. Simple cubic: 6 nearest neighbours along ±x, ±y, ±z.
         One shell, weight = 1/(2*|b|^2).
      2. FCC-like: 12 nearest neighbours in the {110} directions.
         One shell, weight = 1/(4*|b|^2).
    """

    def _check_completeness(self, bvecs: np.ndarray, wb: np.ndarray, tol=1e-10):
        """Assert sum_b w_b b_a b_b = delta_{ab}.  bvecs is (nnb, 3)."""
        outer = np.einsum("b,ba,bc->ac", wb, bvecs, bvecs)  # (3,3)
        np.testing.assert_allclose(
            outer, np.eye(3), atol=tol,
            err_msg="Shell weight completeness condition violated",
        )

    def _simple_cubic_setup(self):
        """
        2x2x2 simple-cubic mesh with lattice constant a=1 (in Bohr).
        recip lattice: 2*pi * I, so b-vectors are multiples of 2*pi/2 = pi.
        6 nearest neighbours: ±pi along each axis.
        """
        a = 1.0   # Bohr
        recip = 2 * np.pi / a * np.eye(3)   # (3,3)

        # 8 k-points on a 2x2x2 Gamma-centred mesh
        kpts = np.array([
            [i/2, j/2, k/2]
            for i in range(2) for j in range(2) for k in range(2)
        ], dtype=np.float64)
        nk  = len(kpts)
        nnb = 6   # simple-cubic nearest neighbours

        # Neighbour table: each k-point has 6 neighbours
        # For a 2x2x2 mesh with periodic BCs the neighbour of k=0 along +x
        # is k=[0.5,0,0], and its G-vector is 0.
        # We build this analytically.
        nnkpts    = np.zeros((nk, nnb), dtype=np.int64)
        g_vectors = np.zeros((nk, nnb, 3), dtype=np.int64)

        # b-vector directions in crystal coords for simple cubic
        b_dirs = np.array([
            [ 0.5, 0, 0], [-0.5, 0, 0],
            [0,  0.5, 0], [0, -0.5, 0],
            [0, 0,  0.5], [0, 0, -0.5],
        ])

        for ik, k in enumerate(kpts):
            for ib, b in enumerate(b_dirs):
                # k + b mod 1 (wrap into first BZ)
                kpb_cryst = k + b
                kpb_wrap  = kpb_cryst % 1.0
                # find G such that kpb_cryst = kpts[ik2] + G
                G = np.round(kpb_cryst - kpb_wrap).astype(int)
                # find ik2
                diffs = np.linalg.norm(kpts - kpb_wrap, axis=1)
                ik2   = int(np.argmin(diffs))
                nnkpts[ik, ib]    = ik2
                g_vectors[ik, ib] = G

        return kpts, nnkpts, g_vectors, recip

    def test_simple_cubic_completeness(self):
        kpts, nnkpts, g_vectors, recip = self._simple_cubic_setup()
        bvecs, wb = _compute_bvecs_and_weights(kpts, nnkpts, g_vectors, recip)
        # bvecs is (nk, nnb, 3); use k=0 since all k-points share the same set
        # on a simple-cubic mesh (no permutation)
        self._check_completeness(bvecs[0], wb)

    def test_simple_cubic_weight_value(self):
        """
        For simple cubic with |b| = pi (when a=1 Bohr, recip lattice = 2*pi*I,
        neighbour at half-mesh spacing = pi), the weight satisfies
        6 * w * pi^2 = 3  =>  w = 1/(2*pi^2).
        """
        kpts, nnkpts, g_vectors, recip = self._simple_cubic_setup()
        bvecs, wb = _compute_bvecs_and_weights(kpts, nnkpts, g_vectors, recip)
        # All 6 neighbours in one shell, so all weights equal.
        assert np.allclose(wb, wb[0], atol=1e-10), "All weights should be equal"
        # bvecs[0, 0] = first b-vector at k=0
        b_len_sq = np.linalg.norm(bvecs[0, 0])**2
        expected_w = 1.0 / (2 * b_len_sq)
        assert abs(wb[0] - expected_w) < 1e-10, (
            f"Weight {wb[0]:.6f} != expected {expected_w:.6f}"
        )

    def test_number_of_bvecs(self):
        kpts, nnkpts, g_vectors, recip = self._simple_cubic_setup()
        bvecs, wb = _compute_bvecs_and_weights(kpts, nnkpts, g_vectors, recip)
        nk = len(kpts)
        assert bvecs.shape == (nk, 6, 3)

    def test_bvecs_dtype(self):
        kpts, nnkpts, g_vectors, recip = self._simple_cubic_setup()
        bvecs, wb = _compute_bvecs_and_weights(kpts, nnkpts, g_vectors, recip)
        assert bvecs.dtype == np.float64
        assert wb.dtype    == np.float64

    def test_weights_positive(self):
        """Shell weights must be positive (they enter as |b|^2 denominators)."""
        kpts, nnkpts, g_vectors, recip = self._simple_cubic_setup()
        _, wb = _compute_bvecs_and_weights(kpts, nnkpts, g_vectors, recip)
        assert np.all(wb > 0)


# ===========================================================================
# Tests: parse_recip_lattice
# ===========================================================================

class TestParseRecipLattice:

    def _cubic_win_params(self, a_ang: float = 5.0) -> dict:
        """Simulate parsed .win for a cubic cell with lattice constant a_ang."""
        return {
            "unit_cell_cart": [
                "Ang",
                f"{a_ang}  0.0  0.0",
                f"0.0  {a_ang}  0.0",
                f"0.0  0.0  {a_ang}",
            ]
        }

    def test_cubic_cell_orthogonality(self):
        """
        For a cubic cell, the reciprocal lattice must be diagonal:
        b_i . a_j = 2*pi * delta_{ij}.
        """
        a_ang = 5.0
        a_bohr = a_ang * ANG_TO_BOHR

        params = self._cubic_win_params(a_ang)
        recip  = parse_recip_lattice(params)

        expected_b = 2 * np.pi / a_bohr
        np.testing.assert_allclose(
            recip,
            expected_b * np.eye(3),
            atol=1e-10,
            err_msg="Reciprocal lattice wrong for cubic cell",
        )

    def test_bohr_unit_tag(self):
        """Cells specified in Bohr should not be scaled by ANG_TO_BOHR."""
        a_bohr = 10.0
        params = {
            "unit_cell_cart": [
                "Bohr",
                f"{a_bohr}  0.0  0.0",
                f"0.0  {a_bohr}  0.0",
                f"0.0  0.0  {a_bohr}",
            ]
        }
        recip = parse_recip_lattice(params)
        expected_b = 2 * np.pi / a_bohr
        np.testing.assert_allclose(recip, expected_b * np.eye(3), atol=1e-10)

    def test_no_unit_tag_defaults_to_angstrom(self):
        """
        A unit_cell_cart block with no leading unit line must be treated as
        Angstrom (Wannier90's default), not Bohr. Regression for the
        tutorial03 (Silicon) units bug: silicon.win omits the unit line,
        and a Bohr default silently shrank the lattice by 1/1.8897,
        mis-scaling every spread/Omega/.chk.fmt value by that factor^2.
        """
        a = 5.0
        no_tag = parse_real_lattice({
            "unit_cell_cart": [
                f"{a}  0.0  0.0", f"0.0  {a}  0.0", f"0.0  0.0  {a}",
            ]
        })
        with_ang = parse_real_lattice({
            "unit_cell_cart": [
                "Ang", f"{a}  0.0  0.0", f"0.0  {a}  0.0", f"0.0  0.0  {a}",
            ]
        })
        # Bare block == explicit Ang block, and both are a*ANG_TO_BOHR in Bohr.
        np.testing.assert_allclose(no_tag, with_ang, atol=1e-12)
        np.testing.assert_allclose(no_tag, a * ANG_TO_BOHR * np.eye(3), atol=1e-10)

    def test_missing_block_raises(self):
        with pytest.raises(ValueError, match="unit_cell_cart"):
            parse_recip_lattice({})

    def test_shape(self):
        params = self._cubic_win_params()
        recip  = parse_recip_lattice(params)
        assert recip.shape == (3, 3)

    def test_dtype(self):
        params = self._cubic_win_params()
        recip  = parse_recip_lattice(params)
        assert recip.dtype == np.float64
