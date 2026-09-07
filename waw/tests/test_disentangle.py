"""
Tests for waw/disentangle.py — Part 6.

Key tests:
  1.  No-op when nb == nw (isolated bands).
  2.  V is semi-unitary after disentanglement.
  3.  Omega_I is non-increasing over sweeps.
  4.  Outer window: bands outside are excluded from V.
  5.  Frozen window: frozen bands are always included in the subspace.
  6.  Frozen window: non-frozen free columns are orthogonal to frozen.
  7.  Known optimal subspace: for diagonal Mmn, selects the band with
      the highest average |M_nn|.
  8.  No frozen, no outer window: converges to the same result as explicit
      all-bands outer window.
  9.  Error on too few bands in outer window.
  10. Error on too many frozen bands.
  11. Omega_I from disentangle matches rotate_overlaps + manual calculation.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.disentangle import (
    disentangle, disentangle_joint, DisentangleResult,
    _init_V_amn, _init_V_index,
)
from waw.core.spread import rotate_overlaps


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

def _random_unitary(nk: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(nk, n, n, dtype=torch.float64, generator=g)
         + 1j * torch.randn(nk, n, n, dtype=torch.float64, generator=g))
    Q, _ = torch.linalg.qr(A)
    return Q   # (nk, n, n)


def _make_entangled_data(nk=8, nb=6, nw=3, nnb=6, seed=1):
    """
    Synthetic entangled system: nb bands, nw Wannier functions.
    Mmn[k,b] is built from random unitary matrices so the Hermitian-
    conjugate symmetry is approximately satisfied per (k, b) pair.
    All k-points share the same simple-cubic wb.
    """
    torch.manual_seed(seed)
    V_full = _random_unitary(nk, nb, seed=seed)

    kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
    for ik in range(nk):
        for ib in range(nnb):
            kb_idx[ik, ib] = (ik + ib + 1) % nk

    Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
    for ik in range(nk):
        for ib in range(nnb):
            ik2 = kb_idx[ik, ib].item()
            Mmn[ik, ib] = V_full[ik].conj().T @ V_full[ik2]

    wb = torch.full((nnb,), 1.0 / (nnb * np.pi**2), dtype=torch.float64)

    # Flat eigenvalues (no physics, just for window tests)
    eig = torch.arange(nb, dtype=torch.float64).unsqueeze(0).expand(nk, -1).clone()

    return Mmn, eig, wb, kb_idx


def _omega_i_direct(V, Mmn, wb, kb_idx) -> float:
    """Reference Omega_I from rotate_overlaps."""
    nk = V.shape[0]
    nw = V.shape[2]
    M_sub = rotate_overlaps(V, Mmn, kb_idx)
    frob2 = M_sub.abs().pow(2).sum(dim=(-1, -2))
    return torch.einsum("b,kb->", wb, nw - frob2).item() / nk


# ===========================================================================
# 1. No-op for isolated bands
# ===========================================================================

class TestIsolatedBands:
    def test_returns_identity(self):
        """nb == nw → V should equal the identity at every k-point."""
        nk, nb = 4, 3
        Mmn    = torch.randn(nk, 4, nb, nb, dtype=torch.complex128)
        eig    = torch.zeros(nk, nb)
        wb     = torch.ones(4, dtype=torch.float64)
        kb_idx = torch.zeros(nk, 4, dtype=torch.long)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nb)

        eye = torch.eye(nb, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(result.V, eye.expand(nk, -1, -1), atol=1e-12, rtol=0)

    def test_marked_converged(self):
        nk, nb = 4, 3
        Mmn    = torch.randn(nk, 4, nb, nb, dtype=torch.complex128)
        eig    = torch.zeros(nk, nb)
        wb     = torch.ones(4, dtype=torch.float64)
        kb_idx = torch.zeros(nk, 4, dtype=torch.long)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nb)
        assert result.converged

    def test_result_type(self):
        nk, nb = 4, 3
        Mmn    = torch.randn(nk, 4, nb, nb, dtype=torch.complex128)
        eig    = torch.zeros(nk, nb)
        wb     = torch.ones(4, dtype=torch.float64)
        kb_idx = torch.zeros(nk, 4, dtype=torch.long)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nb)
        assert isinstance(result, DisentangleResult)


# ===========================================================================
# 2. Semi-unitarity
# ===========================================================================

class TestSemiUnitarity:
    def test_VhV_equals_identity(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        nw = 3
        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=50)
        V   = result.V   # (nk, nb, nw)
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)   # (nk, nw, nw)
        eye = torch.eye(nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_VhV_after_outer_window(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=6, nw=2)
        result = disentangle(
            Mmn, eig, wb, kb_idx, nw=2,
            outer_window=(1.0, 4.5),   # bands 1,2,3,4 (0-indexed)
            n_iter=50,
        )
        V   = result.V
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_VhV_after_frozen_window(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=6, nw=3)
        result = disentangle(
            Mmn, eig, wb, kb_idx, nw=3,
            frozen_window=(0.5, 1.5),   # band 1 frozen
            n_iter=50,
        )
        V   = result.V
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(3, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)


# ===========================================================================
# 3. Omega_I non-increasing
# ===========================================================================

class TestConvergence:
    def test_omega_i_non_increasing(self):
        """Each sweep must not increase Omega_I."""
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle(Mmn, eig, wb, kb_idx, nw=3,
                             n_iter=30, conv_tol=0.0)
        h = result.history
        for i in range(1, len(h)):
            assert h[i] <= h[i-1] + 1e-10, (
                f"Omega_I increased at sweep {i}: {h[i-1]:.6f} → {h[i]:.6f}"
            )

    def test_history_length(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle(Mmn, eig, wb, kb_idx, nw=3,
                             n_iter=20, conv_tol=0.0)
        assert 1 <= len(result.history) <= 20

    def test_omega_i_matches_direct(self):
        """omega_i field must match manual computation via rotate_overlaps."""
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle(Mmn, eig, wb, kb_idx, nw=3, n_iter=30)
        oi_direct = _omega_i_direct(result.V, Mmn, wb, kb_idx)
        assert abs(result.omega_i - oi_direct) < 1e-10


# ===========================================================================
# 4. Outer window: columns of V must lie within the window
# ===========================================================================

class TestOuterWindow:
    def test_V_columns_in_outer_window(self):
        """
        After disentanglement, V[k][:, j] must have non-zero weight only on
        bands within the outer window.  Bands outside have |V[k, i, j]| = 0.
        """
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        outer_window = (1.0, 3.5)   # bands 1, 2, 3

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                             outer_window=outer_window, n_iter=50)
        V = result.V   # (nk, nb, nw)

        nk = Mmn.shape[0]
        eout_min, eout_max = outer_window
        for ik in range(nk):
            outside = (eig[ik] < eout_min) | (eig[ik] > eout_max)
            # All columns of V[ik] must be zero on bands outside the window
            assert V[ik][outside].abs().max().item() < 1e-10, (
                f"k={ik}: V has weight outside outer window"
            )

    def test_error_on_too_few_outer_bands(self):
        """Raise ValueError if outer window contains fewer than nw bands."""
        nb, nw = 6, 4
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="Outer window too narrow"):
            disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                        outer_window=(0.5, 2.5))   # only 2 bands


# ===========================================================================
# 5-6. Frozen window
# ===========================================================================

class TestFrozenWindow:
    def test_frozen_bands_in_subspace(self):
        """
        Frozen bands must be in the span of V(k) at every k-point.
        Check: P @ e_i = e_i for each frozen band index i.
        """
        nb, nw = 6, 3
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        frozen_window = (1.5, 2.5)   # band 2 (eig = 2.0)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                             frozen_window=frozen_window, n_iter=50)
        V = result.V   # (nk, nb, nw)

        nk = Mmn.shape[0]
        froz_min, froz_max = frozen_window
        for ik in range(nk):
            frozen_idx = ((eig[ik] >= froz_min) & (eig[ik] <= froz_max)).nonzero(
                as_tuple=True
            )[0]
            P = V[ik] @ V[ik].conj().T   # (nb, nb) projection
            for idx in frozen_idx:
                e_i = torch.zeros(nb, dtype=torch.complex128)
                e_i[idx] = 1.0
                Pe_i = P @ e_i
                torch.testing.assert_close(Pe_i, e_i, atol=1e-10, rtol=0,
                                           msg=f"Frozen band {idx.item()} not in subspace at k={ik}")

    def test_frozen_columns_orthogonal_to_free(self):
        """
        The frozen columns and free columns of V(k) must be orthogonal
        (they are built to be orthonormal by construction, so V†V = I).
        """
        nb, nw = 6, 3
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                             frozen_window=(0.5, 1.5), n_iter=50)
        V   = result.V
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_error_on_too_many_frozen_bands(self):
        """Raise ValueError if frozen window gives more than nw bands."""
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="Frozen window too wide"):
            disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                        frozen_window=(-0.5, 3.5))   # 4 frozen > nw=2


# ===========================================================================
# 7. Known optimal subspace
# ===========================================================================

class TestKnownOptimum:
    def test_selects_best_band(self):
        """
        When Mmn is block-diagonal with one band having M_nn >> the other,
        disentanglement (nw=1) should select the dominant band.

        Construction: at every k, overlap of band 0 with itself across all
        neighbours is 0.99; band 1 gives only 0.01.
        """
        nk, nnb = 8, 4

        # Band 0 is "good" (high overlap), band 1 is "bad"
        Mmn = torch.zeros(nk, nnb, 2, 2, dtype=torch.complex128)
        Mmn[:, :, 0, 0] = 0.99   # band 0-0 overlap
        Mmn[:, :, 1, 1] = 0.01   # band 1-1 overlap

        eig    = torch.zeros(nk, 2)
        wb     = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=1, n_iter=50)
        V = result.V   # (nk, 2, 1)

        # The selected band must be band 0 (|V[k, 0, 0]|^2 ≈ 1)
        band0_weight = V[:, 0, 0].abs().pow(2)   # (nk,)
        assert band0_weight.min().item() > 0.99, (
            f"Min weight on band 0: {band0_weight.min():.4f} — expected > 0.99"
        )

    def test_omega_i_at_optimum(self):
        """
        For the block-diagonal Mmn above, at the optimal V (band 0 selected),
        Omega_I should be approximately nw - |M_00|^2 * nk * nnb * wb[0].
        """
        nk, nnb = 8, 4
        Mmn = torch.zeros(nk, nnb, 2, 2, dtype=torch.complex128)
        Mmn[:, :, 0, 0] = 0.9   # exact value matters for comparison
        Mmn[:, :, 1, 1] = 0.1

        eig    = torch.zeros(nk, 2)
        wb     = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)

        result = disentangle(Mmn, eig, wb, kb_idx, nw=1, n_iter=100)

        # Expected Omega_I = (1/nk) * nk * nnb * wb * (1 - |M_00|^2)
        #                  = nnb * (1/nnb) * (1 - 0.81) = 1 - 0.81 = 0.19
        expected = (1.0 - 0.9**2)   # sum_b wb * (nw - |M_00|^2) = (1 - 0.81)
        assert abs(result.omega_i - expected) < 1e-6, (
            f"omega_i={result.omega_i:.6f} != expected {expected:.6f}"
        )


# ===========================================================================
# 8. Amn-based initialization
# ===========================================================================

class TestAmnInit:
    """
    Tests for the SVD-based V initialization from trial projections.

    Controlled setup: nk=4, nb=4, nw=2.
    Bands 0 and 3 are the "true" WF bands (Amn ≈ e0, e3 in WF space).
    Bands 1 and 2 have near-zero trial projection.
    Mmn diagonal: M_00 = M_33 = 0.95 (large), M_11 = M_22 = 0.2 (small).
    Index init picks bands {0,1} (first two in energy) → high Omega_I.
    SVD   init picks bands {0,3} (high projection)    → low  Omega_I.
    """

    def setup_method(self):
        self.nk, self.nb, self.nw, self.nnb = 4, 4, 2, 4
        nk, nb, nw, nnb = self.nk, self.nb, self.nw, self.nnb

        # Mmn: diagonal, bands 0 and 3 have large M_nn, bands 1 and 2 small
        Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
        Mmn[:, :, 0, 0] = 0.95
        Mmn[:, :, 1, 1] = 0.20
        Mmn[:, :, 2, 2] = 0.20
        Mmn[:, :, 3, 3] = 0.95
        self.Mmn = Mmn

        # Amn: bands 0→WF0, band 3→WF1; bands 1,2 have near-zero projection
        A = torch.zeros(nk, nb, nw, dtype=torch.complex128)
        A[:, 0, 0] = 0.9
        A[:, 3, 1] = 0.9
        A[:, 1, 0] = 0.05
        A[:, 2, 1] = 0.05
        self.Amn = A

        self.eig    = torch.arange(nb, dtype=torch.float64).unsqueeze(0).expand(nk, -1)
        self.wb     = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)
        self.kb_idx = torch.zeros(nk, nnb, dtype=torch.long)

        outer_mask  = torch.ones(nk, nb, dtype=torch.bool)
        frozen_mask = torch.zeros(nk, nb, dtype=torch.bool)
        self.outer_mask  = outer_mask
        self.frozen_mask = frozen_mask

    def _omega_i(self, V):
        return _omega_i_direct(V, self.Mmn, self.wb, self.kb_idx)

    # ------------------------------------------------------------------
    # Semi-unitarity of both init methods
    # ------------------------------------------------------------------

    def test_index_init_semi_unitary(self):
        V   = _init_V_index(self.outer_mask, self.frozen_mask,
                            self.nk, self.nb, self.nw, torch.complex128)
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(self.nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-12, rtol=0)

    def test_amn_init_semi_unitary(self):
        V   = _init_V_amn(self.Amn, self.outer_mask, self.frozen_mask,
                          self.nk, self.nb, self.nw)
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(self.nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-12, rtol=0)

    # ------------------------------------------------------------------
    # SVD init gives strictly lower initial Omega_I
    # ------------------------------------------------------------------

    def test_amn_init_lower_initial_omega_i(self):
        """
        SVD init should give a lower Omega_I before any sweeps because it
        selects bands 0 and 3 (high M_nn) rather than bands 0 and 1 (low M_11).
        Check via disentangle n_iter=1: history[0] is the pre-sweep Omega_I.
        """
        r_idx = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                            Amn=None, n_iter=1, conv_tol=0.0)
        r_amn = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                            Amn=self.Amn, n_iter=1, conv_tol=0.0)
        assert r_amn.history[0] < r_idx.history[0], (
            f"SVD init Omega_I {r_amn.history[0]:.4f} not lower than "
            f"index init {r_idx.history[0]:.4f}"
        )

    # ------------------------------------------------------------------
    # Backward compatibility: Amn=None must give the same result as before
    # ------------------------------------------------------------------

    def test_amn_none_backward_compat(self):
        """Amn=None (default) must produce the same result as not passing Amn."""
        r1 = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                         n_iter=20, conv_tol=0.0)
        r2 = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                         Amn=None, n_iter=20, conv_tol=0.0)
        assert abs(r1.omega_i - r2.omega_i) < 1e-12

    # ------------------------------------------------------------------
    # Frozen bands preserved by SVD init
    # ------------------------------------------------------------------

    def test_amn_init_frozen_preserved(self):
        """Frozen bands must be exact unit-vector columns after SVD init."""
        nk, nb, nw = self.nk, self.nb, self.nw
        # Freeze band 0 (eig=0)
        frozen_mask = torch.zeros(nk, nb, dtype=torch.bool)
        frozen_mask[:, 0] = True
        outer_mask  = torch.ones(nk, nb, dtype=torch.bool)

        V = _init_V_amn(self.Amn, outer_mask, frozen_mask, nk, nb, nw)

        # Column 0 of V[k] must be e_0 (unit vector for band 0)
        for ik in range(nk):
            col0 = V[ik, :, 0]
            expected = torch.zeros(nb, dtype=torch.complex128)
            expected[0] = 1.0
            torch.testing.assert_close(col0, expected, atol=1e-12, rtol=0)

    # ------------------------------------------------------------------
    # Full disentangle with Amn finds the global minimum
    # ------------------------------------------------------------------

    def test_amn_init_finds_better_minimum(self):
        """
        Index init picks bands {0,1} (first by energy order) and converges
        to the local minimum Omega_I = nw - (M_00^2 + M_11^2) ≈ 1.06.
        SVD   init picks bands {0,3} (by WF projection) and converges to
        the global minimum Omega_I = nw - (M_00^2 + M_33^2) ≈ 0.20.
        The difference (~0.86) is well above any numerical noise.
        """
        r_idx = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                            Amn=None,      n_iter=200, conv_tol=0.0)
        r_amn = disentangle(self.Mmn, self.eig, self.wb, self.kb_idx, nw=2,
                            Amn=self.Amn,  n_iter=200, conv_tol=0.0)
        assert r_amn.omega_i < r_idx.omega_i - 0.5, (
            f"SVD init omega_i={r_amn.omega_i:.4f} not better than "
            f"index init omega_i={r_idx.omega_i:.4f}"
        )


# ===========================================================================
# 9. Projectability disentanglement (Wannier90 dis_windows_proj, tutorial34)
# ===========================================================================

class TestProjectability:
    def test_proj_min_excludes_low_projectability_bands(self):
        """
        Bands with projs = sum_j |Amn[i,j]|^2 < proj_min must get zero
        weight in V, even with no outer_window restricting them.
        """
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        nk = Mmn.shape[0]

        Amn = torch.zeros(nk, nb, nw, dtype=torch.complex128)
        Amn[:, 0, 0] = 0.9    # projs ~0.81
        Amn[:, 1, 1] = 0.9    # projs ~0.81
        Amn[:, 2, 0] = 0.3    # projs ~0.09
        Amn[:, 3, 1] = 0.3    # projs ~0.09
        Amn[:, 5, 0] = 0.001  # projs ~1e-6 -- below proj_min, excluded

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw, Amn=Amn,
                             proj_min=0.01, n_iter=50)
        V = result.V   # (nk, nb, nw)
        assert V[:, 5, :].abs().max().item() < 1e-10, (
            "Band with projs < proj_min still has non-zero weight in V"
        )

    def test_proj_max_freezes_high_projectability_bands(self):
        """
        Bands with projs >= proj_max must be frozen (in the span of V(k))
        even without an explicit frozen_window.
        """
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        nk = Mmn.shape[0]

        Amn = torch.zeros(nk, nb, nw, dtype=torch.complex128)
        Amn[:, 0, 0] = 0.99   # projs ~0.98 -- above proj_max, frozen
        Amn[:, 1, 1] = 0.3
        Amn[:, 2, 0] = 0.3
        Amn[:, 3, 1] = 0.3

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw, Amn=Amn,
                             proj_max=0.85, n_iter=50)
        V = result.V

        P = torch.matmul(V, V.conj().transpose(-1, -2))   # (nk, nb, nb)
        e0 = torch.zeros(nb, dtype=torch.complex128)
        e0[0] = 1.0
        for ik in range(nk):
            Pe0 = P[ik] @ e0
            torch.testing.assert_close(Pe0, e0, atol=1e-8, rtol=0,
                                       msg=f"High-projectability band 0 not frozen at k={ik}")

    def test_error_proj_without_amn(self):
        """proj_min/proj_max without Amn must raise (projectability needs Amn)."""
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="require Amn"):
            disentangle(Mmn, eig, wb, kb_idx, nw=nw, proj_min=0.01, n_iter=10)


# ===========================================================================
# 10. dis_spheres: k-space-localized disentanglement (Wannier90 tutorial20)
# ===========================================================================

def _mp_2x2x2_kpts() -> torch.Tensor:
    return torch.tensor(
        [[i / 2, j / 2, k / 2] for i in range(2) for j in range(2) for k in range(2)],
        dtype=torch.float64,
    )


class TestDisSpheres:
    def test_outside_sphere_uses_fixed_band_window(self):
        """
        At a k-point outside every sphere, V must select EXACTLY the fixed
        contiguous [first_wann, first_wann+nw) band window, regardless of
        outer_window/energetics.
        """
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        kpts = _mp_2x2x2_kpts()
        real_lattice = np.eye(3) * 5.0

        # Sphere only around Gamma (k=0); every other k-point is "outside".
        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=50,
                             kpts=kpts, real_lattice=real_lattice,
                             dis_spheres=[(0.0, 0.0, 0.0, 0.1)],
                             dis_spheres_first_wann=1)
        V = result.V

        outside_idx = [ik for ik in range(8) if kpts[ik].abs().sum() > 1e-12]
        for ik in outside_idx:
            nonzero_rows = V[ik].abs().sum(dim=1).nonzero(as_tuple=True)[0].tolist()
            assert nonzero_rows == [1, 2], (
                f"k={ik} (outside every sphere): expected fixed band window "
                f"[1,2] (first_wann=1, nw=2), got {nonzero_rows}"
            )

    def test_inside_sphere_disentangles_normally(self):
        """At Gamma (inside its own sphere), ordinary disentanglement runs
        (V is not restricted to the fixed window)."""
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        kpts = _mp_2x2x2_kpts()
        real_lattice = np.eye(3) * 5.0

        result = disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=50,
                             kpts=kpts, real_lattice=real_lattice,
                             dis_spheres=[(0.0, 0.0, 0.0, 0.1)],
                             dis_spheres_first_wann=1)
        V = result.V
        gamma_idx = 0   # kpts[0] == (0,0,0)
        nonzero_rows = V[gamma_idx].abs().sum(dim=1).nonzero(as_tuple=True)[0].tolist()
        assert len(nonzero_rows) > nw, (
            "Gamma is inside its own sphere -- expected ordinary "
            "disentanglement (weight spread beyond the fixed 2-band window)"
        )

    def test_error_dis_spheres_without_kpts_or_lattice(self):
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="dis_spheres requires"):
            disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=10,
                        dis_spheres=[(0.0, 0.0, 0.0, 0.1)])

    def test_no_spheres_matches_unset(self):
        """dis_spheres=None (default) must be unaffected -- no behavior change."""
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        kpts = _mp_2x2x2_kpts()
        real_lattice = np.eye(3) * 5.0

        r1 = disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=50)
        r2 = disentangle(Mmn, eig, wb, kb_idx, nw=nw, n_iter=50,
                         kpts=kpts, real_lattice=real_lattice, dis_spheres=None)
        assert abs(r1.omega_i - r2.omega_i) < 1e-12


# ===========================================================================
# 12. Joint (whole-mesh) Riemannian CG solver -- alternative to the per-k
#     Z-matrix coordinate descent above (`disentangle_joint`).
# ===========================================================================

class TestJointSolver:
    def test_matches_known_optimum(self):
        """Same block-diagonal setup as TestKnownOptimum: dominant band 0
        must still be selected (|V[k,0,0]|^2 > 0.99)."""
        nk, nnb = 8, 4
        Mmn = torch.zeros(nk, nnb, 2, 2, dtype=torch.complex128)
        Mmn[:, :, 0, 0] = 0.99
        Mmn[:, :, 1, 1] = 0.01
        eig    = torch.zeros(nk, 2)
        wb     = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)

        result = disentangle_joint(Mmn, eig, wb, kb_idx, nw=1, n_iter=100)
        band0_weight = result.V[:, 0, 0].abs().pow(2)
        assert band0_weight.min().item() > 0.99, (
            f"Min weight on band 0: {band0_weight.min():.4f} — expected > 0.99"
        )

    def test_VhV_equals_identity_no_window(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle_joint(Mmn, eig, wb, kb_idx, nw=3, n_iter=100)
        V   = result.V
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(3, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_VhV_equals_identity_with_frozen_window(self):
        """The harder case: exercises the free-block Gram-Schmidt
        re-orthonormalization against a nontrivial frozen column."""
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=6, nw=3)
        result = disentangle_joint(
            Mmn, eig, wb, kb_idx, nw=3,
            frozen_window=(0.5, 1.5), n_iter=200,
        )
        V   = result.V
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(3, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_frozen_columns_exactly_preserved(self):
        """
        Frozen columns must stay EXACTLY the fixed unit vector, not just
        approximately -- the specific property `_retract_disentangle`'s
        explicit hard-reset/Gram-Schmidt correction exists to guarantee
        (a plain QR retraction does not, in general, preserve an
        already-unit-norm column unchanged; an earlier, rejected version of
        this solver used `torch.linalg.qr` for the free-block
        re-orthonormalization and silently corrupted the free columns
        instead -- see `_gram_schmidt_masked`'s docstring).
        """
        nb, nw = 6, 3
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        frozen_window = (0.5, 1.5)   # band 1 (eig=1.0) frozen at every k
        result = disentangle_joint(
            Mmn, eig, wb, kb_idx, nw=nw,
            frozen_window=frozen_window, n_iter=200,
        )
        V = result.V
        expected = torch.zeros(nb, dtype=torch.complex128)
        expected[1] = 1.0
        for ik in range(Mmn.shape[0]):
            torch.testing.assert_close(V[ik, :, 0], expected, atol=0.0, rtol=0.0)

    def test_frozen_bands_in_subspace_variable_count(self):
        """
        Harder than a single fixed frozen band: a frozen window giving a
        DIFFERENT number of frozen bands at different k-points (0, 1, or 2),
        checking every frozen band is exactly represented in V's span at
        every k regardless of that k's own frozen count.
        """
        nk, nb, nw, nnb = 12, 6, 3, 6
        torch.manual_seed(7)
        V_full = _random_unitary(nk, nb, seed=7)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        for ik in range(nk):
            for ib in range(nnb):
                kb_idx[ik, ib] = (ik + ib + 1) % nk
        Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
        for ik in range(nk):
            for ib in range(nnb):
                ik2 = kb_idx[ik, ib].item()
                Mmn[ik, ib] = V_full[ik].conj().T @ V_full[ik2]
        wb = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)

        # Per-k varying eigenvalues so different k have 0, 1, or 2+ bands in
        # (0.5, 2.5): band 0 fixed at 0.0 (never frozen); bands 1..5 shifted
        # per k so the frozen count genuinely varies. k=0 is forced entirely
        # above the window (guarantees a genuine n_froz=0 case, not left to
        # chance) -- the rest are random for variety.
        torch.manual_seed(8)
        eig = torch.zeros(nk, nb, dtype=torch.float64)
        eig[:, 0] = 0.0
        eig[:, 1:] = torch.rand(nk, nb - 1, dtype=torch.float64) * 6.0
        eig[0, 1:] = torch.rand(nb - 1, dtype=torch.float64) * 3.0 + 3.0   # all > 2.5

        frozen_window = (0.5, 2.5)
        frozen_mask = (eig >= frozen_window[0]) & (eig <= frozen_window[1])
        n_froz = frozen_mask.sum(dim=1)
        assert n_froz.max().item() >= 2 and n_froz.min().item() == 0, (
            "test setup must actually exercise a varying frozen count"
        )

        result = disentangle_joint(
            Mmn, eig, wb, kb_idx, nw=nw,
            frozen_window=frozen_window, n_iter=300,
        )
        V = result.V
        for ik in range(nk):
            frozen_idx = frozen_mask[ik].nonzero(as_tuple=True)[0]
            P = V[ik] @ V[ik].conj().T
            for idx in frozen_idx:
                e_i = torch.zeros(nb, dtype=torch.complex128)
                e_i[idx] = 1.0
                torch.testing.assert_close(
                    P @ e_i, e_i, atol=1e-10, rtol=0,
                    msg=f"Frozen band {idx.item()} not in subspace at k={ik}",
                )
        VhV = torch.matmul(V.conj().transpose(-1, -2), V)
        eye = torch.eye(nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(VhV, eye.expand_as(VhV), atol=1e-10, rtol=0)

    def test_outer_window_excludes_bands_exactly(self):
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        outer_window = (1.0, 3.5)   # bands 1, 2, 3
        result = disentangle_joint(
            Mmn, eig, wb, kb_idx, nw=nw,
            outer_window=outer_window, n_iter=100,
        )
        V = result.V
        eout_min, eout_max = outer_window
        for ik in range(Mmn.shape[0]):
            outside = (eig[ik] < eout_min) | (eig[ik] > eout_max)
            assert V[ik][outside].abs().max().item() < 1e-10, (
                f"k={ik}: V has weight outside outer window"
            )

    def test_omega_i_non_increasing(self):
        """Same property as the Z-matrix solver's history, but for the CG
        trajectory (each accepted step's line search only takes a step that
        does not increase Omega_I)."""
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle_joint(Mmn, eig, wb, kb_idx, nw=3,
                                   n_iter=50, conv_tol=0.0)
        h = result.history
        for i in range(1, len(h)):
            assert h[i] <= h[i - 1] + 1e-8, (
                f"Omega_I increased at sweep {i}: {h[i-1]:.6f} → {h[i]:.6f}"
            )

    def test_omega_i_matches_direct(self):
        Mmn, eig, wb, kb_idx = _make_entangled_data()
        result = disentangle_joint(Mmn, eig, wb, kb_idx, nw=3, n_iter=50)
        oi_direct = _omega_i_direct(result.V, Mmn, wb, kb_idx)
        assert abs(result.omega_i - oi_direct) < 1e-10

    def test_isolated_bands_returns_identity(self):
        nk, nb = 4, 3
        Mmn    = torch.randn(nk, 4, nb, nb, dtype=torch.complex128)
        eig    = torch.zeros(nk, nb)
        wb     = torch.ones(4, dtype=torch.float64)
        kb_idx = torch.zeros(nk, 4, dtype=torch.long)

        result = disentangle_joint(Mmn, eig, wb, kb_idx, nw=nb)
        eye = torch.eye(nb, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(result.V, eye.expand(nk, -1, -1), atol=1e-12, rtol=0)
        assert result.converged

    def test_error_on_too_few_outer_bands(self):
        nb, nw = 6, 4
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="Outer window too narrow"):
            disentangle_joint(Mmn, eig, wb, kb_idx, nw=nw,
                              outer_window=(0.5, 2.5))

    def test_error_on_too_many_frozen_bands(self):
        nb, nw = 6, 2
        Mmn, eig, wb, kb_idx = _make_entangled_data(nb=nb, nw=nw)
        with pytest.raises(ValueError, match="Frozen window too wide"):
            disentangle_joint(Mmn, eig, wb, kb_idx, nw=nw,
                              frozen_window=(-0.5, 3.5))

    def test_reaches_comparable_or_better_omega_i_than_zmatrix(self):
        """
        Not required to beat the Z-matrix solver on every case, but on this
        harder (variable frozen count) synthetic case it should land within
        a reasonable margin -- guards against a silent regression making the
        joint solver systematically worse.
        """
        nk, nb, nw, nnb = 27, 10, 4, 6
        V_full = _random_unitary(nk, nb, seed=3)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        for ik in range(nk):
            for ib in range(nnb):
                kb_idx[ik, ib] = (ik + ib + 1) % nk
        Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
        for ik in range(nk):
            for ib in range(nnb):
                ik2 = kb_idx[ik, ib].item()
                Mmn[ik, ib] = V_full[ik].conj().T @ V_full[ik2]
        wb = torch.full((nnb,), 1.0 / nnb, dtype=torch.float64)
        torch.manual_seed(103)
        eig = torch.rand(nk, nb, dtype=torch.float64).sort(dim=1).values * 10

        res_old  = disentangle(Mmn, eig, wb, kb_idx, nw=nw,
                               frozen_window=(4.5, 5.0), n_iter=500, conv_tol=1e-12)
        res_new  = disentangle_joint(Mmn, eig, wb, kb_idx, nw=nw,
                                     frozen_window=(4.5, 5.0), n_iter=500, conv_tol=1e-12)
        assert res_new.omega_i < res_old.omega_i * 1.2, (
            f"joint solver omega_i={res_new.omega_i:.4f} is more than 20% worse "
            f"than the Z-matrix solver's {res_old.omega_i:.4f}"
        )
