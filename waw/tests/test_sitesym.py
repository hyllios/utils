"""
Tests for waw/core/sitesym.py — symmetry-adapted Wannier functions (tutorial21).

Strategy: hand-construct a small, genuine point group (inversion) acting on
a 4-point 1-D k-mesh, build Z-matrices/gradients that are covariant under it
BY CONSTRUCTION, and check that the symmetrization/broadcast/reduction
primitives reproduce the exact algebraic relations that covariance implies
-- not just "doesn't crash". A naive idempotency check without first
imposing stabilizer-invariance on the input at self-mapped k-points (k=0,
k=0.5 here) is WRONG (caught a real bug this way during development, see
TestExtractSymmetrizedSubspace): symmetrize_zmatrix's job at a stabilized
k IS to project onto the stabilizer-invariant subspace, so only an input
that is already invariant there should come back unchanged.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.sitesym import (
    SiteSymmetry, broadcast_matrix, symmetrize_zmatrix, symmetrize_u_irr,
    extract_symmetrized_subspace, reduce_gradient_to_irr,
)
from waw.core.disentangle import disentangle
from waw.core.optim import minimize_spread_symmetrized
from waw.core.spread import rotate_overlaps


# ===========================================================================
# Fixture: inversion symmetry on a 4-point 1-D mesh
# ===========================================================================

def _inversion_sitesym(nb=2, nw=1, seed=0):
    """
    k = 0, 0.25, 0.5, 0.75. Inversion: 0->0, .25<->.75, .5->.5.
    Irreducible reps: k=0 (idx0), k=0.25 (idx1), k=0.5 (idx2); idx3=k=0.75
    is the image of idx1. d_matrix_band's stabilizer elements (ir=0, ir=2)
    are genuine order-2 involutions (D^2=I); ir=1 (no nontrivial stabilizer)
    uses an arbitrary unitary.
    """
    rng = np.random.default_rng(seed)
    nk, nsym, nkptirr = 4, 2, 3
    ir2ik = torch.tensor([0, 1, 2])
    kptsym = torch.tensor([[0, 1, 2],
                           [0, 3, 2]])

    Dband = torch.zeros(nb, nb, nsym, nkptirr, dtype=torch.complex128)
    Dband[:, :, 0, :] = torch.eye(nb, dtype=torch.complex128).unsqueeze(-1)
    D0 = torch.diag(torch.tensor([1.0, -1.0] + [1.0] * (nb - 2), dtype=torch.complex128))
    D2 = torch.eye(nb, dtype=torch.complex128)
    if nb >= 2:
        D2[[0, 1]] = D2[[1, 0]]   # swap first two rows: an order-2 permutation
    Dband[:, :, 1, 0] = D0
    Dband[:, :, 1, 2] = D2
    A = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
    Q, _ = np.linalg.qr(A)
    Dband[:, :, 1, 1] = torch.tensor(Q, dtype=torch.complex128)

    Dwann = torch.ones(nw, nw, nsym, nkptirr, dtype=torch.complex128)

    sitesym = SiteSymmetry(
        nsymmetry=nsym, nkptirr=nkptirr, num_kpts=nk, num_bands=nb, num_wann=nw,
        ik2ir=torch.tensor([0, 1, 2, 1]), ir2ik=ir2ik, kptsym=kptsym,
        d_matrix_wann=Dwann, d_matrix_band=Dband,
    )
    return sitesym, D0, D2


def _covariant_from_irr(M_irr, D, sitesym, nk):
    """
    Build a full-mesh tensor from irreducible values via M(Rk)=D M(k) D^dagger.
    Guards against re-deriving a SELF-mapped k (kptsym[1,ir]==ir2ik[ir], e.g.
    Gamma or any other high-symmetry fixed point): re-applying the "image"
    formula there would silently overwrite the identity assignment with a
    D-conjugated copy, which only equals the original when M_irr[ir] is
    already stabilizer-invariant -- true for the dedicated invariant-input
    tests, false for a generic random M_irr (a real bug caught here during
    development: TestBroadcastMatrix's random-input test failed until this
    guard was added).
    """
    shape = (nk,) + tuple(M_irr.shape[1:])
    M_full = torch.zeros(shape, dtype=M_irr.dtype)
    for ir in range(sitesym.nkptirr):
        ik = int(sitesym.ir2ik[ir])
        M_full[ik] = M_irr[ir]
        irk = int(sitesym.kptsym[1, ir])
        if irk == ik:
            continue
        Dr = D[:, :, 1, ir]
        M_full[irk] = Dr @ M_irr[ir] @ Dr.conj().transpose(-1, -2)
    return M_full


# ===========================================================================
# symmetrize_zmatrix
# ===========================================================================

class TestSymmetrizeZmatrix:
    def test_idempotent_on_stabilizer_invariant_input(self):
        """
        A Z already invariant under its own stabilizer at self-mapped
        k-points (ir=0, ir=2) must come back EXACTLY unchanged; a Z at a
        k with no nontrivial stabilizer (ir=1) comes back scaled by the
        (here, orbit-size=2) redundant accumulation -- eigenvectors are
        unaffected by a positive scalar, which is all that matters for
        disentanglement, but the exact ratio is checked here as a strong
        regression pin on the algorithm's bookkeeping.
        """
        sitesym, D0, D2 = _inversion_sitesym()
        Z0 = torch.diag(torch.tensor([2.3, -1.1], dtype=torch.complex128))
        Z2 = torch.tensor([[1.7, 0.4], [0.4, 1.7]], dtype=torch.complex128)
        rng = np.random.default_rng(1)
        A1 = torch.tensor(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)),
                          dtype=torch.complex128)
        Z1 = 0.5 * (A1 + A1.conj().T)

        assert torch.allclose(D0.conj().T @ Z0 @ D0, Z0, atol=1e-10)
        assert torch.allclose(D2.conj().T @ Z2 @ D2, Z2, atol=1e-10)

        Z_irr_target = torch.stack([Z0, Z1, Z2])
        Z_full = _covariant_from_irr(Z_irr_target, sitesym.d_matrix_band, sitesym, sitesym.num_kpts)

        Z_irr_sym = symmetrize_zmatrix(Z_full, sitesym)
        torch.testing.assert_close(Z_irr_sym[0], Z_irr_target[0], atol=1e-10, rtol=0)
        torch.testing.assert_close(Z_irr_sym[2], Z_irr_target[2], atol=1e-10, rtol=0)
        torch.testing.assert_close(Z_irr_sym[1], 2.0 * Z_irr_target[1], atol=1e-10, rtol=0)


# ===========================================================================
# broadcast_matrix
# ===========================================================================

class TestBroadcastMatrix:
    def test_round_trips_a_covariant_field(self):
        sitesym, D0, D2 = _inversion_sitesym()
        rng = np.random.default_rng(2)
        Z_irr = torch.tensor(
            rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2)), dtype=torch.complex128,
        )
        Z_full = _covariant_from_irr(Z_irr, sitesym.d_matrix_band, sitesym, sitesym.num_kpts)
        Z_full_bc = broadcast_matrix(Z_irr, sitesym.d_matrix_band, sitesym.d_matrix_band,
                                     sitesym.kptsym, sitesym.ir2ik, sitesym.num_kpts)
        torch.testing.assert_close(Z_full_bc, Z_full, atol=1e-10, rtol=0)

    def test_error_on_unreached_kpoint(self):
        sitesym, _, _ = _inversion_sitesym()
        bad_kptsym = sitesym.kptsym.clone()
        bad_kptsym[1, 1] = 1   # idx3 (k=0.75) is no longer reached by anything
        rng = np.random.default_rng(3)
        Z_irr = torch.tensor(
            rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2)), dtype=torch.complex128,
        )
        with pytest.raises(ValueError, match="not reached"):
            broadcast_matrix(Z_irr, sitesym.d_matrix_band, sitesym.d_matrix_band,
                             bad_kptsym, sitesym.ir2ik, sitesym.num_kpts)


# ===========================================================================
# reduce_gradient_to_irr
# ===========================================================================

class TestReduceGradientToIrr:
    def test_covariant_gradient_scales_by_orbit_size(self):
        sitesym, _, _ = _inversion_sitesym(nw=1)
        G_irr_target = torch.tensor([[[0.7]], [[1.3 + 0.2j]], [[-0.9]]], dtype=torch.complex128)
        G_full = _covariant_from_irr(G_irr_target, sitesym.d_matrix_wann, sitesym, sitesym.num_kpts)

        G_irr = reduce_gradient_to_irr(G_full, sitesym)
        expected_orbit_sizes = [1, 2, 1]   # ir=0: {0}; ir=1: {1,3}; ir=2: {2}
        for ir, orbit_size in enumerate(expected_orbit_sizes):
            torch.testing.assert_close(
                G_irr[ir], orbit_size * G_irr_target[ir], atol=1e-10, rtol=0,
            )


# ===========================================================================
# symmetrize_u_irr / extract_symmetrized_subspace
# ===========================================================================

class TestExtractSymmetrizedSubspace:
    def test_top_eigenvector_is_stabilizer_eigenvector(self):
        """
        Extracting nw=1 from a diagonal, stabilizer-invariant Z at ir=0
        must give exactly the corresponding unit vector (an eigenvector
        of the stabilizer's own D0), not some mixed combination.
        """
        sitesym, D0, _ = _inversion_sitesym()
        Z0 = torch.diag(torch.tensor([2.3, -1.1], dtype=torch.complex128))
        V0 = extract_symmetrized_subspace(Z0, sitesym, nw=1, ir=0)
        expected = torch.tensor([[1.0], [0.0]], dtype=torch.complex128)
        torch.testing.assert_close(V0.abs(), expected.abs(), atol=1e-8, rtol=0)
        # must be an eigenvector of the stabilizer element D0
        torch.testing.assert_close(D0 @ V0, V0, atol=1e-8, rtol=0)

    def test_symmetrize_u_irr_idempotent_on_invariant_input(self):
        sitesym, D0, _ = _inversion_sitesym()
        U = torch.tensor([[1.0], [0.0]], dtype=torch.complex128)   # D0-eigenvector, eigenvalue +1
        U_sym = symmetrize_u_irr(U, sitesym, ir=0, d_left=sitesym.d_matrix_band)
        torch.testing.assert_close(U_sym, U, atol=1e-8, rtol=0)

    def test_symmetrize_u_irr_trivial_stabilizer_just_orthonormalizes(self):
        """ir=1 has no nontrivial stabilizer, so symmetrize_u_irr should
        just Loewdin-orthonormalize U without otherwise changing its span."""
        sitesym, _, _ = _inversion_sitesym()
        rng = np.random.default_rng(4)
        U = torch.tensor(rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1)),
                         dtype=torch.complex128)
        U_sym = symmetrize_u_irr(U, sitesym, ir=1, d_left=sitesym.d_matrix_band)
        torch.testing.assert_close(U_sym.conj().transpose(-1, -2) @ U_sym,
                                   torch.eye(1, dtype=torch.complex128), atol=1e-10, rtol=0)
        # same span as the original (rank-1 projector unchanged)
        P_before = U @ U.conj().transpose(-1, -2) / (U.conj().transpose(-1, -2) @ U).real
        P_after = U_sym @ U_sym.conj().transpose(-1, -2)
        torch.testing.assert_close(P_before, P_after, atol=1e-8, rtol=0)


# ===========================================================================
# End-to-end: disentangle(sitesym=...) + minimize_spread_symmetrized
# ===========================================================================

def _random_involution(n: int, rng) -> torch.Tensor:
    """A Hermitian AND unitary (D^2=I) random matrix -- a valid representation
    of an order-2 group element (e.g. inversion) at a stabilizer point."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    signs = rng.choice([1.0, -1.0], size=n)
    D = Q @ np.diag(signs) @ Q.conj().T
    return torch.tensor(D, dtype=torch.complex128)


def _covariant_gaas_like_system(nb=4, nw=2, seed=0):
    """
    Synthetic disentanglement + spread-minimization problem, fully covariant
    under inversion on the same 4-point mesh as `_inversion_sitesym`: a
    "true" per-k unitary V_full built from independent irreducible-k
    unitaries and broadcast via the group action, then Mmn built from it
    (so Mmn is exactly covariant, not just a mask/label on top of random
    data) -- the strongest test available without a real .dmn/.mmn pair.
    """
    rng = np.random.default_rng(seed)
    nk, nsym, nkptirr, nnb = 4, 2, 3, 2
    ir2ik = torch.tensor([0, 1, 2])
    kptsym = torch.tensor([[0, 1, 2], [0, 3, 2]])

    Dband = torch.zeros(nb, nb, nsym, nkptirr, dtype=torch.complex128)
    Dband[:, :, 0, :] = torch.eye(nb, dtype=torch.complex128).unsqueeze(-1)
    Dband[:, :, 1, 0] = _random_involution(nb, rng)
    Dband[:, :, 1, 2] = _random_involution(nb, rng)
    A = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
    Q, _ = np.linalg.qr(A)
    Dband[:, :, 1, 1] = torch.tensor(Q, dtype=torch.complex128)

    Dwann = torch.zeros(nw, nw, nsym, nkptirr, dtype=torch.complex128)
    Dwann[:, :, 0, :] = torch.eye(nw, dtype=torch.complex128).unsqueeze(-1)
    Dwann[:, :, 1, 0] = _random_involution(nw, rng)
    Dwann[:, :, 1, 2] = _random_involution(nw, rng)
    Aw = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
    Qw, _ = np.linalg.qr(Aw)
    Dwann[:, :, 1, 1] = torch.tensor(Qw, dtype=torch.complex128)

    sitesym = SiteSymmetry(
        nsymmetry=nsym, nkptirr=nkptirr, num_kpts=nk, num_bands=nb, num_wann=nw,
        ik2ir=torch.tensor([0, 1, 2, 1]), ir2ik=ir2ik, kptsym=kptsym,
        d_matrix_wann=Dwann, d_matrix_band=Dband,
    )

    V_full = torch.zeros(nk, nb, nb, dtype=torch.complex128)
    for ir in range(nkptirr):
        Ai = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        Qi, _ = np.linalg.qr(Ai)
        Vi = torch.tensor(Qi, dtype=torch.complex128)
        ik = ir2ik[ir].item()
        V_full[ik] = Vi
        irk = kptsym[1, ir].item()
        if irk != ik:
            V_full[irk] = Dband[:, :, 1, ir] @ Vi

    kb_idx = torch.tensor([[(i + 1) % nk, (i - 1) % nk] for i in range(nk)])
    Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
    for ik in range(nk):
        for ib in range(nnb):
            ik2 = kb_idx[ik, ib].item()
            Mmn[ik, ib] = V_full[ik].conj().transpose(-1, -2) @ V_full[ik2]
    wb = torch.full((nnb,), 1.0 / (nnb * np.pi**2), dtype=torch.float64)
    eig = torch.zeros(nk, nb, dtype=torch.float64)
    bvecs = torch.stack([torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], dtype=torch.float64)
                        for _ in range(nk)])

    return sitesym, Mmn, wb, kb_idx, eig, bvecs


class TestSitesymEndToEnd:
    def test_disentangle_with_sitesym_decreases_omega_i_and_stays_covariant(self):
        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system()
        result = disentangle(Mmn, eig, wb, kb_idx, nw=sitesym.num_wann, n_iter=50,
                             bvecs=bvecs, sitesym=sitesym)
        assert result.history[-1] <= result.history[0] + 1e-8

        # V(Rk) = d_band(R,k) V(k) d_wann(R,k)^dagger (V is semi-unitary,
        # (nb, nw): d_band acts on the band side, d_wann on the Wannier side)
        V = result.V
        for ir in range(sitesym.nkptirr):
            ik = int(sitesym.ir2ik[ir])
            irk = int(sitesym.kptsym[1, ir])
            if irk == ik:
                continue
            Db = sitesym.d_matrix_band[:, :, 1, ir]
            Dw = sitesym.d_matrix_wann[:, :, 1, ir]
            torch.testing.assert_close(Db @ V[ik] @ Dw.conj().transpose(-1, -2), V[irk],
                                       atol=1e-8, rtol=0)

    def test_disentangle_sitesym_rejects_frozen_bands(self):
        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system()
        with pytest.raises(ValueError, match="frozen bands"):
            disentangle(Mmn, eig, wb, kb_idx, nw=sitesym.num_wann, n_iter=10,
                        bvecs=bvecs, sitesym=sitesym,
                        frozen_window=(-1.0, 1.0))   # eig is all zeros -> every band "frozen"

    @pytest.mark.parametrize("optimizer", ["sgd", "cg", "adam"])
    def test_minimize_spread_symmetrized_decreases_omega_and_stays_covariant(self, optimizer):
        """
        All three symmetrized optimizers must decrease Omega and keep the
        converged gauge exactly covariant. NOTE: on a real GaAs validation
        (see core.optim.minimize_spread_symmetrized's docstring), Adam got
        stuck in a bad local minimum where SGD/CG both converged to the
        correct answer -- "decreases" is deliberately the only quantitative
        claim checked here for all three; the real-data cross-check against
        wannier90.x is what actually pins CG/SGD's correctness.
        """
        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system()
        dis_result = disentangle(Mmn, eig, wb, kb_idx, nw=sitesym.num_wann, n_iter=50,
                                 bvecs=bvecs, sitesym=sitesym)
        Mmn_opt = rotate_overlaps(dis_result.V, Mmn, kb_idx)

        nw = sitesym.num_wann
        U_irr_init = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(
            sitesym.nkptirr, -1, -1).clone()
        result = minimize_spread_symmetrized(U_irr_init, sitesym, Mmn_opt, wb, bvecs, kb_idx,
                                             optimizer=optimizer,
                                             lr=0.1, n_iter=100, conv_tol=1e-12, conv_window=5)

        assert result.history[-1] <= result.history[0] + 1e-8

        U = result.U_final
        for ir in range(sitesym.nkptirr):
            ik = int(sitesym.ir2ik[ir])
            irk = int(sitesym.kptsym[1, ir])
            if irk == ik:
                continue
            D = sitesym.d_matrix_wann[:, :, 1, ir]
            torch.testing.assert_close(D @ U[ik] @ D.conj().transpose(-1, -2), U[irk],
                                       atol=1e-8, rtol=0)


# ===========================================================================
# Isolated bands (nb == nw): U's left index is d_matrix_band, NOT
# d_matrix_wann -- a real, confirmed bug (GaAs bond_centered, num_wann=4,
# diverged to Omega_total ~150-185 Ang^2) that the nb>nw tests above cannot
# catch, since genuine disentanglement always puts U entirely in
# Wannier-gauge space by the time minimize_spread_symmetrized runs. This
# fixture uses nb == nw with Dband != Dwann (both nontrivial involutions,
# independently random) specifically so a left/right representation mixup
# is numerically visible.
# ===========================================================================

class TestIsolatedBandsSitesym:
    @pytest.mark.parametrize("optimizer", ["sgd", "cg", "adam"])
    def test_minimize_spread_symmetrized_isolated_bands_needs_d_matrix_band(self, optimizer):
        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system(nb=2, nw=2)
        nw = sitesym.num_wann
        U_irr_init = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(
            sitesym.nkptirr, -1, -1).clone()

        result = minimize_spread_symmetrized(
            U_irr_init, sitesym, Mmn, wb, bvecs, kb_idx,
            optimizer=optimizer, lr=0.1, n_iter=200, conv_tol=1e-12, conv_window=5,
            d_left=sitesym.d_matrix_band,
        )
        assert result.history[-1] <= result.history[0] + 1e-8

        U = result.U_final
        found_nontrivial = False
        for ir in range(sitesym.nkptirr):
            ik = int(sitesym.ir2ik[ir])
            irk = int(sitesym.kptsym[1, ir])
            if irk == ik:
                continue
            found_nontrivial = True
            Db = sitesym.d_matrix_band[:, :, 1, ir]
            Dw = sitesym.d_matrix_wann[:, :, 1, ir]
            # correct convention: U(Rk) = D_band U(k) D_wann^dagger
            torch.testing.assert_close(Db @ U[ik] @ Dw.conj().transpose(-1, -2), U[irk],
                                       atol=1e-6, rtol=0)
            # the old (buggy) D_wann-D_wann convention must NOT hold here,
            # since Dband != Dwann by construction -- guards against a
            # future regression silently reverting to the wrong default
            mismatch = (Dw @ U[ik] @ Dw.conj().transpose(-1, -2) - U[irk]).abs().max().item()
            assert mismatch > 1e-3
        assert found_nontrivial


# ===========================================================================
# core.pipeline.wannierize(sitesym=...) -- the high-level wrapper
# ===========================================================================

class TestWannierizeWithSitesym:
    def test_wannierize_with_sitesym_runs_and_stays_covariant(self):
        from waw.core.pipeline import wannierize
        from waw.core.types import WannierData

        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system()
        nk, nnb, nb, nw = Mmn.shape[0], Mmn.shape[1], Mmn.shape[2], sitesym.num_wann

        rng = np.random.default_rng(5)
        Amn = torch.tensor(rng.normal(size=(nk, nb, nw)) + 1j * rng.normal(size=(nk, nb, nw)),
                           dtype=torch.complex128)
        kpts = torch.zeros(nk, 3, dtype=torch.float64)
        wdata = WannierData(Mmn=Mmn, Amn=Amn, eig=eig, kpts=kpts, bvecs=bvecs, wb=wb,
                           kb_idx=kb_idx, params={})

        real_lattice = np.eye(3) * 5.0
        result = wannierize(
            wdata, nw, (1, 1, 4), real_lattice,
            sitesym=sitesym, optimizer="cg", n_iter=100, dis_n_iter=50,
            conv_tol=1e-12, verbose=False,
        )

        assert result.dis is not None
        U = result.spread.U_final
        for ir in range(sitesym.nkptirr):
            ik = int(sitesym.ir2ik[ir])
            irk = int(sitesym.kptsym[1, ir])
            if irk == ik:
                continue
            D = sitesym.d_matrix_wann[:, :, 1, ir]
            torch.testing.assert_close(D @ U[ik] @ D.conj().transpose(-1, -2), U[irk],
                                       atol=1e-6, rtol=0)

    def test_wannierize_sitesym_ignores_n_restarts(self):
        """n_restarts > 1 with sitesym must not error (restarts are simply
        not supported/applied for the symmetrized path -- a single run)."""
        from waw.core.pipeline import wannierize
        from waw.core.types import WannierData

        sitesym, Mmn, wb, kb_idx, eig, bvecs = _covariant_gaas_like_system()
        nk, nb, nw = Mmn.shape[0], Mmn.shape[2], sitesym.num_wann
        rng = np.random.default_rng(6)
        Amn = torch.tensor(rng.normal(size=(nk, nb, nw)) + 1j * rng.normal(size=(nk, nb, nw)),
                           dtype=torch.complex128)
        kpts = torch.zeros(nk, 3, dtype=torch.float64)
        wdata = WannierData(Mmn=Mmn, Amn=Amn, eig=eig, kpts=kpts, bvecs=bvecs, wb=wb,
                           kb_idx=kb_idx, params={})
        real_lattice = np.eye(3) * 5.0

        result = wannierize(
            wdata, nw, (1, 1, 4), real_lattice,
            sitesym=sitesym, optimizer="cg", n_iter=50, dis_n_iter=30,
            n_restarts=5, verbose=False,
        )
        assert result.spread.U_final.shape == (nk, nw, nw)
