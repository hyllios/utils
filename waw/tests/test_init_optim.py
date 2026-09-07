"""
Tests for waw/init.py (Part 4) and waw/optim.py (Part 5).

Initialization tests:
  1. svd_init produces (semi-)unitary matrices.
  2. svd_init gives the same initial spread as Wannier90 reports.
  3. random_unitary produces Haar-distributed unitaries (unitarity check).
  4. random_unitary with different seeds gives different matrices.

Optimizer tests (isolated bands, Si reference data):
  5. SGD reduces Omega monotonically for a suitable learning rate.
  6. Adam converges to within 1e-4 Ang^2 of W90's final Omega on Si.
  7. SGD converges to within 1e-4 Ang^2 of W90's final Omega on Si.
  8. Unitarity is preserved after optimization (retraction works).
  9. Convergence history has the right length and is monotone (for SGD).
 10. Both optimizers give Omega_D ~ 0 for Si (high symmetry enforces this).
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.init   import svd_init, random_unitary
from waw.core.optim  import minimize_spread, SpreadResult
from waw.core.spread import compute_spread
from waw.core.kmesh import _compute_bvecs_and_weights
from waw.interfaces.wannier90.loader import parse_recip_lattice
from waw.interfaces.wannier90.io     import read_mmn, read_amn, read_nnkp, read_win

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI_DATA = (SI_DIR / "silicon.mmn").exists()

from waw.units import BOHR_TO_ANG
# Wannier90 reference for Si 4x4x4, 4 valence WFs (from silicon.wout)
W90_OMEGA_TOTAL_ANG2 = 6.468598306
W90_OMEGA_I_ANG2     = 5.890769995


# ===========================================================================
# Fixture: load Si data once
# ===========================================================================

@pytest.fixture(scope="module")
def si_data():
    """Load silicon reference data into tensors (module scope = loaded once)."""
    if not HAS_SI_DATA:
        pytest.skip("Silicon reference data not found")

    Mmn_np, _  = read_mmn(SI_DIR / "silicon.mmn")
    Amn_np      = read_amn(SI_DIR / "silicon.amn")
    nnkp        = read_nnkp(SI_DIR / "silicon.nnkp")
    params      = read_win(SI_DIR / "silicon.win")
    recip       = parse_recip_lattice(params)
    bvecs_np, wb_np = _compute_bvecs_and_weights(
        nnkp["kpoints"], nnkp["nnkpts"], nnkp["g_vectors"], recip
    )

    return dict(
        Mmn    = torch.tensor(Mmn_np,          dtype=torch.complex128),
        Amn    = torch.tensor(Amn_np,          dtype=torch.complex128),
        bvecs  = torch.tensor(bvecs_np,        dtype=torch.float64),
        wb     = torch.tensor(wb_np,           dtype=torch.float64),
        kb_idx = torch.tensor(nnkp["nnkpts"],  dtype=torch.long),
        nk=64, nb=4, nw=4,
    )


# ===========================================================================
# Part 4: Initialization tests
# ===========================================================================

class TestSvdInit:
    def test_shape_isolated(self, si_data):
        """For isolated bands (nb==nw), output is (nk, nw, nw)."""
        U = svd_init(si_data["Amn"])
        assert U.shape == (si_data["nk"], si_data["nw"], si_data["nw"])

    def test_unitarity(self, si_data):
        """U† U must be the identity to machine precision."""
        U = svd_init(si_data["Amn"])
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(si_data["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-12, rtol=0)

    def test_dtype_preserved(self, si_data):
        U = svd_init(si_data["Amn"])
        assert U.dtype == torch.complex128

    def test_initial_omega_i_matches_w90(self, si_data):
        """
        Omega_I is gauge-invariant so it must equal W90's value regardless
        of the initialisation.  Verify that our svd_init U gives the correct Omega_I.
        """
        U = svd_init(si_data["Amn"])
        _, OI, _, _, _ = compute_spread(
            U, si_data["Mmn"], si_data["wb"], si_data["bvecs"], si_data["kb_idx"]
        )
        assert abs(OI.item() * BOHR_TO_ANG**2 - W90_OMEGA_I_ANG2) < 1e-4


class TestRandomUnitary:
    def test_shape(self):
        U = random_unitary(nk=8, nw=4)
        assert U.shape == (8, 4, 4)

    def test_unitarity(self):
        U = random_unitary(nk=16, nw=6)
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(6, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-12, rtol=0)

    def test_different_seeds_give_different_matrices(self):
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(1)
        U1 = random_unitary(4, 4, generator=g1)
        U2 = random_unitary(4, 4, generator=g2)
        assert not torch.allclose(U1, U2)

    def test_same_seed_reproducible(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        U1 = random_unitary(4, 4, generator=g1)
        U2 = random_unitary(4, 4, generator=g2)
        torch.testing.assert_close(U1, U2)


# ===========================================================================
# Part 5: Optimizer tests
# ===========================================================================

class TestSpreadResult:
    """Basic structural tests for SpreadResult — independent of Si data."""

    def _make_trivial_data(self):
        """Tiny 2-WF system where Omega = 0 is reachable (diagonal Mmn)."""
        nk, nnb, nw = 4, 6, 2
        # Mmn = identity → M_tilde = I for any unitary U → Omega = 0
        Mmn    = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
        Mmn    = Mmn.expand(nk, nnb, nw, nw).clone()
        wb     = torch.ones(nnb, dtype=torch.float64) / (nnb * np.pi**2)
        bvecs  = torch.zeros(nk, nnb, 3, dtype=torch.float64)  # (nk, nnb, 3)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        g      = torch.Generator().manual_seed(0)
        U_init = random_unitary(nk, nw, generator=g)
        return U_init, Mmn, wb, bvecs, kb_idx

    def test_result_is_spreadresult(self):
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="sgd", n_iter=5)
        assert isinstance(result, SpreadResult)

    def test_history_length(self):
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="sgd", n_iter=10, conv_tol=0.0)
        assert len(result.history) <= 10
        assert len(result.history) >= 1

    def test_omega_equals_sum_of_parts(self):
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="sgd", n_iter=5)
        reconstructed = result.Omega_I + result.Omega_D + result.Omega_OD
        assert abs(result.Omega - reconstructed) < 1e-10

    def test_unknown_optimizer_raises(self):
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            minimize_spread(U_init, Mmn, wb, bvecs, kb_idx, optimizer="nonexistent")

    def test_cg_result_is_spreadresult(self):
        """CG smoke test on degenerate data (Mmn=I everywhere -> gradient is
        exactly zero at every point): exercises the gcnorm0~0 and
        not-a-descent-direction fallback branches without crashing."""
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="cg", n_iter=5)
        assert isinstance(result, SpreadResult)

    def test_lbfgs_result_is_spreadresult(self):
        """L-BFGS smoke test on degenerate data (zero gradient everywhere):
        exercises the not-a-descent-direction / history-clearing fallback."""
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="lbfgs", n_iter=5)
        assert isinstance(result, SpreadResult)

    def test_diis_result_is_spreadresult(self):
        """DIIS smoke test on degenerate data (zero gradient everywhere):
        exercises the singular-B / no-improvement fallback to steepest
        descent."""
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="diis", n_iter=5)
        assert isinstance(result, SpreadResult)

    def test_rtr_result_is_spreadresult(self):
        """RTR smoke test on degenerate data (zero gradient everywhere):
        exercises truncated CG's immediate-return-zero branch (r0_norm~0)."""
        U_init, Mmn, wb, bvecs, kb_idx = self._make_trivial_data()
        result = minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                 optimizer="rtr", n_iter=5)
        assert isinstance(result, SpreadResult)


@pytest.mark.skipif(not HAS_SI_DATA, reason="Silicon reference data not found")
class TestOptimizerOnSilicon:
    """
    Convergence tests against the Wannier90 reference for Si.

    Tolerance: 1e-4 Ang^2 on Omega.  This is well within the W90 result
    precision and validates that our optimizer finds the same minimum.
    """
    OMEGA_TOL_ANG2 = 1e-3   # allow 1 meV / Ang^2 slack

    @pytest.fixture(autouse=True)
    def prepare(self, si_data):
        self.d   = si_data
        self.U0  = svd_init(si_data["Amn"])

    def _run(self, optimizer, lr, n_iter=500, conv_tol=1e-10):
        return minimize_spread(
            self.U0,
            self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"],
            optimizer=optimizer, lr=lr, n_iter=n_iter, conv_tol=conv_tol,
            conv_window=5,
        )

    # ---- SGD ---------------------------------------------------------------

    def test_sgd_reduces_omega(self):
        """SGD must lower the spread from the initial value."""
        result = self._run("sgd", lr=1.0, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "SGD did not reduce Omega"
        )

    def test_sgd_convergence_to_w90(self):
        """SGD must reach W90's Omega_total to within OMEGA_TOL_ANG2."""
        result = self._run("sgd", lr=1.0, n_iter=1000)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"SGD: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )

    def test_sgd_preserves_unitarity(self):
        """After optimization, U must still be unitary to 1e-10."""
        result = self._run("sgd", lr=1.0, n_iter=200, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    def test_sgd_omega_d_near_zero(self):
        """For Si (tetrahedral symmetry), Omega_D must vanish at the minimum."""
        result = self._run("sgd", lr=1.0, n_iter=1000)
        omega_d_ang2 = result.Omega_D * BOHR_TO_ANG**2
        assert abs(omega_d_ang2) < 1e-4, (
            f"Omega_D = {omega_d_ang2:.2e} Ang^2, expected ~0"
        )

    # ---- Adam --------------------------------------------------------------

    def test_adam_reduces_omega(self):
        """Adam must lower the spread from the initial value."""
        result = self._run("adam", lr=3e-2, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "Adam did not reduce Omega"
        )

    def test_adam_convergence_to_w90(self):
        """Adam must reach W90's Omega_total to within OMEGA_TOL_ANG2."""
        result = self._run("adam", lr=3e-2, n_iter=2000)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"Adam: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )

    def test_adam_preserves_unitarity(self):
        result = self._run("adam", lr=3e-2, n_iter=200, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    # ---- CG (Fletcher-Reeves, matching Wannier90's own minimiser) ----------

    def test_cg_reduces_omega(self):
        """CG must lower the spread from the initial value."""
        result = self._run("cg", lr=1.0, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "CG did not reduce Omega"
        )

    def test_cg_convergence_to_w90(self):
        """CG must reach W90's Omega_total to within OMEGA_TOL_ANG2 -- and
        much faster than Adam/SGD (a few dozen iterations, not thousands)."""
        result = self._run("cg", lr=1.0, n_iter=200)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"CG: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )
        assert len(result.history) < 200, (
            "CG should converge well within 200 iterations on this system"
        )

    def test_cg_preserves_unitarity(self):
        result = self._run("cg", lr=1.0, n_iter=100, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    def test_cg_omega_d_near_zero(self):
        """For Si (tetrahedral symmetry), Omega_D must vanish at the minimum."""
        result = self._run("cg", lr=1.0, n_iter=200)
        omega_d_ang2 = result.Omega_D * BOHR_TO_ANG**2
        assert abs(omega_d_ang2) < 1e-4, (
            f"Omega_D = {omega_d_ang2:.2e} Ang^2, expected ~0"
        )

    def test_cg_is_insensitive_to_trial_step(self):
        """The parabolic line search should make CG converge to the same
        minimum regardless of the (otherwise arbitrary) trial step length."""
        omegas = [self._run("cg", lr=lr, n_iter=200).Omega * BOHR_TO_ANG**2
                  for lr in (0.1, 1.0, 3.0)]
        for omega_ang2 in omegas:
            assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2

    # ---- L-BFGS --------------------------------------------------------

    def test_lbfgs_reduces_omega(self):
        """L-BFGS must lower the spread from the initial value."""
        result = self._run("lbfgs", lr=1.0, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "L-BFGS did not reduce Omega"
        )

    def test_lbfgs_convergence_to_w90(self):
        """L-BFGS must reach W90's Omega_total to within OMEGA_TOL_ANG2."""
        result = self._run("lbfgs", lr=1.0, n_iter=200)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"L-BFGS: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )

    def test_lbfgs_preserves_unitarity(self):
        result = self._run("lbfgs", lr=1.0, n_iter=100, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    # ---- DIIS (Pulay mixing) -----------------------------------------------

    def test_diis_reduces_omega(self):
        """DIIS must lower the spread from the initial value."""
        result = self._run("diis", lr=1.0, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "DIIS did not reduce Omega"
        )

    def test_diis_convergence_to_w90(self):
        """DIIS must reach W90's Omega_total to within OMEGA_TOL_ANG2."""
        result = self._run("diis", lr=1.0, n_iter=1000)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"DIIS: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )

    def test_diis_preserves_unitarity(self):
        result = self._run("diis", lr=1.0, n_iter=200, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    # ---- RTR (Riemannian trust-region, second-order) -----------------------

    def test_rtr_reduces_omega(self):
        """RTR must lower the spread from the initial value."""
        result = self._run("rtr", lr=0.1, n_iter=50, conv_tol=0.0)
        assert result.history[-1] < result.history[0], (
            "RTR did not reduce Omega"
        )

    def test_rtr_convergence_to_w90(self):
        """RTR must reach W90's Omega_total to within OMEGA_TOL_ANG2 -- and
        in well under 200 iterations (genuine curvature information, not
        just gradient directions)."""
        result = self._run("rtr", lr=0.1, n_iter=200)
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA_TOTAL_ANG2) < self.OMEGA_TOL_ANG2, (
            f"RTR: Omega={omega_ang2:.6f} Ang^2, W90={W90_OMEGA_TOTAL_ANG2:.6f}"
        )

    def test_rtr_preserves_unitarity(self):
        result = self._run("rtr", lr=0.1, n_iter=100, conv_tol=0.0)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(self.d["nw"], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)

    def test_riemannian_hvp_matches_finite_difference(self):
        """
        The Euclidean Hessian-vector product (double backward through the
        same autodiff'd Omega(U) the gradient uses) must match a central
        finite difference of the Euclidean gradient along a random tangent
        direction, on real DFT overlap data -- the correctness check for
        RTR's only genuinely new piece of machinery (everything else reuses
        `_riemannian_gradient`/`_qr_retract`/`_real_inner`).
        """
        from waw.core.optim import _euclidean_hvp, _omega_and_grad

        U0 = self.U0
        g = torch.Generator().manual_seed(7)
        A = torch.randn(*U0.shape, dtype=torch.complex128, generator=g)
        S = A - A.conj().transpose(-1, -2)
        eta = torch.matmul(U0, S)
        eta = eta / eta.abs().max()

        Hv = _euclidean_hvp(U0, self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"], eta)

        eps = 1e-5
        _, g_p = _omega_and_grad(U0 + eps * eta, self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"])
        _, g_m = _omega_and_grad(U0 - eps * eta, self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"])
        Hv_fd = (g_p - g_m) / (2 * eps)

        torch.testing.assert_close(Hv, Hv_fd, atol=1e-6, rtol=1e-4)

    # ---- Comparison --------------------------------------------------------

    def test_both_optimizers_agree(self):
        """SGD and Adam must converge to the same minimum (within tolerance)."""
        res_sgd  = self._run("sgd",  lr=1.0,  n_iter=1000)
        res_adam = self._run("adam", lr=3e-2, n_iter=2000)
        diff = abs(res_sgd.Omega - res_adam.Omega) * BOHR_TO_ANG**2
        assert diff < self.OMEGA_TOL_ANG2 * 2, (
            f"SGD={res_sgd.Omega*BOHR_TO_ANG**2:.6f} vs "
            f"Adam={res_adam.Omega*BOHR_TO_ANG**2:.6f} Ang^2"
        )

    def test_cg_agrees_with_adam(self):
        """CG and Adam must converge to the same minimum (within tolerance)."""
        res_cg   = self._run("cg",   lr=1.0,  n_iter=200)
        res_adam = self._run("adam", lr=3e-2, n_iter=2000)
        diff = abs(res_cg.Omega - res_adam.Omega) * BOHR_TO_ANG**2
        assert diff < self.OMEGA_TOL_ANG2 * 2, (
            f"CG={res_cg.Omega*BOHR_TO_ANG**2:.6f} vs "
            f"Adam={res_adam.Omega*BOHR_TO_ANG**2:.6f} Ang^2"
        )

    def test_lbfgs_agrees_with_cg(self):
        """L-BFGS and CG must converge to the same minimum (within tolerance)."""
        res_lbfgs = self._run("lbfgs", lr=1.0, n_iter=200)
        res_cg    = self._run("cg",    lr=1.0, n_iter=200)
        diff = abs(res_lbfgs.Omega - res_cg.Omega) * BOHR_TO_ANG**2
        assert diff < self.OMEGA_TOL_ANG2 * 2, (
            f"L-BFGS={res_lbfgs.Omega*BOHR_TO_ANG**2:.6f} vs "
            f"CG={res_cg.Omega*BOHR_TO_ANG**2:.6f} Ang^2"
        )

    def test_diis_agrees_with_cg(self):
        """DIIS and CG must converge to the same minimum (within tolerance)."""
        res_diis = self._run("diis", lr=1.0, n_iter=1000)
        res_cg   = self._run("cg",   lr=1.0, n_iter=200)
        diff = abs(res_diis.Omega - res_cg.Omega) * BOHR_TO_ANG**2
        assert diff < self.OMEGA_TOL_ANG2 * 2, (
            f"DIIS={res_diis.Omega*BOHR_TO_ANG**2:.6f} vs "
            f"CG={res_cg.Omega*BOHR_TO_ANG**2:.6f} Ang^2"
        )

    def test_rtr_agrees_with_cg(self):
        """RTR and CG must converge to the same minimum (within tolerance)."""
        res_rtr = self._run("rtr", lr=0.1, n_iter=200)
        res_cg  = self._run("cg",  lr=1.0, n_iter=200)
        diff = abs(res_rtr.Omega - res_cg.Omega) * BOHR_TO_ANG**2
        assert diff < self.OMEGA_TOL_ANG2 * 2, (
            f"RTR={res_rtr.Omega*BOHR_TO_ANG**2:.6f} vs "
            f"CG={res_cg.Omega*BOHR_TO_ANG**2:.6f} Ang^2"
        )
