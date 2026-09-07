"""
Tests for waw/global_optim.py — Part 7.

  1. Result is a SpreadResult.
  2. n_restarts=1, n_hops=0 with SVD init matches minimize_spread directly.
  3. Returned Omega is ≤ every individual restart's Omega.
  4. Perturbation _perturb_U produces unitary matrices.
  5. Perturbation changes U (not a no-op).
  6. Reproducibility: same seed → same result.
  7. Different seeds → different intermediate paths (not necessarily different final Omega
     for a system with a unique minimum like Si).
  8. n_hops > 0 does not worsen the final Omega vs n_hops=0.
  9. On Si, global_minimize_spread converges to W90's Omega within tolerance.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.global_optim import global_minimize_spread, _perturb_U
from waw.core.optim         import minimize_spread, SpreadResult
from waw.core.init          import svd_init, random_unitary
from waw.core.spread        import compute_spread
from waw.core.kmesh import _compute_bvecs_and_weights
from waw.interfaces.wannier90.loader import parse_recip_lattice
from waw.interfaces.wannier90.io            import read_mmn, read_amn, read_nnkp, read_win

SI_DIR     = Path(__file__).parent / "data" / "silicon"
HAS_SI     = (SI_DIR / "silicon.mmn").exists()
from waw.units import BOHR_TO_ANG
W90_OMEGA  = 6.468598306   # Ang^2


# ===========================================================================
# Fixture: load Si data once
# ===========================================================================

@pytest.fixture(scope="module")
def si_data():
    if not HAS_SI:
        pytest.skip("Silicon reference data not found")
    Mmn_np, _ = read_mmn(SI_DIR / "silicon.mmn")
    Amn_np    = read_amn(SI_DIR / "silicon.amn")
    nnkp      = read_nnkp(SI_DIR / "silicon.nnkp")
    params    = read_win(SI_DIR / "silicon.win")
    recip     = parse_recip_lattice(params)
    bvecs_np, wb_np = _compute_bvecs_and_weights(
        nnkp["kpoints"], nnkp["nnkpts"], nnkp["g_vectors"], recip
    )
    return dict(
        Mmn    = torch.tensor(Mmn_np, dtype=torch.complex128),
        Amn    = torch.tensor(Amn_np, dtype=torch.complex128),
        bvecs  = torch.tensor(bvecs_np, dtype=torch.float64),
        wb     = torch.tensor(wb_np, dtype=torch.float64),
        kb_idx = torch.tensor(nnkp["nnkpts"], dtype=torch.long),
        nk=64, nw=4,
    )


def _trivial_problem(nk=4, nw=2, nnb=4, seed=7):
    """Minimal synthetic spread problem for fast unit tests."""
    Mmn = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
    Mmn = Mmn.expand(nk, nnb, nw, nw).clone()
    wb     = torch.ones(nnb, dtype=torch.float64) / (nnb * np.pi**2)
    bvecs  = torch.zeros(nk, nnb, 3, dtype=torch.float64)
    kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
    g      = torch.Generator().manual_seed(seed)
    U_init = random_unitary(nk, nw, generator=g)
    return U_init, Mmn, wb, bvecs, kb_idx


# ===========================================================================
# 1. Return type
# ===========================================================================

class TestReturnType:
    def test_returns_spreadresult(self):
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()
        result = global_minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                        n_restarts=2, n_iter=5)
        assert isinstance(result, SpreadResult)


# ===========================================================================
# 2. n_restarts=1, n_hops=0 should match minimize_spread
# ===========================================================================

class TestSingleRestart:
    def test_matches_minimize_spread(self):
        """With one restart and no hops, result must equal a direct minimize_spread call."""
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()

        ref = minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                              optimizer="adam", lr=3e-2, n_iter=20, conv_tol=0.0,
                              conv_window=5)
        glo = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                     n_restarts=1, n_hops=0, optimizer="adam",
                                     lr=3e-2, n_iter=20, conv_tol=0.0,
                                     conv_window=5)
        assert abs(ref.Omega - glo.Omega) < 1e-10


# ===========================================================================
# 3. Returned Omega is the best across all restarts
# ===========================================================================

class TestBestRestart:
    def test_omega_is_global_best(self):
        """global_minimize_spread Omega must be ≤ each individual restart's Omega."""
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem(nk=4, nw=2, seed=3)
        n_restarts = 4

        # Collect individual restart omegas
        omegas = []
        omegas.append(
            minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                            n_iter=30, conv_tol=0.0).Omega
        )
        for i in range(1, n_restarts):
            g = torch.Generator().manual_seed(i * 1000)
            U_r = random_unitary(4, 2, generator=g)
            omegas.append(
                minimize_spread(U_r, Mmn, wb, bvecs, kb_idx,
                                n_iter=30, conv_tol=0.0).Omega
            )

        best_manual = min(omegas)
        glo = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                     n_restarts=n_restarts, n_hops=0,
                                     n_iter=30, conv_tol=0.0, seed=0)
        assert glo.Omega <= best_manual + 1e-10


# ===========================================================================
# 4-5. _perturb_U
# ===========================================================================

class TestPerturbU:
    def test_unitary_after_perturbation(self):
        nk, nw = 4, 3
        g = torch.Generator().manual_seed(0)
        U = random_unitary(nk, nw, generator=g)
        g2 = torch.Generator().manual_seed(42)
        U_p = _perturb_U(U, strength=0.5, generator=g2)
        UhU = torch.matmul(U_p.conj().transpose(-1, -2), U_p)
        eye = torch.eye(nw, dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-10, rtol=0)

    def test_perturbation_changes_U(self):
        nk, nw = 4, 3
        g = torch.Generator().manual_seed(0)
        U = random_unitary(nk, nw, generator=g)
        g2 = torch.Generator().manual_seed(1)
        U_p = _perturb_U(U, strength=0.5, generator=g2)
        assert not torch.allclose(U, U_p, atol=1e-6), "Perturbation is a no-op"

    def test_small_perturbation_stays_close(self):
        nk, nw = 4, 3
        g = torch.Generator().manual_seed(0)
        U = random_unitary(nk, nw, generator=g)
        g2 = torch.Generator().manual_seed(2)
        U_p = _perturb_U(U, strength=1e-4, generator=g2)
        diff = (U - U_p).norm().item()
        assert diff < 0.1, f"Small perturbation changed U too much: {diff:.4f}"


# ===========================================================================
# 6. Reproducibility
# ===========================================================================

class TestReproducibility:
    def test_same_seed_same_result(self):
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()
        r1 = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                    n_restarts=3, seed=42, n_iter=20, conv_tol=0.0)
        r2 = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                    n_restarts=3, seed=42, n_iter=20, conv_tol=0.0)
        assert abs(r1.Omega - r2.Omega) < 1e-12


# ===========================================================================
# 7. Basin hopping does not worsen Omega
# ===========================================================================

class TestBasinHopping:
    def test_hops_do_not_worsen(self):
        """n_hops > 0 must give Omega ≤ n_hops=0 (we only accept improvements)."""
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()
        r0 = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                    n_restarts=2, n_hops=0, n_iter=30, seed=0)
        r1 = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                    n_restarts=2, n_hops=3, n_iter=30, seed=0,
                                    hop_strength=0.5)
        assert r1.Omega <= r0.Omega + 1e-10, (
            f"Hops worsened Omega: {r0.Omega:.6f} → {r1.Omega:.6f}"
        )

    def test_hops_U_is_unitary(self):
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()
        result = global_minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                        n_restarts=1, n_hops=2, n_iter=10, seed=5)
        U = result.U_final
        UhU = torch.matmul(U.conj().transpose(-1, -2), U)
        eye = torch.eye(U.shape[1], dtype=torch.complex128).unsqueeze(0)
        torch.testing.assert_close(UhU, eye.expand_as(UhU), atol=1e-8, rtol=0)


# ===========================================================================
# 8. Parallel restarts produce the same result as sequential
# ===========================================================================

class TestParallelRestarts:
    def test_parallel_matches_sequential(self):
        """n_workers=2 must give the same best Omega as n_workers=1."""
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem(nk=4, nw=2, seed=3)
        seq = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                     n_restarts=3, seed=0, n_iter=30, conv_tol=0.0,
                                     n_workers=1)
        par = global_minimize_spread(U_init.clone(), Mmn, wb, bvecs, kb_idx,
                                     n_restarts=3, seed=0, n_iter=30, conv_tol=0.0,
                                     n_workers=2)
        assert abs(seq.Omega - par.Omega) < 1e-10, (
            f"parallel Omega {par.Omega} != sequential {seq.Omega}"
        )

    def test_n_workers_none_auto(self):
        """n_workers=None (auto) must also return a valid SpreadResult."""
        U_init, Mmn, wb, bvecs, kb_idx = _trivial_problem()
        result = global_minimize_spread(U_init, Mmn, wb, bvecs, kb_idx,
                                        n_restarts=2, n_iter=5, n_workers=None)
        assert isinstance(result, SpreadResult)
        assert result.Omega < float("inf")


# ===========================================================================
# 9. Silicon: converges to W90's Omega
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
class TestSiliconGlobal:
    OMEGA_TOL = 1e-3   # Ang^2

    @pytest.fixture(autouse=True)
    def setup(self, si_data):
        self.d   = si_data
        self.U0  = svd_init(si_data["Amn"])

    def test_single_restart_matches_w90(self):
        """n_restarts=1 (SVD init only) must already reach W90's Omega."""
        result = global_minimize_spread(
            self.U0, self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"],
            n_restarts=1, n_hops=0, n_iter=1000,
        )
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA) < self.OMEGA_TOL, (
            f"Omega={omega_ang2:.6f} Ang^2, expected {W90_OMEGA:.6f}"
        )

    def test_multiple_restarts_match_w90(self):
        """Multiple random restarts should all converge to the same unique minimum."""
        result = global_minimize_spread(
            self.U0, self.d["Mmn"], self.d["wb"], self.d["bvecs"], self.d["kb_idx"],
            n_restarts=3, n_hops=0, n_iter=1000,
        )
        omega_ang2 = result.Omega * BOHR_TO_ANG**2
        assert abs(omega_ang2 - W90_OMEGA) < self.OMEGA_TOL, (
            f"Omega={omega_ang2:.6f} Ang^2, expected {W90_OMEGA:.6f}"
        )
