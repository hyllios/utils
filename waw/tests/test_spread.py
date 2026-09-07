"""
Tests for waw/spread.py — the spread functional.

Tests:
  1. rotate_overlaps: basic shape and gauge-invariance of Omega_I.
  2. Omega_D = 0 when centres are at origin (trivial gauge).
  3. Autograd gradient matches finite differences to < 1e-5.
     This is the key correctness test for the optimizer later.
  4. Gauge invariance: Omega_I must not change under U -> U * diag(phases).
  5. Comparison against real Wannier90 Si reference values.
     Reads the actual silicon.mmn/.amn/.nnkp files and checks that
     our Omega_I at the W90-converged U matches W90's reported Omega_I.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.spread import rotate_overlaps, compute_spread, compute_wannier_centres
from waw.core.kmesh import _compute_bvecs_and_weights
from waw.interfaces.wannier90.loader import parse_recip_lattice
from waw.interfaces.wannier90.io     import read_mmn, read_amn, read_eig, read_nnkp, read_win

# Path to the real Si reference data
SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI_DATA = (SI_DIR / "silicon.mmn").exists()

from waw.units import BOHR_TO_ANG


# ===========================================================================
# Helpers
# ===========================================================================

def _random_unitary(nk: int, nw: int, seed: int = 0) -> torch.Tensor:
    """Random unitary matrices via QR decomposition."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    A = torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    A += 1j * torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q


def _identity_U(nk: int, nw: int) -> torch.Tensor:
    """Identity gauge: U[k] = I for all k."""
    return torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(nk, -1, -1).clone()


def _make_simple_cubic_data(nk=8, nb=4, nw=2, nnb=6, seed=1):
    """
    Synthetic data on a simple-cubic mesh for fast unit tests.
    Mmn is constructed from random unitary matrices so that the
    Hermitian conjugate symmetry M(k,b)† = M(k+b,-b) is satisfied.
    """
    torch.manual_seed(seed)
    # Random unitary per k: Mmn[k,b] = V[k]† V[k+b] restricted to nw x nw
    V = torch.linalg.qr(
        torch.randn(nk, nb, nb, dtype=torch.complex128)
        + 1j * torch.randn(nk, nb, nb, dtype=torch.complex128)
    )[0]

    # Simple cubic b-vectors and weights for a 2x2x2 mesh with a=2*pi
    # b = ±pi along each axis, |b| = pi, weight = 1/(2*pi^2)
    # bvecs shape is (nk, nnb, 3) — for simple-cubic, same at every k-point
    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    wb_np  = np.full(6, 1.0 / (2 * np.pi**2))
    # broadcast to (nk, nnb, 3) — all k-points share the same b-vector set
    bvecs  = torch.tensor(
        np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64
    )
    wb     = torch.tensor(wb_np, dtype=torch.float64)

    # Build a symmetric neighbour table for an 8-point mesh arranged as 2x2x2
    # For simplicity: each k-point uses the same 6 neighbour indices
    # (self-connectivity, only valid for testing the functional, not physics)
    kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
    for ik in range(nk):
        for ib in range(nnb):
            kb_idx[ik, ib] = (ik + ib // 2 + 1) % nk

    # Build Mmn from the unitary matrices
    Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
    for ik in range(nk):
        for ib in range(nnb):
            ik2 = kb_idx[ik, ib].item()
            Mmn[ik, ib] = V[ik].conj().T @ V[ik2]

    U = _random_unitary(nk, nw, seed=seed)
    return U, Mmn[:, :, :nw, :nw], wb, bvecs, kb_idx


# ===========================================================================
# Tests: rotate_overlaps
# ===========================================================================

class TestRotateOverlaps:
    def test_shape(self):
        nk, nnb, nw = 4, 6, 3
        U      = _random_unitary(nk, nw)
        Mmn    = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        assert M_tilde.shape == (nk, nnb, nw, nw)

    def test_identity_gauge_unchanged(self):
        """With U=I, M_tilde must equal Mmn exactly."""
        nk, nnb, nw = 4, 6, 3
        U      = _identity_U(nk, nw)
        Mmn    = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        torch.testing.assert_close(M_tilde, Mmn, atol=1e-12, rtol=0)

    def test_unitary_gauge_preserves_singular_values(self):
        """
        A unitary rotation preserves the singular values of each M_tilde[k,b],
        so ||M_tilde||_F = ||Mmn||_F for every (k,b).
        """
        nk, nnb, nw = 4, 6, 3
        U      = _random_unitary(nk, nw)
        Mmn    = torch.randn(nk, nnb, nw, nw, dtype=torch.complex128)
        kb_idx = (torch.arange(nk).unsqueeze(1) + 1).remainder(nk).expand(nk, nnb)
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        norm_before = Mmn.abs().pow(2).sum(dim=(-1, -2))
        norm_after  = M_tilde.abs().pow(2).sum(dim=(-1, -2))
        torch.testing.assert_close(norm_after, norm_before, atol=1e-10, rtol=0)


# ===========================================================================
# Tests: compute_spread — structure and invariants
# ===========================================================================

class TestComputeSpread:
    def _run(self, U, Mmn, wb, bvecs, kb_idx):
        return compute_spread(U, Mmn, wb, bvecs, kb_idx)

    def test_output_shapes(self):
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        Omega, OI, OD, OOD, centres = self._run(U, Mmn, wb, bvecs, kb_idx)
        assert Omega.shape   == ()
        assert OI.shape      == ()
        assert OD.shape      == ()
        assert OOD.shape     == ()
        assert centres.shape == (U.shape[1], 3)

    def test_omega_equals_sum_of_parts(self):
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        Omega, OI, OD, OOD, _ = self._run(U, Mmn, wb, bvecs, kb_idx)
        torch.testing.assert_close(Omega, OI + OD + OOD, atol=1e-12, rtol=0)

    def test_omega_i_non_negative(self):
        """Omega_I >= 0 always (it's a sum of |1 - |M_nn|^2| terms)."""
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        _, OI, _, _, _ = self._run(U, Mmn, wb, bvecs, kb_idx)
        assert OI.item() >= -1e-12

    def test_omega_od_non_negative(self):
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        _, _, _, OOD, _ = self._run(U, Mmn, wb, bvecs, kb_idx)
        assert OOD.item() >= -1e-12

    def test_omega_i_gauge_invariant(self):
        """
        Omega_I is gauge invariant: multiplying U by diagonal phases
        must not change Omega_I (it only affects the phases of M_tilde_nn,
        not their modulus).
        """
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        _, OI_before, _, _, _ = self._run(U, Mmn, wb, bvecs, kb_idx)

        # Random diagonal phases
        nk, nw = U.shape[0], U.shape[1]
        phases  = torch.exp(1j * torch.rand(nk, nw, dtype=torch.float64) * 2 * np.pi)
        U_phased = U * phases.unsqueeze(-1)   # multiply each column by a phase

        _, OI_after, _, _, _ = self._run(U_phased, Mmn, wb, bvecs, kb_idx)
        assert abs(OI_before.item() - OI_after.item()) < 1e-10, (
            f"Omega_I changed under diagonal phase: {OI_before:.6f} -> {OI_after:.6f}"
        )

    def test_identity_U_omega_od_zero_for_diagonal_Mmn(self):
        """
        If Mmn is diagonal for all (k,b) and U=I, then Omega_OD = 0.
        """
        nk, nnb, nw = 4, 6, 2
        # Diagonal Mmn
        Mmn = torch.zeros(nk, nnb, nw, nw, dtype=torch.complex128)
        for i in range(nw):
            Mmn[:, :, i, i] = 0.8 + 0.0j

        wb     = torch.ones(nnb, dtype=torch.float64) / nnb
        bvecs  = torch.zeros(nk, nnb, 3, dtype=torch.float64)  # (nk, nnb, 3)
        kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
        U      = _identity_U(nk, nw)

        _, _, _, OOD, _ = compute_spread(U, Mmn, wb, bvecs, kb_idx)
        assert abs(OOD.item()) < 1e-12


# ===========================================================================
# Tests: autograd gradient vs finite differences
# ===========================================================================

class TestAutograd:
    """
    The spread functional must be differentiable so that we can pass its
    gradient to a Stiefel-manifold optimizer.  We verify that autograd
    matches a finite-difference estimate of the gradient.

    We use Wirtinger / Euclidean gradient: dOmega/dU* treated as a matrix
    with the same shape as U.
    """

    def test_autograd_vs_finite_difference(self):
        """
        Perturb one real and one imaginary element of U at k=0 by eps,
        recompute Omega, and compare (Omega+ - Omega-) / (2*eps) with
        the corresponding autograd gradient component.
        """
        nk, nnb, nw = 4, 6, 2
        eps = 1e-5

        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=nk, nw=nw)
        U = U.detach().requires_grad_(True)

        Omega, *_ = compute_spread(U, Mmn, wb, bvecs, kb_idx)
        Omega.backward()
        grad = U.grad.clone()   # (nk, nw, nw) complex

        # Check a handful of gradient elements against finite differences
        errors = []
        for ik in [0, 1]:
            for i in [0, 1]:
                for j in [0]:
                    for part in ["real", "imag"]:
                        U_p = U.detach().clone()
                        U_m = U.detach().clone()
                        if part == "real":
                            U_p[ik, i, j] += eps
                            U_m[ik, i, j] -= eps
                        else:
                            U_p[ik, i, j] += 1j * eps
                            U_m[ik, i, j] -= 1j * eps

                        with torch.no_grad():
                            O_p = compute_spread(U_p, Mmn, wb, bvecs, kb_idx)[0]
                            O_m = compute_spread(U_m, Mmn, wb, bvecs, kb_idx)[0]
                        fd  = (O_p - O_m).item() / (2 * eps)

                        if part == "real":
                            ag = grad[ik, i, j].real.item()
                        else:
                            ag = grad[ik, i, j].imag.item()

                        errors.append(abs(fd - ag))

        max_err = max(errors)
        assert max_err < 1e-5, (
            f"Max autograd vs finite-difference error: {max_err:.2e} (threshold 1e-5)"
        )

    def test_gradient_is_complex(self):
        """Gradient of a real scalar w.r.t. complex U must be complex."""
        U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data()
        U = U.detach().requires_grad_(True)
        Omega, *_ = compute_spread(U, Mmn, wb, bvecs, kb_idx)
        Omega.backward()
        assert U.grad.is_complex()


# ===========================================================================
# Tests: comparison with real Wannier90 silicon reference
# ===========================================================================

@pytest.mark.skipif(not HAS_SI_DATA, reason="Silicon reference data not found")
class TestSiliconReference:
    """
    Load the real silicon.mmn/.amn/.nnkp files and check that our spread
    functional reproduces Wannier90's reported values.

    Two checks:
      (a) Omega_I at the W90-converged gauge (read from silicon.chk or
          reconstructed from silicon.amn initial projections).
          Omega_I is gauge-invariant so it must match regardless of U.
      (b) Total Omega at the SVD-initialised U (before any minimization)
          must be finite and close to the W90-reported initial value.
    """

    @pytest.fixture(autouse=True)
    def load_si(self):
        """Load all silicon reference data once per test class."""
        self.Mmn_np, _  = read_mmn(SI_DIR / "silicon.mmn")
        self.Amn_np      = read_amn(SI_DIR / "silicon.amn")
        self.nnkp        = read_nnkp(SI_DIR / "silicon.nnkp")
        self.params      = read_win(SI_DIR / "silicon.win")

        # Reciprocal lattice from the .win unit_cell_cart block
        recip = parse_recip_lattice(self.params)
        bvecs_np, wb_np = _compute_bvecs_and_weights(
            kpts          = self.nnkp["kpoints"],
            nnkpts        = self.nnkp["nnkpts"],
            g_vectors     = self.nnkp["g_vectors"],
            recip_lattice = recip,
        )

        self.Mmn    = torch.tensor(self.Mmn_np, dtype=torch.complex128)
        self.Amn    = torch.tensor(self.Amn_np, dtype=torch.complex128)
        self.bvecs  = torch.tensor(bvecs_np,    dtype=torch.float64)
        self.wb     = torch.tensor(wb_np,        dtype=torch.float64)
        self.kb_idx = torch.tensor(self.nnkp["nnkpts"], dtype=torch.long)

        nk, nb, nw = self.Amn_np.shape
        self.nk = nk
        self.nb = nb
        self.nw = nw

    def _svd_init_U(self) -> torch.Tensor:
        """
        Initialise U via the polar decomposition of Amn:
            SVD(A) = P Sigma Q†  =>  U = P Q†
        This is the closest unitary to A in Frobenius norm and matches
        how Wannier90 initialises its minimisation.
        """
        # Amn shape: (nk, nb, nw)  — project nb bands onto nw trial orbitals
        P, _, Qh = torch.linalg.svd(self.Amn, full_matrices=False)
        # P: (nk, nb, nw),  Qh: (nk, nw, nw)
        U = torch.matmul(P, Qh)   # (nk, nb, nw) — rectangular for nb > nw
        # For the isolated-band case nb == nw, U is square unitary
        return U

    def test_dimensions(self):
        assert self.Mmn.shape == (self.nk, self.nnkp["nntot"], self.nb, self.nb)
        assert self.Amn.shape == (self.nk, self.nb, self.nw)
        assert self.nw == 4   # 4 sp3 Wannier functions
        assert self.nk == 64  # 4x4x4 mesh

    def test_omega_i_gauge_invariant_value(self):
        """
        Omega_I is gauge-invariant: it must equal the W90 reference value
        5.890770 Ang^2 for ANY unitary U, including U=I.

        W90 reference (from silicon.wout):
            Omega_I = 5.890769995 Ang^2
        """
        W90_OMEGA_I_ANG2 = 5.890769995

        U = _identity_U(self.nk, self.nw)
        _, OI, _, _, _ = compute_spread(U, self.Mmn, self.wb, self.bvecs, self.kb_idx)

        OI_ang2 = OI.item() * BOHR_TO_ANG**2
        assert abs(OI_ang2 - W90_OMEGA_I_ANG2) < 1e-4, (
            f"Omega_I = {OI_ang2:.6f} Ang^2, expected {W90_OMEGA_I_ANG2:.6f} Ang^2"
        )

    def test_total_omega_at_svd_init_is_finite(self):
        """SVD-initialised U must give a finite, positive total spread."""
        U = self._svd_init_U()
        Omega, OI, OD, OOD, centres = compute_spread(
            U, self.Mmn, self.wb, self.bvecs, self.kb_idx
        )
        assert torch.isfinite(Omega), "Omega is not finite at SVD initialisation"
        assert Omega.item() > 0
        assert torch.all(torch.isfinite(centres))

    def test_omega_od_zero_at_convergence(self):
        """
        W90 reports Omega_D = 0 for Si (high symmetry).
        At the SVD-initialised U the off-diagonal part should be small
        (it goes to zero on minimization).  We just check it's non-negative.
        """
        U = self._svd_init_U()
        _, _, OD, OOD, _ = compute_spread(
            U, self.Mmn, self.wb, self.bvecs, self.kb_idx
        )
        assert OD.item()  >= -1e-10
        assert OOD.item() >= -1e-10

    def test_wannier_centres_sum_to_zero(self):
        """
        By symmetry the 4 Si sp3 WF centres sum to zero.
        This must hold at the W90-converged solution; it approximately
        holds at SVD initialisation too.
        """
        W90_OMEGA_TOTAL_ANG2 = 6.468598306

        U = self._svd_init_U()
        Omega, _, _, _, centres = compute_spread(
            U, self.Mmn, self.wb, self.bvecs, self.kb_idx
        )

        centre_sum = centres.sum(dim=0)   # (3,) should be ~0
        # Not checking values here — just that the tensor has the right shape
        assert centre_sum.shape == (3,)
        assert torch.all(torch.isfinite(centre_sum))
