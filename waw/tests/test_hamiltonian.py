"""
Tests for waw/hamiltonian.py — Part 8.

  1. H_R has the correct shape (nR, nw, nw).
  2. H_R is Hermitian: H(R)† = H(-R)   [time-reversal symmetry]
  3. H(R=0) is real and diagonal for an isolated, non-degenerate system.
  4. Round-trip: Σ_R e^{2πi k·R} H(R)/D(R) reproduces the original H_k at
     the original k-points.
  5. Wigner-Seitz degeneracies sum to Nk (completeness of WS cell).
  6. interpolate_bands at the original k-points reproduces the eigenvalues
     used to build H_R (to within the completeness of the supercell).
  7. interpolate_bands returns shape (nk_interp, nw).
  8. On Si: interpolated eigenvalues at the original k-points match the
     DFT eigenvalues to within the Wannier interpolation accuracy.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import (
    compute_hr, interpolate_bands, _wigner_seitz, HamiltonianR,
    compute_bb_r, compute_cc_r, operator_k, apply_scissors_shift,
)
from waw.core.init        import svd_init
from waw.core.optim       import minimize_spread
from waw.core.kmesh import _compute_bvecs_and_weights
from waw.core.spread import rotate_overlaps, weight_overlaps_by_eigenvalues
from waw.interfaces.wannier90.loader import parse_recip_lattice
from waw.interfaces.wannier90.io          import read_mmn, read_amn, read_nnkp, read_win, read_eig
from waw.units import ANG_TO_BOHR

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.mmn").exists()


# ===========================================================================
# Helpers / synthetic data
# ===========================================================================

def _make_simple_hr_data(nk=8, nw=2, mp_grid=(2, 2, 2), seed=0):
    """
    Synthetic Wannierized system: random unitary U, random real eigenvalues.
    Real lattice is cubic (a=5 Bohr).
    k-points: regular 2x2x2 grid in [0, 1).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    a = 5.0   # Bohr
    real_lattice = a * np.eye(3)

    # k-points on a 2x2x2 grid
    kpts_list = []
    N1, N2, N3 = mp_grid
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                kpts_list.append([i/N1, j/N2, k/N3])
    kpts_np = np.array(kpts_list, dtype=np.float64)
    kpts    = torch.tensor(kpts_np, dtype=torch.float64)

    # Random unitary U
    A = (torch.randn(nk, nw, nw, dtype=torch.float64)
         + 1j * torch.randn(nk, nw, nw, dtype=torch.float64))
    U, _ = torch.linalg.qr(A)

    # Random eigenvalues (sorted, in eV)
    eig_np = np.sort(rng.uniform(-5, 5, size=(nk, nw)), axis=1)
    eig    = torch.tensor(eig_np, dtype=torch.float64)

    return U, eig, kpts, kpts_np, mp_grid, real_lattice


# ===========================================================================
# Wigner-Seitz helper
# ===========================================================================

class TestWignerSeitz:
    def test_degen_sum_equals_nk(self):
        """
        Sum of 1/D(R) over all R must equal Nk.
        This ensures that the Fourier transform is normalized correctly.
        """
        mp_grid = (3, 3, 3)
        real_lattice = 5.0 * np.eye(3)
        R_arr, degen = _wigner_seitz(mp_grid, real_lattice)
        nk = mp_grid[0] * mp_grid[1] * mp_grid[2]
        assert R_arr.shape == (nk, 3)
        total = np.sum(1.0 / degen)
        np.testing.assert_allclose(total, nk, rtol=1e-10,
                                   err_msg="Σ 1/D(R) must equal Nk")

    def test_nR_at_least_nk_and_completeness(self):
        """
        The WS cell includes boundary R-vectors (shared between supercell images),
        so nR >= nk.  The completeness condition Σ 1/D(R) = nk must always hold.
        """
        for grid in [(2, 2, 2), (4, 4, 4), (2, 3, 4)]:
            R_arr, degen = _wigner_seitz(grid, 5.0 * np.eye(3))
            nk = grid[0] * grid[1] * grid[2]
            assert R_arr.shape[0] >= nk, (
                f"nR={R_arr.shape[0]} < nk={nk} for grid {grid}"
            )
            total = np.sum(1.0 / degen)
            np.testing.assert_allclose(total, nk, rtol=1e-10,
                                       err_msg=f"Σ 1/D ≠ nk for grid {grid}")

    def test_degen_positive(self):
        R_arr, degen = _wigner_seitz((2, 2, 2), 5.0 * np.eye(3))
        assert np.all(degen >= 1)

    def test_completeness_on_skewed_lattices(self):
        """
        Sum_R 1/D(R) == Nk on NON-CUBIC cells, which is where it can fail.

        The cubic checks above pass even with too small a search for
        equidistant supercell images, because on a cubic lattice the
        nearest images are all in the +-1 shell. On a skewed cell a
        SECOND-shell image can tie with the first: with a +-1 search the
        strained body-centred-tetragonal cell below (c/a = 1.4365, the
        notebook-13 Al) reported 216.16667 instead of 216, two of its 253
        R-vectors getting degeneracy 3 where the true value is 4. wannier90
        and EPW both search +-2 (EPW/src/wigner.f90).
        """
        a = 5.337733699980888
        c = 1.436488139922566 * a
        h, z = a / 2.0, c / 2.0
        lattices = {
            "bct(strained)": np.array([[h, -h, z], [h, h, z], [-h, -h, z]]),
            "fcc": 0.5 * 4.05 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            "bcc": 0.5 * 2.87 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
            "hex120": np.array([[3.086, 0.0, 0.0],
                                [-1.543, 2.672583, 0.0],
                                [0.0, 0.0, 3.524]]),
            "triclinic": np.array([[3.1, 0.0, 0.0], [1.4, 3.7, 0.0], [0.9, 1.1, 4.3]]),
            # Body-centred tetragonal, SLIGHTLY strained off c/a = sqrt(2)
            # (QE ibrav=7, celldm(3) = 1.4365 -- notebook 13's own Al cell).
            # This is the case that needs the +-2 supercell-image shell: under
            # +-1, R = (-6, 3, 3) and (6, -3, -3) came out with degeneracy 3
            # instead of 4 and the sum rule read 216.1667 rather than 216.
            # Every unstrained lattice above passes either way, which is
            # exactly why the shortfall went unnoticed.
            "bct_strained": 0.5 * 5.337733699980888 * np.array(
                [[1.0, -1.0, 1.436488139922566],
                 [1.0, 1.0, 1.436488139922566],
                 [-1.0, -1.0, 1.436488139922566]]),
        }
        for name, lat in lattices.items():
            for grid in [(3, 3, 3), (4, 4, 2), (6, 6, 4), (6, 6, 6)]:
                _, degen = _wigner_seitz(grid, lat)
                nk = grid[0] * grid[1] * grid[2]
                np.testing.assert_allclose(
                    np.sum(1.0 / degen), nk, rtol=1e-10,
                    err_msg=f"Sum 1/D != Nk for {name} on {grid}",
                )


# ===========================================================================
# compute_hr: structure tests
# ===========================================================================

class TestComputeHR:
    def test_shape(self):
        U, eig, kpts, _, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        nw = U.shape[1]
        nR = hr.H_R.shape[0]
        assert hr.H_R.shape == (nR, nw, nw)
        assert nR >= kpts.shape[0]   # WS cell includes boundary, so nR >= nk

    def test_returns_hamiltonian_r(self):
        U, eig, kpts, _, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        assert isinstance(hr, HamiltonianR)

    def test_H_R_dtype(self):
        U, eig, kpts, _, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        assert hr.H_R.is_complex()

    def test_hermiticity_H_R0(self):
        """H(R=0) must be Hermitian."""
        U, eig, kpts, _, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)

        # Find R=0 in R_vectors
        R0_idx = np.where((hr.R_vectors == 0).all(axis=1))[0]
        assert len(R0_idx) == 1, "R=0 not found in R_vectors"
        H_0 = hr.H_R[R0_idx[0]]   # (nw, nw)
        err = (H_0 - H_0.conj().T).norm().item()
        assert err < 1e-10, f"H(R=0) is not Hermitian: ||H - H†|| = {err:.2e}"

    def test_H_R0_real_part_is_avg_energy(self):
        """
        For a uniform U=I, H(R=0) = (1/Nk) Σ_k diag(ε_k), so the diagonal
        is the average of the eigenvalues.
        """
        nk, nw = 8, 2
        mp_grid = (2, 2, 2)
        real_lattice = 5.0 * np.eye(3)
        kpts_list = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    kpts_list.append([i/2, j/2, k/2])
        kpts = torch.tensor(kpts_list, dtype=torch.float64)

        torch.manual_seed(99)
        eig = torch.rand(nk, nw, dtype=torch.float64) * 10

        U = torch.eye(nw, dtype=torch.complex128).unsqueeze(0).expand(nk, -1, -1)
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)

        R0_idx = np.where((hr.R_vectors == 0).all(axis=1))[0][0]
        H0_diag = hr.H_R[R0_idx].diagonal().real.numpy()
        avg_eig  = eig.mean(dim=0).numpy()
        np.testing.assert_allclose(H0_diag, avg_eig, atol=1e-10)


# ===========================================================================
# Round-trip: Fourier transform and back
# ===========================================================================

class TestRoundTrip:
    def test_interpolate_at_original_kpts_matches_eigenvalues(self):
        """
        Interpolating bands at the original k-points must reproduce the
        DFT eigenvalues used to build H_R (up to ordering).
        """
        U, eig, kpts, kpts_np, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        bands = interpolate_bands(hr, kpts_np)   # (nk, nw)

        eig_np = np.sort(eig.numpy(), axis=1)   # eigenvalues are already in WF basis
        # Eigenvalues of H_k^{WF} = eigenvalues of diag(eig_k) (U is unitary)
        np.testing.assert_allclose(bands, eig_np, atol=1e-8,
                                   err_msg="Round-trip eigenvalues mismatch")

    def test_interpolate_shape(self):
        U, eig, kpts, kpts_np, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)

        kpath = np.linspace([0, 0, 0], [0.5, 0.5, 0.0], 20)
        bands = interpolate_bands(hr, kpath)
        assert bands.shape == (20, U.shape[1])

    def test_interpolate_bands_real(self):
        """Interpolated eigenvalues must be real (H is Hermitian)."""
        U, eig, kpts, kpts_np, mp_grid, real_lattice = _make_simple_hr_data()
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        kpath = np.random.default_rng(0).uniform(0, 1, (10, 3))
        bands = interpolate_bands(hr, kpath)
        # bands is already real numpy array; check dtype
        assert bands.dtype == np.float64 or np.issubdtype(bands.dtype, np.floating)


# ===========================================================================
# BB(R) / CC(R) — orbital-magnetization building blocks (tutorial19)
# ===========================================================================

def _make_bb_cc_data(nk=4, nnb=2, nb=3, nw=2, seed=0):
    """
    Synthetic disentangled system for `compute_bb_r`/`compute_cc_r`:
    random raw Mmn, ab-initio eigenvalues, disentanglement V, converged
    gauge U, a random neighbour-index table/b-vectors/weights, and a
    random raw `.uHu`-like tensor. Not physical (kb_idx/bvecs are not a
    real finite-difference shell) -- only used to check the R-space
    construction against an independent, explicit-loop reference.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mp_grid = (2, 2, 1)
    real_lattice = 5.0 * np.eye(3)

    kpts_np = np.array([[i / 2, j / 2, 0.0] for i in range(2) for j in range(2)])
    kpts = torch.tensor(kpts_np, dtype=torch.float64)

    kb_idx = torch.tensor(rng.integers(0, nk, size=(nk, nnb)), dtype=torch.long)
    wb = torch.tensor(rng.uniform(0.5, 1.5, size=nnb))
    bvecs = torch.tensor(rng.normal(size=(nk, nnb, 3)))

    Mmn = torch.tensor(rng.normal(size=(nk, nnb, nb, nb))
                        + 1j * rng.normal(size=(nk, nnb, nb, nb)))
    eig = torch.tensor(rng.uniform(-5, 5, size=(nk, nb)))
    V, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(nk, nb, nw))
                                        + 1j * rng.normal(size=(nk, nb, nw))))
    U, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(nk, nw, nw))
                                        + 1j * rng.normal(size=(nk, nw, nw))))
    W = torch.bmm(V, U)   # (nk, nb, nw) full converged gauge

    uHu = torch.tensor(rng.normal(size=(nk, nnb, nnb, nb, nb))
                        + 1j * rng.normal(size=(nk, nnb, nnb, nb, nb)))

    return dict(nk=nk, nnb=nnb, nb=nb, nw=nw, mp_grid=mp_grid, real_lattice=real_lattice,
                kpts=kpts, kpts_np=kpts_np, kb_idx=kb_idx, wb=wb, bvecs=bvecs,
                Mmn=Mmn, eig=eig, V=V, U=U, W=W, uHu=uHu)


class TestBBR:
    def test_round_trip_matches_explicit_loop_reference(self):
        """
        Fourier-transforming B_q -> BB(R) -> back to B(k) at the original
        k-points (via `operator_k`) must reproduce an explicit per-k,
        per-neighbour loop computed independently of `compute_bb_r`'s
        einsum -- catches index-order mistakes the einsum could silently
        get wrong (e.g. bra/ket or Cartesian-axis transposition).
        """
        d = _make_bb_cc_data()
        Mmn_w = weight_overlaps_by_eigenvalues(d["Mmn"], d["eig"])
        H_opt = rotate_overlaps(d["V"], Mmn_w, d["kb_idx"])
        H_tilde = rotate_overlaps(d["U"], H_opt, d["kb_idx"])

        B_q_ref = torch.zeros(d["nk"], 3, d["nw"], d["nw"], dtype=torch.complex128)
        for k in range(d["nk"]):
            for b in range(d["nnb"]):
                B_q_ref[k] += 1j * d["wb"][b] * d["bvecs"][k, b][:, None, None] * H_tilde[k, b]

        BB_R = compute_bb_r(H_tilde, d["wb"], d["bvecs"], d["kpts"], d["mp_grid"], d["real_lattice"])
        R_arr, degen = _wigner_seitz(d["mp_grid"], d["real_lattice"])
        B_q_rt = torch.stack(
            [operator_k(BB_R[a], R_arr, degen, d["kpts_np"]) for a in range(3)], dim=1
        )
        torch.testing.assert_close(B_q_rt, B_q_ref, atol=1e-10, rtol=0)

    def test_shape(self):
        d = _make_bb_cc_data()
        Mmn_w = weight_overlaps_by_eigenvalues(d["Mmn"], d["eig"])
        H_opt = rotate_overlaps(d["V"], Mmn_w, d["kb_idx"])
        H_tilde = rotate_overlaps(d["U"], H_opt, d["kb_idx"])
        BB_R = compute_bb_r(H_tilde, d["wb"], d["bvecs"], d["kpts"], d["mp_grid"], d["real_lattice"])
        R_arr, _ = _wigner_seitz(d["mp_grid"], d["real_lattice"])
        assert BB_R.shape == (3, len(R_arr), d["nw"], d["nw"])


class TestCCR:
    def test_round_trip_matches_explicit_loop_reference(self):
        """Same style of check as `TestBBR`, but for the double-neighbour CC(R)."""
        d = _make_bb_cc_data()
        C_q_ref = torch.zeros(d["nk"], 3, 3, d["nw"], d["nw"], dtype=torch.complex128)
        for k in range(d["nk"]):
            for p in range(d["nnb"]):
                kp = d["kb_idx"][k, p].item()
                for q in range(d["nnb"]):
                    kq = d["kb_idx"][k, q].item()
                    Ht = d["W"][kp].conj().T @ d["uHu"][k, p, q] @ d["W"][kq]
                    for a in range(3):
                        for b in range(3):
                            C_q_ref[k, a, b] += (
                                d["wb"][p] * d["bvecs"][k, p, a]
                                * d["wb"][q] * d["bvecs"][k, q, b] * Ht
                            )

        CC_R = compute_cc_r(d["uHu"], d["W"], d["kb_idx"], d["wb"], d["bvecs"],
                            d["kpts"], d["mp_grid"], d["real_lattice"])
        R_arr, degen = _wigner_seitz(d["mp_grid"], d["real_lattice"])
        C_q_rt = torch.stack([
            torch.stack([operator_k(CC_R[a, b], R_arr, degen, d["kpts_np"]) for b in range(3)], dim=1)
            for a in range(3)
        ], dim=1)
        torch.testing.assert_close(C_q_rt, C_q_ref, atol=1e-10, rtol=0)

    def test_shape(self):
        d = _make_bb_cc_data()
        CC_R = compute_cc_r(d["uHu"], d["W"], d["kb_idx"], d["wb"], d["bvecs"],
                            d["kpts"], d["mp_grid"], d["real_lattice"])
        R_arr, _ = _wigner_seitz(d["mp_grid"], d["real_lattice"])
        assert CC_R.shape == (3, 3, len(R_arr), d["nw"], d["nw"])

    def test_hermitian_pair_symmetry(self):
        """
        A physical `.uHu` satisfies uHu[k,q,p] = uHu[k,p,q]^dagger (H_k is
        Hermitian), which forces CC(R)_ba = CC(R)_ab^dagger after Fourier
        transform. Build a uHu obeying that and check the derived CC(R).
        """
        d = _make_bb_cc_data()
        uHu = d["uHu"].clone()
        nk, nnb = d["nk"], d["nnb"]
        for k in range(nk):
            for p in range(nnb):
                for q in range(p + 1, nnb):
                    uHu[k, q, p] = uHu[k, p, q].conj().transpose(-1, -2)
        for k in range(nk):
            for p in range(nnb):
                uHu[k, p, p] = 0.5 * (uHu[k, p, p] + uHu[k, p, p].conj().transpose(-1, -2))

        CC_R = compute_cc_r(uHu, d["W"], d["kb_idx"], d["wb"], d["bvecs"],
                            d["kpts"], d["mp_grid"], d["real_lattice"])
        for a in range(3):
            for b in range(3):
                torch.testing.assert_close(
                    CC_R[b, a], CC_R[a, b].conj().transpose(-1, -2), atol=1e-10, rtol=0,
                )


# ===========================================================================
# Silicon reference (slow, uses real data)
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
class TestSiliconHamiltonian:
    """
    Compute H(R) for Si after convergence and verify that band interpolation
    at the original k-points reproduces the DFT eigenvalues.
    """

    @pytest.fixture(autouse=True)
    def load_si(self):
        self.Mmn_np, _ = read_mmn(SI_DIR / "silicon.mmn")
        self.Amn_np    = read_amn(SI_DIR / "silicon.amn")
        self.eig_np    = read_eig(SI_DIR / "silicon.eig")   # (nk, nb)
        self.nnkp      = read_nnkp(SI_DIR / "silicon.nnkp")
        self.params    = read_win(SI_DIR / "silicon.win")
        recip = parse_recip_lattice(self.params)
        bvecs_np, wb_np = _compute_bvecs_and_weights(
            self.nnkp["kpoints"], self.nnkp["nnkpts"],
            self.nnkp["g_vectors"], recip,
        )
        self.Mmn    = torch.tensor(self.Mmn_np,   dtype=torch.complex128)
        self.Amn    = torch.tensor(self.Amn_np,   dtype=torch.complex128)
        self.bvecs  = torch.tensor(bvecs_np,       dtype=torch.float64)
        self.wb     = torch.tensor(wb_np,           dtype=torch.float64)
        self.kb_idx = torch.tensor(self.nnkp["nnkpts"], dtype=torch.long)
        # real lattice from win (Bohr)
        cell_lines = self.params.get("unit_cell_cart", [])
        unit = "bohr"
        data_lines = cell_lines
        if cell_lines[0].strip().lower() in ("ang", "angstrom"):
            unit = "ang"; data_lines = cell_lines[1:]
        elif cell_lines[0].strip().lower() == "bohr":
            data_lines = cell_lines[1:]
        lat = np.array([[float(x) for x in l.split()[:3]] for l in data_lines[:3]])
        if unit == "ang":
            lat *= ANG_TO_BOHR
        self.real_lattice = lat

    def test_interpolated_bands_close_to_dft(self):
        """
        After converging U, interpolated bands at original k-points must match
        the valence DFT eigenvalues within the Wannier interpolation accuracy.
        For isolated valence bands of Si the interpolation is essentially exact
        (< 1 meV error).
        """
        nw = 4
        U_init = svd_init(self.Amn)
        result = minimize_spread(
            U_init, self.Mmn, self.wb, self.bvecs, self.kb_idx,
            optimizer="adam", lr=3e-2, n_iter=1000,
        )
        U = result.U_final   # (64, 4, 4)

        # eig has shape (64, 4) — valence bands only
        eig_v = torch.tensor(self.eig_np[:, :nw], dtype=torch.float64)
        hr = compute_hr(U, eig_v, torch.tensor(self.nnkp["kpoints"]),
                        mp_grid=(4, 4, 4), real_lattice=self.real_lattice)

        bands = interpolate_bands(hr, self.nnkp["kpoints"])   # (64, 4)

        # Compare to sorted DFT eigenvalues
        eig_sorted = np.sort(self.eig_np[:, :nw], axis=1)
        max_err = np.abs(bands - eig_sorted).max()
        assert max_err < 0.01, (
            f"Max band interpolation error: {max_err*1000:.2f} meV (threshold 10 meV)"
        )


class TestApplyScissorsShift:
    """
    `apply_scissors_shift` transcribes wannier90's own scissors-correction
    block (`get_oper.F90::get_HH_R`, lines ~252-278): H(R) +=
    scissors_shift * P_conduction(R). Since P_conduction(k) commutes with
    H(k) (same eigenbasis U(k) by construction), this must (a) shift
    EXACTLY the top `nw-num_valence_bands` eigenvalues by `scissors_shift`
    at every k, leaving the rest untouched, and (b) leave H(k)'s
    eigenvectors completely unchanged (up to an overall numerical-noise-
    level phase/degenerate-subspace freedom) -- verified directly, not
    assumed.
    """

    def _synthetic_system(self, seed=0):
        torch.manual_seed(seed)
        nk, nw = 5, 4
        mp_grid = (5, 1, 1)
        real_lattice = 10.0 * np.eye(3)
        kpts = torch.tensor(np.stack(
            [np.arange(nk) / nk, np.zeros(nk), np.zeros(nk)], axis=1))
        U, _ = torch.linalg.qr(torch.randn(nk, nw, nw, dtype=torch.complex128))
        eig = torch.sort(torch.randn(nk, nw, dtype=torch.float64) * 2, dim=1).values
        hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
        return hr, U, eig, kpts, mp_grid, real_lattice

    def test_eigenvalues_shift_only_conduction_bands(self):
        hr, U, eig, kpts, mp_grid, real_lattice = self._synthetic_system()
        num_valence, scissors = 2, 0.5

        hr_shifted = apply_scissors_shift(hr, U, kpts, mp_grid, real_lattice,
                                          num_valence, scissors)

        bands0 = interpolate_bands(hr, kpts.numpy())
        bands1 = interpolate_bands(hr_shifted, kpts.numpy())
        diff = bands1 - bands0

        np.testing.assert_allclose(diff[:, :num_valence], 0.0, atol=1e-10)
        np.testing.assert_allclose(diff[:, num_valence:], scissors, atol=1e-10)

    def test_eigenvectors_unchanged(self):
        """H(k)'s eigenvectors (hence the position operator/D_h/anything
        wavefunction-derived) must be IDENTICAL before and after the
        scissors shift, up to a phase per band -- verified by checking
        that projectors |n><n| built from each eigenvector match."""
        hr, U, eig, kpts, mp_grid, real_lattice = self._synthetic_system(seed=1)
        num_valence, scissors = 2, 0.7
        hr_shifted = apply_scissors_shift(hr, U, kpts, mp_grid, real_lattice,
                                          num_valence, scissors)

        R_arr, degen = _wigner_seitz(mp_grid, real_lattice)
        Hk0 = operator_k(hr.H_R, R_arr, degen, kpts.numpy())
        Hk1 = operator_k(hr_shifted.H_R, R_arr, degen, kpts.numpy())

        _, U0 = torch.linalg.eigh(Hk0)
        _, U1 = torch.linalg.eigh(Hk1)

        # per-band projector |n><n|, phase-independent
        P0 = torch.einsum('kin,kjn->knij', U0, U0.conj())
        P1 = torch.einsum('kin,kjn->knij', U1, U1.conj())
        torch.testing.assert_close(P0, P1, atol=1e-8, rtol=0)

    def test_does_not_mutate_input(self):
        hr, U, eig, kpts, mp_grid, real_lattice = self._synthetic_system(seed=2)
        H_R_before = hr.H_R.clone()
        apply_scissors_shift(hr, U, kpts, mp_grid, real_lattice, 2, 1.0)
        torch.testing.assert_close(hr.H_R, H_R_before, atol=0, rtol=0)


def test_wigner_seitz_degeneracies_are_exact_tie_counts_hexagonal():
    """Brute-force check of _wigner_seitz on a HEXAGONAL cell (trigonal Te,
    3x3x4), the case where stock wannier90 3.1 gets it wrong.

    For every R in the WS set, the degeneracy must equal the number of
    BvK images of R that are EXACTLY tied for minimum length. On this cell
    the ties are unambiguous -- e.g. R=(-1,-2,-2) has 6 images equal to
    <1e-9 Bohr with the next shell 10 Bohr farther -- so this is not a
    tolerance judgement.

    Certified against real wannier90 3.1 (2026-07-28): its own te_hr.dat
    for this model carries 45 R-vectors where waw finds 65 (waw's set is a
    strict superset), and 10 shared R have w90 degeneracies smaller than
    the true tie count by exactly the C3 factor of 3 (e.g. 2 vs 6, 1 vs 3).
    Both sets satisfy sum_R 1/degen == N1*N2*N3, so w90's is a consistent
    but SYMMETRY-BREAKING partition; the consequence is that w90/postw90
    interpolate this model differently off-mesh (bands differ by up to
    443 meV on an 8^3 mesh, DOS by 35%). `ws_search_size` does not fix it.
    Same bug class as the one EPW patches with its +-2 image search.
    """
    from ase import Atoms

    from waw.interfaces.ase.structure import real_lattice

    a, c = 4.457, 5.9581176
    atoms = Atoms('Te3',
                  scaled_positions=[[0.274036, 0.274036, 0.0],
                                    [-0.274036, 0.0, 1 / 3],
                                    [0.0, -0.274036, 2 / 3]],
                  cell=[[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]],
                  pbc=True)
    A = real_lattice(atoms)
    mp = np.array([3, 3, 4])
    R_arr, degen = _wigner_seitz(tuple(mp), A)

    assert abs(np.sum(1.0 / degen) - np.prod(mp)) < 1e-10   # partition of unity

    shifts = np.array([(i, j, k) for i in range(-3, 4) for j in range(-3, 4)
                       for k in range(-3, 4)])
    for R, d in zip(R_arr, degen):
        lens = np.linalg.norm((R[None, :] + shifts * mp[None, :]) @ A, axis=1)
        lmin = lens.min()
        n_tied = int((lens < lmin + 1e-9).sum())
        assert d == n_tied, (tuple(R), d, n_tied)
        # R must itself be one of the minimal-length images
        assert np.linalg.norm(R @ A) < lmin + 1e-9, tuple(R)
