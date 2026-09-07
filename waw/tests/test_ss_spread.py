"""
Tests for waw.core.spread's Stengel-Spaldin (SS) alternative localization
functional (`_ss_spread_from_M_tilde`/`compute_ss_spread`), transcribed
from wannierise.F90::wann_omega's "not selective localisation" branch
(lines 2290-2392 -- the formula actually entering the minimized total,
NOT the earlier rave/sheet-based section used only for centre reporting).

No claim is made here about Omega_D_ss vs Omega_D_mv ordering in general
(an earlier, WRONG reading of a different -- reporting-only -- Fortran
section suggested Omega_ss <= Omega_mv always; the authoritative formula
transcribed here is a genuinely different quantity -- a k-averaged
variance of M_nn, not a residual-sum-of-squares around a linear fit -- so
no generic inequality between them is asserted or tested).
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.spread import (
    rotate_overlaps, compute_spread, compute_ss_spread,
    _ss_spread_from_M_tilde, _canonical_b_permutation,
)


def _random_unitary(nk: int, nw: int, seed: int = 0) -> torch.Tensor:
    """Random unitary matrices via QR decomposition."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    A = torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    A += 1j * torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q


def _make_simple_cubic_data(nk=8, nb=4, nw=2, nnb=6, seed=1):
    """Synthetic data on a simple-cubic mesh (same construction as
    test_spread.py's own helper of the same name -- duplicated here rather
    than cross-imported, matching this project's convention of each test
    file defining its own small synthetic-model helpers)."""
    torch.manual_seed(seed)
    V = torch.linalg.qr(
        torch.randn(nk, nb, nb, dtype=torch.complex128)
        + 1j * torch.randn(nk, nb, nb, dtype=torch.complex128)
    )[0]

    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    wb_np = np.full(6, 1.0 / (2 * np.pi ** 2))
    bvecs = torch.tensor(np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64)
    wb = torch.tensor(wb_np, dtype=torch.float64)

    kb_idx = torch.zeros(nk, nnb, dtype=torch.long)
    for ik in range(nk):
        for ib in range(nnb):
            kb_idx[ik, ib] = (ik + ib // 2 + 1) % nk

    Mmn = torch.zeros(nk, nnb, nb, nb, dtype=torch.complex128)
    for ik in range(nk):
        for ib in range(nnb):
            ik2 = kb_idx[ik, ib].item()
            Mmn[ik, ib] = V[ik].conj().T @ V[ik2]

    U = _random_unitary(nk, nw, seed=seed)
    return U, Mmn[:, :, :nw, :nw], wb, bvecs, kb_idx


def test_omega_i_and_omega_od_match_ordinary_mv():
    """Omega_I/Omega_OD are defined identically for SS and MV -- same
    M_tilde must give bit-identical values regardless of which Omega_D
    formula is used."""
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=8, nb=4, nw=2)
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)

    _, Omega_I_mv, _, Omega_OD_mv, _ = compute_spread(U, Mmn, wb, bvecs, kb_idx)
    _, Omega_I_ss, _, Omega_OD_ss = _ss_spread_from_M_tilde(M_tilde, wb, bvecs)

    assert Omega_I_ss.item() == pytest.approx(Omega_I_mv.item(), abs=1e-12)
    assert Omega_OD_ss.item() == pytest.approx(Omega_OD_mv.item(), abs=1e-12)


def test_omega_d_ss_is_nonnegative():
    """Omega_D_ss is a sum of weighted variances -- must be >= 0 always
    (Cauchy-Schwarz: <|X|^2> >= |<X>|^2 for any complex random variable)."""
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=8, nb=4, nw=2, seed=7)
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)
    _, _, Omega_D_ss, _ = _ss_spread_from_M_tilde(M_tilde, wb, bvecs)
    assert Omega_D_ss.item() >= -1e-12


def test_omega_d_ss_vanishes_when_m_nn_is_k_independent():
    """If M_nn(k,b) is IDENTICAL across all k (for every b), its k-variance
    is exactly zero by construction -- Omega_D_ss must vanish."""
    nk, nnb, nw = 6, 6, 2
    torch.manual_seed(3)
    # Same diagonal M_nn value at every k (but nonzero off-diagonal noise,
    # which Omega_D_ss doesn't touch anyway -- only the diagonal enters it).
    M_diag_const = torch.randn(nnb, nw, dtype=torch.complex128)
    M_diag_const = M_diag_const / M_diag_const.abs()  # unit modulus, like a real M_nn
    M_tilde = torch.zeros(nk, nnb, nw, nw, dtype=torch.complex128)
    for k in range(nk):
        for b in range(nnb):
            M_tilde[k, b] = torch.diag(M_diag_const[b])

    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    bvecs = torch.tensor(np.tile(bvecs_np[None, :, :], (nk, 1, 1)), dtype=torch.float64)
    wb = torch.full((nnb,), 1.0 / (2 * np.pi ** 2), dtype=torch.float64)

    _, _, Omega_D_ss, _ = _ss_spread_from_M_tilde(M_tilde, wb, bvecs)
    assert Omega_D_ss.item() == pytest.approx(0.0, abs=1e-10)


def test_canonical_b_permutation_undoes_a_known_shuffle():
    """A deliberately-shuffled per-k b-vector ordering (same physical set,
    permuted) must be reindexed back to the k=0 canonical order."""
    nk, nnb = 5, 6
    bvecs_np = np.array([
        [ np.pi, 0, 0], [-np.pi, 0, 0],
        [0,  np.pi, 0], [0, -np.pi, 0],
        [0, 0,  np.pi], [0, 0, -np.pi],
    ])
    rng = np.random.default_rng(11)
    bvecs_shuffled = np.zeros((nk, nnb, 3))
    true_perm = np.zeros((nk, nnb), dtype=np.int64)
    bvecs_shuffled[0] = bvecs_np
    true_perm[0] = np.arange(nnb)
    for k in range(1, nk):
        p = rng.permutation(nnb)
        bvecs_shuffled[k] = bvecs_np[p]
        # true_perm[k, s] should recover the local index of canonical dir s
        inv = np.zeros(nnb, dtype=np.int64)
        inv[p] = np.arange(nnb)
        true_perm[k] = inv

    bvecs = torch.tensor(bvecs_shuffled, dtype=torch.float64)
    perm = _canonical_b_permutation(bvecs)
    np.testing.assert_array_equal(perm.numpy(), true_perm)


def test_omega_d_ss_invariant_to_per_k_b_vector_shuffle():
    """Physically, Omega_D_ss must not depend on which arbitrary per-k
    index a given physical b-vector happens to sit at -- shuffling bvecs
    (and M_tilde's b-axis identically) must leave Omega_D_ss unchanged."""
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=8, nb=4, nw=2, seed=5)
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)
    _, _, Omega_D_orig, _ = _ss_spread_from_M_tilde(M_tilde, wb, bvecs)

    nk, nnb = bvecs.shape[0], bvecs.shape[1]
    rng = np.random.default_rng(23)
    M_tilde_shuf = M_tilde.clone()
    bvecs_shuf = bvecs.clone()
    for k in range(nk):
        p = torch.from_numpy(rng.permutation(nnb))
        M_tilde_shuf[k] = M_tilde[k, p]
        bvecs_shuf[k] = bvecs[k, p]

    _, _, Omega_D_shuf, _ = _ss_spread_from_M_tilde(M_tilde_shuf, wb, bvecs_shuf)
    assert Omega_D_shuf.item() == pytest.approx(Omega_D_orig.item(), abs=1e-10)


def test_compute_ss_spread_matches_from_m_tilde():
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=8, nb=4, nw=2, seed=9)
    Omega1, OI1, OD1, OOD1 = compute_ss_spread(U, Mmn, wb, bvecs, kb_idx)
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)
    Omega2, OI2, OD2, OOD2 = _ss_spread_from_M_tilde(M_tilde, wb, bvecs)
    assert Omega1.item() == pytest.approx(Omega2.item())
    assert OI1.item() == pytest.approx(OI2.item())
    assert OD1.item() == pytest.approx(OD2.item())
    assert OOD1.item() == pytest.approx(OOD2.item())


def test_ss_gradient_matches_finite_difference():
    """
    Autodiff correctness check -- the whole point of not hand-deriving a
    gradient. Same element-perturbation pattern as
    test_spread.py::test_autograd_vs_finite_difference.
    """
    nk, nw = 6, 3
    eps = 1e-5
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk=nk, nb=3, nw=nw, seed=13)
    U = U.detach().requires_grad_(True)

    Omega, *_ = compute_ss_spread(U, Mmn, wb, bvecs, kb_idx)
    Omega.backward()
    grad = U.grad.clone()

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
                        O_p = compute_ss_spread(U_p, Mmn, wb, bvecs, kb_idx)[0]
                        O_m = compute_ss_spread(U_m, Mmn, wb, bvecs, kb_idx)[0]
                    fd = (O_p - O_m).item() / (2 * eps)

                    ag = grad[ik, i, j].real.item() if part == "real" else grad[ik, i, j].imag.item()
                    errors.append(abs(fd - ag))

    assert max(errors) < 1e-5


SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.mmn").exists()


@pytest.mark.skipif(not HAS_SI, reason="Silicon example files not found")
def test_end_to_end_wannierize_use_ss_functional():
    """
    Full-plumbing smoke test: `interfaces.ase.driver.wannierize(...,
    use_ss_functional=True)` on a real (committed) Si isolated-band overlap
    set -- `use_ss_functional` is forwarded via that driver's `**optim_kwargs`
    straight through to `core.pipeline.wannierize`/`core.optim.
    minimize_spread`, no explicit driver-level plumbing needed.
    """
    from ase import Atoms
    from waw.interfaces.ase import driver
    from waw.interfaces.wannier90.io import read_win, read_nnkp, read_mmn, read_amn, read_eig
    from waw.interfaces.wannier90.loader import parse_real_lattice
    from waw.units import BOHR_TO_ANG

    win = read_win(SI_DIR / "silicon.win")
    real_bohr = parse_real_lattice(win)
    atoms = Atoms("Si2", cell=real_bohr * BOHR_TO_ANG, pbc=True)

    nnkp = read_nnkp(SI_DIR / "silicon.nnkp")
    mmn, _ = read_mmn(SI_DIR / "silicon.mmn")
    amn = read_amn(SI_DIR / "silicon.amn")
    eig = read_eig(SI_DIR / "silicon.eig")
    mp_grid = tuple(int(x) for x in str(win["mp_grid"]).split())

    result = driver.wannierize(
        atoms, mp_grid, nnkp["kpoints"],
        mmn=mmn, amn=amn, eig=eig,
        nnkpts=nnkp["nnkpts"], g_vectors=nnkp["g_vectors"],
        n_iter=200, verbose=False, use_ss_functional=True,
    )
    assert np.isfinite(result.omega_final)
    assert result.omega_final > 0
    assert result.centres_bohr.shape == (int(win["num_wann"]), 3)
    # Regression: WannierResult.omega_final and per-WF spreads_bohr2 must
    # both report the SS functional's own Omega (not silently fall back to
    # the ordinary MV one) -- caught as a real bug while building this
    # notebook: omega_final was unconditionally MV-formula until fixed.
    assert result.spreads_bohr2.sum() == pytest.approx(result.omega_final, rel=1e-6)
