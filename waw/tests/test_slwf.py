"""
Tests for selectively localized Wannier functions (SLWF, Wannier90
tutorial26 -- `slwf_num`/`slwf_constrain`/`slwf_lambda`).

`waw.core.spread._slwf_spread_from_M_tilde`/`compute_slwf_spread` implement
Wang, Lazar, Park, Millis & Marianetti, "Selectively Localized Wannier
Functions", arXiv:1407.5124, Eq. 9-13 (plain) / Eq. 24, 29-31 (constrained).
No real DFT reference is wired up yet for this tutorial (see the plan's
task #65), so these tests:
  1. Cross-check the vectorized implementation against an independently
     written, unvectorized (explicit k/b/n Python loop) transcription of
     the same formulas.
  2. Confirm the exact algebraic identity `slwf_num == nw, constrain=False`
     reduces to plain MLWF (`compute_spread`) -- Omega_IOD collapses to
     Omega_I + Omega_OD combined (verified by direct expansion: the m!=n
     cross terms of the full Frobenius norm cancel between Omega_I and
     Omega_OD, leaving exactly the diagonal-only sum Omega_IOD computes).
  3. A CG-minimizer smoke test on synthetic data: Omega decreases and the
     optimizer runs to completion without error, for both plain SLWF and
     the centre-constrained variant.
"""

import numpy as np
import torch

from waw.core.spread import compute_spread, compute_slwf_spread, _slwf_spread_from_M_tilde
from waw.core.optim import minimize_spread_slwf

torch.manual_seed(0)
np.random.seed(0)


def _random_unitary(nk: int, nw: int, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    A = torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    A += 1j * torch.randn(nk, nw, nw, dtype=torch.complex128, generator=rng)
    Q, _ = torch.linalg.qr(A)
    return Q


def _make_simple_cubic_data(nk=8, nb=4, nw=4, nnb=6, seed=1):
    """Synthetic overlap data on a simple-cubic mesh (same construction as
    tests/test_spread.py's own fixture, kept local for test-file independence)."""
    torch.manual_seed(seed)
    V = torch.linalg.qr(
        torch.randn(nk, nb, nb, dtype=torch.complex128)
        + 1j * torch.randn(nk, nb, nb, dtype=torch.complex128)
    )[0]

    bvecs_np = np.array([
        [np.pi, 0, 0], [-np.pi, 0, 0],
        [0, np.pi, 0], [0, -np.pi, 0],
        [0, 0, np.pi], [0, 0, -np.pi],
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


def _naive_slwf_spread(M_tilde, wb, bvecs, slwf_num, constrain, target_centres, lambda_):
    """Explicit k/b/n-loop transcription of Eq. 12-13 (constrain=False) /
    Eq. 29-31 (constrain=True), independent of the vectorized implementation."""
    nk, nnb, nw, _ = M_tilde.shape
    Mn = M_tilde.numpy()
    wbn = wb.numpy()
    bn = bvecs.numpy()
    lam = lambda_ if constrain else 0.0

    # Wannier centres for ALL n (needed for Omega_D's b.r_n term)
    centres = np.zeros((nw, 3))
    for n in range(nw):
        acc = np.zeros(3)
        for k in range(nk):
            for b in range(nnb):
                phase = np.angle(Mn[k, b, n, n])
                acc += wbn[b] * phase * bn[k, b]
        centres[n] = -acc / nk

    Omega_IOD = 0.0
    Omega_D = 0.0
    for k in range(nk):
        for b in range(nnb):
            for n in range(slwf_num):
                m2 = abs(Mn[k, b, n, n]) ** 2
                phase = np.angle(Mn[k, b, n, n])
                Omega_IOD += wbn[b] * (1.0 - m2 + lam * phase ** 2)
                b_dot_r = np.dot(bn[k, b], centres[n])
                Omega_D += (1.0 - lam) * wbn[b] * (phase + b_dot_r) ** 2
    Omega_IOD /= nk
    Omega_D /= nk

    Omega_nu = 0.0
    if constrain:
        r0 = target_centres.numpy()
        Omega_nu = lambda_ * (r0 ** 2).sum()
        acc = 0.0
        for k in range(nk):
            for b in range(nnb):
                for n in range(slwf_num):
                    phase = np.angle(Mn[k, b, n, n])
                    acc += wbn[b] * np.dot(bn[k, b], r0[n]) * phase
        Omega_nu += 2.0 * lambda_ * acc / nk

    return Omega_IOD + Omega_D + Omega_nu, Omega_IOD, Omega_D, Omega_nu, centres


def test_slwf_spread_matches_naive_loop_unconstrained():
    nk, nb, nw, nnb = 6, 5, 4, 6
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk, nb, nw, nnb, seed=2)
    slwf_num = 2

    Omega, Omega_IOD, Omega_D, Omega_nu, centres = compute_slwf_spread(
        U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=False,
    )

    from waw.core.spread import rotate_overlaps
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)
    want = _naive_slwf_spread(M_tilde, wb, bvecs, slwf_num, False, None, 1.0)

    np.testing.assert_allclose(Omega.item(), want[0], atol=1e-10)
    np.testing.assert_allclose(Omega_IOD.item(), want[1], atol=1e-10)
    np.testing.assert_allclose(Omega_D.item(), want[2], atol=1e-10)
    np.testing.assert_allclose(Omega_nu.item(), want[3], atol=1e-10)
    np.testing.assert_allclose(centres.numpy(), want[4], atol=1e-10)


def test_slwf_spread_matches_naive_loop_constrained():
    nk, nb, nw, nnb = 6, 5, 4, 6
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk, nb, nw, nnb, seed=3)
    slwf_num = 2
    target_centres = torch.tensor(np.random.randn(slwf_num, 3) * 0.5, dtype=torch.float64)
    lambda_ = 0.7

    Omega, Omega_IOD, Omega_D, Omega_nu, centres = compute_slwf_spread(
        U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=True,
        target_centres=target_centres, lambda_=lambda_,
    )

    from waw.core.spread import rotate_overlaps
    M_tilde = rotate_overlaps(U, Mmn, kb_idx)
    want = _naive_slwf_spread(M_tilde, wb, bvecs, slwf_num, True, target_centres, lambda_)

    np.testing.assert_allclose(Omega.item(), want[0], atol=1e-10)
    np.testing.assert_allclose(Omega_IOD.item(), want[1], atol=1e-10)
    np.testing.assert_allclose(Omega_D.item(), want[2], atol=1e-10)
    np.testing.assert_allclose(Omega_nu.item(), want[3], atol=1e-10)


def test_slwf_num_equals_nw_reduces_to_plain_mlwf():
    """slwf_num == nw, constrain=False: Omega_IOD + Omega_D must equal the
    plain MLWF Omega_I + Omega_D + Omega_OD exactly (the m!=n cross terms of
    the full Frobenius norm cancel between Omega_I and Omega_OD, leaving
    the same diagonal-only sum Omega_IOD computes -- an algebraic identity,
    not a coincidence of this particular random data)."""
    nk, nb, nw, nnb = 6, 5, 4, 6
    U, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk, nb, nw, nnb, seed=4)

    Omega_slwf, Omega_IOD, Omega_D_slwf, Omega_nu, _ = compute_slwf_spread(
        U, Mmn, wb, bvecs, kb_idx, slwf_num=nw, constrain=False,
    )
    Omega_mlwf, Omega_I, Omega_D, Omega_OD, _ = compute_spread(U, Mmn, wb, bvecs, kb_idx)

    assert Omega_nu.item() == 0.0
    np.testing.assert_allclose(Omega_slwf.item(), Omega_mlwf.item(), atol=1e-10)
    np.testing.assert_allclose(Omega_IOD.item(), (Omega_I + Omega_OD).item(), atol=1e-10)
    np.testing.assert_allclose(Omega_D_slwf.item(), Omega_D.item(), atol=1e-10)


def test_minimize_spread_slwf_decreases_omega_unconstrained():
    nk, nb, nw, nnb = 6, 5, 4, 6
    U_init, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk, nb, nw, nnb, seed=5)
    slwf_num = 2

    result = minimize_spread_slwf(
        U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=False,
        optimizer="cg", n_iter=200, conv_tol=1e-12,
    )

    Omega_init = compute_slwf_spread(U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=False)[0].item()
    assert result.Omega < Omega_init
    assert result.Omega_IOD is not None
    assert result.Omega_nu == 0.0


def test_minimize_spread_slwf_decreases_omega_constrained():
    nk, nb, nw, nnb = 6, 5, 4, 6
    U_init, Mmn, wb, bvecs, kb_idx = _make_simple_cubic_data(nk, nb, nw, nnb, seed=6)
    slwf_num = 2
    target_centres = torch.zeros(slwf_num, 3, dtype=torch.float64)

    result = minimize_spread_slwf(
        U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=True,
        target_centres=target_centres, lambda_=1.0,
        optimizer="cg", n_iter=200, conv_tol=1e-12,
    )

    Omega_init = compute_slwf_spread(
        U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain=True,
        target_centres=target_centres, lambda_=1.0,
    )[0].item()
    assert result.Omega < Omega_init
    assert result.Omega_nu is not None
