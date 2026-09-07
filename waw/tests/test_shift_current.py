"""
Unit tests for waw.analysis.shift_current's Phase C building blocks --
`_sum_AD_HD`, `_eta_correction_term`, `_generalized_derivative`.

These tests cross-check the vectorized torch implementation against an
independently written, unvectorized (explicit n/m/p Python loop)
transcription of the SAME formulas read directly from wannier90's
berry_get_sc_klist -- this catches indexing/broadcasting bugs distinct
from formula-derivation bugs (both were checked against the Fortran
source directly, see shift_current.py's module docstring).
"""

import numpy as np
import pytest
import torch

from waw.analysis.shift_current import (
    _sum_AD_HD, _eta_correction_term, _generalized_derivative, _accumulate_sc_k,
    _kmesh_spacing, _adaptive_eta, shift_current_tensor, ALPHA_S, BETA_S,
)
from waw.core.distributions import gaussian_smearing
from waw.core.hamiltonian import HamiltonianR
from waw.units import EV_TO_HARTREE

torch.manual_seed(0)
np.random.seed(0)

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u: float) -> HamiltonianR:
    """Same 2-band QWZ model as test_hamiltonian_gauge_position.py."""
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), 2, 2), dtype=np.complex128)
    H_R[0] = (-1j / 2) * SX + 0.5 * SZ
    H_R[1] = (1j / 2) * SX + 0.5 * SZ
    H_R[2] = (-1j / 2) * SY + 0.5 * SZ
    H_R[3] = (1j / 2) * SY + 0.5 * SZ
    H_R[4] = u * SZ

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=2)


def _synthetic_AA_R(hr: HamiltonianR, seed: int = 0) -> torch.Tensor:
    """A random Hermitian-per-component AA_R on the same R-grid as hr, just
    to exercise the rotation machinery -- no physical meaning needed for
    this smoke test."""
    rng = np.random.default_rng(seed)
    nR, nw = hr.H_R.shape[0], hr.nw
    raw = rng.normal(size=(3, nR, nw, nw)) + 1j * rng.normal(size=(3, nR, nw, nw))
    raw = 0.5 * (raw + raw.conj().transpose(0, 1, 3, 2))
    return torch.tensor(raw, dtype=torch.complex128)


def test_shift_current_tensor_runs_and_is_finite():
    """Smoke test on a synthetic 2-band QWZ model (no real DFT reference
    exists yet for tutorial 25/GaAs) -- confirms the full pipeline wiring
    (Phase A/B/C building blocks + k-sum + unit conversion) runs end to end
    and produces finite, real-valued output of the expected shape."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr)
    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    omega = np.linspace(-2.0, 2.0, 5) * EV_TO_HARTREE

    sigma = shift_current_tensor(
        hr, AA_R, recip_lattice, real_lattice, mesh=(4, 4, 1),
        fermi_energy=0.0, omega=omega, eta=0.1 * EV_TO_HARTREE, sc_eta=0.08 * EV_TO_HARTREE,
    )

    assert sigma.shape == (3, 6, 5)
    assert np.all(np.isfinite(sigma))


def test_shift_current_tensor_with_eta_corr_runs_and_is_finite():
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr)
    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    omega = np.linspace(-2.0, 2.0, 5) * EV_TO_HARTREE

    sigma = shift_current_tensor(
        hr, AA_R, recip_lattice, real_lattice, mesh=(4, 4, 1),
        fermi_energy=0.0, omega=omega, eta=0.1 * EV_TO_HARTREE, sc_eta=0.08 * EV_TO_HARTREE,
        sc_use_eta_corr=True,
    )

    assert sigma.shape == (3, 6, 5)
    assert np.all(np.isfinite(sigma))


def _random_complex(*shape):
    return torch.tensor(
        np.random.randn(*shape) + 1j * np.random.randn(*shape), dtype=torch.complex128
    )


def _naive_sum_AD_HD(X_bar, D_h_eta):
    nk, _, nw, _ = X_bar.shape
    Xn = X_bar.numpy()
    Dn = D_h_eta.numpy()
    out = np.zeros((nk, 3, 3, nw, nw), dtype=complex)
    for k in range(nk):
        for c in range(3):
            for a in range(3):
                for n in range(nw):
                    for m in range(nw):
                        s1 = sum(Xn[k, c, n, p] * Dn[k, a, p, m] for p in range(nw)) - Xn[k, c, n, n] * Dn[k, a, n, m]
                        s2 = sum(Dn[k, a, n, p] * Xn[k, c, p, m] for p in range(nw)) - Dn[k, a, n, m] * Xn[k, c, m, m]
                        out[k, c, a, n, m] = s1 - s2
    return out


def test_sum_AD_HD_matches_naive_loop():
    nk, nw = 2, 4
    X_bar = _random_complex(nk, 3, nw, nw)
    D_h_eta = _random_complex(nk, 3, nw, nw)

    got = _sum_AD_HD(X_bar, D_h_eta).numpy()
    want = _naive_sum_AD_HD(X_bar, D_h_eta)
    np.testing.assert_allclose(got, want, atol=1e-10)


def _naive_eta_correction(AA_bar, HH_da_bar, eig, a, c, sc_eta):
    nk, nw = eig.shape
    AAn = AA_bar.numpy()
    HHn = HH_da_bar.numpy()
    eign = eig.numpy()
    out = np.zeros((nk, nw, nw), dtype=complex)
    for k in range(nk):
        for n in range(nw):
            for m in range(nw):
                if n == m:
                    continue
                total = 0.0 + 0.0j
                for p in range(nw):
                    if p == n or p == m:
                        continue
                    denom1 = sc_eta ** 2 / ((eign[k, p] - eign[k, m]) ** 2 + sc_eta ** 2)
                    denom2 = sc_eta ** 2 / ((eign[k, n] - eign[k, p]) ** 2 + sc_eta ** 2)
                    common = 1.0 / (eign[k, n] - eign[k, m])
                    bracket1 = AAn[k, c, n, p] * HHn[k, a, p, m] - (
                        HHn[k, c, n, p] + 1j * (eign[k, n] - eign[k, p]) * AAn[k, c, n, p]
                    ) * AAn[k, a, p, m]
                    bracket2 = HHn[k, a, n, p] * AAn[k, c, p, m] - AAn[k, a, n, p] * (
                        HHn[k, c, p, m] + 1j * (eign[k, p] - eign[k, m]) * AAn[k, c, p, m]
                    )
                    total += common * (-denom1 * bracket1 + denom2 * bracket2)
                out[k, n, m] = total
    return out


def test_eta_correction_matches_naive_loop():
    nk, nw = 2, 5
    AA_bar = _random_complex(nk, 3, nw, nw)
    HH_da_bar = _random_complex(nk, 3, nw, nw)
    eig = torch.tensor(np.sort(np.random.randn(nk, nw) * 2.0, axis=1))
    sc_eta = 0.05

    off_diag = ~np.eye(nw, dtype=bool)
    for a in range(3):
        for c in range(3):
            got = _eta_correction_term(AA_bar, HH_da_bar, eig, a, c, sc_eta).numpy()
            want = _naive_eta_correction(AA_bar, HH_da_bar, eig, a, c, sc_eta)
            np.testing.assert_allclose(got[:, off_diag], want[:, off_diag], atol=1e-10)


def _naive_generalized_derivative(
    AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta, eig, eig_da,
    sc_use_eta_corr, sc_eta,
):
    nk, nw = eig.shape
    AAn = AA_bar.numpy()
    AAda = AA_da_bar.numpy()
    HHda = HH_da_bar.numpy()
    HHdadb = HH_dadb_bar.numpy()
    Dno = D_h_no_eta.numpy()
    Deta = D_h_eta.numpy()
    eign = eig.numpy()
    eigda_n = eig_da.numpy()

    sum_AD = _naive_sum_AD_HD(AA_bar, D_h_eta)
    sum_HD = _naive_sum_AD_HD(HH_da_bar, D_h_eta)

    out = np.zeros((nk, 3, 3, nw, nw), dtype=complex)
    for k in range(nk):
        for a in range(3):
            for c in range(3):
                for n in range(nw):
                    for m in range(nw):
                        if n == m:
                            val = (
                                AAda[k, c, a, n, m]
                                + (AAn[k, c, n, n] - AAn[k, c, m, m]) * Dno[k, a, n, m]
                                + (AAn[k, a, n, n] - AAn[k, a, m, m]) * Dno[k, c, n, m]
                                - 1j * AAn[k, c, n, m] * (AAn[k, a, n, n] - AAn[k, a, m, m])
                                + sum_AD[k, c, a, n, m]
                            )
                        else:
                            inner = (
                                HHdadb[k, c, a, n, m]
                                + sum_HD[k, c, a, n, m]
                                + Dno[k, c, n, m] * (eigda_n[k, n, a] - eigda_n[k, m, a])
                                + Dno[k, a, n, m] * (eigda_n[k, n, c] - eigda_n[k, m, c])
                            )
                            val = (
                                AAda[k, c, a, n, m]
                                + (AAn[k, c, n, n] - AAn[k, c, m, m]) * Dno[k, a, n, m]
                                + (AAn[k, a, n, n] - AAn[k, a, m, m]) * Dno[k, c, n, m]
                                - 1j * AAn[k, c, n, m] * (AAn[k, a, n, n] - AAn[k, a, m, m])
                                + sum_AD[k, c, a, n, m]
                                + 1j * inner / (eign[k, m] - eign[k, n])
                            )
                        if sc_use_eta_corr and n != m:
                            corr = _naive_eta_correction(AA_bar, HH_da_bar, eig, a, c, sc_eta)
                            val = val + corr[k, n, m]
                        out[k, c, a, n, m] = val
    return out


def test_generalized_derivative_matches_naive_loop_no_eta_corr():
    nk, nw = 2, 4
    AA_bar = _random_complex(nk, 3, nw, nw)
    AA_da_bar = _random_complex(nk, 3, 3, nw, nw)
    HH_da_bar = _random_complex(nk, 3, nw, nw)
    HH_dadb_bar = _random_complex(nk, 3, 3, nw, nw)
    D_h_no_eta = _random_complex(nk, 3, nw, nw)
    D_h_eta = _random_complex(nk, 3, nw, nw)
    eig = torch.tensor(np.sort(np.random.randn(nk, nw) * 2.0, axis=1))
    eig_da = torch.tensor(np.random.randn(nk, nw, 3))

    got = _generalized_derivative(
        AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta, eig, eig_da,
    ).numpy()
    want = _naive_generalized_derivative(
        AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta, eig, eig_da,
        sc_use_eta_corr=False, sc_eta=0.0,
    )
    off_diag = ~np.eye(nw, dtype=bool)
    np.testing.assert_allclose(got[:, :, :, off_diag], want[:, :, :, off_diag], atol=1e-10)


def test_generalized_derivative_matches_naive_loop_with_eta_corr():
    nk, nw = 1, 5
    AA_bar = _random_complex(nk, 3, nw, nw)
    AA_da_bar = _random_complex(nk, 3, 3, nw, nw)
    HH_da_bar = _random_complex(nk, 3, nw, nw)
    HH_dadb_bar = _random_complex(nk, 3, 3, nw, nw)
    D_h_no_eta = _random_complex(nk, 3, nw, nw)
    D_h_eta = _random_complex(nk, 3, nw, nw)
    eig = torch.tensor(np.sort(np.random.randn(nk, nw) * 2.0, axis=1))
    eig_da = torch.tensor(np.random.randn(nk, nw, 3))
    sc_eta = 0.05

    got = _generalized_derivative(
        AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta, eig, eig_da,
        sc_use_eta_corr=True, sc_eta=sc_eta,
    ).numpy()
    want = _naive_generalized_derivative(
        AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta, eig, eig_da,
        sc_use_eta_corr=True, sc_eta=sc_eta,
    )
    off_diag = ~np.eye(nw, dtype=bool)
    np.testing.assert_allclose(got[:, :, :, off_diag], want[:, :, :, off_diag], atol=1e-10)


def _naive_accumulate_sc_k(eig, occ, A_H, gen_r_nm, omega, eta):
    nk, nw = eig.shape
    nfreq = omega.shape[0]
    eign = eig.numpy()
    occn = occ.numpy()
    AHn = A_H.numpy()
    genn = gen_r_nm.numpy()
    sigma = eta / np.sqrt(2.0)

    out = np.zeros((nk, 3, 6, nfreq))
    for k in range(nk):
        for n in range(nw):
            for m in range(nw):
                if n == m:
                    continue
                occ_fac = occn[k, n] - occn[k, m]
                if abs(occ_fac) < 1e-10:
                    continue
                r_mn = AHn[k, :, m, n]     # (3,)
                for a in range(3):
                    for bc in range(6):
                        b, c = ALPHA_S[bc], BETA_S[bc]
                        I_nm = np.imag(
                            r_mn[b] * genn[k, c, a, n, m] + r_mn[c] * genn[k, b, a, n, m]
                        )
                        for iw in range(nfreq):
                            delta = (
                                gaussian_smearing(omega[iw] - (eign[k, n] - eign[k, m]), sigma)
                                + gaussian_smearing(omega[iw] - (eign[k, m] - eign[k, n]), sigma)
                            )
                            out[k, a, bc, iw] += occ_fac * I_nm * delta
    return out


def test_accumulate_sc_k_matches_naive_loop():
    nk, nw = 2, 4
    eig = torch.tensor(np.sort(np.random.randn(nk, nw) * 2.0, axis=1))
    occ = torch.tensor((eig.numpy() < 0.0).astype(float))
    A_H = _random_complex(nk, 3, nw, nw)
    gen_r_nm = _random_complex(nk, 3, 3, nw, nw)
    omega = np.linspace(-3.0, 3.0, 7)
    eta = 0.2

    got = _accumulate_sc_k(eig, occ, A_H, gen_r_nm, omega, eta).numpy()
    want = _naive_accumulate_sc_k(eig, occ, A_H, gen_r_nm, omega, eta)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_kmesh_spacing_matches_naive_formula():
    """Delta_k = max_i(|b_i|/mesh_i), `postw90_common.F90::kmesh_spacing_mesh`."""
    recip_lattice = np.array([[2.0, 0.0, 0.0], [0.3, 1.5, 0.0], [0.0, 0.0, 3.0]])
    mesh = (4, 2, 6)

    got = _kmesh_spacing(mesh, recip_lattice)
    want = max(np.linalg.norm(recip_lattice[i]) / mesh[i] for i in range(3))
    assert got == pytest.approx(want)


def _naive_adaptive_eta(del_eig, delta_k, prefactor, max_width, floor=1e-10):
    nk, nw, _ = del_eig.shape
    d = del_eig.numpy()
    out = np.zeros((nk, nw, nw))
    for k in range(nk):
        for n in range(nw):
            for m in range(nw):
                if n == m:
                    out[k, n, m] = 1.0
                    continue
                joint_level_spacing = np.linalg.norm(d[k, n] - d[k, m]) * delta_k
                out[k, n, m] = min(max(joint_level_spacing * prefactor, floor), max_width)
    return out


def test_adaptive_eta_matches_naive_loop():
    nk, nw = 3, 4
    del_eig = torch.tensor(np.random.randn(nk, nw, 3) * 0.3)
    delta_k = 0.7
    prefactor = np.sqrt(2.0)
    max_width = 1.0 / 27.2114079527  # 1 eV in Hartree, arbitrary small cap to exercise clamping

    got = _adaptive_eta(del_eig, delta_k, prefactor=prefactor, max_width=max_width).numpy()
    want = _naive_adaptive_eta(del_eig, delta_k, prefactor, max_width)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_adaptive_eta_is_symmetric_and_diagonal_is_finite():
    del_eig = torch.tensor(np.random.randn(2, 5, 3))
    eta = _adaptive_eta(del_eig, delta_k=0.5)
    np.testing.assert_allclose(eta.numpy(), eta.numpy().transpose(0, 2, 1), atol=1e-12)
    assert np.all(np.isfinite(eta.numpy()))
    diag = torch.diagonal(eta, dim1=-2, dim2=-1).numpy()
    np.testing.assert_allclose(diag, 1.0)


def test_shift_current_tensor_with_adaptive_smearing_runs_and_is_finite():
    """Smoke test for `kubo_adpt_smr=True` (postw90's own default smearing
    mode) on the same synthetic QWZ model as the fixed-width smoke test."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_AA_R(hr)
    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    omega = np.linspace(-2.0, 2.0, 5) * EV_TO_HARTREE

    sigma = shift_current_tensor(
        hr, AA_R, recip_lattice, real_lattice, mesh=(6, 6, 1),
        fermi_energy=0.0, omega=omega, sc_eta=0.08 * EV_TO_HARTREE,
        kubo_adpt_smr=True,
    )

    assert sigma.shape == (3, 6, 5)
    assert np.all(np.isfinite(sigma))
