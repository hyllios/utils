"""
Unit tests for waw.analysis.spin_hall (postw90 berry_task=eval_shc,
shc_method=qiao, Wannier90 tutorial 29): the R-space operator
construction (`build_shc_operators`), the per-k spin-current matrix
(`_js_k_batch`), and an end-to-end smoke test of `spin_hall_conductivity`.

These cross-check the vectorized torch implementation against
independently written, explicit-loop (not vectorized) transcriptions of
the SAME formulas read directly from wannier90's berry.F90/get_oper.F90 --
same standard as tests/test_hamiltonian.py's TestBBR/TestCCR and
tests/test_shift_current.py's naive-loop checks.
"""

import numpy as np
import pytest
import torch

from waw.core.hamiltonian import (
    HamiltonianR, compute_bb_r, compute_operator_r, operator_k, _wigner_seitz,
)
from waw.core.spread import rotate_overlaps, weight_overlaps_by_operator
from waw.analysis.spin_hall import (
    build_shc_operators, _js_k_batch, _shc_per_k_ingredients,
    spin_hall_conductivity, spin_hall_conductivity_ac,
    build_shc_ryoo_operators, _js_k_batch_ryoo, spin_berry_curvature_kpath,
    spin_nernst_conductivity, SpinNernstResult,
)
from waw.analysis.topology import _nernst_mott_integral
from waw.units import EV_TO_HARTREE, K_B_HARTREE, to_si_units

torch.manual_seed(0)
np.random.seed(0)


def _make_shc_data(nk=4, nnb=2, nb=3, nw=2, seed=0):
    """Synthetic disentangled system, same recipe as
    tests/test_hamiltonian.py::_make_bb_cc_data, plus a random Hermitian-
    per-component `.spn`-like spin operator and ab-initio eigenvalues."""
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
    W = torch.bmm(V, U)

    raw = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    raw = 0.5 * (raw + raw.conj().transpose(0, 1, 3, 2))
    spn_bloch = torch.tensor(raw).permute(1, 2, 3, 0).contiguous()   # (nk, nb, nb, 3), Hermitian per component

    return dict(nk=nk, nnb=nnb, nb=nb, nw=nw, mp_grid=mp_grid, real_lattice=real_lattice,
                kpts=kpts, kpts_np=kpts_np, kb_idx=kb_idx, wb=wb, bvecs=bvecs,
                Mmn=Mmn, eig=eig, V=V, U=U, W=W, spn_bloch=spn_bloch)


def _naive_sr_shr(d, c):
    """Explicit per-k, per-neighbour loop reference for SR(R)/SHR(R)'s
    spin-weighted finite-difference sum (component c), independent of
    `weight_overlaps_by_operator`/`rotate_overlaps`/`compute_bb_r`."""
    nk, nnb, nw = d["nk"], d["nnb"], d["nw"]
    Mmn = d["Mmn"].numpy()
    spn = d["spn_bloch"].numpy()
    eig = d["eig"].numpy()
    W = d["W"].numpy()
    wb = d["wb"].numpy()
    bvecs = d["bvecs"].numpy()
    kb_idx = d["kb_idx"].numpy()

    SR_q = np.zeros((nk, 3, nw, nw), dtype=complex)
    SHR_q = np.zeros((nk, 3, nw, nw), dtype=complex)
    for k in range(nk):
        for b in range(nnb):
            kb = kb_idx[k, b]
            S_c = spn[k, :, :, c]                       # (nb, nb)
            SH_c = S_c @ np.diag(eig[k])                 # (nb, nb)
            Mmn_S = S_c @ Mmn[k, b]
            Mmn_SH = SH_c @ Mmn[k, b]
            SR_tilde = W[k].conj().T @ Mmn_S @ W[kb]
            SHR_tilde = W[k].conj().T @ Mmn_SH @ W[kb]
            for a in range(3):
                SR_q[k, a] += 1j * wb[b] * bvecs[k, b, a] * SR_tilde
                SHR_q[k, a] += 1j * wb[b] * bvecs[k, b, a] * SHR_tilde
    return SR_q, SHR_q


def test_sr_shr_r_round_trip_matches_explicit_loop_reference():
    d = _make_shc_data()
    _, SR_R, SHR_R, _ = build_shc_operators(
        d["W"], d["Mmn"], d["kb_idx"], d["spn_bloch"], d["eig"],
        d["wb"], d["bvecs"], d["kpts"], d["mp_grid"], d["real_lattice"],
    )
    R_arr, degen = _wigner_seitz(d["mp_grid"], d["real_lattice"])

    for c in range(3):
        SR_q_ref, SHR_q_ref = _naive_sr_shr(d, c)
        SR_q_rt = torch.stack(
            [operator_k(SR_R[c, a], R_arr, degen, d["kpts_np"]) for a in range(3)], dim=1
        ).numpy()
        SHR_q_rt = torch.stack(
            [operator_k(SHR_R[c, a], R_arr, degen, d["kpts_np"]) for a in range(3)], dim=1
        ).numpy()
        np.testing.assert_allclose(SR_q_rt, SR_q_ref, atol=1e-10)
        np.testing.assert_allclose(SHR_q_rt, SHR_q_ref, atol=1e-10)


def test_sh_r_round_trip_matches_explicit_loop_reference():
    """SH(R) = <0n|sigma.H|Rm>, the PLAIN (no b-vector) operator pattern."""
    d = _make_shc_data()
    _, _, _, SH_R = build_shc_operators(
        d["W"], d["Mmn"], d["kb_idx"], d["spn_bloch"], d["eig"],
        d["wb"], d["bvecs"], d["kpts"], d["mp_grid"], d["real_lattice"],
    )
    R_arr, degen = _wigner_seitz(d["mp_grid"], d["real_lattice"])

    W = d["W"].numpy()
    spn = d["spn_bloch"].numpy()
    eig = d["eig"].numpy()
    for c in range(3):
        SH_o_ref = np.stack([
            W[k].conj().T @ (spn[k, :, :, c] @ np.diag(eig[k])) @ W[k]
            for k in range(d["nk"])
        ])
        SH_o_rt = operator_k(SH_R[..., c], R_arr, degen, d["kpts_np"]).numpy()
        np.testing.assert_allclose(SH_o_rt, SH_o_ref, atol=1e-10)


def test_build_shc_operators_shapes():
    d = _make_shc_data()
    SS_R, SR_R, SHR_R, SH_R = build_shc_operators(
        d["W"], d["Mmn"], d["kb_idx"], d["spn_bloch"], d["eig"],
        d["wb"], d["bvecs"], d["kpts"], d["mp_grid"], d["real_lattice"],
    )
    R_arr, _ = _wigner_seitz(d["mp_grid"], d["real_lattice"])
    nR, nw = len(R_arr), d["nw"]
    assert SS_R.shape == (nR, nw, nw, 3)
    assert SR_R.shape == (3, 3, nR, nw, nw)
    assert SHR_R.shape == (3, 3, nR, nw, nw)
    assert SH_R.shape == (nR, nw, nw, 3)


def _naive_js_k(eig, del_eig_alpha, UU, D_h_alpha, S_k_w, SR_k_w, SHR_k_w, SH_k_w):
    """Explicit per-k, per-(row,col) transcription of berry_get_js_k's
    Qiao branch (berry.F90 lines ~2988-3053), independent of
    `_js_k_batch`'s einsum."""
    nk, nw, _ = UU.shape
    eign = eig.numpy()
    de = del_eig_alpha.numpy()
    UUn = UU.numpy()
    Dn = D_h_alpha.numpy()
    Sn, SRn, SHRn, SHn = (t.numpy() for t in (S_k_w, SR_k_w, SHR_k_w, SH_k_w))

    js_k = np.zeros((nk, nw, nw), dtype=complex)
    for k in range(nk):
        rot = lambda O: UUn[k].conj().T @ O @ UUn[k]
        S_k = rot(Sn[k])
        SH_k = rot(SHn[k])
        SR_alpha_k = -1j * rot(SRn[k])
        SHR_alpha_k = -1j * rot(SHRn[k])
        K_k = SR_alpha_k + S_k @ Dn[k]
        L_k = SHR_alpha_k + SH_k @ Dn[k]
        B_k = np.zeros((nw, nw), dtype=complex)
        for i in range(nw):
            for j in range(nw):
                B_k[i, j] = de[k, j] * S_k[i, j] + eign[k, j] * K_k[i, j] - L_k[i, j]
        js_k[k] = 0.5 * (B_k + B_k.conj().T)
    return js_k


def test_js_k_matches_naive_loop():
    nk, nw = 3, 4
    rng = np.random.default_rng(2)

    eig = torch.tensor(np.sort(rng.normal(size=(nk, nw)), axis=1))
    del_eig_alpha = torch.tensor(rng.normal(size=(nk, nw)))
    UU, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(nk, nw, nw))
                                         + 1j * rng.normal(size=(nk, nw, nw))))

    def rand_c(*shape):
        return torch.tensor(rng.normal(size=shape) + 1j * rng.normal(size=shape))

    D_h_alpha = rand_c(nk, nw, nw)
    S_k_w = rand_c(nk, nw, nw)
    SR_k_w = rand_c(nk, nw, nw)
    SHR_k_w = rand_c(nk, nw, nw)
    SH_k_w = rand_c(nk, nw, nw)

    got = _js_k_batch(eig, del_eig_alpha, UU, D_h_alpha, S_k_w, SR_k_w, SHR_k_w, SH_k_w).numpy()
    want = _naive_js_k(eig, del_eig_alpha, UU, D_h_alpha, S_k_w, SR_k_w, SHR_k_w, SH_k_w)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_js_k_is_hermitian():
    """js_k = (B_k + B_k^dagger)/2 must be exactly Hermitian by construction."""
    d = _make_shc_data(nk=2, nnb=2, nb=3, nw=3, seed=5)
    eig = torch.tensor(np.sort(np.random.default_rng(5).normal(size=(2, 3)), axis=1))
    del_eig_alpha = torch.randn(2, 3, dtype=torch.float64)
    UU, _ = torch.linalg.qr(torch.randn(2, 3, 3, dtype=torch.complex128))
    rand_c = lambda: torch.randn(2, 3, 3, dtype=torch.complex128)
    D_h_alpha, S_k_w, SR_k_w, SHR_k_w, SH_k_w = (rand_c() for _ in range(5))

    js_k = _js_k_batch(eig, del_eig_alpha, UU, D_h_alpha, S_k_w, SR_k_w, SHR_k_w, SH_k_w)
    torch.testing.assert_close(js_k, js_k.conj().transpose(-1, -2), atol=1e-10, rtol=0)


SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u: float) -> HamiltonianR:
    """Same 2-band QWZ model as test_hamiltonian_gauge_position.py/test_shift_current.py."""
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


def _synthetic_hermitian_r(hr, extra_axes=(3,), seed=0):
    rng = np.random.default_rng(seed)
    nR, nw = hr.H_R.shape[0], hr.nw
    shape = (*extra_axes, nR, nw, nw)
    raw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    raw = 0.5 * (raw + raw.conj().swapaxes(-1, -2))
    return torch.tensor(raw, dtype=torch.complex128)


def test_spin_hall_conductivity_runs_and_is_finite():
    """Smoke test on a synthetic 2-band QWZ model with random (non-
    physical, but shape-correct) R-space operators -- confirms the full
    pipeline (interpolation, D_h recovery, js_k, Kubo-like accumulation,
    unit conversion) runs end to end with a finite, real-valued result."""
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=1)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=2).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    SR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=3)
    SHR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=4)
    SH_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=6).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_hall_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energies=np.linspace(-2.0, 2.0, 5) * EV_TO_HARTREE, mesh=(6, 6, 1),
        kubo_adpt_smr=False, eta=0.1 * EV_TO_HARTREE,
    )
    assert res.sigma.shape == (5,)
    assert np.all(np.isfinite(res.sigma))


def test_spin_hall_conductivity_runs_with_adaptive_smearing():
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=1)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=2).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    SR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=3)
    SHR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=4)
    SH_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=6).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_hall_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energies=0.0, mesh=(6, 6, 1), kubo_adpt_smr=True,
    )
    assert np.all(np.isfinite(res.sigma))


def _synthetic_shc_system(seed_offset=0):
    hr = _qwz_hr(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=1 + seed_offset)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=2 + seed_offset).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    SR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=3 + seed_offset)
    SHR_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=4 + seed_offset)
    SH_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=6 + seed_offset).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    return hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice


def _naive_ac_accumulate(eig, dE, prod, mask, eta, fermi_ha, omega_ha):
    """Explicit per-k, per-(n,m,freq) transcription of berry_get_shc_
    klist's `lfreq` branch (berry.F90 lines ~2850-2899): a COMPLEX
    resonance denominator, unlike the Fermi-scan branch's real
    Lorentzian -- independent of `spin_hall_conductivity_ac`'s own
    vectorized einsum/broadcast implementation."""
    nk, nw, _ = dE.shape
    nfreq = len(omega_ha)
    eign = eig.numpy()
    dEn = dE.numpy()
    prodn = prod.numpy()
    maskn = mask.expand(nk, nw, nw).numpy()
    etan = eta.numpy() if torch.is_tensor(eta) else np.full((nk, nw, nw), eta)

    out = np.zeros((nk, nfreq), dtype=complex)
    for k in range(nk):
        for n in range(nw):
            if not (eign[k, n] < fermi_ha):
                continue
            omega_list = np.zeros(nfreq, dtype=complex)
            for m in range(nw):
                if not maskn[k, n, m]:
                    continue
                rfac = dEn[k, n, m]
                for ifreq in range(nfreq):
                    cdum = omega_ha[ifreq] + 1j * etan[k, n, m]
                    cfac = -2.0 / (rfac ** 2 - cdum ** 2)
                    omega_list[ifreq] += cfac * prodn[k, n, m].imag
            out[k] += omega_list
    return out


def test_spin_hall_conductivity_ac_matches_naive_loop():
    hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice = _synthetic_shc_system()
    mesh = (3, 3, 1)
    alpha, beta, gamma = 0, 1, 2
    fermi = 0.0
    omega = np.array([0.0, 0.5, 1.0, 2.0]) * EV_TO_HARTREE
    eta = 0.15 * EV_TO_HARTREE

    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)

    eig, del_eig, dE, prod, mask = _shc_per_k_ingredients(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice, kpts,
        alpha, beta, gamma, degen_thresh_ha=1e-3 * EV_TO_HARTREE, kubo_eigval_max_ha=None,
    )

    want = _naive_ac_accumulate(
        eig, dE, prod, mask, eta, fermi, omega,
    ).sum(axis=0) / kpts.shape[0]

    got = spin_hall_conductivity_ac(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energy=fermi, omega=omega, mesh=mesh,
        alpha=alpha, beta=beta, gamma=gamma, kubo_adpt_smr=False, eta=eta,
    ).sigma

    np.testing.assert_allclose(got, want, atol=1e-8, rtol=1e-8)


def test_spin_hall_conductivity_ac_runs_and_is_finite():
    """Smoke test, fixed AND adaptive smearing -- complex output, correct shape."""
    hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice = _synthetic_shc_system(seed_offset=10)
    omega = np.linspace(0.0, 3.0, 7) * EV_TO_HARTREE

    res_fixed = spin_hall_conductivity_ac(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energy=0.0, omega=omega, mesh=(6, 6, 1), kubo_adpt_smr=False, eta=0.1 * EV_TO_HARTREE,
    )
    assert res_fixed.sigma.shape == (7,)
    assert np.iscomplexobj(res_fixed.sigma)
    assert np.all(np.isfinite(res_fixed.sigma))

    res_adpt = spin_hall_conductivity_ac(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energy=0.0, omega=omega, mesh=(6, 6, 1), kubo_adpt_smr=True,
    )
    assert np.all(np.isfinite(res_adpt.sigma))


# ===========================================================================
# Ryoo method (RPS19) -- new capability, genuine .sIu/.sHu ab-initio data
# ===========================================================================

def _naive_js_k_ryoo(eig, VV0, S_k, SAA_k, SBB_k):
    """Explicit per-k, per-(n,m) transcription of berry.F90 develop branch's
    Ryoo js_k formula (RPS19 Eq. 21/26), independent of `_js_k_batch_ryoo`'s
    own einsum/broadcast implementation. Operates on ALREADY-ROTATED (H(k)
    eigenbasis) matrices -- the rotation itself reuses the same `rot()`
    einsum Qiao's own (already-tested) `_js_k_batch` uses, so this isolates
    just the new algebraic combination."""
    nk, nw, _ = S_k.shape
    eign = eig.numpy() if torch.is_tensor(eig) else eig
    VV0n, S_kn, SAAn, SBBn = VV0, S_k, SAA_k, SBB_k

    js_k = np.zeros((nk, nw, nw), dtype=complex)
    for k in range(nk):
        spinVel0 = VV0n[k] @ S_kn[k] + S_kn[k] @ VV0n[k]
        for n in range(nw):
            for m in range(nw):
                val = spinVel0[n, m]
                val += -1j * (eign[k, m] * SAAn[k, n, m] - SBBn[k, n, m])
                val += 1j * (eign[k, n] * np.conj(SAAn[k, m, n]) - np.conj(SBBn[k, m, n]))
                js_k[k, n, m] = 0.5 * val
    return js_k


def test_js_k_ryoo_matches_naive_loop():
    nk, nw = 3, 4
    rng = np.random.default_rng(3)

    eig = torch.tensor(np.sort(rng.normal(size=(nk, nw)), axis=1))
    UU, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(nk, nw, nw))
                                         + 1j * rng.normal(size=(nk, nw, nw))))

    def rand_c(*shape):
        return torch.tensor(rng.normal(size=shape) + 1j * rng.normal(size=shape))

    # dH_eig_alpha is passed to `_js_k_batch_ryoo` ALREADY in the H(k)
    # eigenbasis (matching how `_shc_per_k_ingredients` builds it before
    # calling this function) -- unlike S_k_w/SAA_k_w/SBB_k_w, which are
    # Wannier-gauge and get rotated internally via `rot()`.
    dH_eig_alpha = rand_c(nk, nw, nw)
    S_k_w = rand_c(nk, nw, nw)
    SAA_k_w = rand_c(nk, nw, nw)
    SBB_k_w = rand_c(nk, nw, nw)

    got = _js_k_batch_ryoo(eig, dH_eig_alpha, UU, S_k_w, SAA_k_w, SBB_k_w).numpy()

    def rot(O):
        return torch.einsum('kni,knm,kmj->kij', UU.conj(), O.to(UU.dtype), UU).numpy()

    want = _naive_js_k_ryoo(eig, dH_eig_alpha.numpy(), rot(S_k_w), rot(SAA_k_w), rot(SBB_k_w))
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_js_k_ryoo_is_hermitian_when_VV0_and_S_k_are():
    """js_k should come out Hermitian when VV0/S_k are Hermitian and the
    SAA/SBB antisymmetric-combination structure is respected -- a basic
    physical sanity check (spin current is a physical Hermitian observable),
    checked on a case built to satisfy it exactly."""
    nk, nw = 2, 3
    rng = np.random.default_rng(4)

    eig = torch.tensor(np.sort(rng.normal(size=(nk, nw)), axis=1))
    UU = torch.eye(nw, dtype=torch.complex128).expand(nk, nw, nw).contiguous()

    def rand_herm(*shape):
        raw = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        raw = 0.5 * (raw + raw.conj().swapaxes(-1, -2))
        return torch.tensor(raw)

    VV0 = rand_herm(nk, nw, nw)
    S_k = rand_herm(nk, nw, nw)

    # Build SAA/SBB so that SBB = 0 and SAA is anti-Hermitian: then
    # -i*(eig(m)SAA(n,m)) + i*(eig(n)*conj(SAA(m,n))) is, for SAA
    # anti-Hermitian (SAA(m,n) = -conj(SAA(n,m))), NOT generically
    # Hermitian unless eig(n)=eig(m) -- so instead just check js_k's
    # spinVel0-only piece (SAA=SBB=0) is exactly Hermitian, isolating the
    # part of the formula this test can pin down unambiguously.
    zero = torch.zeros(nk, nw, nw, dtype=torch.complex128)
    js_k = _js_k_batch_ryoo(eig, VV0, UU, S_k, zero, zero)
    torch.testing.assert_close(js_k, js_k.conj().transpose(-1, -2), atol=1e-10, rtol=0)


SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr_ryoo(u: float) -> HamiltonianR:
    return _qwz_hr(u)


def _synthetic_ryoo_operators(hr, seed_offset=0):
    """Round-trip smoke test data for `build_shc_ryoo_operators`: a random
    W gauge + synthetic .sIu/.sHu-shaped ab-initio arrays (Hermitian per
    Pauli component at coincident k=k+b, as a genuine sIu/sHu would be for
    the b=0 "neighbour")."""
    rng = np.random.default_rng(5 + seed_offset)
    nk, nnb, nb, nw = 4, 2, 3, hr.nw

    W = torch.tensor(rng.normal(size=(nk, nb, nw)) + 1j * rng.normal(size=(nk, nb, nw)))
    W, _ = torch.linalg.qr(W)

    sIu = torch.tensor(rng.normal(size=(nk, nnb, nb, nb, 3)) + 1j * rng.normal(size=(nk, nnb, nb, nb, 3)))
    sHu = torch.tensor(rng.normal(size=(nk, nnb, nb, nb, 3)) + 1j * rng.normal(size=(nk, nnb, nb, nb, 3)))

    kb_idx = torch.tensor(rng.integers(0, nk, size=(nk, nnb)), dtype=torch.long)
    wb = torch.tensor(rng.uniform(0.5, 1.5, size=nnb))
    bvecs = torch.tensor(rng.normal(size=(nk, nnb, 3)))
    kpts = torch.tensor(rng.uniform(size=(nk, 3)))

    return W, sIu, sHu, kb_idx, wb, bvecs, kpts


def test_build_shc_ryoo_operators_shapes():
    hr = _qwz_hr_ryoo(u=1.5)
    W, sIu, sHu, kb_idx, wb, bvecs, kpts = _synthetic_ryoo_operators(hr)
    real_lattice = np.eye(3) * 10.0
    mp_grid = (2, 2, 1)

    SAA_R, SBB_R = build_shc_ryoo_operators(W, sIu, sHu, kb_idx, wb, bvecs, kpts, mp_grid, real_lattice)
    from waw.core.hamiltonian import _wigner_seitz
    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    nR, nw = len(R_arr), hr.nw
    assert SAA_R.shape == (3, 3, nR, nw, nw)
    assert SBB_R.shape == (3, 3, nR, nw, nw)
    assert torch.all(torch.isfinite(SAA_R.real)) and torch.all(torch.isfinite(SAA_R.imag))
    assert torch.all(torch.isfinite(SBB_R.real)) and torch.all(torch.isfinite(SBB_R.imag))


def test_spin_hall_conductivity_ryoo_runs_and_is_finite():
    """Smoke test on a synthetic 2-band QWZ model with random (non-
    physical, but shape-correct) R-space operators, method='ryoo'.
    `SAA_R`/`SBB_R` need to live on the SAME R-grid as `hr` itself (same
    pattern as the existing Qiao smoke test's `_synthetic_hermitian_r`)
    -- `build_shc_ryoo_operators`'s own from-scratch k-mesh/R-grid
    construction is separately shape-tested in
    `test_build_shc_ryoo_operators_shapes` and would use a DIFFERENT,
    unrelated R-grid than `hr`'s own hand-built one here."""
    hr = _qwz_hr_ryoo(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=20)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=21).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    SAA_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=22)
    SBB_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=23)

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_hall_conductivity(
        hr, AA_R, SS_R, recip_lattice=recip_lattice, real_lattice=real_lattice,
        fermi_energies=np.linspace(-2.0, 2.0, 5) * EV_TO_HARTREE, mesh=(6, 6, 1),
        kubo_adpt_smr=False, eta=0.1 * EV_TO_HARTREE,
        method="ryoo", SAA_R=SAA_R, SBB_R=SBB_R,
    )
    assert res.sigma.shape == (5,)
    assert np.all(np.isfinite(res.sigma))


def test_spin_hall_conductivity_rejects_unknown_method():
    hr = _qwz_hr_ryoo(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=22)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=23).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    with pytest.raises(ValueError):
        spin_hall_conductivity(
            hr, AA_R, SS_R, recip_lattice=recip_lattice, real_lattice=real_lattice,
            fermi_energies=0.0, mesh=(4, 4, 1), method="bogus",
        )


# ===========================================================================
# spin_berry_curvature_kpath -- per-k integrand along an explicit path
# ===========================================================================

def test_spin_berry_curvature_kpath_averages_to_conductivity():
    """Averaging the per-k curvature over the SAME uniform mesh
    `spin_hall_conductivity` would use, with fixed (non-adaptive) smearing,
    must exactly reproduce its own sigma -- this is the same accumulation,
    just without the final k-sum/normalize, so this is a decisive
    round-trip check rather than a mere smoke test."""
    hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice = _synthetic_shc_system(seed_offset=30)
    mesh = (4, 4, 1)
    fermi_energy = 0.3 * EV_TO_HARTREE
    eta = 0.1 * EV_TO_HARTREE

    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpath = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)

    curvature = spin_berry_curvature_kpath(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        kpath=kpath, fermi_energy=fermi_energy, eta=eta,
    )
    averaged = curvature.mean()

    ref = spin_hall_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energies=fermi_energy, mesh=mesh, kubo_adpt_smr=False, eta=eta,
    ).sigma[0]

    np.testing.assert_allclose(averaged, ref, atol=1e-10, rtol=1e-8)


def test_spin_berry_curvature_kpath_ryoo_runs_and_is_finite():
    hr = _qwz_hr_ryoo(u=3.0)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=31)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(), seed=32).unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    SAA_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=33)
    SBB_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=34)

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kpath = np.stack([np.linspace(0, 0.5, 20), np.zeros(20), np.zeros(20)], axis=1)

    curvature = spin_berry_curvature_kpath(
        hr, AA_R, SS_R, recip_lattice=recip_lattice, real_lattice=real_lattice,
        kpath=kpath, fermi_energy=0.0, eta=0.1 * EV_TO_HARTREE,
        method="ryoo", SAA_R=SAA_R, SBB_R=SBB_R,
    )
    assert curvature.shape == (20,)
    assert np.all(np.isfinite(curvature))


# ===========================================================================
# spin_nernst_conductivity -- Mott relation applied to the SHC(E) curve
# (PRB 106, 165102 (2022) Eq. 3/4), reusing topology._nernst_mott_integral
# verbatim
# ===========================================================================

def test_spin_nernst_conductivity_end_to_end():
    """
    Shapes, the sigma(E) curve it used, and consistency with a direct Mott
    integral of that same curve -- same standard as this project's own
    `test_anomalous_nernst_conductivity_end_to_end_on_qwz`."""
    hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice = _synthetic_shc_system(seed_offset=40)

    kT_values = [100.0 * K_B_HARTREE, 300.0 * K_B_HARTREE]
    res = spin_nernst_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        mu=0.0, kT_values=kT_values, mesh=(6, 6, 1), kubo_adpt_smr=False,
        eta=0.1 * EV_TO_HARTREE, energy_halfwidth=0.3 * EV_TO_HARTREE, n_energies=21,
    )
    assert isinstance(res, SpinNernstResult)
    assert res.alpha.shape == (2,)
    assert res.sigma_of_E.shape == (21,)
    assert np.all(np.isfinite(res.alpha))

    # reproduces a direct Mott integral of the returned sigma(E)
    direct = _nernst_mott_integral(res.energies, res.sigma_of_E[:, None], 0.0, kT_values[1])[0]
    np.testing.assert_allclose(res.alpha[1], direct, rtol=1e-10)


def test_spin_nernst_conductivity_matches_manual_shc_scan():
    """The orchestration (spin_hall_conductivity's Fermi-energy scan feeding
    _nernst_mott_integral) must give the SAME sigma(E) as calling
    spin_hall_conductivity directly at each energy one at a time."""
    hr, AA_R, SS_R, SR_R, SHR_R, SH_R, real_lattice, recip_lattice = _synthetic_shc_system(seed_offset=41)
    energies = np.linspace(-0.1, 0.1, 11) * EV_TO_HARTREE

    res = spin_nernst_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        mu=0.0, kT_values=200.0 * K_B_HARTREE, mesh=(6, 6, 1), kubo_adpt_smr=False,
        eta=0.1 * EV_TO_HARTREE, energies=energies,
    )

    manual = spin_hall_conductivity(
        hr, AA_R, SS_R, SR_R, SHR_R, SH_R, recip_lattice, real_lattice,
        fermi_energies=energies, mesh=(6, 6, 1), kubo_adpt_smr=False, eta=0.1 * EV_TO_HARTREE,
    ).sigma
    np.testing.assert_allclose(res.sigma_of_E, manual, rtol=1e-12)


def test_spin_nernst_to_si_units_matches_manual_formula():
    """to_si_units("spin_nernst_conductivity") reproduces the SHC-based
    Sommerfeld formula exactly, independent of the actual physics --
    same standard as this project's own `test_anomalous_nernst_to_si_units`."""
    cell_volume_bohr3 = 200.0
    kT_values = np.array([150.0 * K_B_HARTREE, 350.0 * K_B_HARTREE])
    alpha_atomic = np.array([1.5, -2.5])   # Hartree*Bohr^2

    got = to_si_units(alpha_atomic, "spin_nernst_conductivity",
                       cell_volume_bohr3=cell_volume_bohr3, kT_values=kT_values)

    from waw.units import HARTREE_TO_EV, BOHR_TO_ANG, E_CHARGE, HBAR_SI, from_si_units
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    K_scm = BOHR_TO_ANG ** 2 * (1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3) / 2.0)
    T_kelvin = kT_values / K_B_HARTREE
    expected = (-100.0 * HARTREE_TO_EV / T_kelvin) * K_scm * alpha_atomic

    np.testing.assert_allclose(got, expected, rtol=1e-12)

    roundtrip = from_si_units(got, "spin_nernst_conductivity",
                              cell_volume_bohr3=cell_volume_bohr3, kT_values=kT_values)
    np.testing.assert_allclose(roundtrip, alpha_atomic, rtol=1e-10)
