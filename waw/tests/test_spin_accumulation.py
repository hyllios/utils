"""
Unit tests for waw.analysis.spin_accumulation (Shitade & Minamitani,
npj Spintron. 3, 29 (2025) -- SAC gamma^ij_sa).

The load-bearing test here is `test_s_i_na_matches_finite_difference`: it
independently verifies this module's one genuinely new derived identity,

    s^i_na(k) = Re[(S_a(k) @ JJp_i(k))[n,n]]

(module docstring's derivation, reusing `topology._jjp_jjm_from_occ`)
against a literal, from-scratch numerical-derivative transcription of
SM25's own Eq. 2a, s^i_na = (i/2)<u_n|s_a Q_n|d_ki u_n> + c.c., with NO
shared code path -- same standard as tests/test_hamiltonian.py's naive-loop
checks and tutorial33's kdotp finite-difference validation.
"""

import numpy as np
import torch

from waw.core.hamiltonian import HamiltonianR
from waw.analysis._fourier_derivs import h_and_grad_frac_batch
from waw.analysis.topology import _jjp_jjm_from_occ
from waw.analysis.spin_accumulation import (
    spin_accumulation_coefficient, _spin_matrix_eigenbasis, _EPS3,
)
from waw.units import EV_TO_HARTREE, to_si_units

torch.set_default_dtype(torch.float64)


def _random_hermitian_hr(nw=4, seed=0):
    """Small random tight-binding model, several R-vectors, no special symmetry."""
    rng = np.random.default_rng(seed)
    R_list = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 0), (-1, -1, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), nw, nw), dtype=np.complex128)
    H_R[0] = rng.normal(size=(nw, nw))
    H_R[0] = 0.5 * (H_R[0] + H_R[0].T)   # on-site Hermitian (real here, fine)
    for r in range(1, len(R_list)):
        A = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
        H_R[r] = 0.3 * A
    # Hermiticity across +-R pairs: H(-R) = H(R)^dagger
    pair = {1: 2, 3: 4, 5: 6}
    for r1, r2 in pair.items():
        H_R[r2] = H_R[r1].conj().T

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=nw), R_list


def _random_spin_operator_r(hr, seed=1):
    """(nR, nw, nw, 3) complex, each Cartesian component Hermitian per-R-pair
    (same convention `spin_texture.spin_operator_r` produces)."""
    rng = np.random.default_rng(seed)
    nR, nw = hr.H_R.shape[0], hr.nw
    raw = rng.normal(size=(3, nR, nw, nw)) + 1j * rng.normal(size=(3, nR, nw, nw))
    SS_R = np.zeros((nR, nw, nw, 3), dtype=np.complex128)
    for a in range(3):
        S_a = raw[a]
        S_a[0] = 0.5 * (S_a[0] + S_a[0].conj().T)
        pair = {1: 2, 3: 4, 5: 6}
        for r1, r2 in pair.items():
            S_a[r2] = S_a[r1].conj().T
        SS_R[..., a] = S_a
    return torch.tensor(SS_R, dtype=torch.complex128)


def _h_at_k(hr, k_frac):
    H0, _ = h_and_grad_frac_batch(hr, np.asarray([k_frac]))
    return H0[0]


def _spin_op_at_k(SS_R, R_vectors, degen, k_frac):
    phase = np.exp(2j * np.pi * (np.asarray(R_vectors, dtype=np.float64) @ k_frac))
    w = phase / degen
    S = np.einsum('r,rmna->mna', w, SS_R.numpy())
    return S   # (nw, nw, 3)


def _fix_gauge(u_ref, u):
    """Align an eigenvector's arbitrary overall phase to a reference."""
    overlap = np.vdot(u_ref, u)
    phase = overlap / abs(overlap)
    return u / phase


def _naive_s_i_na(hr, SS_R, R_vectors, k0, n, dk=1e-5):
    """Direct finite-difference transcription of Eq. 2a: no shared code
    with `spin_accumulation_coefficient`/`_jjp_jjm_from_occ` at all."""
    degen = hr.degen
    H0 = _h_at_k(hr, k0).numpy()
    eig0, U0 = np.linalg.eigh(H0)
    u_n = U0[:, n]

    S0 = _spin_op_at_k(SS_R, R_vectors, degen, k0)   # (nw,nw,3)

    nw = hr.nw
    Q_n = np.eye(nw) - np.outer(u_n, u_n.conj())
    out = np.zeros((3, 3))   # (cart-i, spin-a)
    for i in range(3):
        kp = k0.copy(); kp[i] += dk
        km = k0.copy(); km[i] -= dk
        Hp = _h_at_k(hr, kp).numpy()
        Hm = _h_at_k(hr, km).numpy()
        _, Up = np.linalg.eigh(Hp)
        _, Um = np.linalg.eigh(Hm)

        up_n = _fix_gauge(u_n, Up[:, n])
        um_n = _fix_gauge(u_n, Um[:, n])
        d_ki_un = (up_n - um_n) / (2 * dk)   # (nw,) derivative of the Bloch eigenvector

        vals = np.array([
            0.5j * np.vdot(u_n, S0[:, :, a] @ Q_n @ d_ki_un) for a in range(3)
        ])
        out[i] = (vals + vals.conjugate()).real
    return out   # (cart-i, spin-a)


def _formula_s_i_na(hr, SS_R, k0, n):
    """The module's own derived formula, evaluated directly (not through
    the full `spin_accumulation_coefficient` k-mesh loop) for a single k."""
    H0, grad = h_and_grad_frac_batch(hr, np.asarray([k0]))
    eig, UU = torch.linalg.eigh(H0)
    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad, UU)

    S_eig = _spin_matrix_eigenbasis(SS_R, hr, np.asarray([k0]), H0, UU)   # (1,3,nw,nw)

    nw = hr.nw
    onehot = torch.zeros(1, nw, dtype=torch.float64)
    onehot[0, n] = 1.0
    JJp_e, _ = _jjp_jjm_from_occ(dH_eig, eig, onehot)

    s_i_na = torch.einsum('kam,kim->kai', S_eig[:, :, n, :], JJp_e[:, :, :, n]).real
    return s_i_na[0].numpy()   # (spin-a, cart-i)


def test_s_i_na_matches_finite_difference():
    hr, R_list = _random_hermitian_hr(nw=4, seed=0)
    SS_R = _random_spin_operator_r(hr, seed=1)
    k0 = np.array([0.137, -0.241, 0.083])

    for n in range(4):
        naive = _naive_s_i_na(hr, SS_R, hr.R_vectors, k0, n, dk=1e-5)   # (cart-i, spin-a)
        formula = _formula_s_i_na(hr, SS_R, k0, n)                     # (spin-a, cart-i)
        np.testing.assert_allclose(formula.T, naive, atol=1e-6, rtol=1e-5)


def test_s_i_na_finite_difference_converges():
    """Same check at a coarser dk should be noticeably less accurate,
    confirming the comparison is a genuine finite-difference limit, not a
    coincidental match."""
    hr, _ = _random_hermitian_hr(nw=4, seed=2)
    SS_R = _random_spin_operator_r(hr, seed=3)
    k0 = np.array([0.05, 0.31, -0.12])
    n = 0

    formula = _formula_s_i_na(hr, SS_R, k0, n)
    fine = _naive_s_i_na(hr, SS_R, hr.R_vectors, k0, n, dk=1e-5)
    coarse = _naive_s_i_na(hr, SS_R, hr.R_vectors, k0, n, dk=1e-2)

    err_fine = np.abs(formula.T - fine).max()
    err_coarse = np.abs(formula.T - coarse).max()
    assert err_fine < 1e-5
    assert err_coarse > err_fine


SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u):
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


def _synthetic_qq_r(hr, seed=0):
    """(3 spin, 3 cart, nR, nw, nw) stand-in for `spin_texture.spin_position_r`'s
    QQ_R. Deliberately NOT Hermitized -- Q^i_a is not a Hermitian operator."""
    rng = np.random.default_rng(seed)
    nR, nw = hr.H_R.shape[0], hr.nw
    raw = rng.normal(size=(3, 3, nR, nw, nw)) + 1j * rng.normal(size=(3, 3, nR, nw, nw))
    return torch.tensor(raw, dtype=torch.complex128)


def test_spin_accumulation_coefficient_runs_and_is_finite():
    hr = _qwz_hr(u=1.5)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=1)
    BB_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=2)
    CC_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=3)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=4).permute(1, 2, 3, 0).contiguous()
    QQ_R = _synthetic_qq_r(hr, seed=5)

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_accumulation_coefficient(
        hr, AA_R, BB_R, CC_R, SS_R, recip_lattice, real_lattice,
        fermi_energies=np.linspace(-3.0, 3.0, 3), mesh=(6, 6, 1),
        sigma=0.3, QQ_R=QQ_R,
    )
    assert res.gamma.shape == (3, 3, 3, 3)
    assert np.all(np.isfinite(res.gamma))

    si = to_si_units(res.gamma, "spin_accumulation",
                      cell_volume_bohr3=abs(np.linalg.det(real_lattice)))
    assert np.all(np.isfinite(si))
    assert si.shape == res.gamma.shape


def test_gamma_is_full_rank2_tensor_not_axial():
    """Regression test for a real bug caught during development: gamma^ij_sa
    is a GENERAL rank-2 tensor in (i,j) (only its second term is
    antisymmetric via the Levi-Civita symbol) -- unlike AHC's curvature,
    it is NOT faithfully encoded by 3 axial components. Confirm the
    diagonal (i==j) entries -- which an axial-vector encoding would
    silently drop -- are generically nonzero."""
    hr = _qwz_hr(u=1.5)
    AA_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=11)
    BB_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=12)
    CC_R = _synthetic_hermitian_r(hr, extra_axes=(3, 3), seed=13)
    SS_R = _synthetic_hermitian_r(hr, extra_axes=(3,), seed=14).permute(1, 2, 3, 0).contiguous()
    QQ_R = _synthetic_qq_r(hr, seed=15)

    real_lattice = np.eye(3) * 10.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_accumulation_coefficient(
        hr, AA_R, BB_R, CC_R, SS_R, recip_lattice, real_lattice,
        fermi_energies=0.0, mesh=(6, 6, 1), sigma=0.3, QQ_R=QQ_R,
    )
    diag = np.array([res.gamma[0, i, i, :] for i in range(3)])
    assert np.abs(diag).max() > 1e-8


def test_eps3_is_the_genuine_levicivita_symbol():
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) < 3:
                    assert _EPS3[i, j, k] == 0.0
    assert _EPS3[0, 1, 2] == _EPS3[1, 2, 0] == _EPS3[2, 0, 1] == 1.0
    assert _EPS3[1, 0, 2] == _EPS3[0, 2, 1] == _EPS3[2, 1, 0] == -1.0


# ---------------------------------------------------------------------------
# C3 covariance -- the regression test for the 2026-07-28 s^i_na correction.
#
# Trigonal Te's measured symptom was that gamma violated the crystal's own C3
# symmetry by ~190% of its magnitude, mesh- and smearing-independently, because
# s^i_na was built from dH/(dE) alone (omitting SM25 Eq. 5a's out-of-subspace
# term and Eq. 6's Berry-connection term), making it gauge-DEPENDENT. A
# gauge-dependent tensor need not respect the point group.
#
# This builds a synthetic model that IS exactly C3 symmetric -- hexagonal
# lattice, three Wannier orbitals transforming as (p_x, p_y, p_z) so the
# orbital representation matrix is the Cartesian rotation itself -- by group
# averaging every operator over C3. gamma must then be C3 invariant.
# ---------------------------------------------------------------------------

_TH = 2 * np.pi / 3
_C3_CART = np.array([[np.cos(_TH), -np.sin(_TH), 0.0],
                     [np.sin(_TH),  np.cos(_TH), 0.0],
                     [0.0, 0.0, 1.0]])
# Hexagonal fractional action: a1 -> a2, a2 -> -a1-a2, so (n1,n2) -> (-n2, n1-n2)
_C3_FRAC = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int64)


def _hex_lattice(a=6.0, c=8.0):
    return np.array([[a, 0.0, 0.0],
                     [-a / 2, a * np.sqrt(3) / 2, 0.0],
                     [0.0, 0.0, c]])


def _c3_closed_r_set():
    """R-set closed under C3 AND under negation (so Hermiticity can be imposed)."""
    R = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, -1, 0),
         (-1, 0, 0), (0, -1, 0), (1, 1, 0)]
    return np.array(R, dtype=np.int64)


def _r_index_map(R_vectors, M):
    """index i -> index of (M @ R_i) within R_vectors."""
    lookup = {tuple(r): i for i, r in enumerate(R_vectors)}
    return np.array([lookup[tuple(M @ r)] for r in R_vectors], dtype=np.int64)


def _c3_symmetrize(op, R_vectors, n_cart_axes):
    """Group-average an operator so that
         O_cart(g.R) = Gamma(g)[ D(g) O(R) D(g)^dag ],
    with D(g) = Gamma(g) = the Cartesian rotation (p-orbital basis).

    `op` axes: (*cart, nR, nw, nw). Returns the same shape.
    """
    fwd = _r_index_map(R_vectors, _C3_FRAC)             # R_{fwd[i]} = C3.R_i
    inv = np.argsort(fwd)                                # R_{inv[i]} = C3^-1.R_i
    out = np.zeros_like(op)
    Dg = np.eye(3)
    gather = np.arange(len(R_vectors))                   # gather[i] -> index of g^-1.R_i
    for _ in range(3):                                   # g = E, C3, C3^2
        rolled = op[..., gather, :, :]                   # O(g^-1 R)
        # D(g) O D(g)^dag on the Wannier indices
        rolled = np.einsum('pq,...qr,rs->...ps', Dg, rolled, Dg.conj().T)
        # Gamma(g) on each Cartesian axis
        for ax in range(n_cart_axes):
            rolled = np.moveaxis(np.tensordot(Dg, np.moveaxis(rolled, ax, 0),
                                              axes=([1], [0])), 0, ax)
        out += rolled / 3.0
        Dg = _C3_CART @ Dg
        gather = inv[gather]
    return out


def _impose_hermiticity(op, R_vectors):
    """O(-R) = O(R)^dagger, elementwise over any leading Cartesian axes."""
    lookup = {tuple(r): i for i, r in enumerate(R_vectors)}
    out = op.copy()
    for i, r in enumerate(R_vectors):
        j = lookup[tuple(-r)]
        avg = 0.5 * (op[..., i, :, :] + op[..., j, :, :].conj().swapaxes(-1, -2))
        out[..., i, :, :] = avg
        out[..., j, :, :] = avg.conj().swapaxes(-1, -2)
    return out


def _impose_cc_hermiticity(cc, R_vectors):
    """CC_R[a,b](-R) = CC_R[b,a](R)^dagger -- the physical relation for
    <0n|(r-R)_a H (r-R)_b|Rm>. WITHOUT this the synthetic CC_R is unphysical
    and m_nk comes out non-covariant (C3 residual 0.3 instead of 1e-14),
    which looks exactly like a code bug but is a broken test input."""
    lookup = {tuple(r): i for i, r in enumerate(R_vectors)}
    out = cc.copy()
    for i, r in enumerate(R_vectors):
        j = lookup[tuple(-r)]
        for a in range(3):
            for b in range(3):
                avg = 0.5 * (cc[a, b, i] + cc[b, a, j].conj().T)
                out[a, b, i] = avg
                out[b, a, j] = avg.conj().T
    return out


def _c3_symmetric_model(seed=0):
    rng = np.random.default_rng(seed)
    R_vectors = _c3_closed_r_set()
    nR, nw = len(R_vectors), 3
    degen = np.ones(nR, dtype=np.int64)

    def rnd(*cart):
        shape = (*cart, nR, nw, nw)
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    H_R = _impose_hermiticity(_c3_symmetrize(rnd(), R_vectors, 0), R_vectors)
    AA_R = _impose_hermiticity(_c3_symmetrize(rnd(3), R_vectors, 1), R_vectors)
    BB_R = _c3_symmetrize(rnd(3), R_vectors, 1)
    CC_R = _impose_cc_hermiticity(_c3_symmetrize(rnd(3, 3), R_vectors, 2), R_vectors)
    SS_R = _impose_hermiticity(_c3_symmetrize(rnd(3), R_vectors, 1), R_vectors)
    QQ_R = _c3_symmetrize(rnd(3, 3), R_vectors, 2)   # not Hermitian by design

    hr = HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                      R_vectors=R_vectors, degen=degen, nw=nw)
    T = lambda x: torch.tensor(x, dtype=torch.complex128)
    return (hr, T(AA_R), T(BB_R), T(CC_R),
            T(np.moveaxis(SS_R, 0, -1).copy()), T(QQ_R))


def _c3_residual(t):
    tp = np.einsum('ip,jq,ar,pqr->ija', _C3_CART, _C3_CART, _C3_CART, t)
    return np.abs(tp - t).max() / max(np.abs(t).max(), 1e-30)


def test_c3_symmetric_model_gives_c3_invariant_gamma():
    """The load-bearing regression test: with every ingredient exactly C3
    symmetric, gamma^ij_sa must be C3 invariant. The pre-2026-07-28 s^i_na
    (dH/(dE) only, no .sIu and no Berry-connection term) fails this."""
    hr, AA_R, BB_R, CC_R, SS_R, QQ_R = _c3_symmetric_model(seed=0)
    real_lattice = _hex_lattice()
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    res = spin_accumulation_coefficient(
        hr, AA_R, BB_R, CC_R, SS_R, recip_lattice, real_lattice,
        fermi_energies=np.linspace(-2.0, 2.0, 5), mesh=(6, 6, 2),
        sigma=0.4, QQ_R=QQ_R,
    )
    residuals = [_c3_residual(res.gamma[f]) for f in range(res.gamma.shape[0])]
    assert max(residuals) < 1e-8, f"C3 residuals: {np.round(residuals, 6)}"


def test_c3_symmetrizer_actually_symmetrizes():
    """Guard the test's own scaffolding: the group average must satisfy the
    covariance relation it claims, or the test above proves nothing."""
    R_vectors = _c3_closed_r_set()
    rng = np.random.default_rng(7)
    op = rng.normal(size=(3, len(R_vectors), 3, 3)) + 1j * rng.normal(size=(3, len(R_vectors), 3, 3))
    sym = _c3_symmetrize(op, R_vectors, 1)
    fwd = _r_index_map(R_vectors, _C3_FRAC)
    lhs = sym[:, fwd, :, :]                                        # O(C3.R)
    rhs = np.einsum('ab,bris->aris', _C3_CART,
                    np.einsum('pq,brqs,st->brpt', _C3_CART, sym, _C3_CART.conj().T))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_old_dh_over_de_route_is_also_c3_covariant():
    """Counter-intuitive but load-bearing: the pre-2026-07-28 s^i_na
    (`Re[(S_a @ JJp_i)[n,n]]`, dH/(dE) only) is ALSO exactly C3 covariant on a
    symmetric model -- it is built from H_R and SS_R, both covariant, so their
    product must be. The omission of SM25 Eq. 5a's out-of-subspace term and
    Eq. 6's Berry-connection term is therefore a CORRECTNESS bug (wrong
    physical quantity), NOT the cause of trigonal Te's C3 violation.

    Recorded as a test so nobody re-derives the tempting-but-wrong story that
    the missing terms broke the symmetry. Te's violation comes from the
    Wannier model itself not being symmetry-adapted (no sitesym): measured
    gauge-invariantly on the cached hexagonal MgB2 models,
    |E(k) - E(C3.k)| reaches 226 meV (nb16) / 17.6 meV (refix v2), which the
    SAC's energy denominators and Fermi-surface deltas amplify to O(1).
    """
    from waw.analysis.spin_accumulation import _spin_matrix_eigenbasis

    hr, AA_R, BB_R, CC_R, SS_R, QQ_R = _c3_symmetric_model(seed=0)
    real_lattice = _hex_lattice()
    inv_recip = np.linalg.inv(2 * np.pi * np.linalg.inv(real_lattice).T)

    ga = np.arange(6) / 6.0
    kpts = np.stack(np.meshgrid(ga, ga, np.arange(2) / 2.0, indexing='ij'), -1).reshape(-1, 3)

    H0, grad_frac = h_and_grad_frac_batch(hr, kpts)
    grad_cart = torch.einsum('ja,kanm->kjnm',
                             torch.as_tensor(inv_recip, dtype=torch.complex128), grad_frac)
    grad_cart = 0.5 * (grad_cart + grad_cart.conj().transpose(-1, -2))
    eig, UU = torch.linalg.eigh(H0)
    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)
    S_eig = _spin_matrix_eigenbasis(SS_R, hr, kpts, H0, UU)

    deps = torch.diagonal(dH_eig, dim1=-2, dim2=-1).real.numpy()
    delta = np.exp(-((eig.numpy() - 0.0) / 0.4) ** 2)

    old = np.zeros((3, 3, 3))
    for n in range(hr.nw):
        onehot = torch.zeros(len(kpts), hr.nw, dtype=torch.float64)
        onehot[:, n] = 1.0
        JJp_e, _ = _jjp_jjm_from_occ(dH_eig, eig, onehot)
        s_i = torch.einsum('kam,kim->kai', S_eig[:, :, n, :], JJp_e[:, :, :, n]).real.numpy()
        old += np.einsum('k,kai,kj->ija', delta[:, n], s_i, deps[:, :, n])

    assert np.abs(old).max() > 1e-6, "term1 should be non-trivial here"
    assert _c3_residual(old) < 1e-8, (
        f"expected the old route to be C3 covariant too; got {_c3_residual(old):.2e}"
    )


def test_full_gamma_matches_direct_eq1_tb_limit():
    """THE anchor test (2026-07-28): full gamma from the module vs a literal,
    from-scratch evaluation of SM25 Eq. 1 in the tight-binding limit
    (point orbitals: AA_R = BB_R = CC_R = QQ_R = 0, on-site spin), where
    every ingredient has an exact eigenvector expression:

        <u_m|d_i u_n>   = dH_mn/(eps_n - eps_m)            (m != n)
        s^i_na          = Re[i sum_m S_nm <u_m|d_i u_n>]
        eps^ijk m_k(n)  = Im sum_m dH^i_nm dH^j_mn/(eps_n - eps_m)

    This pinned BOTH 2026-07-28 kernel corrections: the m-term enters Eq. 1
    as (img-imh)/2 (sign AND factor 2 vs the postw90 accumulation), and
    s^i_na needs the interband S@JJp term on top of the diagonal
    QQ/Berry-connection pieces."""
    hr, _ = _random_hermitian_hr(nw=4, seed=0)
    nR, nw = hr.H_R.shape[0], hr.nw
    rng = np.random.default_rng(11)
    SS_R = np.zeros((nR, nw, nw, 3), dtype=complex)
    for a in range(3):
        S0 = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
        SS_R[0, :, :, a] = 0.5 * (S0 + S0.conj().T)
    SS_R_t = torch.tensor(SS_R)
    Z3 = torch.zeros(3, nR, nw, nw, dtype=torch.complex128)
    QQ_Z = torch.zeros(3, 3, nR, nw, nw, dtype=torch.complex128)

    from waw.core.distributions import gaussian_smearing

    mesh, EF, SIG = (8, 8, 8), 0.3, 0.08
    res = spin_accumulation_coefficient(
        hr, Z3, Z3.clone(), Z3.clone(), SS_R_t,
        2 * np.pi * np.eye(3), np.eye(3),
        fermi_energies=np.array([EF]), mesh=mesh, sigma=SIG, QQ_R=QQ_Z)
    g_mod = res.gamma[0]

    g = np.meshgrid(*[np.arange(m) for m in mesh], indexing='ij')
    kpts = np.stack([x.ravel() for x in g], axis=1) / np.array(mesh)
    H0, grad = h_and_grad_frac_batch(hr, kpts)
    H0 = H0.numpy(); grad = grad.numpy() / (2 * np.pi)   # frac -> cart (recip = 2pi I)
    g_ref = np.zeros((3, 3, 3)); nk = len(kpts)
    for ik in range(nk):
        e, U = np.linalg.eigh(H0[ik])
        dH = np.einsum('pn,apq,qm->anm', U.conj(), grad[ik], U)
        Sk = np.einsum('r,rmna->amn',
                       np.exp(2j * np.pi * (hr.R_vectors @ kpts[ik])) / hr.degen, SS_R)
        Se = np.einsum('pn,apq,qm->anm', U.conj(), Sk, U)
        for n in range(nw):
            d = gaussian_smearing(np.array([e[n] - EF]), SIG)[0]
            if d < 1e-12:
                continue
            JJ = np.array([[dH[i, m, n] / (e[n] - e[m]) if m != n else 0
                            for m in range(nw)] for i in range(3)])
            s_i = np.real(1j * np.einsum('am,im->ai', Se[:, n, :], JJ))
            m_orb = np.array([np.imag(sum(dH[i, n, m] * dH[j, m, n] / (e[n] - e[m])
                                          for m in range(nw) if m != n))
                              for i in range(3) for j in range(3)]).reshape(3, 3)
            s_na = np.real(np.diagonal(Se, axis1=1, axis2=2)[:, n])
            deps = np.real(np.diagonal(dH, axis1=1, axis2=2)[:, n])
            g_ref += d * (np.einsum('ai,j->ija', s_i, deps)
                          - np.einsum('ij,a->ija', m_orb, s_na)) / nk
    # module output carries Eq. 1's -(e/hbar) with e = +|e| (Fig. 2 anchor)
    np.testing.assert_allclose(g_mod, -g_ref, atol=1e-12)
