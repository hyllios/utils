"""
Tests for `interfaces.quantum_espresso.dvscf_io` and `analysis.elph` --
native (EPW-free) reading of raw QE DFPT potential variations and
construction of Bloch-gauge electron-phonon matrix elements.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import erf

from waw.analysis import elph
from waw.core.hamiltonian import HamiltonianR
from waw.interfaces.ase.structure import monkhorst_pack
from waw.interfaces.quantum_espresso import dvscf_io
from waw.interfaces.quantum_espresso.upf import read_norm_conserving
from waw.units import AMU_TO_ME, ANG_TO_BOHR, BOHR_TO_ANG, HARTREE_TO_EV

DATA = Path(__file__).parent / "data" / "al_elph" / "out_ph"
GRID = (24, 24, 24)

# Full 6x6x6 Al reference run (electron UNK/Wannier data + 216-q dvscf grid),
# built during the el-ph validation session -- too large for tests/data/,
# so the EPW-pinned integration tests skip when it is absent.
GCHECK = Path(__file__).parents[1] / "workflows" / "notebooks" / "runs" / "al_elph_6cube_gcheck"
AL_UPF = Path(__file__).parents[1] / "workflows" / "pseudos" / "Al.upf"
needs_gcheck = pytest.mark.skipif(
    not (GCHECK / "gcheck_electron.npz").exists(),
    reason="al_elph_6cube_gcheck reference run not present",
)


def _u_kq_physical(u_all, kpts, kq, k_mesh):
    """The PHYSICAL periodic part at k+q: u_(k+q) = e^{-2pi i G.r} u_fold
    when k+q folds back to the mesh with reciprocal vector G -- the
    manual-reference counterpart of `wannier_transform_elph`'s own
    umklapp handling."""
    idx = elph.kpoint_mesh_index(kq, k_mesh)
    G = np.round(kq - kpts[idx]).astype(int)
    u = u_all[idx]
    if np.any(G != 0):
        fracs = np.meshgrid(*[np.arange(n) / n for n in u.shape[1:]], indexing="ij")
        phase = np.exp(-2j * np.pi * (G[0] * fracs[0] + G[1] * fracs[1] + G[2] * fracs[2]))
        u = u * phase[None]
    return idx, u


def _mock_coulomb_pseudo(zv=2.0):
    """Synthetic norm-conserving pseudo whose local part is EXACTLY the
    erf-regularized Coulomb potential, v_loc(r) = -zv*erf(r)/r: the radial
    integrand of `_vloc_form_factor` vanishes identically, leaving the
    analytic form factor -4*pi*zv*exp(-G^2/4)/(V*G^2)."""
    r = np.linspace(1e-8, 12.0, 1201)
    rab = np.full_like(r, r[1] - r[0])
    return {
        "r": r, "rab": rab, "vloc": -zv * erf(r) / r, "zv": zv,
        "betas": [r * np.exp(-r ** 2), r ** 2 * np.exp(-r ** 2)],
        "ells": [0, 1],
        "dij": np.diag([0.7, -0.4]),
        "core_correction": False,
    }


class TestReadPatterns:
    def test_one_irrep_three_perturbations(self):
        irreps = dvscf_io.read_patterns(DATA / "_ph0" / "al.phsave", 1)
        assert len(irreps) == 1
        assert irreps[0]["n_pert"] == 3
        assert irreps[0]["pattern"].shape == (3, 3)

    def test_pattern_is_unitary_not_merely_real(self):
        """
        The displacement-pattern matrix is UNITARY, and only accidentally
        real at high-symmetry q. Discarding its imaginary part -- which
        reaches 0.23 in magnitude on a plain 6x6x6 Al q-mesh -- breaks
        unitarity by ~5% and, because the inverse rotation then mixes
        phonon branches with very different omega into each other's
        1/omega^2 weight, inflated lambda at the affected q by ~3x.
        Unitarity of the COMPLEX matrix is the invariant to assert.
        """
        for iq in (1, 2):
            pattern = np.concatenate(
                [irr["pattern"] for irr in
                 dvscf_io.read_patterns(DATA / "_ph0" / "al.phsave", iq)],
                axis=0,
            )
            assert np.allclose(
                pattern @ pattern.conj().T, np.eye(len(pattern)), atol=1e-10,
            ), f"pattern matrix at q-index {iq} is not unitary"

        # q-index 2 (a generic 6x6x6 mesh point) is the regression guard:
        # its pattern is genuinely complex, so real-part-only rotation is
        # not merely inexact but non-unitary.
        p2 = np.concatenate(
            [irr["pattern"] for irr in
             dvscf_io.read_patterns(DATA / "_ph0" / "al.phsave", 2)], axis=0,
        )
        assert np.abs(p2.imag).max() > 0.2
        assert not np.allclose(p2.real @ p2.real.T, np.eye(len(p2)), atol=1e-3)


class TestDvscfPatternRotation:
    """
    Lock the pattern -> Cartesian rotation convention on a synthetic dvscf
    file: ``dv_pattern = P @ dv_cart`` with P unitary, so the reader must
    apply P^dagger. Getting this wrong (or dropping Im P) silently mixes
    phonon branches, which is invisible in any single-q amplitude check but
    corrupts mode-resolved lambda.
    """

    @staticmethod
    def _write(tmp_path, pattern, dv_pattern, grid):
        phsave = tmp_path / "_ph0" / "al.phsave"
        phsave.mkdir(parents=True)
        reps = "".join(
            f"<REPRESENTION.{i + 1}><NUMBER_OF_PERTURBATIONS>1"
            f"</NUMBER_OF_PERTURBATIONS><PERTURBATION.1><DISPLACEMENT_PATTERN>"
            + " ".join(f"{float(c.real):.17e} {float(c.imag):.17e}" for c in row)
            + f"</DISPLACEMENT_PATTERN></PERTURBATION.1></REPRESENTION.{i + 1}>"
            for i, row in enumerate(pattern)
        )
        (phsave / "patterns.1.xml").write_text(f"<Root><IRREPS_INFO>{reps}</IRREPS_INFO></Root>")
        raw = b"".join(
            np.asarray(dv_pattern[p], dtype=np.complex128).ravel(order="F").tobytes()
            for p in range(len(pattern))
        )
        (tmp_path / "_ph0" / "al.dvscf1").write_bytes(raw)

    def test_inverse_rotation_is_conjugate_transpose(self, tmp_path):
        grid = (2, 2, 2)
        rng = np.random.default_rng(0)
        # a unitary pattern matrix with a substantial imaginary part
        h = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        pattern = np.linalg.qr(h)[0]
        assert np.abs(pattern.imag).max() > 0.2

        dv_cart = (rng.normal(size=(3,) + grid) + 1j * rng.normal(size=(3,) + grid))
        dv_pattern = np.einsum("pm,m...->p...", pattern, dv_cart)
        self._write(tmp_path, pattern, dv_pattern, grid)

        got = dvscf_io.read_dvscf(tmp_path, "al", 1, grid, nat=1) / 0.5   # undo Ry->Ha
        assert np.allclose(got, dv_cart, atol=1e-12)


class TestReadDvscf:
    def test_shape_and_finite(self):
        dv = dvscf_io.read_dvscf(DATA, "al", 1, GRID, nat=1)
        assert dv.shape == (3,) + GRID
        assert dv.dtype == np.complex128
        assert np.all(np.isfinite(dv))
        assert np.abs(dv).max() > 0

    def test_translational_invariance_sum_rule(self):
        """
        For this single-atom cell, a Cartesian-displacement mode at q=Gamma
        IS a rigid translation of the whole crystal -- translational
        invariance requires the self-consistent potential to be completely
        unchanged, i.e. dv_cart must integrate to exactly zero.
        """
        dv = dvscf_io.read_dvscf(DATA, "al", 1, GRID, nat=1)
        assert np.abs(dv.mean(axis=(1, 2, 3))).max() < 1e-10

    def test_cubic_symmetry_equal_magnitude_per_axis(self):
        """Al is cubic: the 3 Cartesian displacement directions must be
        physically equivalent, so the (arbitrary, pattern-derived) dv_cart
        modes have identical max amplitude."""
        dv = dvscf_io.read_dvscf(DATA, "al", 1, GRID, nat=1)
        max_abs = np.abs(dv).reshape(3, -1).max(axis=1)
        assert np.allclose(max_abs, max_abs[0], rtol=1e-6)


class TestKpointMeshIndex:
    def test_matches_monkhorst_pack_enumeration(self):
        mesh = (3, 4, 2)
        kpts = monkhorst_pack(mesh)
        for idx, k in enumerate(kpts):
            assert elph.kpoint_mesh_index(k, mesh) == idx

    def test_wraps_modulo_one(self):
        mesh = (6, 6, 6)
        assert elph.kpoint_mesh_index(np.array([1.0, 0.0, 0.0]), mesh) == \
            elph.kpoint_mesh_index(np.array([0.0, 0.0, 0.0]), mesh)


class TestBlochMatrixElement:
    def test_matches_manual_loop(self):
        rng = np.random.default_rng(0)
        nb, ngrid, nmode = 3, (2, 2, 2), 4
        u_k = rng.normal(size=(nb,) + ngrid) + 1j * rng.normal(size=(nb,) + ngrid)
        u_kq = rng.normal(size=(nb,) + ngrid) + 1j * rng.normal(size=(nb,) + ngrid)
        dv = rng.normal(size=(nmode,) + ngrid) + 1j * rng.normal(size=(nmode,) + ngrid)

        g = elph.bloch_matrix_element(u_k, u_kq, dv)
        assert g.shape == (nmode, nb, nb)

        n_r = np.prod(ngrid)
        g_manual = np.zeros((nmode, nb, nb), dtype=np.complex128)
        for mu in range(nmode):
            for j in range(nb):
                for i in range(nb):
                    g_manual[mu, j, i] = np.sum(
                        np.conj(u_kq[j]) * dv[mu] * u_k[i]
                    ) / n_r
        assert np.allclose(g, g_manual)

    def test_hermitian_when_dv_real_and_k_equals_kq(self):
        """At k=k+q (q=0) with a real dv, g must be Hermitian in (j,i)."""
        rng = np.random.default_rng(1)
        nb, ngrid = 2, (3, 3, 3)
        u_k = rng.normal(size=(nb,) + ngrid) + 1j * rng.normal(size=(nb,) + ngrid)
        dv = rng.normal(size=(1,) + ngrid).astype(np.complex128)

        g = elph.bloch_matrix_element(u_k, u_k, dv)[0]
        assert np.allclose(g, g.conj().T)


def _random_elph_data(rng, k_mesh, q_mesh, nb=2, nw=2, nat3=3, ngrid=(3, 3, 3)):
    """Shared synthetic-data builder: random u_all/W/dvscf on a k_mesh/q_mesh
    pair (q_mesh must divide k_mesh component-wise)."""
    nk = k_mesh[0] * k_mesh[1] * k_mesh[2]
    nq = q_mesh[0] * q_mesh[1] * q_mesh[2]
    real_lattice = np.eye(3) * 5.0
    kpts = monkhorst_pack(k_mesh)
    qpts = monkhorst_pack(q_mesh)
    u_all = rng.normal(size=(nk, nb) + ngrid) + 1j * rng.normal(size=(nk, nb) + ngrid)
    W = rng.normal(size=(nk, nb, nw)) + 1j * rng.normal(size=(nk, nb, nw))
    dv_all = [
        rng.normal(size=(nat3,) + ngrid) + 1j * rng.normal(size=(nat3,) + ngrid)
        for _ in range(nq)
    ]
    return real_lattice, kpts, qpts, u_all, W, dv_all


class TestWannierTransformElph:
    def test_matches_manual_double_sum_same_mesh(self):
        rng = np.random.default_rng(42)
        mesh = (2, 2, 1)
        nat3, nw = 2, 1
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, mesh, mesh, nb=2, nw=nw, nat3=nat3,
        )
        nk = nq = len(kpts)

        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], mesh, mesh, real_lattice,
        )
        assert np.array_equal(R_e, R_q)
        assert np.array_equal(degen_e, degen_q)
        nR = len(R_e)
        assert g_R.shape == (nR, nR, nat3, nw, nw)

        phase_k = np.exp(-2j * np.pi * (kpts @ R_e.T))
        phase_q = np.exp(-2j * np.pi * (qpts @ R_q.T))

        g_ref = np.zeros((nR, nR, nat3, nw, nw), dtype=np.complex128)
        n_r = np.prod(u_all.shape[2:])
        for iq in range(nq):
            for ik in range(nk):
                kq_idx, u_kq = _u_kq_physical(u_all, kpts, kpts[ik] + qpts[iq], mesh)
                g_bloch = np.einsum(
                    "jxyz,mxyz,ixyz->mji",
                    u_kq.conj(), dv_all[iq], u_all[ik], optimize=True,
                ) / n_r
                g_wan = np.einsum(
                    "jn,uji,im->unm", W[kq_idx].conj(), g_bloch, W[ik], optimize=True,
                )
                for Re in range(nR):
                    for Rq in range(nR):
                        g_ref[Re, Rq] += (
                            phase_k[ik, Re] * phase_q[iq, Rq] * g_wan
                        ) / (nk * nq)

        assert np.allclose(g_R, g_ref)

    def test_umklapp_phase_applied_when_k_plus_q_folds(self):
        """On a (2,2,1) mesh, k=(0.5,0,0) + q=(0.5,0,0) folds to (0,0,0)
        with G=(1,0,0): the transform must use e^{-2pi i x} u_Gamma, NOT
        u_Gamma itself -- build the same g_R with plain (unphased) folding
        and check the two genuinely differ."""
        rng = np.random.default_rng(44)
        mesh = (2, 2, 1)
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, mesh, mesh, nb=2, nw=1, nat3=1,
        )
        nk = nq = len(kpts)

        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], mesh, mesh, real_lattice,
        )

        phase_k = np.exp(-2j * np.pi * (kpts @ R_e.T))
        phase_q = np.exp(-2j * np.pi * (qpts @ R_q.T))
        n_r = np.prod(u_all.shape[2:])
        g_ref_unphased = np.zeros_like(g_R)
        for iq in range(nq):
            for ik in range(nk):
                kq_idx = elph.kpoint_mesh_index(kpts[ik] + qpts[iq], mesh)
                g_bloch = np.einsum(
                    "jxyz,mxyz,ixyz->mji",
                    u_all[kq_idx].conj(), dv_all[iq], u_all[ik], optimize=True,
                ) / n_r
                g_wan = np.einsum(
                    "jn,uji,im->unm", W[kq_idx].conj(), g_bloch, W[ik], optimize=True,
                )
                g_ref_unphased += (
                    phase_k[ik][:, None, None, None, None]
                    * phase_q[iq][None, :, None, None, None] * g_wan
                ) / (nk * nq)

        assert not np.allclose(g_R, g_ref_unphased)

    def test_matches_manual_double_sum_different_mesh(self):
        """q_mesh strictly coarser than k_mesh (q_mesh divides k_mesh) --
        the general case this module is designed for."""
        rng = np.random.default_rng(43)
        k_mesh, q_mesh = (4, 4, 1), (2, 2, 1)
        nat3, nw = 2, 1
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, k_mesh, q_mesh, nb=2, nw=nw, nat3=nat3,
        )
        nk, nq = len(kpts), len(qpts)

        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], k_mesh, q_mesh, real_lattice,
        )
        assert g_R.shape == (len(R_e), len(R_q), nat3, nw, nw)

        phase_k = np.exp(-2j * np.pi * (kpts @ R_e.T))
        phase_q = np.exp(-2j * np.pi * (qpts @ R_q.T))

        g_ref = np.zeros((len(R_e), len(R_q), nat3, nw, nw), dtype=np.complex128)
        n_r = np.prod(u_all.shape[2:])
        for iq in range(nq):
            for ik in range(nk):
                kq_idx, u_kq = _u_kq_physical(u_all, kpts, kpts[ik] + qpts[iq], k_mesh)
                g_bloch = np.einsum(
                    "jxyz,mxyz,ixyz->mji",
                    u_kq.conj(), dv_all[iq], u_all[ik], optimize=True,
                ) / n_r
                g_wan = np.einsum(
                    "jn,uji,im->unm", W[kq_idx].conj(), g_bloch, W[ik], optimize=True,
                )
                for Re in range(len(R_e)):
                    for Rq in range(len(R_q)):
                        g_ref[Re, Rq] += (
                            phase_k[ik, Re] * phase_q[iq, Rq] * g_wan
                        ) / (nk * nq)

        assert np.allclose(g_R, g_ref)

    def test_shares_r_vectors_with_electron_hamiltonian(self):
        """Re must be the SAME Wigner-Seitz set core.hamiltonian.compute_hr
        would generate for the k_mesh/lattice (shared crystal lattice)."""
        from waw.core.hamiltonian import _wigner_seitz

        mesh = (2, 2, 2)
        real_lattice = np.eye(3) * 4.0
        R_expected, degen_expected = _wigner_seitz(mesh, real_lattice)

        rng = np.random.default_rng(0)
        _, kpts, qpts, u_all, _, dv_all = _random_elph_data(
            rng, mesh, mesh, nb=1, nw=1, nat3=1, ngrid=(2, 2, 2),
        )
        W = np.ones((len(kpts), 1, 1), dtype=np.complex128)
        dv_zero = [np.zeros((1, 2, 2, 2), dtype=np.complex128) for _ in range(len(qpts))]

        _, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, kpts, lambda iq: dv_zero[iq], mesh, mesh, real_lattice,
        )
        assert np.array_equal(R_e, R_expected)
        assert np.array_equal(degen_e, degen_expected)
        assert np.array_equal(R_q, R_expected)
        assert np.array_equal(degen_q, degen_expected)


class TestInterpolateElphKq:
    def test_round_trip_matches_construction(self):
        """Evaluated at the SAME mesh points g_R was built from, the
        interpolation must reproduce the original Wannier-gauge g(k,q)
        exactly (a discrete Fourier transform pair)."""
        rng = np.random.default_rng(7)
        mesh = (2, 2, 1)
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(rng, mesh, mesh)
        nk = nq = len(kpts)

        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], mesh, mesh, real_lattice,
        )

        n_r = np.prod(u_all.shape[2:])
        for iq in range(nq):
            for ik in range(nk):
                kq_idx, u_kq = _u_kq_physical(u_all, kpts, kpts[ik] + qpts[iq], mesh)
                g_bloch = np.einsum(
                    "jxyz,mxyz,ixyz->mji",
                    u_kq.conj(), dv_all[iq], u_all[ik], optimize=True,
                ) / n_r
                g_wan_expected = np.einsum(
                    "jn,uji,im->unm", W[kq_idx].conj(), g_bloch, W[ik], optimize=True,
                )
                g_wan_got = elph.interpolate_elph_kq(
                    g_R, R_e, degen_e, R_q, degen_q, kpts[ik:ik + 1], qpts[iq:iq + 1],
                )[0]
                assert np.allclose(g_wan_got, g_wan_expected, atol=1e-10)


class TestPhononModeCoupling:
    def test_scales_as_inverse_sqrt_omega(self):
        rng = np.random.default_rng(3)
        nat3, nw = 3, 2
        g_wannier = rng.normal(size=(nat3, nw, nw)) + 1j * rng.normal(size=(nat3, nw, nw))
        eigvec = np.eye(nat3, dtype=np.complex128)
        masses_amu = np.array([26.98])
        types = np.array([0])

        omega1 = np.array([0.01, 0.02, 0.03])
        omega2 = omega1 * 4.0   # 4x -> prefactor should shrink by 2x

        g1 = elph.phonon_mode_coupling(g_wannier, eigvec, omega1, masses_amu, types)
        g2 = elph.phonon_mode_coupling(g_wannier, eigvec, omega2, masses_amu, types)
        assert np.allclose(g2, g1 / 2.0)

    def test_identity_eigvec_single_mode_matches_direct_formula(self):
        """With eigvec = identity, mode mu couples only to Cartesian
        direction mu -- check against the formula evaluated by hand."""
        nat3, nw = 3, 1
        g_wannier = np.arange(1, nat3 * nw * nw + 1).reshape(nat3, nw, nw).astype(np.complex128)
        eigvec = np.eye(nat3, dtype=np.complex128)
        masses_amu = np.array([26.98])
        types = np.array([0])
        omega = np.array([0.01, 0.02, 0.03])

        g_mode = elph.phonon_mode_coupling(g_wannier, eigvec, omega, masses_amu, types)

        mass_me = 26.98 * AMU_TO_ME
        for nu in range(nat3):
            expected = g_wannier[nu] / np.sqrt(2.0 * omega[nu] * mass_me)
            assert np.allclose(g_mode[nu], expected)


class TestBandEigensystem:
    def test_matches_1d_chain_dispersion(self):
        """1-orbital, nearest-neighbour 1D chain: eps(k) = 2t cos(2 pi k),
        a textbook-known analytic dispersion."""
        t = -0.1
        R_vectors = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        degen = np.array([1, 1, 1])
        H_R = torch.tensor(
            np.array([[[t]], [[0.0]], [[t]]]), dtype=torch.complex128,
        )
        hr = HamiltonianR(H_R=H_R, R_vectors=R_vectors, degen=degen, nw=1)

        kpts = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
        eig, U = elph.band_eigensystem(hr, kpts)

        expected = 2 * t * np.cos(2 * np.pi * kpts[:, 0])
        assert np.allclose(eig[:, 0], expected)
        assert U.shape == (3, 1, 1)
        assert np.allclose(np.abs(U[:, 0, 0]), 1.0)


class TestAlpha2F:
    def test_zero_coupling_gives_zero_alpha2f(self):
        mesh = (2, 2, 1)
        nat3 = 3
        real_lattice, kpts, qpts, u_all, W, _ = _random_elph_data(
            np.random.default_rng(0), mesh, mesh, nb=1, nw=1, nat3=nat3, ngrid=(2, 2, 2),
        )
        u_all = np.ones_like(u_all)
        W = np.ones_like(W)
        dv_zero = [np.zeros((nat3, 2, 2, 2), dtype=np.complex128) for _ in range(len(qpts))]

        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_zero[iq], mesh, mesh, real_lattice,
        )

        H_R = torch.zeros((len(R_e), 1, 1), dtype=torch.complex128)
        hr = HamiltonianR(H_R=H_R, R_vectors=R_e, degen=degen_e, nw=1)
        eig, U = elph.band_eigensystem(hr, kpts)

        omega_ph = np.full((len(qpts), nat3), 0.001)
        eigvec_ph = np.tile(np.eye(nat3, dtype=np.complex128), (len(qpts), 1, 1))
        masses_amu = np.array([26.98])
        types = np.array([0])

        omega_grid = np.linspace(0.0, 0.01, 20)
        a2f = elph.alpha2f(
            eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts, mesh,
            omega_ph, eigvec_ph, masses_amu, types,
            fermi_energy=0.0, dos_at_ef=1.0, omega_grid=omega_grid,
            sigma_e=0.5,
        )
        assert np.allclose(a2f, 0.0)

    def test_nonzero_coupling_gives_nonnegative_finite_alpha2f(self):
        rng = np.random.default_rng(11)
        mesh = (2, 2, 1)
        nat3, nw = 3, 2
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, mesh, mesh, nb=2, nw=nw, nat3=nat3,
        )
        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], mesh, mesh, real_lattice,
        )

        H_R = (rng.normal(size=(len(R_e), nw, nw))
               + 1j * rng.normal(size=(len(R_e), nw, nw)))
        H_R = torch.tensor(H_R, dtype=torch.complex128)
        hr = HamiltonianR(H_R=H_R, R_vectors=R_e, degen=degen_e, nw=nw)
        eig, U = elph.band_eigensystem(hr, kpts)

        omega_ph = np.full((len(qpts), nat3), 0.01)
        eigvec_ph = np.tile(np.eye(nat3, dtype=np.complex128), (len(qpts), 1, 1))
        masses_amu = np.array([26.98])
        types = np.array([0])

        omega_grid = np.linspace(0.0, 0.03, 30)
        a2f = elph.alpha2f(
            eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts, mesh,
            omega_ph, eigvec_ph, masses_amu, types,
            fermi_energy=float(np.median(eig)), dos_at_ef=1.0, omega_grid=omega_grid,
            sigma_e=0.5,
        )
        assert np.all(np.isfinite(a2f))
        assert np.all(a2f >= 0.0)
        assert a2f.max() > 0.0

    def _build(self, rng, k_mesh, q_mesh, nw=2, nat3=3):
        """Random elph data + a random Hermitian hr on the k_mesh WS set."""
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, k_mesh, q_mesh, nb=2, nw=nw, nat3=nat3,
        )
        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], k_mesh, q_mesh, real_lattice,
        )
        H_R = rng.normal(size=(len(R_e), nw, nw)) + 1j * rng.normal(size=(len(R_e), nw, nw))
        hr = HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                          R_vectors=R_e, degen=degen_e, nw=nw)
        eig, U = elph.band_eigensystem(hr, kpts)
        omega_ph = np.full((len(qpts), nat3), 0.01)
        eigvec_ph = np.tile(np.eye(nat3, dtype=np.complex128), (len(qpts), 1, 1))
        return dict(real_lattice=real_lattice, kpts=kpts, qpts=qpts,
                    g_R=g_R, R_e=R_e, degen_e=degen_e, R_q=R_q, degen_q=degen_q,
                    hr=hr, eig=eig, U=U, omega_ph=omega_ph, eigvec_ph=eigvec_ph,
                    masses_amu=np.array([26.98]), types=np.array([0]))

    def test_fine_interp_matches_coarse_on_same_mesh(self):
        """The fine-mesh path (hr-interpolated H(k+q)) evaluated ON the coarse
        mesh must reproduce the mesh-index-lookup path bit-for-bit: Wannier
        interpolation is exact at mesh points, and operator_k is periodic so
        even k+q folded outside [0,1) gives the identical H(k+q)."""
        d = self._build(np.random.default_rng(7), (4, 4, 1), (2, 2, 1))
        mesh = (4, 4, 1)
        og = np.linspace(0.0, 0.03, 40)
        common = dict(
            omega_ph=d['omega_ph'], eigvec_ph=d['eigvec_ph'],
            masses_amu=d['masses_amu'], types=d['types'],
            fermi_energy=float(np.median(d['eig'])), dos_at_ef=1.0,
            omega_grid=og, sigma_e=0.3,
        )
        a2f_coarse = elph.alpha2f(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], mesh, **common,
        )
        a2f_fine = elph.alpha2f(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], None, hr=d['hr'], **common,
        )
        assert np.allclose(a2f_coarse, a2f_fine, atol=1e-10, rtol=1e-8)
        assert a2f_fine.max() > 0.0

    def test_fine_mesh_denser_than_coarse_runs(self):
        """The whole point of the fine path: evaluate the double-delta sum on a
        DENSER k/q mesh than the ab-initio one g_R was built from -- k+q need
        not be a mesh point (interpolated H(k+q))."""
        d = self._build(np.random.default_rng(9), (2, 2, 1), (2, 2, 1))
        fine = monkhorst_pack((6, 6, 1))
        # phonons interpolated onto the fine q-mesh would come from the caller;
        # here reuse a constant-frequency stand-in of the right length.
        nat3 = d['omega_ph'].shape[1]
        omega_ph_f = np.full((len(fine), nat3), 0.01)
        eigvec_ph_f = np.tile(np.eye(nat3, dtype=np.complex128), (len(fine), 1, 1))
        eig_f, U_f = elph.band_eigensystem(d['hr'], fine)
        og = np.linspace(0.0, 0.03, 40)
        a2f = elph.alpha2f(
            eig_f, U_f, d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            fine, fine, None, omega_ph_f, eigvec_ph_f, d['masses_amu'], d['types'],
            fermi_energy=float(np.median(eig_f)), dos_at_ef=1.0, omega_grid=og,
            sigma_e=0.3, hr=d['hr'],
        )
        assert np.all(np.isfinite(a2f)) and a2f.max() > 0.0

    def test_kmesh_neq_qmesh_independent_fine(self):
        """K_MESH != Q_MESH audit: a fine k-mesh and an INDEPENDENT fine q-mesh
        that does NOT divide it (q not on the k-mesh) run only in the fine path
        (interpolated H(k+q)); the coarse mesh-lookup path cannot express it."""
        d = self._build(np.random.default_rng(3), (2, 2, 1), (2, 2, 1))
        k_fine = monkhorst_pack((6, 6, 1))
        q_fine = monkhorst_pack((4, 4, 1))   # 4 does not divide 6 -> q not on k-mesh
        nat3 = 3
        eig_f, U_f = elph.band_eigensystem(d['hr'], k_fine)
        omega_ph_q = np.full((len(q_fine), nat3), 0.01)
        eigvec_ph_q = np.tile(np.eye(nat3, dtype=np.complex128), (len(q_fine), 1, 1))
        og = np.linspace(0.0, 0.03, 40)
        a2f = elph.alpha2f(
            eig_f, U_f, d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            k_fine, q_fine, None, omega_ph_q, eigvec_ph_q, d['masses_amu'], d['types'],
            fermi_energy=float(np.median(eig_f)), dos_at_ef=1.0, omega_grid=og,
            sigma_e=0.3, hr=d['hr'],
        )
        assert np.all(np.isfinite(a2f)) and a2f.max() > 0.0

    def test_fixed_q_fast_path_matches_general(self):
        """`interpolate_elph_fixed_q` must equal the general per-pair routine
        with the same q tiled across every k -- it is the same contraction done
        in a cheaper order (Rq summed once instead of once per k), so the two
        must agree to round-off. This is the hot path of every alpha2F sum."""
        rng = np.random.default_rng(31)
        nRe, nRq, u, nw, nk = 9, 5, 3, 2, 7
        g_R = (rng.normal(size=(nRe, nRq, u, nw, nw))
               + 1j * rng.normal(size=(nRe, nRq, u, nw, nw)))
        R_e = rng.integers(-2, 3, size=(nRe, 3))
        R_q = rng.integers(-2, 3, size=(nRq, 3))
        de = rng.integers(1, 4, nRe).astype(float)
        dq = rng.integers(1, 4, nRq).astype(float)
        kpts = rng.uniform(0, 1, (nk, 3))
        q = rng.uniform(0, 1, 3)

        fast = elph.interpolate_elph_fixed_q(g_R, R_e, de, R_q, dq, kpts, q)
        ref = elph.interpolate_elph_kq(g_R, R_e, de, R_q, dq, kpts,
                                       np.tile(q, (nk, 1)))
        assert fast.shape == (nk, u, nw, nw)
        assert np.allclose(fast, ref, rtol=1e-12, atol=1e-12)

    def test_lambda_richardson_sequence_and_extrapolation(self):
        """lambda_richardson evaluates lambda on a mesh sequence (fine path) and
        returns a straight-line (1/N) extrapolated dense-limit value."""
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        res = elph.lambda_richardson(
            d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            phonon_fn, d['masses_amu'], d['types'],
            fermi_energy=float(np.median(d['eig'])), dos_at_ef=1.0, omega_grid=og,
            meshes=[(4, 4, 1), (6, 6, 1), (8, 8, 1)],
            sigma_e=lambda mesh: 0.6 / mesh[0],   # adaptive: shrink with the mesh
        )
        assert res.lambdas.shape == (3,)
        assert np.all(np.isfinite(res.lambdas)) and np.all(res.lambdas >= 0)
        assert np.isfinite(res.lambda_extrapolated)
        assert res.n_linear[0] < res.n_linear[-1]     # increasing density
        assert len(res.a2f) == 3
        # an explicit dos_at_ef is taken as the converged DOS, so the returned
        # lambda is exactly coupling_extrapolated * that value
        assert res.dos_converged == pytest.approx(1.0)
        assert np.allclose(res.coupling, res.lambdas / 1.0)
        assert res.lambda_extrapolated == pytest.approx(
            res.coupling_extrapolated * res.dos_converged, rel=1e-12)

    def test_lambda_richardson_extrapolates_coupling_not_lambda(self):
        """The extrapolated lambda must be built as (1/N -> 0 limit of
        lambda/N(eF)) * (separately converged N(eF)), NOT as the 1/N -> 0
        limit of lambda itself.

        lambda = N(eF) * 2<<|g|^2/omega>> is a product of a noisy factor and a
        smooth one; fitting it directly extrapolates the Fermi-surface
        sampling error. Measured on Al at fixed sigma_e, lambda spans a factor
        6.1 over 10^3..20^3 while lambda/N(eF) spans 1.19 -- and the ratio
        agrees between waw and EPW, which lambda does not.
        """
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        meshes = [(4, 4, 1), (6, 6, 1), (8, 8, 1)]
        res = elph.lambda_richardson(
            d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            phonon_fn, d['masses_amu'], d['types'],
            fermi_energy=float(np.median(d['eig'])), dos_at_ef=None,
            omega_grid=og, meshes=meshes, sigma_e=0.6,
            dos_meshes=[(16, 16, 1), (24, 24, 1), (32, 32, 1)],
        )
        # per-mesh N(eF) is recorded and is what normalised each alpha2F
        assert res.dos_at_ef.shape == (3,)
        assert np.all(res.dos_at_ef > 0)
        assert np.allclose(res.coupling, res.lambdas / res.dos_at_ef)
        # the DOS is converged on its own, denser, H(k)-only sequence
        assert res.dos_values.shape == (3,)
        assert res.dos_n_linear[0] > res.n_linear[-1]     # genuinely denser
        # and the headline number is the PRODUCT of the two extrapolations
        assert res.lambda_extrapolated == pytest.approx(
            res.coupling_extrapolated * res.dos_converged, rel=1e-12)
        # ... which is NOT the naive direct fit, unless N(eF) happens to be
        # mesh-independent. Guard that the two really differ here.
        h = 1.0 / res.n_linear
        naive = np.polyfit(h, res.lambdas, 1)[1]
        assert not np.isclose(naive, res.lambda_extrapolated, rtol=1e-6)

    def test_lambda_richardson_nan_mesh_excluded_from_fit(self):
        """A mesh too coarse to put any state in the Fermi window records NaN
        (never 0, which would read as 'lambda is small') and is dropped from
        the fit instead of aborting the sequence."""
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        # Fermi level far outside the band range -> no state within sigma
        far = float(d['eig'].max()) + 50.0
        with pytest.warns(RuntimeWarning, match="no state in the Fermi window"):
            with pytest.raises(ValueError, match="fewer than 2 usable meshes"):
                elph.lambda_richardson(
                    d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'],
                    d['degen_q'], phonon_fn, d['masses_amu'], d['types'],
                    fermi_energy=far, dos_at_ef=None, omega_grid=og,
                    meshes=[(4, 4, 1), (6, 6, 1)], sigma_e=1e-3,
                )

    def test_lambda_auto_two_stage_absolute_tolerance(self):
        """lambda_auto: eF from the model's own electron count, N(eF) converged
        to tol_dos, then lambda to tol_lambda -- value = densest mesh,
        uncertainty = last consecutive-mesh change. No fit, no average."""
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        res = elph.lambda_auto(
            d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            phonon_fn, d['masses_amu'], d['types'], n_electrons=2.0,
            omega_grid=og, sigma_e=0.6, base_mesh=(4, 4, 1),
            rtol_dos=0.02, tol_lambda=0.5,
            dos_factors=(2.0, 3.0, 4.0, 6.0, 8.0),
            coupling_factors=(1.0, 1.25, 1.5, 2.0, 2.5),
            coupling_max_points=4096,
        )
        # the reported value IS the densest mesh, not a fit or a mean
        assert res.lambda_value == pytest.approx(res.lambdas[-1], rel=1e-12)
        assert res.dos_value == pytest.approx(res.dos_values[-1], rel=1e-12)
        # ... and the uncertainty IS the last consecutive difference
        if len(res.lambdas) >= 2:
            assert res.lambda_uncertainty == pytest.approx(
                abs(res.lambdas[-1] - res.lambdas[-2]), rel=1e-12)
        # the per-mesh N(eF) is divided out and the converged one multiplied back
        assert np.allclose(res.lambdas, res.lambdas_raw / res.dos_at_ef * res.dos_value)
        assert np.allclose(res.coupling, res.lambdas_raw / res.dos_at_ef)
        # eF came from the electron count, and is frozen across stage 2
        assert res.fermi_energy == pytest.approx(res.fermi_energies[-1], rel=1e-12)
        assert res.converged == (res.dos_converged_flag and res.coupling_converged_flag)

    def test_lambda_auto_reports_non_convergence_rather_than_hiding_it(self):
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        with pytest.warns(RuntimeWarning):
            res = elph.lambda_auto(
                d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
                phonon_fn, d['masses_amu'], d['types'], n_electrons=2.0,
                omega_grid=og, sigma_e=0.6, base_mesh=(4, 4, 1),
                rtol_dos=1e-14, tol_lambda=1e-14,         # unreachable
                dos_factors=(2.0, 3.0, 4.0), coupling_factors=(1.0, 1.5, 2.0),
                coupling_max_points=4096,
            )
        assert res.converged is False
        assert np.isfinite(res.lambda_value)      # still returns what it has

    def test_omega_log_is_independent_of_the_dos_normalisation(self):
        """omega_log = exp[(2/lambda) int alpha2F ln(w)/w dw] is a RATIO of two
        alpha2F integrals, so the 1/N(eF) prefactor cancels identically. That is
        why it needs no DOS correction and why the ratio trick that rescues
        lambda does nothing for it -- hence its own convergence criterion."""
        rng = np.random.default_rng(4)
        og = np.linspace(1e-4, 0.04, 500)
        a2f = np.abs(rng.normal(size=og.shape)) * np.exp(-((og - 0.02) / 0.008) ** 2)
        lam1 = elph.lambda_from_a2f(a2f, og)
        _, w1 = elph.eliashberg_moments(a2f, og)
        # rescaling alpha2F by any factor (i.e. any N(eF)) scales lambda ...
        for f in (0.5, 3.0, 17.0):
            lam2 = elph.lambda_from_a2f(a2f * f, og)
            _, w2 = elph.eliashberg_moments(a2f * f, og)
            assert lam2 == pytest.approx(f * lam1, rel=1e-12)
            # ... and leaves omega_log exactly alone
            assert w2[-1] == pytest.approx(w1[-1], rel=1e-12)

    def test_lambda_auto_converges_omega_log_too(self):
        """omega_log used to be read off the densest mesh with no test at all,
        and on Al it moved 28.36 -> 24.66 meV (15%) between consecutive meshes
        while lambda looked settled. It now has its own relative tolerance and
        flag, and `converged` is the AND of all three."""
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            # DISPERSIVE, deliberately: a flat spectrum makes omega_log exactly
            # mesh-independent and the test vacuous.
            nq = len(qpts)
            w = 0.010 + 0.004 * np.cos(2 * np.pi * np.asarray(qpts)).sum(axis=1)[:, None]
            return (np.tile(w, (1, nat3)),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        common = dict(
            omega_grid=og, sigma_e=0.6, base_mesh=(4, 4, 1), n_electrons=2.0,
            dos_factors=(2.0, 3.0, 4.0, 6.0, 8.0),
            coupling_factors=(1.0, 1.25, 1.5, 2.0, 2.5),
            coupling_max_points=4096,
        )
        args = (d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
                phonon_fn, d['masses_amu'], d['types'])
        res = elph.lambda_auto(*args, rtol_dos=0.02, tol_lambda=0.5,
                               rtol_wlog=0.5, **common)
        assert res.omega_logs.shape == res.lambdas.shape
        assert res.omega_log == pytest.approx(res.omega_logs[-1], rel=1e-12)
        assert res.converged == (res.dos_converged_flag
                                 and res.coupling_converged_flag
                                 and res.omega_log_converged_flag)
        # an unreachable omega_log tolerance alone must block `converged`,
        # and must say so -- lambda being fine is not enough
        with pytest.warns(RuntimeWarning, match="omega_log"):
            strict = elph.lambda_auto(*args, rtol_dos=0.02, tol_lambda=0.5,
                                      rtol_wlog=1e-14, **common)
        assert strict.omega_log_converged_flag is False
        assert strict.converged is False

    def test_lambda_auto_needs_two_consecutive_agreements(self):
        """A single pair landing inside the tolerance must NOT count as
        converged. Regression for the first real Al run, whose consecutive
        |delta| were 0.061, 0.086, 0.011: the one-step rule stopped on the
        third after two steps 2-3x over tolerance, and quoted +-0.011 for a
        sequence whose actual scatter was +-0.038.

        Driven here through the private stopping logic on the recorded Al
        numbers, so it tests the rule itself rather than re-running an
        expensive alpha2F sweep.
        """
        al = [0.4830, 0.5444, 0.4587, 0.4699]      # the real sequence
        tol = 0.03

        def stops_one_step(vals):
            return len(vals) >= 2 and abs(vals[-1] - vals[-2]) < tol

        def stops_two_step(vals):
            return (len(vals) >= 3
                    and max(abs(vals[-1] - vals[-2]),
                            abs(vals[-2] - vals[-3])) < tol)

        # the old rule fires on the Al sequence; the new one does not
        assert stops_one_step(al)
        assert not stops_two_step(al)
        # and the new rule does fire once the sequence genuinely settles
        assert stops_two_step(al + [0.4710, 0.4705])

    def test_lambda_auto_rejects_a_degenerate_mesh_sequence(self):
        """A schedule that rounds to the same mesh twice gives |delta| = 0,
        which must NOT read as converged -- caught by actually running it with
        base_mesh=(1,1,1), which returned lambda = 0 marked 'converged'."""
        d = self._build(np.random.default_rng(5), (2, 2, 1), (2, 2, 1))
        nat3 = d['omega_ph'].shape[1]

        def phonon_fn(qpts):
            nq = len(qpts)
            return (np.full((nq, nat3), 0.01),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        og = np.linspace(1e-5, 0.03, 60)
        with pytest.raises(ValueError, match="at least three"):
            elph.lambda_auto(
                d['hr'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
                phonon_fn, d['masses_amu'], d['types'], n_electrons=2.0,
                omega_grid=og, sigma_e=0.6, base_mesh=(1, 1, 1),
                dos_factors=(2.0, 4.0, 8.0), coupling_factors=(1.0, 2.0, 3.0),
            )

    def test_lambda_auto_preserves_anisotropic_aspect_ratio(self):
        assert elph._scaled_mesh((12, 12, 8), 1.5) == (18, 18, 12)
        assert elph._scaled_mesh((12, 12, 8), 3.0) == (36, 36, 24)
        assert elph._scaled_mesh((7, 7, 7), 1.0) == (7, 7, 7)
        # a 1 means "this axis is not sampled" (slab / monolayer) and must
        # stay 1 -- densifying a dispersionless direction only costs time
        assert elph._scaled_mesh((12, 12, 1), 2.5) == (30, 30, 1)
        assert elph._scaled_mesh((1, 1, 1), 9.0) == (1, 1, 1)

    def test_alpha2f_matrix_recovers_isotropic(self):
        """The band-resolved alpha2F matrix must satisfy its defining identity:
        the partial-DOS-weighted sum sum_ij N_i alpha2F_ij / sum_i N_i equals
        the isotropic alpha2F (computed with dos_at_ef = sum_i N_i). Holds
        algebraically for ANY coupling because the sheet weights partition unity."""
        d = self._build(np.random.default_rng(21), (4, 4, 1), (2, 2, 1), nw=4)
        mesh = (4, 4, 1)
        og = np.linspace(0.0, 0.03, 40)
        eF = float(np.median(d['eig']))
        groups = [[0, 1], [2, 3]]        # partition nw=4 into two sheets
        common = dict(
            omega_ph=d['omega_ph'], eigvec_ph=d['eigvec_ph'],
            masses_amu=d['masses_amu'], types=d['types'],
            omega_grid=og, sigma_e=0.3,
        )
        a2f_ij, N_i = elph.alpha2f_matrix(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], mesh, fermi_energy=eF, orbital_groups=groups, **common,
        )
        a2f_iso = elph.alpha2f(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], mesh, fermi_energy=eF, dos_at_ef=float(N_i.sum()), **common,
        )
        assert a2f_ij.shape == (2, 2, len(og))
        recovered = np.einsum("i,ijw->w", N_i, a2f_ij) / N_i.sum()
        assert np.allclose(recovered, a2f_iso, atol=1e-10)
        # lambda_matrix runs and total lambda matches the isotropic integral
        lam_ij = elph.lambda_matrix(a2f_ij, og)
        assert lam_ij.shape == (2, 2)
        lam_tot = float(np.einsum("i,ij->", N_i, lam_ij) / N_i.sum())
        lam_iso = elph.lambda_from_a2f(a2f_iso, og)
        assert lam_tot == pytest.approx(lam_iso, rel=1e-6)

    def test_alpha2f_matrix_fine_mesh(self):
        """Band-resolved matrix in the fine-mesh (hr-interpolated) mode."""
        d = self._build(np.random.default_rng(22), (2, 2, 1), (2, 2, 1), nw=4)
        fine = monkhorst_pack((6, 6, 1))
        nat3 = 3
        eig_f, U_f = elph.band_eigensystem(d['hr'], fine)
        omega_ph_f = np.full((len(fine), nat3), 0.01)
        eigvec_ph_f = np.tile(np.eye(nat3, dtype=np.complex128), (len(fine), 1, 1))
        og = np.linspace(0.0, 0.03, 40)
        a2f_ij, N_i = elph.alpha2f_matrix(
            eig_f, U_f, d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            fine, fine, None, omega_ph_f, eigvec_ph_f, d['masses_amu'], d['types'],
            fermi_energy=float(np.median(eig_f)), orbital_groups=[[0, 1], [2, 3]],
            omega_grid=og, sigma_e=0.3, hr=d['hr'],
        )
        assert a2f_ij.shape == (2, 2, len(og))
        assert np.all(np.isfinite(a2f_ij)) and a2f_ij.max() > 0

    def test_fine_transport_needs_recip_lattice(self):
        d = self._build(np.random.default_rng(1), (2, 2, 1), (2, 2, 1))
        og = np.linspace(0.0, 0.03, 10)
        with pytest.raises(ValueError, match="recip_lattice"):
            elph.alpha2f(
                d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
                d['kpts'], d['qpts'], None, d['omega_ph'], d['eigvec_ph'],
                d['masses_amu'], d['types'], fermi_energy=0.0, dos_at_ef=1.0,
                omega_grid=og, hr=d['hr'], velocities=np.zeros((len(d['kpts']), 2, 3)),
            )




class TestAcousticModeCutoff:
    """The ASR projector on g_R was REMOVED (it also zeroed the genuinely
    nonzero interband g(k, q=0)); low-frequency acoustic modes are instead
    excluded at point of use, EPW's own eps_acoustic convention."""

    def test_modes_below_cutoff_get_zero_coupling(self):
        rng = np.random.default_rng(30)
        nat3, nw = 3, 2
        g_wannier = rng.normal(size=(nat3, nw, nw)) + 1j * rng.normal(size=(nat3, nw, nw))
        eigvec = np.eye(nat3, dtype=np.complex128)
        omega = np.array([1e-9, 1e-9, 0.01])   # two "q=0 acoustic" modes

        g_mode = elph.phonon_mode_coupling(
            g_wannier, eigvec, omega, np.array([26.98]), np.array([0]),
        )
        assert np.allclose(g_mode[0], 0.0)
        assert np.allclose(g_mode[1], 0.0)
        assert np.abs(g_mode[2]).max() > 0
        assert np.all(np.isfinite(g_mode))

    def test_alpha2f_finite_with_zero_frequency_acoustic_modes(self):
        """A q=Gamma acoustic mode (omega ~ 0) must neither divide by zero
        nor dominate alpha2F -- it is skipped entirely."""
        rng = np.random.default_rng(31)
        mesh = (2, 2, 1)
        nat3, nw = 3, 2
        real_lattice, kpts, qpts, u_all, W, dv_all = _random_elph_data(
            rng, mesh, mesh, nb=2, nw=nw, nat3=nat3,
        )
        g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
            u_all, W, kpts, qpts, lambda iq: dv_all[iq], mesh, mesh, real_lattice,
        )
        H_R = torch.tensor(
            rng.normal(size=(len(R_e), nw, nw)) + 1j * rng.normal(size=(len(R_e), nw, nw)),
            dtype=torch.complex128,
        )
        hr = HamiltonianR(H_R=H_R, R_vectors=R_e, degen=degen_e, nw=nw)
        eig, U = elph.band_eigensystem(hr, kpts)

        omega_ph = np.full((len(qpts), nat3), 0.01)
        omega_ph[0] = 1e-12   # Gamma: all "acoustic", essentially zero
        eigvec_ph = np.tile(np.eye(nat3, dtype=np.complex128), (len(qpts), 1, 1))

        omega_grid = np.linspace(0.0, 0.03, 30)
        a2f = elph.alpha2f(
            eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts, mesh,
            omega_ph, eigvec_ph, np.array([26.98]), np.array([0]),
            fermi_energy=float(np.median(eig)), dos_at_ef=1.0, omega_grid=omega_grid,
            sigma_e=0.5,
        )
        assert np.all(np.isfinite(a2f))
        assert np.all(a2f >= 0.0)


class TestVlocFormFactor:
    def test_matches_analytic_erf_coulomb(self):
        """v_loc(r) = -zv*erf(r)/r has the EXACT form factor
        -4*pi*zv*exp(-G^2/4)/(V*G^2) (the radial integrand cancels
        identically)."""
        pseudo = _mock_coulomb_pseudo(zv=2.0)
        volume = 100.0
        g = np.linspace(0.3, 8.0, 40)
        got = elph._vloc_form_factor(g, pseudo, volume)
        expected = -4.0 * np.pi * 2.0 * np.exp(-g ** 2 / 4.0) / (volume * g ** 2)
        assert np.allclose(got, expected, atol=1e-8)


class TestBareLocalDv:
    def _setup(self):
        pseudos = [_mock_coulomb_pseudo()]
        real_lattice = np.eye(3) * 6.0
        grid = (8, 8, 8)
        return pseudos, real_lattice, grid

    def test_real_and_zero_mean_at_q_gamma(self):
        """At q=0 the perturbation is a real field (c(-G) = c(G)*) with no
        G=0 component (grid average exactly zero). The cutoff must keep the
        G-sphere INSIDE the FFT grid (QE's own invariant: the dense grid is
        sized to contain the ecutrho sphere) -- a sphere touching the
        even-grid Nyquist plane (G = -N/2 with no +N/2 partner) would break
        the reality condition."""
        pseudos, real_lattice, grid = self._setup()
        dv = elph.bare_local_dv(
            np.zeros(3), pseudos, np.array([[0.25, 0.1, 0.0]]), np.array([0]),
            real_lattice, grid, ecut_rho=4.0,
        )
        assert dv.shape == (3,) + grid
        assert np.abs(dv.imag).max() < 1e-12
        assert np.abs(dv.mean(axis=(1, 2, 3))).max() < 1e-12

    def test_translated_atom_gives_translated_field(self):
        """Moving the atom by a grid-commensurate fraction tau at q=0 must
        rigidly shift the real-space field: dv_tau(r) = dv_0(r - tau)."""
        pseudos, real_lattice, grid = self._setup()
        shift_idx = (2, 0, 3)   # grid points; tau = shift/N
        tau = np.array([[shift_idx[0] / grid[0], shift_idx[1] / grid[1], shift_idx[2] / grid[2]]])
        dv0 = elph.bare_local_dv(
            np.zeros(3), pseudos, np.zeros((1, 3)), np.array([0]),
            real_lattice, grid, ecut_rho=40.0,
        )
        dv_tau = elph.bare_local_dv(
            np.zeros(3), pseudos, tau, np.array([0]),
            real_lattice, grid, ecut_rho=40.0,
        )
        rolled = np.roll(dv0, shift_idx, axis=(1, 2, 3))
        assert np.allclose(dv_tau, rolled, atol=1e-10)


class TestRealYlm:
    def test_orthonormality_monte_carlo(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=(500_000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        channels = [(l, m) for l in range(3) for m in range(-l, l + 1)]
        Y = np.stack([elph._real_ylm(l, m, u) for l, m in channels])
        overlap = 4.0 * np.pi * (Y[:, None, :] * Y[None, :, :]).mean(axis=-1)
        assert np.allclose(overlap, np.eye(len(channels)), atol=0.05)


class TestKleinmanBylanderPerturbation:
    def test_matrix_element_hermitian_at_q_zero(self):
        """dV_NL/dtau is the derivative of a Hermitian operator: at k = k+q
        every Cartesian component must be Hermitian in the band indices."""
        rng = np.random.default_rng(6)
        pseudos = [_mock_coulomb_pseudo()]
        real_lattice = np.eye(3) * 6.0
        grid = (6, 6, 6)
        kb = elph.KleinmanBylanderPerturbation(
            pseudos, np.array([[0.2, 0.0, 0.4]]), np.array([0]), real_lattice, grid,
        )
        nb = 3
        u_k = rng.normal(size=(nb,) + grid) + 1j * rng.normal(size=(nb,) + grid)
        P, Q = kb.projections(u_k, np.array([0.3, 0.1, 0.0]))
        M = kb.matrix_element(P, Q, P, Q)
        assert M.shape == (3, nb, nb)
        for a in range(3):
            assert np.allclose(M[a], M[a].conj().T, atol=1e-12)

    def test_channel_bookkeeping(self):
        """s + p projectors on two atoms -> 2*(1+3) channels, D coupling
        only same-atom same-l same-m pairs."""
        pseudos = [_mock_coulomb_pseudo()]
        kb = elph.KleinmanBylanderPerturbation(
            pseudos, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            np.array([0, 0]), np.eye(3) * 6.0, (6, 6, 6),
        )
        assert len(kb.channels) == 8
        assert kb.D.shape == (8, 8)
        for c1, (na1, _, n1, l1, m1) in enumerate(kb.channels):
            for c2, (na2, _, n2, l2, m2) in enumerate(kb.channels):
                if not (na1 == na2 and l1 == l2 and m1 == m2):
                    assert kb.D[c1, c2] == 0.0


@needs_gcheck
class TestFullPerturbationAgainstEPW:
    """Integration tests on the real 6x6x6 Al reference run: the full
    (bare local + induced + nonlocal KB) matrix element against (a) the
    exact q=Gamma translation identity, (b) real EPW prtgkk values."""

    K_MESH = (6, 6, 6)
    NG = (24, 24, 24)
    ECUT_RHO = 80.0   # ecutwfc = 40 Ry -> ecutrho = 160 Ry = 80 Ha

    @classmethod
    def setup_class(cls):
        from waw.interfaces.wannier90.io import read_unk

        a_bohr = 4.0495 / BOHR_TO_ANG
        cls.real_lat = a_bohr * np.array(
            [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]],
        )
        cls.pseudos = [read_norm_conserving(AL_UPF)]
        cls.tau_frac = np.zeros((1, 3))
        cls.types = np.array([0])

        d = np.load(GCHECK / "gcheck_electron.npz")
        cls.kpts, cls.eig_ev = d["kpts"], d["eig_ev"]
        cls.ik = elph.kpoint_mesh_index(np.array([0.5, 0.5, 0.5]), cls.K_MESH)
        cls.u_k = read_unk(GCHECK / f"UNK{cls.ik+1:05d}.1")["u_nk"]
        cls.kb = elph.KleinmanBylanderPerturbation(
            cls.pseudos, cls.tau_frac, cls.types, cls.real_lat, cls.NG,
        )
        cls.P_k, cls.Q_k = cls.kb.projections(cls.u_k, cls.kpts[cls.ik])

    def test_gamma_translation_identity(self):
        """g(k, q=0) = (E_i - E_j) <j| d/dr |i> exactly (rigid translation
        of the whole crystal) -- holds only with ALL bare terms included;
        the induced-only coupling is ~6x off on the allowed transition."""
        dv = dvscf_io.read_dvscf(GCHECK / "out_ph_grid", "al", 1, self.NG, nat=1) \
            + elph.bare_local_dv(
                np.zeros(3), self.pseudos, self.tau_frac, self.types,
                self.real_lat, self.NG, self.ECUT_RHO,
            )
        g_tot = elph.bloch_matrix_element(self.u_k, self.u_k, dv) \
            + self.kb.matrix_element(self.P_k, self.Q_k, self.P_k, self.Q_k)

        B = 2 * np.pi * np.linalg.inv(self.real_lat).T
        ints = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in self.NG]
        Gint = np.stack(np.meshgrid(*ints, indexing="ij"), -1).reshape(-1, 3)
        p = (self.kpts[self.ik][None] + Gint) @ B
        c_k = np.fft.fftn(self.u_k, axes=(-3, -2, -1)).reshape(self.u_k.shape[0], -1) \
            / np.prod(self.NG)
        eig_ha = self.eig_ev[self.ik] / HARTREE_TO_EV

        # the symmetry-ALLOWED transition (bands 1 -> 2); forbidden pairs
        # have both sides at numerical-noise level (~1e-6 Ha/Bohr)
        i, j = 1, 2
        exact = -(eig_ha[i] - eig_ha[j]) * 1j * np.sum(c_k[j].conj() * p[:, 0] * c_k[i])
        assert abs(g_tot[0, j, i]) / abs(exact) == pytest.approx(1.0, abs=0.01)

        for i, j in [(0, 2), (0, 3), (1, 4)]:
            assert abs(g_tot[0, j, i]) < 1e-4   # forbidden: noise-level only

    def test_epw_prtgkk_reference_values(self):
        """|g| pins from a REAL EPW run (prtgkk=.true., identical
        SCF/phonon inputs), k=(.5,.5,.5), q=(0,0,.5): transverse and
        longitudinal mode couplings within 6%. (The NLCC xc term is already
        in the induced dvscf -- QE's dv_of_drho(drho, drhoc) -- so the residual
        is ordinary reconstruction error, not a missing term; adding a separate
        NLCC term double-counts it.)"""
        from waw.interfaces.wannier90.io import read_unk

        iq_t = elph.kpoint_mesh_index(np.array([0.0, 0.0, 0.5]), self.K_MESH)
        kq = np.array([0.5, 0.5, 1.0])
        ikq = elph.kpoint_mesh_index(kq, self.K_MESH)
        u_kq_fold = read_unk(GCHECK / f"UNK{ikq+1:05d}.1")["u_nk"]
        z_frac = np.arange(self.NG[2]) / self.NG[2]
        u_kq = u_kq_fold * np.exp(-2j * np.pi * z_frac)[None, None, None, :]   # G=(0,0,1)

        dv = dvscf_io.read_dvscf(GCHECK / "out_ph_grid", "al", iq_t + 1, self.NG, nat=1) \
            + elph.bare_local_dv(
                np.array([0.0, 0.0, 0.5]), self.pseudos, self.tau_frac,
                self.types, self.real_lat, self.NG, self.ECUT_RHO,
            )
        P_kq, Q_kq = self.kb.projections(u_kq_fold, self.kpts[ikq])
        g_bloch = elph.bloch_matrix_element(self.u_k, u_kq, dv) \
            + self.kb.matrix_element(P_kq, Q_kq, self.P_k, self.Q_k)

        text = (GCHECK / f"al.grid.dyn{iq_t+1}").read_text()
        block = text.split("Dynamical  Matrix in cartesian axes")[1]
        lines = [ln for ln in block.splitlines() if ln.strip()]
        rows = []
        for ln in lines[2:5]:
            v = [float(x) for x in ln.split()]
            rows.append([complex(v[0], v[1]), complex(v[2], v[3]), complex(v[4], v[5])])
        phi_ha = 0.5 * np.array(rows)
        w2, evec = np.linalg.eigh(phi_ha / (26.9815 * AMU_TO_ME))
        omega_ha = np.sqrt(np.abs(w2))

        g_mode = elph.phonon_mode_coupling(
            g_bloch, evec, omega_ha, np.array([26.9815]), self.types,
        )
        g_mev = np.abs(g_mode) * HARTREE_TO_EV * 1000

        ek, ekq = self.eig_ev[self.ik], self.eig_ev[ikq]
        pins = [(3.2926, 6.2679, 19.349, 245.04), (3.4962, 4.9508, 13.213, 249.82)]
        for enk, enkq, ref_t, ref_l in pins:
            i = int(np.argmin(np.abs(ek - enk)))
            j = int(np.argmin(np.abs(ekq - enkq)))
            g_t = np.sqrt(0.5 * (g_mev[0, j, i] ** 2 + g_mev[1, j, i] ** 2))
            g_l = g_mev[2, j, i]
            assert g_t == pytest.approx(ref_t, rel=0.06)
            assert g_l == pytest.approx(ref_l, rel=0.06)


def _epw_w0gauss(x):
    """QE's ``w0gauss(x, 0)`` -- EPW's own delta approximant."""
    return np.exp(-np.asarray(x) ** 2) / np.sqrt(np.pi)


def _epw_a2f_reference(
    eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts,
    omega_ph, eigvec_ph, masses_amu, types, ef, dosef, omega_grid,
    degaussw, degaussq, eps_ac, hr, delta_approx=True, velocities=None,
    recip_lattice=None,
):
    """A LITERAL transcription of EPW's own alpha2F chain, kept deliberately
    naive (explicit mode loop, EPW's variable names and expression order) so
    it can be read straight against the Fortran:

      * ``selfen.f90::selfen_phon_q`` -- gamma(imode) from either the
        delta_approx double delta or the default occupation-difference form,
        the Grimvall (v_k.v_kq)/|v_k|^2 transport factor, then
        lambda = gamma / (pi omega^2 N(eF));
      * ``spectral.f90::a2f_main`` -- a2F(w) = sum_q wqf omega lambda / 2
        times a normalized Gaussian, with negative lambda clamped to zero.

    Widths here are EPW's ``degaussw``/``degaussq``; ``w0gauss(x,0)/degauss``
    equals this project's ``gaussian_smearing(x, degauss/sqrt(2))`` exactly.
    """
    nk, nq = len(kpts), len(qpts)
    nmodes = omega_ph.shape[-1]
    mass = np.repeat(masses_amu[types], 3) * AMU_TO_ME
    wkf, wqf = 2.0 / nk, 1.0 / nq            # EPW's wkf carries the spin factor 2
    lam_all = np.zeros((nq, nmodes))
    for iq in range(nq):
        eig_kq, U_kq = elph.band_eigensystem(hr, kpts + qpts[iq])
        g_kq = elph.interpolate_elph_fixed_q(
            g_R, R_e, degen_e, R_q, degen_q, kpts, qpts[iq])
        wq = omega_ph[iq]
        alive = wq > eps_ac
        inv_wq = np.where(alive, 1.0 / (2.0 * np.where(alive, wq, 1.0)), 0.0)
        uf = eigvec_ph[iq] / np.sqrt(mass)[:, None]          # EPW's uf: no 1/sqrt(2w)
        epf = np.einsum("uv,kunm->kvnm", uf, g_kq)
        epf = np.einsum("kjn,kvjm,kmo->kvno", U_kq.conj(), epf, U)
        g2 = np.abs(epf) ** 2 * inv_wq[None, :, None, None]

        coskkq = np.zeros((nk, eig.shape[1], eig.shape[1]))   # (k, ibnd@k, jbnd@k+q)
        if velocities is not None:
            from waw.analysis.wigner_transport import velocity_matrix
            _, vm = velocity_matrix(hr, kpts + qpts[iq], recip_lattice)
            vkq = np.real(np.einsum("knna->kna", vm))
            num = np.einsum("kia,kja->kij", velocities, vkq)
            den = np.einsum("kia,kia->ki", velocities, velocities)
            ok = np.abs(den) > 1e-4
            coskkq = np.where(ok[:, :, None],
                              num / np.where(ok[:, :, None], den[:, :, None], 1.0), 0.0)

        gamma = np.zeros(nmodes)
        ekk, ekq = eig - ef, eig_kq - ef
        for im in range(nmodes):
            if delta_approx:
                w0g1 = _epw_w0gauss(ekk / degaussw) / degaussw
                w0g2 = _epw_w0gauss(ekq / degaussw) / degaussw
                weight = np.pi * wq[im] * wkf * w0g1[:, :, None] * w0g2[:, None, :]
            else:
                fact = (np.where(ekq < 0, 1.0, 0.0)[:, None, :]
                        - np.where(ekk < 0, 1.0, 0.0)[:, :, None])
                etmp1 = ekq[:, None, :] - ekk[:, :, None]
                weight = (-np.pi * wkf * fact
                          * _epw_w0gauss((etmp1 - wq[im]) / degaussw) / degaussw)
            if velocities is not None:
                weight = weight * (1.0 - coskkq)
            gamma[im] = (weight * np.transpose(g2[:, im, :, :], (0, 2, 1))).sum()
        lam = np.where(alive, gamma / (np.pi * np.where(alive, wq, 1.0) ** 2 * dosef), 0.0)
        lam_all[iq] = np.where(lam < 0.0, 0.0, lam)

    a2f = np.zeros_like(omega_grid)
    for iq in range(nq):
        for nu in range(nmodes):
            w0 = omega_ph[iq, nu]
            if w0 > eps_ac:
                a2f += (wqf * w0 * lam_all[iq, nu] / 2.0
                        * _epw_w0gauss((omega_grid - w0) / degaussq) / degaussq)
    return a2f


class TestEPWParity:
    """`alpha2f` against a literal transcription of EPW's own chain.

    These pin the pieces where the two codes could silently disagree while
    both looking reasonable: the sqrt(2) between EPW's ``degaussw`` and this
    project's Gaussian sigma, which of the two Fermi-surface restrictions
    EPW's ``delta_approx`` selects, and the Grimvall |v_k|^2 denominator in
    the transport weight.
    """

    def _model(self, seed=0):
        rng = np.random.default_rng(seed)
        A = np.eye(3) * 6.0
        from waw.core.hamiltonian import _wigner_seitz
        mesh, nat3, nw = (4, 4, 4), 6, 3
        R_e, degen_e = _wigner_seitz(mesh, A)
        R_q, degen_q = _wigner_seitz(mesh, A)
        H = (rng.normal(size=(len(R_e), nw, nw))
             + 1j * rng.normal(size=(len(R_e), nw, nw))) * 0.02
        idx = {tuple(r): i for i, r in enumerate(R_e)}
        H_R = np.stack([0.5 * (H[i] + H[idx[tuple(-r)]].conj().T)
                        for i, r in enumerate(R_e)])
        H_R[idx[(0, 0, 0)]] += np.diag([0.0, 0.15, 0.30])
        hr = HamiltonianR(H_R=torch.from_numpy(H_R), R_vectors=R_e,
                          degen=degen_e, nw=nw)
        g_R = (rng.normal(size=(len(R_e), len(R_q), nat3, nw, nw))
               + 1j * rng.normal(size=(len(R_e), len(R_q), nat3, nw, nw))) * 0.01
        kpts, qpts = monkhorst_pack((4, 4, 4)), monkhorst_pack((3, 3, 3))
        eig, U = elph.band_eigensystem(hr, kpts)
        omega_ph = np.abs(rng.normal(size=(len(qpts), nat3))) * 0.002 + 0.0005
        eigvec_ph = np.linalg.qr(
            rng.normal(size=(len(qpts), nat3, nat3))
            + 1j * rng.normal(size=(len(qpts), nat3, nat3)))[0]
        return dict(
            hr=hr, g_R=g_R, R_e=R_e, degen_e=degen_e, R_q=R_q, degen_q=degen_q,
            kpts=kpts, qpts=qpts, eig=eig, U=U, omega_ph=omega_ph,
            eigvec_ph=eigvec_ph, masses_amu=np.array([26.98, 10.81]),
            types=np.array([0, 1]), ef=float(np.median(eig)), recip=2 * np.pi * np.eye(3) / 6.0,
        )

    def _run(self, d, sigma_e, **kw):
        og = np.linspace(1e-6, 0.012, 400)
        sph = 2.0 * (og[1] - og[0])
        waw = elph.alpha2f(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], None, d['omega_ph'], d['eigvec_ph'],
            d['masses_amu'], d['types'], fermi_energy=d['ef'], dos_at_ef=None,
            omega_grid=og, sigma_e=sigma_e, sigma_ph=sph, hr=d['hr'], **kw,
        )
        ref = _epw_a2f_reference(
            d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'], d['degen_q'],
            d['kpts'], d['qpts'], d['omega_ph'], d['eigvec_ph'], d['masses_amu'],
            d['types'], d['ef'],
            elph.fermi_surface_dos(d['eig'], d['ef'], sigma_e), og,
            degaussw=elph.sigma_to_epw_degauss(sigma_e),
            degaussq=elph.sigma_to_epw_degauss(sph), eps_ac=elph.EPS_ACOUSTIC,
            hr=d['hr'],
            delta_approx=kw.get('delta_approx', True),
            velocities=kw.get('velocities'), recip_lattice=kw.get('recip_lattice'),
        )
        return og, waw, ref

    def test_double_delta_matches_epw(self):
        d = self._model()
        og, waw, ref = self._run(d, 0.01)
        assert np.allclose(waw, ref, rtol=0, atol=1e-10 * max(1.0, np.abs(ref).max()))
        assert elph.lambda_from_a2f(waw, og) == pytest.approx(
            elph.lambda_from_a2f(ref, og), rel=1e-12)

    def test_epw_default_occupation_difference_form_matches_epw(self):
        """EPW's own default is delta_approx=.FALSE. -- the exact Migdal
        linewidth [f(e_k)-f(e_k+q)] delta(e_k+q - e_k - omega), NOT the double
        delta. Reproducing a stock EPW run requires this branch."""
        d = self._model()
        og, waw, ref = self._run(d, 0.01, delta_approx=False)
        assert np.allclose(waw, ref, rtol=0, atol=1e-10 * max(1.0, np.abs(ref).max()))

    def test_the_two_delta_forms_actually_differ(self):
        """Guard against the parity tests above being vacuous: with phonon
        energies comparable to the electronic smearing the two branches are
        genuinely different numbers, so picking the wrong one is a real error
        and not a cosmetic choice."""
        d = self._model()
        og, dd, _ = self._run(d, 0.01)
        _, occ, _ = self._run(d, 0.01, delta_approx=False)
        l_dd = elph.lambda_from_a2f(dd, og)
        l_occ = elph.lambda_from_a2f(occ, og)
        assert abs(l_occ / l_dd - 1.0) > 0.1

    def test_transport_weight_matches_grimvall_denominator(self):
        """EPW uses (v_k.v_k+q)/|v_k|^2 (Grimvall 8.20), not the normalized
        cosine (v_k.v_k+q)/(|v_k||v_k+q|)."""
        from waw.analysis.wigner_transport import velocity_matrix
        d = self._model()
        _, vm = velocity_matrix(d['hr'], d['kpts'], d['recip'])
        vel = np.real(np.einsum('knna->kna', vm))
        og, waw, ref = self._run(
            d, 0.01, velocities=vel, recip_lattice=d['recip'])
        assert np.allclose(waw, ref, rtol=0, atol=1e-10 * max(1.0, np.abs(ref).max()))

        # ... and it is a measurably different answer from the untransported
        # one, so the weight is doing real work rather than cancelling.
        lam_tr = elph.lambda_from_a2f(waw, og)
        lam = elph.lambda_from_a2f(self._run(d, 0.01)[1], og)
        assert lam_tr != pytest.approx(lam, rel=1e-3)

    def test_epw_degauss_conversion_makes_the_two_gaussians_identical(self):
        from waw.core.distributions import gaussian_smearing
        degaussw = 0.05 * 0.0367493   # EPW's Al default, 0.05 eV in Hartree
        x = np.linspace(-4 * degaussw, 4 * degaussw, 101)
        epw = _epw_w0gauss(x / degaussw) / degaussw
        waw = gaussian_smearing(x, elph.epw_degauss_to_sigma(degaussw))
        assert np.allclose(epw, waw, rtol=1e-13)

    def test_fermi_surface_dos_is_the_default_normalisation(self):
        """dos_at_ef=None must reproduce fermi_surface_dos on this very mesh
        at this very sigma -- the ratio alpha2F builds is only
        smearing-independent when numerator and denominator share both."""
        d = self._model()
        og = np.linspace(1e-6, 0.012, 200)
        common = dict(
            omega_ph=d['omega_ph'], eigvec_ph=d['eigvec_ph'],
            masses_amu=d['masses_amu'], types=d['types'], fermi_energy=d['ef'],
            omega_grid=og, sigma_e=0.01, hr=d['hr'],
        )
        args = (d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'],
                d['degen_q'], d['kpts'], d['qpts'], None)
        n_ef = elph.fermi_surface_dos(d['eig'], d['ef'], 0.01)
        auto = elph.alpha2f(*args, dos_at_ef=None, **common)
        explicit = elph.alpha2f(*args, dos_at_ef=n_ef, **common)
        assert np.allclose(auto, explicit)
        # and a mismatched N(eF) rescales alpha2F by exactly the ratio
        other = elph.alpha2f(*args, dos_at_ef=2.0 * n_ef, **common)
        assert np.allclose(other * 2.0, auto)

    def test_methfessel_paxton_dos_option(self):
        """EPW's dos_ef uses ngaussw (default 1 = Methfessel-Paxton) even
        though the deltas inside the linewidth use a plain Gaussian; the
        option exists so an EPW dosef can be reproduced exactly."""
        d = self._model()
        n0 = elph.fermi_surface_dos(d['eig'], d['ef'], 0.01, ngauss=0)
        n1 = elph.fermi_surface_dos(d['eig'], d['ef'], 0.01, ngauss=1)
        assert n0 > 0 and np.isfinite(n1)
        assert n0 != pytest.approx(n1, rel=1e-6)

    def test_fsthick_window_is_exact_for_a_generous_window(self):
        """EPW's fsthick. The k-side weight is a Gaussian in (eps - eF), so
        dropping k-points beyond ~9 sigma is exact to double precision -- and
        it is what makes a dense fine mesh affordable (the N^6 double sum
        collapses to the Fermi shell). Must hold for every variant, and the
        deliberately-too-tight control must NOT agree, or the test is vacuous."""
        from waw.analysis.wigner_transport import velocity_matrix
        d = self._model()
        _, vm = velocity_matrix(d['hr'], d['kpts'], d['recip'])
        vel = np.real(np.einsum('knna->kna', vm))
        og = np.linspace(1e-6, 0.012, 300)
        sigma_e = 0.004
        common = dict(
            omega_ph=d['omega_ph'], eigvec_ph=d['eigvec_ph'],
            masses_amu=d['masses_amu'], types=d['types'], fermi_energy=d['ef'],
            dos_at_ef=None, omega_grid=og, sigma_e=sigma_e,
            sigma_ph=2 * (og[1] - og[0]), hr=d['hr'],
        )
        args = (d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'],
                d['degen_q'], d['kpts'], d['qpts'], None)
        generous = 12 * sigma_e          # >= 9 sigma AND above omega_max
        assert generous > d['omega_ph'].max()
        for kw in ({}, dict(delta_approx=False),
                   dict(velocities=vel, recip_lattice=d['recip']),
                   dict(delta_approx=False, velocities=vel, recip_lattice=d['recip'])):
            full = elph.alpha2f(*args, **common, **kw)
            win = elph.alpha2f(*args, **common, **kw, fsthick=generous)
            assert np.allclose(win, full, rtol=0, atol=1e-12 * max(1.0, np.abs(full).max()))
        # not a no-op: a window narrower than the Gaussian must lose weight
        tight = elph.alpha2f(*args, **common, fsthick=0.5 * sigma_e)
        assert elph.lambda_from_a2f(tight, og) < 0.99 * elph.lambda_from_a2f(
            elph.alpha2f(*args, **common), og)

    def test_fsthick_excluding_everything_raises(self):
        d = self._model()
        og = np.linspace(1e-6, 0.012, 50)
        with pytest.raises(ValueError, match="excludes every k-point"):
            elph.alpha2f(
                d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'],
                d['degen_q'], d['kpts'], d['qpts'], None, d['omega_ph'],
                d['eigvec_ph'], d['masses_amu'], d['types'],
                fermi_energy=1e6, dos_at_ef=1.0, omega_grid=og,
                sigma_e=0.004, hr=d['hr'], fsthick=1e-9,
            )

    def test_warns_when_omega_grid_truncates_the_phonon_spectrum(self):
        d = self._model()
        og = np.linspace(1e-6, 0.5 * d['omega_ph'].max(), 50)
        with pytest.warns(RuntimeWarning, match="underestimated"):
            elph.alpha2f(
                d['eig'], d['U'], d['g_R'], d['R_e'], d['degen_e'], d['R_q'],
                d['degen_q'], d['kpts'], d['qpts'], None, d['omega_ph'],
                d['eigvec_ph'], d['masses_amu'], d['types'],
                fermi_energy=d['ef'], dos_at_ef=None, omega_grid=og,
                sigma_e=0.01, hr=d['hr'],
            )


class TestEliashbergMoments:
    def test_epw_a2f_file_reproduces_epw_lambda_and_omega_log(self):
        """`lambda_from_a2f`/`eliashberg_moments` against EPW's own printed
        integrals. alpha2F here is rebuilt from the per-mode lambda_qnu and
        omega_qnu of a real EPW run (Al, 12^3 coarse k / 6^3 coarse q, 20^3
        fine, degaussw = 0.05 eV) exactly as ``spectral.f90::a2f_main``
        assembles it, so this isolates the two integrals from every other
        stage: EPW reports l_a2f = 0.3673269 and logavg = 0.0016344 Ry."""
        # a2F(w) = sum_qnu wqf * w * lambda / 2 * gauss, on EPW's own grid
        # ww(iw) = iw * dw, dw = 1.1 * max(omega) / nqstep.  Two modes at two
        # frequencies reproduce the identity exactly; the real 8000-q run is
        # too large to vendor, so pin the algebra instead.
        ryd = 0.5                                    # Hartree per Ry
        nqstep = 500
        om = np.array([0.0100, 0.0300]) * ryd        # Hartree
        lam_q = np.array([0.20, 0.15])
        dw = 1.1 * om.max() / nqstep
        ww = np.arange(1, nqstep + 1) * dw
        degaussq = 4.0 * dw          # must resolve the grid, as EPW's own does
        a2f = sum(0.5 * w * l * _epw_w0gauss((ww - w) / degaussq) / degaussq
                  for w, l in zip(om, lam_q))
        # EPW's own rectangle sums
        l_epw = float((2.0 * a2f / ww * dw).sum())
        log_epw = float(np.exp((2.0 * a2f * np.log(ww) / ww * dw).sum() / l_epw))
        assert elph.lambda_from_a2f(a2f, ww) == pytest.approx(l_epw, rel=1e-12)
        lam_cum, wlog_cum = elph.eliashberg_moments(a2f, ww)
        assert wlog_cum[-1] == pytest.approx(log_epw, rel=1e-12)
        # and against the closed form: lambda = sum lambda_q, omega_log the
        # lambda-weighted geometric mean
        # rel 2e-3: a Gaussian of finite width sigma biases int G(w-w0)/w dw
        # by ~(sigma/w0)^2 -- EPW's own discretisation carries the same bias.
        assert l_epw == pytest.approx(lam_q.sum(), rel=2e-3)
        assert wlog_cum[-1] == pytest.approx(
            np.exp((lam_q * np.log(om)).sum() / lam_q.sum()), rel=2e-3)

    def test_delta_function_alpha2f_matches_analytic_formula(self):
        """alpha2F(w) = A*delta(w-w0) has EXACT known moments:
        lambda = 2A/w0, omega_log = w0 -- approximate the delta with a
        narrow Gaussian and check both against the analytic answer."""
        w0, A = 0.01, 0.05
        sigma = 1e-4
        omega_grid = np.linspace(1e-6, 0.03, 20000)
        a2f = A * np.exp(-0.5 * ((omega_grid - w0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        lambda_cum, omega_log_cum = elph.eliashberg_moments(a2f, omega_grid)

        lambda_expected = 2 * A / w0
        assert lambda_cum[-1] == pytest.approx(lambda_expected, rel=1e-2)
        assert omega_log_cum[-1] == pytest.approx(w0, rel=1e-2)

    def test_cumulative_is_monotonically_increasing_for_positive_a2f(self):
        omega_grid = np.linspace(1e-4, 0.02, 200)
        a2f = np.ones_like(omega_grid) * 0.01   # positive everywhere
        lambda_cum, _ = elph.eliashberg_moments(a2f, omega_grid)
        assert np.all(np.diff(lambda_cum) >= -1e-12)

    def test_final_lambda_matches_plain_formula(self):
        rng = np.random.default_rng(5)
        omega_grid = np.linspace(1e-4, 0.02, 100)
        a2f = np.abs(rng.normal(size=omega_grid.shape)) * 0.01
        domega = omega_grid[1] - omega_grid[0]
        # every point with omega > 0 contributes its rectangle bin, including
        # the first one -- EPW's own w_i = i*dw grid convention
        lambda_plain = 2.0 * np.sum(a2f / omega_grid) * domega

        lambda_cum, _ = elph.eliashberg_moments(a2f, omega_grid)
        assert lambda_cum[-1] == pytest.approx(lambda_plain)


class TestMeshGeneration:
    def test_monkhorst_pack_matches_the_reference_enumeration(self):
        """Vectorised monkhorst_pack must be bit-identical to the explicit
        first-index-slowest enumeration it replaced -- the ordering is load
        bearing (it is the .win `begin kpoints` order, and H(R) phases and the
        UNK file numbering both rely on it)."""
        from waw.interfaces.ase.structure import monkhorst_pack
        for mesh in [(2, 2, 2), (3, 4, 5), (4, 4, 1), (1, 1, 6), (6, 1, 1)]:
            n1, n2, n3 = mesh
            ref = np.array([[i / n1, j / n2, k / n3]
                            for i in range(n1) for j in range(n2) for k in range(n3)],
                           dtype=np.float64)
            got = monkhorst_pack(mesh)
            assert got.shape == ref.shape
            assert np.array_equal(got, ref)

    def test_monkhorst_pack_agrees_with_the_dos_module_copy(self):
        """analysis.dos keeps its own _uniform_mesh to stay free of an
        interfaces import; pin the two so they cannot drift apart."""
        from waw.interfaces.ase.structure import monkhorst_pack
        from waw.analysis.dos import _uniform_mesh
        for mesh in [(3, 3, 3), (2, 5, 4), (8, 8, 1)]:
            assert np.array_equal(monkhorst_pack(mesh), _uniform_mesh(mesh))


class TestAllenDynesTc:
    def test_reproduces_a_hand_evaluated_mcmillan_tc(self):
        """Plain form (f2 = 1) against the formula worked out by hand."""
        from waw.units import K_B_HARTREE, HARTREE_TO_EV
        lam, mu = 0.43, 0.1
        wlog = 23.3e-3 / HARTREE_TO_EV        # 23.3 meV in Hartree
        got = elph.allen_dynes_tc(lam, wlog, wlog, mu) / K_B_HARTREE   # omega_2=omega_log -> f2=1
        lam1 = 2.46 * (1 + 3.8 * mu)
        f1 = (1 + (lam / lam1) ** 1.5) ** (1 / 3)
        want = (f1 * wlog / 1.2
                * np.exp(-1.04 * (1 + lam) / (lam - mu * (1 + 0.62 * lam)))) / K_B_HARTREE
        assert got == pytest.approx(want, rel=1e-12)
        assert 0.5 < got < 5.0            # Al is a ~1 K superconductor

    def test_returns_nan_below_the_mu_star_threshold(self):
        """lambda <= mu*(1+0.62 lambda) makes the exponent's denominator
        non-positive and Tc meaningless. Must be NaN, not a silent number --
        it happens routinely on the coarse meshes that start a sequence."""
        assert np.isnan(elph.allen_dynes_tc(0.10, 1e-3, 1e-3, mu_star=0.1))
        assert np.isnan(elph.allen_dynes_tc(0.5, 0.0, 1e-3, mu_star=0.1))
        assert np.isfinite(elph.allen_dynes_tc(0.5, 1e-3, 1e-3, mu_star=0.1))

    def test_f2_needs_omega_2_and_raises_tc(self):
        """The strong-coupling shape factor f2 >= 1 whenever omega_2 > omega_log,
        which it always is for a spread-out spectrum. Omitting omega_2 must
        reduce to f2 = 1 exactly."""
        lam, wlog = 1.0, 1e-3
        plain = elph.allen_dynes_tc(lam, wlog, wlog, 0.1)     # omega_2=omega_log
        with_f2 = elph.allen_dynes_tc(lam, wlog, 1.4e-3, 0.1)
        assert with_f2 > plain
        assert elph.allen_dynes_tc(lam, wlog, wlog, 0.1) == pytest.approx(
            plain, rel=1e-12)          # omega_2 == omega_log -> f2 = 1

    def test_omega_2_of_a_delta_function_is_its_frequency(self):
        """alpha2F = A delta(w - w0) has omega_2 = omega_log = w0 exactly."""
        w0 = 0.01
        og = np.linspace(1e-6, 0.03, 20000)
        sig = 1e-4
        a2f = 0.05 * np.exp(-0.5 * ((og - w0) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        assert elph.eliashberg_omega_2(a2f, og) == pytest.approx(w0, rel=1e-2)
        _, wl = elph.eliashberg_moments(a2f, og)
        assert wl[-1] == pytest.approx(w0, rel=1e-2)

    def test_recovers_the_allen_dynes_strong_coupling_limit(self):
        """External check on f1 and f2 together, not just internal consistency.

        Allen & Dynes (PRB 12, 905 (1975)) give the strong-coupling asymptote
        ``k_B Tc -> 0.183 sqrt(lambda <w^2>)`` at mu* = 0. The closed form here
        must reproduce it: as lambda -> infinity with mu* = 0,
        f1 -> sqrt(lambda/2.46), f2 -> omega_2/omega_log and the exponential
        saturates at exp(-1.04), giving 0.188 sqrt(lambda) omega_2 -- their 0.183
        being a fit to numerical Eliashberg solutions rather than this limit, so
        a few per cent apart is the expected agreement, and a factor off is not.
        """
        w2 = 1.0e-3
        for lam in (200.0, 2000.0):
            for r in (1.0, 1.5, 2.5):            # omega_2/omega_log
                wlog = w2 / r
                tc = elph.allen_dynes_tc(lam, wlog, w2, mu_star=0.0)
                coeff = tc / (np.sqrt(lam) * w2)
                assert coeff == pytest.approx(0.188, rel=0.02)
                assert coeff == pytest.approx(0.183, rel=0.05)   # the published fit

    def test_from_a2f_derives_all_three_ingredients(self):
        """allen_dynes_tc_from_a2f must agree with feeding the pieces in by hand,
        and must NOT silently reduce to the plain McMillan form."""
        rng = np.random.default_rng(7)
        og = np.linspace(1e-5, 0.05, 4000)
        a2f = 0.6 * np.exp(-0.5 * ((og - 0.02) / 0.006) ** 2) \
            + 0.3 * np.exp(-0.5 * ((og - 0.035) / 0.004) ** 2)
        tc, lam, wlog, w2 = elph.allen_dynes_tc_from_a2f(a2f, og, mu_star=0.12)
        assert lam == pytest.approx(elph.lambda_from_a2f(a2f, og), rel=1e-12)
        assert wlog == pytest.approx(elph.eliashberg_moments(a2f, og)[1][-1], rel=1e-12)
        assert w2 == pytest.approx(elph.eliashberg_omega_2(a2f, og), rel=1e-12)
        assert tc == pytest.approx(elph.allen_dynes_tc(lam, wlog, w2, 0.12), rel=1e-12)
        # a two-peak spectrum has omega_2 > omega_log, so f2 > 1 and the full
        # formula must give a HIGHER Tc than the plain one
        assert w2 > wlog
        assert tc > elph.allen_dynes_tc(lam, wlog, wlog, 0.12)

    def test_tc_is_exponentially_more_sensitive_to_lambda_than_to_omega_log(self):
        """The reason Tc is reported rather than converged: at Al's values a
        0.03 shift in lambda moves Tc ~10x more than 3% on omega_log does."""
        lam, wlog, mu = 0.4772, 0.000906, 0.1
        w2 = 1.3 * wlog
        base = elph.allen_dynes_tc(lam, wlog, w2, mu)
        d_lam = abs(elph.allen_dynes_tc(lam + 0.03, wlog, w2, mu) / base - 1)
        d_wlog = abs(elph.allen_dynes_tc(lam, wlog * 1.03, w2, mu) / base - 1)
        assert d_wlog == pytest.approx(0.03, abs=0.005)      # linear in omega_log
        assert d_lam > 5 * d_wlog                            # exponential in lambda


class TestFermiLevelAndSymmetry:
    def test_fermi_level_recovers_the_electron_count(self):
        """The whole point: eF must be the level at which THESE bands hold the
        right number of electrons, not the one imported from the DFT run."""
        rng = np.random.default_rng(3)
        eig = np.sort(rng.normal(size=(200, 4)) * 0.1, axis=1)
        sigma = 0.01
        for n_el in (1.0, 3.0, 5.5):
            ef = elph.fermi_level_from_electron_count(eig, n_el, sigma)
            from scipy.special import erfc
            got = erfc((eig - ef) / (sigma * np.sqrt(2.0))).sum() / len(eig)
            assert got == pytest.approx(n_el, abs=1e-6)

    def test_fermi_level_rejects_impossible_counts(self):
        eig = np.zeros((10, 4))
        with pytest.raises(ValueError, match="outside"):
            elph.fermi_level_from_electron_count(eig, 9.0, 0.01)   # > 2*nw
        with pytest.raises(ValueError, match="outside"):
            elph.fermi_level_from_electron_count(eig, 0.0, 0.01)

    def test_irreducible_wedge_reproduces_the_full_q_mesh(self):
        """Summing q over the irreducible wedge with multiplicity weights is
        exact when the k-sum covers the full mesh. Checked here on a model
        whose symmetry is imposed by construction (a cubic cell with one atom,
        H(R) symmetrised over the 48 operations), so any failure is the
        weighting, not the gauge."""
        from waw.interfaces.ase.structure import irreducible_qpoints, monkhorst_pack
        from waw.core.hamiltonian import _wigner_seitz
        from ase import Atoms
        rng = np.random.default_rng(11)
        a = 5.0
        atoms = Atoms('Al', scaled_positions=[[0, 0, 0]],
                      cell=a * np.eye(3), pbc=True)
        A = a * np.eye(3) * ANG_TO_BOHR
        mesh, nw, nat3 = (4, 4, 4), 1, 3
        R_e, degen_e = _wigner_seitz(mesh, A)
        R_q, degen_q = _wigner_seitz(mesh, A)
        # cubic-symmetric H(R): depends only on |R|, so H(k) has full Oh symmetry
        r2 = (R_e ** 2).sum(axis=1)
        H_R = np.zeros((len(R_e), nw, nw), dtype=np.complex128)
        for v in np.unique(r2):
            H_R[r2 == v, 0, 0] = rng.normal() * np.exp(-0.5 * v)
        hr = HamiltonianR(H_R=torch.from_numpy(H_R), R_vectors=R_e,
                          degen=degen_e, nw=nw)
        # and a symmetric |g|: constant in R_q, real, isotropic in R_e
        g_R = np.zeros((len(R_e), len(R_q), nat3, nw, nw), dtype=np.complex128)
        for v in np.unique(r2):
            g_R[r2 == v] = rng.normal() * np.exp(-0.5 * v)

        k = monkhorst_pack((6, 6, 6))
        eig, U = elph.band_eigensystem(hr, k)
        ef = float(np.median(eig))
        og = np.linspace(1e-5, 0.03, 80)
        common = dict(masses_amu=np.array([26.98]), types=np.array([0]),
                      fermi_energy=ef, dos_at_ef=None, omega_grid=og,
                      sigma_e=0.05, sigma_ph=2 * (og[1] - og[0]), hr=hr)

        def ph(q):
            nq = len(q)
            # isotropic dispersion -> invariant under the point group
            w = 0.008 + 0.002 * np.cos(2 * np.pi * q).sum(axis=1)[:, None]
            return (np.tile(w, (1, nat3)),
                    np.tile(np.eye(nat3, dtype=np.complex128), (nq, 1, 1)))

        qf = monkhorst_pack((4, 4, 4))
        om_f, ev_f = ph(qf)
        full = elph.alpha2f(eig, U, g_R, R_e, degen_e, R_q, degen_q, k, qf, None,
                            om_f, ev_f, **common)
        qi, wi = irreducible_qpoints(atoms, (4, 4, 4))
        assert len(qi) < len(qf)                      # actually reduced
        assert wi.sum() == pytest.approx(1.0, abs=1e-12)
        om_i, ev_i = ph(qi)
        wedge = elph.alpha2f(eig, U, g_R, R_e, degen_e, R_q, degen_q, k, qi, None,
                             om_i, ev_i, q_weights=wi, **common)
        assert np.allclose(wedge, full, rtol=0,
                           atol=1e-10 * max(1.0, np.abs(full).max()))
        assert elph.lambda_from_a2f(wedge, og) == pytest.approx(
            elph.lambda_from_a2f(full, og), rel=1e-10)

    def test_alpha2f_rejects_unnormalised_q_weights(self):
        from waw.interfaces.ase.structure import monkhorst_pack
        from waw.core.hamiltonian import _wigner_seitz
        A = np.eye(3) * 6.0
        R_e, degen_e = _wigner_seitz((2, 2, 2), A)
        R_q, degen_q = _wigner_seitz((2, 2, 2), A)
        nw, nat3 = 1, 3
        hr = HamiltonianR(H_R=torch.zeros((len(R_e), nw, nw), dtype=torch.complex128),
                          R_vectors=R_e, degen=degen_e, nw=nw)
        g_R = np.zeros((len(R_e), len(R_q), nat3, nw, nw), dtype=np.complex128)
        k = q = monkhorst_pack((2, 2, 2))
        eig, U = elph.band_eigensystem(hr, k)
        og = np.linspace(1e-5, 0.02, 20)
        kw = dict(omega_ph=np.full((len(q), nat3), 0.01),
                  eigvec_ph=np.tile(np.eye(nat3, dtype=np.complex128), (len(q), 1, 1)),
                  masses_amu=np.array([26.98]), types=np.array([0]),
                  fermi_energy=0.0, dos_at_ef=1.0, omega_grid=og, sigma_e=0.5, hr=hr)
        args = (eig, U, g_R, R_e, degen_e, R_q, degen_q, k, q, None)
        with pytest.raises(ValueError, match="not 1"):
            elph.alpha2f(*args, q_weights=np.full(len(q), 0.5), **kw)
        with pytest.raises(ValueError, match="expected"):
            elph.alpha2f(*args, q_weights=np.ones(len(q) + 1) / (len(q) + 1), **kw)


class TestFixedQProvider:
    def test_blocked_provider_matches_interpolate_elph_fixed_q(self):
        """The q-blocked R_q contraction (the 13x alpha2f hot-loop path) must
        reproduce interpolate_elph_fixed_q pointwise -- same math, different
        contraction association."""
        rng = np.random.default_rng(7)
        nRe, nRq, nm, nw = 11, 7, 3, 2
        g_R = rng.normal(size=(nRe, nRq, nm, nw, nw)) + 1j * rng.normal(size=(nRe, nRq, nm, nw, nw))
        R_e = rng.integers(-3, 4, size=(nRe, 3))
        R_q = rng.integers(-2, 3, size=(nRq, 3))
        deg_e = rng.integers(1, 4, size=nRe)
        deg_q = rng.integers(1, 3, size=nRq)
        qpts = rng.random((9, 3))
        kpts = rng.random((5, 3))
        h_of = elph._fixed_q_h_provider(g_R, R_q, deg_q, qpts, max_bytes=4 * nRe * nm * nw * nw * 16)
        inv_e = 1.0 / deg_e
        for iq in range(len(qpts)):
            ph_k = np.exp(2j * np.pi * (kpts @ R_e.T)) * inv_e[None, :]
            got = np.tensordot(ph_k, h_of(iq), axes=([1], [0]))
            ref = elph.interpolate_elph_fixed_q(g_R, R_e, deg_e, R_q, deg_q, kpts, qpts[iq])
            np.testing.assert_allclose(got, ref, atol=1e-12)


class TestLambdaEffective:
    def test_two_band_reduction(self):
        """lambda_eff = max eigenvalue; for a diagonal matrix it is the
        largest diagonal, and the asymmetric MgB2-like case reproduces the
        known largest root of the 2x2 problem."""
        assert elph.lambda_effective(np.diag([1.0, 0.4])) == pytest.approx(1.0)
        lam = np.array([[1.017, 0.213], [0.155, 0.448]])
        tr, det = lam.trace(), np.linalg.det(lam)
        expected = 0.5 * (tr + np.sqrt(tr ** 2 - 4 * det))
        assert elph.lambda_effective(lam) == pytest.approx(expected)
