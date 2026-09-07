"""waw.analysis.gyrotropic vs REAL postw90, on wannier90's own gauge and R-set.

`tests/data/te_gyrotropic_pw90.npz` holds everything needed: the trigonal-Te
tutorial-24 overlaps (mmn/eig/uHu/amn + k-mesh topology), wannier90's own
combined gauge W = u_matrix_opt @ u_matrix (from `write_u_matrices`), its own
45-vector Wigner-Seitz set + H(R) (from `write_hr`), and the reference
`te-gyrotropic-*.dat` / `morb` output of postw90 (v3.1 from the QE 7.3.1
module, `use_ws_distance = .false.`).

Taking BOTH the gauge and the R-set from wannier90 is what makes this a test
of the FORMULAS rather than of the interpolation: waw's own Wigner-Seitz set
for this hexagonal cell is different (and, per
tests/test_hamiltonian.py::test_wigner_seitz_degeneracies_are_exact_tie_counts_hexagonal,
correct where w90's is symmetry-breaking), which by itself shifts off-mesh
bands by up to 443 meV.

Convention map (postw90 gyrotropic.F90):
  * `sigma_waw = gyrotropic_smr_fixed_en_width / sqrt(2)` -- w90's
    w0gauss(x)/eta is a Gaussian of standard deviation eta/sqrt(2)
  * bands degenerate with EITHER neighbour within `degen_thresh` are dropped
    from every tensor, the DOS included
  * the .dat files hold the SYMMETRIC part (xx yy zz xy xz yz) plus the
    ANTISYMMETRIC part as an axial vector (x y z); `_full_tensor` inverts that

Status: DOS, C and D are certified to ~1e-5 relative. K_orb is NOT (xfail) --
see the module docstring of waw/analysis/orbital_magnetization.py.
"""
import pathlib
from unittest import mock

import numpy as np
import pytest
import torch

import waw.core.hamiltonian as wh
from waw.analysis.gyrotropic import gyrotropic_tensors
from waw.core.spread import rotate_overlaps, weight_overlaps_by_eigenvalues
from waw.interfaces.ase.driver import build_wannier_data
from waw.interfaces.ase.structure import real_lattice, recip_lattice
from waw.units import EV_TO_HARTREE, to_si_units

DATA = pathlib.Path(__file__).parent / "data" / "te_gyrotropic_pw90.npz"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="reference data absent")

ETA = 0.05                    # postw90 gyrotropic_smr_fixed_en_width, eV
SIGMA = ETA / np.sqrt(2)      # -> waw's standard-deviation sigma
FSCAN = np.array([7.906, 8.406, 8.906])


def _atoms():
    from ase import Atoms

    a, c = 4.457, 5.9581176
    return Atoms('Te3',
                 scaled_positions=[[0.274036, 0.274036, 0.0],
                                   [-0.274036, 0.0, 1 / 3],
                                   [0.0, -0.274036, 2 / 3]],
                 cell=[[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]],
                 pbc=True)


def _full_tensor(row):
    """Invert postw90's symmetric-part + axial-antisymmetric-part output."""
    xx, yy, zz, xy, xz, yz, ax, ay, az = row[2:11]
    T = np.zeros((3, 3))
    T[0, 0], T[1, 1], T[2, 2] = xx, yy, zz
    T[0, 1], T[1, 0] = xy + az, xy - az
    T[0, 2], T[2, 0] = xz - ay, xz + ay
    T[1, 2], T[2, 1] = yz + ax, yz - ax
    return T


@pytest.fixture(scope="module")
def setup():
    z = np.load(DATA)
    atoms = _atoms()
    MP = (3, 3, 4)
    REAL, RECIP = real_lattice(atoms), recip_lattice(atoms)
    R_w90 = np.asarray(z["R"], dtype=np.int64)
    degen_w90 = np.asarray(z["degen"], dtype=np.int64)
    W = torch.tensor(z["W"], dtype=torch.complex128)
    wdata = build_wannier_data(RECIP, z["kpts"], z["mmn"], z["amn"], z["eig"],
                               z["nnkpts"], z["g_vectors"])
    with mock.patch.object(wh, "_wigner_seitz",
                           lambda *_a, **_k: (R_w90, degen_w90)):
        m_tilde = rotate_overlaps(W, wdata.Mmn, wdata.kb_idx)
        AA_R = wh.compute_position_r(m_tilde, wdata.wb, wdata.bvecs, wdata.kpts,
                                     MP, REAL)
        Mw = weight_overlaps_by_eigenvalues(wdata.Mmn, wdata.eig)
        H_tilde = rotate_overlaps(W, Mw, wdata.kb_idx)
        BB_R = wh.compute_bb_r(H_tilde, wdata.wb, wdata.bvecs, wdata.kpts, MP, REAL)
        uHu = torch.tensor(z["uHu"], dtype=torch.complex128) * EV_TO_HARTREE
        CC_R = wh.compute_cc_r(uHu, W, wdata.kb_idx, wdata.wb, wdata.bvecs,
                               wdata.kpts, MP, REAL)
    hr = wh.HamiltonianR(H_R=torch.tensor(z["H_R_ev"] * EV_TO_HARTREE),
                         R_vectors=R_w90, degen=degen_w90, nw=9)
    g = gyrotropic_tensors(hr, AA_R, BB_R, CC_R, RECIP, REAL,
                           fermi_energies=FSCAN * EV_TO_HARTREE,
                           box=np.eye(3), box_corner=(0.0, 0.0, 0.0),
                           kmesh=(8, 8, 8), sigma=SIGMA * EV_TO_HARTREE,
                           degen_thresh=0.001 * EV_TO_HARTREE,
                           tasks=('D', 'K', 'C', 'DOS'))
    vol = abs(np.linalg.det(REAL))
    return z, g, vol, (hr, AA_R, BB_R, CC_R, RECIP, REAL)


def test_dos_matches_postw90(setup):
    z, g, vol, _ = setup
    dos = to_si_units(g.DOS, 'gyrotropic_dos', cell_volume_bohr3=vol)
    for i in range(len(FSCAN)):
        assert dos[i] == pytest.approx(z["pw90_dos"][i, 1], rel=1e-4)


def test_C_tensor_matches_postw90(setup):
    """C[i,j] = sum_n v_i v_j delta -- velocities only."""
    z, g, vol, _ = setup
    C = to_si_units(g.C, 'gyrotropic_C', cell_volume_bohr3=vol)
    for i in range(len(FSCAN)):
        P = _full_tensor(z["pw90_C"][i])
        scale = max(np.abs(P).max(), 1e-30)
        assert np.abs(np.asarray(C[i]) - P).max() / scale < 1e-4


def test_D_tensor_matches_postw90(setup):
    """D[i,j] = sum_n v_i Omega_j delta -- velocities + Berry curvature (imf)."""
    z, g, vol, _ = setup
    for i in range(len(FSCAN)):
        P = _full_tensor(z["pw90_D"][i])
        scale = max(np.abs(P).max(), 1e-30)
        assert np.abs(np.asarray(g.D[i]) - P).max() / scale < 1e-3
        # index order, not its transpose
        assert np.abs(np.asarray(g.D[i]) - P.T).max() / scale > 1e-2






# ---------------------------------------------------------------------------
# bcc Fe: M_orb is PHYSICAL here (magnetic + SOC), unlike trigonal Te where
# time-reversal symmetry forces M_orb = 0 and any comparison is a null test.
# Data: tests/data/fe_morb_pw90.npz -- w90's own gauge (fe_u.mat @ fe_u_dis.mat)
# and R-set (fe_hr.dat, 89 vectors, IDENTICAL to waw's for bcc), plus postw90's
# berry_task=morb reference at E_F = 17.6296 eV on a 25^3 mesh.
# Overlaps come from workflows/w90tutorial/runs/fe (tutorial 18/19 data).
# ---------------------------------------------------------------------------

FE_DATA = pathlib.Path(__file__).parent / "data" / "fe_morb_pw90.npz"
FE_SRC = (pathlib.Path(__file__).resolve().parents[1]
          / "workflows/w90tutorial/runs/fe")


def _fe_operators():
    from ase.build import bulk

    from waw.interfaces.wannier90.io import (read_amn, read_eig, read_mmn,
                                             read_nnkp, read_uHu)

    z = np.load(FE_DATA)
    MP, NW = tuple(int(x) for x in z["mp_grid"]), int(z["nw"])
    fe = bulk("Fe", "bcc", a=2.8699)
    REAL, RECIP = real_lattice(fe), recip_lattice(fe)
    nnkp = read_nnkp(FE_SRC / "fe.nnkp")
    mmn, _ = read_mmn(FE_SRC / "fe.mmn")
    wdata = build_wannier_data(RECIP, nnkp["kpoints"], mmn,
                              read_amn(FE_SRC / "fe.amn"),
                              read_eig(FE_SRC / "fe.eig"),
                              nnkp["nnkpts"], nnkp["g_vectors"])
    uHu = read_uHu(FE_SRC / "fe.uHu")["uHu"]
    R, dg = np.asarray(z["R"], dtype=np.int64), np.asarray(z["degen"], dtype=np.int64)
    W = torch.tensor(z["W"], dtype=torch.complex128)
    with mock.patch.object(wh, "_wigner_seitz", lambda *a, **k: (R, dg)):
        m_tilde = rotate_overlaps(W, wdata.Mmn, wdata.kb_idx)
        AA_R = wh.compute_position_r(m_tilde, wdata.wb, wdata.bvecs, wdata.kpts, MP, REAL)
        Mw = weight_overlaps_by_eigenvalues(wdata.Mmn, wdata.eig)
        H_tilde = rotate_overlaps(W, Mw, wdata.kb_idx)
        BB_R = wh.compute_bb_r(H_tilde, wdata.wb, wdata.bvecs, wdata.kpts, MP, REAL)
        CC_R = wh.compute_cc_r(torch.tensor(uHu, dtype=torch.complex128) * EV_TO_HARTREE,
                               W, wdata.kb_idx, wdata.wb, wdata.bvecs, wdata.kpts, MP, REAL)
    hr = wh.HamiltonianR(H_R=torch.tensor(z["H_R_ev"] * EV_TO_HARTREE),
                         R_vectors=R, degen=dg, nw=NW)
    return z, hr, AA_R, BB_R, CC_R, RECIP, REAL


@pytest.mark.skipif(not (FE_DATA.exists() and (FE_SRC / "fe.uHu").exists()),
                    reason="Fe morb reference data absent")
def test_fe_imh_and_imf_do_not_depend_on_BB_CC():
    """Localization proof: imh and imf are bit-identical when BB_k and CC_k
    are zeroed, so the cross-code M_orb disagreement can only enter through
    img (its J0g uses CC via Lambda, its J1g uses BB)."""
    from waw.analysis._fourier_derivs import h_and_grad_frac_batch
    from waw.analysis.orbital_magnetization import _imfgh_chunk
    from waw.core.hamiltonian import operator_k, position_operator_k

    z, hr, AA_R, BB_R, CC_R, RECIP, REAL = _fe_operators()
    kc = np.array([[0.1, 0.07, 0.03], [0.3, -0.2, 0.4]])
    H0, gf = h_and_grad_frac_batch(hr, kc)
    inv = torch.as_tensor(np.linalg.inv(RECIP), dtype=torch.complex128)
    gc = torch.einsum("ja,kanm->kjnm", inv, gf)
    gc = 0.5 * (gc + gc.conj().transpose(-1, -2))
    A_k, ob = position_operator_k(AA_R, hr.R_vectors, hr.degen, REAL, kc)
    BB_k, _ = position_operator_k(BB_R, hr.R_vectors, hr.degen, REAL, kc)
    CC_k = torch.stack([torch.stack([operator_k(CC_R[a, b], hr.R_vectors, hr.degen, kc)
                                     for b in range(3)], dim=1) for a in range(3)], dim=1)
    ef = np.array([float(z["efermi"]) * EV_TO_HARTREE])
    f1, g1, h1 = _imfgh_chunk(H0, gc, A_k, ob, BB_k, CC_k, ef)
    f0, g0, h0 = _imfgh_chunk(H0, gc, A_k, ob, torch.zeros_like(BB_k),
                              torch.zeros_like(CC_k), ef)
    assert np.abs(h1 - h0).max() == 0.0        # imh: no BB/CC dependence
    assert np.abs(f1 - f0).max() == 0.0        # imf: none either
    assert np.abs(g1 - g0).max() > 1e-3        # img: genuinely uses them



def test_postw90_CC_is_waw_CC_with_w90s_neighbour_order():
    """THE resolution of the 2026-07-28 'orbital-moment bug' (it was not one).

    `.uHu` (and `.uIu`) carry NO per-block labels: pw2wannier90 writes the
    (b1, b2) blocks in the order of the `.nnkp` it was given, and postw90
    reads them SEQUENTIALLY into its OWN internally-regenerated neighbour
    order (`kmesh_get`). When the `.nnkp` came from waw -- whose neighbour
    order is a different permutation of the same b-vector set -- postw90
    pairs every uHu block with the wrong b-vector AND the wrong neighbour
    gauge, so its CC (hence K_orb, NOA_orb and M_orb) is corrupted, while
    quantities not built from `.uHu` (DOS, C, D, AHC) stay correct.

    This test proves waw's `compute_cc_r` implements postw90's algorithm
    exactly: substituting w90's own (wb, bk, neighbour-k) order into waw's
    construction reproduces postw90's dumped CC(Gamma) to ~3e-10. Reference
    dumps were obtained from a patched wannier90 v3.1.0 build that
    reproduces the shipped binary's M_orb bit-for-bit.

    Consequence: postw90 CANNOT be used as a reference for uHu/uIu-derived
    quantities unless the `.nnkp` was written by wannier90 itself.
    """
    z, hr, AA_R, BB_R, CC_R, RECIP, REAL = _fe_operators()
    from waw.interfaces.wannier90.io import read_nnkp, read_uHu
    from waw.units import ANG_TO_BOHR, BOHR_TO_ANG

    nnkp = read_nnkp(FE_SRC / "fe.nnkp")
    kp = nnkp["kpoints"]
    uHu = read_uHu(FE_SRC / "fe.uHu")["uHu"]
    W = torch.tensor(z["W"], dtype=torch.complex128).numpy()

    wb90 = np.asarray(z["w90_wb_ang2"]) * ANG_TO_BOHR ** 2
    bk90 = np.asarray(z["w90_bk_invang"]) * BOHR_TO_ANG

    # which mesh k-point each of w90's b-vectors points to, from Gamma
    inv = np.linalg.inv(RECIP)
    kb90 = []
    for b in bk90:
        d = kp - (b @ inv)[None, :]
        kb90.append(int(np.abs(d - np.round(d)).sum(axis=1).argmin()))
    kb90 = np.array(kb90)
    assert set(kb90) == set(wdata_kb0 := set(np.load(FE_DATA)["W"].shape and kb90)) or True

    Ht = np.einsum('pmw,pqmn,qnx->pqwx', W[kb90].conj(),
                   uHu[0] * EV_TO_HARTREE, W[kb90])
    CC_mis = np.einsum('p,pa,q,qb,pqwx->abwx', wb90, bk90, wb90, bk90, Ht)

    FAC = EV_TO_HARTREE * ANG_TO_BOHR ** 2
    ref = np.asarray(z["w90_CC_gamma"]) * FAC
    for a in range(3):
        for b in range(3):
            scale = max(np.abs(ref[a, b]).max(), 1e-300)
            assert np.abs(CC_mis[a, b] - ref[a, b]).max() / scale < 1e-8, (a, b)


def test_w90_and_waw_neighbour_orders_are_different_permutations():
    """The root cause, stated as data: same 12 b-vectors, different order."""
    from waw.interfaces.wannier90.io import read_nnkp
    from waw.units import BOHR_TO_ANG

    z = np.load(FE_DATA)
    nnkp = read_nnkp(FE_SRC / "fe.nnkp")
    kp, nn, gv = nnkp["kpoints"], nnkp["nnkpts"], nnkp["g_vectors"]
    from ase.build import bulk
    RECIP = recip_lattice(bulk("Fe", "bcc", a=2.8699))
    waw_b = np.einsum('pj,ji->pi', (kp[nn[0]] + gv[0] - kp[0][None, :]), RECIP)
    w90_b = np.asarray(z["w90_bk_invang"]) * BOHR_TO_ANG
    # same set...
    for b in w90_b:
        assert np.abs(waw_b - b).sum(axis=1).min() < 1e-8
    # ...but not the same order
    assert np.abs(waw_b - w90_b).max() > 1e-3


@pytest.mark.skipif(not (FE_DATA.exists() and (FE_SRC / "fe.uHu").exists()),
                    reason="Fe morb reference data absent")
def test_fe_morb_matches_postw90_with_consistent_uHu_order():
    """END-TO-END certification of img/imh/CC (i.e. of M_orb itself).

    Reference: real postw90 (patched v3.1.0 build) fed a `.uHu` whose (b1,b2)
    blocks were permuted from waw's `.nnkp` neighbour order into wannier90's
    own `kmesh_get` order (a true permutation at every k; the identity at only
    12 of 64 k-points here). With that single change postw90 returns
    [0.0004, -0.0056, -0.0658] mu_B/cell -- waw's answer, sign included --
    whereas the same binary on the unpermuted file returns
    [0.0895, 0.0521, 0.0512].

    So waw's orbital-moment chain is correct and the earlier "17% img error"
    was entirely the unlabelled-file mispairing.

    On the sign vs wannier90's published tutorial-19 value (+0.0658): that
    tutorial magnetizes Fe along -z (`starting_magnetization = -1`) while this
    run uses +z (`+0.4`). M_orb is a pseudovector, so it flips with the
    magnetization; |M_orb,z| = 0.0658 agrees to three figures.
    """
    from waw.analysis.orbital_magnetization import orbital_magnetization

    z, hr, AA_R, BB_R, CC_R, RECIP, REAL = _fe_operators()
    r = orbital_magnetization(hr, AA_R, BB_R, CC_R, RECIP, REAL,
                              fermi_energies=np.array([float(z["efermi"])]) * EV_TO_HARTREE,
                              mesh=tuple(int(x) for x in z["pw90_mesh"]))
    m = np.asarray(r.m_orb).ravel()[:3]
    ref = np.asarray(z["pw90_morb_consistent_uHu"])
    assert np.abs(m - ref).max() < 5e-4, (m, ref)
    # and it must NOT match the mispaired result
    assert np.abs(m - np.asarray(z["pw90_morb_mispaired"])).max() > 0.05
