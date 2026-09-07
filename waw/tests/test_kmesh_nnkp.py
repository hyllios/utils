"""
Tests for waw's in-process replacement of `wannier90.x -pp`
(core.kmesh.generate_nnkp) and the .nnkp writer (io.write_nnkp).

  1. generate_nnkp reproduces a real wannier90 -pp .nnkp: same nntot and the
     same per-k neighbour set (k-index + G-vector), order aside.
  2. The generated b-vectors satisfy the finite-difference completeness
     relation Σ_b w_b b_α b_β = δ_αβ (simple-cubic and the Si FCC mesh).
  3. write_nnkp round-trips through read_nnkp.
  4. End to end: driving the core from the generated topology (with the shipped
     overlaps reordered to the generated b-vector order) reproduces the
     Wannier90-path Ω exactly -- i.e. the generated .nnkp is interoperable with
     real overlaps.
"""

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.kmesh import generate_nnkp, _compute_bvecs_and_weights, _nnkp_from_mmn
from waw.interfaces.ase.structure import monkhorst_pack
from waw.interfaces.wannier90.io import (
    read_win, read_nnkp, read_mmn, read_amn, read_eig, write_nnkp,
)
from waw.interfaces.projections import spd_projections
from waw.interfaces.wannier90.loader import (
    load, parse_real_lattice, parse_recip_lattice,
)
from waw.units import BOHR_TO_ANG

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.nnkp").exists()


def _completeness_residual(bvecs_k0, wb):
    """max |Σ_b w_b b_a b_c - δ_ac| for the k=0 b-vectors."""
    M = np.einsum("b,ba,bc->ac", wb, bvecs_k0, bvecs_k0)
    return np.abs(M - np.eye(3)).max()


# ---------------------------------------------------------------------------
# 1. generate_nnkp vs a real wannier90 -pp .nnkp
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SI, reason="Silicon .nnkp not found")
def test_generate_nnkp_matches_wannier90_pp():
    win   = read_win(SI_DIR / "silicon.win")
    recip = parse_recip_lattice(win)
    mp    = tuple(int(x) for x in str(win["mp_grid"]).split())
    ref   = read_nnkp(SI_DIR / "silicon.nnkp")

    gen = generate_nnkp(ref["kpoints"], recip, mp)

    assert gen["nntot"] == ref["nntot"]
    for ik in range(len(ref["kpoints"])):
        gen_set = {(int(gen["nnkpts"][ik, b]), *map(int, gen["g_vectors"][ik, b]))
                   for b in range(gen["nntot"])}
        ref_set = {(int(ref["nnkpts"][ik, b]), *map(int, ref["g_vectors"][ik, b]))
                   for b in range(ref["nntot"])}
        assert gen_set == ref_set, ik


# ---------------------------------------------------------------------------
# 2. Completeness of the generated shells
# ---------------------------------------------------------------------------

def test_generated_nnkp_completeness_simple_cubic():
    a = 5.0
    recip = (2 * np.pi / a) * np.eye(3)     # cubic recip (Bohr^-1)
    mp = (4, 4, 4)
    kpts = monkhorst_pack(mp)
    gen = generate_nnkp(kpts, recip, mp)
    assert gen["nntot"] == 6                # 6 nearest neighbours, one shell
    bvecs, wb = _compute_bvecs_and_weights(kpts, gen["nnkpts"], gen["g_vectors"], recip)
    assert _completeness_residual(bvecs[0], wb) < 1e-10


@pytest.mark.skipif(not HAS_SI, reason="Silicon files not found")
def test_generated_nnkp_completeness_silicon():
    win   = read_win(SI_DIR / "silicon.win")
    recip = parse_recip_lattice(win)
    mp    = tuple(int(x) for x in str(win["mp_grid"]).split())
    kpts  = read_nnkp(SI_DIR / "silicon.nnkp")["kpoints"]
    gen   = generate_nnkp(kpts, recip, mp)
    bvecs, wb = _compute_bvecs_and_weights(kpts, gen["nnkpts"], gen["g_vectors"], recip)
    assert _completeness_residual(bvecs[0], wb) < 1e-10


# ---------------------------------------------------------------------------
# 3. write_nnkp / read_nnkp round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SI, reason="Silicon files not found")
def test_write_nnkp_roundtrip(tmp_path):
    win   = read_win(SI_DIR / "silicon.win")
    recip = parse_recip_lattice(win)
    real  = parse_real_lattice(win)
    mp    = tuple(int(x) for x in str(win["mp_grid"]).split())
    kpts  = read_nnkp(SI_DIR / "silicon.nnkp")["kpoints"]
    gen   = generate_nnkp(kpts, recip, mp)

    path = tmp_path / "si.nnkp"
    write_nnkp(path, real * BOHR_TO_ANG, recip / BOHR_TO_ANG, kpts,
               gen["nnkpts"], gen["g_vectors"])
    back = read_nnkp(path)

    assert back["nntot"] == gen["nntot"]
    np.testing.assert_array_equal(back["nnkpts"], gen["nnkpts"])
    np.testing.assert_array_equal(back["g_vectors"], gen["g_vectors"])
    np.testing.assert_allclose(back["kpoints"], kpts, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. End to end: generated topology + reordered overlaps == W90-path result
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SI, reason="Silicon files not found")
def test_generated_nnkp_reproduces_reference_omega():
    from waw.interfaces.ase.driver import build_wannier_data
    from waw.core import wannierize as core_wannierize

    win   = read_win(SI_DIR / "silicon.win")
    recip = parse_recip_lattice(win)
    real  = parse_real_lattice(win)
    mp    = tuple(int(x) for x in str(win["mp_grid"]).split())

    mmn, kpb_map = read_mmn(SI_DIR / "silicon.mmn")   # wannier90-order blocks
    amn = read_amn(SI_DIR / "silicon.amn")
    eig = read_eig(SI_DIR / "silicon.eig")            # eV
    nk, nnb = mmn.shape[0], mmn.shape[1]

    # Neighbour topology in the .mmn's own block order, and waw's own order.
    shipped = _nnkp_from_mmn(kpb_map, win, nk, nnb)
    kpts    = shipped["kpoints"]
    gen     = generate_nnkp(kpts, recip, mp)

    # Permute each k's Mmn blocks from shipped order into the generated order,
    # so block ib pairs with the generated b-vector ib (what pw2wannier90 would
    # have produced had it consumed our generated .nnkp).
    mmn_gen = np.empty_like(mmn)
    for ik in range(nk):
        shipped_pairs = [(int(shipped["nnkpts"][ik, b]),
                          tuple(int(x) for x in shipped["g_vectors"][ik, b]))
                         for b in range(nnb)]
        for ib in range(gen["nntot"]):
            key = (int(gen["nnkpts"][ik, ib]),
                   tuple(int(x) for x in gen["g_vectors"][ik, ib]))
            mmn_gen[ik, ib] = mmn[ik, shipped_pairs.index(key)]

    wd_gen = build_wannier_data(recip, kpts, mmn_gen, amn, eig,
                                gen["nnkpts"], gen["g_vectors"])
    r_gen = core_wannierize(wd_gen, 4, mp, real, n_iter=500, conv_tol=1e-10,
                            seed=0, verbose=False)

    wd_ref = load(SI_DIR / "silicon", recip, params=win)
    r_ref = core_wannierize(wd_ref, 4, mp, real, n_iter=500, conv_tol=1e-10,
                            seed=0, verbose=False)

    np.testing.assert_allclose(r_gen.omega_final, r_ref.omega_final, rtol=1e-9)


def test_spd_projections_shells():
    """spd_projections expands shells into the Wannier90 (l, mr) ordering."""
    specs = spd_projections((0.0, 0.0, 0.0), "s;p;d")
    assert len(specs) == 9
    lm = [(l, mr) for (_, l, mr, _, _, _, _) in specs]
    assert lm == [(0, 1), (1, 1), (1, 2), (1, 3),
                  (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
    # s;d skips the p shell
    assert len(spd_projections((0, 0, 0), "s;d")) == 6


def test_write_nnkp_projection_blocks(tmp_path):
    """write_nnkp emits an explicit analytic projections block, and the
    auto_projections block pw2wannier90 needs for scdm_proj."""
    mp = (2, 2, 2)
    kpts = monkhorst_pack(mp)
    real = np.eye(3) * 5.0
    recip = 2 * np.pi * np.linalg.inv(real).T
    gen = generate_nnkp(kpts, recip, mp)

    # analytic s;d projections
    p = tmp_path / "a.nnkp"
    write_nnkp(p, real, recip, kpts, gen["nnkpts"], gen["g_vectors"],
               projections=spd_projections((0, 0, 0), "s;d"))
    text = p.read_text()
    assert "begin projections" in text
    # 6 projection specs -> count line "6" then two lines each
    assert text.split("begin projections")[1].split()[0] == "6"

    # SCDM auto_projections
    p2 = tmp_path / "b.nnkp"
    write_nnkp(p2, real, recip, kpts, gen["nnkpts"], gen["g_vectors"],
               num_proj=0, auto_projections=6)
    t2 = p2.read_text()
    assert "begin auto_projections" in t2
    assert t2.split("begin auto_projections")[1].split()[:2] == ["6", "0"]
