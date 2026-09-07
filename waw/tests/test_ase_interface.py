"""
Tests for the ASE-based interface (waw.interfaces.ase).

The ASE interface is the new front door to the same atomic-unit core the
Wannier90 interface drives.  These tests check that:

  1. structure.real_lattice / recip_lattice (ASE Atoms, Angstrom) reproduce the
     Wannier90 loader's atomic-unit lattices exactly.
  2. monkhorst_pack gives the Γ-centred m/N crystal mesh.
  3. nnkp_from_kpb_map reconstructs the .nnkp neighbour topology from a .mmn
     k-pair map.
  4. build_wannier_data assembles a WannierData bit-identical to the one the
     Wannier90 loader.load produces from the same files (same eig eV→Ha
     conversion, same b-vectors/weights).
  5. save_npz / load_npz round-trip the problem arrays.
  6. End to end: the ASE-native wannierize reproduces the Wannier90-path
     wannierize (same core, only the front door differs) on isolated-band Si.
"""

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

ase = pytest.importorskip("ase")
from ase import Atoms

from waw.interfaces.ase import structure, driver
from waw.interfaces.wannier90.io import read_win, read_nnkp, read_mmn, read_amn, read_eig
from waw.interfaces.wannier90.loader import (
    load, parse_real_lattice, parse_recip_lattice,
)
from waw.units import BOHR_TO_ANG

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.mmn").exists()

pytestmark = pytest.mark.skipif(not HAS_SI, reason="Silicon example files not found")


def _si_atoms():
    """ase.Atoms for the Si cell (lattice from the .win, Bohr → Angstrom)."""
    win = read_win(SI_DIR / "silicon.win")
    real_bohr = parse_real_lattice(win)
    return Atoms("Si2", cell=real_bohr * BOHR_TO_ANG, pbc=True)


def _si_arrays():
    """Read the Si overlaps + neighbour topology as plain numpy arrays."""
    win = read_win(SI_DIR / "silicon.win")
    nnkp = read_nnkp(SI_DIR / "silicon.nnkp")
    mmn, _ = read_mmn(SI_DIR / "silicon.mmn")
    amn = read_amn(SI_DIR / "silicon.amn")
    eig = read_eig(SI_DIR / "silicon.eig")   # eV
    mp_grid = tuple(int(x) for x in str(win["mp_grid"]).split())
    return win, nnkp, mmn, amn, eig, mp_grid


# ---------------------------------------------------------------------------
# 1. Structure helpers vs the Wannier90 loader
# ---------------------------------------------------------------------------

def test_real_and_recip_lattice_match_w90():
    win = read_win(SI_DIR / "silicon.win")
    atoms = _si_atoms()
    np.testing.assert_allclose(structure.real_lattice(atoms),
                               parse_real_lattice(win), atol=1e-12)
    np.testing.assert_allclose(structure.recip_lattice(atoms),
                               parse_recip_lattice(win), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Γ-centred Monkhorst-Pack mesh
# ---------------------------------------------------------------------------

def test_monkhorst_pack_gamma_centred():
    mesh = structure.monkhorst_pack((2, 2, 2))
    assert mesh.shape == (8, 3)
    np.testing.assert_allclose(mesh[0], [0.0, 0.0, 0.0])
    # first index slowest, last fastest (matches .win kpoints ordering)
    np.testing.assert_allclose(mesh[1], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(mesh[2], [0.0, 0.5, 0.0])
    assert mesh.min() == 0.0 and mesh.max() == 0.5


# ---------------------------------------------------------------------------
# 3. nnkp reconstruction from the .mmn k-pair map
# ---------------------------------------------------------------------------

def test_nnkp_from_kpb_map_matches_nnkp_file():
    nnkp = read_nnkp(SI_DIR / "silicon.nnkp")
    mmn, kpb_map = read_mmn(SI_DIR / "silicon.mmn")
    nk, nnb = nnkp["nnkpts"].shape
    nnkpts, g_vectors = driver.nnkp_from_kpb_map(kpb_map, nk, nnb)
    np.testing.assert_array_equal(nnkpts, nnkp["nnkpts"])
    np.testing.assert_array_equal(g_vectors, nnkp["g_vectors"])


# ---------------------------------------------------------------------------
# 4. build_wannier_data == loader.load
# ---------------------------------------------------------------------------

def test_build_wannier_data_matches_loader():
    win, nnkp, mmn, amn, eig, _ = _si_arrays()
    recip = parse_recip_lattice(win)

    wd_ase = driver.build_wannier_data(
        recip, nnkp["kpoints"], mmn, amn, eig,
        nnkp["nnkpts"], nnkp["g_vectors"],
    )
    wd_w90 = load(SI_DIR / "silicon", recip, params=win)

    # Same shapes / dims
    assert (wd_ase.nk, wd_ase.nb, wd_ase.nw, wd_ase.nnb) == \
           (wd_w90.nk, wd_w90.nb, wd_w90.nw, wd_w90.nnb)
    # Same tensors, including eig (both converted eV→Ha) and b-vectors/weights
    for name in ("Mmn", "Amn", "eig", "kpts", "bvecs", "wb", "kb_idx"):
        np.testing.assert_allclose(
            getattr(wd_ase, name).numpy(), getattr(wd_w90, name).numpy(),
            atol=1e-12, err_msg=name,
        )


# ---------------------------------------------------------------------------
# 5. npz round-trip
# ---------------------------------------------------------------------------

def test_npz_roundtrip(tmp_path):
    win, nnkp, mmn, amn, eig, mp_grid = _si_arrays()
    real = parse_real_lattice(win)

    out = driver.save_npz(
        tmp_path / "si",
        kpts=nnkp["kpoints"], mmn=mmn, amn=amn, eig=eig,
        nnkpts=nnkp["nnkpts"], g_vectors=nnkp["g_vectors"],
        real_lattice=real, mp_grid=mp_grid,
    )
    assert out.exists()

    d = driver.load_npz(out)
    np.testing.assert_array_equal(d["mmn"], mmn)
    np.testing.assert_array_equal(d["amn"], amn)
    np.testing.assert_allclose(d["eig"], eig)
    np.testing.assert_array_equal(d["nnkpts"], nnkp["nnkpts"])
    np.testing.assert_allclose(d["real_lattice"], real)
    assert d["mp_grid"] == mp_grid


# ---------------------------------------------------------------------------
# 6. End to end: ASE front door reproduces the Wannier90-path result
# ---------------------------------------------------------------------------

def test_ase_wannierize_matches_w90_path():
    from waw import wannierize as w90_wannierize

    win, nnkp, mmn, amn, eig, mp_grid = _si_arrays()
    atoms = _si_atoms()

    r_ase = driver.wannierize(
        atoms, mp_grid, nnkp["kpoints"],
        mmn=mmn, amn=amn, eig=eig,
        nnkpts=nnkp["nnkpts"], g_vectors=nnkp["g_vectors"],
        nw=4, n_iter=500, conv_tol=1e-10, seed=0, verbose=False,
    )
    r_w90 = w90_wannierize(
        SI_DIR / "silicon", nw=4,
        n_iter=500, conv_tol=1e-10, seed=0,
        write_outputs=False, verbose=False,
    )

    # Same core, same seed/settings → the two front doors agree to ~machine eps.
    np.testing.assert_allclose(r_ase.omega_final, r_w90.omega_final, rtol=1e-9)
    # …and both reproduce the wannier90 reference Ω (6.468598 Ang²).
    assert abs(r_ase.omega_final * BOHR_TO_ANG**2 - 6.468598306) < 1e-2


# ---------------------------------------------------------------------------
# 7. band_path / band_path_segments (ASE's standard k-path)
# ---------------------------------------------------------------------------

def test_band_path_gives_ase_standard_path_and_kpts():
    from ase.build import bulk
    atoms = bulk("Si", "diamond", a=5.43)

    bp = structure.band_path(atoms, npoints=50)
    assert bp.kpts.shape == (50, 3)
    x, xspecial, labels = bp.get_linear_kpoint_axis()
    assert x.shape == (50,)
    assert len(xspecial) == len(labels)
    assert labels[0] == "G"                 # Setyawan-Curtarolo fcc path starts at Gamma


def test_band_path_segments_matches_parse_kpoint_path_format():
    from ase.build import bulk
    from waw.analysis.kpath import parse_kpoint_path, build_kpath

    atoms = bulk("Si", "diamond", a=5.43)
    segments_str = structure.band_path_segments(atoms)
    assert all(len(s.split()) == 8 for s in segments_str)   # "L1 x y z L2 x y z"

    # round-trips through the existing waw.analysis.kpath machinery unchanged
    segments = parse_kpoint_path(segments_str)
    kpath = build_kpath(segments, structure.recip_lattice(atoms), n_points=30)
    assert kpath.kpts.shape[1] == 3
    assert kpath.tick_labels[0] == "G"

    # matches the plain ASE BandPath's own special points exactly
    bp = structure.band_path(atoms)
    for label1, k1, label2, k2 in segments:
        np.testing.assert_allclose(k1, bp.special_points[label1], atol=1e-8)
        np.testing.assert_allclose(k2, bp.special_points[label2], atol=1e-8)
