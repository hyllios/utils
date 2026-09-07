"""VASP interface input writers / readers (pure text, no VASP run)."""

import numpy as np
import pytest
from ase import Atoms

from waw.interfaces import vasp
from waw.interfaces.vasp.pipeline import _ldau_arrays


def _nii2_cell():
    atoms = Atoms("Ni2I4",
                  scaled_positions=[[0, 0, 0.0], [0, 0, 0.5],
                                    [0.25, 0.5, 0.375], [0.75, 0.5, 0.125],
                                    [0.25, 0.5, 0.875], [0.75, 0.5, 0.625]],
                  cell=[[3.9, 0, 0], [-1.95, 3.37, 0], [0, -4.49, 13.09]], pbc=True)
    atoms.set_tags([1, 2, 0, 0, 0, 0])   # Ni1(up), Ni2(dn), I...
    return atoms


def test_incar_formatting(tmp_path):
    incar = {"ENCUT": 520, "LNONCOLLINEAR": True, "LSORBIT": True,
             "LDAUL": [-1, 2], "LDAUU": [0.0, 3.24], "ISYM": -1}
    text = vasp.write_incar(incar, tmp_path / "INCAR").read_text()
    assert "ENCUT = 520" in text
    assert "LNONCOLLINEAR = .TRUE." in text
    assert "LDAUL = -1 2" in text
    assert "LDAUU = 0.0 3.24" in text
    assert "ISYM = -1" in text


def test_poscar_species_order_and_magmom(tmp_path):
    atoms = _nii2_cell()
    species, counts, sort_index = vasp.write_poscar(atoms, tmp_path / "POSCAR")
    assert species == ["I", "Ni"] and counts == [4, 2]     # ASE groups sorted
    # noncollinear MAGMOM: 3 per atom, in POSCAR order (I's first, then Ni up/dn)
    mag = vasp.noncollinear_magmom(atoms, {1: 2.0, 2: -2.0}, sort_index)
    assert len(mag) == 3 * 6
    assert mag[:12] == [0.0] * 12                          # 4 I atoms, no moment
    assert mag[12:15] == [0.0, 0.0, 2.0]                   # Ni1 up (+z)
    assert mag[15:18] == [0.0, 0.0, -2.0]                  # Ni2 down (-z)


def test_ldau_arrays_liechtenstein():
    L, U, J = _ldau_arrays(["I", "Ni"], {"Ni": {"L": 2, "U": 3.24, "J": 0.68}})
    assert L == [-1, 2] and U == [0.0, 3.24] and J == [0.0, 0.68]


def test_wannier_win(tmp_path):
    proj = ["f=0.0,0.0,0.0:d", "f=0.25,0.5,0.375:p"]
    text = vasp.write_wannier_win(proj, 44, tmp_path / "wannier90.win", spinors=True).read_text()
    assert "num_wann = 44" in text
    assert "spinors = .true." in text
    assert "num_iter = 0" in text and "dis_num_iter = 0" in text   # VASP writes overlaps only
    assert "f=0.0,0.0,0.0:d" in text and "end projections" in text


def test_kpoints_gamma_mesh(tmp_path):
    text = vasp.write_kpoints((6, 6, 4), tmp_path / "KPOINTS").read_text()
    assert "Gamma" in text and "6 6 4" in text


def test_potcar_concatenation_order(tmp_path):
    pot = tmp_path / "pot"
    for s, tag in [("I", b"POTCAR-I\n"), ("Ni", b"POTCAR-Ni\n")]:
        (pot / s).mkdir(parents=True)
        (pot / s / "POTCAR").write_bytes(tag)
    vasp.write_potcar(["I", "Ni"], pot, tmp_path / "POTCAR")
    assert (tmp_path / "POTCAR").read_bytes() == b"POTCAR-I\nPOTCAR-Ni\n"


def test_read_kpoints_ibzkpt(tmp_path):
    (tmp_path / "IBZKPT").write_text(
        "Auto mesh\n2\nReciprocal\n"
        "  0.0 0.0 0.0  1\n  0.5 0.0 0.0  1\n"
    )
    k = vasp.read_kpoints(tmp_path)
    assert k.shape == (2, 3)
    assert np.allclose(k[1], [0.5, 0.0, 0.0])
