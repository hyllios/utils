"""Staleness of the reused .amn/.mmn/.eig against the settings asked for now.

`generate_overlaps` used to gate reuse on "the files exist and both QE jobs said
JOB DONE", which says nothing about WHAT they were made with. Step 2 rewrites the
.nnkp unconditionally, so changing `projections` and rerunning in the same
directory left a new .nnkp beside an .amn built from the old projections, and the
Wannierization proceeded on it silently. `_check_overlap_block_order` cannot see
it: the projections do not change the neighbour block order.
"""
import json

import numpy as np
import pytest
from ase.build import bulk

from waw.interfaces.quantum_espresso.pipeline import (
    NSCF_KEYS, _fingerprint_verdict, _jsonable, _overlap_fingerprint,
)

ATOMS = bulk("Si", "diamond", a=5.43)
BASE = dict(mp_grid=(4, 4, 4), nbnd=12, ecutwfc=30.0, exclude_bands=(),
            num_wann=4, projections=[((0.0, 0.0, 0.0), 0, 1, 1)],
            spinors=False, gamma_only=False, nscf_symmetry=False,
            nscf_electrons=None, system_extra=None, hubbard=None,
            hubbard_projector="ortho-atomic", atom_proj=False,
            atom_proj_ext=False, atom_proj_dir=None, atom_proj_exclude=(),
            scdm_entanglement="isolated", scdm_mu=None, scdm_sigma=None,
            spin_component=None, sym_ops=None,
            write_flags=dict(spn=False, unk=False, uHu=False, sHu=False,
                             sIu=False, dmn=False))


def fp(**over):
    return _overlap_fingerprint(atoms=ATOMS, **{**BASE, **over})


def write(path, **over):
    path.write_text(json.dumps(fp(**over), indent=1, sort_keys=True))


def test_identical_settings_reuse_everything(tmp_path):
    p = tmp_path / "s.waw-provenance.json"
    write(p)
    assert _fingerprint_verdict(p, fp(), True) == (False, False)


def test_changed_projections_rerun_pw2wannier90_but_not_the_nscf(tmp_path):
    """The whole point: new projections must invalidate the .amn.

    They do not touch the NSCF, whose wavefunctions stay valid, so paying for it
    again would be waste -- but reusing the .amn is wrong, not merely stale.
    """
    p = tmp_path / "s.waw-provenance.json"
    write(p)
    stale_nscf, stale_ovl = _fingerprint_verdict(
        p, fp(projections=[((0.25, 0.25, 0.25), 1, 1, 1)]), True)
    assert (stale_nscf, stale_ovl) == (False, True)


@pytest.mark.parametrize("key,value", [
    ("num_wann", 8),
    ("exclude_bands", (1, 2, 3, 4)),
    ("scdm_entanglement", "erfc"),
    ("scdm_mu", 7.5),
    ("atom_proj", True),
    ("spin_component", "up"),
    ("spinors", True),
    ("write_flags", dict(spn=True, unk=False, uHu=False, sHu=False,
                         sIu=False, dmn=False)),
])
def test_overlap_side_settings_invalidate_only_the_overlaps(tmp_path, key, value):
    p = tmp_path / "s.waw-provenance.json"
    write(p)
    assert _fingerprint_verdict(p, fp(**{key: value}), True) == (False, True)


@pytest.mark.parametrize("key,value", [
    ("mp_grid", (6, 6, 6)),
    ("nbnd", 20),
    ("ecutwfc", 60.0),
    ("nscf_symmetry", True),
    ("gamma_only", True),
    ("nscf_electrons", {"conv_thr": 1e-10}),
    ("system_extra", {"noncolin": True}),
    ("hubbard", {"Si-3p": 3.0}),
])
def test_wavefunction_side_settings_invalidate_both(tmp_path, key, value):
    p = tmp_path / "s.waw-provenance.json"
    write(p)
    assert _fingerprint_verdict(p, fp(**{key: value}), True) == (True, True)
    assert key in NSCF_KEYS


def test_a_moved_atom_invalidates_both(tmp_path):
    p = tmp_path / "s.waw-provenance.json"
    write(p)
    moved = ATOMS.copy()
    moved.positions[1] += 0.01
    assert _fingerprint_verdict(
        p, _overlap_fingerprint(atoms=moved, **BASE), True) == (True, True)


def test_missing_record_warns_instead_of_guessing(tmp_path):
    """A run predating the check cannot be judged -- say so, do not assume."""
    p = tmp_path / "s.waw-provenance.json"
    with pytest.warns(RuntimeWarning, match="cannot be checked"):
        assert _fingerprint_verdict(p, fp(), True) == (False, False)


def test_absent_files_need_no_verdict(tmp_path):
    """Nothing to reuse, so nothing to warn about; both stages run regardless."""
    p = tmp_path / "s.waw-provenance.json"
    assert _fingerprint_verdict(p, fp(), False) == (False, False)


def test_corrupt_record_reruns_everything(tmp_path):
    p = tmp_path / "s.waw-provenance.json"
    p.write_text("{not json")
    assert _fingerprint_verdict(p, fp(), True) == (True, True)


def test_fingerprint_is_json_stable_and_order_insensitive():
    """Equal settings must compare equal through a JSON round trip, including
    numpy scalars/arrays and dicts written in a different order."""
    a = fp(system_extra={"nosym": True, "noinv": False},
           projections=[((0.0, 0.0, 0.0), np.int64(0), 1, 1)])
    b = fp(system_extra={"noinv": False, "nosym": True},
           projections=[((0.0, 0.0, 0.0), 0, 1, 1)])
    assert json.loads(json.dumps(a)) == json.loads(json.dumps(b))
    assert _jsonable(np.arange(3)) == [0, 1, 2]


def test_no_warning_when_nothing_is_being_reused(tmp_path, recwarn):
    """`files_present` means "about to be reused", not "exists on disk". A
    rerun_nscf=True call regenerates everything, so the absent-provenance warning
    has nothing to be uncertain about -- it fired there at first, on the very
    first real run (notebook 02), which is how this was found."""
    p = tmp_path / "s.waw-provenance.json"
    assert _fingerprint_verdict(p, fp(), False) == (False, False)
    assert not [w for w in recwarn if "cannot be checked" in str(w.message)]
