"""Run-directory manifests and the layout convention."""
import json
import pytest

from waw.utils import runs as R


def test_layout_is_code_compound_purpose(tmp_path):
    d = R.run_dir(tmp_path, "vasp", "cui", "bulk_eos")
    assert d.relative_to(tmp_path).parts == ("runs", "vasp", "cui", "bulk_eos")
    assert d.is_dir()


def test_variants_nest_instead_of_becoming_siblings(tmp_path):
    """Seven top-level al_scdm_auto_r*_s*_n* directories is what the alternative
    looks like; a sweep belongs inside one purpose."""
    a = R.run_dir(tmp_path, "qe", "al", "scdm_scan", variant="r0_s4_n4")
    b = R.run_dir(tmp_path, "qe", "al", "scdm_scan", variant="r1_s6_n4")
    assert a.parent == b.parent


@pytest.mark.parametrize("kw,match", [
    (dict(code="abinit"), "unknown code"),
    (dict(compound="CuI"), "lowercase"),
    (dict(purpose="bulk eos"), "lowercase"),
])
def test_bad_keys_are_rejected(tmp_path, kw, match):
    base = dict(code="vasp", compound="cui", purpose="bulk_eos")
    with pytest.raises(ValueError, match=match):
        R.run_dir(tmp_path, **{**base, **kw})


def test_manifest_round_trip_and_created_is_preserved(tmp_path):
    d = R.run_dir(tmp_path, "siesta", "fe", "bands")
    R.stamp(d, code="siesta", compound="fe", purpose="bands", owner="nb18")
    first = R.read(d)["created"]
    R.stamp(d, code="siesta", compound="fe", purpose="bands", owner="nb18",
            note="rerun")
    again = R.read(d)
    assert again["created"] == first          # creation date survives a rewrite
    assert again["note"] == "rerun"
    assert again["status"] == "live"


def test_status_is_validated(tmp_path):
    d = R.run_dir(tmp_path, "qe", "al", "elph")
    with pytest.raises(ValueError, match="status"):
        R.stamp(d, code="qe", compound="al", purpose="elph", owner="nb13",
                status="probably-fine")


def test_corrupt_manifest_reads_as_absent(tmp_path):
    d = R.run_dir(tmp_path, "qe", "al", "elph")
    (d / R.MANIFEST).write_text("{not json")
    assert R.read(d) is None


def test_survey_finds_runs_and_flags_the_unmanifested(tmp_path):
    a = R.run_dir(tmp_path, "vasp", "cui", "slab")
    (a / "OUTCAR").write_text("General timing")
    R.stamp(a, code="vasp", compound="cui", purpose="slab", owner="nb30")
    b = R.run_dir(tmp_path, "qe", "mystery", "whatever")
    (b / "x.out").write_text("JOB DONE")
    assert {s["path"] for s in R.survey(tmp_path)} == {a, b}
    assert [s["path"] for s in R.survey(tmp_path, unmanifested_only=True)] == [b]


def test_prunable_accounting(tmp_path):
    """The regenerable bulk is what makes the space reclaimable: on this repo
    137 GB of 480 sits in files above 100 MB, essentially all wavefunctions."""
    d = R.run_dir(tmp_path, "vasp", "cui", "scf")
    (d / "WAVECAR").write_bytes(b"x" * 5000)
    (d / "hr.dat").write_bytes(b"y" * 10)      # derived, small, NOT prunable
    R.stamp(d, code="vasp", compound="cui", purpose="scf", owner="nb30")
    n, files = R.prunable_bytes(d)
    assert n == 5000 and [f.name for f in files] == ["WAVECAR"]


# --- naming: the spelling has to be the one a human would look under ---------

@pytest.mark.parametrize("symbols,expected", [
    (["Mg", "B", "B"], "mgb2"),                       # not b2mg
    (["Ba", "Ti", "O", "O", "O"], "batio3"),          # not bao3ti
    (["Ni", "I", "I"], "nii2"),                       # not i2ni
    (["Ga", "As"], "gaas"),                           # not asga
    (["Ag"] * 2 + ["Nd", "Cd"], "ag2ndcd"),           # pymatgen says NdCdAg2
    (["C"] * 6 + ["H"] * 6, "c6h6"),                  # reducing gives CH: no one
])
def test_compound_is_spelled_conventionally(symbols, expected):
    assert R.compound_name(symbols) == expected


def test_formula_is_reduced_by_gcd_not_by_ase():
    """`get_chemical_formula(mode="reduce")` returns the UNREDUCED string.

    Four formula units of Ag2NdCd came back as ndcdag2ndcdag2ndcdag2ndcdag2,
    which is a plausible-looking directory name and pure noise.
    """
    assert R.compound_name((["Ag"] * 2 + ["Nd", "Cd"]) * 4) == "ag2ndcd"


def test_an_unparseable_formula_is_refused_rather_than_guessed():
    assert R.compound_name([]) is None
    assert R.compound_name([f"X{i}" for i in range(40)]) is None


# --- describing a directory --------------------------------------------------

def test_a_run_in_the_layout_states_its_own_identity(tmp_path):
    d = R.run_dir(tmp_path, "qe", "mgb2", "elph")
    assert R.describe_path(d) == {"code": "qe", "compound": "mgb2",
                                  "purpose": "elph"}


def test_a_variant_keeps_its_parent_purpose(tmp_path):
    d = R.run_dir(tmp_path, "vasp", "cui", "slab", "iodine_vacancy")
    assert R.describe_path(d)["purpose"] == "slab/iodine_vacancy"


def test_outside_the_layout_the_compound_comes_from_the_structure(tmp_path):
    d = tmp_path / "some_old_scratch_dir"
    d.mkdir()
    (d / "POSCAR").write_text(
        "MgB2\n1.0\n3.08 0 0\n-1.54 2.67 0\n0 0 3.52\nMg B\n1 2\nDirect\n"
        "0 0 0\n0.333 0.667 0.5\n0.667 0.333 0.5\n")
    got = R.describe_path(d)
    assert got["compound"] == "mgb2"
    assert got["purpose"] == "some_old_scratch_dir"


# --- autostamp: what the code itself knows, written on every run -------------

def test_autostamp_fills_identity_from_the_layout(tmp_path):
    d = R.run_dir(tmp_path, "siesta", "nio", "ctrl")
    R.autostamp(d, code="siesta", settings={"ncores": 4})
    m = R.read(d)
    assert (m["code"], m["compound"], m["purpose"]) == ("siesta", "nio", "ctrl")
    assert m["owner"] == "unassigned" and m["settings"]["ncores"] == 4


def test_autostamp_never_overwrites_what_a_human_set(tmp_path):
    d = R.run_dir(tmp_path, "qe", "al", "elph")
    R.stamp(d, code="qe", compound="al", purpose="elph", owner="notebook 24",
            status="archived", note="keep")
    R.autostamp(d, code="qe", settings={"ncores": 16})
    m = R.read(d)
    assert m["owner"] == "notebook 24" and m["status"] == "archived"
    assert m["note"] == "keep" and m["settings"]["ncores"] == 16


def test_autostamp_never_raises_and_loses_a_finished_run(tmp_path):
    """A manifest-writing failure must not destroy hours of DFT."""
    assert R.autostamp(tmp_path / "does_not_exist", code="qe") is None
    assert R.autostamp(tmp_path, code="not_a_code") is None
