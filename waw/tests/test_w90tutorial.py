"""
Smoke tests for the waw-native W90 tutorial notebooks (workflows/w90tutorial/).

These do NOT run DFT (that needs the QE toolchain and minutes of compute); they
check that the committed notebooks are well-formed, waw-native (no wannier90.x /
.win / .chk), executed (carry outputs), and that `waw.interfaces.quantum_espresso`
(the direct-input QE driver every notebook imports as `qe`) imports and its
public API is intact -- enough to catch a broken commit in CI.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
TUTDIR = REPO / "workflows" / "w90tutorial"
NOTEBOOKS = sorted(TUTDIR.glob("*.ipynb"))
HAS_TUT = TUTDIR.exists() and len(NOTEBOOKS) > 0

pytestmark = pytest.mark.skipif(not HAS_TUT, reason="w90tutorial notebooks not present")


def test_expected_notebooks_present():
    # filenames are numbered after the OFFICIAL Wannier90 tutorial they
    # reimagine (not creation order); "bonus_..." has no official number.
    names = {p.name for p in NOTEBOOKS}
    for expected in ("01_gaas_isolated.ipynb", "02_lead_isolated.ipynb",
                     "03_silicon_disentangle_bands.ipynb",
                     "04_copper_fermi_surface.ipynb", "05_diamond_wannier_functions.ipynb",
                     "07_silane_gamma_only.ipynb", "08_iron_spin_polarized.ipynb",
                     "09_batio3_bulk.ipynb",
                     "10_graphite_disentangle.ipynb",
                     "11_silicon_select_projections.ipynb",
                     "12_benzene_gamma_only.ipynb",
                     "13_cnt_transport.ipynb",
                     "14_na_chain_transport.ipynb",
                     "16_silicon_thermoelectrics.ipynb", "17_iron_soc_spin_texture.ipynb",
                     "18_iron_berry_ahc.ipynb", "19_iron_orbital_magnetization.ipynb",
                     "20_lavo3_dis_spheres.ipynb", "21_gaas_sitesym.ipynb",
                     "22_copper_sitesym.ipynb",
                     "23_silicon_yambo_gw.ipynb",
                     "24_tellurium_gyrotropic.ipynb",
                     "25_gaas_shift_current.ipynb",
                     "26_gaas_selective_localization.ipynb",
                     "27_silicon_scdm.ipynb",
                     "28_diamond_cube.ipynb",
                     "29_platinum_spin_hall.ipynb",
                     "30_gaas_ac_spin_hall.ipynb",
                     "31_platinum_soc_scdm.ipynb",
                     "32_tungsten_projectability_scdm.ipynb",
                     "33_bc2n_kdotp.ipynb",
                     "34_graphene_projectability_disentangle.ipynb",
                     "35_silicon_ext_proj.ipynb",
                     "36_silicon_ss_functional.ipynb",
                     "37_iron_translational_invariance.ipynb",
                     "bonus_aluminium_chain_transport.ipynb"):
        assert expected in names, f"missing tutorial notebook {expected}"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_waw_native_and_executed(nb_path):
    nb = json.loads(nb_path.read_text())

    # valid nbformat v4 on the waw kernel
    assert nb.get("nbformat") == 4
    assert nb["metadata"]["kernelspec"]["name"] == "waw"

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"

    src_all = "\n".join("".join(c["source"]) for c in code_cells)
    # waw-native: no wannier90 minimiser, no .win/.chk file interface
    for banned in ("wannier90.x", ".win", ".chk"):
        assert banned not in src_all, f"{nb_path.name} references {banned!r}"
    # it IS the waw pipeline -- either the high-level wannierize() wrapper,
    # or (tutorial21's site-symmetry notebook, not yet wired into that
    # wrapper) the lower-level disentangle()/minimize_spread_symmetrized()
    # core calls directly.
    assert "import waw" in src_all
    assert ("wannierize(" in src_all
           or ("disentangle(" in src_all and "minimize_spread_symmetrized(" in src_all))

    # executed: at least one code cell carries outputs, and none errored
    assert any(c.get("outputs") for c in code_cells), "notebook was not executed"
    for c in code_cells:
        for o in c.get("outputs", []):
            assert o.get("output_type") != "error", \
                f"{nb_path.name} has an error output: {o.get('ename')}"


def test_pseudos_committed():
    pdir = REPO / "workflows" / "pseudos"   # shared across workflows/, not w90tutorial-specific
    for upf in ("Ga.upf", "As.upf", "Si.upf", "Fe-sp_r.upf", "Cu.upf", "C.upf", "Al.upf", "H.upf",
                "Ba.upf", "Ti.upf", "O.upf", "Na.upf", "Pb.upf", "La.upf", "V.upf",
                "Sr.upf", "Mn.upf", "W.upf"):
        assert (pdir / upf).exists(), f"missing committed pseudo {upf}"


def test_qe_helper_imports_and_api():
    from waw.interfaces import quantum_espresso as qe
    from waw.interfaces.quantum_espresso import io as qe_io
    import inspect
    params = inspect.signature(qe.generate_overlaps).parameters
    for expected in ("ecutwfc", "scf_kpts", "nbnd", "num_wann",
                     "exclude_bands", "scdm_entanglement", "write_spn",
                     "gamma_only", "projections", "spin_component"):
        assert expected in params
    # the MPI helper forces single-threaded QE (no ranks*threads oversubscription)
    env = qe_io._mpi_env()
    assert env["OMP_NUM_THREADS"] == "1"


def test_qe_read_fermi_energy(tmp_path):
    """read_fermi_energy handles the three pw.x output forms (no DFT run)."""
    from waw.interfaces import quantum_espresso as qe

    metal = tmp_path / "m.out"
    metal.write_text("...\n     the Fermi energy is    17.3554 ev\n     JOB DONE.\n")
    assert qe.read_fermi_energy(metal) == pytest.approx(17.3554)

    insulator = tmp_path / "i.out"
    insulator.write_text("     highest occupied, lowest unoccupied level (ev):"
                         "     6.2350    6.9428\n")
    assert qe.read_fermi_energy(insulator) == pytest.approx(0.5 * (6.2350 + 6.9428))

    vbm = tmp_path / "v.out"
    vbm.write_text("     highest occupied level (ev):     8.8289\n")
    assert qe.read_fermi_energy(vbm) == pytest.approx(8.8289)

    with pytest.raises(ValueError):
        (tmp_path / "empty.out").write_text("nothing here\n")
        qe.read_fermi_energy(tmp_path / "empty.out")


def test_qe_read_bands_eigenvalues(tmp_path):
    """read_bands_eigenvalues parses a real pw.x 'bands (ev)' block
    (verbosity='high'), transcribed verbatim from an actual run."""
    from waw.interfaces import quantum_espresso as qe

    out = tmp_path / "bands.out"
    out.write_text(
        "     End of band structure calculation\n"
        "\n"
        "          k = 0.0000 0.0000 0.0000 (   531 PWs)   bands (ev):\n"
        "\n"
        "    -3.2305  20.4426  20.4426  20.4426  21.8734  21.8734  21.8734  24.2967\n"
        "\n"
        "          k =-0.1000-0.1000 0.1000 (   495 PWs)   bands (ev):\n"
        "\n"
        "    -2.9600  17.1583  19.7096  19.7096  22.5558  23.0115  23.0115  25.9109\n"
        "\n"
        "     Writing all to output data dir ./out/al.save/ :\n"
    )
    bands = qe.read_bands_eigenvalues(out, nbnd=8)
    assert bands.shape == (2, 8)
    assert bands[0, 0] == pytest.approx(-3.2305)
    assert bands[0, -1] == pytest.approx(24.2967)
    assert bands[1, 0] == pytest.approx(-2.9600)
    assert bands[1, -1] == pytest.approx(25.9109)

    (tmp_path / "empty2.out").write_text("nothing here\n")
    with pytest.raises(ValueError):
        qe.read_bands_eigenvalues(tmp_path / "empty2.out", nbnd=8)


def test_gamma_only_half_shell_nnkp():
    """The Gamma-only neighbour topology is the 3-vector half shell (not the
    full +-b 6-vector shell), matching what pw2wannier90 requires (no DFT)."""
    import numpy as np
    from waw.interfaces import quantum_espresso as qe

    nnkp = qe.gamma_only_half_shell_nnkp()
    assert nnkp["nnkpts"].shape == (1, 3)
    assert (nnkp["nnkpts"] == 0).all()          # the sole k-point is its own neighbour
    assert nnkp["g_vectors"].shape == (1, 3, 3)
    np.testing.assert_array_equal(nnkp["g_vectors"][0], np.eye(3, dtype=np.int64))


def test_write_pw_input_gamma_kpoints_card(tmp_path):
    """kpoints=('gamma',) emits QE's real-wavefunction K_POINTS {gamma} card
    (no DFT run needed -- this is pure text formatting)."""
    from waw.interfaces import quantum_espresso as qe
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
    path = qe.write_pw_input(
        tmp_path / "gamma.in", atoms,
        control={"calculation": "scf", "prefix": "x", "outdir": "./out", "pseudo_dir": "."},
        system={"ecutwfc": 30}, electrons={"conv_thr": 1e-8},
        pseudopotentials={"H": "H.upf"}, kpoints=("gamma",),
    )
    text = path.read_text()
    assert "K_POINTS {gamma}\n" in text
    assert "K_POINTS automatic" not in text
    assert "K_POINTS crystal" not in text


def test_write_pw_input_magnetic_sublattices(tmp_path):
    """ASE tags split one element into distinct QE species (Ni1/Ni2) so an
    A-type antiferromagnet can carry opposite per-species
    starting_magnetization; untagged atoms keep the bare symbol. Pure text
    formatting, no QE run."""
    from waw.interfaces import quantum_espresso as qe
    from ase import Atoms

    # 2 Ni (antiparallel sublattices) + 2 I in a c-stacked cell
    atoms = Atoms("Ni2I2",
                  scaled_positions=[[0, 0, 0.0], [0, 0, 0.5],
                                    [0, 0, 0.25], [0, 0, 0.75]],
                  cell=[[3.9, 0, 0], [0, 3.9, 0], [0, 0, 20.0]], pbc=True)
    atoms.set_tags([1, 2, 0, 0])   # Ni1, Ni2, I, I

    mag = qe.magnetization_keys(atoms, {"Ni1": 0.6, "Ni2": -0.6})
    # species order is sorted(labels) = ['I', 'Ni1', 'Ni2'] -> indices 1,2,3
    assert mag == {"starting_magnetization(2)": 0.6,
                   "starting_magnetization(3)": -0.6}

    path = qe.write_pw_input(
        tmp_path / "afm.in", atoms,
        control={"calculation": "scf", "prefix": "x", "outdir": "./out", "pseudo_dir": "."},
        system={"ecutwfc": 40, "noncolin": True, "lspinorb": True, **mag},
        electrons={"conv_thr": 1e-8},
        pseudopotentials={"Ni": "Ni-rel.upf", "I": "I-rel.upf"},
        kpoints=("automatic", (2, 2, 1), (0, 0, 0)),
    )
    text = path.read_text()
    assert "ntyp = 3" in text
    assert "nat = 4" in text
    # two Ni species lines, one shared pseudo per element
    assert " Ni1 " in text and "Ni-rel.upf" in text
    assert " Ni2 " in text
    assert text.count("Ni-rel.upf") == 2   # Ni1 and Ni2 both point at it
    # positions use the split labels
    assert text.count(" Ni1 ") >= 2 and text.count(" Ni2 ") >= 2
    assert "starting_magnetization(2) = 0.6" in text
    assert "starting_magnetization(3) = -0.6" in text


def test_write_pw_input_hubbard_card(tmp_path):
    """DFT+U emits QE's modern HUBBARD card; an element head expands to every
    matching split species (Ni1/Ni2), and no card is written when hubbard is
    omitted. Pure text formatting, no QE run."""
    from waw.interfaces import quantum_espresso as qe
    from ase import Atoms

    atoms = Atoms("Ni2I2",
                  scaled_positions=[[0, 0, 0.0], [0, 0, 0.5],
                                    [0, 0, 0.25], [0, 0, 0.75]],
                  cell=[[3.9, 0, 0], [0, 3.9, 0], [0, 0, 20.0]], pbc=True)
    atoms.set_tags([1, 2, 0, 0])   # Ni1, Ni2, I

    common = dict(
        control={"calculation": "scf", "prefix": "x", "outdir": "./out", "pseudo_dir": "."},
        system={"ecutwfc": 40}, electrons={"conv_thr": 1e-8},
        pseudopotentials={"Ni": "Ni-rel.upf", "I": "I-rel.upf"},
        kpoints=("automatic", (2, 2, 1), (0, 0, 0)),
    )
    text = qe.write_pw_input(tmp_path / "u.in", atoms,
                             hubbard={"Ni-3d": 3.24}, **common).read_text()
    assert "HUBBARD (ortho-atomic)" in text
    assert "U Ni1-3d 3.24" in text and "U Ni2-3d 3.24" in text
    assert "U I-" not in text   # I gets no U

    # Liechtenstein form: a J line per Ni species (activates lda_plus_u_kind=1)
    textJ = qe.write_pw_input(tmp_path / "uj.in", atoms,
                              hubbard={"Ni-3d": {"U": 3.24, "J": 0.68}}, **common).read_text()
    assert "U Ni1-3d 3.24" in textJ and "J Ni1-3d 0.68" in textJ
    assert "U Ni2-3d 3.24" in textJ and "J Ni2-3d 0.68" in textJ

    # no hubbard -> no card
    text0 = qe.write_pw_input(tmp_path / "nou.in", atoms, **common).read_text()
    assert "HUBBARD" not in text0


def test_write_pw_input_untagged_unchanged(tmp_path):
    """Backward compatibility: a cell with no tags produces the same bare
    sorted(set(symbols)) species list as before (no accidental relabeling)."""
    from waw.interfaces import quantum_espresso as qe
    from ase import Atoms

    atoms = Atoms("OH2", scaled_positions=[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
                  cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)
    path = qe.write_pw_input(
        tmp_path / "h2o.in", atoms,
        control={"calculation": "scf", "prefix": "x", "outdir": "./out", "pseudo_dir": "."},
        system={"ecutwfc": 30}, electrons={"conv_thr": 1e-8},
        pseudopotentials={"H": "H.upf", "O": "O.upf"}, kpoints=("gamma",),
    )
    text = path.read_text()
    assert "ntyp = 2" in text
    assert " H " in text and " O " in text
    assert "H1" not in text and "O1" not in text


def test_generate_overlaps_gamma_only_projections_skip_scdm(tmp_path, monkeypatch):
    """End-to-end (no QE binaries): gamma_only=True + explicit `projections`
    must (a) skip SCDM entirely in the pw2wannier90 namelist -- pw2wannier90.x
    has no SCDM support for gamma-only wavefunctions, a hard QE-side
    limitation -- and (b) route the .nnkp through the half-shell topology,
    not generate_nnkp's full +-b shell."""
    import numpy as np
    from waw.interfaces import quantum_espresso as qe
    # generate_overlaps (in .pipeline) resolves run_pw/run_pw2wannier90/
    # read_mmn/read_amn/read_eig as names bound in ITS OWN module namespace
    # (imported there via `from .io import ...` / `from waw.interfaces.
    # wannier90.io import ...`), not the package's `__init__`-level
    # re-exports -- patch them on `qe_pipeline`, not on `qe` itself.
    from waw.interfaces.quantum_espresso import pipeline as qe_pipeline
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[[6, 0, 0], [0, 6, 0], [0, 0, 6]], pbc=True)

    monkeypatch.setattr(qe_pipeline, "run_pw", lambda *a, **k: None)
    monkeypatch.setattr(qe_pipeline, "read_fermi_energy", lambda *a, **k: -5.0)
    captured = {}
    def fake_pw2wan(workdir, seedname, inp, **kw):
        captured["inp"] = inp
    monkeypatch.setattr(qe_pipeline, "run_pw2wannier90", fake_pw2wan)
    monkeypatch.setattr(qe_pipeline, "read_mmn", lambda p: (np.zeros((1, 3, 1, 1), dtype=complex), None))
    monkeypatch.setattr(qe_pipeline, "read_amn", lambda p: np.zeros((1, 1, 1), dtype=complex))
    monkeypatch.setattr(qe_pipeline, "read_eig", lambda p: np.zeros((1, 1)))

    sp3 = [((0.0, 0.0, 0.0), -3, mr, 1, (0., 0., 1.), (1., 0., 0.), 1.0) for mr in (1,)]
    ov = qe.generate_overlaps(
        atoms, (1, 1, 1), tmp_path, "x",
        ecutwfc=30, scf_kpts=(1, 1, 1), nbnd=1, num_wann=1,
        pseudopotentials={"H": "H.upf"}, pseudo_dir=tmp_path,
        gamma_only=True, projections=sp3,
    )
    assert not any(k.startswith("scdm") for k in captured["inp"])   # SCDM fully skipped
    np.testing.assert_array_equal(ov["kpts"], np.zeros((1, 3)))     # forced to Gamma

    nnkp_text = (tmp_path / "x.nnkp").read_text()
    assert "begin projections" in nnkp_text                        # analytic, not auto_projections
    assert "begin auto_projections" not in nnkp_text
    assert "    3\n" in nnkp_text.split("begin nnkpts")[1].split("end nnkpts")[0]  # nntot = 3


def test_generate_overlaps_spin_component(tmp_path, monkeypatch):
    """spin_component='up'/'down' must reach pw2wannier90's inputpp namelist
    verbatim (no DFT run -- run_pw/run_pw2wannier90/readers are stubbed)."""
    import numpy as np
    from waw.interfaces import quantum_espresso as qe
    from waw.interfaces.quantum_espresso import pipeline as qe_pipeline
    from ase.build import bulk

    atoms = bulk("Fe", "bcc", a=2.8699)
    monkeypatch.setattr(qe_pipeline, "run_pw", lambda *a, **k: None)
    monkeypatch.setattr(qe_pipeline, "read_fermi_energy", lambda *a, **k: 17.0)
    captured = {}
    monkeypatch.setattr(qe_pipeline, "run_pw2wannier90",
                         lambda workdir, seedname, inp, **kw: captured.setdefault("inp", inp))
    monkeypatch.setattr(qe_pipeline, "read_mmn", lambda p: (np.zeros((1, 8, 1, 1), dtype=complex), None))
    monkeypatch.setattr(qe_pipeline, "read_amn", lambda p: np.zeros((1, 1, 1), dtype=complex))
    monkeypatch.setattr(qe_pipeline, "read_eig", lambda p: np.zeros((1, 1)))

    qe.generate_overlaps(
        atoms, (1, 1, 1), tmp_path, "fe_up",
        ecutwfc=30, scf_kpts=(1, 1, 1), nbnd=1, num_wann=1,
        pseudopotentials={"Fe": "Fe.upf"}, pseudo_dir=tmp_path,
        spin_component="up",
    )
    assert captured["inp"]["spin_component"] == "up"


def test_generate_overlaps_sym_ops(tmp_path, monkeypatch):
    """sym_ops must write a `.sym` file and set pw2wannier90's read_sym=.true.
    (no DFT run -- run_pw/run_pw2wannier90/readers are stubbed), Wannier90
    tutorial22's mechanism for a Wannier centre that's a fixed point of only
    a SUBGROUP of the crystal's full symmetry."""
    import numpy as np
    from waw.interfaces import quantum_espresso as qe
    from waw.interfaces.quantum_espresso import pipeline as qe_pipeline
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.615)
    monkeypatch.setattr(qe_pipeline, "run_pw", lambda *a, **k: None)
    monkeypatch.setattr(qe_pipeline, "read_fermi_energy", lambda *a, **k: 17.0)
    captured = {}
    monkeypatch.setattr(qe_pipeline, "run_pw2wannier90",
                         lambda workdir, seedname, inp, **kw: captured.setdefault("inp", inp))
    monkeypatch.setattr(qe_pipeline, "read_mmn", lambda p: (np.zeros((1, 8, 1, 1), dtype=complex), None))
    monkeypatch.setattr(qe_pipeline, "read_amn", lambda p: np.zeros((1, 1, 1), dtype=complex))
    monkeypatch.setattr(qe_pipeline, "read_eig", lambda p: np.zeros((1, 1)))

    rotations = np.tile(np.eye(3), (3, 1, 1))
    translations = np.zeros((3, 3))

    qe.generate_overlaps(
        atoms, (1, 1, 1), tmp_path, "cu",
        ecutwfc=30, scf_kpts=(1, 1, 1), nbnd=1, num_wann=1,
        pseudopotentials={"Cu": "Cu.upf"}, pseudo_dir=tmp_path,
        sym_ops=(rotations, translations),
    )
    assert captured["inp"]["read_sym"] is True
    sym_text = (tmp_path / "cu.sym").read_text()
    assert sym_text.split()[0] == "3"


def test_generate_overlaps_atom_proj_ext(tmp_path, monkeypatch):
    """atom_proj_ext=True must set pw2wannier90's atom_proj=.true. +
    atom_proj_ext=.true. + atom_proj_dir, and atom_proj_exclude (if given)
    must reach the namelist as space-separated ints, not repr()'s
    Python-list "[..]" syntax (no DFT run -- run_pw/run_pw2wannier90/
    readers are stubbed), Wannier90 tutorial 35's external-projector path."""
    import numpy as np
    from waw.interfaces import quantum_espresso as qe
    from waw.interfaces.quantum_espresso import pipeline as qe_pipeline
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43)
    monkeypatch.setattr(qe_pipeline, "run_pw", lambda *a, **k: None)
    monkeypatch.setattr(qe_pipeline, "read_fermi_energy", lambda *a, **k: 6.0)
    captured = {}
    monkeypatch.setattr(qe_pipeline, "run_pw2wannier90",
                         lambda workdir, seedname, inp, **kw: captured.setdefault("inp", inp))
    monkeypatch.setattr(qe_pipeline, "read_mmn", lambda p: (np.zeros((1, 8, 1, 1), dtype=complex), None))
    monkeypatch.setattr(qe_pipeline, "read_amn", lambda p: np.zeros((1, 1, 1), dtype=complex))
    monkeypatch.setattr(qe_pipeline, "read_eig", lambda p: np.zeros((1, 1)))

    ext_dir = tmp_path / "ext_proj"
    qe.generate_overlaps(
        atoms, (1, 1, 1), tmp_path, "si",
        ecutwfc=30, scf_kpts=(1, 1, 1), nbnd=1, num_wann=1,
        pseudopotentials={"Si": "Si.upf"}, pseudo_dir=tmp_path,
        atom_proj_ext=True, atom_proj_dir=ext_dir, atom_proj_exclude=[5, 6, 7],
    )
    inp = captured["inp"]
    assert inp["atom_proj"] is True
    assert inp["atom_proj_ext"] is True
    assert inp["atom_proj_dir"] == str(ext_dir)
    assert inp["atom_proj_exclude"] == [5, 6, 7]

    from waw.interfaces.quantum_espresso.io import _namelist
    text = _namelist("INPUTPP", inp)
    assert "atom_proj_exclude = 5 6 7" in text
    assert "[5, 6, 7]" not in text
