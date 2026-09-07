"""SIESTA NAO -> Löwdin HamiltonianR (waw.interfaces.siesta).

Needs the completed bcc-Fe SIESTA run in workflows/notebooks/runs/fe_siesta
(fe.fdf + fe.HSX, produced by the notebook-18 pipeline) and sisl; skipped
when absent.
"""
import pathlib

import numpy as np
import pytest

sisl = pytest.importorskip("sisl")

from waw.analysis.elph import band_eigensystem
from waw.analysis.exchange import heisenberg_exchange
from waw.units import HARTREE_TO_EV

W = pathlib.Path(__file__).resolve().parents[1] / "workflows/notebooks/runs/fe_siesta"

pytestmark = pytest.mark.skipif(not (W / "fe.HSX").exists(),
                                reason="fe_siesta run absent")


@pytest.fixture(scope="module")
def H():
    from waw.interfaces import siesta as sst

    return sst.load_hamiltonian(W / "fe.fdf")


def test_lowdin_on_mesh_parity(H):
    """The Löwdin model reproduces the generalized eigenproblem EXACTLY
    on its own mesh, for both spin channels."""
    from waw.interfaces import siesta as sst

    kt = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0.5, 0.5]])
    for sp in (0, 1):
        hr = sst.lowdin_hamiltonian(H, (4, 4, 4), spin=sp)
        e_w = band_eigensystem(hr, kt)[0] * HARTREE_TO_EV
        e_s = np.array([H.eigh(k, spin=sp) for k in kt])
        assert np.abs(e_w - e_s).max() < 1e-6


def test_fermi_reference_is_zero(H):
    """TSHS stores H - E_F S: the occupied d manifold must sit below 0."""
    from waw.interfaces import siesta as sst

    hr = sst.lowdin_hamiltonian(H, (6, 6, 6), spin=0)
    eig = band_eigensystem(hr, np.zeros((1, 3)))[0][0] * HARTREE_TO_EV
    assert (eig < 0).sum() >= 8      # 4 semicore + 4s + majority d at Gamma


def test_exchange_moment_and_symmetry(H):
    """Native LKAG on the Löwdin channels: the all-orbital moment equals
    SIESTA's spin moment, and the NAO basis is symmetry-adapted (J1
    symmetric over the 8 bcc neighbours with no gauge tricks)."""
    from ase.build import bulk

    from waw.interfaces import siesta as sst

    cell = bulk("Fe", "bcc", a=2.8699).get_cell()[:]
    hr = {sp: sst.lowdin_hamiltonian(H, (8, 8, 8), spin=sp) for sp in (0, 1)}
    n = 8
    g = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    kpts = np.stack([x.ravel() for x in g], axis=1) / n
    R1 = np.array([R for R in
                   [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
                   if R != (0, 0, 0) and abs(np.linalg.norm(np.array(R) @ cell) - 2.485) < 0.01])
    res = heisenberg_exchange(hr[0], hr[1], kpts, 0.0,
                              [list(range(19))], R1, nz=60)
    assert res.magmoms[0] == pytest.approx(2.31, abs=0.05)
    j1 = np.array([res.J[(tuple(R), 0, 0)] for R in R1]) * HARTREE_TO_EV * 1e3
    assert len(j1) == 8
    assert np.ptp(j1) < 0.1          # symmetry-adapted basis: tiny spread
    assert j1.mean() > 1.0           # ferromagnetic nearest-neighbour J


def test_write_fdf_extra_overrides_the_defaults(tmp_path):
    """
    `extra` must win over the built-in lines, whatever the fdf spelling.

    In fdf the FIRST occurrence of a label wins, and labels ignore case and
    the separators '.', '-', '_'. write_fdf emitted "MaxSCFIterations 200"
    before `extra`, so a caller asking for "Max.SCF.Iterations 400" silently
    got 200 -- which is how an SCF that needed more cycles kept stopping at
    exactly the default and looking like a convergence failure.
    """
    from ase.build import bulk
    from waw.interfaces.siesta import write_fdf

    fe = bulk("Fe", "bcc", a=2.87)
    pd = tmp_path / "ps"
    pd.mkdir()
    (pd / "Fe.psml").write_text("<psml/>")
    p = write_fdf(tmp_path / "fe.fdf", fe, label="fe", pseudo_dir=pd,
                  kgrid=(2, 2, 2),
                  extra={"Max.SCF.Iterations": 400, "scf-dm-tolerance": "1.e-6"})
    text = p.read_text()
    assert "MaxSCFIterations 200" not in text
    assert "Max.SCF.Iterations 400" in text
    assert "SCF.DM.Tolerance 1.0e-5" not in text
    assert "scf-dm-tolerance 1.e-6" in text
    # untouched defaults survive, and blocks are not damaged
    assert "DM.MixingWeight 0.15" in text
    assert "%block kgrid.MonkhorstPack" in text and "%endblock" in text
    assert text.count("SaveHS T") == 1


def test_scf_consistency_check_separates_healthy_from_corrupt_runs(tmp_path):
    """
    The guard against this cluster's SIESTA >= 8-rank corruption.

    siesta/5.4.1-gcc-13.2.0 can return a wrong Hamiltonian for some
    (n_orbitals, ncores, BlockSize) combinations while still printing "SCF
    Convergence by DM": the SCF trace looks self-consistent line by line, but
    the final summary's Eharris and Etot -- equal at self-consistency by
    construction -- come out 830-2310 eV apart. Every healthy run measured on
    this build gives exactly 0.00.

    Three distinct failures must not be confused, because each sends the reader
    somewhere different. Fixtures are cut from real outputs in runs/:
    oxide_np4 and ag2ndcd_sc_AFM_X (healthy), narrow_species2_cubic (corrupt,
    np8 BlockSize 17), ag2ndcd_fsm0 (merely unconverged, 0.50 eV), np4_FM
    (unfinished), np4_AFM_X (healthy but with interleaved foreign lines).
    """
    from waw.interfaces.siesta.io import check_scf_consistency

    def _out(e_harris, e_tot, *, converged=True, finished=True, trailing=""):
        t = (f"   scf:    1   -5000.000000   -5000.000000   -5000.100000\n"
             f"   scf:   17   {e_harris:.6f}   {e_harris:.6f}   {e_harris:.6f}\n")
        if converged:
            t += "SCF Convergence by DM+H criterion\n"
        t += (f"siesta: Final energy (eV):\n"
              f"siesta:       Kinetic =    2469.629698\n"
              f"siesta:       Eharris =   {e_harris:.6f}\n"
              f"siesta:       Etot    =   {e_tot:.6f}\n"
              f"siesta:         Total =   {e_tot:.6f}\n" + trailing)
        if finished:
            t += "siesta: End of run\n"
        return t

    healthy = tmp_path / "clean.out"
    healthy.write_text(_out(-16704.444747, -16704.444747))
    assert check_scf_consistency(healthy) == pytest.approx(0.0, abs=1e-9)

    # the corruption: converged marker present, energies keV apart
    corrupt = tmp_path / "corrupt.out"
    corrupt.write_text(_out(-21259.090442, -18949.685989))
    with pytest.raises(RuntimeError, match="parallel corruption"):
        check_scf_consistency(corrupt)
    # the gap, not the magnitude, decides -- a loose tolerance must let it pass
    assert check_scf_consistency(corrupt, tol_ev=1e4) == pytest.approx(2309.40, abs=0.01)

    # ordinary non-convergence must NOT be reported as the parallel bug
    unconverged = tmp_path / "unconverged.out"
    unconverged.write_text(_out(-59926.482421, -59925.985287, converged=False))
    with pytest.raises(RuntimeError, match="did NOT converge"):
        check_scf_consistency(unconverged)

    # an unfinished run cannot be certified: mid-SCF Eharris/Etot differ by ~1
    # keV in ANY healthy run, so reading them early would flag everything
    partial = tmp_path / "partial.out"
    partial.write_text(_out(-58484.982235, -59495.624001, finished=False))
    with pytest.raises(RuntimeError, match="does not reach the end"):
        check_scf_consistency(partial)

    # A SIESTA .out can carry INTERLEAVED lines from a second process (a killed
    # run whose fd still held an offset when a relaunch truncated the file), so
    # the textually last `scf:` line may belong to a different run entirely.
    # Reading one adjacent summary pair is immune -- comparing the last scf line
    # against the summary gave a 0.891 eV false positive on a healthy run.
    interleaved = tmp_path / "interleaved.out"
    interleaved.write_text(_out(
        -59940.465428, -59940.465428,
        trailing="      scf:   56   -59939.306099   -59939.574188   -59939.638104\n"))
    assert check_scf_consistency(interleaved) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------- launcher policy --
def test_siesta_launcher_default_is_plain_mpirun():
    from waw.interfaces.siesta import siesta_launcher
    assert siesta_launcher(4) == ["mpirun", "-np", "4"]


def test_siesta_launcher_env_override(monkeypatch):
    """A submission script must be able to choose the launcher without editing
    the library: MPI flags are site-specific (a login node wants --bind-to none
    to stop ranks piling onto one core; a batch node under SLURM may want srun),
    so baking them in would force a library edit per cluster."""
    from waw.interfaces.siesta import siesta_launcher
    monkeypatch.setenv("WAW_SIESTA_LAUNCHER", "srun -n 4 --cpu-bind=cores")
    assert siesta_launcher(4) == ["srun", "-n", "4", "--cpu-bind=cores"]


def test_siesta_launcher_argument_beats_env(monkeypatch):
    from waw.interfaces.siesta import siesta_launcher
    monkeypatch.setenv("WAW_SIESTA_LAUNCHER", "mpirun -np 99")
    assert siesta_launcher(4, ["mpirun", "-np", "2"]) == ["mpirun", "-np", "2"]


def test_siesta_launcher_empty_env_falls_back_to_default(monkeypatch):
    """An exported-but-empty variable is 'unset', not a launcher of no words."""
    from waw.interfaces.siesta import siesta_launcher
    monkeypatch.setenv("WAW_SIESTA_LAUNCHER", "")
    assert siesta_launcher(6) == ["mpirun", "-np", "6"]


# ---------------------------------------------------------------------------
# What the loaded model carries. `lowdin_hamiltonian` used to compute
# real_lattice for the Wigner-Seitz set and then drop it, so every SIESTA model
# came back with real_lattice/mp_grid/centres all None. Consequences were
# silent: `interpolate_bands(ws="auto")` fell back to the plain sum on every one
# of them, and `analysis.surface`/`analysis.floquet` could not be used at all
# without the caller rebuilding orbital positions by hand.
# ---------------------------------------------------------------------------

def test_loaded_model_knows_its_cell_and_mesh(H):
    from waw.interfaces import siesta as sst

    hr = sst.lowdin_hamiltonian(H, (4, 4, 4), spin=0)
    assert hr.mp_grid == (4, 4, 4)
    assert hr.real_lattice is not None
    from waw.units import ANG_TO_BOHR
    assert np.allclose(hr.real_lattice, np.asarray(H.geometry.cell) * ANG_TO_BOHR)


def test_centres_are_opt_in_and_enable_use_ws_distance(H):
    """Default stays centre-less (the loader does not fabricate position matrix
    elements); centres="atomic" is exact for NAOs and switches ws_distance on."""
    from waw.interfaces import siesta as sst

    plain = sst.lowdin_hamiltonian(H, (4, 4, 4), spin=0)
    assert plain.centres is None and plain.ws_distance() is None

    withc = sst.lowdin_hamiltonian(H, (4, 4, 4), spin=0, centres="atomic")
    assert withc.centres.shape == (withc.nw, 3)
    assert withc.ws_distance() is not None


def test_orbital_centres_sit_on_their_own_atom(H):
    from waw.interfaces.siesta.loader import orbital_centres
    from waw.units import ANG_TO_BOHR

    cen = orbital_centres(H)
    g = H.geometry
    assert cen.shape == (g.no, 3)
    for io in (0, g.no // 2, g.no - 1):
        assert np.allclose(cen[io], np.asarray(g.xyz[g.o2a(io)]) * ANG_TO_BOHR)


def test_centres_argument_is_validated(H):
    from waw.interfaces import siesta as sst

    with pytest.raises(ValueError, match="centres must be"):
        sst.lowdin_hamiltonian(H, (2, 2, 2), spin=0, centres="wannier")


def test_ws_distance_is_the_identity_for_a_one_atom_cell(H):
    """bcc Fe has a single atom, so every NAO centre coincides and the
    use_ws_distance remapping is exactly the identity -- this fixture therefore
    CANNOT show the difference the correction makes. It does show that switching
    it on is harmless, and that it is on: `ws_distance()` is built, and the two
    interpolations agree to machine precision because the centres are equal, not
    because the correction was skipped. The case where it changes bands needs
    inequivalent centres and is covered in test_ws_distance.py.
    """
    from waw.interfaces import siesta as sst
    from waw.core.hamiltonian import interpolate_bands

    assert H.geometry.na == 1
    hr = sst.lowdin_hamiltonian(H, (4, 4, 4), spin=0, centres="atomic")
    assert hr.ws_distance() is not None
    assert np.allclose(hr.centres, hr.centres[0])
    off = np.array([[0.1, 0.07, 0.03], [0.31, 0.19, 0.44]])
    d = np.abs(interpolate_bands(hr, off)
               - interpolate_bands(hr, off, ws=None)).max() * HARTREE_TO_EV
    assert d < 1e-9, f"coincident centres must not change anything, got {d} eV"


# ---------------------------------------------------------------------------
# A multi-atom NAO model, where use_ws_distance actually does something. The
# reference is not another interpolation but SIESTA's own generalized
# eigenproblem, which is available at ANY k -- so "better" here is measured, not
# asserted from wannier90's default.
# ---------------------------------------------------------------------------

W2 = pathlib.Path(__file__).resolve().parents[1] / "workflows/notebooks/runs/ctrl_NiO"


@pytest.mark.skipif(not (W2 / "m.HSX").exists(), reason="ctrl_NiO run absent")
def test_ws_distance_beats_the_plain_sum_off_mesh_on_a_multi_atom_cell():
    from waw.interfaces import siesta as sst
    from waw.core.hamiltonian import interpolate_bands

    H = sst.load_hamiltonian(W2 / "m.fdf")
    assert H.geometry.na > 1                      # inequivalent centres exist
    hr = sst.lowdin_hamiltonian(H, (6, 6, 6), spin=0, centres="atomic")
    off = np.random.default_rng(0).uniform(0, 1, (25, 3))
    exact = np.array([H.eigh(k, spin=0) for k in off])                  # eV
    ws = interpolate_bands(hr, off) * HARTREE_TO_EV
    plain = interpolate_bands(hr, off, ws=None) * HARTREE_TO_EV
    # high NAO virtuals are basis-tail artefacts and interpolate badly whatever
    # is done, so judge on the physically meaningful window
    m = exact < 5.0
    err_ws = np.abs(ws - exact)[m].mean()
    err_plain = np.abs(plain - exact)[m].mean()
    assert err_ws < err_plain, (f"ws_distance made it worse: {err_ws*1e3:.1f} vs "
                               f"{err_plain*1e3:.1f} meV")
    assert err_plain / err_ws > 1.2               # and by a real margin
