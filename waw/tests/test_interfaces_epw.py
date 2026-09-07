"""Cross-code test: waw machinery on EPW's own Wannier-space data.

Skips unless a real EPW run directory is present (400 MB of epmatwp is not
fixture material). The pinned numbers are EPW 5.8.1's OWN stdout values from
the Al 12^3 phonselfen reference run (/tmp/al_epw_test), so every assertion
here is a genuine two-code agreement, not a self-consistency check.
"""

from pathlib import Path

import numpy as np
import pytest

EPW_DIR = Path("/tmp/al_epw_test/epw_12x12x12")

pytestmark = pytest.mark.skipif(
    not (EPW_DIR / "epwdata.fmt").exists(),
    reason="EPW Al reference run not present",
)


@pytest.fixture(scope="module")
def model():
    from waw.interfaces.epw import load_epw_model
    return load_epw_model(EPW_DIR, (12, 12, 12), (6, 6, 6))


def test_wigner_seitz_counts_match_epwdata(model):
    # load_epw_model already asserts them; reaching here IS the test, but
    # keep the numbers visible for the Al reference
    assert len(model["R_e"]) == 1957
    assert len(model["R_q"]) == 279


def test_epw_fine_mesh_fermi_level_and_dos(model):
    """epw_fermi_level + MP1 dos reproduce EPW's printed fine-mesh values:
    'Fermi energy is calculated from the fine k-mesh: Ef = 7.777617 eV' and
    'DOS = 0.226032 states/spin/eV/Unit Cell'."""
    from waw.analysis import elph
    from waw.interfaces.ase.structure import monkhorst_pack
    from waw.units import EV_TO_HARTREE, HARTREE_TO_EV

    eig, _ = elph.band_eigensystem(model["hr"], monkhorst_pack((12, 12, 12)))
    degauss = 0.05 * EV_TO_HARTREE
    ef = elph.epw_fermi_level(eig, 3.0, degauss, ngauss=1)
    assert ef * HARTREE_TO_EV == pytest.approx(7.777617, abs=1e-5)
    dos = elph.fermi_surface_dos(eig, ef, elph.epw_degauss_to_sigma(degauss),
                                 ngauss=1)
    assert dos * EV_TO_HARTREE == pytest.approx(0.226032, abs=2e-5)


def test_epw_phonons_from_rdw(model):
    """rdw interpolation reproduces EPW's own fine-mesh phonon maximum
    (1.1 * max = its a2f grid top, dw = 0.0984389 meV in the .a2f file)."""
    from waw.interfaces.ase.structure import monkhorst_pack
    from waw.units import HARTREE_TO_EV

    om, ev = model["phonon_fn"](monkhorst_pack((12, 12, 12)))
    assert om.max() * HARTREE_TO_EV * 1e3 == pytest.approx(44.7450, abs=2e-3)
    # eigenvector unitarity
    eye = np.einsum("qim,qin->qmn", ev.conj(), ev)
    assert np.abs(eye - np.eye(ev.shape[-1])).max() < 1e-10


def test_lambda_matches_epw_on_a_q_subset(model):
    """Mode-resolved lambda_qnu against EPW's stdout 'lambda___' lines for a
    handful of q, using ONLY EPW's own data -- the machinery-parity check.
    Full-mesh agreement (1728 q): lambda_sum 0.4838 double-delta / 0.0003188
    occupation-difference, both matching EPW (see the 2026-07-27 parity
    scripts); here a subset keeps the runtime test-suite friendly."""
    from waw.analysis import elph
    from waw.interfaces.ase.structure import monkhorst_pack
    from waw.units import EV_TO_HARTREE

    # EPW stdout, run rerun_dd_true (delta_approx=.true., degaussw=0.05 eV,
    # fsthick=6 eV): per-mode lambda summed at two representative q
    refs = {
        (6, 6, 6): 0.000022,                            # L point, FS-disjoint
        (3, 0, 3): 6.752877,                            # [0.25, 0, 0.25]
    }
    k = monkhorst_pack((12, 12, 12))
    eig, U = elph.band_eigensystem(model["hr"], k)
    degauss = 0.05 * EV_TO_HARTREE
    ef = elph.epw_fermi_level(eig, 3.0, degauss, ngauss=1)
    sig = elph.epw_degauss_to_sigma(degauss)
    dos = elph.fermi_surface_dos(eig, ef, sig, ngauss=1)
    q = np.array([[i / 12, j / 12, l / 12] for (i, j, l) in refs])
    om, ev = model["phonon_fn"](q)
    og = np.linspace(1e-6, 0.002, 60)
    _, lam_qnu = elph.alpha2f(
        eig, U, model["g_R"], model["R_e"], model["degen_e"], model["R_q"],
        model["degen_q"], k, q, None, om, ev,
        model["crystal"]["masses_amu"], model["crystal"]["types"],
        fermi_energy=ef, dos_at_ef=dos, omega_grid=og, sigma_e=sig,
        hr=model["hr"], fsthick=6.0 * EV_TO_HARTREE, return_qnu=True)
    for (idx, ref), got in zip(refs.items(), lam_qnu.sum(axis=1)):
        assert got == pytest.approx(ref, rel=2e-3, abs=5e-5), (idx, ref, got)
