"""LSWT magnon bands (waw.analysis.magnon) vs TB2J's magnon module.

The parity tests need the TB2J pickles archived under
workflows/notebooks/runs/{fe,nii2}_exchange/tb2j*/ and TB2J installed;
skipped when absent.  TB2J regularizes its Cholesky with +1e-6 eV, so
away-from-Gamma agreement is asserted at 2e-3 meV (its jitter floor),
and Gamma Goldstone modes are asserted natively (~0) rather than against
TB2J's shift artifact.
"""
import pathlib

import numpy as np
import pytest

from waw.analysis.magnon import magnon_bands
from waw.units import EV_TO_HARTREE, HARTREE_TO_EV

RUNS = pathlib.Path(__file__).resolve().parents[1] / "workflows/notebooks/runs"
FE_T = RUNS / "fe_exchange/tb2j/TB2J_d21"
NI_T = RUNS / "nii2_exchange/tb2j/TB2Jv2_downfolded"

tb2j = pytest.importorskip("TB2J.magnon.magnon3")


def _reference(path, qpts):
    from TB2J.io_exchange import SpinIO

    exc = SpinIO.load_pickle(path=str(path), fname="TB2J.pickle")
    mag = tb2j.Magnon.from_TB2J_results(path=str(path))
    mm = np.atleast_2d(np.asarray(mag.magmom, dtype=float))
    if mm.shape[1] != 3:
        m3 = np.zeros((mm.size, 3))
        m3[:, 2] = mm.ravel()
        mm = m3
    mag.magmom = mm
    mag.Snorm = np.linalg.norm(mm, axis=-1) / 2  # upstream Magnon skips this
    ref = mag._magnon_energies(qpts) * 1e3
    J = {(tuple(R), i, j): v * EV_TO_HARTREE
         for (R, i, j), v in exc.exchange_Jdict.items()}
    return J, mm, ref


def test_ferromagnet_analytic_single_site():
    """1D FM chain, one J: omega(q) = 4S J~ (1 - cos 2 pi q) with
    J~ = J/S^2 * S = J/S (TB2J normalization chain)."""
    Jval = 1e-3
    J = {((1, 0, 0), 0, 0): Jval, ((-1, 0, 0), 0, 0): Jval}
    m = np.array([2.0])          # S = 1
    q = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    w = magnon_bands(J, m, q)
    S = 1.0
    ana = 4 * (Jval / S) * (1 - np.cos(2 * np.pi * q[:, 0]))
    assert np.allclose(w[:, 0], ana, atol=5e-9)   # Cholesky jitter 1e-9 Ha


@pytest.mark.skipif(not (FE_T / "TB2J.pickle").exists(), reason="Fe pickle absent")
def test_fe_matches_tb2j():
    rng = np.random.default_rng(3)
    qpts = np.vstack([np.zeros(3), rng.uniform(-0.5, 0.5, (6, 3))])
    J, mm, ref = _reference(FE_T, qpts)
    nat = magnon_bands(J, mm, qpts) * HARTREE_TO_EV * 1e3
    assert abs(nat[0, 0]) < 1e-4                       # Goldstone at Gamma
    assert np.abs(nat[1:] - ref[1:]).max() < 2e-3      # TB2J jitter floor


@pytest.mark.skipif(not (NI_T / "TB2J.pickle").exists(), reason="NiI2 pickle absent")
def test_nii2_afm_matches_tb2j():
    rng = np.random.default_rng(3)
    qpts = np.vstack([np.zeros(3), rng.uniform(-0.5, 0.5, (6, 3))])
    J, mm, ref = _reference(NI_T, qpts)
    nat = magnon_bands(J, mm, qpts) * HARTREE_TO_EV * 1e3
    assert nat.shape == (7, 2)
    assert np.abs(nat[0]).max() < 0.1                  # AFM Goldstone pair
    stable = ref[1:].min(axis=1) > 0
    assert np.abs(nat[1:][stable] - ref[1:][stable]).max() < 2e-2
    # unstable (omega<0) qs: both codes fall back to shifted Cholesky --
    # paths differ in jitter, so only sign/rough agreement is meaningful
    assert np.abs(nat[1:][~stable] - ref[1:][~stable]).max() < 0.5


# --------------------------------------------------------------------------
# downfolding weakly polarized sublattices.
#
# A magnon energy goes as J/S and the LSWT matrix elements as
# J_ij/sqrt(S_i S_j), so a nearly-zero moment produces a huge flat branch
# rather than a small correction. `downfold` eliminates such sites
# adiabatically. For a collinear ferromagnet the anomalous block vanishes and
# the elimination is an ordinary Schur complement, so it must reproduce the
# kept branches of the full calculation exactly in the well-separated limit.
# --------------------------------------------------------------------------

def _two_sublattice(j_mm, j_ml, j_ll, m_strong, m_weak):
    """FM chain-like model: strong sublattice 0, weak sublattice 1."""
    J = {}
    for R in ((1, 0, 0), (-1, 0, 0)):
        J[(R, 0, 0)] = j_mm
        J[(R, 1, 1)] = j_ll
    for R in ((0, 0, 0), (1, 0, 0), (-1, 0, 0)):
        J[(R, 0, 1)] = j_ml
        J[(R, 1, 0)] = j_ml
    return J, np.array([m_strong, m_weak])


def test_downfold_reproduces_the_acoustic_branch():
    """The downfolded branch must lie on the full model's kept branch."""
    J, mm = _two_sublattice(1e-4, 3e-5, 1e-6, 3.5, 0.04)
    q = np.stack([np.linspace(0, 0.5, 25), np.zeros(25), np.zeros(25)], axis=1)
    full = magnon_bands(J, mm, q)
    down = magnon_bands(J, mm, q, downfold=[1])
    assert full.shape == (25, 2) and down.shape == (25, 1)
    # the weak sublattice's branch sits far above the strong one; the
    # adiabatic elimination is exact as that ratio grows, so the separation
    # is the precondition and the agreement is the claim
    assert full[:, 1].min() > 10 * max(full[:, 0].max(), 1e-12)
    assert np.abs(down[:, 0] - full[:, 0]).max() < 5e-3 * np.ptp(full[:, 0])


def test_downfold_is_not_the_same_as_omitting_the_sublattice():
    """Downfolding keeps the mediated path; deleting the site throws it away."""
    J, mm = _two_sublattice(1e-4, 8e-5, 1e-6, 3.5, 0.04)
    q = np.stack([np.linspace(0, 0.5, 25), np.zeros(25), np.zeros(25)], axis=1)
    down = magnon_bands(J, mm, q, downfold=[1])
    bare = magnon_bands({k: v for k, v in J.items() if k[1] == 0 == k[2]},
                        mm[:1], q)
    assert np.ptp(down[:, 0]) > 1.05 * np.ptp(bare[:, 0])


def test_downfold_rejects_bad_input():
    J, mm = _two_sublattice(1e-4, 3e-5, 1e-6, 3.5, 0.04)
    q = np.zeros((1, 3))
    with pytest.raises(ValueError, match="cannot downfold every"):
        magnon_bands(J, mm, q, downfold=[0, 1])
    with pytest.raises(ValueError, match="must lie in"):
        magnon_bands(J, mm, q, downfold=[5])


def test_downfold_none_is_the_unchanged_calculation():
    J, mm = _two_sublattice(1e-4, 3e-5, 1e-6, 3.5, 0.04)
    q = np.stack([np.linspace(0, 0.5, 9), np.zeros(9), np.zeros(9)], axis=1)
    np.testing.assert_allclose(magnon_bands(J, mm, q),
                               magnon_bands(J, mm, q, downfold=[]))


# --------------------------------------------------------------------------
# Magnetic ground state from the LSWT stability condition
# (Romero et al., npj Comput. Mater.)
# --------------------------------------------------------------------------

def _chain(j1, j2=0.0):
    J = {}
    for R in ((1, 0, 0), (-1, 0, 0)):
        J[(R, 0, 0)] = j1
    if j2:
        for R in ((2, 0, 0), (-2, 0, 0)):
            J[(R, 0, 0)] = j2
    return J


def test_propagation_vector_ferromagnet_is_stable():
    from waw.analysis.magnon import propagation_vector
    r = propagation_vector(_chain(+1e-4), np.array([2.0]), mesh=(16, 1, 1))
    assert r["stable"]
    np.testing.assert_allclose(r["q"], 0.0, atol=1e-4)


def test_propagation_vector_finds_the_antiferromagnet():
    from waw.analysis.magnon import propagation_vector
    r = propagation_vector(_chain(-1e-4), np.array([2.0]), mesh=(16, 1, 1))
    assert not r["stable"]
    assert abs(abs(r["q"][0]) - 0.5) < 1e-3
    assert r["lambda_min"] < r["lambda_at_0"]      # the mode is cheaper


def test_propagation_vector_finds_a_spiral_from_frustration():
    """J1 FM with J2 AFM strong enough: the classical spiral has
    cos(2 pi q) = -J1/(4 J2) -- note J2 SIGNED, not |J2| -- and it is
    incommensurate, the case where a commensurate cell can only ever be an
    approximant. Also cross-checks the LSWT route against the classical
    spiral energy: argmin lambda(q) must equal argmax J(q)."""
    from waw.analysis.magnon import propagation_vector, commensurate_supercell
    j1, j2 = 1e-4, -1e-4                       # |J2/J1| = 1 > 1/4
    r = propagation_vector(_chain(j1, j2), np.array([2.0]), mesh=(64, 1, 1))
    q_exact = np.arccos(-j1 / (4 * j2)) / (2 * np.pi)      # 0.2098
    assert not r["stable"]
    assert abs(abs(r["q"][0]) - q_exact) < 5e-3, (r["q"][0], q_exact)
    # the same q maximises the classical J(q)
    qs = np.linspace(0, 0.5, 2001)
    Jq = 2 * j1 * np.cos(2 * np.pi * qs) + 2 * j2 * np.cos(4 * np.pi * qs)
    assert abs(qs[np.argmax(Jq)] - abs(r["q"][0])) < 5e-3
    c = commensurate_supercell(r["q"], max_denominator=4)
    assert not c["exact"]                      # honest about being approximate


def test_commensurate_supercell_signs():
    from waw.analysis.magnon import commensurate_supercell
    c = commensurate_supercell([0.5, 0.0, 0.0])
    assert list(c["supercell"]) == [2, 1, 1]
    assert sorted(c["signs"].tolist()) == [-1, 1]
    assert c["exact"]
    # fcc type-I ordering doubles two axes and gives 2 up / 2 down
    c = commensurate_supercell([0.5, 0.5, 0.0])
    assert list(c["supercell"]) == [2, 2, 1]
    assert c["signs"].sum() == 0 and c["exact"]
