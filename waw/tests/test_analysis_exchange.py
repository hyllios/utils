"""LKAG Heisenberg exchange (waw.analysis.exchange).

The TB2J-parity test needs the Fe reference data in /tmp/fe_tb2j_col
(fe_{up,dn}_hr.dat exported from waw + TB2J_d21/exchange.out produced by
wann2J.py on those exact files) and is skipped when absent.  On identical
input and with TB2J's conventions replicated (degen ignored, band-column
state truncation at E_F + 5.1 eV, CFR nz=100 at 600 K) the two codes
agree to 5e-8 eV.
"""
import pathlib
import re

import numpy as np
import pytest
import torch

from waw.analysis.exchange import cfr_contour, heisenberg_exchange, KT_600K
from waw.core.distributions import fermi_dirac
from waw.core.hamiltonian import HamiltonianR
from waw.units import EV_TO_HARTREE, HARTREE_TO_EV

FE_DIR = pathlib.Path("/tmp/fe_tb2j_col")


def test_cfr_contour_reproduces_fermi_occupation():
    """Plain CFR gives f(e0) - 1/2; the constant half-moment lives on the
    omitted z -> i*inf arc and cancels in exchange integrands."""
    kT = KT_600K
    z, w = cfr_contour(100, kT)
    for e0_ev in (-2.0, -0.2, 0.05, 0.3):
        e0 = e0_ev * EV_TO_HARTREE
        val = np.imag(-np.pi / 2.0 * np.sum(w / (z - e0)))
        occ = -val / np.pi + 0.5
        ref = fermi_dirac(np.array([e0]), 0.0, kT)[0]
        assert abs(occ - ref) < 1e-8, (e0_ev, occ, ref)


def _random_channel(rng, nw, shift=0.0):
    R_half = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    R, H = [(0, 0, 0)], []
    H0 = rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
    H0 = 0.1 * (H0 + H0.conj().T)
    H0 += np.diag(np.arange(nw) * 0.3 + shift)
    H.append(H0)
    for Rv in R_half:
        h = 0.05 * (rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw)))
        R += [Rv, tuple(-x for x in Rv)]
        H += [h, h.conj().T]
    R = np.array(R, dtype=np.int64)
    return HamiltonianR(H_R=torch.tensor(np.array(H), dtype=torch.complex128),
                        R_vectors=R, degen=np.ones(len(R), dtype=np.int64), nw=nw)


def test_exchange_pair_symmetry():
    """J_ij(R) == J_ji(-R) exactly, on an asymmetric synthetic model."""
    rng = np.random.default_rng(7)
    hr_up = _random_channel(rng, 4, shift=-0.05)
    hr_dn = _random_channel(rng, 4, shift=+0.05)
    n = 6
    g = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    kpts = np.stack([x.ravel() for x in g], axis=1) / n
    R_list = np.array([(1, 0, 0), (-1, 0, 0), (1, 1, 0), (-1, -1, 0)])
    res = heisenberg_exchange(hr_up, hr_dn, kpts, efermi=0.3,
                              orbital_groups=[[0, 1], [2, 3]],
                              R_list=R_list, nz=40)
    for R in ((1, 0, 0), (1, 1, 0)):
        Rm = tuple(-x for x in R)
        assert res.J[(R, 0, 1)] == pytest.approx(res.J[(Rm, 1, 0)], abs=1e-12)
        assert res.J[(R, 1, 0)] == pytest.approx(res.J[(Rm, 0, 1)], abs=1e-12)


@pytest.mark.skipif(not FE_DIR.exists(), reason="Fe TB2J reference data absent")
def test_fe_tb2j_parity():
    from waw.interfaces.wannier90.io import read_hr

    hrs = {}
    for ch in ("up", "dn"):
        d = read_hr(FE_DIR / f"fe_{ch}_hr.dat")
        hrs[ch] = HamiltonianR(
            H_R=torch.tensor(d["H_R"] * EV_TO_HARTREE),
            R_vectors=d["R_vectors"],
            # TB2J ignores the WS degeneracy column of hr.dat; replicate
            degen=np.ones_like(d["degen"]),
            nw=d["nw"],
        )
    ef = 17.6223 * EV_TO_HARTREE

    ref = {}
    txt = (FE_DIR / "TB2J_d21/exchange.out").read_text()
    for m in re.finditer(
            r"Fe1\s+Fe1\s+\(\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s+(-?[\d.]+)", txt):
        ref[(int(m[1]), int(m[2]), int(m[3]))] = float(m[4])
    assert len(ref) > 100

    n = 21
    g = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    kpts = np.stack([x.ravel() for x in g], axis=1) / n
    kpts -= np.round(kpts)

    R_list = np.array(sorted(ref.keys()))
    res = heisenberg_exchange(
        hrs["up"], hrs["dn"], kpts, ef,
        orbital_groups=[[4, 5, 6, 7, 8]], R_list=R_list, nz=100,
        state_window=(-1e9, ef + 5.1 * EV_TO_HARTREE))

    assert res.magmoms[0] == pytest.approx(2.4459, abs=2e-4)
    for R, j_ref in ref.items():
        j = res.J[(R, 0, 0)] * HARTREE_TO_EV * 1e3
        assert j == pytest.approx(j_ref, abs=1e-3), R


@pytest.mark.skipif(
    not pathlib.Path(__file__).resolve().parents[1]
        .joinpath("workflows/notebooks/runs/fe_siesta/fe.HSX").exists(),
    reason="fe_siesta run absent")
def test_siesta_nonortho_vs_siesta2j():
    """Non-orthogonal LKAG (G=(zS-H)^-1) on the raw SIESTA NAO H/S vs
    TB2J's siesta2J on identical data. With TB2J's conventions replicated
    (its own Gaussian-occupation E_F, band truncation at E_F+5.1 eV) the
    shells agree to <1.5 meV; the residual is Fermi-level placement --
    metallic LKAG here has dJ1/dE_F ~ 155 meV per eV, so a +-10 meV E_F
    difference moves J1 by +-1.5 meV. The Löwdin-orthogonal route is a
    DIFFERENT effective-J definition (Fe all-orbital J1 ~ 3.5 vs ~10):
    compare like with like."""
    sisl = pytest.importorskip("sisl")
    import re as _re

    from ase.build import bulk

    from waw.analysis.exchange import heisenberg_exchange_nonortho
    from waw.interfaces import siesta as sst

    W = pathlib.Path(__file__).resolve().parents[1] / "workflows/notebooks/runs/fe_siesta"
    H = sst.load_hamiltonian(W / "fe.fdf")
    n = 12
    g = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    kpts = np.stack([x.ravel() for x in g], axis=1) / n
    kpts -= np.round(kpts)
    nk, nb = len(kpts), H.no
    Hk = {s: np.zeros((nk, nb, nb), complex) for s in (0, 1)}
    Sk = np.zeros((nk, nb, nb), complex)
    for ik, k in enumerate(kpts):
        for s in (0, 1):
            Hk[s][ik] = np.asarray(H.Hk(k, format="array", spin=s)) * EV_TO_HARTREE
        Sk[ik] = np.asarray(H.Sk(k, format="array"))

    txt = (W / "TB2J_siesta/exchange.out").read_text()
    ref = {}
    for m in _re.finditer(r"Fe1\s+Fe1\s+\(\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
                          r"\s+(-?[\d.]+)\s+\([^)]+\)\s+([\d.]+)", txt):
        ref[(int(m[1]), int(m[2]), int(m[3]))] = (float(m[4]), float(m[5]))
    R_list = np.array(sorted(ref.keys()))

    EF_TB = 0.0114 * EV_TO_HARTREE     # TB2J's own Gaussian-occupation E_F
    res = heisenberg_exchange_nonortho(
        Hk[0], Hk[1], Sk, kpts, EF_TB, [list(range(nb))], R_list, nz=100,
        state_window=(-1e9, EF_TB + 5.1 * EV_TO_HARTREE))
    assert res.magmoms[0] == pytest.approx(2.31, abs=0.05)
    cell = bulk("Fe", "bcc", a=2.8699).get_cell()[:]
    shells = {}
    for R, (jr, d) in ref.items():
        j = res.J[(tuple(R), 0, 0)] * HARTREE_TO_EV * 1e3
        shells.setdefault(round(d, 3), []).append((j, jr))
    for d in sorted(shells)[:3]:
        ours = np.mean([x[0] for x in shells[d]])
        theirs = np.mean([x[1] for x in shells[d]])
        assert abs(ours - theirs) < 1.5, (d, ours, theirs)
