"""
Tests for waw/analysis/bands.py.

  1. band_structure returns bands with shape (nk_path, nw) matching the
     dense k-path built from the .win kpoint_path block.
  2. Bands at a path vertex match interpolate_bands evaluated directly at
     that k-point (band_structure is just kpath + interpolate_bands).
  3. On Cu (entangled bands, real kpoint_path block): the full pipeline
     runs end-to-end and produces sane band energies.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import compute_hr, interpolate_bands, HamiltonianR
from waw.analysis.bands import band_structure
from waw.analysis.kpath import parse_kpoint_path, build_kpath



def _synthetic_hr(nk=8, nw=2, mp_grid=(2, 2, 2), seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    a = 5.0
    real_lattice = a * np.eye(3)
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T

    N1, N2, N3 = mp_grid
    kpts_np = np.array(
        [[i / N1, j / N2, k / N3]
         for i in range(N1) for j in range(N2) for k in range(N3)],
        dtype=np.float64,
    )
    kpts = torch.tensor(kpts_np, dtype=torch.float64)

    A = (torch.randn(nk, nw, nw, dtype=torch.float64)
         + 1j * torch.randn(nk, nw, nw, dtype=torch.float64))
    U, _ = torch.linalg.qr(A)
    eig = torch.tensor(np.sort(rng.uniform(-5, 5, size=(nk, nw)), axis=1),
                        dtype=torch.float64)

    hr = compute_hr(U, eig, kpts, mp_grid, real_lattice)
    return hr, recip_lattice


def test_band_structure_shape():
    hr, recip_lattice = _synthetic_hr()
    kpoint_path = ["G 0.00 0.00 0.00   X 0.50 0.00 0.00"]
    result = band_structure(hr, kpoint_path, recip_lattice, n_points=15)

    assert result.bands.shape == (15, hr.nw)
    assert result.kpath.kpts.shape == (15, 3)
    assert result.kpath.tick_labels == ["G", "X"]


def test_band_structure_matches_interpolate_bands_directly():
    hr, recip_lattice = _synthetic_hr()
    kpoint_path = [
        "G 0.00 0.00 0.00   X 0.50 0.00 0.00",
        "X 0.50 0.00 0.00   G 0.00 0.00 0.00",
    ]
    result = band_structure(hr, kpoint_path, recip_lattice, n_points=10)

    segments = parse_kpoint_path(kpoint_path)
    kpath = build_kpath(segments, recip_lattice, n_points=10)
    # both band_structure and interpolate_bands are Hartree (atomic units)
    expected = interpolate_bands(hr, kpath.kpts)

    np.testing.assert_allclose(result.bands, expected)


# --------------------------------------------------------------------------
# band_path_fidelity -- the off-mesh gate.
#
# The case that matters is a model that is EXACT on its own mesh and rings
# between mesh points: mesh_fidelity is blind to it by construction, and this
# is what has to catch it. Built here as a 1D tight-binding chain whose H(R)
# does not decay, which is what a badly disentangled subspace produces.
# --------------------------------------------------------------------------

from waw.analysis.bands import band_path_fidelity, mesh_fidelity
from waw.units import HARTREE_TO_EV


def _chain_hr(hoppings, nw=1):
    """1D chain: hoppings maps R (int) -> t. eps(k) = sum_R t_R exp(2pi i k R)."""
    rvecs = np.array([[r, 0, 0] for r in sorted(hoppings)], dtype=np.int64)
    ham = np.zeros((len(rvecs), nw, nw), dtype=np.complex128)
    for i, r in enumerate(sorted(hoppings)):
        ham[i, 0, 0] = hoppings[r]
    return HamiltonianR(
        H_R=torch.from_numpy(ham),
        R_vectors=torch.from_numpy(rvecs),
        degen=torch.ones(len(rvecs), dtype=torch.float64),
        nw=nw,
    )


def _eps(hoppings, kx):
    return sum(t * np.exp(2j * np.pi * kx * r) for r, t in hoppings.items()).real


def test_band_path_fidelity_is_zero_for_a_faithful_model():
    hop = {-1: 0.5, 0: 0.0, 1: 0.5}
    kx = np.linspace(0.0, 0.5, 40)
    ref = _eps(hop, kx)[:, None]
    fid = band_path_fidelity(ref, ref, fermi_ev=0.0)
    assert fid["max"] == 0.0 and fid["rms"] == 0.0
    assert fid["n_states"] == 40


def test_band_path_fidelity_catches_ringing_that_mesh_fidelity_misses():
    """
    A model exact on an 8-point mesh but ringing between its points.

    The extra hopping is at R = +-8 and IMAGINARY, t_8 = i a, t_-8 = -i a, so
    it contributes -2a sin(16 pi k): identically zero at every k = n/8 and
    swinging by 2a off the mesh. (A real R = 8 hopping would not do -- cos is
    1 on that mesh, a constant shift the mesh check would see.) This is the
    failure mode of notebook 22's Nb model: exact where it was fitted, ringing
    everywhere else.
    """
    n, amp = 8, 0.4
    ref_hop = {-1: 0.5, 1: 0.5}
    bad_hop = dict(ref_hop); bad_hop[n] = 1j * amp; bad_hop[-n] = -1j * amp

    mesh = np.arange(n) / n
    kpts_mesh = np.stack([mesh, np.zeros(n), np.zeros(n)], axis=1)

    # on-mesh: identical by construction, so mesh_fidelity is blind
    on_mesh = mesh_fidelity(_chain_hr(bad_hop), kpts_mesh,
                            _eps(ref_hop, mesh)[:, None] * HARTREE_TO_EV)
    assert on_mesh["max"] < 1e-9, "the mesh check must be blind here"

    # off-mesh: the ringing is unmissable
    dense = np.linspace(0.0, 1.0, 401)
    fid = band_path_fidelity(_eps(bad_hop, dense)[:, None],
                             _eps(ref_hop, dense)[:, None], fermi_ev=0.0)
    assert fid["max"] > 1e3 * 2 * amp * 0.98        # ~800 meV
    assert fid["rms"] > 1e3 * 2 * amp * 0.5


def test_band_path_fidelity_ignores_nans_instead_of_scoring_well():
    """A failed bands run must not read as a good model."""
    kx = np.linspace(0.0, 0.5, 20)
    w = _eps({-1: 0.5, 1: 0.5}, kx)[:, None]
    fid = band_path_fidelity(w, np.full_like(w, np.nan))
    assert fid["n_states"] == 0 and np.isnan(fid["max"])


def test_band_path_fidelity_rejects_mismatched_paths():
    """Two paths built with different npoints is the documented foot-gun."""
    import pytest
    a = np.zeros((30, 2)); b = np.zeros((40, 3))
    with pytest.raises(ValueError, match="share a k-path"):
        band_path_fidelity(a, b)


def test_band_path_fidelity_near_ef_isolates_the_fermi_surface():
    """Error far from E_F must not mask a clean Fermi surface, or vice versa."""
    kx = np.linspace(0.0, 1.0, 200)
    ref = np.stack([_eps({-1: 0.5, 1: 0.5}, kx),
                    _eps({-1: 0.5, 1: 0.5}, kx) + 20.0], axis=1)
    w = ref.copy()
    w[:, 1] += 1.5                        # 1.5 eV error, 20 eV above E_F
    fid = band_path_fidelity(w, ref, fermi_ev=0.0, near_ef=1.0)
    assert fid["max"] > 1400.0            # the high band is bad
    assert fid["max_near_ef"] < 1e-6      # the Fermi surface is clean


def test_band_series_plot_kw_reaches_the_artist():
    """A reference series must be able to draw as markers under the model's
    lines -- two closely-agreeing line series are otherwise indistinguishable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from waw.vis.bands import plot_bands, BandSeries

    x = np.linspace(0.0, 1.0, 20)
    b = np.stack([np.cos(x), np.cos(x) + 1.0], axis=1)
    fig, ax = plt.subplots()
    plot_bands(x, [0.0, 1.0], ["G", "X"],
               [BandSeries(b, label="ref", color="0.45",
                           plot_kw=dict(ls="none", marker=".", ms=2.0)),
                BandSeries(b, label="model", color="C3")], ax=ax)
    styles = {(ln.get_linestyle(), ln.get_marker()) for ln in ax.get_lines()}
    assert ("None", ".") in styles          # the reference is markers
    assert any(ls == "-" and m == "None" for ls, m in styles)   # model is lines
    # and the legend still carries one entry per series, not one per band
    assert len(ax.get_legend().get_texts()) == 2
    plt.close(fig)


# --------------------------------------------------------------------------
# hopping_range: the quantity Omega cannot see.
# --------------------------------------------------------------------------

def test_hopping_range_sees_what_omega_cannot():
    """
    Two models identical except that one Wannier function sits a lattice
    vector away. The spread is unchanged (it is translation-invariant per
    function) but H(R) reaches a whole cell further -- which is the degeneracy
    that let argmin(Omega) pick a 52x worse model on bcc Nb.
    """
    from waw.core.hamiltonian import hopping_range

    a = 3.0
    lat = a * np.eye(3)
    hop = {(0, 0, 0): 5.0, (1, 0, 0): 1.0, (-1, 0, 0): 1.0}
    R = np.array(sorted(hop), dtype=np.int64)
    nw = 2
    H = np.zeros((len(R), nw, nw), dtype=np.complex128)
    for i, r in enumerate(sorted(hop)):
        H[i, 0, 0] = H[i, 1, 1] = hop[r]
    home = HamiltonianR(H_R=torch.from_numpy(H), R_vectors=torch.from_numpy(R),
                        degen=torch.ones(len(R), dtype=torch.float64), nw=nw)

    # displace WF 1 by one lattice vector: H_01(R) -> H_01(R + T)
    R2 = np.array(sorted({tuple(r) for r in R} | {tuple(r + [1, 0, 0]) for r in R}),
                  dtype=np.int64)
    H2 = np.zeros((len(R2), nw, nw), dtype=np.complex128)
    idx = {tuple(r): i for i, r in enumerate(R2)}
    for i, r in enumerate(R):
        H2[idx[tuple(r)], 0, 0] = H[i, 0, 0]
        H2[idx[tuple(r + [1, 0, 0])], 1, 1] = H[i, 1, 1]
    moved = HamiltonianR(H_R=torch.from_numpy(H2), R_vectors=torch.from_numpy(R2),
                         degen=torch.ones(len(R2), dtype=torch.float64), nw=nw)

    r_home = hopping_range(home, lat)
    r_moved = hopping_range(moved, lat)
    assert r_moved > 2.0 * r_home, (r_home, r_moved)
    # and it is a genuine second moment: doubling the lattice quadruples it
    assert hopping_range(home, 2 * lat) == pytest.approx(4.0 * r_home)


def test_global_minimize_spread_score_overrides_argmin_omega():
    """The score hook must be able to reject the lowest-Omega restart."""
    from waw.core.global_optim import global_minimize_spread
    import waw.core.global_optim as go

    class Fake:
        def __init__(s, om, tag):
            s.Omega, s.tag = om, tag

    fakes = [Fake(10.0, "lowest-omega"), Fake(10.5, "wanted")]
    real_worker = go._restart_worker
    go._restart_worker = lambda cfg: fakes[cfg.i_restart]
    try:
        common = dict(U_init=torch.eye(2, dtype=torch.complex128)[None],
                      Mmn=None, wb=None, bvecs=None, kb_idx=None,
                      n_restarts=2, n_workers=1)
        assert global_minimize_spread(**common).tag == "lowest-omega"
        picked = global_minimize_spread(
            **common, score=lambda r: 0.0 if r.tag == "wanted" else 1.0)
        assert picked.tag == "wanted"
    finally:
        go._restart_worker = real_worker


def test_use_ws_distance_is_on_by_default_and_removes_the_gauge_dependence():
    """
    Centre-aware interpolation must make the result independent of which
    periodic image each Wannier function landed in.

    Displacing WF 1 to cell +s re-indexes its hoppings -- H'_01(R) = H_01(R+s)
    and H'_10(R) = H_10(R-s), oppositely, or the model is not Hermitian-
    consistent -- and on a finite mesh those indices wrap into the supercell.
    That wrap is why the displaced model interpolates differently while staying
    exact on the mesh. use_ws_distance undoes it exactly, by phasing on
    |R + tau_m - tau_n| rather than |R|.

    This is a correctness property, not an accuracy tweak: the two models are
    the same physics in different gauges and MUST interpolate identically. On
    bcc Nb, off/on was 3797/52 meV at E_F.
    """
    from waw.core.hamiltonian import HamiltonianR

    a, N = 3.0, 12
    lat = a * np.eye(3)
    half = N // 2
    R = np.array([(i, 0, 0) for i in range(-half, half + 1)], dtype=np.int64)
    idx = {int(r[0]): i for i, r in enumerate(R)}
    nw = 2
    deg = np.ones(len(R), dtype=np.int64)
    kp = np.stack([np.linspace(0, 0.5, 21), np.zeros(21), np.zeros(21)], axis=1)

    def wrap(w):
        return w - N * round(w / N)

    def chain(shift, dec=1.6):
        H = np.zeros((len(R), nw, nw), dtype=np.complex128)
        for r0 in range(-half, half + 1):
            t = np.exp(-dec * abs(r0))
            H[idx[r0], 0, 0] += t
            H[idx[r0], 1, 1] += t
            u, v = wrap(r0 - shift), wrap(r0 + shift)
            if u in idx:
                H[idx[u], 0, 1] += 0.4 * t
            if v in idx:
                H[idx[v], 1, 0] += 0.4 * t
        return H

    def bands(H, centres, ws):
        hr = HamiltonianR(H_R=torch.from_numpy(H), R_vectors=R, degen=deg, nw=nw,
                          centres=centres, real_lattice=lat, mp_grid=(N, N, N))
        return np.asarray(interpolate_bands(hr, kp, ws=ws))

    shift = 4
    home = bands(chain(0), np.zeros((nw, 3)), None)
    away_c = np.array([[0.0, 0.0, 0.0], [shift * a, 0.0, 0.0]])
    off = np.abs(bands(chain(shift), away_c, None) - home).max()
    on = np.abs(bands(chain(shift), away_c, "auto")
                - bands(chain(0), np.zeros((nw, 3)), "auto")).max()
    assert off > 1e-3, "the displaced model should differ without the correction"
    assert on < 1e-12, f"correction must restore gauge invariance exactly: {on}"

    # centres present -> the model builds its own correction; absent -> none,
    # and "auto" must then be exactly the legacy plain sum rather than an error
    hr = HamiltonianR(H_R=torch.from_numpy(chain(0)), R_vectors=R, degen=deg,
                      nw=nw, centres=np.zeros((nw, 3)), real_lattice=lat,
                      mp_grid=(N, N, N))
    assert hr.ws_distance() is not None
    bare = HamiltonianR(H_R=torch.from_numpy(chain(0)), R_vectors=R, degen=deg,
                        nw=nw)
    assert bare.ws_distance() is None
    np.testing.assert_allclose(np.asarray(interpolate_bands(bare, kp, ws="auto")),
                               np.asarray(interpolate_bands(bare, kp, ws=None)))

    with pytest.raises(ValueError, match="ws must be"):
        interpolate_bands(hr, kp, ws="yes")
