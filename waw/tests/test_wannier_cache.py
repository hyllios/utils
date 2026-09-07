"""save_wannier_result / load_wannier_result round-trip."""

from types import SimpleNamespace

import numpy as np
import torch

from waw import save_wannier_result, load_wannier_result
from waw.core.pipeline import WannierResult
from waw.core.hamiltonian import HamiltonianR, interpolate_bands


def _toy_result(nw=3, seed=0):
    rng = np.random.default_rng(seed)
    Rs = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                  dtype=np.int64)
    H = 0.4 * (rng.normal(size=(len(Rs), nw, nw))
               + 1j * rng.normal(size=(len(Rs), nw, nw)))
    # Hermiticity H(-R)=H(R)^H
    idx = {tuple(r): i for i, r in enumerate(Rs)}
    for i, r in enumerate(Rs):
        j = idx[tuple(-r)]
        H[i] = 0.5 * (H[i] + H[j].conj().T)
        H[j] = H[i].conj().T
    hr = HamiltonianR(H_R=torch.from_numpy(H), R_vectors=Rs,
                      degen=np.ones(len(Rs), np.int64), nw=nw)
    return WannierResult(
        spread=None, hr=hr, dis=SimpleNamespace(omega_i=12.34),
        wdata=None, spreads_bohr2=rng.uniform(0.5, 1.5, nw),
        omega_final=23.45, centres_bohr=rng.normal(size=(nw, 3)),
        m_tilde=None,
    )


def test_roundtrip_preserves_fields(tmp_path):
    r = _toy_result()
    p = save_wannier_result(r, tmp_path / "wann")
    assert p.exists() and p.suffix == ".npz"
    r2 = load_wannier_result(p)

    assert r2.hr.nw == r.hr.nw
    assert np.array_equal(r2.hr.R_vectors, r.hr.R_vectors)
    assert np.array_equal(r2.hr.degen, r.hr.degen)
    assert torch.allclose(r2.hr.H_R, r.hr.H_R)
    assert np.allclose(r2.centres_bohr, r.centres_bohr)
    assert np.allclose(r2.spreads_bohr2, r.spreads_bohr2)
    assert r2.omega_final == r.omega_final
    assert r2.dis.omega_i == r.dis.omega_i


def test_loaded_hr_interpolates_identically(tmp_path):
    r = _toy_result(seed=2)
    r2 = load_wannier_result(save_wannier_result(r, tmp_path / "w.npz"))
    kpts = np.random.default_rng(1).uniform(0, 1, (25, 3))
    # the cached model deliberately loses its lattice/mesh, so compare the
    # plain sum on both sides rather than let "auto" mean different things
    b1 = interpolate_bands(r.hr, kpts, ws=None)
    b2 = interpolate_bands(r2.hr, kpts, ws=None)
    assert np.allclose(b1, b2)


def test_none_centres_and_dis(tmp_path):
    r = _toy_result()
    r.centres_bohr = None
    r.dis = None
    r2 = load_wannier_result(save_wannier_result(r, tmp_path / "w2.npz"))
    assert r2.centres_bohr is None
    assert r2.dis is None


def test_gauge_roundtrip_rebuilds_W(tmp_path):
    """The gauge (dis.V + spread.U_final) must survive the round-trip: it is
    what `analysis.elph.wannier_transform_elph` needs to rotate the el-ph
    matrix element from the Bloch to the Wannier gauge (W = V @ U_final).
    Without it an el-ph notebook cannot resume from a cache."""
    rng = np.random.default_rng(7)
    nk, nb, nw = 6, 5, 3
    r = _toy_result(nw=nw, seed=3)
    V = torch.from_numpy(rng.normal(size=(nk, nb, nw))
                         + 1j * rng.normal(size=(nk, nb, nw)))
    U_final = torch.from_numpy(rng.normal(size=(nk, nw, nw))
                               + 1j * rng.normal(size=(nk, nw, nw)))
    r.spread = SimpleNamespace(U_final=U_final)
    r.dis = SimpleNamespace(omega_i=12.34, V=V)

    r2 = load_wannier_result(save_wannier_result(r, tmp_path / "gauge.npz"))
    assert r2.spread is not None and r2.dis.V is not None
    assert torch.allclose(r2.spread.U_final, U_final)
    assert torch.allclose(r2.dis.V, V)
    assert r2.dis.omega_i == 12.34
    W = torch.einsum("kbw,kwn->kbn", V, U_final)
    W2 = torch.einsum("kbw,kwn->kbn", r2.dis.V, r2.spread.U_final)
    assert torch.allclose(W, W2)


def test_gaugeless_cache_still_loads(tmp_path):
    """A cache written before the gauge was stored must still load (spread None),
    so callers can detect it and redo the minimization rather than crash."""
    r = _toy_result(seed=5)          # _toy_result has spread=None, dis without V
    r2 = load_wannier_result(save_wannier_result(r, tmp_path / "old.npz"))
    assert r2.spread is None
    assert r2.dis is not None and r2.dis.V is None
    assert r2.dis.omega_i == r.dis.omega_i
