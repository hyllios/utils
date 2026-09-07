"""
Tests for waw/window.py — automatic energy window selection.

  1. Return type is WindowResult.
  2. outer_window is a 2-tuple of floats with E_min < E_max.
  3. At every k-point, ≥ nw bands fall inside the outer window.
  4. frozen_window is None or a valid (E_min, E_max) inside outer_window.
  5. At every k-point, ≤ nw bands fall inside the frozen window.
  6. proj shape is (nk, nb) and values are in [0, 1].
  7. threshold=0 includes all bands in the outer window.
  8. High frozen_threshold (> 1.0) gives frozen_window=None.
  9. n_outer_bands ≥ nw.
 10. Synthetic: target bands with p≈1 are correctly selected as outer/frozen.
 11. Synthetic entangled: non-target bands with p≈0 are excluded from frozen.
 12. On Si data: all 4 valence bands end up inside the outer window.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.window import auto_window, WindowResult
from waw.core.types import WannierData

SI_DIR = Path(__file__).parent / "data" / "silicon"
HAS_SI = (SI_DIR / "silicon.mmn").exists()


# ===========================================================================
# Helper: build a minimal WannierData with controlled Amn
# ===========================================================================

def _make_wdata(
    nk: int, nb: int, nw: int,
    proj_target: float = 0.9,
    proj_noise:  float = 0.05,
    seed:        int   = 0,
) -> WannierData:
    """
    Synthetic WannierData where the first nw bands have high Amn overlap
    (projectability ≈ proj_target) and the remaining nb-nw bands have
    near-zero overlap (projectability ≈ 0).

    eig is a linearly spaced band structure from -5 to +5 eV;
    target bands occupy the lower nw slots.
    """
    rng = np.random.default_rng(seed)

    # Eigenvalues: nw target bands below 0 eV, rest above 0 eV
    eig = np.zeros((nk, nb), dtype=np.float64)
    for ik in range(nk):
        eig[ik, :nw] = np.linspace(-5.0, -0.1, nw) + rng.uniform(-0.1, 0.1, nw)
        if nb > nw:
            eig[ik, nw:] = np.linspace(0.5, 5.0, nb - nw) + rng.uniform(-0.1, 0.1, nb - nw)

    # Amn: target bands → near-identity block; non-target → near-zero
    A = np.zeros((nk, nb, nw), dtype=np.complex128)
    for ik in range(nk):
        # Target block: diagonal ≈ proj_target^0.5 + noise
        block = np.eye(nw) * proj_target ** 0.5 + proj_noise * (
            rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
        )
        A[ik, :nw, :] = block
        # Non-target bands: tiny random overlap
        if nb > nw:
            A[ik, nw:, :] = 0.01 * (
                rng.standard_normal((nb - nw, nw))
                + 1j * rng.standard_normal((nb - nw, nw))
            )

    # Minimal overlap / weight tensors (unused by auto_window)
    Mmn    = torch.zeros(nk, 1, nb, nb, dtype=torch.complex128)
    kb_idx = torch.zeros(nk, 1, dtype=torch.long)
    wb     = torch.ones(1, dtype=torch.float64)
    bvecs  = torch.zeros(nk, 1, 3, dtype=torch.float64)
    kpts   = torch.zeros(nk, 3, dtype=torch.float64)

    return WannierData(
        Mmn    = Mmn,
        Amn    = torch.tensor(A, dtype=torch.complex128),
        eig    = torch.tensor(eig, dtype=torch.float64),
        kpts   = kpts,
        bvecs  = bvecs,
        wb     = wb,
        kb_idx = kb_idx,
    )


# ===========================================================================
# 1. Return type
# ===========================================================================

def test_returns_window_result():
    wdata = _make_wdata(nk=8, nb=4, nw=2)
    result = auto_window(wdata, nw=2)
    assert isinstance(result, WindowResult)


# ===========================================================================
# 2. outer_window is (E_min, E_max) with E_min < E_max
# ===========================================================================

def test_outer_window_valid():
    wdata = _make_wdata(nk=8, nb=4, nw=2)
    result = auto_window(wdata, nw=2)
    assert len(result.outer_window) == 2
    assert result.outer_window[0] < result.outer_window[1]


# ===========================================================================
# 3. At least nw bands per k inside outer_window
# ===========================================================================

def test_outer_window_covers_nw_bands_per_k():
    nw = 2
    wdata = _make_wdata(nk=16, nb=6, nw=nw)
    result = auto_window(wdata, nw=nw)
    eig = wdata.eig.numpy()
    E_lo, E_hi = result.outer_window
    n_in = ((eig >= E_lo) & (eig <= E_hi)).sum(axis=1)
    assert (n_in >= nw).all(), (
        f"Some k-points have < nw={nw} bands in outer window: min={n_in.min()}"
    )


# ===========================================================================
# 4. frozen_window is None or a valid sub-interval of outer_window
# ===========================================================================

def test_frozen_window_valid_or_none():
    wdata = _make_wdata(nk=8, nb=4, nw=2)
    result = auto_window(wdata, nw=2)
    if result.frozen_window is not None:
        fl, fh = result.frozen_window
        ol, oh = result.outer_window
        assert fl < fh,  "frozen_window E_min >= E_max"
        assert fl >= ol, "frozen_window extends below outer_window"
        assert fh <= oh, "frozen_window extends above outer_window"


# ===========================================================================
# 5. At most nw frozen bands per k-point
# ===========================================================================

def test_frozen_window_at_most_nw_bands():
    nw = 2
    wdata = _make_wdata(nk=8, nb=4, nw=nw, proj_target=0.99, proj_noise=0.001)
    result = auto_window(wdata, nw=nw, frozen_threshold=0.5)
    if result.frozen_window is not None:
        eig = wdata.eig.numpy()
        fl, fh = result.frozen_window
        n_froz = ((eig >= fl) & (eig <= fh)).sum(axis=1)
        assert (n_froz <= nw).all(), (
            f"Some k-points have > nw={nw} frozen bands: max={n_froz.max()}"
        )


# ===========================================================================
# 6. proj shape and range
# ===========================================================================

def test_proj_shape_and_range():
    nk, nb, nw = 8, 4, 2
    wdata = _make_wdata(nk=nk, nb=nb, nw=nw)
    result = auto_window(wdata, nw=nw)
    assert result.proj.shape == (nk, nb)
    assert result.proj.min() >= 0.0
    assert result.proj.max() <= 1.0 + 1e-12   # globally normalised


# ===========================================================================
# 7. threshold=0 → all bands in outer window
# ===========================================================================

def test_zero_threshold_all_bands():
    nk, nb, nw = 8, 6, 2
    wdata = _make_wdata(nk=nk, nb=nb, nw=nw)
    result = auto_window(wdata, nw=nw, threshold=0.0)
    eig = wdata.eig.numpy()
    assert result.outer_window[0] <= eig.min() + 1e-6
    assert result.outer_window[1] >= eig.max() - 1e-6


# ===========================================================================
# 8. frozen_threshold > 1 → frozen_window is None
# ===========================================================================

def test_high_frozen_threshold_gives_none():
    wdata = _make_wdata(nk=8, nb=4, nw=2)
    result = auto_window(wdata, nw=2, frozen_threshold=1.1)
    assert result.frozen_window is None
    assert result.n_frozen_bands == 0


# ===========================================================================
# 9. n_outer_bands >= nw
# ===========================================================================

def test_n_outer_at_least_nw():
    for nw in (1, 2, 3):
        wdata = _make_wdata(nk=8, nb=6, nw=nw)
        result = auto_window(wdata, nw=nw)
        assert result.n_outer_bands >= nw


# ===========================================================================
# 10. Synthetic: high-proj target bands → correct outer selection
# ===========================================================================

def test_target_bands_in_outer_window():
    """The nw target bands (lower energy, p≈0.9) must all be in outer_window."""
    nw = 2
    wdata = _make_wdata(nk=16, nb=6, nw=nw, proj_target=0.9, proj_noise=0.02)
    result = auto_window(wdata, nw=nw, threshold=0.10)
    eig = wdata.eig.numpy()

    # The nw target bands sit below 0 eV; outer_window should cover them
    target_eig_max = eig[:, :nw].max()
    target_eig_min = eig[:, :nw].min()
    assert result.outer_window[0] <= target_eig_min, "outer_window misses target band bottom"
    assert result.outer_window[1] >= target_eig_max, "outer_window misses target band top"


# ===========================================================================
# 11. Synthetic: non-target bands excluded from frozen window
# ===========================================================================

def test_non_target_bands_not_frozen():
    """Non-target bands (p≈0) must not appear in the frozen window."""
    nw = 2
    # proj_target=0.95 → target bands high-proj; non-target ≈ 0
    wdata  = _make_wdata(nk=16, nb=6, nw=nw, proj_target=0.95, proj_noise=0.01)
    result = auto_window(wdata, nw=nw, frozen_threshold=0.80)

    if result.frozen_window is not None:
        eig = wdata.eig.numpy()
        fl, fh = result.frozen_window
        # non-target bands sit above 0 eV → none should be in frozen window
        non_target_eig = eig[:, nw:]
        in_frozen = ((non_target_eig >= fl) & (non_target_eig <= fh)).any()
        assert not in_frozen, "Non-target bands leaked into frozen_window"


# ===========================================================================
# 12. Silicon: outer_window covers all 4 valence bands
# ===========================================================================

@pytest.mark.skipif(not HAS_SI, reason="Silicon reference data not found")
def test_si_outer_covers_all_valence_bands():
    """For Si isolated valence bands, all 4 must land in the outer window."""
    from waw.interfaces.wannier90.loader import load, parse_recip_lattice
    from waw.interfaces.wannier90.io   import read_win

    params = read_win(SI_DIR / "silicon.win")
    recip  = parse_recip_lattice(params)
    wdata  = load(SI_DIR / "silicon", recip)

    result = auto_window(wdata, nw=4, threshold=0.05)

    eig = wdata.eig.numpy()
    E_lo, E_hi = result.outer_window
    # All 4 bands must be inside the outer window at every k-point
    n_in = ((eig >= E_lo) & (eig <= E_hi)).sum(axis=1)
    assert (n_in >= 4).all(), (
        f"outer_window {result.outer_window} misses bands at some k: "
        f"min count = {n_in.min()}"
    )

    # proj should be normalised (max = 1)
    assert abs(result.proj.max() - 1.0) < 1e-12

    # n_outer_bands should be 4 (all valence bands included)
    assert result.n_outer_bands == 4
