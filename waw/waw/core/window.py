"""
Automatic energy window selection for Wannier disentanglement.

Computes per-band, per-k projectability from the Amn trial-projection matrix
and derives outer / frozen energy windows for the disentanglement step.

Definition
----------
    p_n(k) = Σ_m |A_mn(k)|²      (sum over nw trial functions)

Normalized globally so that the most WF-like band at its best k has p = 1:
    p_n(k) ← p_n(k) / max_{k,n} p_n(k)

Selection rules
---------------
  outer window  : bands where max_k p_n(k) > threshold
                  ("looks like a WF at any k-point")
  frozen window : bands where min_k p_n(k) > frozen_threshold
                  ("looks like a WF at every k-point")

Both windows are returned as energy tuples (E_min, E_max) in Hartree (the core
atomic-unit convention, matching the disentangler and the core wannierize
driver), with an additional buffer to avoid placing the boundary on a band
edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import WannierData


@dataclass
class WindowResult:
    """
    Output of auto_window().

    Attributes
    ----------
    outer_window   : (E_min, E_max) Hartree — outer disentanglement window
    frozen_window  : (E_min, E_max) Hartree or None if no bands qualify
    proj           : (nk, nb) globally normalised projectability p_n(k)
    mean_proj      : (nb,)  mean_k p_n(k)
    max_proj       : (nb,)  max_k  p_n(k)  — drives outer-window selection
    min_proj       : (nb,)  min_k  p_n(k)  — drives frozen-window selection
    n_outer_bands  : bands whose max_proj > threshold (before energy fallback)
    n_frozen_bands : bands whose min_proj > frozen_threshold
    """

    outer_window:   tuple[float, float]
    frozen_window:  tuple[float, float] | None
    proj:           np.ndarray
    mean_proj:      np.ndarray
    max_proj:       np.ndarray
    min_proj:       np.ndarray
    n_outer_bands:  int
    n_frozen_bands: int


def auto_window(
    wdata:            WannierData,
    nw:               int | None = None,
    threshold:        float = 0.10,
    frozen_threshold: float = 0.85,
    buffer:           float = 0.002,
) -> WindowResult:
    """
    Automatically select outer and frozen energy windows from projectability.

    Parameters
    ----------
    wdata : WannierData
        Loaded input data; only ``Amn`` and ``eig`` are used.
    nw : int, optional
        Target number of Wannier functions.  Defaults to ``wdata.nw``.
    threshold : float
        Minimum normalised max_k projectability for a band to enter the outer
        window (default 0.10).  Lower → wider window.
    frozen_threshold : float
        Minimum normalised min_k projectability for a band to be frozen
        (default 0.85).  Higher → fewer frozen bands.
    buffer : float
        Energy buffer (Hartree) added to both sides of each window so the
        boundary never lands exactly on a band edge (default 0.002 Ha ≈ 0.05 eV).

    Returns
    -------
    WindowResult

    Notes
    -----
    The outer window is guaranteed to contain at least *nw* bands at every
    k-point; if the projectability criterion yields fewer, the window is
    widened automatically.

    The frozen window is clamped to lie strictly inside the outer window.
    If more than *nw* bands satisfy the frozen criterion (rare), only the
    *nw* with the highest min_proj are frozen.
    """
    if nw is None:
        nw = wdata.nw

    A   = wdata.Amn.detach().cpu().numpy()    # (nk, nb, nw_trials)
    eig = wdata.eig.detach().cpu().numpy()    # (nk, nb)
    nk, nb = eig.shape

    # ---- Raw projectability: p_n(k) = Σ_m |A_mn(k)|² -------------------
    proj_raw = (np.abs(A) ** 2).sum(axis=-1)   # (nk, nb)

    # Globally normalise to [0, 1] so thresholds are code-independent.
    p_max = proj_raw.max()
    proj  = proj_raw / p_max if p_max > 0 else proj_raw.copy()

    mean_proj = proj.mean(axis=0)              # (nb,)
    max_proj  = proj.max(axis=0)              # (nb,)
    min_proj  = proj.min(axis=0)              # (nb,)

    # ---- Outer window: include if WF-like at ANY k -----------------------
    outer_flag = max_proj > threshold          # (nb,) bool
    n_outer    = int(outer_flag.sum())

    if n_outer < nw:
        # Threshold too strict — fall back to the nw bands with highest max_proj.
        outer_flag             = np.zeros(nb, dtype=bool)
        outer_flag[np.argsort(max_proj)[-nw:]] = True
        n_outer                = nw

    outer_E_min = float(eig[:, outer_flag].min()) - buffer
    outer_E_max = float(eig[:, outer_flag].max()) + buffer

    # Guarantee ≥ nw bands inside the outer window at every k-point.
    eig_sorted  = np.sort(eig, axis=1)        # (nk, nb) ascending per k
    outer_E_min = min(outer_E_min, float(eig_sorted[:, 0].min())        - buffer)
    outer_E_max = max(outer_E_max, float(eig_sorted[:, nw - 1].max())   + buffer)

    # ---- Frozen window: freeze only if WF-like at ALL k ------------------
    frozen_flag = min_proj > frozen_threshold  # (nb,) bool
    n_frozen    = int(frozen_flag.sum())

    if n_frozen > nw:
        # More frozen candidates than WFs — keep only the nw most WF-like.
        frozen_flag              = np.zeros(nb, dtype=bool)
        frozen_flag[np.argsort(min_proj)[-nw:]] = True
        n_frozen                 = nw

    if n_frozen == 0:
        frozen_window = None
    else:
        froz_E_min = float(eig[:, frozen_flag].min()) - buffer
        froz_E_max = float(eig[:, frozen_flag].max()) + buffer

        # Clamp to lie strictly inside the outer window.
        froz_E_min = max(froz_E_min, outer_E_min)
        froz_E_max = min(froz_E_max, outer_E_max)

        if froz_E_min < froz_E_max:
            frozen_window = (froz_E_min, froz_E_max)
        else:
            frozen_window = None
            n_frozen      = 0

    return WindowResult(
        outer_window   = (outer_E_min, outer_E_max),
        frozen_window  = frozen_window,
        proj           = proj,
        mean_proj      = mean_proj,
        max_proj       = max_proj,
        min_proj       = min_proj,
        n_outer_bands  = n_outer,
        n_frozen_bands = n_frozen,
    )
