"""
Wannierization-window diagnostic plot: DFT bands, Wannier-interpolated
bands, and the outer/frozen disentanglement windows drawn where they can
be SEEN.

Motivation (2026-07-28, trigonal Te): a wrong window survived four working
sessions because nothing ever drew it. The outer window silently included
the Te-5s manifold, so the frozen window held exactly ``num_wann`` bands at
every k-point, disentanglement had zero freedom, and the returned "p
model" contained no p-conduction band at all -- every downstream quantity
above E_F was built on the wrong subspace. One glance at this plot shows
both defects: a detached manifold inside the shaded outer window, and the
frozen edge slicing through (or below) the bands it was meant to protect.
`core.disentangle` now also warns programmatically on the zero-freedom
case; this plot is the human-facing half of that check.

Use it in every notebook's Wannierization section.
"""

from __future__ import annotations

import numpy as np

from .bands import break_at_path_jumps

__all__ = ["plot_wannierization_windows"]


def plot_wannierization_windows(
    ax,
    xcoords: np.ndarray,
    dft_bands_ev: np.ndarray,
    wann_bands_ev: np.ndarray | None = None,
    *,
    outer_window: tuple[float, float] | None = None,
    frozen_window: tuple[float, float] | None = None,
    fermi_energy: float | None = None,
    xticks=None,
    xticklabels=None,
    ylim: tuple[float, float] | None = None,
):
    """
    Draw DFT bands (points), Wannier bands (lines), and the windows
    (shaded spans) on ``ax``, plotted relative to E_F (this project's band-
    plot convention). Pass ABSOLUTE eV for the bands AND the windows AND
    ``fermi_energy``; the shift happens here, consistently, so the windows
    land exactly where they act. If ``fermi_energy`` is None everything is
    drawn absolute.

    Args:
      ax             : matplotlib axes.
      xcoords        : (nk,) linear k-path coordinate (see
                       `interfaces.ase.structure.band_path`).
      dft_bands_ev   : (nk, nbands) ab-initio eigenvalues along the path.
      wann_bands_ev  : (nk, nw) Wannier-interpolated bands, or None.
      outer_window   : (min, max) eV -- shaded light; the disentanglement
                       search space. Anything inside it is a candidate for
                       the model, including manifolds you forgot about.
      frozen_window  : (min, max) eV -- shaded darker; bands inside are
                       kept exactly. If this together with a detached
                       manifold fills num_wann slots everywhere,
                       disentanglement has no freedom (see module doc).
      fermi_energy   : eV, drawn as a dashed line.
      ylim           : y-limits. Default: the span of the WANNIER bands plus
                       15% padding (see the comment below for why it is
                       neither the window bounds nor the full DFT set).
                       Shaded windows are clipped to the visible range, so an
                       open-ended bound fills to the axis edge instead of
                       stretching the axis.

    Wannier lines are split at k-path discontinuities
    (`bands.break_at_path_jumps`) so a comma in the path string does not draw
    a band plunging vertically through the figure.
    """
    shift = fermi_energy if fermi_energy is not None else 0.0
    if outer_window is not None:
        outer_window = (outer_window[0] - shift, outer_window[1] - shift)
    if frozen_window is not None:
        frozen_window = (frozen_window[0] - shift, frozen_window[1] - shift)
    dft_bands_ev = np.asarray(dft_bands_ev) - shift
    if wann_bands_ev is not None:
        wann_bands_ev = np.asarray(wann_bands_ev) - shift

    # The y-range frames the WANNIER bands -- the thing under test -- and
    # never the windows or the full DFT set.
    #
    # Not the windows: they are routinely given open-ended sentinel bounds
    # (-1e3, -1e6, ...) meaning "no lower bound", which stretched the axis to
    # -1000 eV and flattened every band into a line at the top.
    #
    # Not the full DFT set either: a bands run returns every band pw.x
    # computed, including the semicore states `exclude_bands` removed before
    # disentanglement. Na chain (tutorial 14) has those at -56 and -26 eV
    # while the model spans ~4 eV around E_F, so ranging over the DFT bands
    # squashed the comparison just as badly. Bands outside the frame are
    # cropped, which is correct: they are not part of the model.
    if ylim is None:
        ref = wann_bands_ev if wann_bands_ev is not None else dft_bands_ev
        ref = np.asarray(ref)
        ref = ref[np.isfinite(ref)]
        if ref.size:
            lo, hi = float(ref.min()), float(ref.max())
            pad = max(0.15 * (hi - lo), 1.5)      # generous, so nearby DFT
            ylim = (lo - pad, hi + pad)           # structure stays visible
    if ylim is not None:
        ax.set_ylim(*ylim)

    def _shade(window, color, alpha, label):
        """Shade a window, clipped to the visible y-range so an open-ended
        bound fills to the axis edge instead of rescaling the plot."""
        y0, y1 = ax.get_ylim()
        lo, hi = max(window[0], y0), min(window[1], y1)
        if hi <= lo:
            return                                  # window is off-screen
        ax.axhspan(lo, hi, color=color, alpha=alpha, label=label, zorder=0)
        for y in window:                            # only edges that are visible
            if y0 < y < y1:
                ax.axhline(y, color=color, lw=0.8, alpha=min(1.0, 2 * alpha + 0.4),
                           zorder=1)

    if outer_window is not None:
        _shade(outer_window, "C0", 0.08, "outer window")
    if frozen_window is not None:
        _shade(frozen_window, "C2", 0.15, "frozen window")

    for ib in range(dft_bands_ev.shape[1]):
        ax.plot(xcoords, dft_bands_ev[:, ib], "o", ms=1.8, color="0.35",
                label="DFT" if ib == 0 else None, zorder=2)
    if wann_bands_ev is not None:
        # no lines across k-path discontinuities (see bands.break_at_path_jumps)
        _xb, _wb = break_at_path_jumps(xcoords, wann_bands_ev)
        for ib in range(_wb.shape[1]):
            ax.plot(_xb, _wb[:, ib], "-", lw=1.1, color="C3",
                    label="Wannier" if ib == 0 else None, zorder=3)
    if fermi_energy is not None:
        ax.axhline(0.0, color="k", lw=0.9, ls="--", label=r"$E_F$", zorder=4)

    if xticks is not None:
        ax.set_xticks(xticks)
        for x in xticks:
            ax.axvline(x, color="0.9", lw=0.5, zorder=0)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(float(xcoords[0]), float(xcoords[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)          # re-assert: axhline/axhspan can autoscale
    ax.set_ylabel(r"$E - E_F$ (eV)" if fermi_energy is not None else "E (eV)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    return ax
