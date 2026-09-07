"""
Generic band-structure-vs-k-path plotter.

Unit-agnostic and representation-agnostic: works off whatever
``(xcoords, xticks, xticklabels)`` triple the caller derived, from either
ASE's ``BandPath.get_linear_kpoint_axis()`` (``waw.interfaces.ase.
structure.band_path``) or waw's own ``analysis.kpath.KPath``
(``dists``/``tick_dists``/``tick_labels``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BandSeries:
    """
    One eigenvalue series to plot against a shared k-path axis.

    ``color_by`` (same shape as ``bands``) switches this series from a plain
    line plot to a per-band colour-mapped scatter -- e.g. spin_texture's
    ``<S_z>`` in [-1, 1], or (later) a phonon/atom-projection fatband weight.

    ``plot_kw`` is passed through to ``ax.plot`` for every band of this
    series, so a reference series can be drawn as markers under the model's
    lines instead of being hidden by them -- two line series that agree
    closely are indistinguishable, which defeats the purpose of overlaying
    them. E.g. ``plot_kw=dict(ls="none", marker=".", ms=2.0, zorder=1)``.
    Ignored when ``color_by`` is set (that branch scatters already).
    """
    bands:    np.ndarray
    label:    str | None = None
    color:    str | None = None
    color_by: np.ndarray | None = None
    cmap:     str = "coolwarm"
    vmin:     float | None = None
    vmax:     float | None = None
    plot_kw:  dict | None = None


def break_at_path_jumps(xcoords, values):
    """
    Insert NaN at k-path discontinuities so plotted lines are not drawn
    straight across them.

    Standard k-paths are not a single connected walk: ASE writes the breaks as
    commas (tetragonal is ``'GXMGZRAZ,XR,MA'``) and places the two sides of a
    break at the SAME linear coordinate. Joining them with a line adds a
    vertical segment that looks like a band plunging through the whole plot --
    visible in tutorial 14 (Na chain) at X and M, where the path jumps twice.
    Splitting the series at those points removes the artifact without
    touching any real dispersion.

    Args:
      xcoords : (nk,) linear k-path coordinate, non-decreasing except at breaks.
      values  : (nk,) or (nk, nbands) values sharing that axis.

    Returns:
      (xcoords, values) with a NaN separator inserted at every break; the
      inputs unchanged when the path is connected.
    """
    x = np.asarray(xcoords, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    breaks = np.flatnonzero(np.diff(x) <= 0.0)
    if breaks.size == 0:
        return x, v
    return (np.insert(x, breaks + 1, np.nan),
            np.insert(v, breaks + 1, np.nan, axis=0))


def _as_series_list(series) -> list[BandSeries]:
    if isinstance(series, BandSeries):
        return [series]
    if isinstance(series, (list, tuple)):
        return list(series)
    return [BandSeries(bands=np.asarray(series))]


def plot_bands(
    xcoords, xticks, xticklabels,
    series,
    *,
    ax=None,
    figsize=(6.5, 4.2),
    ref_line: float | None = None,
    ref_label: str = "E_F",
    ref_color: str = "C3",
    shade_window: tuple[float, float] | None = None,
    shade_label: str = "frozen window",
    ylim=None,
    xlim=None,
    ylabel: str = "E (eV)",
    title: str | None = None,
    colorbar_label: str | None = None,
    legend: bool = True,
):
    """
    Plot one or more band series against a shared k-path.

    Args:
      xcoords, xticks, xticklabels : the shared k-path linear axis (e.g.
                     ``bp.get_linear_kpoint_axis()`` for an ASE ``BandPath``
                     ``bp``, or ``(kpath.dists, kpath.tick_dists,
                     kpath.tick_labels)`` for a waw ``KPath``)
      series : a plain ``(nk, nbands)`` array, a single ``BandSeries``, or a
               list of ``BandSeries`` (e.g. spin up/down, each its own
               color/label) -- see ``BandSeries`` for the fatband/spin-color
               ``color_by`` mechanism.
      ax     : an existing Axes to draw into; a new figure is created if None.
      ref_line : draw a dashed horizontal reference line at this y-value
                 (e.g. E_F, or 0 if bands are already shifted by E_F).
      shade_window : (y_min, y_max) axhspan shading (e.g. a frozen
                     disentanglement energy window).
      colorbar_label : label for the colorbar drawn when any series sets
                       ``color_by``.
      legend : draw a legend if any series/ref_line/shade_window has a label.

    Returns the matplotlib Axes.
    """
    series_list = _as_series_list(series)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if shade_window is not None:
        ax.axhspan(shade_window[0], shade_window[1], color="C1", alpha=0.10,
                   label=shade_label, zorder=0)

    mappable = None
    for i, s in enumerate(series_list):
        bands = np.asarray(s.bands)
        nb = bands.shape[1]
        color = s.color or f"C{i}"
        if s.color_by is not None:
            for ib in range(nb):
                mappable = ax.scatter(
                    xcoords, bands[:, ib], c=np.asarray(s.color_by)[:, ib],
                    cmap=s.cmap, vmin=s.vmin, vmax=s.vmax, s=6,
                )
        else:
            band_labels = [s.label] + [None] * (nb - 1)
            xb, bb = break_at_path_jumps(xcoords, bands)   # no lines across path breaks
            kw = dict(color=color, lw=1.0)
            kw.update(s.plot_kw or {})
            for ib in range(nb):
                ax.plot(xb, bb[:, ib], label=band_labels[ib], **kw)

    if ref_line is not None:
        ax.axhline(ref_line, color=ref_color, ls="--", lw=0.9, label=ref_label, zorder=1)

    for x in xticks:
        ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(*(xlim if xlim is not None else (xcoords[0], xcoords[-1])))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if mappable is not None:
        ax.figure.colorbar(mappable, ax=ax, label=colorbar_label)
    elif legend:
        handles, labels = ax.get_legend_handles_labels()
        if any(labels):
            ax.legend(loc="best")

    return ax
