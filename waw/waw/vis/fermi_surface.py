"""
Interactive Fermi-surface plot: Plotly Mesh3d isosurface sheets
(toggleable via the legend) plus the Brillouin-zone wireframe.

Renderer only -- no unit conversion, no interpolation, no marching cubes;
see `waw.analysis.fermi_surface.fermi_surface_sheets` for that half.

Display via `show_plotly` (below), not the bare `fig.show()`: Plotly's
default Jupyter renderer depends on the viewing frontend having a
matching extension registered and silently renders nothing otherwise
(e.g. a reopened, previously-executed `.ipynb`). `show_plotly` embeds the
whole plotly.js library inline instead, trading file size (~5 MB per
plot) for a guaranteed offline-safe render.
"""

from __future__ import annotations

import numpy as np

from ..analysis.fermi_surface import FermiSheet
from .brillouin_zone import bz_edges

_DEFAULT_COLORS = [
    "#c85a3f", "#3f7dc8", "#4fae5b", "#c8a13f",
    "#8a5ac8", "#3fc8bb", "#c83f8a", "#8ac83f",
]


def _bz_edge_trace(recip_lattice: np.ndarray, color: str, loops=None):
    """One Scatter3d line trace for the whole BZ wireframe (all facet
    loops concatenated with None-separators -- a single legend entry,
    not one per facet). `loops` lets the caller pass a precomputed
    `bz_edges(recip_lattice)` result instead of recomputing it."""
    import plotly.graph_objects as go

    if loops is None:
        loops = bz_edges(recip_lattice)

    xs, ys, zs = [], [], []
    for loop in loops:
        xs.extend(loop[:, 0].tolist() + [None])
        ys.extend(loop[:, 1].tolist() + [None])
        zs.extend(loop[:, 2].tolist() + [None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=3),
        name="Brillouin zone", showlegend=True, hoverinfo="skip",
    )


def plot_fermi_surface(
    sheets: list[FermiSheet],
    recip_lattice: np.ndarray,
    *,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    bz_color: str = "black",
    title: str | None = None,
    show_bz: bool = True,
    opacity: float = 1.0,
):
    """
    Render Fermi-surface sheets (from `fermi_surface_sheets`) plus the
    first-BZ wireframe (`brillouin_zone.bz_edges`) as an interactive
    Plotly figure -- drag to rotate/zoom, click a legend entry to toggle
    that sheet (or the BZ itself) on/off.

    Args:
      sheets       : output of `waw.analysis.fermi_surface.fermi_surface_sheets`
      recip_lattice: (3, 3) rows = b1,b2,b3, same units as `sheets`'
                     vertices (Bohr^-1 for waw's own convention)
      labels       : legend name per sheet; default "band {band_index}"
      colors       : fill color per sheet; default a fixed qualitative
                     palette, cycling if there are more sheets than colors
      bz_color     : BZ wireframe line color
      show_bz      : draw the BZ wireframe (default True)
      opacity      : Mesh3d opacity, shared by all sheets (1.0 = opaque)

    Returns a `plotly.graph_objects.Figure` (caller decides `.show()`,
    further layout tweaks, or `write_html`).
    """
    import plotly.graph_objects as go

    labels = labels if labels is not None else [f"band {s.band_index}" for s in sheets]
    colors = colors if colors is not None else _DEFAULT_COLORS

    data = []
    for i, sheet in enumerate(sheets):
        v, f = sheet.vertices, sheet.faces
        # Smooth (not flat) shading requires FermiSheet's vertices to be
        # welded (`analysis.fermi_surface._weld_vertices`) so Plotly can
        # average face normals into real per-vertex normals.
        data.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=colors[i % len(colors)],
            opacity=opacity,
            name=labels[i], showlegend=True,
            flatshading=False,
            lighting=dict(ambient=0.35, diffuse=0.8, specular=0.7, roughness=0.4, fresnel=0.15),
            lightposition=dict(x=10_000, y=10_000, z=5_000),
        ))

    # Fixed axis ranges from the BZ's own extent + aspectmode="cube",
    # rather than auto-ranged axes: auto-ranging refits to whichever
    # traces are currently visible, so toggling a legend entry would
    # rescale/re-zoom the whole scene.
    bz_loops = bz_edges(recip_lattice)
    bz_extent = np.concatenate([loop[:-1] for loop in bz_loops], axis=0)
    lim = float(np.abs(bz_extent).max()) * 1.05

    if show_bz:
        data.append(_bz_edge_trace(recip_lattice, bz_color, loops=bz_loops))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False, range=[-lim, lim]),
            yaxis=dict(visible=False, range=[-lim, lim]),
            zaxis=dict(visible=False, range=[-lim, lim]),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )
    return fig


def show_plotly(fig, height: str = "720px"):
    """
    Display a Plotly `Figure` as fully self-contained inline HTML (see
    module docstring). Still the full interactive widget: rotate/zoom/
    legend-toggle all work.

    `height` sets `to_html`'s `default_height` explicitly rather than its
    own `"100%"` default: a percentage height only resolves if an
    ancestor element already has an explicit height, which a Jupyter
    output cell does not, so the default silently renders a zero-height
    div.

    Returns an `IPython.display.HTML` object -- put this as a cell's
    last expression (or call `IPython.display.display` on it) to render.
    """
    from IPython.display import HTML
    return HTML(fig.to_html(include_plotlyjs=True, full_html=False, default_height=height))
