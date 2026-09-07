"""
Unit tests for the generic Fermi-surface visualization pipeline:
`waw.vis.brillouin_zone.bz_edges` (BZ wireframe geometry, any crystal),
`waw.analysis.fermi_surface.fermi_surface_sheets` (isosurface computation),
and `waw.vis.fermi_surface.plot_fermi_surface` (Plotly rendering).

No DFT needed -- synthetic tight-binding models only, matching this
project's usual unit-test convention (real DFT cross-validation happens
in the tutorial notebooks, e.g. the Copper Fermi surface).
"""

import numpy as np
import torch

from ase.build import bulk

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.fermi_surface import fermi_surface_sheets, FermiSheet, _weld_vertices
from waw.vis import bz_edges, plot_fermi_surface, show_plotly


def _cubic_recip(a: float = 1.0) -> np.ndarray:
    real = np.eye(3) * a
    return 2 * np.pi * np.linalg.inv(real).T


def _simple_cubic_one_band_hr(t: float = -1.0) -> HamiltonianR:
    """H(k) = 2t*(cos kx + cos ky + cos kz), one band, real hoppings
    along the 6 nearest-neighbour simple-cubic directions."""
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)
    H_R = np.zeros((len(R_list), 1, 1), dtype=np.complex128)
    H_R[:6, 0, 0] = t
    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=1)


def _num_unique_edges(loops: list[np.ndarray]) -> int:
    edges = set()
    for loop in loops:
        n = len(loop) - 1   # last vertex repeats the first
        for i in range(n):
            a = tuple(np.round(loop[i], 6))
            b = tuple(np.round(loop[i + 1], 6))
            edges.add(tuple(sorted((a, b))))
    return len(edges)


def test_bz_edges_simple_cubic_is_a_cube():
    recip = _cubic_recip()
    loops = bz_edges(recip)

    assert len(loops) == 6   # a cube has 6 faces
    for loop in loops:
        assert loop.shape[1] == 3
        np.testing.assert_allclose(loop[0], loop[-1])   # closed loop
    assert _num_unique_edges(loops) == 12   # a cube has 12 edges

    all_vertices = np.concatenate([loop[:-1] for loop in loops], axis=0)
    np.testing.assert_allclose(np.abs(all_vertices).max(), np.pi, atol=1e-8)


def test_bz_edges_hexagonal_is_a_prism():
    """Generic-crystal sanity check: a hexagonal (non-cubic) lattice's
    first BZ is a hexagonal prism -- 8 faces (2 hexagons + 6
    rectangles), 18 edges, 12 vertices (Euler's formula V-E+F=2)."""
    atoms = bulk("C", "hcp", a=2.46, c=6.7)
    recip = 2 * np.pi * np.linalg.inv(np.array(atoms.cell)).T
    loops = bz_edges(recip)

    assert len(loops) == 8
    assert _num_unique_edges(loops) == 18


def test_fermi_surface_sheets_synthetic_cubic_band():
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()

    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(24, 24, 24))

    assert len(sheets) == 1
    sheet = sheets[0]
    assert isinstance(sheet, FermiSheet)
    assert sheet.band_index == 0
    assert sheet.vertices.ndim == 2 and sheet.vertices.shape[1] == 3
    assert sheet.faces.ndim == 2 and sheet.faces.shape[1] == 3
    assert sheet.faces.max() < sheet.vertices.shape[0]
    assert np.all(np.isfinite(sheet.vertices))

    # a generous BZ-containment check (whole-triangle keep/discard can
    # leave individual vertices a little outside a razor-exact bound)
    assert np.abs(sheet.vertices).max() <= np.pi * 1.05

    # vertices must be WELDED (shared across adjacent triangles), not a
    # triangle soup -- otherwise Plotly can't smooth-shade the surface
    # (see FermiSheet's docstring). A real closed mesh has roughly
    # nv ~ nf/2 (Euler's formula for a closed/near-closed triangulated
    # surface), nowhere near the 3*nf a fully unwelded soup would have.
    assert sheet.vertices.shape[0] < 0.75 * 3 * sheet.faces.shape[0]


def test_weld_vertices_merges_shared_edge():
    """Two triangles sharing an edge: as a soup that's 6 vertices; welded
    it must be exactly the 4 distinct points, with faces correctly
    remapped to reference them."""
    # triangle A: (0,0,0),(1,0,0),(0,1,0); triangle B: (1,0,0),(1,1,0),(0,1,0)
    soup = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 0, 0], [1, 1, 0], [0, 1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    v, f = _weld_vertices(soup, faces)

    assert v.shape[0] == 4
    # every welded face, reconstructed via its vertex coordinates, must
    # reproduce the original (unwelded) triangle's coordinates
    np.testing.assert_allclose(v[f[0]], soup[faces[0]])
    np.testing.assert_allclose(v[f[1]], soup[faces[1]])
    # the shared edge (1,0,0)-(0,1,0) must now use the SAME two indices
    # in both faces
    shared_a = {f[0][1], f[0][2]}      # triangle A's (1,0,0), (0,1,0)
    shared_b = {f[1][0], f[1][2]}      # triangle B's (1,0,0), (0,1,0)
    assert shared_a == shared_b


def test_fermi_surface_sheets_empty_outside_band_range():
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()

    # band range is [-6, 6] in |t|=1 units; 100 is far outside
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=100.0, mesh=(10, 10, 10))
    assert sheets == []


def test_fermi_surface_sheets_selects_requested_bands_only():
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()

    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(16, 16, 16), bands=[0])
    assert [s.band_index for s in sheets] == [0]

    sheets_none = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(16, 16, 16), bands=[])
    assert sheets_none == []


def test_plot_fermi_surface_trace_count_and_types():
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(16, 16, 16))

    fig = plot_fermi_surface(sheets, recip, title="test")

    assert len(fig.data) == len(sheets) + 1   # + 1 for the BZ wireframe trace
    assert fig.data[0].type == "mesh3d"
    assert fig.data[-1].type == "scatter3d"
    assert fig.data[-1].name == "Brillouin zone"


def test_plot_fermi_surface_without_bz():
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(16, 16, 16))

    fig = plot_fermi_surface(sheets, recip, show_bz=False)
    assert len(fig.data) == len(sheets)
    assert all(t.type == "mesh3d" for t in fig.data)


def test_plot_fermi_surface_axis_range_independent_of_bz_visibility():
    """The scene's axis ranges must be fixed up front (from the BZ's own
    extent) and identical whether or not the BZ trace is included --
    otherwise Plotly's `aspectmode="data"` auto-refit would rescale/
    re-zoom the whole scene whenever the BZ's legend entry is toggled,
    changing the mesh's apparent shading/contrast even though the
    Mesh3d trace itself never changed (the actual root cause of a real
    "looks flat with the BZ on" bug report)."""
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(16, 16, 16))

    fig_with_bz = plot_fermi_surface(sheets, recip, show_bz=True)
    fig_without_bz = plot_fermi_surface(sheets, recip, show_bz=False)

    assert fig_with_bz.layout.scene.aspectmode == "cube"
    assert fig_with_bz.layout.scene.xaxis.range is not None
    for axis in ("xaxis", "yaxis", "zaxis"):
        assert getattr(fig_with_bz.layout.scene, axis).range == \
            getattr(fig_without_bz.layout.scene, axis).range

    # toggling visibility on an ALREADY-BUILT figure (simulating a live
    # legend click) must not be able to change the range either, since
    # it's a static layout property, not auto-fit from visible traces
    fig_with_bz.data[-1].visible = "legendonly"
    for axis in ("xaxis", "yaxis", "zaxis"):
        assert getattr(fig_with_bz.layout.scene, axis).range == \
            getattr(fig_without_bz.layout.scene, axis).range


def test_show_plotly_embeds_plotlyjs_inline():
    """`show_plotly` must embed the whole plotly.js library inline
    (`include_plotlyjs=True`) rather than relying on `fig.show()`'s
    default Jupyter mimetype renderer -- that renderer depends on the
    viewing frontend having a matching extension (or a CDN fetch), and
    empirically renders as nothing once an executed notebook is reopened
    elsewhere. This test guards against silently regressing back to a
    renderer-dependent, CDN-dependent, or otherwise non-self-contained
    display path."""
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(12, 12, 12))
    fig = plot_fermi_surface(sheets, recip)

    html_obj = show_plotly(fig)

    html = html_obj.data
    assert "<script" in html
    assert "plotly" in html.lower()
    assert 'src="https://cdn' not in html   # must be embedded, not a CDN <script src=...> fetch
    assert len(html) > 1_000_000            # the embedded plotly.js library alone is several MB


def test_show_plotly_uses_explicit_pixel_height():
    """A percentage `height:100%` div only resolves to something nonzero
    if an ancestor already has an explicit height -- a Jupyter output
    cell does not, so `to_html`'s own `default_height="100%"` default
    silently renders a zero-height (invisible) plot. `show_plotly` must
    override this with an explicit pixel height."""
    hr = _simple_cubic_one_band_hr()
    recip = _cubic_recip()
    sheets = fermi_surface_sheets(hr, recip, fermi_energy=0.0, mesh=(12, 12, 12))
    fig = plot_fermi_surface(sheets, recip)

    import re

    def _outer_div_style(html):
        m = re.search(r'<div style="([^"]*)">', html)
        assert m, "no wrapper <div style=...> found"
        return m.group(1).replace(" ", "")

    style = _outer_div_style(show_plotly(fig).data)
    assert "height:100%" not in style
    assert "height:720px" in style

    style_custom = _outer_div_style(show_plotly(fig, height="400px").data)
    assert "height:400px" in style_custom
