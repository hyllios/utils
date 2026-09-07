"""
K-path construction for band-structure interpolation.

Turns a Wannier90 `kpoint_path` block (as parsed by `io.read_win`, i.e.
`params["kpoint_path"]`) into a dense list of k-points plus the cumulative
distance along the path, for use with `hamiltonian.interpolate_bands` and
for labelling band-structure plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KPath:
    """A dense k-path with high-symmetry vertex bookkeeping for plotting."""
    kpts:        np.ndarray    # (nk, 3)  crystal coordinates
    dists:       np.ndarray    # (nk,)    cumulative distance, Bohr^-1
    tick_dists:  np.ndarray    # (n_vertices,) distances of the high-symmetry points
    tick_labels: list[str]     # (n_vertices,) e.g. ["G", "X", "W", "L|G", "K"]


def parse_kpoint_path(
    lines: list[str],
) -> list[tuple[str, np.ndarray, str, np.ndarray]]:
    """
    Parse the raw lines of a .win `kpoint_path` block.

    Each line has the form::

        LABEL1  k1x k1y k1z   LABEL2  k2x k2y k2z

    Returns a list of (label1, kpt1, label2, kpt2) segments.
    """
    segments = []
    for line in lines:
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"Malformed kpoint_path line: {line!r}")
        label1, k1 = parts[0], np.array(parts[1:4], dtype=np.float64)
        label2, k2 = parts[4], np.array(parts[5:8], dtype=np.float64)
        segments.append((label1, k1, label2, k2))
    return segments


def build_kpath(
    segments:      list[tuple[str, np.ndarray, str, np.ndarray]],
    recip_lattice: np.ndarray,
    n_points:      int = 100,
    total_points:  int | None = None,
) -> KPath:
    """
    Build a dense k-point path from a list of high-symmetry segments.

    Segments that are contiguous (segment i's end k-point equals segment
    i+1's start k-point) are merged into a single continuous run so the
    shared point isn't duplicated. Discontinuous jumps (e.g. W-L, G-K)
    are kept back-to-back at the same path distance, with the tick label
    combined as "L|G".

    Args:
      segments     : output of parse_kpoint_path
      recip_lattice: (3, 3) reciprocal lattice rows in Bohr^-1, used to
                     convert segment lengths to physical (Cartesian) units
      n_points     : number of k-points per segment (endpoints inclusive)
      total_points : if given, spread this many points over the whole path
                     in proportion to each segment's Cartesian length
                     (minimum 2 per segment), instead of `n_points` each.
                     Prefer this on cells with very anisotropic reciprocal
                     vectors: `dists` is a true Cartesian length, so a fixed
                     count per segment oversamples the short ones and makes
                     their dispersion look like a near-vertical jump. On the
                     c-doubled NiI2 magnetic cell (a=3.9, c=39 Ang), G-A is
                     1.6% of the path length but would get the same 60 points
                     as G-M -- 0.4 eV of dispersion drawn across 1.6% of the
                     axis. This is ASE's `bandpath(npoints=...)` convention.

    Returns:
      KPath with concatenated kpts/dists and vertex tick positions/labels.
    """
    if not segments:
        raise ValueError("Empty kpoint_path")

    counts = [n_points] * len(segments)
    if total_points is not None:
        lens = np.array([np.linalg.norm((k2 - k1) @ recip_lattice)
                         for _, k1, _, k2 in segments])
        if lens.sum() <= 0.0:
            raise ValueError("Path has zero total length; cannot distribute total_points")
        counts = [max(2, int(round(total_points * L / lens.sum()))) for L in lens]

    kpts_all:  list[np.ndarray] = []
    dists_all: list[np.ndarray] = []
    tick_dists  = [0.0]
    tick_labels = [segments[0][0]]

    dist0    = 0.0
    prev_k2  = None

    for (label1, k1, label2, k2), n_seg in zip(segments, counts):
        is_continuous = prev_k2 is not None and np.allclose(k1, prev_k2, atol=1e-8)
        if prev_k2 is not None and not is_continuous:
            tick_labels[-1] = f"{tick_labels[-1]}|{label1}"

        t = np.linspace(0.0, 1.0, n_seg)
        seg_kpts  = k1[None, :] + t[:, None] * (k2 - k1)[None, :]
        seg_len   = np.linalg.norm((k2 - k1) @ recip_lattice)
        seg_dists = dist0 + t * seg_len

        if is_continuous:
            seg_kpts  = seg_kpts[1:]
            seg_dists = seg_dists[1:]

        kpts_all.append(seg_kpts)
        dists_all.append(seg_dists)

        dist0 += seg_len
        tick_dists.append(dist0)
        tick_labels.append(label2)
        prev_k2 = k2

    return KPath(
        kpts        = np.concatenate(kpts_all, axis=0),
        dists       = np.concatenate(dists_all, axis=0),
        tick_dists  = np.array(tick_dists, dtype=np.float64),
        tick_labels = tick_labels,
    )
