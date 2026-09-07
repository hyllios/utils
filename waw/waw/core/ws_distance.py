"""
use_ws_distance: Wannier-centre-aware minimal-image assignment of the
real-space operator vectors, wannier90's default interpolation refinement.

Plain Wannier interpolation places every matrix element O_{ij}(R) at the
same lattice vector R. But the two Wannier functions i (in the home cell)
and j (in cell R) sit at their own centres tau_i, tau_j; the physically
correct separation is R + tau_j - tau_i, and the smoothest interpolant
uses, for each (i, j, R), the periodic image of that separation lying in
the Wigner-Seitz cell of the mp_grid supercell. When that image is on a
WS-cell face, all equidistant images are kept and the matrix element
split equally between them (degeneracy averaging).

Transcribed from wannier90's src/ws_distance.F90::ws_translate_dist and
R_wz_sc, and applied at interpolation time the way postw90's
pw90common_fourier_R_to_k does after operator_wigner_setup expands the
grid. Equivalent to that expanded-grid approach, but done as an
(i, j, R)-dependent phase in the per-k interpolation loop instead
(waw already loops per-k), so no separate expanded R-grid is built.

UNITS. Atomic (Bohr) throughout, like the rest of `core`: centres and
real_lattice in Bohr, R_vectors in integer lattice coordinates. `tol` is
a length in Bohr for the WS-boundary degeneracy test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# wannier90's ndegenx: max WS-boundary degeneracy it will tolerate.
_NDEGENX = 8


@dataclass
class WsDistance:
    """
    Precomputed use_ws_distance data for a fixed (H(R) grid, WF centres).

    shiftedR : (nR, nw, nw, ndegenx, 3) int64 -- for element (i, j) at grid
               vector R (index r), the up-to-ndegenx WS-minimal-image
               integer lattice vectors R + s. Only the first ndeg[r,i,j]
               slots are valid (mask).
    ndeg     : (nR, nw, nw) int64 -- WS-boundary degeneracy per (R, i, j).
    mask     : (nR, nw, nw, ndegenx) bool -- valid-slot mask.
    """
    shiftedR: np.ndarray
    ndeg:     np.ndarray
    mask:     np.ndarray

    def phase(self, kpt: np.ndarray) -> np.ndarray:
        """
        The (i, j)-dependent WS interpolation phase at a single k-point
        (crystal coords): (1/ndeg) * sum_ideg exp(2πi k·(R+s_ideg)),
        shape (nR, nw, nw) complex. Multiply by O_R and the usual
        1/degen(R) and sum over R to get O(k) -- see `interpolate_bands`.
        """
        rdotk = 2.0 * np.pi * (self.shiftedR @ np.asarray(kpt, dtype=np.float64))  # (nR,nw,nw,ndegenx)
        ph = np.where(self.mask, np.exp(1j * rdotk), 0.0)
        return ph.sum(axis=-1) / self.ndeg   # (nR, nw, nw)


def build_ws_distance(
    R_vectors:      np.ndarray,
    centres_cart:   np.ndarray,
    mp_grid:        tuple[int, int, int],
    real_lattice:   np.ndarray,
    ws_search_size: int = 2,
    tol:            float = 1e-5,
) -> WsDistance:
    """
    Precompute use_ws_distance minimal-image assignments.

    Args:
      R_vectors     : (nR, 3) int    H(R) grid vectors (integer lattice coords)
      centres_cart  : (nw, 3) float  Wannier centres in Bohr (Cartesian)
      mp_grid       : (N1,N2,N3)     Monkhorst-Pack grid (supercell size)
      real_lattice  : (3, 3) float   rows = lattice vectors a1,a2,a3 in Bohr
      ws_search_size : supercell search half-width (wannier90 default 2)
      tol           : WS-boundary degeneracy tolerance, Bohr

    Returns a `WsDistance`.
    """
    R_vectors = np.asarray(R_vectors, dtype=np.int64)
    centres_cart = np.asarray(centres_cart, dtype=np.float64)
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    N = np.asarray(mp_grid, dtype=np.int64)
    nR = R_vectors.shape[0]
    nw = centres_cart.shape[0]

    # Candidate supercell translations s = (a*N1, b*N2, c*N3), Cartesian.
    # wannier90 searches [-ws_search_size-1, ws_search_size+1] per axis.
    w = ws_search_size + 1
    abc = np.array([[a, b, c]
                    for a in range(-w, w + 1)
                    for b in range(-w, w + 1)
                    for c in range(-w, w + 1)], dtype=np.int64)      # (S, 3)
    shift_int = abc * N[None, :]                                     # (S, 3) integer lattice shift
    shift_cart = shift_int @ real_lattice                           # (S, 3) Bohr
    S = shift_int.shape[0]

    # Target separation T = R_cart + tau_j - tau_i for every (R, i, j).
    R_cart = R_vectors @ real_lattice                               # (nR, 3)
    dtau = centres_cart[None, :, :] - centres_cart[:, None, :]      # (nw, nw, 3) = tau_j - tau_i
    T = R_cart[:, None, None, :] + dtau[None, :, :, :]              # (nR, nw, nw, 3)

    M = nR * nw * nw
    Tf = T.reshape(M, 3)                                            # (M, 3)

    # Distance^2 of each candidate image, for every element -- (S, M).
    # cand = Tf + shift_cart[s];  dist2 = |cand|^2
    dist2 = ((Tf[None, :, :] + shift_cart[:, None, :]) ** 2).sum(axis=-1)   # (S, M)
    min_dist = np.sqrt(dist2.min(axis=0))                          # (M,)
    degen_mask = np.abs(np.sqrt(dist2) - min_dist[None, :]) < tol   # (S, M) bool

    ndeg_flat = degen_mask.sum(axis=0)                             # (M,)
    if ndeg_flat.max() > _NDEGENX:
        raise ValueError(
            f"use_ws_distance: WS degeneracy {ndeg_flat.max()} exceeds ndegenx={_NDEGENX} "
            "(unexpected -- check WF centres / lattice)"
        )

    # Scatter the degenerate shifted integer R-vectors into slots.
    R_of_elem = np.repeat(R_vectors, nw * nw, axis=0)              # (M, 3), R for each flat element
    shiftedR_flat = np.zeros((M, _NDEGENX, 3), dtype=np.int64)
    slot = np.cumsum(degen_mask, axis=0) - 1                       # (S, M) 0-based slot per True
    s_idx, m_idx = np.nonzero(degen_mask)
    shiftedR_flat[m_idx, slot[s_idx, m_idx], :] = R_of_elem[m_idx] + shift_int[s_idx]

    shiftedR = shiftedR_flat.reshape(nR, nw, nw, _NDEGENX, 3)
    ndeg = ndeg_flat.reshape(nR, nw, nw)
    mask = np.arange(_NDEGENX)[None, None, None, :] < ndeg[..., None]

    return WsDistance(shiftedR=shiftedR, ndeg=ndeg, mask=mask)
