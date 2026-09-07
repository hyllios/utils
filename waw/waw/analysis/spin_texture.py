"""
Spin texture from a spin-orbit-coupled (noncollinear, `spinors = true`)
Wannier Hamiltonian: per-band spin expectation values, interpolated the
same way band energies are, from pw2wannier90's `.spn` file (Pauli
spin-operator matrix elements between ab-initio Bloch states,
`write_spn=.true.`).

Transcribed from wannier90's `get_oper.F90::get_SS_R` (real-space spin
operator SS(R) via the same W(k) gauge rotation + Wigner-Seitz Fourier
transform as H(R), see `core.hamiltonian.compute_operator_r`) and
`spin.F90::spin_get_nk` (diagonalize H(k), rotate the axis-projected spin
operator into that eigenbasis, take the diagonal). Matches wannier90's
`kpath_bands_colour = spin` band-structure output.

Energies are Hartree; spin expectation values are dimensionless (Pauli
eigenvalues +-1, matching wannier90's `spn_nk` convention, not +-hbar/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, compute_operator_r
from .bands import BandStructure
from .kpath import build_kpath, parse_kpoint_path


def spin_operator_r(
    W:            torch.Tensor,
    spn_bloch:    np.ndarray,
    kpts:         torch.Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> torch.Tensor:
    """
    Real-space spin operator SS(R), on the SAME R-vector grid `compute_hr`
    uses for H(R) (same mp_grid/real_lattice) -- so SS_R[..., c] can be
    Fourier-interpolated with the exact machinery used for H(R).

    Args:
      W            : (nk, nb, nw) complex, the SAME full converged gauge
                     passed to `compute_hr` (V@U for entangled bands, U
                     alone for isolated bands)
      spn_bloch    : (nk, nb, nb, 3) complex, `read_spn(...)["spn"]` --
                     ab-initio-basis Pauli matrix elements, same band
                     window/ordering as W (i.e. matching `wdata.eig`)
      kpts, mp_grid, real_lattice : same as `compute_hr`

    Returns SS_R: (nR, nw, nw, 3) complex.
    """
    spn_t = torch.as_tensor(spn_bloch, dtype=W.dtype, device=W.device)
    components = [
        compute_operator_r(W, spn_t[..., c], kpts, mp_grid, real_lattice)
        for c in range(3)
    ]
    return torch.stack(components, dim=-1)   # (nR, nw, nw, 3)


def spin_position_r(
    W:            torch.Tensor,
    sIu:          np.ndarray | torch.Tensor,
    kb_idx:       torch.Tensor,
    wb:           torch.Tensor,
    bvecs:        torch.Tensor,
    kpts:         torch.Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> torch.Tensor:
    """
    Real-space spin-position operator QQ(R)[a, i] = <0n|sigma_a (r-R)_i|Rm>,
    the Wannier-gauge form of

        Q^i_a(k) = i <u(k)|sigma_a|d_ki u(k)>
                 = i sum_b wb b_i <u_m(k)|sigma_a|u_n(k+b)>

    from the genuine ab-initio `.sIu` matrix elements. This is the quantity
    `analysis.spin_accumulation`'s `s^i_na` needs (SM25 Eq. 5a/6) and the
    SAA half of `spin_hall.build_shc_ryoo_operators` -- both call here.

    The b-sum needs no `- sigma_a(k)` subtraction term even though
    `|d_ki u> ~ sum_b wb b_i (|u(k+b)> - |u(k)>)`: the b-vector completeness
    condition `sum_b wb b_i = 0` kills it, exactly as it kills the `- delta_mn`
    in `compute_position_r`'s Berry connection.

    NOT Hermitized (unlike `compute_position_r`): Q^i_a is not a Hermitian
    operator -- SM25 Eq. 2b/7 take `Re[...]` of the final diagonal instead --
    so this uses `compute_bb_r`'s non-Hermitized b-weighted transform.

    Args:
      W        : (nk, nb, nw) complex, FULL converged gauge (V@U_final)
      sIu      : (nk, nnb, nb, nb, 3) complex, `read_sIu(...)["data"]`,
                 dimensionless (Pauli +-1), same band window/ordering as Mmn
      kb_idx, wb, bvecs, kpts, mp_grid, real_lattice : as `compute_bb_r`

    Returns QQ_R: (3(spin a), 3(cart i), nR, nw, nw) complex, raw (not
    degen-divided) -- same storage convention as AA_R/BB_R; reconstruct with
    `core.hamiltonian.operator_k`, which applies 1/degen.
    """
    from ..core.hamiltonian import compute_bb_r
    from ..core.spread import rotate_overlaps

    sIu_t = torch.as_tensor(sIu, dtype=W.dtype, device=W.device)
    return torch.stack([
        compute_bb_r(rotate_overlaps(W, sIu_t[..., c], kb_idx),
                     wb, bvecs, kpts, mp_grid, real_lattice)
        for c in range(3)
    ], dim=0)   # (3(spin), 3(cart), nR, nw, nw)


def interpolate_spin(
    hr:    HamiltonianR,
    SS_R:  torch.Tensor,
    kpts:  np.ndarray,
    axis:  tuple[float, float, float] = (0.0, 0.0, 1.0),
    ws=None,
) -> np.ndarray:
    """
    Per-band spin expectation <n(k)|S.axis|n(k)> along an arbitrary set of
    k-points, matching wannier90's `spin_get_nk`: diagonalize H(k), project
    the interpolated spin operator onto `axis` (normalized; default z),
    take the diagonal in H(k)'s eigenbasis.

    SS_R must be on the same R_vectors/degen grid as `hr` (built with the
    same mp_grid/real_lattice, see `spin_operator_r`).

    `ws` (a `core.ws_distance.WsDistance`) must be the same for H(k) and
    S(k) so eigenvectors and the spin operator share a gauge; pass the
    same `ws` used for `interpolate_bands`.

    Returns spin: (nk, nw) real, dimensionless (Pauli eigenvalues +-1).
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    SS_R_t = SS_R if torch.is_tensor(SS_R) else torch.as_tensor(SS_R)
    SS_R_axis = (SS_R_t.to(hr.H_R.dtype) @ torch.as_tensor(axis, dtype=hr.H_R.dtype)).contiguous()  # (nR,nw,nw)

    from ..core.hamiltonian import operator_k
    H_k = operator_k(hr.H_R, hr.R_vectors, hr.degen, kpts, ws=ws)          # (nk,nw,nw) torch
    S_k = operator_k(SS_R_axis, hr.R_vectors, hr.degen, kpts, ws=ws)

    # <n(k)|S.axis|n(k)> in H(k)'s eigenbasis, batched over all k at once
    H_k = 0.5 * (H_k + H_k.conj().transpose(-1, -2))
    _, U = torch.linalg.eigh(H_k)                                          # (nk,nw,nw)
    S_rot_diag = torch.einsum('kni,knm,kmi->ki', U.conj(), S_k, U)         # (nk,nw)
    return S_rot_diag.real.cpu().numpy()


@dataclass
class SpinColoredBands:
    """Band structure with a per-band spin expectation value alongside energy."""
    bands:      BandStructure
    spin:       np.ndarray   # (nk, nw), dimensionless (Pauli eigenvalues +-1)
    axis:       np.ndarray   # (3,) the (normalized) quantization axis used


def spin_colored_bands(
    hr:            HamiltonianR,
    SS_R:          torch.Tensor,
    kpoint_path:   list[str],
    recip_lattice: np.ndarray,
    n_points:      int = 100,
    axis:          tuple[float, float, float] = (0.0, 0.0, 1.0),
    ws=None,
) -> SpinColoredBands:
    """
    `bands.band_structure` plus the per-band spin expectation value along
    the same k-path -- matches wannier90's `kpath_bands_colour = spin`.

    `ws` (a `core.ws_distance.WsDistance`) applies use_ws_distance
    (wannier90's default) to both the bands and the spin -- see
    `interpolate_spin`.
    """
    segments = parse_kpoint_path(kpoint_path)
    kpath = build_kpath(segments, recip_lattice, n_points=n_points)
    from ..core.hamiltonian import interpolate_bands
    bands = BandStructure(kpath=kpath, bands=interpolate_bands(hr, kpath.kpts, ws=ws))
    spin = interpolate_spin(hr, SS_R, kpath.kpts, axis=axis, ws=ws)
    axis_arr = np.asarray(axis, dtype=np.float64)
    return SpinColoredBands(bands=bands, spin=spin, axis=axis_arr / np.linalg.norm(axis_arr))
