"""
Real-space Wannier Hamiltonian and band interpolation.

After Wannierization, the tight-binding Hamiltonian in the Wannier basis is:

    H_{mn}(R) = (1/Nk) Σ_k e^{-2πi k·R} [U†(k) diag(ε_k) U(k)]_{mn}

where
  * k is a crystal-coord k-point (components in [0, 1))
  * R = (R1, R2, R3) is a supercell lattice vector (integer components)
  * k·R = k1*R1 + k2*R2 + k3*R3   (inner product in fractional units)
  * ε_k are the DFT eigenvalues in Hartree
  * U(k) is the (nw × nw) converged gauge matrix

Bands at any k-point are recovered by the inverse transform:

    H(k) = Σ_R e^{2πi k·R} H(R) / D(R)

where D(R) is the Wigner-Seitz degeneracy weight.

This module provides:
  compute_hr       — Fourier transform to real space
  interpolate_bands — eigenvalues of H(k) on an arbitrary k-path
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import numpy as np

from ..units import EV_TO_HARTREE as _EV_TO_HARTREE
from torch import Tensor


# ---------------------------------------------------------------------------
# Wigner-Seitz supercell and R-vectors
# ---------------------------------------------------------------------------

def _wigner_seitz(mp_grid: tuple[int, int, int], real_lattice: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all supercell R-vectors and their Wigner-Seitz degeneracies.

    For an N1×N2×N3 MP mesh the supercell lattice vectors are N_i * a_i.
    A point R belongs to the Wigner-Seitz cell of the origin if |R| ≤ |R + T|
    for all supercell translations T. The degeneracy D(R) is the number of
    equidistant images (weight 1/D is applied in H(k)).

    Args:
      mp_grid     : (N1, N2, N3) Monkhorst-Pack grid
      real_lattice: (3, 3) rows = lattice vectors a1, a2, a3 in Bohr

    Returns:
      R_vectors: (nR, 3) int64   lattice vectors in direct coordinates
      degen    : (nR,)   int64   Wigner-Seitz degeneracy D(R)
    """
    N1, N2, N3 = mp_grid

    # Candidate R-vectors: all primitive lattice points in [-N_i, N_i].
    # The WS cell of the supercell (N_i*a_i) is centered at zero, so we
    # need a range larger than the primary cell [0, N_i).
    candidates = np.array(
        [[r1, r2, r3]
         for r1 in range(-N1, N1 + 1)
         for r2 in range(-N2, N2 + 1)
         for r3 in range(-N3, N3 + 1)],
        dtype=np.int64,
    )  # ((2N1+1)*(2N2+1)*(2N3+1), 3)

    # Supercell translation vectors in Cartesian. The +-2 shell (125 images)
    # is wannier90's/EPW's own range (EPW/src/wigner.f90 loops i1,i2,i3 over
    # -2..2), NOT the +-1 shell: on a skewed cell a SECOND-shell supercell
    # image can tie with the first, and missing it undercounts that R's
    # degeneracy. Silent on cubic/hexagonal cells but real -- on the strained
    # bct Al of notebook 13 (c/a = 1.4365) exactly two of 253 R-vectors came
    # out with degen 3 instead of 4 under +-1, overweighting them by 4/3.
    # The invariant that catches this is sum_R 1/degen(R) == N1*N2*N3, which
    # says the WS cell tiles the supercell; it read 216.1667 vs 216 there.
    rng = (-2, -1, 0, 1, 2)
    shift_direct = np.array(
        [[t1 * N1, t2 * N2, t3 * N3] for t1 in rng for t2 in rng for t3 in rng],
        dtype=np.int64,
    )  # (125, 3)
    shift_cart = shift_direct @ real_lattice   # (125, 3)

    # Vectorized WS check over all candidates at once.
    # candidates_cart : (nR_cand, 3)
    # images          : (nR_cand, 125, 3) — all supercell images of each candidate
    candidates_cart = candidates @ real_lattice                        # (nR_cand, 3)
    images          = candidates_cart[:, None, :] + shift_cart[None]  # (nR_cand, 125, 3)
    dists2          = (images ** 2).sum(axis=-1)                      # (nR_cand, 125)
    d_self          = (candidates_cart ** 2).sum(axis=-1)             # (nR_cand,)
    d_min           = dists2.min(axis=-1)                             # (nR_cand,)

    # Tolerance must SCALE: an absolute 1e-6 Bohr^2 is meaningless against
    # |R|^2 ~ 1e3 Bohr^2 on a large mesh, where ties are only equal to
    # rounding. EPW compares sorted distances with a relative eps6 likewise.
    tol = 1e-6 * np.maximum(d_min, 1.0)
    in_ws  = d_self <= d_min + tol
    degen  = (dists2 <= (d_min + tol)[:, None]).sum(axis=-1)          # (nR_cand,)

    return (candidates[in_ws].astype(np.int64),
            degen[in_ws].astype(np.int64))


# ---------------------------------------------------------------------------
# Real-space Hamiltonian
# ---------------------------------------------------------------------------

@dataclass
class HamiltonianR:
    """
    Real-space Wannier Hamiltonian H(R).

    ``centres``/``real_lattice``/``mp_grid`` are what `use_ws_distance` needs.
    When all three are present the interpolation is centre-aware BY DEFAULT
    (wannier90's own default), which is not a refinement but a correctness
    fix: without it, two gauges whose Wannier functions differ by lattice
    vectors -- identical Omega, identical bands on the mesh -- interpolate
    completely differently. On bcc Nb that was a factor 73 at E_F, 3797 meV
    against 52. With it the two agree to 0.7 meV, as they must, the difference
    between them being a gauge choice.
    """
    H_R:       Tensor        # (nR, nw, nw)  complex
    R_vectors: np.ndarray    # (nR, 3)       int64   direct coords
    degen:     np.ndarray    # (nR,)         int64   W-S degeneracy
    nw:        int
    centres:      np.ndarray | None = None   # (nw, 3) Bohr, Cartesian
    real_lattice: np.ndarray | None = None   # (3, 3)  Bohr, rows
    mp_grid:      tuple | None      = None

    def ws_distance(self):
        """Cached `WsDistance` for this model, or None if centres are absent."""
        if getattr(self, "_ws_cache", "miss") != "miss":
            return self._ws_cache
        ws = None
        if (self.centres is not None and self.real_lattice is not None
                and self.mp_grid is not None):
            from .ws_distance import build_ws_distance
            ws = build_ws_distance(self.R_vectors, np.asarray(self.centres),
                                   tuple(self.mp_grid),
                                   np.asarray(self.real_lattice))
        object.__setattr__(self, "_ws_cache", ws)
        return ws


def hopping_range(hr, real_lattice) -> float:
    """
    Second moment of the hopping range, <R^2>_H, in Bohr^2 -- how far the
    real-space Hamiltonian actually reaches.

        <R^2>_H = sum_R |R|^2 ||H(R)||_F^2 / sum_R ||H(R)||_F^2

    THE QUANTITY Omega CANNOT SEE. A Wannier spread is translation-invariant
    per function, so two gauges whose Wannier functions differ by lattice
    vectors have identical Omega and completely different H(R). It is exactly
    that degeneracy that let `global_minimize_spread`'s argmin(Omega) return a
    model with a 52x worse interpolation at E_F (see its docstring). On the two
    bcc Nb minima:

        model            Omega    Omega~   <R^2>_H     off-mesh @ E_F
        n_restarts=1    10.4241   1.3938   0.562 A^2         73 meV
        n_restarts=3    10.4235   1.3932   7.839 A^2       3797 meV

    Omega separates them by a factor 1.00006; this by 14. A model whose H(R)
    does not decay rings between mesh points while staying exact on them, so
    this is the cheap health number to look at when an interpolated quantity
    misbehaves.

    Caveats. H(R) on a finite mesh is periodic with the superlattice, so this
    measures concentration WITHIN the Wigner-Seitz cell, not true decay -- it
    is a discriminator, not a localization length. And a metal has a physical
    floor set by its Friedel oscillations, so a "large" value is only
    meaningful against another model of the same system on the same mesh.

    Args:
      hr           : HamiltonianR
      real_lattice : (3, 3) real-space lattice rows, Bohr

    Returns <R^2>_H in Bohr^2 (multiply by BOHR_TO_ANG**2 for Ang^2).
    """
    import numpy as np

    R = np.asarray(hr.R_vectors, dtype=np.float64)
    H = hr.H_R.detach().cpu().numpy() if hasattr(hr.H_R, "detach") else np.asarray(hr.H_R)
    deg = np.asarray(hr.degen, dtype=np.float64)
    w = (np.abs(H / deg[:, None, None]) ** 2).sum(axis=(1, 2))
    d2 = (R @ np.asarray(real_lattice, dtype=np.float64)) ** 2
    tot = w.sum()
    return float((w * d2.sum(axis=1)).sum() / tot) if tot > 0 else float("nan")


def compute_hr(
    U:            Tensor,
    eig:          Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> HamiltonianR:
    """
    Compute the real-space Hamiltonian H(R) from the converged gauge U.

    Args:
      U           : (nk, nw, nw) complex  converged gauge matrices
      eig         : (nk, nw) real          Kohn-Sham eigenvalues in Hartree
                    (only the nw bands selected by U; pass eig[:, :nw] or
                    disentangled eigenvalues)
      kpts        : (nk, 3) real           k-points in crystal coordinates
      mp_grid     : (N1, N2, N3)           Monkhorst-Pack grid dimensions
      real_lattice: (3, 3) float64         lattice vectors as rows in Bohr

    Returns:
      HamiltonianR with H_R (nR, nw, nw), R_vectors, degen, nw.
    """
    nk, nw = U.shape[0], U.shape[-1]

    # H_k^{WF} = U†(k) diag(ε_k) U(k)
    # eig shape: (nk, nw); broadcast to (nk, nw, nw) diagonal
    H_k = torch.matmul(
        U.conj().transpose(-1, -2),
        eig.to(dtype=U.dtype).unsqueeze(-1) * U,   # diag(ε) U = ε_n * U[:,n]
    )   # (nk, nw, nw)

    # Wigner-Seitz R-vectors
    R_arr, degen = _wigner_seitz(mp_grid, real_lattice)
    nR = len(R_arr)

    # Phase factors e^{-2πi k·R}: (nk, nR)
    # k in crystal coords (nk, 3), R in direct integers (nR, 3)
    # k·R = sum_i k_i * R_i (fractional dot product)
    k_np = kpts.detach().cpu().numpy()           # (nk, 3)
    phase_np = np.exp(-2j * np.pi * (k_np @ R_arr.T))   # (nk, nR)
    phase = torch.tensor(phase_np, dtype=U.dtype, device=U.device)   # (nk, nR)

    # H(R) = (1/Nk) Σ_k e^{-2πi k·R} H_k^{WF}
    # einsum: (nk, nR) × (nk, nw, nw) → (nR, nw, nw)
    H_R = torch.einsum("kr,kmn->rmn", phase, H_k) / nk

    return HamiltonianR(H_R=H_R, R_vectors=R_arr, degen=degen, nw=nw,
                        real_lattice=np.asarray(real_lattice),
                        mp_grid=tuple(mp_grid))


def apply_scissors_shift(
    hr:                  HamiltonianR,
    U_final:             Tensor,
    kpts:                Tensor,
    mp_grid:             tuple[int, int, int],
    real_lattice:        np.ndarray,
    num_valence_bands:   int,
    scissors_shift:      float,
) -> HamiltonianR:
    """
    Rigidly shift the conduction manifold's eigenvalues by `scissors_shift`
    (Hartree) while leaving H(k)'s eigenvectors unchanged (postw90's
    `scissors_shift`, e.g. opening a DFT-underestimated gap before SHC).

    Adds `scissors_shift * P_conduction(R)` to `H_R`, where
    `P_conduction(k) = U(k)^dagger @ diag(0,...,0,1,...,1) @ U(k)` is the
    projector onto the conduction subspace of the converged gauge `U(k)`
    (0 for the first `num_valence_bands` bands, 1 for the rest).
    `P_conduction(k)` commutes with `H(k)` (same eigenbasis `U(k)`), so this
    shifts exactly the top `nw - num_valence_bands` eigenvalues without
    touching eigenvectors. `P_conduction(R)` has the same
    `U^dagger diag(.) U` form as `compute_hr`'s `H(R)`, so it reuses
    `compute_hr` directly.

    Args:
      hr                : the (un-scissored) converged HamiltonianR
      U_final           : (nk, nw, nw) complex, the same converged gauge
                          used to build `hr`
      kpts, mp_grid, real_lattice : same as used to build `hr`
      num_valence_bands : number of low-lying valence WFs left unshifted
      scissors_shift    : Hartree (postw90's own value is in eV; convert
                          at the caller's interface boundary)

    Returns a new HamiltonianR (does not mutate `hr`).
    """
    nk, nw = U_final.shape[0], U_final.shape[-1]
    proj = torch.zeros(nw, dtype=torch.float64)
    proj[num_valence_bands:] = 1.0
    proj_k = proj[None, :].expand(nk, -1)

    P_cond = compute_hr(U_final, proj_k, kpts, mp_grid, real_lattice)

    return HamiltonianR(
        H_R=hr.H_R + scissors_shift * P_cond.H_R,
        R_vectors=hr.R_vectors, degen=hr.degen, nw=hr.nw,
    )


def compute_operator_r(
    W:            Tensor,
    O_k:          Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
) -> Tensor:
    """
    Fourier-transform a general ab-initio-basis (Bloch-gauge) Hermitian
    operator into the Wannier gauge and real space: the same W†(k) O_k W(k)
    rotation and Wigner-Seitz Fourier transform as `compute_hr`, generalized
    from diag(eigenvalues) to an arbitrary per-k Hermitian matrix (e.g. the
    .spn spin operator, see `analysis/spin_texture.py`).

    Args:
      W    : (nk, nb, nw) complex  full converged gauge (V@U for
             entangled bands, U alone for isolated bands)
      O_k  : (nk, nb, nb) complex  operator matrix elements between
             ab-initio Bloch states, same band window/ordering as W
      kpts, mp_grid, real_lattice : same as `compute_hr`

    Returns O_R: (nR, nw, nw) complex, on the same R_vectors/degen grid as
    `compute_hr`; call `_wigner_seitz(mp_grid, real_lattice)` directly for
    R_vectors/degen, or reuse an existing `HamiltonianR.R_vectors`/`.degen`.
    """
    nk, nw = W.shape[0], W.shape[-1]

    # O_k^{WF} = W†(k) O_k W(k)
    O_k_wf = torch.matmul(W.conj().transpose(-1, -2), torch.matmul(O_k, W))   # (nk, nw, nw)

    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    k_np = kpts.detach().cpu().numpy()
    phase_np = np.exp(-2j * np.pi * (k_np @ R_arr.T))
    phase = torch.tensor(phase_np, dtype=W.dtype, device=W.device)

    O_R = torch.einsum("kr,kmn->rmn", phase, O_k_wf) / nk
    return O_R


def _bond_midpoint_phase(bvecs: Tensor, centres: Tensor) -> Tensor:
    """
    exp(i * b . r0(m,n)), r0(m,n) = (centres[m]+centres[n])/2 -- the
    bond-midpoint-referenced phase correction wannier90's
    ``transl_inv_full`` applies to q-space position-type matrix elements
    before any k->R Fourier transform, making quantities built from the
    result (AA(R)/BB(R)/CC(R) and derived quantities like orbital
    magnetization) invariant to a rigid shift of the origin/atomic basis --
    see `compute_position_r`'s ``centres=`` docstring.

    Args:
      bvecs  : (nk, nnb, 3) real, Cartesian b-vectors (Bohr^-1)
      centres: (nw, 3) real, Wannier centres (Bohr)

    Returns (nk, nnb, nw, nw) complex unit-modulus phase factor.
    """
    centres_r = centres.to(torch.float64)
    r0 = 0.5 * (centres_r[:, None, :] + centres_r[None, :, :])          # (nw, nw, 3)
    b_dot_r0 = torch.einsum('kba,mna->kbmn', bvecs.to(torch.float64), r0)  # (nk, nnb, nw, nw)
    return torch.polar(torch.ones_like(b_dot_r0), b_dot_r0)             # complex128


def _r_dependent_shell_phase(bvecs_canonical: Tensor, R_arr: np.ndarray, factor: float = -0.5) -> Tensor:
    """
    exp(factor * i * b_shell . R), one factor per (b-shell, R) pair -- the
    second ``transl_inv_full`` phase correction, applied after the k->R
    Fourier transform, per b-shell, before summing shells (depends on R,
    so it does not commute with collapsing the b-sum before the transform).

    Args:
      bvecs_canonical: (nnb, 3) real, Cartesian b-vectors in the same
                       canonical (k=0) ordering as `wb` (Bohr^-1)
      R_arr          : (nR, 3) real, Cartesian Wigner-Seitz R-vectors (Bohr)
      factor         : -0.5 for AA(R)/BB(R)/CC(R)'s "phase2"; CC(R)
                       additionally needs a combined (b1+b2) version, built
                       by the caller from two calls' exponents added
                       together (equivalently, pass the summed b-vector)

    Returns (nnb, nR) complex unit-modulus phase factor.
    """
    R_t = torch.as_tensor(R_arr, dtype=torch.float64)
    b_dot_R = torch.einsum('ba,ra->br', bvecs_canonical.to(torch.float64), R_t)   # (nnb, nR)
    return torch.polar(torch.ones_like(b_dot_R), factor * b_dot_R)


def compute_position_r(
    M_tilde:      Tensor,
    wb:           Tensor,
    bvecs:        Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
    centres:      Tensor | None = None,
) -> Tensor:
    """
    Real-space Wannier position operator AA(R)_a = <0n|r_a|Rm>, in Bohr,
    via the Marzari-Vanderbilt finite-difference Berry connection (Eq. 44,
    PRB 56, 12847, 1997):

        A_a(k) = i * sum_b wb * b_a * Mtilde^{k,b}           (a = x,y,z Cartesian)
        A_a(k) -> Hermitian part: 0.5*(A_a(k) + A_a(k)^dagger)
        AA(R)_a = (1/Nk) sum_k e^{-2pi i k.R} A_a(k)

    `M_tilde` must already be in the final Wannier gauge (W = V@U for
    entangled bands, U alone for isolated bands) -- exactly what
    `spread.rotate_overlaps` returns / `WannierResult.m_tilde` stores. No
    further gauge rotation happens here (unlike `compute_operator_r`, which
    rotates a separately-supplied ab-initio-basis operator by W).

    ``centres`` (wannier90's ``transl_inv_full``): when given (the Wannier
    centres, Bohr), apply the translational-invariance correction instead
    of the plain formula above, making AA(R)/BB(R)/CC(R) (and quantities
    derived from them, e.g. M_orb/AHC) invariant to a rigid shift of the
    atomic basis/origin:

    1. Multiply M_tilde by a bond-midpoint phase before forming A_q, per
       (k,b,m,n) (`_bond_midpoint_phase`): `M_tilde *= exp(i*b.r0(m,n))`,
       `r0(m,n) = (centres[m]+centres[n])/2`.
    2. Fourier-transform each b-shell separately (rather than collapsing
       the b-sum first) and multiply by a second, R-dependent phase
       `exp(-i*0.5*b_shell.R)` (`_r_dependent_shell_phase`) before summing
       shells.
    3. Set the R=0 diagonal to the actual Wannier centres directly
       (`AA_R[a, i, i, R=0] = centres[i, a]`, exact by Eq. 31 MV97).

    No Hermitization is applied in this mode.

    Returns AA_R: (3, nR, nw, nw) complex, on the same R_vectors/degen
    grid `compute_hr`/`_wigner_seitz(mp_grid, real_lattice)` produce. Raw
    (not degen-divided) -- same storage convention as `compute_hr`'s H_R;
    use `position_operator_k` to reconstruct A(k)/Omega_bar(k), which
    applies 1/degen there.
    """
    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    k_np = kpts.detach().cpu().numpy()
    phase_np = np.exp(-2j * np.pi * (k_np @ R_arr.T))
    phase = torch.tensor(phase_np, dtype=M_tilde.dtype, device=M_tilde.device)
    nk = M_tilde.shape[0]
    nw = M_tilde.shape[-1]

    if centres is None:
        A_q = 1j * torch.einsum(
            'b,kba,kbmn->kamn', wb.to(M_tilde.dtype), bvecs.to(M_tilde.dtype), M_tilde,
        )   # (nk, 3, nw, nw)
        A_q = 0.5 * (A_q + A_q.conj().transpose(-1, -2))
        AA_R = torch.einsum('kr,kamn->ramn', phase, A_q) / nk   # (nR, 3, nw, nw)
        return AA_R.permute(1, 0, 2, 3).contiguous()            # (3, nR, nw, nw)

    phase1 = _bond_midpoint_phase(bvecs, centres).to(M_tilde.dtype)   # (nk, nnb, nw, nw)
    M_corrected = M_tilde * phase1

    A_q_b = 1j * torch.einsum(
        'b,kba,kbmn->kbamn', wb.to(M_tilde.dtype), bvecs.to(M_tilde.dtype), M_corrected,
    )   # (nk, nnb, 3, nw, nw)

    AA_R_b = torch.einsum('kr,kbamn->bramn', phase, A_q_b) / nk   # (nnb, nR, 3, nw, nw)

    bvecs_canonical = bvecs[0]                                    # (nnb, 3)
    phase2 = _r_dependent_shell_phase(bvecs_canonical, R_arr).to(M_tilde.dtype)   # (nnb, nR)

    AA_R = torch.einsum('br,bramn->ramn', phase2, AA_R_b)          # (nR, 3, nw, nw)
    AA_R = AA_R.permute(1, 0, 2, 3).contiguous()                   # (3, nR, nw, nw)

    r0_idx = int(np.nonzero((R_arr == 0).all(axis=1))[0][0])
    diag = torch.diag_embed(centres.to(M_tilde.dtype).T)           # (3, nw, nw)
    AA_R[:, r0_idx, :, :] = diag

    return AA_R


def compute_bb_r(
    H_tilde:      Tensor,
    wb:           Tensor,
    bvecs:        Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
    centres:      Tensor | None = None,
    H_R:          Tensor | None = None,
) -> Tensor:
    """
    Real-space quantity BB(R)_a = <0n|H(r-R)_a|Rm>, the Fourier transform of
    BB_a(k) = i <u_k|H_k|del_a u_k>, needed (with `compute_cc_r`) for the
    orbital magnetization (postw90 `berry_task = morb`; CTVR06/LVTS12). Same
    finite-difference Berry-connection sum as `compute_position_r`'s AA(R),
    but on `H_tilde` instead of `M_tilde`, and not Hermitized (BB is
    generally non-Hermitian: H(r-R) singles out the bra side).

    `H_tilde` is the Wannier-gauge-rotated, per-neighbour matrix element
    W(k)^dagger [diag(eigval(k)) . Mmn(k,b)] W(k+b): exactly `M_tilde` but
    with the raw overlap first left-multiplied, band-index by band-index,
    by the ab-initio eigenvalues at k (H_k is diagonal in its own Bloch
    eigenbasis, so <u_{m,k}|H_k = eigval[k,m] * <u_{m,k}|). Build it as:

        Mmn_weighted = eig[:, None, :, None] * Mmn        # weight bra index
        H_opt        = rotate_overlaps(V, Mmn_weighted, kb_idx)   # if disentangled
        H_tilde      = rotate_overlaps(U_final, H_opt, kb_idx)

    (skip the V step for isolated bands, mirroring how `Mmn_opt`/`M_tilde`
    are built in `core.pipeline.wannierize`).

    ``centres``/``H_R`` (wannier90's ``transl_inv_full`` -- see
    `compute_position_r`'s `centres=` docstring for the general mechanism):
    when given, apply the same bond-midpoint phase1 + per-shell R-dependent
    phase2 as AA(R), plus an extra additive term BB(R) alone needs:

        BB(R)_a += (r0_a(m,n) - 0.5*R_a) * HH(R)(m,n)

    (`r0(m,n) = (centres[m]+centres[n])/2`) -- `H_R` (the same real-space
    Hamiltonian `compute_hr` builds) is required when `centres` is given
    (raises if omitted).

    Returns BB_R: (3, nR, nw, nw) complex, on the same R_vectors/degen grid
    as `compute_hr`/`compute_position_r`. Raw (not degen-divided); use
    `core.hamiltonian.operator_k` per Cartesian component to reconstruct
    BB(k).
    """
    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    k_np = kpts.detach().cpu().numpy()
    phase_np = np.exp(-2j * np.pi * (k_np @ R_arr.T))
    phase = torch.tensor(phase_np, dtype=H_tilde.dtype, device=H_tilde.device)
    nk = H_tilde.shape[0]

    if centres is None:
        B_q = 1j * torch.einsum(
            'b,kba,kbmn->kamn', wb.to(H_tilde.dtype), bvecs.to(H_tilde.dtype), H_tilde,
        )   # (nk, 3, nw, nw)
        BB_R = torch.einsum('kr,kamn->ramn', phase, B_q) / nk   # (nR, 3, nw, nw)
        return BB_R.permute(1, 0, 2, 3).contiguous()            # (3, nR, nw, nw)

    if H_R is None:
        raise ValueError("compute_bb_r: H_R is required when centres is given (transl_inv_full).")

    phase1 = _bond_midpoint_phase(bvecs, centres).to(H_tilde.dtype)   # (nk, nnb, nw, nw)
    H_corrected = H_tilde * phase1

    B_q_b = 1j * torch.einsum(
        'b,kba,kbmn->kbamn', wb.to(H_tilde.dtype), bvecs.to(H_tilde.dtype), H_corrected,
    )   # (nk, nnb, 3, nw, nw)
    BB_R_b = torch.einsum('kr,kbamn->bramn', phase, B_q_b) / nk   # (nnb, nR, 3, nw, nw)

    bvecs_canonical = bvecs[0]
    phase2 = _r_dependent_shell_phase(bvecs_canonical, R_arr).to(H_tilde.dtype)   # (nnb, nR)
    BB_R = torch.einsum('br,bramn->ramn', phase2, BB_R_b)          # (nR, 3, nw, nw)
    BB_R = BB_R.permute(1, 0, 2, 3).contiguous()                   # (3, nR, nw, nw)

    centres_r = centres.to(torch.float64)
    r0 = 0.5 * (centres_r[:, None, :] + centres_r[None, :, :])      # (nw, nw, 3)
    r0_minus_halfR = r0[None, :, :, :] - 0.5 * torch.as_tensor(R_arr, dtype=torch.float64)[:, None, None, :]
    r0_minus_halfR = r0_minus_halfR.permute(3, 0, 1, 2).to(H_tilde.dtype)   # (3, nR, nw, nw)
    BB_R = BB_R + r0_minus_halfR * H_R[None, :, :, :]

    return BB_R


def _negate_R_index(R_arr: np.ndarray) -> np.ndarray:
    """For each row R in R_arr, the index of -R in R_arr -- assumes R_arr
    is symmetric under negation (true for any Wigner-Seitz R-vector list
    on a Bravais lattice). Needed by `compute_cc_r`'s `transl_inv_full`
    correction, which reuses `BB_R` evaluated at -R."""
    neg = -R_arr
    idx = np.zeros(len(R_arr), dtype=np.int64)
    for i, r in enumerate(neg):
        matches = np.nonzero((np.abs(R_arr - r) < 1e-8).all(axis=1))[0]
        if len(matches) == 0:
            raise ValueError(f"-R not found in R_arr for R={R_arr[i]} -- not a symmetric WS grid?")
        idx[i] = matches[0]
    return idx


def compute_cc_r(
    uHu:          Tensor,
    W:            Tensor,
    kb_idx:       Tensor,
    wb:           Tensor,
    bvecs:        Tensor,
    kpts:         Tensor,
    mp_grid:      tuple[int, int, int],
    real_lattice: np.ndarray,
    centres:      Tensor | None = None,
    BB_R:         Tensor | None = None,
    H_R:          Tensor | None = None,
) -> Tensor:
    """
    Real-space quantity CC(R)_ab = <0|r_a.H.(r-R)_b|R>, the Fourier
    transform of CC_ab(k) = <del_a u_k|H_k|del_b u_k>, needed (with
    `compute_bb_r`) for the orbital magnetization.

    Unlike AA(R)/BB(R) (built from the .mmn overlap, which only connects k
    to its own neighbours), CC needs the ab-initio matrix element
    <u_{k+b1}|H_k|u_{k+b2}> between two (possibly different) neighbours of
    k -- the `.uHu` file (`interfaces.wannier90.io.read_uHu`,
    `pw2wannier90 write_uHu=.true.`) -- since this cannot be recovered from
    Mmn + eigenvalues the way `compute_bb_r`'s H_tilde can.

    Args:
      uHu    : (nk, nnb, nnb, nb, nb) complex, `read_uHu`'s "uHu" array,
               uHu[k, p, q, m, n] = <u_{m,k+b(p)}|H_k|u_{n,k+b(q)}>
               (ab-initio band basis, 0-based m, n; p, q index the same
               neighbour-shell ordering as `kb_idx`/`bvecs`).
      W      : (nk, nb, nw) complex, the full converged gauge (V@U_final
               for disentangled bands, U_final alone for isolated bands).
      kb_idx : (nk, nnb) long, neighbour-index table (same as `rotate_overlaps`).
      wb, bvecs, kpts, mp_grid, real_lattice : as `compute_position_r`.

    ``centres``/``BB_R``/``H_R`` (wannier90's ``transl_inv_full`` -- see
    `compute_position_r`'s `centres=` docstring for the general mechanism):
    when given, apply the same 2-phase correction as AA(R)/BB(R), but with
    both b-vectors (b1=p, b2=q) involved:

    1. Before the k->R transform: `H_tilde[k,p,q,m,n] *=
       exp(i*(b_q.r0(m,n) - b_p.r0(m,n)))`.
    2. Fourier-transform each (p,q) shell pair separately, then multiply by
       `exp(-i*0.5*(b_p+b_q).R)` before summing shell pairs.
    3. Three extra additive terms (CC(R) alone needs, beyond AA/BB's single
       extra term):

           CC(R)_ab += (r0_a(m,n) + 0.5*R_a) * BB(R)_b(m,n)
           CC(R)_ab += conj(BB(R)_a(n,m)|_{-R}) * (r0_b(m,n) - 0.5*R_b)
           CC(R)_ab += (r0_a(m,n) + 0.5*R_a) * R_b * HH(R)(m,n)

    `BB_R` must already be the `transl_inv_full`-corrected one (`compute_bb_r`
    with the same `centres`/`H_R`). `H_R` is the same real-space Hamiltonian
    `compute_hr` builds.

    Returns CC_R: (3, 3, nR, nw, nw) complex, on the same R_vectors/degen
    grid as `compute_hr`. Raw (not degen-divided).
    """
    Wg = W[kb_idx]   # (nk, nnb, nb, nw): Wg[k, p] = W(k + b_p)

    # H_tilde[k,p,q] = W(k+b_p)^dagger . uHu[k,p,q] . W(k+b_q)   (nk,nnb,nnb,nw,nw)
    H_tilde = torch.einsum(
        'kpmw,kpqmn,kqnx->kpqwx',
        Wg.conj().to(uHu.dtype), uHu, Wg.to(uHu.dtype),
    )

    wb_c = wb.to(uHu.dtype)
    bvecs_c = bvecs.to(uHu.dtype)

    R_arr, _ = _wigner_seitz(mp_grid, real_lattice)
    k_np = kpts.detach().cpu().numpy()
    phase_np = np.exp(-2j * np.pi * (k_np @ R_arr.T))
    phase = torch.tensor(phase_np, dtype=uHu.dtype, device=uHu.device)
    nk = uHu.shape[0]

    if centres is None:
        C_q = torch.einsum(
            'p,kpa,q,kqb,kpqwx->kabwx', wb_c, bvecs_c, wb_c, bvecs_c, H_tilde,
        )   # (nk, 3, 3, nw, nw)
        CC_R = torch.einsum('kr,kabmn->rabmn', phase, C_q) / nk   # (nR, 3, 3, nw, nw)
        return CC_R.permute(1, 2, 0, 3, 4).contiguous()           # (3, 3, nR, nw, nw)

    if BB_R is None or H_R is None:
        raise ValueError("compute_cc_r: BB_R and H_R are required when centres is given (transl_inv_full).")

    phase1 = _bond_midpoint_phase(bvecs, centres).to(uHu.dtype)   # (nk, nnb, nw, nw), exp(i*b.r0)
    # exp(i*(b_q.r0 - b_p.r0)) = conj(phase1_p) * phase1_q
    H_corrected = H_tilde * phase1.conj()[:, :, None, :, :] * phase1[:, None, :, :, :]

    C_q_pq = torch.einsum(
        'p,kpa,q,kqb,kpqwx->kpqabwx', wb_c, bvecs_c, wb_c, bvecs_c, H_corrected,
    )   # (nk, nnb, nnb, 3, 3, nw, nw)
    CC_R_pq = torch.einsum('kr,kpqabmn->pqrabmn', phase, C_q_pq) / nk   # (nnb,nnb,nR,3,3,nw,nw)

    bvecs_canonical = bvecs[0]
    shell_phase = _r_dependent_shell_phase(bvecs_canonical, R_arr).to(uHu.dtype)   # (nnb, nR)
    phase2_pq = shell_phase[:, None, :] * shell_phase[None, :, :]                  # (nnb, nnb, nR)

    CC_R = torch.einsum('pqr,pqrabmn->rabmn', phase2_pq, CC_R_pq)   # (nR, 3, 3, nw, nw)
    CC_R = CC_R.permute(1, 2, 0, 3, 4).contiguous()                 # (3, 3, nR, nw, nw)

    centres_r = centres.to(torch.float64)
    r0 = 0.5 * (centres_r[:, None, :] + centres_r[None, :, :])   # (nw, nw, 3)
    R_t = torch.as_tensor(R_arr, dtype=torch.float64)             # (nR, 3)

    r0_plus_halfR  = (r0[None, :, :, :] + 0.5 * R_t[:, None, None, :]).permute(3, 0, 1, 2).to(uHu.dtype)  # (3,nR,nw,nw)
    r0_minus_halfR = (r0[None, :, :, :] - 0.5 * R_t[:, None, None, :]).permute(3, 0, 1, 2).to(uHu.dtype)  # (3,nR,nw,nw)
    Rb = R_t.to(uHu.dtype)   # (nR, 3)

    term1 = r0_plus_halfR[:, None, :, :, :] * BB_R[None, :, :, :, :]                       # (3,3,nR,nw,nw)

    neg_idx = _negate_R_index(R_arr)
    BB_R_negR_T = BB_R[:, neg_idx, :, :].conj().transpose(-1, -2)                          # (3,nR,nw,nw), (m,n)->(n,m)
    term2 = BB_R_negR_T[:, None, :, :, :] * r0_minus_halfR[None, :, :, :, :]               # (3,3,nR,nw,nw)

    term3 = r0_plus_halfR[:, None, :, :, :] * Rb.T[None, :, :, None, None] * H_R[None, None, :, :, :]

    return CC_R + term1 + term2 + term3


def position_operator_k(
    AA_R:         Tensor,
    R_vectors:    np.ndarray,
    degen:        np.ndarray,
    real_lattice: np.ndarray,
    kpts:         np.ndarray,
) -> tuple:
    """
    Reconstruct, at an arbitrary set of k-points, the two Fourier-transform
    conventions of the position operator that WYSV06/postw90 need:

        A(k)_a        = sum_R e^{2pi i k.R} AA_R(R)_a / degen(R)          ("OO_true")
        Omega_bar(k)_a = i * sum_R e^{2pi i k.R} (R x AA_R(R))_a / degen(R)  ("OO_pseudo")

    with the cyclic cross product (a=x: R_y*AA_R_z - R_z*AA_R_y, etc).
    Omega_bar is the J0 (band-Kubo-equivalent) WYSV06 curvature term;
    A(k) feeds the J1/J2 position-operator correction terms.

    Args:
      AA_R      : (3, nR, nw, nw) complex, from `compute_position_r`
      R_vectors : (nR, 3) int, direct coords (same grid as AA_R)
      degen     : (nR,) Wigner-Seitz degeneracies
      real_lattice: (3, 3) rows = lattice vectors in Bohr
      kpts      : (nk, 3) crystal-coordinate k-points

    Returns (A_k, omega_bar_k): each (nk, 3, nw, nw) complex, Bohr / Bohr^2.
    """
    crvec = R_vectors.astype(np.float64) @ real_lattice   # (nR, 3) Cartesian Bohr
    inv_degen = 1.0 / np.asarray(degen, dtype=np.float64)

    phase_np = np.exp(2j * np.pi * (np.asarray(kpts) @ np.asarray(R_vectors).T))   # (nk, nR)
    phase = torch.tensor(phase_np, dtype=AA_R.dtype, device=AA_R.device)
    w = torch.tensor(inv_degen, dtype=AA_R.dtype, device=AA_R.device)

    A_k = torch.einsum('kr,r,armn->kamn', phase, w, AA_R)   # (nk, 3, nw, nw)

    cyc = ((1, 2), (2, 0), (0, 1))
    crvec_t = torch.tensor(crvec, dtype=AA_R.dtype, device=AA_R.device)   # (nR, 3)
    nk = phase.shape[0]
    nw = AA_R.shape[-1]
    omega_bar_k = torch.empty((nk, 3, nw, nw), dtype=AA_R.dtype, device=AA_R.device)
    for comp, (j, l) in enumerate(cyc):
        cross_R = crvec_t[:, j, None, None] * AA_R[l] - crvec_t[:, l, None, None] * AA_R[j]   # (nR,nw,nw)
        omega_bar_k[:, comp] = 1j * torch.einsum('kr,r,rmn->kmn', phase, w, cross_R)

    return A_k, omega_bar_k


def position_operator_derivative_k(
    AA_R:          Tensor,
    R_vectors:     np.ndarray,
    degen:         np.ndarray,
    recip_lattice: np.ndarray,
    kpts:          np.ndarray,
) -> Tensor:
    """
    Cartesian k-derivative of the Wannier-gauge position operator, dA(k)_c/
    dk_b, for a k-stack: the analytic-Fourier-series derivative of
    `position_operator_k`'s `A(k) = sum_R e^{2pi i k.R} AA_R(R)/degen(R)`
    ("OO_true"), needed by shift current's `AA_da_bar` (IATS18 Eq. 34).

    Same differentiation pattern as `analysis._fourier_derivs.
    h_and_grad_frac_batch` applied to H(R): one extra factor of
    `2*pi*i*R_b` per real-space lattice vector, then Jacobian-rotated to
    Cartesian via `inv(recip_lattice)` (kpts are fractional/crystal
    coordinates, so the raw Fourier-series derivative is fractional too).
    Not Hermitized on reconstruction, matching `position_operator_k`.

    Args:
      AA_R         : (3, nR, nw, nw) complex, from `compute_position_r`
      R_vectors    : (nR, 3) int, direct coords (same grid as AA_R)
      degen        : (nR,) Wigner-Seitz degeneracies
      recip_lattice: (3, 3) rows = reciprocal lattice vectors, Bohr^-1 (2pi convention)
      kpts         : (nk, 3) crystal-coordinate k-points

    Returns dA_dk: (nk, 3, 3, nw, nw) complex, Bohr (axes: b=Cartesian
    derivative direction, c=Cartesian position-operator component).
    """
    inv_degen = 1.0 / np.asarray(degen, dtype=np.float64)
    R_np = np.asarray(R_vectors, dtype=np.float64)

    phase_np = np.exp(2j * np.pi * (np.asarray(kpts) @ R_np.T))   # (nk, nR)
    phase = torch.tensor(phase_np, dtype=AA_R.dtype, device=AA_R.device)
    w = torch.tensor(inv_degen, dtype=AA_R.dtype, device=AA_R.device)          # (nR,)
    R_t = torch.tensor(R_np, dtype=AA_R.dtype, device=AA_R.device)             # (nR, 3)

    grad_coeff = (phase * w[None, :])[:, :, None] * (2j * np.pi * R_t[None, :, :])   # (nk, nR, 3)
    dA_dk_frac = torch.einsum('krb,crmn->kbcmn', grad_coeff, AA_R)   # (nk, 3, 3, nw, nw)

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=AA_R.dtype, device=AA_R.device)
    dA_dk = torch.einsum('jb,kbcmn->kjcmn', inv_recip, dA_dk_frac)   # Cartesian derivative direction
    return dA_dk


def _degenerate_aware_band_velocities(
    dH_eig: Tensor, eig: Tensor, degen_thresh: float,
) -> Tensor:
    """
    Band velocities dE_n/dk, using the degenerate-perturbation-theory
    correction (WYSV06/YWVS07 Eq. 27/31) for groups of adjacent
    (numerically-)degenerate bands: for a contiguous run of bands [i,j]
    with adjacent gaps all below `degen_thresh`, diagonalize `dH_eig`'s
    (i:j+1, i:j+1) submatrix and take the eigenvalues as that group's band
    velocities (a single diagonal element is ill-defined within a
    degenerate subspace, but the submatrix eigenvalues are gauge-invariant
    and well-defined); isolated bands keep the plain Hellmann-Feynman
    diagonal.

    Only loops (in Python) over k-points with some adjacent
    near-degeneracy; the rest get the vectorized diagonal directly.

    Args:
      dH_eig: (nk, 3, nw, nw) complex, H(k)-eigenbasis-rotated dH/dk
      eig   : (nk, nw) real, Hartree, ascending
      degen_thresh: energy gap below which adjacent bands are grouped

    Returns del_eig: (nk, nw, 3) real, Hartree*Bohr.
    """
    nk, _, nw, _ = dH_eig.shape
    del_eig = torch.diagonal(dH_eig, dim1=-2, dim2=-1).real.transpose(-1, -2).contiguous()  # (nk,nw,3)
    if nw < 2:
        return del_eig

    eig_np = eig.detach().cpu().numpy()
    gaps = eig_np[:, 1:] - eig_np[:, :-1]              # (nk, nw-1)
    k_idx = np.nonzero((gaps < degen_thresh).any(axis=1))[0]

    for k in k_idx:
        i = 0
        while i < nw:
            j = i
            while j + 1 < nw and gaps[k, j] < degen_thresh:
                j += 1
            if j > i:
                for a in range(3):
                    sub = dH_eig[k, a, i:j + 1, i:j + 1]
                    vals = torch.linalg.eigvalsh(sub)
                    del_eig[k, i:j + 1, a] = vals.real
            i = j + 1
    return del_eig


def hamiltonian_gauge_position(
    H0:        Tensor,
    grad_cart: Tensor,
    A_k:       Tensor,
    degen_thresh: float = 1.0e-7 * _EV_TO_HARTREE,   # 0.1 ueV
    sc_eta: float | None = None,
) -> tuple:
    """
    Rotate the Wannier-gauge position matrix `A_k` (from `position_operator_k`)
    into the H(k) eigenbasis, with the WYSV06 Eq. 24/25 degenerate-subspace
    correction:

        D_h(n,m) = (U^dagger . dH/dk . U)(n,m) / (E_m - E_n)   n != m, |E_m-E_n| > degen_thresh
        D_h(n,m) = 0                                            otherwise (incl. n == m)
        A_H = U^dagger . A_k . U + i * D_h

    Needed by the gyrotropic module's Dw/tildeD and NOA tensors, which use
    the band-pair-resolved `A_H(n,m)` directly (unlike quantities that only
    trace Wannier-gauge operators against an occupation projector).

    `degen_thresh` default (3.6749e-9 Hartree ~ 1e-7 eV) is also used as the
    adjacent-band grouping threshold for `_degenerate_aware_band_velocities`,
    which is always applied (identical to the plain Hellmann-Feynman
    diagonal away from degeneracies).

    `sc_eta` (optional, shift current, IATS18): when given, also returns a
    second, eta-regularized Hamiltonian-gauge position matrix
    `D_h(n,m) = dH_eig(n,m) * Re[1/(E_m-E_n+i*sc_eta)]` (no degeneracy skip
    -- eta regularizes near-degenerate denominators), needed alongside the
    no-eta `D_h` (IATS18 Eq. 30/32/34); returned as an extra 5th value only
    when `sc_eta` is requested, so existing 4-tuple callers are unaffected.

    Args:
      H0       : (nk, nw, nw) complex, Wannier-gauge H(k), Hartree
      grad_cart: (nk, 3, nw, nw) complex, dH/dk_cart, Wannier gauge, Hartree*Bohr
      A_k      : (nk, 3, nw, nw) complex, Wannier-gauge position operator
                 (`position_operator_k`'s first return value), Bohr
      sc_eta   : optional, Hartree -- if given, also return `D_h_eta`

    Returns (eig, UU, del_eig, A_H) normally, or (eig, UU, del_eig, A_H,
    D_h_eta) if `sc_eta` is given:
      eig    : (nk, nw) real, Hartree, ascending
      UU     : (nk, nw, nw) complex, H0's eigenvectors (columns)
      del_eig: (nk, nw, 3) real, Hartree*Bohr -- the (degenerate-aware) band
               velocity dE_n/dk
      A_H    : (nk, 3, nw, nw) complex, Bohr, the Hamiltonian-gauge position
               matrix with the (no-eta) degenerate correction applied
      D_h_eta: (nk, 3, nw, nw) complex, Bohr, the eta-regularized D_h alone
               (not yet added to A_H_eig -- callers needing `AA_bar + i*D_h_eta`
               combine it themselves, matching how the Fortran keeps `D_h`
               and `D_h_no_eta` as separate named quantities throughout)
    """
    eig, UU = torch.linalg.eigh(H0)                                    # (nk,nw), (nk,nw,nw)

    dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)   # (nk,3,nw,nw)
    del_eig = _degenerate_aware_band_velocities(dH_eig, eig, degen_thresh)

    dE = eig[:, None, :] - eig[:, :, None]                              # (nk,nw,nw): E_m - E_n
    nw = eig.shape[-1]
    off_diag = ~torch.eye(nw, dtype=torch.bool, device=eig.device)
    non_degen = off_diag[None, :, :] & (dE.abs() > degen_thresh)
    inv_dE = torch.where(non_degen, 1.0 / torch.where(dE == 0, torch.ones_like(dE), dE),
                         torch.zeros_like(dE))

    A_H_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)    # (nk,3,nw,nw)
    D_h = dH_eig * inv_dE[:, None, :, :].to(dH_eig.dtype)
    A_H = A_H_eig + 1j * D_h

    if sc_eta is None:
        return eig, UU, del_eig, A_H

    denom = dE.to(dH_eig.dtype) + 1j * sc_eta                           # (nk,nw,nw)
    inv_dE_eta = (1.0 / denom).real.to(dH_eig.dtype)                    # Re[1/(E_m-E_n+i*eta)]
    D_h_eta = dH_eig * inv_dE_eta[:, None, :, :] * off_diag[None, None, :, :].to(dH_eig.dtype)
    return eig, UU, del_eig, A_H, D_h_eta


# ---------------------------------------------------------------------------
# Band interpolation
# ---------------------------------------------------------------------------

def operator_k(
    O_R:       Tensor,
    R_vectors: np.ndarray,
    degen:     np.ndarray,
    kpts:      np.ndarray,
    ws=None,
) -> Tensor:
    """
    Fourier-interpolate a real-space operator O(R) to a set of k-points:

        O(k) = Σ_R e^{2πi k·R} O(R) / D(R)

    Shared by band and spin-operator interpolation. If `ws` (a
    `core.ws_distance.WsDistance`) is given, use_ws_distance is applied:
    the plain phase e^{2πi k·R} is replaced by the (i, j)-dependent
    WS-minimal-image phase `ws.phase(k)` (see that module) -- the two
    Wannier functions' centres are accounted for, matching wannier90's
    default `use_ws_distance = .true.`.

    Args:
      O_R      : (nR, nw, nw) complex operator on the H(R) grid
      R_vectors: (nR, 3) int  same grid as O_R
      degen    : (nR,)  Wigner-Seitz degeneracies D(R)
      kpts     : (nk, 3) crystal-coordinate k-points
      ws       : optional WsDistance (use_ws_distance data), or None

    Returns O(k): (nk, nw, nw) complex.
    """
    inv_degen = torch.tensor(1.0 / np.asarray(degen), dtype=O_R.dtype, device=O_R.device)
    if ws is None:
        # Chunked over k, and in torch rather than numpy, for two reasons that
        # both bite hard on the dense meshes the el-ph convergence sweeps use:
        #
        #   * the (nk, nR) phase array is never materialised whole. At a 220^3
        #     mesh against 1957 R-vectors it would be 333 GB;
        #   * numpy's `exp` on a large complex array is SINGLE-threaded whatever
        #     BLAS is set to, while torch's respects `torch.set_num_threads`.
        #     Measured at nk = 2e5, nR = 1957: 17.7 s numpy vs 1.2 s torch, a
        #     14.6x difference on identical arithmetic. This was 100% of the
        #     runtime of a Fermi-surface DOS sweep before the change.
        #
        # The 1/D(R) factor is folded into O_R once instead of riding along in
        # the per-chunk contraction.
        kpts_t = torch.as_tensor(np.ascontiguousarray(kpts, dtype=np.float64),
                                 device=O_R.device)
        R_t = torch.as_tensor(np.asarray(R_vectors, dtype=np.float64), device=O_R.device)
        O_R_scaled = O_R * inv_degen[:, None, None]
        nk, nR = kpts_t.shape[0], R_t.shape[0]
        nw = O_R.shape[-1]
        out = torch.empty((nk, nw, nw), dtype=O_R.dtype, device=O_R.device)
        # ~0.5 GB of complex128 phase per chunk
        chunk = max(1, int(3.0e7 // max(1, nR)))
        two_pi = 2.0 * float(np.pi)
        for s0 in range(0, nk, chunk):
            s1 = min(s0 + chunk, nk)
            ang = torch.matmul(kpts_t[s0:s1], R_t.T).mul_(two_pi)     # (blk, nR) real
            ph = torch.polar(torch.ones_like(ang), ang).to(O_R.dtype)
            out[s0:s1] = torch.tensordot(ph, O_R_scaled, dims=([1], [0]))
        return out

    # use_ws_distance: (i, j)-dependent phase, per k (ws.phase is (nR, nw, nw))
    nk = len(kpts)
    nw = O_R.shape[-1]
    O_k = torch.empty((nk, nw, nw), dtype=O_R.dtype, device=O_R.device)
    for ik in range(nk):
        wsph = torch.tensor(ws.phase(kpts[ik]), dtype=O_R.dtype, device=O_R.device)   # (nR,nw,nw)
        O_k[ik] = torch.einsum("r,rmn,rmn->mn", inv_degen, wsph, O_R)
    return O_k


def interpolate_bands(
    hr:   HamiltonianR,
    kpts: np.ndarray,
    ws="auto",
) -> np.ndarray:
    """
    Interpolate band energies on an arbitrary k-path.

    H(k) = Σ_R e^{2πi k·R} H(R) / D(R)
    Eigenvalues of H(k) are the interpolated band energies.

    Args:
      hr  : HamiltonianR from compute_hr
      kpts: (nk_interp, 3) float64  k-points in crystal coordinates
      ws  : "auto" (default) uses the model's own `hr.ws_distance()` when its
            Wannier centres are known, i.e. use_ws_distance ON, matching
            wannier90's default. Pass an explicit `WsDistance` to override, or
            None to force the plain Wigner-Seitz sum. Turning it off is a
            correctness regression, not a speed/accuracy trade: see
            `HamiltonianR`.

    Returns:
      bands: (nk_interp, nw) float64  eigenvalues in Hartree (sorted ascending)
    """
    if isinstance(ws, str):
        if ws != "auto":
            raise ValueError(f"ws must be 'auto', None, or a WsDistance; got {ws!r}")
        ws = hr.ws_distance()
        if ws is None:
            # "auto" degrading to the plain WS sum is a correctness regression,
            # so say so rather than silently returning worse bands. It happens
            # whenever the model lacks centres/real_lattice/mp_grid -- e.g. a
            # SIESTA model loaded without centres="atomic". Pass ws=None to
            # choose the plain sum deliberately and silence this.
            warnings.warn(
                "interpolate_bands(ws='auto'): this model has no Wannier "
                "centres, real_lattice or mp_grid, so the Wigner-Seitz "
                "distance correction (wannier90's use_ws_distance, ON by "
                "default there) is NOT applied. Off-mesh bands will be worse. "
                "Give the model centres, or pass ws=None to say you meant the "
                "plain sum.", RuntimeWarning, stacklevel=2)
    H_k = operator_k(hr.H_R, hr.R_vectors, hr.degen, kpts, ws=ws)   # (nk_interp, nw, nw)

    # Diagonalize (H_k is Hermitian up to numerical noise; use eigh)
    H_k_herm = (H_k + H_k.conj().transpose(-1, -2)) / 2
    bands, _ = torch.linalg.eigh(H_k_herm)   # (nk_interp, nw) real

    return bands.real.detach().cpu().numpy()
