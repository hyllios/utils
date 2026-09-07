"""
Spread functional for maximally-localized Wannier functions.

Implements the Marzari-Vanderbilt spread functional (PRB 56, 12847, 1997):

    Omega   = Omega_I + Omega_D + Omega_OD
    Omega_I = (1/Nk) sum_{k,b} w_b [ Nw - sum_n |M_tilde_nn^{k,b}|^2 ]
    Omega_D = (1/Nk) sum_{k,b} w_b sum_n [ -Im(ln M_tilde_nn^{k,b}) - b.<r>_n ]^2
    Omega_OD= (1/Nk) sum_{k,b} w_b sum_{m!=n} |M_tilde_mn^{k,b}|^2

where M_tilde^{k,b} = U^(k)† M^{k,b} U^{k+b} are the rotated overlap matrices.

The function is written as pure PyTorch so that autograd can differentiate
through it with respect to U.  No gradients are implemented by hand.
"""

import torch
from torch import Tensor


def rotate_overlaps(
    U: Tensor,
    Mmn: Tensor,
    kb_idx: Tensor,
) -> Tensor:
    """
    Compute rotated overlap matrices M_tilde^{k,b} = U^(k)† M^{k,b} U^{k+b}.

    Args:
      U      : (nk, nw, nw)       complex  unitary gauge matrices
      Mmn    : (nk, nnb, nb, nb)  complex  raw overlap matrices from DFT
      kb_idx : (nk, nnb)          long     index of k+b for each (k, b) pair

    Returns:
      M_tilde: (nk, nnb, nw, nw)  complex  rotated overlaps

    When there is no disentanglement (nb == nw), U acts directly on the full
    band space.  With disentanglement U is (nk, nw, nw) in the subspace
    already selected by V; M^{k,b} has already been projected: M = V† Mmn V.
    """
    nk, nnb = kb_idx.shape

    U_kb = U[kb_idx]                                          # (nk, nnb, nw, nw)
    U_k  = U.unsqueeze(1).expand(-1, nnb, -1, -1)            # (nk, nnb, nw, nw)
    M_tilde = torch.matmul(U_k.conj().transpose(-1, -2), Mmn)
    M_tilde = torch.matmul(M_tilde, U_kb)
    return M_tilde   # (nk, nnb, nw, nw)


def weight_overlaps_by_eigenvalues(Mmn: Tensor, eig: Tensor) -> Tensor:
    """
    Left-multiply each overlap block's bra (band) index by the eigenvalue
    at that k-point: Mmn_weighted[k,b,m,n] = eig[k,m] * Mmn[k,b,m,n].

    Building block for BB(R) = <0n|H(r-R)|Rm> (`core.hamiltonian.compute_bb_r`).
    Feed the result through the same two `rotate_overlaps` calls used to
    build `M_tilde` (V then U_final) to get `compute_bb_r`'s `H_tilde`.

    Args:
      Mmn : (nk, nnb, nb, nb) complex  raw overlaps from DFT
      eig : (nk, nb)          real     eigenvalues (Hartree), same band
            window/ordering as Mmn's bra index

    Returns (nk, nnb, nb, nb) complex.
    """
    return eig.to(Mmn.dtype)[:, None, :, None] * Mmn


def weight_overlaps_by_operator(Mmn: Tensor, O_bloch: Tensor) -> Tensor:
    """
    Left-multiply each overlap block's bra (band) index by an arbitrary
    per-k operator matrix:

        Mmn_weighted[k,b,m,n] = sum_p O_bloch[k,m,p] * Mmn[k,b,p,n]

    Generalizes `weight_overlaps_by_eigenvalues` (diagonal O_bloch =
    diag(eig)) to non-diagonal operators such as the `.spn` Pauli matrices,
    needed for spin Hall conductivity SR(R)/SHR(R)/SH(R) (postw90
    `berry_task = eval_shc`, `shc_method = qiao`; QZYZ18).

    Feed the result through the same two `rotate_overlaps` calls used to
    build `M_tilde` (V then U_final) to get a Wannier-gauge tilde array for
    `compute_position_r`/`compute_bb_r` (see `analysis.spin_hall`).

    Args:
      Mmn     : (nk, nnb, nb, nb) complex  raw overlaps from DFT
      O_bloch : (nk, nb, nb)      complex  operator (e.g. one Cartesian/
                Pauli component of `.spn`), same band window/ordering as
                Mmn's bra index

    Returns (nk, nnb, nb, nb) complex.
    """
    return torch.einsum('kmp,kbpn->kbmn', O_bloch.to(Mmn.dtype), Mmn)


def _guided_phase(M_diag: Tensor, bvecs: Tensor, rguide: Tensor) -> Tensor:
    """
    Branch-consistent replacement for ``torch.angle(M_diag)``, using a
    per-WF guiding-centre reference to resolve the 2*pi ambiguity in
    Im(ln M_nn) -- wannier90's ``guiding_centres`` mechanism.

    ``torch.angle`` is confined to (-pi, pi], so once a WF's true
    accumulated phase b.r_n exceeds that range, the naive principal-branch
    phase can no longer distinguish "correct position" from "off by one
    lattice vector" and Omega_D balloons (the MLWF "runaway centre"
    pathology). Pre-rotating by the expected phase before calling
    ``angle`` and adding it back recovers the correct branch:

        sheet[k,b,n] = bvecs[k,b] . rguide[n]
        phase[k,b,n] = angle(M_diag[k,b,n] * exp(i*sheet[k,b,n])) - sheet[k,b,n]

    Recovers plain ``torch.angle(M_diag)`` when rguide is exactly 0.
    ``rguide`` must be a fixed reference (no grad) -- refresh it
    periodically via ``refine_guiding_centres``, not every step, and never
    backprop through it.

    Args:
      M_diag: (nk, nnb, nw) complex  diagonal of the rotated overlaps
      bvecs : (nk, nnb, 3)  real     Cartesian b-vectors (Bohr^-1)
      rguide: (nw, 3)       real     guiding centres (Bohr)

    Returns:
      phase: (nk, nnb, nw) real, unwrapped
    """
    sheet = torch.einsum("kba,na->kbn", bvecs, rguide)
    shifted = M_diag * torch.polar(torch.ones_like(sheet), sheet)
    return torch.angle(shifted) - sheet


def refine_guiding_centres(M_diag: Tensor, bvecs: Tensor, rguide_init: Tensor) -> Tensor:
    """
    Sequential branch-consistent least-squares refinement of the guiding
    centres (wannier90's ``wann_phases``): process each (k, b) equation in
    turn, picking the branch of Im(ln M_nn) closest to the running rguide
    estimate (an arbitrary branch for the first 3, needed to pin down a
    non-degenerate frame), refitting rguide by least squares once >= 3
    linearly-independent b-vectors are available.

    The first 3 equations always take M_nn's own principal-branch phase,
    ignoring ``rguide_init`` -- so correctness needs |b.r_true| < pi for
    those. This is preventive rather than correcting an arbitrary jump:
    call frequently so drift never accumulates that far, with
    ``rguide_init`` the previous cycle's estimate.

    Args:
      M_diag     : (nk, nnb, nw) complex  diagonal of the rotated overlaps
      bvecs      : (nk, nnb, 3)  real     Cartesian b-vectors (Bohr^-1)
      rguide_init: (nw, 3)       real     starting guess (Bohr), used only
                   to pick branches for the 4th+ equation of each WF
    Returns:
      rguide: (nw, 3) real, refined guiding centres (Bohr)
    """
    nk, nnb, nw = M_diag.shape
    b_flat = bvecs.reshape(nk * nnb, 3)
    M_flat = M_diag.reshape(nk * nnb, nw)
    n_eq = b_flat.shape[0]

    rguide = rguide_init.clone()
    for n in range(nw):
        smat = torch.zeros(3, 3, dtype=bvecs.dtype, device=bvecs.device)
        svec = torch.zeros(3, dtype=bvecs.dtype, device=bvecs.device)
        r = rguide[n].clone()
        for idx in range(n_eq):
            b = b_flat[idx]
            Mv = M_flat[idx, n]
            if idx < 3:
                xx = -torch.angle(Mv)
            else:
                xx0 = torch.dot(b, r)
                xx = xx0 - torch.angle(Mv * torch.exp(1j * xx0))
            smat = smat + torch.outer(b, b)
            svec = svec + b * xx
            if idx >= 2:
                det = torch.det(smat)
                if torch.abs(det) > 1e-6:
                    r = torch.linalg.solve(smat, svec)
        rguide[n] = r
    return rguide


def _spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute all spread components from pre-rotated overlaps (single pass).

    M_diag and phase are extracted once and shared between centres, Omega_I,
    Omega_OD, and Omega_D — no redundant diagonal extraction.

    Args:
      M_tilde: (nk, nnb, nw, nw)  complex  rotated overlaps
      wb     : (nnb,)              real     shell weights
      bvecs  : (nk, nnb, 3)       real     Cartesian b-vectors (Bohr^-1)
      rguide : (nw, 3) real, optional  guiding centres (Bohr).  When given,
               use the branch-consistent ``_guided_phase`` instead of the
               naive ``torch.angle`` (prevents MLWF runaway centres; see
               ``_guided_phase``).  None reproduces the original formula.

    Returns:
      (Omega, Omega_I, Omega_D, Omega_OD, centres)
    """
    nk = M_tilde.shape[0]
    nw = M_tilde.shape[-1]

    M_diag      = torch.diagonal(M_tilde, dim1=-2, dim2=-1)   # (nk, nnb, nw)
    M_diag_abs2 = M_diag.abs().pow(2)
    phase       = _guided_phase(M_diag, bvecs, rguide) if rguide is not None else torch.angle(M_diag)

    # Wannier centres: -(1/Nk) sum_{k,b} w_b b^{k,b} Im(ln M_nn)
    weighted_phase = wb[None, :, None] * phase                 # (nk, nnb, nw)
    centres = -torch.einsum("kbn,kba->na", weighted_phase, bvecs) / nk

    # Omega_I
    M_frob2  = M_tilde.abs().pow(2).sum(dim=(-1, -2))         # (nk, nnb)
    Omega_I  = torch.einsum("b,kb->", wb, nw - M_frob2) / nk

    # Omega_OD
    Omega_OD = torch.einsum("b,kb->", wb, M_frob2 - M_diag_abs2.sum(-1)) / nk

    # Omega_D
    b_dot_r  = torch.einsum("kba,na->kbn", bvecs, centres)    # (nk, nnb, nw)
    residual = -phase - b_dot_r
    Omega_D  = torch.einsum("b,kbn->", wb, residual.pow(2)) / nk

    return Omega_I + Omega_D + Omega_OD, Omega_I, Omega_D, Omega_OD, centres


def _canonical_b_permutation(bvecs: Tensor) -> Tensor:
    """
    For each k, find the local b-index matching bvecs[0]'s (canonical)
    direction ordering.

    Every k-point sees the same set of physical b-vectors, just permuted
    per k (see `core.kmesh`). Ordinary Omega_I/Omega_D/Omega_OD don't care,
    but the Stengel-Spaldin Omega_D (`_ss_spread_from_M_tilde`) averages
    M_nn across k for a fixed physical b-vector, so the permutation must
    be undone first (wannier90's `nnord` table does the same reindexing).

    Returns perm: (nk, nnb) long, s.t. bvecs[k, perm[k, s]] == bvecs[0, s]
    for every k (matched by nearest Cartesian vector).
    """
    nk = bvecs.shape[0]
    ref = bvecs[0].unsqueeze(0).expand(nk, -1, -1)   # (nk, nnb, 3)
    dist = torch.cdist(ref, bvecs)                    # (nk, nnb_ref, nnb_local)
    return dist.argmin(dim=-1)                        # (nk, nnb)


def _ss_spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Stengel-Spaldin alternative localization functional (PRB 73, 075121,
    2006; wannier90's `use_ss_functional`). Omega_I/Omega_OD are identical
    to the ordinary MV formula; only Omega_D differs, as a k-averaged
    variance of the diagonal overlap per orbital/shell (no branch-cut log
    needed):

        Omega_D = sum_n sum_b wb(b) * [<|M_nn(k,b)|^2>_k - |<M_nn(k,b)>_k|^2]

    Since |<X>|^2 <= <|X|^2> (Jensen/Cauchy-Schwarz), this term is
    systematically <= the ordinary MV Omega_D for identical overlaps.

    M_nn must be reindexed to a canonical b-vector ordering before
    averaging over k (`_canonical_b_permutation`). Does not return
    centres -- use `compute_wannier_centres` if needed.

    Args:
      M_tilde: (nk, nnb, nw, nw)  complex  rotated overlaps
      wb     : (nnb,)              real     shell weights (k=0 ordering)
      bvecs  : (nk, nnb, 3)       real     Cartesian b-vectors, k-specific (Bohr^-1)

    Returns:
      (Omega, Omega_I, Omega_D, Omega_OD)
    """
    nk = M_tilde.shape[0]
    nw = M_tilde.shape[-1]

    M_diag      = torch.diagonal(M_tilde, dim1=-2, dim2=-1)   # (nk, nnb, nw)
    M_diag_abs2 = M_diag.abs().pow(2)
    M_frob2     = M_tilde.abs().pow(2).sum(dim=(-1, -2))       # (nk, nnb)

    Omega_I  = torch.einsum("b,kb->", wb, nw - M_frob2) / nk
    Omega_OD = torch.einsum("b,kb->", wb, M_frob2 - M_diag_abs2.sum(-1)) / nk

    perm = _canonical_b_permutation(bvecs)                                     # (nk, nnb)
    M_diag_c = torch.gather(M_diag, 1, perm.unsqueeze(-1).expand(-1, -1, nw))  # (nk, nnb, nw)

    mean_M    = M_diag_c.mean(dim=0)             # (nnb, nw) complex
    mean_abs2 = M_diag_c.abs().pow(2).mean(dim=0)  # (nnb, nw) real
    variance  = mean_abs2 - mean_M.abs().pow(2)    # (nnb, nw) real, >= 0

    Omega_D = torch.einsum("b,bn->", wb, variance)

    return Omega_I + Omega_D + Omega_OD, Omega_I, Omega_D, Omega_OD


def compute_wannier_centres(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
) -> Tensor:
    """
    Compute Wannier centres <r>_n from the rotated overlap matrices.

    The MV formula (Eq. 31):
        <r>_n = -(1/Nk) sum_{k,b} w_b * b^{k,b} * Im(ln M_tilde_nn^{k,b})

    The b-vectors are k-dependent (the index ordering in the .nnkp file differs
    per k-point on a folded BZ), so bvecs must be the full (nk, nnb, 3) array.

    Args:
      M_tilde: (nk, nnb, nw, nw)  complex  rotated overlaps
      wb     : (nnb,)              real     shell weights
      bvecs  : (nk, nnb, 3)       real     Cartesian b-vectors, k-specific (Bohr^-1)

    Returns:
      centres: (nw, 3)  real  Wannier centres in Bohr.  See MV97 Eq. 31.
    """
    nk    = M_tilde.shape[0]
    phase = torch.angle(torch.diagonal(M_tilde, dim1=-2, dim2=-1))  # (nk, nnb, nw)
    return -torch.einsum("kbn,kba->na", wb[None, :, None] * phase, bvecs) / nk


def compute_spread(
    U: Tensor,
    Mmn: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    kb_idx: Tensor,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the MV spread functional and its three components.

    This function is fully differentiable: torch.autograd.grad(Omega, U)
    gives the Euclidean gradient needed for Stiefel-manifold optimization.

    Args:
      U      : (nk, nw, nw)       complex  unitary gauge matrices (the var)
      Mmn    : (nk, nnb, nb, nb)  complex  overlap matrices
      wb     : (nnb,)             real     shell weights
      bvecs  : (nk, nnb, 3)       real     Cartesian b-vectors, k-specific
      kb_idx : (nk, nnb)          long     neighbour k-index table
      rguide : (nw, 3) real, optional  guiding centres (Bohr); see
               ``_guided_phase``/``refine_guiding_centres``.

    Returns:
      Omega, Omega_I, Omega_D, Omega_OD, centres (nw, 3)

    Note on units: bvecs in Bohr^-1 → spread in Bohr^2.
    """
    return _spread_from_M_tilde(rotate_overlaps(U, Mmn, kb_idx), wb, bvecs, rguide)


def compute_spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute spread components from pre-rotated overlaps (skips rotate_overlaps).

    Use when M_tilde = rotate_overlaps(U, Mmn, kb_idx) has already been built
    to avoid the redundant batched matmul.  Same return signature as compute_spread.
    """
    return _spread_from_M_tilde(M_tilde, wb, bvecs, rguide)


def compute_ss_spread(
    U: Tensor,
    Mmn: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    kb_idx: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the Stengel-Spaldin spread functional and its components (see
    `_ss_spread_from_M_tilde`). Fully differentiable, same autodiff usage
    as `compute_spread`.

    Returns (Omega, Omega_I, Omega_D, Omega_OD) -- no `centres` (unlike
    `compute_spread`); use `compute_wannier_centres` separately if wanted.
    """
    return _ss_spread_from_M_tilde(rotate_overlaps(U, Mmn, kb_idx), wb, bvecs)


def compute_ss_spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SS spread components from pre-rotated overlaps (skips
    rotate_overlaps). Same return signature as `compute_ss_spread`.
    """
    return _ss_spread_from_M_tilde(M_tilde, wb, bvecs)


def compute_pm_spread(
    U: Tensor,
    Aat: Tensor,
    atom_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Pipek-Mezey localization functional (Pipek & Mezey, J. Chem. Phys. 90,
    4916 (1989)), adapted to periodic Wannier functions: MAXIMIZE the sum
    over Wannier functions of the sum of squared Mulliken atomic charges --
    minimized here as its negative, Omega_PM = -sum_n sum_A Q_A[n]^2, with

        Q_A[n] = sum_{i in A} |<phi_i|w_n>|^2,
        <phi_i|w_n> = (1/Nk) sum_k sum_m U_mn(k) * conj(Aat[k,m,i])

    (the conjugate on Aat, not on U, matches wannier90's own A_mn(k) =
    <psi_mk|trial_n> .amn convention -- the same one `svd_init` already
    relies on -- verified against real converged Si data: the diagonal
    self-overlap |<phi_n|w_n>| comes out large and uniform across the 4
    symmetric sp3 bonds with this sign, small and asymmetric with the
    other).

    ``Aat`` : (nk, nw, n_orbitals) complex, the (fixed, U-independent)
    overlap of each band in the already-disentangled nw-dimensional
    subspace with each atomic pseudo-orbital (pw2wannier90's
    ``atom_proj``/``atom_proj_ext``, read the same way as an ordinary
    ``.amn`` and projected into the Wannier-gauge subspace exactly like
    `rotate_overlaps` projects `Mmn` when disentangling).
    ``atom_index`` : (n_orbitals,) long, the 0-based atom each orbital
    column belongs to (`interfaces.quantum_espresso.upf.atom_proj_column_atoms`).

    Unlike the Marzari-Vanderbilt/Stengel-Spaldin functionals, this has no
    Omega_I/Omega_D/Omega_OD decomposition (it isn't a spread at all -- it
    doesn't even reference Mmn/bvecs) and no branch-cut phase ambiguity (no
    logarithm anywhere), so `guiding_centres` is meaningless for it.

    Returns (Omega_PM, Q) -- Q is (nw, n_atoms) real, the Mulliken charges
    themselves (useful for inspection/plotting; sums to <= 1 per Wannier
    function in general, not exactly 1, since the atomic pseudo-orbitals
    are not a complete basis).
    """
    nk = U.shape[0]
    proj = torch.einsum("kmn,kmi->ni", U, Aat.conj()) / nk   # (nw, n_orbitals)
    charge = proj.abs().pow(2)
    n_atoms = int(atom_index.max().item()) + 1
    Q = torch.zeros(charge.shape[0], n_atoms, dtype=torch.float64, device=U.device)
    Q.index_add_(1, atom_index, charge)
    Omega_PM = -Q.pow(2).sum()
    return Omega_PM, Q


def _slwf_spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    slwf_num: int,
    constrain: bool = False,
    target_centres: Tensor | None = None,
    lambda_: float = 1.0,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Selectively-localized Wannier functions (SLWF, `slwf_num`/
    `slwf_constrain`/`slwf_lambda`), from Wang, Lazar, Park, Millis &
    Marianetti, arXiv:1407.5124, Eq. 9-13 (plain) / Eq. 24, 29-31
    (`constrain=True`).

    Only the first `slwf_num` of `nw` Wannier functions ("objective" WFs)
    are localized; the remaining "spectator" WFs are excluded from Omega
    (they still couple in through the unitary gauge but their spread is
    never minimized). `slwf_num = nw` recovers plain MLWF
    (`compute_spread`) exactly.

        Omega_IOD = (1/Nk) sum_{n<J'} sum_{k,b} wb * (1 - |M_nn|^2 [+ lambda*(Im ln M_nn)^2 if constrain])
        Omega_D   = (1-lambda if constrain else 1)/Nk * sum_{n<J'} sum_{k,b} wb * (Im ln M_nn + b.r_n)^2
        Omega_nu  = lambda * sum_{n<J'} r0n^2 + (2*lambda/Nk) * sum_{k,b} wb * sum_{n<J'} (b.r0n) * Im ln M_nn
                    (0 when constrain=False)

    `lambda` is only active when `constrain=True`.

    Args:
      M_tilde       : (nk, nnb, nw, nw)  complex  rotated overlaps
      wb            : (nnb,)              real     shell weights
      bvecs         : (nk, nnb, 3)        real     Cartesian b-vectors (Bohr^-1)
      slwf_num      : J' <= nw, number of objective Wannier functions
      constrain     : whether to add the centre-constraint penalty
      target_centres: (J', 3) real, Bohr -- desired centres r0n for the OWFs
                      (required when constrain=True)
      lambda_       : Lagrange multiplier (wannier90's `slwf_lambda`, default 1.0)
      rguide        : optional guiding centres, same convention as `compute_spread`

    Returns (Omega, Omega_IOD, Omega_D, Omega_nu, centres): the first four
    scalars, `centres` (nw, 3) real -- computed for ALL nw WFs (the usual MV
    centre formula), even though only the first `slwf_num` enter Omega.
    """
    if constrain and target_centres is None:
        raise ValueError("target_centres is required when constrain=True")

    M_diag      = torch.diagonal(M_tilde, dim1=-2, dim2=-1)   # (nk, nnb, nw)
    M_diag_abs2 = M_diag.abs().pow(2)
    phase       = _guided_phase(M_diag, bvecs, rguide) if rguide is not None else torch.angle(M_diag)

    nk = M_tilde.shape[0]
    weighted_phase = wb[None, :, None] * phase
    centres = -torch.einsum("kbn,kba->na", weighted_phase, bvecs) / nk

    phase_a  = phase[:, :, :slwf_num]            # (nk, nnb, J')
    m2_a     = M_diag_abs2[:, :, :slwf_num]
    lam      = lambda_ if constrain else 0.0

    Omega_IOD = torch.einsum("b,kbn->", wb, 1.0 - m2_a + lam * phase_a.pow(2)) / nk

    b_dot_r  = torch.einsum("kba,na->kbn", bvecs, centres[:slwf_num])   # (nk, nnb, J')
    residual = -phase_a - b_dot_r
    Omega_D  = (1.0 - lam) * torch.einsum("b,kbn->", wb, residual.pow(2)) / nk

    if constrain:
        b_dot_r0 = torch.einsum("kba,na->kbn", bvecs, target_centres)   # (nk, nnb, J')
        Omega_nu = (
            lambda_ * target_centres.pow(2).sum()
            + 2.0 * lambda_ * torch.einsum("b,kbn->", wb, b_dot_r0 * phase_a) / nk
        )
    else:
        Omega_nu = torch.zeros((), dtype=Omega_IOD.dtype, device=Omega_IOD.device)

    Omega = Omega_IOD + Omega_D + Omega_nu
    return Omega, Omega_IOD, Omega_D, Omega_nu, centres


def compute_slwf_spread(
    U: Tensor,
    Mmn: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    kb_idx: Tensor,
    slwf_num: int,
    constrain: bool = False,
    target_centres: Tensor | None = None,
    lambda_: float = 1.0,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """SLWF counterpart of `compute_spread` -- see `_slwf_spread_from_M_tilde`.
    Fully differentiable: torch.autograd.grad(Omega, U) gives the Euclidean
    gradient (no hand-derived gradient formula needed, same as plain MLWF)."""
    return _slwf_spread_from_M_tilde(
        rotate_overlaps(U, Mmn, kb_idx), wb, bvecs, slwf_num, constrain, target_centres, lambda_, rguide,
    )


def compute_slwf_spread_from_M_tilde(
    M_tilde: Tensor,
    wb: Tensor,
    bvecs: Tensor,
    slwf_num: int,
    constrain: bool = False,
    target_centres: Tensor | None = None,
    lambda_: float = 1.0,
    rguide: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Skip-rotation counterpart of `compute_slwf_spread`, mirroring
    `compute_spread_from_M_tilde`."""
    return _slwf_spread_from_M_tilde(M_tilde, wb, bvecs, slwf_num, constrain, target_centres, lambda_, rguide)
