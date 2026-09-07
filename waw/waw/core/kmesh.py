"""
k-mesh geometry: finite-difference b-vectors and shell weights.

The b-vectors are the Cartesian displacements from each k-point to its
nearest neighbours in the extended Brillouin zone.  Their weights w_b must
satisfy the completeness relation:

    sum_b  w_b * b_alpha * b_beta  =  delta_{alpha,beta}

(Eq. 11 in Marzari & Vanderbilt, PRB 56, 12847 (1997)).  This is the
finite-difference approximation to the identity operator on gradient space,
and it is the key numerical condition that makes the spread functional
well-defined.

Pure numerics in atomic units: recip_lattice in Bohr^-1 in, b-vectors in
Bohr^-1 out.  No file I/O, no physical-unit conversions.
"""

from __future__ import annotations

import numpy as np


def _compute_bvecs_and_weights(
    kpts: np.ndarray,
    nnkpts: np.ndarray,
    g_vectors: np.ndarray,
    recip_lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Cartesian b-vectors and their finite-difference shell weights.

    The b-vectors connect each k-point to its neighbours in the extended BZ:
        b^(k,ib) = k_{k+b} + G_{k,ib} - k_k
    in crystal coordinates, then converted to Cartesian (Bohr^-1) by
    multiplying by the reciprocal lattice matrix.

    On a regular MP mesh the b-vectors at different k are a permutation of the
    same fixed set — but the index ordering (which ib corresponds to which
    direction) differs per k. Returns the full (nk, nnb, 3) array so the
    spread functional can pair each M_{mn}(k, ib) with the correct Cartesian
    direction.

    Shell weights w_b are determined by solving the completeness condition:
        sum_b  w_b * b_alpha * b_beta  =  delta_{alpha,beta}
    per unique shell.  Because every k-point sees the same set of directions
    (just permuted), a single weight per index (using k=0 ordering) satisfies
    the condition at every k when the spread sums over all b.

    Args:
      kpts         : (nk, 3)       k-points in crystal coordinates
      nnkpts       : (nk, nnb)     neighbour k-index table (0-based)
      g_vectors    : (nk, nnb, 3)  G-vector folding back into first BZ
      recip_lattice: (3, 3)        reciprocal lattice vectors as rows (Bohr^-1)

    Returns:
      bvecs: (nk, nnb, 3)  Cartesian b-vectors, k-point specific
      wb   : (nnb,)        shell weights (same ordering as k=0 b-vectors)
    """
    nk, nnb = nnkpts.shape

    # b-vectors in crystal coordinates: b[k,b] = kpts[k+b] + G[k,b] - kpts[k]
    b_cryst = kpts[nnkpts] + g_vectors - kpts[:, None, :]   # (nk, nnb, 3)

    # Convert to Cartesian: b_cart = b_cryst @ recip_lattice
    bvecs = b_cryst @ recip_lattice   # (nk, nnb, 3)

    # Determine unique shells from k=0 (representative; all k have same set).
    bvecs_k0  = bvecs[0]                             # (nnb, 3)
    b_lengths = np.linalg.norm(bvecs_k0, axis=1)
    unique_lengths = np.unique(np.round(b_lengths, decimals=6))

    # Build the completeness-condition linear system.
    alpha_beta_pairs = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    n_shells = len(unique_lengths)
    A   = np.zeros((6, n_shells), dtype=np.float64)
    rhs = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])   # delta_{alpha,beta}

    for s, length in enumerate(unique_lengths):
        shell_mask = np.abs(b_lengths - length) < 1e-5
        shell_b    = bvecs_k0[shell_mask]
        for row, (a, b) in enumerate(alpha_beta_pairs):
            A[row, s] = np.sum(shell_b[:, a] * shell_b[:, b])

    weights_per_shell, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)

    if rank < n_shells:
        raise ValueError(
            f"Shell weight matrix is rank-deficient ({rank} < {n_shells}). "
            "The b-vector shells do not satisfy the completeness condition — "
            "check that .nnkp was generated with enough neighbour shells."
        )
    residual_norm = float(np.linalg.norm(A @ weights_per_shell - rhs))
    if residual_norm > 1e-6:
        raise ValueError(
            f"Shell weight completeness condition not satisfied "
            f"(residual = {residual_norm:.2e}). "
            "Check .nnkp neighbour shells."
        )

    # Assign the per-shell weight to each b-vector index (k=0 ordering).
    wb = np.zeros(nnb, dtype=np.float64)
    for s, length in enumerate(unique_lengths):
        shell_mask      = np.abs(b_lengths - length) < 1e-5
        wb[shell_mask]  = weights_per_shell[s]

    return bvecs, wb


def _shell_completeness_matrix(shell_bvecs_cart: list[np.ndarray]) -> np.ndarray:
    """
    Build the (6, n_shells) completeness matrix A whose columns are
    [Σ b_x², Σ b_y², Σ b_z², Σ b_x b_y, Σ b_x b_z, Σ b_y b_z] summed over the
    b-vectors of each shell.  The completeness condition is A w = [1,1,1,0,0,0].
    """
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    A = np.zeros((6, len(shell_bvecs_cart)), dtype=np.float64)
    for s, shell in enumerate(shell_bvecs_cart):
        for row, (a, b) in enumerate(pairs):
            A[row, s] = np.sum(shell[:, a] * shell[:, b])
    return A


def generate_nnkp(
    kpts:          np.ndarray,
    recip_lattice: np.ndarray,
    mp_grid:       tuple[int, int, int],
    *,
    supercell:          int   = 4,
    search_shells:      int   = 36,
    length_tol:         float = 1e-6,
    completeness_tol:   float = 1e-6,
) -> dict:
    """
    Compute the k-point neighbour topology from scratch — the job of
    ``wannier90.x -pp`` — so no external Wannier90 call is needed.

    Given a Γ-centred Monkhorst-Pack mesh and the reciprocal lattice, this finds
    the finite-difference b-vector shells that satisfy the completeness relation
    (Σ_b w_b b_α b_β = δ_αβ, Marzari-Vanderbilt Eq. 11) by adding shells of
    increasing |b| until the condition is met (skipping shells linearly
    dependent on those already chosen), then builds, for every k-point, the
    neighbour k-index and folding G-vector of each selected b-vector.

    The shell *selection* depends only on b-vector length ratios and directions,
    so ``recip_lattice`` may be in any consistent unit (the result is identical
    in Bohr^-1 or Angstrom^-1).

    Parameters
    ----------
    kpts : (nk, 3) float64
        k-points in crystal coordinates on a Γ-centred m/N mesh.
    recip_lattice : (3, 3) float64
        Reciprocal lattice rows (2π convention).
    mp_grid : (N1, N2, N3)
        The Monkhorst-Pack grid the k-points lie on.
    supercell : int
        Half-width of the integer neighbour search box (db_i ∈ [-supercell,
        supercell]); enlarge for meshes whose first shells are far out.
    search_shells : int
        Maximum number of distinct-length shells to consider.

    Returns
    -------
    dict with the same shape as ``read_nnkp``'s output:
      kpoints  : (nk, 3)        the input k-points
      nnkpts   : (nk, nntot)    neighbour k-index table (0-based)
      g_vectors: (nk, nntot, 3) folding G-vectors
      nntot    : int            neighbours per k-point
    """
    kpts  = np.asarray(kpts, dtype=np.float64)
    recip = np.asarray(recip_lattice, dtype=np.float64)
    N     = tuple(int(x) for x in mp_grid)
    nk    = len(kpts)

    # --- candidate b-vectors: displacements db_i / N_i to other mesh points ---
    rng = range(-supercell, supercell + 1)
    dbs = np.array([(d1, d2, d3) for d1 in rng for d2 in rng for d3 in rng
                    if (d1, d2, d3) != (0, 0, 0)], dtype=np.int64)
    b_cryst = dbs / np.array(N, dtype=np.float64)          # (M, 3)
    b_cart  = b_cryst @ recip                              # (M, 3)
    lengths = np.linalg.norm(b_cart, axis=1)

    # --- group candidates into shells of equal |b|, increasing length ---
    unique_lengths = np.unique(np.round(lengths, decimals=6))
    shells_db   = []   # integer db-vectors per shell
    shells_cart = []   # Cartesian b-vectors per shell
    for L in unique_lengths[:search_shells]:
        mask = np.abs(lengths - L) < length_tol
        shells_db.append(dbs[mask])
        shells_cart.append(b_cart[mask])

    # --- greedily add shells until the completeness condition holds ---
    rhs = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    accepted_cart: list[np.ndarray] = []
    accepted_db:   list[np.ndarray] = []
    for shell_cart, shell_db in zip(shells_cart, shells_db):
        trial = accepted_cart + [shell_cart]
        A = _shell_completeness_matrix(trial)
        w, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
        if rank < len(trial):
            continue   # this shell is linearly dependent on the accepted ones
        accepted_cart = trial
        accepted_db.append(shell_db)
        if np.linalg.norm(A @ w - rhs) < completeness_tol:
            break
    else:
        raise ValueError(
            "Could not satisfy the b-vector completeness condition within "
            f"{len(shells_db)} shells (supercell={supercell}). Increase "
            "`supercell` or `search_shells`."
        )

    selected_db = np.concatenate(accepted_db, axis=0)   # (nntot, 3) int
    nntot = len(selected_db)

    # --- neighbour table: exact integer lookup on the m/N grid ---
    index_map = {
        tuple(int(round(kpts[j, d] * N[d])) % N[d] for d in range(3)): j
        for j in range(nk)
    }
    nnkpts    = np.zeros((nk, nntot), dtype=np.int64)
    g_vectors = np.zeros((nk, nntot, 3), dtype=np.int64)
    for ik in range(nk):
        k = kpts[ik]
        for ib, db in enumerate(selected_db):
            kpb = k + db / np.array(N, dtype=np.float64)
            key = tuple(int(round(kpb[d] * N[d])) % N[d] for d in range(3))
            ik2 = index_map[key]
            nnkpts[ik, ib]    = ik2
            g_vectors[ik, ib] = np.round(kpb - kpts[ik2]).astype(np.int64)

    return {
        "kpoints":   kpts,
        "nnkpts":    nnkpts,
        "g_vectors": g_vectors,
        "nntot":     nntot,
    }


def _nnkp_from_mmn(
    kpb_map: list,
    win_params: dict,
    nk: int,
    nnb: int,
) -> dict:
    """
    Reconstruct the nnkp dict from the .mmn k-pair map and .win k-points.

    Used as a fallback when no .nnkp file is present.  The .mmn file contains
    all the same information: k-point neighbour indices and G-vectors are
    listed in the block headers.  K-points come from the begin kpoints block
    in the .win file.
    """
    kpts_lines = win_params.get("kpoints", [])
    if not kpts_lines:
        raise ValueError("begin kpoints block missing from .win file; cannot reconstruct nnkp")
    kpoints = np.array(
        [[float(x) for x in line.split()[:3]] for line in kpts_lines[:nk]],
        dtype=np.float64,
    )
    if len(kpoints) != nk:
        raise ValueError(
            f"Expected {nk} k-points in .win but found {len(kpoints)}"
        )

    nnkpts    = np.zeros((nk, nnb), dtype=np.int64)
    g_vectors = np.zeros((nk, nnb, 3), dtype=np.int64)
    # ib_counter tracks how many neighbors have been filled for each k-point,
    # so the assignment is correct regardless of the order entries appear in kpb_map.
    ib_counter: dict[int, int] = {}
    for (ik, ik2, g1, g2, g3) in kpb_map:
        ib = ib_counter.get(ik, 0)
        nnkpts[ik, ib]    = ik2
        g_vectors[ik, ib] = (g1, g2, g3)
        ib_counter[ik]    = ib + 1

    return {
        "kpoints":  kpoints,
        "nnkpts":   nnkpts,
        "g_vectors": g_vectors,
        "nntot":    nnb,
    }
