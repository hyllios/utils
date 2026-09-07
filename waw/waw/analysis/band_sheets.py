"""
Identify the *physical* bands / Fermi-surface sheets of a Wannier model, for
band-resolved (multiband) quantities like the Eliashberg alpha2F matrix
(`analysis.elph.alpha2f_matrix`): MgB2's sigma/pi two-band coupling, Pb's
three-sheet coupling, etc.

Two complementary tools, matching the two regimes discussed in the design:

- `band_character_weights` -- a SOFT partition of unity ``w_i(n,k)`` assigning
  each (k, band) a fractional membership on sheet ``i`` from its Wannier-orbital
  character ``|U_{on}(k)|^2``. Needs no global band tracking, so it handles a
  band crossing E_F several times, eigenvalue-order swaps at crossings, and
  genuine hybridization (a 50/50 point splits fractionally) all gracefully.
  This is the robust default the alpha2F matrix uses.

- `follow_bands` -- HARD connected-band labels by eigenvector-overlap tracking
  across the mesh (optionally by a symmetry eigenvalue when the channels are
  symmetry-separated, e.g. MgB2 sigma/pi under the horizontal mirror). For
  visualization (colouring a band structure / Fermi surface by sheet) and for a
  hard assignment where bands are cleanly separable; degrades to the soft
  weights where a global label would be topologically ambiguous.

``U`` throughout is `analysis.elph.band_eigensystem`'s eigenvector array,
shape ``(nk, nw, nw)`` with ``U[k, o, n]`` = amplitude of Wannier orbital ``o``
in band ``n`` at k (columns = bands).
"""

from __future__ import annotations

import numpy as np


def band_character_weights(U: np.ndarray, orbital_groups: list[list[int]]) -> np.ndarray:
    """
    Soft sheet-membership weights from Wannier-orbital character:

      w_i(n, k) = sum_{o in group i} |U[k, o, n]|^2.

    Parameters
    ----------
    U : (nk, nw, nw) complex
        Band eigenvectors in the Wannier basis (`band_eigensystem`).
    orbital_groups : list of lists of int
        Wannier-orbital indices forming each sheet ``i`` (e.g. MgB2:
        ``[[sigma orbitals], [pi orbitals]]``). Groups SHOULD partition
        ``range(nw)`` so the weights sum to 1 over sheets (columns of ``U`` are
        normalised); orbitals left out of every group are dropped and the
        weights then sum to < 1 (a deliberate "other" remainder).

    Returns
    -------
    (nk, n_sheets, nw) float64 -- ``w[k, i, n]``.
    """
    P = np.abs(np.asarray(U)) ** 2                       # (nk, orb, band)
    groups = [np.asarray(g, dtype=np.int64) for g in orbital_groups]
    all_idx = np.concatenate(groups) if groups else np.array([], dtype=np.int64)
    if len(all_idx) != len(np.unique(all_idx)):
        raise ValueError("band_character_weights: orbital_groups overlap "
                         "(an orbital appears in more than one sheet).")
    return np.stack([P[:, g, :].sum(axis=1) for g in groups], axis=1)


def sheet_dos(eig: np.ndarray, fermi_energy: float, weights: np.ndarray,
              sigma: float) -> np.ndarray:
    """Per-sheet density of states at ``fermi_energy`` (states/Hartree, per spin
    channel, on the given k-mesh):

      N_i(eF) = (1/Nk) sum_{k,n} w_i(n,k) delta(eps_kn - eF).

    ``eig`` (nk, nw) Hartree, ``weights`` (nk, n_sheets, nw) from
    `band_character_weights`, ``sigma`` Gaussian broadening (Hartree). The
    unweighted total sum over sheets equals the ordinary DOS."""
    from ..core.distributions import gaussian_smearing
    delta = gaussian_smearing(np.asarray(eig) - fermi_energy, sigma)   # (nk, nw)
    return np.einsum("kin,kn->i", weights, delta) / eig.shape[0]


def sheet_drude_weight(eig: np.ndarray, velocities: np.ndarray,
                       fermi_energy: float, weights: np.ndarray,
                       sigma: float) -> np.ndarray:
    """
    Per-sheet Drude weight for each Cartesian direction (per spin channel):

      D_a,i = (1/Nk) sum_{k,n} w_i(n,k) v_a(n,k)^2 delta(eps_kn - eF)
            = N_i <v_a^2>_i,      Hartree * Bohr^2

    This is the numerator of the Boltzmann conductivity: with a relaxation time
    tau, sigma_aa = (num_elec_per_state) e^2 D_a tau / Omega, so summed over
    sheets it fixes the Drude plasma frequency, Omega_p,a^2 = 4 pi e^2
    (num_elec_per_state) D_a / Omega. Checking that against the free-electron
    value is the quickest way to catch a bad Wannier interpolation: it is a
    single-Fermi-surface integral and converges far faster than lambda, so a
    D that is off says the velocities or the mesh are wrong, not the coupling.

    ``velocities`` (nk, nw, 3) Hartree*Bohr, ``weights`` (nk, n_sheets, nw)
    from `band_character_weights`. Returns (3, n_sheets).
    """
    from ..core.distributions import gaussian_smearing
    eig = np.asarray(eig, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.shape[:2] != eig.shape or velocities.shape[2] != 3:
        raise ValueError(
            f"sheet_drude_weight: velocities must be {(*eig.shape, 3)}; "
            f"got {velocities.shape}")
    delta = gaussian_smearing(eig - fermi_energy, sigma)               # (nk, nw)
    return np.einsum("kin,kn,kna->ai", weights, delta, velocities ** 2,
                     optimize=True) / eig.shape[0]


def sheets_crossing_fermi(eig: np.ndarray, fermi_energy: float,
                          weights: np.ndarray, sigma: float,
                          rel_threshold: float = 0.02) -> np.ndarray:
    """Indices of sheets with non-negligible Fermi-level DOS (i.e. that actually
    cross E_F): those whose `sheet_dos` exceeds ``rel_threshold`` times the
    largest sheet's. Rows of the alpha2F matrix for sheets NOT here are ~0."""
    N_i = sheet_dos(eig, fermi_energy, weights, sigma)
    keep = N_i > rel_threshold * N_i.max()
    return np.flatnonzero(keep)


def follow_bands(U: np.ndarray, path_neighbors: np.ndarray | None = None,
                 seed: int = 0) -> np.ndarray:
    """
    Connected physical-band labels by eigenvector-overlap tracking. At each step
    bands are matched to the previous k-point by MAXIMUM overlap
    ``|<n(k)|m(k')>|^2 = |[U(k)^H U(k')]_{nm}|^2`` (Hungarian assignment) -- the
    eigenvectors stay continuous through an eigenvalue crossing, so this follows
    the physical band across it.

    Parameters
    ----------
    U : (nk, nw, nw) complex -- band eigenvectors (`band_eigensystem`) along an
        ORDERED sequence of k-points (a band-structure path); consecutive
        entries must be close in k for the overlap match to be meaningful.
    path_neighbors : (nk,) int or None
        For a plain 1D path leave None (each k matches k-1). For a general mesh
        pass, per k, the index of an already-labelled neighbour to match against
        (a spanning-tree parent; -1 for the seed) -- lets the caller propagate
        labels over a 2D/3D mesh from a seed.
    seed : int -- index whose band order defines the reference labels (identity).

    Returns
    -------
    label : (nk, nw) int64 -- ``label[k, n]`` = physical-band id of the n-th
        eigenvalue-sorted band at k. Ids are the band columns of the seed.

    Note: a globally consistent hard labelling exists only when the bands are
    separable; at a genuine (non-symmetry-protected) degeneracy the assignment
    there is arbitrary -- use `band_character_weights` for quantities that must
    stay well-defined across such points.
    """
    from scipy.optimize import linear_sum_assignment

    U = np.asarray(U)
    nk, nw = U.shape[0], U.shape[2]
    label = np.full((nk, nw), -1, dtype=np.int64)
    label[seed] = np.arange(nw)

    if path_neighbors is None:
        order = list(range(seed + 1, nk)) + list(range(seed - 1, -1, -1))
        parent = {k: (k - 1 if k > seed else k + 1) for k in order}
    else:
        parent = {k: int(path_neighbors[k]) for k in range(nk) if k != seed}
        order = [k for k in range(nk) if k != seed]
        # process in order of already-labelled parents (assumes a valid tree)
        order.sort(key=lambda k: 0 if parent[k] == seed else 1)

    for k in order:
        p = parent[k]
        if label[p, 0] == -1:
            continue
        M = np.abs(U[p].conj().T @ U[k]) ** 2          # (band_p, band_k)
        row, col = linear_sum_assignment(-M)           # maximize overlap
        label[k, col] = label[p, row]
    return label
