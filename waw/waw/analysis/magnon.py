"""
Magnon spectra from Heisenberg exchange couplings (linear spin-wave theory).

Holstein-Primakoff to quadratic order around a (generally non-collinear,
here any set of moment directions) classical ground state, diagonalized by
Colpa's paraunitary method (J. H. P. Colpa, Physica A 93, 327 (1978)).
Conventions follow TB2J's magnon module exactly so that spectra computed
from the same J's coincide: E = -sum_{i!=j} J_ij e_i.e_j (unit vectors,
both orderings counted), S_i = |m_i|/2, J(q) = sum_R J(R) e^{-2 pi i q.R}.
Certified against TB2J's `Magnon` on identical exchange data
(tests/test_analysis_magnon.py).

All energies in Hartree.
"""
from __future__ import annotations

import numpy as np


def _rotation_arrays(directions: np.ndarray):
    """TB2J's (U, V) local-frame arrays for moment unit vectors (ns, 3)."""
    uz = np.array([0.0, 0.0, 1.0])
    ns = len(directions)
    U = np.zeros((ns, 3), dtype=complex)
    V = np.zeros((ns, 3))
    for a, v in enumerate(directions):
        n = np.cross(uz, v)
        nn = np.linalg.norm(n)
        if nn > 1e-12:
            n = n / nn
            A = np.stack([uz, np.cross(n, uz), n])
            B = np.stack([v, np.cross(n, v), n])
            R = A.T @ B
            U[a] = R[0] + 1j * R[1]
            V[a] = R[2]
        else:
            e3 = v
            axis = np.array([1.0, 0.0, 0.0])
            if np.linalg.norm(np.cross(axis, e3)) < 1e-12:
                axis = np.array([0.0, 1.0, 0.0])
            e1 = axis - (axis @ e3) * e3
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(e3, e1)
            U[a] = e1 + 1j * e2
            V[a] = e3
    return U, V


def lswt_matrix(J: dict, magmoms: np.ndarray, qpts: np.ndarray) -> np.ndarray:
    """
    The LSWT grand matrix h(q) of Toth & Lake, (nq, 2ns, 2ns) complex Hermitian.

        h(q) = [[A(q) - C,      B(q)     ],
                [B^dag(q),  Abar(-q) - C ]]

    `magnon_bands` diagonalizes it with Colpa's method; the magnetic
    ground-state search minimizes its eigenvalues directly, because h(q)
    positive semi-definite for all q IS the stability condition of the
    assumed magnetic structure (npj Comput. Mater., Romero et al.). A
    negative eigenvalue at q names a spin-wave mode cheaper than the
    reference state, and that q is the propagation vector to adopt.

    Args as `magnon_bands`. Hartree.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    if magmoms.ndim == 1:
        m = np.zeros((len(magmoms), 3))
        m[:, 2] = magmoms
        magmoms = m
    ns = len(magmoms)
    Snorm = np.linalg.norm(magmoms, axis=1) / 2.0
    directions = magmoms / np.linalg.norm(magmoms, axis=1)[:, None]

    pairs = [(R, i, j) for (R, i, j) in J if not (i == j and R == (0, 0, 0))]
    Rlist = sorted({R for (R, _, _) in pairs})
    Rindex = {R: k for k, R in enumerate(Rlist)}
    JR = np.zeros((len(Rlist), ns, ns))
    for (R, i, j) in pairs:
        JR[Rindex[R], i, j] = J[(R, i, j)]
    JR = JR / (Snorm[:, None] * Snorm[None, :])
    Rarr = np.array(Rlist, dtype=float)

    qpts = np.atleast_2d(np.asarray(qpts, dtype=float))
    phase = np.exp(-2j * np.pi * (qpts @ Rarr.T))
    Jq = np.einsum("qr,rij->qij", phase, JR)
    Jmq = np.einsum("qr,rij->qij", phase.conj(), JR)

    U, V = _rotation_arrays(directions)
    UU_c = U @ U.conj().T
    UU = U @ U.T
    VV = V @ V.T

    J0 = -np.einsum("rij->ij", JR)
    Jq = -Jq
    Jmq = -Jmq

    Ssqrt = np.sqrt(Snorm)
    SS = Ssqrt[:, None] * Ssqrt[None, :]
    A1 = Jmq * UU_c[None] * SS[None]
    A2 = Jq.conj() * UU_c.conj()[None] * SS[None]
    B = Jmq * UU[None] * SS[None]
    C = np.diag(((2.0 * J0 * VV) @ Snorm))

    nq = len(qpts)
    H = np.zeros((nq, 2 * ns, 2 * ns), dtype=complex)
    H[:, :ns, :ns] = A1 - C
    H[:, :ns, ns:] = B
    H[:, ns:, :ns] = B.swapaxes(1, 2).conj()
    H[:, ns:, ns:] = A2 - C
    return H


def magnon_bands(
    J: dict,
    magmoms: np.ndarray,
    qpts: np.ndarray,
    downfold: "list[int] | None" = None,
) -> np.ndarray:
    """
    LSWT magnon energies from isotropic Heisenberg couplings.

    Args:
      J        : {(R, i, j): J_iso (Hartree)} in the TB2J sign convention
                 (J > 0 ferromagnetic), R a 3-tuple of ints, i/j magnetic
                 sublattice indices -- e.g. `ExchangeResult.J` from
                 `analysis.exchange.heisenberg_exchange`, or parsed TB2J
                 output. Both (R, i, j) and (-R, j, i) entries should be
                 present (as both producers emit them). On-site (0,i,i)
                 entries are ignored.
      magmoms  : (ns,) signed z moments or (ns, 3) moment vectors, mu_B;
                 spin magnitudes are S_i = |m_i|/2 (TB2J convention).
      qpts     : (nq, 3) fractional q-points.
      downfold : sublattice indices to eliminate ADIABATICALLY instead of
                 keeping as branches -- for weakly polarized atoms (ligands,
                 induced moments) whose spins follow the strong ones rather
                 than precessing independently. Returns only the kept
                 branches, with the folded sites' coupling absorbed into
                 them. See the note below on why this is usually what you
                 want for a small induced moment.

    WEAKLY POLARIZED SUBLATTICES.  A magnon energy scales as J/S, and the LSWT
    matrix elements here go as J_ij/sqrt(S_i S_j), so a site with a small
    moment does not give a small correction -- it gives a huge, flat, spurious
    branch.  On Ag2NdCd the Ag sites carry 0.039 mu_B against Nd's 3.5, and
    treating them as independent spins puts two flat branches at 60-124 meV
    while the physical Nd magnons live below 10 meV.  Those branches are the
    precession of a nearly-zero moment: fast, decoupled, and not what a
    neutron sees.  Three defensible treatments, in increasing completeness:

      1. leave them out of the exchange model entirely -- their polarization
         still enters the Green's functions that produce J, but their spins
         are frozen when the moments are rotated;
      2. keep them as full branches (`downfold=None`) -- correct only when
         their moments are comparable to the others';
      3. compute their couplings and pass them here as `downfold` -- the
         induced moments then mediate coupling between the strong sites
         without contributing branches of their own.

    (1) and (3) differ by exactly the ligand-mediated path, which is not
    small: on NiI2 downfolding the iodine moments changed J1 by ~90x.

    Returns
    -------
    omega : (nq, ns) float64, Hartree -- magnon branches, ascending. A
      Cholesky failure of the LSWT grand matrix (classical reference state
      not a local minimum -- e.g. a frustrated lattice past its spiral
      instability) falls back to shifting by the most negative eigenvalue,
      TB2J-style; the resulting branches signal the instability rather
      than crash.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    if magmoms.ndim == 1:
        m = np.zeros((len(magmoms), 3))
        m[:, 2] = magmoms
        magmoms = m
    ns = len(magmoms)
    Snorm = np.linalg.norm(magmoms, axis=1) / 2.0
    directions = magmoms / np.linalg.norm(magmoms, axis=1)[:, None]

    pairs = [(R, i, j) for (R, i, j) in J if not (i == j and R == (0, 0, 0))]
    Rlist = sorted({R for (R, _, _) in pairs})
    Rindex = {R: k for k, R in enumerate(Rlist)}
    JR = np.zeros((len(Rlist), ns, ns))
    for (R, i, j) in pairs:
        JR[Rindex[R], i, j] = J[(R, i, j)]
    JR = JR / (Snorm[:, None] * Snorm[None, :])
    Rarr = np.array(Rlist, dtype=float)

    qpts = np.atleast_2d(np.asarray(qpts, dtype=float))
    phase = np.exp(-2j * np.pi * (qpts @ Rarr.T))          # (nq, nR)
    Jq = np.einsum("qr,rij->qij", phase, JR)
    Jmq = np.einsum("qr,rij->qij", phase.conj(), JR)

    U, V = _rotation_arrays(directions)
    UU_c = U @ U.conj().T          # (ns, ns): U_i . U_j*
    UU = U @ U.T                   # U_i . U_j
    VV = V @ V.T                   # V_i . V_j

    J0 = -np.einsum("rij->ij", JR)
    Jq = -Jq
    Jmq = -Jmq

    Ssqrt = np.sqrt(Snorm)
    SS = Ssqrt[:, None] * Ssqrt[None, :]
    A1 = Jmq * UU_c[None] * SS[None]
    A2 = Jq.conj() * UU_c.conj()[None] * SS[None]
    B = Jmq * UU[None] * SS[None]
    C = np.diag(((2.0 * J0 * VV) @ Snorm))

    nq = len(qpts)
    H = np.zeros((nq, 2 * ns, 2 * ns), dtype=complex)
    H[:, :ns, :ns] = A1 - C
    H[:, :ns, ns:] = B
    H[:, ns:, :ns] = B.swapaxes(1, 2).conj()
    H[:, ns:, ns:] = A2 - C

    if downfold:
        fold = sorted(set(int(i) for i in downfold))
        if any(i < 0 or i >= ns for i in fold):
            raise ValueError(f"downfold indices must lie in [0, {ns})")
        keep = [i for i in range(ns) if i not in fold]
        if not keep:
            raise ValueError("cannot downfold every sublattice")
        # eliminate the particle AND hole component of each folded site, at
        # omega -> 0: H_eff = H_KK - H_KF H_FF^-1 H_FK (Schur complement).
        # Exact for a collinear ferromagnet, where the anomalous block B
        # vanishes and this is an ordinary Hermitian elimination; elsewhere it
        # is the adiabatic (fast-mode) approximation, good when the folded
        # branches sit well above the kept ones -- which is precisely the
        # regime a small induced moment puts them in.
        K = np.array(keep + [i + ns for i in keep])
        F = np.array(fold + [i + ns for i in fold])
        Hd = np.zeros((nq, len(K), len(K)), dtype=complex)
        for iq in range(nq):
            h = H[iq]
            hff = h[np.ix_(F, F)]
            Hd[iq] = h[np.ix_(K, K)] - h[np.ix_(K, F)] @ np.linalg.solve(
                hff, h[np.ix_(F, K)])
        H = Hd
        ns = len(keep)

    g = np.diag(np.concatenate([np.ones(ns), -np.ones(ns)]))
    omega = np.zeros((nq, ns))
    for iq in range(nq):
        h = H[iq]
        shift = 0.0
        try:
            K = np.linalg.cholesky(h)
        except np.linalg.LinAlgError:
            try:
                K = np.linalg.cholesky(h + 1e-9 * np.eye(2 * ns))
            except np.linalg.LinAlgError:
                shift = float(np.linalg.eigvalsh(h).min())
                K = np.linalg.cholesky(h - (shift - 1e-9) * np.eye(2 * ns))
        w = np.linalg.eigvalsh(K.conj().T @ g @ K)
        omega[iq] = w[ns:] + shift
    return omega


# --------------------------------------------------------------------------
# Magnetic ground state from the LSWT stability condition
#
# Romero et al., "Systematic determination of a material's magnetic ground
# state from first principles", npj Comput. Mater. The reference magnetic
# structure is a ground state only if the LSWT grand matrix h(q) is positive
# semi-definite for every q. A negative eigenvalue at q_min names a spin-wave
# mode cheaper than the reference, and q_min is the propagation vector to
# adopt -- in the cell commensurate with it, q_min becomes 0 by construction.
# Iterating (DFT -> LKAG -> here -> new cell -> DFT ...) is the self-consistent
# loop; this module supplies the step that turns imaginary magnons into the
# next cell to try.
# --------------------------------------------------------------------------

def propagation_vector(J: dict, magmoms: np.ndarray, mesh=12, refine: bool = True):
    """
    The q that minimizes the lowest eigenvalue of h(q) -- the instability of
    the assumed magnetic structure, and the propagation vector of the next one.

    Args:
      J, magmoms : as `magnon_bands`, describing the REFERENCE structure
      mesh       : int or (n1, n2, n3), the Gamma-centred scan grid
      refine     : polish the best mesh point with Nelder-Mead

    Returns a dict:
      ``q``         (3,) propagation vector, fractional
      ``lambda_min`` lowest eigenvalue of h(q) at that q (Hartree)
      ``lambda_at_0`` the same at q = 0
      ``stable``    True when the reference structure is already a ground
                    state -- q is 0 (to `tol`) and lambda_min >= 0
    A stable answer means the reference survives; anything else is the cell to
    build next. Note this tests the reference you PASSED: a ferromagnetic
    magmoms array tests the ferromagnet.
    """
    from scipy.optimize import minimize

    mesh = (mesh, mesh, mesh) if np.isscalar(mesh) else tuple(mesh)
    grids = [np.arange(n) / n for n in mesh]
    qs = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)

    def lam(q):
        h = lswt_matrix(J, magmoms, np.atleast_2d(q))
        return float(np.linalg.eigvalsh(h[0]).min())

    lams = np.array([float(np.linalg.eigvalsh(h).min())
                     for h in lswt_matrix(J, magmoms, qs)])
    k = int(np.argmin(lams))
    q_best, lam_best = qs[k].copy(), float(lams[k])

    if refine:
        r = minimize(lam, q_best, method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-14, maxiter=400))
        if r.fun < lam_best:
            q_best, lam_best = np.mod(r.x, 1.0), float(r.fun)

    lam0 = lam(np.zeros(3))
    q_wrapped = np.where(q_best > 0.5, q_best - 1.0, q_best)
    stable = bool(np.allclose(q_wrapped, 0.0, atol=1e-4)
                  and lam_best > -1e-10)
    return dict(q=q_wrapped, lambda_min=lam_best, lambda_at_0=lam0,
                stable=stable)


def commensurate_supercell(q, max_denominator: int = 12, tol: float = 1e-4):
    """
    The smallest diagonal supercell commensurate with propagation vector `q`,
    and the collinear spin pattern on it.

    Each component q_a is approximated by the fraction n_a/d_a with the
    smallest denominator within `tol`; the supercell is diag(d_1, d_2, d_3)
    and site at lattice translation R carries sign cos(2 pi q.R). Signs are
    collinear, so this covers the commensurate collinear orders (FM, the
    A/C/G-type and fcc type-I/II families) and NOT spirals -- for an
    incommensurate q, `d` will hit `max_denominator` and the returned
    ``exact`` flag is False, meaning the cell is an approximant.

    Returns a dict: ``supercell`` (3,) int, ``signs`` (n_cells,) +-1,
    ``translations`` (n_cells, 3) int, ``exact`` bool, ``q_used`` (3,).
    """
    from fractions import Fraction

    q = np.asarray(q, dtype=float)
    fr = [Fraction(float(x)).limit_denominator(max_denominator) for x in q]
    d = np.array([f.denominator for f in fr], dtype=int)
    q_used = np.array([float(f) for f in fr])
    exact = bool(np.allclose(q_used, q, atol=tol))

    tr = np.stack(np.meshgrid(*[np.arange(n) for n in d], indexing="ij"),
                  axis=-1).reshape(-1, 3)
    phase = np.cos(2 * np.pi * (tr @ q_used))
    signs = np.where(phase >= 0, 1, -1).astype(int)
    return dict(supercell=d, signs=signs, translations=tr, exact=exact,
                q_used=q_used)
