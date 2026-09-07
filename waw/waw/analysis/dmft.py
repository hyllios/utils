"""
Dynamical mean-field theory on a Wannier Hamiltonian: local correlation beyond
any static mean field.

  Reference physics: A. Georges, G. Kotliar, W. Krauth and M. J. Rozenberg,
  Rev. Mod. Phys. 68, 13 (1996) for the method; M. Caffarel and W. Krauth,
  Phys. Rev. Lett. 72, 1545 (1994) for the exact-diagonalization solver with a
  discretised bath; R. Bulla, Phys. Rev. Lett. 83, 136 (1999) for the numbers
  the Mott transition here is checked against.

WHAT THIS IS. DMFT replaces the lattice self-energy by a LOCAL one,
Sigma(k, z) -> Sigma(z), and fixes it by demanding that the lattice's local
Green's function equal that of an Anderson impurity embedded in a
self-consistent bath:

    G_loc(z) = (1/N_k) sum_k [z + mu - H(k) - Sigma(z)]^-1        (lattice)
    G_imp(z) = impurity solver applied to  G_0^-1 = z + mu - Delta(z)
    Sigma(z) = G_0^-1(z) - G_imp^-1(z)                            (Dyson)
    Delta(z) <- z + mu - Sigma(z) - G_loc^-1(z)                   (self-consistency)

The approximation is the locality of Sigma, exact in infinite dimensions. What
it buys over any static scheme (LDA+U, Hartree-Fock) is FREQUENCY dependence:
quasiparticle renormalisation Z, finite lifetimes, and Hubbard bands, none of
which a static potential can produce. A Mott insulator here is a genuine one --
gapped without symmetry breaking -- rather than a band insulator of an ordered
state.

RELATION TO `analysis.cpa`. The outer loop is structurally the CPA loop already
in this package: a local, non-Hermitian, energy-dependent self-energy driven to
self-consistency against a k-integrated local Green's function. CPA gets Sigma
from averaging over disorder; DMFT gets it from solving an interacting
impurity. Everything downstream of "we have Sigma(z)" is shared.

UNITS. Atomic units throughout: energies and U in Hartree, `beta` in inverse
Hartree. Matsubara frequencies are the fermionic ones, i w_n = i (2n+1) pi / beta.

WHAT COMES OUT, AND WHAT IT COSTS. With Sigma(z) in hand every one-particle
quantity in this package becomes a correlated one -- A(k, w) comparable with
photoemission (including the incoherent Hubbard bands DFT does not have), and
H(k) + Sigma(0), the topological Hamiltonian, whose invariants are those of the
interacting system. The costs are real and are not hidden here: U and J are
INPUTS (constrained RPA is the way to stop them being dials); the
double-counting correction between DFT and DMFT is genuinely ambiguous, though
for a metal it largely amounts to a shift of mu; and the exact-diagonalization
solver represents the bath by a handful of discrete levels, which is a
controlled but visible approximation -- `fit_bath` reports its own residual and
`dmft_selfconsistency` propagates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "matsubara_frequencies",
    "bethe_green",
    "AndersonParameters",
    "ImpuritySolution",
    "solve_anderson_ed",
    "solve_hubbard_i",
    "solve_ipt",
    "fit_bath",
    "quasiparticle_weight",
    "pade_continuation",
    "DMFTResult",
    "dmft_bethe",
    "local_green_function",
    "dmft_selfconsistency",
    "dmft_scan",
    "StaticUResult",
    "static_u_bethe",
]


def dmft_scan(points, *, common=None, continuation: bool = False,
              verbose: bool = False) -> list:
    """
    Run a series of `dmft_bethe` solves, SEQUENTIALLY.

    Deliberately serial. A process pool over these is tempting -- the points
    are independent and a phase diagram is dozens of them -- but a pool created
    in an interpreter that has already done heavy multi-threaded torch work
    stalls: under `fork` it deadlocks at once (the child inherits OpenMP's locks
    but not the threads holding them), and under `spawn` the workers run for a
    while and then go idle. A cold parent is fine at any width, a warm one is
    not, and "warm" is exactly what a script that solves anything before
    scanning will be. Waiting is better than debugging that.

    What sequential buys back is CONTINUATION. With ``continuation=True`` each
    point starts from the previous one's converged hybridisation, which is what
    traces a hysteresis loop: sweeping U upward finds the spinodal where the
    metal dies, sweeping downward finds where the insulator does, and the two
    differ. Restarting every point from scratch finds only one of them. A pool
    cannot do this at all -- the dependence is the point.

    Args:
      points : iterable of dicts of `dmft_bethe` keyword arguments, e.g.
               ``[{'U': u} for u in np.arange(0.5, 3.4, 0.1)]``
      common : keyword arguments merged into every point
      continuation : feed each solution forward as the next point's `delta0`
      verbose : print one line per point

    Returns a list of `DMFTResult`, in the order given.
    """
    out, delta = [], None
    for i, pt in enumerate(points):
        kw = dict(common or {}, **pt)
        if continuation and delta is not None:
            kw.setdefault("delta0", delta)
        r = dmft_bethe(**kw)
        delta = r.delta_iw
        out.append(r)
        if verbose:
            tag = " ".join(f"{k}={v:g}" for k, v in pt.items()
                           if isinstance(v, (int, float)))
            print(f"  [{i + 1}/{len(points) if hasattr(points, '__len__') else '?'}] "
                  f"{tag}  Z = {r.Z:+.4f}  n = {r.occupation:.4f}  "
                  f"{'conv' if r.converged else 'NOT conv'} in {r.n_iter} it",
                  flush=True)
    return out


# --------------------------------------------------------------------------
# Grids and free Green's functions
# --------------------------------------------------------------------------

def matsubara_frequencies(beta: float, n_iw: int) -> np.ndarray:
    """Fermionic Matsubara frequencies w_n = (2n+1) pi / beta, n = 0 .. n_iw-1.

    Returned REAL; the Green's functions take ``1j * w``. Only the positive
    branch is stored -- everything here obeys G(-i w) = conj(G(i w)), and
    carrying both halves is a common source of double counting in tail fits.
    """
    if beta <= 0:
        raise ValueError("matsubara_frequencies: beta must be > 0")
    return (2 * np.arange(int(n_iw)) + 1) * np.pi / float(beta)


def bethe_green(z, half_bandwidth: float) -> np.ndarray:
    """Local Green's function of the Bethe lattice, semicircular DOS.

    ``G(z) = 2 (z - s sqrt(z^2 - D^2)) / D^2`` with the branch chosen so that
    ``Im G < 0`` in the upper half plane and ``G ~ 1/z`` at large |z|. This is
    the Hilbert transform of ``rho(e) = 2 sqrt(D^2 - e^2) / (pi D^2)``, and the
    reason the Bethe lattice is the standard DMFT testbed: the self-consistency
    collapses to ``Delta = (D/2)^2 G`` with no k-sum at all.
    """
    z = np.asarray(z, dtype=np.complex128)
    D = float(half_bandwidth)
    root = np.sqrt(z ** 2 - D ** 2)
    # pick the branch that decays: G -> 1/z
    root = np.where(np.real(np.conj(z) * root) < 0, -root, root)
    return 2.0 * (z - root) / D ** 2


# --------------------------------------------------------------------------
# The Anderson impurity, and two solvers
# --------------------------------------------------------------------------

@dataclass
class AndersonParameters:
    """Single-orbital Anderson impurity: one correlated level plus a discrete bath.

    ``H = eps_imp sum_s n_s + U n_up n_dn
         + sum_p eps_bath[p] sum_s n_ps
         + sum_p V[p] sum_s (d_s^+ c_ps + h.c.)``

    ``eps_imp`` is measured from the chemical potential; particle-hole symmetry
    at half filling is ``eps_imp = -U/2`` with a bath symmetric about zero.
    """
    eps_imp: float
    U: float
    eps_bath: np.ndarray
    V: np.ndarray

    @property
    def n_bath(self) -> int:
        return len(np.atleast_1d(self.eps_bath))

    def hybridization(self, z) -> np.ndarray:
        """Delta(z) = sum_p |V_p|^2 / (z - eps_p)."""
        z = np.asarray(z, dtype=np.complex128)
        e = np.atleast_1d(np.asarray(self.eps_bath, dtype=np.float64))
        v = np.atleast_1d(np.asarray(self.V, dtype=np.float64))
        if len(e) == 0:
            return np.zeros_like(z)
        return (v ** 2 / (z[:, None] - e[None, :])).sum(axis=1)


@dataclass
class ImpuritySolution:
    """What a solver returns on the Matsubara axis."""
    iw:        np.ndarray     # (n_iw,) real Matsubara frequencies
    g_iw:      np.ndarray     # (n_iw,) impurity Green's function
    sigma_iw:  np.ndarray     # (n_iw,) impurity self-energy
    occupation: float         # <n> per spin
    double_occupancy: float   # <n_up n_dn>
    ground_energy: float


# ---- Fock-space machinery for the exact-diagonalisation solver -------------

def _sector_states(n_sites: int, n_up: int, n_dn: int) -> np.ndarray:
    """Fock states of a (N_up, N_dn) sector as integers.

    Spin-orbital p = site + n_sites * spin, bit p of the integer. The ordering
    fixes the Jordan-Wigner signs used by `_apply_c`; changing it changes every
    matrix element's sign and nothing else, so it must be used consistently.
    """
    out = []
    for up in combinations(range(n_sites), n_up):
        mu = sum(1 << i for i in up)
        for dn in combinations(range(n_sites), n_dn):
            out.append(mu | sum(1 << (i + n_sites) for i in dn))
    return np.array(sorted(out), dtype=np.int64)


def _apply_c(state: int, p: int) -> tuple[int, int]:
    """Annihilate spin-orbital p of a single state. (new_state, sign); 0 if empty."""
    if not (state >> p) & 1:
        return 0, 0
    sign = -1 if bin(state & ((1 << p) - 1)).count("1") % 2 else 1
    return state ^ (1 << p), sign


def _occupations(states: np.ndarray, n_orb: int) -> np.ndarray:
    """(dim, n_orb) occupation bits of every state at once."""
    return (states[:, None] >> np.arange(n_orb)[None, :]) & 1


def _hopping_block(states: np.ndarray, a: int, b: int):
    """Matrix elements of c_a^+ c_b over a sorted sector, vectorised.

    Returns (rows, cols, signs). The Jordan-Wigner sign is
    (-1)^{popcount(s & mask_b)} for the annihilation and
    (-1)^{popcount(t & mask_a)} for the creation, with t = s ^ (1<<b) -- the
    intermediate state, not the original. Using s for both is the classic
    fermion-sign bug and it only shows up when a and b straddle occupied
    orbitals.
    """
    occ_b = (states >> b) & 1
    empty_a = ~((states >> a) & 1) & 1
    sel = np.nonzero(occ_b & empty_a)[0]
    if not len(sel):
        return sel, sel, np.zeros(0, dtype=np.float64)
    src = states[sel]
    t = src ^ (np.int64(1) << np.int64(b))
    # np.bitwise_count returns UINT8, so `1 - 2*(count & 1)` wraps to 255
    # instead of -1. Cast first. The bug is invisible for n_bath = 1, where
    # the impurity and the single bath orbital are adjacent and no hop can
    # straddle an occupied one, so every sign is +1 anyway.
    def _parity(x):
        return 1 - 2 * (np.bitwise_count(x).astype(np.int64) & 1)
    sgn_b = _parity(src & ((np.int64(1) << np.int64(b)) - 1))
    sgn_a = _parity(t & ((np.int64(1) << np.int64(a)) - 1))
    dest = t | (np.int64(1) << np.int64(a))
    rows = np.searchsorted(states, dest)
    return rows, sel, (sgn_a * sgn_b).astype(np.float64)


def _hamiltonian(states: np.ndarray, n_sites: int, eps: np.ndarray,
                 U: float, V: np.ndarray) -> np.ndarray:
    """Dense H in one (N_up, N_dn) sector. eps[0] is the impurity level."""
    dim = len(states)
    H = np.zeros((dim, dim), dtype=np.float64)
    occ = _occupations(states, 2 * n_sites)
    H[np.diag_indices(dim)] = (occ * np.tile(eps, 2)[None, :]).sum(axis=1) \
        + U * occ[:, 0] * occ[:, n_sites]
    for sp in (0, 1):
        imp = n_sites * sp
        for q in range(1, n_sites):
            bath = q + n_sites * sp
            for a, b in ((imp, bath), (bath, imp)):
                r, c, sg = _hopping_block(states, a, b)
                if len(r):
                    np.add.at(H, (r, c), V[q - 1] * sg)
    return H


def _batched_eigh(mats: list) -> list:
    """Diagonalise many small dense matrices, grouped by size so torch can
    batch them.

    DMFT's sectors are numerous and individually tiny -- 36 of them at
    n_bath = 4, the largest 100x100 -- so LAPACK threading buys nothing and the
    cost is per-call overhead. Stacking equal-sized sectors into one batched
    `torch.linalg.eigh` turns that into a single threaded call per size class.
    """
    import torch
    out = [None] * len(mats)
    by_dim: dict = {}
    for i, m in enumerate(mats):
        by_dim.setdefault(m.shape[0], []).append(i)
    for d, idx in by_dim.items():
        stack = torch.from_numpy(np.stack([mats[i] for i in idx]))
        w, v = torch.linalg.eigh(stack)
        w, v = w.numpy(), v.numpy()
        for j, i in enumerate(idx):
            out[i] = (w[j], v[j])
    return out


def _lehmann_sum(iw: np.ndarray, poles: np.ndarray, weights: np.ndarray,
                 *, merge_tol: float = 1e-10, chunk: int = 1 << 23) -> np.ndarray:
    """``sum_p w_p / (i w_n - d_p)`` for real poles, as real matmuls.

    Two things make this fast enough to stop being the bottleneck.

    REAL ARITHMETIC. With z = i w and d real,
    ``1/(iw - d) = -(d + iw)/(d^2 + w^2)``, so one real matrix
    ``M[n,p] = 1/(d_p^2 + w_n^2)`` serves both parts and the whole sum is
    ``M @ [w*d, w]`` -- a single real GEMM. A complex reciprocal of the same
    array costs several times more memory traffic and cannot use GEMM at all.

    POLE MERGING. The Lehmann representation of a sector-resolved ED spectrum
    is riddled with exact degeneracies, and summing weights over identical
    poles first is exact, not an approximation. `merge_tol` is in Hartree; at
    the default it only merges poles that are degenerate to round-off.
    """
    import torch
    if not len(poles):
        return np.zeros(len(iw), dtype=np.complex128)
    key = np.round(poles / merge_tol).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    d = np.bincount(inv, weights=poles) / np.bincount(inv)
    w = np.bincount(inv, weights=weights)

    wn2 = torch.from_numpy(np.ascontiguousarray(iw ** 2))
    re = torch.zeros(len(iw), dtype=torch.float64)
    im = torch.zeros(len(iw), dtype=torch.float64)
    step = max(1, chunk // max(len(iw), 1))
    for i in range(0, len(d), step):
        dc = torch.from_numpy(np.ascontiguousarray(d[i:i + step]))
        wc = torch.from_numpy(np.ascontiguousarray(w[i:i + step]))
        M = 1.0 / (wn2[:, None] + (dc * dc)[None, :])
        re -= M @ (wc * dc)
        im += M @ wc
    return re.numpy() - 1j * np.asarray(iw) * im.numpy()


def _sector_energy_bound(n_up: int, n_dn: int, eps_sorted: np.ndarray,
                         U: float) -> float:
    """Rigorous lower bound on a sector's ground-state energy.

    The hopping is off-diagonal and traceless in each sector, so the lowest
    eigenvalue is bounded below by filling the lowest single-particle levels
    and ignoring the (non-negative) U -- minus the largest the hopping can
    shift it, which for the bound below is folded into the caller's margin.
    Used only to SKIP sectors whose Boltzmann weight cannot matter; a bound
    that is too loose costs time, never accuracy.
    """
    return float(eps_sorted[:n_up].sum() + eps_sorted[:n_dn].sum())


def solve_anderson_ed(params: AndersonParameters, beta: float, iw: np.ndarray,
                      *, mu: float = 0.0, e_cut: float = 40.0) -> ImpuritySolution:
    """
    Exact diagonalisation of the single-impurity Anderson model, finite T.

    Every ``(N_up, N_dn)`` sector is diagonalised densely and the Green's
    function assembled from the Lehmann representation,

        G(iw) = (1/Z) sum_{mn} |<m|d|n>|^2 (e^{-b E_m} + e^{-b E_n})
                                / (iw + E_m - E_n),

    which is exact for the discretised model -- the ONLY approximation in this
    solver is that the bath has `n_bath` levels rather than a continuum. That
    approximation is visible in the result (a finite number of poles) and is
    what `fit_bath`'s residual measures; it is not hidden inside a stochastic
    error bar the way a QMC solver's would be.

    Cost is set by the largest sector: n_bath = 4 gives a 100-dimensional
    worst case, n_bath = 6 about 1200. Beyond ~8 use a Lanczos solver instead.

    Args:
      beta  : inverse temperature, 1/Hartree
      iw    : (n_iw,) REAL Matsubara frequencies from `matsubara_frequencies`
      mu    : chemical potential, subtracted from every level
      e_cut : drop Boltzmann weights below exp(-e_cut) relative to the ground
              state. Purely a speed knob; 40 is far past double precision.
    """
    nb = params.n_bath
    n_sites = nb + 1
    eps = np.concatenate([[params.eps_imp], np.atleast_1d(params.eps_bath)]) - mu
    V = np.atleast_1d(np.asarray(params.V, dtype=np.float64))
    if len(V) != nb:
        raise ValueError("solve_anderson_ed: V and eps_bath must have equal length")

    keys, mats = [], []
    for n_up in range(n_sites + 1):
        for n_dn in range(n_sites + 1):
            st = _sector_states(n_sites, n_up, n_dn)
            if not len(st):
                continue
            keys.append(((n_up, n_dn), st))
            mats.append(_hamiltonian(st, n_sites, eps, params.U, V))
    diag = _batched_eigh(mats)
    sectors = {k: (st, w, v) for (k, st), (w, v) in zip(keys, diag)}

    e0 = min(w.min() for _, w, _ in sectors.values())
    Z = sum(np.exp(-beta * (w - e0)).sum() for _, w, _ in sectors.values())

    z = 1j * np.asarray(iw, dtype=np.float64)
    n_imp = 0.0
    docc = 0.0
    all_poles, all_weights = [], []
    for (n_up, n_dn), (st, w, v) in sectors.items():
        boltz = np.exp(-beta * (w - e0))
        occ = _occupations(st, n_sites + 1)
        occ_up = occ[:, 0].astype(np.float64)
        d_occ = (occ[:, 0] & ((st >> n_sites) & 1)).astype(np.float64)
        v2 = v * v
        n_imp += float(boltz @ (v2.T @ occ_up)) / Z
        docc += float(boltz @ (v2.T @ d_occ)) / Z

        # d_up connects (n_up, n_dn) -> (n_up - 1, n_dn); annihilating orbital 0
        # carries no Jordan-Wigner sign, there being no lower-indexed orbital
        tgt = (n_up - 1, n_dn)
        if tgt not in sectors:
            continue
        st2, w2, v2e = sectors[tgt]
        src = np.nonzero(st & 1)[0]
        if not len(src):
            continue
        rows = np.searchsorted(st2, st[src] ^ 1)
        D = np.zeros((len(st2), len(st)), dtype=np.float64)
        D[rows, src] = 1.0
        A = v2e.T @ D @ v                       # <m|d|n> in the eigenbasis
        b2 = np.exp(-beta * (w2 - e0))
        wt = (A * A) * (b2[:, None] + boltz[None, :]) / Z
        de = w2[:, None] - w[None, :]           # E_m - E_n
        keep = wt > np.exp(-e_cut)
        if keep.any():
            all_weights.append(wt[keep])
            all_poles.append(-de[keep])         # pole of 1/(z + E_m - E_n)
    poles = np.concatenate(all_poles) if all_poles else np.zeros(0)
    weights = np.concatenate(all_weights) if all_weights else np.zeros(0)
    g = _lehmann_sum(np.asarray(iw, dtype=np.float64), poles, weights)

    g0_inv = z + mu - params.eps_imp - params.hybridization(z)
    return ImpuritySolution(iw=np.asarray(iw), g_iw=g, sigma_iw=g0_inv - 1.0 / g,
                            occupation=n_imp, double_occupancy=docc,
                            ground_energy=e0)


def solve_hubbard_i(eps_imp: float, U: float, beta: float, iw: np.ndarray,
                    *, mu: float = 0.0, delta_iw=None) -> ImpuritySolution:
    """
    Hubbard-I: the atomic Green's function, with the hybridisation entering
    only through Dyson.

    ``G_at(z) = (1-n)/(z - e) + n/(z - e - U)`` with ``e = eps_imp - mu`` and n
    the atomic occupation per spin. Sigma is then read off the atomic problem
    and used in the lattice, which is EXACT in the atomic limit and is the
    right zeroth order for a well-localised, strongly correlated shell (rare
    earths, and Mott insulators generally). It has no quasiparticle peak by
    construction -- it cannot describe a correlated metal, and using it for one
    is the standard way to produce a spurious gap.
    """
    z = 1j * np.asarray(iw, dtype=np.float64)
    e = eps_imp - mu
    # atomic partition function over |0>, |up>, |dn>, |up dn>
    w = np.array([0.0, e, e, 2 * e + U])
    w -= w.min()
    p = np.exp(-beta * w)
    p /= p.sum()
    n = p[1] + p[3]                     # <n_sigma>
    docc = p[3]
    g_at = (1.0 - n) / (z - e) + n / (z - e - U)
    sigma = (z - e) - 1.0 / g_at
    if delta_iw is None:
        g = g_at
    else:
        g = 1.0 / (z - e - np.asarray(delta_iw) - sigma)
    return ImpuritySolution(iw=np.asarray(iw), g_iw=g, sigma_iw=sigma,
                            occupation=n, double_occupancy=docc,
                            ground_energy=float(min(0.0, e, 2 * e + U)))


# --------------------------------------------------------------------------
# Bath fitting
# --------------------------------------------------------------------------

def _iw_to_tau(g_iw: np.ndarray, iw: np.ndarray, beta: float,
               n_tau: int) -> tuple[np.ndarray, np.ndarray]:
    """Matsubara -> imaginary time, with the 1/(iw) tail handled analytically.

    ``G(tau) = -1/2 + (2/beta) sum_{n>=0} Re[e^{-i w_n tau}(G(i w_n) - 1/(i w_n))]``

    The subtraction is not optional. G(iw) ~ 1/(iw) decays too slowly for a
    truncated sum to converge, and its exact transform is the constant -1/2;
    summing the raw series instead produces Gibbs ringing at tau = 0 and beta,
    which is exactly where G(tau)^3 is largest.
    """
    tau = np.linspace(0.0, beta, n_tau)
    sub = g_iw - 1.0 / (1j * iw)
    out = np.empty(n_tau)
    step = max(1, (1 << 22) // max(len(iw), 1))          # bound the matrix
    for i in range(0, n_tau, step):
        ph = np.exp(-1j * np.outer(tau[i:i + step], iw))
        out[i:i + step] = np.real(ph @ sub)
    return tau, -0.5 + (2.0 / beta) * out


def _tau_to_iw(f_tau: np.ndarray, tau: np.ndarray, iw: np.ndarray) -> np.ndarray:
    """Imaginary time -> Matsubara by Simpson quadrature on a uniform grid."""
    n = len(tau)
    if n % 2 == 0:
        raise ValueError("_tau_to_iw: Simpson needs an odd number of tau points")
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (tau[1] - tau[0]) / 3.0
    wf = w * f_tau
    out = np.empty(len(iw), dtype=np.complex128)
    step = max(1, (1 << 22) // max(n, 1))
    for i in range(0, len(iw), step):
        out[i:i + step] = np.exp(1j * np.outer(iw[i:i + step], tau)) @ wf
    return out


def solve_ipt(delta_iw: np.ndarray, U: float, beta: float, iw: np.ndarray, *,
              n_tau: int | None = None) -> ImpuritySolution:
    """
    Iterated perturbation theory: the impurity self-energy to second order in U,
    evaluated on the Weiss field.

    ``Sigma(tau) = U^2 G_0(tau)^3``, one Fourier transform each way and a cube
    in between -- no bath fitting, no diagonalisation, and a cost independent of
    how much structure the hybridisation has.

    Written for the PARTICLE-HOLE SYMMETRIC case (half filling), where it is
    unreasonably good, and the reason is worth stating: it is exact to O(U^2) by
    construction, and it is ALSO exact in the atomic limit. Put Delta = 0 and
    G_0 = 1/(iw), so G_0(tau) = -1/2 and Sigma(tau) = -U^2/8, whose transform is
    U^2/(4 i w) -- precisely the exact atomic self-energy U/2 + (U/2)^2/(iw).
    An approximation that is exact at both ends of the coupling range
    interpolates between them far better than its order suggests, which is why
    IPT found the Mott transition before any numerically exact solver did.

    Away from half filling that accident is lost and plain IPT is unreliable;
    the interpolative "modified IPT" of Kajueter and Kotliar exists for that
    case and is not implemented here. This function does not check the symmetry
    it assumes -- pass it a symmetric problem.
    """
    iw = np.asarray(iw, dtype=np.float64)
    if n_tau is None:
        # Simpson converges as h^4 and the integrand oscillates as e^{i w tau},
        # so the grid has to resolve the HIGHEST Matsubara frequency, not the
        # physics. 8 points per iw keeps the atomic-limit check at ~1e-5; 4 was
        # 1e-3, which is visible in Z.
        n_tau = 8 * len(iw) + 1
    n_tau = int(n_tau) | 1                     # Simpson needs odd
    g0 = 1.0 / (1j * iw - np.asarray(delta_iw, dtype=np.complex128))
    tau, g0_tau = _iw_to_tau(g0, iw, beta, n_tau)
    sigma2 = _tau_to_iw(U ** 2 * g0_tau ** 3, tau, iw)
    sigma = 0.5 * U + sigma2
    g = 1.0 / (1j * iw - np.asarray(delta_iw) - sigma2)
    n = _bethe_occupation(g, beta)
    # <n_up n_dn> from the Galitskii-Migdal-like relation is not available at
    # this order without the two-particle function; report the Hartree estimate
    # and let the caller not rely on it
    return ImpuritySolution(iw=iw, g_iw=g, sigma_iw=sigma, occupation=n,
                            double_occupancy=float("nan"),
                            ground_energy=float("nan"))


def fit_bath(delta_iw: np.ndarray, iw: np.ndarray, n_bath: int, *,
             eps0=None, V0=None, n_restart: int = 4, seed: int = 0,
             weight_power: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit ``Delta(iw) = sum_p V_p^2 / (iw - eps_p)`` to a target hybridisation.

    Returns ``(eps_bath, V, residual)`` with the residual the weighted rms
    deviation in Hartree. The weight ``1/w_n^weight_power`` biases the fit to
    low frequency, which is where the physics is: a bath that is excellent at
    high frequency and wrong at the first few Matsubara points gets Z wrong.

    Bath fitting is the one uncontrolled step in ED-DMFT. Watch the residual
    across iterations -- if it grows as the loop converges the bath is too
    small to represent the emerging structure, and Z will drift with n_bath
    rather than converge.
    """
    from scipy.optimize import least_squares
    iw = np.asarray(iw, dtype=np.float64)
    target = np.asarray(delta_iw, dtype=np.complex128)
    wt = 1.0 / iw ** weight_power

    def resid(x):
        e, v = x[:n_bath], x[n_bath:]
        model = (v ** 2 / (1j * iw[:, None] - e[None, :])).sum(axis=1)
        d = (model - target) * wt
        return np.concatenate([d.real, d.imag])

    def jac(x):
        # Analytic, and worth it: Levenberg-Marquardt with a numerical Jacobian
        # spends (2 n_bath + 1) residual evaluations per step, which was the
        # single largest cost in the whole DMFT loop.
        e, v = x[:n_bath], x[n_bath:]
        den = 1j * iw[:, None] - e[None, :]
        de = (v ** 2 / den ** 2) * wt[:, None]          # d Delta / d eps_p
        dv = (2.0 * v / den) * wt[:, None]              # d Delta / d V_p
        J = np.concatenate([de, dv], axis=1)
        return np.concatenate([J.real, J.imag], axis=0)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restart)):
        if r == 0 and eps0 is not None and V0 is not None:
            x0 = np.concatenate([np.asarray(eps0, float), np.asarray(V0, float)])
        else:
            scale = max(iw[0] * 4, 0.1)
            x0 = np.concatenate([rng.normal(0.0, scale, n_bath),
                                 rng.uniform(0.05, 0.5, n_bath)])
        sol = least_squares(resid, x0, jac=jac, method="lm", max_nfev=20000)
        if best is None or sol.cost < best.cost:
            best = sol
    e, v = best.x[:n_bath], np.abs(best.x[n_bath:])
    order = np.argsort(e)
    res = float(np.sqrt(2 * best.cost / len(iw)))
    return e[order], v[order], res


# --------------------------------------------------------------------------
# Observables and continuation
# --------------------------------------------------------------------------

def quasiparticle_weight(sigma_iw: np.ndarray, iw: np.ndarray,
                         n_fit: int = 4) -> float:
    """
    ``Z = [1 - d Im Sigma / d w |_{w->0}]^-1``, from a polynomial fit through
    the lowest `n_fit` Matsubara points.

    Fitting rather than using the first point alone matters: Im Sigma(i w_n) is
    linear only as w -> 0, and at the temperatures ED-DMFT is run at the first
    Matsubara frequency is not always inside that regime. Z < 0 coming out is
    not a bug in the fit -- it is the signature of an insulator, where Im Sigma
    diverges as 1/w instead of vanishing.
    """
    iw = np.asarray(iw, dtype=np.float64)[:n_fit]
    s = np.imag(np.asarray(sigma_iw)[:n_fit])
    slope = np.polyfit(iw, s, min(3, n_fit - 1))[-2] if n_fit > 2 else s[0] / iw[0]
    return float(1.0 / (1.0 - slope))


def pade_continuation(z_in: np.ndarray, f_in: np.ndarray,
                      z_out: np.ndarray, *, n_points: int | None = None
                      ) -> np.ndarray:
    """
    Analytic continuation by Pade (Vidberg-Serene recursion).

    Continuation from Matsubara to the real axis is ill-posed, and Pade is the
    cheap option, not the safe one: it is exact for the data it is given and
    can be wildly wrong between the points, especially with noise or too many
    coefficients. Use it to LOOK at spectra, and take quantitative statements
    (Z, occupations, energies) from the Matsubara axis where they are stable.

    `n_points` truncates the input; sweeping it and keeping only features that
    survive is the standard sanity check.
    """
    z = np.asarray(z_in, dtype=np.complex128)
    f = np.asarray(f_in, dtype=np.complex128)
    if n_points is not None:
        z, f = z[:n_points], f[:n_points]
    n = len(z)
    g = np.zeros((n, n), dtype=np.complex128)
    g[0] = f
    for i in range(1, n):
        with np.errstate(divide="ignore", invalid="ignore"):
            g[i, i:] = (g[i - 1, i - 1] - g[i - 1, i:]) / \
                       ((z[i:] - z[i - 1]) * g[i - 1, i:])
    a = np.array([g[i, i] for i in range(n)])
    a = np.nan_to_num(a)

    w = np.asarray(z_out, dtype=np.complex128)
    A_prev, A_cur = np.zeros_like(w), np.full_like(w, a[0])
    B_prev, B_cur = np.ones_like(w), np.ones_like(w)
    for i in range(1, n):
        t = a[i] * (w - z[i - 1])
        A_prev, A_cur = A_cur, A_cur + t * A_prev
        B_prev, B_cur = B_cur, B_cur + t * B_prev
        big = np.abs(B_cur) > 1e100
        if big.any():
            A_cur[big] /= 1e100
            B_cur[big] /= 1e100
            A_prev[big] /= 1e100
            B_prev[big] /= 1e100
    return A_cur / B_cur


# --------------------------------------------------------------------------
# The self-consistency loops
# --------------------------------------------------------------------------

def _anderson_update(hist_x, hist_f, x, f, mixing, depth, trust=0.5):
    """One residual-based Anderson (DIIS) step on the fixed point g(x) = x.

    LINEAR MIXING CANNOT FIX AN UNSTABLE MAP, and the DMFT loop has one. Damped
    iteration replaces the Jacobian eigenvalue lambda by 1 - a + a*lambda, which
    for real lambda > 1 exceeds 1 for EVERY a in (0, 1]: no amount of damping
    stabilises it, it only slows the divergence. Measured on the Bethe lattice
    with the IPT solver at U = 2.25 D, beta = 100: the residual falls to 4e-5,
    turns around and grows by a factor 1.4 per iteration, giving lambda ~ 2.0,
    and mixing 0.4, 0.2 and 0.1 all end in the wrong phase. The metallic
    solution is repelling under simple iteration even where it exists, and a
    cold start only appears to converge because it trips the tolerance on the
    way past -- which is why a 3e-6 change in the starting hybridisation flipped
    metal to insulator.

    Anderson extrapolates over a history of residuals instead of damping, and
    handles lambda > 1. Safeguards, all of which earned their place in this
    package's Eliashberg solver: `rcond` well above machine precision (the
    residual differences go nearly collinear near a transition and an
    unregularised solve extrapolates straight into the trivial fixed point),
    and a trust region capping the step at `trust` times the current iterate.
    """
    hist_x.append(np.asarray(x).copy())
    hist_f.append(np.asarray(f).copy())
    if len(hist_x) > depth + 1:
        hist_x.pop(0)
        hist_f.pop(0)
    if len(hist_f) >= 2:
        dF = np.stack([hist_f[i + 1] - hist_f[i]
                       for i in range(len(hist_f) - 1)], axis=1)
        dX = np.stack([hist_x[i + 1] - hist_x[i]
                       for i in range(len(hist_x) - 1)], axis=1)
        try:
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=1e-8)
        except np.linalg.LinAlgError:
            gamma = None
        if gamma is not None and np.all(np.isfinite(gamma)):
            corr = mixing * f - (dX + mixing * dF) @ gamma
            limit = trust * max(float(np.max(np.abs(x))), 1e-300)
            big = float(np.max(np.abs(corr)))
            if big > limit:
                corr = corr * (limit / big)
            if np.all(np.isfinite(corr)):
                return x + corr
            hist_x.clear()
            hist_f.clear()
    return x + mixing * f


@dataclass
class DMFTResult:
    iw:        np.ndarray
    g_iw:      np.ndarray
    sigma_iw:  np.ndarray
    delta_iw:  np.ndarray
    eps_bath:  np.ndarray
    V:         np.ndarray
    Z:         float
    occupation: float
    double_occupancy: float
    converged: bool
    n_iter:    int
    history:   list
    bath_residual: float
    residual:  float = float("nan")   # the SMALLEST |dDelta| reached
    diverged:  bool = False           # the residual turned around and grew


def dmft_bethe(U: float, beta: float, *, half_bandwidth: float = 1.0,
               n_bath: int = 4, n_iw: int = 200, mu=None, max_iter: int = 40,
               tol: float = 1e-4, mix: float = 0.6, verbose: bool = False,
               seed: int = 0, delta0=None, solver: str = "ed",
               acceleration: str = "anderson", history: int = 5,
               n_stable: int = 3) -> DMFTResult:
    """
    DMFT for the single-band Hubbard model on the Bethe lattice, ED solver.

    The self-consistency is ``Delta = (D/2)^2 G``, with no k-sum -- which is
    exactly why this is the standard testbed. At half filling (`mu = U/2`,
    imposed by default) it undergoes the Mott transition, and the value of
    U_c is a number from the literature rather than anything chosen here.

    Args:
      U    : Hubbard interaction, Hartree
      beta : inverse temperature, 1/Hartree
      half_bandwidth : D. Everything scales with it; D = 1 makes U dimensionless
      mix  : mixing parameter. With `acceleration='anderson'` (the default)
             this is only the fallback linear step; with 'linear' it is the
             whole update, and see `_anderson_update` for why that is not
             enough near the transition -- the map is genuinely unstable there
             and no mixing parameter fixes it.
      acceleration : 'anderson' (default) or 'linear'.
      history : Anderson history depth.
      n_stable : how many CONSECUTIVE iterations must satisfy |dDelta| < tol
             before the loop is called converged. One is not enough, and that
             is not pedantry: where the fixed point is repelling the iterate
             passes THROUGH it, dipping below any tolerance for an iteration or
             two on the way past. Accepting that dip is what produced spurious
             "converged" metallic solutions whose position depended on where
             the sweep started. Three is cheap -- a genuinely convergent run
             satisfies it for free -- and it turns a silent wrong answer into
             converged=False.
      solver : 'ed' (exact diagonalisation with a discretised bath) or 'ipt'
             (second-order perturbation theory on the Weiss field). IPT needs
             no bath and is much faster; the two are independent approximations
             and agreeing to O(U^2) is a real cross-check of both.
      delta0 : (n_iw,) starting hybridisation, for continuation. Default is the
             non-interacting one, i.e. always approaching from the metallic
             side. Feeding the previous U's converged Delta is what lets the
             two spinodals U_c1 and U_c2 be found separately -- restarting
             from scratch at every U finds only one of them.
    """
    t = 0.5 * half_bandwidth
    iw = matsubara_frequencies(beta, n_iw)
    z = 1j * iw
    # ONE convention, everywhere below: every energy is measured from mu. The
    # bath levels that come out of `fit_bath` are already in that frame, so the
    # solver must NOT subtract mu again -- doing so shifts the bath twice and
    # silently dopes the impurity away from half filling.
    if mu is None:
        mu = 0.5 * U                       # half filling
    eps_imp = -mu                          # = -U/2, the particle-hole-symmetric point

    if acceleration not in ("anderson", "linear"):
        raise ValueError("acceleration must be 'anderson' or 'linear'")
    delta = (t ** 2 * bethe_green(z, half_bandwidth) if delta0 is None
             else np.asarray(delta0, dtype=np.complex128).copy())
    hist_x, hist_f = [], []
    best_res, best_delta, n_below = np.inf, delta.copy(), 0
    eps_b = V_b = None
    hist = []
    converged = False
    res_b = np.nan
    for it in range(max_iter):
        if solver == "ipt":
            sol = solve_ipt(delta, U, beta, iw)
            eps_b = V_b = np.zeros(0)
            res_b = 0.0
        elif solver == "ed":
            eps_b, V_b, res_b = fit_bath(delta, iw, n_bath, eps0=eps_b, V0=V_b,
                                         n_restart=(4 if eps_b is None else 1),
                                         seed=seed + it)
            par = AndersonParameters(eps_imp=eps_imp, U=U, eps_bath=eps_b, V=V_b)
            sol = solve_anderson_ed(par, beta, iw, mu=0.0)
        else:
            raise ValueError("solver must be 'ed' or 'ipt'")
        delta_new = t ** 2 * sol.g_iw
        diff = float(np.max(np.abs(delta_new - delta)))
        hist.append(dict(iter=it, diff=diff, Z=quasiparticle_weight(sol.sigma_iw, iw),
                         n=sol.occupation, docc=sol.double_occupancy,
                         bath_residual=res_b))
        if verbose:
            h = hist[-1]
            print(f"  it {it:2d}  |dDelta| {diff:.2e}  Z {h['Z']:+.4f}  "
                  f"n {h['n']:.4f}  docc {h['docc']:.4f}  bath {res_b:.2e}")
        if diff < best_res:
            best_res, best_delta = diff, delta.copy()
        n_below = n_below + 1 if diff < tol else 0
        if n_below >= n_stable:
            converged = True
            break
        if acceleration == "anderson":
            delta = _anderson_update(hist_x, hist_f, delta, delta_new - delta,
                                     mix, history)
        else:
            delta = delta + mix * (delta_new - delta)
    diverged = (not converged) and diff > 10.0 * best_res
    if not converged:
        # hand back the CLOSEST approach, not wherever the iteration wandered
        # to. Under continuation the returned Delta seeds the next point, and
        # propagating a diverged iterate poisons the whole sweep.
        delta = best_delta
        import warnings as _w
        _w.warn(
            f"dmft_bethe: not converged after {it + 1} iterations at U = {U:g}, "
            f"beta = {beta:g} (best |dDelta| = {best_res:.2e}, last = {diff:.2e})"
            + ("; the residual TURNED AROUND and grew, so the fixed point is "
               "repelling under this iteration -- see `_anderson_update`"
               if diverged else ""), RuntimeWarning, stacklevel=2)
    return DMFTResult(iw=iw, g_iw=sol.g_iw, sigma_iw=sol.sigma_iw, delta_iw=delta,
                      eps_bath=eps_b, V=V_b,
                      Z=quasiparticle_weight(sol.sigma_iw, iw),
                      occupation=sol.occupation,
                      double_occupancy=sol.double_occupancy,
                      converged=converged, n_iter=it + 1, history=hist,
                      bath_residual=res_b, residual=best_res, diverged=diverged)


def local_green_function(h_k: np.ndarray, z, sigma, mu: float = 0.0,
                         weights=None) -> np.ndarray:
    """
    ``G_loc(z) = sum_k w_k [z + mu - H(k) - Sigma(z)]^-1``, the orbital-resolved
    local Green's function of a Wannier Hamiltonian.

    Args:
      h_k   : (nk, nw, nw) H(k), Hartree -- from `core.hamiltonian.operator_k`
      z     : (nz,) complex frequencies
      sigma : (nz,) or (nz, nw, nw) local self-energy; a scalar per frequency is
              broadcast onto the diagonal
      weights : (nk,) k-point weights, default uniform

    Returns (nz, nw, nw).
    """
    h_k = np.asarray(h_k, dtype=np.complex128)
    nk, nw = h_k.shape[0], h_k.shape[-1]
    z = np.atleast_1d(np.asarray(z, dtype=np.complex128))
    s = np.asarray(sigma, dtype=np.complex128)
    if s.ndim == 1:
        s = s[:, None, None] * np.eye(nw)[None]
    w = (np.full(nk, 1.0 / nk) if weights is None
         else np.asarray(weights, dtype=np.float64) / np.sum(weights))
    eye = np.eye(nw)[None, None]
    M = (z[:, None, None, None] + mu) * eye - h_k[None] - s[:, None]
    return np.einsum('k,zkij->zij', w, np.linalg.inv(M))


def dmft_selfconsistency(h_k: np.ndarray, U: float, beta: float, *,
                         orbital: int = 0, n_bath: int = 4, n_iw: int = 200,
                         mu: float = 0.0, double_counting: float = 0.0,
                         max_iter: int = 40, tol: float = 1e-4, mix: float = 0.6,
                         weights=None, verbose: bool = False, seed: int = 0
                         ) -> DMFTResult:
    """
    Single-site, single-orbital DMFT on an explicit H(k).

    `orbital` selects which Wannier orbital carries the interaction; the rest
    are spectators that still feel Sigma through the k-integration. Use
    `double_counting` to remove whatever part of U is already in the DFT
    Hamiltonian -- there is no first-principles answer to what that is, and for
    a metal the leading effect is a shift of mu, so scan it rather than trust
    one prescription.
    """
    iw = matsubara_frequencies(beta, n_iw)
    z = 1j * iw
    nw = h_k.shape[-1]
    # the correlated orbital's on-site level, measured from mu (see dmft_bethe:
    # one frame throughout, and the solver is handed energies already shifted)
    eps_imp = float(np.real(np.mean(h_k[:, orbital, orbital]))) - mu
    sigma = np.zeros(len(iw), dtype=np.complex128)
    eps_b = V_b = None
    hist, converged, res_b = [], False, np.nan
    sol = None
    for it in range(max_iter):
        s_full = np.zeros((len(iw), nw, nw), dtype=np.complex128)
        s_full[:, orbital, orbital] = sigma - double_counting
        g_loc = local_green_function(h_k, z, s_full, mu=mu, weights=weights)
        g_ii = g_loc[:, orbital, orbital]
        g0_inv = 1.0 / g_ii + sigma - double_counting
        delta = z - eps_imp - g0_inv
        eps_b, V_b, res_b = fit_bath(delta, iw, n_bath, eps0=eps_b, V0=V_b,
                                     n_restart=(4 if eps_b is None else 1),
                                     seed=seed + it)
        par = AndersonParameters(eps_imp=eps_imp, U=U, eps_bath=eps_b, V=V_b)
        sol = solve_anderson_ed(par, beta, iw, mu=0.0)
        diff = float(np.max(np.abs(sol.sigma_iw - sigma)))
        hist.append(dict(iter=it, diff=diff,
                         Z=quasiparticle_weight(sol.sigma_iw, iw),
                         n=sol.occupation, bath_residual=res_b))
        if verbose:
            h = hist[-1]
            print(f"  it {it:2d}  |dSigma| {diff:.2e}  Z {h['Z']:+.4f}  "
                  f"n {h['n']:.4f}  bath {res_b:.2e}")
        sigma = mix * sol.sigma_iw + (1 - mix) * sigma
        if diff < tol:
            converged = True
            break
    return DMFTResult(iw=iw, g_iw=sol.g_iw, sigma_iw=sigma, delta_iw=delta,
                      eps_bath=eps_b, V=V_b, Z=quasiparticle_weight(sigma, iw),
                      occupation=sol.occupation,
                      double_occupancy=sol.double_occupancy,
                      converged=converged, n_iter=it + 1, history=hist,
                      bath_residual=res_b)


# --------------------------------------------------------------------------
# The static mean field, for comparison
# --------------------------------------------------------------------------

@dataclass
class StaticUResult:
    """Hartree (DFT+U-like) mean field on the same lattice as `dmft_bethe`."""
    U:          float
    order:      str
    magnetisation: float
    n_up:       float
    n_dn:       float
    gap:        float          # Hartree; Um for the antiferromagnet
    Z:          float          # exactly 1: a static Sigma has no slope
    converged:  bool
    n_iter:     int


def _bethe_occupation(g_iw: np.ndarray, beta: float) -> float:
    """n from a Matsubara Green's function.

    ``n = 1/2 + (2/beta) sum_{n>=0} Re G(i w_n)``. Only the REAL part is summed,
    which is why no tail correction is needed: G ~ 1/(iw) is purely imaginary at
    high frequency, so Re G already falls as 1/w^2.
    """
    return 0.5 + (2.0 / beta) * float(np.sum(np.real(g_iw)))


def static_u_bethe(U: float, beta: float, *, half_bandwidth: float = 1.0,
                   order: str = "para", n_iw: int = 2048, m_init: float = 0.3,
                   max_iter: int = 400, tol: float = 1e-10, mix: float = 0.5
                   ) -> StaticUResult:
    """
    Static Hubbard mean field on the half-filled Bethe lattice -- the DFT+U
    treatment of exactly the problem `dmft_bethe` solves.

    ``Sigma_s = U (n_{-s} - 1/2)``: a NUMBER, not a function of frequency. That
    single fact decides everything below.

    * ``order='para'`` forbids symmetry breaking. Then n_up = n_dn = 1/2 at half
      filling, Sigma vanishes identically, and the result is the
      non-interacting band structure FOR EVERY U. A static potential has no
      mechanism to move spectral weight into Hubbard bands, so it cannot make a
      paramagnetic Mott insulator at all -- not a poor description of one, none.
    * ``order='ferro'`` allows n_up != n_dn on one sublattice. A moment appears
      above the Stoner threshold ``U rho(0) = 1``, i.e. ``U = pi D / 2`` for the
      semicircular density of states.
    * ``order='afm'`` allows a two-sublattice antiferromagnet, which on a
      bipartite lattice is the case that CAN gap. It gaps for arbitrarily small
      U -- ``int rho(e)/|e| de`` diverges logarithmically for this DOS, so the
      gap equation has a solution at any coupling -- and the gap is ``U m``,
      opening exponentially from U = 0. That is a SLATER insulator, produced by
      magnetic order, and it is a qualitatively different object from a Mott
      insulator appearing at a finite U_c with no broken symmetry.

    Comparing this with `dmft_bethe` on one set of axes is the cleanest
    statement of what the frequency dependence of Sigma buys, because
    everything else -- lattice, filling, U, temperature -- is held identical.

    Note this is a mean field on the LATTICE MODEL, not the DFT+U available
    through `interfaces.quantum_espresso` (`hubbard=`) or `interfaces.vasp`
    (`ldau=`): the charge density never re-relaxes. It is the right object for a
    controlled methodological comparison and not a substitute for the other.
    """
    t = 0.5 * half_bandwidth
    iw = matsubara_frequencies(beta, n_iw)
    z = 1j * iw
    if order not in ("para", "ferro", "afm"):
        raise ValueError("order must be 'para', 'ferro' or 'afm'")

    m = 0.0 if order == "para" else float(m_init)
    converged = False
    for it in range(max_iter):
        h = 0.5 * U * m                      # +-h is the exchange splitting
        if order == "afm":
            # sublattice B is sublattice A with the spins exchanged, so the two
            # coupled equations close on one site
            gup = np.zeros_like(z)
            gdn = np.zeros_like(z)
            for _ in range(200):
                gup_new = 1.0 / (z + h - t ** 2 * gdn)
                gdn_new = 1.0 / (z - h - t ** 2 * gup)
                if max(np.abs(gup_new - gup).max(),
                       np.abs(gdn_new - gdn).max()) < 1e-14:
                    gup, gdn = gup_new, gdn_new
                    break
                gup, gdn = gup_new, gdn_new
        else:
            gup = bethe_green(z + h, half_bandwidth)
            gdn = bethe_green(z - h, half_bandwidth)
        n_up = _bethe_occupation(gup, beta)
        n_dn = _bethe_occupation(gdn, beta)
        m_new = n_up - n_dn
        if order == "para":
            m_new = 0.0
        if abs(m_new - m) < tol:
            m = m_new
            converged = True
            break
        m = mix * m_new + (1 - mix) * m

    gap = abs(U * m) if order == "afm" else 0.0
    return StaticUResult(U=float(U), order=order, magnetisation=float(m),
                         n_up=float(n_up), n_dn=float(n_dn), gap=float(gap),
                         Z=1.0, converged=converged, n_iter=it + 1)
