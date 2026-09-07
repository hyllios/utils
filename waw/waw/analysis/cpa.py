"""
Coherent Potential Approximation (CPA) for substitutional alloys in a
Wannier basis, via the configurationally-averaged one-electron Green's
function and the Dyson equation (KKR-CPA analog; Soven /
Velicky-Kirkpatrick-Ehrenreich).

A binary (or n-ary) alloy A_x B_{1-x} with one substitutional site per
primitive cell is described by per-species Wannier Hamiltonians H_alpha(R)
on a common R-grid. The medium is a virtual-crystal (VCA) average for the
hopping, with the on-site (R=0) block carrying the species disorder:

    H_ref(R) = sum_alpha x_alpha H_alpha(R)        (VCA, all R)
    v_alpha  = H_alpha(0) - H_ref(0)               (on-site fluctuation)

i.e. single-site *diagonal* disorder; off-diagonal (hopping) disorder
(Blackman-Esterling-Berk) is not implemented.

Single-site CPA replaces every random site by one coherent, non-Hermitian,
k-independent local self-energy Sigma(z) added to the reference on-site:

    G(k, z)  = [ z I - H_ref(k) - Sigma(z) ]^-1
    G_loc(z) = (1/Nk) sum_k G(k, z)                (cell-diagonal block)

The scattering off a real alpha atom embedded in the medium is
(v_alpha - Sigma); its single-site t-matrix and the CPA condition are

    t_alpha = (v_alpha - Sigma) [ I - G_loc (v_alpha - Sigma) ]^-1
    sum_alpha x_alpha t_alpha = 0                  (CPA condition)

driven to self-consistency by Sigma <- Sigma + tbar (I + G_loc tbar)^-1,
tbar = sum_alpha x_alpha t_alpha. For a single band this reduces to the
VKE scalar equation Sigma = <v> - G_loc (v_A - Sigma)(v_B - Sigma), and in
the atomic limit (no hopping) to the exact split DOS
sum_alpha x_alpha delta(E - V_alpha).

Outputs: disorder-averaged DOS = -1/pi Im Tr G_loc, and the Bloch spectral
function A_B(k, E) = -1/pi Im Tr [z I - H_ref(k) - Sigma(z)]^-1 (Im Sigma
is the disorder lifetime). For a collinear magnetic alloy run one spin
channel at a time (independent nw x nw CPA each).

Atomic units throughout (Hartree, Bohr); convert at the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, operator_k
from ..units import BOHR_RADIUS_M, E_CHARGE, HBAR_SI
from .dos import _uniform_mesh


# --------------------------------------------------------------------------
# Alloy model: VCA reference hopping + per-species on-site fluctuations
# --------------------------------------------------------------------------

@dataclass
class AlloyModel:
    """Single-site diagonal-disorder alloy in a Wannier basis."""
    H_ref:          HamiltonianR         # VCA reference (medium hopping + <V>)
    v_species:      list[np.ndarray]     # per-species on-site fluctuation (nw, nw)
    concentrations: np.ndarray           # (n_species,) sum to 1
    nw:             int


def _as_hr_list(hamiltonians) -> list[HamiltonianR]:
    hrs = list(hamiltonians)
    R0 = np.asarray(hrs[0].R_vectors)
    nw = hrs[0].nw
    for h in hrs[1:]:
        if h.nw != nw:
            raise ValueError("all species Hamiltonians must share nw")
        if not np.array_equal(np.asarray(h.R_vectors), R0):
            raise ValueError("all species Hamiltonians must share the R-grid")
    return hrs


def virtual_crystal(hamiltonians, concentrations) -> HamiltonianR:
    """Virtual-crystal (VCA) Hamiltonian H_ref(R) = sum_alpha x_alpha
    H_alpha(R) on the shared R-grid. A cheap baseline and the CPA medium's
    hopping (Sigma = 0 recovers this)."""
    hrs = _as_hr_list(hamiltonians)
    x = np.asarray(concentrations, dtype=np.float64)
    x = x / x.sum()
    H_R = sum(xa * h.H_R for xa, h in zip(x, hrs))
    # carry the lattice through: `cpa_conductivity` needs the cell volume,
    # and losing it here forces every caller to pass it back in by hand
    return HamiltonianR(H_R=H_R, R_vectors=hrs[0].R_vectors,
                        degen=hrs[0].degen, nw=hrs[0].nw,
                        real_lattice=getattr(hrs[0], 'real_lattice', None))


def build_alloy(hamiltonians, concentrations) -> AlloyModel:
    """Assemble an `AlloyModel` from per-species Wannier Hamiltonians on a
    common R-grid. The VCA average is the medium; each species' on-site
    (R=0) block minus the VCA on-site is its scattering potential."""
    hrs = _as_hr_list(hamiltonians)
    x = np.asarray(concentrations, dtype=np.float64)
    x = x / x.sum()
    if len(x) != len(hrs):
        raise ValueError("concentrations and Hamiltonians length mismatch")
    H_ref = virtual_crystal(hrs, x)

    R = np.asarray(hrs[0].R_vectors)
    i0 = int(np.where((R == 0).all(axis=1))[0][0])
    V_ref = H_ref.H_R[i0].detach().cpu().numpy()
    v_species = [h.H_R[i0].detach().cpu().numpy() - V_ref for h in hrs]
    return AlloyModel(H_ref=H_ref, v_species=v_species,
                      concentrations=x, nw=hrs[0].nw)


# --------------------------------------------------------------------------
# Single-site CPA self-consistency
# --------------------------------------------------------------------------

def _cpa_single_energy(H_k, v_list, x_list, z, Sigma0, max_iter, tol, mix):
    """Converge the coherent self-energy Sigma at a single complex energy z.
    H_k: (nk, nw, nw) VCA H(k); returns (Sigma, G_loc, residual, iters)."""
    nw = H_k.shape[-1]
    I = np.eye(nw, dtype=np.complex128)
    Sigma = Sigma0.copy()
    res = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        G = np.linalg.inv(z * I[None] - H_k - Sigma[None])   # (nk, nw, nw)
        Gloc = G.mean(axis=0)                                # (nw, nw)
        tbar = np.zeros((nw, nw), dtype=np.complex128)
        for x, v in zip(x_list, v_list):
            d = v - Sigma
            t = d @ np.linalg.inv(I - Gloc @ d)
            tbar += x * t
        res = np.abs(tbar).max()
        if res < tol:
            break
        dSigma = tbar @ np.linalg.inv(I + Gloc @ tbar)
        Sigma = Sigma + mix * dSigma
    return Sigma, Gloc, res, it


@dataclass
class CPAResult:
    """Converged CPA medium on an energy grid (atomic units)."""
    energies:  np.ndarray    # (nE,) Hartree
    Sigma:     np.ndarray    # (nE, nw, nw) complex coherent self-energy
    G_loc:     np.ndarray    # (nE, nw, nw) complex site-diagonal Green's fn
    dos:       np.ndarray    # (nE,) states / Hartree / cell (per spin channel)
    residual:  np.ndarray    # (nE,) final |sum_a x_a t_a|
    iters:     np.ndarray    # (nE,) iterations to convergence
    eta:       float


def coherent_potential(
    model:    AlloyModel,
    mesh:     tuple[int, int, int],
    energies: np.ndarray,
    eta:      float = 1e-3,
    max_iter: int = 300,
    tol:      float = 1e-8,
    mix:      float = 1.0,
) -> CPAResult:
    """Solve the single-site CPA equations on `mesh` for each energy.

    Energies are swept in order with warm-started Sigma (continuation), so
    a fine, ordered grid converges fast. `eta` (Hartree) is the small
    imaginary part E + i eta at which G is evaluated.
    """
    energies = np.asarray(energies, dtype=np.float64)
    kpts = _uniform_mesh(mesh)
    H_k = operator_k(model.H_ref.H_R, model.H_ref.R_vectors,
                     model.H_ref.degen, kpts).detach().cpu().numpy()   # (nk, nw, nw)
    v_list = [np.asarray(v, dtype=np.complex128) for v in model.v_species]
    x_list = list(model.concentrations)
    nw = model.nw

    nE = len(energies)
    Sigma_E = np.empty((nE, nw, nw), dtype=np.complex128)
    Gloc_E = np.empty((nE, nw, nw), dtype=np.complex128)
    dos = np.empty(nE, dtype=np.float64)
    residual = np.empty(nE, dtype=np.float64)
    iters = np.empty(nE, dtype=np.int64)

    Sigma = np.zeros((nw, nw), dtype=np.complex128)   # VCA start at first energy
    for ie, E in enumerate(energies):
        z = E + 1j * eta
        Sigma, Gloc, res, it = _cpa_single_energy(
            H_k, v_list, x_list, z, Sigma, max_iter, tol, mix)
        Sigma_E[ie] = Sigma
        Gloc_E[ie] = Gloc
        dos[ie] = -np.trace(Gloc).imag / np.pi
        residual[ie] = res
        iters[ie] = it
    return CPAResult(energies=energies, Sigma=Sigma_E, G_loc=Gloc_E,
                     dos=dos, residual=residual, iters=iters, eta=eta)


# --------------------------------------------------------------------------
# Bloch spectral function A_B(k, E)
# --------------------------------------------------------------------------

@dataclass
class CPABlochSpectralFunction:
    """Disorder-averaged Bloch spectral function A_B(k, E) (atomic units)."""
    kpath:    np.ndarray     # (nk, 3) crystal-coord k-path
    energies: np.ndarray     # (nE,) Hartree
    A:        np.ndarray     # (nk, nE) A_B = -1/pi Im Tr G(k, E)
    dos:      np.ndarray     # (nE,) averaged DOS (per cell / spin)
    cpa:      CPAResult


def cpa_bloch_spectral_function(
    model:    AlloyModel,
    kpath:    np.ndarray,
    energies: np.ndarray,
    mesh:     tuple[int, int, int],
    eta:      float = 1e-3,
    cpa:      CPAResult | None = None,
    **cpa_kwargs,
) -> CPABlochSpectralFunction:
    """A_B(k, E) = -1/pi Im Tr [z I - H_ref(k) - Sigma(z)]^-1 along `kpath`,
    with Sigma(z) from the converged CPA medium (computed on `mesh`, or
    reused if `cpa` is passed). Same eta for the medium and the map."""
    energies = np.asarray(energies, dtype=np.float64)
    kpath = np.asarray(kpath, dtype=np.float64)
    if cpa is None:
        cpa = coherent_potential(model, mesh, energies, eta=eta, **cpa_kwargs)
    elif not np.array_equal(cpa.energies, energies):
        raise ValueError("supplied CPAResult energies differ from `energies`")

    # The Dyson inversion is not CPA-specific -- `analysis.spectral` owns it,
    # and takes the CPA medium's local Sigma(z) like any other self-energy.
    from .spectral import bloch_spectral_function

    sf = bloch_spectral_function(model.H_ref, kpath, energies, cpa.Sigma,
                                 eta=eta)
    return CPABlochSpectralFunction(kpath=kpath, energies=energies, A=sf.A,
                                    dos=cpa.dos, cpa=cpa)


# --------------------------------------------------------------------------
# Kubo-Greenwood conductivity of the CPA medium, with Velicky vertex
# corrections
# --------------------------------------------------------------------------

@dataclass
class CPAConductivity:
    """dc conductivity of a CPA medium (atomic units, e^2/(hbar a_0))."""
    sigma:        float   # with vertex corrections if they were requested
    sigma_bubble: float   # the bare bubble, for comparison
    energy:       float   # Hartree, where it was evaluated
    ward:         float   # spectral radius of U.chi -- must be <= 1
    vertex_frac:  float   # (sigma - sigma_bubble) / sigma_bubble

    def resistivity_microohm_cm(self) -> float:
        """1 / sigma converted from e^2/(hbar a_0) to microOhm cm."""
        return 1.0e8 / (self.sigma * _SIGMA_AU_TO_SI)


#: Atomic unit of conductivity, e^2 / (hbar a_0), in S/m.
_SIGMA_AU_TO_SI = (E_CHARGE ** 2) / (HBAR_SI * BOHR_RADIUS_M)


def cpa_conductivity(
    model:        AlloyModel,
    cpa:          CPAResult,
    energy_index: int,
    recip_lattice: np.ndarray,
    mesh:         tuple[int, int, int],
    *,
    direction:          int = 0,
    vertex_corrections: bool = True,
    cell_volume_bohr3:  float | None = None,
    kchunk:             int = 2000,
) -> CPAConductivity:
    """
    Kubo-Greenwood dc conductivity of the converged CPA medium,

        sigma_aa = (pi / V N_k) sum_k Tr[ v_a A(k,E) v_a A(k,E) ]  + vertex,
        A(k,E)   = (i/2pi) [ G(k,E) - G(k,E)^dagger ],

    with the Velicky ladder (Phys. Rev. 184, 614 (1969)) resummed in the
    nw^2-dimensional pair space:

        Lambda = (1 - U chi)^-1 U Phi,
        Phi_(mu nu)             = <[G^R v_a G^A]_(mu nu)>_k
        chi_(rho sig),(mu nu)   = <G^R_(rho mu) G^A_(nu sig)>_k
                                  - G^R_loc,(rho mu) G^A_loc,(nu sig)
        U_(mu nu),(rho sig)     = sum_s x_s (t_s)_(mu rho) conj((t_s)_(nu sig))

    THE SUBTRACTION IN chi IS NOT OPTIONAL and is the whole content of Butler,
    Phys. Rev. B 31, 3260 (1985), Eq. (67): the ladder propagator is X^(0n)
    with the SITE-DIAGONAL n = 0 term excluded, because same-site repeated
    scattering is already resummed inside the single-site t-matrix. Leaving it
    in double-counts that scattering, and the symptom is unmistakable once you
    look for it -- the ladder's spectral radius comes out at 1.5-3.9 instead
    of 1, so (1 - U chi)^-1 is not summing anything. With the subtraction the
    radius is 1 to five decimals, which is the Ward identity of particle
    conservation (the diffusion pole). `ward` is returned so a caller can
    check it; anything above 1 means the result is meaningless.

    Because 1 - U chi is therefore SINGULAR by construction, it is solved by
    least squares rather than inverted: the current vertex is orthogonal to
    the conserving mode, so the physical answer lives in the complement.

    Consistency requirements, all of which bite silently:

      * `mesh` and the CPA's own mesh should match, and `cpa.G_loc` is used
        for the t-matrices rather than a re-integrated one -- rebuilding them
        from a different mesh breaks the CPA condition sum_s x_s t_s = 0 and
        with it the ladder;
      * ``cpa.energies[energy_index]`` must be the energy of interest. Put it
        ON the grid: evaluating G a few meV from where Sigma was computed
        breaks the same condition.

    Parameters
    ----------
    direction : Cartesian component a of v_a (0, 1, 2).
    vertex_corrections : set False for the bare bubble.
    cell_volume_bohr3 : taken from ``model.H_ref.real_lattice`` if omitted.

    Returns
    -------
    CPAConductivity, atomic units.
    """
    from ._fourier_derivs import h_and_grad_cart_batch

    if cell_volume_bohr3 is None:
        if getattr(model.H_ref, "real_lattice", None) is None:
            raise ValueError(
                "cpa_conductivity: cell_volume_bohr3 not given and "
                "model.H_ref carries no real_lattice to take it from."
            )
        cell_volume_bohr3 = abs(np.linalg.det(model.H_ref.real_lattice))

    E = float(cpa.energies[energy_index])
    Sigma = cpa.Sigma[energy_index]
    Gloc = cpa.G_loc[energy_index]
    eta = cpa.eta
    nw = model.nw
    kpts = _uniform_mesh(mesh)
    nk = len(kpts)
    I = np.eye(nw, dtype=np.complex128)

    bub = 0.0
    Phi = np.zeros((nw, nw), dtype=np.complex128)
    chi = np.zeros((nw, nw, nw, nw), dtype=np.complex128)
    for lo in range(0, nk, kchunk):
        kk = kpts[lo:lo + kchunk]
        H0, gc = h_and_grad_cart_batch(model.H_ref, kk, recip_lattice)
        H0 = H0.detach().cpu().numpy()
        va = gc.detach().cpu().numpy()[:, direction]
        GR = np.linalg.inv((E + 1j * eta) * I[None] - H0 - Sigma[None])
        GA = GR.conj().transpose(0, 2, 1)
        A = (1j / (2 * np.pi)) * (GR - GA)
        bub += np.einsum("kij,kjl,klm,kmi->", va, A, va, A, optimize=True).real
        if vertex_corrections:
            Phi += np.einsum("kij,kjl,klm->im", GR, va, GA, optimize=True)
            chi += np.einsum("krm,kns->rsmn", GR, GA, optimize=True)
    sigma_bub = np.pi * bub / (cell_volume_bohr3 * nk)
    if not vertex_corrections:
        return CPAConductivity(sigma=sigma_bub, sigma_bubble=sigma_bub,
                               energy=E, ward=float("nan"), vertex_frac=0.0)

    Phi /= nk
    chi /= nk
    chi -= np.einsum("rm,ns->rsmn", Gloc, Gloc.conj().T)     # Butler Eq. (67)

    ts = [v_a - Sigma for v_a in model.v_species]
    ts = [d @ np.linalg.inv(I - Gloc @ d) for d in ts]
    U = np.zeros((nw, nw, nw, nw), dtype=np.complex128)
    for x, t in zip(model.concentrations, ts):
        U += x * np.einsum("mr,ns->mnrs", t, t.conj())

    Um = U.reshape(nw * nw, nw * nw)
    Cm = chi.reshape(nw * nw, nw * nw)
    M = Um @ Cm
    ward = float(np.abs(np.linalg.eigvals(M)).max())
    Lam = np.linalg.lstsq(np.eye(nw * nw) - M, Um @ Phi.reshape(-1),
                          rcond=1e-10)[0]
    dsig = float(np.real(np.vdot(Phi.reshape(-1), Lam))
                 / (2.0 * np.pi * cell_volume_bohr3))
    return CPAConductivity(sigma=sigma_bub + dsig, sigma_bubble=sigma_bub,
                           energy=E, ward=ward,
                           vertex_frac=dsig / sigma_bub if sigma_bub else 0.0)
