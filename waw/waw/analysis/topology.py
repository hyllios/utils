"""
Berry curvature, Chern numbers, and anomalous Hall conductivity from the
Wannier Hamiltonian.

Berry curvature and Chern numbers are topological: in fractional
(crystal) k-space they're metric-independent, so berry_curvature()/
chern_number() need no recip_lattice. berry_curvature_cartesian() and
anomalous_hall_conductivity() do need recip_lattice since AHC is a
metric-dependent transport coefficient, and use the same inv_recip
Jacobian transform as effective_mass.py's Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, position_operator_k
from ..core.distributions import minus_fermi_deriv
from ..units import (
    BOHR_TO_ANG, ANG_TO_BOHR, HARTREE_TO_EV, EV_TO_HARTREE,
    E_CHARGE, HBAR_SI, K_B_HARTREE, register_si_unit, register_from_si_unit,
)
from ._fourier_derivs import h_and_k_derivatives_frac, h_and_grad_frac_batch
from .shift_current import (
    _kmesh_spacing,
    KUBO_ADPT_SMR_FAC_DEFAULT, KUBO_ADPT_SMR_MAX_DEFAULT, _ADAPTIVE_ETA_FLOOR,
)

# k-points per batch for the vectorized curvature/AHC paths; bounds peak
# memory of the (chunk, 3, nw, nw) intermediates.
_KCHUNK = 4096

_AXIAL_PAIRS = ((1, 2), (2, 0), (0, 1))   # (alpha, beta) for axial x, y, z
_DEGEN_TOL = 1e-14


def _adaptive_eta_per_band(
    v_n: torch.Tensor, delta_k: float,
    prefactor: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    max_width: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> torch.Tensor:
    """
    Per-band, per-k adaptive smearing width for the AHC's Fermi-level
    step function (YWVS07 adaptive smearing, generalized from
    `shift_current._adaptive_eta`'s band-*pair* relative-velocity width
    -- appropriate for a delta(omega-(E_n-E_m)) at finite frequency --
    to a single band's own group velocity, the natural width for
    smoothing where band n individually crosses the Fermi level in a DC
    (omega=0) quantity like the AHC):

        eta_n(k) = clamp(|v_n(k)| * Delta_k * prefactor, floor, max_width)

    Args:
      v_n     : (nk, nw, 3) real, Hartree*Bohr -- band group velocities
                (diagonal of the Cartesian-rotated H(k) gradient)
      delta_k : Bohr^-1, from `shift_current._kmesh_spacing`
      prefactor, max_width: as in `shift_current._adaptive_eta`

    Returns eta_n: (nk, nw) real, Hartree.
    """
    speed = torch.linalg.norm(v_n, dim=-1)
    return torch.clamp(speed * delta_k * prefactor, min=_ADAPTIVE_ETA_FLOOR, max=max_width)


def _smooth_occupation(eig: torch.Tensor, fermi_energy: float, eta: torch.Tensor) -> torch.Tensor:
    """
    Gaussian-smoothed (error-function) occupation step, with a per-
    (band,k) adaptive width `eta` in place of a fixed smearing
    temperature:

        occ_n(k) = 0.5 * [1 - erf((E_n(k) - E_F) / (eta_n(k) * sqrt(2)))]

    matching the `sigma = eta/sqrt(2)` Gaussian convention already used
    for smearing elsewhere in this project (e.g.
    `shift_current._accumulate_sc_k`). Reduces to the sharp T=0 step
    `(E_n(k) < E_F)` in the eta -> 0 limit.

    Args:
      eig: (nk, nw) real, Hartree
      fermi_energy: scalar, Hartree
      eta: (nk, nw) real, Hartree (per-band adaptive width, e.g. from
           `_adaptive_eta_per_band`)

    Returns occ: (nk, nw) real in [0, 1].
    """
    return 0.5 * (1.0 - torch.erf((eig - fermi_energy) / (eta * np.sqrt(2.0))))


@dataclass
class BerryCurvature:
    """Berry curvature of every band on a set of fractional k-points."""
    kpts:      np.ndarray   # (nk, 3) fractional coordinates
    plane:     tuple        # (a, b) crystal directions of this curvature component
    curvature: np.ndarray   # (nk, nw), Omega_n^{ab}(k)


@dataclass
class BerryCurvatureCartesian:
    """Cartesian (axial-vector) Berry curvature of every band, in Bohr^2."""
    kpts:      np.ndarray   # (nk, 3) fractional coordinates
    curvature: np.ndarray   # (nk, nw, 3), Bohr^2: (Omega^yz, Omega^zx, Omega^xy) = (Omega_x, Omega_y, Omega_z)


@dataclass
class ChernNumber:
    """Chern number of one or more band groups, via the FHS lattice method."""
    plane:       tuple    # (a, b) crystal directions integrated over
    fixed_index: int      # the third crystal direction, held fixed
    fixed_value: float
    mesh:        tuple    # (N_a, N_b) mesh density used
    groups:      tuple    # tuple of tuples; 0-based band indices in each group
    chern:       np.ndarray   # (len(groups),) Chern number of each group (exact integer up to numerical noise)


@dataclass
class AHCResult:
    """
    Anomalous Hall conductivity vs. Fermi energy, atomic units. Convert
    with ``waw.units.to_si_units(result.sigma, "hall_conductivity",
    cell_volume_bohr3=abs(np.linalg.det(real_lattice)))`` for S/cm.
    """
    fermi_energies: np.ndarray   # (nf,) Hartree
    sigma:          np.ndarray   # (nf, 3) Bohr^2, bare curvature average (axial: yz, zx, xy)
    mesh:           tuple        # (N_a, N_b, N_c) k-mesh density used


@dataclass
class AnomalousNernstResult:
    """
    Intrinsic anomalous Nernst conductivity vs. temperature, atomic units.
    Convert with ``waw.units.to_si_units(result.alpha, "anomalous_nernst",
    cell_volume_bohr3=...)`` for A/(m K). The Kelvin temperature axis is
    recovered from ``kT_values`` via ``waw.units.K_B_HARTREE`` (kT/k_B).
    """
    kT_values:  np.ndarray     # (nT,) Hartree
    mu:         float          # Hartree, chemical potential the scan is taken at
    alpha:      np.ndarray     # (nT, 3) atomic units, axial vector (alpha_yz, alpha_zx, alpha_xy)
    energies:   np.ndarray     # (nE,) Hartree, the sigma^A(E) grid the Mott integral used
    sigma_of_E: np.ndarray     # (nE, 3) Bohr^2, the AHC-vs-energy curve the transform used
    mesh:       tuple          # (N_a, N_b, N_c) k-mesh density used


def _omega_tensor_frac_batch(H0: torch.Tensor, grad_frac: torch.Tensor) -> tuple:
    """
    Batched fractional Berry-curvature tensor for a stack of k-points:

        Omega_ab[k, a, b, n] = -2 sum_{m != n} Im[G_a[n,m] G_b[m,n]] / (E_n-E_m)^2

    with G_a = U^dagger (dH/dk_a) U in the H(k) eigenbasis. Bands within a
    degenerate group (denominator below `_DEGEN_TOL`) get 0 -- per-band
    curvature is ill-defined at an exact degeneracy; only the group sum
    is meaningful.

    Returns (eig, omega): eig (nk, nw) real, omega (nk, 3, 3, nw) real.
    """
    eig, U = torch.linalg.eigh(H0)                                   # (nk,nw),(nk,nw,nw)
    G = torch.einsum('kni,kanm,kmj->kaij', U.conj(), grad_frac, U)   # (nk,3,nw,nw)

    dE = eig[:, :, None] - eig[:, None, :]                           # (nk,nw,nw) E_n - E_m
    denom = dE * dE
    inv = torch.where(denom < _DEGEN_TOL, torch.zeros_like(denom), 1.0 / denom)

    nk, _, nw, _ = G.shape
    omega = torch.zeros((nk, 3, 3, nw), dtype=eig.dtype, device=eig.device)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        # prod[k,n,m] = G_a[k,n,m] * G_b[k,m,n]
        prod = G[:, a] * G[:, b].transpose(-1, -2)
        w = -2.0 * (prod.imag * inv).sum(dim=-1)                     # sum over m -> (nk,nw)
        omega[:, a, b] = w
        omega[:, b, a] = -w
    return eig, omega


def berry_curvature(
    hr: HamiltonianR, kpts_frac: np.ndarray, plane: tuple = (0, 1),
) -> BerryCurvature:
    """
    Berry curvature Omega_n^{ab}(k) for every band n, at a set of
    fractional k-points, for the (a, b) pair of crystal directions given
    by `plane` (default (0, 1): the first two crystal axes).

        Omega_n^{ab}(k) = -2 sum_{m != n} Im[<n|dH/dk_a|m><m|dH/dk_b|n>]
                            / (E_n - E_m)^2

    Dimensionless when integrated over fractional k in [0,1) x [0,1)
    (see `chern_number`) -- no recip_lattice is needed since this is a
    purely topological quantity.
    """
    a, b = plane
    kpts = np.asarray(kpts_frac, dtype=np.float64)
    nk = len(kpts)
    curvature = np.empty((nk, hr.nw))

    for lo in range(0, nk, _KCHUNK):
        chunk = kpts[lo:lo + _KCHUNK]
        H0, grad = h_and_grad_frac_batch(hr, chunk)
        _, omega = _omega_tensor_frac_batch(H0, grad)
        curvature[lo:lo + _KCHUNK] = omega[:, a, b].cpu().numpy()

    return BerryCurvature(kpts=kpts, plane=plane, curvature=curvature)


def berry_curvature_cartesian(
    hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray,
) -> BerryCurvatureCartesian:
    """
    Cartesian axial-vector Berry curvature Omega_n(k) = (Omega_n^x,
    Omega_n^y, Omega_n^z) = (Omega_n^{yz}, Omega_n^{zx}, Omega_n^{xy}),
    in Bohr^2, per band, from the plain band-Kubo formula (perturbation
    theory in H(k) alone, no position-operator input).

    Summed over occupied bands this equals the "J2" term of the
    WYSV06/CTVR06 decomposition (Eq. 51 / LVTS12 Eq. 6): occ-occ cross
    terms cancel by antisymmetry of dH/dk, leaving only the occ-unocc sum
    J2 also contains. It is not the full quantity postw90 reports for
    `berry_task = ahc` when the Wannier functions have nonzero internal
    position structure (AA(R) != 0 for R != 0) -- the full answer is
    J0 + J1 + J2, with J0/J1 built from the position operator AA(R) (see
    `wannier_interpolated_curvature`). J2 alone is exact only for an
    idealized atomic-like Wannier basis (AA(R) = 0 for R != 0); otherwise
    it's a useful diagnostic, not a drop-in replacement for postw90.

    Obtained from the fractional-space tensor via the same Jacobian
    transform effective_mass.py uses for the Hessian:

        Omega_cart[i, j] = sum_ab inv_recip[i, a] inv_recip[j, b] Omega_frac[a, b]

    `recip_lattice` in Bohr^-1 (core convention; rows b1, b2, b3).
    """
    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.float64)

    kpts = np.asarray(kpts_frac, dtype=np.float64)
    nk = len(kpts)
    curvature = np.empty((nk, hr.nw, 3))

    for lo in range(0, nk, _KCHUNK):
        chunk = kpts[lo:lo + _KCHUNK]
        H0, grad = h_and_grad_frac_batch(hr, chunk)
        _, omega_frac = _omega_tensor_frac_batch(H0, grad)          # (nc,3,3,nw)
        # cart[k,comp,n] = sum_ab inv_recip[i,a] inv_recip[j,b] omega_frac[k,a,b,n]
        cart = torch.stack([
            torch.einsum('a,b,kabn->kn', inv_recip[i], inv_recip[j], omega_frac)
            for (i, j) in _AXIAL_PAIRS
        ], dim=-1)                                                  # (nc, nw, 3)
        curvature[lo:lo + _KCHUNK] = cart.cpu().numpy()

    return BerryCurvatureCartesian(kpts=kpts, curvature=curvature)


def _eigvecs_at(hr: HamiltonianR, k_frac: np.ndarray, band_group: tuple) -> np.ndarray:
    """Eigenvectors for the given (sorted-by-energy) band indices at k_frac."""
    H0, _, _ = h_and_k_derivatives_frac(hr, np.asarray(k_frac, dtype=np.float64))
    _, eigvecs = np.linalg.eigh(H0)
    return eigvecs[:, list(band_group)]   # (nw, len(band_group))


def _link_variable(V1: np.ndarray, V2: np.ndarray) -> complex:
    """U(1) link det(V1^dagger V2)/|det(...)|; reduces to a plain overlap for a single band."""
    d = np.linalg.det(V1.conj().T @ V2)
    return d / abs(d)


def _lattice_chern_number(
    hr: HamiltonianR, band_group: tuple, plane: tuple,
    fixed_index: int, fixed_value: float, mesh: tuple,
) -> float:
    """
    Chern number of the subspace spanned by `band_group`, via the discrete
    Fukui-Hatsugai-Suzuki (FHS) lattice method: gauge-invariant plaquette
    Berry phases summed over the mesh. Exactly quantized to an integer
    multiple of 2*pi regardless of mesh density, and robust to any
    unitary gauge choice within `band_group` at each k-point, including
    exact internal degeneracies (unlike a per-band Kubo-formula sum,
    whose energy denominator is ill-defined at a degeneracy).
    """
    a, b = plane
    Na, Nb = mesh

    V = np.empty((Na, Nb, hr.nw, len(band_group)), dtype=np.complex128)
    for i in range(Na):
        for j in range(Nb):
            k = np.zeros(3)
            k[a] = i / Na
            k[b] = j / Nb
            k[fixed_index] = fixed_value
            V[i, j] = _eigvecs_at(hr, k, band_group)

    total_phase = 0.0
    for i in range(Na):
        for j in range(Nb):
            ip, jp = (i + 1) % Na, (j + 1) % Nb
            Ux_ij  = _link_variable(V[i, j],  V[ip, j])
            Uy_ipj = _link_variable(V[ip, j], V[ip, jp])
            Ux_ijp = _link_variable(V[i, jp], V[ip, jp])
            Uy_ij  = _link_variable(V[i, j],  V[i, jp])
            plaquette = Ux_ij * Uy_ipj * np.conj(Ux_ijp) * np.conj(Uy_ij)
            total_phase += np.angle(plaquette)

    return total_phase / (2 * np.pi)


def chern_number(
    hr:           HamiltonianR,
    plane:        tuple = (0, 1),
    fixed_index:  int | None = None,
    fixed_value:  float = 0.0,
    mesh:         tuple = (20, 20),
    groups:       tuple | None = None,
) -> ChernNumber:
    """
    Chern number of one or more band groups, over the 2D torus spanned by
    the two crystal directions in `plane`, at a fixed value of the third
    direction, via the Fukui-Hatsugai-Suzuki lattice method (see
    `_lattice_chern_number`).

    Args:
      plane       : (a, b) crystal directions to integrate over
      fixed_index : the third crystal direction, held fixed; defaults to
                    whichever index isn't in `plane`
      fixed_value : fractional coordinate to hold that direction at
      mesh        : (N_a, N_b) uniform mesh density
      groups      : sequence of band-index tuples, each treated as one
                    (possibly multi-band) group. Defaults to each band as
                    its own singleton group ((0,), (1,), ..., (nw-1,)).
                    Pass e.g. groups=((0, 1, 2),) to get the single joint
                    invariant of a degenerate/occupied manifold -- the
                    robust choice whenever bands within the group of
                    interest are (near-)degenerate anywhere on the mesh.

    A gapped 3D crystal should give the same Chern number for any
    `fixed_value` (it's a topological invariant of the whole 2D slice);
    if it varies with fixed_value, the slices aren't all gapped (e.g. a
    Weyl point separates them), which is itself useful information.
    """
    a, b = plane
    if fixed_index is None:
        fixed_index = ({0, 1, 2} - {a, b}).pop()
    if groups is None:
        groups = tuple((n,) for n in range(hr.nw))

    chern = np.array([
        _lattice_chern_number(hr, g, plane, fixed_index, fixed_value, mesh)
        for g in groups
    ])

    return ChernNumber(plane=plane, fixed_index=fixed_index, fixed_value=fixed_value,
                        mesh=mesh, groups=tuple(groups), chern=chern)


def _jjp_jjm_batch(dH_eig: torch.Tensor, eig: torch.Tensor,
                    fermi_energy: float) -> tuple:
    """
    Batched JJ+/JJ- (one Fermi energy), in the H(k) eigenbasis, for a
    k-stack (wannier90's `wham_get_JJp_JJm_list`):

        JJp[n, m] = i dH_eig[n,m] / (E_m - E_n)   (E_n > Ef, E_m < Ef)
        JJm[m, n] = i dH_eig[m,n] / (E_n - E_m)   (same pair, transposed slot)

    dH_eig: (nk,3,nw,nw); eig: (nk,nw), Hartree; fermi_energy: Hartree.
    Returns (JJp, JJm) each (nk,3,nw,nw) complex, in the H(k) eigenbasis.
    """
    En = eig[:, :, None]                         # (nk,nw,1) row index
    Em = eig[:, None, :]                         # (nk,1,nw) col index
    dE = eig[:, :, None] - eig[:, None, :]       # (nk,nw,nw) E_n - E_m

    maskP = (En > fermi_energy) & (Em < fermi_energy)   # n unocc, m occ
    invP = torch.where(maskP, -1.0 / dE, torch.zeros_like(dE))   # 1/(E_m - E_n)
    JJp = 1j * dH_eig * invP[:, None, :, :]

    maskM = (En < fermi_energy) & (Em > fermi_energy)   # row occ, col unocc
    invM = torch.where(maskM, -1.0 / dE, torch.zeros_like(dE))   # 1/(E_col - E_row)
    JJm = 1j * dH_eig * invM[:, None, :, :]
    return JJp, JJm


def _jjp_jjm_from_occ(dH_eig: torch.Tensor, eig: torch.Tensor, occ: torch.Tensor) -> tuple:
    """
    JJ+/JJ- built from an explicit per-band occupation weight `occ`
    (nk,nw) rather than a single Fermi-energy threshold -- needed for
    postw90 `gyrotropic`'s per-band (not per-Fermi-sea) curvature/
    orbital-moment extraction (a "fake occupation" with only one band
    set to 1). Generalizes `_jjp_jjm_batch`, reducing to it for
    `occ = (eig < fermi)`.

    wannier90's `wham_get_JJp_JJm_list` occ-branch:

        JJp[i,j] = i dH_eig[i,j] / (E_j - E_i)   occ[i] < 0.5 (unocc), occ[j] > 0.5 (occ)
        JJm[i,j] = i dH_eig[i,j] / (E_j - E_i)   occ[i] > 0.5 (occ),   occ[j] < 0.5 (unocc)

    Args:
      dH_eig: (nk,3,nw,nw) complex, H(k)-eigenbasis-rotated dH/dk
      eig   : (nk,nw) real, Hartree (or any consistent energy unit)
      occ   : (nk,nw) real, 0/1 (or any values that binarize at 0.5)

    Returns (JJp, JJm), each (nk,3,nw,nw) complex, H(k) eigenbasis.
    """
    dE = eig[:, None, :] - eig[:, :, None]        # (nk,nw,nw): E_j - E_i
    occ_row = occ[:, :, None]                     # (nk,nw,1): occ[i]
    occ_col = occ[:, None, :]                     # (nk,1,nw): occ[j]

    maskP = (occ_row < 0.5) & (occ_col > 0.5)     # i unocc, j occ
    maskM = (occ_row > 0.5) & (occ_col < 0.5)     # i occ, j unocc
    invP = torch.where(maskP, 1.0 / dE, torch.zeros_like(dE))
    invM = torch.where(maskM, 1.0 / dE, torch.zeros_like(dE))

    JJp = 1j * dH_eig * invP[:, None, :, :]
    JJm = 1j * dH_eig * invM[:, None, :, :]
    return JJp, JJm


def wannier_interpolated_curvature(
    hr: HamiltonianR, AA_R: "torch.Tensor", recip_lattice: np.ndarray,
    real_lattice: np.ndarray, kpts_frac: np.ndarray, fermi_energies, *,
    delta_k: float | None = None,
    adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> np.ndarray:
    """
    Occupied-band-summed Berry curvature -2Im[f(k)] = J0 + J1 + J2 (WYSV06
    Eq. 51 / LVTS12 Eq. 6), the full quantity postw90 evaluates for both
    `kpath_task = curv` / `kslice_task = curv` and `berry_task = ahc`
    (see `anomalous_hall_conductivity`, built on top of this function).

    wannier90's `berry_get_imfgh_klist`:

        J0 = Re Tr[f . Omega_bar(k)]
        J1 = -2 Im[ Tr(A(alpha).JJp(beta)) + Tr(JJm(alpha).A(beta)) ]
        J2 = -2 Im[ Tr(JJm(alpha).JJp(beta)) ]

    for each axial (alpha, beta) = (y,z)/(z,x)/(x,y) pair, where `f` is
    the Wannier-gauge occupation projector (Sum_{n occ} |n><n|), A(k)/
    Omega_bar(k) come from `core.hamiltonian.position_operator_k`
    ("OO_true"/"OO_pseudo" Fourier transforms of AA_R), and JJp/JJm come
    from `_jjp_jjm_batch` on the same Cartesian H(k) gradient
    `berry_curvature_cartesian` uses, rotated back to the Wannier gauge.

    Returns this quantity's raw sign, not postw90's `-curv.dat` printing
    convention (which negates it).

    Args:
      hr           : Hartree, core convention
      AA_R         : (3, nR, nw, nw) complex, from `compute_position_r`,
                     on the same R_vectors/degen grid as `hr`
      recip_lattice, real_lattice: Bohr^-1/Bohr (core convention)
      kpts_frac    : (nk, 3) fractional k-points
      fermi_energies: scalar or (nf,) array, Hartree
      delta_k      : Bohr^-1, characteristic k-mesh spacing (e.g. from
                     `shift_current._kmesh_spacing`). `None` (default):
                     sharp T=0 step at each Fermi energy, exactly as
                     before. If given: YWVS07 **adaptive smearing** --
                     the Fermi-level step for each band is smoothed by a
                     per-(band,k) width set by that band's own group
                     velocity (`_adaptive_eta_per_band`/
                     `_smooth_occupation`), tempering the 1/(E_n-E_m)
                     divergence in J1/J2 for band pairs straddling E_F
                     that dominates this quantity's notoriously slow
                     uniform-mesh convergence. A prototype generalization
                     of `shift_current`'s own adaptive-smearing scheme
                     (there for a band-*pair* delta function at finite
                     frequency; here for a single band's DC Fermi-level
                     crossing) -- not yet cross-validated against real
                     wannier90/postw90's own `kubo_adpt_smr` AHC output,
                     only against the sharp-step limit and empirical
                     mesh-convergence behaviour.
      adpt_smr_fac, adpt_smr_max: as in `shift_current._adaptive_eta`,
                     only used when `delta_k` is given.

    Returns (nk, nf, 3) real, Bohr^2 (axial vector: yz, zx, xy components).
    """
    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    nf = len(fermi_energies)
    kpts = np.asarray(kpts_frac, dtype=np.float64)
    nk = len(kpts)

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.complex128)
    result = np.empty((nk, nf, 3))

    for lo in range(0, nk, _KCHUNK):
        kc = kpts[lo:lo + _KCHUNK]
        H0, grad = h_and_grad_frac_batch(hr, kc)                    # (nc,nw,nw),(nc,3,nw,nw)
        grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad)  # dH/dk_cart, Wannier gauge

        eig, UU = torch.linalg.eigh(H0)                            # (nc,nw),(nc,nw,nw)
        dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)

        # A(k) (OO_true) and Omega_bar(k) (OO_pseudo) for this chunk
        A_k, omega_bar_k = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)

        if delta_k is not None:
            v_n = torch.diagonal(dH_eig, dim1=-2, dim2=-1).real.movedim(1, 2)  # (nc,nw,3)
            eta_n = _adaptive_eta_per_band(v_n, delta_k, adpt_smr_fac, adpt_smr_max)  # (nc,nw)

        for ife, fe in enumerate(fermi_energies):
            if delta_k is None:
                occ = (eig < fe).to(UU.dtype)                      # (nc,nw)
                U_occ = UU * occ[:, None, :]                        # zero the unoccupied columns
                f_proj = U_occ @ U_occ.conj().transpose(-1, -2)    # (nc,nw,nw) Wannier gauge
                JJp_e, JJm_e = _jjp_jjm_batch(dH_eig, eig, float(fe))
            else:
                occ_real = _smooth_occupation(eig, float(fe), eta_n)          # (nc,nw) real
                occ = occ_real.to(UU.dtype)
                f_proj = torch.einsum('kin,kn,kjn->kij', UU, occ, UU.conj())   # (nc,nw,nw)
                JJp_e, JJm_e = _jjp_jjm_from_occ(dH_eig, eig, occ_real)

            JJp = torch.einsum('kin,kanm,kjm->kaij', UU, JJp_e, UU.conj())   # -> Wannier gauge
            JJm = torch.einsum('kin,kanm,kjm->kaij', UU, JJm_e, UU.conj())

            for comp, (alpha, beta) in enumerate(_AXIAL_PAIRS):
                # traces via einsum('kij,kji->k', X, Y) = Tr(X @ Y) per k
                J0 = torch.einsum('kij,kji->k', f_proj, omega_bar_k[:, comp]).real
                J1 = -2.0 * (
                    torch.einsum('kij,kji->k', A_k[:, alpha], JJp[:, beta])
                    + torch.einsum('kij,kji->k', JJm[:, alpha], A_k[:, beta])
                ).imag
                J2 = -2.0 * torch.einsum('kij,kji->k', JJm[:, alpha], JJp[:, beta]).imag
                result[lo:lo + _KCHUNK, ife, comp] = (J0 + J1 + J2).cpu().numpy()

    return result


def anomalous_hall_conductivity(
    hr: HamiltonianR, AA_R: "torch.Tensor", recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energies, mesh: tuple = (20, 20, 20), *,
    curv_adpt_kmesh: int = 1,
    curv_adpt_thresh_ang2: float = 100.0,
    kubo_adpt_smr: bool = False,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> AHCResult:
    """
    Anomalous Hall conductivity via the full WYSV06 Berry-curvature
    formula (`wannier_interpolated_curvature`, J0+J1+J2), averaged over a
    uniform k-mesh:

        sigma^{ab}_atomic = (1/Nk) sum_k sum_{n: E_n < E_F} [J0+J1+J2]_n^{ab}(k)

    reported as the axial vector (sigma_yz, sigma_zx, sigma_xy) for a scan
    of Fermi energies at once (like postw90's `fermi_energy_min/max/step`),
    in bare Bohr^2 atomic units, no cell-volume/e^2-hbar prefactor applied
    (see `AHCResult` for conversion to S/cm via `waw.units.to_si_units`).

    The 1/(E_n-E_m) divergence for band pairs straddling E_F (in
    `wannier_interpolated_curvature`'s J1/J2) makes a plain uniform-mesh
    sum highly sensitive to `mesh` being fine enough: a handful of
    near-degenerate k-points can carry curvature spikes of 1e3-1e5 Bohr^2
    and swing the whole average (measured on bcc Fe against a real
    postw90 run on IDENTICAL data: regular scan energies agree to 0.1
    S/cm while spike-dominated ones differ by 1000 S/cm between two
    equally-valid gauges of the same model). Two mitigations:

    * ``curv_adpt_kmesh > 1``: postw90's own **adaptive kmesh
      refinement** (``berry_curv_adpt_kmesh``, berry.F90): every k whose
      curvature vector exceeds ``curv_adpt_thresh_ang2`` (postw90's
      ``berry_curv_adpt_kmesh_thresh``, default 100 Ang^2) at some Fermi
      energy is re-evaluated on a centred ``n^3`` sub-mesh of its own
      mesh cell, and the spike replaced by the sub-mesh average for the
      Fermi energies that triggered. Faithful to postw90's semantics
      (per-Fermi-energy trigger and replacement).
    * ``kubo_adpt_smr=True`` (a **prototype**, see
      `wannier_interpolated_curvature`'s own caveat): YWVS07 adaptive
      smearing of the Fermi-level step itself.

    `hr` in Hartree; `AA_R` from `compute_position_r` on the same
    R-grid as `hr`; `real_lattice`/`recip_lattice` in Bohr/Bohr^-1 (core
    convention); `fermi_energies` (scalar or array) in Hartree.
    """
    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    Na, Nb, Nc = mesh

    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)

    delta_k = _kmesh_spacing(mesh, recip_lattice) if kubo_adpt_smr else None
    smr_kw = dict(delta_k=delta_k, adpt_smr_fac=kubo_adpt_smr_fac,
                  adpt_smr_max=kubo_adpt_smr_max)
    curvature_bohr2 = wannier_interpolated_curvature(
        hr, AA_R, recip_lattice, real_lattice, kpts, fermi_energies, **smr_kw,
    )   # (Nk, nf, 3)

    if curv_adpt_kmesh > 1:
        thresh_bohr2 = curv_adpt_thresh_ang2 / BOHR_TO_ANG ** 2
        trig = np.linalg.norm(curvature_bohr2, axis=2) > thresh_bohr2   # (Nk, nf)
        ktrig = np.flatnonzero(trig.any(axis=1))
        if ktrig.size:
            n = int(curv_adpt_kmesh)
            frac = (np.arange(n) + 0.5) / n - 0.5                       # centred sub-cells
            oi, oj, ok = np.meshgrid(frac / Na, frac / Nb, frac / Nc, indexing='ij')
            offsets = np.stack([oi.ravel(), oj.ravel(), ok.ravel()], axis=-1)  # (n^3, 3)
            sub_kpts = (kpts[ktrig][:, None, :] + offsets[None]).reshape(-1, 3)
            sub_curv = wannier_interpolated_curvature(
                hr, AA_R, recip_lattice, real_lattice, sub_kpts, fermi_energies, **smr_kw,
            ).reshape(len(ktrig), n ** 3, len(fermi_energies), 3).mean(axis=1)
            # replace ONLY the (k, fermi) entries that triggered -- postw90's
            # per-Fermi-energy bookkeeping (berry.F90 ladpt loop)
            mask = trig[ktrig][:, :, None]                              # (ntrig, nf, 1)
            curvature_bohr2[ktrig] = np.where(mask, sub_curv, curvature_bohr2[ktrig])

    sigma_bohr2 = curvature_bohr2.mean(axis=0)   # (nf, 3)

    return AHCResult(fermi_energies=fermi_energies, sigma=sigma_bohr2, mesh=mesh)


def _hall_conductivity_si_factor(cell_volume_bohr3: float) -> float:
    """
    `1e8 * e^2 / (hbar * V_cell[Ang^3])` (wannier90's `berry.F90`
    convention, V_cell in Angstrom^3, 1e8 converts Angstrom to cm).
    Shared by the "hall_conductivity" and "anomalous_nernst" SI factors.
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    return -1.0e8 * E_CHARGE ** 2 / (HBAR_SI * cell_volume_ang3)


@register_si_unit("hall_conductivity")
def _hall_conductivity_to_si(sigma_bohr2, *, cell_volume_bohr3: float):
    """Bohr^2 curvature average -> S/cm (Wannier90's own AHC convention)."""
    return np.asarray(sigma_bohr2) * BOHR_TO_ANG ** 2 * _hall_conductivity_si_factor(cell_volume_bohr3)


@register_from_si_unit("hall_conductivity")
def _hall_conductivity_from_si(sigma_si, *, cell_volume_bohr3: float):
    """Inverse of `_hall_conductivity_to_si` -- same `cell_volume_bohr3` kwarg."""
    return np.asarray(sigma_si) / (BOHR_TO_ANG ** 2 * _hall_conductivity_si_factor(cell_volume_bohr3))


def _nernst_mott_integral(energies: np.ndarray, curvature_bohr2: np.ndarray,
                          mu: float, kT: float) -> np.ndarray:
    """
    Generalized Mott relation, evaluated numerically for one temperature,
    in atomic units (Hartree energies, bare Bohr^2 curvature):

        alpha_ij(mu, T) = -(1/(e T)) int dE (E - mu) (-df/dE)(E;mu,T) sigma_ij(E)

    (Xiao, Yao & Niu, PRL 97, 026603 (2006), Eq. 7). ``curvature_bohr2``
    is the T=0 anomalous-Hall Berry curvature vs. Fermi level E, in bare
    Bohr^2, on the ``energies`` (Hartree) grid.

    Resolution: the kernel ``(E-mu)(-df/dE)`` has its extrema at
    ``+-2.4 kT`` and dies by ``+-6 kT`` -- at low temperature it is far
    NARROWER than any energy grid sized for the sigma(E) sweep (a 10 meV
    grid against a 2 meV-kT kernel leaves 1-3 points under each lobe, and
    the trapezoid value is then grid-alignment noise of order 100%).
    sigma(E) itself is smooth on that scale, so the fix is interpolation,
    not more (expensive) sigma evaluations: when the grid is coarser than
    ``kT/4``, sigma is cubic-spline resampled onto a fine grid before the
    convolution. Silent before 2026-07-27: the linear-in-E test case
    integrates EXACTLY on any grid, so only a curved sigma(E) at low T
    exposes it (now pinned by a quartic-Sommerfeld test).

    Returns the atomic-units integral (Hartree*Bohr^2, (…, 3-or-n)), not
    yet A/(m K); see `waw.units.to_si_units(..., "anomalous_nernst", ...)`
    for that.
    """
    energies = np.asarray(energies, dtype=np.float64)
    curvature_bohr2 = np.asarray(curvature_bohr2)
    spacing = np.diff(energies).max()
    if spacing > 0.25 * kT:
        from scipy.interpolate import CubicSpline

        n_fine = int(np.ceil((energies[-1] - energies[0]) / (kT / 8.0))) + 1
        e_fine = np.linspace(energies[0], energies[-1], n_fine)
        curvature_bohr2 = CubicSpline(energies, curvature_bohr2, axis=0)(e_fine)
        energies = e_fine
    w = minus_fermi_deriv(energies, mu, kT)                     # -df/dE, 1/Hartree, (nE,)
    kernel = (energies - mu) * w                                # (nE,), dimensionless
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # np>=2.0 renamed trapz
    return _trapz(kernel[:, None] * curvature_bohr2, energies, axis=0)


def anomalous_nernst_conductivity(
    hr: HamiltonianR, AA_R: "torch.Tensor", recip_lattice: np.ndarray,
    real_lattice: np.ndarray, *, mu: float, kT_values,
    mesh: tuple = (25, 25, 25), energies: np.ndarray | None = None,
    energy_halfwidth: float = 0.4 * EV_TO_HARTREE, n_energies: int = 81,
    curv_adpt_kmesh: int = 1, curv_adpt_thresh_ang2: float = 100.0,
) -> AnomalousNernstResult:
    """
    Intrinsic anomalous Nernst conductivity alpha^A_ij(mu, T), atomic
    units -- the transverse thermoelectric (Berry-curvature) response,
    the thermoelectric companion of the anomalous Hall conductivity.
    Convert via `waw.units.to_si_units(result.alpha, "anomalous_nernst",
    cell_volume_bohr3=...)` for A/(m K) -- see `AnomalousNernstResult`.

    Computed from the energy-resolved AHC via the generalized Mott relation
    (`_nernst_mott_integral`; Xiao-Yao-Niu 2006): the T=0 AHC
    ``sigma^A_ij(E)`` is evaluated on an energy grid around `mu` (via
    `anomalous_hall_conductivity`), then convolved with the Sommerfeld
    kernel ``(E-mu)(-df/dE)`` at each temperature. In the low-T limit this
    is the Mott formula ``alpha_xy = -(pi^2/3)(k_B^2 T/e) dsigma_xy/dE``.

    The energy grid defaults to ``mu +/- max(energy_halfwidth, 8 kT_max)``
    with `n_energies` points, wide enough to capture the ``-df/dE``
    window at the highest requested temperature. Pass `energies` to
    override.

    Args:
      hr, AA_R, recip_lattice, real_lattice : as for `anomalous_hall_conductivity`
      mu        : chemical potential (Hartree) to take the Nernst scan at (E_F)
      kT_values : scalar or array of kT (Hartree), the atomic-units
                  temperature axis; converted to Kelvin at SI-conversion
                  time via `waw.units.K_B_HARTREE`
      mesh      : k-mesh for the underlying AHC (needs to be dense)
      curv_adpt_kmesh, curv_adpt_thresh_ang2 : passed to the underlying
                  `anomalous_hall_conductivity` -- postw90-style curvature-
                  spike refinement; strongly recommended for metals

    Returns AnomalousNernstResult (alpha in atomic units, axial vector
    yz/zx/xy), also carrying the sigma^A(E) curve it used.
    """
    kT_values = np.atleast_1d(np.asarray(kT_values, dtype=np.float64))
    if energies is None:
        half = max(energy_halfwidth, 8.0 * float(kT_values.max()))
        energies = np.linspace(mu - half, mu + half, n_energies)
    energies = np.asarray(energies, dtype=np.float64)

    ahc = anomalous_hall_conductivity(hr, AA_R, recip_lattice, real_lattice,
                                      energies, mesh=mesh,
                                      curv_adpt_kmesh=curv_adpt_kmesh,
                                      curv_adpt_thresh_ang2=curv_adpt_thresh_ang2)
    sigma_of_E = ahc.sigma                                       # (nE, 3) Bohr^2

    alpha = np.stack([_nernst_mott_integral(energies, sigma_of_E, mu, float(kt))
                      for kt in kT_values], axis=0)              # (nT, 3), Hartree*Bohr^2

    return AnomalousNernstResult(kT_values=kT_values, mu=float(mu), alpha=alpha,
                                 energies=energies, sigma_of_E=sigma_of_E, mesh=mesh)


@register_si_unit("anomalous_nernst")
def _anomalous_nernst_to_si(alpha_atomic, *, cell_volume_bohr3: float, kT_values):
    """
    Atomic-units (Hartree*Bohr^2, one row per `kT_values` entry) -> A/(m K):
    `alpha[A/(m K)] = -(100/T) int dE_eV (E-mu)(-df/dE) sigma[S/cm]`, with
    T_K = kT/k_B and the same Bohr^2->S/cm factor `hall_conductivity` uses.
    The extra `HARTREE_TO_EV` factor rescales the Hartree-based integration
    variable to the eV-based one the S/cm formula is defined in.
    """
    alpha_atomic = np.asarray(alpha_atomic, dtype=np.float64)
    T_kelvin = np.asarray(kT_values, dtype=np.float64) / K_B_HARTREE
    K_scm = BOHR_TO_ANG ** 2 * _hall_conductivity_si_factor(cell_volume_bohr3)
    prefac = -100.0 * HARTREE_TO_EV / T_kelvin                     # (nT,)
    prefac = prefac.reshape(prefac.shape + (1,) * (alpha_atomic.ndim - 1))
    return prefac * K_scm * alpha_atomic


@register_from_si_unit("anomalous_nernst")
def _anomalous_nernst_from_si(alpha_si, *, cell_volume_bohr3: float, kT_values):
    """Inverse of `_anomalous_nernst_to_si` -- same `cell_volume_bohr3`/`kT_values` kwargs."""
    alpha_si = np.asarray(alpha_si, dtype=np.float64)
    T_kelvin = np.asarray(kT_values, dtype=np.float64) / K_B_HARTREE
    K_scm = BOHR_TO_ANG ** 2 * _hall_conductivity_si_factor(cell_volume_bohr3)
    prefac = -100.0 * HARTREE_TO_EV / T_kelvin                     # (nT,)
    prefac = prefac.reshape(prefac.shape + (1,) * (alpha_si.ndim - 1))
    return alpha_si / (prefac * K_scm)
