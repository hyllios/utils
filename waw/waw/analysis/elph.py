"""
Bloch-gauge electron-phonon matrix elements from raw QE DFPT output --
built natively, the same way EPW itself does (not via `ph.x`'s own
`electron_phonon='Wannier'`/`elph_mat` feature, which crashes with a
heap-corruption error in this QE version; see `interfaces.quantum_espresso
.dvscf_io`'s docstring for the full diagnosis).

  g^mu_ji(k,q) = <psi_(k+q,j)| dV_(q,mu) |psi_(k,i)>

where mu = na*3 + ipol is a Cartesian atomic-displacement index (same
combined-index convention as `analysis.phonon`) and psi_k are Bloch
states with periodic part u_nk(r) read from pw2wannier90's UNK files
(`interfaces.wannier90.io.read_unk`).

The FULL perturbation dV_(q,mu) has three pieces (QE assembles the same
three in ``PHonon/PH/dvqpsi_us.f90`` + ``solve_linter.f90``):

  1. the INDUCED self-consistent response (Hartree + xc reaction of the
     valence density) -- this, and ONLY this, is what ``ph.x`` writes to
     ``fildvscf`` (`interfaces.quantum_espresso.dvscf_io.read_dvscf`);
  2. the BARE local-pseudopotential gradient, -grad V_loc, built in
     G-space from the UPF's own radial v_loc(r) (`bare_local_dv`);
  3. the BARE nonlocal Kleinman-Bylander derivative, a separable (not
     grid-local) operator built from the UPF's beta projectors
     (`KleinmanBylanderPerturbation`).

Treating the dvscf file as the whole perturbation (this module's original
sin) inflates |g| by 1.5-2.1x for Al -- worst-case material for the
omission, since the famous local/nonlocal cancellation in simple metals
collapses the transverse-mode coupling from ~530 meV (bare-local+induced)
to the true ~19 meV only once the nonlocal term is included. With all
three terms, |g| reproduces real EPW (``prtgkk``) to 1-5%.

NLCC core-correction xc term (Al.upf has ``core_correction=T``): this is
ALREADY in the induced ``dvscf`` and must NOT be added separately. QE builds
the induced potential with ``dv_of_drho(drho, drhoc)`` in its DFPT SCF loop
(``LR_Modules/dfpt_kernels.f90``), i.e. the xc response is taken to the TOTAL
density change ``drho_val + drho_core``, and that potential is what ``ph.x``
writes to ``fildvscf``. Reconstructing the term and adding it (the machinery
exists: `nlcc_dv` / `core_charge_variation` / `analysis.xc.xc_kernel_action` /
`interfaces.quantum_espresso.io.read_charge_density`) DOUBLE-COUNTS it and
wrecks the transverse-mode cancellation (verified: g_T +197% vs EPW). The 1-5%
residual is ordinary reconstruction error (grid, KB, ph.x's own convergence),
not a missing NLCC term. Those helpers are kept only for codes/paths whose
``dvscf`` is genuinely NLCC-free.

Standard DFPT convention (Baroni, de Gironcoli, Dal Corso & Giannozzi,
Rev. Mod. Phys. 73, 515 (2001)) stores dV_q(r) as the PERIODIC part of a
Bloch-form perturbation, dV(r) = dV_q(r)*e^{iq.r} -- so in
<psi_(k+q,j)|dV(r)|psi_(k,i)>, the perturbation's own e^{iq.r} exactly
cancels the -(k+q)+k+q = 0 phase mismatch between the two Bloch states,
leaving a PLAIN periodic-cell integral of u_(k+q,j)^*(r) dV_q(r) u_(k,i)(r)
-- no extra q-phase bookkeeping needed.

Normalization: `read_unk`'s u_nk is NOT L2-normalized (integral |u_nk|^2 dV
over one cell = cell volume, not 1 -- see that module's own docstring).
The properly-normalized Bloch state is u_nk/sqrt(Omega_cell), so

  g = integral [u_kq,j^*/sqrt(Omega)] dV_q [u_k,i/sqrt(Omega)] dr
    = (1/Omega) * integral u_kq,j^* dV_q u_k,i dr
    ~ (1/Omega) * (Omega/N_r) * sum_grid u_kq,j^* dV_q u_k,i
    = (1/N_r) * sum_grid u_kq,j^* dV_q u_k,i

the cell-volume factors cancel exactly, leaving a plain grid average (no
cell volume needed at all) -- verified independently by the discretized
integral's own dimensional consistency (dV_q already has units of energy
per grid point, same convention QE itself uses for V_scf(r)).

Electron k-mesh and phonon q-mesh need NOT be the same density (`k_mesh`/
`q_mesh` are separate throughout this module) -- only that `q_mesh`
divides `k_mesh` component-wise, so every q-point coincides exactly with
a k-mesh point and k+q always lands back on the k-mesh. Re (electron
real space) and Rq (phonon real space) then use their OWN, generally
different-sized, Wigner-Seitz R-sets (`core.hamiltonian._wigner_seitz`
applied to `k_mesh` and `q_mesh` respectively).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from scipy.special import erf, erfc, spherical_jn

from ..core.distributions import (
    EPW_DEGAUSS_TO_SIGMA,
    epw_degauss_to_sigma,
    fermi_dirac,
    gaussian_smearing,
    sigma_to_epw_degauss,
    w0gauss,
    wgauss,
)
from .dos import (
    epw_fermi_level,
    fermi_level_from_electron_count,
    fermi_level_spin_channels,
    fermi_surface_dos,
)
from .eliashberg import (
    allen_dynes_tc,
    lambda_effective,
    allen_dynes_tc_from_a2f,
    eliashberg_moments,
    eliashberg_omega_2,
    lambda_from_a2f,
    lambda_matrix,
)
from ..core.hamiltonian import HamiltonianR, _wigner_seitz, operator_k
from ..units import AMU_TO_ME, CM1_TO_HARTREE

#: Modes below this frequency (Hartree; 5 cm^-1 ~ 0.62 meV) are excluded
#: from normal-mode conversion and alpha2F sums: the acoustic modes'
#: 1/sqrt(2*omega) prefactor diverges against a coupling that vanishes
#: only as O(q).
#:
#: This is a DELIBERATELY STRICTER guard than EPW's current default, which is
#: NOT 5 cm^-1: ``EPW/src/readin.f90:588`` sets ``eps_acoustic = 0.1d0`` cm^-1
#: (converted to Ry at line 1352) with the comment "the default of
#: eps_acoustic was changed from 5 cm-1 to 0.1 cm-1 (05/2025)". Reproducing a
#: stock modern EPW run therefore means passing `EPS_ACOUSTIC_EPW`, not this
#: value -- and the 50x is not cosmetic, since the window between the two is
#: exactly where the near-Gamma acoustic branch's lambda ~ |g|^2/omega^2 blows
#: up. That is why the stricter number is kept as the default here.
#:
#: This is a GUARD, NOT A CURE, and it is worth being explicit about what
#: it does not fix. alpha2F on a q-mesh INTERPOLATED beyond the ab-initio
#: one is only as good as the coarse mesh's ability to represent the
#: acoustic vertex vanishing as O(q) near Gamma. On too coarse a mesh
#: |g_acoustic| stays finite while omega -> 0, and lambda ~ |g|^2/omega^2
#: explodes: measured on MgB2 interpolated from a 4x4x2 coarse q-mesh, a
#: single near-Gamma q (-1/8, 1/8, 0) with omega_ac = 0.7 meV contributed
#: lambda = 52, and 33% of the total came from below 10 meV against 0% on
#: the ab-initio mesh itself. 0.7 meV is 5.65 cm^-1, i.e. ABOVE this
#: threshold -- it survives the guard. The real fix is a coarse q-mesh
#: dense enough for the interpolation; treat a fine-mesh lambda whose
#: weight piles up at low omega as an artifact, not a result.
#:
#: NOT an ASR projector on g itself -- INTERband g(k,q=0) is genuinely
#: nonzero (= (E_i-E_j)<j|d/dr|i>, the translation identity).
EPS_ACOUSTIC = 5.0 * CM1_TO_HARTREE

#: EPW's own present-day ``eps_acoustic`` default (0.1 cm^-1, EPW >= 05/2025).
#: Pass this as ``eps_acoustic=`` to match a stock MODERN EPW run exactly; see
#: `EPS_ACOUSTIC` for why it is not the default here. NOTE the EPW installed
#: on this cluster (v5.8.1, QE 7.3.1 module) predates the change: its
#: ``eps_acustic`` default IS 5 cm^-1, i.e. `EPS_ACOUSTIC` -- matching THAT
#: binary means NOT passing this value (verified against
#: ``EPW/src/epw_readin.f90:498`` of the qe-7.3.1 tag).
EPS_ACOUSTIC_EPW = 0.1 * CM1_TO_HARTREE

def kpoint_mesh_index(k_frac: np.ndarray, mesh: tuple[int, int, int]) -> int:
    """
    Flat index of a fractional k/q-vector (mod 1) into
    `interfaces.ase.structure.monkhorst_pack(mesh)`'s own enumeration
    order (first component slowest). Used to find k+q's index in the
    ELECTRON k-mesh (q_mesh must divide k_mesh component-wise so k+q
    always lands back on it).
    """
    n1, n2, n3 = mesh
    i = int(round(k_frac[0] * n1)) % n1
    j = int(round(k_frac[1] * n2)) % n2
    k = int(round(k_frac[2] * n3)) % n3
    return (i * n2 + j) * n3 + k


def bloch_matrix_element(u_k: np.ndarray, u_kq: np.ndarray, dv_cart: np.ndarray) -> np.ndarray:
    """
    g[mu, j, i] = <u_(k+q,j)| dv_cart[mu] |u_(k,i)>, Hartree/Bohr (dv_cart
    is dV/du_mu, a response to a physical Cartesian displacement in Bohr,
    not yet the dimensionless phonon-normal-mode coupling -- see
    `phonon_mode_coupling` for the conversion to genuine Hartree units).

    Parameters
    ----------
    u_k, u_kq : (nb, ngx, ngy, ngz) complex128
        Periodic Bloch parts at k and k+q (`interfaces.wannier90.io.
        read_unk`'s ``u_nk``), same real-space FFT grid as ``dv_cart``.
    dv_cart : (3*nat, ngx, ngy, ngz) complex128
        Cartesian-basis potential variation at this q, Hartree/Bohr
        (`interfaces.quantum_espresso.dvscf_io.read_dvscf`).

    Returns
    -------
    (3*nat, nb_kq, nb_k) complex128
    """
    n_r = np.prod(u_k.shape[1:])
    return np.einsum("jxyz,mxyz,ixyz->mji", u_kq.conj(), dv_cart, u_k, optimize=True) / n_r


def _simpson(fr: np.ndarray, rab: np.ndarray) -> np.ndarray:
    """
    QE-style Simpson quadrature over the UPF radial grid: integral f(r) dr
    = sum f(r_i) rab_i w_i / 3 with 1,4,2,...,4,1 weights (odd point count;
    the last point is dropped from an even-length grid, exactly QE's own
    ``simpson``). ``fr`` may be batched, shape (..., nr).
    """
    n = len(rab) if len(rab) % 2 == 1 else len(rab) - 1
    w = np.zeros(n)
    w[0] = w[n - 1] = 1.0
    w[1:n - 1:2] = 4.0
    w[2:n - 1:2] = 2.0
    return (fr[..., :n] * (rab[:n] * w)).sum(axis=-1) / 3.0


def _vloc_form_factor(gnorms: np.ndarray, pseudo: dict, volume: float) -> np.ndarray:
    """
    Local-pseudopotential Fourier form factor v_loc(|G|), Hartree, on an
    array of |G| > 0 (Bohr^-1) -- QE's ``vloc_of_g`` (vloc_mod.f90): the
    long-range Coulomb tail -zv/r is regularized by ADDING zv*erf(r)/r
    inside the radial transform and subtracting its analytic transform
    4*pi*zv*e^{-G^2/4}/(volume*G^2) afterwards (e^2 = 1, atomic units).

    ``pseudo``: one species' dict from `interfaces.quantum_espresso.upf.
    read_norm_conserving` (vloc already in Hartree).
    """
    r, rab, vloc, zv = pseudo["r"], pseudo["rab"], pseudo["vloc"], pseudo["zv"]
    aux = r * vloc + zv * erf(r)
    out = np.empty(len(gnorms), dtype=np.float64)
    chunk = 2048   # bounds the (chunk, nr) sin matrix
    for s in range(0, len(gnorms), chunk):
        g = gnorms[s:s + chunk]
        out[s:s + chunk] = _simpson(np.sin(np.outer(g, r)) * aux, rab) / g
    g2 = gnorms ** 2
    return (4.0 * np.pi / volume) * out \
        - (4.0 * np.pi * zv / volume) * np.exp(-g2 / 4.0) / g2


def _core_form_factor(gnorms: np.ndarray, pseudo: dict, volume: float) -> np.ndarray:
    """Fourier transform of the partial core charge, rho_core(|q+G|), on an
    array of |q+G| >= 0 (Bohr^-1) -- QE's ``interp_rhc`` (upflib/rhoc_mod.f90):

      rho_core(g) = (4 pi / Omega) integral r^2 rho_atc(r) j0(g r) dr,

    j0 = sin(x)/x (j0(0)=1). ``pseudo`` is one species' `upf.
    read_norm_conserving` dict (needs ``rho_atc``)."""
    r, rab, rho_atc = pseudo["r"], pseudo["rab"], pseudo["rho_atc"]
    aux = r ** 2 * rho_atc
    out = np.empty(len(gnorms), dtype=np.float64)
    small = gnorms < 1e-10
    out[small] = _simpson(aux, rab)                       # j0(0) = 1
    g = gnorms[~small]
    if len(g):
        chunk = 2048
        vals = np.empty(len(g))
        for s in range(0, len(g), chunk):
            gg = g[s:s + chunk]
            vals[s:s + chunk] = _simpson(np.sin(np.outer(gg, r)) * aux, rab) / gg
        out[~small] = vals
    return (4.0 * np.pi / volume) * out


def _qg_grid(q_frac, real_lattice, grid_shape, ecut_rho):
    """Shared (q+G) setup for the density/potential G-space builders: returns
    (qG_frac, qG_cart, qG2, mask, B, volume, ntot)."""
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    B = 2.0 * np.pi * np.linalg.inv(real_lattice).T
    volume = abs(np.linalg.det(real_lattice))
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in grid_shape]
    Gint = np.stack(np.meshgrid(*ints, indexing="ij"), axis=-1).reshape(-1, 3)
    qG_frac = np.asarray(q_frac, dtype=np.float64)[None, :] + Gint
    qG = qG_frac @ B
    qG2 = (qG ** 2).sum(axis=1)
    mask = qG2 <= 2.0 * ecut_rho
    ntot = int(np.prod(grid_shape))
    return qG_frac, qG, qG2, mask, B, volume, ntot


def core_charge_density(
    pseudos: list[dict], tau_frac: np.ndarray, types: np.ndarray,
    real_lattice: np.ndarray, grid_shape: tuple[int, int, int], ecut_rho: float,
) -> np.ndarray:
    """Ground-state partial core charge rho_core(r) (electrons/Bohr^3) on
    ``grid_shape`` -- the NLCC core density summed over atoms (q=0):
    ``rho_core(r) = sum_na IFFT[ rho_core_s(|G|) e^{-iG.tau_na} ]``. Add to the
    valence density to form the total density the xc kernel is evaluated at."""
    qG_frac, _, qG2, mask, _, volume, ntot = _qg_grid(
        (0.0, 0.0, 0.0), real_lattice, grid_shape, ecut_rho)
    gnorm = np.sqrt(qG2)
    ff_by_type = {}
    for s in sorted({int(t) for t in types}):
        if pseudos[s].get("rho_atc") is None:
            ff_by_type[s] = np.zeros(len(qG2))
            continue
        ff = np.zeros(len(qG2))
        ff[mask] = _core_form_factor(gnorm[mask], pseudos[s], volume)
        ff_by_type[s] = ff
    tau_frac = np.asarray(tau_frac, dtype=np.float64)
    rhog = np.zeros(len(qG2), dtype=np.complex128)
    for na in range(len(tau_frac)):
        struct = np.exp(-2j * np.pi * (qG_frac @ tau_frac[na]))
        rhog += ff_by_type[int(types[na])] * struct
    return (np.fft.ifftn(rhog.reshape(grid_shape)) * ntot).real


def core_charge_variation(
    q_frac: np.ndarray, pseudos: list[dict], tau_frac: np.ndarray,
    types: np.ndarray, real_lattice: np.ndarray,
    grid_shape: tuple[int, int, int], ecut_rho: float,
) -> np.ndarray:
    """Change of the partial core charge under each Cartesian atomic
    displacement (QE's ``addcore``), periodic part at q, ``(3*nat,)+grid``:

      drho_core[na*3+a](r) = IFFT[ -i (q+G)_a rho_core_s(|q+G|)
                                   e^{-i(q+G).tau_na} ],

    the density analogue of `bare_local_dv` (same structure/sign convention:
    displacement +e_a gives -d_a rho_core). Feeds the NLCC xc term."""
    qG_frac, qG, qG2, mask, _, volume, ntot = _qg_grid(
        q_frac, real_lattice, grid_shape, ecut_rho)
    gnorm = np.sqrt(qG2)
    ff_by_type = {}
    for s in sorted({int(t) for t in types}):
        if pseudos[s].get("rho_atc") is None:
            ff_by_type[s] = np.zeros(len(qG2))
            continue
        ff = np.zeros(len(qG2))
        ff[mask] = _core_form_factor(gnorm[mask], pseudos[s], volume)
        ff_by_type[s] = ff
    tau_frac = np.asarray(tau_frac, dtype=np.float64)
    nat = len(tau_frac)
    drhoc = np.zeros((3 * nat,) + tuple(grid_shape), dtype=np.complex128)
    for na in range(nat):
        struct = np.exp(-2j * np.pi * (qG_frac @ tau_frac[na]))
        base = ff_by_type[int(types[na])] * struct
        for a in range(3):
            cg = -1j * qG[:, a] * base
            drhoc[na * 3 + a] = np.fft.ifftn(cg.reshape(grid_shape)) * ntot
    return drhoc


def nlcc_dv(
    q_frac: np.ndarray, pseudos: list[dict], tau_frac: np.ndarray,
    types: np.ndarray, real_lattice: np.ndarray,
    grid_shape: tuple[int, int, int], ecut_rho: float,
    rho_val: np.ndarray,
) -> np.ndarray:
    """NLCC xc core-correction, ``dv_NLCC[na*3+a] = f_xc[rho_tot] .
    drho_core[na*3+a]`` (Hartree, ``(3*nat,)+grid``) -- the xc kernel
    (`analysis.xc.xc_kernel_action`, PBE) acting on the displaced core charge
    (`core_charge_variation`) at the ground-state total density.

    WARNING -- do NOT add this to a QE-derived perturbation. QE already folds
    this exact term into the induced ``dvscf`` (`dv_of_drho(drho, drhoc)` in its
    DFPT SCF loop); adding it again double-counts and destroys the transverse
    el-ph cancellation (g_T +197% vs EPW, verified). This function is provided
    only for codes/paths whose ``dvscf`` is genuinely NLCC-free. Returns 0 for
    atoms whose pseudo has no core correction.

    ``rho_val`` : (grid) real, the SCF valence density (electrons/Bohr^3), e.g.
    from `interfaces.quantum_espresso.io.read_charge_density`.
    """
    from .xc import xc_kernel_action
    rho_core = core_charge_density(pseudos, tau_frac, types, real_lattice,
                                   grid_shape, ecut_rho)
    rho_tot = np.asarray(rho_val, dtype=np.float64) + rho_core
    drhoc = core_charge_variation(q_frac, pseudos, tau_frac, types,
                                  real_lattice, grid_shape, ecut_rho)
    B = 2.0 * np.pi * np.linalg.inv(np.asarray(real_lattice, float)).T
    dv = np.zeros_like(drhoc)
    for mu in range(drhoc.shape[0]):
        if not np.any(drhoc[mu]):
            continue
        dv[mu] = xc_kernel_action(rho_tot, drhoc[mu], B, q_frac=q_frac)
    return dv


def bare_local_dv(
    q_frac: np.ndarray,
    pseudos: list[dict],
    tau_frac: np.ndarray,
    types: np.ndarray,
    real_lattice: np.ndarray,
    grid_shape: tuple[int, int, int],
    ecut_rho: float,
) -> np.ndarray:
    """
    BARE local-pseudopotential part of the phonon perturbation (periodic
    Bloch part at q), the term QE's ``dvscf`` file does NOT contain:

      dv[na*3+a](r) = sum_G -i (q+G)_a v_loc,s(na)(|q+G|)
                      e^{-i(q+G).tau_na} e^{iG.r}

    Hartree/Bohr on ``grid_shape``, same shape/convention as
    `interfaces.quantum_espresso.dvscf_io.read_dvscf` -- ADD the two.
    Validated at q=Gamma against pp.x's -grad V_scf,local (correlation
    1.00000) and, summed with the induced + nonlocal terms, against real
    EPW |g| to 1-5%.

    Parameters
    ----------
    q_frac : (3,) fractional q-vector.
    pseudos : list of per-species dicts (`upf.read_norm_conserving`).
    tau_frac : (nat, 3) fractional atomic positions.
    types : (nat,) 0-based species index per atom.
    real_lattice : (3, 3) Bohr.
    ecut_rho : Hartree -- G-sphere cutoff |q+G|^2 <= 2*ecut_rho (QE's dense
        grid; = 4*ecutwfc for norm-conserving runs. QE input ecutwfc is in
        Ry, so ecut_rho[Ha] = 2 * ecutwfc[Ry]).
    """
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    B = 2.0 * np.pi * np.linalg.inv(real_lattice).T
    volume = abs(np.linalg.det(real_lattice))
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in grid_shape]
    Gint = np.stack(np.meshgrid(*ints, indexing="ij"), axis=-1).reshape(-1, 3)
    qG_frac = np.asarray(q_frac, dtype=np.float64)[None, :] + Gint
    qG = qG_frac @ B
    qG2 = (qG ** 2).sum(axis=1)
    mask = (qG2 <= 2.0 * ecut_rho) & (qG2 > 1e-12)

    uniq, inv = np.unique(np.round(np.sqrt(qG2[mask]), 10), return_inverse=True)
    vl_by_type = {}
    for s in sorted({int(t) for t in types}):
        vl = np.zeros(len(qG2))
        vl[mask] = _vloc_form_factor(uniq, pseudos[s], volume)[inv]
        vl_by_type[s] = vl

    tau_frac = np.asarray(tau_frac, dtype=np.float64)
    nat = len(tau_frac)
    ntot = int(np.prod(grid_shape))
    dv = np.zeros((3 * nat,) + tuple(grid_shape), dtype=np.complex128)
    for na in range(nat):
        struct = np.exp(-2j * np.pi * (qG_frac @ tau_frac[na]))
        base = vl_by_type[int(types[na])] * struct
        for a in range(3):
            cg = -1j * qG[:, a] * base
            dv[na * 3 + a] = np.fft.ifftn(cg.reshape(grid_shape)) * ntot
    return dv


def _real_ylm(l: int, m: int, u: np.ndarray) -> np.ndarray:
    """Real spherical harmonics on unit vectors ``u`` (N, 3), QE ``ylmr2``
    conventions, l <= 2 (covers every projector in this project's
    norm-conserving pseudopotentials)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    if l == 0:
        return np.full(len(u), 0.5 * np.sqrt(1.0 / np.pi))
    if l == 1:
        return {0: z, 1: x, -1: y}[m] * np.sqrt(3.0 / (4.0 * np.pi))
    if l == 2:
        return {
            0: 0.25 * np.sqrt(5.0 / np.pi) * (3.0 * z * z - 1.0),
            1: np.sqrt(15.0 / (4.0 * np.pi)) * z * x,
            -1: np.sqrt(15.0 / (4.0 * np.pi)) * z * y,
            2: 0.25 * np.sqrt(15.0 / np.pi) * (x * x - y * y),
            -2: 0.5 * np.sqrt(15.0 / np.pi) * x * y,
        }[m]
    raise NotImplementedError(f"_real_ylm: l={l} > 2 not implemented")


class KleinmanBylanderPerturbation:
    """
    BARE nonlocal (Kleinman-Bylander) part of the phonon perturbation --
    a separable operator, NOT a grid-local potential, so it enters the
    Bloch matrix element directly rather than being added to dv(r):

      dV_NL/dtau_{na,a} = sum_{n,n'} D_{nn'} ( |d_a beta_n><beta_n'|
                                             + |beta_n><d_a beta_n'| )

    assembled from per-k-point projections (independent of q):

      P_c(i)   = <beta_c|psi_ki>
               = sum_G c_i(G) conj(beta_hat_c(p)) e^{ip.tau_c} / sqrt(V)
      Q_{a,c}(i) = dP_c(i)/dtau_a = i sum_G c_i(G) p_a [same] / sqrt(V)

    with p = k+G, beta_hat = 4pi (-i)^l Y_lm(p^) F_nl(|p|) and F_nl(p) =
    integral beta_n(r) r j_l(pr) dr, tabulated once on a fine |p| grid
    (QE's own ``tab_beta`` strategy) and linearly interpolated.

    Projections depend only on the k-MESH-point index: for a folded
    k+q = k_fold + G0, the physical momentum set {k+q+G} == {k_fold+G'}
    and c_{k+q}(G) = c_fold(G+G0), so P/Q computed at the folded point
    are EXACTLY those of the true k+q -- no umklapp phase needed on the
    nonlocal side (unlike the grid-local terms in
    `wannier_transform_elph`).
    """

    def __init__(
        self,
        pseudos: list[dict],
        tau_frac: np.ndarray,
        types: np.ndarray,
        real_lattice: np.ndarray,
        grid_shape: tuple[int, int, int],
        dq: float = 0.01,
    ):
        self.real_lattice = np.asarray(real_lattice, dtype=np.float64)
        self.B = 2.0 * np.pi * np.linalg.inv(self.real_lattice).T
        self.volume = abs(np.linalg.det(self.real_lattice))
        self.grid_shape = tuple(grid_shape)
        ints = [np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in grid_shape]
        self.Gint = np.stack(
            np.meshgrid(*ints, indexing="ij"), axis=-1,
        ).reshape(-1, 3).astype(np.float64)

        pmax = np.linalg.norm(self.Gint @ self.B, axis=1).max() \
            + np.linalg.norm(self.B, axis=1).sum()
        self.p_tab = np.arange(0.0, pmax + 2 * dq, dq)

        self.tau_frac = np.asarray(tau_frac, dtype=np.float64)
        self.nat = len(self.tau_frac)
        self.channels: list[tuple[int, int, int, int, int]] = []   # (na, s, n, l, m)
        self.F_tab: dict[tuple[int, int], np.ndarray] = {}
        for na, s in enumerate(types):
            ps = pseudos[int(s)]
            for n, l in enumerate(ps["ells"]):
                if (int(s), n) not in self.F_tab:
                    jl = spherical_jn(l, np.outer(self.p_tab, ps["r"]))
                    self.F_tab[(int(s), n)] = _simpson(
                        jl * (ps["betas"][n] * ps["r"]), ps["rab"],
                    )
                for m in range(-l, l + 1):
                    self.channels.append((na, int(s), n, l, m))

        nchan = len(self.channels)
        self.D = np.zeros((nchan, nchan))
        for c1, (na1, s1, n1, l1, m1) in enumerate(self.channels):
            for c2, (na2, s2, n2, l2, m2) in enumerate(self.channels):
                if na1 == na2 and l1 == l2 and m1 == m2:
                    self.D[c1, c2] = pseudos[s1]["dij"][n1, n2]
        self.chan_atom = np.array([c[0] for c in self.channels], dtype=np.int64)

    def projections(self, u_k: np.ndarray, k_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Beta projections of the periodic Bloch parts at one k-point.

        u_k : (nb, ngx, ngy, ngz), `read_unk` normalization (integral
        |u|^2 over the cell = cell volume).

        Returns P (nchan, nb), Q (3, nchan, nb).
        """
        nb = u_k.shape[0]
        ntot = int(np.prod(self.grid_shape))
        c = np.fft.fftn(u_k, axes=(-3, -2, -1)).reshape(nb, -1) / ntot
        p_frac = np.asarray(k_frac, dtype=np.float64)[None, :] + self.Gint
        p = p_frac @ self.B
        pn = np.linalg.norm(p, axis=1)
        phat = p / np.maximum(pn, 1e-12)[:, None]

        nchan = len(self.channels)
        P = np.empty((nchan, nb), dtype=np.complex128)
        Q = np.empty((3, nchan, nb), dtype=np.complex128)
        inv_sqrt_v = 1.0 / np.sqrt(self.volume)
        for ci, (na, s, n, l, m) in enumerate(self.channels):
            F = np.interp(pn, self.p_tab, self.F_tab[(s, n)])
            if l > 0:
                F = np.where(pn < 1e-12, 0.0, F)
            beta_hat = 4.0 * np.pi * ((-1j) ** l) * _real_ylm(l, m, phat) * F
            struct = np.exp(2j * np.pi * (p_frac @ self.tau_frac[na]))
            v = beta_hat.conj() * struct * inv_sqrt_v   # (N,)
            P[ci] = c @ v
            for a in range(3):
                Q[a, ci] = c @ (1j * p[:, a] * v)
        return P, Q

    def matrix_element(
        self, P_kq: np.ndarray, Q_kq: np.ndarray, P_k: np.ndarray, Q_k: np.ndarray,
    ) -> np.ndarray:
        """
        g_NL[na*3+a, j, i] = <psi_(k+q,j)| dV_NL/dtau_{na,a} |psi_(k,i)>
        from `projections` at k+q (bra) and k (ket) -- (3*nat, nb_kq, nb_k),
        Hartree/Bohr, same convention as `bloch_matrix_element` (add the two).
        """
        nb_kq, nb_k = P_kq.shape[1], P_k.shape[1]
        M = np.zeros((3 * self.nat, nb_kq, nb_k), dtype=np.complex128)
        for na in range(self.nat):
            sel = np.flatnonzero(self.chan_atom == na)
            D = self.D[np.ix_(sel, sel)]
            for a in range(3):
                M[na * 3 + a] = (Q_kq[a][sel].conj().T @ D @ P_k[sel]) \
                    + (P_kq[sel].conj().T @ D @ Q_k[a][sel])
        return M


def wannier_transform_elph(
    u_all: np.ndarray,
    W: np.ndarray,
    kpts: np.ndarray,
    qpts: np.ndarray,
    read_dvscf_q,
    k_mesh: tuple[int, int, int],
    q_mesh: tuple[int, int, int],
    real_lattice: np.ndarray,
    pseudos: list[dict] | None = None,
    tau_frac: np.ndarray | None = None,
    types: np.ndarray | None = None,
    ecut_rho: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Double real-space Wannier transform of the electron-phonon coupling:

      g(Re, Rq)[mu, n, m] = (1/Nk) (1/Nq) sum_{k,q} e^{-2pi i k.Re}
          e^{-2pi i q.Rq} [W(k+q)^dagger @ g_bloch(k,q)[mu] @ W(k)][n, m]

    mirroring `core.hamiltonian.compute_hr`'s own W(k)^dagger H(k) W(k)
    Wigner-Seitz Fourier transform, done twice (electron k -> Re, phonon
    q -> Rq). The phonon side is left in the Cartesian atomic-displacement
    index `mu` (no separate "phonon gauge" rotation) -- exactly the same
    combined-index convention `analysis.phonon.phonon_hamiltonian` already
    uses. Re and Rq use SEPARATE Wigner-Seitz R-sets, from `k_mesh` and
    `q_mesh` respectively (same crystal lattice; `q_mesh` need not equal
    `k_mesh`, only divide it component-wise).

    Umklapp: when k+q folds outside [0,1) back to a mesh point k_fold with
    a nonzero reciprocal vector G0 (k+q = k_fold + G0), the true periodic
    part is u_(k+q) = e^{-iG0.r} u_(k_fold) -- the stored u alone is the
    WRONG integrand for the grid-local dv product (a 10-50% pointwise |g|
    error). The phase is applied here on the fly; the nonlocal term needs
    none (see `KleinmanBylanderPerturbation`).

    When ``pseudos`` (+ ``tau_frac``/``types``/``ecut_rho``) is given, the
    FULL perturbation is assembled: induced (``read_dvscf_q``) + bare
    local (`bare_local_dv`, added to dv per q) + nonlocal KB
    (`KleinmanBylanderPerturbation`, added to the Bloch matrix element per
    (k, q)). Without it, the coupling is induced-only -- 1.5-2.1x too
    large for Al; see the module docstring.

    Parameters
    ----------
    u_all : (nk, nb, ngx, ngy, ngz) complex128
        Periodic Bloch parts at every k on the mesh, mesh order matching
        ``kpts`` (`interfaces.wannier90.io.read_unk`'s ``u_nk``, stacked).
    W : (nk, nb, nw) complex128
        Full converged electron gauge (V@U_final for entangled bands,
        U_final alone for isolated bands -- `core.hamiltonian.
        compute_operator_r`'s own convention).
    kpts, qpts : (nk, 3), (nq, 3) float64
        Fractional k-points (electron mesh) and q-points (phonon mesh).
    read_dvscf_q : callable(int) -> (3*nat, ngx, ngy, ngz) complex128
        Returns the Cartesian-basis potential variation at q-mesh index
        ``iq`` (0-based, ``qpts[iq]``) -- e.g. a closure around
        `interfaces.quantum_espresso.dvscf_io.read_dvscf`.
    k_mesh, q_mesh : (N1, N2, N3)
        Electron and phonon mesh dimensions -- `q_mesh` must divide
        `k_mesh` component-wise (e.g. k_mesh=(12,12,12), q_mesh=(6,6,6)),
        so every q-point coincides exactly with a k-mesh point.
    real_lattice : (3, 3) float64
        Crystal lattice vectors in Bohr.
    pseudos, tau_frac, types, ecut_rho : optional, all-or-none
        Per-species UPF data (`interfaces.quantum_espresso.upf.
        read_norm_conserving`), fractional positions (nat, 3), 0-based
        species index (nat,), dense-grid cutoff in Hartree (see
        `bare_local_dv`) -- enables the full bare+induced perturbation.
    (The NLCC xc core-correction is NOT added: QE already includes it in the
    induced ``dvscf`` read via ``read_dvscf_q`` -- see the module docstring and
    `nlcc_dv`.)

    Returns
    -------
    g_R : (nR_e, nR_q, 3*nat, nw, nw) complex128, indexed [Re, Rq, mu, n, m]
    R_e : (nR_e, 3) int64
    degen_e : (nR_e,) int64
    R_q : (nR_q, 3) int64
    degen_q : (nR_q,) int64
        Wigner-Seitz degeneracies (NOT yet divided out -- apply at
        interpolation time, matching `core.hamiltonian.operator_k`'s own
        convention).
    """
    full_perturbation = pseudos is not None
    if full_perturbation and (tau_frac is None or types is None or ecut_rho is None):
        raise ValueError(
            "wannier_transform_elph: pseudos requires tau_frac, types and "
            "ecut_rho too (all-or-none)."
        )

    R_e, degen_e = _wigner_seitz(k_mesh, real_lattice)
    R_q, degen_q = _wigner_seitz(q_mesh, real_lattice)
    nR_e, nR_q = len(R_e), len(R_q)
    nk, nq = len(kpts), len(qpts)
    nw = W.shape[-1]
    grid_shape = u_all.shape[2:]
    n_r = int(np.prod(grid_shape))

    phase_k = np.exp(-2j * np.pi * (np.asarray(kpts) @ R_e.T))   # (nk, nR_e)
    phase_q = np.exp(-2j * np.pi * (np.asarray(qpts) @ R_q.T))   # (nq, nR_q)

    kb = P_all = Q_all = None
    if full_perturbation:
        kb = KleinmanBylanderPerturbation(
            pseudos, tau_frac, types, real_lattice, grid_shape,
        )
        proj = [kb.projections(u_all[ik], kpts[ik]) for ik in range(nk)]
        P_all = np.stack([p for p, _ in proj])   # (nk, nchan, nb)
        Q_all = np.stack([q for _, q in proj])   # (nk, 3, nchan, nb)

    # per-axis grid coordinates for the umklapp phase e^{-2pi i G.r}
    grid_ix = np.meshgrid(
        *[np.arange(n) / n for n in grid_shape], indexing="ij",
    )

    # q-independent: build the flat torch view of u_all ONCE, not per q
    # (torch.from_numpy shares the buffer; the reshape is a view because
    # read_unk/np.stack give a C-contiguous (nk, nb, ngx, ngy, ngz) array)
    nb_all = u_all.shape[1]
    u_all_t = torch.from_numpy(u_all.reshape(nk, nb_all, n_r))

    nat3 = None
    g_R = None
    for iq in range(nq):
        dv = read_dvscf_q(iq)   # (3*nat, ngx, ngy, ngz) -- induced part
        if nat3 is None:
            nat3 = dv.shape[0]
            g_R = np.zeros((nR_e, nR_q, nat3, nw, nw), dtype=np.complex128)
        if full_perturbation:
            dv = dv + bare_local_dv(
                qpts[iq], pseudos, tau_frac, types, real_lattice,
                grid_shape, ecut_rho,
            )
            # NOTE: the NLCC xc core-correction is deliberately NOT added here.
            # QE already folds it into the induced ``dvscf`` this reads (its
            # SCF loop builds the induced potential with ``dv_of_drho(drho,
            # drhoc)`` -- LR_Modules/dfpt_kernels.f90 -- so f_xc.drho_core is in
            # the written fildvscf). Adding `nlcc_dv` on top double-counts it
            # and wrecks the delicate transverse-mode cancellation (verified:
            # g_T +197% vs EPW). See `nlcc_dv`'s docstring.

        kq_idx = np.empty(nk, dtype=np.int64)
        Gvecs = np.empty((nk, 3), dtype=np.int64)
        for ik in range(nk):
            kq = kpts[ik] + qpts[iq]
            kq_idx[ik] = kpoint_mesh_index(kq, k_mesh)
            Gvecs[ik] = np.round(kq - kpts[kq_idx[ik]]).astype(np.int64)

        # g_bloch[k,m,j,i] = (1/n_r) sum_r u_kq*[k,j,r] dv[m,r] u_k[k,i,r].
        #
        # This is the pipeline's hot spot, and it is MEMORY-bound, not
        # flop-bound: for a real mesh the arithmetic is ~40 GFLOP per q against
        # ~10 GB of array traffic. Three things therefore matter far more than
        # the GEMM itself:
        #   * torch, not numpy, for the elementwise work -- numpy runs gathers,
        #     conj and broadcast multiplies SINGLE-threaded whatever BLAS is
        #     set to, while torch's respect `torch.set_num_threads`, so the
        #     bandwidth-bound passes actually use the cores;
        #   * no materialised u_all[kq_idx]: the k+q gather happens per chunk,
        #     never as a full (nk, nb, n_r) reordered copy;
        #   * no materialised conjugate: torch's `.conj()` is a lazy view that
        #     `matmul` consumes directly, so the dv weight is folded into the
        #     (already transposed) u_k factor instead.
        nb_ = nb_all
        dv_t = torch.from_numpy(np.ascontiguousarray(dv.reshape(nat3, n_r)))
        kq_t = torch.from_numpy(kq_idx)

        # umklapp phases: one per distinct nonzero G, plus a per-k index into them
        uniq_G = np.unique(Gvecs[np.any(Gvecs != 0, axis=1)], axis=0)
        phase_of_k = np.full(nk, -1, dtype=np.int64)
        phases = []
        for gi, G in enumerate(uniq_G):
            phase_of_k[(Gvecs == G).all(axis=1)] = gi
            phases.append(torch.from_numpy(np.exp(-2j * np.pi * (
                G[0] * grid_ix[0] + G[1] * grid_ix[1] + G[2] * grid_ix[2]
            )).reshape(n_r)))

        g_t = torch.empty((nk, nat3, nb_, nb_), dtype=torch.complex128)
        # Chunk for CACHE, not for a memory ceiling. The per-mode
        # `u_k * dv[m]` temporary is (chunk, n_r, nb) and is written then
        # immediately consumed by the GEMM, so the whole cost is whether it
        # survives in cache: measured on Al (nb=12, n_r=13824) a 500 MB
        # temporary costs 8.7 s/q while a ~16 MB one costs 2.7 s/q -- a 3.2x
        # difference from chunk size alone, with identical arithmetic. Target
        # ~16 MB rather than a fixed count, since the optimum chunk differs
        # per system (Al likes 4-8, MgB2 with its larger grid likes 1-2).
        k_chunk = max(1, int(16.0e6 // max(1, nb_ * n_r * 16)))
        for s in range(0, nk, k_chunk):
            e = min(s + k_chunk, nk)
            ukq = u_all_t[kq_t[s:e]]                       # (blk, nb_j, n_r) gather
            pk = phase_of_k[s:e]
            for gi in np.unique(pk[pk >= 0]):
                sel = torch.from_numpy(np.flatnonzero(pk == gi))
                ukq[sel] *= phases[gi]                     # in-place, threaded
            ukq_c = ukq.conj()                             # lazy view, no copy
            uk = u_all_t[s:e].transpose(1, 2)              # (blk, n_r, nb_i)
            for m in range(nat3):
                torch.matmul(ukq_c, uk * dv_t[m][:, None], out=g_t[s:e, m])
        g_bloch = g_t.numpy()
        g_bloch /= n_r   # (nk, 3*nat, nb_kq, nb_k)

        if full_perturbation:
            P_kq, Q_kq = P_all[kq_idx], Q_all[kq_idx]
            for na in range(kb.nat):
                sel = np.flatnonzero(kb.chan_atom == na)
                D = kb.D[np.ix_(sel, sel)]
                g_bloch[:, na * 3:na * 3 + 3] += np.einsum(
                    "kacj,cd,kdi->kaji", Q_kq[:, :, sel].conj(), D, P_all[:, sel],
                    optimize=True,
                ) + np.einsum(
                    "kcj,cd,kadi->kaji", P_kq[:, sel].conj(), D, Q_all[:, :, sel],
                    optimize=True,
                )

        W_kq = W[kq_idx]   # (nk, nb, nw)
        g_wan = np.einsum(
            "kjn,kuji,kim->kunm", W_kq.conj(), g_bloch, W, optimize=True,
        )   # (nk, 3*nat, nw, nw)

        g_semi = np.einsum("kR,kunm->Runm", phase_k, g_wan, optimize=True) / nk   # (nR_e, 3*nat, nw, nw)
        g_R += np.einsum("Q,Runm->RQunm", phase_q[iq], g_semi, optimize=True) / nq

    return g_R, R_e, degen_e, R_q, degen_q


def interpolate_elph_kq(
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    kpts: np.ndarray, qpts: np.ndarray,
) -> np.ndarray:
    """
    Fourier-interpolate the doubly-real-space coupling back to a list of
    (k, q) pairs, Wannier gauge:

      g(k,q)[mu,n,m] = sum_{Re,Rq} e^{2pi i k.Re}/D(Re) e^{2pi i q.Rq}/D(Rq)
                       g_R[Re,Rq,mu,n,m]

    -- the electron-phonon analogue of `core.hamiltonian.operator_k`,
    applied independently to each of the two R-indices (Re/Rq use their
    OWN Wigner-Seitz sets, from the electron k-mesh and phonon q-mesh
    respectively). Evaluated at the SAME mesh points `wannier_transform_
    elph` was built from, this is an EXACT round-trip (the whole point
    of a discrete Fourier transform pair) -- the strongest available
    correctness check before any real dense-mesh interpolation is
    attempted.

    Parameters
    ----------
    kpts, qpts : (n, 3) float64, matched pairs (kpts[i], qpts[i])

    Returns
    -------
    (n, 3*nat, nw, nw) complex128
    """
    kpts = np.atleast_2d(np.asarray(kpts, dtype=np.float64))
    qpts = np.atleast_2d(np.asarray(qpts, dtype=np.float64))
    inv_degen_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    inv_degen_q = 1.0 / np.asarray(degen_q, dtype=np.float64)

    phase_k = np.exp(2j * np.pi * (kpts @ R_e.T)) * inv_degen_e[None, :]   # (npairs, nR_e)
    phase_q = np.exp(2j * np.pi * (qpts @ R_q.T)) * inv_degen_q[None, :]   # (npairs, nR_q)

    return np.einsum("bP,bQ,PQumn->bumn", phase_k, phase_q, g_R, optimize=True)


def interpolate_elph_fixed_q(
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    kpts: np.ndarray, q_frac: np.ndarray,
) -> np.ndarray:
    """`interpolate_elph_kq` for the common case of ONE q shared by every k --
    the access pattern of every alpha2F sum (a q-loop over all k).

    Mathematically identical to
    ``interpolate_elph_kq(..., kpts, np.tile(q_frac, (len(kpts), 1)))`` but
    vastly cheaper. The general routine takes a per-pair ``(b, Q)`` phase and
    can only evaluate ``sum_{P,Q} e^{ikR_P} e^{iqR_Q} g_R`` per pair, costing
    ``nk * nR_e * nR_q * nat3 * nw^2``; with q fixed the Rq sum does not depend
    on k, so contracting it ONCE and then over Re costs
    ``nR_e*nR_q*nat3*nw^2 + nk*nR_e*nat3*nw^2`` -- a factor ~``nR_q`` less work
    (100x+ on a real mesh). Both contractions go through ``tensordot``, i.e.
    BLAS GEMM, so they are also multithreaded rather than a serial einsum loop.

    Returns ``(nk, 3*nat, nw, nw)`` complex128, Wannier gauge.
    """
    kpts = np.atleast_2d(np.asarray(kpts, dtype=np.float64))
    q_frac = np.asarray(q_frac, dtype=np.float64).reshape(3)
    inv_degen_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    inv_degen_q = 1.0 / np.asarray(degen_q, dtype=np.float64)

    ph_q = np.exp(2j * np.pi * (R_q @ q_frac)) * inv_degen_q          # (nR_q,)
    h = np.tensordot(ph_q, g_R, axes=([0], [1]))                      # (nR_e, u, n, m)
    ph_k = np.exp(2j * np.pi * (kpts @ R_e.T)) * inv_degen_e[None, :]  # (nk, nR_e)
    return np.tensordot(ph_k, h, axes=([1], [0]))                     # (nk, u, n, m)



def _fixed_q_h_provider(g_R, R_q, degen_q, qpts, max_bytes: float = 2.0e9):
    """Blocked R_q contraction for the alpha2F q-loops.

    ``interpolate_elph_fixed_q``'s first contraction sweeps the ENTIRE
    multi-GB ``g_R`` array once per q -- memory-bound, and measured at 92%
    of the whole alpha2f hot loop on MgB2-sized data (1.8 s of a 1.9 s
    per-q iteration). Contracting a BLOCK of q at once against a
    once-reordered ``(nR_q, nR_e*modes*nw*nw)`` copy turns that into one
    BLAS GEMM and one g_R sweep per ~hundred q: ~7x on the full alpha2f.
    Identical mathematics, reassociated contraction order.

    Returns ``h_of(iq) -> (nR_e, 3*nat, nw, nw)``, the per-q intermediate
    `interpolate_elph_fixed_q` calls ``h``.
    """
    g_R = np.asarray(g_R)
    nR_e, nR_q = g_R.shape[0], g_R.shape[1]
    rest = g_R.shape[2:]
    gR_flat = np.ascontiguousarray(g_R.transpose(1, 0, 2, 3, 4)).reshape(nR_q, -1)
    inv_deg = 1.0 / np.asarray(degen_q, dtype=np.float64)
    R_q = np.asarray(R_q, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    qblk = max(1, int(max_bytes // (gR_flat.shape[1] * 16)))
    cache: dict = {}

    def h_of(iq: int) -> np.ndarray:
        b = (iq // qblk) * qblk
        if b not in cache:
            cache.clear()
            ph = np.exp(2j * np.pi * (qpts[b:b + qblk] @ R_q.T)) * inv_deg[None, :]
            cache[b] = (ph @ gR_flat).reshape(-1, nR_e, *rest)
        return cache[b][iq - b]

    return h_of


def phonon_mode_coupling(
    g_wannier_kq: np.ndarray,
    eigvec_q: np.ndarray,
    omega_q_hartree: np.ndarray,
    masses_amu: np.ndarray,
    types: np.ndarray,
    eps_acoustic: float = EPS_ACOUSTIC,
) -> np.ndarray:
    """
    Convert the Cartesian-atomic-displacement-basis coupling to the
    genuine (Hartree-valued) phonon-normal-mode coupling:

      g^nu_nm(k,q) = (1/sqrt(2 omega_q,nu)) sum_mu
                     [eigvec_q,nu[mu] / sqrt(M_mu)] * g_wannier_kq[mu,n,m]

    Derivation (atomic units, hbar=1): the ionic Cartesian displacement
    from phonon mode (q,nu) is u_mu(q) = sqrt(1/(2 M_mu omega_qnu)) *
    eigvec_q,nu[mu] (standard harmonic-oscillator normal-mode
    quantization; `eigvec_q,nu` is `analysis.phonon.interpolate_phonons`'s
    own mass-weighted-dynamical-matrix eigenvector, orthonormal, mu = na*3
    + ipol). The coupling Hamiltonian dV = sum_mu (dV/du_mu) u_mu then
    gives the formula above directly. Dimensional check: [1/sqrt(M*omega)]
    is a LENGTH in atomic units (from E_h = hbar^2/(m_e a_0^2) => a_0 =
    1/sqrt(m_e*E_h)), so (Hartree/Bohr) * Bohr = Hartree -- the result
    genuinely comes out in Hartree, unlike `bloch_matrix_element`'s raw
    Hartree/Bohr.

    Parameters
    ----------
    g_wannier_kq : (3*nat, nw, nw) complex128, Hartree/Bohr
        `interpolate_elph_kq`'s output at one (k, q) pair (or
        `wannier_transform_elph`'s per-k,q slice), Wannier gauge.
    eigvec_q : (3*nat, 3*nat) complex128
        `analysis.phonon.interpolate_phonons`'s ``eigvecs[iq]`` at this q
        -- columns are modes, ``eigvec_q[mu, nu]``.
    omega_q_hartree : (3*nat,) float64
        Phonon angular frequencies at this q, in Hartree (NOT
        `analysis.phonon`'s own mixed-unit "omega2_au" -- convert via
        ``units.cm1_to_hartree(freq_cm1)``).
    masses_amu : (ntyp,) float64
        Physical atomic masses in amu (`interfaces.quantum_espresso.
        phonon_io.read_force_constants`'s own ``masses_amu``).
    types : (nat,) int
        0-based type index of each atom, same order as the mu = na*3+ipol
        combined index (`interfaces.quantum_espresso.phonon_io`'s own
        ``types``).
    eps_acoustic : Hartree
        Modes with omega < eps_acoustic (the q=0 acoustic translations)
        get an exactly-zero coupling instead of a divergent
        1/sqrt(2*omega) against a near-zero frequency -- EPW's own
        ``eps_acoustic`` point-of-use convention (see `EPS_ACOUSTIC`).

    Returns
    -------
    (3*nat, nw, nw) complex128, Hartree
    """
    omega = np.asarray(omega_q_hartree, dtype=np.float64)
    alive = omega > eps_acoustic
    mass_mu_me = np.repeat(masses_amu[types], 3) * AMU_TO_ME   # (3*nat,) atomic units
    prefactor = eigvec_q / np.sqrt(mass_mu_me)[:, None]        # (mu, nu)
    prefactor = prefactor / np.sqrt(2.0 * np.where(alive, omega, 1.0))[None, :]
    prefactor = prefactor * alive[None, :]
    return np.einsum("uv,unm->vnm", prefactor, g_wannier_kq, optimize=True)


def band_eigensystem(hr: HamiltonianR, kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize H(k) (Wannier gauge) at a list of k-points.

    Returns
    -------
    eig : (nk, nw) float64, Hartree
    U   : (nk, nw, nw) complex128, columns = band eigenvectors (rotate
          Wannier gauge -> band gauge, same convention as `core.hamiltonian
          .compute_hr`'s own U)
    """
    H_k = operator_k(hr.H_R, hr.R_vectors, hr.degen, np.asarray(kpts))
    H_k = 0.5 * (H_k + H_k.conj().transpose(-1, -2))
    eig, U = torch.linalg.eigh(H_k)
    return eig.detach().cpu().numpy(), U.detach().cpu().numpy()


def alpha2f(
    eig: np.ndarray,
    U: np.ndarray,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    kpts: np.ndarray, qpts: np.ndarray, k_mesh: tuple[int, int, int],
    omega_ph: np.ndarray, eigvec_ph: np.ndarray,
    masses_amu: np.ndarray, types: np.ndarray,
    fermi_energy: float, dos_at_ef: float | None,
    omega_grid: np.ndarray,
    sigma_e: float = 0.01, sigma_ph: float | None = None,
    velocities: np.ndarray | None = None,
    eps_acoustic: float = EPS_ACOUSTIC,
    hr: HamiltonianR | None = None,
    recip_lattice: np.ndarray | None = None,
    delta_approx: bool = True,
    temperature: float = 0.0,
    fsthick: float | None = None,
    q_weights: np.ndarray | None = None,
    return_qnu: bool = False,
) -> np.ndarray:
    """
    Isotropic Eliashberg spectral function alpha2F(omega) (Allen, Phys.
    Rev. B 6, 2577 (1972), Eq. 2 combined with the phonon linewidth
    formula gamma_qnu = 2*pi*omega_qnu*sum_(k,mn)|g^nu_mn(k,q)|^2
    delta(eps_km-eF) delta(eps_(k+q)n-eF) / Nk, giving

      alpha2F(w) = (1/N(eF)) (1/Nk) (1/Nq) sum_(k,q,mn,nu)
                   |g^nu_mn(k,q)|^2 delta(eps_km-eF) delta(eps_(k+q)n-eF)
                   delta(w - w_qnu)

    Spin: this project's electron structure is non-spin-polarized/no-SOC,
    so both the coupling sum and `dos_at_ef` (from `analysis.dos.
    density_of_states`, itself per-spin-channel -- see that function's
    own docstring) implicitly represent ONE spin channel; the physical
    (both-spin) factor of 2 appears identically in the numerator (real
    scattering sums over both spins) and in N(eF) (true DOS counts both
    spins), so it cancels exactly -- no explicit spin-degeneracy factor
    is needed here.

    If `velocities` is given (diagonal group velocities v_n(k), Hartree*
    Bohr, e.g. `analysis.wigner_transport.velocity_matrix`'s diagonal),
    each (k,q,m,n) term is additionally weighted by the transport factor

      1 - (v_km . v_(k+q)n) / |v_km|^2

    (Grimvall Eq. 8.20; Allen, Phys. Rev. B 17, 3725 (1978)), giving the
    transport spectral function alpha2F_tr(omega) instead of the plain
    superconducting one. Note the denominator is |v_k|^2, NOT
    |v_km||v_(k+q)n| -- the factor is a velocity-RELAXATION weight, not a
    cosine, and EPW says so in as many words (``selfen.f90``: "coskkq =
    (vk dot vkq) / |vk|^2 appears in Grimvall 8.20; this is different from:
    coskkq = (vk dot vkq) / |vk||vkq|"). The two agree only on a Fermi
    surface where |v| happens to be constant; the normalized-cosine version
    this function used previously is a different quantity.

    Two forms of the Fermi-surface restriction are available, mirroring
    EPW's ``delta_approx`` switch (see that parameter).

    Parameters
    ----------
    eig, U : from `band_eigensystem(hr, kpts)` -- band eigenvalues/gauge
        at the SAME `kpts` (electron mesh) `g_R` was built from.
    g_R, R_e, degen_e, R_q, degen_q : from `wannier_transform_elph`.
    kpts, qpts : (nk, 3), (nq, 3) float64, fractional -- electron k-mesh
        and phonon q-mesh (`q_mesh` dividing `k_mesh`, see
        `wannier_transform_elph`).
    k_mesh : electron mesh dimensions -- used ONLY in the coarse-mesh mode
        (``hr is None``) for the k+q mesh-index lookup ``eig[kq_idx]``,
        which requires every ``k+q`` to coincide with a ``kpts`` mesh point
        (so ``qpts`` must lie on the ``kpts`` mesh). Ignored when ``hr`` is
        given (see below). May be ``None`` in that case.
    omega_ph : (nq, 3*nat) float64, Hartree, phonon frequencies at `qpts`
        (`units.cm1_to_hartree` of `analysis.phonon.interpolate_phonons`'s
        ``freq_cm1``, evaluated at `qpts`).
    eigvec_ph : (nq, 3*nat, 3*nat) complex128, phonon eigenvectors at
        `qpts` (`interpolate_phonons`'s ``eigvecs``).
    masses_amu, types : as `phonon_mode_coupling`.
    fermi_energy : Hartree.
    dos_at_ef : states/Hartree at `fermi_energy` (per spin channel, see
        Spin note above), or ``None`` -- STRONGLY preferred -- to have it
        computed internally by `fermi_surface_dos(eig, fermi_energy,
        sigma_e)`, i.e. on this very k-mesh with this very broadening.
        alpha2F is a ratio of two delta-function sums and is only
        smearing-independent when both share a mesh and a width, which is
        what EPW does (it recomputes ``dosef`` from the fine-mesh ``etf``
        with ``degaussw0`` inside every smearing loop). Passing a
        separately-converged N(eF) -- denser mesh, or different sigma --
        rescales lambda by exactly the mismatch.
    omega_grid : (nE,) float64, Hartree, output frequency grid. Must span the
        phonon spectrum: any mode above ``omega_grid[-1]`` contributes
        nothing and would silently lower lambda, so that case warns.
    sigma_e : Hartree, electronic delta-function Gaussian broadening. This is
        the sqrt(2)-narrower statistician's sigma, NOT EPW's ``degaussw`` --
        convert with `epw_degauss_to_sigma`.
    sigma_ph : Hartree, phonon delta-function Gaussian broadening
        (defaults to twice the `omega_grid` spacing).
    delta_approx : bool
        Which Fermi-surface restriction to use. Named after EPW's input flag
        of the same name, but read the asymmetry below before flipping it --
        the DEFAULT HERE (``True``) is the right one for alpha2F, and it is
        what EPW itself uses wherever alpha2F feeds superconductivity.

        * ``True``: the double delta ``delta(eps_km - eF)
          delta(eps_(k+q)n - eF)``, i.e. Allen's formula as written in this
          docstring's header. This IS the definition of the Eliashberg
          function: the Eliashberg equations pin both electrons to the Fermi
          surface, and Allen's ``lambda_qnu = gamma_qnu / (pi N(eF)
          omega_qnu^2)`` -- the relation any gamma-based route must use to
          get lambda -- is derived in exactly that omega_qnu -> 0 limit.
          EPW's own superconductivity module agrees: ``supercond.f90
          ::evaluate_a2f_lambda`` builds ``weight = wkfs * wqf * w0g(ibnd,
          ik) * w0g(jbnd, k+q)`` with ``w0g = w0gauss((e - ef0)/degaussw,
          0)/degaussw``, a plain double delta, and normalises by ``dosef``.
        * ``False``: the exact Migdal linewidth ``[f(eps_km) -
          f(eps_(k+q)n)] delta(eps_(k+q)n - eps_km - omega_qnu)``. This is
          the correct expression for the PHONON LINEWIDTH gamma_qnu -- a
          different, directly measurable observable (neutron/Raman) -- and
          it is what ``selfen.f90::selfen_phon_q`` computes by default.
          Feeding that gamma through Allen's omega -> 0 relation, as
          ``spectral.f90::a2f_main`` then does, mixes a finite-omega
          linewidth into a quantity whose definition assumes omega -> 0.
          Per-mode contributions can come out negative here and are clamped
          to zero, exactly as EPW's own "sanity check" does.

        So: use the default for alpha2F, lambda, omega_log and Tc. Use
        ``False`` only to reproduce EPW's ``phonselfen``/``a2f=.true.``
        printout specifically, or when the phonon linewidth itself is what
        you want.

        The choice is not academic, and the occupation-difference form is
        also far harder to converge: it needs (k, k+q) pairs that straddle
        eF AND are separated by ~omega_qnu, which a coarse mesh essentially
        never supplies. Measured with EPW itself on Al (identical vertex,
        only the flag toggled): a 12^3 fine mesh gives lambda = 0.484 from
        the double delta but 0.0003 from the occupation-difference form.
        EPW's published 0.368 for the same system comes from the latter at
        20^3 -- three orders of magnitude above its own 12^3 value, i.e.
        still far from converged, while the double delta is already stable.
    temperature : Hartree (k_B T), used only when ``delta_approx=False`` for
        the occupation factors; ``0.0`` is the exact step function
        (EPW's ``temps`` is typically a fraction of a kelvin).
    fsthick : Hartree or None
        Fermi-surface window -- EPW's ``fsthick``. Skip every k (and every
        k+q) whose bands all lie further than this from ``fermi_energy``.
        Those terms are suppressed by ``exp(-(fsthick/sigma_e)^2)``, so at
        ``fsthick >= 9 * sigma_e`` the restriction is exact to double
        precision while cutting the (k,q) double sum by the ratio of the
        window to the bandwidth. This is the difference between an N^6 sum
        and a feasible one, and the reason EPW reaches 40^3 fine meshes:
        for Al at ``sigma_e = 35 meV`` a 10-sigma window keeps ~4% of the
        mesh, a ~25x saving with no measurable change in alpha2F.

        ``None`` (default) keeps every k-point. If you set it, make it
        comfortably larger than both ``9 * sigma_e`` and (for
        ``delta_approx=False``, where the delta is on ``eps_(k+q) - eps_k -
        omega``) the maximum phonon energy -- too small a window silently
        truncates the sum, which is why this is opt-in rather than on by
        default.
    q_weights : (nq,) float64 or None
        Per-q weights summing to 1, replacing the uniform ``1/nq``. Pass the
        output of `interfaces.ase.structure.irreducible_qpoints` together
        with its ``q_irr`` to sum over the irreducible wedge instead of the
        full q-mesh -- exact, because the q-summand is invariant under the
        crystal point group once the k-sum covers the complete mesh (see
        that function for the argument, and note it requires the k-mesh to
        stay FULL). Up to 48x for fcc; this is EPW's ``mp_mesh_k`` saving and
        it is what makes a dense fine mesh affordable.
        ``None`` means the uniform ``1/nq`` of a full mesh.
    velocities : (nk, nw, 3) float64 or None, diagonal group velocities
        at `kpts` in the SAME band gauge as `U` -- enables the
        transport-weighted variant.
    eps_acoustic : Hartree
        Modes with omega < eps_acoustic (the q=0 acoustic translations)
        are excluded from the sum entirely -- EPW's own point-of-use
        convention (see `EPS_ACOUSTIC`), replacing the earlier (wrong)
        ASR projector on g_R, which also zeroed the genuinely-nonzero
        INTERband g(k, q=0).
    hr : HamiltonianR or None
        When given, ``eig``/``U`` at ``k+q`` are obtained by DIRECTLY
        Wannier-interpolating H(k+q) (`band_eigensystem`) instead of the
        coarse ``eig[kq_idx]`` mesh lookup -- EPW's fine-mesh
        (``nkf``/``nqf``) trick for the double-delta Fermi-surface sums.
        This lifts BOTH restrictions of the coarse mode: ``kpts`` and
        ``qpts`` may be arbitrary, independent, DENSE meshes (``qpts`` need
        not lie on the ``kpts`` mesh, and ``k_mesh`` is unused). ``eig``/
        ``U`` passed in are still the k-side eigensystem at ``kpts`` (i.e.
        ``band_eigensystem(hr, kpts)``), ``g_R`` is the mesh-independent
        real-space coupling from `wannier_transform_elph`, and ``omega_ph``/
        ``eigvec_ph`` are the phonons interpolated at the (fine) ``qpts``.
        The result converges to the true Brillouin-zone double integral as
        the fine meshes are densified (see `lambda_richardson`); the coarse
        mode evaluates the same sum on the sparse ab-initio mesh and is far
        from converged for the notoriously slow lambda.
    recip_lattice : (3, 3) float64 or None
        Reciprocal lattice (2*pi/Bohr), required ONLY for the
        transport-weighted variant in fine-mesh mode (``hr`` and
        ``velocities`` both given): the group velocities at ``k+q`` are
        recomputed by `analysis.wigner_transport.velocity_matrix` at the
        interpolated ``k+q`` rather than a mesh lookup.
    return_qnu : bool
        Also return the mode-resolved coupling ``lambda_qnu`` (nq, n_modes)
        -- EPW's per-q ``lambda___( nu )`` stdout numbers,
        ``lambda_qnu = 2 s_qnu / (omega_qnu N(eF))``, UNCLAMPED (EPW also
        prints the raw value and clamps only inside ``a2f_main``; the
        occupation-difference form can genuinely go negative). EPW's
        headline ``lambda :`` is then ``sum_q q_weights * clip(lambda_qnu,
        0) . sum(axis=modes)``, with no phonon smearing and no omega grid
        involved -- the sharpest single number for cross-code parity.
        Caveat for per-mode comparison: within a DEGENERATE phonon
        multiplet the split between modes is eigenvector-gauge arbitrary
        (EPW additionally averages over the multiplet); only multiplet
        SUMS are comparable.

    Returns
    -------
    (nE,) float64  -- or ``(a2f, lambda_qnu)`` when ``return_qnu``.
    """
    if hr is not None and velocities is not None and recip_lattice is None:
        raise ValueError(
            "alpha2f: fine-mesh transport variant (hr + velocities) needs "
            "recip_lattice for the velocity interpolation at k+q."
        )
    nk, nq = len(kpts), len(qpts)
    n_modes = omega_ph.shape[-1]
    if sigma_ph is None:
        sigma_ph = 2.0 * (omega_grid[1] - omega_grid[0])
    if q_weights is None:
        q_weights = np.full(nq, 1.0 / nq)
    else:
        q_weights = np.asarray(q_weights, dtype=np.float64)
        if q_weights.shape != (nq,):
            raise ValueError(
                f"alpha2f: q_weights has shape {q_weights.shape}, expected ({nq},)."
            )
        if not np.isclose(q_weights.sum(), 1.0, rtol=0, atol=1e-8):
            raise ValueError(
                f"alpha2f: q_weights sum to {q_weights.sum():.10g}, not 1 -- "
                "they must be normalised star multiplicities."
            )
    if dos_at_ef is None:
        dos_at_ef = fermi_surface_dos(eig, fermi_energy, sigma_e)
    # A mesh too coarse for the broadening can leave NO state inside the
    # Fermi window at all, and then alpha2F is 0/0. Returning zeros (or
    # silent NaNs downstream in omega_log) would read as "lambda is small"
    # when the truth is "this mesh cannot resolve the Fermi surface".
    if not dos_at_ef > 0.0:
        raise ValueError(
            f"alpha2f: N(eF) = {dos_at_ef:g} -- no state within the smearing "
            f"of fermi_energy on this k-mesh ({nk} points, sigma_e = "
            f"{sigma_e:g} Ha). Densify the mesh or widen sigma_e; alpha2F is "
            "0/0 here, not small."
        )

    _warn_grid_truncation(omega_ph, omega_grid, eps_acoustic)

    mass_mu_me = np.repeat(masses_amu[types], 3) * AMU_TO_ME

    use_interp = hr is not None
    kpts = np.asarray(kpts, dtype=np.float64)

    # Fermi-window restriction (EPW's fsthick). The k-side factor is a
    # Gaussian in (eps_kn - eF), so a k whose every band sits further than
    # fsthick from eF contributes a number of order exp(-(fsthick/sigma)^2)
    # -- below double precision already at ~9 sigma. Dropping those k is
    # therefore exact to roundoff, and it is what makes a dense fine mesh
    # affordable at all: for a simple metal only a thin shell of k-points
    # carries any weight (Al, 4 bands over 12 eV, fsthick = 10 sigma_e:
    # ~4% of the mesh survives, a ~25x saving in the (k,q) double sum,
    # which is the whole reason EPW can reach 40^3 where a brute-force
    # N^6 sum cannot). ``None`` keeps every k -- the safe default.
    if fsthick is None:
        k_sel = np.arange(nk)
    else:
        k_sel = np.flatnonzero(np.min(np.abs(eig - fermi_energy), axis=1) < fsthick)
        if k_sel.size == 0:
            raise ValueError(
                f"alpha2f: fsthick={fsthick:g} Ha excludes every k-point "
                "(no band within the window of fermi_energy)."
            )
    eig_w, U_w, kpts_w = eig[k_sel], U[k_sel], kpts[k_sel]

    a2f = np.zeros_like(omega_grid, dtype=np.float64)
    lam_qnu = np.zeros((nq, n_modes), dtype=np.float64)
    _h_of = _fixed_q_h_provider(g_R, R_q, degen_q, qpts)
    _inv_deg_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    _R_e_T = np.asarray(R_e, dtype=np.float64).T
    for iq in range(nq):
        if use_interp:
            # EPW's fine-mesh trick: interpolate H(k+q) directly (k, q may be
            # arbitrary, independent, dense meshes -- no mesh-index lookup).
            kq_pts = kpts_w + qpts[iq]
            eig_kq, U_kq = band_eigensystem(hr, kq_pts)
        else:
            kq_idx = np.array([kpoint_mesh_index(kpts_w[i] + qpts[iq], k_mesh)
                               for i in range(len(k_sel))])
            eig_kq, U_kq = eig[kq_idx], U[kq_idx]

        # ... and the k+q side carries its own Gaussian, so the same window
        # applies there. Applying it per q (not once) is what keeps the
        # surviving set small: only pairs with BOTH ends near eF matter.
        if fsthick is not None:
            keep = np.min(np.abs(eig_kq - fermi_energy), axis=1) < fsthick
            if not keep.any():
                continue
            eig_kq, U_kq = eig_kq[keep], U_kq[keep]
            if use_interp:
                kq_pts = kq_pts[keep]
            else:
                kq_idx = kq_idx[keep]
            eig_q, U_q, kpts_q = eig_w[keep], U_w[keep], kpts_w[keep]
        else:
            eig_q, U_q, kpts_q = eig_w, U_w, kpts_w

        # fixed-q fast path, q-blocked: the Rq contraction comes from
        # `_fixed_q_h_provider` (one g_R sweep per q-block), the Re side is
        # a plain BLAS GEMM -- identical result to interpolate_elph_fixed_q.
        ph_k = np.exp(2j * np.pi * (kpts_q @ _R_e_T)) * _inv_deg_e[None, :]
        g_kq = np.tensordot(ph_k, _h_of(iq), axes=([1], [0]))
        #   (nk_win, 3*nat, nw, nw), Wannier gauge, Hartree/Bohr

        alive = omega_ph[iq] > eps_acoustic   # (n_modes,)
        prefactor = eigvec_ph[iq] / np.sqrt(mass_mu_me)[:, None]
        prefactor = prefactor / np.sqrt(
            2.0 * np.where(alive, omega_ph[iq], 1.0),
        )[None, :]   # (mu, nu)
        prefactor = prefactor * alive[None, :]
        g_mode = np.einsum("uv,kunm->kvnm", prefactor, g_kq, optimize=True)   # (nk, n_modes, nw, nw)

        g_band = np.einsum(
            "kjn,kvjm,kmo->kvno", U_kq.conj(), g_mode, U_q, optimize=True,
        )   # (nk_win, n_modes, nw_kq_band, nw_k_band)

        # g_band axes are (n = k+q band, o = k band): pair each band's own
        # delta -- delta_kq on n, delta_k on o (matches the velocity pairing
        # below). The isotropic sum is invariant under this vs the swapped
        # pairing by full-BZ k<->k+q symmetry, but the band-resolved
        # alpha2f_matrix and the transport term need the correct assignment.
        if delta_approx:
            delta_kq = gaussian_smearing(eig_kq - fermi_energy, sigma_e)   # (nk, nw)
            delta_k = gaussian_smearing(eig_q - fermi_energy, sigma_e)     # (nk, nw)
            weight = delta_kq[:, :, None] * delta_k[:, None, :]            # (nk, nw_kq, nw_k)
        else:
            # EPW's default: [f(eps_k) - f(eps_k+q)] delta(eps_k+q - eps_k -
            # omega_qnu), so the weight is per-MODE. The 1/omega_qnu below
            # replaces the double delta's implicit one (see the delta_approx
            # note in this function's docstring for the derivation).
            f_k = fermi_dirac(eig_q - fermi_energy, 0.0, temperature)     # (nk, nw)
            f_kq = fermi_dirac(eig_kq - fermi_energy, 0.0, temperature)   # (nk, nw)
            docc = f_k[:, None, :] - f_kq[:, :, None]                      # (nk, nw_kq, nw_k)
            de = eig_kq[:, :, None] - eig_q[:, None, :]                    # (nk, nw_kq, nw_k)
            inv_w = np.where(alive, 1.0 / np.where(alive, omega_ph[iq], 1.0), 0.0)
            weight = (
                docc[:, None, :, :]
                * gaussian_smearing(de[:, None, :, :] - omega_ph[iq][None, :, None, None], sigma_e)
                * inv_w[None, :, None, None]
            )                                                              # (nk, n_modes, nw_kq, nw_k)

        if velocities is not None:
            if use_interp:
                from .wigner_transport import velocity_matrix
                _, vmat_kq = velocity_matrix(hr, kq_pts, recip_lattice)
                v_kq = np.real(np.einsum("knna->kna", vmat_kq))   # (nk, nw, 3)
            else:
                v_kq = velocities[kq_idx]    # (nk, nw, 3)
            v_k = velocities[k_sel][keep] if fsthick is not None else velocities
            # Grimvall 8.20 / EPW selfen.f90: (v_k . v_k+q) / |v_k|^2, with
            # |v_k|^2 -- NOT |v_k||v_k+q| -- in the denominator, and the whole
            # factor set to zero (so 1 - cos -> 1) where |v_k|^2 underflows.
            vdot = np.einsum("kna,kma->knm", v_kq, v_k)          # (nk, nw_kq, nw_k)
            vsq_k = np.einsum("kma,kma->km", v_k, v_k)            # (nk, nw_k)
            denom = np.broadcast_to(vsq_k[:, None, :], vdot.shape)
            cos_kkq = np.divide(
                vdot, denom, out=np.zeros_like(vdot), where=np.abs(denom) > 1e-4,
            )
            tr = 1.0 - cos_kkq
            weight = weight * (tr if delta_approx else tr[:, None, :, :])

        g2 = np.abs(g_band) ** 2                                  # (nk, n_modes, nw_kq, nw_k)
        if delta_approx:
            s_nu = np.einsum("kvno,kno->v", g2, weight, optimize=True) / nk
        else:
            s_nu = np.einsum("kvno,kvno->v", g2, weight, optimize=True) / nk
        lam_qnu[iq] = np.where(
            alive, 2.0 * s_nu / (np.where(alive, omega_ph[iq], 1.0) * dos_at_ef), 0.0)
        # EPW clamps a negative per-mode lambda to zero ("sanity check",
        # spectral.f90). Vacuous for the positive-definite double delta;
        # the occupation-difference form genuinely can go negative.
        s_nu = np.maximum(s_nu, 0.0)

        for nu in range(n_modes):
            a2f += (q_weights[iq] * s_nu[nu]
                    * gaussian_smearing(omega_grid - omega_ph[iq, nu], sigma_ph))

    a2f = a2f / dos_at_ef
    if return_qnu:
        return a2f, lam_qnu
    return a2f


def _warn_grid_truncation(omega_ph, omega_grid, eps_acoustic):
    """alpha2F weight at a frequency outside `omega_grid` is simply lost, and
    lambda drops with no other symptom. Say so rather than let it pass."""
    live = np.asarray(omega_ph)[np.asarray(omega_ph) > eps_acoustic]
    if live.size and live.max() > omega_grid[-1]:
        import warnings
        warnings.warn(
            f"alpha2f: omega_grid stops at {omega_grid[-1]:.6g} Ha but the "
            f"phonon spectrum reaches {live.max():.6g} Ha -- the modes above "
            "the grid contribute nothing and lambda is underestimated. EPW "
            "sizes its own grid as 1.1 * max(omega).",
            RuntimeWarning, stacklevel=3,
        )


def alpha2f_matrix(
    eig: np.ndarray, U: np.ndarray,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    kpts: np.ndarray, qpts: np.ndarray, k_mesh: tuple[int, int, int] | None,
    omega_ph: np.ndarray, eigvec_ph: np.ndarray,
    masses_amu: np.ndarray, types: np.ndarray,
    fermi_energy: float, orbital_groups: list[list[int]],
    omega_grid: np.ndarray,
    sigma_e: float = 0.01, sigma_ph: float | None = None,
    eps_acoustic: float = EPS_ACOUSTIC,
    hr: HamiltonianR | None = None,
    fsthick: float | None = None,
    q_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-resolved (multiband) Eliashberg spectral function -- an ``(n_sheets,
    n_sheets, nE)`` matrix ``alpha2F_ij(omega)``. This is the two-band picture
    of MgB2 (sigma/pi) or the multi-sheet picture of Pb: partition the Fermi
    surface into sheets ``i`` and resolve which sheet the initial (k) and final
    (k+q) electron sit on,

      alpha2F_ij(w) = (1/N_i) (1/Nk Nq) sum_{k,q,mn,nu}
                      w_i(m,k) w_j(n,k+q) |g^nu_mn(k,q)|^2
                      delta(eps_km - eF) delta(eps_(k+q)n - eF) delta(w - w_qnu),

    with ``w_i(n,k)`` the soft Wannier-character sheet weights
    (`analysis.band_sheets.band_character_weights`) and ``N_i`` the per-sheet
    Fermi-level DOS. The row normalisation is by ``N_i`` (the k-side sheet), the
    Golubov-Mazin convention, so ``N_i lambda_ij = N_j lambda_ji`` and the
    DOS-weighted sum recovers the isotropic result exactly:

      alpha2F(w) = sum_ij N_i alpha2F_ij(w) / sum_i N_i          (built-in check)

    and ``lambda = sum_ij (N_i / N) lambda_ij`` with ``lambda_ij = 2 integral
    alpha2F_ij / w`` (`lambda_from_a2f` per element).

    Args as `alpha2f` (same fine-mesh ``hr`` interpolation and coarse-mesh
    fallback), plus ``orbital_groups`` (Wannier-orbital indices per sheet). The
    soft weights need no band tracking -- a band crossing E_F multiple times,
    or an index swap at a crossing, is handled by construction; see
    `analysis.band_sheets`. Returns ``(alpha2F_ij, N_i)``.

    ``fsthick`` and ``q_weights`` work exactly as in `alpha2f` (Fermi window
    and irreducible-wedge star weights); the wedge reduction stays exact for
    the sheet-resolved sums as long as ``orbital_groups`` is itself invariant
    under the point group (sigma/pi in MgB2 -- in-plane vs p_z -- is).

    The built-in check above is an identity only against an `alpha2f` whose
    own N(eF) is the SAME number: call it with ``dos_at_ef=None`` (which makes
    it `fermi_surface_dos` on this mesh at this ``sigma_e``, i.e. ``N_i.sum()``
    exactly) or with ``dos_at_ef=float(N_i.sum())``. An independently
    converged N(eF) rescales the isotropic curve and the identity fails by
    that ratio.
    """
    from .band_sheets import band_character_weights, sheet_dos

    nk, nq = len(kpts), len(qpts)
    n_modes = omega_ph.shape[-1]
    n_sheets = len(orbital_groups)
    if sigma_ph is None:
        sigma_ph = 2.0 * (omega_grid[1] - omega_grid[0])
    _warn_grid_truncation(omega_ph, omega_grid, eps_acoustic)
    mass_mu_me = np.repeat(masses_amu[types], 3) * AMU_TO_ME
    kpts = np.asarray(kpts, dtype=np.float64)
    if q_weights is None:
        q_weights = np.full(nq, 1.0 / nq)
    else:
        q_weights = np.asarray(q_weights, dtype=np.float64)
        if q_weights.shape != (nq,) or not np.isclose(q_weights.sum(), 1.0, rtol=0, atol=1e-8):
            raise ValueError("alpha2f_matrix: q_weights must be (nq,) normalised star weights.")

    w_k = band_character_weights(U, orbital_groups)        # (nk, n_sheets, nw)
    N_i = sheet_dos(eig, fermi_energy, w_k, sigma_e)        # (n_sheets,)
    use_interp = hr is not None

    # Fermi window (EPW's fsthick), exactly as `alpha2f`: k whose bands all
    # sit further than fsthick from eF contribute below double precision.
    if fsthick is None:
        k_sel = np.arange(nk)
    else:
        k_sel = np.flatnonzero(np.min(np.abs(eig - fermi_energy), axis=1) < fsthick)
        if k_sel.size == 0:
            raise ValueError("alpha2f_matrix: fsthick excludes every k-point.")
    eig_w, U_w, kpts_w, w_k_w = eig[k_sel], U[k_sel], kpts[k_sel], w_k[k_sel]

    a2f = np.zeros((n_sheets, n_sheets, len(omega_grid)), dtype=np.float64)
    _h_of = _fixed_q_h_provider(g_R, R_q, degen_q, qpts)
    _inv_deg_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    _R_e_T = np.asarray(R_e, dtype=np.float64).T
    for iq in range(nq):
        if use_interp:
            kq_pts = kpts_w + qpts[iq]
            eig_kq, U_kq = band_eigensystem(hr, kq_pts)
        else:
            kq_idx = np.array([kpoint_mesh_index(kpts_w[ik] + qpts[iq], k_mesh)
                               for ik in range(len(k_sel))])
            eig_kq, U_kq = eig[kq_idx], U[kq_idx]

        if fsthick is not None:
            keep = np.min(np.abs(eig_kq - fermi_energy), axis=1) < fsthick
            if not keep.any():
                continue
            eig_kq, U_kq = eig_kq[keep], U_kq[keep]
            eig_q, U_q, kpts_q, w_kk = eig_w[keep], U_w[keep], kpts_w[keep], w_k_w[keep]
        else:
            eig_q, U_q, kpts_q, w_kk = eig_w, U_w, kpts_w, w_k_w
        w_kq = band_character_weights(U_kq, orbital_groups)   # (nkw, n_sheets, nw)

        ph_k = np.exp(2j * np.pi * (kpts_q @ _R_e_T)) * _inv_deg_e[None, :]
        g_kq = np.tensordot(ph_k, _h_of(iq), axes=([1], [0]))
        alive = omega_ph[iq] > eps_acoustic
        pref = eigvec_ph[iq] / np.sqrt(mass_mu_me)[:, None]
        pref = pref / np.sqrt(2.0 * np.where(alive, omega_ph[iq], 1.0))[None, :]
        pref = pref * alive[None, :]
        g_mode = np.einsum("uv,kunm->kvnm", pref, g_kq, optimize=True)
        g_band = np.einsum("kjn,kvjm,kmo->kvno", U_kq.conj(), g_mode, U_q, optimize=True)
        g2 = np.abs(g_band) ** 2                              # (nkw, n_modes, nw_kq, nw_k)

        delta_kq = gaussian_smearing(eig_kq - fermi_energy, sigma_e)   # (nkw, nw)
        delta_k = gaussian_smearing(eig_q - fermi_energy, sigma_e)     # (nkw, nw)
        wq = w_kq * delta_kq[:, None, :]     # (nkw, n_sheets, nw_kq)  j = k+q side
        wk = w_kk * delta_k[:, None, :]      # (nkw, n_sheets, nw_k)   i = k side
        # s[nu, i, j] = sum_{k,n,o} g2[k,nu,n,o] wk[k,i,o] wq[k,j,n]
        s = np.einsum("kvno,kio,kjn->vij", g2, wk, wq, optimize=True) / nk
        for nu in range(n_modes):
            a2f += q_weights[iq] * s[nu][:, :, None] * gaussian_smearing(
                omega_grid - omega_ph[iq, nu], sigma_ph)[None, None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        a2f = np.where(N_i[:, None, None] > 0, a2f / N_i[:, None, None], 0.0)
    return a2f, N_i


class TransportSpectralMatrix(NamedTuple):
    """
    Velocity-weighted, sheet-resolved el-ph spectral functions for the
    band-resolved Boltzmann equation (`analysis.elph_boltzmann`).

    ``f_out`` and ``f_in`` are the two weightings that the familiar
    ``alpha2F_tr`` bundles into one number, kept SEPARATE because they enter
    the multiband collision matrix differently (diagonal vs off-diagonal --
    see `analysis.elph_boltzmann.lova_conductivity`). Both are absolute
    per-cell sums, NOT normalised by a DOS: the LOVA matrix needs their
    magnitude, and dividing by N_i (as `alpha2f_matrix` does) would destroy
    the Drude bookkeeping.
    """
    omega: np.ndarray      # (nE,) Hartree, the frequency grid
    f_out: np.ndarray      # (3, n_sheets, n_sheets, nE), weight v_a(k)^2
    f_in:  np.ndarray      # (3, n_sheets, n_sheets, nE), weight v_a(k) v_a(k+q)
    drude: np.ndarray      # (3, n_sheets) Hartree*Bohr^2, N_i <v_a^2>_i per cell
    dos:   np.ndarray      # (n_sheets,) per-sheet N_i(eF), for reference


def alpha2f_transport_matrix(
    eig: np.ndarray, U: np.ndarray,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    kpts: np.ndarray, qpts: np.ndarray, k_mesh: tuple[int, int, int] | None,
    omega_ph: np.ndarray, eigvec_ph: np.ndarray,
    masses_amu: np.ndarray, types: np.ndarray,
    fermi_energy: float, orbital_groups: list[list[int]],
    omega_grid: np.ndarray, velocities: np.ndarray,
    sigma_e: float = 0.01, sigma_ph: float | None = None,
    eps_acoustic: float = EPS_ACOUSTIC,
    hr: HamiltonianR | None = None,
    recip_lattice: np.ndarray | None = None,
    fsthick: float | None = None,
    q_weights: np.ndarray | None = None,
) -> TransportSpectralMatrix:
    """
    Sheet-resolved el-ph spectral functions carrying the VELOCITY weights the
    Boltzmann equation needs, for each Cartesian direction a:

      F^out_a,ij(w) = (1/Nk Nq) sum_{k,q,mn,nu} w_i(m,k) w_j(n,k+q) |g|^2
                      v_a(m,k)^2      delta(eps_km-eF) delta(eps_(k+q)n-eF)
                                      delta(w - w_qnu)
      F^in_a,ij(w)  = same, with      v_a(m,k) v_a(n,k+q)

    plus the per-sheet Drude weights D_a,i = (1/Nk) sum_km w_i(m,k)
    v_a(m,k)^2 delta(eps_km - eF) = N_i <v_a^2>_i.

    WHY TWO OF THEM, and why not just a per-pair alpha2F_tr. With one sheet,
    Allen's transport spectral function is exactly

      alpha2F_tr(w) = [F^out(w) - F^in(w)] / D                          (*)

    (Allen, Phys. Rev. B 17, 3725 (1978); Grimvall Ch. 7) -- the difference is
    all that survives because out- and in-scattering enter the single-band
    collision term with opposite sign. In BAND space that cancellation is not
    local: expanding the variational collision term with the trial function
    Phi_k = Phi_i v_a(k) gives

      M_ij = delta_ij sum_l A_il - C_ij,

    i.e. the out-weight appears on the DIAGONAL summed over all final sheets l,
    while the in-weight sits off-diagonal. A pre-subtracted alpha2F_tr per pair
    has already thrown that structure away, and with it the physics: interband
    scattering that preserves the current direction (v_a(k) v_a(k+q) > 0)
    relaxes the current less than its out-scattering rate alone implies.

    NOTE THE CONVENTION DIFFERENCE FROM `alpha2f(velocities=...)`. That
    function applies the DIMENSIONLESS factor 1 - (v_k.v_k+q)/|v_k|^2 and
    normalises by N(eF), which is EPW's transport-lambda convention; every k on
    the Fermi surface then carries equal weight. Eq. (*) instead weights each k
    by v_a(k)^2, which is what the Drude/variational algebra produces. The two
    agree only where |v| is constant over the Fermi surface. Neither is wrong;
    they are different averages, and only the second closes the Boltzmann
    equation. Use this function for transport, that one for comparing lambda_tr
    with EPW.

    Args as `alpha2f_matrix`, plus:
      velocities : (nk, nw, 3) diagonal group velocities at ``kpts``, Hartree*
        Bohr (`analysis.wigner_transport.velocity_matrix`'s diagonal, or
        `analysis.boltzmann.band_velocities`).
      recip_lattice : required in fine-mesh mode (``hr`` given), where the
        velocities at k+q are recomputed by `velocity_matrix` at the
        interpolated points rather than looked up on the mesh.

    Returns a `TransportSpectralMatrix`. Feed it to
    `analysis.elph_boltzmann.lova_conductivity`.

    Only the DIAGONAL Cartesian components are formed, so this gives
    sigma_xx/yy/zz. Off-diagonal sigma_ab would need the v_a v_b cross weights
    (same algebra, three more arrays); it is not implemented.
    """
    from .band_sheets import band_character_weights, sheet_dos, sheet_drude_weight

    nk, nq = len(kpts), len(qpts)
    n_modes = omega_ph.shape[-1]
    n_sheets = len(orbital_groups)
    if sigma_ph is None:
        sigma_ph = 2.0 * (omega_grid[1] - omega_grid[0])
    _warn_grid_truncation(omega_ph, omega_grid, eps_acoustic)
    use_interp = hr is not None
    if use_interp and recip_lattice is None:
        raise ValueError(
            "alpha2f_transport_matrix: fine-mesh mode (hr given) needs "
            "recip_lattice to recompute the velocities at k+q.")
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.shape != (nk, eig.shape[1], 3):
        raise ValueError(
            f"alpha2f_transport_matrix: velocities must be (nk, nw, 3) = "
            f"{(nk, eig.shape[1], 3)}; got {velocities.shape}")
    mass_mu_me = np.repeat(masses_amu[types], 3) * AMU_TO_ME
    kpts = np.asarray(kpts, dtype=np.float64)
    if q_weights is None:
        q_weights = np.full(nq, 1.0 / nq)
    else:
        q_weights = np.asarray(q_weights, dtype=np.float64)
        if q_weights.shape != (nq,) or not np.isclose(q_weights.sum(), 1.0, rtol=0, atol=1e-8):
            raise ValueError(
                "alpha2f_transport_matrix: q_weights must be (nq,) normalised.")

    w_k = band_character_weights(U, orbital_groups)          # (nk, n_sheets, nw)
    N_i = sheet_dos(eig, fermi_energy, w_k, sigma_e)          # (n_sheets,)

    # Drude weights on the FULL mesh -- these are one-Fermi-surface sums and
    # need no q, so the fsthick window below must not touch them.
    drude = sheet_drude_weight(eig, velocities, fermi_energy, w_k, sigma_e)

    if fsthick is None:
        k_sel = np.arange(nk)
    else:
        k_sel = np.flatnonzero(np.min(np.abs(eig - fermi_energy), axis=1) < fsthick)
        if k_sel.size == 0:
            raise ValueError("alpha2f_transport_matrix: fsthick excludes every k.")
    eig_w, U_w, kpts_w, w_k_w = eig[k_sel], U[k_sel], kpts[k_sel], w_k[k_sel]
    v_k_w = velocities[k_sel]

    f_out = np.zeros((3, n_sheets, n_sheets, len(omega_grid)))
    f_in = np.zeros((3, n_sheets, n_sheets, len(omega_grid)))
    _h_of = _fixed_q_h_provider(g_R, R_q, degen_q, qpts)
    _inv_deg_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    _R_e_T = np.asarray(R_e, dtype=np.float64).T
    for iq in range(nq):
        if use_interp:
            kq_pts = kpts_w + qpts[iq]
            eig_kq, U_kq = band_eigensystem(hr, kq_pts)
        else:
            kq_idx = np.array([kpoint_mesh_index(kpts_w[ik] + qpts[iq], k_mesh)
                               for ik in range(len(k_sel))])
            eig_kq, U_kq = eig[kq_idx], U[kq_idx]

        if fsthick is not None:
            keep = np.min(np.abs(eig_kq - fermi_energy), axis=1) < fsthick
            if not keep.any():
                continue
            eig_kq, U_kq = eig_kq[keep], U_kq[keep]
            eig_q, U_q, kpts_q = eig_w[keep], U_w[keep], kpts_w[keep]
            w_kk, v_k = w_k_w[keep], v_k_w[keep]
            if use_interp:
                kq_pts = kq_pts[keep]
            else:
                kq_idx = kq_idx[keep]
        else:
            eig_q, U_q, kpts_q = eig_w, U_w, kpts_w
            w_kk, v_k = w_k_w, v_k_w
        w_kq = band_character_weights(U_kq, orbital_groups)

        if use_interp:
            from .wigner_transport import velocity_matrix
            _, vmat_kq = velocity_matrix(hr, kq_pts, recip_lattice)
            v_kq = np.real(np.einsum("knna->kna", vmat_kq))       # (nkw, nw, 3)
        else:
            v_kq = velocities[kq_idx]

        ph_k = np.exp(2j * np.pi * (kpts_q @ _R_e_T)) * _inv_deg_e[None, :]
        g_kq = np.tensordot(ph_k, _h_of(iq), axes=([1], [0]))
        alive = omega_ph[iq] > eps_acoustic
        pref = eigvec_ph[iq] / np.sqrt(mass_mu_me)[:, None]
        pref = pref / np.sqrt(2.0 * np.where(alive, omega_ph[iq], 1.0))[None, :]
        pref = pref * alive[None, :]
        g_mode = np.einsum("uv,kunm->kvnm", pref, g_kq, optimize=True)
        g_band = np.einsum("kjn,kvjm,kmo->kvno", U_kq.conj(), g_mode, U_q, optimize=True)
        g2 = np.abs(g_band) ** 2                          # (nkw, n_modes, nw_kq, nw_k)

        delta_kq = gaussian_smearing(eig_kq - fermi_energy, sigma_e)
        delta_k = gaussian_smearing(eig_q - fermi_energy, sigma_e)
        wq = w_kq * delta_kq[:, None, :]     # (nkw, n_sheets, nw_kq)  j = k+q side
        wk = w_kk * delta_k[:, None, :]      # (nkw, n_sheets, nw_k)   i = k side

        lorentz = np.stack([gaussian_smearing(omega_grid - omega_ph[iq, nu], sigma_ph)
                            for nu in range(n_modes)])                     # (n_modes, nE)
        for a in range(3):
            # out: weight v_a(k)^2, a function of the k-side band index only
            g2w = g2 * (v_k[:, :, a] ** 2)[:, None, None, :]
            s_out = np.einsum("kvno,kio,kjn->vij", g2w, wk, wq, optimize=True) / nk
            # in: weight v_a(k) v_a(k+q), coupling the two band indices
            vprod = v_kq[:, :, a][:, :, None] * v_k[:, :, a][:, None, :]   # (nkw, nw_kq, nw_k)
            g2w = g2 * vprod[:, None, :, :]
            s_in = np.einsum("kvno,kio,kjn->vij", g2w, wk, wq, optimize=True) / nk
            f_out[a] += q_weights[iq] * np.einsum("vij,vw->ijw", s_out, lorentz,
                                                  optimize=True)
            f_in[a] += q_weights[iq] * np.einsum("vij,vw->ijw", s_in, lorentz,
                                                 optimize=True)
    return TransportSpectralMatrix(omega=np.asarray(omega_grid, dtype=np.float64),
                                   f_out=f_out, f_in=f_in, drude=drude, dos=N_i)


class LambdaConvergence(NamedTuple):
    """Result of `lambda_richardson`.

    ``lambda_extrapolated`` is built as ``coupling_extrapolated *
    dos_converged``, NOT by extrapolating ``lambdas`` directly -- see that
    function's docstring for why the direct fit is ill-conditioned.
    """
    lambda_extrapolated: float
    n_linear: np.ndarray        # (nmesh,) effective linear mesh dimension N = (N1 N2 N3)^(1/3)
    lambdas: np.ndarray         # (nmesh,) lambda(N) as computed on each mesh
    sigma_e: np.ndarray         # (nmesh,) electronic smearing actually used at each mesh
    a2f: list                   # per-mesh alpha2F(omega_grid)
    #: (nmesh,) N(eF) actually used to normalise alpha2F on each mesh
    dos_at_ef: np.ndarray = None
    #: (nmesh,) lambda(N) / N(eF)(N) = 2 <<|g|^2/omega>>, the DOS-independent
    #: double-Fermi-surface average -- this is the well-conditioned sequence
    coupling: np.ndarray = None
    #: 1/N -> 0 intercept of `coupling`
    coupling_extrapolated: float = float("nan")
    #: N(eF) converged on its own (cheap, H(k)-only) dense mesh sequence
    dos_converged: float = float("nan")
    #: (n_dos_mesh,) linear sizes and N(eF) values of that sequence
    dos_n_linear: np.ndarray = None
    dos_values: np.ndarray = None


def lambda_richardson(
    hr: HamiltonianR,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    phonon_fn,
    masses_amu: np.ndarray, types: np.ndarray,
    fermi_energy: float, dos_at_ef: float | None,
    omega_grid: np.ndarray,
    meshes: list[tuple[int, int, int]],
    sigma_e,
    *,
    sigma_ph: float | None = None,
    eps_acoustic: float = EPS_ACOUSTIC,
    delta_approx: bool = True,
    temperature: float = 0.0,
    fsthick: float | None = None,
    dos_meshes: list[tuple[int, int, int]] | None = None,
    verbose: bool = False,
) -> LambdaConvergence:
    """
    Fine-mesh convergence + Richardson extrapolation of the el-ph mass
    enhancement lambda, the notoriously slowest-converging el-ph quantity
    (a simultaneous double-delta Fermi-surface integral on BOTH the k and
    k+q meshes). Evaluates `alpha2f` in its fine-mesh mode (interpolated
    H(k+q), `hr` given) on each mesh in ``meshes``.

    DO NOT extrapolate lambda directly -- extrapolate the coupling.
    ---------------------------------------------------------------
    This function used to fit ``lambda(N)`` against ``h = 1/N`` and return the
    intercept. That fit is ill-conditioned, and measurably so. lambda is a
    RATIO,

        lambda = 2 integral alpha2F/omega = N(eF) * 2 <<|g|^2/omega>>,

    of which the two factors converge at wildly different rates. The
    double-Fermi-surface average ``2 <<|g|^2/omega>>`` is a smooth average of a
    smooth quantity; ``N(eF)`` at finite smearing is a sum of delta functions
    over however many states happen to land in a thin shell, and on an
    affordable mesh it is dominated by sampling noise. lambda inherits that
    noise in full. Measured on Al (fixed sigma_e = 35 meV, EPW's degaussw):

        code   N    N(eF)    lambda   lambda/N(eF)
         waw  10   0.0605    0.1288       2.13
         waw  12   0.0541    0.1368       2.53
         waw  14   0.1650    0.3668       2.22
         EPW  12   0.2260    0.4838       2.14
         EPW  20   0.3538    0.7920       2.24

    lambda spans a factor 6.1 across those rows; lambda/N(eF) spans 1.19, and
    agrees between the two CODES to within that same scatter. Fitting a
    straight line to the lambda column extrapolates the DOS sampling error,
    not the physics -- and since the sequence is not even monotonic, the
    intercept is close to meaningless.

    So this routine instead:

      1. forms the DOS-independent coupling ``rho(N) = lambda(N) / N(eF)(N)``
         on each mesh (both from the SAME mesh and smearing, so the noise
         cancels in the ratio rather than compounding);
      2. extrapolates ``rho`` to ``h = 1/N -> 0``;
      3. converges ``N(eF)`` separately on its own, much denser mesh sequence
         (``dos_meshes``) -- cheap, because it needs only H(k), never g;
      4. returns ``lambda_extrapolated = rho_extrapolated * dos_converged``.

    All the intermediates are returned so the two extrapolations can be
    inspected independently, which is the point: if ``coupling`` is smooth and
    ``dos_values`` is not, you know exactly which one is limiting you.

    Smearing: pass ``sigma_e`` as a scalar to converge the mesh at fixed
    broadening (recommended -- it isolates the mesh error, and step 1 is then
    a clean h -> 0 limit), or as a callable ``mesh -> sigma`` to shrink it with
    the mesh. Note the callable form moves BOTH error sources at once and was
    the other half of why the old fit misbehaved.

    Parameters
    ----------
    hr : HamiltonianR
        Wannier electron Hamiltonian (for H(k) and H(k+q) interpolation).
    g_R, R_e, degen_e, R_q, degen_q : from `wannier_transform_elph` -- the
        mesh-independent real-space coupling (built ONCE on the coarse
        ab-initio mesh; reused at every fine mesh here).
    phonon_fn : callable(qpts_frac (nq,3)) -> (omega_ph (nq,nmodes) Hartree,
        eigvec_ph (nq,nmodes,nmodes) complex) -- phonons interpolated onto an
        arbitrary q-mesh, e.g. a closure around
        `analysis.phonon.interpolate_phonons` (+ `units.cm1_to_hartree`).
    masses_amu, types, fermi_energy, omega_grid, sigma_ph, eps_acoustic,
    delta_approx, temperature : as `alpha2f`.
    dos_at_ef : leave at ``None`` (the sane default) so N(eF) is rebuilt on
        every mesh with that mesh's own ``sigma_e``, exactly as `alpha2f`
        does and EPW does. A FIXED N(eF) across a sequence whose smearing is
        deliberately shrinking is inconsistent by construction: the
        numerator's delta-delta sum tracks sigma_e while the denominator does
        not, so each lambda(N) carries a different systematic bias and the
        1/N -> 0 straight line extrapolates that drift, not the physics.
        Pass a float only to reproduce that older behaviour deliberately.
    meshes : list of (N1,N2,N3), increasing density (>= 2 meshes).
    sigma_e : float or callable(mesh)->float, electronic smearing (Hartree).
    fsthick : as `alpha2f` -- strongly recommended here, since it is what
        makes the denser end of the sequence affordable at all.
    dos_meshes : list of (N1,N2,N3) or None
        Dense meshes on which N(eF) is converged for step 3. These cost only
        an H(k) diagonalisation per point (no g, no phonons), so they can and
        should be far denser than ``meshes``. Default: 2x, 3x and 4x the
        linear size of the densest entry in ``meshes``. Ignored when an
        explicit ``dos_at_ef`` float is supplied, in which case that value is
        used as the converged DOS and only the coupling is extrapolated.
    verbose : print each mesh as it completes. Worth turning on: the dense end
        of the sequence takes tens of minutes per mesh, and without it the
        call is silent from start to finish.

    Returns
    -------
    LambdaConvergence
    """
    meshes = [tuple(int(x) for x in m) for m in meshes]
    if len(meshes) < 2:
        raise ValueError("lambda_richardson needs >= 2 meshes to extrapolate.")

    from ..interfaces.ase.structure import monkhorst_pack

    n_lin, lambdas, sig_used, a2f_all, dos_used = [], [], [], [], []
    for mesh in meshes:
        sig = float(sigma_e(mesh)) if callable(sigma_e) else float(sigma_e)
        kpts = monkhorst_pack(mesh)
        qpts = monkhorst_pack(mesh)
        eig, U = band_eigensystem(hr, kpts)
        omega_ph, eigvec_ph = phonon_fn(qpts)
        n_ef = (float(dos_at_ef) if dos_at_ef is not None
                else fermi_surface_dos(eig, fermi_energy, sig))
        if not n_ef > 0.0:
            # Too coarse to resolve the Fermi surface at this smearing. Record
            # NaN and carry on rather than aborting the whole sequence -- but
            # NaN, not 0, so it cannot be mistaken for a converging point and
            # cannot silently enter the extrapolation fit.
            import warnings
            warnings.warn(
                f"lambda_richardson: mesh {mesh} at sigma_e = {sig:g} Ha has no "
                "state in the Fermi window; lambda recorded as NaN and excluded "
                "from the fit.", RuntimeWarning, stacklevel=2,
            )
            n_lin.append((mesh[0] * mesh[1] * mesh[2]) ** (1.0 / 3.0))
            lambdas.append(np.nan)
            dos_used.append(np.nan)
            sig_used.append(sig)
            a2f_all.append(np.full_like(omega_grid, np.nan))
            continue
        a2f = alpha2f(
            eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts, None,
            omega_ph, eigvec_ph, masses_amu, types,
            fermi_energy=fermi_energy, dos_at_ef=n_ef, omega_grid=omega_grid,
            sigma_e=sig, sigma_ph=sigma_ph, eps_acoustic=eps_acoustic, hr=hr,
            delta_approx=delta_approx, temperature=temperature, fsthick=fsthick,
        )
        n_lin.append((mesh[0] * mesh[1] * mesh[2]) ** (1.0 / 3.0))
        lambdas.append(lambda_from_a2f(a2f, omega_grid))
        dos_used.append(n_ef)
        sig_used.append(sig)
        a2f_all.append(a2f)
        if verbose:
            print(f"  [lambda_richardson] {mesh}  sigma_e={sig:.5g}  "
                  f"N(eF)={n_ef:.6g}  lambda={lambdas[-1]:.4f}  "
                  f"lambda/N(eF)={lambdas[-1] / n_ef:.4f}", flush=True)

    n_lin = np.asarray(n_lin)
    lambdas = np.asarray(lambdas)
    dos_used = np.asarray(dos_used)

    # Step 1-2: extrapolate the DOS-INDEPENDENT coupling rho = lambda/N(eF),
    # not lambda. NaN meshes (Fermi surface unresolved) are excluded rather
    # than poisoning the whole fit to NaN.
    coupling = lambdas / dos_used
    h = 1.0 / n_lin
    ok = np.isfinite(coupling)
    if ok.sum() < 2:
        raise ValueError(
            "lambda_richardson: fewer than 2 usable meshes (the rest could not "
            "resolve the Fermi surface at their smearing); nothing to extrapolate."
        )
    _, rho_inf = np.polyfit(h[ok], coupling[ok], 1)

    # Step 3: converge N(eF) on its own, much denser sequence -- H(k) only, so
    # it costs a rounding error next to the (k,q) double sums above. An
    # explicit dos_at_ef is taken as already converged.
    if dos_at_ef is not None:
        dos_conv = float(dos_at_ef)
        dos_n, dos_v = np.array([np.nan]), np.array([dos_conv])
    else:
        if dos_meshes is None:
            nmax = max(max(m) for m in meshes)
            dos_meshes = [(f * nmax,) * 3 for f in (2, 3, 4)]
        dos_meshes = [tuple(int(x) for x in m) for m in dos_meshes]
        sig_ref = (float(sigma_e(meshes[-1])) if callable(sigma_e)
                   else float(sigma_e))
        dos_n, dos_v = [], []
        for m in dos_meshes:
            e, _ = band_eigensystem(hr, monkhorst_pack(m))
            dos_n.append((m[0] * m[1] * m[2]) ** (1.0 / 3.0))
            dos_v.append(fermi_surface_dos(e, fermi_energy, sig_ref))
            if verbose:
                print(f"  [lambda_richardson dos] {m}  N(eF)={dos_v[-1]:.6g}",
                      flush=True)
        dos_n, dos_v = np.asarray(dos_n), np.asarray(dos_v)
        if len(dos_v) >= 2:
            _, dos_conv = np.polyfit(1.0 / dos_n, dos_v, 1)
        else:
            dos_conv = float(dos_v[-1])
        dos_conv = float(dos_conv)

    return LambdaConvergence(
        lambda_extrapolated=float(rho_inf * dos_conv),
        n_linear=n_lin, lambdas=lambdas, sigma_e=np.asarray(sig_used), a2f=a2f_all,
        dos_at_ef=dos_used, coupling=coupling,
        coupling_extrapolated=float(rho_inf), dos_converged=dos_conv,
        dos_n_linear=dos_n, dos_values=dos_v,
    )


class AutoLambdaResult(NamedTuple):
    """Result of `lambda_auto`. Both stages are reported separately so the
    answer can be audited rather than trusted; ``lambda_value`` is the
    DENSEST-mesh estimate and ``lambda_uncertainty`` the last consecutive-mesh
    change, i.e. the convergence criterion's own residual."""
    lambda_value: float
    lambda_uncertainty: float    #: worse of the last two consecutive changes
    omega_log: float             #: Hartree, from the densest mesh
    omega_log_uncertainty: float #: worse of the last two consecutive changes
    omega_log_converged_flag: bool
    omega_logs: np.ndarray       #: omega_log(N) along the coupling sequence
    #: Allen-Dynes Tc as an ENERGY k_B*Tc in Hartree, and its uncertainty
    #: PROPAGATED from lambda_uncertainty and omega_log_uncertainty. Not a
    #: convergence criterion -- reported so the exponential sensitivity of
    #: Tc to lambda is visible rather than implicit.
    tc: float
    tc_uncertainty: float
    tc_uncertainty_from_lambda: float
    tc_uncertainty_from_omega_log: float
    mu_star: float
    omega_2: float               #: Hartree, for Allen-Dynes f2
    converged: bool              #: ALL THREE criteria met
    fermi_energy: float          #: Hartree, from the electron count, then frozen
    # --- stage 1: eF and N(eF), H(k) only ---
    dos_value: float
    dos_uncertainty: float
    dos_converged_flag: bool
    dos_meshes: list
    dos_values: np.ndarray
    fermi_energies: np.ndarray   #: eF(N) along the DOS sequence
    # --- stage 2: the (k,q) double sum ---
    coupling_converged_flag: bool
    meshes: list
    n_linear: np.ndarray
    lambdas: np.ndarray          #: lambda(N) = [lambda_raw(N)/N(eF)(N)] * dos_value
    lambdas_raw: np.ndarray      #: lambda(N) as alpha2F gave it on that mesh
    dos_at_ef: np.ndarray        #: per-mesh N(eF), used to cancel sampling noise
    coupling: np.ndarray         #: lambda_raw(N) / N(eF)(N)
    a2f: list


def _scaled_mesh(base: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale a mesh keeping its aspect ratio, so an anisotropic cell such as
    MgB2's 12x12x8 refines as 18x18x12 rather than towards a cube.

    A base entry of 1 stays 1: it means that direction is deliberately not
    sampled (a slab, a monolayer, a molecule in a box), and densifying it
    would add cost while sampling a dispersionless axis."""
    return tuple(1 if b == 1 else max(2, int(round(factor * b))) for b in base)


def lambda_auto(
    hr: HamiltonianR,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    phonon_fn,
    masses_amu: np.ndarray, types: np.ndarray,
    n_electrons: float,
    omega_grid: np.ndarray,
    sigma_e: float,
    base_mesh: tuple[int, int, int],
    *,
    rtol_dos: float = 3e-3,
    tol_lambda: float = 0.03,
    rtol_wlog: float = 0.03,
    mu_star: float = 0.1,
    sigma_ph: float | None = None,
    eps_acoustic: float = EPS_ACOUSTIC,
    delta_approx: bool = True,
    temperature: float = 0.0,
    fsthick: float | str | None = "auto",
    qpoint_fn=None,
    dos_factors=(2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0),
    coupling_factors=(1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 2.8, 3.3, 4.0),
    dos_max_points: int = 12_000_000,
    coupling_max_points: int = 200_000,
    verbose: bool = False,
) -> AutoLambdaResult:
    """
    Drive lambda to an ABSOLUTE tolerance, in the two stages the quantity
    factorises into. Deliberately the simplest thing that works:

      Stage 1  For each mesh: fix eF by the model's OWN electron count
               (`fermi_level_from_electron_count`), then evaluate N(eF).
               Densify until N(eF) is stable to ``rtol_dos`` RELATIVE on TWO
               CONSECUTIVE refinements (see below). Costs
               one H(k) diagonalisation per k-point -- no g, no phonons -- so
               it can be genuinely converged. Freeze eF at its densest value.
      Stage 2  At that frozen eF, densify the (k,q) double sum until BOTH
               ``|lambda(N) - lambda(N_prev)| < tol_lambda`` AND
               ``|w_log(N) - w_log(N_prev)| < rtol_wlog * w_log(N)`` hold on
               TWO CONSECUTIVE refinements, where
               ``lambda(N) = [lambda_raw(N) / N(eF)(N)] * N(eF)_converged``.

    omega_log is converged separately, and needs no DOS correction. It is a
    RATIO of two alpha2F integrals,
    ``exp[(2/lambda) integral alpha2F ln(w)/w dw]``, so the 1/N(eF) prefactor
    cancels identically between numerator and denominator: its mesh noise is
    pure alpha2F SHAPE, not Fermi-surface sampling of the DOS. That also means
    the ratio trick that rescues lambda does nothing for it, which is why it
    gets its own criterion. It was previously just read off the densest mesh,
    and on Al it moved 28.36 -> 24.66 meV (15%) between consecutive meshes
    while lambda looked settled -- silently, since nothing tested it.

    Two consecutive agreements, not one. These sequences are noisy and
    non-monotonic, so a single pair landing inside the tolerance is a
    coincidence waiting to happen -- and it did, on the first real run: Al
    gave consecutive |delta| of 0.061, 0.086, 0.011 and a one-step rule
    declared victory on the third, after two steps 2-3x over tolerance,
    quoting +-0.011 for a column whose actual scatter was +-0.038. Requiring
    the last THREE values to agree within the tolerance costs one or two
    extra meshes and removes that failure mode without introducing a model.

    Dividing by the per-mesh N(eF) and multiplying by the converged one is the
    whole trick: lambda = N(eF) * 2<<|g|^2/omega>> is a product of a noisy
    factor and a smooth one, and on Al the raw lambda(N) swings by a factor 4
    over 10^3..18^3 (and is NOT monotonic -- 18^3 drops after 16^3 because its
    N(eF) collapsed) while the ratio spans only 1.30.

    Value = densest mesh. Uncertainty = the last consecutive-mesh change.
    -------------------------------------------------------------------
    Not an extrapolation, and not an average, both of which were tried and
    measured on Al first:

      estimator                     rho   stat SE  poss.bias   lambda  total
      extrapolate (1/N intercept)  0.0673   0.0176    0        0.389   0.101
      average of the sequence      0.0808   0.0035    0.0135   0.467   0.081
      densest mesh                 0.0717   0.0081    0.0101   0.414   0.075

    Extrapolating pays a leverage factor of 2.6 in variance to fit a slope
    that is not even significant (t = 0.79, p = 0.49). Averaging looks five
    times more precise but only because its error has moved into an invisible
    bias term -- add the bias the undetectable trend could hide and it is no
    better than anything else. The densest mesh is the only estimator whose
    bias provably vanishes as N grows and which assumes no model at all, and
    the consecutive-mesh difference is an honest, assumption-free error bar.
    (`lambda_richardson` still offers the extrapolation as a diagnostic.)

    Fixed smearing, on purpose. The true lambda is a DOUBLE limit, mesh
    ``N -> infinity`` AND ``sigma_e -> 0``. This converges the mesh at the
    ``sigma_e`` given, so the result is "lambda at this broadening,
    mesh-converged". Run it at two or three sigma_e and compare to address the
    other limit; each is cheap once mesh-converged. (For Al the DOS at 35 and
    71 meV agree to 0.11%, so there the second limit is already reached.)

    Parameters
    ----------
    hr, g_R, R_e, degen_e, R_q, degen_q, phonon_fn, masses_amu, types,
    omega_grid, sigma_ph, eps_acoustic, delta_approx, temperature, fsthick :
        as `alpha2f`.
    n_electrons : electrons per cell carried by these Wannier bands, both
        spins (3.0 for Al's sp manifold). Used to fix eF -- see
        `fermi_level_from_electron_count` for why the DFT Fermi level is the
        wrong thing to import.
    sigma_e : Hartree, FIXED. This project's Gaussian sigma, not EPW's
        ``degaussw`` -- convert with `epw_degauss_to_sigma`.
    base_mesh : (N1, N2, N3) whose ASPECT RATIO is preserved as meshes grow.
    rtol_dos : RELATIVE tolerance on N(eF), required on two consecutive
        refinements. The ``3e-3`` default is about "stable in the third
        significant digit". Deliberately relative rather than absolute:
        N(eF) is dimensionful, `analysis` carries it in states/Hartree while it
        is habitually quoted in states/eV, and the factor 27.2114 between them
        makes an absolute threshold a silent unit trap (it just runs to the end
        of the schedule and warns). A relative one is unambiguous.
    tol_lambda : ABSOLUTE tolerance on lambda, required on two consecutive
        refinements. Absolute is right here because lambda is dimensionless
        and O(0.1-1), so ``0.03`` means what it says.
    rtol_wlog : RELATIVE tolerance on omega_log, required on two consecutive
        refinements. Relative because omega_log is an energy. The ``0.03``
        default is set by Allen-Dynes, where ``Tc`` is LINEAR in omega_log, so
        3% here buys 3% of Tc.
    mu_star : Coulomb pseudopotential for the reported Allen-Dynes Tc. NOT a
        convergence criterion -- Tc and its propagated error are reported
        only. Deliberately so: Tc is EXPONENTIAL in lambda (at Al's values
        0.03 in lambda is 28% of Tc, against 3% from 3% of omega_log), so a
        few-per-cent target on Tc would demand lambda to ~0.003 -- a ~130x
        larger mesh, chasing a term that the 1-5% systematic error in the
        vertex |g| already swamps (that alone is 20-90% of Tc). Read
        ``tc_uncertainty`` as the MESH contribution, not a total error bar.
    qpoint_fn : callable(mesh) -> (qpts, q_weights) or None. Pass
        ``lambda m: irreducible_qpoints(atoms, m)`` (from
        `interfaces.ase.structure`) to sum q over the irreducible wedge --
        exact, and 24x faster already at 12^3 for fcc (measured 2.5e-4
        relative on lambda against the full BZ). ``None`` uses the full mesh
        with uniform weights.
    dos_factors, coupling_factors : mesh scale factors for the two stages.
    dos_max_points, coupling_max_points : per-mesh k-point ceilings; a
        schedule entry above its ceiling is skipped rather than attempted.
    verbose : print each mesh as it completes.

    Returns
    -------
    AutoLambdaResult -- ``converged`` is the AND of the two stage flags. False
    means the schedule or the budget ran out, never that the answer is fine.
    """
    import warnings

    from ..interfaces.ase.structure import monkhorst_pack

    sigma_e = float(sigma_e)
    if sigma_ph is None:
        sigma_ph = 2.0 * (omega_grid[1] - omega_grid[0])

    def _log(m):
        if verbose:
            print(m, flush=True)

    # ------------------------------------------------------------------
    # Stage 1: eF and N(eF). H(k) only, so push until it really converges.
    # ------------------------------------------------------------------
    dos_meshes, dos_vals, efs = [], [], []
    dos_ok, dos_unc = False, float("inf")
    for f in dos_factors:
        m = _scaled_mesh(base_mesh, f)
        if m[0] * m[1] * m[2] > dos_max_points:
            continue
        if dos_meshes and m == dos_meshes[-1]:
            continue          # schedule rounded to the same mesh: no new information
        eig_d, _ = band_eigensystem(hr, monkhorst_pack(m))
        ef = fermi_level_from_electron_count(eig_d, n_electrons, sigma_e)
        n_ef = fermi_surface_dos(eig_d, ef, sigma_e)
        dos_meshes.append(m); dos_vals.append(n_ef); efs.append(ef)
        if len(dos_vals) >= 2:
            dos_unc = abs(dos_vals[-1] - dos_vals[-2])
        _log(f"  [dos] {m}  eF={ef:.8f}  N(eF)={n_ef:.6g}  |d|={dos_unc:.3g}")
        if len(dos_vals) >= 3:
            d1 = abs(dos_vals[-1] - dos_vals[-2])
            d2 = abs(dos_vals[-2] - dos_vals[-3])
            if max(d1, d2) < rtol_dos * abs(n_ef):
                dos_unc = max(d1, d2)      # quote the worse of the two
                dos_ok = True
                break
    if len(dos_vals) < 3:
        raise ValueError(
            f"lambda_auto: the DOS stage produced {len(dos_vals)} distinct mesh(es). "
            "Convergence needs at least three (the tolerance must hold on two "
            "consecutive refinements) -- check dos_factors and base_mesh (a base_mesh "
            "entry of 1 is held fixed by design, so (1,1,1) never grows)."
        )
    dos_value, fermi_energy = float(dos_vals[-1]), float(efs[-1])
    if not dos_value > 0.0:
        raise ValueError(
            f"lambda_auto: N(eF) = {dos_value:g} is not positive -- no state within "
            "sigma_e of the Fermi level even on the densest DOS mesh."
        )
    if not dos_ok:
        warnings.warn(
            f"lambda_auto: N(eF) did not reach rtol_dos={rtol_dos:g} (last "
            f"relative change {dos_unc / abs(dos_value):.3g}); lambda inherits "
            "that error.",
            RuntimeWarning, stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Stage 2: the (k,q) double sum at the frozen eF.
    # ------------------------------------------------------------------
    meshes, n_lin, lam_raw, lam_scaled, dos_used, a2f_all = [], [], [], [], [], []
    wlogs = []
    coup_ok, lam_unc = False, float("inf")
    wlog_ok, wlog_unc = False, float("inf")
    for f in coupling_factors:
        m = _scaled_mesh(base_mesh, f)
        if m[0] * m[1] * m[2] > coupling_max_points:
            continue
        if meshes and m == meshes[-1]:
            continue          # schedule rounded to the same mesh: no new information
        kpts = monkhorst_pack(m)
        qpts, q_w = (monkhorst_pack(m), None) if qpoint_fn is None else qpoint_fn(m)
        eig, U = band_eigensystem(hr, kpts)
        omega_ph, eigvec_ph = phonon_fn(qpts)
        n_ef = fermi_surface_dos(eig, fermi_energy, sigma_e)
        if not n_ef > 0.0:
            warnings.warn(
                f"lambda_auto: mesh {m} has no state within sigma_e of eF; skipped.",
                RuntimeWarning, stacklevel=2,
            )
            continue
        fs = (12.0 * sigma_e + float(np.max(omega_ph))) if fsthick == "auto" else fsthick
        a2f = alpha2f(
            eig, U, g_R, R_e, degen_e, R_q, degen_q, kpts, qpts, None,
            omega_ph, eigvec_ph, masses_amu, types,
            fermi_energy=fermi_energy, dos_at_ef=n_ef, omega_grid=omega_grid,
            sigma_e=sigma_e, sigma_ph=sigma_ph, eps_acoustic=eps_acoustic, hr=hr,
            delta_approx=delta_approx, temperature=temperature, fsthick=fs,
            q_weights=q_w,
        )
        lr = lambda_from_a2f(a2f, omega_grid)
        meshes.append(m)
        n_lin.append((m[0] * m[1] * m[2]) ** (1.0 / 3.0))
        lam_raw.append(lr); dos_used.append(n_ef); a2f_all.append(a2f)
        lam_scaled.append(lr / n_ef * dos_value)
        wlogs.append(float(eliashberg_moments(a2f, omega_grid)[1][-1]))
        if len(lam_scaled) >= 2:
            lam_unc = abs(lam_scaled[-1] - lam_scaled[-2])
            wlog_unc = abs(wlogs[-1] - wlogs[-2])
        _log(f"  [lambda] {m}  nq={len(qpts)}  lambda_raw={lr:.4f}  N(eF)={n_ef:.6g}"
             f"  lambda={lam_scaled[-1]:.4f}  |d|={lam_unc:.3g}"
             f"  w_log={wlogs[-1]:.6g}  |dw|={wlog_unc:.3g}")
        if len(lam_scaled) >= 3:
            d1 = abs(lam_scaled[-1] - lam_scaled[-2])
            d2 = abs(lam_scaled[-2] - lam_scaled[-3])
            w1 = abs(wlogs[-1] - wlogs[-2])
            w2 = abs(wlogs[-2] - wlogs[-3])
            lam_hit = max(d1, d2) < tol_lambda
            wlog_hit = (np.isfinite(wlogs[-1])
                        and max(w1, w2) < rtol_wlog * abs(wlogs[-1]))
            if lam_hit and wlog_hit:
                lam_unc = max(d1, d2)      # quote the worse of the two
                wlog_unc = max(w1, w2)
                coup_ok, wlog_ok = True, True
                break
    if len(lam_scaled) < 3:
        raise ValueError(
            f"lambda_auto: the coupling stage produced {len(lam_scaled)} distinct "
            "mesh(es). Convergence needs at least three (the tolerance must hold on "
            "two consecutive refinements) -- check coupling_factors, "
            "coupling_max_points and base_mesh (a base_mesh entry of 1 is held "
            "fixed by design, so (1,1,1) never grows)."
        )
    if not (coup_ok and wlog_ok):
        which = []
        if not coup_ok:
            which.append(f"lambda (tol_lambda={tol_lambda:g}, last change {lam_unc:.3g})")
        if not wlog_ok:
            which.append(f"omega_log (rtol_wlog={rtol_wlog:g}, last change {wlog_unc:.3g})")
        warnings.warn(
            "lambda_auto: not converged in " + " and ".join(which)
            + "; the coupling schedule or budget ran out.",
            RuntimeWarning, stacklevel=2,
        )

    # Allen-Dynes Tc and its MESH error, propagated from the two convergence
    # residuals by finite difference (the derivatives are strongly nonlinear,
    # so an analytic linearisation would be its own approximation). Added in
    # quadrature, which ASSUMES the two are independent -- they are not, both
    # coming from the same alpha2F, so treat this as a lower bound.
    om2 = eliashberg_omega_2(a2f_all[-1], omega_grid)
    lam_f, wlog_f = float(lam_scaled[-1]), float(wlogs[-1])
    tc = allen_dynes_tc(lam_f, wlog_f, om2, mu_star)
    d_lam = abs(allen_dynes_tc(lam_f + lam_unc, wlog_f, om2, mu_star) - tc)
    d_wlog = abs(allen_dynes_tc(lam_f, wlog_f + wlog_unc, om2, mu_star) - tc)
    tc_unc = float(np.hypot(d_lam, d_wlog))

    return AutoLambdaResult(
        lambda_value=float(lam_scaled[-1]),
        lambda_uncertainty=float(lam_unc),
        omega_log=float(wlogs[-1]),
        omega_log_uncertainty=float(wlog_unc),
        omega_log_converged_flag=bool(wlog_ok),
        omega_logs=np.asarray(wlogs),
        tc=float(tc), tc_uncertainty=tc_unc,
        tc_uncertainty_from_lambda=float(d_lam),
        tc_uncertainty_from_omega_log=float(d_wlog),
        mu_star=float(mu_star), omega_2=float(om2),
        converged=bool(dos_ok and coup_ok and wlog_ok),
        fermi_energy=fermi_energy,
        dos_value=dos_value, dos_uncertainty=float(dos_unc),
        dos_converged_flag=bool(dos_ok), dos_meshes=dos_meshes,
        dos_values=np.asarray(dos_vals), fermi_energies=np.asarray(efs),
        coupling_converged_flag=bool(coup_ok), meshes=meshes,
        n_linear=np.asarray(n_lin), lambdas=np.asarray(lam_scaled),
        lambdas_raw=np.asarray(lam_raw), dos_at_ef=np.asarray(dos_used),
        coupling=np.asarray(lam_raw) / np.asarray(dos_used), a2f=a2f_all,
    )
