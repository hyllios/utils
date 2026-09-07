"""
Nonlinear shift current (postw90 `berry_task = sc`).

Ibanez-Azpiroz, Tsirkin & Souza, "Ab initio calculation of the shift
photocurrent by Wannier interpolation", PRB 97, 245143 (2018) ("IATS18").
Follows wannier90's `berry_get_sc_klist` term for term so unit factors and
index conventions match:

    sigma^abc(0;w,-w) = -i*pi*e^3/(4*hbar^2) INT[dk] SUM_{n,m} f_nm *
                        (I^abc_mn + I^acb_mn) * [delta(w_mn-w) + delta(w_nm-w)]   (Eq. 8)

    I^abc_mn = r^b_mn . r^{c;a}_nm                                              (Eq. 5)

`r_mn` (the dipole matrix element) is `AA_bar(m,n) + i*D_h_no_eta(m,n)`,
i.e. `hamiltonian_gauge_position`'s combined `A_H(m,n)`. The
generalized/covariant derivative `r^{c;a}_nm` (`gen_r_nm` in the Fortran)
is an 8-term combination (IATS18 Eq. 34 + Eq. 30 + Eq. 32) built from:
  - `AA_bar`      = the plain H-gauge-rotated position operator, U^dag.A_W.U
                    (`hamiltonian_gauge_position`'s `A_H` minus its own
                    `i*D_h` piece -- kept separate throughout this module)
  - `AA_da_bar`   = the H-gauge-rotated first k-derivative of the position
                    operator (`core.hamiltonian.position_operator_derivative_k`,
                    rotated by the same U(k), no product-rule dU/dk term)
  - `HH_da_bar`   = `hamiltonian_gauge_position`'s `dH_eig` (H-gauge
                    band-velocity/gradient matrix)
  - `HH_dadb_bar` = the H-gauge-rotated second k-derivative of H
                    (`analysis._fourier_derivs.h_and_hess_cart_batch`,
                    rotated the same way)
  - `D_h_no_eta`  = `hamiltonian_gauge_position`'s (no-eta) `D_h`
  - `D_h` (eta)   = `hamiltonian_gauge_position`'s `sc_eta`-regularized
                    `D_h_eta` (needed alongside the no-eta one)
  - `eig_da`      = `hamiltonian_gauge_position`'s (degenerate-aware) `del_eig`

Index convention for the 4-index quantities `AA_da_bar`/`HH_dadb_bar`:
shape `(nk, 3, 3, nw, nw)` with axis 1 = `c` (the free position/gradient
component of `gen_r_nm`), axis 2 = `a` (the outer generalized-derivative
direction), i.e. `AA_da_bar[:, c, a]` <-> Fortran's `AA_da_bar(n, m, c, a)`.
All other 3-index quantities (`AA_bar`/`HH_da_bar`/`D_h_no_eta`/`D_h_eta`)
are `(nk, 3, nw, nw)`, axis 1 = Cartesian component, same `[n, m]` layout
as `hamiltonian_gauge_position`'s `A_H`/`D_h`.

`sc_phase_conv`: two mathematically equivalent Fourier-sum phase
conventions (IATS18 SS III.4.2); GaAs uses `sc_phase_conv=2` (no
Wannier-centre phases), matching waw's own Fourier convention throughout.

`sc_use_eta_corr`: an additional finite-`sc_eta` correction (Eq. 19 of PRB
103, 247101 (2021)), transcribed from the Fortran's per-intermediate-band
sum; not independently verified against that 2021 paper itself.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.hamiltonian import (
    HamiltonianR, position_operator_k, position_operator_derivative_k,
    hamiltonian_gauge_position,
)
from ..units import (
    HARTREE_TO_EV, EV_TO_HARTREE, BOHR_TO_ANG, ANG_TO_BOHR, E_CHARGE, HBAR_SI,
    register_si_unit, register_from_si_unit,
)
from ._fourier_derivs import h_and_hess_cart_batch, KCHUNK
from .dos import _uniform_mesh

# Wannier90's defaults for postw90's adaptive-smearing keywords
# (also imported by spin_hall.py). KUBO_ADPT_SMR_MAX_DEFAULT is Hartree.
KUBO_ADPT_SMR_FAC_DEFAULT = float(np.sqrt(2.0))
KUBO_ADPT_SMR_MAX_DEFAULT = 1.0 * EV_TO_HARTREE

# Numerical-only floor for the adaptive smearing width (see `_adaptive_eta`'s
# docstring): avoids exact-zero joint level spacing at symmetry-enforced
# equal-velocity band pairs, negligible next to any physical smearing width.
_ADAPTIVE_ETA_FLOOR = 1e-10

# Independent (b,c) index pairs for the symmetric-under-exchange shift
# current tensor (Fortran's alpha_S/beta_S, 1-indexed there):
# bc = 0..5 <-> (xx, yy, zz, xy, xz, yz)
ALPHA_S = (0, 1, 2, 0, 0, 1)
BETA_S = (0, 1, 2, 1, 2, 2)


def _sum_AD_HD(X_bar: torch.Tensor, D_h_eta: torch.Tensor) -> torch.Tensor:
    """
    IATS18 Eq. 30 (X_bar=HH_da_bar) / Eq. 32 (X_bar=AA_bar), wannier90's
    `sum_AD`/`sum_HD`:

        sum(c,a)[n,m] = (sum_p X_bar(n,p,c).D_h(p,m,a) - X_bar(n,n,c).D_h(n,m,a))
                       - (sum_p D_h(n,p,a).X_bar(p,m,c) - D_h(n,m,a).X_bar(m,m,c))

    `D_h` here is the eta-regularized `D_h`, not `D_h_no_eta` (the two are
    distinct throughout this module).

    Args:
      X_bar  : (nk, 3, nw, nw) complex -- AA_bar or HH_da_bar
      D_h_eta: (nk, 3, nw, nw) complex -- the sc_eta-regularized D_h

    Returns (nk, 3, 3, nw, nw) complex, axes (c, a, n, m).
    """
    nk, _, nw, _ = X_bar.shape
    diagX = torch.diagonal(X_bar, dim1=-2, dim2=-1)   # (nk, 3, nw): X_bar(n,n,c)

    out = torch.zeros(nk, 3, 3, nw, nw, dtype=X_bar.dtype, device=X_bar.device)
    for c in range(3):
        for a in range(3):
            term1 = torch.matmul(X_bar[:, c], D_h_eta[:, a])            # sum_p X_bar(n,p,c) D_h(p,m,a)
            term1 = term1 - diagX[:, c][:, :, None] * D_h_eta[:, a]     # - X_bar(n,n,c) D_h(n,m,a)

            term2 = torch.matmul(D_h_eta[:, a], X_bar[:, c])            # sum_p D_h(n,p,a) X_bar(p,m,c)
            term2 = term2 - D_h_eta[:, a] * diagX[:, c][:, None, :]     # - D_h(n,m,a) X_bar(m,m,c)

            out[:, c, a] = term1 - term2
    return out


def _eta_correction_term(
    AA_bar: torch.Tensor, HH_da_bar: torch.Tensor, eig: torch.Tensor,
    a: int, c: int, sc_eta: float,
) -> torch.Tensor:
    """
    The optional `sc_use_eta_corr` finite-`sc_eta` correction (Eq. 19,
    PRB 103, 247101 (2021)), wannier90's `sc_use_eta_corr` branch -- a sum
    over intermediate bands `p not in {n, m}`:

        corr(c)[n,m] = SUM_{p != n,m}  1/(eig(n)-eig(m)) * (
            - eta^2/((eig(p)-eig(m))^2+eta^2)
              * ( AA_bar(n,p,c).HH_da_bar(p,m,a)
                  - (HH_da_bar(n,p,c) + i.(eig(n)-eig(p)).AA_bar(n,p,c)).AA_bar(p,m,a) )
            + eta^2/((eig(n)-eig(p))^2+eta^2)
              * ( HH_da_bar(n,p,a).AA_bar(p,m,c)
                  - AA_bar(n,p,a).(HH_da_bar(p,m,c) + i.(eig(p)-eig(m)).AA_bar(p,m,c)) )
        )

    Full additive correction to `gen_r_nm(c)` for fixed outer direction
    `a`; already contains its own `1/(eig(n)-eig(m))` prefactor (sign:
    `eig(n)-eig(m)`, not `eig(m)-eig(n)`), so the caller adds it directly.

    The n==m diagonal is meaningless (the `common` prefactor diverges
    there) -- callers must ignore/mask it.

    Returns (nk, nw, nw) complex, axes (n, m).
    """
    nk, nw = eig.shape
    eig_row = eig[:, :, None]      # (nk, nw, 1): eig(n), broadcasts over m
    eig_col = eig[:, None, :]      # (nk, 1, nw): eig(m), broadcasts over n
    dE = eig_row - eig_col
    eye = torch.eye(nw, dtype=torch.bool, device=eig.device)
    # Zero the n==m diagonal here (not just via the caller's off-diagonal
    # mask): 1/(eig(n)-eig(m)) is a division by zero there, and a
    # downstream 0*NaN in `_accumulate_sc_k` would otherwise poison the
    # whole per-k sum despite the diagonal's own weight being zero.
    common = torch.where(eye[None, :, :], torch.zeros_like(dE), 1.0 / torch.where(dE == 0, torch.ones_like(dE), dE))
    common = common.to(AA_bar.dtype)

    corr = torch.zeros(nk, nw, nw, dtype=AA_bar.dtype, device=AA_bar.device)
    for p in range(nw):
        eig_p = eig[:, p]                                    # (nk,)

        denom1 = (sc_eta ** 2) / ((eig_p[:, None] - eig) ** 2 + sc_eta ** 2)   # (nk,nw) fn of m: (eig(p)-eig(m))^2
        denom2 = (sc_eta ** 2) / ((eig - eig_p[:, None]) ** 2 + sc_eta ** 2)   # (nk,nw) fn of n: (eig(n)-eig(p))^2
        denom1 = denom1.to(AA_bar.dtype)
        denom2 = denom2.to(AA_bar.dtype)

        AA_np_c = AA_bar[:, c, :, p]     # (nk,nw) fn of n: AA_bar(n,p,c)
        AA_pm_c = AA_bar[:, c, p, :]     # (nk,nw) fn of m: AA_bar(p,m,c)
        HH_np_c = HH_da_bar[:, c, :, p]  # (nk,nw) fn of n: HH_da_bar(n,p,c)
        HH_pm_c = HH_da_bar[:, c, p, :]  # (nk,nw) fn of m: HH_da_bar(p,m,c)

        AA_np_a = AA_bar[:, a, :, p]     # (nk,nw) fn of n: AA_bar(n,p,a)
        AA_pm_a = AA_bar[:, a, p, :]     # (nk,nw) fn of m: AA_bar(p,m,a)
        HH_np_a = HH_da_bar[:, a, :, p]  # (nk,nw) fn of n: HH_da_bar(n,p,a)
        HH_pm_a = HH_da_bar[:, a, p, :]  # (nk,nw) fn of m: HH_da_bar(p,m,a)

        en_minus_ep = eig - eig_p[:, None]     # (nk,nw) fn of n: eig(n)-eig(p)
        ep_minus_em = eig_p[:, None] - eig      # (nk,nw) fn of m: eig(p)-eig(m)

        bracket1 = (
            AA_np_c[:, :, None] * HH_pm_a[:, None, :]
            - (HH_np_c[:, :, None] + 1j * en_minus_ep[:, :, None].to(AA_bar.dtype) * AA_np_c[:, :, None])
            * AA_pm_a[:, None, :]
        )
        bracket2 = (
            HH_np_a[:, :, None] * AA_pm_c[:, None, :]
            - AA_np_a[:, :, None] * (HH_pm_c[:, None, :] + 1j * ep_minus_em[:, None, :].to(AA_bar.dtype) * AA_pm_c[:, None, :])
        )

        contrib = common * (-denom1[:, None, :] * bracket1 + denom2[:, :, None] * bracket2)

        contrib[:, p, :] = 0.0
        contrib[:, :, p] = 0.0
        corr = corr + contrib
    return corr


def _generalized_derivative(
    AA_bar: torch.Tensor, AA_da_bar: torch.Tensor,
    HH_da_bar: torch.Tensor, HH_dadb_bar: torch.Tensor,
    D_h_no_eta: torch.Tensor, D_h_eta: torch.Tensor,
    eig: torch.Tensor, eig_da: torch.Tensor,
    sc_use_eta_corr: bool = False, sc_eta: float = 0.0,
) -> torch.Tensor:
    """
    IATS18's generalized/covariant derivative r^{c;a}_nm (`gen_r_nm` in the
    Fortran):

        gen_r_nm(c,a) = AA_da_bar(n,m,c,a)
            + (AA_bar(n,n,c)-AA_bar(m,m,c)).D_h_no_eta(n,m,a)
            + (AA_bar(n,n,a)-AA_bar(m,m,a)).D_h_no_eta(n,m,c)
            - i.AA_bar(n,m,c).(AA_bar(n,n,a)-AA_bar(m,m,a))
            + sum_AD(c,a)[n,m]
            + i.( HH_dadb_bar(n,m,c,a) + sum_HD(c,a)[n,m]
                  + D_h_no_eta(n,m,c).(eig_da(n,a)-eig_da(m,a))
                  + D_h_no_eta(n,m,a).(eig_da(n,c)-eig_da(m,c)) ) / (eig(m)-eig(n))
            [+ eta-correction, if sc_use_eta_corr]

    (only the last i*(...) group is divided by (eig(m)-eig(n))).

    Args:
      eig_da: (nk, nw, 3) real -- `hamiltonian_gauge_position`'s `del_eig`.

    Returns (nk, 3, 3, nw, nw) complex, axes (c, a, n, m).
    """
    nk, nw = eig.shape
    sum_AD = _sum_AD_HD(AA_bar, D_h_eta)     # (nk,3,3,nw,nw): (c,a,n,m)
    sum_HD = _sum_AD_HD(HH_da_bar, D_h_eta)

    diagAA = torch.diagonal(AA_bar, dim1=-2, dim2=-1)   # (nk,3,nw): AA_bar(n,n,c)
    dE = eig[:, None, :] - eig[:, :, None]              # (nk,nw,nw): E_m - E_n at [n,m]
    eye = torch.eye(nw, dtype=torch.bool, device=eig.device)
    inv_dE = torch.where(eye[None, :, :], torch.zeros_like(dE), 1.0 / torch.where(dE == 0, torch.ones_like(dE), dE))
    inv_dE = inv_dE.to(AA_bar.dtype)

    gen_r = torch.zeros(nk, 3, 3, nw, nw, dtype=AA_bar.dtype, device=AA_bar.device)
    for c in range(3):
        for a in range(3):
            diag_diff_a = diagAA[:, a][:, :, None] - diagAA[:, a][:, None, :]   # AA_bar(n,n,a)-AA_bar(m,m,a)
            diag_diff_c = diagAA[:, c][:, :, None] - diagAA[:, c][:, None, :]

            term_outer = (
                AA_da_bar[:, c, a]                                            # AA_da_bar(n,m,c,a)
                + diag_diff_c * D_h_no_eta[:, a]
                + diag_diff_a * D_h_no_eta[:, c]
                - 1j * AA_bar[:, c] * diag_diff_a
                + sum_AD[:, c, a]
            )

            inner = (
                HH_dadb_bar[:, c, a]                                          # HH_dadb_bar(n,m,c,a)
                + sum_HD[:, c, a]
                + D_h_no_eta[:, c] * (eig_da[:, :, a][:, :, None] - eig_da[:, :, a][:, None, :])
                + D_h_no_eta[:, a] * (eig_da[:, :, c][:, :, None] - eig_da[:, :, c][:, None, :])
            )
            term_last = 1j * inner * inv_dE

            gen_r_ca = term_outer + term_last

            if sc_use_eta_corr:
                gen_r_ca = gen_r_ca + _eta_correction_term(AA_bar, HH_da_bar, eig, a, c, sc_eta)

            gen_r[:, c, a] = gen_r_ca
    return gen_r


def _dipole_matrix_element(A_H: torch.Tensor) -> torch.Tensor:
    """
    IATS18's dipole matrix element r_mn = AA_bar(m,n) + i*D_h_no_eta(m,n) is
    exactly `core.hamiltonian.hamiltonian_gauge_position`'s combined `A_H`.

    Returns A_H itself, axes (n, m) as always in this module (so `r_mn`
    itself is `A_H[..., m, n]`, transposed relative to `A_H`'s own `[n, m]`
    storage layout).
    """
    return A_H


def _kmesh_spacing(mesh: tuple[int, int, int], recip_lattice: np.ndarray) -> float:
    """
    Interpolation-mesh spacing Delta_k needed for adaptive smearing
    (YWVS07 Eq. 34-35), wannier90's `kmesh_spacing_mesh`: the largest of
    the three per-direction spacings |b_i|/mesh_i, where b_i are the
    reciprocal lattice vectors (rows of `recip_lattice`).

    Args:
      mesh         : (n1, n2, n3) uniform full-BZ mesh dimensions
      recip_lattice: (3, 3) rows = reciprocal lattice vectors, Bohr^-1

    Returns Delta_k, Bohr^-1 (a plain float, not k-dependent).
    """
    b = np.asarray(recip_lattice, dtype=np.float64)
    m = np.asarray(mesh, dtype=np.float64)
    return float(np.max(np.linalg.norm(b, axis=1) / m))


def _adaptive_eta(
    del_eig: torch.Tensor, delta_k: float,
    prefactor: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    max_width: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> torch.Tensor:
    """
    Per-(n,m,k) adaptive smearing width (`kubo_adpt_smr`), wannier90's
    adaptive branch:

        eta_smr(n,m,k) = min(|eig_da(m,k,:) - eig_da(n,k,:)| * Delta_k * prefactor,
                              max_width)

    (`|.|` = Euclidean norm of the 3-vector velocity difference, symmetric
    in n,m.)

    The n==m diagonal is exactly 0 before the `min(...)`, and is masked to
    a safe placeholder rather than used (their contribution is always
    multiplied by an exactly-zero weight in `_accumulate_sc_k` anyway).

    Off-diagonal entries are clamped to a tiny positive floor
    (`_ADAPTIVE_ETA_FLOOR`) rather than allowed to hit exact 0: two
    different bands can have exactly the same group velocity by symmetry
    at a high-symmetry k-point (e.g. VBM/CBM at Gamma), which would
    otherwise divide by zero downstream. The floor is negligible next to
    any physically meaningful smearing width (~1e-10 Hartree ~ 3e-9 eV).

    Args:
      del_eig: (nk, nw, 3) real, Hartree*Bohr -- band velocities
               (`hamiltonian_gauge_position`'s own `del_eig`)
      delta_k: Bohr^-1, from `_kmesh_spacing`
      prefactor, max_width: `kubo_adpt_smr_fac`/`kubo_adpt_smr_max`, Hartree
               (already unit-converted by the caller)

    Returns eta_nm: (nk, nw, nw) real, Hartree -- axes (n, m).
    """
    nk, nw, _ = del_eig.shape
    vdiff = del_eig[:, :, None, :] - del_eig[:, None, :, :]     # (nk,nw,nw,3): v(n)-v(m)
    joint_level_spacing = torch.linalg.norm(vdiff, dim=-1) * delta_k   # (nk,nw,nw)
    eta_nm = torch.clamp(joint_level_spacing * prefactor, min=_ADAPTIVE_ETA_FLOOR, max=max_width)

    eye = torch.eye(nw, dtype=torch.bool, device=del_eig.device)
    eta_nm = torch.where(eye[None, :, :], torch.ones_like(eta_nm), eta_nm)
    return eta_nm


def _accumulate_sc_k(
    eig: torch.Tensor, occ: torch.Tensor,
    A_H: torch.Tensor, gen_r_nm: torch.Tensor,
    omega: np.ndarray, eta: float | torch.Tensor,
    kubo_eigval_max: float | None = None,
) -> torch.Tensor:
    """
    Per-k contribution to the shift-current tensor, IATS18 Eq. 8's
    integrand (wannier90's `berry_get_sc_klist` (n,m) double loop):

        sc_k(a, bc, w) = SUM_{n!=m} occ_fac(n,m) * I_nm(a,bc) *
                         [delta(w-(eig(n)-eig(m))) + delta(w-(eig(m)-eig(n)))]

        occ_fac(n,m) = occ(n) - occ(m)         (T=0 step function)
        I_nm(a,bc)   = Im[ r_mn(b).gen_r_nm(c,a) + r_mn(c).gen_r_nm(b,a) ]
        r_mn         = A_H(m,n)   (`A_H` from hamiltonian_gauge_position)

    `delta` is a Gaussian (`core.distributions.gaussian_smearing` with
    `sigma = eta/sqrt(2)`, matching wannier90's default smearing
    convention). Unlike the Fortran, this evaluates the full frequency
    grid rather than truncating to a window around each delta
    (mathematically equivalent up to exponentially small Gaussian tails,
    simpler and not performance-critical here).

    `eta` may be a plain scalar (fixed-width smearing,
    `kubo_smr_fixed_en_width`) or a `(nk, nw, nw)` tensor (per-(n,m,k)
    adaptive width from `_adaptive_eta`, `kubo_adpt_smr`).

    `kubo_eigval_max` (optional): the Fortran's `if (eig(m) >
    kubo_eigval_max .or. eig(n) > kubo_eigval_max) cycle` band cutoff;
    `None` disables it (includes all bands).

    Args:
      eig     : (nk, nw) real
      occ     : (nk, nw) real, T=0 step function (1 occupied / 0 empty)
      A_H     : (nk, 3, nw, nw) complex, axes (n, m)
      gen_r_nm: (nk, 3, 3, nw, nw) complex, axes (c, a, n, m)
      omega   : (nfreq,) real, same energy unit as eig/eta
      eta     : scalar OR (nk, nw, nw) real, same energy unit as eig/omega

    Returns (nk, 3, 6, nfreq) real.
    """
    nk, nw = eig.shape
    nfreq = omega.shape[0]
    if isinstance(eta, torch.Tensor):
        sigma = (eta / np.sqrt(2.0))[:, :, :, None]           # (nk,nw,nw,1), broadcasts over frequency
    else:
        sigma = eta / np.sqrt(2.0)

    occ_fac = occ[:, :, None] - occ[:, None, :]           # (nk,nw,nw): occ(n)-occ(m)
    eye = torch.eye(nw, dtype=torch.bool, device=eig.device)
    mask = (occ_fac.abs() > 1e-10) & (~eye[None, :, :])
    if kubo_eigval_max is not None:
        below_max = eig <= kubo_eigval_max
        mask = mask & below_max[:, :, None] & below_max[:, None, :]

    dE_nm = eig[:, :, None] - eig[:, None, :]              # (nk,nw,nw): eig(n)-eig(m)

    omega_t = torch.as_tensor(np.asarray(omega, dtype=np.float64), dtype=eig.dtype, device=eig.device)
    x1 = omega_t[None, None, None, :] - dE_nm[:, :, :, None]        # w - (eig(n)-eig(m))
    x2 = omega_t[None, None, None, :] + dE_nm[:, :, :, None]        # w - (eig(m)-eig(n))
    # torch-native counterpart of `core.distributions.gaussian_smearing`
    # (same formula; avoids feeding torch tensors through a numpy-only helper)
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    delta = norm * (torch.exp(-0.5 * (x1 / sigma) ** 2) + torch.exp(-0.5 * (x2 / sigma) ** 2))   # (nk,nw,nw,nfreq)

    weight = (occ_fac * mask.to(occ_fac.dtype))[:, :, :, None] * delta   # (nk,nw,nw,nfreq)

    r_mn = A_H.transpose(-1, -2)          # (nk,3,nw,nw): r_mn(b)[n,m] = A_H(m,n)[n,m] = A_H[m,n] transposed

    sc_k = torch.zeros(nk, 3, 6, nfreq, dtype=eig.dtype, device=eig.device)
    for a in range(3):
        for bc, (b, c) in enumerate(zip(ALPHA_S, BETA_S)):
            I_nm = (r_mn[:, b] * gen_r_nm[:, c, a] + r_mn[:, c] * gen_r_nm[:, b, a]).imag   # (nk,nw,nw)
            sc_k[:, a, bc] = torch.einsum('knm,knmf->kf', I_nm.to(weight.dtype), weight)
    return sc_k


def shift_current_tensor(
    hr: HamiltonianR, AA_R: torch.Tensor,
    recip_lattice: np.ndarray, real_lattice: np.ndarray,
    mesh: tuple[int, int, int],
    fermi_energy: float,
    omega: np.ndarray,
    eta: float = 0.05 * EV_TO_HARTREE,
    sc_eta: float = 0.04 * EV_TO_HARTREE,
    sc_use_eta_corr: bool = False,
    degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    kubo_eigval_max: float | None = None,
    kubo_adpt_smr: bool = False,
    kubo_adpt_smr_fac: float = KUBO_ADPT_SMR_FAC_DEFAULT,
    kubo_adpt_smr_max: float = KUBO_ADPT_SMR_MAX_DEFAULT,
) -> np.ndarray:
    """
    The full postw90 `berry_task=sc` pipeline on a uniform full-BZ mesh,
    IATS18 Eq. 8 (wannier90's `berry_get_sc_klist` per-k body plus its
    `sc_list = sc_list + sc_k_list*kweight` k-sum; the Fortran's own final
    SI unit conversion has moved to `waw.units.to_si_units(result,
    "shift_current", cell_volume_bohr3=...)`).

    Wires together this module's per-k building blocks: `h_and_hess_cart_batch`
    (H0/HH_da_bar/HH_dadb_bar), `position_operator_k`/
    `position_operator_derivative_k` (A_k/AA_da_bar), `hamiltonian_gauge_position`
    (eig/UU/del_eig/A_H/D_h_eta). `AA_bar` (the plain rotation, without
    `hamiltonian_gauge_position`'s own `i*D_h` addition) and `D_h_no_eta`
    (recovered as `-i*(A_H - AA_bar)`) are recomputed directly here rather
    than extending that function's return signature.

    Works entirely in Hartree/Bohr and returns the bare atomic-units
    result; no eV/Angstrom/SI conversion happens in this function.

    `mesh`: a uniform Gamma-centred full-BZ mesh (`dos._uniform_mesh`),
    kweight = 1/prod(mesh) for every k-point (a plain, unweighted mesh --
    no symmetry reduction, matching every other capability in this
    codebase).

    Args:
      hr, AA_R            : Hartree/Bohr, HamiltonianR + position-operator R-matrix
      recip_lattice, real_lattice: Bohr^-1 / Bohr
      mesh                : (n1, n2, n3) uniform full-BZ mesh dimensions
      fermi_energy        : scalar, Hartree
      omega               : (nfreq,) real, Hartree
      eta, sc_eta         : scalars, Hartree (fixed-width Gaussian smearing /
                             D_h regularization width, respectively).
                             `eta` is ignored when `kubo_adpt_smr=True`.
      kubo_eigval_max     : optional band-energy cutoff, Hartree
      kubo_adpt_smr       : use adaptive (per band-pair, per-k) smearing
                             (YWVS07 recipe, `_adaptive_eta`) instead of the
                             fixed `eta` width -- matches postw90's own
                             `kubo_adpt_smr` default of `.true.`
      kubo_adpt_smr_fac   : adaptive-width prefactor (`kubo_adpt_smr_fac`),
                             dimensionless, default sqrt(2)
      kubo_adpt_smr_max   : adaptive-width cap (`kubo_adpt_smr_max`), Hartree,
                             default 1 eV (`KUBO_ADPT_SMR_MAX_DEFAULT`)

    Returns sigma: (3, 6, nfreq) real, Bohr^3*Hartree^-1 (raw, atomic) --
    axes (a, bc), bc indexing the 6 independent symmetric (b,c) pairs via
    `ALPHA_S`/`BETA_S` (xx, yy, zz, xy, xz, yz).
    """
    kpts_frac = _uniform_mesh(mesh)
    nk_total = kpts_frac.shape[0]
    nfreq = len(omega)

    omega = np.asarray(omega, dtype=np.float64)
    delta_k = _kmesh_spacing(mesh, recip_lattice) if kubo_adpt_smr else None

    sc_sum = torch.zeros(3, 6, nfreq, dtype=torch.float64)

    for start in range(0, nk_total, KCHUNK):
        kc = kpts_frac[start:start + KCHUNK]

        H0, grad_cart, hess_cart = h_and_hess_cart_batch(hr, kc, recip_lattice)
        A_k, _ = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)
        dA_dk = position_operator_derivative_k(AA_R, hr.R_vectors, hr.degen, recip_lattice, kc)

        eig, UU, del_eig, A_H, D_h_eta = hamiltonian_gauge_position(
            H0, grad_cart, A_k, degen_thresh=degen_thresh, sc_eta=sc_eta,
        )

        AA_bar = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), A_k, UU)          # (nk,3,nw,nw)
        D_h_no_eta = -1j * (A_H - AA_bar)

        HH_da_bar = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)   # dH_eig

        # dA_dk's own axes are (b=deriv dir, c=component); this module's
        # AA_da_bar convention is (c=component, a=deriv dir) -- transpose
        # after the simple same-U rotation (no product-rule dU/dk term
        # needed, see module docstring).
        dA_dk_rot = torch.einsum('kni,kbcnm,kmj->kbcij', UU.conj(), dA_dk, UU)
        AA_da_bar = dA_dk_rot.transpose(1, 2)

        HH_dadb_bar = torch.einsum('kni,kabnm,kmj->kabij', UU.conj(), hess_cart, UU)   # symmetric in (a,b)

        occ = (eig <= fermi_energy).to(torch.float64)

        gen_r_nm = _generalized_derivative(
            AA_bar, AA_da_bar, HH_da_bar, HH_dadb_bar, D_h_no_eta, D_h_eta,
            eig, del_eig, sc_use_eta_corr=sc_use_eta_corr, sc_eta=sc_eta,
        )

        if kubo_adpt_smr:
            eta_use = _adaptive_eta(del_eig, delta_k, prefactor=kubo_adpt_smr_fac, max_width=kubo_adpt_smr_max)
        else:
            eta_use = eta

        sc_k = _accumulate_sc_k(eig, occ, A_H, gen_r_nm, omega, eta_use, kubo_eigval_max=kubo_eigval_max)
        sc_sum += sc_k.sum(dim=0).to(torch.float64)

    kweight = 1.0 / nk_total
    sc_raw_atomic = sc_sum * kweight                              # Bohr^3 . Hartree^-1

    return sc_raw_atomic.numpy()


@register_si_unit("shift_current")
def _shift_current_to_si(sc_atomic, *, cell_volume_bohr3: float):
    """
    Bohr^3*Hartree^-1 -> A/V^2. wannier90's final SI conversion:
    `fac = eV_seconds*pi*e^3/(4*hbar^2*V_c)`, `eV_seconds = hbar/e`, applied
    to the intermediate Bohr^3.Hartree^-1 -> Ang^3.eV^-1 rescale
    (`BOHR_TO_ANG**3 / HARTREE_TO_EV`).
    """
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    sc_raw_eVAng3 = np.asarray(sc_atomic) * (BOHR_TO_ANG ** 3) / HARTREE_TO_EV

    eV_seconds = HBAR_SI / E_CHARGE
    fac = eV_seconds * np.pi * E_CHARGE ** 3 / (4.0 * HBAR_SI ** 2 * cell_volume_ang3)

    return fac * sc_raw_eVAng3


@register_from_si_unit("shift_current")
def _shift_current_from_si(sc_si, *, cell_volume_bohr3: float):
    """Inverse of `_shift_current_to_si` -- same `cell_volume_bohr3` kwarg."""
    cell_volume_ang3 = cell_volume_bohr3 * BOHR_TO_ANG ** 3
    eV_seconds = HBAR_SI / E_CHARGE
    fac = eV_seconds * np.pi * E_CHARGE ** 3 / (4.0 * HBAR_SI ** 2 * cell_volume_ang3)

    sc_raw_eVAng3 = np.asarray(sc_si) / fac
    return sc_raw_eVAng3 * (ANG_TO_BOHR ** 3) * HARTREE_TO_EV
