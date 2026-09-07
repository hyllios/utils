"""
Spin accumulation coefficient (SAC), Shitade & Minamitani, npj Spintron.
3, 29 (2025) ("SM25"), https://doi.org/10.1038/s44306-025-00096-x.

The SAC gamma^ij_sa characterizes the spin accumulation <Delta s_a> = g^ij_sa
d_xi E_j induced at a surface by the spin Hall effect, evaluated as a bulk
Fermi-surface property (SM25 Eq. 1):

    gamma^ij_sa = -(e/hbar) sum_n int d^3k/(2pi)^3 [-f'(eps_n(k))]
                  [ s^i_na(k) . deps_n(k)/dk_j  -  s_na(k) . eps^ijk m_nk(k) ]

**Overall-sign caveat (2026-07-28).** Transcribed above EXACTLY as SM25 Eq. 1
reads in the article's MathML (`pdf/Wannier interpolation of spin accumulation
coefficient _ npj Spintronics.html` -- use that, the equations extract exactly;
do NOT OCR the PDF). Note the `[-f'(eps)]`: with e=hbar=1 and -f' -> +delta,
Eq. 1 gives `gamma = -sum_n int delta(eps_n-E_F)[...]`, whereas the code below
accumulates `+gaussian_smearing(...)*[...]`, i.e. **-gamma(SM25)** if `e` means
+|e|. SM25 do not state whether `e` is the elementary charge or the signed
electron charge, and flipping the wrong way would double the error, so the code
is left as-is and the ambiguity recorded here. It does not affect any symmetry
or convergence statement (a global sign leaves the D3/C3 analysis invariant),
only the reported sign of gamma; settle it against a published Te number before
quoting signs. The same applies to `_sac_to_si`, already marked tentative.
**RESOLVED 2026-07-28 (late, superseding the evening anchor which was made
with the pre-correction kernel): SM25's `e` is the ELEMENTARY charge.** With
the exact-anchor-corrected kernel (m-term and s^i interband fixes below), the
Fig. 2 anchors (gamma_1(E_F-1.3)<0, gamma_1(E_F+0.8)>0, gamma_4 mostly>0)
match only when the -(e/hbar) prefactor is applied literally with e=+|e|,
i.e. gamma = -sum_n int delta[...]; the returned gamma carries this minus and
IS SM25's plotted convention. Fig. 2 also fixes the SI unit of gamma:
(hbar/e) S/cm, the SAME unit as sigma_alpha -- use the spin-Hall conversion
factor, not an independent one.

with f'(eps) = df/deps (negative-valued; -f'(eps) -> delta(eps-E_F) at low
smearing, the SAME Fermi-surface weight `gyrotropic.py`'s D/K/C/DOS tensors
already use via `gaussian_smearing`). Since f'(eps) = -gaussian_smearing(eps
-E_F, sigma) and e=hbar=1 in Hartree atomic units, the leading -(e/hbar)
and the f'->-gaussian_smearing substitution cancel: the code below sums
+gaussian_smearing(eps_n-E_F,sigma)*[...] directly (no separate prefactor
multiplication needed) and that IS gamma^ij_sa in Hartree atomic units --
same "raw Bohr^2, no cell-volume/e/hbar factors yet" convention as
`topology.anomalous_hall_conductivity`'s `sigma_bohr2` (see `_sac_to_si`
for the tentative SI conversion, and its docstring for why it is only
order-of-magnitude, not independently anchored the way AHC's S/cm is).
The three per-band ingredients (SM25
Eq. 2a-2c) are all obtained via postw90's "fake occupation" trick (only
band n occupied, `topology._jjp_jjm_from_occ`) -- the SAME trick already
used by `gyrotropic.gyrotropic_tensors` for its own per-band curvature/
orbital-moment isolation, so this module is a direct sibling of that one:

    s_na(k)   = <u_n|s_a|u_n>                     -- spin expectation
    s^i_na(k) = (i/2)<u_n|s_a Q_n|d_ki u_n> + c.c. -- spin magnetic
                                                       quadrupole moment
    eps^ijk m_nk(k) = -(i/2)<d_ki u_n|[eps_n-H(k)]|d_kj u_n> + c.c.
                                                    -- orbital magnetic moment
                       (RESOLVED 2026-07-28 late: the 2-band TB-limit anchor
                       -- exact eigenvector algebra gives eps^{ijk}m_k =
                       Im sum_{m!=n} dH^i_nm dH^j_mn/(eps_n-eps_m) -- pins
                       `_imfgh_chunk`'s imh-img at exactly -2x SM25's Eq. 2c
                       m, so the kernel uses m_nk = (img-imh)/2. The
                       postw90-certified orbital-moment code is untouched;
                       only this module's use of it carries the conversion.
                       A relative sign error between Eq. 1's two terms is
                       invisible to the D3/C3 symmetry tests -- both terms
                       are separately covariant -- which is why this needed
                       its own anchor.)

`m_nk` is `orbital_magnetization._imfgh_chunk`'s per-band `(img_n - imh_n)/2`
(gyrotropic.py's `orb_nk` ingredient, rescaled to SM25's Eq. 2c convention).
`s_na` generalizes `spin_texture.interpolate_spin` from "diagonal only" to
the full H(k)-eigenbasis-rotated spin matrix (needed for `s^i_na` below
too, so computed once and reused).

**s^i_na: SM25's Eq. 4-7, not perturbation theory (2026-07-28 CORRECTION).**
It is tempting to expand Eq. 2b over the eigenbasis, `<u_n|s_a Q_n|d_ki u_n>
= sum_{m!=n} S_a[n,m] <u_m|d_ki u_n>`, and substitute non-degenerate
perturbation theory `<u_m|d_ki u_n> = <u_m|dH/dk_i|u_n>/(eps_n-eps_m)` (i.e.
`topology._jjp_jjm_from_occ`'s one-hot JJp), giving
`s^i_na = Re[(S_a@JJp_i)[n,n]]`. **That is what this module used to do and it
is WRONG** -- two terms are missing, and the result is gauge-DEPENDENT:

1. SM25's own text splits the projector as `Q_n(k) = Q(k) + Q_in(k)` with
   `Q_in = |u> g <u|`, where `Q` projects OUT OF THE WANNIER SUBSPACE and
   `Q_in` covers the in-subspace unoccupied states. The eigenbasis sum over
   `m != n` captures only `Q_in`; the `Q` part (Eq. 5a,
   `Qtilde^i_a = i<u|s_a Q|d_ki u>`) needs genuine ab-initio matrix elements.
2. In a Wannier interpolation `<u_m|d_ki u_n>` is not dH/(dE) alone -- per
   WYSV06 it carries the AA_R Berry connection too (`Abar_i` + the D-matrix),
   which is Eq. 6's explicit `-S_a f A_i f` term.

Both omissions break the gauge invariance SM25 stress for their Eqs. (3)/(4),
so the old route returned the WRONG PHYSICAL QUANTITY.

**It is NOT, however, the cause of trigonal Te's D3/C3 symmetry violation** --
a tempting story that was tested and refuted (2026-07-28). On an exactly
C3-symmetric synthetic model both the old and the new s^i_na are C3 covariant
to ~1e-14 (`tests/test_spin_accumulation.py::
test_old_dh_over_de_route_is_also_c3_covariant`): the old expression is built
from H_R and SS_R, both covariant, so their product is too. Te's violation
(C3 residual 1.9, i.e. 190% of |gamma|, independent of k-mesh and smearing,
with only ~23% of the tensor norm inside the 4-dimensional D3-invariant
subspace) comes instead from **the Wannier model itself not being
symmetry-adapted** -- the `sitesym` gap. Measured gauge-invariantly on the
cached hexagonal MgB2 models, the interpolated bands break C3 by up to 226 meV
(nb16 window) and 17.6 meV (corrected window); the SAC's energy denominators
and Fermi-surface deltas amplify a few-meV band asymmetry to O(1), the same
"spiky Fermi-surface BZ integral" pathology documented for Fe3RuN's AHC. The
Wigner-Seitz R-set is NOT at fault (closed under C3, degeneracies consistent,
sum_R 1/degen == N1*N2*N3 exactly for both the (6,6,8) and (3,3,4) Te meshes).

What is implemented (SM25 Eq. (6) with a one-band projector f = |n><n|;
2026-07-28 correction #3 -- the first version of this correction dropped a
term, caught by the TB-limit anchor test):

    s^i_na = Re[(U^dag Q^W_ai U)_nn] - Re[S_nn Abar^H_nn]
             + Re[sum_{m!=n} S^H_nm D^i_mn]

    Q^W_ai(k) = i<u|s_a|d_ki u>  <- ab initio, pw2wannier90 `.sIu`,
                                    interpolated via `spin_texture.
                                    spin_position_r`'s QQ_R
    Abar_i(k) = i<u|d_ki u>      <- the AA_R Berry connection
    S_a(k)    = <u|s_a|u>        <- SS_R, the same spin operator as s_na
    D^i_mn    = dH_mn/(eps_n-eps_m)  <- `topology._jjp_jjm_from_occ`

The first two terms are the out-of-subspace + Berry-connection content the
old dH/(dE)-only route missed; the THIRD is the old route itself (in-subspace
interband), whose TB limit (QQ_R = AA_R = 0) is pinned against a literal
finite-difference transcription of Eq. 2b. All three pieces are separately
gauge covariant. The intermediate version of this correction claimed the
D-matrix piece "drops out of the diagonal" -- that confused the product of
diagonals with the diagonal of the operator product (SM25's tr[i S_a g
dtilde f] is a trace over off-diagonal products and carries D); the
TB-limit full-gamma test now pins the complete combination. This is
cheaper than the old route.

Status: a genuinely new capability -- not yet in any released Wannier90/
postw90 (SM25 report their own from-scratch implementation). Validated
here against (1) the D3/D3h point-group tensor-structure identities SM25
themselves use (Eq. 8 for trigonal tellurium: only 4 independent
components survive, the rest must vanish), and (2) gauge invariance under
the number of spread-minimization iterations -- SM25's own validation,
comparing "0" vs "1000" extra CG steps and checking gamma is unchanged --
see tests/test_analysis_spin_accumulation.py. Not yet cross-validated
against a real wannier90/postw90 reference number (none exists to compare
against for this quantity at the time of writing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR, position_operator_k, operator_k
from ..core.distributions import gaussian_smearing
from ..units import (
    EV_TO_HARTREE, BOHR_TO_ANG, E_CHARGE, HBAR_SI,
    register_si_unit, register_from_si_unit,
)
from ._fourier_derivs import h_and_grad_frac_batch, KCHUNK
from .orbital_magnetization import _imfgh_chunk
from .topology import _jjp_jjm_from_occ

__all__ = ["SpinAccumulationResult", "spin_accumulation_coefficient"]

# Genuine Levi-Civita symbol, NOT an axial-vector shorthand: unlike AHC's
# sigma^ab (antisymmetric, faithfully encoded by 3 axial components) or
# m_nk itself (see below), gamma^ij_sa is a general rank-2 tensor in (i,j)
# -- only its second term (s_na * eps^ijk m_nk) is antisymmetric; the first
# term (s^i_na . deps_n/dk_j) is a generic product with no i<->j symmetry.
# All 9 (i,j) components must be evaluated; `topology._AXIAL_PAIRS` (which
# only enumerates 3 of them) does NOT apply here.
_EPS3 = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPS3[_i, _j, _k] = 1.0
    _EPS3[_j, _i, _k] = -1.0


@dataclass
class SpinAccumulationResult:
    """
    SAC gamma^ij_sa vs. Fermi energy, raw atomic units -- **Bohr^2**, the
    same "bare curvature-mean, no e/hbar/cell-volume factors yet" convention
    as `topology.AHCResult.sigma` (the leading -(e/hbar) of Eq. 1 is already
    folded in via the f'(eps)->+gaussian_smearing substitution, see the
    module docstring; e=hbar=1 in Hartree atomic units so no further
    multiplication is needed for the atomic-unit value itself).

    gamma is the FULL rank-2 (i,j) tensor (not an axial 3-vector like AHC's
    curvature -- see `_EPS3`'s docstring note above), spin axis a as the
    last dimension: shape (nf, 3, 3, 3), axes (i, j, spin-a), all Cartesian
    (x,y,z) = (0,1,2).
    """
    fermi_energies: np.ndarray   # (nf,) Hartree
    gamma:          np.ndarray   # (nf, 3, 3, 3) raw atomic units, Bohr^2 -- axes (i, j, spin-a)
    mesh:           tuple


def _spin_matrix_eigenbasis(SS_R: torch.Tensor, hr: HamiltonianR, kc: np.ndarray,
                            H0: torch.Tensor, UU: torch.Tensor) -> torch.Tensor:
    """
    The full (not just diagonal) spin operator matrix in the H(k)
    eigenbasis, for all 3 Cartesian spin axes at once -- the generalization
    of `spin_texture.interpolate_spin` (which only keeps the diagonal)
    needed for `s^i_na`'s off-diagonal matrix elements too.

    Returns S_eig: (nc, 3, nw, nw) complex -- axes (spin-axis a, n, m).
    """
    S_k = torch.stack([
        operator_k(SS_R[..., a], hr.R_vectors, hr.degen, kc) for a in range(3)
    ], dim=1)   # (nc, 3, nw, nw), Wannier gauge
    return torch.einsum('kni,kanm,kmj->kaij', UU.conj(), S_k, UU)   # -> H(k) eigenbasis


def spin_accumulation_coefficient(
    hr: HamiltonianR, AA_R: torch.Tensor, BB_R: torch.Tensor, CC_R: torch.Tensor,
    SS_R: torch.Tensor, recip_lattice: np.ndarray, real_lattice: np.ndarray,
    fermi_energies, mesh: tuple[int, int, int],
    sigma: float = 0.02 * EV_TO_HARTREE, degen_thresh: float = 1e-3 * EV_TO_HARTREE,
    QQ_R: torch.Tensor | None = None,
) -> SpinAccumulationResult:
    """
    The SAC gamma^ij_sa (SM25 Eq. 1) on a uniform full-BZ k-mesh, for a
    scan of Fermi energies at once (matching `anomalous_hall_conductivity`'s
    convention -- this is a bulk-averaged quantity over the whole BZ, unlike
    `gyrotropic_tensors`'s arbitrary-box convention, since SM25's own
    postw90 calculations use full uniform meshes, e.g. 256^3 for Te).

    Args:
      hr, AA_R, BB_R, CC_R: Hartree/Bohr, same R-grid (as `orbital_magnetization`)
      SS_R         : (nR, nw, nw, 3) complex, from `spin_texture.spin_operator_r`,
                     same R-grid as `hr`
      recip_lattice, real_lattice: Bohr^-1/Bohr
      fermi_energies: scalar or (nf,) array, Hartree
      mesh         : (N1, N2, N3) uniform full-BZ k-mesh
      sigma        : Gaussian smearing width standing in for -f'(eps) ->
                     delta(eps-E_F) (SM25/postw90's `smearing`, e.g. 0.02 eV)
      degen_thresh : adjacent-band skip threshold (s^i_na/m_nk are only
                     meaningful for a non-degenerate band), Hartree
      QQ_R         : (3(spin a), 3(cart i), nR, nw, nw) complex, the
                     spin-position operator from `spin_texture.
                     spin_position_r` (needs pw2wannier90's `.sIu`, i.e.
                     `generate_overlaps(..., write_sIu=True)`), same R-grid
                     as `hr`. REQUIRED -- see the module docstring's
                     "2026-07-28 correction": s^i_na cannot be reconstructed
                     from dH/(dE) alone.

    Returns SpinAccumulationResult (raw atomic units, Bohr^2, same bare
    convention as AHC's `sigma_bohr2`; see `_sac_to_si` -- tentative only,
    not independently anchored -- for an SI conversion).
    """
    if QQ_R is None:
        raise ValueError(
            "QQ_R is required. s^i_na (SM25 Eq. 2b) is NOT reconstructible from "
            "dH/(eps_n-eps_m) alone: SM25's own Eq. 5a/6 show the band-n projector "
            "splits as Q_n = Q + Q_in, and the out-of-Wannier-subspace part Q needs "
            "the ab-initio <u|sigma_a|d_ki u> matrix elements (pw2wannier90 .sIu), "
            "while the Berry-connection term -S_a f A_i f is needed on top. Omitting "
            "them makes gamma gauge-DEPENDENT, i.e. the wrong physical quantity "
            "(note: NOT the cause of trigonal Te's C3 violation -- that is the "
            "non-symmetry-adapted Wannier model, see the module docstring). "
            "Build it with "
            "`spin_texture.spin_position_r` from "
            "`generate_overlaps(..., write_sIu=True)['sIu']`."
        )

    fermi_energies = np.atleast_1d(np.asarray(fermi_energies, dtype=np.float64))
    nf = len(fermi_energies)

    ga, gb, gc = (np.arange(N, dtype=np.float64) / N for N in mesh)
    kpts = np.stack(np.meshgrid(ga, gb, gc, indexing='ij'), axis=-1).reshape(-1, 3)
    nk = len(kpts)

    gamma_sum = np.zeros((nf, 3, 3, 3))   # (nf, i, j, spin-a)
    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=torch.complex128)

    for lo in range(0, nk, KCHUNK):
        kc = kpts[lo:lo + KCHUNK]
        H0, grad_frac = h_and_grad_frac_batch(hr, kc)
        grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad_frac)
        grad_cart = 0.5 * (grad_cart + grad_cart.conj().transpose(-1, -2))

        A_k, omega_bar_k = position_operator_k(AA_R, hr.R_vectors, hr.degen, real_lattice, kc)
        BB_k, _ = position_operator_k(BB_R, hr.R_vectors, hr.degen, real_lattice, kc)
        CC_k = torch.stack([torch.stack([
            operator_k(CC_R[a, b], hr.R_vectors, hr.degen, kc) for b in range(3)
        ], dim=1) for a in range(3)], dim=1)

        eig, UU = torch.linalg.eigh(H0)                 # (nc,nw),(nc,nw,nw), Hartree
        dH_eig = torch.einsum('kni,kanm,kmj->kaij', UU.conj(), grad_cart, UU)
        eig_np = eig.cpu().numpy()
        nc, nw = eig_np.shape

        S_eig = _spin_matrix_eigenbasis(SS_R, hr, kc, H0, UU)   # (nc,3,nw,nw)

        # s^i_na, SM25 Eq. (6) with a single-band projector f = |n><n| and
        # Eq. (7)'s Hamiltonian-gauge d_ki f = 0:
        #
        #   s^i_na = Re[ Q^i_a(n,n) - S_a(n,n) . A_i(n,n) ]   (H(k) eigenbasis)
        #
        # tr[S_a f A_i f] with f = |n><n| is the PRODUCT OF DIAGONALS, not the
        # matrix product -- that combination is what cancels the non-covariant
        # i U^dag S_a dU pieces of Q^i_a and A_i separately (each transforms as
        # X^(h) = U^dag X^(w) U + i U^dag (S_a or 1) dU), so the trace is gauge
        # invariant while neither term is. All bands at once: no per-band loop
        # and no JJp needed (the old dH/(dE) route needed one).
        QQ_k = torch.stack([torch.stack([
            operator_k(QQ_R[a, i], hr.R_vectors, hr.degen, kc) for i in range(3)
        ], dim=1) for a in range(3)], dim=1)              # (nc,3(a),3(i),nw,nw)
        q_dia = torch.einsum('kpn,kaipq,kqn->kain', UU.conj(), QQ_k, UU).real
        a_dia = torch.einsum('kpn,kipq,kqn->kin', UU.conj(), A_k, UU).real
        s_dia = torch.diagonal(S_eig, dim1=-2, dim2=-1).real      # (nc,3(a),nw)
        # (nc, a, i, nw) - (nc, a, 1, nw)*(nc, 1, i, nw)
        s_i_all = (q_dia - s_dia[:, :, None, :] * a_dia[:, None, :, :]).cpu().numpy()

        skip = np.zeros((nc, nw), dtype=bool)
        if nw > 1:
            close = (eig_np[:, 1:] - eig_np[:, :-1]) <= degen_thresh
            skip[:, 1:] |= close
            skip[:, :-1] |= close

        for n in range(nw):
            if skip[:, n].all():
                continue
            onehot = torch.zeros(nc, nw, dtype=torch.float64)
            onehot[:, n] = 1.0

            # m_nk: per-band orbital moment. `_imfgh_chunk`'s imh-img is the
            # postw90/WYSV06 accumulation convention; the 2-band TB-limit
            # anchor (tests/test_spin_accumulation.py::
            # test_full_gamma_matches_direct_eq1_tb_limit) pins it at exactly
            # -2x SM25's Eq. 2c m, so Eq. 1 needs m_nk = (img - imh)/2.
            # (2026-07-28 correction #2: before this the m-term entered with
            # the wrong sign AND double weight.)
            _, img_n, imh_n = _imfgh_chunk(
                H0, grad_cart, A_k, omega_bar_k, BB_k, CC_k,
                np.zeros(1), occ_override=onehot[None],
            )
            m_nk = 0.5 * (img_n[:, 0, :] - imh_n[:, 0, :])   # (nc,3) numpy, Hartree*Bohr^2

            # s_na: per-band spin expectation (diagonal of S_eig)
            s_na = S_eig[:, :, n, n].real.cpu().numpy()   # (nc,3)

            # s^i_na (2026-07-28 correction #3): the diagonal-only
            # q_dia - s_dia*a_dia above is INCOMPLETE. Rotating the operator
            # products shows the exact one-band formula is
            #   s^i_na = Re[(U^dag Q^W U)_nn] - Re[S_nn Abar^H_nn]
            #            + Re[sum_{m!=n} S^H_nm D^i_mn]
            # -- the first two terms are s_i_all, and the third is precisely
            # the old S@JJp interband term (whose TB limit is pinned by the
            # naive Eq.-2b finite-difference tests). The earlier claim that
            # "the D-matrix piece drops out of the diagonal" confused the
            # product of diagonals with the diagonal of the product: SM25's
            # tr[i S_a g dtilde_f] term is a trace over OFF-diagonal products
            # and carries the D matrix.
            JJp_e, _ = _jjp_jjm_from_occ(dH_eig, eig, onehot)
            s_i_corr = torch.einsum('kam,kim->kai',
                                    S_eig[:, :, n, :], JJp_e[:, :, :, n]).real
            s_i_na = s_i_all[:, :, :, n] + s_i_corr.cpu().numpy()

            deps_dk = dH_eig[:, :, n, n].real.cpu().numpy()   # (nc,3) Hartree*Bohr -- d(eps_n)/dk_i

            # Full (i,j) tensor -- NOT an axial vector (see `_EPS3`'s note):
            # term1(i,j,a) = s^i_na . deps_n/dk_j (generic, all 9 (i,j))
            # term2(i,j,a) = s_na . eps^ijk m_nk^k (antisymmetric, via _EPS3)
            term1 = np.einsum('nai,nj->nija', s_i_na, deps_dk)          # (nc,3,3,3)
            eps_m = np.einsum('ijk,nk->nij', _EPS3, m_nk)               # (nc,3,3)
            term2 = np.einsum('nij,na->nija', eps_m, s_na)              # (nc,3,3,3)
            integrand = term1 - term2

            for ife in range(nf):
                arg = eig_np[:, n] - fermi_energies[ife]
                delta = gaussian_smearing(arg, sigma)          # (nc,) 1/Hartree, stands in for -f'(eps)
                delta = np.where(skip[:, n], 0.0, delta) / nk
                gamma_sum[ife] += np.einsum('n,nija->ija', delta, integrand)

    # global sign: Eq. 1's -(e/hbar) prefactor with e the ELEMENTARY charge
    # (settled by the Fig. 2 anchors re-run on the exact-anchor-corrected
    # kernel, 2026-07-28 late: without this minus all three anchors flip).
    return SpinAccumulationResult(fermi_energies=fermi_energies, gamma=-gamma_sum, mesh=mesh)


@register_si_unit("spin_accumulation")
def _sac_to_si(gamma_atomic, *, cell_volume_bohr3: float):
    """
    Bohr^2 (raw a.u. output -- already includes Eq. 1's -(e/hbar) prefactor
    via the f'(eps)->+gaussian_smearing substitution, see the module
    docstring) -> SI. Mirrors `topology._hall_conductivity_si_factor`
    exactly, except Eq. 1's prefactor is e^1/hbar, not AHC's e^2/hbar:

        gamma[SI] = gamma[Bohr^2] * (Bohr_in_m)^2 * e / (hbar * V[m^3])

    CAVEAT, unlike AHC's S/cm (independently anchored to wannier90's own
    `berry.F90` convention): SAC has no released wannier90/postw90
    implementation to check the absolute SI scale against, and SM25 itself
    reports a tau (relaxation-time)-scaled g^ij_sa = tau*gamma in A/(V.m)
    rather than gamma alone -- consistent with this factor if tau is in
    seconds, but NOT independently verified. Treat the atomic-unit
    `SpinAccumulationResult.gamma` as the trustworthy output; this SI
    conversion is order-of-magnitude, not a checked exact match.
    """
    bohr_to_m = BOHR_TO_ANG * 1e-10
    cell_volume_m3 = cell_volume_bohr3 * bohr_to_m ** 3
    return np.asarray(gamma_atomic) * bohr_to_m ** 2 * E_CHARGE / (HBAR_SI * cell_volume_m3)


@register_from_si_unit("spin_accumulation")
def _sac_from_si(gamma_si, *, cell_volume_bohr3: float):
    """Inverse of `_sac_to_si` -- same `cell_volume_bohr3` kwarg, same caveat."""
    bohr_to_m = BOHR_TO_ANG * 1e-10
    cell_volume_m3 = cell_volume_bohr3 * bohr_to_m ** 3
    return np.asarray(gamma_si) * HBAR_SI * cell_volume_m3 / (bohr_to_m ** 2 * E_CHARGE)
