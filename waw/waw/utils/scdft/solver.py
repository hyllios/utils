"""
Band-resolved isotropic SCDFT gap equation with the Sanna-Pellegrini-Gross
exchange-correlation functional (Phys. Rev. Lett. 125, 057001 (2020)).

Their Eq. (1) is a BCS-like equation for the Kohn-Sham pairing potential,

    Delta_k = -Z_k Delta_k - (1/2) sum_k' K_kk' [tanh(beta E_k'/2)/E_k'] Delta_k'

with E_k = sqrt(xi_k^2 + Delta_k^2), and xc kernels Z (their Eq. 10) and
K^ph (their Eq. 11) built from the I and J functions in `functions.py`.
K^ee is taken as a constant mu per band pair here, in place of their static-RPA
screened Coulomb matrix elements.

ISOTROPIC BAND-RESOLVED REDUCTION. The k-sums become an energy integral per
band, sum_k' -> sum_j integral dxi', and the electron-phonon matrix elements
become alpha^2F_ij(omega). Keeping the same convention as the Eliashberg module
-- alpha^2F_ij carries the density of states of band j -- makes

    sum_{k' in j, eta} |g|^2 X(xi_k', omega) -> integral dxi' domega
                                                 alpha^2F_ij(omega) X(xi', omega)

under the usual assumption of a DOS constant over the window. mu_ij is
dimensionless (it already contains N_j), exactly as mu* is.

ONE SIMPLIFICATION WORTH NOTING. K^ph carries 1/tanh(beta E_k'/2) and Eq. (1)
multiplies by tanh(beta E_k'/2)/E_k', so that factor cancels identically. Not
forming it avoids a 0/0 at E' -> 0 that has no business being there.

Units: Hartree throughout, temperatures in Kelvin at the interface.

STATUS. The kernels are CERTIFIED; the model's energy window is a parameter.

CERTIFIED (2026-07-29). Eqs. (10) and (11), and the isotropic reduction below,
were checked against a completely independent evaluation that uses NO closed
form at all: Eqs. (5)-(9) summed directly over Matsubara frequencies for the
Einstein model. Both agree elementwise to machine precision -- the linearized
K^ph operator to 1.000000 on every (xi, xi') pair, and Z to all printed digits.
That check is `tests/test_utils_scdft.py::test_*_against_direct_matsubara`, and
it covers the gamma1*gamma2*omega prefactor, the eight s1s2s3 signs, the J
arguments, the isotropic normalisation and the omega weighting in one go. A
correct Tc is weak evidence; this is not.

Z comes out POSITIVE here where the direct route gives its negative. LM2005
Eq. (78) writes the same kernel with +1/tanh while the Letter's Eq. (10) has
-1/tanh, so the sign convention of I differs between the two papers; the
magnitudes agree exactly and the physical requirement 1 + Z > 0 fixes the sign.

THE 1/E' IS THE GAMMA3 ENERGY, sqrt(xi'^2 + gamma3 Delta'^2). Eq. (11)'s
denominator tanh[(beta/2)E_k'] and Eq. (1)'s tanh(beta E'/2)/E' cancel whatever
they are, so the only thing that survives is 1/E' -- and the direct Matsubara
route says unambiguously that it is the gamma3 one, to 1e-10 across omega_ph,
lambda, T and gamma3. Using the bare sqrt(xi'^2 + Delta'^2) there is a 5.6%
error at gamma3 = 1.33 and 14.5% at gamma3 = 1.95. It is invisible at Delta = 0,
where both reduce to |xi'|, which is why no Tc test can see it.

The Coulomb term keeps the BARE E', and that is not an inconsistency: it comes
from the anomalous density chi_k = Delta_sk tanh(beta E_k/2)/(2 E_k), built from
the KS propagator F_s, which the Letter defines with the bare E_k. gamma3 enters
only through the ansatz Eq. (9), i.e. only the phonon term.

THE ENERGY WINDOW IS NOT A DETAIL -- IT WAS THE Fig. 2 DISCREPANCY. Both kernels
have power-law tails in xi', so their integrals converge only as 1/L in the
window L = `cutoff`. The old default L = 30 omega_max was nowhere near
converged, and the error is lambda-dependent: on the Fig. 2 model it raised Tc
by 3.0% at lambda = 0.4 and 6.4% at lambda = 2.8. That, and not a missing term,
was the monotonic Tc drift against the published Fig. 2. The default is now
400 omega_max (~0.3% from the L -> infinity limit); the remaining error is clean
1/L, so 2*Tc(2L) - Tc(L) extrapolates it away. Z(0)/lambda converges to 1.838,
not the 1.773 the 30 omega_max grid gave.

For a real material the window is physics, not convergence: LM2005 (Sec. IV)
points out that the tails are an artefact of replacing g_{k,k'} by its Fermi-
surface value, and that the true |g|^2 decays away from E_F and cuts them. Set
`cutoff` to the band structure's scale there. In the Einstein model, where
|g|^2 is constant by construction, L is a parameter of the model.

BEWARE THE DEFAULT WHEN COMPARING TWO PHONON ENERGIES. Being a multiple of
omega_max, it makes L different for each -- 8 eV at omega_log = 20 meV against
32 eV at 80 meV -- which is the same trap `mu_cutoff` was introduced to close
for the Coulomb term. Any comparison ACROSS omega_ph must pass `cutoff`
explicitly and identically. There is no better default available: omega_max is
the only scale the inputs carry.

Notebook 20 now pins both windows to its model's band edge, `cutoff = mu_cutoff
= E_FERMI`. It was never affected by the old default, which is worth knowing
because it narrows a separate open item: `_prepare` raises `cutoff` to at least
`mu_cutoff`, and 30 omega_max was below 5 eV for both its phonon energies, so
BOTH families always ran at exactly 5 eV. Its numbers are unchanged to every
digit printed. Therefore the residual it reports against McMillan at identical
mu = 0.164 over +-5 eV,

    omega_log = 80 meV:  lambda 1.0 -> 0.913,  1.3 -> 0.911,  2.0 -> 0.965
    omega_log = 20 meV:  lambda 1.0 -> 0.969,  1.3 -> 0.952,  2.0 -> 0.991

is NOT a window artefact and remains open. Note the two families are not on the
same footing for a different reason: the notebook quotes mu* at a fixed 0.1 eV
for both, while McMillan wants mu* at the phonon scale -- fine at 80 meV,
inconsistent at 20 meV. Settle that before reading the spread as physics.

AGAINST THE PUBLISHED Fig. 2 (digitised into tests/data/scdft/). With the
window converged, Tc_SCDFT/Tc_Eliashberg has the paper's shape, which it did not
before: 1.31 at lambda = 0.4 (paper 1.33), crossing below 1 near lambda = 1
(paper 1.2), a minimum of 0.959 (paper 0.976), then a return toward 1. Our
Eliashberg Tc matches theirs to 0.3-0.8% with no lambda trend, so the reference
is sound.

THE RESIDUAL 1-3% IS ON THE PAPER'S SIDE, NOT HERE (settled 2026-07-31). An
earlier note here said an unknown ~3 eV window absorbs it -- REFUTED: the raw
data behind the Letter's Fig. 1 (pdf/fig1.agr, from Antonio) runs on a log grid
out to +-27 eV, i.e. the authors' own window matches our converged one. At
matched windows the published SCDFT Tc stays 2-3.5% above these equations.
Three facts pin the blame:

1. This implementation now matches SCTK (github.com/mitsuaki1987/sctk,
   scdft_kernel = 2) -- a third, fully independent implementation whose kernels
   are symbolic Matsubara closed forms -- to 1e-10 pointwise at finite T and at
   T = 0 (tests test_*_matches_sctk_closed_form). Three independent derivations
   agree on what Eqs. (10)-(11) are; any correct solver must get our numbers.
2. The constants are EXACTLY the printed ones, so they cannot absorb the
   residual. A. Sanna confirmed (2026-07-31, via Miguel) that his code uses
   gamm = 1.330, 3.800, 1.33000, and that "gamm(3) only affects the value of the
   gap. Tc is set just by gamm(1) and gamm(2)". A previous version of this note
   proposed that the printed 3.8 was a rounding of SCTK's 3.88 and that this
   explained part of the gap -- WRONG, and retracted: 3.88 is SCTK's own choice.
   The gamma3 remark is also a structural check this module passes, since
   `linearized_eigenvalue` takes Delta_s -> 0, where E' = |xi'| and gamma3 drops
   out of Tc identically.
3. The authors' two publications disagree with EACH OTHER by the same margin:
   at lambda = 2, mu = 0, Tc_SCDFT/Tc_Eliashberg is 0.993 in PRL 125 (60 meV,
   27 eV window) but 0.979 in the Nature Rev. Phys. 6, 570 (2024) SI Fig. 2
   model (20 meV, 5 eV window), while the window physics says the FIRST should
   be the lower one (we get 0.963 and 0.965 -- nearly equal, as they must be).
   Their published curves carry ~1% internal wiggle on top.

WHAT IS LEFT UNEXPLAINED, stated plainly. With the constants pinned exactly and
the kernels certified three ways, waw's Tc sits 1.5-2.8% BELOW the published
Fig. 2 at matched windows, and nothing in this module accounts for it. The
remaining candidates are all on the far side of the printed equations: their
electronic window for Fig. 2 specifically (fig1.agr fixes Fig. 1's at +-27 eV,
not necessarily Fig. 2's), their Eliashberg reference, or their numerics. Fitting
it would need gamma1 ~ 1.34-1.35, which the author's own values exclude, so do
NOT tune the constants to close it.

Against the review's model (E_F = 5 eV, digitised from its SI Fig. 2 vector
data; the mu* = 0 family, where nothing is ambiguous): waw with gamma2 = 3.88
is 1.0-2.1% below their SCDFT curve at lambda >= 1, the same offset as against
the PRL. Their McMillan mu* = 0 curve matches the analytic formula to 0.4%,
which certifies the digitisation. NB their mu* = 0.1 families quote mu* at the
PHONON scale (mu*(omega_log), McMillan's convention): mapping it that way
reproduces both phonon energies to 1-2%, while a fixed 0.1 eV reference is off
by 5-10% at 20 meV -- that was notebook 20's old "omega_ph bend".

*** Fig. 1 IS NOT A TEST OF THIS FUNCTIONAL. *** Its curves cannot be
reproduced by Eqs. (1), (10), (11), and chasing them is what sent an earlier
hunt after a nonexistent bug in Z. Two independent reasons:

1. The Letter's own text says Fig. 1 is "the numerically exact Delta_s(xi) for
   the model", i.e. Eq. (5) with the true interacting Eliashberg Green's
   functions. gamma3 was then fitted to approximate it. It is the target, not
   the output.

2. Its central feature -- Delta_s(E_F) -> 0 as T -> 0 -- is structurally absent
   from the PARTIALLY LINEARIZED equation that Eqs. (1), (10), (11) are (the
   Supplemental Material says so: the xi_k side is taken at Delta_k -> 0, which
   is why those kernels depend on xi_k alone). At xi -> 0 the two routes give

       full     D(1 + lambda) + (lambda/beta) sinh(beta D) = S
       partial  D(1 + 2 lambda)                            = S

   The exponential sinh is what pins beta*D to a constant, i.e. D ~ kT -> 0;
   partial linearization deletes it, leaving D finite. Solved numerically with
   the direct Matsubara route: full gives D = 7.8, 5.5, 3.5, 2.1 meV at
   T/Tc = 0.5, 0.25, 0.125, 0.0625 (vanishing), partial gives 11.4, 12.1, 12.2,
   12.2 meV (flat). Note the same deleted term is exactly the LM2005 factor of
   two in Z: (1 + lambda) is McMillan's, (1 + 2 lambda) is Z^{ph,sym}'s. So
   Z(0)/lambda ~ 1.8 is a CONSEQUENCE of the partial linearization, not a bug --
   which is what LM2005 Eq. (78) reports and never explains.

   For the record, Fig. 1's Delta_s(E_F) is exactly 2 kT artanh(2 chi_F): the
   digitised curve gives chi_F = 0.2365 constant to 4 digits below T/Tc = 0.3.
   That is just chi_k = Delta_sk tanh(beta E_k/2)/(2 E_k) at xi = 0, so it says
   nothing about the kernels either.

WEAK COUPLING IS THE FUNCTIONAL'S OWN BEHAVIOUR. The Supplemental Material says
so directly: "relative deviations from the Eliashberg results are bigger in the
very weak coupling limit. In this regime our functional slightly overestimates
Tc". gamma3 was fitted over 0.4 < lambda < 3.0. Ours overshoots by the same
amount in the same direction (1.31 vs their 1.33 at lambda = 0.4).

The SM also confirms the isotropic normalisation used here. Its Eq. (1) writes
|g|^2 = lambda*omega_ph/(2 N_F) and sum_k' -> N_F integral dxi'; the N_F
cancels to leave (lambda*omega_ph/2) integral dxi', which is exactly
alpha^2F = lambda*omega_ph/2 * delta(omega - omega_ph) fed through
`_omega_nodes`.

COULOMB: mu AND mu* ARE DIFFERENT INPUTS. mu is the unrenormalized
pseudopotential over the full band and is what SCDFT takes, because SCDFT keeps
the whole energy dependence. Eliashberg truncates at omega_c and needs the
Morel-Anderson mu*(omega_c) = mu / [1 + mu ln(E_F/omega_c)]. The Letter's own
comparison uses mu = 0.223 and 0.394 over +-10 eV, giving mu* = 0.11 and 0.14 at
a 0.1 eV cutoff. Handing mu* to this solver understates the repulsion and raises
Tc.

COULOMB CUTOFF. mu is meaningful only together with a window, exactly as mu* is
only meaningful together with omega_c, so the kernel is the windowed constant

    W_ij(xi, xi') = mu_ij Theta(L - |xi|) Theta(L - |xi'|),   L = `mu_cutoff`

and both Heaviside factors matter: the one on xi' cuts the log-divergent
integral, the one on xi kills the repulsion outside the window (which is what
leaves the two-square-well structure). This is a separate window from `cutoff`;
before it existed, mu silently meant something different at every phonon energy.

Two things that are NOT bugs -- stop rediscovering them:

* Z(xi->0)/lambda ~ 1.8. Explained above: it is the partial linearization, and
  LM2005 Eq. (78) reports the same value.
* The diagonal Coulomb "B-term" of LM2005 Eqs. (33)-(35) is ABSENT ON PURPOSE
  and must stay absent. Per Antonio (via Miguel, 2026-07-29) it was removed
  because it gave nonsensical physics. The Coulomb term is otherwise UNCHANGED
  since LM2005, so that paper is a valid reference for K^ee -- but not for the
  B-term. A constant mu in the off-diagonal gap kernel with nothing diagonal is
  the correct structure.

STILL OPEN:

1. The PHYSICAL gap functional (Letter Eq. 14, review Eqs. 10-14: Delta_n from
   S11, S12 and J12, with the gap value 1.95 replacing gamma3) is fully
   specified and is the right thing to compare with experiment. Not implemented.

2. K^ee is a constant mu here, not the static-RPA screened interaction W_kk'.

3. `solve_delta_s` is the partially linearized equation, which is what the
   Letter recommends ("solving the fully non-linear gap equation ... requires a
   much higher computational cost, without (slightly) improving the accuracy").
   Its Delta_s therefore does NOT vanish at E_F, by construction -- see above.
   The nonlinear iteration is plain linear mixing and shows the critical slowing
   down near Tc that `eliashberg.nonlinear` solves with Anderson acceleration;
   porting that across is the obvious next step.

CHECKED AGAINST THE REVIEW. The supplementary information of Nature Rev. Phys.
6, 570 (2024) reprints these kernels (its Eqs. 3-7) and they are identical, term
for term, to what is implemented here, with gamma1 = gamma3 = 1.33 and
gamma2 = 3.8. It also settles two questions this module used to carry:

* Delta_s is NOT expected to resemble the measured gap. "In SCDFT the KS and
  experimental gaps exhibit very different behaviors" -- the physical gap is a
  separate functional OF Delta_s (review Eqs. 10-14), so the shape of Delta_s
  around E_F is not evidence of an error.
* gamma4 = 1.95 belongs to that gap functional, not to Eq. (5).
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np

from waw.units import K_B_HARTREE

from ..eliashberg.kernels import as_band_matrix
from .functions import di_dxi, i_function, j_function, i_akashi, j_akashi, j_lueders

__all__ = ["SCDFT_GAMMAS_KERNEL", "SCDFT_GAMMAS_TC", "SCDFT_GAMMA4_GAP",
           "ScdftResult", "energy_grid",
           "z_kernel", "gap_operator", "tc_scdft", "solve_delta_s"]

#: (gamma1, gamma2, gamma3) of the xc kernels. Sanna-Pellegrini-Gross give
#: gamma1 = gamma3 = 1.33, gamma2 = 3.8, confirmed in the supplementary
#: information of Nature Rev. Phys. 6, 570 (2024) (its Eqs. 4-7).
SCDFT_GAMMAS_KERNEL = (1.33, 3.8, 1.33)
SCDFT_GAMMAS_TC = SCDFT_GAMMAS_KERNEL            # backwards-compatible alias
#: What SCTK uses (src/sctk_kernel_weight.f90): gamma2 = 3.88, not the printed
#: 3.8. This is SCTK's OWN value and does not come from the functional's authors:
#: A. Sanna confirmed (2026-07-31, via Miguel) that his code runs the printed
#: constants exactly -- gamm = 1.330, 3.800, 1.33000 -- so 3.8 is NOT a rounding
#: of 3.88 and this tuple is not a better estimate of the published fit. Kept
#: only because it is the right choice when reproducing SCTK specifically, and
#: because the 0.3-0.7% Tc shift it produces measures the sensitivity to gamma2.
SCDFT_GAMMAS_SCTK = (1.33, 3.88, 1.33)
#: gamma4 = 1.95 belongs to the PHYSICAL-gap functional, not to the kernels: it
#: enters E^gamma4 = sqrt(xi^2 + gamma4^2 Delta_s^2) inside that functional
#: (review Eqs. 10-14), which is a separate object from Eq. (5)'s gamma3.
SCDFT_GAMMA4_GAP = 1.95


@dataclass
class ScdftResult:
    """Kohn-Sham pairing potential on the energy grid."""
    xi:          np.ndarray    # (n_xi,) Hartree, relative to E_F, excludes 0
    weights:     np.ndarray    # (n_xi,) quadrature weights for integral dxi
    delta_s:     np.ndarray    # (n_xi, nb) Hartree
    z:           np.ndarray    # (n_xi, nb) the Z kernel
    temperature: float         # Kelvin
    converged:   bool
    n_iterations: int

    @property
    def delta_s_max(self) -> np.ndarray:
        """max |Delta_s| per band -- the peak of the potential, not a gap."""
        return np.max(np.abs(self.delta_s), axis=0)

    def delta_s_at_fermi(self) -> np.ndarray:
        """Delta_s interpolated to xi = 0. The paper's exact result is that this
        VANISHES as T -> 0, which is why Delta_s must not be read as a gap."""
        out = []
        for j in range(self.delta_s.shape[1]):
            out.append(np.interp(0.0, self.xi, self.delta_s[:, j]))
        return np.array(out)


def energy_grid(kT: float, cutoff: float, n_points: int = 240):
    """
    Symmetric grid in xi with quadrature weights, dense near E_F.

    Half the points sit linearly within +-20 kT, where the tanh and Fermi
    factors vary, and half run logarithmically out to ``cutoff``. xi = 0 is
    excluded: Z and K^ph both carry 1/tanh(beta xi/2), which diverges there
    (the numerators vanish, but there is no reason to evaluate 0/0).
    """
    n_half = max(n_points // 4, 8)
    inner = np.linspace(0.02 * kT, 20.0 * kT, n_half)
    outer = np.geomspace(21.0 * kT, cutoff, n_points // 2 - n_half)
    pos = np.concatenate([inner, outer])
    xi = np.concatenate([-pos[::-1], pos])
    # trapezoid weights on the (non-uniform) grid
    w = np.zeros_like(xi)
    w[1:-1] = 0.5 * (xi[2:] - xi[:-2])
    w[0] = 0.5 * (xi[1] - xi[0])
    w[-1] = 0.5 * (xi[-1] - xi[-2])
    return xi, w


def _omega_nodes(omega, a2f):
    """
    Quadrature nodes for integral domega alpha^2F_ij(omega) (...).

    A single frequency is taken as an Einstein mode whose ``a2f`` is the already
    integrated weight lambda*omega/2, so ``modes`` and continuous spectra go
    through the same code path.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    a = as_band_matrix(a2f, len(omega))
    if len(omega) == 1:
        return omega, a
    dw = np.zeros_like(omega)
    dw[1:-1] = 0.5 * (omega[2:] - omega[:-2])
    dw[0] = 0.5 * (omega[1] - omega[0])
    dw[-1] = 0.5 * (omega[-1] - omega[-2])
    return omega, a * dw[None, None, :]


def z_kernel(xi, w_xi, omega, a2f_w, beta: float, functional: str = "spg",
             dos_ratio=None):
    """
    The renormalisation kernel, reduced to energies.

    ``dos_ratio`` is N(xi')/N(0) on the same grid as ``xi``, the DOS weight that
    appears in the xi' integral of Luders' Eq. (79) and of Akashi-Arita
    Eq. (40). ``None`` means a constant DOS (all ones), which is what the
    symmetrised functionals assume anyway and reproduces this function's
    behaviour before the weight existed.

    ``functional="akashi"`` is Akashi and Arita, Phys. Rev. B 88, 014514 (2013),
    their Eq. (40),

        Z^ph,new(xi) = 1/tanh(beta xi/2) int dw a2F(w)
                       int dxi' [N(xi')/N(0)] [I(xi,xi',w) - 2 J(xi,xi',w)]

    with I and J their Eqs. (41)-(44) (`functions.i_akashi`, `j_akashi`). This
    is the UNSYMMETRISED kernel: they prove the divergence that forced Luders et
    al. to symmetrise cancels between I and 2J, so the antisymmetric part of the
    DOS survives. Because `i_akashi` is odd in xi', the I term contributes
    NOTHING for a particle-hole symmetric DOS -- pass a `dos_ratio` with an
    antisymmetric component or this functional differs from `lm2005` only in the
    smoothing.

    SIGN CAVEAT, flagged rather than silently patched. Their Eq. (40) carries
    ``+1/tanh`` while the previous kernel [their Eq. (39), Luders Eq. (79)]
    carries ``-1/tanh``; and their property (iii) states the xi'-symmetric part
    of ``[I - 2J]`` agrees with ``J_Luders(xi,xi') + J_Luders(xi,-xi')``. Taken
    together those give Z^new = -Z^ph in the symmetric limit, which contradicts
    their own property (i), ``lim_{xi->0} Z^new = lim Z^ph = lambda``. One of the
    three statements carries a sign slip in the published paper. This
    implementation follows Eq. (40) VERBATIM; `tests/test_utils_scdft_akashi.py`
    measures which of (i) and (iii) then hold, and `sign` below lets a caller
    flip the prefactor to test the alternative reading.

    ``functional="spg"`` (default) is Sanna-Pellegrini-Gross Eq. (10),

        Z_i(xi) = -[1/tanh(beta xi/2)] sum_j integral dxi' domega
                   alpha^2F_ij(omega) d/dxi [I(xi, xi', w) + I(xi, -xi', w)]

    which is the SYMMETRISED kernel of Luders et al., their Eq. (78). The
    xi-derivative uses the closed form `functions.di_dxi`, so nothing here is
    approximated that the paper gives exactly.

    ``functional="lm2005"`` is instead Luders Eq. (79), with ``I'`` replaced by
    the ``J`` of their Eqs. (80)-(81). Luders et al. report that Eq. (78) gives
    Z(0) ~ 2*lambda -- "twice the value expected from the comparison to
    McMillan's formula" -- and Eq. (79) is their repair of it, keeping only the
    broad, weakly temperature-dependent part. SPG did NOT adopt that repair:
    they state they use "the symmetric approximation proposed in Ref. [10]",
    i.e. Eq. (78). So the default here carries Z(0) = 2*lambda and
    ``lm2005`` carries Z(0) ~ lambda; this is a real difference between the two
    published functionals, not a choice of implementation.
    """
    if functional not in ("spg", "lm2005", "akashi"):
        raise ValueError("functional must be 'spg', 'lm2005' or 'akashi', got "
                         f"{functional!r}")
    nb = a2f_w.shape[0]
    if dos_ratio is None:
        wq = np.asarray(w_xi, dtype=np.float64)
    else:
        dos_ratio = np.asarray(dos_ratio, dtype=np.float64)
        if dos_ratio.shape != np.shape(xi):
            raise ValueError(
                f"z_kernel: dos_ratio must match xi {np.shape(xi)}; got "
                f"{dos_ratio.shape}")
        wq = np.asarray(w_xi, dtype=np.float64) * dos_ratio
    acc = np.zeros((len(xi), nb))
    for k, wk in enumerate(omega):
        if functional == "spg":
            # d/dxi of the symmetrised bracket, in closed form: (n_xi, n_xi')
            block = (di_dxi(xi[:, None], xi[None, :], wk, beta)
                     + di_dxi(xi[:, None], -xi[None, :], wk, beta))
        elif functional == "lm2005":
            block = (j_lueders(xi[:, None], xi[None, :], wk, beta)
                     + j_lueders(xi[:, None], -xi[None, :], wk, beta))
        else:
            block = (i_akashi(xi[:, None], xi[None, :], wk, beta)
                     - 2.0 * j_akashi(xi[:, None], xi[None, :], wk, beta))
        contrib = block @ wq                                 # integral dxi'
        for i in range(nb):
            acc[:, i] += float(a2f_w[i, :, k].sum()) * contrib
    # Eq. (40) carries +1/tanh; Eqs. (78)/(79) carry -1/tanh. See the SIGN
    # CAVEAT above -- this is the paper's own asymmetry, not a choice here.
    sign = 1.0 if functional == "akashi" else -1.0
    return sign * acc / np.tanh(0.5 * beta * xi)[:, None]


def gap_operator(xi, w_xi, omega, a2f_w, mu, beta: float, delta_s,
                 gammas=SCDFT_GAMMAS_KERNEL, mu_cutoff: float | None = None,
                 functional: str = "spg"):
    """
    The right-hand side operator of their Eq. (1), excluding -Z*Delta.

    Returns ``M`` of shape ``(n_xi, nb, n_xi, nb)`` such that the phonon plus
    Coulomb contribution is ``einsum('xiyj,yj->xi', M, delta_s)``. ``delta_s``
    enters only through E' = sqrt(xi'^2 + gamma3 Delta'^2) in K^ph and through
    E' in the tanh factor of the Coulomb term, so passing zeros gives the
    LINEARIZED operator.

    The Coulomb kernel is the windowed constant

        W_ij(xi, xi') = mu_ij Theta(L - |xi|) Theta(L - |xi'|),   L = mu_cutoff

    so mu is only meaningful together with L, exactly as mu* is only meaningful
    together with omega_c. Both Heaviside factors matter: the one on xi' cuts
    the log-divergent xi' integral, and the one on xi kills the repulsion
    outside the window, which is what leaves the two-square-well structure.

    The xi' quadrature must SPAN the window: `_prepare` grows the grid to L when
    L exceeds it. Closing the tail analytically instead -- tanh -> 1, E' = |xi'|,
    so integral_{X<|xi'|<L} dxi' Delta'/|xi'| = ln(L/X)[Delta(X) + Delta(-X)] --
    looks safe but is NOT: Delta_s is not flat there. The Coulomb term is indeed
    xi-independent inside the window (a scalar times Theta), but
    Delta_s = [phonon + Coulomb]/(1 + z), and z^ph decays slowly, so Delta_s
    only reaches its plateau logarithmically. Measured on an Einstein mode at
    L = 5 eV, that tail form drifts Tc by +2.6% at X = 30 w0 and +4.8% at
    X = 10 w0, monotonically toward the spanned-grid answer.
    """
    g1, g2, g3 = gammas
    nb = a2f_w.shape[0]
    n = len(xi)
    delta_s = np.zeros((n, nb)) if delta_s is None else np.asarray(delta_s)

    e_g3 = np.sqrt(xi[:, None] ** 2 + g3 * delta_s ** 2)        # (n_xi', nb)
    e_bare = np.sqrt(xi[:, None] ** 2 + delta_s ** 2)           # for the tanh
    coth = 1.0 / np.tanh(0.5 * beta * xi)                       # (n_xi,)

    if functional not in ("spg", "lm2005", "akashi"):
        raise ValueError("functional must be 'spg', 'lm2005' or 'akashi', got "
                         f"{functional!r}")
    # Akashi and Arita propose a new Z ONLY; for the pairing kernel their model
    # calculation uses "the nk-averaged form for K^ph [Eq. (23) in Ref. GrossII]",
    # i.e. the Gross/Luders one. So "akashi" = new Z + LM2005 K^ph, which is what
    # makes it a controlled change against "lm2005": the two differ in Z alone.
    if functional == "akashi":
        functional = "lm2005"

    M = np.zeros((n, nb, n, nb))

    if functional == "lm2005":
        # K^ph of Luders et al., their Eq. (74):
        #   K^ph_ij = 2/(tanh(beta xi_i/2) tanh(beta E_j/2)) sum |g|^2
        #             [I(xi_i, xi_j, W) - I(xi_i, -xi_j, W)]
        # The -1/2 cancels the 2, leaving
        #     -coth(beta xi/2) [I - I] * tanh(beta E'/2)/(E' tanh(beta xi'/2)).
        # The tanh(beta xi'/2) in Eq. (74) must NOT be cancelled against Eq. (1)'s
        # tanh(beta E'/2)/E' the way SPG's can: [I(xi,xi') - I(xi,-xi')] is ODD in
        # xi', and so is 1/tanh(beta xi'/2), so their ratio is even -- which is
        # what an even gap function needs. Dropping the 1/tanh leaves an operator
        # odd in xi', which annihilates every even Delta and gives rho == 0
        # identically. As Delta -> 0 the surviving factor is the SIGNED 1/xi',
        # not 1/|xi'|.
        #
        # There is also no factor omega here: Eq. (74) sums |g|^2, unlike SPG's
        # Eq. (11) which sums omega|g|^2. Luders derived Eq. (74) for the
        # LINEARIZED gap equation; keeping tanh(beta E'/2)/E' for finite Delta is
        # the usual way to reuse it and reduces to their form as Delta -> 0.
        tanh_xi = np.tanh(0.5 * beta * xi)
        for k, wk in enumerate(omega):
            blk = (i_function(xi[:, None], xi[None, :], wk, beta)
                   - i_function(xi[:, None], -xi[None, :], wk, beta))
            for i in range(nb):
                for j in range(nb):
                    a = float(a2f_w[i, j, k])
                    if a == 0.0:
                        continue
                    fac = (w_xi * np.tanh(0.5 * beta * e_bare[:, j])
                           / (e_bare[:, j] * tanh_xi))
                    M[:, i, :, j] += (-a) * (coth[:, None] * blk * fac[None, :])
        return _add_coulomb(M, xi, w_xi, mu, beta, e_bare, nb, mu_cutoff)

    # --- phonon part: the tanh(beta E'/2) of K^ph cancels Eq. (1)'s, leaving
    #     -(g1 g2/2) coth(beta xi/2) integral dxi' (Delta'/E') integral dw w a2F S
    for k, wk in enumerate(omega):
        s_sum = np.zeros((n, n, nb))
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                for s3 in (1.0, -1.0):
                    sign = s1 * s2 * s3
                    for j in range(nb):
                        s_sum[:, :, j] += sign * j_function(
                            xi[:, None], s1 * e_g3[None, :, j],
                            s2 * wk, s3 * g2 * wk, beta)
        for i in range(nb):
            for j in range(nb):
                a = float(a2f_w[i, j, k]) * wk
                if a == 0.0:
                    continue
                # 1/E' is the GAMMA3 energy sqrt(xi'^2 + gamma3 Delta'^2), the
                # same one J's second argument carries. Eq. (11)'s denominator
                # tanh[(beta/2)E_k'] and Eq. (1)'s tanh(beta E'/2)/E' cancel
                # whatever they are, so all that survives is 1/E', and the
                # direct Matsubara evaluation of Eq. (5)-(9) says that E' is the
                # gamma3 one to 1e-10 (tests/test_utils_scdft.py). Using the
                # bare E' there is a 5.6% error at gamma3 = 1.33, invisible in
                # the linearized limit where both reduce to |xi'|.
                M[:, i, :, j] += (-0.5 * g1 * g2 * a) * (
                    coth[:, None] * s_sum[:, :, j] * (w_xi / e_g3[:, j])[None, :])

    return _add_coulomb(M, xi, w_xi, mu, beta, e_bare, nb, mu_cutoff)


def _add_coulomb(M, xi, w_xi, mu, beta, e_bare, nb, mu_cutoff):
    """
    -(1/2) integral dxi' W_ij tanh(beta E'/2)/E' Delta', added in place.

    Shared by every functional: LM2005 and SPG both take the Coulomb kernel to
    be the statically screened interaction, with no mass renormalisation.
    """
    lam_c = float(np.max(np.abs(xi))) if mu_cutoff is None else float(mu_cutoff)
    inside = np.abs(xi) <= lam_c                                 # Theta(L - |xi|)
    tanh_over_e = np.tanh(0.5 * beta * e_bare) / e_bare          # (n_xi', nb)
    quad = (w_xi * inside)[:, None] * tanh_over_e                # cut at L in xi'
    for i in range(nb):
        for j in range(nb):
            if mu[i, j] == 0.0:
                continue
            M[:, i, :, j] += (-0.5 * mu[i, j]) * inside[:, None] * quad[None, :, j]
    return M


def _leading_eigenvalue(op, n, nb):
    """Largest real eigenvalue of the (n*nb) square gap operator."""
    m = op.reshape(n * nb, n * nb)
    vals, vecs = np.linalg.eig(m)
    k = int(np.argmax(vals.real))
    vec = vecs[:, k].real.reshape(n, nb)
    if vec.size and vec.flat[np.argmax(np.abs(vec))] < 0:
        vec = -vec
    return float(vals.real[k]), vec


def _prepare(omega, a2f, mu, kT, cutoff, n_points, mu_cutoff=None):
    om, a2f_w = _omega_nodes(omega, a2f)
    nb = a2f_w.shape[0]
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.full((nb, nb), float(mu)) if mu.ndim == 0 else mu
    if mu.shape != (nb, nb):
        raise ValueError(f"mu must be a scalar or ({nb}, {nb}); got {mu.shape}")
    if cutoff is None:
        # Both kernels have power-law tails in xi', so the xi' integral converges
        # only as 1/L. 30 omega_max -- the old default -- overestimates Tc by
        # 3.0% at lambda = 0.4 and 6.4% at lambda = 2.8 on the Fig. 2 model, a
        # lambda-dependent error that reads as a physical drift. 400 sits ~0.3%
        # from the limit; the residual is clean 1/L, so 2*Tc(2L) - Tc(L)
        # extrapolates it. For a real material set `cutoff` to the band scale
        # instead: the tails are an artefact of a Fermi-surface-averaged g.
        cutoff = 400.0 * float(np.max(om))
    if mu_cutoff is not None:
        cutoff = max(cutoff, float(mu_cutoff))   # the xi' grid must span W
    if mu_cutoff is not None and float(mu_cutoff) <= 20.0 * kT:
        warnings.warn(
            f"mu_cutoff = {mu_cutoff} is inside the thermal window "
            f"(20 kT = {20.0 * kT}): the Coulomb window is narrower than the "
            f"thermal smearing of the Fermi surface. Legitimate, but check that "
            f"enough grid points fall inside it.", RuntimeWarning, stacklevel=3)
    xi, w_xi = energy_grid(kT, cutoff, n_points)
    return om, a2f_w, mu, xi, w_xi, nb


def linearized_eigenvalue(omega, a2f, mu, kT: float, *, cutoff: float | None = None,
                          n_points: int = 200, gammas=SCDFT_GAMMAS_KERNEL,
                          mu_cutoff: float | None = None,
                          functional: str = "spg", dos_ratio=None):
    """
    Leading eigenvalue of the SCDFT gap equation linearized in Delta_s.

    Setting Delta_s = 0 in the kernels makes their Eq. (1) linear, so Tc is
    where this crosses 1 -- the same device as `eliashberg.linearized`.

    ``dos_ratio`` is a CALLABLE ``N(xi)/N(0)``, not an array: the xi grid depends
    on T and on the cutoff, so it is built inside and the ratio evaluated on it.
    ``None`` means a constant DOS.

    WHERE THE DOS WEIGHT IS AND IS NOT APPLIED. It enters the Z kernel only,
    which is exactly the scope of Akashi and Arita's Eq. (40). The pairing and
    Coulomb xi' integrals keep this module's standing convention, in which the
    DOS of the final band is already absorbed into alpha^2F_ij and into the
    dimensionless mu. So `dos_ratio` isolates the effect that paper is about and
    nothing else. For a STRONGLY asymmetric system that is incomplete -- the
    same N(xi')/N(0) belongs in every xi' integral -- and this should be revisited
    before trusting it on, say, a doped semiconductor.
    """
    om, a2f_w, mu_m, xi, w_xi, nb = _prepare(omega, a2f, mu, kT, cutoff, n_points,
                                             mu_cutoff)
    beta = 1.0 / kT
    dr = None if dos_ratio is None else np.asarray(dos_ratio(xi), dtype=np.float64)
    z = z_kernel(xi, w_xi, om, a2f_w, beta, functional, dos_ratio=dr)
    M = gap_operator(xi, w_xi, om, a2f_w, mu_m, beta, None, gammas, mu_cutoff,
                     functional)
    op = M / (1.0 + z)[:, :, None, None]
    return _leading_eigenvalue(op, len(xi), nb) + (xi, z)


def tc_scdft(omega, a2f, mu, *, cutoff: float | None = None, n_points: int = 200,
             gammas=SCDFT_GAMMAS_KERNEL, t_min: float = 0.2, t_max: float = 600.0,
             tol: float = 1e-3, mu_cutoff: float | None = None,
             functional: str = "spg", dos_ratio=None):
    """
    Tc (Kelvin) from the linearized SCDFT gap equation.

    Bisects the leading eigenvalue of the linearized operator through 1.
    ``dos_ratio``: see `linearized_eigenvalue`.
    """
    def rho(t):
        return linearized_eigenvalue(omega, a2f, mu, t * K_B_HARTREE,
                                     cutoff=cutoff, n_points=n_points,
                                     gammas=gammas, mu_cutoff=mu_cutoff,
                                     functional=functional,
                                     dos_ratio=dos_ratio)[0]
    lo, hi = t_min, t_max
    if rho(lo) < 1.0:
        return 0.0
    if rho(hi) > 1.0:
        raise RuntimeError(f"rho > 1 at t_max = {t_max} K; raise t_max")
    while (hi - lo) > tol * max(hi, 1.0):
        mid = 0.5 * (lo + hi)
        if rho(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_delta_s(omega, a2f, mu, kT: float, *, cutoff: float | None = None,
                  n_points: int = 200, gammas=SCDFT_GAMMAS_KERNEL,
                  delta_init: float | None = None, mixing: float = 0.4,
                  tol: float = 1e-8, max_iter: int = 500,
                  mu_cutoff: float | None = None,
                  functional: str = "spg") -> ScdftResult:
    """
    Self-consistent (nonlinear) solution of their Eq. (1) for Delta_s(xi).

    Delta_s enters the kernels only through E' = sqrt(xi'^2 + gamma3 Delta'^2),
    so each iteration rebuilds the operator at the current Delta_s.

    NOTE Delta_s is the Kohn-Sham PAIRING POTENTIAL, not the gap. The paper's
    central exact result is that it has a local minimum at E_F and vanishes
    there as T -> 0; the physical gap comes from their Eq. (14) instead.
    """
    om, a2f_w, mu_m, xi, w_xi, nb = _prepare(omega, a2f, mu, kT, cutoff, n_points,
                                             mu_cutoff)
    beta = 1.0 / kT
    z = z_kernel(xi, w_xi, om, a2f_w, beta, functional)
    scale = delta_init if delta_init is not None else 2.0 * kT
    delta = np.full((len(xi), nb), float(scale))

    converged, it = False, 0
    for it in range(1, max_iter + 1):
        M = gap_operator(xi, w_xi, om, a2f_w, mu_m, beta, delta, gammas, mu_cutoff,
                         functional)
        new = np.einsum("xiyj,yj->xi", M, delta) / (1.0 + z)
        ref = max(float(np.max(np.abs(delta))), float(np.max(np.abs(new))))
        if ref == 0.0:
            converged, delta = True, np.zeros_like(delta)
            break
        res = float(np.max(np.abs(new - delta))) / ref
        delta = mixing * new + (1.0 - mixing) * delta
        if res < tol:
            converged = True
            break
    return ScdftResult(xi=xi, weights=w_xi, delta_s=delta, z=z,
                       temperature=kT / K_B_HARTREE, converged=converged,
                       n_iterations=it)
