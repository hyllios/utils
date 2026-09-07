"""
Electron-phonon-limited electrical conductivity from the isotropic,
BAND-RESOLVED Boltzmann equation -- Allen's lowest-order variational
approximation (LOVA) generalised to several Fermi-surface sheets.

  P. B. Allen, Phys. Rev. B 17, 3725 (1978); G. Grimvall, "The
  Electron-Phonon Interaction in Metals" (1981), Ch. 7; J. M. Ziman,
  "Electrons and Phonons" (1960), Ch. 7 for the variational principle.

WHAT THIS SOLVES. The linearised Boltzmann equation in the degenerate,
equilibrium-phonon limit, restricted to trial deviation functions that are
isotropic on each sheet and proportional to the velocity,

    phi_k = Phi_i v_a(k)        for k on sheet i, field along a,

leaving one unknown Phi_i per sheet. Substituting into the variational
resistivity functional and minimising gives a linear system in sheet space,

    sum_j M_a,ij(T) Phi_j = D_a,i,      sigma_a = (e^2/Omega) sum_i D_a,i Phi_i

    M_a,ij(T) = delta_ij sum_l A_a,il(T) - C_a,ij(T)
    A_a,ij(T) = integral dw F^out_a,ij(w) K(w, T)
    C_a,ij(T) = integral dw F^in_a,ij(w)  K(w, T)
    K(w, T)   = (4 pi kT / w) [y / sinh y]^2,        y = w / (2 kT)

with F^out (weight v_a(k)^2) and F^in (weight v_a(k) v_a(k+q)) from
`elph.alpha2f_transport_matrix`, and D_a,i = N_i <v_a^2>_i its Drude weights.

Minimising over n_sheets parameters is EXACT within that trial space, so the
variational step costs nothing; the approximation is entirely the ansatz.
Because a larger trial space can only lower the resistivity, the band-resolved
answer always has sigma >= the one-Phi (isotropic) answer, which is recovered
by `lump_sheets` -- that inequality is a useful internal check and is tested.

THE SINGLE-SHEET LIMIT is Allen's textbook formula by construction: with one
sheet, M = A - C = D * integral dw alpha2F_tr(w) K(w,T) = D / tau_tr, since
alpha2F_tr = (F^out - F^in)/D, so sigma = e^2 D tau_tr / Omega -- Drude with
the transport lifetime. Two limits check the kernel: at high T the bracket
tends to 1 and 1/tau_tr -> 2 pi lambda_tr kT (linear resistivity, the standard
slope-to-lambda_tr relation), while at low T a Debye alpha2F_tr ~ w^4 gives
Bloch-Gruneisen rho ~ T^5. Both are tested.

WHAT IS NOT IN HERE, and would be wrong to read out of it:

* No energy resolution of the deviation function -- the degenerate limit is
  taken. So this gives sigma(T) and nothing else: Seebeck, the Lorenz number
  and any Wiedemann-Franz violation live precisely in the d/d(eps) structure
  that is integrated out here. `analysis.boltzmann` (constant relaxation time)
  has those, with its own approximation instead.
* Phonons are in equilibrium: no phonon drag. That is safe well above the
  drag regime and questionable in very clean samples at low T.
* Anisotropy WITHIN a sheet is gone. For transport that is a sharper loss than
  for T_c, because sigma weights v^2 and a neck or a hot spot can dominate.
* Only diagonal sigma_aa (see `elph.alpha2f_transport_matrix`).
* Umklapp/normal distinction, and the phonon-limited Hall coefficient, are
  outside the model.

IMPURITIES enter by Matthiessen on the diagonal: an elastic rate 1/tau_imp
adds D_a,i / tau_imp to M_a,ii. It also regularises the singular case below.

A SINGULAR M IS PHYSICS, NOT A BUG. If every scattering channel preserves the
current (F^in = F^out, pure forward scattering), then M = 0 and sigma diverges:
with no momentum sink the current never decays. The same happens when two
sheets exchange carriers with equal velocities and nothing else scatters. The
solver reports that instead of returning a meaningless number; add
`impurity_rate` to get the finite clean-limit answer.

Units: atomic units in, Kelvin at the temperature interface. `sigma` is in the
same convention as `analysis.boltzmann` (Hartree * atomic_time / Bohr, i.e.
1/Bohr with hbar = 1), so `waw.units.to_si_units(sigma, "electrical_
conductivity")` gives 1/(Ohm m) and `resistivity_si` wraps the reciprocal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import K_B_HARTREE, to_si_units

__all__ = ["LovaTransport", "transport_kernel", "lump_sheets",
           "lova_conductivity", "alpha2f_tr_effective", "lambda_tr",
           "stack_spin_channels", "spin_resolved_conductivity"]


@dataclass
class LovaTransport:
    """Solution of the band-resolved LOVA Boltzmann equation."""
    temperatures: np.ndarray   # (nT,) Kelvin
    sigma:        np.ndarray   # (nT, 3) atomic units (see module docstring)
    phi:          np.ndarray   # (nT, 3, n_sheets) the deviation amplitudes
    #: (nT, 3, n_sheets, n_sheets) collision matrix actually solved
    collision:    np.ndarray
    drude:        np.ndarray   # (3, n_sheets) as given
    cell_volume:  float        # Bohr^3
    num_elec_per_state: float = 2.0

    @property
    def sigma_si(self) -> np.ndarray:
        """Electrical conductivity in 1/(Ohm m), shape (nT, 3)."""
        return to_si_units(self.sigma, "electrical_conductivity")

    @property
    def resistivity_si(self) -> np.ndarray:
        """Resistivity in Ohm m, shape (nT, 3). Infinite where sigma is 0."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.sigma_si

    @property
    def resistivity_microohm_cm(self) -> np.ndarray:
        """Resistivity in micro-Ohm cm, the unit resistivity is quoted in."""
        return self.resistivity_si * 1e8

    def tau_transport(self) -> np.ndarray:
        """
        Effective transport lifetime (atomic time), shape (nT, 3), defined by
        inverting Drude on the total: sigma = (e^2/Omega) sum_i D_i tau_eff.
        Equals the true tau_tr exactly for one sheet; for several it is a
        summary number, since each sheet relaxes at its own rate.
        """
        d_tot = self.num_elec_per_state * self.drude.sum(axis=1)      # (3,)
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sigma * self.cell_volume / d_tot[None, :]

    def plasma_frequency(self) -> np.ndarray:
        """
        Drude plasma frequency Omega_p,a = sqrt(4 pi e^2 n_eff D_a / Omega) in
        Hartree, shape (3,). Independent of temperature and of the coupling, so
        comparing it with the free-electron value sqrt(4 pi n) is the cheapest
        check that the Wannier velocities and the Fermi-surface mesh are sound
        BEFORE reading any resistivity: sigma scales as Omega_p^2.
        """
        return np.sqrt(4.0 * np.pi * self.num_elec_per_state
                       * self.drude.sum(axis=1) / self.cell_volume)


def transport_kernel(omega: np.ndarray, kT: float) -> np.ndarray:
    """
    The LOVA temperature kernel K(w,T) = (4 pi kT / w) [y/sinh y]^2, y = w/2kT.

    Both factors are singular-looking and neither is a problem: [y/sinh y]^2
    -> 1 as w -> 0, leaving K ~ 4 pi kT / w, and the spectral functions vanish
    at least as fast as w^2 (as w^4 once the velocity weights are in), so the
    integrand is finite. w <= 0 entries return 0 so a grid that starts at zero
    can be passed as-is.

    The exponential suppression for w >> kT is what makes the resistivity fall
    off as T^5 rather than T^3: [y/sinh y]^2 ~ 4 y^2 exp(-2y).
    """
    omega = np.asarray(omega, dtype=np.float64)
    out = np.zeros_like(omega)
    pos = omega > 0.0
    if not np.any(pos):
        return out
    y = omega[pos] / (2.0 * kT)
    # y/sinh(y) underflows gracefully to 0 for large y; guard the small-y 0/0
    ratio = np.where(y < 1e-8, 1.0 - y ** 2 / 6.0, y / np.sinh(np.minimum(y, 700.0)))
    ratio = np.where(y > 700.0, 0.0, ratio)
    out[pos] = 4.0 * np.pi * kT * ratio ** 2 / omega[pos]
    return out


def stack_spin_channels(*channels):
    """
    Assemble per-spin-channel spectral matrices into ONE block-diagonal object,
    for the electrical conductivity of a collinear magnet.

    THE PHYSICS. Without spin-orbit coupling the electron-phonon interaction is
    spin-diagonal: an electron cannot flip its spin by absorbing a phonon. The
    two channels therefore have separate band structures, separate Fermi
    surfaces, separate lambda_tr -- and they never scatter into each other, so
    they conduct in PARALLEL,

        sigma = sigma_up + sigma_down,

    which is the two-current model (Fert & Campbell, J. Phys. F 6, 849 (1976)).
    Stacking the channels as sheets with ZERO inter-channel blocks makes the
    collision matrix block diagonal, and the ordinary solve then produces that
    parallel sum with no special-casing: the diagonal out-scattering sum over
    final sheets picks up only same-spin partners because the other blocks are
    zero, exactly as spin conservation requires.

    *** AND NOW THE FACTOR THAT IS EASY TO GET WRONG. *** Each channel's bands
    hold ONE electron, not two, so a spin-resolved calculation must be solved
    with ``num_elec_per_state=1``, NOT the 2 that is right for a non-magnetic
    band. The spin factor does not cancel in transport (see
    `lova_conductivity`), so using 2 here would double sigma. This is the
    documented failure mode to watch for when comparing against other codes:
    a machinery written for the non-magnetic case carries a hard-wired spin
    degeneracy of 2, and switching it to a spin-polarised calculation without
    removing that factor silently doubles the conductivity. Miguel's warning
    that EPW "has some issue with spin and electrical conductivity" is exactly
    this shape of bug, so treat any cross-code comparison of a magnet's sigma as
    suspect until the other code's spin bookkeeping has been checked explicitly.
    `test_spin_degenerate_ferromagnet_reproduces_the_nonmagnetic_answer` pins
    ours: two identical channels at occupancy 1 must equal one sheet at
    occupancy 2, to machine precision.

    With SOC the premise fails -- spin-flip scattering appears (Elliott-Yafet)
    and the channels are coupled, so this stacking is no longer exact. It is
    also silent about spin-dependent IMPURITY scattering, which in real magnets
    often dominates the channel asymmetry.

    Args:
      *channels : two or more `elph.TransportSpectralMatrix`, one per spin
        channel, all on the SAME omega grid. Each may itself be sheet-resolved;
        the sheets are concatenated in the order given.

    Returns a `TransportSpectralMatrix` whose sheet index runs over
    (channel, sheet). Solve it with ``num_elec_per_state=1``.
    """
    from .elph import TransportSpectralMatrix

    if len(channels) < 2:
        raise ValueError("stack_spin_channels: give at least two channels.")
    omega = np.asarray(channels[0].omega, dtype=np.float64)
    for c in channels[1:]:
        if not np.array_equal(np.asarray(c.omega, dtype=np.float64), omega):
            raise ValueError(
                "stack_spin_channels: all channels must share one omega grid; "
                "rebuild them with the same `omega_grid`.")
    sizes = [c.f_out.shape[1] for c in channels]
    n = sum(sizes)
    nE = len(omega)
    f_out = np.zeros((3, n, n, nE))
    f_in = np.zeros((3, n, n, nE))
    drude = np.zeros((3, n))
    dos = np.zeros(n)
    at = 0
    for c, s in zip(channels, sizes):
        sl = slice(at, at + s)
        f_out[:, sl, sl, :] = c.f_out
        f_in[:, sl, sl, :] = c.f_in          # zero off-block => spin conserved
        drude[:, sl] = c.drude
        dos[sl] = c.dos
        at += s
    return TransportSpectralMatrix(omega=omega, f_out=f_out, f_in=f_in,
                                   drude=drude, dos=dos)


def spin_resolved_conductivity(spectral_up, spectral_down, temperatures,
                               cell_volume: float, **kwargs):
    """
    Conductivity of a collinear magnet, and its decomposition by spin channel.

    Convenience wrapper: stacks the two channels (`stack_spin_channels`), solves
    with the correct ``num_elec_per_state=1``, and also solves each channel
    alone so the per-channel currents and the spin asymmetry come back with it.
    Any other `lova_conductivity` keyword is forwarded, but passing
    ``num_elec_per_state`` is refused -- getting that factor right is the whole
    point of this wrapper.

    Returns ``(total, up, down)`` as `LovaTransport` objects. Since the channels
    are independent, ``total.sigma == up.sigma + down.sigma`` to machine
    precision; the spin polarisation of the current is
    ``(up.sigma - down.sigma) / total.sigma``.
    """
    if "num_elec_per_state" in kwargs:
        raise ValueError(
            "spin_resolved_conductivity fixes num_elec_per_state = 1 (one "
            "electron per band per spin channel). Do not override it: that "
            "factor does not cancel, and a stray 2 doubles sigma.")
    both = stack_spin_channels(spectral_up, spectral_down)
    total = lova_conductivity(both, temperatures, cell_volume,
                              num_elec_per_state=1.0, **kwargs)
    up = lova_conductivity(spectral_up, temperatures, cell_volume,
                           num_elec_per_state=1.0, **kwargs)
    down = lova_conductivity(spectral_down, temperatures, cell_volume,
                             num_elec_per_state=1.0, **kwargs)
    return total, up, down


def lump_sheets(spectral):
    """
    Collapse a `TransportSpectralMatrix` to a single sheet -- the isotropic
    (one deviation amplitude for the whole Fermi surface) limit.

    Sums the sheet indices of both spectral functions and of the Drude weight.
    Because it shrinks the variational trial space, its conductivity is a LOWER
    bound on the band-resolved one; comparing the two measures how much the
    band resolution buys.
    """
    from .elph import TransportSpectralMatrix

    return TransportSpectralMatrix(
        omega=spectral.omega,
        f_out=spectral.f_out.sum(axis=(1, 2))[:, None, None, :],
        f_in=spectral.f_in.sum(axis=(1, 2))[:, None, None, :],
        drude=spectral.drude.sum(axis=1)[:, None],
        dos=np.atleast_1d(spectral.dos.sum()),
    )


def lova_conductivity(spectral, temperatures, cell_volume: float, *,
                      impurity_rate: float = 0.0,
                      num_elec_per_state: float = 2.0,
                      directions=(0, 1, 2)) -> LovaTransport:
    """
    Solve the band-resolved LOVA Boltzmann equation for sigma(T).

    Args:
      spectral      : `elph.TransportSpectralMatrix` (its `f_out`, `f_in`,
                      `drude` and `omega`).
      temperatures  : (nT,) Kelvin. Must be > 0; the kernel is proportional to
                      T and the model has no T = 0 limit at zero impurity rate.
      cell_volume   : unit-cell volume in Bohr^3 (the spectral functions and
                      Drude weights are per-cell sums).
      impurity_rate : elastic scattering rate 1/tau_imp in Hartree (hbar = 1),
                      added to the diagonal as D_i/tau_imp. Matthiessen.
      num_elec_per_state : spin occupancy of each band, 2 for non-spinor
                      Wannier functions (the default, and the same convention
                      and name as `analysis.boltzmann`). THIS FACTOR DOES NOT
                      CANCEL, unlike in lambda: electron-phonon scattering is
                      spin-diagonal, so the collision matrix and hence tau_tr
                      are per-spin quantities and carry no factor, while the
                      current sums over both spin channels. Omitting it halves
                      sigma -- caught on Al, where the Drude plasma frequency
                      came out at 1/sqrt(2) of its known value.
      directions    : which Cartesian components to solve (default all three).

    Returns `LovaTransport`; `sigma_si` and `resistivity_microohm_cm` convert.
    """
    f_out = np.asarray(spectral.f_out, dtype=np.float64)
    f_in = np.asarray(spectral.f_in, dtype=np.float64)
    drude = np.asarray(spectral.drude, dtype=np.float64)
    omega = np.asarray(spectral.omega, dtype=np.float64)
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=np.float64))
    if np.any(temperatures <= 0.0):
        raise ValueError("lova_conductivity: temperatures must be > 0 K.")
    n_sheets = f_out.shape[1]
    if f_out.shape != f_in.shape or drude.shape != (3, n_sheets):
        raise ValueError(
            f"lova_conductivity: shape mismatch -- f_out {f_out.shape}, f_in "
            f"{f_in.shape}, drude {drude.shape} (expected (3, {n_sheets})).")

    nT = len(temperatures)
    sigma = np.zeros((nT, 3))
    phi = np.zeros((nT, 3, n_sheets))
    coll = np.zeros((nT, 3, n_sheets, n_sheets))
    for it, T in enumerate(temperatures):
        K = transport_kernel(omega, T * K_B_HARTREE)
        # trapezoid over the frequency grid, the same rule alpha2F is built on
        A = np.trapezoid(f_out * K, omega, axis=-1)     # (3, ns, ns)
        C = np.trapezoid(f_in * K, omega, axis=-1)
        for a in directions:
            M = np.diag(A[a].sum(axis=1)) - C[a]
            if impurity_rate:
                M = M + np.diag(drude[a] * impurity_rate)
            coll[it, a] = M
            d = drude[a]
            if not np.any(d > 0.0):
                continue                # this direction carries no current
            try:
                p = np.linalg.solve(M, d)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"lova_conductivity: the collision matrix is singular at "
                    f"T = {T} K, direction {a}. That is physical, not a "
                    f"numerical accident: it means no channel relaxes the "
                    f"current (in-scattering cancels out-scattering exactly, "
                    f"e.g. pure forward scattering, or sheets that only trade "
                    f"carriers of equal velocity), so sigma diverges. Pass "
                    f"impurity_rate > 0 for the finite clean-limit answer."
                ) from exc
            phi[it, a] = p
            sigma[it, a] = num_elec_per_state * float(d @ p) / cell_volume
    return LovaTransport(temperatures=temperatures, sigma=sigma, phi=phi,
                         collision=coll, drude=drude, cell_volume=float(cell_volume),
                         num_elec_per_state=float(num_elec_per_state))


def alpha2f_tr_effective(spectral, direction: int = 0) -> np.ndarray:
    """
    Allen's transport spectral function alpha2F_tr(w) = [F^out - F^in]/D for
    the LUMPED Fermi surface, i.e. the single-sheet reduction of `spectral`.

    This is the quantity whose first inverse moment is lambda_tr, and the one
    to compare against a literature alpha2F_tr -- with the caveat in
    `elph.alpha2f_transport_matrix` that its k-weighting is v_a^2, not the flat
    Fermi-surface average EPW's transport lambda uses.
    """
    num = (spectral.f_out - spectral.f_in)[direction].sum(axis=(0, 1))
    d = float(spectral.drude[direction].sum())
    if d <= 0.0:
        raise ValueError("alpha2f_tr_effective: zero Drude weight -- no "
                         "carriers with velocity along this direction.")
    return num / d


def lambda_tr(spectral, direction: int = 0) -> float:
    """
    Transport coupling lambda_tr = 2 integral dw alpha2F_tr(w)/w, from
    `alpha2f_tr_effective`. This is the number that sets the high-temperature
    resistivity slope: 1/tau_tr -> 2 pi lambda_tr kT.
    """
    a2f = alpha2f_tr_effective(spectral, direction)
    w = spectral.omega
    pos = w > 0.0
    return float(2.0 * np.trapezoid(a2f[pos] / w[pos], w[pos]))
