"""
Superconducting critical fields from a tight-binding H(R), following

  "Towards the discovery of high critical magnetic field superconductors",
  arXiv:2601.21044 (2026) -- the University of Florida workflow (Hennig et al.),

with the lower critical field taken from its reference for that quantity,

  E. H. Brandt, "Properties of the ideal Ginzburg-Landau vortex lattice",
  Phys. Rev. B 68, 054506 (2003); the interpolation formulas quoted below are
  Eqs. (56)-(57) of Brandt's review arXiv:0806.1058.

THE PAPER'S CHAIN, and where this module improves on it.

Their ingredients are a Fermi-surface average of the velocity, the density of
states at E_F, an Eliashberg gap, and lambda_ep:

    <vF^2>   = (1/(A_FS hbar^2)) int_FS dn (grad_k eps)^2                (Eq. 1)
    xi(0)    = hbar sqrt(<vF^2>) / (pi <Delta(0)>)                       (Eq. 2)
    lambda_L = sqrt( 3 / (mu_0 e^2 n(E_F) <vF^2>) )                      (Eq. 4)
    kappa    = lambda_L / xi                                             (Eq. 8)
    Hc2      = Phi_0 / (2 pi xi^2)                                       (Eq. 9)
    Hc1      = Phi_0 [ln kappa + alpha(kappa)] / (4 pi lambda_L^2)       (Eq. 10)
    Hc       = Phi_0 / (2 sqrt2 pi lambda_L xi)                          (Eq. 11)

and the electron-phonon renormalisation, applied to both lengths,

    n(E_F) -> n(E_F)(1 + lambda_ep),   vF -> vF/(1 + lambda_ep)          (Eq. 7)
    lambda_L -> lambda_L sqrt(1 + lambda_ep),  xi -> xi/(1 + lambda_ep)  (Eq. 6)

**Their Eq. 4 is an isotropic approximation, and this module can do better.**
It is the London relation lambda^-2 = mu_0 e^2 (n/m*) with n/m* replaced by
n(E_F)<vF^2>/3, which is exact only for a free-electron gas -- and the average
in Eq. 1 is weighted by Fermi-surface AREA, whereas the London superfluid weight
is weighted by the density of states:

    <f>_area = int dS f / int dS          = sum_k delta(eps-mu) |v| f / ...
    <f>_dos  = int (dS/|v|) f / int dS/|v| = sum_k delta(eps-mu) f / ...

The two coincide only where |v| is constant over the Fermi surface. So
`fermi_surface_averages` returns both, and the recommended route for lambda_L is
not Eq. 4 at all but the superfluid weight of `analysis.superfluid`, which is the
exact anisotropic tensor and carries the quantum-geometric term Eq. 4 has no
representation for. `critical_fields` therefore takes lambda_L as an argument:
feed it either, and compare.

UNITS: atomic units throughout (Hartree, Bohr, hbar = e = m_e = 1), per the
project convention. Fields come out in the atomic unit of magnetic flux density,
hbar/(e a_0^2) = 2.350517e5 T; convert at the boundary with
`waw.units.to_si_units(B, "magnetic_flux_density")`. In these units the flux
quantum is Phi_0 = h/2e = pi, so Eq. 9 is simply Hc2 = 1/(2 xi^2).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..core.distributions import gaussian_smearing
from ..units import register_si_unit, register_from_si_unit, AU_B_FIELD_TESLA

PHI0_AU = np.pi                      # h/2e with h = 2 pi, e = 1
KAPPA_TYPE_BOUNDARY = 1.0 / np.sqrt(2.0)

# Brandt Eq. (56): alpha(kappa) = alpha_inf + exp[-c0 - c1 ln k - c2 (ln k)^2],
# with |error| <= 7.6e-4. Reproduces hc1 = 1 at kappa = 1/sqrt(2) and
# alpha -> 0.49693 for kappa >> 1 -- both verified in the tests.
BRANDT56 = dict(alpha_inf=0.49693, c0=0.41477, c1=0.775, c2=0.1303, eps=0.00076)


@dataclass
class CriticalFields:
    """All in atomic units (fields in hbar/(e a_0^2)); lengths in Bohr."""
    lambda_L: float
    xi: float
    kappa: float
    Hc: float
    Hc1: float
    Hc2: float
    type_ii: bool
    alpha_brandt: float

    def to_tesla(self) -> dict:
        d = asdict(self)
        for k in ("Hc", "Hc1", "Hc2"):
            d[k + "_tesla"] = d[k] * AU_B_FIELD_TESLA
        return d


def alpha_brandt(kappa, form: str = "eq56") -> np.ndarray:
    """
    alpha(kappa) of Brandt's lower-critical-field formula.

    `form="eq56"` is the accurate interpolation (error <= 7.6e-4); `"eq57"` is
    his simpler one, alpha = 0.5 + (1 + ln 2)/(2 kappa - sqrt2 + 2), which is
    exact at kappa = 1/sqrt2 and within ~1% elsewhere. Note the `+ 2` in that
    denominator: without it the expression diverges at exactly the type-I/II
    boundary, so a version missing it is wrong there.

    The Florida paper writes Hc1 with alpha(kappa) but does not print the
    expression, citing Brandt -- hence both forms here, and the tests pin them
    against the two limits Brandt states.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    if np.any(kappa <= 0):
        raise ValueError("alpha_brandt: kappa must be positive")
    if form == "eq57":
        return 0.5 + (1.0 + np.log(2.0)) / (2.0 * kappa - np.sqrt(2.0) + 2.0)
    if form != "eq56":
        raise ValueError(f"alpha_brandt: form must be 'eq56' or 'eq57', got {form!r}")
    p, L = BRANDT56, np.log(kappa)
    return p["alpha_inf"] + np.exp(-p["c0"] - p["c1"] * L - p["c2"] * L ** 2)


def fermi_surface_averages(eig, velocities, mu, sigma, volume, n_spin=2):
    """
    n(E_F) and the two Fermi-surface averages of v^2, by Gaussian smearing.

    Args:
      eig        : (nk, nb) band energies, Hartree
      velocities : (nk, nb, 3) band velocities dE/dk, atomic units
      mu         : Hartree
      sigma      : Hartree, the smearing width representing delta(eps - mu)
      volume     : unit-cell volume, Bohr^3
      n_spin     : 2 for a spin-degenerate calculation (the paper's n(E_F) is the
                   TOTAL density of states, normalised by the cell volume)

    Returns dict with
      dos          : n(E_F), states / (Hartree Bohr^3)
      v2_dos       : <v^2> weighted by the density of states -- the average that
                     enters the London/Drude weight
      v2_area      : <v^2> weighted by Fermi-surface AREA -- the paper's Eq. 1
      v_rms_dos/area, area_over_dos : diagnostics; area_over_dos = <|v|>_dos,
                     the factor by which the two weightings differ
    """
    eig = np.asarray(eig, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if vel.shape[:2] != eig.shape or vel.shape[2] != 3:
        raise ValueError(f"velocities must be (nk, nb, 3) matching eig {eig.shape}, "
                         f"got {vel.shape}")
    nk = eig.shape[0]
    w = np.asarray(gaussian_smearing(eig - mu, sigma))      # (nk, nb)
    v2 = (vel ** 2).sum(axis=-1)                            # (nk, nb)
    vmag = np.sqrt(v2)

    wsum = w.sum()
    if wsum <= 0:
        raise ValueError("no spectral weight at mu -- widen sigma or check mu")
    dos = n_spin * wsum / nk / volume
    wa = w * vmag                                           # area measure

    # full <v_a v_b> tensors as well as the scalar traces: an anisotropic
    # material has a direction-dependent coherence length and upper critical
    # field, and collapsing to one number throws that away
    tens_dos = np.einsum('kb,kba,kbc->ac', w, vel, vel) / wsum
    tens_area = np.einsum('kb,kba,kbc->ac', wa, vel, vel) / wa.sum()
    return dict(
        dos=float(dos),
        v2_dos=float((w * v2).sum() / wsum),
        v2_area=float((wa * v2).sum() / wa.sum()),
        v_rms_dos=float(np.sqrt((w * v2).sum() / wsum)),
        v_rms_area=float(np.sqrt((wa * v2).sum() / wa.sum())),
        area_over_dos=float((w * vmag).sum() / wsum),
        v2_tensor_dos=tens_dos,
        v2_tensor_area=tens_area,
        v2_diag_dos=np.diag(tens_dos).copy(),
        v2_diag_area=np.diag(tens_area).copy(),
        anisotropy=float(np.diag(tens_dos).max() / max(np.diag(tens_dos).min(),
                                                       1e-300)),
    )


def coherence_length(v2, delta, lambda_ep=0.0):
    """
    xi(0) = hbar sqrt(<vF^2>) / (pi <Delta(0)>), Eq. 2, with the electron-phonon
    renormalisation xi -> xi/(1 + lambda_ep) of Eq. 6.

    v2 in atomic units, delta in Hartree. Returns Bohr.
    """
    if delta <= 0:
        raise ValueError("coherence_length: delta must be positive")
    return float(np.sqrt(v2) / (np.pi * delta) / (1.0 + lambda_ep))


def london_depth_dft(dos, v2, lambda_ep=0.0):
    """
    lambda_L(0) = sqrt(3 / (mu_0 e^2 n(E_F) <vF^2>)), Eq. 4, with
    lambda_L -> lambda_L sqrt(1 + lambda_ep) of Eq. 6.

    In atomic units mu_0 = 4 pi alpha^2 with alpha the fine-structure constant
    (c = 1/alpha, eps_0 = 1/4pi), so this is
    lambda_L = sqrt(3 / (4 pi alpha^2 n <v^2>)).

    This is the paper's isotropic route and is kept for comparison; prefer the
    superfluid weight (see the module docstring). dos in states/(Hartree Bohr^3),
    v2 in atomic units. Returns Bohr.
    """
    from ..units import FINE_STRUCTURE
    mu0_au = 4.0 * np.pi * FINE_STRUCTURE ** 2
    if dos <= 0 or v2 <= 0:
        raise ValueError("london_depth_dft: dos and v2 must be positive")
    return float(np.sqrt(3.0 / (mu0_au * dos * v2)) * np.sqrt(1.0 + lambda_ep))


def critical_fields(lambda_L, xi, alpha_form: str = "eq56") -> CriticalFields:
    """
    kappa, Hc, Hc1, Hc2 from a penetration depth and a coherence length.

    lambda_L, xi in Bohr; fields returned in the atomic unit of magnetic flux
    density (2.350517e5 T). Eqs. 8-11 of arXiv:2601.21044:

        kappa = lambda_L / xi
        Hc2   = Phi_0/(2 pi xi^2)                        = 1/(2 xi^2)
        Hc1   = Phi_0 [ln kappa + alpha]/(4 pi lambda^2) = [ln k + a]/(4 lambda^2)
        Hc    = Phi_0/(2 sqrt2 pi lambda xi)             = 1/(2 sqrt2 lambda xi)

    Hc1 is only meaningful for a type-II superconductor; for kappa < 1/sqrt2 it
    is returned as nan rather than as a number that looks like a field, since
    ln(kappa) < 0 there and the vortex lattice does not exist.
    """
    lambda_L, xi = float(lambda_L), float(xi)
    if lambda_L <= 0 or xi <= 0:
        raise ValueError("critical_fields: lambda_L and xi must be positive")
    kappa = lambda_L / xi
    type_ii = kappa > KAPPA_TYPE_BOUNDARY
    a = float(alpha_brandt(kappa, form=alpha_form))
    hc2 = PHI0_AU / (2.0 * np.pi * xi ** 2)
    hc = PHI0_AU / (2.0 * np.sqrt(2.0) * np.pi * lambda_L * xi)
    hc1 = (PHI0_AU * (np.log(kappa) + a) / (4.0 * np.pi * lambda_L ** 2)
           if type_ii else float("nan"))
    return CriticalFields(lambda_L=lambda_L, xi=xi, kappa=kappa, Hc=hc,
                          Hc1=hc1, Hc2=hc2, type_ii=bool(type_ii),
                          alpha_brandt=a)


def coherence_length_axes(v2_diag, delta, lambda_ep=0.0):
    """Per-axis xi_a from the DIAGONAL of <v_a v_b>, Eq. 2 applied componentwise.

    The paper's Eq. 2 uses one scalar <vF^2>, i.e. an isotropic xi. For an
    anisotropic Fermi surface the natural generalisation keeps the axis
    resolution: xi_a = hbar sqrt(3 <v_a^2>) / (pi Delta), normalised so that an
    isotropic system (3<v_a^2> = <v^2>) reproduces the scalar result exactly.
    """
    v2_diag = np.asarray(v2_diag, dtype=np.float64)
    if v2_diag.shape != (3,):
        raise ValueError(f"v2_diag must have shape (3,), got {v2_diag.shape}")
    return np.array([coherence_length(3.0 * v2, delta, lambda_ep)
                     for v2 in v2_diag])


@dataclass
class AnisotropicCriticalFields:
    """Field-direction-resolved critical fields, atomic units.

    Hc2[c] and Hc1[c] are for the field along axis c and use the coherence
    lengths / penetration depths TRANSVERSE to it, which is the standard
    anisotropic Ginzburg-Landau result:

        Hc2[c] = Phi_0 / (2 pi xi_a xi_b),     (a, b) the other two axes
        Hc1[c] = Phi_0 [ln kappa_c + alpha(kappa_c)] / (4 pi lambda_a lambda_b)
        kappa_c = sqrt(lambda_a lambda_b / (xi_a xi_b))

    Hc is NOT direction-resolved: it is thermodynamic, fixed by the condensation
    energy, so a single value is correct and it is taken from the isotropic
    formula. This whole class is an extension BEYOND arXiv:2601.21044, which is
    isotropic throughout -- use `critical_fields` for numbers comparable to
    theirs.
    """
    lambda_axes: np.ndarray
    xi_axes: np.ndarray
    kappa_axes: np.ndarray       # kappa_c for field along c
    Hc1_axes: np.ndarray
    Hc2_axes: np.ndarray
    type_ii_axes: np.ndarray


def critical_fields_axes(lambda_axes, xi_axes, alpha_form: str = "eq56"):
    """Anisotropic Ginzburg-Landau fields per field direction (see the dataclass)."""
    lam = np.asarray(lambda_axes, dtype=np.float64)
    xi = np.asarray(xi_axes, dtype=np.float64)
    if lam.shape != (3,) or xi.shape != (3,):
        raise ValueError("lambda_axes and xi_axes must both have shape (3,)")
    if np.any(lam <= 0) or np.any(xi <= 0):
        raise ValueError("lambda_axes and xi_axes must be positive")
    hc1, hc2, kap, tii = (np.zeros(3), np.zeros(3), np.zeros(3),
                          np.zeros(3, dtype=bool))
    for c in range(3):
        a, b = [i for i in range(3) if i != c]
        kap[c] = np.sqrt(lam[a] * lam[b] / (xi[a] * xi[b]))
        tii[c] = kap[c] > KAPPA_TYPE_BOUNDARY
        hc2[c] = PHI0_AU / (2.0 * np.pi * xi[a] * xi[b])
        hc1[c] = (PHI0_AU * (np.log(kap[c]) + float(alpha_brandt(kap[c], alpha_form)))
                  / (4.0 * np.pi * lam[a] * lam[b]) if tii[c] else float("nan"))
    return AnisotropicCriticalFields(lambda_axes=lam, xi_axes=xi, kappa_axes=kap,
                                     Hc1_axes=hc1, Hc2_axes=hc2, type_ii_axes=tii)


@register_si_unit("magnetic_flux_density")
def au_to_tesla(B):
    """Atomic unit of magnetic flux density -> Tesla (hbar/(e a_0^2))."""
    return np.asarray(B, dtype=np.float64) * AU_B_FIELD_TESLA


@register_from_si_unit("magnetic_flux_density")
def tesla_to_au(B):
    """Tesla -> atomic unit of magnetic flux density."""
    return np.asarray(B, dtype=np.float64) / AU_B_FIELD_TESLA
