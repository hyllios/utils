"""
ARPES photoemission matrix elements from Wannier functions.

The bare surface spectral function ``A(k_par, E)`` (``analysis.surface``) is
the single-particle quantity; an ARPES experiment measures

    I(k_par, E) ~ |sum_a M_a(k_f) <a|psi>|^2  ->  M† A M   (spin-integrated),

where ``M_a`` is the one-electron dipole matrix element from Wannier orbital
``a`` to the photoelectron final state. This module builds ``M_a`` in the
**sudden approximation with a plane-wave (free-electron) final state** -- the
standard "tight-binding / Wannier ARPES" level (Moser, *J. Electron Spectrosc.*
**214**, 29 (2017); Day, Zwartsenberg, Elfimov & Damascelli, *chinook*, *npj
Quantum Materials* **4**, 54 (2019)). Contract the result with
``analysis.surface``'s top-layer Green's function via its ``matrix_element=``
argument.

For a Wannier orbital of cubic-harmonic character ``(l, m)`` centred at
``r_a``, with photoelectron momentum ``k_f`` (``|k_f|`` fixed by the photon
energy, direction fixed by ``k_par``):

    M_a(k_f) = e^{i k_f . r_a}  sum_{L=l±1, M} (-i)^L 4pi B_{lL}(|k_f|)
                                              Y*_LM(k_f_hat) <Y_LM|eps.r_hat|K_lm>

capturing (i) the **structure-factor phase** ``e^{i k_f.r_a}`` (BZ-dependent
intensity, "dark corridors"), (ii) the **dipole selection rules** ``l -> l±1``
and the **polarization** through the angular coupling
``<Y_LM|eps.r_hat|K_lm>``, and (iii) the **radial / cross-section** factor
``B_{lL}`` (a single-zeta Slater estimate; sets the ``l±1`` branching ratio and
the smooth ``h*nu`` envelope).

Approximations (documented, honest): plane-wave final state (no multiple
scattering / one-step, no inner-potential refraction), spin-diagonal matrix
element applied to each Wannier function's dominant spatial ``(l, m)`` character
(exact for a spinless model; for spinor/SOC Wannier functions this is the
spin-integrated dominant-orbital approximation), and single-zeta Slater radial
functions (relative, not absolute, cross-sections).

Units: lengths Bohr, momenta 1/Bohr, energies eV on the public API.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y, spherical_jn

from ..units import EV_TO_HARTREE


# --------------------------------------------------------------------------
# Angular quadrature and cubic harmonics
# --------------------------------------------------------------------------

def _quadrature(ntheta: int = 48, nphi: int = 96):
    """Gauss-Legendre in cos(theta) x uniform in phi on the unit sphere."""
    x, wx = np.polynomial.legendre.leggauss(ntheta)   # cos(theta) in [-1, 1]
    theta = np.arccos(x)
    phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(wx, np.full(nphi, 2 * np.pi / nphi))   # sum W = 4pi
    return TH.ravel(), PH.ravel(), W.ravel()


# W90 real cubic harmonics as explicit polynomials in the direction cosines
# (dx, dy, dz) = (sin th cos ph, sin th sin ph, cos th); normalized so that
# the integral of |K|^2 over the sphere is 1. Keyed by (l, mr) in the
# Wannier90 mr ordering used by `interfaces.projections.spd_projections`.
def _cubic_harmonic(l: int, mr: int, th, ph):
    dx = np.sin(th) * np.cos(ph)
    dy = np.sin(th) * np.sin(ph)
    dz = np.cos(th)
    if l == 0:
        return np.full_like(th, np.sqrt(1.0 / (4 * np.pi)))
    if l == 1:
        c = np.sqrt(3.0 / (4 * np.pi))
        return {1: c * dz, 2: c * dx, 3: c * dy}[mr]
    if l == 2:
        return {
            1: np.sqrt(5.0 / (16 * np.pi)) * (3 * dz**2 - 1),
            2: np.sqrt(15.0 / (4 * np.pi)) * dx * dz,
            3: np.sqrt(15.0 / (4 * np.pi)) * dy * dz,
            4: np.sqrt(15.0 / (16 * np.pi)) * (dx**2 - dy**2),
            5: np.sqrt(15.0 / (4 * np.pi)) * dx * dy,
        }[mr]
    raise NotImplementedError(f"cubic harmonic l={l} not implemented (need s/p/d)")


def _Y(l, m, th, ph):
    # scipy sph_harm_y(n=l, m, theta=polar, phi=azimuth); Condon-Shortley phase
    return sph_harm_y(l, m, th, ph)


def _complex_coeffs(l: int, mr: int, quad):
    """c_m = <Y_lm | K_{l,mr}> so that K = sum_m c_m Y_lm (numerical)."""
    th, ph, w = quad
    K = _cubic_harmonic(l, mr, th, ph)
    return {m: np.sum(w * np.conj(_Y(l, m, th, ph)) * K) for m in range(-l, l + 1)}


def _dipole_coupling(l: int, quad):
    """<Y_LM | r_hat_i | Y_lm> for L=l±1, i in {x,y,z}. Numerical quadrature.

    Returns dict[(L, M, m, i)] -> complex.
    """
    th, ph, w = quad
    rhat = {"x": np.sin(th) * np.cos(ph),
            "y": np.sin(th) * np.sin(ph),
            "z": np.cos(th)}
    out = {}
    for L in (l - 1, l + 1):
        if L < 0:
            continue
        for M in range(-L, L + 1):
            YLM_c = np.conj(_Y(L, M, th, ph))
            for m in range(-l, l + 1):
                Ylm = _Y(l, m, th, ph)
                for i, ri in rhat.items():
                    out[(L, M, m, i)] = np.sum(w * YLM_c * ri * Ylm)
    return out


# --------------------------------------------------------------------------
# Radial (cross-section) factor: single-zeta Slater
# --------------------------------------------------------------------------

# Rough single-zeta Slater exponents (1/Bohr) for the valence shells we use.
_SLATER_ZETA = {("Ni", 3, 2): 3.6, ("I", 5, 1): 2.8}
_SHELL_N = {2: {"default": 3}, 1: {"default": 3}}   # fallback principal n by l


def _radial_B(l: int, L: int, kf: float, n: int, zeta: float) -> float:
    """B_{lL}(kf) = \\int_0^inf j_L(kf r) r^3 R_nl(r) dr, single-zeta Slater
    R_nl(r) ~ r^{n-1} e^{-zeta r} (normalized \\int R^2 r^2 dr = 1)."""
    r = np.linspace(1e-6, 40.0 / zeta, 4000)
    R = r ** (n - 1) * np.exp(-zeta * r)
    R /= np.sqrt(np.trapezoid(R**2 * r**2, r))
    return float(np.trapezoid(spherical_jn(L, kf * r) * r**3 * R, r))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def wannier_lm_from_projections(projections, spinor: bool = True):
    """(l, mr) per Wannier function, in Hamiltonian basis order.

    ``projections`` is the list of specs passed to ``generate_overlaps`` /
    ``write_nnkp`` (each ``(centre, l, mr, r, zaxis, xaxis, zona)``). For a
    spinor run each spec yields two consecutive WFs (spin up then down), so
    each ``(l, mr)`` is repeated -- matching the ``.amn`` column order and
    hence ``result.hr`` / ``result.wannier_centres``.
    """
    lm = []
    for spec in projections:
        l, mr = int(spec[1]), int(spec[2])
        lm.extend([(l, mr)] * (2 if spinor else 1))
    return lm


def photoemission_matrix_element(
    kpar_cart: np.ndarray,
    wf_centres: np.ndarray,
    wf_lm,
    polarization,
    photon_energy: float,
    *,
    work_function: float = 4.5,
    elements=None,
    radial: dict | None = None,
    quad=None,
) -> np.ndarray:
    """Plane-wave-final-state dipole ARPES matrix element ``M (nk, nw)``.

    Args:
      kpar_cart    : (nk, 2) in-plane photoelectron momentum, Cartesian, 1/Bohr
                     (surface normal assumed +z).
      wf_centres   : (nw, 3) Wannier centres, Cartesian, Bohr
                     (e.g. ``result.wannier_centres``).
      wf_lm        : length-nw list of ``(l, mr)`` cubic-harmonic labels
                     (``wannier_lm_from_projections``).
      polarization : length-3 (complex) light polarization vector in the same
                     Cartesian frame (real for linear, complex for circular).
      photon_energy: eV. Final-state kinetic energy KE = photon_energy -
                     work_function fixes ``|k_f|``; the small variation of
                     ``|k_f|`` across the eV-scale binding-energy window is
                     neglected.
      elements     : optional length-nw list of element symbols, to pick the
                     Slater shell/exponent per WF (defaults: guess from l).
      radial       : optional {(l, L): B} overrides for the radial factors.
      quad         : optional (theta, phi, w) quadrature (``_quadrature()``).

    Returns ``M`` (nk, nw) complex, ready for ``surface_spectral_function(...,
    matrix_element=M)``.
    """
    kpar = np.asarray(kpar_cart, dtype=np.float64)
    centres = np.asarray(wf_centres, dtype=np.float64)
    nk, nw = kpar.shape[0], centres.shape[0]
    if len(wf_lm) != nw:
        raise ValueError(f"wf_lm has {len(wf_lm)} entries, expected nw={nw}")
    eps = np.asarray(polarization, dtype=np.complex128)
    quad = quad or _quadrature()

    KE = photon_energy - work_function
    if KE <= 0:
        raise ValueError("photon_energy must exceed work_function")
    kf_mag = np.sqrt(2.0 * KE * EV_TO_HARTREE)          # 1/Bohr

    # final-state momenta: (kx, ky, kz>=0) with |k_f| = kf_mag
    kperp2 = kf_mag**2 - (kpar[:, 0]**2 + kpar[:, 1]**2)
    kz = np.sqrt(np.clip(kperp2, 0.0, None))
    kf = np.stack([kpar[:, 0], kpar[:, 1], kz], axis=1)  # (nk, 3)
    emitted = kperp2 > 0                                 # outside -> evanescent
    khat = np.zeros_like(kf)
    khat[emitted] = kf[emitted] / kf_mag

    # precompute, per distinct (l, mr): the angular emission amplitude
    #   D_LM^a = sum_m c_m <Y_LM | eps.r_hat | Y_lm>,   for L = l±1
    # and the radial factors B_{lL}(kf_mag).
    def _shell(l, elem):
        if elem is not None and (elem, {2: 3, 1: 5, 0: 4}.get(l), l) in _SLATER_ZETA:
            n = {2: 3, 1: 5, 0: 4}[l]
            return n, _SLATER_ZETA[(elem, n, l)]
        for (e, n, ll), z in _SLATER_ZETA.items():   # any element with this l
            if ll == l:
                return n, z
        return (l + 1), 3.0

    cache = {}
    for a, (l, mr) in enumerate(wf_lm):
        elem = elements[a] if elements is not None else None
        key = (l, mr, elem)
        if key in cache:
            continue
        coeffs = _complex_coeffs(l, mr, quad)
        G = _dipole_coupling(l, quad)
        n, zeta = _shell(l, elem)
        Ls = [L for L in (l - 1, l + 1) if L >= 0]
        B = {}
        for L in Ls:
            B[L] = (radial or {}).get((l, L)) or _radial_B(l, L, kf_mag, n, zeta)
        # D[L] : (2L+1,) amplitude vector over M for this orbital
        D = {}
        for L in Ls:
            Dv = np.zeros(2 * L + 1, dtype=np.complex128)
            for Mi, M in enumerate(range(-L, L + 1)):
                s = 0.0j
                for m in range(-l, l + 1):
                    gi = (eps[0] * G[(L, M, m, "x")] + eps[1] * G[(L, M, m, "y")]
                          + eps[2] * G[(L, M, m, "z")])
                    s += coeffs[m] * gi
                Dv[Mi] = s
            D[L] = Dv
        cache[key] = (Ls, B, D)

    # spherical harmonics of the emission directions, per needed L
    Lmax = max(l + 1 for l, _ in wf_lm)
    th_f = np.arccos(np.clip(khat[:, 2], -1.0, 1.0))
    ph_f = np.arctan2(khat[:, 1], khat[:, 0])
    Ycache = {}
    for L in range(Lmax + 1):
        Ycache[L] = np.stack([_Y(L, M, th_f, ph_f) for M in range(-L, L + 1)], axis=1)

    M = np.zeros((nk, nw), dtype=np.complex128)
    phase = np.exp(1j * (kf @ centres.T))               # (nk, nw) e^{i k_f . r_a}
    for a, (l, mr) in enumerate(wf_lm):
        elem = elements[a] if elements is not None else None
        Ls, B, D = cache[(l, mr, elem)]
        ang = np.zeros(nk, dtype=np.complex128)
        for L in Ls:
            YLM = np.conj(Ycache[L])                     # (nk, 2L+1)
            ang += ((-1j) ** L) * 4 * np.pi * B[L] * (YLM @ D[L])
        M[:, a] = phase[:, a] * ang
    M[~emitted, :] = 0.0
    return M
