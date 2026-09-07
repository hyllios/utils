"""
GGA exchange-correlation potential and its linear response (the xc kernel
action), for the non-linear-core-correction (NLCC) term of the phonon
perturbation (`analysis.elph`).

QE's DFPT NLCC contribution to the *bare* perturbation is
``dV_xc^core = f_xc[rho_tot] . drho_core`` (``PHonon/PH/compute_dvloc`` ->
``LR_Modules/dv_of_drho_xc``): the exchange-correlation potential's linear
response to the rigidly-displaced atomic core charge, evaluated at the
ground-state total density ``rho_tot = rho_val + rho_core``. For a GGA (this
project uses PBE) the kernel is gradient-dependent; rather than port QE's
Rydberg-unit ``dgcxc``/``dgradcorr`` array bookkeeping, we evaluate the
standard closed-form GGA kernel action directly from libxc's first and second
derivatives (`pylibxc`) with FFT-based gradients/divergence, and self-validate
it as the finite-difference derivative of `xc_potential_gga` (a rigorous,
QE-independent check -- see `tests/test_analysis_xc.py`).

All atomic (Hartree) units: densities in electrons/Bohr^3, potentials in
Hartree, lengths in Bohr. libxc is unit-agnostic, so no Rydberg ``e2`` factor
appears (unlike QE).

Convention (spin-unpolarized): with the energy density
``eps = rho * e_xc(rho, sigma)``, ``sigma = |grad rho|^2``, libxc returns
``vrho = d eps/d rho``, ``vsigma = d eps/d sigma`` and the seconds ``v2rho2``,
``v2rhosigma``, ``v2sigma2``; the GGA potential is

    V_xc = vrho - 2 div( vsigma grad rho ),

and its response to a (Bloch, wavevector q) perturbation ``drho`` is

    dV_xc = v2rho2 drho + v2rhosigma dsigma
            - 2 div[ vsigma grad(drho) + (v2rhosigma drho + v2sigma2 dsigma) grad rho ],
    dsigma = 2 grad rho . grad(drho),

with the perturbation's gradients/divergence taken at ``q+G`` (the ground-state
``rho`` at ``G``). This is exactly what ``dv_of_drho_xc`` computes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# FFT gradient / divergence on the periodic FFT grid
# ---------------------------------------------------------------------------

def _gfrac(grid_shape):
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in grid_shape]
    return np.stack(np.meshgrid(*ints, indexing="ij"), axis=-1)   # (nx,ny,nz,3)


def _qg_cart(grid_shape, recip_lattice, q_frac=(0.0, 0.0, 0.0)):
    """(nx,ny,nz,3) Cartesian (q+G) in 1/Bohr; rows of ``recip_lattice`` are
    the reciprocal vectors b_i (2*pi/Bohr)."""
    gf = _gfrac(grid_shape).astype(np.float64) + np.asarray(q_frac, float)
    return gf @ np.asarray(recip_lattice, float)


def _nyquist_mask(grid):
    """False at the Nyquist frequency of each even-length axis. A spectral
    first derivative must drop the Nyquist mode (its +N/2 vs -N/2 sign is
    ambiguous, `numpy.fft.fftfreq` picks -N/2), else the derivative of a real
    field acquires a spurious imaginary part."""
    m = np.ones(grid, dtype=bool)
    for a, n in enumerate(grid):
        if n % 2 == 0:
            idx = [slice(None)] * len(grid)
            idx[a] = n // 2
            m[tuple(idx)] = False
    return m


def _grad(f, qg_cart):
    """Gradient of a periodic field ``f`` (real or complex, shape grid) whose
    periodic part carries wavevector q: returns (3, *grid) complex."""
    F = np.fft.fftn(f) * _nyquist_mask(f.shape)
    return np.stack([np.fft.ifftn(1j * qg_cart[..., a] * F) for a in range(3)], axis=0)


def _div(V, qg_cart):
    """Divergence of a periodic vector field ``V`` (3, *grid) at wavevector q."""
    mask = _nyquist_mask(V.shape[1:])
    out = np.zeros(V.shape[1:], dtype=np.complex128)
    for a in range(3):
        out += np.fft.ifftn(1j * qg_cart[..., a] * np.fft.fftn(V[a]) * mask)
    return out


# ---------------------------------------------------------------------------
# libxc PBE derivatives
# ---------------------------------------------------------------------------

def _pbe_functionals():
    import pylibxc
    return (pylibxc.LibXCFunctional("gga_x_pbe", "unpolarized"),
            pylibxc.LibXCFunctional("gga_c_pbe", "unpolarized"))


def pbe_derivatives(rho, sigma, *, fxc=False):
    """PBE (x+c) derivatives on flat arrays ``rho``, ``sigma`` (electrons/Bohr^3,
    (electrons/Bohr^3/Bohr)^2). Returns dict with ``vrho``, ``vsigma`` and, if
    ``fxc``, ``v2rho2``, ``v2rhosigma``, ``v2sigma2`` -- PBE = gga_x_pbe +
    gga_c_pbe summed. Density floored at a tiny positive value (libxc returns
    0-derivatives in vacuum; guards divide-by-rho)."""
    fx, fc = _pbe_functionals()
    rho = np.clip(np.asarray(rho, float).ravel(), 1e-12, None)
    sigma = np.clip(np.asarray(sigma, float).ravel(), 0.0, None)
    inp = {"rho": rho, "sigma": sigma}
    ox = fx.compute(inp, do_exc=False, do_vxc=True, do_fxc=fxc)
    oc = fc.compute(inp, do_exc=False, do_vxc=True, do_fxc=fxc)
    out = {"vrho": (ox["vrho"] + oc["vrho"]).ravel(),
           "vsigma": (ox["vsigma"] + oc["vsigma"]).ravel()}
    if fxc:
        out["v2rho2"] = (ox["v2rho2"] + oc["v2rho2"]).ravel()
        out["v2rhosigma"] = (ox["v2rhosigma"] + oc["v2rhosigma"]).ravel()
        out["v2sigma2"] = (ox["v2sigma2"] + oc["v2sigma2"]).ravel()
    return out


# ---------------------------------------------------------------------------
# GGA potential and its linear response
# ---------------------------------------------------------------------------

def xc_potential_gga(rho, recip_lattice):
    """PBE V_xc(r) (Hartree) for a real ground-state density ``rho`` (*grid,
    electrons/Bohr^3). ``V_xc = vrho - 2 div(vsigma grad rho)``."""
    grid = rho.shape
    Gc = _qg_cart(grid, recip_lattice)
    grho = _grad(rho, Gc).real                      # (3,*grid)
    sigma = (grho ** 2).sum(axis=0)
    d = pbe_derivatives(rho, sigma)
    vrho = d["vrho"].reshape(grid)
    vsigma = d["vsigma"].reshape(grid)
    flux = vsigma[None] * grho                      # (3,*grid)
    return (vrho - 2.0 * _div(flux, Gc).real)


def xc_kernel_action(rho, drho, recip_lattice, *, q_frac=(0.0, 0.0, 0.0)):
    """Linear response dV_xc(r) of the PBE potential to a perturbation ``drho``
    (periodic part at wavevector ``q_frac``), at ground-state density ``rho``
    (real, *grid). This is QE's ``dv_of_drho_xc(drho)`` -- for the NLCC term
    pass ``drho = drho_core`` and ``rho = rho_val + rho_core``.

    Returns dV_xc (*grid) complex, Hartree. Validated as the finite-difference
    derivative of `xc_potential_gga` at q=0 (`tests/test_analysis_xc.py`)."""
    grid = rho.shape
    Gc = _qg_cart(grid, recip_lattice)                       # for rho (q=0)
    qGc = _qg_cart(grid, recip_lattice, q_frac)              # for drho (q)
    grho = _grad(rho, Gc).real                               # (3,*grid) real
    sigma = (grho ** 2).sum(axis=0)
    d = pbe_derivatives(rho, sigma, fxc=True)
    vsigma = d["vsigma"].reshape(grid)
    v2rho2 = d["v2rho2"].reshape(grid)
    v2rhosigma = d["v2rhosigma"].reshape(grid)
    v2sigma2 = d["v2sigma2"].reshape(grid)

    gdrho = _grad(drho, qGc)                                 # (3,*grid) complex
    dsigma = 2.0 * (grho * gdrho).sum(axis=0)               # complex
    dvrho = v2rho2 * drho + v2rhosigma * dsigma
    dvsigma = v2rhosigma * drho + v2sigma2 * dsigma
    flux = vsigma[None] * gdrho + dvsigma[None] * grho       # (3,*grid) complex
    return dvrho - 2.0 * _div(flux, qGc)
