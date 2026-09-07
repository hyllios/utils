"""
The Akashi-Arita renormalisation kernel, Phys. Rev. B 88, 014514 (2013)
[arXiv:1305.0390], Eqs. (40)-(44): SCDFT without particle-hole symmetrisation.

The paper states three properties of Z^ph,new, and they are falsifiable, so they
are the tests:

  (i)   lim_{xi->0} Z^new(xi) = lim_{xi->0} Z^ph(xi) ~ lambda
  (ii)  its temperature dependence resembles Z^ph's
  (iii) for a symmetric DOS it reduces to the previous symmetrised kernel

plus the claim the paper is actually FOR: with an asymmetric DOS the kernel
becomes asymmetric in xi and is larger where the DOS is larger (their Fig. 3),
which the symmetrised kernel cannot represent at all.

A SIGN SLIP IN THE PAPER, resolved here by measurement rather than assumption.
Eq. (40) carries +1/tanh where the previous kernel [Eq. (39), Luders Eq. (79)]
carries -1/tanh, and property (iii) says the xi'-symmetric part of [I - 2J]
"agrees with J(xi,xi') + J(xi,-xi')". Since `i_akashi` is odd in xi', that
symmetric part is -[J_ak(xi') + J_ak(-xi')], so the two statements together give
Z^new = -Z^ph, contradicting property (i). Implemented verbatim from Eq. (40),
property (i) HOLDS and the reduction (iii) holds to four digits -- so the +1/tanh
is right and the minus sign is missing from the statement of (iii). This is the
same family of convention mismatch already noted in `solver.py` between LM2005
Eq. (78) and Sanna-Pellegrini-Gross Eq. (10).
"""

import numpy as np
import pytest

from waw.units import EV_TO_HARTREE, K_B_HARTREE
from waw.utils.scdft.functions import (i_akashi, j_akashi, j_lueders, p_smooth)
from waw.utils.scdft.solver import _omega_nodes, energy_grid, z_kernel

LAMBDA = 1.0
W_D = 400.0 / 219474.6313705          # 400 cm^-1 in Hartree, their model
SIGMA = W_D / 10.0                    # their width


def _model_a2f(lam=LAMBDA):
    """Their Eq. (46): a2F = (lam/2 sigma sqrt(pi)) w exp[-((w-wD)/sigma)^2],
    renormalised so 2*int a2F/w is exactly lam."""
    w = np.linspace(W_D - 5 * SIGMA, W_D + 5 * SIGMA, 201)
    a2f = lam / (2 * SIGMA * np.sqrt(np.pi)) * w * np.exp(-((w - W_D) / SIGMA) ** 2)
    a2f *= lam / (2 * np.trapezoid(a2f / w, w))
    return _omega_nodes(w, a2f[None, None, :])


def _grid(T):
    return energy_grid(K_B_HARTREE * T, 200 * W_D, 400)


def _step_dos(xi, ef_off, ratio=6.0, d=W_D / 2):
    """Their step-like model DOS: N+/N- = ratio over a width d, N flat beyond
    5 eV, normalised to N(0) = 1."""
    s = np.clip(0.5 + (xi - ef_off) / d, 0.0, 1.0)
    N = 1.0 + (ratio - 1.0) * s
    N = np.where(np.abs(xi) > 5.0 * EV_TO_HARTREE, np.interp(0.0, xi, N), N)
    return N / np.interp(0.0, xi, N)


def _z(functional, T, dos_ratio=None, lam=LAMBDA):
    om, a2f_w = _model_a2f(lam)
    xi, wq = _grid(T)
    z = z_kernel(xi, wq, om, a2f_w, 1.0 / (K_B_HARTREE * T),
                 functional=functional, dos_ratio=dos_ratio)[:, 0]
    return xi, z


# --------------------------------------------------------------- the kernels

def test_i_akashi_is_odd_in_xi_prime():
    """Eq. (41) is antisymmetric in xi' by construction. Consequence, and the
    reason the new kernel needs an asymmetric DOS to do anything: for a
    symmetric N(xi') the whole I term integrates to zero."""
    beta = 1.0 / (K_B_HARTREE * 10.0)
    xi, xip, w = 0.3 * W_D, 0.7 * W_D, W_D
    a = i_akashi(xi, xip, w, beta)
    b = i_akashi(xi, -xip, w, beta)
    assert np.isclose(a, -b, rtol=1e-12, atol=0.0)


def test_j_akashi_reduces_to_lueders_at_the_removable_pole():
    """At xi' = xi - w the extra xi/(xi'+w) factor is exactly 1, so Akashi's
    Jt must coincide with Luders' there -- this is the cancellation their
    paper proves, checked pointwise."""
    beta = 1.0 / (K_B_HARTREE * 10.0)
    w = W_D
    for xi in (0.5 * W_D, 2.0 * W_D, -1.3 * W_D):
        xip = xi - w                              # => xi'+w = xi, ratio = 1
        # compare the single-branch pieces: J = Jt(w) - Jt(-w) for both, and at
        # this xi' only the +w branch sits at the removable pole
        ja = j_akashi(xi, xip, w, beta)
        jl = j_lueders(xi, xip, w, beta)
        assert np.isfinite(ja) and np.isfinite(jl)
        assert abs(ja - jl) < 0.05 * max(abs(jl), 1e-12) + 1e-9, (xi, ja, jl)


@pytest.mark.parametrize("T", [0.01, 1.0, 10.0, 100.0])
def test_kernels_are_finite_everywhere_including_coincidences(T):
    """The paper's central claim is numerical stability with no divergence down
    to the low-temperature limit. Sweep xi' across every coincidence:
    xi' = +-xi +- w, +-w, and xi = 0."""
    beta = 1.0 / (K_B_HARTREE * T)
    xi = np.array([0.0, 1e-12, 0.3 * W_D, W_D, 3 * W_D, -0.3 * W_D, -W_D])
    xip = np.concatenate([
        np.linspace(-4 * W_D, 4 * W_D, 401),
        np.array([W_D, -W_D, 0.0, 0.3 * W_D - W_D, 0.3 * W_D + W_D]),
    ])
    for w in (W_D, 0.5 * W_D):
        I = i_akashi(xi[:, None], xip[None, :], w, beta)
        J = j_akashi(xi[:, None], xip[None, :], w, beta)
        assert np.isfinite(I).all(), f"I not finite at T={T}, w={w}"
        assert np.isfinite(J).all(), f"J not finite at T={T}, w={w}"


def test_p_smooth_is_even_saturating_and_vanishing():
    beta = 1.0 / (K_B_HARTREE * 10.0)
    x = np.array([-1.0, -0.1, 0.1, 1.0]) * W_D
    assert np.allclose(p_smooth(x, beta), p_smooth(-x, beta))
    assert p_smooth(0.0, beta) == 0.0
    assert np.isclose(p_smooth(10.0 * W_D, beta), 1.0, atol=1e-12)


# ------------------------------------------------- the paper's properties

@pytest.mark.parametrize("T", [0.01, 1.0, 10.0])
def test_property_i_z_at_ef_is_lambda(T):
    """(i) Z^new(xi->0) ~ lambda, the value Luders et al. call desirable.
    This is also what fixes the +1/tanh sign of Eq. (40)."""
    xi, z = _z("akashi", T)
    z0 = z[xi > 0][0]
    assert 0.9 * LAMBDA < z0 < 1.05 * LAMBDA, (T, z0)


@pytest.mark.parametrize("T", [0.01, 1.0, 10.0])
def test_property_iii_reduces_to_lm2005_for_symmetric_dos(T):
    """(iii) With a particle-hole symmetric DOS the new kernel must reproduce
    the old symmetrised one. It does so essentially exactly, because the I term
    is odd in xi' and drops out."""
    xi, za = _z("akashi", T)
    _, zl = _z("lm2005", T)
    p = xi > 0
    assert np.isclose(za[p][0], zl[p][0], rtol=2e-3), (T, za[p][0], zl[p][0])
    # and across the whole range, not just at E_F
    scale = np.abs(zl).max()
    assert np.abs(za - zl).max() < 5e-3 * scale, (T, np.abs(za - zl).max(), scale)


def test_property_ii_temperature_dependence_tracks_lm2005():
    """(ii) Similar T dependence. Both fall with T; the ratio stays put."""
    r = []
    for T in (0.01, 1.0, 10.0, 50.0):
        xi, za = _z("akashi", T)
        _, zl = _z("lm2005", T)
        p = xi > 0
        r.append(za[p][0] / zl[p][0])
    assert max(r) - min(r) < 5e-3, r


# --------------------------------------------- what the paper is actually for

def test_asymmetric_dos_makes_the_new_kernel_asymmetric():
    """The point of the paper. With their step DOS the new kernel is asymmetric
    in xi while the symmetrised one is EXACTLY symmetric and cannot represent
    the effect at all."""
    T = 10.0
    xi, za = _z("akashi", T, dos_ratio=None)
    dr = _step_dos(xi, -0.3 * W_D)
    xi, za = _z("akashi", T, dos_ratio=dr)
    _, zl = _z("lm2005", T, dos_ratio=dr)
    im, ip = int(np.argmin(np.abs(xi + W_D))), int(np.argmin(np.abs(xi - W_D)))

    def asym(z):
        return (z[ip] - z[im]) / (z[ip] + z[im])

    assert abs(asym(zl)) < 1e-10, f"lm2005 should be exactly symmetric, got {asym(zl)}"
    assert abs(asym(za)) > 0.05, f"akashi should be asymmetric, got {asym(za)}"
    # their Fig. 3: Z is LARGER where the DOS is larger (here at +xi)
    assert za[ip] > za[im]
    assert dr[ip] > dr[im]


def test_property_i_survives_an_asymmetric_dos():
    """Their stronger statement: lim_{xi->0}[Z^new - Z^ph] ~ 0 even when the DOS
    is strongly asymmetric, so the asymmetry shows up away from E_F, not at it."""
    T = 10.0
    xi, _ = _z("akashi", T)
    dr = _step_dos(xi, -0.3 * W_D)
    _, za = _z("akashi", T, dos_ratio=dr)
    _, zl = _z("lm2005", T, dos_ratio=dr)
    p = xi > 0
    assert np.isclose(za[p][0], zl[p][0], rtol=5e-3), (za[p][0], zl[p][0])


def test_dos_ratio_default_matches_explicit_ones():
    """`dos_ratio=None` must be exactly a constant DOS, so adding the weight
    cannot have changed any existing result."""
    T = 10.0
    xi, za = _z("akashi", T)
    _, zb = _z("akashi", T, dos_ratio=np.ones_like(xi))
    assert np.array_equal(za, zb)
    xi, zl = _z("lm2005", T)
    _, zm = _z("lm2005", T, dos_ratio=np.ones_like(xi))
    assert np.array_equal(zl, zm)


def test_z_kernel_rejects_bad_dos_ratio_and_functional():
    T = 10.0
    om, a2f_w = _model_a2f()
    xi, wq = _grid(T)
    beta = 1.0 / (K_B_HARTREE * T)
    with pytest.raises(ValueError, match="dos_ratio must match"):
        z_kernel(xi, wq, om, a2f_w, beta, "akashi", dos_ratio=np.ones(3))
    with pytest.raises(ValueError, match="functional must be"):
        z_kernel(xi, wq, om, a2f_w, beta, "nope")
