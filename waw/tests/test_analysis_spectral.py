"""
Tests for waw/analysis/spectral.py and waw/analysis/elph_selfenergy.py.

spectral.py
  1. Sigma = None gives Lorentzians centred on the band energies, with the
     sum rule int A dE = nw.
  2. A constant real Sigma shifts every peak by exactly that amount.
  3. A constant imaginary Sigma sets the width: HWHM = eta + |Im Sigma|.
  4. A local Sigma and the same Sigma broadcast over k agree exactly.
  5. Wrong shapes are rejected with a message naming the expected ones.
  6. quasiparticle_shift on a linear Sigma = a + b E reproduces the analytic
     Z = 1/(1-b) and E_qp = eps + Z(a + b eps).

elph_selfenergy.py
  7. One flat band + one Einstein mode + constant coupling is analytic:
     Sigma(w) = g2 [ f/(w - e0 + w0 + i eta) + (1-f)/(w - e0 - w0 + i eta) ].
     Checked against the closed form, which pins the mode prefactor
     1/sqrt(2 M w0), the Bose/Fermi factors and the branch signs at once.
  8. Retardation: Im Sigma <= 0 everywhere (the numerators n+f and n+1-f are
     non-negative and every denominator carries +i eta).
  9. lambda = -dRe Sigma/dw > 0, and for the analytic model equals
     2 g2 / w0^2 at w = e0.
 10. Sigma vanishes with the coupling.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.spectral import (bloch_spectral_function,
                                   quasiparticle_shift)
from waw.analysis.elph_selfenergy import (fan_migdal_self_energy,
                                          mass_enhancement)
from waw.core.distributions import bose_einstein, fermi_dirac
from waw.units import AMU_TO_ME


def _flat_hr(nw=1, e0=0.0):
    """nw-orbital Hamiltonian with only an on-site block: flat bands at e0."""
    H = np.zeros((1, nw, nw), dtype=np.complex128)
    H[0] = np.diag(e0 + 0.01 * np.arange(nw))
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=np.zeros((1, 3), dtype=np.int64),
                        degen=np.ones(1, dtype=np.int64), nw=nw)


def _dispersive_hr(nw=2, t=0.05, a=5.0):
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64)
    H = np.zeros((3, nw, nw), dtype=np.complex128)
    H[0] = np.diag(np.linspace(-0.1, 0.1, nw))
    H[1] = -t * np.eye(nw)
    H[2] = -t * np.eye(nw)
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=R,
                        degen=np.ones(3, dtype=np.int64), nw=nw,
                        real_lattice=a * np.eye(3))


# ---------------------------------------------------------------- spectral

def test_bare_spectral_function_peaks_and_sum_rule():
    hr = _dispersive_hr(nw=2)
    k = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    E = np.linspace(-0.6, 0.6, 4001)
    eta = 2e-3
    sf = bloch_spectral_function(hr, k, E, None, eta=eta)
    from waw.core.hamiltonian import operator_k
    Hk = operator_k(hr.H_R, hr.R_vectors, hr.degen, k).detach().cpu().numpy()
    # A Lorentzian's tail outside a finite window carries real weight, so the
    # sum rule on [Emin, Emax] is not nw -- it is the analytic partial
    # integral, which is a sharper test than nw with a loose tolerance.
    for ik in range(len(k)):
        ev = np.linalg.eigvalsh(Hk[ik])
        want = sum((np.arctan((E[-1] - e) / eta)
                    - np.arctan((E[0] - e) / eta)) / np.pi for e in ev)
        assert abs(np.trapezoid(sf.A[ik], E) - want) < 1e-5
        peak = E[np.argmax(sf.A[ik])]
        assert np.min(np.abs(ev - peak)) < 5e-3


def test_constant_real_self_energy_shifts_peaks():
    hr = _dispersive_hr(nw=1)
    k = np.array([[0.1, 0.2, 0.3]])
    E = np.linspace(-0.6, 0.6, 6001)
    shift = 0.137
    A0 = bloch_spectral_function(hr, k, E, None, eta=1e-3).A
    sig = np.full((len(E), 1, 1), shift + 0j)
    A1 = bloch_spectral_function(hr, k, E, sig, eta=1e-3).A
    assert abs((E[np.argmax(A1[0])] - E[np.argmax(A0[0])]) - shift) < 3e-4


def test_imaginary_self_energy_sets_the_width():
    hr = _flat_hr(nw=1, e0=0.0)
    k = np.array([[0.0, 0.0, 0.0]])
    E = np.linspace(-0.2, 0.2, 20001)
    eta, gam = 1e-4, 5e-3
    sig = np.full((len(E), 1, 1), -1j * gam)
    A = bloch_spectral_function(hr, k, E, sig, eta=eta).A[0]
    half = A.max() / 2
    above = E[A >= half]
    hwhm = 0.5 * (above.max() - above.min())
    assert abs(hwhm - (eta + gam)) < 2e-4


def test_local_and_kresolved_self_energy_agree():
    hr = _dispersive_hr(nw=2)
    k = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.0], [0.5, 0.5, 0.0]])
    E = np.linspace(-0.5, 0.5, 301)
    rng = np.random.default_rng(0)
    loc = (rng.normal(size=(len(E), 2, 2))
           - 1j * rng.uniform(1e-4, 1e-2, size=(len(E), 2, 2))) * 0.01
    loc = 0.5 * (loc + loc.conj().transpose(0, 2, 1)) - 1j * 1e-3 * np.eye(2)
    A_loc = bloch_spectral_function(hr, k, E, loc).A
    A_k = bloch_spectral_function(
        hr, k, E, np.repeat(loc[:, None], len(k), axis=1)).A
    assert np.allclose(A_loc, A_k, rtol=0, atol=1e-12)


def test_self_energy_shape_is_validated():
    hr = _dispersive_hr(nw=2)
    k = np.zeros((3, 3))
    E = np.linspace(-0.5, 0.5, 11)
    with pytest.raises(ValueError, match="expected"):
        bloch_spectral_function(hr, k, E, np.zeros((11, 5, 5), dtype=complex))


def test_quasiparticle_shift_linear_self_energy():
    eig = np.array([[-0.1, 0.05]])
    E = np.linspace(-0.5, 0.5, 2001)
    a, b = 0.02, -0.3
    sig = (a + b * E)[:, None, None] * np.ones((1, 1, 2)) + 0j
    E_qp, Z = quasiparticle_shift(eig, sig, E)
    Z_exact = 1.0 / (1.0 - b)
    assert np.allclose(Z, Z_exact, rtol=1e-6)
    assert np.allclose(E_qp, eig + Z_exact * (a + b * eig), atol=1e-6)
    # the exact root of E = eps + Re Sigma(E) is available in closed form here
    E_root, _ = quasiparticle_shift(eig, sig, E, linear=False)
    assert np.allclose(E_root, (eig + a) / (1.0 - b), atol=1e-5)


# --------------------------------------------------------- elph self-energy

W0 = 5.0e-3          # Hartree, Einstein mode, well above EPS_ACOUSTIC
MASS = 20.0          # amu
E0 = 0.0             # flat band
CVAL = 3.0e-3        # Wannier-gauge coupling, Hartree/Bohr


def _einstein_inputs(coupling=CVAL, nq=4):
    """One orbital, one atom, one Einstein frequency, q-independent coupling."""
    hr = _flat_hr(nw=1, e0=E0)
    R_e = np.zeros((1, 3), dtype=np.int64)
    R_q = np.zeros((1, 3), dtype=np.int64)
    g_R = np.full((1, 1, 3, 1, 1), coupling, dtype=np.complex128)
    degen_e = np.ones(1, dtype=np.int64)
    degen_q = np.ones(1, dtype=np.int64)
    rng = np.random.default_rng(1)
    qpts = rng.uniform(size=(nq, 3))
    omega = np.full((nq, 3), W0)
    evec = np.broadcast_to(np.eye(3), (nq, 3, 3)).copy().astype(np.complex128)
    return (hr, g_R, R_e, degen_e, R_q, degen_q, qpts, omega, evec,
            np.array([MASS]), np.array([0]))


def _analytic_g2():
    """3 modes x |c|^2 / (2 M w0), the mode prefactor squared."""
    return 3.0 * CVAL ** 2 / (2.0 * MASS * AMU_TO_ME * W0)


def test_fan_migdal_matches_the_einstein_closed_form():
    hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty = _einstein_inputs()
    E = np.linspace(-0.05, 0.05, 501)
    eta = 2e-4
    EF = 0.02                       # above the band: the level is occupied
    se = fan_migdal_self_energy(
        np.zeros((1, 3)), hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty,
        E, fermi_energy=EF, temperature=0.0, eta=eta)
    g2 = _analytic_g2()
    f = 1.0                         # e0 < EF
    want = g2 * (f / (E - E0 + W0 + 1j * eta)
                 + (1.0 - f) / (E - E0 - W0 + 1j * eta))
    assert np.allclose(se.sigma[:, 0, 0], want, rtol=1e-10, atol=1e-14)


def test_fan_migdal_empty_band_takes_the_other_branch():
    hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty = _einstein_inputs()
    E = np.linspace(-0.05, 0.05, 501)
    eta = 2e-4
    EF = -0.02                      # below the band: the level is empty
    se = fan_migdal_self_energy(
        np.zeros((1, 3)), hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty,
        E, fermi_energy=EF, temperature=0.0, eta=eta)
    want = _analytic_g2() / (E - E0 - W0 + 1j * eta)
    assert np.allclose(se.sigma[:, 0, 0], want, rtol=1e-10, atol=1e-14)


def test_imaginary_part_is_never_positive():
    hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty = _einstein_inputs()
    E = np.linspace(-0.05, 0.05, 401)
    for T in (0.0, 300.0):
        se = fan_migdal_self_energy(
            np.zeros((1, 3)), hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m,
            ty, E, fermi_energy=0.0, temperature=T, eta=1e-4)
        assert se.sigma.imag.max() <= 1e-18


def test_mass_enhancement_matches_the_analytic_slope():
    hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty = _einstein_inputs()
    E = np.linspace(-0.05, 0.05, 4001)
    se = fan_migdal_self_energy(
        np.zeros((1, 3)), hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty,
        E, fermi_energy=E0, temperature=0.0, eta=1e-5)
    # at w = e0 with f = 1/2 the two branches contribute g2/(2 w0^2) each
    lam = mass_enhancement(se, E0)[0, 0]
    assert lam > 0
    assert abs(lam - _analytic_g2() / W0 ** 2) < 0.02 * _analytic_g2() / W0 ** 2


def test_zero_coupling_gives_zero_self_energy():
    hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty = _einstein_inputs(
        coupling=0.0)
    E = np.linspace(-0.05, 0.05, 51)
    se = fan_migdal_self_energy(
        np.zeros((1, 3)), hr, g_R, R_e, deg_e, R_q, deg_q, q, om, ev, m, ty,
        E, fermi_energy=0.0, temperature=0.0, eta=1e-4)
    assert np.allclose(se.sigma, 0.0)


def test_shared_occupation_functions_at_zero_temperature():
    """elph_selfenergy uses core.distributions rather than its own copies;
    the T = 0 limits are the part that has to be a limit and not a 0/0."""
    w = np.array([1e-3, 5e-3])
    assert np.allclose(bose_einstein(w, 0.0), 0.0)
    assert bose_einstein(w, 1e-3)[0] > bose_einstein(w, 1e-3)[1]
    e = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(fermi_dirac(e, 0.0, 0.0), [1.0, 0.5, 0.0])
    assert abs(fermi_dirac(np.array([0.0]), 0.0, 1e-3)[0] - 0.5) < 1e-12


def test_fermi_surface_average_warns_when_dominated_by_outliers():
    """lambda_nk is heavy-tailed (no upper bound as an intermediate state
    approaches eF with a soft mode), so a sparse sample is dominated by
    outliers and the average is not converged. That regime must announce
    itself rather than return a plausible-looking number."""
    import warnings as _w
    from waw.analysis.elph_selfenergy import (ElphSelfEnergy,
                                              fermi_surface_average_lambda)
    nE, nk, nb = 61, 200, 1
    E = np.linspace(-0.03, 0.03, nE)
    eig = np.linspace(-2e-3, 2e-3, nk).reshape(nk, nb)
    # a self-energy whose slope at w=0 is huge for one state and small for
    # the rest: Sigma = -c_nk * w reproduces lambda = c_nk exactly
    c = np.full((nk, nb), 0.5)
    c[nk // 2] = 4000.0
    sig = (-E[:, None, None] * c[None]).astype(np.complex128)
    se = ElphSelfEnergy(kpts=np.zeros((nk, 3)), energies=E, sigma=sig,
                        eig=eig, temperature=0.0, eta=1e-5)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        fermi_surface_average_lambda(se, 0.0, 1e-2)
    assert any("dominated by a handful" in str(r.message) for r in rec)

    c[nk // 2] = 0.5                       # homogeneous: no warning
    sig = (-E[:, None, None] * c[None]).astype(np.complex128)
    se = ElphSelfEnergy(kpts=np.zeros((nk, 3)), energies=E, sigma=sig,
                        eig=eig, temperature=0.0, eta=1e-5)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        lam = fermi_surface_average_lambda(se, 0.0, 1e-2)
    assert not rec
    assert abs(lam - 0.5) < 1e-6


def test_mass_enhancement_refuses_finite_temperature():
    """-dReSigma/dw is a T = 0 quantity: at finite T the Bose factor makes
    the real-axis poles live and the derivative diverges as eta -> 0 instead
    of converging. Returning a number there would be worse than refusing."""
    from waw.analysis.elph_selfenergy import ElphSelfEnergy, mass_enhancement
    E = np.linspace(-0.02, 0.02, 21)
    se = ElphSelfEnergy(kpts=np.zeros((2, 3)), energies=E,
                        sigma=np.zeros((21, 2, 1), dtype=complex),
                        eig=np.zeros((2, 1)), temperature=300.0, eta=1e-4)
    with pytest.raises(ValueError, match="zero-temperature quantity"):
        mass_enhancement(se, 0.0)
    se_0 = ElphSelfEnergy(kpts=np.zeros((2, 3)), energies=E,
                          sigma=np.zeros((21, 2, 1), dtype=complex),
                          eig=np.zeros((2, 1)), temperature=0.0, eta=1e-4)
    assert np.allclose(mass_enhancement(se_0, 0.0), 0.0)
