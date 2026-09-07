"""
Tests for waw/analysis/topology.py.

Uses the Qi-Wu-Zhang (QWZ) model, the standard minimal 2-band lattice
Chern insulator:

    H(k) = sin(kx) sx + sin(ky) sy + (u + cos(kx) + cos(ky)) sz

(sx, sy, sz Pauli matrices; kx = 2*pi*kx_frac etc., matching this
codebase's Fourier convention exactly, so it maps onto R = (+-1,0,0),
(0,+-1,0), (0,0,0) hoppings with no extra factors needed).

Known phase diagram: Chern number of the lower band is 0 for |u| > 2
(trivial) and +-1 for 0 < |u| < 2, with opposite sign for u > 0 vs
u < 0 (exact sign convention depends on orientation choices we didn't
want to get wrong from memory -- see the mass-tensor cross-term mistake
earlier in this project). So instead of hard-coding an absolute sign,
these tests check convention-independent invariants:

  - integer quantization of the Chern number
  - sum over both bands = 0 (exact identity for any gapped 2-band model)
  - |C| = 1 in the topological regime, 0 in the trivial regime
  - C(u) = -C(-u) (sign flips under mass reversal)
  - independence from the (physically irrelevant, since this model has
    no z-hopping at all) fixed kz plane
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.topology import (
    berry_curvature, chern_number, berry_curvature_cartesian,
    anomalous_hall_conductivity, wannier_interpolated_curvature,
    _nernst_mott_integral, anomalous_nernst_conductivity,
    AnomalousNernstResult, _jjp_jjm_batch, _jjp_jjm_from_occ,
)
from waw.units import BOHR_TO_ANG, EV_TO_HARTREE, K_B_HARTREE, E_CHARGE, HBAR_SI, to_si_units

SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _qwz_hr(u: float) -> HamiltonianR:
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    H_R = np.zeros((len(R_list), 2, 2), dtype=np.complex128)
    H_R[0] = (-1j / 2) * SX + 0.5 * SZ    # R = (+1, 0, 0)
    H_R[1] = (1j / 2) * SX + 0.5 * SZ     # R = (-1, 0, 0)
    H_R[2] = (-1j / 2) * SY + 0.5 * SZ    # R = (0, +1, 0)
    H_R[3] = (1j / 2) * SY + 0.5 * SZ     # R = (0, -1, 0)
    H_R[4] = u * SZ                       # R = (0, 0, 0)

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=2)


@pytest.mark.parametrize("u", [1.0, -1.0, 1.5])
def test_chern_number_quantized_in_topological_regime(u):
    hr = _qwz_hr(u)
    result = chern_number(hr, plane=(0, 1), mesh=(40, 40))
    np.testing.assert_allclose(np.abs(result.chern), [1.0, 1.0], atol=0.02)


@pytest.mark.parametrize("u", [3.0, -3.0])
def test_chern_number_trivial_regime(u):
    hr = _qwz_hr(u)
    result = chern_number(hr, plane=(0, 1), mesh=(40, 40))
    np.testing.assert_allclose(result.chern, [0.0, 0.0], atol=0.02)


def test_chern_numbers_sum_to_zero():
    for u in (1.0, -1.0, 3.0):
        hr = _qwz_hr(u)
        result = chern_number(hr, plane=(0, 1), mesh=(30, 30))
        np.testing.assert_allclose(result.chern.sum(), 0.0, atol=1e-8)


def test_chern_number_sign_flips_with_mass_reversal():
    result_pos = chern_number(_qwz_hr(1.0), plane=(0, 1), mesh=(40, 40))
    result_neg = chern_number(_qwz_hr(-1.0), plane=(0, 1), mesh=(40, 40))
    # Lower band (index 0) should have opposite Chern number under u -> -u.
    np.testing.assert_allclose(result_pos.chern[0], -result_neg.chern[0], atol=0.02)


def test_chern_number_independent_of_fixed_kz():
    """QWZ model has no z-hopping at all, so the result must not depend on
    which kz plane the (kx, ky) torus is evaluated at."""
    hr = _qwz_hr(1.0)
    c0 = chern_number(hr, plane=(0, 1), fixed_value=0.0, mesh=(25, 25))
    c1 = chern_number(hr, plane=(0, 1), fixed_value=0.37, mesh=(25, 25))
    np.testing.assert_allclose(c0.chern, c1.chern, atol=1e-10)


def test_berry_curvature_shape_and_fields():
    hr = _qwz_hr(1.0)
    kpts = np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]])
    result = berry_curvature(hr, kpts, plane=(0, 1))
    assert result.curvature.shape == (2, 2)
    assert result.plane == (0, 1)
    np.testing.assert_allclose(result.kpts, kpts)


# ===========================================================================
# Everywhere-degenerate stress test: two decoupled, IDENTICAL QWZ copies.
#
# The lower/upper energy of copy A and copy B are exactly degenerate at
# *every* k-point (not just an isolated accidental point), so numpy.eigh's
# eigenvectors for the degenerate lower pair are an arbitrary (and, across
# neighbouring mesh points, generally discontinuous) rotation mixing copy
# A and copy B -- a single-band Chern number computed naively from one of
# those eigenvectors would be numerically meaningless. The joint (group)
# Chern number of the whole degenerate pair together must still come out
# exactly right, since FHS is gauge-invariant to any such rotation.
# ===========================================================================

def _doubled_qwz_hr(u: float) -> HamiltonianR:
    R_list = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 0)]
    R_vectors = np.array(R_list, dtype=np.int64)
    degen = np.ones(len(R_list), dtype=np.int64)

    block = [(-1j / 2) * SX + 0.5 * SZ, (1j / 2) * SX + 0.5 * SZ,
             (-1j / 2) * SY + 0.5 * SZ, (1j / 2) * SY + 0.5 * SZ, u * SZ]

    H_R = np.zeros((len(R_list), 4, 4), dtype=np.complex128)
    for i, b in enumerate(block):
        H_R[i, 0:2, 0:2] = b   # copy A
        H_R[i, 2:4, 2:4] = b   # copy B (identical, decoupled)

    return HamiltonianR(H_R=torch.tensor(H_R, dtype=torch.complex128),
                         R_vectors=R_vectors, degen=degen, nw=4)


def test_group_chern_number_robust_to_everywhere_degeneracy():
    hr = _doubled_qwz_hr(1.0)

    single = chern_number(_qwz_hr(1.0), plane=(0, 1), mesh=(30, 30))
    doubled = chern_number(hr, plane=(0, 1), mesh=(30, 30), groups=((0, 1),))

    # The lower 2-dim subspace (copy A + copy B's lower bands, exactly
    # degenerate at every k) must have exactly twice the single-copy
    # lower-band Chern number, regardless of eigh's arbitrary internal gauge.
    np.testing.assert_allclose(doubled.chern[0], 2 * single.chern[0], atol=0.02)


# ===========================================================================
# berry_curvature_cartesian / anomalous_hall_conductivity
# ===========================================================================

def test_berry_curvature_cartesian_isotropic_scaling():
    """
    With an isotropic recip_lattice = gamma*I, the Jacobian is just 1/gamma
    in every direction, so the Cartesian z-component (built from the (x,y)
    fractional plane) must equal the plain fractional Omega_xy rescaled by
    1/gamma**2 -- an explicit, hand-checkable instance of the general
    Omega_cart = inv_recip . inv_recip . Omega_frac transform.
    """
    hr = _qwz_hr(1.0)
    kpts = np.array([[0.1, 0.2, 0.0], [0.3, 0.45, 0.0], [0.05, 0.9, 0.0]])
    gamma = 2.3

    frac = berry_curvature(hr, kpts, plane=(0, 1)).curvature
    cart = berry_curvature_cartesian(hr, kpts, recip_lattice=gamma * np.eye(3)).curvature

    assert cart.shape == (3, 2, 3)
    np.testing.assert_allclose(cart[:, :, 2], frac / gamma ** 2, rtol=1e-10)
    # x, y components vanish identically: this model has no z-hopping, so
    # dH/dk_frac_z = 0 and any curvature tensor component touching z is 0.
    np.testing.assert_allclose(cart[:, :, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(cart[:, :, 1], 0.0, atol=1e-12)


@pytest.mark.parametrize("u", [1.0, -1.0])
def test_ahc_matches_closed_form_prediction_from_chern_number(u):
    """
    Cross-check anomalous_hall_conductivity's k-sum + unit-conversion
    pipeline against an independent closed-form prediction, rather than
    against itself: choose recip_lattice = 2*pi*I / real_lattice = I
    (Bohr, cell volume 1 Bohr^3), and AA_R = 0 (no position-operator
    input -- this synthetic tight-binding model has no ab-initio Mmn
    data to build a real one from), so that J0 = J1 = 0 identically and
    the full formula reduces to J2 alone, for which

        (1/Nk) sum_k Omega_frac_xy,occ(k) --> -2*pi*C   (Nk -> infinity)

    (verified numerically: chern_number's discrete FHS plaquette-loop
    orientation is the opposite handedness from the continuum Kubo-sum
    Omega_xy sign convention -- a real, meaningful sign difference
    between the two methods, not a bug in either one; only the relative
    sign matters for this cross-check). C here is the same quantized
    Chern number checked elsewhere in this file, giving the closed form

        sigma_z = 1e8 * e^2 * C / (2*pi * hbar * BOHR_TO_ANG)

    Fermi energy = 0 sits exactly in the QWZ gap (checked analytically:
    the Bloch vector (sin kx, sin ky, u+cos kx+cos ky) is nonzero on any
    regular mesh for u=1, so no k-point has an eigenvalue pinned at 0),
    so only the lower band is occupied everywhere.
    """
    hr = _qwz_hr(u)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)
    AA_R = torch.zeros(3, len(hr.R_vectors), hr.nw, hr.nw, dtype=torch.complex128)

    C = chern_number(hr, plane=(0, 1), mesh=(60, 60)).chern[0]
    assert abs(abs(C) - 1.0) < 0.02   # sanity: topological regime

    result = anomalous_hall_conductivity(
        hr, AA_R, recip_lattice, real_lattice, fermi_energies=0.0, mesh=(80, 80, 1),
    )
    sigma_si = to_si_units(result.sigma, "hall_conductivity",
                           cell_volume_bohr3=abs(np.linalg.det(real_lattice)))

    # postw90-style curvature-triggered adaptive refinement must leave a
    # smooth, already-converged integrand unchanged (threshold in Ang^2;
    # anything it refines is re-averaged over the sub-mesh, which for the
    # gapped QWZ model reproduces the same quantized answer)
    result_adpt = anomalous_hall_conductivity(
        hr, AA_R, recip_lattice, real_lattice, fermi_energies=0.0, mesh=(80, 80, 1),
        curv_adpt_kmesh=3, curv_adpt_thresh_ang2=1.0,
    )
    sigma_adpt = to_si_units(result_adpt.sigma, "hall_conductivity",
                             cell_volume_bohr3=abs(np.linalg.det(real_lattice)))
    np.testing.assert_allclose(sigma_adpt[0, 2], sigma_si[0, 2], rtol=0.02)

    sigma_z_expected = 1.0e8 * E_CHARGE ** 2 * C / (2 * np.pi * HBAR_SI * BOHR_TO_ANG)

    assert result.sigma.shape == (1, 3)
    np.testing.assert_allclose(sigma_si[0, 2], sigma_z_expected, rtol=0.05)
    # No x/z-hopping in this model => no yz/zx curvature => those AHC
    # components must vanish (up to the finite-mesh Kubo-sum noise floor).
    np.testing.assert_allclose(sigma_si[0, 0], 0.0, atol=1e-6 * abs(sigma_z_expected))
    np.testing.assert_allclose(sigma_si[0, 1], 0.0, atol=1e-6 * abs(sigma_z_expected))


def test_ahc_fermi_energy_scan_and_sign_flip():
    """A Fermi level scan should track: deep below both bands -> 0 (nothing
    occupied), in the gap -> the topological value, deep above both bands
    -> 0 again (both bands occupied, contributions cancel per the
    sum-to-zero identity already checked for chern_number). The QWZ
    bandwidth is |vector| <= sqrt(2**2 + 3**2) in Hartree-like units, so
    +-500 Hartree is safely outside both bands."""
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)
    AA_R = torch.zeros(3, len(hr.R_vectors), hr.nw, hr.nw, dtype=torch.complex128)

    result = anomalous_hall_conductivity(
        hr, AA_R, recip_lattice, real_lattice,
        fermi_energies=[-500.0, 0.0, 500.0], mesh=(50, 50, 1),
    )
    assert result.sigma.shape == (3, 3)
    np.testing.assert_allclose(result.sigma[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.sigma[2], 0.0, atol=1e-8)
    assert abs(result.sigma[1, 2]) > 1e-6


def test_occupied_curvature_sum_equals_J2_term():
    """
    Closed-form identity, checked pointwise (no mesh/Riemann-sum
    convergence involved, so this should hold to ~machine precision):
    summed over the occupied bands, the plain band-Kubo curvature
    (`berry_curvature_cartesian`, "no position operator" formula) is
    EXACTLY the "J2" term of the WYSV06/CTVR06 decomposition.

    Proof sketch (see berry_curvature_cartesian's docstring): the full
    per-band Kubo sum runs over ALL other bands m != n. Splitting that
    sum by occupation and summing over occupied n, the occ-occ cross
    terms cancel pairwise by the Hermiticity of dH/dk (term(o,o') =
    -term(o',o) once E_o != E_o'), leaving only occ-unocc cross terms --
    which are algebraically identical (same energy denominator squared)
    to J2's own occ-unocc sum. With AA_R = 0, wannier_interpolated_curvature
    reduces to J0 (=0, needs AA_R) + J1 (=0, needs AA_R) + J2, so it
    should reproduce the occupied sum exactly.
    """
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)
    AA_R = torch.zeros(3, len(hr.R_vectors), hr.nw, hr.nw, dtype=torch.complex128)
    fermi_energy = 0.0

    kpts = np.array([[0.13, 0.27, 0.0], [0.41, 0.05, 0.0], [0.7, 0.83, 0.0]])

    cart = berry_curvature_cartesian(hr, kpts, recip_lattice).curvature   # (nk, nw, 3)
    from waw.core.hamiltonian import interpolate_bands
    bands = interpolate_bands(hr, kpts)
    occ_mask = bands < fermi_energy
    occ_sum = np.einsum('kn,knc->kc', occ_mask.astype(np.float64), cart)   # (nk, 3)

    full = wannier_interpolated_curvature(hr, AA_R, recip_lattice, real_lattice, kpts, fermi_energy)
    full = full[:, 0, :]   # (nk, 3), single Fermi energy

    np.testing.assert_allclose(full, occ_sum, atol=1e-10)


# ===========================================================================
# Anomalous Nernst effect (transverse thermoelectric conductivity, Mott)
# ===========================================================================

def test_nernst_mott_matches_sommerfeld_for_linear_curvature():
    """
    The atomic-units Mott integral must reproduce the closed-form
    Sommerfeld limit ``I = (pi^2/3) kT^2 dsigma/dE`` (the `-(1/(eT))`
    physical prefactor is NOT part of `_nernst_mott_integral`'s own
    return value -- see its docstring -- it's applied later by
    `to_si_units("anomalous_nernst", ...)`, which needs actual Kelvin).
    For a *linear* curvature_xy(E) = a + b (E - mu) the Sommerfeld
    expansion is exact (all higher derivatives vanish), so numeric ==
    analytic to machine precision at any kT.
    """
    mu, a, b = 0.15, 2.0, -8.0            # Hartree, Bohr^2, slope Bohr^2/Hartree
    # wide grid so the (E-mu)^2 (-df/dE) tail is not truncated even at the largest kT
    E = np.linspace(mu - 0.5, mu + 0.5, 801)
    curvature = (a + b * (E - mu))[:, None]

    for kT in (0.002, 0.006, 0.02):    # Hartree (~630 K, ~1900 K, ~6300 K)
        I_num = _nernst_mott_integral(E, curvature, mu, kT)[0]
        I_exact = (np.pi**2 / 3.0) * kT**2 * b
        # residual is pure numerical-integration error (the identity is exact)
        np.testing.assert_allclose(I_num, I_exact, rtol=1e-4)


def test_nernst_mott_low_T_kernel_on_a_coarse_grid():
    """
    The blind spot of the linear test above: for a CURVED sigma(E) at a kT
    much smaller than the grid spacing, a plain trapezoid over the coarse
    grid samples the (E-mu)(-df/dE) kernel with 1-3 points per lobe and
    returns grid-alignment noise. The spline resampling inside
    `_nernst_mott_integral` must recover the closed form: for
    sigma(E) = c (E-mu)^3,

        I = c * integral x^3 (E-mu)(-df/dE) = c * (7 pi^4 / 15) kT^4.

    Grid: 10 meV-ish spacing (0.0004 Ha); kT down to 25 K-ish (1e-4 Ha),
    i.e. spacing = 4 kT -- exactly the notebook-07 regime that was silently
    wrong before 2026-07-27.
    """
    mu, c = 0.15, 5.0e3
    E = np.linspace(mu - 0.015, mu + 0.015, 76)      # spacing 0.0004 Ha ~ 11 meV
    curvature = (c * (E - mu) ** 3)[:, None]

    for kT in (1e-4, 2.5e-4, 1e-3):                  # ~32 K, 79 K, 316 K
        I_num = _nernst_mott_integral(E, curvature, mu, kT)[0]
        I_exact = c * (7.0 * np.pi ** 4 / 15.0) * kT ** 4
        np.testing.assert_allclose(I_num, I_exact, rtol=2e-3)


def test_nernst_mott_energy_independent_curvature_gives_zero():
    """A constant curvature_xy(E) has zero energy derivative -> zero Nernst
    signal (the (E-mu)(-df/dE) kernel is odd about mu)."""
    E = np.linspace(0.14, 0.16, 401)
    curvature = np.full((len(E), 1), 3.5)
    I_num = _nernst_mott_integral(E, curvature, 0.15, 0.01)[0]
    assert abs(I_num) < 1e-9


def test_anomalous_nernst_conductivity_end_to_end_on_qwz():
    """
    End-to-end alpha^A(mu, T) on the QWZ model: shapes, the sigma^A(E) curve
    it used, and consistency with a direct Mott integral of that same curve.
    (QWZ is 2D so only the z/xy component is nonzero.)
    """
    hr = _qwz_hr(1.0)
    real_lattice = np.eye(3)
    recip_lattice = 2 * np.pi * np.eye(3)
    AA_R = torch.zeros(3, len(hr.R_vectors), hr.nw, hr.nw, dtype=torch.complex128)

    kT_values = [100.0 * K_B_HARTREE, 300.0 * K_B_HARTREE]
    res = anomalous_nernst_conductivity(
        hr, AA_R, recip_lattice, real_lattice,
        mu=0.0, kT_values=kT_values, mesh=(24, 24, 1),
        energy_halfwidth=0.3 * EV_TO_HARTREE, n_energies=41,
    )
    assert isinstance(res, AnomalousNernstResult)
    assert res.alpha.shape == (2, 3)
    assert res.sigma_of_E.shape == (41, 3)
    # reproduces a direct Mott integral of the returned sigma^A(E)
    direct = _nernst_mott_integral(res.energies, res.sigma_of_E, 0.0, kT_values[1])
    np.testing.assert_allclose(res.alpha[1], direct, rtol=1e-10)


def test_anomalous_nernst_to_si_units():
    """to_si_units("anomalous_nernst") reproduces the original eV/S-cm/Kelvin
    Mott formula exactly, for a synthetic (non-QWZ) case where the answer
    can be computed both ways independently."""
    cell_volume_bohr3 = 123.4
    kT_values = np.array([200.0 * K_B_HARTREE, 400.0 * K_B_HARTREE])
    alpha_atomic = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # Hartree*Bohr^2

    got = to_si_units(alpha_atomic, "anomalous_nernst",
                       cell_volume_bohr3=cell_volume_bohr3, kT_values=kT_values)

    from waw.units import HARTREE_TO_EV
    from waw.analysis.topology import _hall_conductivity_si_factor
    K_scm = BOHR_TO_ANG ** 2 * _hall_conductivity_si_factor(cell_volume_bohr3)
    T_kelvin = kT_values / K_B_HARTREE
    expected = (-100.0 * HARTREE_TO_EV / T_kelvin)[:, None] * K_scm * alpha_atomic

    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_jjp_jjm_from_occ_matches_threshold_form():
    """
    `_jjp_jjm_from_occ` (postw90 gyrotropic's generalization, needed for its
    per-band "fake occupation" trick) must reproduce `_jjp_jjm_batch` EXACTLY
    when `occ` is built the ordinary way, `occ = (eig < fermi)` -- the two
    are the same formula, re-derived from an explicit occupation array
    instead of a Fermi-energy threshold (see `_jjp_jjm_from_occ`'s docstring).
    """
    torch.manual_seed(0)
    nk, nw = 5, 4
    hermitian = torch.randn(nk, nw, nw, dtype=torch.complex128)
    hermitian = hermitian + hermitian.conj().transpose(-1, -2)
    eig, UU = torch.linalg.eigh(hermitian)

    dH_eig = torch.randn(nk, 3, nw, nw, dtype=torch.complex128)
    dH_eig = dH_eig + dH_eig.conj().transpose(-1, -2)   # Hermitian, as the real quantity is

    for fe in (-1.5, 0.0, 2.3):
        JJp_ref, JJm_ref = _jjp_jjm_batch(dH_eig, eig, float(fe))
        occ = (eig < fe).to(torch.float64)
        JJp_new, JJm_new = _jjp_jjm_from_occ(dH_eig, eig, occ)
        torch.testing.assert_close(JJp_new, JJp_ref)
        torch.testing.assert_close(JJm_new, JJm_ref)
