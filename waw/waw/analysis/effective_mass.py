"""
Band-extremum search and effective-mass tensor analysis for semiconductors.

Given a converged Wannier Hamiltonian H(R), locates the valence-band
maxima and conduction-band minima over the Brillouin zone and computes
the effective-mass tensor at each extremum from a numerical Hessian of
the interpolated band energy (central finite differences in Cartesian
k-space, Bohr^-1).

The extremum is located per band index (i.e. per sorted eigenvalue of
H(k), as returned by `interpolate_bands`). If a target band is
near-degenerate with any other band at the extremum (e.g. the unsplit
heavy-/light-hole valence top of a zinc-blende semiconductor without
spin-orbit coupling), `BandExtremum.degenerate_with` flags it — the
single mass tensor from `analyze_effective_mass` mixes the group via the
sorted-eigenvalue assignment and shouldn't be trusted there. Use
`degenerate_effective_mass`/`masses_along` instead, which computes the
proper (quasi-)degenerate k.p effective-mass operator via second-order
Loewdin partitioning against the remaining ("remote") bands, correctly
splitting e.g. heavy-hole/light-hole masses by direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR, interpolate_bands
from .dos import _uniform_mesh
from ._fourier_derivs import h_and_k_derivatives_frac


@dataclass
class BandExtremum:
    """A stationary point of one interpolated band."""
    kpt:             np.ndarray   # (3,) crystal coordinates
    energy:          float        # Hartree
    band_index:      int
    kind:            str          # "min" or "max"
    degenerate_with: tuple        # neighbouring band indices near-degenerate at kpt


@dataclass
class EffectiveMass:
    """Effective-mass tensor at a band extremum, plus derived scalars."""
    extremum:          BandExtremum
    mass_tensor:       np.ndarray   # (3,3) m_e units, Cartesian axes
    principal_masses:  np.ndarray   # (3,) eigenvalues, m_e units
    principal_axes:    np.ndarray   # (3,3) columns = Cartesian eigenvectors
    dos_mass:          float        # (m1 m2 m3)^(1/3)
    conductivity_mass: float        # 3 / (1/m1 + 1/m2 + 1/m3)
    anisotropy:        float        # max(mi) / min(mi)


@dataclass
class SemiconductorBandEdges:
    """Valence/conduction band-edge summary for a semiconductor."""
    vbm:    list
    cbm:    list
    gap:    float   # Hartree, global cbm energy - global vbm energy
    direct: bool     # True if the global VBM and CBM sit at the same k-point


# ---------------------------------------------------------------------------
# Finite-difference gradient / Hessian in Cartesian k
# ---------------------------------------------------------------------------

def _grad_hess(
    hr: HamiltonianR, k0_frac: np.ndarray, band_index: int,
    recip_lattice: np.ndarray, h: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Central finite-difference gradient and Hessian of E_{band_index}(k) in
    Cartesian k (Bohr^-1), evaluated at k0_frac (crystal coordinates).
    """
    inv_recip = np.linalg.inv(recip_lattice)

    stencil = [np.zeros(3)]
    for a in range(3):
        for s in (+1.0, -1.0):
            d = np.zeros(3); d[a] = s * h
            stencil.append(d)
    pairs = ((0, 1), (0, 2), (1, 2))
    for a, b in pairs:
        for sa in (+1.0, -1.0):
            for sb in (+1.0, -1.0):
                d = np.zeros(3); d[a] = sa * h; d[b] = sb * h
                stencil.append(d)

    kpts = k0_frac[None, :] + np.array(stencil) @ inv_recip
    E = interpolate_bands(hr, kpts)[:, band_index]

    E0 = float(E[0])
    Ep = {a: E[1 + 2 * a] for a in range(3)}
    Em = {a: E[2 + 2 * a] for a in range(3)}

    grad = np.array([(Ep[a] - Em[a]) / (2 * h) for a in range(3)])

    hess = np.zeros((3, 3))
    for a in range(3):
        hess[a, a] = (Ep[a] - 2 * E0 + Em[a]) / h ** 2

    off = 7
    for i, (a, b) in enumerate(pairs):
        Epp, Epm, Emp, Emm = E[off + 4 * i: off + 4 * i + 4]
        val = (Epp - Epm - Emp + Emm) / (4 * h ** 2)
        hess[a, b] = hess[b, a] = val

    return E0, grad, hess


def _refine_extremum(
    hr: HamiltonianR, k0_frac: np.ndarray, band_index: int,
    recip_lattice: np.ndarray, h: float, tol: float = 1e-8, max_iter: int = 20,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Newton refinement of a stationary point in Cartesian k."""
    inv_recip = np.linalg.inv(recip_lattice)
    k = np.array(k0_frac, dtype=np.float64)

    for _ in range(max_iter):
        E0, grad, hess = _grad_hess(hr, k, band_index, recip_lattice, h)
        if np.linalg.norm(grad) < tol:
            break
        try:
            step_cart = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        k = k + step_cart @ inv_recip

    E0, grad, hess = _grad_hess(hr, k, band_index, recip_lattice, h)
    return k, E0, hess


def _degenerate_group(eigvals: np.ndarray, band_index: int, tol: float) -> tuple:
    """
    Transitive closure of bands within `tol` of each other, starting from
    band_index (e.g. a triply-degenerate point: checking only band_index's
    immediate neighbour would miss the third band unless chained).
    """
    group = {band_index}
    changed = True
    while changed:
        changed = False
        for i in range(len(eigvals)):
            if i in group:
                continue
            if any(abs(eigvals[i] - eigvals[j]) < tol for j in group):
                group.add(i)
                changed = True
    return tuple(sorted(group))


# ---------------------------------------------------------------------------
# Extremum search over the Brillouin zone
# ---------------------------------------------------------------------------

def _mesh_extrema_candidates(
    hr: HamiltonianR, band_index: int, kind: str, mesh: tuple,
) -> np.ndarray:
    """Grid points that are local extrema w.r.t. their 6 nearest neighbours."""
    kpts = _uniform_mesh(mesh)
    E = interpolate_bands(hr, kpts)[:, band_index].reshape(mesh)

    mask = np.ones(mesh, dtype=bool)
    for axis in range(3):
        if kind == "max":
            mask &= E >= np.roll(E, 1, axis=axis)
            mask &= E >= np.roll(E, -1, axis=axis)
        else:
            mask &= E <= np.roll(E, 1, axis=axis)
            mask &= E <= np.roll(E, -1, axis=axis)

    idx = np.argwhere(mask)
    return idx / np.array(mesh, dtype=np.float64)


def find_band_extrema(
    hr:            HamiltonianR,
    band_index:    int,
    kind:          str,
    recip_lattice: np.ndarray,
    mesh:          tuple = (10, 10, 10),
    h:             float = 1e-3,
    k_tol:         float = 1e-4,
    energy_tol:    float = 1e-7,
    degeneracy_tol: float = 4e-5,
) -> list:
    """
    Locate all inequivalent stationary points of `band_index` that are
    local minima ("min") or maxima ("max"), refined to high precision.

    Candidates are found on a coarse uniform mesh (grid points that are
    extremal w.r.t. their 6 nearest neighbours), then refined by Newton
    iteration on the Cartesian-k gradient/Hessian. Points whose refined
    Hessian isn't consistent with the requested kind (e.g. Newton wandered
    to a saddle) are discarded.

    Returns a list of BandExtremum, sorted with the global extremum first
    (lowest energy for "min", highest for "max"). All energies (extremum
    energies, energy_tol dedup threshold, degeneracy_tol) are in Hartree
    (degeneracy_tol default 4e-5 Ha ~ 1 meV).
    """
    if kind not in ("min", "max"):
        raise ValueError('kind must be "min" or "max"')

    candidates = _mesh_extrema_candidates(hr, band_index, kind, mesh)

    extrema: list = []
    for k0 in candidates:
        k_ref, E0, hess = _refine_extremum(hr, k0, band_index, recip_lattice, h)
        eigval_hess = np.linalg.eigvalsh(hess)
        if kind == "min" and np.any(eigval_hess < -1e-6):
            continue
        if kind == "max" and np.any(eigval_hess > 1e-6):
            continue

        k_ref = k_ref % 1.0
        if any(
            np.linalg.norm(((ex.kpt - k_ref + 0.5) % 1.0) - 0.5) < k_tol
            and abs(ex.energy - E0) < energy_tol
            for ex in extrema
        ):
            continue

        all_bands = interpolate_bands(hr, k_ref[None, :])[0]
        group = _degenerate_group(all_bands, band_index, degeneracy_tol)
        degenerate_with = tuple(m for m in group if m != band_index)

        extrema.append(BandExtremum(
            kpt=k_ref, energy=E0, band_index=band_index, kind=kind,
            degenerate_with=degenerate_with,
        ))

    extrema.sort(key=lambda e: e.energy, reverse=(kind == "max"))
    return extrema


# ---------------------------------------------------------------------------
# Effective-mass tensor
# ---------------------------------------------------------------------------

def effective_mass_tensor(
    hr: HamiltonianR, extremum: BandExtremum, recip_lattice: np.ndarray,
    h: float = 1e-3,
) -> np.ndarray:
    """
    Effective-mass tensor (m_e units, Cartesian axes) at a BandExtremum,
    from the numerical Hessian of the interpolated band energy.

    m*^-1 = d^2E/dk^2 in atomic units (Hartree, Bohr^-1): with hbar = 1
    and E in Hartree, the inverse Hessian IS the mass in m_e directly --
    no conversion factor at all.  Valence maxima are sign-flipped so the
    reported hole mass is positive.
    """
    _, _, hess_ha = _grad_hess(hr, extremum.kpt, extremum.band_index,
                                recip_lattice, h)
    sign = -1.0 if extremum.kind == "max" else 1.0
    mass_tensor = sign * np.linalg.inv(hess_ha)
    return (mass_tensor + mass_tensor.T) / 2


def analyze_effective_mass(
    hr: HamiltonianR, extremum: BandExtremum, recip_lattice: np.ndarray,
    h: float = 1e-3,
) -> EffectiveMass:
    """Effective-mass tensor at a BandExtremum plus its principal-axis analysis."""
    mass_tensor = effective_mass_tensor(hr, extremum, recip_lattice, h=h)
    principal_masses, principal_axes = np.linalg.eigh(mass_tensor)

    m1, m2, m3 = principal_masses
    dos_mass = float(np.cbrt(m1 * m2 * m3))
    conductivity_mass = float(3.0 / (1.0 / m1 + 1.0 / m2 + 1.0 / m3))
    anisotropy = float(principal_masses.max() / principal_masses.min())

    return EffectiveMass(
        extremum=extremum,
        mass_tensor=mass_tensor,
        principal_masses=principal_masses,
        principal_axes=principal_axes,
        dos_mass=dos_mass,
        conductivity_mass=conductivity_mass,
        anisotropy=anisotropy,
    )


# ---------------------------------------------------------------------------
# Semiconductor band-edge summary
# ---------------------------------------------------------------------------

def semiconductor_band_edges(
    hr:            HamiltonianR,
    n_valence:     int,
    recip_lattice: np.ndarray,
    mesh:          tuple = (10, 10, 10),
    h:             float = 1e-3,
    k_tol:         float = 1e-4,
) -> SemiconductorBandEdges:
    """
    Locate the valence-band maxima and conduction-band minima of a
    semiconductor Wannier Hamiltonian and analyze their effective masses.

    Args:
      hr           : HamiltonianR spanning both valence and (for a
                     meaningful gap/CBM) some conduction bands
      n_valence    : number of occupied bands; band index n_valence - 1
                     is searched for the VBM, n_valence for the CBM
      recip_lattice: (3, 3) reciprocal lattice rows, Bohr^-1
      mesh         : coarse search mesh for locating extrema
      h            : finite-difference step in Cartesian k (Bohr^-1)
      k_tol        : deduplication / direct-gap tolerance in crystal coords

    Returns:
      SemiconductorBandEdges with a list of EffectiveMass results for
      each inequivalent VBM/CBM valley, the fundamental gap, and whether
      it is direct.
    """
    if n_valence >= hr.nw:
        raise ValueError(
            f"n_valence={n_valence} leaves no conduction band in this "
            f"Hamiltonian (nw={hr.nw}); Wannierize with extra bands to "
            f"capture the conduction-band minimum."
        )

    vbm_extrema = find_band_extrema(hr, n_valence - 1, "max", recip_lattice,
                                     mesh=mesh, h=h, k_tol=k_tol)
    cbm_extrema = find_band_extrema(hr, n_valence, "min", recip_lattice,
                                     mesh=mesh, h=h, k_tol=k_tol)

    vbm = [analyze_effective_mass(hr, e, recip_lattice, h=h) for e in vbm_extrema]
    cbm = [analyze_effective_mass(hr, e, recip_lattice, h=h) for e in cbm_extrema]

    gap = cbm[0].extremum.energy - vbm[0].extremum.energy
    direct = bool(np.linalg.norm(
        ((cbm[0].extremum.kpt - vbm[0].extremum.kpt + 0.5) % 1.0) - 0.5
    ) < k_tol)

    return SemiconductorBandEdges(vbm=vbm, cbm=cbm, gap=float(gap), direct=direct)


# ---------------------------------------------------------------------------
# Degenerate (quasi-degenerate / Loewdin) k.p effective mass
# ---------------------------------------------------------------------------

@dataclass
class DegenerateEffectiveMass:
    """
    Effective-mass operator for a group of (near-)degenerate bands at a
    single k-point, from second-order quasi-degenerate (Loewdin) k.p
    perturbation theory.

    Unlike EffectiveMass, curvature here isn't a single tensor: within
    the degenerate subspace it depends on direction *and* mixes the
    bands (e.g. heavy-hole/light-hole splitting). Use `masses_along` to
    get the per-branch masses along a specific Cartesian direction.
    """
    extremum:              BandExtremum
    band_group:            tuple   # 0-based band indices in the group
    linear_term:           np.ndarray   # (3, D, D) complex, <i|dH/dk_a|j>; ~0 at a true extremum
    inverse_mass_operator: dict         # {(a, b): (D, D) complex} for a <= b in 0,1,2; Hartree, Bohr^-2


def _h_and_k_derivatives_cartesian(
    hr: HamiltonianR, k0_frac: np.ndarray, recip_lattice: np.ndarray,
) -> tuple:
    """
    Analytic H(k), dH/dk and d^2H/dk^2 in Cartesian k (Bohr^-1) at
    k0_frac, from the fractional-space derivatives (see _fourier_derivs).
    """
    H0, grad_frac, hess_frac = h_and_k_derivatives_frac(hr, k0_frac)

    inv_recip = np.linalg.inv(recip_lattice)
    grad_cart = np.einsum('ja,anm->jnm', inv_recip, grad_frac)
    hess_cart = np.einsum('ja,lb,abnm->jlnm', inv_recip, inv_recip, hess_frac)

    for j in range(3):
        grad_cart[j] = (grad_cart[j] + grad_cart[j].conj().T) / 2
        for l in range(3):
            hess_cart[j, l] = (hess_cart[j, l] + hess_cart[j, l].conj().T) / 2

    return H0, grad_cart, hess_cart


def degenerate_effective_mass(
    hr: HamiltonianR, extremum: BandExtremum, recip_lattice: np.ndarray,
    degeneracy_tol: float = 4e-5,
) -> DegenerateEffectiveMass:
    """
    Effective-mass operator at a (possibly degenerate) band extremum via
    second-order quasi-degenerate (Loewdin) k.p perturbation theory:

        H_eff_ij(q) = E0 dij + sum_a q_a <i|dH/dk_a|j>
                      + sum_{a,b} q_a q_b Q_ab[i,j]

        Q_ab[i,j] = (1/2){ <i|d^2H/dk_a dk_b|j>
                     + sum_{m not in group} (<i|dH/dk_a|m><m|dH/dk_b|j>
                                            + <i|dH/dk_b|m><m|dH/dk_a|j>)
                            / (E0 - E_m) }

    for a small Cartesian displacement q from extremum.kpt (Bohr^-1). The
    band group is the full transitive closure of near-degenerate bands
    (energy difference < degeneracy_tol) around extremum.band_index, so
    e.g. a triply-degenerate point is handled as a group of 3.

    `inverse_mass_operator` is in atomic units (Hartree, Bohr^-2); use
    `masses_along` to get m_e-unit per-branch masses along a direction.
    """
    H0, grad_cart, hess_cart = _h_and_k_derivatives_cartesian(
        hr, extremum.kpt, recip_lattice
    )
    eigvals, eigvecs = np.linalg.eigh(H0)

    group = _degenerate_group(eigvals, extremum.band_index, degeneracy_tol)
    remote = [m for m in range(len(eigvals)) if m not in group]
    E0 = eigvals[list(group)].mean()

    Ga  = np.einsum('ni,anm,mj->aij', eigvecs.conj(), grad_cart, eigvecs)
    # 'li' (not 'ni'): hess_cart's own two matrix indices are labelled l,m
    # here (a,b are its two Cartesian indices) -- eigvecs.conj()'s row index
    # must contract against hess_cart's FIRST matrix index (l), the same
    # role 'n' plays against grad_cart's single matrix index above. Using
    # 'ni' left that row index uncontracted (a dangling sum with nothing to
    # pair against), silently corrupting every Hab entry -- caught by
    # cross-checking against `analyze_effective_mass` on an exactly
    # degenerate, decoupled multi-band model where the right answer
    # (m_e = 1 for every branch) is known analytically.
    Hab = np.einsum('li,ablm,mj->abij', eigvecs.conj(), hess_cart, eigvecs)

    D = len(group)
    linear_term = np.array([
        [[Ga[a, gi, gj] for gj in group] for gi in group]
        for a in range(3)
    ])

    inverse_mass_operator = {}
    for a in range(3):
        for b in range(a, 3):
            q_ab = np.zeros((D, D), dtype=np.complex128)
            for i, gi in enumerate(group):
                for j, gj in enumerate(group):
                    val = Hab[a, b, gi, gj]
                    for m in remote:
                        val += (
                            Ga[a, gi, m] * Ga[b, m, gj]
                            + Ga[b, gi, m] * Ga[a, m, gj]
                        ) / (E0 - eigvals[m])
                    q_ab[i, j] = val / 2
            inverse_mass_operator[(a, b)] = q_ab

    return DegenerateEffectiveMass(
        extremum=extremum, band_group=group,
        linear_term=linear_term, inverse_mass_operator=inverse_mass_operator,
    )


def masses_along(result: DegenerateEffectiveMass, q_cart: np.ndarray) -> np.ndarray:
    """
    Per-branch effective masses (m_e units) along a Cartesian direction
    q_cart (need not be normalized), from the degenerate inverse-mass
    operator. Valence maxima are sign-flipped to report positive hole
    masses, matching `analyze_effective_mass`. Returned sorted ascending.

    `inverse_mass_operator`'s Q_ab already carries the Taylor-series 1/2
    (see `degenerate_effective_mass`'s docstring: H_eff(q) - E0 = sum_ab
    q_a q_b Q_ab[i,j], directly the second-order energy, no extra factor),
    so matching the standard E(q) = E0 + q^2/(2m) convention needs mass =
    1/(2 * eigenvalue) here -- confirmed against `analyze_effective_mass`
    on an exactly-solvable single-band (no-degeneracy) tight-binding model,
    where the missing factor of 2 previously made every reported mass
    exactly double the true value.
    """
    q_hat = np.asarray(q_cart, dtype=np.float64)
    q_hat = q_hat / np.linalg.norm(q_hat)

    D = len(result.band_group)
    inv_mass = np.zeros((D, D), dtype=np.complex128)
    for (a, b), q_ab in result.inverse_mass_operator.items():
        weight = q_hat[a] * q_hat[b] * (1.0 if a == b else 2.0)
        inv_mass += weight * q_ab
    inv_mass = (inv_mass + inv_mass.conj().T) / 2

    sign = -1.0 if result.extremum.kind == "max" else 1.0
    with np.errstate(divide="ignore"):
        masses = sign / (2.0 * np.linalg.eigvalsh(inv_mass))
    return np.sort(masses)
