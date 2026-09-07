"""
Floquet engineering of a Wannier Hamiltonian: what a periodic drive does to the
band structure, and what a time-resolved ARPES experiment would see.

  Reference physics: J. H. Shirley, Phys. Rev. 138, B979 (1965) for the Sambe
  construction; T. Oka and H. Aoki, Phys. Rev. B 79, 081406(R) (2009) for
  light-induced topology in graphene; Y. H. Wang, H. Steinberg, P. Jarillo-
  Herrero and N. Gedik, Science 342, 453 (2013) for the measured Floquet-Bloch
  sidebands on Bi2Se3 that `floquet_spectral_function` is written to reproduce.

WHAT THIS COMPUTES. Under a monochromatic drive the Hamiltonian is periodic in
time, H(k, t + T) = H(k, t), and Floquet's theorem replaces the eigenvalue
problem with one in the extended (Sambe) space of photon sectors m:

    [H_F]_{mn}(k) = H^{(m-n)}(k) + m*omega*delta_{mn},
    H^{(p)}(k)    = (1/T) integral dt e^{i p omega t} H(k, t)

whose eigenvalues are the QUASI-ENERGIES: energies modulo omega, defined in a
Floquet zone just as crystal momentum is defined in a Brillouin zone. A driven
metal is not "heated bands"; it is a genuinely different band structure, and it
can have a different topology from the material at rest.

THE PEIERLS SUBSTITUTION IS EXACT AND CHEAP HERE, which is what makes a Wannier
Hamiltonian the right object to drive. Coupling to light is k -> k + A(t), and
because H(k) = sum_R e^{i k.R} H(R) is a finite Fourier sum, the time average
above can be done ANALYTICALLY rather than by sampling the period. For circular
polarisation A(t) = A0 (cos wt, sin wt, 0), writing A(t).R = A0 rho_R
cos(wt - phi_R) with rho_R = |R_perp| and phi_R = atan2(R_y, R_x), the
Jacobi-Anger expansion gives

    H^{(p)}(k) = sum_R e^{i k.R} H(R) * i^p J_p(A0 rho_R) e^{i p phi_R}

exactly: one Bessel function per lattice vector, no time grid, no error beyond
the photon truncation. Each photon block is the undriven H(R) reweighted by
J_p(A0 rho_R), so the drive reaches further in R the stronger it is -- which is
the real-space statement of "the drive mixes distant orbitals".

The p = 0 block is H(R) J_0(A0 rho_R): the famous dynamical band-narrowing
(coherent destruction of tunnelling) falls out as J_0 passing through zero.

CONVENTIONS AND UNITS. Atomic units throughout: `omega` in Hartree, `amplitude`
= A0 in inverse Bohr, so the Peierls phase A0*rho is dimensionless. A0 is the
vector-potential amplitude in the gauge where the phase is A.R; convert from a
field strength with `drive_amplitude_from_field`. Helicity +1/-1 selects the
sense of circular rotation and therefore the SIGN of the induced Chern number;
`polarization='linear'` drives along x only.

TRUNCATION. Photon sectors are kept for |m| <= n_photon. Convergence is not
optional: the number needed grows with A0*rho_max, and an under-truncated
calculation is wrong in a way that looks like physics (spurious gaps at zone
crossings). `floquet_convergence` sweeps it for you. Rule of thumb: n_photon
must comfortably exceed A0 * max|R_perp| over the R with non-negligible H(R).

WHAT IT IS NOT. This is the ideal-drive, infinite-time, non-interacting limit:
no pulse envelope, no dephasing, no electron-electron scattering, and no
occupation dynamics -- Floquet states are computed, not populated. A real
pump-probe experiment sees a transient with finite bandwidth, so compare
SIDEBAND POSITIONS and GAPS (which this gets) rather than intensities (which
depend on the pulse and on how the states got filled).
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from ..core.hamiltonian import HamiltonianR
from ..units import HARTREE_TO_EV

__all__ = ["drive_amplitude_from_field", "floquet_blocks", "floquet_hamiltonian",
           "floquet_quasi_energies", "floquet_spectral_function",
           "floquet_convergence"]


def drive_amplitude_from_field(field_v_per_m: float, omega_ev: float) -> float:
    """
    Peierls amplitude A0 (1/Bohr) from a peak electric field and photon energy.

    A0 = E/omega in atomic units, so a stronger field or a LOWER frequency
    drives harder -- the mid-infrared is the usual choice for exactly this
    reason, and it is why the Bi2Se3 experiment used 120 meV rather than a
    visible photon.

    Args:
      field_v_per_m : peak field of the pump, V/m
      omega_ev      : photon energy, eV
    """
    from ..units import AU_FIELD_V_PER_M, EV_TO_HARTREE
    e_au = float(field_v_per_m) / AU_FIELD_V_PER_M
    w_au = float(omega_ev) * EV_TO_HARTREE
    if w_au <= 0:
        raise ValueError("drive_amplitude_from_field: omega must be > 0")
    return e_au / w_au


def _drive_axes(drive_plane):
    """Orthonormal (e1, e2) spanning the plane the field lives in."""
    if drive_plane is None:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    e = np.asarray(drive_plane, dtype=np.float64)
    if e.shape != (2, 3):
        raise ValueError("drive_plane must be two Cartesian vectors, shape (2, 3)")
    e1 = e[0] / np.linalg.norm(e[0])
    e2 = e[1] - np.dot(e[1], e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        raise ValueError("drive_plane vectors are parallel")
    return e1, e2 / n2


def _polarisation_geometry(d_cart, polarization, helicity, drive_plane=None):
    """rho and phi of the drive plane, for a set of BOND vectors.

    The plane is the FIELD's, not the crystal's: a slab whose surface normal is
    not Cartesian z needs `drive_plane` set to its two in-plane directions, or
    the "in-plane" drive quietly acquires an out-of-plane component and the
    induced gap comes out wrong.
    """
    e1, e2 = _drive_axes(drive_plane)
    d1, d2 = d_cart @ e1, d_cart @ e2
    if polarization == "circular":
        return np.hypot(d1, d2), np.arctan2(helicity * d2, d1)
    if polarization == "linear":
        # A(t) = A0 e1 cos(wt): A.d = A0 d1 cos(wt), i.e. rho = |d1| with the
        # sign of d1 carried by phi = 0 or pi.
        return np.abs(d1), np.where(d1 < 0, np.pi, 0.0)
    raise ValueError(f"polarization must be 'circular' or 'linear', got {polarization!r}")


def _bond_vectors(hr, real_lattice, use_centres):
    """
    Peierls bond vectors d[R, m, n], Cartesian Bohr.

    The phase an electron picks up hopping between two orbitals is A dotted
    into the vector JOINING THEM, not into the lattice vector alone. With
    H_mn(R) = <w_m(0)|H|w_n(R)> and H(k) = sum_R e^{ik.R} H(R), that vector is

        d = R + tau_n - tau_m

    which reduces to R when every Wannier centre sits at the same point. The
    difference is not cosmetic: an INTRA-CELL hopping (R = 0) has d = 0 under
    the R-only convention and therefore feels no field at all, so graphene's
    third nearest-neighbour bond would be left undriven and the dynamical
    band-narrowing would come out at the wrong amplitude (by the ratio of the
    lattice constant to the bond length). Pass ``use_centres=False`` for the
    R-only convention that much of the Floquet tight-binding literature uses.
    """
    R_cart = np.asarray(hr.R_vectors, dtype=np.float64) @ np.asarray(real_lattice)
    if not use_centres or hr.centres is None:
        return np.broadcast_to(R_cart[:, None, None, :],
                               (len(R_cart), hr.nw, hr.nw, 3))
    tau = np.asarray(hr.centres, dtype=np.float64)              # (nw, 3) Bohr
    return R_cart[:, None, None, :] + tau[None, None, :, :] - tau[None, :, None, :]


def floquet_blocks(hr: HamiltonianR, kpts, *, amplitude: float, n_photon: int,
                   real_lattice=None, polarization: str = "circular",
                   helicity: int = 1, use_centres: bool = True,
                   drive_plane=None) -> np.ndarray:
    """
    The Fourier blocks H^{(p)}(k) of the driven Hamiltonian, p = -2n..+2n.

    Returns ``(n_p, nk, nw, nw)`` complex with ``n_p = 4*n_photon + 1``, indexed
    so that element ``[p + 2*n_photon]`` is H^{(p)}. Blocks up to |p| = 2*n_photon
    are needed because the Sambe matrix couples sectors m and n with
    |m - n| <= 2*n_photon.

    The Bessel weights make the cost identical to one undriven interpolation per
    block: no time integration anywhere.
    """
    if real_lattice is None:
        real_lattice = hr.real_lattice
    if real_lattice is None:
        raise ValueError(
            "floquet_blocks needs the real lattice (Bohr) to place R in Cartesian "
            "space for the Peierls phase; pass real_lattice= or use an hr that "
            "carries it.")
    d = _bond_vectors(hr, real_lattice, use_centres)              # (nR, nw, nw, 3)
    rho, phi = _polarisation_geometry(d, polarization, helicity, drive_plane)
    H_R = hr.H_R.detach().cpu().numpy() / np.asarray(hr.degen)[:, None, None]
    kpts = np.atleast_2d(np.asarray(kpts, dtype=np.float64))
    phase_k = np.exp(2j * np.pi * (kpts @ np.asarray(hr.R_vectors).T))   # (nk, nR)

    n_p = 4 * n_photon + 1
    out = np.zeros((n_p, len(kpts), hr.nw, hr.nw), dtype=np.complex128)
    for idx, p in enumerate(range(-2 * n_photon, 2 * n_photon + 1)):
        w = (1j ** p) * jv(p, amplitude * rho) * np.exp(1j * p * phi)  # (nR,nw,nw)
        out[idx] = np.einsum("kR,Rij,Rij->kij", phase_k, w, H_R, optimize=True)
    return out


def floquet_hamiltonian(hr: HamiltonianR, kpts, *, amplitude: float,
                        omega: float, n_photon: int, real_lattice=None,
                        polarization: str = "circular", helicity: int = 1,
                        use_centres: bool = True, drive_plane=None) -> np.ndarray:
    """
    Sambe-space Floquet Hamiltonian, ``(nk, (2n+1)*nw, (2n+1)*nw)`` complex.

    Photon sectors run m = -n_photon .. +n_photon in blocks of ``nw``, so the
    m-th block of an eigenvector is its m-th harmonic amplitude -- which is
    exactly the weight the sideband at ``quasi_energy + m*omega`` carries in
    photoemission (see `floquet_spectral_function`).

    Args:
      omega     : drive frequency, Hartree
      amplitude : Peierls A0, 1/Bohr (`drive_amplitude_from_field` converts)
    """
    blocks = floquet_blocks(hr, kpts, amplitude=amplitude, n_photon=n_photon,
                            real_lattice=real_lattice, polarization=polarization,
                            helicity=helicity, use_centres=use_centres,
                            drive_plane=drive_plane)
    nk, nw = blocks.shape[1], hr.nw
    nm = 2 * n_photon + 1
    HF = np.zeros((nk, nm * nw, nm * nw), dtype=np.complex128)
    for i, m in enumerate(range(-n_photon, n_photon + 1)):
        for j, n in enumerate(range(-n_photon, n_photon + 1)):
            HF[:, i * nw:(i + 1) * nw, j * nw:(j + 1) * nw] = \
                blocks[(m - n) + 2 * n_photon]
        HF[:, i * nw:(i + 1) * nw, i * nw:(i + 1) * nw] += \
            m * omega * np.eye(nw)[None, :, :]
    return HF


def floquet_quasi_energies(hr: HamiltonianR, kpts, *, amplitude: float,
                           omega: float, n_photon: int, real_lattice=None,
                           polarization: str = "circular", helicity: int = 1,
                           fold: bool = False, use_centres: bool = True,
                           drive_plane=None):
    """
    Quasi-energies (Hartree) and Floquet states.

    Returns ``(eps, states)`` with ``eps`` of shape ``(nk, (2n+1)*nw)`` sorted
    ascending and ``states`` the corresponding eigenvectors.

    ``fold=True`` maps every quasi-energy into the first Floquet zone
    [-omega/2, omega/2). Do that for a topological analysis, where the zone
    structure is the point, and NOT for comparison with photoemission, where
    the physical replicas at eps + m*omega are what is measured.

    Only the central sectors are trustworthy: sectors near +-n_photon are
    truncated and their states leak. Use `floquet_convergence` to choose
    n_photon, and read observables from the m = 0 neighbourhood.
    """
    HF = floquet_hamiltonian(hr, kpts, amplitude=amplitude, omega=omega,
                             n_photon=n_photon, real_lattice=real_lattice,
                             polarization=polarization, helicity=helicity,
                             use_centres=use_centres, drive_plane=drive_plane)
    HF = 0.5 * (HF + np.conj(np.swapaxes(HF, -1, -2)))
    eps, vecs = np.linalg.eigh(HF)
    if fold:
        eps = (eps + 0.5 * omega) % omega - 0.5 * omega
        order = np.argsort(eps, axis=1)
        eps = np.take_along_axis(eps, order, axis=1)
        vecs = np.take_along_axis(vecs, order[:, None, :], axis=2)
    return eps, vecs


def floquet_spectral_function(hr: HamiltonianR, kpts, energies, *,
                              amplitude: float, omega: float, n_photon: int,
                              broadening: float, real_lattice=None,
                              polarization: str = "circular", helicity: int = 1,
                              orbital_weights=None, use_centres: bool = True,
                              drive_plane=None) -> np.ndarray:
    """
    A(k, E) as time-resolved ARPES would measure it: the Floquet-Bloch sidebands.

    Each Floquet state alpha contributes at EVERY replica energy
    ``eps_alpha - m*omega``, with weight equal to the norm of its m-th photon
    block -- the probe absorbs the electron out of one harmonic of the dressed
    state, so the harmonic content IS the sideband intensity:

        A(k,E) = sum_alpha sum_m |phi_alpha^{(m)}|^2 L(E - eps_alpha + m omega)

    The SIGN of that shift is fixed by the Sambe convention, not free. With
    [H_F]_{mn} = H^{(m-n)} + m omega delta_{mn} the component phi^{(m)} carries
    the time dependence e^{-i(eps - m omega)t}, so the pole is at eps - m omega.
    Getting it backwards is invisible in the driven spectrum -- it merely
    relabels sidebands -- but it breaks the undriven limit, where the Sambe copy
    living in sector m has eps = e_0 + m omega and must fold back exactly onto
    e_0 rather than spreading to e_0 + 2 m omega.

    with L a normalised Lorentzian of half-width `broadening`. In the undriven
    limit each state is pure m = 0 and this collapses to the ordinary spectral
    function; as the drive turns on, weight bleeds into m = +-1, +-2 ... and
    replicas appear above and below every band, gapping where they cross. That
    crossing gap is the observable that distinguishes genuine Floquet-Bloch
    dressing from mere photo-assisted emission, and it is what Wang et al.
    resolved on Bi2Se3.

    Args:
      energies   : (nE,) Hartree, the analyser energy axis
      broadening : Lorentzian half-width, Hartree (experimental resolution plus
                   the inverse pulse duration; 5-20 meV is realistic)
      orbital_weights : (nw,) optional real weights to project the intensity
                   onto selected Wannier orbitals (e.g. a surface layer)

    Returns ``(nk, nE)``.
    """
    eps, vecs = floquet_quasi_energies(
        hr, kpts, amplitude=amplitude, omega=omega, n_photon=n_photon,
        real_lattice=real_lattice, polarization=polarization, helicity=helicity,
        use_centres=use_centres, drive_plane=drive_plane)
    nk, ndim = eps.shape
    nw, nm = hr.nw, 2 * n_photon + 1
    energies = np.asarray(energies, dtype=np.float64)
    if broadening <= 0:
        raise ValueError("floquet_spectral_function: broadening must be > 0")
    ow = (np.ones(nw) if orbital_weights is None
          else np.asarray(orbital_weights, dtype=np.float64))

    # weight[k, alpha, m] = sum_orb |phi_alpha^{(m),orb}|^2
    amp2 = (np.abs(vecs) ** 2).reshape(nk, nm, nw, ndim)
    weight = np.einsum("kmoa,o->kam", amp2, ow, optimize=True)      # (nk, ndim, nm)

    A = np.zeros((nk, len(energies)))
    m_vals = np.arange(-n_photon, n_photon + 1)
    for im, m in enumerate(m_vals):
        centre = eps - m * omega                                   # (nk, ndim)
        d = energies[None, None, :] - centre[:, :, None]
        lor = (broadening / np.pi) / (d ** 2 + broadening ** 2)
        A += np.einsum("ka,kae->ke", weight[:, :, im], lor, optimize=True)
    return A


def floquet_convergence(hr: HamiltonianR, kpts, *, amplitude: float,
                        omega: float, photon_range=(1, 2, 3, 4, 5),
                        real_lattice=None, polarization: str = "circular",
                        helicity: int = 1, n_track: int = 4,
                        use_centres: bool = True, drive_plane=None,
                        e_ref: float | None = None) -> dict:
    """
    Photon-truncation convergence: quasi-energies of the states you care about
    versus ``n_photon``.

    Under-truncation does not announce itself -- it produces plausible-looking
    gaps at zone-boundary crossings -- so this is not optional. Returns a dict
    with ``n_photon``, the tracked quasi-energies ``eps`` (n_settings, nk,
    n_track), and ``max_shift_meV``, the largest change from the previous
    setting. Converged means that shift is small against whatever gap you
    intend to quote.

    The states tracked are the ones whose weight sits mostly in the m = 0
    photon sector -- the physical band in the central Floquet zone. That choice
    is what makes the diagnostic mean anything: the Sambe spectrum is unbounded
    and gets DENSER as sectors are added, so both "the middle of the sorted
    spectrum" and "the states nearest an energy" select different physical
    states at each truncation and report tens of meV of drift on a system that
    is perfectly converged. ``e_ref`` (Hartree) picks which of the m = 0 states
    to report; without it they are taken from the middle of that set.

    The set is chosen ONCE, at the first truncation, and afterwards each tracked
    state is followed to the nearest quasi-energy of the next calculation.
    Re-selecting at every truncation instead makes ``max_shift_meV`` report the
    arrival of new states rather than the movement of old ones -- on a slab that
    reads as 40-80 meV of "drift" while the states themselves are stable to
    0.1 meV.
    """
    out_eps, shifts, prev = [], [], None
    for n in photon_range:
        eps, vecs = floquet_quasi_energies(
            hr, kpts, amplitude=amplitude, omega=omega, n_photon=n,
            real_lattice=real_lattice, polarization=polarization,
            helicity=helicity, use_centres=use_centres, drive_plane=drive_plane)
        nk, ndim = eps.shape
        nm, nw = 2 * n + 1, hr.nw
        w0 = (np.abs(vecs.reshape(nk, nm, nw, ndim)[:, n]) ** 2).sum(axis=1)  # (nk, ndim)
        central = np.argsort(-w0, axis=1)[:, :nw]
        e_central = np.sort(np.take_along_axis(eps, central, axis=1), axis=1)
        if prev is not None:
            # follow the states already being tracked, do not re-select
            idx = np.abs(e_central[:, None, :] - prev[:, :, None]).argmin(axis=2)
            take = np.take_along_axis(e_central, idx, axis=1)
        elif e_ref is None:
            mid = nw // 2
            take = e_central[:, max(0, mid - n_track // 2):
                             max(0, mid - n_track // 2) + n_track]
        else:
            order = np.argsort(np.abs(e_central - e_ref), axis=1)[:, :n_track]
            take = np.sort(np.take_along_axis(e_central, order, axis=1), axis=1)
        out_eps.append(take)
        shifts.append(np.nan if prev is None
                      else float(np.abs(take - prev).max()) * HARTREE_TO_EV * 1e3)
        prev = take
    return {"n_photon": list(photon_range), "eps": np.array(out_eps),
            "max_shift_meV": shifts}
