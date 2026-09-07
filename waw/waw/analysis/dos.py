"""
Density of states via Wannier interpolation on a dense k-mesh.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR, interpolate_bands
from ..core.distributions import gaussian_smearing


@dataclass
class DOS:
    """Gaussian-broadened density of states (atomic units)."""
    energies: np.ndarray   # (n_energies,) Hartree
    dos:      np.ndarray   # (n_energies,) states/Hartree, per cell


def _uniform_mesh(mesh: tuple[int, int, int]) -> np.ndarray:
    """Gamma-centred uniform k-mesh in crystal coordinates, (prod(mesh), 3)."""
    N1, N2, N3 = mesh
    i, j, k = np.meshgrid(np.arange(N1), np.arange(N2), np.arange(N3), indexing="ij")
    return np.stack([i.ravel() / N1, j.ravel() / N2, k.ravel() / N3], axis=-1)


def density_of_states(
    hr:         HamiltonianR,
    mesh:       tuple[int, int, int],
    energies:   np.ndarray | None = None,
    n_energies: int   = 500,
    sigma:      float = 0.002,
    e_pad:      float = 0.02,
) -> DOS:
    """
    Density of states by Gaussian-broadened Wannier interpolation.

    Diagonalizes H(k) on a dense uniform k-mesh and bins the eigenvalues
    with Gaussian smearing:

        DOS(E) = (1/Nk) sum_{k,n} exp(-(E - eps_kn)^2 / 2 sigma^2) / (sigma sqrt(2 pi))

    Everything is in atomic units, like the rest of `analysis`: the energy
    grid in Hartree and the DOS in states/Hartree.  Convert at the caller
    (energies * HARTREE_TO_EV; dos * EV_TO_HARTREE gives states/eV) when
    eV output is wanted.

    Args:
      hr        : HamiltonianR from compute_hr (Hartree)
      mesh      : (N1, N2, N3) dense interpolation mesh, e.g. (20, 20, 20)
      energies  : explicit energy grid in Hartree; if None, built automatically
                  from [min(eig) - e_pad, max(eig) + e_pad] with n_energies points
      n_energies: number of grid points when `energies` is None
      sigma     : Gaussian smearing width in Hartree (default 0.002 Ha ~ 54 meV)
      e_pad     : padding in Hartree added to the automatic energy range

    Returns:
      DOS with the energy grid (Hartree) and states/Hartree per cell (i.e.
      per the set of bands in hr, not normalized to a formula unit).
    """
    kpts  = _uniform_mesh(mesh)
    bands = interpolate_bands(hr, kpts)   # (nk, nw), Hartree

    if energies is None:
        energies = np.linspace(bands.min() - e_pad, bands.max() + e_pad, n_energies)

    nk    = bands.shape[0]
    diff  = energies[:, None, None] - bands[None, :, :]   # (nE, nk, nw)
    dos   = gaussian_smearing(diff, sigma)
    dos   = dos.sum(axis=(1, 2)) / nk

    return DOS(energies=energies, dos=dos)


def fermi_surface_dos(
    eig: np.ndarray, fermi_energy: float, sigma: float, ngauss: int = 0,
) -> float:
    """
    Fermi-level density of states from an already-computed band eigenvalue
    array, per spin channel and per unit cell (states/Hartree):

      N(eF) = (1/Nk) sum_(k,n) delta_sigma(eps_kn - eF)

    This exists so `analysis.elph.alpha2f`'s normalising ``N(eF)`` can be
    evaluated on EXACTLY the same k-mesh and with EXACTLY the same broadening
    as the Fermi-surface deltas in its own numerator, which is what EPW does
    (``selfen.f90`` recomputes ``dosef = dos_ef(ngaussw, degaussw0, ...)/2``
    from the fine-mesh ``etf`` inside every smearing loop). It is NOT
    optional bookkeeping: at finite smearing alpha2F is a RATIO of two
    delta-function sums, and only when numerator and denominator share the
    mesh and the broadening does that ratio become smearing-independent to
    leading order. Supplying a separately-converged N(eF) -- a denser mesh,
    or a different sigma -- rescales lambda by the mismatch.

    Parameters
    ----------
    eig : (nk, nw) float64, Hartree -- `analysis.elph.band_eigensystem`'s
        eigenvalues at the SAME kpts `alpha2f` sums over.
    fermi_energy : Hartree.
    sigma : Hartree, this project's Gaussian width (see
        `core.distributions.epw_degauss_to_sigma` to convert an EPW
        ``degaussw``).
    ngauss : 0 (plain Gaussian, this project's default and the one EPW uses
        for the deltas inside the linewidth) or 1 (Methfessel-Paxton, which
        is what EPW's ``ngaussw`` defaults to for the DOS specifically --
        set it to 1 to reproduce an EPW run's ``dosef`` exactly).

    Returns
    -------
    float, states/Hartree per spin per cell.
    """
    from ..core.distributions import sigma_to_epw_degauss, w0gauss

    eig = np.asarray(eig, dtype=np.float64)
    if ngauss == 0:
        d = gaussian_smearing(eig - fermi_energy, sigma)
    else:
        degauss = sigma_to_epw_degauss(sigma)
        d = w0gauss((eig - fermi_energy) / degauss, ngauss) / degauss
    return float(d.sum() / eig.shape[0])


def fermi_level_from_electron_count(
    eig: np.ndarray, n_electrons: float, sigma: float,
    *, band_degeneracy: float = 2.0, tol: float = 1e-10, max_iter: int = 200,
) -> float:
    """
    Fermi level of a Wannier-interpolated band structure, fixed by its OWN
    electron count rather than imported from the underlying DFT run:

      sum_(k,n) 2 * f_sigma(eps_kn - eF) / Nk = n_electrons

    with ``f_sigma`` the Gauss-smeared occupation (the erfc complementary to
    `fermi_surface_dos`'s Gaussian delta, so the two are consistent).

    This is NOT a convenience. A Wannier model built from a disentangled
    subspace is its own band structure, and its Fermi level is generally NOT
    the SCF one: on the Al of workflow 13 the QE value of 7.7436 eV puts
    2.9583 electrons in the 4-orbital sp manifold instead of 3, i.e. the
    Fermi surface is drawn 0.10 eV off. Every Fermi-surface delta in
    `analysis.elph.alpha2f`, and N(eF) itself, is then evaluated at the
    wrong energy. EPW does the same thing (``efnew``/``efermig`` recomputed
    from the interpolated eigenvalues on each fine mesh -- `epw_fermi_level`
    replicates that exactly).

    The electron count is a VOLUME integral and converges far faster than
    N(eF), which is a Fermi-SURFACE integral: on that Al model the count is
    stable to 1e-4 by a 80^3 mesh while N(eF) is still settling at 150^3. So
    determine eF once on a modest mesh, then hold it fixed.

    Parameters
    ----------
    eig : (nk, nw) float64, Hartree.
    n_electrons : electrons per cell carried by these bands -- e.g. 3.0 for
        the Al sp manifold, 35.0 for Fe3RuN's 46-spinor manifold.
    sigma : Hartree, the same broadening used for `fermi_surface_dos`.
    band_degeneracy : electrons per filled band: 2.0 (default) for a
        spin-unpolarized calculation whose bands are doubly degenerate,
        **1.0 for spinor (noncollinear/SOC) models** -- forgetting this
        places E_F a band too high and is silent (the count still
        converges, to the wrong level). Half of this project's materials
        are spinor; pass it explicitly there.

    Returns
    -------
    float, Hartree.
    """
    from scipy.special import erfc

    eig = np.asarray(eig, dtype=np.float64)
    nk = eig.shape[0]
    if not 0.0 < n_electrons < band_degeneracy * eig.shape[1]:
        raise ValueError(
            f"fermi_level_from_electron_count: n_electrons={n_electrons} is "
            f"outside (0, {band_degeneracy * eig.shape[1]}) for "
            f"{eig.shape[1]} bands at band_degeneracy={band_degeneracy}."
        )

    def count(ef):
        return band_degeneracy * 0.5 * float(
            erfc((eig - ef) / (sigma * np.sqrt(2.0))).sum()) / nk

    lo, hi = float(eig.min()) - 50.0 * sigma, float(eig.max()) + 50.0 * sigma
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if count(mid) < n_electrons:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def fermi_level_spin_channels(
    eig_by_spin: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    n_electrons: float, sigma: float,
    *, tol: float = 1e-10, max_iter: int = 200,
) -> float:
    """
    The single Fermi level shared by the spin channels of a collinear magnet,
    fixed by their COMBINED electron count:

      sum_s (1/Nk_s) sum_(k,n) f_sigma(eps^s_kn - eF) = n_electrons

    A collinear run produces one Wannier model per spin, and it is tempting to
    hand each of them its own `fermi_level_from_electron_count`. That is
    wrong: a ferromagnet has ONE chemical potential, and its moment is exactly
    the difference in how the two channels fill at that one energy. Solving
    the channels separately imposes equal filling on them and so sets the
    moment to zero by construction -- silently, since each channel's count
    converges perfectly well to its own (wrong) target.

    Each channel carries ONE electron per state, unlike
    `fermi_level_from_electron_count`'s ``band_degeneracy=2`` default: the
    factor of two a spin-unpolarized model gets from degeneracy is here
    supplied by there being two channels.

    The same reason that function exists applies here -- the SCF Fermi level
    of the underlying DFT run is generally NOT the Fermi level of the
    disentangled subspace, because the smearing function, its width, the
    k-mesh and the band set all differ. On the fcc Co of workflow 24 the QE
    value of 18.7350 eV sits 25 meV below the model's own and puts 8.97
    electrons in the two manifolds instead of 9.

    Parameters
    ----------
    eig_by_spin : sequence or mapping of (nk, nw) float64 arrays, Hartree --
        one per spin channel, as returned by `analysis.elph.band_eigensystem`.
        Channels may carry different numbers of Wannier functions (a common
        outcome of disentangling the majority and minority manifolds
        separately); each is normalised by its own ``nk``.
    n_electrons : electrons per cell carried by ALL the channels together --
        e.g. 17.0 for fcc Co's two 6-orbital s+d models under a pseudo with
        3s3p semicore, not 8.5 per channel.
    sigma : Hartree, the same broadening used for `fermi_surface_dos`.

    Returns
    -------
    float, Hartree.
    """
    from scipy.special import erfc

    if isinstance(eig_by_spin, Mapping):
        eig_by_spin = list(eig_by_spin.values())
    eigs = [np.asarray(e, dtype=np.float64) for e in eig_by_spin]
    if len(eigs) < 2:
        raise ValueError(
            "fermi_level_spin_channels: expected one eigenvalue array per spin "
            f"channel, got {len(eigs)}. A spin-unpolarized model wants "
            "fermi_level_from_electron_count instead."
        )
    n_max = float(sum(e.shape[1] for e in eigs))
    if not 0.0 < n_electrons < n_max:
        raise ValueError(
            f"fermi_level_spin_channels: n_electrons={n_electrons} is outside "
            f"(0, {n_max}) for channels carrying "
            f"{[int(e.shape[1]) for e in eigs]} Wannier functions at one "
            "electron per state. This is the per-CELL count summed over "
            "channels, not the count per channel."
        )

    def count(ef):
        return sum(
            0.5 * float(erfc((e - ef) / (sigma * np.sqrt(2.0))).sum()) / e.shape[0]
            for e in eigs
        )

    lo = min(float(e.min()) for e in eigs) - 50.0 * sigma
    hi = max(float(e.max()) for e in eigs) + 50.0 * sigma
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if count(mid) < n_electrons:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def epw_fermi_level(
    eig: np.ndarray, n_electrons: float, degauss: float, ngauss: int = 1,
    *, band_degeneracy: float = 2.0, tol: float = 1e-10, max_iter: int = 300,
) -> float:
    """
    EPW's fine-mesh Fermi level, replicating QE's ``efermig`` (``PW/src/
    efermig.f90``) exactly -- EPW recomputes ``efnew = efermig(etf, ...,
    degaussw, ngaussw, ...)`` on every fine mesh ("Fermi energy is
    calculated from the fine k-mesh" in its stdout), and since
    ``ngaussw`` DEFAULTS TO 1 (Methfessel-Paxton) that energy is neither
    the SCF one nor `fermi_level_from_electron_count`'s Gaussian-erfc
    one. On a model whose N(E) varies strongly near eF (any coarse fine
    mesh), everything downstream -- the double deltas, N(eF), lambda --
    is pinned to THIS energy, so matching EPW starts here. Measured on
    the Al model: EPW 12^3 gives 7.777617 eV where the SCF value is
    7.7423 eV, and lambda evaluated at the latter is 3.5x smaller.

    Algorithm (faithful to QE 7.3.1):

      1. bisection on the electron count with PLAIN GAUSSIAN smearing
         (``ngauss=0``) regardless of the requested type, tolerance 1e-10
         on the count, 300 iterations max;
      2. if the requested type is 0 or -99 (monotonic occupations), or the
         count already matches: done;
      3. otherwise (MP / cold smearing, non-monotonic): Newton minimisation
         of ``(N(eF) - nelec)^2`` with the ACTUAL smearing, accepted at the
         looser ``eps_cold_MP = 1e-2``, falling back to bisection with the
         actual smearing if it fails.

    Parameters
    ----------
    eig : (nk, nw) float64, Hartree -- interpolated eigenvalues on the fine
        mesh (uniform weights ``band_degeneracy/nk`` are applied here,
        exactly EPW's ``wkf``).
    n_electrons : electrons per cell (EPW's ``nelec``).
    band_degeneracy : 2.0 (default) for spin-unpolarized bands, **1.0 for
        spinor models** -- QE/EPW's own ``wkf`` sums to 1 for noncolin
        runs; the same silent-misplacement trap as
        `fermi_level_from_electron_count`.
    degauss : Hartree -- EPW's ``degaussw`` AS IS (the w0gauss/wgauss width;
        no sqrt(2) conversion, unlike `alpha2f`'s ``sigma_e``).
    ngauss : QE smearing type (EPW's ``ngaussw``): 0 Gaussian, 1
        Methfessel-Paxton (EPW's default), -99 Fermi-Dirac.

    Returns
    -------
    float, Hartree.
    """
    from ..core.distributions import w0gauss, wgauss

    eig = np.asarray(eig, dtype=np.float64)
    nk = eig.shape[0]

    def count(ef, ng):
        return band_degeneracy * float(wgauss((ef - eig) / degauss, ng).sum()) / nk

    lo = float(eig.min()) - 10.0 * degauss
    hi = float(eig.max()) + 10.0 * degauss

    def bisect(ng, lo, hi):
        ef = 0.5 * (lo + hi)
        for _ in range(max_iter):
            ef = 0.5 * (lo + hi)
            f = count(ef, ng) - n_electrons
            if abs(f) < tol:
                break
            if f < -tol:
                lo = ef
            else:
                hi = ef
        return ef

    ef = bisect(0 if ngauss != -99 else -99, lo, hi)
    if ngauss in (0, -99) or abs(count(ef, ngauss) - n_electrons) < tol:
        return ef

    # Newton on (N - nelec)^2 with the actual (non-monotonic) smearing, QE's
    # own refinement -- dN/deF is the smeared DOS, d2N/deF2 its derivative.
    def dcount(ef):
        return band_degeneracy * float(
            (w0gauss((eig - ef) / degauss, ngauss) / degauss).sum()) / nk

    x0 = ef
    for _ in range(max_iter):
        f = count(x0, ngauss) - n_electrons
        f1 = 2.0 * f * dcount(x0)
        h = 1e-4 * degauss
        f2 = 2.0 * ((dcount(x0)) ** 2
                    + f * (dcount(x0 + h) - dcount(x0 - h)) / (2.0 * h))
        if abs(f2) <= tol:
            break
        x = x0 - f1 / abs(f2)
        if abs(x - x0) < tol or abs(count(x, ngauss) - n_electrons) < tol:
            x0 = x
            break
        x0 = x
    if abs(count(x0, ngauss) - n_electrons) < 1e-2:
        return x0
    return bisect(ngauss, lo, hi)
