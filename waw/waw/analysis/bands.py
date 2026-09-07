"""
Band-structure interpolation along a high-symmetry k-path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR, interpolate_bands
from .kpath import KPath, build_kpath, parse_kpoint_path


@dataclass
class BandStructure:
    """Interpolated Wannier bands along a high-symmetry k-path."""
    kpath: KPath
    bands: np.ndarray   # (nk, nw) Hartree, sorted ascending


def band_structure(
    hr:            HamiltonianR,
    kpoint_path:   list[str],
    recip_lattice: np.ndarray,
    n_points:      int = 100,
) -> BandStructure:
    """
    Interpolate Wannier bands along a Wannier90 `kpoint_path` block.

    Args:
      hr           : HamiltonianR from waw.core.hamiltonian.compute_hr
      kpoint_path  : raw lines of the .win `kpoint_path` block, i.e.
                     `read_win(...)["kpoint_path"]`
      recip_lattice: (3, 3) reciprocal lattice rows in Bohr^-1
      n_points     : k-points per path segment

    Returns:
      BandStructure with the k-path (for plot x-axis/tick labels) and the
      interpolated eigenvalues in Hartree (atomic units, like everything
      in `analysis`; convert with units.HARTREE_TO_EV at the caller when
      eV output is wanted).
    """
    segments = parse_kpoint_path(kpoint_path)
    kpath    = build_kpath(segments, recip_lattice, n_points=n_points)
    bands    = interpolate_bands(hr, kpath.kpts)   # Hartree, like hr
    return BandStructure(kpath=kpath, bands=bands)


def mesh_fidelity(hr, kpts, eig_dft_ev, *, window=None, fermi_ev=None) -> dict:
    """
    How faithfully a Wannier model reproduces the ab-initio eigenvalues at the
    k-points it was BUILT from -- the quantitative companion to a
    DFT-vs-Wannier band plot, and the check that catches a bad model when the
    plot is unavailable (no charge density left for a bands run).

    On the coarse mesh a correct model is exact wherever bands are frozen, so
    a nonzero deviation there localizes the damage immediately. It is what
    exposed NiI2's partially-frozen model (0.0000 meV on the 20 frozen bands
    but 131 meV -- 24% of their bandwidth -- on the two unfrozen ones that
    carried the exchange), so ALWAYS look at it inside the window you care
    about, not just overall.

    *** THIS CANNOT SEE INTERPOLATION ERROR, AND IS NOT A MODEL-QUALITY GATE. ***
    Inside the frozen window a Wannier model reproduces its own mesh by
    construction, so a good number here is very nearly a tautology -- notebook
    21 returns exactly 0.000/0.000 meV. It says nothing about the space BETWEEN
    mesh points, which is where an interpolated model is actually used. Notebook
    22's QE route passed this at 6.4 meV max / 0.34 meV rms while its bands rang
    by several eV between mesh points and its superfluid weight came out 23x too
    large. Pair it with `band_path_fidelity`, which measures the off-mesh error;
    neither one substitutes for the other.

    Args:
      hr         : HamiltonianR (Hartree)
      kpts       : (nk, 3) the model's own mesh, crystal coordinates
      eig_dft_ev : (nk, nb) ab-initio eigenvalues in eV, same k order
                   (e.g. ``generate_overlaps(...)['eig']``)
      window     : (lo, hi) eV -- restrict the comparison to this absolute
                   energy range (pass the frozen window to assert exactness)
      fermi_ev   : if given, also report the deviation within +-1 eV of E_F

    Returns a dict of deviations in meV: ``max``, ``rms`` and ``n_states``,
    plus ``max_near_ef`` when ``fermi_ev`` is given. Each Wannier level is
    compared with the closest ab-initio eigenvalue at the same k-point, so
    the metric does not assume the two are index-aligned.
    """
    import numpy as np

    from .elph import band_eigensystem
    from ..units import HARTREE_TO_EV

    kpts = np.asarray(kpts, dtype=np.float64)
    eig_dft_ev = np.asarray(eig_dft_ev, dtype=np.float64)
    ew = band_eigensystem(hr, kpts)[0] * HARTREE_TO_EV        # (nk, nw)

    dev, energy = [], []
    for ik in range(len(kpts)):
        for e in ew[ik]:
            dev.append(np.abs(eig_dft_ev[ik] - e).min())
            energy.append(e)
    dev = np.asarray(dev) * 1e3          # meV
    energy = np.asarray(energy)

    sel = np.ones(len(dev), dtype=bool)
    if window is not None:
        sel &= (energy >= window[0]) & (energy <= window[1])
    out = {
        "max": float(dev[sel].max()) if sel.any() else float("nan"),
        "rms": float(np.sqrt((dev[sel] ** 2).mean())) if sel.any() else float("nan"),
        "n_states": int(sel.sum()),
    }
    if fermi_ev is not None:
        near = np.abs(energy - fermi_ev) <= 1.0
        out["max_near_ef"] = float(dev[near].max()) if near.any() else float("nan")
    return out


def band_path_fidelity(bands_w_ev, bands_ref_ev, *, window=None, fermi_ev=None,
                       near_ef=1.0) -> dict:
    """
    Off-mesh interpolation error: Wannier bands against an ab-initio reference
    along a k-path. THE model-quality gate, and the one `mesh_fidelity` cannot
    be.

    A Wannier model is exact on its own mesh inside the frozen window, so
    `mesh_fidelity` is close to a tautology there. What decides whether the
    model is usable is its behaviour BETWEEN mesh points, and the only way to
    see that is to evaluate it off-mesh against an independent calculation.
    Every DFT-vs-Wannier band plot already has both arrays in hand; this turns
    that picture into a number so the failure cannot be missed by not looking.

    A model whose H(R) does not decay -- typically because the disentangled
    subspace changes character between neighbouring k-points -- rings between
    mesh points at mesh frequency while staying exact on it. That is invisible
    to `mesh_fidelity` and glaring here. It is how notebook 22's Nb model
    reached a superfluid weight 23x too large.

    Args:
      bands_w_ev   : (nk, nw) interpolated Wannier eigenvalues, eV
      bands_ref_ev : (nk, nb) ab-initio eigenvalues on the SAME k-path, eV.
                     nb need not equal nw and the two need not be index
                     aligned: each Wannier level is matched to the closest
                     reference level at that k-point. NaNs are ignored, so a
                     failed bands run degrades to n_states = 0 rather than to a
                     falsely good number.
      window       : (lo, hi) eV, restrict to an absolute energy range
      fermi_ev     : if given, also report the deviation within ``near_ef`` of
                     E_F -- the number that matters for a Fermi-surface
                     property
      near_ef      : half-width in eV of that E_F band (default 1.0)

    Returns a dict in meV: ``max``, ``rms``, ``n_states``, plus ``max_near_ef``
    and ``rms_near_ef`` when ``fermi_ev`` is given.

    There is no universal pass mark -- it scales with the bandwidth being
    interpolated -- but for a Fermi-surface property, an error near E_F that is
    a sizeable fraction of the superconducting gap, the smearing, or any other
    small energy the result depends on means the model cannot support it.
    """
    import numpy as np

    w = np.asarray(bands_w_ev, dtype=np.float64)
    ref = np.asarray(bands_ref_ev, dtype=np.float64)
    if w.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"expected (nk, n) arrays; got {w.shape} and {ref.shape}")
    if len(w) != len(ref):
        raise ValueError(
            f"the two band sets must share a k-path: {len(w)} vs {len(ref)} "
            "k-points. Take the sparse path as an index subset of one dense "
            "path rather than building two paths with different npoints.")

    dev, energy = [], []
    for ik in range(len(w)):
        good = ref[ik][np.isfinite(ref[ik])]
        if not good.size:
            continue
        for e in w[ik]:
            if not np.isfinite(e):
                continue
            dev.append(np.abs(good - e).min())
            energy.append(e)
    dev = np.asarray(dev) * 1e3                      # meV
    energy = np.asarray(energy)

    sel = np.ones(len(dev), dtype=bool)
    if window is not None:
        sel &= (energy >= window[0]) & (energy <= window[1])
    out = {
        "max": float(dev[sel].max()) if sel.any() else float("nan"),
        "rms": float(np.sqrt((dev[sel] ** 2).mean())) if sel.any() else float("nan"),
        "n_states": int(sel.sum()),
    }
    if fermi_ev is not None:
        near = sel & (np.abs(energy - fermi_ev) <= near_ef)
        out["max_near_ef"] = float(dev[near].max()) if near.any() else float("nan")
        out["rms_near_ef"] = (float(np.sqrt((dev[near] ** 2).mean()))
                              if near.any() else float("nan"))
    return out
