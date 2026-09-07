"""
Surface spectral function A(k_par, E) of a semi-infinite crystal cleaved
along an arbitrary (hkl) plane, by Lopez-Sancho/Lopez-Sancho/Rubio
iterative Green's-function decimation (`_sancho_rubio`, shared with
`analysis/transport.py`'s ballistic conductance).

Downfolding: the semi-infinite bulk below the surface is folded onto the
surface principal layer through the decimation self-energy Sigma(k_par, E):

    G_s(k_par, E) = [ (E + i eta) I - H00(k_par) - Sigma(k_par, E) ]^-1
    A(k_par, E)   = -1/pi Im Tr G_s(k_par, E)

Sigma resums all bulk layers below the surface, so A includes surface
states plus the surface-projected bulk continuum.

Geometry (arbitrary Miller index): for a surface normal to the (hkl)
plane, build a stacked surface cell with two in-plane lattice vectors
c1, c2 and a stacking vector c3 advancing one interlayer period along the
normal. The integer transformation C = [c1; c2; c3] (det = +-1) comes from
ASE's `ext_gcd`-based algorithm (`ase.build.general_surface`), applied to
the cell H(R) lives in. Miller indices are in the basis of the
`real_lattice` you pass -- the primitive Wannier cell, not necessarily the
conventional cubic cell used by ASE's own `surface()`.

The bulk H(R) is re-indexed into layers by m = R . C^-1: m[0], m[1] are
in-plane cell indices (conjugate to the 2D surface k_par), m[2] is the
layer index along c3. H00(k_par)/H01(k_par) are built like transport.py's
principal-layer blocks but k_par-dependent (Fourier over the two in-plane
directions).

Termination: `termination=0` is the natural cell-boundary termination.
For a multi-sublayer surface cell (e.g. perovskite (001): BaO vs TiO2
plane), pass WF centres (`wf_centres`): WFs are grouped into sublayers by
height along the surface normal (`wf_sublayers`), and `termination`
(0 .. n_sublayer-1) picks which sublayer is exposed. A cut plane is placed
just above the chosen sublayer and each hopping element H(R)[i,j] is
re-assigned to layer m_stack + kappa_j - kappa_i, kappa_w = floor(f_w -
cut) -- a bulk-preserving relabeling, so A_bulk is termination-independent
while A_surface changes.

Units: atomic units throughout (Hartree energies, Bohr lengths).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import gcd

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR
from .transport import _sancho_rubio, _ETA, _NTERX, _EPS7


# --------------------------------------------------------------------------
# Batched (multi-core) Green's-function decimation over k-points
# --------------------------------------------------------------------------

def _eval_layer_batched(layer_H: dict, layer: int, kpath: np.ndarray, nw: int) -> np.ndarray:
    """`_eval_layer` for a whole k-path at once -> (nk, nw, nw) complex."""
    nk = kpath.shape[0]
    out = np.zeros((nk, nw, nw), dtype=np.complex128)
    terms = layer_H.get(layer)
    if terms:
        m = np.array([[m1, m2] for m1, m2, _ in terms], dtype=np.float64)   # (nt, 2)
        blks = np.stack([blk for _, _, blk in terms])                       # (nt, nw, nw)
        phase = np.exp(2j * np.pi * (kpath @ m.T))                          # (nk, nt)
        out = np.einsum("kt,tij->kij", phase, blks)
    return out


def _blocks_batched(layers, kpath: np.ndarray):
    """(H00, H01) for every k at once, each (nk, npl*nw, npl*nw) complex."""
    nw, npl = layers.nw, layers.num_pl
    nb, nk = npl * nw, kpath.shape[0]
    Hl = {d: _eval_layer_batched(layers.layer_H, d, kpath, nw)
          for d in range(-npl, npl + 1)}
    H00 = np.zeros((nk, nb, nb), dtype=np.complex128)
    H01 = np.zeros((nk, nb, nb), dtype=np.complex128)
    for j in range(npl):
        for i in range(npl):
            H00[:, j * nw:(j + 1) * nw, i * nw:(i + 1) * nw] = Hl[i - j]
    for j in range(1, npl + 1):
        for i in range(j):
            H01[:, (j - 1) * nw:j * nw, i * nw:(i + 1) * nw] = Hl[i - (j - 1) + npl]
    return H00, H01


_BATCHED_LU_WARNED = []


def _batched_inv(A: torch.Tensor) -> torch.Tensor:
    """`torch.linalg.inv` over a stack, surviving the batched-LU bug.

    torch's batched complex LU raises `Pivots given to lu_solve must all be
    greater or equal to 1` (preceded by `Intel oneMKL ERROR: Parameter 6 was
    incorrect on entry to ZLASWP`) for ANY thread count above 1 on some
    torch/MKL builds -- reproduced here at 4 and 32 threads, fine at 1, on
    non-singular input. Inverting one matrix at a time uses the ordinary
    threaded LAPACK path, which is unaffected, so fall back to that rather than
    propagate a failure that is not the caller's doing.

    The recovered result is the right one (1e-11 against `backend="loop"` on the
    CuI(111) 222x222 layer), but MKL prints its own `Parameter 6 ... ZLASWP`
    line to stderr from C before torch raises, once per failing call, and that
    cannot be silenced from Python. Flooded stderr during a `backend="batched"`
    run is therefore cosmetic; `backend="loop"` avoids it.
    """
    try:
        return torch.linalg.inv(A)
    except RuntimeError as exc:
        if "Pivots given to lu_solve" not in str(exc):
            raise
        if not _BATCHED_LU_WARNED:
            _BATCHED_LU_WARNED.append(True)
            warnings.warn(
                "torch's batched LU failed on this build (a known bug at more "
                "than one thread, not a singular matrix); inverting per "
                "k-point instead. Results are unaffected. backend='loop' "
                "avoids this path entirely and is usually faster anyway.",
                RuntimeWarning, stacklevel=3)
        return torch.stack([torch.linalg.inv(a) for a in A])


def _sancho_rubio_batched(H00: torch.Tensor, H01: torch.Tensor, z: complex,
                          nterx: int = _NTERX, eps: float = _EPS7):
    """Batched (over k) Lopez-Sancho/Rubio decimation; H00/H01 are (nk, n, n)
    torch complex. Iterates until every k-point's increment is below ``eps``
    (so the result matches the per-k serial `_sancho_rubio` to precision)."""
    nk, n, _ = H00.shape
    I = torch.eye(n, dtype=H00.dtype, device=H00.device).expand(nk, n, n)
    g0 = _batched_inv(z * I - H00)
    tau = g0 @ H01.conj().transpose(-1, -2)
    taut = g0 @ H01
    T, Tt = tau.clone(), taut.clone()
    tsum, tsumt = taut.clone(), tau.clone()
    for _ in range(nterx):
        s2 = _batched_inv(I - tau @ taut - taut @ tau)
        tau2 = s2 @ (tau @ tau)
        taut2 = s2 @ (taut @ taut)
        T = T + tsum @ tau2
        tsum = tsum @ taut2
        Tt = Tt + tsumt @ taut2
        tsumt = tsumt @ tau2
        tau, taut = tau2, taut2
        r = tau2.abs().sum(dim=(-1, -2)).max().item()
        rt = taut2.abs().sum(dim=(-1, -2)).max().item()
        if r < eps and rt < eps:
            break
    return T, Tt


# --------------------------------------------------------------------------
# Geometry: (hkl) -> integer surface-cell transformation C = [c1; c2; c3]
# --------------------------------------------------------------------------

def surface_transformation(real_lattice: np.ndarray, miller) -> np.ndarray:
    """
    Integer transformation C (3x3, rows c1, c2 in-plane, c3 stacking) such
    that the surface-cell vectors are C @ real_lattice, for the (hkl)
    surface. Uses ASE's ext_gcd algorithm; `miller` is in the basis of
    `real_lattice` (the primitive Wannier cell -- see module docstring).

    det(C) = +-1, so np.linalg.inv(C) is integer (used to re-index H(R)).
    """
    from ase.build.general_surface import ext_gcd

    miller = np.asarray(miller, dtype=int)
    if miller.shape != (3,) or not miller.any():
        raise ValueError(f"invalid Miller index {miller}")
    # reduce by gcd so (220) == (110)
    g = gcd(gcd(abs(int(miller[0])), abs(int(miller[1]))), abs(int(miller[2])))
    miller = miller // g
    h, k, l = (int(x) for x in miller)
    h0, k0, l0 = (miller == 0)

    if (h0 and k0) or (h0 and l0) or (k0 and l0):     # two indices zero
        if not h0:
            c1, c2, c3 = (0, 1, 0), (0, 0, 1), (1, 0, 0)
        elif not k0:
            c1, c2, c3 = (0, 0, 1), (1, 0, 0), (0, 1, 0)
        else:
            c1, c2, c3 = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        return np.array([c1, c2, c3], dtype=np.int64)

    a1, a2, a3 = np.asarray(real_lattice, dtype=np.float64)
    p, q = ext_gcd(k, l)
    k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3), l * a2 - k * a3)
    k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3), l * a2 - k * a3)
    if abs(k2) > 1e-10:
        i = -int(round(k1 / k2))
        p, q = p + i * l, q - i * k
    a, b = ext_gcd(p * k + q * l, h)
    c1 = (p * k + q * l, -p * h, -q * h)
    c2 = np.array((0, l, -k)) // abs(gcd(l, k))
    c3 = (b, a * p, a * q)
    C = np.array([c1, c2, c3], dtype=np.int64)
    if abs(round(np.linalg.det(C))) != 1:
        raise ValueError(f"surface transformation for {miller} is not unimodular (det={np.linalg.det(C)})")
    return C


# --------------------------------------------------------------------------
# k_par-resolved principal-layer blocks
# --------------------------------------------------------------------------

@dataclass
class SurfaceLayers:
    """
    k_par-dependent principal-layer Hamiltonian builder for a semi-infinite
    (hkl) surface.  `.blocks(kpar)` returns (H00, H01), each
    (num_pl*nw, num_pl*nw), ready for `_sancho_rubio`.
    """
    C:        np.ndarray                # (3,3) int surface transformation
    C_inv:    np.ndarray                # (3,3) int inverse (R -> surface indices)
    layer_H:  dict                      # layer index l -> list of (m1, m2, H/degen)
    num_pl:   int                       # stacked cells per principal layer
    nw:       int
    max_layer: int
    face:     str = "top"

    @property
    def exposed(self) -> slice:
        """The cell of the principal layer the vacuum is on.

        NOT block 0 when face="top" and num_pl > 1. The blocks are ordered by
        increasing layer index (H00[j, i] = H(layer i - layer j)) and H01
        couples the principal layer to the one ABOVE, so the surface self-energy
        H01^dag Tt attaches the stack BELOW and the exposed cell is the LAST
        block. Taking block 0 there returns the first BURIED cell instead:
        checked against an explicit 3000-site chain, block 0 reproduced the
        second cell's LDOS to 1e-14 while differing from the outermost cell's by
        79% of its peak.
        """
        j = 0 if self.face == "bottom" else self.num_pl - 1
        return slice(j * self.nw, (j + 1) * self.nw)

    def _H_layer(self, layer: int, kpar) -> np.ndarray:
        """Sum_{R in this layer} H(R)/degen * exp(2pi i (k1 m1 + k2 m2))."""
        return _eval_layer(self.layer_H, layer, kpar, self.nw)

    def blocks(self, kpar) -> tuple[np.ndarray, np.ndarray]:
        nw, npl = self.nw, self.num_pl
        nb = npl * nw
        Hl = {d: self._H_layer(d, kpar) for d in range(-(npl), npl + 1)}
        H00 = np.zeros((nb, nb), dtype=np.complex128)
        H01 = np.zeros((nb, nb), dtype=np.complex128)
        for j in range(npl):
            for i in range(npl):
                H00[j * nw:(j + 1) * nw, i * nw:(i + 1) * nw] = Hl[i - j]
        for j in range(1, npl + 1):
            for i in range(j):
                H01[(j - 1) * nw:j * nw, i * nw:(i + 1) * nw] = Hl[i - (j - 1) + npl]
        return H00, H01


def wf_sublayers(real_lattice: np.ndarray, miller, wf_centres: np.ndarray,
                 tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    Group the Wannier functions into sublayers by their height along the
    (hkl) surface normal, for termination selection.

    Args:
      wf_centres : (nw, 3) WF centres in Bohr (Cartesian), e.g.
                   `WannierResult.centres_bohr`.
      tol        : fractional-height clustering tolerance.

    Returns:
      frac_z  : (nw,) each WF's stacking-fractional coordinate in [0, 1)
                (its position along the surface stacking vector c3).
      heights : (n_sublayer,) sorted unique sublayer heights.
    """
    C = surface_transformation(real_lattice, miller)
    B = C @ np.asarray(real_lattice, dtype=np.float64)     # surface cell (rows c1,c2,c3)
    frac = np.asarray(wf_centres, dtype=np.float64) @ np.linalg.inv(B)   # surface-cell frac
    frac_z = np.mod(frac[:, 2], 1.0)
    order = np.argsort(frac_z)
    heights = []
    for f in frac_z[order]:
        if not heights or abs(f - heights[-1]) > tol:
            heights.append(f)
    # merge a cluster that wraps around 1 ~ 0
    if len(heights) > 1 and (1.0 - heights[-1]) + heights[0] < tol:
        heights = heights[:-1]
    return frac_z, np.array(heights)


def build_surface_layers(
    hr:           HamiltonianR,
    real_lattice: np.ndarray,
    miller,
    hr_cutoff:    float = 1e-6,
    termination:  int = 0,
    wf_centres:   np.ndarray | None = None,
    face:         str = "top",
) -> SurfaceLayers:
    """
    Re-index the bulk H(R) into k_par-dependent surface layers for the
    (hkl) surface (see module docstring). `hr_cutoff` (Hartree) prunes
    negligible hoppings and sets the principal-layer thickness.

    `termination` selects which sublayer is exposed (needs `wf_centres`
    to resolve sublayers -- see `wf_sublayers`); 0 is the natural
    cell-boundary termination (no WF centres needed). See module
    docstring for the cut-plane / kappa relabeling.
    """
    if face not in ("top", "bottom"):
        raise ValueError(f"face must be 'top' or 'bottom', got {face!r}")
    C = surface_transformation(real_lattice, miller)
    C_inv = np.linalg.inv(C)
    C_inv_int = np.rint(C_inv).astype(np.int64)
    if not np.allclose(C_inv, C_inv_int, atol=1e-8):
        raise ValueError("surface transformation inverse is not integer (non-unimodular C)")

    H_R = hr.H_R.detach().cpu().numpy()
    R = np.asarray(hr.R_vectors, dtype=np.int64)
    degen = np.asarray(hr.degen, dtype=np.float64)
    nw = hr.nw

    kappa = np.zeros(nw, dtype=np.int64)
    if termination != 0 or wf_centres is not None:
        if wf_centres is None:
            raise ValueError("termination != 0 requires wf_centres to resolve sublayers")
        frac_z, heights = wf_sublayers(real_lattice, miller, wf_centres)
        ns = len(heights)
        if not (0 <= termination < ns):
            raise ValueError(f"termination {termination} out of range for {ns} sublayers {heights}")
        h = heights[termination]
        if face == "top":
            # cut just ABOVE the exposed sublayer, so it ends up the highest
            # one in the principal layer (which is the end the vacuum is on)
            h_other = heights[termination + 1] if termination + 1 < ns \
                else heights[0] + 1.0
        else:
            # face="bottom": cut just BELOW it, making it the lowest
            h_other = heights[termination - 1] if termination > 0 \
                else heights[-1] - 1.0
        cut = 0.5 * (h + h_other)
        kappa = np.floor(frac_z - cut).astype(np.int64)   # 0 above the cut, -1 below

    layer_H, max_layer = _layerize(H_R, R, degen, C_inv_int, hr_cutoff, kappa)
    # principal-layer thickness: largest populated |layer|
    num_pl = max((abs(lay) for lay in layer_H), default=0)
    num_pl = max(num_pl, 1)
    _warn_if_cutoff_set_num_pl(H_R, R, degen, C_inv_int, hr_cutoff, kappa,
                               num_pl, max_layer)
    return SurfaceLayers(C=C, C_inv=C_inv_int, layer_H=layer_H,
                         num_pl=num_pl, nw=nw, max_layer=max_layer,
                         face=face)


def _warn_if_cutoff_set_num_pl(H_R, R, degen, C_inv_int, cutoff, kappa,
                               num_pl, max_layer, rel_tol=1e-4):
    """Item 5: `hr_cutoff` silently decides the principal-layer thickness.

    The decimation assumes couplings reach at most one principal layer, so
    num_pl is read off the layers that SURVIVE the cutoff -- raise the cutoff
    and num_pl drops, silently truncating the Hamiltonian rather than the
    numerical noise it is meant to trim. On CuI(111) hr_cutoff 1e-3 gave
    num_pl 1 and a spectral function wrong by 44% of its peak, where 1e-4 and
    1e-5 both gave num_pl 2 and agreed to 0.04%. Nothing in the result says so,
    hence this.
    """
    if cutoff <= 0:
        return
    full, full_max = _layerize(H_R, R, degen, C_inv_int, 0.0, kappa)
    if full_max <= max_layer:
        return
    dropped = max(np.abs(blk).max()
                  for lay, terms in full.items() if abs(lay) > max_layer
                  for _, _, blk in terms)
    # Judged RELATIVE to the couplings kept, not on "anything survives beyond
    # the cut". A Loewdin H(R) has an exponential tail that never reaches
    # exactly zero, so an absolute test fires at every cutoff and says nothing:
    # on CuI(111) it complained equally about the 2.2e-4 Ha coupling that moved
    # the spectral function by 44% and the 2.6e-6 Ha one worth 0.04%.
    kept = max((np.abs(blk).max()
                for lay, terms in full.items() if lay != 0
                for _, _, blk in terms), default=0.0)
    if kept <= 0 or dropped <= rel_tol * kept:
        return
    warnings.warn(
        f"hr_cutoff={cutoff:g} Ha discarded couplings reaching layer "
        f"+-{full_max}, the largest {dropped:.2e} Ha against {kept:.2e} Ha of "
        f"retained interlayer coupling ({dropped / kept:.1e} of it), leaving "
        f"num_pl={num_pl}. That is a truncation of the Hamiltonian rather than "
        f"of numerical noise: the decimation treats a coupling reaching beyond "
        f"one principal layer as absent. Lower hr_cutoff until the spectral "
        f"function stops moving.", RuntimeWarning, stacklevel=3)


def build_slab(
    hr:           HamiltonianR,
    real_lattice: np.ndarray,
    miller,
    n_layers:     int,
    hr_cutoff:    float = 1e-6,
    termination:  int = 0,
    wf_centres:   np.ndarray | None = None,
    vacuum:       float = 20.0,
) -> HamiltonianR:
    """
    Cut a FINITE slab of `n_layers` surface cells out of the bulk H(R) and
    return it as an ordinary 2D-periodic `HamiltonianR`.

    Where `surface_spectral_function` folds the semi-infinite bulk into a self
    energy, this keeps the slab explicit: the result is a Hamiltonian like any
    other, with ``n_layers * hr.nw`` orbitals, in-plane R vectors ``(m1, m2, 0)``
    and Wannier centres placed at ``tau_w + L * c3``. That matters whenever an
    analysis needs the OPERATOR rather than a Green's function -- a Floquet
    drive (`waw.analysis.floquet`) needs the real-space bond vectors to build
    Peierls phases, and a self-energy has none.

    Both surfaces are exposed, so a topological surface state appears twice,
    once per face, degenerate up to the residual tunnelling through the slab.
    Project onto the outermost layer (`slab_layer_weights`) to isolate one.

    Args:
      n_layers : surface cells stacked along c3. Must exceed the principal
                 layer thickness or the two faces hybridise; the returned
                 object carries ``.slab_num_pl`` so you can check.
      vacuum   : Bohr of empty space added to the third lattice vector. It only
                 has to keep periodic images from touching, since every R has
                 ``m3 = 0`` by construction and no hopping crosses it.

    Returns a `HamiltonianR` with extra attributes ``slab_n_layers``,
    ``slab_nw_cell`` and ``slab_num_pl``.
    """
    if n_layers < 1:
        raise ValueError("build_slab: n_layers must be >= 1")
    layers = build_surface_layers(hr, real_lattice, miller, hr_cutoff=hr_cutoff,
                                  termination=termination, wf_centres=wf_centres)
    nw, C = layers.nw, layers.C
    if n_layers <= layers.num_pl:
        warnings.warn(
            f"build_slab: n_layers={n_layers} does not exceed the principal-layer "
            f"thickness {layers.num_pl}; the two faces will hybridise and any "
            f"surface state you find is a finite-size artefact.", stacklevel=2)

    B = np.asarray(C, dtype=np.float64) @ np.asarray(real_lattice, dtype=np.float64)
    c3 = B[2]

    # collect the in-plane R set over all layers
    inplane = sorted({(m1, m2) for terms in layers.layer_H.values()
                      for (m1, m2, _) in terms})
    index = {rv: i for i, rv in enumerate(inplane)}
    nw_slab = n_layers * nw
    H = np.zeros((len(inplane), nw_slab, nw_slab), dtype=np.complex128)
    for lay, terms in layers.layer_H.items():
        # a hopping from layer L to layer L + lay survives only if both are in
        for L in range(n_layers):
            Lp = L + lay
            if not (0 <= Lp < n_layers):
                continue
            for m1, m2, blk in terms:
                H[index[(m1, m2)], L * nw:(L + 1) * nw, Lp * nw:(Lp + 1) * nw] += blk

    R_vectors = np.array([[m1, m2, 0] for (m1, m2) in inplane], dtype=np.int64)
    tau = (np.zeros((nw, 3)) if hr.centres is None
           else np.asarray(hr.centres, dtype=np.float64))
    if hr.centres is not None and (termination != 0 or wf_centres is not None):
        frac_z, heights = wf_sublayers(real_lattice, miller,
                                       wf_centres if wf_centres is not None else tau)
        ns = len(heights)
        h = heights[termination]
        h_next = heights[termination + 1] if termination + 1 < ns else heights[0] + 1.0
        kappa = np.floor(frac_z - 0.5 * (h + h_next)).astype(np.int64)
        tau = tau - kappa[:, None] * c3[None, :]
    centres = np.concatenate([tau + L * c3[None, :] for L in range(n_layers)], axis=0)

    n3 = np.linalg.norm(c3)
    lat = np.array([B[0], B[1], c3 * (n_layers + max(vacuum / max(n3, 1e-30), 1.0))])
    out = HamiltonianR(
        H_R=torch.as_tensor(H, dtype=torch.complex128),
        R_vectors=R_vectors, degen=np.ones(len(inplane), dtype=np.int64),
        nw=nw_slab, centres=centres, real_lattice=lat,
        mp_grid=(hr.mp_grid[0] if hr.mp_grid is not None else 1,
                 hr.mp_grid[1] if hr.mp_grid is not None else 1, 1))
    out.slab_n_layers = n_layers
    out.slab_nw_cell = nw
    out.slab_num_pl = layers.num_pl
    return out


def slab_layer_weights(hr_slab: HamiltonianR, n_surface_cells: int = 1,
                       face: str = "bottom") -> np.ndarray:
    """
    Orbital weights selecting the outermost `n_surface_cells` of a slab, for
    projecting a spectral function onto one face.

    ``face='bottom'`` keeps layers 0.., ``'top'`` the last ones. A slab has two
    surfaces and they carry the same states, so a plot made without this shows
    every surface feature twice.
    """
    n, nwc = hr_slab.slab_n_layers, hr_slab.slab_nw_cell
    w = np.zeros(n * nwc)
    if face == "bottom":
        w[:n_surface_cells * nwc] = 1.0
    elif face == "top":
        w[-n_surface_cells * nwc:] = 1.0
    else:
        raise ValueError("face must be 'bottom' or 'top'")
    return w


def _layerize(op_R: np.ndarray, R: np.ndarray, degen: np.ndarray,
              C_inv_int: np.ndarray, cutoff: float,
              kappa: np.ndarray | None = None) -> tuple[dict, int]:
    """Group op(R)/degen by stacking layer m[2]=R.C^-1 (see module docstring).
    With `kappa` (per-WF integer cell shift, for a chosen termination), each
    element (i,j) is re-assigned to layer m[2] + kappa_j - kappa_i.
    Returns {layer -> [(m1, m2, block), ...]} and the max |layer| populated."""
    m = R @ C_inv_int
    nw = op_R.shape[-1]
    if kappa is None or not np.any(kappa):
        deltas = np.array([0])
        kd = None
    else:
        kd = kappa[None, :] - kappa[:, None]     # (nw, nw): delta for element [i, j]
        deltas = np.unique(kd)
    layer_H: dict = {}
    max_layer = 0
    for idx in range(R.shape[0]):
        block = op_R[idx] / degen[idx]
        if np.abs(block).max() <= cutoff:
            continue
        m1, m2, ms = int(m[idx, 0]), int(m[idx, 1]), int(m[idx, 2])
        for d in deltas:
            sub = block if kd is None else np.where(kd == d, block, 0.0)
            if np.abs(sub).max() <= cutoff:
                continue
            lay = ms + int(d)
            max_layer = max(max_layer, abs(lay))
            layer_H.setdefault(lay, []).append((m1, m2, sub))
    return layer_H, max_layer


def _eval_layer(layer_H: dict, layer: int, kpar, nw: int) -> np.ndarray:
    """Sum_{R in this layer} block * exp(2pi i (k1 m1 + k2 m2))  -> (nw, nw)."""
    out = np.zeros((nw, nw), dtype=np.complex128)
    terms = layer_H.get(layer)
    if terms is not None:
        k1, k2 = kpar[0], kpar[1]
        for m1, m2, blk in terms:
            out += blk * np.exp(2j * np.pi * (k1 * m1 + k2 * m2))
    return out


# --------------------------------------------------------------------------
# Spectral function
# --------------------------------------------------------------------------

@dataclass
class SurfaceSpectralFunction:
    """A(k_par, E) surface spectral function (atomic units)."""
    kpath:      np.ndarray   # (nk, 2) surface-BZ fractional coords
    energies:   np.ndarray   # (nE,) Hartree
    A_surface:  np.ndarray   # (nk, nE) top-surface spectral function
    A_bulk:     np.ndarray   # (nk, nE) bulk-projected spectral function
    num_pl:     int
    C:          np.ndarray
    A_up:       np.ndarray | None = None   # (nk, nE) majority-spin (+axis) surface weight
    A_dn:       np.ndarray | None = None   # (nk, nE) minority-spin (-axis) surface weight
    A_arpes:    np.ndarray | None = None   # (nk, nE) matrix-element-weighted intensity M† A_top M


def surface_spectral_function(
    hr:            HamiltonianR,
    real_lattice:  np.ndarray,
    miller,
    kpath:         np.ndarray,
    energies:      np.ndarray,
    eta:           float = _ETA,
    hr_cutoff:     float = 1e-6,
    termination:   int = 0,
    nterx:         int = _NTERX,
    spin_op_r=None,
    spin_axis:     tuple[float, float, float] = (0.0, 0.0, 1.0),
    wf_centres:    np.ndarray | None = None,
    matrix_element: np.ndarray | None = None,
    backend:       str = "loop",
    face:          str = "top",
) -> SurfaceSpectralFunction:
    """
    Surface spectral function A(k_par, E) = -1/pi Im Tr G_s along a 2D
    k_par path, for the semi-infinite (hkl) surface (see module docstring).

    Args:
      hr, real_lattice : bulk Wannier Hamiltonian (atomic units) + lattice (Bohr)
      miller           : (h, k, l), in the primitive `real_lattice` basis
      kpath            : (nk, 2) surface-BZ fractional coords (conjugate to c1, c2)
      energies         : (nE,) Hartree
      eta              : Green's-function broadening (Hartree)
      hr_cutoff        : hopping prune / principal-layer threshold (Hartree)
      termination      : which sublayer to expose (needs wf_centres; see
                         `build_surface_layers` / `wf_sublayers`). 0 = the
                         natural cell-boundary termination.
      wf_centres       : (nw, 3) WF centres in Bohr, required for termination != 0.
      spin_op_r        : optional real-space spin operator SS(R), (nR, nw, nw, 3),
                         on the same R grid as `hr` (e.g. from
                         `spin_texture.spin_operator_r`). When given, the surface
                         spectral function is spin-resolved into majority/minority
                         channels along `spin_axis` (default +z):
                         A_up/dn = -1/pi Im Tr[(I +- S.axis)/2 . G_s], projected
                         with the surface-layer (top) spin operator block.
      spin_axis        : quantization axis for the majority/minority split.
      matrix_element   : optional (nk, nw) complex ARPES matrix element M over
                         the surface-cell Wannier orbitals (hr basis order).
                         When given, the intensity A_arpes = -1/pi Im[M† G_top M]
                         = M† A_top M is computed -- the matrix-element-weighted
                         (photon-energy/polarization-dependent) simulated ARPES
                         signal, cf. `analysis.arpes.photoemission_matrix_element`.
      backend          : "loop" (default) inverts per k-point, letting LAPACK
                         thread inside each inversion. "batched" stacks the
                         k-path into one batched inversion instead. "loop" is
                         the default because batched is not actually faster
                         here -- on a 222x222 CuI(111) principal layer, loop at
                         32 threads took 8.7 s where batched at 1 thread took
                         17.1 s -- and because torch's batched complex LU is
                         broken above one thread on some builds (see
                         `_batched_inv`, which recovers from it rather than
                         letting it propagate). The two agree to machine
                         precision; pick "batched" only if measurement on your
                         own problem says it wins.

    Returns a `SurfaceSpectralFunction` with the top-surface A_surface and the
    bulk-projected A_bulk (nk, nE); A_up/A_dn are populated iff spin_op_r given
    (A_up + A_dn == A_surface); A_arpes iff matrix_element given.
    """
    layers = build_surface_layers(hr, real_lattice, miller, hr_cutoff=hr_cutoff,
                                  termination=termination, wf_centres=wf_centres,
                                  face=face)
    kpath = np.asarray(kpath, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    nk, nE = kpath.shape[0], energies.shape[0]
    npl, nw = layers.num_pl, layers.nw

    A_surf = np.empty((nk, nE), dtype=np.float64)
    A_bulk = np.empty((nk, nE), dtype=np.float64)
    # The cell the vacuum is on -- the LAST block for face="top", the first for
    # face="bottom". See `SurfaceLayers.exposed`; hardcoding block 0 returned a
    # buried cell whenever num_pl > 1.
    top = layers.exposed

    # ARPES matrix-element weighting: intensity I = M† A_top M with A_top the
    # top-layer spectral matrix (-1/pi Im G_top). M (nk, nw) is the per-k
    # photoemission matrix element over the surface-cell Wannier orbitals (same
    # basis order as hr), e.g. from `analysis.arpes.photoemission_matrix_element`.
    Me = None
    if matrix_element is not None:
        Me = np.asarray(matrix_element, dtype=np.complex128)
        if Me.shape != (nk, nw):
            raise ValueError(f"matrix_element must be (nk, nw)=({nk}, {nw}), got {Me.shape}")
        A_arpes = np.empty((nk, nE), dtype=np.float64)

    # Spin projector: surface-layer (layer 0) block of S.axis, k_par-dependent.
    spin_layer0 = None
    if spin_op_r is not None:
        ss = spin_op_r.detach().cpu().numpy() if hasattr(spin_op_r, "detach") else np.asarray(spin_op_r)
        axis = np.asarray(spin_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        ss_axis = ss @ axis                                  # (nR, nw, nw)
        R = np.asarray(hr.R_vectors, dtype=np.int64)
        degen = np.asarray(hr.degen, dtype=np.float64)
        spin_layer_H, _ = _layerize(ss_axis, R, degen, layers.C_inv, cutoff=0.0)
        spin_layer0 = spin_layer_H
        A_up = np.empty((nk, nE), dtype=np.float64)
        A_dn = np.empty((nk, nE), dtype=np.float64)

    if backend == "batched":
        # k-points vectorized as stacked (nk, nb, nb) torch tensors: the
        # decimation and Green's-function inversions run as batched BLAS over
        # the whole k-path (multi-core), looping only over the nE energies.
        H00_np, H01_np = _blocks_batched(layers, kpath)
        cdt = torch.complex128
        H00t = torch.from_numpy(H00_np).to(cdt)
        H01t = torch.from_numpy(H01_np).to(cdt)
        H01th = H01t.conj().transpose(-1, -2)
        nb = H00t.shape[-1]
        Ib = torch.eye(nb, dtype=cdt).expand(nk, nb, nb)
        if spin_layer0 is not None:
            Sz0 = torch.from_numpy(_eval_layer_batched(spin_layer0, 0, kpath, nw)).to(cdt)
            Inw = torch.eye(nw, dtype=cdt).expand(nk, nw, nw)
            P_up_b = 0.5 * (Inw + Sz0)
            P_dn_b = 0.5 * (Inw - Sz0)
        Me_t = torch.from_numpy(Me).to(cdt) if Me is not None else None
        for ie, E in enumerate(energies):
            z = complex(E, eta)
            T, Tt = _sancho_rubio_batched(H00t, H01t, z, nterx=nterx, eps=_EPS7)
            # sigma_s attaches the semi-infinite stack on the far side of the
            # exposed face: below it for face="top" (H01^dag Tt), above it for
            # face="bottom" (H01 T). sigma_b is the other one, and adding both
            # restores the bulk.
            sigma_s, sigma_b = ((H01th @ Tt, H01t @ T) if face == "top"
                                else (H01t @ T, H01th @ Tt))
            Gs = _batched_inv(z * Ib - H00t - sigma_s)
            Gb = _batched_inv(z * Ib - H00t - sigma_s - sigma_b)
            Gtop = Gs[:, top, top]
            A_surf[:, ie] = (-torch.einsum("kii->k", Gtop).imag / np.pi).numpy()
            A_bulk[:, ie] = (-torch.einsum("kii->k", Gb[:, top, top]).imag / np.pi).numpy()
            if spin_layer0 is not None:
                A_up[:, ie] = (-torch.einsum("kij,kji->k", P_up_b, Gtop).imag / np.pi).numpy()
                A_dn[:, ie] = (-torch.einsum("kij,kji->k", P_dn_b, Gtop).imag / np.pi).numpy()
            if Me_t is not None:
                A_arpes[:, ie] = (-torch.einsum("ki,kij,kj->k", Me_t.conj(), Gtop, Me_t).imag / np.pi).numpy()
    else:
        for ik in range(nk):
            H00, H01 = layers.blocks(kpath[ik])
            I = np.eye(H00.shape[0], dtype=np.complex128)
            if spin_layer0 is not None:
                Sz0 = _eval_layer(spin_layer0, 0, kpath[ik], nw)     # (nw, nw) top-layer S.axis
                P_up = 0.5 * (np.eye(nw) + Sz0)
                P_dn = 0.5 * (np.eye(nw) - Sz0)
            for ie, E in enumerate(energies):
                z = E + 1j * eta
                T, Tt = _sancho_rubio(H00, H01, z, nterx=nterx, eps=_EPS7)
                # Surface self-energy from the semi-infinite bulk on the far
                # side of the exposed face: below it for face="top", above it
                # for face="bottom". Adding the other one back gives the bulk.
                sigma_s, sigma_b = ((H01.conj().T @ Tt, H01 @ T) if face == "top"
                                    else (H01 @ T, H01.conj().T @ Tt))
                Gs = np.linalg.inv(z * I - H00 - sigma_s)
                Gb = np.linalg.inv(z * I - H00 - sigma_s - sigma_b)
                Gtop = Gs[top, top]
                A_surf[ik, ie] = -np.trace(Gtop).imag / np.pi
                A_bulk[ik, ie] = -np.trace(Gb[top, top]).imag / np.pi
                if spin_layer0 is not None:
                    A_up[ik, ie] = -np.trace(P_up @ Gtop).imag / np.pi
                    A_dn[ik, ie] = -np.trace(P_dn @ Gtop).imag / np.pi
                if Me is not None:
                    m = Me[ik]
                    A_arpes[ik, ie] = -np.imag(m.conj() @ Gtop @ m) / np.pi

    return SurfaceSpectralFunction(kpath=kpath, energies=energies,
                                   A_surface=A_surf, A_bulk=A_bulk,
                                   num_pl=npl, C=layers.C,
                                   A_up=(A_up if spin_layer0 is not None else None),
                                   A_dn=(A_dn if spin_layer0 is not None else None),
                                   A_arpes=(A_arpes if Me is not None else None))
