"""
UPF (v2) pseudopotential parsing for pw2wannier90's ``atom_proj_ext``: read
a pseudopotential's own radial grid and pseudo-atomic-orbital radial
wavefunctions, and re-serialize them in the plain-text format
pw2wannier90's ``atom_proj_ext=.true.``/``atom_proj_dir=...`` expects,
handing its own projectors back through the external-projector file path
instead of the embedded-in-UPF one.

File format for ``<atom_proj_dir>/<Species>.dat`` (``read_atomproj`` in
``pw2wannier90.f90``, module ``atproj``): ``ngrid, nproj``, then the ``l``
list, then per radial point ``xgrid(i), rgrid(i), (radial(i,j),
j=1,nproj)`` where ``rgrid = exp(xgrid)`` -- the ``ln(r)`` column is
required, a 2-column ``r``-only format makes pw2wannier90.x crash with a
Fortran "End of file" error:

    line 1: <n_radial_points> <n_l_channels>
    line 2: <l values present, space-separated, ascending>
    remaining <n_radial_points> lines:
        ln(r)  r  R_{l[0]}(r)  R_{l[1]}(r)  ...   (one radial-function
                                    column per l value, in line 2's order)

UPF tags are plain-text with quoted attributes but not strictly XML
(attributes may be split one-per-line) -- parsed here with small DOTALL
regexes rather than a real XML parser.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TAG_RE_TEMPLATE = r'<{name}\b(.*?)>(.*?)</{name}>'


def _find_tag(text: str, name: str) -> tuple[str, str]:
    """Return (attrs_text, body_text) for the FIRST ``<name ...>...</name>``
    block (DOTALL so multi-line attribute lists and data blocks match)."""
    pattern = re.compile(_TAG_RE_TEMPLATE.format(name=re.escape(name)), re.DOTALL)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"{name!r} tag not found in UPF file")
    return m.group(1), m.group(2)


def _attr(attrs_text: str, key: str) -> str:
    # \b anchors the match to a WHOLE attribute name -- without it, e.g.
    # key="l" would spuriously match inside `label="3S"` (which literally
    # ends in "l=").
    m = re.search(r'\b' + re.escape(key) + r'\s*=\s*"([^"]*)"', attrs_text)
    if not m:
        raise ValueError(f"attribute {key!r} not found in {attrs_text!r}")
    return m.group(1).strip()


def _read_beta_j(text: str, nproj: int):
    """j of every KB projector from ``PP_SPIN_ORB``, or None if not present.

    ``<PP_RELBETA.n index="n" lll="l" jjj="j"/>`` -- self-closing, so it needs
    its own regex rather than `_find_tag`.
    """
    if "<PP_SPIN_ORB" not in text:
        return None
    jjj = np.full(nproj, np.nan)
    for m in re.finditer(r"<PP_RELBETA\.(\d+)\b([^>]*?)/?>", text):
        n = int(m.group(1))
        raw = _attr(m.group(2), "jjj")
        if 1 <= n <= nproj and raw:
            jjj[n - 1] = float(raw)
    return None if np.isnan(jjj).any() else jjj


def average_j_channels(betas, ells, dij, jjj):
    """
    Collapse the j = l +- 1/2 projectors of a FULLY RELATIVISTIC pseudopotential
    into scalar-relativistic ones -- Quantum ESPRESSO's ``average_pp``.

    A pseudopotential with ``has_so="T"`` carries two KB projectors per l > 0,
    one for each j. Run collinear (``lspinorb=.false.``) QE averages them before
    doing anything, so its bands, its density and its dvscf are all
    scalar-relativistic. Any code that then rebuilds a bare electron-phonon
    vertex from the RAW file must average them too, or it counts every l > 0
    channel twice.

    That is not a small effect. On fcc Co (Co-rel.upf: 10 projectors, ells
    [0,0,1,1,1,1,2,2,2,2]) the unaveraged vertex gave lambda = 3.9 / 4.8 against
    a literature 0.3-0.5, while Nb -- whose pseudopotential is scalar and so
    passes through this untouched -- validated at 1.20. The signature is a
    pseudopotential-dependent error in lambda with everything else (charge,
    Wannier fidelity, dvscf spin convention) verifiably correct.

    The averaging is QE's, weighting each j by its multiplicity 2j+1:

        D     = [(l+1) D_{j=l+1/2} + l D_{j=l-1/2}] / (2l+1)
        beta  = [(l+1) sqrt(D_{j=l+1/2}/D) beta_{j=l+1/2}
                 + l   sqrt(D_{j=l-1/2}/D) beta_{j=l-1/2}] / (2l+1)

    The sqrt(D/D_avg) factors are what make |beta> D <beta| average correctly
    rather than |beta> alone; dropping them is a subtler version of the same
    bug.

    Returns ``(betas, ells, dij)`` with the l > 0 pairs collapsed. Assumes the
    KB matrix is diagonal, which is true of norm-conserving pseudopotentials
    (checked here) and not of ultrasoft ones -- QE refuses those too.
    """
    dij = np.asarray(dij, dtype=np.float64)
    if np.abs(dij - np.diag(np.diag(dij))).max() > 1e-12 * max(np.abs(dij).max(), 1.0):
        raise ValueError("average_j_channels: KB matrix is not diagonal; "
                         "j-averaging is defined here only for that case")
    ells = list(ells)
    d = np.diag(dij).copy()
    out_b, out_l, out_d = [], [], []
    n, i = len(ells), 0
    while i < n:
        l = ells[i]
        if l == 0:
            out_b.append(betas[i]); out_l.append(l); out_d.append(d[i])
            i += 1
            continue
        if i + 1 >= n or ells[i + 1] != l:
            raise ValueError(f"average_j_channels: projector {i} (l={l}) has no "
                             f"j partner; unexpected UPF ordering")
        a, b = i, i + 1                       # a, b are the two j channels
        if abs(jjj[a] - (l + 0.5)) < 1e-7:    # put j = l+1/2 first
            jp, jm = a, b
        else:
            jp, jm = b, a
        if abs(jjj[jp] - (l + 0.5)) > 1e-7 or abs(jjj[jm] - (l - 0.5)) > 1e-7:
            raise ValueError(f"average_j_channels: projectors {a},{b} are not a "
                             f"j = l +- 1/2 pair for l = {l}")
        d_avg = ((l + 1) * d[jp] + l * d[jm]) / (2 * l + 1)
        rp, rm = d[jp] / d_avg, d[jm] / d_avg
        if rp < 0 or rm < 0:
            raise ValueError("average_j_channels: D coefficients of a j pair "
                             "have opposite signs; QE's averaging is undefined")
        out_b.append(((l + 1) * np.sqrt(rp) * np.asarray(betas[jp])
                      + l * np.sqrt(rm) * np.asarray(betas[jm])) / (2 * l + 1))
        out_l.append(l)
        out_d.append(d_avg)
        i += 2
    return out_b, out_l, np.diag(np.array(out_d, dtype=np.float64))


def read_norm_conserving(upf_path, *, average_j: bool = True) -> dict:
    """
    Parse the pieces of a norm-conserving UPF v2 pseudopotential needed to
    build the BARE ionic phonon perturbation for electron-phonon coupling
    (`analysis.elph.bare_local_dv` / `analysis.elph.
    KleinmanBylanderPerturbation`): QE's ``dvscf`` output is the INDUCED
    (self-consistent response) part only -- the bare local gradient and the
    nonlocal Kleinman-Bylander derivative must be reconstructed from the
    pseudopotential itself (QE does this in ``PHonon/PH/dvqpsi_us.f90``).

    Returns a dict (energies converted Ry -> Hartree at this boundary,
    matching `dvscf_io.read_dvscf`'s own convention; radial grid in Bohr):

      r, rab : (nr,) float64 -- radial grid and integration weights (Bohr)
      vloc   : (nr,) float64 -- local potential v_loc(r), HARTREE
      zv     : float         -- valence charge
      betas  : list of (nr,) float64 -- UPF's r*beta_n(r) projectors (raw
               UPF radial normalization, as `PP_DIJ` expects)
      ells   : list of int   -- angular momentum of each projector

    ``average_j`` (default True) collapses the j = l +- 1/2 projectors of a
    fully relativistic pseudopotential into scalar-relativistic ones, exactly
    as QE does for a collinear calculation -- see `average_j_channels`. Pass
    False only if you intend to handle the spin-orbit projectors yourself.
      dij    : (nbeta, nbeta) float64 -- KB couplings, HARTREE
      core_correction : bool -- NLCC flag. When True, the exchange-
               correlation response to the rigidly-displaced core charge is
               an additional bare-perturbation term NOT reconstructed by
               `analysis.elph` (a documented ~1-5% |g| residual for Al).
    """
    text = Path(upf_path).read_text()
    m = re.search(r"<PP_HEADER\b(.*?)/?>", text, re.DOTALL)
    if not m:
        raise ValueError(f"{upf_path}: PP_HEADER not found")
    header = m.group(1)
    if _attr(header, "pseudo_type").strip().upper() != "NC":
        raise ValueError(
            f"{upf_path}: pseudo_type={_attr(header, 'pseudo_type')!r} -- only "
            "norm-conserving pseudopotentials are supported (no USPP/PAW "
            "augmentation terms are reconstructed)"
        )
    zv = float(_attr(header, "z_valence"))
    nproj = int(_attr(header, "number_of_proj"))
    core_correction = _attr(header, "core_correction").strip().upper().startswith("T")

    _, r_body = _find_tag(text, "PP_R")
    r = np.array(r_body.split(), dtype=np.float64)
    _, rab_body = _find_tag(text, "PP_RAB")
    rab = np.array(rab_body.split(), dtype=np.float64)
    _, vloc_body = _find_tag(text, "PP_LOCAL")
    vloc_ry = np.array(vloc_body.split(), dtype=np.float64)

    betas, ells = [], []
    for n in range(1, nproj + 1):
        m = re.search(
            rf"<PP_BETA\.{n}\b(.*?)>(.*?)</PP_BETA\.{n}>", text, re.DOTALL,
        )
        if not m:
            raise ValueError(f"{upf_path}: PP_BETA.{n} not found (expected {nproj})")
        ells.append(int(_attr(m.group(1), "angular_momentum")))
        beta = np.array(m.group(2).split(), dtype=np.float64)
        betas.append(beta[: len(r)])

    _, dij_body = _find_tag(text, "PP_DIJ")
    dij_ry = np.array(dij_body.split(), dtype=np.float64).reshape(nproj, nproj)

    # A fully relativistic pseudopotential carries j = l +- 1/2 projector pairs.
    # QE averages them for any collinear run, so everything it produces is
    # scalar-relativistic; a bare vertex rebuilt from the raw file must match.
    jjj = _read_beta_j(text, nproj)
    if average_j and jjj is not None:
        betas, ells, dij_h = average_j_channels(betas, ells, 0.5 * dij_ry, jjj)
        dij_ry = 2.0 * dij_h

    # partial core charge rho_atc(r) for the non-linear core correction (NLCC),
    # PP_NLCC -- present only when core_correction=T. QE convention: the core
    # charge density rho_core(r) = rho_atc(|r|) (electrons/Bohr^3, spherical),
    # total core charge = integral 4 pi r^2 rho_atc dr. Used by analysis.elph's
    # NLCC term (analysis.xc kernel on the displaced core charge).
    rho_atc = None
    if core_correction:
        m = re.search(r"<PP_NLCC\b(.*?)>(.*?)</PP_NLCC>", text, re.DOTALL)
        if m:
            rho_atc = np.array(m.group(2).split(), dtype=np.float64)[: len(r)]

    return {
        "r": r, "rab": rab, "vloc": 0.5 * vloc_ry, "zv": zv,
        "betas": betas, "ells": ells, "dij": 0.5 * dij_ry,
        "core_correction": core_correction,
        "rho_atc": rho_atc,
    }


def read_pswfc(upf_path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Parse a UPF pseudopotential's radial grid (``PP_MESH/PP_R``) and its
    pseudo-atomic-orbital radial wavefunctions (``PP_PSWFC/PP_CHI.i``, each
    tagged with an ``l="..."`` attribute).

    Returns ``{l: (r, chi)}`` -- both ``(n_radial_points,)`` float64, `r`
    in Bohr (UPF's native unit; pw2wannier90's ``atom_proj_ext`` expects
    the same, no Angstrom conversion at this boundary).

    If a UPF carries more than one ``PP_CHI`` block for the same `l` (e.g.
    a semicore + valence channel), only the last one encountered is kept --
    the ``atom_proj_ext`` format has room for one radial function per l.
    """
    text = Path(upf_path).read_text()
    _, r_body = _find_tag(text, "PP_R")
    r = np.array(r_body.split(), dtype=np.float64)

    _, pswfc_body = _find_tag(text, "PP_PSWFC")
    radial: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for m in re.finditer(r'<(PP_CHI\.\d+)\b(.*?)>(.*?)</\1>', pswfc_body, re.DOTALL):
        attrs_text, body = m.group(2), m.group(3)
        l = int(_attr(attrs_text, "l"))
        chi = np.array(body.split(), dtype=np.float64)
        if chi.shape != r.shape:
            raise ValueError(
                f"{upf_path}: PP_CHI l={l} has {chi.shape[0]} points, "
                f"PP_R has {r.shape[0]}"
            )
        radial[l] = (r, chi)
    return radial


def write_atom_proj_ext(dir_path, species_to_radial: dict[str, dict[int, tuple]]) -> None:
    """
    Write one ``<Species>.dat`` file per entry of `species_to_radial` into
    `dir_path` (created if needed), in pw2wannier90's ``atom_proj_ext``
    format (see module docstring).

    `species_to_radial`: ``{species: {l: (r, chi)}}`` -- typically
    `read_pswfc`'s own return value, keyed by species symbol. All `l`
    channels for one species MUST share the same radial grid `r` (true by
    construction when they come from a single UPF's own `PP_MESH`).

    ``xgrid = ln(r)`` requires `r` > 0 throughout. Some UPFs use a linear
    ``PP_MESH`` starting at ``r[0] = 0``; the leading point is then dropped
    for both `r` and every `chi` column, since ``chi(r) = r * R_l(r)`` is
    identically 0 there for every `l`, so no information is lost.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    for species, radial in species_to_radial.items():
        ls = sorted(radial)
        r = radial[ls[0]][0]
        for l in ls[1:]:
            if not np.array_equal(radial[l][0], r):
                raise ValueError(
                    f"{species}: l={l} radial grid differs from l={ls[0]}'s"
                )
        columns = np.stack([radial[l][1] for l in ls], axis=1)   # (npts, nl)

        if r[0] <= 0.0:
            # linear mesh starting at r=0: ln(r) undefined there, and
            # chi(0)=0 for every l by construction (chi = r*R_l(r)) --
            # drop that one point rather than fabricate an x value.
            r, columns = r[1:], columns[1:]

        x = np.log(r)   # pw2wannier90's own xgrid; rgrid = exp(xgrid) = r

        lines = [f"{len(r)} {len(ls)}", " ".join(str(l) for l in ls)]
        for row_x, row_r, row_chi in zip(x, r, columns):
            lines.append(" ".join(
                [f"{row_x:.15e}", f"{row_r:.15e}"] + [f"{c:.15e}" for c in row_chi]
            ))

        (dir_path / f"{species}.dat").write_text("\n".join(lines) + "\n")


def atom_proj_column_atoms(atoms, species_to_radial: dict) -> "np.ndarray":
    """
    0-based atom index owning each column of the ``atom_proj``/
    ``atom_proj_ext`` overlap matrix (pw2wannier90's own ``.amn``-like
    output when ``atom_proj=.true.``), needed for atom-resolved quantities
    like the Pipek-Mezey functional's Mulliken charges
    (``core.spread.compute_pm_spread``).

    Column order, confirmed directly from ``pw2wannier90.f90``'s
    ``atomproj_wfc``/``atomic_wfc___`` (module ``atproj``): outer loop over
    atoms in structure order (``atoms``' own iteration order), then that
    atom's species' ``l`` channels in ascending order (matching
    ``write_atom_proj_ext``'s own ``sorted(radial)``, which is what
    ``read_atomproj`` parses back in file order with no re-sorting), then
    ``2*l + 1`` magnetic-quantum-number columns per channel. E.g. two Si
    atoms with s+p channels each (``read_pswfc``'s keys ``{0, 1}``) give
    columns ``[Si0-s, Si0-px, Si0-py, Si0-pz, Si1-s, Si1-px, Si1-py, Si1-pz]``
    -> atom index array ``[0,0,0,0,1,1,1,1]``.

    ``species_to_radial``: ``{species: {l: (r, chi)}}``, the same dict
    `write_atom_proj_ext` takes (typically built from `read_pswfc` per
    species).
    """
    symbols = atoms.get_chemical_symbols()
    index = []
    for atom_idx, species in enumerate(symbols):
        for l in sorted(species_to_radial[species]):
            index += [atom_idx] * (2 * l + 1)
    return np.array(index, dtype=np.int64)
