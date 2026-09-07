"""Run directories that say what they are.

THE PROBLEM THIS SOLVES. Run directories are gitignored working state, so
nothing records what produced them or who still needs them. Measured on this
repo before the convention existed: 480 GB across 178 directories in a flat
namespace, of which 98 (~92 GB) were referenced by no notebook, script or test.
That number is not a delete list, and the reason it is not is the whole point --
the single largest of them, `ca_elph` at 51.6 GB, is unreferenced only because
the notebook that owns the campaign writes to `ca_gw_nb`. Deciding what to keep
required reading notebooks and guessing. It should require reading one file.

THE LAYOUT is `runs/<code>/<compound>/<purpose>/`, at the REPOSITORY ROOT rather
than under one workflow tree, because the same compound is used from both the
notebooks and the tutorials and an SCF worth tens of GB should be shared instead
of duplicated. Parameter sweeps are subdirectories of one purpose, never
siblings of it: seven top-level `al_scdm_auto_r*_s*_n*` directories is what the
alternative looks like.

Every run directory carries a `RUN.json` written by `stamp`. It answers, without
archaeology: which code, which compound, what for, who owns it, is it still
live, and which of its files are cheap to regenerate.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

CODES = ("qe", "vasp", "siesta", "pyscf", "epw", "w90ref", "model")
STATUSES = ("live", "archived", "superseded")
MANIFEST = "RUN.json"

# Regenerable from a completed run plus its inputs, and the bulk of the space:
# 137 GB of this repo's 480 sits in 231 files above 100 MB, essentially all
# wavefunctions. The derived artefacts an analysis re-reads -- hr.dat, the .npz
# caches, .mmn/.amn/.eig -- are small and are NOT listed here.
PRUNABLE = ("WAVECAR", "CHGCAR", "CHG", "*.wfc*", "*.hub*", "WAVEDER",
            "vasprun.xml", "PROCAR", "*.save/wfc*")


# How these compounds are written by the people who look for them. The
# structure file decides WHICH compound a run holds; this table only decides how
# it is spelled, because a formula reduced alphabetically is unfindable: MgB2
# lands under `b2mg`, BaTiO3 under `bao3ti`, NiI2 under `i2ni`. Anything absent
# falls back to pymatgen's electronegativity ordering, which gets the
# conventional spelling right by itself except for a few intermetallics (it
# calls Ag2NdCd "NdCdAg2"). Add to this table rather than accepting a name you
# would not think to look under.
NAMES = ["Ag2NdCd", "MgCoNi2O4", "Fe3RuN", "Bi2Se3", "NiI2", "MgB2", "BaTiO3",
         "SrVO3", "LaVO3", "SrMnO3", "GaAs", "CoSi", "CuI", "PdBi2", "SiH4",
         # C6H6 because reducing a MOLECULE in a box throws away its identity:
         # benzene's reduced formula is CH, which names nothing.
         "C6H6",
         "BC2N", "NiO", "CoO", "MgO", "Al", "Fe", "Cu", "Nb", "Co", "Te",
         "Ca", "Si", "Pt", "Ni", "Na", "Pb", "C", "W"]

# The structure files a run might carry, and the format ASE needs told. SIESTA
# is the reason for the explicit ones: ASE will not guess `.fdf` or
# `.STRUCT_OUT` from the extension, so a whole class of runs looks nameless.
_STRUCTURE_FILES = (("POSCAR", "vasp"), ("CONTCAR", "vasp"), ("*.cif", None),
                    ("*.scf.in", "espresso-in"), ("*.in", "espresso-in"),
                    ("*.STRUCT_OUT", "struct_out"), ("*.XV", None),
                    ("*.fdf", "siesta-in"), ("*/POSCAR", "vasp"),
                    ("*/*.STRUCT_OUT", "struct_out"), ("*/*.scf.in", "espresso-in"))


def compound_name(symbols) -> str | None:
    """Empirical formula, reduced and spelled the conventional way.

    NOT `get_chemical_formula(mode="reduce")`: it warns that an empirical
    formula is unavailable in that mode and then returns the unreduced string,
    so a 4-formula-unit cell comes back as `ndcdag2ndcdag2ndcdag2ndcdag2` -- a
    plausible-looking directory name that is pure noise. Dividing the counts by
    their gcd is unambiguous; the element ORDER then comes from `NAMES`.
    """
    from collections import Counter
    from functools import reduce as _reduce
    from math import gcd
    import re

    from ase.data import chemical_symbols

    c = Counter(symbols)
    # Real elements only. Handed junk, pymatgen parses "X0X1X2..." as its dummy
    # species and confidently answers "X" -- a name for nothing, which is worse
    # than admitting the structure could not be read.
    if not c or any(s not in chemical_symbols for s in c):
        return None
    g = _reduce(gcd, c.values())
    plain = "".join(f"{el}{n // g if n // g > 1 else ''}"
                    for el, n in sorted(c.items()))
    name = plain
    try:
        from pymatgen.core import Composition
        target = Composition(plain).reduced_composition
        for n in NAMES:
            if Composition(n).reduced_composition == target:
                name = n
                break
        else:
            name = Composition(plain).reduced_formula
    except Exception:                                           # noqa: BLE001
        pass
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    # a formula longer than this is a parse gone wrong, not a compound
    return name if 0 < len(name) <= 16 else None


def compound_from_structure(d) -> str | None:
    """The compound a run holds, read from whatever structure file it contains.

    Far more reliable than parsing a directory name: `ctrl_NiO`,
    `diagvar_nio_np16` and `aluminum_superfluid` all carry the compound in a
    form no prefix table will match, and the POSCAR beside them states it.
    """
    from ase.io import read as ase_read

    d = Path(d)
    for pat, fmt in _STRUCTURE_FILES:
        for f in sorted(d.glob(pat))[:1]:
            try:
                atoms = ase_read(f) if fmt is None else ase_read(f, format=fmt)
            except Exception:                                   # noqa: BLE001
                continue
            if len(atoms):
                return compound_name(atoms.get_chemical_symbols())
    return None


def describe_path(d) -> dict:
    """(code, compound, purpose) for a directory, from its place in the layout.

    A run already living at runs/<code>/<compound>/<purpose> states all three,
    and that beats any inference. Anywhere else, only the directory name is
    available and the compound has to come from the structure.
    """
    d = Path(d).resolve()
    parts = d.parts
    if "runs" in parts:
        i = len(parts) - 1 - parts[::-1].index("runs")
        tail = parts[i + 1:]
        if len(tail) >= 3 and tail[0] in CODES:
            return {"code": tail[0], "compound": tail[1], "purpose": "/".join(tail[2:])}
    return {"code": None, "compound": compound_from_structure(d), "purpose": d.name}


def autostamp(d, *, code: str, settings: dict | None = None, **extra) -> Path | None:
    """Stamp a run with what the CODE ITSELF knows, on every run.

    The driver knows which code ran, which structure it ran on and with what
    settings; it cannot know the purpose or who will need the output. Recording
    the first three the moment the run happens is what stops a directory from
    ever again needing archaeology, and `owner` stays "unassigned" until a human
    or `stamp` says otherwise.

    NEVER raises. A finished DFT run must not be lost to a manifest-writing
    error, so any failure here is silent and the run stands unstamped.
    """
    try:
        d = Path(d)
        known = describe_path(d)
        old = read(d) or {}
        return stamp(d, code=code,
                     compound=old.get("compound") or known["compound"] or "unknown",
                     purpose=old.get("purpose") or known["purpose"],
                     owner=old.get("owner", "unassigned"),
                     status=old.get("status", "live"),
                     note=old.get("note", ""),
                     settings={**old.get("settings", {}), **(settings or {})},
                     **extra)
    except Exception:                                           # noqa: BLE001
        return None


def run_dir(root, code: str, compound: str, purpose: str, variant: str = "") -> Path:
    """`root`/runs/<code>/<compound>/<purpose>[/<variant>], created."""
    if code not in CODES:
        raise ValueError(f"unknown code {code!r}; expected one of {CODES}")
    for part, name in ((compound, "compound"), (purpose, "purpose")):
        if not part or part != part.strip().lower().replace(" ", "_"):
            raise ValueError(f"{name} {part!r} must be lowercase, no spaces "
                             f"(it becomes a directory name)")
    d = Path(root) / "runs" / code / compound / purpose
    if variant:
        d = d / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp(d, *, code: str, compound: str, purpose: str, owner: str,
          settings: dict | None = None, status: str = "live",
          prunable=PRUNABLE, note: str = "", **extra) -> Path:
    """Write/refresh `RUN.json`. `owner` is the notebook or script that needs it.

    Called on every run, so the manifest tracks the directory rather than
    describing it once and going stale. `created` is preserved across rewrites;
    `updated` is not.
    """
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    if code not in CODES:
        raise ValueError(f"unknown code {code!r}; expected one of {CODES}")
    p = Path(d) / MANIFEST
    old = {}
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except ValueError:
            pass
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec = {"code": code, "compound": compound, "purpose": purpose,
           "owner": owner, "status": status, "note": note,
           "created": old.get("created", now), "updated": now,
           "host": socket.gethostname(), "prunable": list(prunable),
           "settings": settings or {}, **extra}
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return p


def read(d) -> dict | None:
    p = Path(d) / MANIFEST
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def survey(root, unmanifested_only: bool = False) -> list[dict]:
    """Every run directory under `root`, with size, age and manifest.

    A directory is a "run" if it holds a manifest or looks like output (an
    OUTCAR, a .out, a .fdf ...). Directories WITHOUT a manifest are what needs
    triage, which is what `unmanifested_only` selects.
    """
    root = Path(root)
    marks = ("OUTCAR", "RUN.json")
    seen = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any((d / m).exists() for m in marks) or list(d.glob("*.out")) \
                or list(d.glob("*.fdf")):
            if any(str(d).startswith(str(s["path"]) + os.sep) for s in seen):
                continue                      # a subdir of a run already listed
            m = read(d)
            if unmanifested_only and m is not None:
                continue
            seen.append({"path": d, "manifest": m, "bytes": _size(d),
                         "mtime": d.stat().st_mtime})
    return seen


def prunable_bytes(d) -> tuple[int, list[Path]]:
    """Size and list of files this run declares regenerable."""
    d = Path(d)
    pats = (read(d) or {}).get("prunable", PRUNABLE)
    files = [f for pat in pats for f in d.rglob(pat) if f.is_file()]
    return sum(f.stat().st_size for f in files), files
