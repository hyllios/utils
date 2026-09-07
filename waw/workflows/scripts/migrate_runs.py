#!/usr/bin/env python
"""Move run directories into runs/<code>/<compound>/<purpose>, reversibly.

    python workflows/scripts/migrate_runs.py            # print the plan, move nothing
    python workflows/scripts/migrate_runs.py --apply
    python workflows/scripts/migrate_runs.py --undo

NOTHING BREAKS WHILE THIS HAPPENS. Every moved directory leaves a symlink behind
at its old path, so the 29 notebooks that hardcode
`runs/<name>` keep working untouched and can be updated one at a time, or never.
The move itself is a rename within one filesystem, so it is instantaneous
regardless of the 470 GB involved -- no copying, and `--undo` puts everything
back by reading the manifests.

The code is inferred from the files present, the compound and purpose from the
directory name, and both are recorded in RUN.json. Inference that is not
confident says so and the directory is left alone: a wrong guess here is a
directory nobody can find again, which is the problem being fixed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from waw.utils import runs as R

OLD_ROOTS = ("workflows/notebooks/runs", "workflows/w90tutorial/runs")

# How these compounds are written by the people who look for them. The
# structure file decides WHICH compound a run holds; this table only decides how
# it is spelled, because a formula reduced alphabetically is unfindable: MgB2
# lands under `b2mg`, BaTiO3 under `bao3ti`, NiI2 under `i2ni`. Anything absent
# here falls back to pymatgen's electronegativity ordering, which gets the
# conventional spelling right on its own for everything except a few
# intermetallics (it calls Ag2NdCd "NdCdAg2").
NAMES = R.NAMES

# Longest first, so nii2_vasp is not read as "ni". Only real compound spellings
# belong here: putting `graphene`/`diamond` in made them get stripped as if they
# were the compound, and since all four carbon tutorials then had an empty
# purpose they all aimed at runs/qe/c/main -- three would have been skipped.
COMPOUNDS = sorted({n.lower() for n in NAMES} | {"aluminum"}, key=len, reverse=True)


def infer_code(d: Path) -> str | None:
    has = lambda *p: any(next(d.glob(x), None) for x in p)          # noqa: E731
    if has("OUTCAR", "vasprun.xml", "*/OUTCAR"):
        return "vasp"
    if has("*.fdf", "*.HSX", "*/*.fdf"):
        return "siesta"
    if has("*.epw*", "epw.out", "*/epw.out"):
        return "epw"
    # A bare .fc is q2r.x output and nothing else writes one; two directories
    # held only that and so looked like they came from no code at all.
    if has("*.dyn*", "*.ph.out", "_ph0", "*.fc", "*/*.dyn*"):
        return "qe"
    if has("*.scf.out", "*.save", "*.nscf.out", "*.pw2wan.out", "*/*.scf.out"):
        return "qe"
    if has("*.wout", "*.chk", "wannier90*"):
        return "w90ref"
    if has("*.npz", "*.json") and not has("*.out"):
        return "model"
    return None


compound_from_structure = R.compound_from_structure


def infer(name: str, d: Path | None = None) -> tuple[str | None, str]:
    """(compound, purpose) for a run directory."""
    low = name.lower()
    struct = compound_from_structure(d) if d is not None else None
    if struct:
        # Strip the compound out of the name, so `diagvar_nio_np16` -> purpose
        # `diagvar_np16` rather than repeating it. Several spellings have to go:
        # the same run may be named for the alphabetical formula (ag2cdnd) or
        # the conventional one (ag2ndcd).
        #
        # WHOLE TOKENS ONLY. Substring matching quietly destroys the purpose it
        # is meant to shorten: `cu` inside gaas_shift_current gave `shift_rrent`,
        # `te` inside graphite gave `graphi`, and both look like plausible run
        # names, so nothing complains.
        aliases = {struct} | {c for c in COMPOUNDS if c in low}
        tokens = [t for t in re.split(r"[_-]+", low) if t and t not in aliases]
        return struct, "_".join(tokens) or "main"
    for c in COMPOUNDS:
        if low == c:
            return c, "main"
        if low.startswith(c + "_"):
            return c, low[len(c) + 1:] or "main"
    return None, low


def plan(root: Path) -> list[dict]:
    out = []
    for base in OLD_ROOTS:
        b = root / base
        if not b.exists():
            continue
        for d in sorted(p for p in b.iterdir() if p.is_dir() and not p.is_symlink()):
            code = infer_code(d)
            compound, purpose = infer(d.name, d)
            tutorial = "w90tutorial" in base
            reason = None
            if code is None:
                reason = "cannot tell which code produced it"
            elif compound is None:
                reason = "cannot tell the compound from the name"
            dest = None
            if reason is None:
                dest = R.run_dir.__wrapped__(root, code, compound, purpose) \
                    if hasattr(R.run_dir, "__wrapped__") else \
                    root / "runs" / code / compound / purpose
                if tutorial:
                    dest = root / "runs" / code / compound / f"w90tutorial_{purpose}"
            out.append(dict(src=d, dest=dest, code=code, compound=compound,
                            purpose=purpose, skip=reason))
    return _resolve_collisions(out)


def _resolve_collisions(items: list[dict]) -> list[dict]:
    """Two runs must never aim at one destination.

    `apply` refuses to overwrite, so a collision costs nothing but leaves the
    loser sitting at its old path while the summary says everything moved --
    exactly the silent partial result this whole exercise is meant to end. When
    it happens, fall back to the full source name, which is unique per root by
    construction, and only give up if even that repeats.
    """
    from collections import Counter

    dests = Counter(str(i["dest"]) for i in items if i["dest"] is not None)
    for i in items:
        if i["dest"] is None or dests[str(i["dest"])] == 1:
            continue
        i["purpose"] = re.sub(r"[^a-z0-9._]+", "_", i["src"].name.lower()).strip("_")
        i["dest"] = i["dest"].parent / (
            f"w90tutorial_{i['purpose']}" if i["dest"].name.startswith("w90tutorial_")
            else i["purpose"])
    again = Counter(str(i["dest"]) for i in items if i["dest"] is not None)
    for i in items:
        if i["dest"] is not None and again[str(i["dest"])] > 1:
            i["skip"] = f"collides with another run at {i['dest'].name}"
    return items


def apply(items, root: Path, dry: bool) -> None:
    moved = 0
    for it in items:
        if it["skip"]:
            continue
        src, dest = it["src"], it["dest"]
        if dest.exists():
            print(f"  SKIP {src.name}: destination exists ({dest.relative_to(root)})")
            continue
        if dry:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)                       # same filesystem: instant
        os.symlink(os.path.relpath(dest, src.parent), src)
        R.stamp(dest, code=it["code"], compound=it["compound"],
                purpose=it["purpose"], owner="unassigned",
                note=f"migrated from {src.relative_to(root)}; a symlink remains "
                     f"at the old path so existing notebooks keep working",
                migrated_from=str(src.relative_to(root)))
        moved += 1
    if not dry:
        print(f"\nmoved {moved} directories, each with a symlink left behind")


def migrated(root: Path):
    """Every migrated run directory: runs/<code>/<compound>/<purpose>, no deeper.

    NOT `rglob("*")`. These directories hold 470 GB of wavefunctions across
    millions of files on a network filesystem, and enumerating all of them to
    find 172 manifests three levels down does not finish in any useful time.
    """
    base = root / "runs"
    return sorted(d for d in base.glob("*/*/*") if d.is_dir()) if base.exists() else []


def undo(root: Path) -> None:
    n = 0
    for d in migrated(root):
        m = R.read(d)
        if not m or "migrated_from" not in m:
            continue
        old = root / m["migrated_from"]
        if old.is_symlink():
            old.unlink()
        old.parent.mkdir(parents=True, exist_ok=True)
        os.rename(d, old)
        (old / R.MANIFEST).unlink(missing_ok=True)
        n += 1
    print(f"restored {n} directories to their original paths")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    if a.undo:
        undo(a.root)
        return 0

    items = plan(a.root)
    ok = [i for i in items if not i["skip"]]
    bad = [i for i in items if i["skip"]]
    by_code: dict[str, int] = {}
    for i in ok:
        by_code[i["code"]] = by_code.get(i["code"], 0) + 1
    print(f"{len(items)} directories: {len(ok)} placeable, {len(bad)} need a decision")
    print("  by code: " + ", ".join(f"{k} {v}" for k, v in sorted(by_code.items())))
    print("\nexamples of the mapping:")
    for i in ok[:12]:
        print(f"  {i['src'].name:34s} -> {i['dest'].relative_to(a.root)}")
    if bad:
        print(f"\nleft alone ({len(bad)}) -- name them or move them by hand:")
        for i in bad[:15]:
            print(f"  {i['src'].name:34s} : {i['skip']}")
        if len(bad) > 15:
            print(f"  ... and {len(bad)-15} more")
    apply(items, a.root, dry=not a.apply)
    if not a.apply:
        print("\nthis was a dry run; add --apply to move (and --undo to reverse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
