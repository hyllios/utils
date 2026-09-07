#!/usr/bin/env python
"""Inventory the run directories and say which are claimed by something.

    python workflows/scripts/triage_runs.py            # summary + candidates
    python workflows/scripts/triage_runs.py --csv out.csv

"Claimed" means the directory name appears in a notebook, script or test. That
is a WEAK test and deliberately reported as such: a notebook that builds its
path with an f-string, or that was renamed since, will look unclaimed. On this
repo the largest apparently-unclaimed directory was 51.6 GB of live campaign
whose notebook simply writes to a differently-named folder. Treat the output as
a list to review, never as a delete list -- and fix the ambiguity for good by
having runs carry `RUN.json` (see `waw.utils.runs`).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

from waw.utils import runs as R

SEARCH_GLOBS = ("workflows/**/*.ipynb", "workflows/**/*.py", "tests/*.py",
                "waw/**/*.py")


def corpus(root: Path) -> str:
    out = []
    for g in SEARCH_GLOBS:
        for p in root.glob(g):
            if ".ipynb_checkpoints" in str(p):
                continue
            out.append(p.read_text(errors="replace"))
    return "\n".join(out)


def _touched(r: Path) -> float:
    """When this run was last WORKED ON, from its contents, not its inode.

    Writing a manifest into a directory updates that directory's mtime, so
    stamping 172 runs at once reset every age to zero and made "how old is this"
    -- the signal triage leans on hardest -- unusable. The outputs inside keep
    their own timestamps.
    """
    ts = [f.stat().st_mtime for f in r.glob("*")
          if f.is_file() and f.name != R.MANIFEST]
    return max(ts) if ts else r.stat().st_mtime


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()

    text = corpus(a.root)
    rows, gb = [], 1 << 30
    for base in ("workflows/notebooks/runs", "workflows/w90tutorial/runs",
                 "runs/*/*"):
        for d in sorted(a.root.glob(base)) if "*" in base else [a.root / base]:
            if not d.exists():
                continue
            # Skip symlinks or every migrated run is counted TWICE -- once at its
            # real path and once through the compatibility link left behind, which
            # is how 470 GB first reported itself as 940.
            for r in sorted(p for p in d.iterdir()
                            if p.is_dir() and not p.is_symlink()):
                man = R.read(r)
                # Notebooks still refer to these by the name they had before the
                # migration, so a run is claimed under EITHER name -- checking
                # only the new one dropped the count from 79 to 33 without a
                # single reference having changed.
                aka = {r.name, Path((man or {}).get("migrated_from", "")).name} - {""}
                claimed = any(re.search(rf"[\"'/]{re.escape(n)}[\"'/\s]", text)
                              for n in aka)
                pb, _ = R.prunable_bytes(r)
                rows.append(dict(
                    path=str(r.relative_to(a.root)), name=r.name,
                    gb=round(R._size(r) / gb, 2), prunable_gb=round(pb / gb, 2),
                    claimed=claimed, manifest=bool(man),
                    owner=(man or {}).get("owner", ""),
                    status=(man or {}).get("status", ""),
                    age_days=round((time.time() - _touched(r)) / 86400, 1)))

    tot = sum(r["gb"] for r in rows)
    unclaimed = [r for r in rows if not r["claimed"] and not r["manifest"]]
    prune = sum(r["prunable_gb"] for r in rows)
    print(f"{len(rows)} run directories, {tot:.0f} GB total")
    print(f"  with a manifest      : {sum(r['manifest'] for r in rows):4d}")
    print(f"  named in the codebase: {sum(r['claimed'] for r in rows):4d}")
    print(f"  NEITHER (review these): {len(unclaimed):3d}  "
          f"({sum(r['gb'] for r in unclaimed):.0f} GB)")
    print(f"  declared regenerable : {prune:.0f} GB  (needs manifests to be useful)")

    smell = re.compile(r"_(stale|old|final|interrupted|tmp|test|bak|copy|v\d)$|"
                       r"_(gcheck|check)\d*$", re.I)
    flagged = [r for r in rows if smell.search(r["name"])]
    if flagged:
        print(f"\nnames that advertise themselves as disposable ({len(flagged)}):")
        for r in sorted(flagged, key=lambda x: -x["gb"])[:10]:
            print(f"  {r['gb']:8.1f} GB  {r['age_days']:6.0f} d  {r['name']}")

    print(f"\nlargest unclaimed, unmanifested:")
    for r in sorted(unclaimed, key=lambda x: -x["gb"])[:15]:
        print(f"  {r['gb']:8.1f} GB  {r['age_days']:6.0f} d  {r['name']}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda x: -x["gb"]))
        print(f"\nfull table -> {a.csv}  (add a 'keep/archive/delete' column and "
              f"hand it back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
