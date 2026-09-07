# Run directories

`runs/<code>/<compound>/<purpose>[/<variant>]` — 470 GB of gitignored working
state, of which **315 GB is declared regenerable** (wavefunctions: `WAVECAR`,
`CHGCAR`, `*.wfc*`, …). Every directory carries a `RUN.json`, and those
manifests **are tracked** — they are the only record of what the data is, and
they survive a machine that does not.

```
runs/qe/al/elph_alpha2f/RUN.json    <- tracked
runs/qe/al/elph_alpha2f/al.save/    <- ignored
```

## Reading and writing them

```python
from waw.utils import runs as R

d = R.run_dir(root, "qe", "mgb2", "elph")        # creates it, validates the name
R.stamp(d, code="qe", compound="mgb2", purpose="elph", owner="notebook 16")
R.read(d)["owner"]
R.survey(root, unmanifested_only=True)           # what still needs triage
R.prunable_bytes(d)                              # what is cheap to recompute
```

`run_pw`, `run_vasp` and `run_siesta` call `R.autostamp` themselves, so a new
run records its code, compound and settings without being asked. `owner` and
`purpose` are the two things a driver cannot know; set them with `stamp`.

## Conventions that are not obvious

**The compound comes from the structure file, not the directory name.** Names
like `ctrl_NiO`, `diagvar_nio_np16` and `aluminum_superfluid` carry it in a form
no prefix table matches, and the POSCAR beside them states it exactly.

**The spelling is the conventional one, from `R.NAMES`.** An alphabetically
reduced formula is a correct composition and a useless name: MgB2 becomes
`b2mg`, BaTiO3 `bao3ti`, NiI2 `i2ni`. Add to that table rather than accepting a
name you would not think to look under.

**Sweeps nest under one purpose, never beside it.** Seven top-level
`al_scdm_auto_r*_s*_n*` directories is what the alternative looked like.

**`runs/` is at the repository root**, not under one workflow tree: the same
compound is used from the notebooks and from the tutorials, and an SCF worth
tens of GB should be shared rather than duplicated.

## The symlinks in the old locations

`workflows/notebooks/runs/` and `workflows/w90tutorial/runs/` now hold only
symlinks into here, one per migrated directory, so notebooks that hardcode
`runs/<old_name>` keep working. They can be updated one at a time, or never. The
manifest records the previous path in `migrated_from`.

`workflows/scripts/`:

| script | what it does |
|---|---|
| `triage_runs.py` | inventory: size, age, what is regenerable, what nothing references |
| `migrate_runs.py` | the move that created this layout; `--undo` reverses it |
| `make_index.py` | regenerates `workflows/notebooks/INDEX.md` |

`triage_runs.py`'s "claimed" column is a **weak** test — it greps for the
directory name — and is a list to review, never a delete list. The largest
apparently-unclaimed directory here is 51.6 GB of live campaign whose notebook
simply writes to a differently-named folder.
