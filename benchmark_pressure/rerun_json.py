#!/usr/bin/env python

import sys, os, json
from tqdm import tqdm

from umlip import umlip
from pymatgen.entries.computed_entries import ComputedStructureEntry

import json

if len(sys.argv) != 5 and len(sys.argv) != 6:
  print("Arguments please (dim, model, source_json, dest_json, pressure)")
  sys.exit()


dim = sys.argv[1].lower() # exp: "1D"
model = sys.argv[2].lower() # exp: "m3gnet-2d"
pressure = float(sys.argv[5]) if len(sys.argv) == 6 else None

# initialize mlip
mlip = umlip(model=model, cell_mask=dim)

thick = {"0d": 20.0, "1d": 12.5, "2d": 7.5, "3d": 1.0}[dim] # Not used in 3D
fix_symmetry = True if dim in ('2d', '3d') else False # in 1D fix symmetry fails due to large gradients

to_ds = []

## Loading the test data, usually, this is connected to our database, but one can find
## all our database on https://alexandria.icams.rub.de/
from_ds = []
with open(sys.argv[3],'r') as f:
    json_data = json.load(f)
    data = [ComputedStructureEntry.from_dict(i) for i in json_data["entries"]]
    from_ds.extend(data)

fmax_vs_iter = {}
for entry in tqdm(from_ds, desc="Relaxing structures"):
  try:
    entry.structure.remove_site_property("forces")
    entry.structure.remove_site_property("charge")
    entry.structure.remove_site_property("magmom")
  except:
    pass

  fmax = []
  try:
    res = mlip.relax_structure(entry.structure, \
      thick=thick, check_connected=False, fix_symmetry=fix_symmetry, pressure=pressure, fmax_vs_iter=fmax)
  except Exception as e:
    print(e)
    res = None

  fmax_vs_iter[entry.data["mat_id"]] = fmax

  if res is None: # not converged
    continue

  structure, energy, stress = res
  data = {}
  data["mat_id"] = entry.data["mat_id"]
  data["stress"] = 0.0 #stress 
  data["energy_total"] = 0.0 #energy

  try:
    cse = ComputedStructureEntry(structure, \
          energy=energy, \
          correction=0.0,
          data=data)

    to_ds.append(cse)
  except Exception as e:
    print(e)

with open(sys.argv[4], 'w') as fh:
    mdict = [a.as_dict() for a in to_ds]
    data = {"entries": mdict}
    json.dump(data, fh)

with open(os.path.splitext(sys.argv[4])[0] + "_trajectory.json", 'w', encoding='utf-8') as f:
    json.dump(fmax_vs_iter, f, ensure_ascii=False, indent=2)
