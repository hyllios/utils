#!/usr/bin/env python

import sys, os
import multiprocessing
import numpy as np
import pandas as pd
import bz2, json
from pymatgen.core import Structure

atom_size = 3.0
dim = "3d"

def parse_entry_3d(entry):
  structure = Structure.from_dict(entry["structure"])
  return {
          "mat_id": entry["data"]["mat_id"], 
          "volume_per_atom": structure.volume/len(structure),
          "energy_per_atom": entry["energy"]/len(structure)
         }


def parse_json(json_file):
  with open(json_file) as fh:
    json_data = json.loads(fh.read())

  global dim
  if dim.lower() == '3d':
    return [parse_entry_3d(entry) for entry in json_data["entries"]]
  else:
    print(f"Error dim {dim} does not exist")
    sys.exit(1)


def get_params():
    from argparse import ArgumentParser

    parent_parser = ArgumentParser(usage="Extract properties to benchmark")
    parent_parser.add_argument('--json', type = str, default = "",
        help='Directory with the jsons')
    parent_parser.add_argument('--csv', type = str, default = "benchmark.csv",
        help='Destination csv file')
    parent_parser.add_argument('--dim', type = str, default = "2d",
        help='Dimensionality of the structure')

    return parent_parser.parse_args()

if __name__ == '__main__':
    params = get_params()
    dim = params.dim

    jsons = [f"{params.json}/{i}" for i in os.listdir(params.json) if "json.bz2" in i and "traj" not in i]
    with multiprocessing.Pool(4) as pool:
      zipped_out = pool.map(parse_json, jsons)

    mlip = pd.DataFrame([x for xs in zipped_out for x in xs])

    mlip.to_csv(f"{params.csv}", index=False)

