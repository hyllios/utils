#!/usr/bin/env python

import sys, os
from pymatgen.core.composition import Composition
from phase_diagram import PDEntry, PhaseDiagram, PDPlotter

# These are the elements to plot
func_names = ["pbe", "ps", "scan"]
functional = func_names.index(sys.argv[1]) + 4

material = sys.argv[2]
composition = Composition(material)
elements = [i.symbol for i in composition.elements]

fh = open("ps_scan_summary.txt", "r")
entries = [PDEntry(composition, float(sys.argv[3]), name=material)]
for line in fh:
  fields = line.split()
  comp = Composition(fields[0])

  elm = set([i.symbol for i in comp.elements])
  if not all([i in elements for i in elm]):
    continue

  e  = float(fields[functional])*comp.num_atoms

  new_entry = PDEntry(comp, e, name=fields[0])
  entries.append(new_entry)

pd = PhaseDiagram(entries)

decomp, e_above_hull = pd.get_decomp_and_e_above_hull(entries[0])
print("e_above_hull  = ", e_above_hull)
print("decomposition = ", decomp)
