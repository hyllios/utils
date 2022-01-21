#!/usr/bin/env python

# see https://www.degruyter.com/document/doi/10.1515/pac-2016-1107/html

import sys, os
import numpy as np
from pymatgen.core.composition import Composition
from phase_diagram import PDEntry, PhaseDiagram, PDPlotter
from scipy.optimize import linprog

# These are the elements to plot
func_names = ["pbe", "ps", "scan"]
functional = func_names.index(sys.argv[1]) + 4

reactants = [Composition(i) for i in sys.argv[2:]]

# find all chemical elements involved in the reaction
elements = []
for compound in reactants:
  elements.extend([el.symbol for el in compound.elements])
elements = list(set(elements))

# read data from summary file
fh = open("ps_scan_summary.txt", "r")
pd_entries = []
for line in fh:
  fields = line.split()
  comp = Composition(fields[0])

  elm = set([i.symbol for i in comp.elements])
  if not all([i in elements for i in elm]):
    continue

  # transform to energy per unit cell
  e  = float(fields[functional])*comp.num_atoms
  pd_entries.append(PDEntry(comp, e, name=comp.reduced_formula))

pd = PhaseDiagram(pd_entries)
entries = list(pd.stable_entries)
#entries = list(pd_entries)

# now we build the matrices required for scipy
entries_mat  = np.zeros((len(elements)+1, len(entries)))
entries_e    = np.zeros(len(entries))
b_eq         = np.zeros(len(elements)+1)
nreactants   = 0

for i, entry in enumerate(entries):
  factor = 1.0
  if entry.composition in reactants:
    nreactants += 1 # we count the reactants found
    factor = -1.0
    r_index = i
    
  for el, amt in entry.composition.get_el_amt_dict().items():
    entries_mat[elements.index(el), i] = factor*amt
  entries_e[i] = factor*entry.energy

if nreactants != len(reactants):
   print("Problem: one of the reactants is not on convex hull")
   print("Elements: ", elements)
   print("Entries:")
   for entry in entries:
     print("  " + entry.composition.reduced_formula)
   sys.exit(1)
  
# fix one reactant to 1
entries_mat[-1, r_index] = 1
b_eq[-1] = 1.0

# and now we solve the linear programming problem
res = linprog(c=entries_e, A_eq=entries_mat, b_eq=b_eq)
min_x = min([x for x in res.x if x > 1e-3])

if res.message != 'Optimization terminated successfully.':
  print(res)

print("Entries:")
left_str  = []
right_str = []
for i, x in enumerate(res.x):
  if x < 1e-3:
    continue

  print("  " + entries[i].composition.reduced_formula + ": " +
        " %.3f"%entries[i].energy + " eV")
        
  mstr = "%g %s"%(x/min_x, entries[i].composition.reduced_formula)
  if entries[i].composition in reactants:
    left_str.append(mstr)
  else:
    right_str.append(mstr)

print("Reaction: " + " + ".join(left_str) + " -> " + " + ".join(right_str))
print("Reaction energy: %.3f"%(res.fun/min_x) + " eV")
