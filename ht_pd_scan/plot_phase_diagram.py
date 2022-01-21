#!/usr/bin/env python

import sys, os
from pymatgen.core.composition import Composition
from phase_diagram import PDEntry, PhaseDiagram, PDPlotter

# These are the elements to plot
func_names = ["pbe", "ps", "scan"]
functional = func_names.index(sys.argv[1]) + 4
elements = sys.argv[2:]

fh = open("ps_scan_summary.txt", "r")
entries = []
for line in fh:
  fields = line.split()
  comp = Composition(fields[0])

  elm = set([i.symbol for i in comp.elements])
  if not all([i in elements for i in elm]):
    continue

  if fields[functional] == "nan":
    print("Entry ", fields[0], "does not have a", sys.argv[1], "energy!", file=sys.stderr)
    continue
  e  = float(fields[functional])*comp.num_atoms

  entries.append(PDEntry(comp, e, name=comp.reduced_formula))

pd = PhaseDiagram(entries)

plotkwargs = {
  "markerfacecolor": (0.2157, 0.4941, 0.7216),
  "markersize": 10,
  "linewidth": 2
}
show_unstable = 10 if len(elements) == 2 else 0.05
plotter = PDPlotter(pd, show_unstable=show_unstable, **plotkwargs)
plot = plotter.get_plot(label_unstable=False)

f = plot.gcf()
if len(elements) == 2:
  f.set_size_inches((12, 10))
  f.subplots_adjust(left=0.13, bottom=0.1)
else:
  f.set_size_inches((11, 10))
  f.subplots_adjust(left=0.1, right=0.99, bottom=0.01, top=0.99)

plot.savefig("pd.pdf")
#import tikzplotlib
#tikzplotlib.save("pd.tex")

