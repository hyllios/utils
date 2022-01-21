These are a couple of scripts that we used to create and analyze the data
calculated with PBEsol and SCAN:

* _ps\_scan\_summary.txt_: summary of the data
* _run\_ht\_scan.py_: the scripts used to perform the calculations. This uses _converge.py_ that contains a couple of utility functions
* _e\_above\_hull.py_: calculates the distance to the convex hull
* _plot\_phase\_diagram.py_ plots the binary or ternary phase diagrams. This uses a slightly hacked version of pymatgen's _phase\_diagram.py_ that outputs commands (to the stardard output) to make a .tikz plot.
* _compute\_reaction.py_: computes reaction energies

References:
Dataset: https://archive.materialscloud.org/record/2021.164
pymatgen: https://pymatgen.org/
ase: https://wiki.fysik.dtu.dk/ase/
cif2cell: https://github.com/kmu/cif2cell