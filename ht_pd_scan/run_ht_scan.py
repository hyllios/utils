#!/usr/bin/env python

import sys, os, re, bz2, shutil
import numpy as np
from converge import *
from ase import *
from ase.calculators.vasp import Vasp, float_keys, exp_keys, string_keys, int_keys
from pymatgen.io.vasp.outputs import Vasprun

def check_converged():
  if os.path.isfile('vasprun.xml'):
    fname = 'vasprun.xml'
  else:
    fname = 'vasprun.xml.bz2'

  try:
    x = Vasprun(fname)
    if x.converged:
      return True
    else:
      return False
  except:
    return False

  
def clean_vasp():
  """Delete unecessary vasp files"""
  import os
  files = ['CHG', 'CHGCAR', 'CONTCAR', 'IBZKPT', 'OSZICAR',
           'PCDAT', 'WAVECAR', 'XDATCAR', 'POTCAR', 'PROCAR', 'ase-sort.dat',
           'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2', 'REPORT']
  for f in files:
    try:
      os.remove(f)
    except OSError:
      pass
    
          
def compress_vasp(prefix="", only_vasprun=False, remove=False):
  if only_vasprun:
    files_to_bz2 = ("vasprun.xml",)
  else:
    files_to_bz2 = ("EIGENVAL", "DOSCAR", "OUTCAR", "vasprun.xml")

  for i in files_to_bz2:
    with open(i, 'rb') as data:
      tarbz2contents = bz2.compress(data.read(), 9) 
    fh = open(prefix + i + ".bz2", "wb")
    fh.write(tarbz2contents)
    fh.close()
    if remove:
      os.remove(i)

      
def pprint_info(vasprun_file, f):
  from pymatgen.analysis.elasticity.stress import Stress
  from pymatgen.core.units import Unit

  kB_to_eV_A3 = 1e-1*Unit("GPa").get_conversion_factor(Unit("eV ang^-3"))

  vasprun = Vasprun(vasprun_file)
  step = vasprun.ionic_steps[-1]

  print >>f, "\n==== End geometry:"
  print >>f, step["structure"].lattice.matrix
  print >>f, ""
  print >>f, step["structure"].frac_coords
  print >>f, "\nForces:"
  print >>f, np.array(step["forces"])
  print >>f, "\nStress:"

  # convert stress to Voigt notation and to eV/Ang^3
  stress = Stress(step["stress"]).voigt * kB_to_eV_A3
  print >>f, stress


def check_forces(vasprun_file, threshold, f):
  from pymatgen.analysis.elasticity.stress import Stress
  from pymatgen.core.units import Unit

  kB_to_eV_A3 = 1e-1*Unit("GPa").get_conversion_factor(Unit("eV ang^-3"))

  vasprun = Vasprun(vasprun_file)
  step = vasprun.ionic_steps[-1]

  # check forces
  for fi in step["forces"]:
    fabs = np.sqrt(fi[0]**2 + fi[1]**2 + fi[2]**2)
    if fabs > threshold:
      print >>f, "Error: Forces not converged"
      sys.exit(1)

  # check stresses
  stress = Stress(step["stress"]).voigt * kB_to_eV_A3
  for fi in stress:
    if abs(fi) > threshold:
      print >>f, "Error: Stress not converged"
      sys.exit(1)


def preliminary_geo(primitive_cell, vasp_params, f):
  # save a couple of default values
  ispin = vasp_params["ispin"]
  algo = vasp_params["algo"]

  print >>f, "Preliminary geometry optimization:\n"
  # run preliminary geometry optimization

  # most prototypes are too small: let us increase the cell a bit
  cell = 1.15*primitive_cell.get_cell()
  primitive_cell.set_cell(cell, scale_atoms=True)
 
  for iter in range(8):
    if (iter < 3):
      pre_kmax   = 1000 #300
      pre_prec   = "Normal"
      pre_ediffg = -5.0e-2
      pre_ediff  = 1.e-4
      pre_algo   = "normal"
    elif (iter < 5):
      pre_kmax   = 1000 #500
      pre_prec   = "Normal"
      pre_ediffg = -1.0e-2
      pre_ediff  = 1.e-6
      pre_algo   = "fast"
    else:
      pre_kmax   = 1000
      pre_prec   = "Accurate"
      pre_ediffg = -5.0e-3
      pre_ediff  = 1.e-7
      pre_algo   = "fast"

    # some elements need special treatment
    if (algo != "Fast"):
      pre_algo = algo

    vasp_params["algo"]  = pre_algo
    vasp_params["ispin"] = 1
    kpt_1, kpt_2, kpt_3 = kpoints_max(primitive_cell, return_kgrid=True, kppa=pre_kmax)
        
    try:
      calc = Vasp(kpts=(kpt_1, kpt_2, kpt_3), prec=pre_prec,
                  ibrion=2, isif=3, ediff=pre_ediff, ediffg=pre_ediffg, nsw=50, **vasp_params)
      calc.calculate(primitive_cell)

      pre_energy = calc.read_energy()[0] / primitive_cell.get_number_of_at

      pre_iter   = 0
      for line in reversed(open("OUTCAR").readlines()):
        m = re.search("Iteration(.+?)\(", line)
        if m:
          pre_iter = int(m.group(1))
          break
    except:
      pre_energy = 0.0
      pre_iter   = 100
      # we expand the structure by 15%
      cell = 1.15*primitive_cell.get_cell()
      primitive_cell.set_cell(cell, scale_atoms=True)
      if iter < 4:
        pass

    clean_vasp()

    print >>f, iter, "\t", pre_energy, "\t", pre_iter

    if (iter >= 5) and (pre_iter < 20):
      print >>f, "\n"
      break

  # restore defaults
  vasp_params["algo"]  = algo
  vasp_params["ispin"] = ispin

######################
# the workflow starts here
vasp_params = {
  "xc": "PBE",
  "gga": "PS", # this is PBE_SOL
  "isym": 2, # run with (2) or without (0) symmetries
  "symprec": 1e-4,
  "gamma": True,
  "kpar": 4,
  "algo": "A",
  "ispin": 2,
}

# kppa
kppa   = 2000
ibrion = 2

# allow overriding of parameters
if os.path.isfile("vasp_params"):
  with open("vasp_params") as f:
    for line in f:
      if not ":" in line:
        continue
      fields = line.split(":")
      if fields[0] == "kppa":
        kppa = int(fields[1])
        continue
      if fields[0] == "ibrion":
        ibrion = int(fields[1])
        continue
      if fields[0] in float_keys or fields[0] in exp_keys:
        fields[1] = float(fields[1])
      elif fields[0] in int_keys:
        fields[1] = int(fields[1])
      else:
        fields[1] = fields[1].rstrip()
      vasp_params[fields[0]] = fields[1]

# read structure from file
basename, extension = os.path.splitext(sys.argv[1])
if os.path.isfile("POSCAR"):
  primitive_cell = read_file("POSCAR")
  str_type = "a" # do not perform preliminary geometry optimization
else:
  primitive_cell = read_file(sys.argv[1])
  str_type = sys.argv[2]

# increase number of bands for certain special chemical elements
sym = primitive_cell.get_chemical_symbols()
special_sym = ["Gd", "Ce", "Eu", "Yb", "Pu", "U", "Np"]
if any(i in sym for i in special_sym) and not ("nbands" in  vasp_params):
  # find number of valence electrons
  calc = Vasp(**vasp_params)
  calc.initialize(primitive_cell)
  calc.write_potcar()
  nel_potcar = dict(calc.get_default_number_of_electrons())
  for k in nel_potcar:
    if k in special_sym:
      nel_potcar[k] *= 2
  nelect = sum([nel_potcar[sym] for sym in primitive_cell.get_chemical_symbols()])

  vasp_params["nbands"] = 0.6*nelect

# calculations start here
f = open('convergence', 'w', 0)

# check if geometry optimization is done
if not os.path.isfile(basename + '_opt.cif'):
  if (str_type == 'p'):
    clean_vasp() # just in case garbage is lying around
    preliminary_geo(primitive_cell, vasp_params, f)

  # find converged k-point sampling
  if vasp_params["ispin"] == 2:
    primitive_cell.set_initial_magnetic_moments([1] * primitive_cell.get_number_of_atoms())
  kpt_1, kpt_2, kpt_3 = kpoints_max(primitive_cell, kppa=kppa, return_kgrid=True)

  print >>f, "Standard geometry optimization:\n"
  # run standard geometry optimization

  # now we run the geometry optimization with vasp
  print >>f, "\n==== Starting geometry:"
  print >>f, primitive_cell.get_cell(), "\n\n", primitive_cell.get_scaled_positions()

  calc = Vasp(kpts=(kpt_1, kpt_2, kpt_3), prec='High', ibrion=ibrion, isif=3, ediff=1E-07, ediffg=-0.005, nsw=250, **vasp_params)
  calc.calculate(primitive_cell)
  compress_vasp(prefix="GEO1_", only_vasprun=True)
  clean_vasp()

  # now we redo it
  kpt_1, kpt_2, kpt_3 = kpoints_max(primitive_cell, kppa=kppa, return_kgrid=True)

  calc = Vasp(kpts=(kpt_1, kpt_2, kpt_3), prec='High', ibrion=ibrion, isif=3, ediff=1E-07, ediffg=-0.005, nsw=250, **vasp_params)
  calc.calculate(primitive_cell)
  compress_vasp(prefix="GEO2_", only_vasprun=True)

  calc.calculate(primitive_cell) # we run this twice
  compress_vasp(prefix="GEO3_", only_vasprun=True)

  if not check_converged():
    print >>f, "Did not succeed in converging geometry"
    sys.exit(1)
    
  # write optimized cell
  primitive_cell.write(basename + '_opt.cif', format='cif', long_format = True)
  shutil.copyfile("KPOINTS", "GEO_KPOINTS")
  if kppa != 8000:
    os.remove("WAVECAR") # can not restart as number of k-points will change

# set magmons and increase number of k-points
if vasp_params["ispin"] == 2:
  primitive_cell.set_initial_magnetic_moments([1] * primitive_cell.get_number_of_atoms())
kpt_1, kpt_2, kpt_3 = kpoints_max(primitive_cell, kppa=8000, return_kgrid=True)

# set more accurate vasp parameters for the final calculations
vasp_params["encut"] = 520
vasp_params["enaug"] = 659.424
vasp_params["nelm"] = 200

print >>f, "\n\n### PBE_SOL calculation:"

# check if PBE_SOL calculation is done
if not os.path.isfile("PS_vasprun.xml.bz2"):
  calc = Vasp(kpts=(kpt_1, kpt_2, kpt_3), prec='High', lorbit=11,
              ediff=5E-07, emin=-15, emax=15, nsw=0, ibrion=-1, **vasp_params)
  calc.calculate(primitive_cell)

  if not check_converged():
    print >>f, "Did not succeed in converging PBE_SOL"
    sys.exit(1)

  # we now copy keep the relevant PBE_SOL files
  compress_vasp(prefix="PS_")

# print PBE_SOL info
pprint_info("PS_vasprun.xml.bz2", f)
check_forces("PS_vasprun.xml.bz2", 5e-2, f)

print >>f, "\n\n### SCAN calculation:"

# we now rerun vasp to get the ground-state with scan
del vasp_params['gga']
vasp_params["metagga"] = "SCAN"
vasp_params["lasph"] = True
vasp_params["lmaxtau"] = 6

# check if PBE_SOL calculation is done
if not os.path.isfile("SCAN_vasprun.xml.bz2"):
  calc = Vasp(kpts=(kpt_1, kpt_2, kpt_3), prec='High', lorbit=11,
              ediff=5E-07, emin=-15, emax=15, nsw=0, ibrion=-1, **vasp_params)
  calc.calculate(primitive_cell)

  if not check_converged():
    print >>f, "Did not succeed in converging SCAN"
    sys.exit(1)

  # we now copy keep the relevant SCAN files
  compress_vasp(prefix="SCAN_", remove=True)

# print SCAN info
pprint_info("SCAN_vasprun.xml.bz2", f)

f.close()
clean_vasp()
