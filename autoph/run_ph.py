import os, re, json, math
from matplotlib import gridspec
from pw_utils import MyPWInput as PWInput
from pw_utils import check_imfreq, read_dyn, PostProcessInput, ibrav_to_cell
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.units import bohr_to_angstrom, Ry_to_eV
from ase.io import read
import pandas as pd
import numpy as np

from onnes_bot import send_discord_message_with_images

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot

CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))

config = {
  "PSEUDO_DIR": "/espresso/runs/nc-sr-04_pbesol_stringent_upf/",
  "QE": {'pw': "mpirun -n 32 pw.x ",
      'dos': "dos.x ", 
      'pdos': "projwfc.x ", 
      'sumpdos' : "sumpdos.x ",
      'ph': "mpirun -n 32 ph.x ",
      'q2r': "q2r.x ",
      'matdyn': "matdyn.x ",
      'bands': "bands.x ",
      'pp': "pp.x "}
      }

inp_params = {
  'control': {
    'prefix': 'qe',
    'outdir': 'outdir',
    'tprnfor': True,
    'tstress': True,
    'etot_conv_thr': 1.0E-8,
    'forc_conv_thr': 1.0E-6,
    'restart_mode': "from_scratch"
  },
  'system': {
    'smearing': 'mp',
    'occupations': 'smearing',
    'degauss': 0.2/Ry_to_eV,
    'tot_charge': 0.0
  },
  'electrons': {
    'diagonalization': 'david',
    'mixing_mode': 'plain',
    'mixing_beta': 0.7,
    'conv_thr':  1.0E-12,
  },
  'ions': {
    'ion_dynamics': 'bfgs'
  },
  'cell': {
    'press_conv_thr': 0.05,
    'cell_dynamics': 'bfgs',
    'press': 0.0
  }
}

pdos_params = {
  'projwfc': {
    'outdir': 'outdir',
    'prefix': 'qe',
    'degauss': 0.008,
    'Emin': 0, 
    'Emax': 0, 
    'DeltaE': 0.01
  }
}

dos_params = {
  'dos': {
    'outdir': 'outdir',
    'prefix': 'qe',
    'degauss': 0.008,
    'fildos': 'qe.el-dos',
    'Emin': 0,
    'Emax': 0,
    'DeltaE': 0.01
  }
}

bands_params = {
  'bands': {
    'prefix': "qe",
    'outdir': 'outdir',
    'filband': "el-bands.dat"
  }
}

elf_params = {
  'inputPP': {
    'prefix': 'qe',
    'outdir': 'outdir',
    'plot_num': 8,
    'filplot': 'qe.elf.dat'
  },
  'plot': {
    'iflag': 3,
    'output_format': 6,
    'fileout': 'elf.cube'
  }
}

ph_params = {
  'inputph': {
    'reduce_io' : True,
    'search_sym' : False,
    'recover': False,
    'prefix': 'qe',
    'fildyn': 'qe.dyn',
    'outdir': 'outdir',
    'ldisp': True,
    'trans': True,
    'fildvscf': 'dvscf',
    'diagonalization': 'david',
    'electron_phonon': 'interpolated',
    'el_ph_sigma': 0.005,
    'el_ph_nsigma': 10,
    'nq1': 0,
    'nq2': 0,
    'nq3': 0,
    'tr2_ph':  1.0E-15,
    'start_q': 0,
    'last_q': 0,
    'niter_ph': 100,
    'alpha_mix(1)': 0.2
  }
}

q2r_params = {
  'input': {
    'la2F': True,
    'fildyn': 'qe.dyn',
    'zasr': 'simple', 
    'flfrc': 'qe.fc'
  }
}

matdyn_params = {
  'input': {
    'la2F' : True,
    'asr':'simple',  
    'flfrc':'qe.fc', 
    'flfrq':'qe.freq',
    'q_in_cryst_coord': True,
    'dos': True,
    'nk1': 1,
    'nk2': 1,
    'nk3': 1
  }
}

matdyn_bs_params = {
  'input': {                  
    'asr':'simple',  
    'flfrc':'qe.fc', 
    'flfrq':'qe.freq',
    'q_in_cryst_coord': True
  }
}

def check_jobdone(filename):
  if not os.popen(f"grep -a 'JOB DONE' {filename}").read().strip():
    sys.exit(f"Error: {filename} did not finish successfully!!!")

def to_latex(string):
  if "\\" in string:
    string = f"$\\{'_'.join(string[1:].split('_'))}$"
  if "_" in string:
    l, n = string.split('_')
    string = f"{l}_{n}" if "\\" in string else f"{l}$_{n}$"
  return string

def standardize_cell(structure, symprec=0.1):
  import spglib
  # Atomic positions have to be specified by scaled positions for spglib.
  lattice = structure.lattice.matrix
  scaled_positions = structure.frac_coords
  numbers = [i.specie.Z for i in structure.sites]
  cell = (lattice, scaled_positions, numbers)
  lattice, scaled_positions, numbers = \
    spglib.standardize_cell(cell, to_primitive=True, symprec=symprec)
  s = Structure(lattice, numbers, scaled_positions)
  return s.get_sorted_structure()

def get_cutoff(composition):
  import requests

  jsonfile = os.path.join(CONFIG_PATH, f"{PP_VERSION}")
  if not os.path.isfile(jsonfile):
    dojo = requests.get(f"https://raw.githubusercontent.com/abinit/pseudo_dojo/master/website/{PP_VERSION}").json()
    json.dump(dojo, open(jsonfile, "w"))
  else:
    with open(jsonfile) as fh:
      dojo = json.load(fh)

  cutoffs = [dojo[i.symbol]['hh'] for i in composition]

  # convert to Rydbergs
  return max(cutoffs)*2

def get_espresso_structure(structure, dim=3):
  """ Get structure in a format suitable for QE
  SPG info from: https://msestudent.com/list-of-space-groups/

  This routine works also for 2D, except for spg_2d = 3, 4, 5, 6, 7, 8
  """
  from pymatgen.core.lattice import Lattice
  from pymatgen.core.structure import PeriodicSite, Structure
  
  # first we determine the conventional structure
  sym = SpacegroupAnalyzer(structure, symprec=1e-5)
  std_struct  = sym.get_conventional_standard_structure(international_monoclinic=True)

  spg = sym.get_space_group_number()

  # spglib gives us a lattice where the a axis is along y and the b axis is 
  # along x. We switch x and y to make espresso happy. As the resulting unit
  # cell has negative volume, we also change z->-z.
  if spg >= 3 and spg <= 15:
    new_lat = std_struct.lattice.matrix.copy()
    for i in (0, 1, 2):
      new_lat[i, 0], new_lat[i, 1] = new_lat[i, 1], new_lat[i, 0]
      new_lat[i, 2] = -new_lat[i, 2]
    new_coords = std_struct.cart_coords.copy()
    for i in new_coords:
      i[0], i[1] = i[1], i[0]
      i[2] = -i[2]
    std_struct = Structure(new_lat, std_struct.species, new_coords, coords_are_cartesian=True)

  lat = std_struct.lattice

  # This is the structure in espresso input format
  espresso_in = {}
  espresso_in["celldm(1)"] = lat.a/bohr_to_angstrom

  if spg in [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218, 221, 222, 223, 224]:
    # Simple cubic
    espresso_in["ibrav"] = 1
    
  elif spg in [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]:
    # Face-centered cubic
    espresso_in["ibrav"] = 2

  elif spg in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:
    espresso_in["ibrav"] = 3
    
  elif ((spg >= 168) and (spg <= 194)) or spg in [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 162, 163, 164, 165]:
    # Hexagonal
    espresso_in["ibrav"] = 4
    espresso_in["celldm(3)"] = lat.c/lat.a
    
  elif spg in [146, 148, 155, 160, 161, 166, 167]:
    # Rhombohedral
    espresso_in["ibrav"] = 5
    aR       = math.sqrt(lat.a**2/3 + lat.c**2/9)
    cosgamma = (2*lat.c**2 - 3*lat.a**2)/(2*lat.c**2 + 6*lat.a**2)
    espresso_in["celldm(1)"] = aR/bohr_to_angstrom
    espresso_in["celldm(4)"] = cosgamma

  elif spg in [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]:
    # Simple Tetragonal
    espresso_in["ibrav"] = 6
    espresso_in["celldm(3)"] = lat.c/lat.a
    
  elif spg in [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142]:
    # Body-Centered Tetragonal
    espresso_in["ibrav"] = 7
    espresso_in["celldm(3)"] = lat.c/lat.a

  elif spg in [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]:
    #  Simple Orthorhombic
    espresso_in["ibrav"] = 8
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a

  elif spg in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
    # Base-Centered Orthorhombic
    espresso_in["ibrav"] = 9
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a
    
  elif spg in [22, 42, 43, 69, 70]:
    # Face-Centered Orthorhombic
    espresso_in["ibrav"] = 10
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a
    
  elif spg in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
    # Body-Centered Orthorhombic
    espresso_in["ibrav"] = 11
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a

  elif spg in [5, 8, 9, 12, 15]:
    # Base-Centered Monoclinic
    espresso_in["ibrav"] = -13
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a
    espresso_in["celldm(5)"] = np.cos(np.deg2rad(lat.beta))

  elif spg in [3, 4, 6, 7, 10, 11, 13, 14]:
    # Simple Monoclinic
    espresso_in["ibrav"] = -12
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a
    espresso_in["celldm(5)"] = np.cos(np.deg2rad(lat.beta))

  elif spg in [1, 2]:
    # Triclinic
    espresso_in["ibrav"] = 14
    espresso_in["celldm(2)"] = lat.b/lat.a
    espresso_in["celldm(3)"] = lat.c/lat.a
    espresso_in["celldm(4)"] = np.cos(np.deg2rad(lat.alpha))
    espresso_in["celldm(5)"] = np.cos(np.deg2rad(lat.beta))
    espresso_in["celldm(6)"] = np.cos(np.deg2rad(lat.gamma))

  else:
    raise NotImplementedError(f"ibrav not defined for spg {spg}")

  # get espresso unit cell lattice
  cell = ibrav_to_cell(espresso_in)[-1]

  # get sites for the new lattice
  new_sites = []
  latt = Lattice(cell)
  for s in std_struct:
    new_s = PeriodicSite(
      s.specie,
      s.coords,
      latt,
      to_unit_cell=True,
      coords_are_cartesian=True,
      properties=s.properties,
    )

    # tolerance and validate_proximity suggested by Max Großmann
    if not any([new_s.is_periodic_image(ns, tolerance=new_s.position_atol) for ns in new_sites]):
      new_sites.append(new_s)

  prim_struct = Structure.from_sites(new_sites, to_unit_cell=True, validate_proximity=True)

  if False: # Debug code
    print(structure)
    print(prim_struct)
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher()
    print(spg, sm.fit(prim_struct, structure))
    sys.exit()
 
  return espresso_in, prim_struct


def get_kpoints(structure, kppa=1500, _2D=False):
  """ Get kgrid for a given kkpa
  """
  if _2D:
    """Returns the max number of kpoints allowed during the
    kpoints convergence test. Based on the routine used in
    pymatgen.
    """
    from math import sqrt

    ase_structure = AseAtomsAdaptor.get_atoms(structure)

    area = ase_structure.get_volume()/ase_structure.get_cell()[2,2]

    rcell = ase_structure.cell.reciprocal()
    lengths = [sqrt(sum(map(lambda y: y**2, rcell[i])))
              for i in range(2)]
    ngrid = kppa/area

    mult = sqrt(ngrid / (lengths[0] * lengths[1]))
    num_div = [int(round((lengths[i] * mult))) for i in range(2)]
    num_div = [i if i > 0 else 1 for i in num_div]

    return num_div[0], num_div[1], 1

  else:
    from pymatgen.io.vasp.inputs import Kpoints
    kpts = Kpoints.automatic_density(structure, kppa, force_gamma=True)

    kpts_grid = [i if i % 2 == 0 else i + 1 for i in kpts.kpts[0]]

    return kpts_grid


def optimize_geometry(structure, input_params, kppa, _2D=False):
  """ Optimize the geometry using QE
  """
  inpparams = input_params.copy()
  inpparams["control"]["calculation"] = "vc-relax"
  inpparams["electrons"]["electron_maxstep"] = 200
  for i in range(2):
    a, structure = get_espresso_structure(structure)

    if _2D:
      inpparams["cell"]["cell_dofree"] = "2Dxy"
      inpparams["system"]["assume_isolated"] = "2D"

    inpparams["kpoints_grid"] = get_kpoints(structure, kppa=kppa, _2D=_2D)
    inpparams["system"].update(a)

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file(f"geoopt_{i}.in")
    os.system(QE["pw"] + " < " + f"geoopt_{i}.in" + " > " + f"geoopt_{i}.out")

    x = read(f"geoopt_{i}.out", format="espresso-out")
    adp = AseAtomsAdaptor()
    structure = adp.get_structure(x)
    
  structure.to(fmt="cif", filename="geo_opt.cif")
  return structure


def run_scf(structure, kpts, input_params, bandkpts, kpt_multiplier, _2D=False):
  """
  """
  inpparams = input_params.copy()
  a, structure = get_espresso_structure(structure)
  inpparams["system"].update(a)
  inpparams["control"]["calculation"] = "scf"
  # run with la2f and 4*ktps
  if not os.path.isfile("scf_fine.out"):
    inpparams["kpoints_grid"] = [4*i for i in kpts]
    if _2D:
      inpparams["kpoints_grid"][2] = 1
      inpparams["system"]["assume_isolated"] = "2D"

    inpparams["system"]["la2f"] = True

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file("scf_fine.in")
    os.system(QE["pw"] + " < " + "scf_fine.in" + " > " + "scf_fine.out")
  check_jobdone("scf_fine.out")

  if kpt_multiplier == 2 and not os.path.isfile("vfermi.frmsf"):
    os.system(QE["fermi_velocity"] + " < " + "scf_fine.in")

  # calculate el DOS and el-pDOS
  symbols = set([i.symbol for i in structure.species])
  efermi = read("scf_fine.out", format="espresso-out").calc.eFermi
  if not os.path.isfile("el-dos.out"):
    dos_params['dos']['Emin'] = efermi-20
    dos_params['dos']['Emax'] = efermi+10
    inp = PostProcessInput(dos_params)
    inp.write_file("el-dos.in")
    os.system(QE["dos"] + " < " + "el-dos.in" + " > " + "el-dos.out")
  check_jobdone("el-dos.out")

  if not os.path.isfile("el-pdos.out"):
    pdos_params['projwfc']['Emin'] = efermi-20
    pdos_params['projwfc']['Emax'] = efermi+10
    inp = PostProcessInput(pdos_params)
    inp.write_file("el-pdos.in")

    os.system(QE["pdos"] + " < " + "el-pdos.in" + " > " + "el-pdos.out")

  for s in symbols:
    if not os.path.isfile(f"atom_{s}_tot.dat"):
      os.system(QE["sumpdos"] +  "*\\(" + s + "\\)*" + " > " + f"atom_{s}_tot.dat")
  check_jobdone("el-pdos.out")

  # calculate bands
  if not os.path.isfile("el-bs.out"):
    inpparams = input_params.copy()
    inpparams["system"].update(a)
    inpparams["control"]["calculation"] = "bands"  
    inpparams["system"]["la2f"] = False
    inpparams["electrons"]["diagonalization"] = "cg" # "david" crashes sometimes
    inpparams["kpoints_grid"] = bandkpts
    inpparams["kpoints_mode"] = "crystal"
    inp = PWInput(structure=structure, **inpparams)
    inp.write_file("el-bs.in")
    os.system(QE["pw"] + " < " + "el-bs.in" + " > " + "el-bs.out")
  check_jobdone("el-bs.out")
  
  if not os.path.isfile("el-bands.out"):
    inp = PostProcessInput(bands_params)
    inp.write_file("el-bands.in")
    os.system(QE["bands"] + " < " + "el-bands.in" + " > " + "el-bands.out")    
  check_jobdone("el-bands.out")

  # run without la2f
  if not os.path.isfile("scf_coarse.out"):
    inpparams["kpoints_grid"] = [kpt_multiplier*i for i in kpts]
    if _2D:
      inpparams["kpoints_grid"][2] = 1
      inpparams["system"]["assume_isolated"] = "2D"

    inpparams["system"]["la2f"] = False
    inpparams["electrons"]["diagonalization"] = "david"
    inpparams["kpoints_mode"] = "automatic"
    inpparams["control"]["calculation"] = "scf"

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file("scf_coarse.in")
    os.system(QE["pw"] + " < " + "scf_coarse.in" + " > " + "scf_coarse.out")
  check_jobdone("scf_coarse.out")

  # calculate ELF
  if not os.path.isfile("elf.out") and not inpparams["system"].get("lspinorb", False):
    inp = PostProcessInput(elf_params)
    inp.write_file("elf.in")
    os.system(QE["pp"] + " < " + "elf.in" + " > " + "elf.out")    
  if not inpparams["system"].get("lspinorb", False): check_jobdone("elf.out")


def run_ph(kpts, qpts, is_metal, imcheck=True, _2D=False):
  # initialize variables
  q = 0
  dyn = {}

  # if dyn0 file is present read already performed q-points
  if os.path.isfile("qe.dyn0"):
    dyn = read_dyn("./")

  ph_params["inputph"]["nq1"] = qpts[0]
  ph_params["inputph"]["nq2"] = qpts[1]
  ph_params["inputph"]["nq3"] = qpts[2]

  while True:
    q += 1
    if ("nqp" in dyn) and (q > dyn["nqp"]):
      break
    if os.path.isfile(f"ph_{q}.out"):
      continue

    ph_params["inputph"]["start_q"] = q
    ph_params["inputph"]["last_q"] = q
    inp = PostProcessInput(ph_params)
    inp.write_file(f"ph_{q}.in")

    os.system(QE["ph"] + " < " + f"ph_{q}.in" + " > " + f"ph_{q}.out")
    check_jobdone(f"ph_{q}.out")

    nneg, dyn = check_imfreq("./", img_threshold=-45, prefix="qe", _2D=False)
    if imcheck and nneg != 0:
      sys.exit(f"Imaginary frequencies found at qpt {q}!!!")

  if not is_metal:
    q2r_params["input"]["la2F"] = False
    matdyn_params["input"]["la2F"] = False

  if not os.path.isfile("q2r.out"):
    if _2D:
      q2r_params["input"]["loto_2d"] = True
    inp = PostProcessInput(q2r_params)
    inp.write_file(f"q2r.in")
    os.system(QE["q2r"] + " < " + "q2r.in" + " > " + "q2r.out") 

  if not os.path.isfile("matdyn.out"):
    matdyn_params["input"]["nk1"] = kpts[0]*4
    matdyn_params["input"]["nk2"] = kpts[1]*4
    matdyn_params["input"]["nk3"] = kpts[2]*4

    if _2D:
      matdyn_params["input"]["nk3"] = 1      
      q2r_params["input"]["loto_2d"] = True

    inp = PostProcessInput(matdyn_params)
    inp.write_file(f"matdyn.in")
    os.system(QE["matdyn"] + " < " + "matdyn.in" + " > " + "matdyn.out")


def run_phbs(bandkpts):
  """
  
  """
  if not os.path.isfile("matdyn_bs.out"):
    inp = PostProcessInput(matdyn_bs_params, kpts=bandkpts)
    inp.write_file(f"matdyn_bs.in")
    os.system(QE["matdyn"] + " < " + "matdyn_bs.in" + " > " + "matdyn_bs.out")


def get_band_kpts(structure, _2D=False):

  from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
  from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo

  s = SpacegroupAnalyzer(structure)

  prim_struct = s.get_primitive_standard_structure(
      international_monoclinic=False)

  k = KPathSetyawanCurtarolo(prim_struct)

  if _2D: # remove out-of-plane k-points
    new_path = []
    for seg in k.kpath["path"]:
      new_seg = [kpt for kpt in seg if k.kpath["kpoints"][kpt][2] == 0.]
      if(len(new_seg) > 1): new_path.append(new_seg)
    k.kpath["path"] = new_path

  bandkpts, labels = k.get_kpoints(coords_are_cartesian=True,
                                     line_density=40)
  rec_latt = structure.lattice.reciprocal_lattice

  a = [labels[0]]
  b = [rec_latt.get_fractional_coords(bandkpts[0])]
  for j, i in enumerate(labels[1:]):
    if not i.strip() or i != a[-1]:
      a.append(i)
      b.append(rec_latt.get_fractional_coords(bandkpts[j + 1]))

  b = [i.tolist() for i in b]

  return b, a


def make_elbs_plot(structure, labels):
  """

  """
  matplotlib.rc('xtick', labelsize=20) 
  matplotlib.rc('ytick', labelsize=20)
  pyplot.rcParams['axes.labelsize'] = 22

  pyplot.rcParams['font.size'] = '18'
  pyplot.rcParams['axes.linewidth'] = 2
  pyplot.rcParams['legend.handlelength'] = .75

  symbols = set([i.symbol for i in structure.species])
  labels = [[i+1, j] for i,j in enumerate(labels) if j.strip()]

  bands = []
  with open("el-bands.dat.gnu") as f:
    lines = []
    for line in f.readlines():
      try:
        lines.append(float(line.strip().split()[1]))
      except IndexError:
        bands.append(lines)
        lines = []
  bands = pd.DataFrame(np.array(bands).T.tolist(), columns=[str(i) for i in range(len(bands))])

  with open(f'qe.el-dos') as f:
    alllines = f.readlines()
    lines = [list(map(float, i.split())) for i in alllines[1:]]
    efermi = float(alllines[0].split()[-2])
  edos = pd.DataFrame(lines, columns=['Energy', 'dos', 'intDOS'])
  edos["Energy"] -= efermi

  pdos = pd.DataFrame()
  for i in symbols:
    with open(f'atom_{i}_tot.dat') as f:
      lines = [list(map(float, i.split())) for i in f.readlines()[1:]]
    pdos = pd.concat([pdos, pd.DataFrame(lines, columns=["Energy", i])], axis=1)
  pdos = pdos.loc[:,~pdos.columns.duplicated()]
  pdos["Energy"] -= efermi

  ###

  fig = pyplot.figure()
  spec = gridspec.GridSpec(ncols=2, nrows=1,
                           width_ratios=[3, 1])

  ax1 = fig.add_subplot(spec[0])
  ax2 = fig.add_subplot(spec[1])

  ylimmin = -6
  ylimmax = 4

  ax1.plot(bands-efermi, c='tab:blue', lw=2)
  for i in labels:
      ax1.axvline(bands.index[int(i[0])-1], c='black', lw=2)
  ax1.set_xlim(bands.index[0], bands.index[-1])
  ax1.set_ylim(ylimmin, ylimmax)
  ax1.set_xticks([bands.index[int(i[0])-1] for i in labels])
  ax1.set_xticklabels([to_latex(i[1]) for i in labels])
  ax1.axhline(0, color='tab:red')
  ax1.set_ylabel('Energy [eV]')

  pdos = pdos[(pdos.Energy > ylimmin) & (pdos.Energy < ylimmax)]
  edos = edos[(edos.Energy > ylimmin) & (edos.Energy < ylimmax)]
  for i in symbols:
    ax2.plot(pdos[i], pdos.Energy, label=i, lw=2)
  ax2.plot(edos.dos, edos.Energy, color="black", label='Total', lw=2)
  ax2.set_ylim([ylimmin, ylimmax])
  ax2.set_xlim([0,  edos.dos.max()*1.05])
  ax2.axhline(0, color='tab:red')
  ax2.set_yticks([])
  ax2.set_xticks([])  
  ax2.legend()
  pyplot.tight_layout()
  pyplot.subplots_adjust(wspace=0, hspace=0)
  pyplot.savefig('el-bs.pdf')

def make_phbs_plot(structure, labels, a2f_file, is_metal):
    """

    """
    from scipy.integrate import cumulative_trapezoid

    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    pyplot.rcParams['axes.labelsize'] = 22
    pyplot.rcParams['font.size'] = '18'
    pyplot.rcParams['axes.linewidth'] = 2
    pyplot.rcParams['legend.handlelength'] = .75

    symbols = [i.symbol for i in structure.species]
    labels = [[i+1, j] for i,j in enumerate(labels) if j.strip()]
    nmodes = len(symbols)*3
  
    with open(f'qe.freq.gp') as f:
        lines = [list(map(float, i.split())) for i in f.readlines()]
    bs = pd.DataFrame(lines, columns=['Freq']+list(map(str, range(len(lines[0])-1))))
    bs.set_index('Freq', inplace=True)

    with open(f'matdyn.dos') as f:
        lines = [list(map(float, i.split())) for i in f.readlines()[1:]]
    pdos = pd.DataFrame(lines, columns=['Freq', 'tdos'] + symbols)

    fig = pyplot.figure(figsize=(10, 6.6))
    spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[5, 2, 2])

    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax3 = fig.add_subplot(spec[2])

    max_freq = int(bs.values.max()*1.05)
    for i in set(symbols):
        if pdos.columns.to_list().count(i) > 1:
            ax2.plot(pdos[i].values.sum(axis=1), pdos.Freq.values, label=i, lw=2)
        else:
            ax2.plot(pdos[i].values, pdos.Freq.values, label=i, lw=2)

    ax2.set_ylim(0, max_freq)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.legend()

    if is_metal:
      with open(a2f_file) as f:
        lines = f.readlines()
        lines = [[float(j) for j in i.split()] for i in lines[5:-1]]
      a2f = pd.DataFrame(lines, columns=["Freq", "tot"] + ["m%s" % i for i in range(nmodes)])

      mesh = a2f.Freq.values*219474.63/2 # Ry to cm-1
      ff = a2f.tot/mesh
      la = 2*cumulative_trapezoid(ff[1:], x=mesh[1:], initial=0.0)
      ax3.plot(a2f.tot.values, mesh, label="a$^2$F($\\omega$)", lw=2)

      ax3.plot(la, mesh[1:], label="$\\lambda$($\\omega$)", lw=2)
      ax3.set_ylim(0, max_freq)
      ax3.set_yticks([])
      ax3.legend()

    ax1.plot(np.arange(bs.shape[0]), bs.values, c='tab:blue', lw=2)
    for i in labels:
        ax1.axvline(int(i[0])-1, c='black', lw=2)
    ax1.set_xlim(0, bs.shape[0])
    ax1.set_ylim(0, max_freq)
    ax1.set_xticks([int(i[0])-1 for i in labels])
    ax1.set_xticklabels([to_latex(i[1]) for i in labels])
    ax1.set_ylabel('Frequency [cm$^{-1}$]')

    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.savefig('ph-bs.pdf')


def get_bandgap():
    nelectrons = int(round(float(os.popen("grep 'renormalised to' scf_fine.out").read().split()[-1]),0))
    fermi_band =  nelectrons // 2 - 1 + nelectrons%2
    nkpts = int(os.popen("grep 'number of k points=' el-bs.out").read().split()[4])
    efermi = float(os.popen(f"grep 'Fermi' scf_fine.out").read().split()[-2])

    bands = np.loadtxt("el-bands.dat.gnu")[:,1]
    nbands = bands.size // nkpts
    bands = bands.reshape([nbands, nkpts]).T

    val = bands[:, fermi_band].max()
    cond = bands[:, fermi_band+1].min()
    if (efermi < val) or (cond < val):
        return 0.0
    else:
        return cond-val

def parse_cmd_line():
  import argparse
  parser = argparse.ArgumentParser(description="Calculates electron-phonon coupling")

  parser.add_argument('structure', default='start.cif',
                    type=str, nargs='?', help='structure to calculate')
  parser.add_argument('-c', '--config', default=os.path.join(CONFIG_PATH, "config.json"),
                    type=str, help='Location of json config file')
  parser.add_argument('-a', '--accurate', action='store_true',
                    help='Increase accuracy')
  parser.add_argument('--soc', action='store_true',
                    help='Use spin orbit coupling')
  parser.add_argument('-cut', '--cutoff', default=1.0, 
                    type=float, help='Multiplicative factor for wf cutoff')
  parser.add_argument('-cg', '--conjugate_gradients', action='store_true',
                    help='Use conjugate gradients instead of Davidson')
  parser.add_argument('--charge', type=float, default=0.0,
                    help='Add or remove to total charge; Defaults to 0')
  parser.add_argument('--pressure', type=float, default=0.0,
                    help='Target pressure in GPa; Defaults to 0')
  parser.add_argument('--qpts', type=str, default="",
                    help='Q-points, e.g. 8x8x6; Defaults to 0')
  parser.add_argument('--metal_threshold', type=float, default=0.1,
                    help='Gap threshold to consider a metal (in eV)')
  parser.add_argument('--only_metals', action='store_true',
                    help='Stops the workflow if a semiconductor is found')
  parser.add_argument('--no_imcheck', action='store_true',
                    help='Stops the workflow if imaginary frequencies are found')
  parser.add_argument('--2D', dest='_2D', action='store_true',
                    help='Assumes 2D system with c the non-periodic axis')


  args = parser.parse_args()
  return args


if __name__ == "__main__":
    import sys
    from allen_dynes import AllenDynes

    args = parse_cmd_line()

    config_file = args.config
    if os.path.isfile(config_file):
      config = json.load(open(config_file))
    else:
      json.dump(config, open(config_file, "w"))

    PSEUDO_DIR = config["PSEUDO_DIR"]
    QE = config["QE"]
    PP_VERSION = f"{[i for i in PSEUDO_DIR.split('/') if i.strip()][-1].split('_upf')[0]}.json"

    a2f = "a2F.dos5"

    ANNOUNCE_TC_THRESHOLD = 20
    DISCORD_TOKEN = os.environ.get("DISCORD_ONNES_TOKEN", None)
    CHANNEL_ID = os.environ.get("DISCORD_ONNES_CHANNEL_ID", None)
    if DISCORD_TOKEN and CHANNEL_ID:
      DISCORD_ANNOUCE = True
    else:
      DISCORD_ANNOUCE = False
      print("Discord BOT disabled, check DISCORD_ONNES_TOKEN and DISCORD_ONNES_CHANNEL_ID")

    structure = Structure.from_file(args.structure)
    structure = standardize_cell(structure, symprec=0.1)

    inp_params["system"]["ecutwfc"] = get_cutoff(structure.composition)*args.cutoff
    inp_params["pseudo"] = {i.symbol:f"{i}.upf" for i in structure.composition}
    inp_params["control"]["pseudo_dir"] = PSEUDO_DIR
    inp_params["system"]["tot_charge"] = args.charge
    inp_params["cell"]["press"] = args.pressure * 10 # convert GPa to kbar
    if args.conjugate_gradients:
      inp_params["electrons"]["diagonalization"] = "cg"
      ph_params["inputph"]["diagonalization"] = "cg"

    if args.accurate:
      KPPA = 3.0*(6*np.pi)**2 if args._2D else 3000
      kpt_multiplier = 2
    else:
      KPPA = 1.5*(6*np.pi)**2 if args._2D else 1500
      kpt_multiplier = 1

    if args.soc:
      inp_params["system"]["lspinorb"] = True
      inp_params["system"]["noncolin"] = True

    if os.path.isfile("geo_opt.cif"):
      structure = Structure.from_file("geo_opt.cif")
    else:
      structure = optimize_geometry(structure, inp_params, kppa=KPPA, _2D=args._2D)

    kpts = get_kpoints(structure, kppa=KPPA)
    if args.qpts != "":
      qpts = [int(i) for i in re.split(r'\D+', args.qpts)]
    else:
      qpts = [i//2 for i in kpts]

    # If kpath already present, use it; ortherwise get it and save for restarts
    if os.path.isfile("kpath.dat"):
      bandkpts, labels = json.load(open("kpath.dat"))
    else:
      bandkpts, labels = get_band_kpts(structure, _2D=args._2D)
      json.dump([bandkpts, labels], open("kpath.dat", "w"))

    run_scf(structure, kpts, inp_params, bandkpts, kpt_multiplier, _2D=args._2D)
    make_elbs_plot(structure, labels)

    gap = get_bandgap()
    if gap > args.metal_threshold:
      print(f"Material has a gap of {gap:0.2f}!!")
      metal = False
      if args.only_metals:
        sys.exit("Semiconductor found, stopping.")
    else:
      metal = True
      if gap <= args.metal_threshold:
        print(f"Small gap ({gap:0.2f}) detected! Assuming metal...")

    run_ph(kpts, qpts, is_metal=metal, imcheck=(not args.no_imcheck), _2D=args._2D)
    run_phbs(bandkpts)
    make_phbs_plot(structure, labels, a2f, is_metal=metal)

    # calculate average lambda, wlog, Tc ignoring smallest and largest smearings
    if metal:
      epc = {}
      for i in range(2, 10):
        ad = AllenDynes(f'a2F.dos{i}')
        for prop in ["la", "wlog", "Tc"]:
          epc[prop] = epc.get(prop, []) + [getattr(ad, prop)]
      with open("Tc.dat", "w") as f:
        for key, values in epc.items():
          m = np.mean(values, axis=0)
          s = np.std(values, axis=0)
          if isinstance(m, float):
            f.write(f"{key}: {m:.3f}±{s:.3f}\n")
          else:
            f.write(f"Tc_AD: {m[0]:.3f}±{s[0]:.3f}\n")
            f.write(f"Tc_OPT: {m[1]:.3f}±{s[1]:.3f}\n")
      if DISCORD_ANNOUCE:
        if m[0] > ANNOUNCE_TC_THRESHOLD:
          # Send discord notification
          send_discord_message_with_images()
    else:
      with open("Tc.dat", "w") as f:
        f.write("la: 0.0\nwlog: 0.0\nTc_AD: 0.0\nTc_OPT: 0.0\n")
