import os, json, math
from matplotlib import gridspec
from pw_utils import MyPWInput as PWInput
from pw_utils import check_imfreq, read_dyn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.units import bohr_to_angstrom, Ry_to_eV
from ase.io.espresso import ibrav_to_cell
from ase.io import read
import pandas as pd
import numpy as np

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
    'degauss': 0.2/Ry_to_eV
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
    'cell_dynamics': 'bfgs'
  }
}

pdos_params = """&projwfc
  outdir='outdir',
  prefix='qe',
  degauss = 0.008,
  Emin=%s, Emax=%s, DeltaE=0.01
 /
"""

dos_params = """&dos
  outdir='outdir',
  prefix='qe',
  degauss = 0.008,
  fildos='qe.el-dos'
  Emin=%s, Emax=%s, DeltaE=0.01
 /
"""

bands_params = """&bands
  prefix="qe"
  outdir='outdir',
  filband="el-bands.dat"
/
"""

elf_params = """&inputPP
   prefix   = 'qe',
   outdir   = 'outdir'
   plot_num = 8
   filplot  = 'qe.elf.dat'
/
&plot
   iflag         = 3,
   output_format = 6,
   fileout       = 'elf.cube'
/
"""

ph_params = """&inputph
  reduce_io = .true.,
  search_sym = .false.,
  recover  = .false.,
  prefix   = 'qe',
  fildyn   = 'qe.dyn',
  outdir   = 'outdir'
  ldisp    = .true.,
  trans    = .true.,
  fildvscf = 'dvscf',
  diagonalization = 'david',
  electron_phonon = 'interpolated',
  el_ph_sigma = 0.005,
  el_ph_nsigma = 10,
  nq1 = %s,
  nq2 = %s,
  nq3 = %s,
  tr2_ph   =  1.0d-15,
  start_q = %s
  last_q = %s
  niter_ph = 100,
  alpha_mix(1) = 0.2
 /
"""

q2r_params = """&input
 la2F = .true.,
 fildyn='qe.dyn',
 zasr='simple', 
 flfrc='qe.fc'
/
"""

matdyn_params = """&input
 la2F = .true.,
 asr='simple',  
 flfrc='qe.fc', 
 flfrq='qe.freq'
 q_in_cryst_coord=.true.,
 dos = .true.
 nk1 = %s
 nk2 = %s
 nk3 = %s
/
"""

matdyn_bs_params = """&input
 asr='simple',  
 flfrc='qe.fc', 
 flfrq='qe.freq'
 q_in_cryst_coord=.true.,
/
"""

def check_jobdone(filename):
  if not os.popen(f"grep 'JOB DONE' {filename}").read().strip():
    sys.exit(f"Error: {filename} did no finish successfully!!!")

def to_latex(string):
  if "\\" in string:
    string = f"$\{'_'.join(string[1:].split('_'))}$"
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
      json.dump(dojo, open(jsonfile, "w"))

  cutoffs = [dojo[i.symbol]['hh'] for i in composition]

  # convert to Rydbergs
  return max(cutoffs)*2

def get_espresso_structure(structure):
  """ Get structure in a format suitable for QE
  SPG info from: https://msestudent.com/list-of-space-groups/
  """
  from pymatgen.core.lattice import Lattice
  from pymatgen.core.structure import PeriodicSite, Structure
  
  # first we determine the conventional structure
  sym = SpacegroupAnalyzer(structure, symprec=1e-5)
  std_struct  = sym.get_conventional_standard_structure(international_monoclinic=False)

  spg = sym.get_space_group_number()

  # This is the structure in espresso input format
  espresso_in = {}
  espresso_in["celldm(1)"] = std_struct.lattice.a/bohr_to_angstrom

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
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a
    
  elif spg in [146, 148, 155, 160, 161, 166, 167]:
    # Rhombohedral
    espresso_in["ibrav"] = 5
    aR       = math.sqrt(std_struct.lattice.a**2/3 + std_struct.lattice.c**2/9)
    cosgamma = (2*std_struct.lattice.c**2 - 3*std_struct.lattice.a**2)/(2*std_struct.lattice.c**2 + 6*std_struct.lattice.a**2)
    espresso_in["celldm(1)"] = aR/bohr_to_angstrom
    espresso_in["celldm(4)"] = cosgamma

  elif spg in [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 111, 112, 113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]:
    # Simple Tetragonal
    espresso_in["ibrav"] = 6
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a
    
  elif spg in [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142]:
    # Body-Centered Tetragonal
    espresso_in["ibrav"] = 7
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a

  elif spg in [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]:
    #  Simple Orthorhombic
    espresso_in["ibrav"] = 8
    espresso_in["celldm(2)"] = std_struct.lattice.b/std_struct.lattice.a
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a

  elif spg in [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68]:
    # Base-Centered Orthorhombic
    espresso_in["ibrav"] = 9
    espresso_in["celldm(2)"] = std_struct.lattice.b/std_struct.lattice.a
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a
    
  elif spg in [22, 42, 43, 69, 70]:
    # Face-Centered Orthorhombic
    espresso_in["ibrav"] = 10
    espresso_in["celldm(2)"] = std_struct.lattice.b/std_struct.lattice.a
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a
    
  elif spg in [23, 24, 44, 45, 46, 71, 72, 73, 74]:
    # Body-Centered Orthorhombic
    espresso_in["ibrav"] = 11
    espresso_in["celldm(2)"] = std_struct.lattice.b/std_struct.lattice.a
    espresso_in["celldm(3)"] = std_struct.lattice.c/std_struct.lattice.a

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
    if not any(map(new_s.is_periodic_image, new_sites)):
      new_sites.append(new_s)

  return espresso_in, Structure.from_sites(new_sites)


def get_kpoints(structure, kppa=1500):
  """ Get kgrid for a given kkpa
  """
  from pymatgen.io.vasp.inputs import Kpoints

  kpts = Kpoints.automatic_density(structure, kppa, force_gamma=True)

  # round kpoints to the even number above
  return [i if i % 2 == 0 else i+1 for i in kpts.kpts[0]]

def optimize_geometry(structure, input_params, kppa):
  """ Optimize the geometry using QE
  """
  inpparams = input_params.copy()
  inpparams["control"]["calculation"] = "vc-relax"
  inpparams["electrons"]["electron_maxstep"] = 200
  for i in range(2):
    a, structure = get_espresso_structure(structure)
    inpparams["kpoints_grid"] = get_kpoints(structure, kppa=kppa)
    inpparams["system"].update(a)

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file(f"geoopt_{i}.in")
    os.system(QE["pw"] + " < " + f"geoopt_{i}.in" + " > " + f"geoopt_{i}.out")

    x = read(f"geoopt_{i}.out", format="espresso-out")
    adp = AseAtomsAdaptor()
    structure = adp.get_structure(x)
    
  structure.to(fmt="cif", filename="geo_opt.cif")
  return structure

def run_scf(structure, kpts, input_params, bandkpts, kpt_multiplier):
  """
  """
  inpparams = input_params.copy()
  a, structure = get_espresso_structure(structure)
  inpparams["system"].update(a)
  inpparams["control"]["calculation"] = "scf"
  # run with la2f and 4*ktps
  if not os.path.isfile("scf_fine.out"):
    inpparams["kpoints_grid"] = [4*i for i in kpts]
    inpparams["system"]["la2f"] = True

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file("scf_fine.in")
    os.system(QE["pw"] + " < " + "scf_fine.in" + " > " + "scf_fine.out")
  check_jobdone("scf_fine.out")

  # calculate el DOS and el-pDOS
  symbols = set([i.symbol for i in structure.species])
  efermi = read("scf_fine.out", format="espresso-out").calc.eFermi
  if not os.path.isfile("el-dos.out"):
    with open("el-dos.in", "w") as f:
      f.write(dos_params % (efermi-20, efermi+10))
    os.system(QE["dos"] + " < " + "el-dos.in" + " > " + "el-dos.out")
  check_jobdone("el-dos.out")

  if not os.path.isfile("el-pdos.out"):
    with open("el-pdos.in", "w") as f:
      f.write(pdos_params % (efermi-20, efermi+10))
    os.system(QE["pdos"] + " < " + "el-pdos.in" + " > " + "el-pdos.out")
    for s in symbols:
      os.system(QE["sumpdos"] +  f"*\({s}\)*" + " > " + f"atom_{s}_tot.dat")
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
    with open("el-bands.in", "w") as f:
      f.write(bands_params)
    os.system(QE["bands"] + " < " + "el-bands.in" + " > " + "el-bands.out")    
  check_jobdone("el-bands.out")

  # run without la2f
  if not os.path.isfile("scf_coarse.out"):
    inpparams["kpoints_grid"] = [kpt_multiplier*i for i in kpts]
    inpparams["system"]["la2f"] = False
    inpparams["electrons"]["diagonalization"] = "david"
    inpparams["kpoints_mode"] = "automatic"
    inpparams["control"]["calculation"] = "scf"

    inp = PWInput(structure=structure, **inpparams)
    inp.write_file("scf_coarse.in")
    os.system(QE["pw"] + " < " + "scf_coarse.in" + " > " + "scf_coarse.out")
  check_jobdone("scf_coarse.out")

  # calculate ELF
  if not os.path.isfile("elf.out"):
    with open("elf.in", "w") as f:
      f.write(elf_params)
    os.system(QE["pp"] + " < " + "elf.in" + " > " + "elf.out")    
  check_jobdone("elf.out")

def run_ph(kpts):
  # initialize variables
  q = 0
  dyn = {}

  # if dyn0 file is present read already eprformed q-points
  if os.path.isfile("qe.dyn0"):
    dyn = read_dyn("./")

  while True:
    q += 1
    if ("nqp" in dyn) and (q > dyn["nqp"]):
      break
    if os.path.isfile(f"ph_{q}.out"):
      continue
    with open(f"ph_{q}.in", "w") as f:
      f.write(ph_params % tuple([i//2 for i in kpts] + [q, q]))
    os.system(QE["ph"] + " < " + f"ph_{q}.in" + " > " + f"ph_{q}.out")
    check_jobdone(f"ph_{q}.out")

    nneg, dyn = check_imfreq("./", img_threshold=-35, prefix="qe")
    if nneg != 0:
      sys.exit(f"Imaginary frequencies found at qpt {q}!!!")


  if not os.path.isfile("q2r.out"):
    with open("q2r.in", "w") as f:
      f.write(q2r_params)
    os.system(QE["q2r"] + " < " + "q2r.in" + " > " + "q2r.out") 

  if not os.path.isfile("matdyn.out"):
    with open("matdyn.in", "w") as f:
      f.write(matdyn_params % tuple([i*4 for i in kpts]))
    os.system(QE["matdyn"] + " < " + "matdyn.in" + " > " + "matdyn.out")

def run_phbs(bandkpts):
  """
  
  """
  if not os.path.isfile("matdyn_bs.out"):
    with open("matdyn_bs.in", "w") as f:
      f.write(matdyn_bs_params)
      f.write(f"{len(bandkpts)} crystal\n")
      for i,j,k in bandkpts:
        f.write(f"{i:12.10f} {j:12.10f} {k:12.10f}\n")
      f.write("\n")
    os.system(QE["matdyn"] + " < " + "matdyn_bs.in" + " > " + "matdyn_bs.out")

def get_band_kpts(structure):
  from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
  from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo

  s = SpacegroupAnalyzer(structure)
  prim_struct = s.get_primitive_standard_structure(international_monoclinic=False)

  k = KPathSetyawanCurtarolo(prim_struct)
  bandkpts, labels = k.get_kpoints(coords_are_cartesian=True, line_density=40)

  rec_latt = structure.lattice.reciprocal_lattice

  # convert to fractional coordinates of the espresso lattice
  # bands.x does not like repeated consecutive kpts
  a = [labels[0]]
  b = [rec_latt.get_fractional_coords(bandkpts[0])]
  for j,i in enumerate(labels[1:]):
    if not i.strip() or i != a[-1]:
      a.append(i)
      b.append(rec_latt.get_fractional_coords(bandkpts[j+1]))
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
  #ax2.set_ylabel("Energy [eV]")
  #ax2.xlabel("DOS [states/eV]")
  pyplot.tight_layout()
  pyplot.subplots_adjust(wspace=0, hspace=0)
  pyplot.savefig('el-bs.pdf')

def make_phbs_plot(structure, labels, a2f_file):
    """

    """
    from scipy.integrate import cumtrapz

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

    with open(a2f_file) as f:
      lines = f.readlines()
      lines = [[float(j) for j in i.split()] for i in lines[5:-1]]
    a2f = pd.DataFrame(lines, columns=["Freq", "tot"] + ["m%s" % i for i in range(nmodes)])

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

    mesh = a2f.Freq.values*219474.63/2 # Ry to cm-1
    ff = a2f.tot/mesh
    la = 2*cumtrapz(ff[1:], x=mesh[1:], initial=0.0)
    ax3.plot(a2f.tot.values, mesh, label=f"a$^2$F($\omega$)", lw=2)

    ax3.plot(la, mesh[1:], label=f"$\lambda$($\omega$)", lw=2)
    ax3.set_ylim(0, max_freq)
    ax3.set_yticks([])
    ax3.legend()

    ax1.plot(bs.index.values, bs.values, c='tab:blue', lw=2)
    for i in labels:
        ax1.axvline(bs.index[int(i[0])-1], c='black', lw=2)
    ax1.set_xlim(bs.index[0], bs.index[-1])
    ax1.set_ylim(0, max_freq)
    ax1.set_xticks([bs.index[int(i[0])-1] for i in labels])
    ax1.set_xticklabels([to_latex(i[1]) for i in labels])
    ax1.set_ylabel('Frequency [cm$^{-1}$]')

    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.savefig('ph-bs.pdf')


def parse_cmd_line():
  import argparse
  parser = argparse.ArgumentParser(description="Calculates electron-phonon coupling")

  parser.add_argument('structure', default='start.cif',
                    type=str, nargs='?', help='structure to calculate')
  parser.add_argument('-c', '--config', default=os.path.join(CONFIG_PATH, "config.json"),
                    type=str, help='Location of json config file')
  parser.add_argument('-a', '--accurate', action='store_true',
                    help='Increase accuracy')
  parser.add_argument('-cg', '--conjugate_gradients', action='store_true',
                    help='Use conjugate gradients instead of Davidson')

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

    structure = Structure.from_file(args.structure)
    structure = standardize_cell(structure, symprec=0.1)

    inp_params["system"]["ecutwfc"] = get_cutoff(structure.composition)
    inp_params["pseudo"] = {i.symbol:f"{i}.upf" for i in structure.composition}
    inp_params["control"]["pseudo_dir"] = PSEUDO_DIR
    if args.conjugate_gradients:
      inp_params["electrons"]["diagonalization"] = "cg"
      ph_params = ph_params.replace("david", "cg")

    if args.accurate:
      KPPA = 3000
      kpt_multiplier = 2
    else:
      KPPA = 1500
      kpt_multiplier = 1

    if os.path.isfile("geo_opt.cif"):
      structure = Structure.from_file("geo_opt.cif")
    else:
      structure = optimize_geometry(structure, inp_params, kppa=KPPA)

    kpts = get_kpoints(structure, kppa=KPPA)
    bandkpts, labels = get_band_kpts(structure)
    run_scf(structure, kpts, inp_params, bandkpts, kpt_multiplier)
    make_elbs_plot(structure, labels)
    run_ph(kpts)
    run_phbs(bandkpts)
    make_phbs_plot(structure, labels, a2f)

    # calculate average lambda, wlog, Tc ignoring smallest and largest smearings
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

