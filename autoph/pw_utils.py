from pymatgen.io.pwscf import PWInput
from pymatgen.core import Element
import sys, re
import numpy as np

class MyPWInput(PWInput):
    """
    Initializes a PWSCF input file.
    Args:
        structure (Structure): Input structure. For spin-polarized calculation,
            properties (e.g. {"starting_magnetization": -0.5,
            "pseudo": "Mn.pbe-sp-van.UPF"}) on each site is needed instead of
            pseudo (dict).
        pseudo (dict): A dict of the pseudopotentials to use. Default to None.
        control (dict): Control parameters. Refer to official PWSCF doc
            on supported parameters. Default to {"calculation": "scf"}
        system (dict): System parameters. Refer to official PWSCF doc
            on supported parameters. Default to None, which means {}.
        electrons (dict): Electron parameters. Refer to official PWSCF doc
            on supported parameters. Default to None, which means {}.
        ions (dict): Ions parameters. Refer to official PWSCF doc
            on supported parameters. Default to None, which means {}.
        cell (dict): Cell parameters. Refer to official PWSCF doc
            on supported parameters. Default to None, which means {}.
        kpoints_mode (str): Kpoints generation mode. Default to automatic.
        kpoints_grid (sequence): The kpoint grid. Default to (1, 1, 1).
        kpoints_shift (sequence): The shift for the kpoints. Defaults to
            (0, 0, 0).
    """    
    def __str__(self):
        out = []
        site_descriptions = {}

        if self.pseudo is not None:
            site_descriptions = self.pseudo
        else:
            c = 1
            for site in self.structure:
                name = None
                for k, v in site_descriptions.items():
                    if site.properties == v:
                        name = k

                if name is None:
                    name = site.specie.symbol + str(c)
                    site_descriptions[name] = site.properties
                    c += 1

        def to_str(v):
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, float):
                return f"{str(v).replace('e', 'd')}"
            if isinstance(v, bool):
                if v:
                    return ".TRUE."
                return ".FALSE."
            return v

        for k1 in ["control", "system", "electrons", "ions", "cell"]:
            v1 = self.sections[k1]
            out.append(f"&{k1.upper()}")
            sub = []
            for k2 in sorted(v1.keys()):
                if isinstance(v1[k2], list):
                    n = 1
                    for l in v1[k2][: len(site_descriptions)]:
                        sub.append(f"  {k2}({n}) = {to_str(v1[k2][n - 1])}")
                        n += 1
                else:
                    sub.append(f"  {k2} = {to_str(v1[k2])}")
            if k1 == "system":
                if "ibrav" not in self.sections[k1]:
                    sub.append("  ibrav = 0")
                if "nat" not in self.sections[k1]:
                    sub.append(f"  nat = {len(self.structure)}")
                if "ntyp" not in self.sections[k1]:
                    sub.append(f"  ntyp = {len(site_descriptions)}")
            sub.append("/")
            out.append(",\n".join(sub))

        out.append("ATOMIC_SPECIES")
        for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
            e = re.match(r"[A-Z][a-z]?", k).group(0)
            if self.pseudo is not None:
                p = v
            else:
                p = v["pseudo"]
            out.append(f"  {k}  {Element(e).atomic_mass:.4f} {p}")

        out.append("ATOMIC_POSITIONS crystal")
        if self.pseudo is not None:
            for site in self.structure:
                out.append(f"  {site.specie} {site.a:.8} {site.b:.8} {site.c:.8}")
        else:
            for site in self.structure:
                name = None
                for k, v in sorted(site_descriptions.items(), key=lambda i: i[0]):
                    if v == site.properties:
                        name = k
                out.append(f"  {name} {site.a:.8} {site.b:.8} {site.c:.8}")

        out.append(f"K_POINTS {self.kpoints_mode}")
        if self.kpoints_mode == "automatic":
            kpt_str = [f"{i}" for i in self.kpoints_grid]
            kpt_str.extend([f"{i}" for i in self.kpoints_shift])
            out.append(f"  {' '.join(kpt_str)}")
        elif self.kpoints_mode == "crystal_b":
            out.append(f" {str(len(self.kpoints_grid))}")
            for i in range(len(self.kpoints_grid)):
                kpt_str = [f"{entry:.4f}" for entry in self.kpoints_grid[i]]
                out.append(f" {' '.join(kpt_str)}")
        elif self.kpoints_mode == "crystal":
            out.append(f" {str(len(self.kpoints_grid))}")
            for i in range(len(self.kpoints_grid)):
                kpt_str = [f"{entry:.10f}" for entry in self.kpoints_grid[i]]
                out.append(f" {' '.join(kpt_str)}" + " 1.0")
        elif self.kpoints_mode == "gamma":
            pass

        # Difference to original
        if "ibrav" not in self.sections["system"]:
            out.append("CELL_PARAMETERS angstrom")
            for vec in self.structure.lattice.matrix:
                out.append(f"  {vec[0]:.15f} {vec[1]:.15f} {vec[2]:.15f}")
        return "\n".join(out)+"\n\n"

def read_dyn(mdir, prefix="qe"):
  dyn = {}
  
  # we start by reading the k-points
  fh = open(mdir + "/" + prefix + ".dyn0")
  dyn["qgrid"] = [int(i) for i in fh.readline().split()]
  dyn["nqp"]   = int(fh.readline())

  dyn["qpts"]  = []
  for qpt in range(dyn["nqp"]):
    dyn["qpts"].append([round(float(i), 5) for i in fh.readline().split()])
  fh.close()

  # read now the rest of the .dyn files
  dyn["dyn"] = []
  dyn["species"] = []
  dyn["sites"] = []
  for kpt in range(dyn["nqp"]):
    dyn_file = {}
    try:
      fh = open(mdir + "/" + prefix + ".dyn" + str(kpt+1))
    except Exception:
      continue

    # skip first two lines, and check if file is empty
    if len(fh.readline()) == 0:
      continue
    fh.readline()

    fields = fh.readline().split()
    if kpt == 0:
      dyn["ntyp"]   = int(fields[0])
      dyn["nat"]    = int(fields[1])
      dyn["ibrav"]  = int(fields[2])
      dyn["celldm"] = [float(i) for i in fields[3:]]
    if dyn["ibrav"] == 0:
        #skip bravais lattice
        [fh.readline() for i in range(4)]

    for i in range(dyn["ntyp"]):
      fields = fh.readline().split()
      symbol = re.sub(r"[ ']", "", fields[1])
      if kpt == 0:
        dyn["species"].append([symbol, float(fields[3])])

    for i in range(dyn["nat"]):
      fields = fh.readline().split()
      if kpt == 0:
        dyn["sites"].append([dyn["species"][int(fields[1]) - 1][0],
                             float(fields[2]), float(fields[3]), float(fields[4])])

    # a dyn file constains all dynamic matrices for equivalent q-points
    dyn_file["dynmatrix"] = []
    while True:
      fh.readline()
      if fh.readline().strip() != "Dynamical  Matrix in cartesian axes":
        break

      fh.readline()

      dynmatrix_q = {}
      
      fields = fh.readline().split()
      dynmatrix_q["qpt"] = [round(float(i), 5) for i in fields[3:6]]
      fh.readline()

      dynmatrix = np.zeros((dyn["nat"], dyn["nat"], 3, 3), dtype=complex)
      for ia in range(dyn["nat"]):
        for ja in range(dyn["nat"]):
          fh.readline()
          for k in range(3):
            fields = fh.readline().split()
            dynmatrix[ia, ja, k, 0] = float(fields[0]) + float(fields[1])*1j
            dynmatrix[ia, ja, k, 1] = float(fields[2]) + float(fields[3])*1j
            dynmatrix[ia, ja, k, 2] = float(fields[4]) + float(fields[5])*1j

      dynmatrix_q["dynmatrix"] = dynmatrix.tolist()
      dyn_file["dynmatrix"].append(dynmatrix_q)

    fh.readline()
    fields = fh.readline().split()
    dyn_file["qpt"] = [round(float(i), 5) for i in fields[3:6]]
    fh.readline(); fh.readline()
    
    freq = []
    vecs = []
    for i in range(3*dyn["nat"]):
      fields = fh.readline().split()
      freq.append(float(fields[7])) # in cm^-1

      vec = np.zeros((dyn["nat"], 3), dtype=complex)
      for ia in range(dyn["nat"]):
        fields = fh.readline().split()
        vec[ia, 0] = float(fields[1]) + float(fields[2])*1j
        vec[ia, 1] = float(fields[3]) + float(fields[4])*1j
        vec[ia, 2] = float(fields[5]) + float(fields[6])*1j
      vecs.append(vec.tolist())

    dyn_file["freqs"] = freq
    dyn_file["vecs"]  = vecs

    dyn["dyn"].append(dyn_file)
  return dyn

def check_imfreq(mdir="./", img_threshold=-25, prefix="qe"):
    dyn = read_dyn(mdir, prefix=prefix)
    nimag = 0
    for i in dyn["dyn"]:
        # is it Gamma
        nimag_qpt = sum([1 for f in i["freqs"] if f < 0.0])
        if i["qpt"][0] == 0. and i["qpt"][1] == 0. and i["qpt"][2] == 0.:
            nok = min(3, sum([1 for f in i["freqs"] if f < 0.0 and f > img_threshold]))
            nimag_qpt = nimag_qpt - nok
        nimag += nimag_qpt
    return nimag, dyn

def read_a2Fdos(mdir, fname="a2F.dos10"):
    fh = open(mdir + "/" + fname)

    # skip first 5 lines
    for i in range(5):
        fh.readline()

    a2Fdos = {"freqs": [], "total": [], "modes": []}
    for line in fh.readlines():
        fields = line.split()
        
        if fields[0] == "lambda":
            break

        a2Fdos["freqs"].append(float(fields[0]))
        a2Fdos["total"].append(float(fields[1]))
        a2Fdos["modes"].append([float(i) for i in fields[2:]])

    return a2Fdos
