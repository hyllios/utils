from pymatgen.io.pwscf import PWInput
from pymatgen.core import Element
import sys, re
import numpy as np

class PostProcessInput(object):
    """
    
    """
    def __init__(self, params_dict, kpts=None):
        self.sections = params_dict
        self.kpts = kpts
    
    def write_file(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))

    def __str__(self):
        out = []
        site_descriptions = {}

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

        for k1, v1 in self.sections.items():
            out.append(f"&{k1.upper()}")
            sub = []
            for k2, v2 in sorted(v1.items()):
                if isinstance(v2, list):
                    n = 1
                    for l in v2[: len(site_descriptions)]:
                        sub.append(f"  {k2}({n}) = {to_str(v2[n - 1])}")
                        n += 1
                else:
                    sub.append(f"  {k2} = {to_str(v2)}")
            sub.append("/")
            out.append(",\n".join(sub))

        if self.kpts is not None:
            out.append(f"{len(self.kpts)}")
            for i,j,k in self.kpts:
                out.append(f"{i:12.10f} {j:12.10f} {k:12.10f}")
        return "\n".join(out)+"\n\n"

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

def check_imfreq(mdir="./", img_threshold=-25, prefix="qe", _2D=False):
    dyn = read_dyn(mdir, prefix=prefix)
    nimag = 0
    for i in dyn["dyn"]:
        nimag_qpt = sum([1 for f in i["freqs"] if f < 0.0])

        # is it Gamma
        if i["qpt"][0] == 0. and i["qpt"][1] == 0. and i["qpt"][2] == 0.:
            nok = min(3, sum([1 for f in i["freqs"] if f < 0.0 and f > img_threshold]))
            nimag_qpt = nimag_qpt - nok

        # we have to give some slack to the flexural mode
        elif _2D and i["freqs"][0] > img_threshold/2:
            nimag_qpt = max(0, nimag_qpt - 1)

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

def ibrav_to_cell(system):
    """
    Convert a value of ibrav to a cell. Any unspecified lattice dimension
    is set to 0.0, but will not necessarily raise an error. Also return the
    lattice parameter.

    Parameters
    ----------
    system : dict
        The &SYSTEM section of the input file, containing the 'ibrav' setting,
        and either celldm(1)..(6) or a, b, c, cosAB, cosAC, cosBC.

    Returns
    -------
    alat, cell : float, np.array
        Cell parameter in Angstrom, and
        The 3x3 array representation of the cell.

    Raises
    ------
    KeyError
        Raise an error if any required keys are missing.
    NotImplementedError
        Only a limited number of ibrav settings can be parsed. An error
        is raised if the ibrav interpretation is not implemented.

    MALM: added ibrav=-13
    """
    from ase.units import create_units
    # Quantum ESPRESSO uses CODATA 2006 internally
    units = create_units('2006')

    if 'celldm(1)' in system and 'a' in system:
        raise KeyError('do not specify both celldm and a,b,c!')
    elif 'celldm(1)' in system:
        # celldm(x) in bohr
        alat = system['celldm(1)'] * units['Bohr']
        b_over_a = system.get('celldm(2)', 0.0)
        c_over_a = system.get('celldm(3)', 0.0)
        cosab = system.get('celldm(4)', 0.0)
        cosac = system.get('celldm(5)', 0.0)
        cosbc = 0.0
        if system['ibrav'] == 14:
            cosbc = system.get('celldm(4)', 0.0)
            cosac = system.get('celldm(5)', 0.0)
            cosab = system.get('celldm(6)', 0.0)
    elif 'a' in system:
        # a, b, c, cosAB, cosAC, cosBC in Angstrom
        alat = system['a']
        b_over_a = system.get('b', 0.0) / alat
        c_over_a = system.get('c', 0.0) / alat
        cosab = system.get('cosab', 0.0)
        cosac = system.get('cosac', 0.0)
        cosbc = system.get('cosbc', 0.0)
    else:
        raise KeyError("Missing celldm(1) or a cell parameter.")

    if system['ibrav'] == 1:
        cell = np.identity(3) * alat
    elif system['ibrav'] == 2:
        cell = np.array([[-1.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0],
                         [-1.0, 1.0, 0.0]]) * (alat / 2)
    elif system['ibrav'] == 3:
        cell = np.array([[1.0, 1.0, 1.0],
                         [-1.0, 1.0, 1.0],
                         [-1.0, -1.0, 1.0]]) * (alat / 2)
    elif system['ibrav'] == -3:
        cell = np.array([[-1.0, 1.0, 1.0],
                         [1.0, -1.0, 1.0],
                         [1.0, 1.0, -1.0]]) * (alat / 2)
    elif system['ibrav'] == 4:
        cell = np.array([[1.0, 0.0, 0.0],
                         [-0.5, 0.5 * 3**0.5, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 5:
        tx = ((1.0 - cosab) / 2.0)**0.5
        ty = ((1.0 - cosab) / 6.0)**0.5
        tz = ((1 + 2 * cosab) / 3.0)**0.5
        cell = np.array([[tx, -ty, tz],
                         [0, 2 * ty, tz],
                         [-tx, -ty, tz]]) * alat
    elif system['ibrav'] == -5:
        ty = ((1.0 - cosab) / 6.0)**0.5
        tz = ((1 + 2 * cosab) / 3.0)**0.5
        a_prime = alat / 3**0.5
        u = tz - 2 * 2**0.5 * ty
        v = tz + 2**0.5 * ty
        cell = np.array([[u, v, v],
                         [v, u, v],
                         [v, v, u]]) * a_prime
    elif system['ibrav'] == 6:
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 7:
        cell = np.array([[1.0, -1.0, c_over_a],
                         [1.0, 1.0, c_over_a],
                         [-1.0, -1.0, c_over_a]]) * (alat / 2)
    elif system['ibrav'] == 8:
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, b_over_a, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 9:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [-1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -9:
        cell = np.array([[1.0 / 2.0, -b_over_a / 2.0, 0.0],
                         [1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == 10:
        cell = np.array([[1.0 / 2.0, 0.0, c_over_a / 2.0],
                         [1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [0.0, b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 11:
        cell = np.array([[1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                         [-1.0 / 2.0, b_over_a / 2.0, c_over_a / 2.0],
                         [-1.0 / 2.0, -b_over_a / 2.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == 12:
        sinab = (1.0 - cosab**2)**0.5
        cell = np.array([[1.0, 0.0, 0.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         [0.0, 0.0, c_over_a]]) * alat
    elif system['ibrav'] == -12:
        sinac = (1.0 - cosac**2)**0.5
        cell = np.array([[1.0, 0.0, 0.0],
                         [0.0, b_over_a, 0.0],
                         [c_over_a * cosac, 0.0, c_over_a * sinac]]) * alat
    elif system['ibrav'] == 13:
        sinab = (1.0 - cosab**2)**0.5
        cell = np.array([[1.0 / 2.0, 0.0, -c_over_a / 2.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         [1.0 / 2.0, 0.0, c_over_a / 2.0]]) * alat
    elif system['ibrav'] == -13:
        sinac = (1.0 - cosac**2)**0.5
        cell = np.array([[ 1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [-1.0 / 2.0, b_over_a / 2.0, 0.0],
                         [c_over_a * cosac, 0.0, c_over_a * sinac]]) * alat
    elif system['ibrav'] == 14:
        sinab = (1.0 - cosab**2)**0.5
        v3 = [c_over_a * cosac,
              c_over_a * (cosbc - cosac * cosab) / sinab,
              c_over_a * ((1 + 2 * cosbc * cosac * cosab
                           - cosbc**2 - cosac**2 - cosab**2)**0.5) / sinab]
        cell = np.array([[1.0, 0.0, 0.0],
                         [b_over_a * cosab, b_over_a * sinab, 0.0],
                         v3]) * alat
    else:
        raise NotImplementedError('ibrav = {0} is not implemented'
                                  ''.format(system['ibrav']))

    return alat, cell

