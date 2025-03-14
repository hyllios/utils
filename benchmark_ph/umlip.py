import sys, os
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from copy import deepcopy


DEG_IN_RAD = np.pi / 180
TOL = np.pi / 180 * 1e-2  # 0.01 degree converted to radians
Z = np.array([0.0, 0.0, 1.0])


class PlaneException(ValueError):
  pass


class umlip():
  def __init__(self, model=None, cell_mask=None) -> None:

    self.model = model
    self.calculator = None

    self.ase_adaptor = AseAtomsAdaptor()

    mask_3D = [True, True,  True,  True,  True, True]
    mask_2D = [True, True, False, False, False, True]
    mask_1D = [False, False, True, False, False, False]

    # for retro compatibility
    self.adjust_vacuum_1D = adjust_vacuum_1D
    self.adjust_vacuum_2D = adjust_vacuum_2D
    self.adjust_vacuum_3D = adjust_vacuum_3D
    self.is_connected = is_connected

    if cell_mask is None:
      cell_mask = "3d"

    if isinstance(cell_mask, str):
      if cell_mask.lower() == "3d":
        self.cell_mask = mask_3D
        self.adjust_vacuum = adjust_vacuum_3D
      elif cell_mask.lower() == "2d":
        self.cell_mask = mask_2D
        self.adjust_vacuum = adjust_vacuum_2D
      elif cell_mask.lower() == "1d":
        self.cell_mask = mask_1D
        self.adjust_vacuum = adjust_vacuum_1D
      else:
        print(f"cell_mask {cell_mask} is not recognized")
        sys.exit(1)
    else:
      self.cell_mask = cell_mask


  def init_calculator(self, model=None):
    import json

    if model is None:
      return None

    # read model definitions from file
    umlip_defs = os.path.join(os.path.dirname(__file__), "umlip_defs.json")
    with open(umlip_defs) as f:
       umlip = json.load(f)

    if not model in umlip:
      print(f"Model {model} not found!")
      sys.exit()

    model_arch = umlip[model]["arch"]
    model_path = umlip[model]["path"].replace("$HOME", os.environ["HOME"])
    
    device = "cpu"
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
      device = "cuda"

    if model_arch == "m3gnet":
      import tensorflow as tf
      from m3gnet.models import M3GNet, Potential, M3GNetCalculator

      # for m3gnet to run in serial
      tf.config.threading.set_intra_op_parallelism_threads(1)
      tf.config.threading.set_inter_op_parallelism_threads(1)

      m3gnet     = M3GNet.load(model_path)
      potential  = Potential(model=m3gnet)

      return M3GNetCalculator(potential=potential, stress_weight=0.01)

    elif model_arch == "mace":
      from mace.calculators import MACECalculator

      return MACECalculator(model_paths=model_path, device=device)

    elif model_arch == "orbital":
      from orb_models.forcefield import pretrained
      from orb_models.forcefield.calculator import ORBCalculator

      orbff = pretrained.orb_v2(weights_path=model_path, device=device)
      return ORBCalculator(orbff, device=device)

    elif model_arch == "fairchem":
      from fairchem.core import OCPCalculator

      return OCPCalculator(checkpoint_path=model_path, cpu=(device=="cpu"))
    
    elif model_arch == "sevenn":
      from sevenn.sevennet_calculator import SevenNetCalculator

      return SevenNetCalculator(model_path, device=device)

    elif model_arch == "chgnet":
      from chgnet.model.model import CHGNet
      from chgnet.model.dynamics import CHGNetCalculator

      chgnet = CHGNet.load()
      return CHGNetCalculator(model=chgnet, device='cuda')
        
    elif model_arch == "mattersim":
      from mattersim.forcefield import MatterSimCalculator

      return MatterSimCalculator(load_path=model_path, device=device)
    
    elif model_arch == "grace":
      from tensorpotential.calculator import TPCalculator

      return TPCalculator(model=model_path)

    elif model_arch == "deepmd":
      from deepmd.calculator import DP
      
      return DP(model=model_path)

    else:
      print(f"Model architecture '{model_arch}' not found")
      sys.exit()


  def relax_structure(self, struct, fmax=0.040, thick=7.5, relax_cell=True, check_cell=True, 
                      check_connected=True, fix_symmetry=False, pressure=None):
    from ase.stress import voigt_6_to_full_3x3_stress
    from ase.optimize import BFGS, FIRE
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.units import GPa

    # get the pressure in eV/A^3 units
    if pressure is not None:
      scalar_pressure = pressure * GPa
    else:
      scalar_pressure = 0.0

    # we need something to chek that the MLIP (especially MACE)
    # does not diverge to infinity
    def observe_forces(atoms):
      fmax2 = (atoms.get_forces()**2).sum(axis=1).max()
      if fmax2 > 10000.0**2:
        raise Exception("Error forces are unphysical")

    def set_calculator(ase_struct):
      ase_struct.set_calculator(self.calculator)
      if fix_symmetry:
        ase_struct.set_constraint(FixSymmetry(ase_struct))
      if relax_cell:
        ase_struct = FrechetCellFilter(ase_struct, mask=self.cell_mask, scalar_pressure=scalar_pressure)
      return ase_struct

    if self.calculator is None:
      self.calculator = self.init_calculator(model=self.model)

    if isinstance(struct, Atoms):
      ase_struct = struct.copy()
    else:
      ase_struct = self.ase_adaptor.get_atoms(struct)

    for i in range(15):
      if self.adjust_vacuum is not None:
        ase_struct = self.adjust_vacuum(ase_struct, thick=thick*1.5, check_cell=check_cell, check_connected=check_connected)
      if ase_struct is None: return None

      ase_struct = set_calculator(ase_struct)

      optimizer = FIRE(ase_struct)
      optimizer.insert_observer(observe_forces, atoms=optimizer.atoms)
      try:
        optimizer.run(fmax=fmax/2, steps=500)
      except Exception as error:
        print(error)
        return None

      forces = optimizer.atoms.get_forces()
      fmax2  = (forces**2).sum(axis=1).max()

      # remove all filters from ase_struct
      if relax_cell:
        ase_struct = ase_struct.atoms

      if fmax2 < (fmax/2)**2: break # converge to half the force required
      if fmax2 > 10000.0**2: return None # not converged

    if self.adjust_vacuum is not None:
      ase_struct = self.adjust_vacuum(ase_struct, thick=thick, check_cell=check_cell, check_connected=check_connected)
    if ase_struct is None: return None

    ase_struct = set_calculator(ase_struct)
    forces  = ase_struct.get_forces()
    fmax2  = (forces**2).sum(axis=1).max()
    if fmax2 > fmax**2: # Unconverged
      print(f"fmax: {fmax2} > {fmax**2}")
      return None

    # remove all filters from ase_struct
    if relax_cell:
      ase_struct = ase_struct.atoms

    # we have now to get the real forces
    forces  = ase_struct.get_forces()
    if isinstance(forces, np.ndarray):
      forces = forces.tolist() # get rid of numpy arrays

    stress = ase_struct.get_stress(voigt=False)
    # convert stress to vasp convention (negative of stress in kB)
    stress /= -0.1 * GPa

    if isinstance(stress, np.ndarray):
      stress = stress.tolist() # get rid of numpy arrays

    energy = ase_struct.get_potential_energy()

    if isinstance(energy, np.ndarray):
      energy = energy.tolist()[0] # get rid of numpy arrays
    if isinstance(energy, np.float64) or isinstance(energy, np.float32):
      energy = float(energy)

    if isinstance(struct, Atoms):
      struct = ase_struct
    else:
      struct = self.ase_adaptor.get_structure(ase_struct)
      struct.add_site_property("forces", forces)

      # chgnet creates a float32 property called final_magmom that can not be serialized
      # so we remove it
      if "final_magmom" in struct.site_properties:
        struct.remove_site_property("final_magmom")      

    # We removed all the filters above, so we have to add the PV term to the enthalpy here
    if pressure is not None:
      energy += scalar_pressure*ase_struct.get_volume()

    return struct, energy, stress


def check_struct_ab(structure: Atoms, tol=TOL):
  "Check the ab plane is normal to the z axis"
  a, b, _ = structure.get_cell()
  d = np.cross(a, b)
  d /= np.linalg.norm(d)
  # sin(a x b, z) < tol
  return np.linalg.norm(np.cross(d, Z)) < tol


def check_struct_c(structure: Atoms, tol=TOL):
  "Check the the c axis is parallel to the z axis"
  _, _, c = structure.get_cell()
  # sin(c, z) < tol
  return np.linalg.norm(np.cross(Z, c / np.linalg.norm(c))) < tol


def straighten(structure: Atoms):
  """Correct the orientation and orthogonality of the cell.

  Ensure:
  - a along x axis
  - c along z axis
  - b in in xy plane

  :param structure: a :class:`ase.Atoms`
  :returns: a new :class:`ase.Atoms` instance with fixed cell
  """
  a, b, c, alph, bet, gam = structure.get_cell_lengths_and_angles()
  alph, bet, gam = alph * DEG_IN_RAD, bet * DEG_IN_RAD, gam * DEG_IN_RAD

  src_mat = np.array(structure.get_cell())

  # dest_mat is a pure rotation of src_mat
  # it isn't deformed
  dest_mat = np.zeros((3, 3))
  dest_mat[0, 0] = a
  dest_mat[1, 0] = np.cos(gam) * b
  dest_mat[1, 1] = np.sin(gam) * b
  dest_mat[2, 0] = c * np.cos(bet)
  dest_mat[2, 1] = c / np.sin(gam) * (np.cos(alph) - np.cos(bet) * np.cos(gam))
  dest_mat[2, 2] = np.sqrt(c * c - dest_mat[2, 0] ** 2 - dest_mat[2, 1] ** 2)

  cart2src = np.linalg.inv(src_mat.T)
  dest2cart = dest_mat.T

  # new_lat has the same a and b as dest_mat but we straighten the c
  new_latt = dest_mat.copy()
  new_latt[2, :] = [0.0, 0.0, c]

  new_struct = structure.copy()
  for s in new_struct:
    s.position = (dest2cart @ cart2src @ s.position[:, None])[:, 0]

  new_struct.set_cell(new_latt, scale_atoms=False)

  return new_struct

def is_connected(ase_struct_, supercell=(2, 2, 1), scale=1.5):
  '''Check if a structure is fully connected.
  The number of connected components in the graph is the dimension of the nullspace
  of the Laplacian and the algebraic multiplicity of the 0 eigenvalue.
  https://en.wikipedia.org/wiki/Laplacian_matrix
  '''
  from ase.data import covalent_radii

  # remove any existing constraint from the structure
  ase_struct = ase_struct_.copy()
  ase_struct.set_constraint()

  # create the supercell
  atoms_obj = ase_struct * supercell

  n = len(atoms_obj)
  lap_m = np.zeros([n, n])

  numbers = atoms_obj.get_atomic_numbers()

  # create connectivity matrix
  for i in range(n-1):
    for j in range(i+1, n):
      dis = atoms_obj.get_distance(i, j, mic=True)
      if dis < scale*(covalent_radii[numbers[i]] + covalent_radii[numbers[j]]):
        lap_m[i][j] = -1
        lap_m[j][i] = -1

  for i in range(n):
    lap_m[i, i] = abs(sum(lap_m[i]))

  # Get eigenvalues and check how many zeros;
  eigenvalues = np.linalg.eigh(lap_m)[0]

  nzeros = sum([1 for i in eigenvalues if round(i, 5) == 0.0])
  return (nzeros == 1)


def adjust_vacuum_3D(ase_struct, thick=None, check_cell=True, check_connected=False, tol_angle=TOL):
  uc = ase_struct.get_cell()

  # Sanity check
  cell_params = uc.cellpar()
  if check_cell and (cell_params[0] > 30.0 or cell_params[1] > 30.0 or cell_params[2] > 30.0):
    print(f"Error: cell vector too large {cell_params[0:3]}")
    return None

  # check if structure is really two-dimensional
  if check_connected and not is_connected(ase_struct, supercell=(2, 2, 2)):
    print(f"Cell is not connected")
    return None

  return ase_struct


def adjust_vacuum_2D(ase_struct, thick=7.5, check_cell=True, check_connected=True, tol_angle=TOL):
  '''Adjust vacuum to 15 Angstrom. Makes the c axis orthogonal
  to a and b. Returns None is structure is thicker than thick'''

  a, b, c, _, _, _ = ase_struct.get_cell_lengths_and_angles()

  # Sanity check
  if check_cell and (a > 30.0 or b > 30.0):
    print(f"Error: cell vector too large {(a, b, c)}")
    return None

  scaled_pos = ase_struct.get_scaled_positions()
  # centering everything properly before any change
  # 1. ensure that the bottom of the slab is in the middle of the cell
  #   This is important in case the slab is already split on both sides of a boundary.
  scaled_pos[:, 2] -= np.min(scaled_pos[:, 2]) + 0.5
  # 2. ensure that all atoms are in the cell
  scaled_pos[:, :] = np.remainder(scaled_pos[:, :], 1.0)
  # 3. center the slab in the middle of the cell
  scaled_pos[:, 2] += 0.5 - np.mean(scaled_pos[:, 2])

  ase_struct.set_scaled_positions(scaled_pos)

  # Check and fix title of ab and orthogonality of c
  if (
    not check_struct_ab(ase_struct, tol=tol_angle)
    or not check_struct_c(ase_struct, tol=tol_angle)
  ):
    ase_struct = straighten(ase_struct)

  zs = ase_struct.positions[:, 2]
  thickness = np.max(zs) - np.min(zs)

  # check if the resulting structure is "atomically thin"
  if (thick is not None) and thickness > thick:
    print(f"Error: thickness {thickness}")
    return None

  # increase vaccum
  ase_struct.cell[2, 2] = thickness + 15.0

  # center again after the change of axis
  scaled_pos = ase_struct.get_scaled_positions()
  scaled_pos[:, 2] += 0.5 - np.mean(scaled_pos[:, 2])
  ase_struct.set_scaled_positions(scaled_pos)

  # check if structure is really two-dimensional
  if check_connected and not is_connected(ase_struct, supercell=(2, 2, 1)):
    print(f"Cell is not connected")
    return None

  return ase_struct


def adjust_vacuum_1D(ase_struct, thick=None, check_cell=True, check_connected=False, tol_angle=TOL):
  '''Adjust vacuum to 15 Angstrom and make c axis orthogonal
  to a and b.'''

  uc = ase_struct.get_cell()

  # Sanity check
  cell_params = uc.cellpar()
  if check_cell and cell_params[2] > 30.0:
    print(f"Error: cell vector too large {cell_params[2]}")
    return None

  # center positions around (x, y)=1/2
  pos = ase_struct.get_positions(wrap=True)

  # check if the resulting structure is "atomically thin"
  if thick is not None:
    if max(pos[:,0]) - min(pos[:,0]) > thick:
      print(f"Error: A-AXIS {max(pos[:,0]), min(pos[:,0])}")
      return None
    if max(pos[:,1]) - min(pos[:,1]) > thick:
      print(f"Error: B-AXIS {max(pos[:,1]), min(pos[:,1])}")
      return None

  pos[:, 0] += -np.mean(pos[:, 0]) + 0.5*(uc[0, 0] + uc[1, 0])
  pos[:, 1] += -np.mean(pos[:, 1]) + 0.5*(uc[0, 1] + uc[1, 1])
  ase_struct.set_positions(pos)

  # lattice
  xmean = np.mean(pos[:, 0])
  ymean = np.mean(pos[:, 1])
  rmax = 0.0
  for p in pos:
    r = np.sqrt((p[0] - xmean)**2 + (p[1] - ymean)**2)
    if r > rmax: rmax = r

  a = 15 + rmax; c = uc[2, 2]
  uc = np.array([[a, 0., 0.],
                  [-a/2., a*np.sqrt(3.)/2., 0.],
                  [0., 0., c]])
  ase_struct.set_cell(uc, scale_atoms=False)

  # recenter again
  pos = ase_struct.get_positions(wrap=True)
  pos[:, 0] += -np.mean(pos[:, 0]) + 0.5*(uc[0, 0] + uc[1, 0])
  pos[:, 1] += -np.mean(pos[:, 1]) + 0.5*(uc[0, 1] + uc[1, 1])

  ase_struct.set_positions(pos)

  # check if structure is really one-dimensional
  if check_connected and not is_connected(ase_struct, supercell=(1, 1, 2)):
    return None

  return ase_struct


