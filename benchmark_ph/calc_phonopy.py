#!/usr/bin/env python

import sys, os, glob, random, yaml
import numpy as np
from ase import Atoms
from pathlib import Path

import phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

from umlip import umlip

def run_phonopy(ph_ref, mlip, distance=0.01, relax=True):
    # create ase structure
    ase_cell = Atoms(
        cell=ph_ref.unitcell.cell,
        symbols=ph_ref.unitcell.symbols,
        scaled_positions=ph_ref.unitcell.scaled_positions,
        pbc=True)

    # relax unit cell
    if relax:
        relaxed = mlip.relax_structure(ase_cell, fmax=0.005, check_cell=False, check_connected=False, fix_symmetry=True)
        if relaxed is None:
            return None
        ase_cell = relaxed[0]
    else:
        # get the energy from the reference
        relaxed = [ase_cell, None, None]

    # create phonopy structure
    ph_atoms = phonopy.structure.atoms.PhonopyAtoms(
        cell=ase_cell.get_cell(),
        scaled_positions=ase_cell.get_scaled_positions(),
        symbols=ase_cell.get_chemical_symbols()
        )
    ph_mlip = phonopy.Phonopy(ph_atoms, 
        supercell_matrix=ph_ref.supercell_matrix,
        primitive_matrix=ph_ref.primitive_matrix
        )

    # get forces in supercells
    forcesets = []
    ph_mlip.generate_displacements(distance=distance, is_diagonal=False)
    for supercell in ph_mlip.supercells_with_displacements:
        scell = Atoms(
            cell=supercell.cell,
            symbols=supercell.symbols,
            scaled_positions=supercell.scaled_positions,
            pbc=True)
        scell.calc = mlip.calculator

        forces = scell.get_forces()
        drift_force = forces.sum(axis = 0) 
        for force in forces:
            force -= drift_force / forces.shape[0]

        forcesets.append(forces)

    ph_mlip.forces = forcesets
    ph_mlip.produce_force_constants()
    ph_mlip.symmetrize_force_constants()

    return ph_mlip, relaxed

def get_params():
    from argparse import ArgumentParser

    parent_parser = ArgumentParser(usage="Benchmark phonopy phonons")
    parent_parser.add_argument('--model', type = str, default = "m3gnet",
        help='Model to run')
    parent_parser.add_argument('--ref', type = str, default = "",
        help='Folder with reference yaml files')
    parent_parser.add_argument('--dest', type = str, default = "",
        help='Destination folder')
    parent_parser.add_argument('--distance', type = float, default = 0.01,
        help='Displacement to use')
    parent_parser.add_argument('--relax', action='store_true')

    return parent_parser.parse_args()

# programs starts here
params = get_params()

# initialize mlip
mlip = umlip(model=params.model)

ymls = glob.glob(f"{params.ref}*.yaml.bz2")
random.shuffle(ymls)

if params.dest == "":
    dest = params.model
else:
    dest = params.dest

for yml in ymls:
    print(f"Running: {yml}")
    name = os.path.basename(yml).replace('.bz2', '')

    if os.path.isfile(f"{dest}/{name}"):
        continue

    # load reference structure
    ph_ref = phonopy.load(yml)

    r = run_phonopy(ph_ref, mlip, distance=params.distance, relax=params.relax)
    if r is None:
        print("Error: Relaxation failed")
        continue # relaxation failed

    ph_mlip, relaxed = r
    Path(dest).mkdir(parents=True, exist_ok=True)
    ph_mlip.save(filename=f"{dest}/{name}", settings={'force_constants': True})

    # calculate thermal properties at 300 K
    print("... computing thermal properties")
    ph_mlip.init_mesh()
    ph_mlip.run_mesh()
    ph_mlip.run_thermal_properties(temperatures=(0, 75, 150, 300, 600))
    res = ph_mlip.get_thermal_properties_dict()

    commensurate_q = get_commensurate_points(ph_mlip.supercell_matrix)
    phonon_freqs = np.array([ph_mlip.get_frequencies(q) for q in commensurate_q])
    group_velocities = np.array([ph_mlip.get_group_velocity_at_q(q) for q in commensurate_q])

    # we now add extra properties to the yaml
    print("... saving extra properties")
    with open(f"{dest}/{name}", 'r') as f:
        ph_e = yaml.load(f, yaml.FullLoader)

    ph_e["nsites"] = len(relaxed[0])
    ph_e["energy"] = relaxed[1]
    ph_e["volume"] = float(relaxed[0].get_volume())

    ph_e["free_e"] = res['free_energy'].tolist()
    ph_e["entropy"] = res['entropy'].tolist()
    ph_e["heat_capacity"] = res['heat_capacity'].tolist()
    ph_e["temperatures"] = res['temperatures'].tolist()

    ph_e["phonon_freq"] = phonon_freqs.tolist()
    ph_e["group_velocities"] = group_velocities.tolist()

    with open(f"{dest}/{name}", 'w') as f:
        yaml.dump(ph_e, f)

print("Done!")
