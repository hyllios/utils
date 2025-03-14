#!/usr/bin/env python

import sys, os, glob, random
import numpy as np
import phonopy, yaml, ase
from ase.io import read as ase_read
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

# programs starts here
models = ("pbe", "pbesol")

ymls = glob.glob(sys.argv[1])

for yml in ymls:
    print(f"Running: {yml}")
    name = os.path.basename(yml)

    for model in models:
        if not os.path.isfile(f"{yml}/{model}.yaml"):
            continue

        if os.path.isfile(f"reference/{model}/{name}.yaml"):
            continue

        # load reference structure
        ph_ref = phonopy.load(f"{yml}/{model}.yaml")

        ph_ref.nac_params = None # We do not want the response to the e-field
        ph_ref.produce_force_constants()
        ph_ref.symmetrize_force_constants()
        ph_ref.save(filename=f"reference/{model}/{name}.yaml", settings={'force_constants': True})

        # calculate thermal properties at 300 K
        print("... computing thermal properties")
        ph_ref.init_mesh()
        ph_ref.run_mesh()
        ph_ref.run_thermal_properties(temperatures=(0, 75, 150, 300, 600))
        res = ph_ref.get_thermal_properties_dict()

        commensurate_q = get_commensurate_points(ph_ref.supercell_matrix)
        phonon_freqs = np.array([ph_ref.get_frequencies(q) for q in commensurate_q])
        group_velocities = np.array([ph_ref.get_group_velocity_at_q(q) for q in commensurate_q])

        # we now add extra properties to the yaml
        print("... saving extra properties")
        with open(f"reference/{model}/{name}.yaml", 'r') as f:
            ph_e = yaml.load(f, yaml.FullLoader)

        if model == "pbe":
            ase_pbe = ase_read(f"{yml}/relax_1/vasprun.xml.bz2")
            ph_e["energy"] = float(ase_pbe.get_potential_energy())
        else:
            ph_e["energy"] = float('nan')
        ph_e["volume"] = float(ph_ref.unitcell.volume)
        ph_e["nsites"] = len(ph_ref.unitcell)

        ph_e["free_e"] = res['free_energy'].tolist()
        ph_e["entropy"] = res['entropy'].tolist()
        ph_e["heat_capacity"] = res['heat_capacity'].tolist()
        ph_e["temperatures"] = res['temperatures'].tolist()

        ph_e["phonon_freq"] = phonon_freqs.tolist()
        ph_e["group_velocities"] = group_velocities.tolist()

        with open(f"reference/{model}/{name}.yaml", 'w') as f:
            yaml.dump(ph_e, f)

print("Done!")
