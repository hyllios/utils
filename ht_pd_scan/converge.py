import os, sys
from ase import *

def kpoints_max(structure, kppa=1000, return_kgrid=False):
    """Returns the max number of kpoints allowed during the
    kpoints convergence test. Based on the routine used in
    pymatgen.

    "struture" must be an ase.atoms object.

    """
    from math import sqrt, ceil

    if(os.path.exists('in.kpt')):    
        f = open('in.kpt', 'r')
        kptr = f.readline().split()
        f.close()

        if int(kptr[0]) != -1:
          return int(kptr[0]), int(kptr[1]), int(kptr[2])

    rcell = structure.get_reciprocal_cell()
    lengths = [sqrt(sum(map(lambda y: y**2, rcell[i]))) \
               for i in xrange(3)]
    ngrid = kppa / structure.get_number_of_atoms()
    mult = (ngrid / (lengths[0] * lengths[1] * lengths[2])) ** (1.0/3.0)
    num_div = [int(round((lengths[i] * mult))) for i in xrange(3)]
    num_div = [i if i > 0 else 1 for i in num_div]
    if return_kgrid:
        return num_div[0], num_div[1], num_div[2]
    else:
        return num_div[0]*num_div[1]*num_div[2]

def converge_kpoints(atoms, fh=sys.stdout, tol=1e-3*27.211, vasp_params={}, kppa=1000):
    """converge the k-point sampling"""

    from math import sqrt, ceil
    from ase.calculators.vasp import Vasp

    if(os.path.exists('in.kpt')):    
        f = open('in.kpt', 'r')
        kptr = f.readline().split()
        f.close()

        if int(kptr[0]) != -1:
          return int(kptr[0]), int(kptr[1]), int(kptr[2])
        else: # return MP default
	  return kpoints_max(atoms, kppa=1000, return_kgrid=True)

    rcell = atoms.get_reciprocal_cell()
    vol   = 1.0/atoms.get_volume()
    nat   = atoms.get_number_of_atoms()
    
    # lets get the cell ratios
    a1 = sqrt(rcell[0][0]**2 + rcell[0][1]**2 + rcell[0][2]**2)
    a2 = sqrt(rcell[1][0]**2 + rcell[1][1]**2 + rcell[1][2]**2)
    a3 = sqrt(rcell[2][0]**2 + rcell[2][1]**2 + rcell[2][2]**2)

    if fh != None:
        print >>fh, "==== Convergence with respect to k-points:"

    kpt = 1
    E1, E2 = 0.0, 0.0
    kpt_1  = [0, 0, 0]
    kpt_2  = [0, 0, 0]

    kpt_max = kpoints_max(atoms, kppa=kppa)
    
    while(True):
        if kpt == 1:
            kpt_new = [1, 1, 1]
        else:
            factor  = pow(kpt/vol, 1.0/3.0)
            kpt_new = [int(max(ceil(factor*a1), 1)), int(max(ceil(factor*a2), 1)), int(max(ceil(factor*a3), 1))]

        if kpt_new != kpt_1:
            calc = Vasp(kpts=(kpt_new[0], kpt_new[1], kpt_new[2]), ediff=1E-05, **vasp_params)
            atoms.set_calculator(calc)
            #try:  # sometimes vasp crashes for some k-point sets
            E_new = atoms.get_potential_energy() / nat

            os.remove('WAVECAR')

            if fh != None:
                fh.write("target = " + str(kpt) + "\t")
                fh.write("test = " + str(kpt_new[0]) + "x" + str(kpt_new[1]) + "x" + str(kpt_new[2]) + " = " +
                         str(kpt_new[0]*kpt_new[1]*kpt_new[2]) + "\tEnergy = " +
                         str(E_new) + "\n")

            converged = ((E1 != 0.0) and (abs(E_new - E1) < tol) and (abs(E_new - E2) < tol))
            if converged or 2*kpt > kpt_max:
                if not converged:
                    if abs(E_new - E1) < tol:
                        kpt_2 = kpt_1
                    else:
                        kpt_2 = kpt_new
                if fh != None:
                    fh.write("\nConverged : " + str(kpt_2[0]) + 'x' + str(kpt_2[1]) + 'x' + str(kpt_2[2]) + '\n')
                break
            #except:
            #    pass

            kpt_2 = kpt_1[:]
            kpt_1 = kpt_new[:]

            E2 = E1
            E1 = E_new
    
        kpt = kpt*2
        if(kpt > 10000):
          fh.write("\nToo many k-points. Aborting")
          sys.exit("Too many k-points. Aborting")

    calc.clean()

    # make sure that we have at least the MP k-points
    kpt_min = kpoints_max(atoms, kppa=1000, return_kgrid=True)
    if(kpt_min[0] > kpt_2[0] or kpt_min[1] > kpt_2[1] or kpt_min[2] > kpt_2[2]):
        fh.write("\nUsing MP : " + str(kpt_min[0]) + 'x' + str(kpt_min[1]) + 'x' + str(kpt_min[2]) + '\n')
        return kpt_min[0], kpt_min[1], kpt_min[2]
    else:
        return kpt_2[0], kpt_2[1], kpt_2[2]


def read_file(in_file):
    """reads a file and calculates the primitive unit cell"""

    import os
    import subprocess
    from copy import copy
    from ase import io
    import numpy as np

    basename, extension = os.path.splitext(in_file)
    basename = os.path.basename(basename)
    if (basename == 'POSCAR') or (basename == 'CONTCAR'):
        form = 'vasp'
    elif extension == '.ascii':
        form = 'v_sim'
    else:
        form = extension[1:]

    # generate the primitive unit cell
    if form == 'cif':
        cifReplace = {
            "data_ ": "data_",
            "_space_group_symop_operation_xyz": "_symmetry_equiv_pos_as_xyz",
            " 0.33333 ": " 0.333333333333333 ",
            " 0.66667 ": " 0.666666666666667 ",
            # now for the H-M symbols
            "P 2/m 2/m 2/m": "Pmmm", # 47
            "P 21/m 2/m 2/a": "Pmma", # 51
            "P 2/n 21/n 2/a": "Pnna", # 52
            "P 2/m 2/n 21/a": "Pmna", # 53
            "P 21/c 2/c 2/a": "Pcca", # 54
            "P 21/b 21/a 2/m": "Pbam", # 55
            "P 2/b 21/c 21/m": "Pbcm", # 57
            "P 21/n 21/n 2/m": "Pnnm", # 58
            "P 21/m 2/n 21/m (origin choice 1)": "Pmnm", # 59
            "P 21/b 2/c 21/n": "Pbcn", # 60
            "P 21/n 21/m 21/a": "Pnma", # 62
            "C 2/m 2/c 21/m": "Cmcm", # 63
            "C 2/m 2/c 21/a": "Cmca", # 64
            "C 2/m 2/m 2/m": "Cmmm", # 65
            "C 2/c 2/c 2/m": "Cccm", # 66
            "C 2/m 2/m 2/a":  "Cmma", # 67
            "B 2/b 2/a 2/b (origin choice 1)": "Bbab", # 68
            "F 2/m 2/m 2/m": "Fmmm", # 69
            "F 2/d 2/d 2/d (origin choice 1)": "Fddd", # 70
            "I 2/m 2/m 2/m": "Immm", # 71
            "I 2/b 2/a 2/m": "Ibam", # 72
            "I 21/m 21/m 21/a": "Imma", # 74
            "P 4/m 2/m 2/m": "P4/mmm", # 123
            "P 4/m 21/b 2/m": "P4/mbm", # 127
            "P 4/n 21/m 2/m (origin choice 1)": "P4/nmm", # 129
            "P 42/m 2/m 2/c": "P42/mmc", # 131
            "P 42/m 21/b 2/c": "P42/mbc", # 135
            "P 42/m 21/n 2/m": "P42/mnm", # 136
            "I 4/m 2/m 2/m": "I4/mmm", # 139
            "I 4/m 2/c 2/m": "I4/mcm", # 140
            "I 41/a 2/m 2/d (origin choice 1)" : "I41/amd", # 141
            "R 3 (hexagonal axes)": "R3", # 146
            "R -3 (hexagonal axes)": "R-3", # 148
            "R 3 2 (hexagonal axes)": "R32", # 155
            "R 3 m (hexagonal axes)": "R3m", # 160
            "R 3 c (hexagonal axes)": "R3c", # 161
            "P -3 2/m 1": "P-3m1", # 164
            "R -3 2/m (hexagonal axes)": "R-3m", # 166
            "R -3 2/c (hexagonal axes)": "R-3c", # 167
            "P 6/m 2/m 2/m": "P6/mmm", # 191
            "P 63/m 2/m 2/c":  "P63/mmc", # 194
            "P 2/m -3": "Pm-3", # 200
            "P 4/m -3 2/m": "Pm-3m", # 221
            "P 42/m -3 2/n": "Pm-3n", # 223
            "F 4/m -3 2/m": "Fm-3m", # 225
            "F 41/d -3 2/m (origin choice 1)": "Fd-3m", # 227
            "I 4/m -3 2/m": "Im-3m", # 229
        }    

        # we do not trust the reading of cif files of ase so we use cif2cell
        # to convert them to abinit format first
        pid  = str(os.getpid())

        # do a bit of search and replace so that cif2cell can read cif
        infile = open(in_file)
        outfile = open("/tmp/" + pid + ".cif", 'w')

        for line in infile:
            for src, target in cifReplace.iteritems():
                line = line.replace(src, target)
            outfile.write(line)

        infile.close()
        outfile.close()

        try:
          subprocess.call("$MHM_BASE_DIR/python/cif2cell /tmp/" + pid + ".cif -p abinit -o /tmp/" + pid + ".abinit", shell=True)
          cell = io.read("/tmp/" + pid + ".abinit", format="abinit")
          os.remove("/tmp/" + pid + ".cif")
          os.remove("/tmp/" + pid + ".abinit")
	except:
	  cell = io.read(in_file, format='cif')
    else:
        cell = io.read(in_file, format=form)

    # check if volume is negative
    ucell = cell.get_cell()
    if np.linalg.det(ucell) < 0.0:
        cpos = cell.get_scaled_positions()
        tmp = copy(cpos[:, 1])
        cpos[:, 1] = copy(cpos[:, 2])
        cpos[:, 2] = copy(tmp)

        tmp = copy(ucell[1, :])
        ucell[1, :] = copy(ucell[2, :])
        ucell[2, :] = copy(tmp)

        symbols = cell.get_chemical_symbols()

        cell = Atoms(symbols=symbols, scaled_positions=cpos, cell=ucell)

    return cell

