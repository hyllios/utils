#!/usr/bin/env python

import os, sys
from autoph.pw_utils import read_a2Fdos

ELIASHBERGX = "/nfs/data-019/marques/software/bin/eliashberg.x"

def run_eliashberg(mdir, mus=0.1, factor_w=1.0, factor_a=1.0):

    a2Fdos = read_a2Fdos(mdir, fname="a2F.dos6")
    n_freq = sum([1 for i in a2Fdos["freqs"] if i > 0])

    # create eliash.in
    Ry2cmm1 = 13.605692*1000*8.06554
    fh = open("elias.in", "w")
    fh.write("# {} 1.0 1 cm-1 0\n# 1 1\n".format(n_freq))
    for f, a in zip(a2Fdos["freqs"], a2Fdos["total"]):
        f *= Ry2cmm1*factor_w
        if f < 10.0: # but a2F=0 for frequencies smaller than 10 cm-1
          a = 0.0
        a *= factor_a
        if f > 0:
            fh.write("{} {}\n".format(f, a))
    fh.close()

    input = """
    0.000001 0.3 .true.  ! convergence, mixing, boh
    1000    ! max number of self-consistent iteratuions 
    1000.   ! max freq of Matsubara points  wmax
    500     ! max number of Matsubara points 
    .true.  ! Mass Renormalization (Z) 
    .true. ! Compute the gap
    {}    ! mu*
    500     ! IMPORTANT CUTOFF FOR COULOMB INTERACTION
    -30     ! A) If A>0: A=number of Temps; If A<0: A= max number of temps and their choice is automatic
    -1      ! B) If A>0: B are the temperatures (as many lines as needed); &
                If A<0 and B<0 fully automatic choice of temps 
                If A<0 and B>0 B is the Tc to be used as guess for the automatic choice algorithm
    """.format(mus)

    fh = open("input", "w")
    fh.write(input)
    fh.close()

    os.system(ELIASHBERGX + " > eliashberg.out")

    fh = open("eliashberg.out")
    tc_line = fh.readlines()[-1]
    if tc_line.startswith(" estimated Tc :"):
        tc = float(tc_line.split()[-1])
    else:
        tc = 0

    return tc

tc = run_eliashberg(sys.argv[1], mus=float(sys.argv[2]), factor_w=float(sys.argv[3]), factor_a=float(sys.argv[4]))
print("{:.3f}".format(tc))
