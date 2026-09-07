`out_ph/`: raw `ph.x` DFPT output for fcc Al at q=Gamma (q-point index 1
of a 6x6x6 mesh), `ibrav=2, celldm(1)=4.0495 Ang` (this project's own
electron-phonon-coupling pipeline, `scratchpad/al_elph_test/full_run.py`
-- a **plain** phonon run, `fildvscf='dvscf'`, no `electron_phonon`/
`dvscf_star` set).

  - `_ph0/al.dvscf1`: the single irrep at q=Gamma (all 3 acoustic modes,
    O_h symmetry groups them into one 3-dimensional representation),
    raw self-consistent-potential variation on the (24,24,24) FFT grid,
    in the irrep's own symmetry-adapted pattern basis (NOT bare
    Cartesian atomic displacements).
  - `_ph0/al.phsave/patterns.1.xml`: the pattern metadata (number of
    perturbations + `DISPLACEMENT_PATTERN` rotation matrix) needed to
    convert `al.dvscf1` to the Cartesian atomic-displacement basis.

Used by `tests/test_analysis_elph.py` to validate
`interfaces.quantum_espresso.dvscf_io.read_dvscf`/`read_patterns`
against a genuine, non-trivial physics identity: for this single-atom
cell, a "Cartesian displacement" mode at q=Gamma IS a rigid translation
of the whole crystal, so translational invariance requires the
resulting `dv_cart` to integrate to exactly zero over the cell (checked
to ~1e-15, far tighter than DFPT's own `tr2_ph=1e-14` convergence
threshold would explain by coincidence).

This bypasses `ph.x`'s own `electron_phonon='Wannier'`/`elph_mat`
feature entirely (crashes with a heap-corruption error in QE 7.3.1,
confirmed serial and parallel) -- see `dvscf_io`'s module docstring.
