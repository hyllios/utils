# SIESTA 5.4.1 returns a wrong Hamiltonian at some MPI rank counts

Reproducer for the bug that made every `MgCoNi2O4` calculation in this repo
garbage (see `waw/interfaces/siesta/io.py::check_scf_consistency`). Kept here
so the finding survives the run directories, which are gitignored, and so the
upstream report below can be filed verbatim.

`ni4o4.fdf` is rocksalt NiO in an 8-atom cubic cell, non-spin-polarised,
2 species, 128 orbitals. Needs `Ni.psml` and `O.psml` (PseudoDojo
nc-fr-04/standard; the copies this repo uses are documented in
`workflows/PSEUDOS.md`) in the working directory.

## The one-line check (seconds, no SCF needed)

The corruption is visible in the FIRST SCF iteration, i.e. in the Hamiltonian
built from the superposition of atomic densities -- a deterministic quantity
with no mixing history in it. Run the same input at 4 and 8 ranks and compare
the first-diagonalisation band-structure energy:

    module load siesta/5.4.1-gcc-13.2.0-dfgltdg
    for np in 4 8; do
      mkdir -p np$np && cp ni4o4.fdf Ni.psml O.psml np$np/
      ( cd np$np && mpirun --mca pml ob1 --bind-to none -np $np siesta \
            < ni4o4.fdf > out )
    done
    grep -H "siesta: Ebs" np*/out          # <- differs by 1796 eV
    grep -H "^ *scf: *1 " np*/out

Measured here:

    np   BlockSize    Ebs (1st diag)   scf iteration 1: Eharris / E_KS / dDmax
     4   33 (deflt)      -3402.4315    -19998.9010  -20026.6106     0.93
     8   17 (deflt)      -5198.6958    -19792.1633  -20475.8482   669.16
     8   16 (forced)     -3402.4315    -19998.9010  -20026.6106     0.93

np=4 and np=8-with-BlockSize-16 agree to every printed digit; the np=8 default
does not. Adding the single line `BlockSize 16` to the np=8 input is the whole
difference. Note also the absurd first dDmax (669 vs 0.93) -- the initial
density matrix is not even normalised.

## The converged runs

Letting them run to self-consistency (this input, non-spin-polarised):

    np   BlockSize    E_KS (SCF)     summary Total    gap       max|F|
     4   33 (deflt)   -20049.7012      -20049.7012    0.00      0.000
     8   16 (forced)  -20049.7012      -20049.7012    0.00      0.000
     8   17 (deflt)   ~ -21253.2       (oscillates, will not converge)

and, from the spin-polarised production runs of the same cell (`Spin
polarized` with all-Ni moments +2, everything else identical):

    np   BlockSize     E_KS (SCF)   summary Total    gap      max|F| (eV/Ang)
     1   64 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     2   65 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     4   33 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     5   26 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     6   22 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     7   19 (deflt)   -20050.2421     -20050.2421     0.00        0.000
     8   17 (deflt)   -21259.0904     -18949.6860  2309.40      175.417
     8   16 (forced)  -20050.2421     -20050.2421     0.00        0.000
     8   15 (forced)  -20050.2421     -20050.2421     0.00        0.000

The np=8 default result is wrong in four independent ways at once:

1. `Eharris` and `Etot` agree line by line during the SCF, but the post-SCF
   summary energy is 2309 eV from the converged `E_KS`. The run still prints
   `SCF Convergence by DM`, so nothing warns.
2. E_KS lands 1209 eV *below* 4x the 2-atom NiO answer, i.e. below the
   variational bound.
3. Forces reach 175 eV/Ang on a structure where symmetry forbids any force,
   and the pressure comes out ~1.5e5 GPa.
4. The Mulliken residual piles onto whichever atom is *last* in the
   coordinate block (+11 e on O8 here); permuting the coordinate list moves
   it and changes E_KS by ~1200 eV.

## Scope

* Not the eigensolver: `Diag.Algorithm D&C`, `MRRR` and
  `Diag.ParallelOverK T` all give the identical wrong number, and the last of
  those diagonalises each k serially. The corruption is present in the first
  diagonalisation of the atomic-superposition H (see `Ebs` above), so it is
  in the setup or distribution of H/S, not in the SCF or the solver.
* Not spin: `Spin non-polarized` (as shipped here) and `Spin polarized` break
  identically. Not `Diag.ProcessorY`, not the mesh cutoff (456-1824 Ry),
  not `ElectronicTemperature` (0.025-0.3 eV).
* `np >= 8` is necessary but not sufficient, and we could find no predicate
  for which combinations break. At 128 orbitals np8/BS17 is broken while
  BS15 and BS16 are clean; at 118 orbitals (a 4-species oxide, same cell
  shape) BS15 *is* broken and BS 2 and 6 are clean, as is np16/BS20 at 320
  orbitals. Divisibility of n_orbitals by BlockSize, parity of BlockSize, and
  the "Orbital distribution balance" SIESTA prints all fail to separate the
  two populations.

## Detection

`|last scf E_KS - summary "siesta: Total"|`. Across ~70 runs on this build it
is exactly 0.00 for every healthy run and 830-2310 eV for every corrupt one.
`waw.interfaces.siesta.io.check_scf_consistency` implements it and
`run_siesta` calls it after every run.

## Environment

    siesta 5.4.1, module siesta/5.4.1-gcc-13.2.0-dfgltdg (spack)
    GNU 13.2.0, flags -fallow-argument-mismatch -O3 -march=native
    OpenMPI 5.0.3 (openmpi/5.0.3-gcc-11.4.1-vc5rpuu), NetCDF-4 MPI-IO,
    internal ELPA, PEXSI available
    linux-almalinux9-zen4, 192 cores
