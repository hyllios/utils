`diam.ifc`: real-space interatomic force constants for diamond, generated
by `q2r.x` on a 3x3x3 q-mesh -- copied verbatim (unmodified) from Quantum
ESPRESSO's own bundled example suite,
`PHonon/examples/example19/reference/diam.ifc` (QE 7.3.1, GPL-licensed,
redistributed here for testing only, same convention as this project's
own PseudoDojo pseudopotential redistribution -- see workflows/PSEUDOS.md).

Used by `tests/test_analysis_phonon.py` as an independent, real
(non-synthetic) reference: diamond's zone-center optical phonon is a very
well known experimental quantity (~1332-1333 cm^-1, Raman-active), so a
correct Fourier-interpolation + mass-weighting + diagonalization pipeline
should reproduce it (this project's own PBE-DFPT value: 1337.9 cm^-1,
matching to <0.5%) -- a decisive end-to-end check that doesn't depend on
any of waw's own DFT runs.
