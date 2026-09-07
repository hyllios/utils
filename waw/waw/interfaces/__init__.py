"""
waw.interfaces — front doors to the atomic-unit core.

- ``wannier90``: the legacy Wannier90 file interface (.win/.mmn/.amn/.eig/…),
  kept for compatibility.
- ``ase``: the new ASE-based interface (structures, k-meshes, an Espresso DFT
  driver), preferring numpy for on-disk data.

Each interface owns its physical-unit conversions (eV, Angstrom); the core
sees only atomic units.
"""
