# waw.utils

Self-contained tools that sit on top of the library: each subpackage solves one
problem end to end and carries its own CLI. They may use anything in `waw`
(units, analysis helpers); nothing in the rest of `waw` imports them, so they
stay optional.

## `eliashberg` — band-resolved isotropic Eliashberg solver

Takes an Eliashberg spectral function — a single spectrum α²F(ω) or a
band-resolved matrix α²F_ij(ω) — plus a Coulomb pseudopotential μ\* (scalar or
matrix) and returns **Tc from the linearized equations**.

```python
from waw.utils.eliashberg import tc_linearized

res = tc_linearized(a2f, omega, mu_star=0.11, omega_c=omega_c)
print(res.tc, 'K', res.lambda_matrix)
```

`a2f` is `(n_omega,)` or `(nb, nb, n_omega)` and `omega` is in Hartree — the
footing `waw.analysis.elph.alpha2f` and `alpha2f_matrix` already produce, so
their output feeds straight in.

From the command line, reading either an `.npz` holding `omega`/`a2f` or a
columnar text file (omega, then the α²F blocks in row-major (i, j) order):

```
python -m waw.utils.eliashberg a2f.npz --mu 0.11 --omega-c 500meV
python -m waw.utils.eliashberg a2f.dat --mu 0.11 --unit cm-1
python -m waw.utils.eliashberg a2f.npz --mu 0.11 --scan 0.08 0.16 5
```

### Why linearized

As T → Tc the gap vanishes and the coupled equations collapse to a linear
eigenvalue problem, Δ = M(T)Δ. Tc is then where the leading eigenvalue ρ(T)
crosses 1 — a sharp, well-conditioned root, monotonic in T. Extracting Tc from
the full equations instead means watching Δ(T) → 0 exactly where the nonlinear
iteration converges slowest, which is why the reference implementation has to
fit a model curve to Δ(T).

### Validation

Cross-checked against an independent Fortran solver of the *full* nonlinear
equations on two systems (μ\* = 0.11, ω_c = 500 meV):

| system | bands | reference Tc | this solver | difference |
|---|---|---|---|---|
| CaC₆ | 1 | 14.149 K | 14.099 K | −0.35 % |
| MgB₂ | 2 (σ/π) | 32.011 K | 31.907 K | −0.33 % |

Its α²F for both, in atomic units, is archived with those settings in
`tests/data/eliashberg/*.npz`, so the comparison is reproducible without the
other code.

Both land slightly *below* the reference by the same amount, which is the
expected signature of its gap-curve extrapolation overshooting — a discrepancy
differing in sign or size between the two systems would instead point at a
formulation error. λ agrees with the reference's printed value to 4×10⁻¹⁶ when
its rectangle quadrature is reproduced (3.4×10⁻⁶ with the trapezoid rule used
by default). MgB₂ comes out s++ with λ_σσ = 0.812 dominant, as it should.

Also verified internally: power iteration matches full diagonalisation to
1×10⁻¹¹ (the kernel is *not* symmetric, so the dominant eigenvalue is not
variational); ρ(T) is monotonic through 1; Tc falls monotonically with μ\* and
reaches 0 when μ\* overwhelms the coupling; and Tc is converged in the Matsubara
count to 0.01 % at the default settings.

### Conventions

- **Frequencies in Hartree**, temperatures out in Kelvin, matching
  `waw/core` and `waw/analysis`. The CLI converts at the file boundary via
  `--unit`.
- **Band index order**: `a2f[i, j]` couples band *i* (solved for) to band *j*
  (summed over), so α²F_ij carries the DOS of band *j*. `isotropic_average`
  therefore weights on the row index. This is the reference code's convention;
  transposing it changes the answer on an asymmetric matrix like MgB₂'s
  (λ_σπ = 0.199 vs λ_πσ = 0.142).
- **The Coulomb cutoff defaults to 10 × the maximum phonon frequency.** That
  is the convention [EPW's documentation](https://docs.epw-code.org/Inputs/Inputs.html)
  calls common, and its own Pb
  [tutorial](https://docs.epw-code.org/tutorials/tutorial_04/index.html) uses
  exactly that (phonons to ~10 meV, `wscut` = 100 meV); its MgB₂ example is
  ~5×, and EPW's hard default is 1 eV. The literature also uses 3× (widely, in
  the Heaviside-θ form) up to 10×.

- **μ\* is cutoff-dependent**, and the spread of conventions above is not
  harmless: on MgB₂ at fixed μ\* = 0.11, Tc runs from 30.4 K at 3× to 34.1 K
  at 10×. A μ\* is only meaningful together with the `omega_c` it was quoted
  at. `rescale_mu_star` moves one between cutoffs by the Morel–Anderson law,
  1/μ\*(ω₂) = 1/μ\*(ω₁) + ln(ω₁/ω₂), which removes most of the drift (CaC₆
  +8.5 % → −0.5 % over a 6.7× range; MgB₂ +6.9 % → +3.4 %) but not all of it,
  the form being approximate — more so for a multiband system carrying one
  scalar μ\* in every channel.

- **The Matsubara extent is a separate knob** (default 40 × the phonon
  maximum, never below 4 × `omega_c`). EPW uses a single `wscut` for both, so
  raising its cutoff to converge the phonon sum also redefines μ\*; keeping
  them apart lets the numerics converge without touching the meaning of μ\*.
  Pass `omega_max_matsubara=omega_c` for strict EPW parity.

- The Coulomb sum gives the last Matsubara point a fractional weight so the
  effective cutoff sits exactly at `omega_c` rather than jumping between grid
  points as T varies.
- ω must be strictly positive — α²F/ω diverges at 0, so drop the zero point
  (and any acoustic frequencies below an `eps_acoustic`-style cutoff) first.

### Not yet implemented

The **full nonlinear equations**: Δ_n and Z_n at finite temperature, hence
Δ(T), the two distinct MgB₂ gaps, and Padé continuation to the real axis for
the spectral gap. The kernels in `kernels.py` are shared with that step, so it
builds on this rather than replacing it.

## `scdft` — density-functional theory for superconductors

The band-resolved isotropic SCDFT gap equation with the Sanna–Pellegrini–Gross
exchange–correlation functional ([Phys. Rev. Lett. 125, 057001
(2020)](https://doi.org/10.1103/PhysRevLett.125.057001)), taking the same
inputs as the Eliashberg module — α²F (single or band matrix) plus a
dimensionless Coulomb parameter.

```python
from waw.utils.scdft import tc_scdft, solve_delta_s

tc  = tc_scdft(omega, a2f, mu=0.1)                    # linearized
res = solve_delta_s(omega, a2f, 0.1, kT)              # self-consistent Δ_s(ξ)
```

Δ_s is the Kohn–Sham **pairing potential**, not an excitation gap: the paper's
own exact result is that it has a local minimum at E_F rather than a maximum.

Two implementation notes, because the paper's formulas cannot be used
literally. Their Eq. (12) carries `exp(beta*xi)` with `beta*xi` ~ 400 at 30 K
and ξ = 1 eV, so it overflows before any physics; multiplying the Fermi and Bose
prefactors in gives a bounded equivalent. And their Eq. (13) is, term for term,
the second-order **Newton divided difference** of the Fermi function over
`{ξ−ω, E, γ}` — which is why its three apparent poles are all removable, and
computing it as a divided difference gets that right by construction.

**Validated**: the special functions to 1e-11 against the printed formulas
(including correct limits where those are 0/0); Tc against Eliashberg for the
paper's Einstein model at λ ≥ 1 (within 2 %); the Fig. 1 local minimum at E_F.
**Open**, and stated in `solver.py`'s docstring rather than hidden: weak-coupling
Tc runs high (65 % at λ = 0.3), Δ_s(E_F) does not vanish as T → 0 as their
Fig. 1 shows, Z(ξ→0) comes out 0.6π·λ rather than λ, and their Eq. (14) gap
function is not implemented.
