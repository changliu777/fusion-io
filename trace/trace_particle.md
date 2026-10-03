# `trace_particle`

`trace_particle` integrates one M3D-C1 guiding-center orbit or scans particle
motion periods over energy, magnetic moment or Lambda, and canonical toroidal
momentum.
The equations follow `particle.f90` and use RK4 integration in
`(R, phi, Z, v_parallel)`.

## Build

From the Fusion-IO source directory:

```bash
cmake -S . -B build
cmake --build build --target trace_particle -j4
```

The resulting executable is `build/trace/trace_particle`.

## Units

- `--energy`: keV
- `--mu`: keV/T
- `--lambda`: dimensionless `Lambda=mu*B0/E`
- `--pphi`: canonical `P_phi/q` in Wb
- `--R` and `--Z`: m
- `--phi` and `--phase`: degrees
- `--dt`: s
- `--mass`: proton-mass units
- `--charge`: elementary-charge units

The default mass and charge come from `ion_mass` and `z_ion` in the M3D-C1
file.

## Magnetic Field Parts

The orbit magnetic field can include the equilibrium field, perturbed field,
or both:

```bash
# Equilibrium only
trace_particle ... --equilibrium 1 --perturbed 0

# Perturbation only, amplified by 100
trace_particle ... --equilibrium 0 --perturbed 1 --factor 100

# Equilibrium plus an amplified perturbation
trace_particle ... --equilibrium 1 --perturbed 1 --factor 100
```

Both field parts are enabled by default. `--factor` multiplies only the
perturbed part through `FIO_LINEAR_SCALE`; `--scale` is an alias. The
`--phase` option applies the requested toroidal phase shift. Equilibrium
fields are still used to invert `P_phi/q` and locate low-field-side starting
points during a `P_phi/q` scan.

## Single Orbit

```bash
build/trace/trace_particle \
  -m3dc1 C1.h5 \
  --energy 20 \
  --mu 1 \
  --R 2.0 \
  --Z 0.0 \
  --sigma 1 \
  --dt 2e-9 \
  --steps 10000 \
  --output particle_orbit.out
```

If `R` or `Z` is omitted, the corresponding magnetic-axis coordinate is used.
`--sigma 1` selects positive initial `v_parallel`; `--sigma -1` selects the
opposite direction. `--sigma 0` traces both directions as a two-point scan.

The orbit table contains time, position, parallel and perpendicular velocity,
energy, magnetic moment, and magnetic-field amplitude. The final comment lines
contain `toroidal_period_s` and `poloidal_period_s`.

## Energy and Pitch Scan

Ranges are linearly spaced, include both endpoints, and use the final argument
as the number of points:

```bash
mpirun -n 4 build/trace/trace_particle \
  -m3dc1 C1.h5 \
  --energy 5 50 10 \
  --mu 0 5 8 \
  --R 2.0 \
  --Z 0.0 \
  --dt 2e-9 \
  --steps 20000 \
  --output particle_period_scan.out
```

Add `--sigma 0` to run every energy, magnetic-moment, and `P_phi/q` point
for both signs of `v_parallel`. The period table includes a `sigma` column,
and Poincare files retain their deterministic `out<scan_index>` names.

`--lambda` is mutually exclusive with `--mu` and accepts the same scalar or
three-value range syntax:

```bash
mpirun -n 4 build/trace/trace_particle \
  -m3dc1 C1.h5 \
  --energy 5 50 10 \
  --lambda 0 1.2 25 \
  --pphi 0.15 0.30 20 \
  --sigma 0 \
  --output particle_period_scan.out
```

For every energy and Lambda value, the code calculates

```text
mu_keV_per_T = lambda * energy_keV / B0_T
B0_T = abs(bzero) * b0_norm
```

where the Fusion-IO M3D-C1 source has already converted `b0_norm` to tesla.
This keeps an energy/Lambda scan rectangular.

A scalar can replace either range, for example:

```bash
--energy 20 --mu 0 5 8
```

MPI ranks process flattened scan points cyclically. Each particle orbit remains
serial because each RK4 step depends on the previous step. Rank 0 gathers,
orders, and writes the results. In single-orbit mode, only rank 0 traces and
writes the orbit.

## Period and Poincare Output

The output switches follow `trace`:

- `-qout 1` calculates and writes motion periods; it is enabled by default.
- `-pout 1` writes drift-orbit Poincare crossings; it is disabled by default.
- `-t TRANSITS` stops after the requested number of toroidal transits; zero
  disables the transit limit.
- `-a ANGLE` selects the first output plane in degrees.
- `-n NPLANES` uses equally spaced toroidal output planes.

Enable either output or both:

```bash
# Periods only
trace_particle ... -qout 1 -pout 0

# Poincare crossings for 100 toroidal transits
trace_particle ... -qout 0 -pout 1 -t 100 -a 0 -n 1

# Both, using planes at 0 and 180 degrees
trace_particle ... -qout 1 -pout 1 -a 0 -n 2
```

Poincare files are named `out<scan_index>`. Their columns are:

```text
phi_deg R_m Z_m theta_deg time_s
```

The first three columns match the `trace` `out*` convention and can be read by
`m3dc1.read_poincare` or `m3dc1.plot_poincare`:

```python
from m3dc1 import plot_poincare

fig, ax = plot_poincare(directory=".", color=True, iso=True)
```

With `-t`, each orbit stops after its unwrapped toroidal displacement reaches
`2*pi*TRANSITS`; `--steps` remains the maximum-step safety limit. Without
`-t`, Poincare output runs for all `--steps`, while a period-only scan stops
early after both periods are measured.

## Pphi Scan

For a `P_phi/q` scan, do not supply `R` or `Z`:

```bash
mpirun -n 8 build/trace/trace_particle \
  -m3dc1 C1.h5 \
  --pphi 0.15 0.30 10 \
  --energy 10 50 5 \
  --mu 0 5 6 \
  --dt 2e-9 \
  --steps 20000 \
  --output particle_period_scan.out
```

For each `(P_phi/q, E, mu)` combination, initialization fixes `Z=Z_axis` and
solves for the outermost valid `R>R_axis` satisfying

```text
P_phi/q = psi + (m/q) v_parallel I/B.
```

This selects the low-field-side midplane branch. Equilibrium `psi`, `I`, and
`B` are used for this canonical-momentum solve; orbit integration uses the
requested total field. The output includes the solved `initial_R_m` so the
initialization can be checked.

## Motion Periods

The poloidal period is the time required to return to the initial poloidal ray
with the same crossing direction. It is reported as `orbit_period_s`: the
transit period for a passing particle and the full bounce period for a trapped
particle.

An orbit is classified as `trapped` after `v_parallel` reverses sign. A
complete poloidal cycle without a sign reversal is classified as `passing`.
It remains `unknown` if integration ends before either condition is
established. With `--sigma 0`, the two signs are distinct co- and
counter-passing orbits, but the two trapped rows normally represent different
starting phases of the same trapped orbit and should not both be counted in a
phase-space Jacobian.

The toroidal period is calculated from the average toroidal angular advance
over that completed poloidal or bounce cycle. Consequently, in the field-line
limit,

```text
q = T_poloidal / T_toroidal.
```

A low-energy, zero-`mu` comparison over five flux surfaces agreed with the
`trace` q profile to a mean relative difference of `0.073%` and a maximum of
`0.158%`.

## Scan Output

An energy/pitch scan writes:

```text
energy_keV mu_keV_per_T lambda sigma orbit_type toroidal_period_s
poloidal_period_s orbit_period_s jacobian_relative_keV_s_per_T steps status
```

A `P_phi/q` scan writes:

```text
pphi_over_q_Wb initial_R_m energy_keV mu_keV_per_T lambda sigma orbit_type
toroidal_period_s poloidal_period_s orbit_period_s
jacobian_relative_keV_s_per_T steps status
```

The relative COM Jacobian is

```text
jacobian_relative = orbit_period_s * energy_keV / B0_T
```

with units `keV s/T`. To obtain the fixed-energy Jacobian for the normalized
coordinate `x=(P_phi/q-psi0)/(psi0-psi_edge)`, multiply by the constant
`abs(q*(psi0-psi_edge))`. Constants common to the entire plotted surface may
be omitted when only the distribution shape is needed.

Possible statuses are:

- `complete`: both periods were measured.
- `incomplete`: the configured step limit was reached first.
- `invalid_E_lt_muB`: the fixed starting point has no real `v_parallel`.
- `invalid_Pphi`: no accessible low-field-side root exists.
- `lost_or_singular`: the orbit left the field mesh or encountered singular
  `B_star_parallel`.
- `output_error`: a rank could not create its Poincare file.

Increase `--steps` when valid slow particles are reported as `incomplete`.
