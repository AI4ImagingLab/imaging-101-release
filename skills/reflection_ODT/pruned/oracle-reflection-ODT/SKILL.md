---
name: oracle-reflection-ODT
description: "Task-specific reconstruction guidance for reflection-mode optical diffraction tomography from multi-angle reflected-light intensity data."
---

# oracle-reflection-ODT Oracle Skill

Use this skill when preparing a submission for the reflection-mode ODT case. It records the public I/O contract, numerical conventions, reflection BPM equations, solver settings, and submission checks needed for a faithful reconstruction.

## Normalize physical units and tensor dtypes

Convert metadata to the normalized BPM convention before building optical operators. Use `res = tuple(res_um_i * n0 / wavelength_um for i in x,y,z)`. With the public metadata this is approximately `(0.6677848101265823, 0.6677848101265823, 2.4367088607594938)`. Keep array order `(Nz, Ny, Nx)` and lateral axes `(y, x)` for images. Use complex128 for fields and Fourier-domain operators and float64 for real volumes, frequencies, amplitudes, and losses when running the iterative solver. Select `cuda` only if available; CPU is valid but slower.

## Illumination rings, ordering, and tilted plane waves

Simulate and compare angles in the metadata ring order: BF NA 0.130 with 8 angles, BF NA 0.251 with 16 angles, DF NA 0.420 with 16 angles, then DF NA 0.560 with 16 angles, for a total of 56 images. Within each ring, angle `m` has `theta = 2*pi*m/n_angles_in_ring`, direction cosines `c_a = (NA/n0)*cos(theta)` and `c_b = (NA/n0)*sin(theta)`. Truncate each to a periodic Fourier-compatible integer grid by `c_a = int(c_a * nx * res[0]) / (nx * res[0])` and `c_b = int(c_b * ny * res[1]) / (ny * res[1])`. The incident field is `exp(2j*pi*c_b*res[1]*y)[:,None] * exp(2j*pi*c_a*res[0]*x)[None,:]` on pixel coordinates starting at 0.

## Reflection-mode round-trip layer traversal

For one angle, start with its tilted incident field and traverse all four slices downward. For each slice index `iz=0..Nz-1`: propagate by `dz_layer`, scatter with `dn[iz]` using `dz_layer`, then propagate by `dz_gap`. Reflect at the mirror by multiplying the whole field by `-1`. Traverse upward in reverse order `iz=Nz-1..0`: propagate by `dz_gap`, propagate by `dz_layer`, then scatter with `dn[iz]` using `dz_layer`. Back-propagate to the focal plane using `total_depth = Nz*(dz_gap + dz_layer)` and propagation distance `-total_depth`. Then apply the pupil in Fourier space and return intensity `abs(u)**2`. This exact round-trip ordering, including two gap traversals per layer and the mirror phase, is central.
