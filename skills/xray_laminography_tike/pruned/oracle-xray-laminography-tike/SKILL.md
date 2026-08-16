---
name: oracle-xray-laminography-tike
description: "Reconstruct a 3D complex-valued volume from parallel-beam/laminographic projections by minimizing a least-squares data-fidelity objective with a conjugate-gradient solver built on a Fourier-slice NUFFT forward/adjoint pair."
---

# oracle-xray-laminography-tike Oracle Skill

This skill reconstructs a 128x128x128 complex volume from a single case of simulated laminographic projection data. The reconstruction poses a linear inverse problem min_V ||P_tilt[V] - d||^2 and solves it with conjugate gradient descent. The forward operator P_tilt maps a volume to per-angle 2D complex projections using the generalized Fourier Slice Theorem: a 3D FFT of the volume is sampled along tilted/rotated planes, then transformed back to detector space. The adjoint scatters projection Fourier data back onto the 3D grid. Each operation below is an independently removable piece: the public contract, run-local input loading, the tilt/rotation frequency-grid geometry, the NUFFT primitives and FFT phase conventions, the forward and adjoint projection assembly, the cost and gradient, the CG step-length estimate, the Dai-Yuan conjugate-gradient update with backtracking line search, the multi-round reconstruction driver, GPU acceleration notes, output writing, and submission validation. Preserve every constant, axis order, FFT normalization, sign convention, and normalization factor exactly as stated; these determine whether the reconstruction converges and matches the expected volume semantics.

## Tilt/rotation frequency-grid geometry (Fourier slice coordinates)

## Laminographic frequency-grid geometry

The forward model samples the 3D Fourier transform of the volume along tilted, rotated planes (generalized Fourier Slice Theorem). For each rotation angle theta_p and each 2D detector frequency `(ku, kv)`, compute a 3D frequency point `f = (f0, f1, f2)`:

```
f0 =  kv * sin(tilt)
f1 = -ku * sin(theta) + kv * cos(theta) * cos(tilt)
f2 =  ku * cos(theta) + kv * sin(theta) * cos(tilt)
```

Conventions (must match exactly):
- Frequency index axis: `k = (arange(n) - n//2) / n`, giving values in `[-0.5, 0.5)`. Use the SAME `k` for both `ku` and `kv`.
- Build `(ku, kv)` with `meshgrid(k, k, indexing='xy')`, then `ravel()` each to length `n*n`. So the fast layout iterates `kv` over rows (`y`) and `ku` over columns (`x`); index `idx = p*n*n + y*n + x`.
- Output `xi` has shape `(R*n*n, 3)`, dtype float32, ordered `[angle, y(row/v), x(col/u)]` flattened.
- `ctilt = cos(tilt)`, `stilt = sin(tilt)`. At tilt = pi/2: `stilt=1`, `ctilt≈0`, so `f0=kv`, `f1=-ku*sin(theta)`, `f2=ku*cos(theta)` (standard tomography slice normal to detector, kv maps straight to the z-frequency).

This grid is reused unchanged by both the forward and adjoint operators (the adjoint negates it). See the resource for a vectorized implementation.
