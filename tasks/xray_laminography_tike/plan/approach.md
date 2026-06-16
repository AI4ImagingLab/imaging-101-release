# Solution Approach

## Problem Statement

Recover a 3D complex-valued volume V (128 x 128 x 128, complex64) from 128 simulated
laminographic projections. The tilt angle is pi/2, which corresponds to standard
parallel-beam tomography.

## Forward Model

The laminographic forward model maps a 3D volume to a set of 2D projections:

    d(theta) = P_tilt[V](theta)

For each rotation angle theta, the operator integrates the volume along rays determined
by the tilt geometry. In Fourier space, this is equivalent to extracting tilted planes
from the 3D FFT of V (generalized Fourier Slice Theorem).

At tilt = pi/2, the model reduces to the standard Radon transform: for each angle,
integrate along the beam direction (perpendicular to the detector).

The forward operator is implemented using a simplified NUFFT (non-uniform FFT) with
nearest-neighbor interpolation for GPU computation via CuPy. No external reconstruction
library is required.

## Conjugate Gradient Solver

The reconstruction minimizes the data-fidelity cost:

    min_V || P_tilt[V] - d ||^2

using the conjugate gradient (CG) algorithm. CG requires:

1. The forward operator: P_tilt[V] -> d  (src.physics_model.forward_project)
2. The adjoint operator: P_tilt^H[d] -> V  (src.physics_model.adjoint_project)

The CG iteration updates V by computing the gradient (via adjoint), maintaining
conjugate search directions, and performing line searches.

## Reconstruction Strategy

The reconstruction is organized as 5 rounds of 4 CG iterations each (20 total):

- **Initial guess:** zeros (128 x 128 x 128, complex64)
- **Round 1:** CG iterations 1-4, cost drops from ~1.24 to ~0.15
- **Round 2:** CG iterations 5-8, cost drops to ~0.05
- **Round 3:** CG iterations 9-12, cost drops to ~0.025
- **Round 4:** CG iterations 13-16, cost drops to ~0.017
- **Round 5:** CG iterations 17-20, cost drops to ~0.013

Between rounds, the current volume estimate is passed as the initial guess for the
next round, allowing monitoring of the cost convergence.

## Expected Results

- **Cost convergence:** from ~1.24 (round 1) to ~0.013 (round 5)
- **NCC vs ground truth:** ~0.96 (high correlation)
- **NRMSE vs ground truth:** ~0.005 (low error)

The reconstruction is expected to closely match the ground truth because:
1. The data is noise-free (simulated)
2. Full angular coverage (128 angles over [0, pi))
3. Standard tomography geometry (tilt = pi/2) provides good coverage of Fourier space
