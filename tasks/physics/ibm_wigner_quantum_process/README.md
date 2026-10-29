# IBM Wigner Quantum Process Tomography

> Reconstruct an unknown single-qubit unitary process from noisy auxiliary-qubit interferometric scans over a sphere, combining weak and partially suppressed measurement channels.
> Domain: physics | Keywords: quantum tomography, phase-space imaging, inverse problems | Difficulty: Hard

## Background

Quantum operations transform quantum states coherently. Their action must be characterized to understand laboratory control errors and to compare implemented operations with a desired design. Measurements of a single output state cannot determine a general operation: several independently prepared probes or auxiliary quantum systems are needed.

Interferometric experiments with auxiliary qubits encode an operation into measurable expectation values. Spherical scanning provides a phase-space representation in which scalar and vector harmonics describe a single-qubit process. Finite measurement counts, decoherence and imperfect control reduce contrast and mix measurement channels, making recovery more difficult than reconstructing an ideal circuit.

## Problem Description

Use the complete measured 50-direction experiment to reconstruct a complex unitary process matrix. Express a determinant-one representative as

```math
U= D I+i(A X+B Y+C Z), \qquad A^2+B^2+C^2+D^2=1.
```

Here I is the 2-by-2 identity, X, Y and Z are Pauli matrices, and A, B, C, D are real quaternion coordinates. For direction n=(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)), put v=A n_x+B n_y+C n_z. The ideal complex expectation values in scalar rank 0 and vector rank 1 satisfy

```math
e_{0,I}=D^2, \quad e_{1,I}=iDv, \qquad
e_{0,j}=-iq_jD, \quad e_{1,j}=q_jv, \quad j\in\{X,Y,Z\},
```

where q_X=A, q_Y=B and q_Z=C. The real and imaginary parts are complementary auxiliary-spin measurements; each complex value combines measured Pauli expectations. Experimental errors perturb these ideal relations, including attenuation and offsets. The four controlled reference operations I, X, Y, Z are known probe operations, not the unknown process. No ideal target matrix is supplied.

Save **output/reconstruction.npz** with key **reconstruction**, shape **(1, 2, 2)**, dtype **complex64 or complex128**, dimensionless unitary amplitudes. Axes are batch, computational output basis (0,1), computational input basis (0,1). Preserve complex amplitudes and unit physical scale. An arbitrary common phase is physically irrelevant. Evaluation removes the
determinant phase to obtain either equivalent SU(2) representative, extracts its
real quaternion, and evaluates all ideal expectations above. The sphere-average
complex mean squared error is sum over rank/control/direction of weight times
absolute squared expectation residual, divided by8. Every direction and both
complex quadratures contribute. It may exceed an independently attained standard
fit by at most1e-6; equal or better fits are accepted independently of iterative
projection history. The Frobenius error of U-dagger U minus identity must be at
most1e-6. No entrywise phase adjustment or amplitude rescaling is fitted. The
ideal-model residual retains hardware attenuation and offsets; this criterion
is not a calibrated gate-error confidence bound.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| expectation | (1, 2, 4, 50) | complex128 | Dimensionless measured complex expectations; axes batch, scalar/vector rank, controlled I/X/Y/Z, direction. |
| polar_angle | (1, 50) | float64 | Polar angle theta in radians from positive z. |
| azimuth | (1, 50) | float64 | Azimuth phi in radians measured from positive x toward positive y. |
| quadrature_weight | (1, 50) | float64 | Dimensionless Lebedev sphere-average integration weights, summing to one. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Physical platform | IBM superconducting quantum processor | not applicable | Published experimental auxiliary-qubit process scan. |
| Measured direction count | 50 | direction | Complete supplied Lebedev acquisition, no direction removed. |
| Circuit size | 3 | qubit | One process qubit and auxiliary mapping/readout qubits. |
| Controlled reference order | I, X, Y, Z | dimensionless | Known identity and Pauli controls. |
| Complex convention | first expectation + i times complementary expectation | dimensionless | Scalar rank and spin-correlation vector rank. |
| Angular geometry | full sphere | radian | Polar theta in [0,pi], periodic azimuth. |
| Integration convention | sphere average | dimensionless | Integral over solid angle divided by 4pi. |
| Ideal process convention | determinant-one representative, common phase unobservable | dimensionless | U=D I+i(A X+B Y+C Z). |
| Raw-shot uncertainty | unavailable in supplied processed archive | not applicable | No invented count variance or Gaussian noise level is imposed. |
| Device errors | retained in observations | dimensionless | No target-fitted calibration matrix is provided. |

## Method Hints

A spherical-harmonic projection can initialize the real quaternion, followed
by constrained least squares on the complete measured expectation model.
Normalize the quaternion to enforce SU(2), and preserve all complex measurement
channels. Equivalent Euler-angle or unit-quaternion optimizers are allowed.
Compare residuals and multiple minima; process unitarity does not remove
experimental attenuation or make the ideal response a hardware calibration.

## References

- Devra, Van Damme, vom Ende, Malvetti and Glaser. *Theory and Experimental Demonstration of Wigner Tomography of Unknown Unitary Quantum Gates*. [arXiv:2411.05404v2](https://arxiv.org/abs/2411.05404v2), 2024; [published article](https://doi.org/10.1007/s11128-026-05269-z).
