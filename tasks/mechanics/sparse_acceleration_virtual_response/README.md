# Sparse Acceleration Virtual Response

> Reconstruct displacement, velocity and acceleration throughout a ten-level structure from noisy accelerations at five levels while the excitation is unknown.
> Domain: mechanics | Keywords: virtual sensing, inverse dynamics, Bayesian estimation | Difficulty: Hard

## Background

Structural monitoring often measures motion at only a few accessible locations. Accelerometers provide a practical signal, but the motion of uninstrumented parts can control fatigue and serviceability. Mechanical coupling allows information to travel between locations, making a model useful for estimating the complete response.

Unknown excitation and measurement noise make this inverse problem difficult. Independent time integration accumulates low-frequency errors, and interpolation between sensors does not enforce structural dynamics. Virtual sensing combines partial measurements with a mechanical model to infer unobserved response histories and their uncertainty.

## Problem Description

The bound case contains author-code-generated synthetic accelerations for a ten-degree-of-freedom shear structure. Every degree of freedom describes one horizontal floor displacement relative to a fixed base. Infer the full response over the supplied reconstruction time grid, including all unmeasured floors.

The physical model is

```math
M\ddot u(t)+C\dot u(t)+Ku(t)=f(t),\qquad y(t)=S\ddot u(t)+\epsilon(t).
```

Here $u(t)$ is the ten-component displacement in metres, $\dot u(t)$ velocity in m/s, $\ddot u(t)$ acceleration in m/s², $M$ the known mass matrix in kg, $C$ the viscous damping matrix in N s/m, $K$ the stiffness matrix in N/m, $f(t)$ the unknown force vector in N, $S$ the known five-row sensor selection matrix, and $y(t)$ the observed accelerations in m/s². The measurement error $\epsilon(t)$ has independent zero-mean Gaussian components with the variance stated below. The matrices are supplied calibration, established before response generation. Unobserved excitation and low-frequency integration ambiguity require a dynamical prior; no force history is supplied.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly the following three real arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 100001, 10) | float32 or float64 | Displacement in m; axes batch, reconstruction time, floor 1 through 10 |
| velocity | (1, 100001, 10) | float32 or float64 | Velocity in m/s; the same axes and floor order |
| acceleration | (1, 100001, 10) | float32 or float64 | Acceleration in m/s²; the same axes and floor order |

All values must be finite. Preserve the supplied time and floor order. Quantities are relative to the fixed base, with the supplied physical scale and sign; no per-floor centring, drift removal, time shift, rescaling or temporal interpolation is applied in evaluation. Each of the ten floor histories is evaluated separately for each physical quantity using cosine NCC without mean subtraction and RMS error divided by that reference history's full dynamic range. All 60 metric conditions must pass. This prevents strong channels from hiding missing unmeasured responses.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| accelerations | (1, 20001, 5) | float64 | Measured accelerations in m/s²; axes batch, observation time, sensor |
| observation_time | (1, 20001) | float64 | Observation times in s |
| reconstruction_time | (1, 100001) | float64 | Required output times in s |
| mass | (1, 10, 10) | float64 | Calibrated mass matrix in kg, ordered by floor |
| stiffness | (1, 10, 10) | float64 | Calibrated stiffness matrix in N/m, ordered by floor |
| damping | (1, 10, 10) | float64 | Calibrated viscous damping matrix in N s/m, ordered by floor |
| sensor_matrix | (1, 5, 10) | float64 | Dimensionless selection matrix mapping floor accelerations to sensor order |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Structural coordinates | Ten horizontal floor motions | m | Floor order increases from the fixed base to the roof |
| Base condition | Zero displacement | m | Fixed support below floor 1 |
| Observation interval | 0 to 1000 inclusive | s | Complete supplied record |
| Observation spacing | 0.05 | s | 20 Hz acceleration measurements |
| Output spacing | 0.01 | s | 100 Hz required response grid |
| Measured floors | 1, 3, 5, 7, 9 | index | Sensor order agrees with the rows of sensor_matrix |
| Floor mass | 200 | kg | Diagonal mass calibration |
| Inter-level spring constant | 5000 | N/m | Fixed-base chain with the supplied roof boundary entry |
| Damping calibration | Supplied full matrix | N s/m | Use its entries without replacing it by an assumed scalar damping law |
| Measurement noise variance | 0.0001 | (m/s²)² | Independent sensor noise |
| Additional supplied force measurement | None | N | Excitation is latent |

## Method Hints

A modally reduced Gaussian-process latent-force model can combine spatial mechanics and temporal regularity. For mass-normalized mode shapes $\Phi$ satisfying $K\Phi=M\Phi\Omega^2$ and $\Phi^TM\Phi=I$, approximate displacement by $u=\Phi r$, where $r$ contains modal coordinates and $\Omega$ contains angular natural frequencies. A compact reference uses the first three modes and independent latent modal inputs.

The Matérn-5/2 covariance provides a smooth input prior:

```math
k(\tau)=a^2\left(1+\frac{\sqrt5|\tau|}{\ell}+\frac{5\tau^2}{3\ell^2}\right)\exp\left(-\frac{\sqrt5|\tau|}{\ell}\right),
```

where $\tau$ is time lag in s, $a$ is the positive modal-input scale, and $\ell$ a positive correlation time in s. Its finite-dimensional state-space representation can be joined with the mechanical dynamics. Match modelled stationary sensor distributions to empirical acceleration distributions to estimate the prior scales, then use Gaussian filtering and backward smoothing. This uses measurements before and after a prediction time and constrains all physical response quantities together.

## References

Joanna Zou, Eliz-Mari Lourens and Alice Cicirello, *Virtual sensing of subsoil strain response in monopile-based offshore wind turbines via Gaussian process latent force models*, Mechanical Systems and Signal Processing 200, 110488 (2023). https://doi.org/10.1016/j.ymssp.2023.110488

Author preprint, version 2: https://arxiv.org/abs/2207.05901v2
