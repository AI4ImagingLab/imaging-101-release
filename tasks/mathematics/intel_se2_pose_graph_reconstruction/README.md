# Intel SE(2) Pose Graph Reconstruction

> Reconstruct a planar moving sensor's position and heading from real relative wheel and laser constraints while resolving accumulated drift and loop closure inconsistency.
> Domain: mathematics | Keywords: geometric inverse problems, sensor localization, nonlinear least squares | Difficulty: Hard

## Background

Mobile sensing reconstructs spatial relationships by recording motion and matching observations acquired at different locations. A sensor's location and orientation determine how locally measured geometry relates to a common spatial coordinate system. Recovering these transformations supports coherent environmental mapping and navigation.

Small motion errors accumulate along a trajectory. When the sensor revisits a place, the resulting relative constraint ties distant times together and exposes that drift. Redundant observations generally disagree because wheel motion and range matching are imperfect, making consistent global geometry an inverse problem.

## Problem Description

The observations describe a real indoor mobile sensing trajectory. Pose number i is a planar rigid transform T(i), with world position t(i)=(x,y) in metres and counterclockwise heading theta(i) in radians. A relative observation Z(ij) expresses pose j in pose i's coordinate frame:

```math
Z(ij)\simeq T(i)^{-1}T(j).
```

For rotation R(a) through angle a, measured local translation z and measured relative heading phi, define the canonical observation residual:

```math
e(ij)=\begin{bmatrix}R(\phi)^\mathsf{T}\big[R(\theta(i))^\mathsf{T}(t(j)-t(i))-z\big]\\
\operatorname{wrap}(\theta(j)-\theta(i)-\phi)\end{bmatrix}.
```

Here wrap selects the equivalent angle in (-pi,pi]. The supplied symmetric information matrix weights this residual in the measured transform's coordinate convention. Relative measurements cannot determine a common global translation or rotation; fix pose zero exactly to (0,0,0). Every observation must be used. The constraints are processed measurements, and their supplied weights are not an independently established noise calibration.

Save `output/reconstruction.npz` with exactly one key `reconstruction`, shape `(1, 1228, 3)`, real `float64`. Axes are batch, chronological pose, coordinate `(x,y,theta)`. Positions are in metres, headings in radians in (-pi,pi], and pose zero must be exactly zero. Preserve the measured metric scale. Evaluation compares the maximum Euclidean position difference and maximum circular heading difference to a standard method, together with the complete weighted observation error sum e-transpose information e. The gauge is fixed by the anchor; no fitted alignment, rescaling, or coordinate permutation is applied. There is no independently measured absolute trajectory truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| edge_indices | (1, 1483, 2) | int64 | Source and destination zero-based pose identifiers; dimensionless |
| relative_poses | (1, 1483, 3) | float64 | Local measured `(dx,dy,dtheta)`; metres, metres, radians |
| information | (1, 1483, 3, 3) | float64 | Symmetric residual information; entries have inverse products of residual-coordinate units, yielding dimensionless weighted error |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Intel Research Lab, Seattle | text | Real mobile wheel odometry and laser range matching, distributed as a processed pose graph |
| Nodes | 1228 | count | Chronological sensor poses |
| Constraints | 1483 | count | All supplied relative observations, including 1227 sequential and 256 loop edges |
| Gauge anchor | Pose 0 = (0,0,0) | m, m, rad | Coordinate convention; no absolute position measurement |
| Frame convention | Right-handed planar x/y, counterclockwise heading | text | Relative translation expressed in the source pose frame |
| Observation residual | Z inverse times predicted relative transform | text | Translation coordinates and wrapped rotation as defined above |
| Information | Provided full positive definite matrices | mixed inverse squared units | Preserved nominal weights; large dynamic range is intrinsic to the processed graph |
| Temporal sampling | Not provided | unknown | Pose index is ordering, not calibrated elapsed time |
| Boundary | None | text | Finite connected graph, no periodic world boundary |

## Method Hints

Use anchored sparse nonlinear least squares on the rigid motion graph. Composing sequential relative transforms provides an observation-only initial trajectory. Gauss-Newton linearizes the canonical residual, solves a sparse normal system for free pose increments, and wraps heading updates. Consistent residual coordinates and information matrices are essential: changing the residual transform order changes the weighted objective. The highly unequal supplied weights also make early linear systems numerically delicate.

## References

- Kummerle et al., *g2o: A General Framework for Graph Optimization*, ICRA 2011. [Paper](https://doi.org/10.1109/ICRA.2011.5979949).
- Carlone and Censi, *From Angular Manifolds to the Integer Lattice: Guaranteed Orientation Estimation With Application to Pose Graph Optimization*, IEEE Transactions on Robotics 30(2), 2014. [Processed datasets and acquisition attribution](https://lucacarlone.mit.edu/datasets/).

The bound method is a current maintained author implementation of the paper's canonical planar pose-graph objective, not a reproduction of the paper's exact 2011 experimental dataset.
