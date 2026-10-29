# Multicamera Volumetric Particle Trajectories

> Reconstruct three-dimensional particle positions and temporal trajectories from four synchronized camera image sequences with dense overlapping particle projections.
> Domain: mechanics | Keywords: particle tracking, volumetric imaging, inverse geometry | Difficulty: Hard

## Background

Fluid transport, mixing and dispersion depend on the motion of individual fluid elements. Small tracer particles provide optical markers whose positions can be recorded with cameras. Following the same particle through time gives a Lagrangian description of flow and allows velocity and acceleration to be examined along trajectories.

A single camera collapses depth. Multiple calibrated views constrain three-dimensional position through intersecting viewing rays, but dense particle images produce many geometrically plausible false correspondences. Overlap also shifts detected image peaks. Long trajectories add temporal information while introducing the risk of incorrect connections and fragmentation.

## Problem Description

The observations are the author's existing simulated four-camera sequence, with100 synchronized frames and native1024×1024 images. Recover particle positions in the complete40mm cube and the temporal connections between detections. The input is reused exactly, without cropping, image resampling or synthesizing replacement observations.

For a world position vector r in millimetres, camera c first forms normalized perspective coordinates from R(c)r+t(c). Apply radial and tangential distortion, then the supplied camera matrix K(c), to obtain image coordinates (u,v), with u increasing by column and v by row. The image model is a superposition of localized Gaussian particle projections, subject to the author's rasterization and finite intensity representation:

```math
I_{fc}(x,y)\simeq\sum_j A_c
\exp\!\left[-B_c(x-u_{fcj})^2-C_c(y-v_{fcj})^2\right].
```

Here f is the frame, c the camera, j the particle, I the observed count value, (x,y) the integer pixel centre, and A,B,C the camera-dependent peak and inverse squared width. These Gaussian parameters are not supplied as hidden calibration: they can be estimated from the actual images. The finite projection window and source method's overlap approximation are reconstruction conventions, rather than an assertion that every measured pixel equals this continuous model exactly.

For normalized coordinates (a,b), define radius squared s=a²+b². With the distortion array ordered (k1,k2,p1,p2,k3), the distorted coordinates are:

```math
\begin{aligned}
d&=1+k_1s+k_2s^2+k_3s^3,\\
a'&=ad+2p_1ab+p_2(s+2a^2),\\
b'&=bd+p_1(s+2b^2)+2p_2ab,\\
u&=K_{00}a'+K_{02},\qquad v=K_{11}b'+K_{12}.
\end{aligned}
```

Save `output/reconstruction.npz` relative to the task workspace, with all three required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| positions | (1,N,3) | float64 | World position estimates in mm, coordinate order X,Y,Z, all finite |
| frame_index | (1,N) | int16 | Frame number0 through99, dimensionless; one entry per position |
| track_id | (1,N) | int32 | Nonnegative segment label, dimensionless; equal labels assert temporal identity |

N is the returned number of point records and must be positive. Labels are arbitrary within a submission and may be renamed freely. A label/frame pair must be unique. Distinct segments need distinct labels; labels must not be reused for unrelated trajectories. Row order is arbitrary. No spatial rotation, translation, scaling or hidden alignment is performed. The cube defines acquisition and true-particle support; noisy position estimates may lie outside it and remain fully scored. No postfit coordinate clipping or deletion is applied.

Evaluation uses independent simulated positions and identities: unique nearest point matching within0.1mm in each frame, point recall and precision, RMS Euclidean position error in mm, and recall/precision of consecutive-frame same-segment temporal edges. Matching candidates are processed by increasing distance with stable input-row tie order; a truth point can be owned only once. Missing points reduce recall; extra points reduce precision. Singleton labels have no temporal edges and receive zero edge precision. All five metrics must pass. Unmatched detections do not contribute to conditional position RMS, so the separate coverage/precision criteria are essential. No-match RMS is assigned0.1mm.

## Data Description

All visible arrays are in `data/raw_data.npz`, loadable with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1,100,4,1024,1024) | uint32 | Native nonnegative intensity counts, axes batch,frame,camera,row,column; ADU |
| camera_matrix | (1,4,3,3) | float64 | Camera intrinsic matrix K; focal entries and principal points in pixels |
| rotation | (1,4,3,3) | float64 | World-to-camera matrix R, dimensionless |
| inverse_rotation | (1,4,3,3) | float64 | Supplied algebraic inverse of R, dimensionless; rounded R is not assumed exactly orthogonal |
| translation | (1,4,3) | float64 | World-to-camera translation t in mm |
| camera_centres | (1,4,3) | float64 | Supplied camera origins in world XYZ, mm |
| distortion | (1,4,5) | float64 | Ordered k1,k2,p1,p2,k3 distortion coefficients, dimensionless normalized-coordinate convention |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Data nature | Author-existing simulated tracer sequence | — | Not a real measurement and not a newly generated maintainer simulation |
| Frames / cameras |100 /4 | — | Complete supplied sequence and camera set |
| Image size |1024×1024 | pixel | Native rows and columns; no crop or binning |
| Pixel coordinates | Zero-based integer pixel centres | pixel | X image axis is column, Y image axis is row |
| World bounds | X,Y,Z each[-20,20] | mm | Complete interrogation volume, inclusive endpoints |
| Frame interval |1 | s | Author runtime time convention; velocity derived from it is mm/s |
| Intensity convention |0..255 | ADU | Native TIFF numeric values retained losslessly as uint32 |
| Camera order | Original cameras1..4 mapped to indices0..3 | — | Geometry and images use the same order |
| Perspective axes | Rr+t, divide first two coordinates by third | mm then normalized | Supplied signs, including negative vertical focal entries, must be retained |
| Ray origin | Supplied camera centre | mm | Backproject direction using supplied inverse_rotation |
| Depth / boundary | Perspective finite camera plane; no periodic wrapping | — | Out-of-volume particles may leave the measurement region |
| Known noise model | No separately calibrated noise variance supplied | ADU | Estimate consistency against actual observations; do not assume added Gaussian noise |
| Validated CPU environment | Windows x64, MSVC19.29 C++17 compiler plus declared Python numerical subset | — | The reference uses locally readable standard numerical C++ kernels, compiled without network or upstream packages |

## Method Hints

Iterative particle reconstruction combines subpixel peak localization, calibrated viewing-ray intersection and competition between candidate correspondences. Enforcing exclusive ownership of a two-dimensional detection suppresses false three-dimensional particles. Local image-domain coordinate refinement can then reduce disagreement with observed particle projections.

Temporal prediction uses existing trajectory information to propose the next location before reconstructing newly appearing particles from residual images. A coarse displacement field can initialize short tracks; adaptive linear prediction and local refinement extend established tracks. Separate spatial accuracy and temporal connectivity are needed because a collection of accurate isolated points is not a trajectory reconstruction.

## References

Tan, Salibindla, Masuk and Ni, *Introducing OpenLPT: new method of removing ghost particles and high-concentration particle shadow tracking*, Experiments in Fluids61:47(2020), [doi:10.1007/s00348-019-2875-2](https://doi.org/10.1007/s00348-019-2875-2). The selected modern author implementation differs from the paper-era configuration; the current method and a qualified coordinate-state correction are documented in maintainer evidence. This benchmark does not claim reproduction of the paper's performance tables.
