# Asynchronous Acoustic Sensor Geometry

> Recover ten microphone positions and fourteen sound-event positions from measured arrival-time differences and source displacements despite unknown sensor clocks.
> Domain: mathematics | Keywords: acoustic imaging, sensor calibration, nonlinear inverse problems | Difficulty: Hard

## Background

Sound travels a distance in a finite time, so arrival times encode the geometry between a sound source and a microphone. Combining several receivers makes it possible to locate a source and to calibrate the receiver geometry. Applications include acoustic mapping, robot audition and distributed sensing in rooms.

Independent recording devices have different time origins and slightly different sampling clocks. A timing discrepancy can therefore represent either a spatial displacement or a clock error. Repeated sound events and independent motion measurements help separate these effects, but the resulting geometric inverse problem remains nonlinear and sensitive to initialization.

## Problem Description

There are ten microphones and fourteen ordered emission events. Let microphone position be $m_i$, source position be $s_j$, propagation distance be $d_{ij}=\lVert m_i-s_j\rVert_2$, sound speed be $c$, sensor offset be $\tau_i$, and dimensionless fractional clock drift be $\delta_i$. Time $t_j$ is the cumulative known emission interval, with $t_0=0$. The available successive-event delay $a_{ij}$ at one microphone and between-microphone delay $b_{ij}$ for one event obey the first-order clock model

```math
a_{ij}=\frac{d_{i,j+1}-d_{ij}}{c}+(1+\delta_i)(t_{j+1}-t_j)+\epsilon^a_{ij},
\qquad
b_{ij}=\frac{d_{ij}-d_{0j}}{c}+\tau_i+(\delta_i-\delta_0)t_j+\epsilon^b_{ij}.
```

Here microphone index $i$ starts at zero; the second equation applies to $i=1,\ldots,9$. The errors $\epsilon^a$ and $\epsilon^b$ represent timing errors. The source displacement measurement is $u_j=s_{j+1}-s_j+\epsilon^u_j$, where $\epsilon^u$ is displacement error. This clock approximation neglects drift times propagation delay and drift times offset. Offsets and drifts are nuisance unknowns rather than supplied calibration. First offset $\tau_0=0$ fixes the unobservable common time origin; common drift remains observable through successive-event delays.

Use a right-handed Cartesian frame fixed by the first three ordered sources: the first is the origin, the second has positive x coordinate and zero y/z, and the third has positive y and zero z. Observed displacements already use this frame. Do not apply an additional alignment, scale fit or coordinate rotation. The acquisition is one real indoor configuration, with microphone coordinates and source coordinates both unknown. Nonconvex distance geometry, clock-position coupling and noisy odometry make recovery difficult.

Save `output/reconstruction.npz` containing the sole required key `reconstruction`, real float64 shape `(1, 24, 3)`, in metres. Axes are batch, ordered point, Cartesian x/y/z. Points 0–9 are microphones in the observation-channel order; points 10–23 are sound events in emission order. Return physical coordinates with the gauge above; do not normalize or take magnitudes. Clock nuisance estimates need not be submitted. Evaluation measures uncentered cosine similarity and reference-dynamic-range normalized coordinate RMS error separately for microphones and sources, plus timing consistency after fitting the linear clock nuisances from the submitted geometry, and source-displacement RMS error. All groups must pass; evaluation performs no spatial alignment.

## Data Description

All arrays are in `data/raw_data.npz` and have one batch. These are author-provided processed measurements from real recordings and odometry, rather than newly synthesized data. Independent experiment height settings are included in the measured displacement coordinates; unprocessed audio is outside this task.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| tdoa_successive | (1, 13, 10) | float64 | Successive-event arrival-time difference, event interval then microphone; seconds. Includes the emission interval. |
| tdoa_between | (1, 9, 14) | float64 | Arrival time at microphone 1–9 minus microphone 0 for each ordered event; seconds. |
| emission_intervals | (1, 14) | float64 | Seconds; first entry is zero, subsequent entries are positive elapsed emission intervals. |
| displacement | (1, 13, 3) | float64 | Measured next-source minus current-source displacement in the fixed x/y/z frame; metres. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sound speed | 340 | m/s | Constant propagation speed used by the acquisition model. |
| Microphones / events | 10 / 14 | count | Full selected-channel configuration and complete ordered event sequence. |
| Timing origin | First emission | s | Cumulative sum of the supplied intervals defines emission time. |
| Relative clock origin | First microphone offset = 0 | s | Gauge only; its drift is not fixed. |
| Spatial origin and axes | First three source events | m | Origin, positive x axis, positive xy half-plane as defined above. |
| Motion coordinates | Cartesian right-handed | m | Includes independently known two-plane source height settings, 0.80 m and 0.43 m before the observation-frame transform. |
| Propagation | Direct-path constant-speed Euclidean | — | First-order fractional clock model; no reflected-wave reconstruction. |
| Error scales | Not independently calibrated | — | Timing and odometry contain measurement error; no certified noise variance is supplied. |
| Boundaries | Unbounded room-coordinate model | — | No periodic wrap, image boundary or hidden geometric prior is required. |

## Method Hints

Joint nonlinear least squares can combine travel-time and displacement constraints. An analytic distance Jacobian and weighted Gauss–Newton updates make the sensor-clock coupling explicit. A fixed coordinate gauge removes rigid-motion ambiguity. Independent spatial restarts help expose competing stationary points; choose them using observed-data fit rather than target coordinates. Nominal error weights are modeling choices, not measured uncertainty certificates.

## References

Zhang, Wang, Li and Kong. *Asynchronous Microphone Array Calibration Using Hybrid TDOA Information*. arXiv:2403.05791v1 (2024). https://arxiv.org/abs/2403.05791v1
