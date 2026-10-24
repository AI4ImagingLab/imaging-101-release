# Hurricane Florence Dual-Doppler Wind Volume

> Recover a three-dimensional atmospheric wind vector field from two real radar radial-velocity volumes with incomplete viewing geometry and falling hydrometeors.
> Domain: earth science | Keywords: radar tomography, vector-field inversion, variational imaging | Difficulty: Hard

## Background

Weather radar measures the motion of scattering hydrometeors along its beam. A single radial velocity cannot determine the full air velocity: horizontal motion, vertical air motion and particle fall speed all contribute to the observed Doppler shift. Simultaneous views from separated radars provide complementary projections of the same atmospheric flow.

Volumetric wind imaging helps study convection, storm dynamics and transport. Its limitations include incomplete radar coverage, weak sensitivity to vertical motion, differing scan times and errors in the assumed terminal fall speed. Physical constraints and explicit prior assumptions are needed where the viewing geometry does not resolve every vector component.

## Problem Description

Recover eastward u, northward v and upward w wind volumes over the supplied9by21by21 Cartesian grid. The observations are gridded NOAA NEXRAD radar measurements over Hurricane Florence on14September2018around06:24UTC, supplied by the method authors. Original velocity dealiasing and radar-to-grid interpolation are already part of the supplied observations. Historical retrieved wind fields are not inputs.

```math
d_r=\cos\epsilon_r\sin\alpha_r\,u+\cos\epsilon_r\cos\alpha_r\,v+
\sin\epsilon_r\,[w-|v_{t,r}|]+e_r.
```

At each voxel, d is measured outward radial velocity in m/s; r identifies the radar, alpha is azimuth clockwise from north, epsilon elevation, vt empirical terminal particle fall speed, and e residual measurement/model error. Azimuth and elevation are supplied in radians. A binary validity array identifies usable, jointly observed beam-crossing locations. Invalid measurement/calibration entries are filled with0and contribute no radial data constraint.

The terminal speed is vt=A*10^(B*Z)*(1.2/rho)^0.4m/s, where Z is reflectivity in dBZ and rho=exp(−z/10000m)kg/m³ the adopted density profile. Below the5000mfreezing level, (A,B) is(−2.6,.0107)forZ<55,(−2.5,.013)for55<=Z<60,and(−3.95,.0148)forZ>60. At or above freezing it is(−.817,.0063)forZ<33,(−2.5,.013)for33<=Z<49,and(−3.95,.0148)forZ>49. These empirical branches define the selected forward model. Exact endpoint values60below freezing and49above freezing retain zero terminal speed under the supplied reference convention; no valid bound observation lies at these endpoints.

The inverse uses anelastic continuity, divergence of rho*(u,v,w) approximately0. Vertical velocity is fixed0at the lowest and highest analysis planes. The specified conditional reconstruction also uses a weak zero-mean Gaussian assumption for horizontal wind over the whole grid. This is a regularizer assumption, not measured wind calibration or an external numerical forecast. It removes otherwise weakly constrained modes; vertical motion remains sensitive to the continuity and prior assumptions. No physical wind truth is claimed.

Save `output/reconstruction.npz` with all three required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 9, 21, 21) | float64 | Real eastward air velocity u, m/s |
| northward | (1, 9, 21, 21) | float64 | Real northward air velocity v, m/s |
| vertical | (1, 9, 21, 21) | float64 | Real upward air velocity w, m/s |

Axes are `(batch, height z, northward y, eastward x)`, increasing with index. Preserve absolute velocity signs and scale. Evaluate each component separately with uncentered cosine similarity and RMS error normalized by that component's reference dynamic range. Every metric must pass. No mask, crop, magnitude conversion, alignment, gain or offset is fitted during evaluation; the entire declared volume is scored. Agreement establishes the specified conditional inversion, not independent physical accuracy.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`; all arrays are real float64.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| radial_velocity | (1, 2, 9, 21, 21) | float64 | Observed outward Doppler radial velocities, m/s; radar,z,y,x |
| reflectivity | (1, 2, 9, 21, 21) | float64 | Radar reflectivity used for terminal-speed model, dBZ |
| azimuth | (1, 2, 9, 21, 21) | float64 | Beam azimuth clockwise from north, radians |
| elevation | (1, 2, 9, 21, 21) | float64 | Beam elevation above local horizontal, radians |
| validity | (1, 2, 9, 21, 21) | float64 | Binary0/1usable-observation weights, dimensionless |
| x | (1, 21) | float64 | Eastward source-grid coordinate from projection origin, m |
| y | (1, 21) | float64 | Northward source-grid coordinate from projection origin, m |
| z | (1, 9) | float64 | Height coordinate of analysis planes, m |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Radars/order | NEXRAD A,NEXRAD B | identifier | Two actual gridded observation volumes |
| Observation interval |14 September 2018, 06:24:17 to 06:24:50 UTC | UTC | Radar scan start times; frozen-time approximation |
| Radar A location |34.77590942 N,76.87619019 W,43 | degree,degree,m | Source grid_ltx metadata |
| Radar B location |33.989151 N,78.42910767 W,44 | degree,degree,m | Source grid_mhx metadata |
| Projection origin |33.989151 N,78.42910767 W | degree | Cartesian azimuthal equidistant origin |
| Coordinates | Source Cartesian projected grid, east/north/up | dimensionless | Supplied x,y,z arrays fix exact positions |
| Grid spacing | 2250,1714.29,1000 | m | x,y,z regular spacing respectively |
| x range | 24375to69375 | m | Increasing eastward coordinate |
| y range | 68000to102286 | m | Increasing northward coordinate |
| z range | 0to8000 | m | Increasing altitude coordinate |
| Beam-crossing acceptance |30 to150inclusive | degree | Known geometry condition included in validity |
| Effective Earth radius for beam geometry |(4/3)*6371.1 | km | Original supplied beam-angle convention |
| Density scale height |10000 | m | Exponential anelastic density profile |
| Surface density normalization |1 | kg/m³ | rho=exp(−z/10000m)in the selected empirical model |
| Terminal-speed reference density |1.2 | kg/m³ | Density correction factor |
| Freezing height |5000 | m | Piecewise terminal-speed law transition |
| Vertical boundary |w=0at first and last supplied zplanes | m/s | Analysis-plane impermeability approximation |
| Horizontal boundaries | One-sided finite differences | dimensionless | No periodic wrapping or padded winds |
| Noise information | No independent voxel covariance supplied | m/s | Relative radial cost normalized by observed valid radial RMS |
| Missing values |0where validity=0 | dimensionless | Placeholders carry no radial constraint |

## Method Hints

Use a variational vector-field inverse combining the normalized radial-velocity mismatch, a discrete anelastic-continuity penalty and the declared weak horizontal prior. Retain the vertical boundary condition. Compute gradients with the exact transpose of the chosen finite differences or automatic differentiation; a negative-gradient shortcut is generally incorrect at grid boundaries. Bounded limited-memory optimization can handle the full volume. Compare radial fit against a zero-air-wind baseline and inspect sensitivity of the vertical field to prior and continuity weights.

The selected conditional target minimizes the following fixed objective; these weights define the requested field rather than an unspecified family of regularized answers:

```math
J=\frac{\sum m_r(v_r-d_r)^2}{\sigma^2}
+\frac{256\,\mathrm{s}^2}{2}\sum\delta^2
+10^{-4}\,\mathrm{s}^2\mathrm{m}^{-2}\sum(u^2+v^2),\qquad
\delta=D_xu+D_yv+D_zw+w\frac{D_z\rho}{\rho},\qquad
\sigma^2=\frac{\sum m_rd_r^2}{\sum m_r}.
```

Here m is the supplied validity weight, vr the predicted radial velocity, d the observation, sigma the valid observed radial RMS, and delta the discrete anelastic divergence. Sums include every supplied grid point and, in the radial term, both radars. D uses centered interior first differences and one-sided first differences at both endpoints, divided by the supplied coordinate spacing. Bound each air-velocity component to −100 through100m/s, with w fixed0on both end height planes. There are no other regularizers. The requested result is the converged minimizer of this declared conditional objective; no particular optimizer iteration count is required.

In observation displays, blank white voxels denote invalid radial data with zero weight. Wind reconstruction covers the entire declared grid; neutral white in a signed wind colormap denotes velocity near zero, not an uncomputed voxel.

## References

- Jackson et al. (2020), *PyDDA: A Pythonic Direct Data Assimilation Framework for Wind Retrievals*. [Article](https://doi.org/10.5334/jors.264).
- Potvin et al. (2012), variational multiple-Doppler retrieval. [Article](https://doi.org/10.1175/JTECH-D-11-00019.1).
- Shapiro et al. (2009), mass-conservation constraints. [Article](https://doi.org/10.1175/2009JTECHA1256.1).
- [Institutional article copy](https://repository.library.noaa.gov/view/noaa/32197/noaa_32197_DS1.pdf).
