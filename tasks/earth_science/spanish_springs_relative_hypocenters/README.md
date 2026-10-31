# Spanish Springs Relative Hypocenters

> Reconstruct earthquake hypocenters and origin-time corrections from real differential P/S arrival times, an initial catalog and independent velocity-derived travel-time calibration.
> Domain: earth_science | Keywords: seismic imaging, source localization, relative relocation | Difficulty: Hard

## Background

Earthquake hypocenters reveal the geometry of active faults and the evolution of seismic sequences. A seismic phase arrival time depends on the source location, origin time and propagation speed through the Earth. Absolute locations are affected by uncertain structure and noisy arrival picking, so nearby events can appear spatially blurred in a catalog.

Waveform cross correlation measures small arrival-time differences between related earthquakes at a common station. Shared propagation paths reduce some common errors, making these differential measurements useful for resolving fine fault structures. The resulting inverse problem remains nonlinear: origin-time and position changes can compensate each other, stations have limited angular coverage and some events have weak observational links.

## Problem Description

Relocate all 1616 events of a real Spanish Springs, Nevada sequence. Each supplied differential observation links catalog events i and j, one station s and a phase p (P or S). Its sign is second event minus first. For hypocenter coordinates q and origin correction h in seconds,

```math
b_{ijsp}=T_p(r_{js},z_j)+h_j-T_p(r_{is},z_i)-h_i+e_{ijsp}.
```

Here b is the supplied differential travel-time measurement in seconds after the original catalog-origin convention; z is depth in kilometres, e is error, and T is the known table-interpolated travel time. Unknown latitude and longitude are in degrees. Distances use the local geographic approximation

```math
r_{is}=g\sqrt{(\phi_i-\phi_s)^2+
[(\lambda_i-\lambda_s)\cos((\phi_i+\phi_j)/(2a))]^2}.
```

Latitude is phi, longitude lambda, g is kilometres per degree and a is degrees per radian. For each observed pair, the same mean event latitude is used for both source-to-station ranges. The supplied initial catalog is a reference position and an anchoring prior, not an independently known final hypocenter truth.

T is bilinear interpolation on the supplied regular range/depth grids, separately for each phase. For a bracketing cell, let f and h be the fractional range and depth coordinates, and A, B, C, D the lower-range/lower-depth, upper-range/lower-depth, lower-range/upper-depth and upper-range/upper-depth table values:

```math
T=(A+f(B-A))+h\left[(C+f(D-C))-(A+f(B-A))\right].
```

At an upper grid boundary use the last interior cell with fraction one. A zero table corner invokes the following known calibration convention: independently in each bracketing depth row choose the adjacent pair of nonzero range samples whose midpoint is nearest the query range, use that line for range extrapolation, then interpolate between the two depths. Ties use the lower range pair. This convention matters near the origin; no table values are inferred from final locations.

Define an undirected graph from every row of event_pairs, including weak correlations. Within EACH connected component, impose that the arithmetic mean change of latitude, longitude and depth from initial_catalog is zero, and the mean origin correction is zero, up to numerical precision. Isolated events therefore retain their catalog values. Spatial anchoring is an explicit conditional prior: overall hypocenter translation is generally not an exact observation symmetry with fixed stations. A common origin correction within a connected component is an exact differential-time gauge, fixed by its zero mean. No fitted clustering or hidden labels define these constraints.

Save `output/reconstruction.npz` with `reconstruction`, real float64 shape `(1, 1616, 4)`. Axes: batch, original event row, quantity ordered (latitude [degree], longitude [degree], depth [km], origin-time correction [s]). Positive depth points downward from the model surface; positive correction means a later origin than the catalog. Latitude/longitude are absolute geographic coordinates, without centering or alignment. Depth must lie within 0–26 km, origin corrections within−50–50 s, and queried source-to-station ranges within 0–200 km. Do not reorder event rows.

Evaluation independently checks the physical residual for EVERY qualified observation and both phases separately, together with every connected component's spatial centroid and mean time correction. The fixed qualification mask uses observed correlation>=0.6 and initial-catalog pair-centroid-to-station range<=80 km. This latter range uses the mean of station and pair-centroid latitude for its cosine, with the same g and a constants; it is fixed before fitting. Each phase metric is the mean absolute residual in seconds with equal weight per included observation. No observation is removed based on fitted residual or final cluster membership. Spatial centroid offsets are expressed in kilometres using g and the initial component mean latitude. No comparison forces a particular optimizer, hierarchical merger path or final cluster identity.

The joint physical acceptance bounds are P mean absolute residual <=0.020105228534259745 s, S mean absolute residual <=0.017063631931293554 s, maximum component centroid displacement <=0.00005 km and maximum absolute component mean origin correction <=0.000001 s. All four conditions must hold; lower likelihood residuals remain admissible.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| initial_catalog | (1, 1616, 4) | float64 | Reference latitude [degree], longitude [degree], depth [km], initial origin correction0[s]. |
| event_ids | (1, 1616) | int64 | Original event identifiers [dimensionless]; event_pairs uses row indices instead. |
| station_positions | (1, 51, 2) | float64 | Station latitude and longitude [degree]; no elevation correction is included. |
| event_pairs | (1, 50214, 2) | int64 | Zero-based initial_catalog row indices i, j [dimensionless] for each observed event pair. |
| observation_bounds | (1, 50214, 2) | int64 | Zero-based start-inclusive/end-exclusive slices [dimensionless] into the observation arrays for each pair. |
| station_index | (1, 608577) | int64 | Zero-based station_positions row for every observation [dimensionless]. |
| phase | (1, 608577) | int32 | Phase code1=P,2=S [dimensionless]. |
| differential_times | (1, 608577) | float64 | Measured differential times [s], sign j minus i. |
| cross_correlation | (1, 608577) | float64 | Observed waveform correlation coefficient [dimensionless]. |
| travel_time_table | (1, 101, 27, 2) | float32 | Independent velocity-model-derived travel times [s], axes range/depth/P-S. Native calibration precision retained; finite-precision spherical-to-flat-Earth calibration. |
| distance_grid | (1, 101) | float32 | Table horizontal ranges0..200[km]. |
| depth_grid | (1, 27) | float32 | Table source depths0..26[km]. |
| velocity_model | (1, 18, 2) | float64 | Independent model depth[km] and P speed[km/s]; repeated depths mark discontinuities. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sequence | Spanish Springs, Nevada,2012–2015 | 1 | Real catalog and waveform-derived relative arrivals. |
| Geographic range constant g |111.1949266|km/degree| Local range convention in the declared physical evaluator. |
| Angular constant a |180/3.1415927|degree/radian| Explicit source geographic convention. |
| Depth datum |0|km| Velocity-model surface, depth positive down; stations treated at this datum. |
| Table range spacing |2|km| Uniform101 range grid. |
| Table depth spacing |1|km| Uniform27 depth grid. |
| S-speed convention |P speed/1.732, serialized to four decimal places|km/s| Independent one-dimensional S calibration derived from P model. |
| Travel-time convention |First-arrival P/S table from known one-dimensional velocity |s| Table is known acquisition response, not fitted to relocated events. |
| Differential sign |T(j)+h(j)−T(i)−h(i)|s| Original first-minus-second text convention has been converted explicitly. |
| Qualified correlation |>=0.6|1| Fixed observation-quality condition, not a fitted residual cut. |
| Qualified initial range |<=80|km| Fixed initial pair-centroid/station geometry condition. |
| Anchoring |Each full observation-graph component centroid fixed to initial catalog; mean origin correction zero|mixed stated units| Explicit conditional prior and time gauge; no final fitted clustering input. |
| Observation precision |Original decimal values stored in float64|s/1/degree/km| No observation rounding; the known travel-time calibration retains its nativefloat32 representation. |

## Method Hints

A robust relative-location objective can reduce sensitivity to spurious cross-correlation measurements. Similarity-weighted hierarchical grouping can process the most reliable event relationships first, optimize relative positions of linked groups while preserving their internal geometry, and retain stable catalog anchors. A local grid search can avoid requiring derivatives of a travel-time interpolant. More general constrained differential-time inversions are also admissible.

Origin-time differences can often be eliminated analytically for a fixed spatial configuration: the median differential residual minimizes an equally weighted absolute-error objective. Respect the graph-level conditional anchors and assess both P and S observation consistency; finer-looking fault geometry alone is not evidence of a valid reconstruction.

## References

Trugman, D.T. and Shearer, P.M. (2017), GrowClust: A Hierarchical Clustering Algorithm for Relative Earthquake Relocation, with Application to the Spanish Springs and Sheldon, Nevada, Earthquake Sequences. Seismological Research Letters88,379–391. DOI[10.1785/0220160188](https://doi.org/10.1785/0220160188).
