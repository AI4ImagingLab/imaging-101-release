# North Anatolian Fault Creep Distribution

> Recover distributed fault slip rates and nuisance deformation coefficients from three real satellite line-of-sight velocity maps and ground-station velocities through an elastic half-space model.
> Domain: earth science | Keywords: geodetic imaging, elastic inversion, fault mechanics | Difficulty: Hard

## Background

An active fault can move slowly without producing a large earthquake. This creep deforms the surrounding ground. Satellite radar measures the component of ground motion along its viewing direction, while geodetic stations measure horizontal motion at individual locations. Combining viewing directions and ground observations provides information about slip beneath the surface.

Elastic deformation spreads the contribution of a buried slipping region over a broad area. Nearby fault elements can therefore produce similar observations, and sensitivity decreases with depth. Measurement correlations, reference-frame offsets and local subsidence also affect the inferred distribution. Imaging fault creep requires a conditional physical model and carefully stated assumptions about these additional contributions.

## Problem Description

The observations are existing, author-curated real velocity products near the creeping central North Anatolian Fault. Recover uniform right-lateral slip rates on the supplied1292triangles and seven rectangles, together with a local subsidence coefficient and five velocity-reference offsets. The five deep rectangles share one slip rate. Geometry is known; dip-slip and opening are fixed to zero.

The forward model is

```math
\boldsymbol d=G\boldsymbol s+\boldsymbol b a+T\boldsymbol c+\boldsymbol\epsilon.
```

Here the3554-vector of observations is denoted by $\boldsymbol d$, the1299slip rates by $\boldsymbol s$, and the scalar formal deflation coefficient by $a$. The matrix $G$ contains the surface east/north/up responses to unit uniform right-lateral slip in a homogeneous isotropic elastic half-space with a traction-free surface. Satellite rows project these responses onto the supplied line-of-sight vectors. Ground-station rows select east and north components. The vector $\boldsymbol b$ is the projected fixed empirical subsidence basis defined below. The matrix $T$ adds one constant to each of the three radar tracks and one translation to each ground-station component; $\boldsymbol c$ contains these five offsets. The noise vector is $\boldsymbol\epsilon$.

For a surface point with horizontal displacement $(X,Y)$ in metres from the specified basin center, define

```math
\boldsymbol b_{\mathrm{ENU}}(X,Y)
=-10^6\frac{(1-\nu)r^3}{\mu\left(X^2+Y^2+h^2\right)^{3/2}}
\begin{pmatrix}X\\Y\\h\end{pmatrix}.
```

The radius parameter is $r$, the depth is $h$, Poisson's ratio is $\nu$, and the shear modulus is $\mu$. This basis is in millimetres per kilopascal: displacement in metres per pascal is multiplied by1000millimetres/metre and1000pascals/kilopascal. Thus $a$ is formally in kPa/year and positive values represent deflation. The chosen radius exceeds the depth, violating the small-cavity approximation. Treat this as a fixed empirical Mogi-shaped subsidence basis; its coefficient is not a measured geological pressure rate.

Save `output/reconstruction.npz` with all three float64 arrays:

| Output key | Shape | Units and axis order |
|---|---|---|
| `slip_rate` | `(1, 1299)` | mm/year;1292main triangles,5deep rectangles,east rectangle,west rectangle; right-lateral positive |
| `basin_coefficient` | `(1, 1)` | formal kPa/year multiplying the stated empirical deflation basis |
| `datum_offset` | `(1, 5)` | mm/year; radar tracks87,65,167,then GNSS east,north |

All entries must be finite. All1292main-fault values are assessed using cosine similarity without mean subtraction and RMS error divided by their reference dynamic range. Deep and remote slip rates, the basin coefficient and each datum are assessed separately using maximum absolute error in their declared units. Evaluation also checks the noise-weighted observation residual. There is no alignment, rescaling, cropping or magnitude conversion. The benchmark target is the conditional estimate defined in Method Hints, rather than independently measured subsurface truth.

To define the selected conditional target, concatenate the slips, basin coefficient and datums into $\boldsymbol m$. Minimize

```math
\frac12(\boldsymbol d-A\boldsymbol m)^T C_d^{-1}(\boldsymbol d-A\boldsymbol m)
+\frac12(\boldsymbol m-\boldsymbol m_0)^T C_m^{-1}(\boldsymbol m-\boldsymbol m_0),
\qquad A=[G\;\boldsymbol b\;T],\quad \boldsymbol m\geq\boldsymbol m_0.
```

The observation covariance is $C_d$. The prior vector $\boldsymbol m_0$ is zero for slips and the basin coefficient, and−100mm/year for each datum. The deep rates are tied exactly and have shared prior variance100(mm/year)². The two remote rates each have prior variance0.01(mm/year)²; basin and datum variances are1in their respective squared units.

The main-fault prior covariance between triangle centers $i,j$ is

```math
(C_m)_{ij}
=\frac{(20\ell_0)^2}{\ell_x\ell_z}
\exp\left(-\frac{H_{ij}}{2\ell_x}-\frac{V_{ij}}{2\ell_z}\right),
\qquad \ell_x=10\ \mathrm{km},\quad \ell_z=1\ \mathrm{km}.
```

The amplitude20has units mm/year. The horizontal and vertical center separations are $H_{ij}$ and $V_{ij}$. Each center is its three vertices' arithmetic mean. For each coordinate axis, divide the span of its distinct center values by the number of distinct values; $\ell_0$ is the Euclidean norm of these three spacings in km. All other prior cross-covariances are zero. These conditions specify a unique regularized estimate while retaining the uncertainty of the physical interpretation.

## Data Description

All arrays in `data/raw_data.npz` have a leading batch dimension of one.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `fault_triangles` | `(1, 1292, 3, 3)` | float64 | Known main-fault vertices; last axis east,north,positive-down depth in km |
| `deep_rectangles` | `(1, 5, 4, 3)` | float64 | Known deep-fault vertices in km, same coordinate convention |
| `remote_rectangles` | `(1, 2, 4, 3)` | float64 | Known east then west fault vertices in km |
| `sar_xy` | `(1, 3476, 2)` | float64 | Radar observation east/north coordinates in km |
| `sar_los` | `(1, 3476, 3)` | float64 | Dimensionless east,north,up line-of-sight projection vectors |
| `sar_velocity` | `(1, 3476)` | float64 | Curated radar line-of-sight velocities in mm/year |
| `sar_track_index` | `(1, 3476)` | int32 | Dimensionless track labels0,1,2 for87,65,167 |
| `sar_lonlat` | `(1, 3476, 2)` | float64 | Radar longitude/latitude in degrees |
| `gnss_xy` | `(1, 39, 2)` | float64 | Ground-station east/north coordinates in km |
| `gnss_lonlat` | `(1, 39, 2)` | float64 | Ground-station longitude/latitude in degrees |
| `gnss_velocity` | `(1, 39, 2)` | float64 | Curated ground-station east/north velocities in mm/year |
| `gnss_std` | `(1, 39, 2)` | float64 | Ground-station east/north standard deviations in mm/year |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Product | Three Sentinel-1 LOS velocity tracks and39GNSS stations | — | Existing curated/downsampled real observations |
| Radar track order | 87,65,167 | — | Contiguous blocks of1341,770,1365observations |
| Vector order | All radar rows,then39east,then39north GNSS values | — | The3554observation coordinates |
| Coordinate projection | WGS84 transverse Mercator; origin33°E,40.8°N; scale1; false easting/northing0 | degree,m | Supplied XY coordinates are already converted to km |
| Surface boundary | Observation depth0; traction-free ground | km | Homogeneous elastic half-space |
| Poisson ratio | 0.25 | dimensionless | Elastic response convention |
| Shear modulus | 30 | GPa | Formal subsidence basis |
| Basin center | East−30.343562481406668,north8.946455159060937 | km | Same projected coordinate frame |
| Basin depth/radius | 3000/6000 | m | Fixed empirical basis geometry |
| Radar correlated-noise amplitudes | 0.6000870639815176,0.19437212612551358,1.481715399840633 | mm/year | Track87,65,167,respectively |
| Radar correlation lengths | 2.073427161379951,1.5885688062110617,6.51170955211404 | km | Exponential spatial covariance |
| Radar total standard deviations | 2.8757212667307055,0.72886891566135,4.180839551249299 | mm/year | Set the covariance diagonal |
| Cross-track/GNSS noise correlation | Zero | dimensionless | Separate instrument blocks; independent supplied GNSS component errors |

For distinct pixels within one radar track, covariance is its amplitude squared times the exponential of negative horizontal separation divided by correlation length. Diagonal entries are the squared total standard deviations. GNSS covariance is diagonal with the supplied component variances. Positive LOS velocity means positive dot product with the supplied vector.

## Method Hints

Analytical elastic dislocation elements give the response of each known patch. A rectangle can be represented exactly by two consistently oriented triangles. Gaussian-prior generalized least squares and shifted nonnegative least squares provide useful comparisons.

## References

- [Daily to Centennial Behavior of Aseismic Slip Along the Central Section of the North Anatolian Fault](https://doi.org/10.1029/2022JB026018),2023. This benchmark selects the authors' simplified homogeneous-half-space tutorial estimator; it does not reproduce the article's layered-medium posterior analysis.
- [Algorithms for the calculation of exact displacements,strains,and stresses for triangular dislocation elements in a uniform elastic half space](https://doi.org/10.1016/j.cageo.2006.12.003),2007.
- [Internal deformation due to shear and tensile faults in a half-space](https://doi.org/10.1785/BSSA0820021018),1992.
