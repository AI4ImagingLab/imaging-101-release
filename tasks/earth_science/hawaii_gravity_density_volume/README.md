# Hawaii Gravity Density Volume

> Reconstruct a three-dimensional effective subsurface density-contrast field from irregular real gravity measurements whose depth sensitivity is broad and nonunique.
> Domain: earth science | Keywords: gravity imaging, volumetric inversion, spatial regularization | Difficulty: Hard

## Background

Spatial changes in density perturb the gravitational acceleration measured at the surface. Dense intrusions, volcanic structures and low-density deposits can therefore leave detectable gravity signatures even when they are hidden underground. Surveys record the sum of contributions from many depths and lateral positions.

Recovering a density volume from surface gravity is strongly nonunique: different deep and shallow structures can produce nearly the same acceleration. Latitude, station elevation and regional fields also affect the measurements. A useful interpretation must specify these reductions, the finite modeling domain and the spatial assumptions that select a stable inverse.

## Problem Description

Use 1019 real historical USGS stations in the Kilauea region of the island of Hawai'i. The task reconstructs an effective density contrast in a fixed sea-level-and-below rectangular prism volume. It is a conditional standard inverse under the stated reductions, not a claim that the recovered values are independently measured true geology or a reproduction of the historical survey's published geological model.

Let g be measured downward gravity in mGal, h orthometric station elevation in m, and latitude a in radians. Normal gravity is n(a)=978032.53359(1+0.00193185265241 sin(a)^2)/sqrt(1-0.00669437999013 sin(a)^2) mGal. Define the simple-Bouguer quantity b by

```math
b=g-n(a)+0.3086h-\left(2\pi G\rho_{\mathrm{B}}h\right)10^5.
```

Here G is the gravitational constant and rhoB=2670 kg/m^3. The factor 10^5 converts only the SI Bouguer-slab acceleration term to mGal. The linear height correction and infinite-slab Bouguer correction are adopted approximations; detailed terrain and curvature corrections are unavailable and are not silently assumed.

Set x=RE cos(a0)(longitude-longitude0), y=RE(latitude-latitude0), using angular differences in radians and the reference coordinates below. T has three columns [1,x/10000,y/10000]. P=I-T(T^T T)^(-1)T^T removes a regional plane. Apply P to both measured residuals and forward predictions, so an arbitrary nuisance plane is consistently marginalized:

```math
d=-Pb,\qquad d=PK\rho+\epsilon,\qquad
K_{ij}=10^8G\int_{V_j}\frac{z-z_i}{[(x-x_i)^2+(y-y_i)^2+(z-z_i)^2]^{3/2}}\,dV.
```

Here d is the upward residual gravity, rho is the vector of constant prism density contrasts in g/cm^3, V_j is prism j, (x_i,y_i,z_i) is station i, and epsilon represents observational and modeling discrepancies after projection. The factor 10^8 combines g/cm^3 to kg/m^3 and m/s^2 to mGal. Positive density below a receiver produces negative upward gravity. The Earth-radius projection is a local Cartesian approximation, not a geodetic reprojection requiring another library.

Save output/reconstruction.npz with key reconstruction, real float64 shape (1,8,16,16), axes batch,z,y,x, units g/cm^3. The z axis increases upward from the deepest cell center -18750 m to -1250 m. The y and x centers run from -22500 m to 22500 m, increasing north and east. Cells are 2500 m thick and 3000 m wide. All 2048 values are required. Signed positive and negative contrasts are allowed. No cropping, rescaling, coordinate shift or alignment is performed in evaluation. The full volume is scored using cosine NCC and RMS error divided by the reference dynamic range; the projected forward residual in mGal is also checked.

Gravity alone cannot identify a unique depth distribution, especially after regional-plane removal. Use spatial regularity to select a standard volume and explain the adopted model. The historical source metadata does not supply formal instrumental uncertainty estimates.

The requested conditional volume is the unique minimizer, over all signed cell contrasts, of

```math
\frac{\|PK\rho-d\|_2^2}{5^2}
+0.001\,V\left[\frac{\|\rho\|_2^2}{3000^2}
+\|D_x\rho\|_2^2+\|D_y\rho\|_2^2+\|D_z\rho\|_2^2\right].
```

Here the data residual is numerically expressed in mGal, its adopted scale is 5 mGal (a weighting convention rather than measured instrumental noise), and V=3000*3000*2500 m^3 is each cell volume. Each D is the adjacent-cell difference divided by its physical spacing in metres, counting each interior face once and contributing no exterior boundary row. The zero reference contrast defines the smallness term. Density is represented numerically in g/cm^3; these coefficients specify the conditional regularization scale and remove ambiguity among equally plausible depth distributions. No positivity constraint is imposed.

## Data Description

The visible arrays are selected unchanged from the original historical data table. No target density, truth model or fitted density-derived calibration is supplied.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| latitude | (1,1019) | float64 | WGS84 latitude, decimal degrees north. |
| longitude | (1,1019) | float64 | WGS84 longitude, decimal degrees east; west is negative. |
| elevation | (1,1019) | float64 | Orthometric elevation above sea level, m. |
| observed_gravity | (1,1019) | float64 | Original measured downward gravity, mGal; 1 mGal=10^-5 m/s^2. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Latitude selection | 19.25 < latitude < 19.55 | deg north | Native irregular survey region. |
| Longitude selection | -155.45 < longitude < -155.10 | deg east | Native irregular survey region. |
| Cartesian origin latitude | 19.4 | deg north | a0 for the local projection. |
| Cartesian origin longitude | -155.275 | deg east | longitude0 for the local projection. |
| Projection Earth radius | 6371000 | m | RE in x/y conversion. |
| Vertical origin | orthometric sea level | m | Station z=elevation; source prism top at z=0. |
| Volume x bounds | -24000 to 24000 | m | 16 uniform east-west cells. |
| Volume y bounds | -24000 to 24000 | m | 16 uniform south-north cells. |
| Volume z bounds | -20000 to 0 | m | 8 uniform bottom-to-top cells. |
| Cell widths x,y,z | 3000,3000,2500 | m | Fixed physical discretization. |
| Gravitational constant | 6.67430e-11 | m^3 kg^-1 s^-2 | Newtonian gravity convention. |
| Slab reduction density | 2670 | kg/m^3 | rhoB, equivalent to 2.67 g/cm^3. |
| Linear free-air coefficient | 0.3086 | mGal/m | Height-reduction approximation. |
| Normal-gravity equatorial value | 978032.53359 | mGal | n(a) convention. |
| Normal-gravity numerator coefficient | 0.00193185265241 | 1 | Somigliana latitude dependence. |
| Normal-gravity eccentricity squared | 0.00669437999013 | 1 | Ellipsoidal latitude convention. |
| Regional nuisance basis | 1,x/10000,y/10000 | 1 | Unweighted orthogonal plane projection, same on data and model. |
| Exterior model | zero density contrast outside the finite box | g/cm^3 | Does not assert absence of external geological structures. |
| Survey lineage | historical stations through 1999, published dataset 2017 | year | Earlier GS stations and later LaCoste-Romberg measurements. |

## Method Hints

The prism acquisition operator is linear in density, but broad depth sensitivity makes its inverse ill-conditioned. A weighted least-squares fit with cell-volume-weighted smallness and three spatial-gradient penalties selects a smooth finite-volume solution. Natural zero exterior normal derivatives provide a regularization boundary convention. Compute the prism kernel explicitly and include the same regional projection in its likelihood and gradient. A Newton step with a general conjugate-gradient linear solve is suitable for this quadratic objective; use physically scaled regularization rather than interpreting a shallow concentrated fit as uniquely determined geology.

## References

- Kauahikaua (2017), Gravity Data for Island of Hawai'i.csv, USGS data release. [Dataset](https://doi.org/10.5066/F7V1230Q). Public domain; source metadata states no access or use constraints.
- Kauahikaua, Hildenbrand and Webring (2000), Deep magmatic structures of Hawaiian volcanoes, imaged by three-dimensional gravity models. [Survey context](https://doi.org/10.1130/0091-7613(2000)28%3C883:DMSOHV%3E2.0.CO;2). The present fixed model is not asserted to reproduce that geological interpretation.
- Cockett et al. (2015), SimPEG: an open source framework for simulation and gradient based parameter estimation in geophysical applications. [Method framework](https://doi.org/10.1016/j.cageo.2015.09.015). MIT licensed numerical extraction; included notices preserve source attribution.
