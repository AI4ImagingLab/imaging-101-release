# Aluminium Crack-Tip Stress and Stress Intensity

> Reconstruct the elastic stress field and two stress-intensity factors near a fatigue crack from measured surface displacements, despite differentiation noise and finite sampling.
> Domain: mechanics | Keywords: fracture mechanics, displacement imaging, inverse elasticity | Difficulty: Hard

## Background

Optical displacement measurements follow the motion of a specimen's surface during loading. In a thin elastic specimen, the measured in-plane deformation contains information about the stresses driving crack growth. This supports fatigue assessment and experiments on lightweight structural materials.

Direct differentiation amplifies measurement noise. Near a crack, stresses vary sharply with distance, while the two crack faces permit displacement discontinuities. Elastic crack-tip fields provide a physically constrained connection between measured motion and the unobserved stress field.

## Problem Description

This task uses one measured aluminium fatigue-crack nodemap. Infer the in-plane elastic stress field and opening/sliding stress-intensity factors in the specified crack coordinate system. The known crack-tip position and orientation define this coordinate system; they are geometric calibration, not quantities to infer. An isotropic plane-stress model applies.

Let Cartesian position relative to the crack tip be $(x,y)=(r\cos\phi,r\sin\phi)$, where $r$ is in mm and $\phi$ in radians. Let $u$ and $v$ denote displacement along these axes in mm. A finite elastic eigenfield expansion relates measured displacements to unknown coefficients $a(n)$ and $b(n)$ for orders $n\in\lbrace-1,1,2,3,4,5\rbrace$. Define $h=n/2$, $s=(-1)^n$, $G=E/[2(1+\nu)]$ and $\kappa=(3-\nu)/(1+\nu)$, where $E$ is Young's modulus and $\nu$ Poisson's ratio. The forward displacement components are sums over the listed orders:

```math
u(r,\phi)=\sum_n\frac{r^h}{2G}\left[a(n)\left((\kappa+s+h)\cos(h\phi)-h\cos((h-2)\phi)\right)+b(n)\left((-\kappa+s-h)\sin(h\phi)+h\sin((h-2)\phi)\right)\right],
```

```math
v(r,\phi)=\sum_n\frac{r^h}{2G}\left[a(n)\left((\kappa-s-h)\sin(h\phi)+h\sin((h-2)\phi)\right)+b(n)\left((\kappa+s-h)\cos(h\phi)+h\cos((h-2)\phi)\right)\right].
```

Rigid translation contributes an additional constant displacement and does not affect stress. The desired stresses obey plane-stress elasticity: $\sigma_{xx}=E(\partial_xu+\nu\partial_yv)/(1-\nu^2)$, $\sigma_{yy}=E(\nu\partial_xu+\partial_yv)/(1-\nu^2)$, and $\sigma_{xy}=G(\partial_yu+\partial_xv)$. They are in MPa. Coefficient units are $\mathrm{MPa}\,\mathrm{mm}^{1-n/2}$. The singular-order coefficients give $K_I=\sqrt{2\pi/1000}\,a(1)$ and $K_{II}=-\sqrt{2\pi/1000}\,b(1)$ in $\mathrm{MPa}\sqrt{\mathrm m}$. The negative sign fixes the sliding-mode convention. Finite noisy sampling and overlapping eigenfields make the inversion sensitive to truncation and fitting geometry.

Save `output/reconstruction.npz` with both required real arrays, using float32 or float64:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 3, 231, 559) | float32 or float64 | Stress in MPa; axes batch, component, radius, azimuth; component order xx, yy, xy in the crack frame |
| stress_intensity | (1, 2) | float32 or float64 | Opening then sliding intensity in MPa sqrt(m) |

Output sampling is fixed: radius index $i=0,\ldots,230$ uses $r=0.7695+0.01i$ mm; azimuth index $j=0,\ldots,558$ uses $\phi=-\pi+\pi/9+0.01j$ rad. Every output value must be finite. Evaluation compares both complete arrays using cosine similarity (without mean subtraction) and RMS error divided by each reference array's full dynamic range. No alignment, scaling, cropping or component permutation is applied. Reference agreement measures consistency with the selected elastic standard method; it does not establish independently measured stress accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| coordinates | (1, 44581, 2) | float64 | Measured point positions in the laboratory frame, x then y, in mm |
| displacements | (1, 44581, 2) | float64 | Measured in-plane displacement vectors, laboratory x then y, in mm |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Aluminium fatigue-crack test, right crack tip, one nodemap | — | Full supplied valid optical point set |
| Sensor / camera | ARAMIS 12M (FG), 12MP | — | Author nodemap acquisition header |
| Focal length / distance | 50.0 / 315.0 | mm / mm | Camera geometry reported in author header |
| Camera angle | 24.992663911979694 | degree | Camera geometry reported in author header |
| Facet size / distance | 19 / 15 | pixel / pixel | DIC sampling settings |
| Exposure time | 0.002997 | s | Stage 52, reference stage 1 |
| Tip position | (15.39, 0.40) | mm | Laboratory coordinates |
| Tip orientation | -0.27 | degree | Crack-axis angle from laboratory x |
| Coordinate transformation | Translate by tip, then rotate vectors and positions by +0.27 degrees | degree | Crack x points forward; crack faces lie on negative x |
| Young's modulus E | 72000 | MPa | Isotropic elastic calibration |
| Poisson ratio nu | 0.33 | dimensionless | Isotropic elastic calibration |
| Mechanical state | Plane stress | — | Thin-sheet surface model |
| Crack-face condition | Zero normal and shear traction | MPa | Elastic eigenfields satisfy this away from the tip |
| Inner / outer radius | 0.7695 / 3.078 | mm | Annular evaluation region, outer endpoint excluded |
| Crack-face exclusion | 20 | degree | Angular gap from each face |
| Radius / angle increments | 0.01 / 0.01 | mm / rad | Evaluation-grid sampling |
| Noise distribution | Not supplied | — | Measured data; no invented noise variance |

## Method Hints

An overdetermined linear fit of elastic crack-tip eigenfields can replace unstable direct differentiation. The reference estimator uses unweighted least squares on the declared polar grid: interpolate the scattered displacement components linearly within their Delaunay triangles and subtract the interpolated tip displacement. Exclude grid points outside the observed convex hull rather than extrapolating them. Handle coordinate rotation consistently, then derive stresses analytically from the fitted displacement expansion. Higher and supersingular orders capture finite-region effects; retaining only the leading singular field can bias the inferred stress intensities. A rank-revealing decomposition helps diagnose numerical degeneracy.

## References

Strohmann et al., *Next generation fatigue crack growth experiments of aerospace materials*, Scientific Reports 14, 14075 (2024). https://doi.org/10.1038/s41598-024-63915-x

M. L. Williams, *On the stress distribution at the base of a stationary crack*, Journal of Applied Mechanics 24, 109–114 (1957). https://doi.org/10.1115/1.4011454
