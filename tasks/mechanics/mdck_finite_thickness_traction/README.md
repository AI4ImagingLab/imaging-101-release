# MDCK Finite-Thickness Traction Microscopy

> Reconstruct a cell colony's signed in-plane traction field from experimentally measured substrate displacements despite elastic spatial spreading and amplified measurement noise.
> Domain: mechanics | Keywords: elastic imaging, traction microscopy, inverse mechanics | Difficulty: Hard

## Background

Adherent cells transmit force to their surroundings through molecular attachments. These forces influence migration, wound closure, tissue organization and the response of cells to mechanical environments. A compliant substrate acts as a mechanical sensor: cell forces deform it, and fluorescent particles embedded in the material reveal the deformation.

The displacement at a point depends on forces across a surrounding region, because elastic stresses spread through the substrate. Traction microscopy converts the measured deformation into the surface forces that caused it. Accounting for substrate thickness matters when deformation wavelengths approach the thickness of the bonded elastic layer.

## Problem Description

The input is a complete measured displacement field from an MDCK epithelial cell colony on a planar elastic substrate. Displacements were measured by cross-correlation PIV of fluorescent-bead images before and after cell detachment; this task starts from those displacement observations and reconstructs tractions, not image registration or the cell sheet's internal stress.

In Fourier space the linear elastic observation model is

```math
\widehat{\boldsymbol u}(\boldsymbol k)
=\boldsymbol G(\boldsymbol k;E,\nu,h)\widehat{\boldsymbol t}(\boldsymbol k)
+\widehat{\boldsymbol\epsilon}(\boldsymbol k),
\qquad \boldsymbol G=\left[\alpha_\perp(\boldsymbol I-\boldsymbol n\boldsymbol n^T)
+\alpha_\parallel\boldsymbol n\boldsymbol n^T\right]^{-1}.
```

Here $\boldsymbol u=(u_x,u_y)$ is the surface displacement in µm, $\boldsymbol t=(t_x,t_y)$ is surface traction in Pa, $\boldsymbol k=(k_x,k_y)$ is angular spatial frequency in rad/µm, a hat denotes the discrete Fourier transform, $\boldsymbol I$ is the 2×2 identity, and $\boldsymbol n=\boldsymbol k/|\boldsymbol k|$. The known elastic parameters are Young's modulus $E$, Poisson ratio $\nu$ and layer thickness $h$. For nonzero $k=|\boldsymbol k|$, let $r=kh$ and

```math
\alpha_\perp=\frac{Ek}{2(1+\nu)\tanh r},\qquad
\alpha_\parallel=\frac{Ek\,\gamma}{2(1-\nu^2)},\qquad
\gamma=\frac{(3-4\nu)+(1-2\nu)^2\operatorname{sech}^2r+r^2\operatorname{sech}^2r}
{(3-4\nu)\tanh r+r\operatorname{sech}^2r}.
```

Both $\alpha$ values have units Pa/µm. The substrate is isotropic and linearly elastic, with its lower surface bonded and its upper surface loaded in-plane. Out-of-plane traction and displacement are excluded. Rigid image translation is uninformative and is removed; zero-frequency traction is fixed to zero. High spatial frequencies amplify displacement noise, requiring a physically stated smoothing convention. Finite field of view and noise prevent exact force recovery.

Save `output/reconstruction.npz` with the following array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 2, 48, 49) | float64 | Real signed traction in Pa; axes batch, component (x,y), increasing physical y, increasing physical x; retain absolute force scale. |

Evaluation jointly compares both signed vector components using uncentred cosine NCC and dynamic-range NRMSE: RMS error divided by the maximum minus minimum of the comparison traction. No centring, cropping, alignment, vector-magnitude conversion, or fitted rescaling is applied. This real-data task measures agreement with a standard finite-layer interpretation rather than independently known true forces. Use a 6.0 µm Gaussian standard deviation in the final traction field to define the spatial resolution of the requested interpretation.

## Data Description

`data/raw_data.npz` contains measured observations only:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| displacement | (1, 2, 48, 49) | float64 | Measured ux,uy in original fluorescent-image pixels; multiply by 0.201 µm/pixel. Axes batch, component (x,y), increasing physical y, increasing physical x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Fluorescent-image pixel size | 0.201 | µm/pixel | Metric calibration of measured displacement amplitudes. |
| Displacement sample pitch | 8.040000000000001 | µm | Stored float64 value of 40 × 0.201; equal x/y spacing. |
| PIV interrogation width | 100 | image pixels | Gaussian subpixel cross-correlation of bead image pairs. |
| PIV overlap | 60 | image pixels | Adjacent measurement centres differ by 40 image pixels. |
| Original bead-image size | 1989 × 2034 | image pixels | Entire experimental image pair underlying the observed field. |
| Displacement array size | 48 × 49 | grid samples | Entire PIV measurement field, without selecting a favourable region. |
| Young's modulus E | 49000.0 | Pa | Polyacrylamide substrate elasticity. |
| Poisson ratio ν | 0.49 | dimensionless | Nearly incompressible substrate. |
| Layer thickness h | 300.0 | µm | Bonded elastic substrate thickness. |
| Coordinates | x right; y upward | dimensionless | Displacement field rows increase physically upward; original image rows were reversed without changing signed component values. |
| Origin | Lower-left displacement sample | dimensionless | Translation-invariant elastic model; absolute image origin is not required. |
| Noise | No calibrated pointwise variance | image pixels | PIV peak-ratio validation and local-mean outlier replacement precede the supplied observation. |
| Boundary convention | Zero-padded even square; periodic FFT | dimensionless | Place data at the final rows/columns of a 50×50 grid, then crop to the same support. |

The physical y convention and exact PIV sampling step are stated explicitly. This calibrated application of the published elastic method does not claim an unchanged pixelwise reproduction of a tutorial that uses approximate image-size-based sampling.

## Method Hints

Finite-layer Fourier traction cytometry diagonalizes the isotropic elastic response into modes parallel and perpendicular to spatial frequency. Remove component-wise mean displacement, invert these mode responses, then apply Gaussian smoothing at the declared physical resolution. Treat the Fourier zero mode separately and preserve signed component conventions. On the50-by50 real discrete operator, set the mixed spectral tensor entries to zero along either Nyquist axis except their joint corner, whose physical tensor value is retained. After cropping to48-by49, the6 micrometer Gaussian resolution is a separable normalized sampled kernel with sigma=6/8.040000000000001 grid samples and integer offsets -3 through3. Use half-sample symmetric reflection at the cropped field edges (the edge sample is repeated), without filtering across vector components. Compare the finite-layer result with the infinite-half-space limit to assess the importance of substrate thickness.

## References

Bauer, A., Prechová, M., Fischer, L., Thievessen, I., Gregor, M., and Fabry, B. (2021). *pyTFM: A tool for traction force and monolayer stress microscopy*. PLOS Computational Biology 17(6), e1008364. [doi:10.1371/journal.pcbi.1008364](https://doi.org/10.1371/journal.pcbi.1008364).

Bauer, A. (2020). Experimental data for pyTFM. [doi:10.5281/zenodo.4047040](https://doi.org/10.5281/zenodo.4047040). Data and derived displacement observations are distributed under CC BY-SA 4.0.
