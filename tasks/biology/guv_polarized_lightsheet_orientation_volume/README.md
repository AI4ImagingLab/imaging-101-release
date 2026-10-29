# Three-dimensional dipole orientation from polarized light-sheet volumes

> Reconstruct a three-dimensional spatio-angular fluorophore distribution from registered polarized two-view light-sheet volumes despite optical blur and incomplete angular information.
> Domain: biology | Keywords: fluorescence microscopy, polarization imaging, inverse problems | Difficulty: Hard

## Background

Fluorophores absorb light preferentially when their transition dipoles align with the incident electric field. Polarization-dependent fluorescence therefore contains information about molecular organization as well as fluorophore abundance. This is useful for studying membrane labels, aligned fibers and cytoskeletal structures in three dimensions.

A light sheet excites a section of a specimen while an orthogonal objective records fluorescence. Repeating this measurement with different illumination polarizations, sheet tilts and viewing directions provides complementary orientation information. Finite numerical aperture blurs the spatial signal, and some angular combinations remain poorly measured even when many illumination states are available.

## Problem Description

The sample is an FM1-43-labelled giant unilamellar vesicle. The supplied observations have already been deskewed and registered to a common coordinate system. Use the complete author-defined analysis region; its physical extent is approximately10.01 by7.15 by10.40 micrometres. No independent molecular-orientation truth is supplied.

Let the spatial coordinate be $\mathbf r=(x,y,z)$, the unit dipole direction be $\mathbf s$, and the even-band angular distribution be

```math
f(\mathbf r,\mathbf s)=\sum_{j=0}^{14} f_j(\mathbf r)Y_j(\mathbf s).
```

Here $Y_j$ are orthonormal real spherical harmonics ordered by degree $\ell=0,2,4$, then order $m=-\ell,\ldots,\ell$. For the standard complex Condon–Shortley harmonics $\mathcal Y_{\ell m}(\theta,\phi)$, with polar angle $\theta$ from positive $z$ and azimuth $\phi$ from positive $x$ toward positive $y$, the real convention is

```math
Y_{\ell m}=\begin{cases}
\operatorname{Re}[\mathcal Y_{\ell m}+(-1)^m\mathcal Y_{\ell,-m}]/\sqrt{2},&m>0,\\
\operatorname{Re}\mathcal Y_{\ell0},&m=0,\\
\operatorname{Re}\left[\frac{\mathcal Y_{\ell m}-(-1)^m\mathcal Y_{\ell,-m}}{\sqrt{2}\,i}\right],&m<0.
\end{cases}
```

The last expression means the real part after division by $i=\sqrt{-1}$. In particular, $Y_0=1/\sqrt{4\pi}$. The coefficient fields have relative instrument-intensity units; integrating the distribution over directions gives $\sqrt{4\pi}f_0$. Absolute molecular concentration is not determined.

For a spatial frequency $\boldsymbol\nu$, selected polarization state $p$ and view $v$, the measurement model is

```math
G_{pv}(\boldsymbol\nu)=\sum_{j=0}^{14}H_{jpv}(\boldsymbol\nu)F_j(\boldsymbol\nu)+E_{pv}(\boldsymbol\nu).
```

$F_j$ and $G_{pv}$ are three-dimensional discrete Fourier transforms, and $E_{pv}$ represents measurement noise and model discrepancy. The forward kernel combines dipole illumination $3(\mathbf p\cdot\mathbf s)^2/(4\pi)$ with the finite-aperture detection response, projected onto the same real harmonics. For a normalized transverse frequency radius $r$, define $u=\min(|r|/2,1)$ and

```math
A_1(r)=\frac{2}{\pi}\big[\arccos u-u\sqrt{1-u^2}\big],\qquad
A_2(r)=\frac{2}{\pi}\big[\arccos u-(3-r^2/2)u\sqrt{1-u^2}\big].
```

For a detector along $z$, its nonzero angular coefficients are $d_0=A_1+\alpha^2A_2/4$ and $d_{20}=(-A_1+\alpha^2A_2/2)/\sqrt5$, where $\alpha=\mathrm{NA}/n$, $\mathrm{NA}$ is the detector numerical aperture and $n$ is the sample refractive index. For the other view rotate the detector by90 degrees about $y$, mapping its quadrupole coefficient into $d_{20}=-d_{20}^{(z)}/2$ and $d_{22}=\sqrt3d_{20}^{(z)}/2$. Each view also has an axial multiplier $\exp[-q^2/(2\sigma^2)]$, where $q$ is normalized axial frequency and $\sigma$ is its Gaussian width. Transverse and axial frequencies are normalized by the model wavelength divided by the corresponding numerical aperture. Each view's transverse angular transfer array is divided by its largest absolute entry.

The calibration lake is an independently recorded spatially and angularly uniform specimen. Fit its mean counts versus polarization angle with $a+b\cos(2\theta)+c\sin(2\theta)$ for each tilt/view, and use the mean fitted maximum angle as the polarization offset. The source convention locates each maximum on1000 equally spaced angles from0 to180 degrees. Divide observations by65535 before calibration. If $c_{pv}$ are the lake means and $h_{pv}$ are the unit-$f_0$ forward-model mean intensities normalized by their largest entry, calibrated data are $g_{pv}=g^{\rm raw}_{pv}c_{00}h_{pv}/c_{pv}$. This fixes a common relative scale; the global normalization differs from dividing the model intensities by their first entry by one known scalar.

The desired conditional estimator uses the ALL acquisition group, with the originally declared registration-problem state1 excluded in each view. All other20 states per view and all spatial voxels are retained. Spatial frequencies follow the full periodic DFT grid, including both partners of each odd-size conjugate pair. The target is the regularized coefficient estimate under this model; incomplete angular information prevents interpreting every component as independent biological truth.

Save `output/reconstruction.npz` with exactly this array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1,77,55,80,15) | float64 | Real signed coefficient fields in relative instrument-intensity units; axes batch,X,Y,Z,SH in the stated order. |

Every coefficient is evaluated over the full volume using cosine NCC, dynamic-range NRMSE and maximum absolute error. Predicted intensities must also explain the full calibrated observations. There is no fitted intensity rescaling, alignment, spatial mask or positivity clipping. Preserve signed coefficients and any negative angular lobes of the selected linear estimator.

For a flattened estimated coefficient field $x$ and its reference $y$, NCC is $x\cdot y/(\|x\|_2\|y\|_2)$ without mean subtraction, NRMSE is $\sqrt{\operatorname{mean}[(x-y)^2]}/(\max y-\min y)$, and maximum absolute error is $\max|x-y|$. The observation residual is $\|g_{\rm predicted}-g\|_2/\|g\|_2$, including every selected state and spatial voxel.

## Data Description

`data/raw_data.npz` contains only the registered observations and same-session calibration counts:

This author archive records seven polarizer settings per tilt, including separate0 and180 degree measurements of the same nominal orientation. It therefore contains42 recorded volumes rather than the paper's36 distinct nominal configurations. The selected author analysis retains40 recorded volumes after its declared registration exclusion; redundant orientations are not silently collapsed or counted as additional independent angular information.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observations | (1,77,55,80,21,2) | uint16 | Real registered camera counts, in digital counts; batch,X,Y,Z,state,view. |
| lake_calibration | (1,30,30,30,21,2) | uint16 | Real uniform fluorescent-slide calibration counts; batch,X,Y,Z,state,view, in digital counts. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Specimen | FM1-43-labelled GUV | — | Vesicle orientation experiment. |
| Observation provenance | S-BIAD1055,20200116_GUV/Sample4/Reg | — | Author-provided registered/deskewed real volumes. |
| Analysis ROI | X[128:205],Y[705:760],Z[56:136] | registered voxel | Original author's fixed zero-based half-open analysis region. |
| Voxel spacing |130,130,130 | nm | Common registered X,Y,Z spacing. |
| Observation views | A then B | — | View A detector axis positiveZ; view B detector axis positiveX in the analysis model. |
| Detection NA |1.1,0.67 | dimensionless | Selected author analysis model, A then B. |
| Sample refractive index |1.33 | dimensionless | Known immersion/sample convention. |
| Model wavelength |561 | nm | Author's effective emission/transfer-function wavelength, not a measured spectral reconstruction. |
| Axial Gaussian sigma |0.25 | normalized frequency | Each view's axial transfer multiplier. |
| Polarizer angles |0,45,60,90,120,135,180 | degree | Seven states per tilt before subtracting the lake-fitted offset. |
| Tilt order |0,+1,−1 | — | States0:7,7:14,14:21 respectively. |
| Measured sheet tilts A |0,+8.9,−9.2 | degree | Direction rotations about Z. |
| Measured sheet tilts B |0,+7.7,−6.5 | degree | Direction rotations about X. |
| Polarization at tilt0 |A:(0,cosθ,−sinθ);B:(−sinθ,cosθ,0) | unit vector | θ is the polarizer angle after offset correction. |
| Polarization at tiltδ |A:(sinδcosθ,cosδcosθ,−sinθ);B:(−sinθ,cosδcosθ,sinδcosθ) | unit vector | Normalize each direction before evaluating the dipole response. |
| Used ALL states |0 and2–20 in each view | zero-based index | State1 has the author's declared registration issue and remains visible in the input. |
| Calibration acquisition |20200116_Calibration/LS_488 | — | Same-session independent fluorescent-slide acquisition; central30³ author analysis cube. |
| Known noise variance |Not supplied | digital counts² | No Gaussian/Poisson variance calibration is invented. |
| Boundary convention |Periodic spatial DFT; complete signed X/Y frequencies | — | No extra padding or implicit crop; output on the exact author ROI grid. |

## Method Hints

Use a spherical-harmonic decomposition to separate angular sensitivity from the spatial Fourier transform. Products of illumination and detection harmonics can be evaluated by their triple-product surface integrals. The two orthogonal views provide complementary angular and axial sensitivity, while sheet tilting reduces orientation blind spots.

A Tikhonov-regularized singular-system inverse gives the selected conditional solution. With $H=U\operatorname{diag}(s)V^*$ in coefficient-by-measurement orientation, apply $U\operatorname{diag}[s/(s^2+\eta)]V^*$ to each measured Fourier vector; $s$ are nonnegative singular values, $V^*$ is the conjugate transpose, and the ALL estimator uses $\eta=0.7$. Singular values at or below $10^{-7}$ contribute zero. This defines the reference convention in weak or null angular directions without claiming recovery of unmeasured molecular structure.

## References

1. Chandler et al. *Volumetric imaging of the3D orientation of cellular structures with a polarized fluorescence light-sheet microscope*. Proceedings of the National Academy of Sciences (2025), including its supplementary model and calibration description. [Published paper](https://doi.org/10.1073/pnas.2406679122).
2. [BioImageArchive S-BIAD1055](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1055), author-shared real imaging/calibration data and published analysis ROI, CC BY4.0.
