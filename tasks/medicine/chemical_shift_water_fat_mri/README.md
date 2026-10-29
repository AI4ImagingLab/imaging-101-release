# Chemical-Shift Water–Fat MRI

> Reconstruct separate three-dimensional water and fat signal images, fat signal fraction, off-resonance field and transverse decay rate from three complex MRI echoes despite chemical-shift ambiguity and field inhomogeneity.
> Domain: medicine | Keywords: chemical shift imaging, quantitative MRI, spectral separation | Difficulty: Hard

## Background

Water and fat protons resonate at slightly different frequencies in a magnetic field. Recording complex MRI images at several echo times captures the resulting interference. Separating these signals supports assessment of tissue composition and visualization of anatomy that would otherwise be obscured by fat signal.

Fat has several spectral peaks rather than a single resonance. Spatial variation in the main magnetic field adds another phase evolution, while transverse relaxation decreases signal amplitude. These effects can resemble one another when only a few echoes are acquired. A spatially coherent field estimate helps distinguish chemical species, but low signal and rapid field variation can still cause water–fat swaps. Signal fraction is an imaging-model quantity and need not equal an independently calibrated tissue fat concentration.

## Problem Description

The observations are the complete author-supplied case17 demonstration volume from a fat–water reconstruction challenge. They contain three complex echoes and four adjacent spatial slices. No anatomical orientation, subject diagnosis or additional calibration scan is supplied. Reconstruct all five quantities below on the entire supplied grid.

Let the complex observation at spatial voxel position $\mathbf r$ and echo index $n=0,1,2$ be $s_n(\mathbf r)$. A first-echo-referenced signal model is

```math
s_n(\mathbf r)=e^{i2\pi f(\mathbf r)n\Delta t}
e^{-R(\mathbf r)(t_n-t_0)}
\left[W(\mathbf r)+C(\mathbf r)
\sum_{p=1}^{6}a_p e^{i2\pi\gamma B_0(c_p-c_w)t_n}\right]
+\epsilon_n(\mathbf r).
```

Here $i^2=-1$, $t_n=t_0+n\Delta t$ is echo time in seconds, $W$ and $C$ are complex water and fat spectral-coefficient images in arbitrary signal units, $R$ is the shared transverse decay rate in inverse seconds, and $f$ is the field offset in Hz. The main field $B_0$ is in tesla, $\gamma$ is the proton gyromagnetic ratio in MHz/T, and the chemical shifts $c_p,c_w$ are in ppm, so $\gamma B_0(c_p-c_w)$ is in Hz. The known relative fat-peak weights are $a_p$. Measurement error $\epsilon$ is complex; no voxelwise noise covariance is provided. The common phase at the first echo is absorbed into $W$ and $C$. The model excludes separate species decay rates, eddy-current corrections and an additional unknown background term.

The requested water amplitude is $|W|$. The requested aggregate fat amplitude is $|C\sum_p a_p|$, retaining the supplied peak weights without renormalizing them. The fat signal fraction is100 times aggregate fat amplitude divided by the sum of the water and aggregate fat amplitudes, with a numerical machine-epsilon denominator guard. It is not asserted to be a calibrated proton-density fat fraction. The requested fieldmap is $f/(\gamma B_0)$ in ppm. Equally spaced echoes make field estimates periodic with period $1/\Delta t$ Hz. Use the fixed representative interval $[0,1/(\gamma B_0\Delta t))$ ppm; no phase unwrapping or spatial shift is part of the output contract.

Save **`output/reconstruction.npz`** relative to the task workspace, containing every required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| water | (1, 101, 101, 4) | float32 | Nonnegative water coefficient magnitude, arbitrary signal units, referenced to the first-echo decay normalization. |
| fat | (1, 101, 101, 4) | float32 | Nonnegative aggregate fat coefficient magnitude including the sum of all supplied spectral weights, same signal units as water. |
| fat_fraction | (1, 101, 101, 4) | float32 | Fat signal fraction in percent, from0 to100. |
| fieldmap | (1, 101, 101, 4) | float64 | Field offset in ppm in the fixed alias interval above. |
| r2star | (1, 101, 101, 4) | float64 | Shared transverse signal-decay rate in s$^{-1}$. |

All arrays are real and finite, ordered **batch,x,y,z**, using the supplied voxel grid without cropping or padding. Spatial index order is preserved; anatomical axis orientation is unavailable. Complex amplitudes are internal nuisance quantities, while their magnitudes above are the specified outputs. Evaluation independently profiles the two complex spectral coefficients from all three raw echoes at the submitted field/decay. Submitted water and fat magnitudes must agree with these coefficients to maximum absolute error divided by the maximum observed echo magnitude at most 1e-6. Fat fraction must agree with the submitted magnitudes to 1e-4 percentage points. Thus every reported quantity is checked; a magnitude-only echo calculation cannot replace the complex acquisition model. The normalized full complex profiled SSE and the spatial field energy defined below may each exceed an attainable conditional fit by at most 1e-6. These are predeclared numerical objective budgets, not measured noise. The field uses 100 equally spaced labels on the stated single-period interval; decay is an integer from 0 through 144 s^-1. Alternative supported fits are accepted without voxelwise identity to one implementation. No independent tissue-composition accuracy is certified.

## Data Description

`data/raw_data.npz` contains only the observed complex echo images. No separate target image, fitted field or target-derived calibration is available as input.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| echoes | (1, 3, 101, 101, 4) | complex64 | Complex MRI signal in arbitrary signal units; axes batch,echo,x,y,z; stored real and imaginary parts jointly retain the acquisition phase convention. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Source example | Challenge case17, author-distributed observation volume | dimensionless | Full volume; anatomical orientation and subject details were not supplied in the example. |
| Echo times | 0.00287,0.00607,0.00927 | s | Three equally spaced complex echoes. |
| Echo spacing | 0.0032 | s | Difference between consecutive echo times. |
| Main magnetic field | 1.494 | T | Nominal field for chemical-shift conversion. |
| Proton gyromagnetic ratio | 42.576 | MHz/T | Multiplied by ppm chemical shifts to obtain Hz. |
| Voxel spacing | 1.5,1.5,5.0 | mm | Spacing along stored x,y,z axes. |
| Temperature convention | 37 | degrees Celsius | Known author-example model calibration; water shift follows1.3+3.748−0.01085 times temperature. |
| Water chemical shift | 4.64655 | ppm | Computed using the stated temperature convention, represented in float32 before spectral modeling. |
| Fat chemical shifts | 5.3,4.31,2.76,2.1,1.3,0.9 | ppm | Six fat resonances, represented in float32. |
| Fat relative weights | 0.048,0.039,0.004,0.128,0.693,0.087 | dimensionless | Retain original float32 weights and their supplied sum rather than renormalizing. |
| Precession convention | Positive complex exponential | dimensionless | Echo data are not conjugated before reconstruction. |
| Spatial boundary convention | Nonperiodic in x,y,z | dimensionless | Spatial neighbors do not wrap across volume edges. |
| Coil combination and intensity calibration | Not supplied | dimensionless | Absolute spin density cannot be inferred. |
| Noise covariance | Not supplied | squared signal unit | Use unweighted complex residuals; do not infer a hidden noise map. |

## Method Hints

Use variable projection to eliminate the linear complex water/fat coefficients for each candidate field and decay rate. The residual as a function of field has several minima, so choosing the best local residual independently can exchange water and fat in neighboring voxels. A spatial field-label energy with a smoothness penalty can compare competing minima jointly. Pairwise candidate costs must be supplied in the same voxel-endpoint order as their corresponding labels; interchanging endpoints without transposing the mixed-state costs changes the physical energy.

The mathematical spatial prior is fixed as follows. For label l=0,...,99, let f(l)=l/(100 Delta-t) Hz. Define J(l,v) as the minimum squared complex three-echo residual at voxel v over the two complex coefficients, with shared decay fixed at 40 s^-1. Let l*(v) be the lowest-index global minimum of J. Define periodic label curvature

```math
c_v=\max\{0,J(l^*_v-1,v)+J(l^*_v+1,v)-2J(l^*_v,v)\},\qquad
w_{uv}=0.1\,\min(c_u,c_v)/d_{uv},
```

where J indices wrap modulo 100, d is voxel separation in mm, and edges join immediate x/y/z neighbors only, once each, without boundary wrapping. A zero or roundoff-negative curvature gives zero weight; it supplies no artificial smoothing information. The static finest-lattice field energy is

```math
E(l)=\sum_v J(l_v,v)+\sum_{(u,v)}w_{uv}\min(|l_u-l_v|,100-|l_u-l_v|)^2.
```

Normalize E by the squared Euclidean norm of all observed complex echoes. This same norm normalizes the final profiled SSE at the submitted decay, whose complex coefficients are independently minimized at each voxel. The R=40 curvature prior and final integer decay domain are mathematical target conventions, not known tissue decay or optimizer stopping settings. Frequencies differing by an echo-sampling period describe identical modulation; report their canonical label in the required interval. Graph-label optimization and variable projection can solve the coupled discrete ambiguity without requiring a particular update sequence.

## References

- Berglund,J. *Multi-scale graph cut algorithm for water/fat separation*. Proceedings ISMRM23,3653(2015). [Complete conference paper](https://cds.ismrm.org/protected/15MProceedings/PDFfiles/3653.pdf).
- Berglund,J. and Skorpil,M. *Multi-scale graph-cut algorithm for efficient water-fat separation*. Magnetic Resonance in Medicine78,941–949(2017). [Journal article](https://doi.org/10.1002/mrm.26479).
