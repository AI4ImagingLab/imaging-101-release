# Mean-Signal Diffusion Kurtosis Imaging

> Reconstruct maps of diffusion, non-Gaussian displacement kurtosis, and baseline signal from real multi-shell human-brain diffusion MRI despite directional variation and low high-b signal.
> Domain: medicine | Keywords: diffusion imaging, tissue microstructure, inverse estimation | Difficulty: Hard

## Background

Diffusion MRI probes water displacement through magnetic-field gradient encoding. Freely diffusing water produces approximately exponential attenuation with diffusion weighting. Membranes, cellular compartments, and heterogeneous tissue environments cause departures from this simple behavior, creating sensitivity to tissue microstructure without invasive labeling.

The orientation of axon bundles strongly affects directional signals. Averaging diffusion-weighted measurements across encoding directions offers rotationally invariant contrast and can reduce dependence on fiber crossings. Such measurements are useful for studying brain tissue organization, aging, and disease, while their interpretation remains limited by noise, finite diffusion times, and model assumptions.

## Problem Description

The observations are three complete neighboring native slices from a real human-brain acquisition. Each diffusion shell contains 33 encoding directions. The acquisition includes one zero-weighting volume and fifteen nonzero shells. The unknowns are mean-signal diffusivity D, mean-signal kurtosis K, and the extrapolated zero-weighting amplitude S0 at every voxel. These differ from a directional diffusion tensor or an orientation distribution.

For b in seconds per square millimeter, the powder-averaged signal follows the finite-b cumulant approximation:


```math
\overline S(b)=S_0\exp\left(-bD+\frac{b^2D^2K}{6}\right).
```


Sbar is the arithmetic mean over directions in a shell, S0 is in detector ADU, D is in square millimeters per second, and K is dimensionless. Measurements are magnitude MRI data; noise bias, motion, and partial volumes can violate the expansion. The expansion is not an exact high-b tissue compartment model. Shell means must preserve the deposited b-value convention. The supplied directions provide the experimental angular sampling.

Save **output/reconstruction.npz** with all three real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 3, 96, 96) | float64 | Mean-signal kurtosis K, dimensionless; axes batch, ordered native slice, native x, native y |
| diffusivity | (1, 3, 96, 96) | float64 | Mean-signal diffusivity D, mm squared per second; identical axes |
| baseline | (1, 3, 96, 96) | float64 | Fitted zero-weighting S0, ADU; identical axes |

Define an observation-only tissue support as the recorded b=0 signal greater than 40 ADU, before any model fit. Set all three outputs to zero outside this support. Preserve every fit within this support, including negative kurtosis caused by noise or model mismatch; no fitted-value screening or clipping is authorized. Each complete array is evaluated by relative Euclidean error against an independently reproduced standard-method result. The fitted forward model must also reproduce the observed shell means within a fixed absolute root-mean-square residual in ADU over the same public support. No registration, crop, rescaling, or alignment is applied.

## Data Description

All 496 diffusion encodings are included. Spatial slices are original zero-based indices 8, 9, and 10; the full 96 by 96 detector field is retained. Data are the deposited magnitude observations without additional denoising, motion correction, resampling, or synthesized noise.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| signals | (1, 3, 96, 96, 496) | int16 | MRI magnitude observations, ADU; axes batch, native slice, native x, native y, encoding |
| bvalues | (1, 496) | float64 | Diffusion weighting in s per mm squared, matched to encoding axis |
| gradient_directions | (1, 496, 3) | float64 | Unitless Cartesian encoding direction components from deposited table; zero for b=0 |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Scanner | Siemens Trio | dimensionless | Human 3 T system |
| Field strength | 3 | tesla | Static magnetic field |
| Receiver | 32-channel head coil | channel | Acquisition receiver |
| Sequence | Double-spin-echo diffusion EPI with CSF suppression | dimensionless | Magnitude image acquisition |
| b shells | 0; 200 through 3000 in increments of 200 | s/mm squared | One b=0 image and 33 directions at every nonzero shell |
| Voxel spacing | 2.5, 2.5, 2.500000476837158 | mm | Stored native NIfTI x,y,z spacing, including original precision |
| Repetition time | 7200 | ms | TR |
| Echo time | 116 | ms | TE |
| Inversion time | 2100 | ms | CSF suppression TI |
| Original volume | 96 by 96 by 19 | voxel | Native geometry |
| Included slices | 8, 9, 10 | zero-based index | Three central native planes, no in-plane crop |
| Phase encoding | Anterior to posterior | dimensionless | EPI acquisition direction |
| Angular sampling | 24-point spherical design plus 9 fast-kurtosis directions | dimensionless | 33-direction shell sampling; use supplied vectors |
| Reported b=0 SNR | Approximately 39 | dimensionless | Depositor tissue-average estimate; no per-voxel noise standard deviation supplied |
| Spatial boundary | Independent voxel estimation | dimensionless | No spatial interpolation or wraparound |

Native voxel indices (i,j,k) map to world millimeters through the acquisition affine below; k takes values 8,9,10 for the output slices. Vector components follow the deposited encoding table and are used only for identifying directional sampling; the powder-average inverse is rotationally invariant.


```math
\begin{pmatrix}X\\Y\\Z\\1\end{pmatrix}=
\begin{pmatrix}
-2.497706413269043&0.08375494927167892&0.06668981164693832&113.64055633544922\\
0.069154292345047&2.4509189128875732&-0.4880725145339966&-104.14154052734375\\
0.0817319005727768&0.48577991127967834&2.4509875774383545&-31.504009246826172\\
0.0&0.0&0.0&1.0
\end{pmatrix}
\begin{pmatrix}i\\j\\k\\1\end{pmatrix}.
```


X,Y,Z denote world coordinates; i,j,k denote native zero-based voxel indices. Absolute orientation is not an evaluation alignment degree of freedom.

## Method Hints

The evaluated estimator is fixed weighted log-linear least squares. For each exact b shell let n(b) be its number of measurements and y(b) the arithmetic shell mean floored at 0.0001 ADU. Minimize sum over all 16 shells of n(b)y(b)²[log(y(b))-(-b D+b² P/6+c)]² over the three unconstrained coefficients D,P,c. Then report K=P/D² and S0=exp(c), retaining signed K. Weights use the observed means once and are not repeatedly updated from fitted signals; there is no sample-count normalization or spatial penalty. This objective selects the standard conditional estimate and differs from unweighted log fitting or nonlinear signal-domain least squares.

Use the mean-signal diffusion-kurtosis family: powder average within each shell and fit the quadratic expansion of log signal. A weighted linear estimate accounts for the number of directional measurements and the changing uncertainty after the logarithm. This method estimates D, D squared times K, and log S0 together. The distinction between non-Gaussian signal curvature and a tissue compartment fraction matters: the former is requested here. A lower-order Gaussian attenuation fit and alternate shell subsets are informative checks of model adequacy.

## References

Henriques, R. N., et al. *Diffusional Kurtosis Imaging in the Diffusion Imaging in Python Project*. Frontiers in Human Neuroscience 15, 675433 (2021). [Publication](https://doi.org/10.3389/fnhum.2021.675433).

Hansen, B., and Jespersen, S. N. *Data for evaluation of fast kurtosis strategies, b-value optimization and exploration of diffusion MRI contrast*. Scientific Data 3, 160072 (2016). [Publication](https://doi.org/10.1038/sdata.2016.72). [Deposited data](https://doi.org/10.5061/dryad.9bc43), CC0.
