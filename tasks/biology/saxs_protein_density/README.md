# Protein Density from Solution X-Ray Scattering

> Reconstruct a normalized three-dimensional protein electron-density map from real solution scattering intensities despite loss of Fourier phases and molecular orientation information.
> Domain: biology | Keywords: solution scattering, phase retrieval, molecular imaging | Difficulty: Hard

## Background

X-rays scatter from the spatial distribution of electrons in a molecule. In a dilute solution, molecules tumble through many orientations, so the detector records an orientationally averaged scattering pattern. Such measurements study biological macromolecules in solution without requiring a crystal and can reveal molecular size, overall organization and changes of conformation.

Orientation averaging removes much of the information present in a three-dimensional diffraction measurement. Fourier phases are also unavailable. A single solution scattering curve therefore cannot determine a unique three-dimensional molecule. Reconstruction uses physical constraints to find density distributions consistent with the measurements; fine structural features require independent experimental evidence.

## Problem Description

The sample is lysozyme in solution. Reconstruct a complete three-dimensional normalized contrast electron density from the supplied real experimental curve and its observation-derived smooth fit. The forward model is


```math
F(\mathbf q)=\int \rho(\mathbf r)\exp(-i\mathbf q\cdot\mathbf r)\,d^3\mathbf r,
\qquad
I(q)/I(0)=\frac{1}{4\pi}\int_{S^2}|F(q\boldsymbol\omega)|^2\,d\boldsymbol\omega,
\qquad \int\rho(\mathbf r)\,d^3\mathbf r=1.
```


Here position $\mathbf r$ is in angstroms, spatial frequency $\mathbf q$ is in inverse angstroms, $q=|\mathbf q|=4\pi\sin(\theta)/\lambda$, $2\theta$ is scattering angle and $\lambda$ the X-ray wavelength. Unit vector $\boldsymbol\omega$ indexes directions. Density $\rho$ is in inverse cubic angstroms under unit integrated-density normalization; absolute electron count and concentration are not supplied. Intensity $I$ and its standard uncertainty are in the dataset's arbitrary intensity unit. Independent Gaussian errors with the supplied pointwise standard deviations define the scattering residual convention; full noise covariance was not supplied.

The acquisition-derived smooth curve supplies the unmeasured zero-angle limit and a maximum dimension estimate. The latter guides compactness but is not an exact support mask. A large periodic computation box embeds a compact particle; its exterior density should be small. At frequencies beyond the measurement, a declared $q^{-4}$ intensity continuation is a regularizing assumption, not extra measured information. The forward evaluation averages all Cartesian Fourier samples in each spherical shell. Shell boundaries have spacing $2\pi/240-10^{-8}$ inverse angstroms and start at zero; each representative frequency is the mean radius of the shell samples. Intensities on these shell coordinates are cubically interpolated to all measured frequencies, with the same cubic extrapolation over the small final partial shell. No fitted intensity multiplier or background offset is used after normalization by the supplied smooth curve's $I(0)$.

Save **`output/reconstruction.npz`** relative to the task workspace with key **`reconstruction`**, a finite real **float64** array of shape **`(1, 64, 64, 64)`**, ordered **batch, x, y, z**, in **angstrom$^{-3}$**. Grid coordinate index $j$ corresponds to $(j-32)3.75$ angstrom along each axis. Voxel volume is $3.75^3$ cubic angstroms; integrate by multiplying the density sum by that volume. Preserve the signed final density; the standard method's final Fourier projection can reintroduce small negative tails and density outside its real-space support. These are quantitatively evaluated, not silently clipped away.

Evaluation also compares the complete predicted scattering curve with the standard-method scattering reference at every measured frequency: the RMS discrepancy divided by the reference curve dynamic range must meet a fixed nonzero numerical tolerance. This invariant comparison does not impose the reference map's undetermined spatial phase. Evaluation checks the complete map through its all-measured-frequency standardized scattering residual, integrated normalization, negative-to-positive mass ratio, compactness, connected high-density core, declared high-frequency continuation and scattering radius of gyration. The scattering residual is the mean squared residual divided by the supplied variance after fixed zero-angle normalization. The radius check compares $\sqrt{-3\log(I(q_1)/I(0))}/q_1$ at the first nonzero shell $q_1$ against a weighted straight-line fit of $\log I$ versus $q^2$ over actual measured $q\leq0.08$ inverse angstroms, with weights $I/\sigma$. Here $\sigma$ denotes the supplied intensity standard deviation.

For compactness only, an independent coordinate gauge first moves the largest absolute-density voxel to index32 on every axis by integer periodic translation, then moves the absolute-density center of mass to that center by rounded integer translation. The positive mass beyond radius40 angstrom is measured as a soft support residual. A core is defined by a periodic Gaussian of the positive density with standard deviation1.5 voxels and threshold20% of its maximum; the fraction of its positive mass outside its largest 26-connected component is evaluated. These checks use no reference molecular model. The high-frequency criterion is the RMS relative discrepancy from the stated $q^{-4}$ continuation over every shell above the measured range.

**Non-uniqueness is part of the problem.** Translation, rigid orientation and inversion of handedness are not determined by the curve, and more general distinct three-dimensional densities can share it. No comparison against one random phase-retrieval map defines shape truth. Passing establishes scattering/physical consistency at this resolution; it does not establish a unique molecular structure, atomic coordinates or correct internal atomic detail.

## Data Description

The complete supplied profile contains485 frequencies. Of these468 have experimental measurements;17 low-frequency entries are a smooth extrapolation only. The mask separates them explicitly. The smooth fit was obtained from the same experimental curve by indirect Fourier transformation before density reconstruction. It is an observation-derived preprocessing result and contains no hidden spatial model.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| q | (1, 485) | float64 | Scattering-vector magnitude in angstrom$^{-1}$, including extrapolated low-frequency samples. |
| intensity | (1, 485) | float64 | Experimental intensity in arbitrary intensity units; zero placeholders where measured_mask is false are not measurements. |
| sigma | (1, 485) | float64 | Experimental one-standard-deviation uncertainty in the same intensity unit; extrapolated entries carry the source's mean early uncertainty and are not used as measured observations. |
| smoothed_intensity | (1, 485) | float64 | Observation-derived GNOM smooth intensity in the same arbitrary intensity unit, including zero-angle extrapolation. |
| measured_mask | (1, 485) | bool | Dimensionless indicator of actual measurement; false marks only the extrapolated low-frequency entries. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Lysozyme solution | dimensionless | Author-provided real SAXS example |
| Actual measured q range | 0.00982 to 0.2793 | angstrom$^{-1}$ | All468 measured points remain evaluated |
| Profile q range including extrapolation | 0 to 0.2793 | angstrom$^{-1}$ | All485 supplied samples |
| Zero-angle smooth intensity | 0.04591 | arbitrary intensity unit | Fixed normalization value from smoothed_intensity at q=0 |
| Maximum dimension estimate | 40.0 | angstrom | Observation-derived GNOM estimate; soft compactness information, not a hidden atomic model |
| Source indirect transform | GNOM4.6,2017-05-04 | dimensionless | Supplied profile header provenance |
| Box side | 240.0 | angstrom | Periodic Fourier box embedding the compact object |
| Grid side | 64 | voxel | Same count on x,y,z axes |
| Voxel spacing | 3.75 | angstrom | Uniform isotropic spacing; origin at index32 |
| Density normalization | 1.0 | dimensionless | Total integrated normalized contrast density |
| Particle orientation | Not measured | dimensionless | Random tumbling produces the spherical intensity average |
| Instrument angular resolution and covariance | Not supplied | dimensionless | No extra smearing kernel or covariance is inferred from a hidden model |
| Absolute electron count/concentration | Not supplied | dimensionless | Output is a normalized map, not absolute electrons per volume |

## Method Hints

Iterative structure-factor retrieval alternates reciprocal-space spherical-intensity constraints with real-space positivity and an adaptively shrinking compact support. If $F_k$ is the current discrete Fourier transform, $J_k$ the shell-averaged squared amplitude and $J_{\mathrm{obs}}$ the target intensity on that shell, the amplitude projection is


```math
F_{k+1/2}(\mathbf q)=F_k(\mathbf q)\sqrt{\frac{J_{\mathrm{obs}}(|\mathbf q|)}{J_k(|\mathbf q|)}}.
```


Index $k$ denotes iteration. This rescales amplitudes while retaining phases. An inverse Fourier transform then permits real-space constraints, and a blurred-density threshold can update the support. Real FFT implementations must count interior conjugate partners twice and zero/Nyquist planes once; otherwise shell averages depend on the selected Fourier half-space. Multiple independent starts help reveal uncertainty and non-uniqueness. A final reciprocal-space projection may violate positivity slightly, so distinguish the constrained iterate from that final output rather than claiming both constraints hold exactly.

## References

- Grant,T.D. *Ab initio electron density determination directly from solution scattering data*. Nature Methods15,191–193(2018). [Paper](https://doi.org/10.1038/nmeth.4581).
- DENSS author documentation. The real lysozyme GNOM profile accompanies the author's software distribution.
