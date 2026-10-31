# Core-loss Electron Single-scattering Reconstruction

> Recover a Gaussian-resolved single-scattering Mn core-loss spectrum from measured paired electron energy-loss spectra whose plural scattering obscures the edge structure.
> Domain: physics | Keywords: electron spectroscopy, deconvolution, spectral imaging | Difficulty: Hard

## Background

Fast electrons traversing matter can lose energy to several excitations during the same passage. An electron energy-loss spectrum records their combined energy transfers. Its core-level features help distinguish elemental composition and local electronic structure, while its low-loss part records the instrumental resolution and additional low-energy excitations.

Multiple scattering blends core-level features with the low-loss distribution. Recovering a single-scattering spectrum improves quantitative interpretation in electron microscopy. Finite detector windows, continuum subtraction and instrumental broadening limit the recoverable detail, so the requested resolution and window conventions are part of this problem.

## Problem Description

The two supplied spectra are measured counts per energy channel from a paired acquisition on 4 September 2012. Recover the Gaussian-resolved Mn core-loss single-scattering spectrum defined below. No independently measured unbroadened single-scattering truth is available. Continuum, reference measurement and finite-window conventions define the requested quantity; a particular inverse algorithm is not required.

Write $l_k,c_k$ for the measured low-loss and core-loss counts, $E^L_k,E^C_k$ for their nominal energies in eV, and $h=0.2$ eV for the bin width. All indices start at zero. The measured zero-loss shift is

```math
k_*=\operatorname{argmax}_k l_k,\qquad
\delta=E^L_{k_*},\qquad
x^L_k=E^L_k-\delta,\qquad x^C_k=E^C_k-\delta.
```

For a single spectrum there is no count interpolation. Given an energy $e$ and grid origin $x_0$, its channel index is the nearest integer to $(e-x_0)/h$, with ties to even. The first maximum defines $k_*$ if a tie occurs.

The continuum is $b(x)=hAx^{-r}$ in counts/channel, where $r$ is dimensionless and $A$ has units counts times eV to power $r-1$. Its two-area calibration protocol is the following mathematical definition. For count vector $y$ on energy grid $x$, convert the specified interval endpoints to indices $i,j$, reduce $j$ by one if $i+j$ is odd, and let $m=(i+j)/2$. Define the two observed count sums $U_1,U_2$ and energy endpoints $a,d,z$ by

```math
U_1=\sum_{k=i}^{m-1}y_k,\qquad
U_2=\sum_{k=m}^{j-1}y_k,\qquad
a=x_i,\quad d=x_j,\quad z=x_m.
```

Then

```math
r=\frac{2(\log U_1-\log U_2)}{\log d-\log a},
\qquad
A=\frac{(1-r)U_2}{d^{1-r}-z^{1-r}}.
```

Apply this definition to $c$ on corrected energies $[600,638]$ eV. Subtract $b$ on the core grid and retain the 844 measured channels beginning at 600 eV, ending at 768.6 eV. Denote their background-subtracted counts by $v$.

Define extended low-loss and core arrays by preserving $l$ and $v$ and appending 100 bins $hAx^{-r}$ on the right. Determine each tail's $A,r$ with the same two-area definition on its final $w$ original channels: interval endpoint indices $n-w,n-1$, where $n$ is its pre-extension length, $w=100$ for low loss and $w=20$ for core loss. Retain negative exponents. For each extended length $n'$ set

```math
q=\max\left(20,\operatorname{round}(0.02n')\right),\qquad
H_j=\frac{1-\cos(2\pi j/(2q-1))}{2},
\quad j=0,\ldots,2q-1.
```

Multiply the first $q$ bins by the first half of $H$ and the final $q$ bins by its second half. Preserve each energy origin and zero-pad on the right to length 3000. Denote these tapered, padded count arrays by $L,V$.

The elastic integration limit $T$ is determined from the original corrected low-loss interval $[1,11)$ eV. Compute its first derivative with a length-five, degree-three local polynomial fit; use one-sided polynomial evaluation at its two edges. Choose the first energy whose derivative has minimum absolute value. The elastic count $I_0$ is the sum of extended tapered low-loss bins preceding the rounded index of $T$. It is measured counts, with no extra factor of $h$ and no substitution of total low-loss area.

Let $W$ be the separation, in eV, of the two half-maximum roots of the interpolating cubic spline through the extended tapered low-loss spectrum (zero smoothing, not-a-knot endpoints). With $\sigma=W/2.3548$ eV, define the sampled Gaussian density $G$, units $1/\mathrm{eV}$, on the padded low-loss energy grid:

```math
G_k=\frac{1}{\sqrt{2\pi}\sigma}
\exp\left[-\frac{(x^L_0+kh)^2}{2\sigma^2}\right],
\qquad k=0,\ldots,2999.
```

The unknown padded single-scattering density $S$, in counts/eV, obeys

```math
L\circledast S=I_0\,G\circledast V,
\qquad
(a\circledast b)_k=\sum_{j=0}^{2999}a_jb_{(k-j)\bmod 3000}.
```

Here $\circledast$ is circular discrete convolution without another bin-width factor. The measured low-loss convolution spectrum is nonsingular. The low-loss and Gaussian origins coincide, preserving core-loss indexing. Only the first 844 entries of $S$ are requested; the other entries specify the finite-window boundary model. No positivity clipping or arbitrary intensity normalization is part of this target.

Save `output/reconstruction.npz` with required key `reconstruction`, shape `(1, 844)`, real `float32` or `float64`, units counts/eV, axes `(batch, increasing corrected core energy)`. Evaluation uses both full-window dynamic-range NRMSE and relative integrated-intensity error:

```math
\mathrm{NRMSE}=
\frac{\sqrt{844^{-1}\sum_{k=0}^{843}(\widehat S_k-S_k)^2}}
{\max_k S_k-\min_k S_k},
\qquad
\epsilon_I=
\frac{\left|\sum_{k=0}^{843}(\widehat S_k-S_k)\right|}
{\left|\sum_{k=0}^{843}S_k\right|}.
```

Both boundaries are $10^{-5}$. Equal bin-width factors cancel in the intensity ratio. There is no fitted shift, rescaling, cropping or alignment during evaluation.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| low_loss_counts | (1, 2048) | float64 | Original low-loss measurements, counts/channel; negative recorded noise is preserved. |
| core_loss_counts | (1, 2048) | float64 | Original core-loss measurements, counts/channel. |
| low_loss_energy | (1, 2048) | float64 | Nominal low-loss channel energies, eV. |
| core_loss_energy | (1, 2048) | float64 | Nominal core-loss channel energies, eV. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition date | 2012-09-04 | calendar date | Original paired spectrum headers. |
| Channel spacing | 0.2 | eV | Both uniformly spaced binned spectra. |
| Nominal low-loss origin | -40 | eV | Before visible zero-loss correction. |
| Nominal core-loss origin | 360 | eV | Before the same zero-loss correction. |
| Continuum interval | 600 to 638 | eV | Corrected energy bounds for background calibration. |
| Output origin and extent | 600 to 768.6 inclusive | eV | 844 measured core-loss channels. |
| Tail extent | 100 | channels | Defined mathematical continuation on the right. |
| Periodic model length | 3000 | channels | Circular convolution length after right zero padding. |
| Resolution | Measured low-loss FWHM | eV | Determined from the visible reference, not a hidden fitted constant. |
| Absolute intensity | Elastic-count scaling $I_0$ | counts | Preserves absolute spectral density; no unit-area renormalization. |

Beam energy and collection angle are not present in the acquisition headers and are not needed for this convolution target. No dielectric function, cross section or specimen thickness is requested.

## Method Hints

A Fourier-ratio method diagonalizes convolution, and Gaussian reconvolution controls instrumental resolution. Work with the specified count and density conventions throughout. The continuum and reference calibration above define the data model; they should be freshly determined from the visible measurements.

## References

- F. Wang, R. F. Egerton and M. Malac, “Fourier-ratio deconvolution techniques for electron energy-loss spectroscopy (EELS),” *Ultramicroscopy* 109 (2009), 1245–1249. [DOI](https://doi.org/10.1016/j.ultramic.2009.05.011).
- R. F. Egerton, F. Wang, M. Malac, M. S. Moreno and F. Hofer, “Fourier-ratio deconvolution and its Bayesian equivalent,” *Micron* 39 (2008), 642–647. [DOI](https://doi.org/10.1016/j.micron.2007.10.004).
