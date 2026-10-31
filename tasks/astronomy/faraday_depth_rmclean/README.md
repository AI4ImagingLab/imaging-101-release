# Faraday Depth Reconstruction from Broadband Polarization

> Reconstruct a complex line of sight polarized emission spectrum from noisy broadband Stokes measurements whose finite positive wavelength coverage creates strong response sidelobes.
> Domain: astronomy | Keywords: polarization tomography, inverse Fourier imaging, sparse deconvolution | Difficulty: Medium

## Background

Linearly polarized radio radiation rotates as it travels through magnetized ionized plasma. The rotation angle increases with wavelength squared and with the integrated product of electron density and the magnetic field component along the ray. Broadband polarization therefore contains information about the distribution of emitting and rotating material along a line of sight.

Faraday depth tomography is used to investigate magnetic fields in galaxies, galaxy clusters and the interstellar medium. Only a bounded interval of positive wavelength squared is measured, so the recovered depth response has a broad central lobe and sidelobes. A reconstructed depth spectrum is a resolution limited representation, rather than a unique three dimensional physical distance map.

## Problem Description

This task uses a complete supplied **simulated** broadband polarization observation, with all 288 measured channels retained. It is not a real sky acquisition. Complex polarization is P = Q + i U, where Q and U are signed Stokes flux densities in Jy/beam. For frequency nu, define t = (c/nu)^2 in square metres, with c the vacuum speed of light. The discrete acquisition model is

```math
P_k=\sum_{j=0}^{400} C_j\exp\{2i\phi_j(t_k-t_0)\}+\epsilon_k,
\qquad
t_0=\frac{\sum_k w_k t_k}{\sum_k w_k},\qquad w_k=\sigma_k^{-2}.
```

Here k indexes frequency channels, j indexes depth bins, phi is Faraday depth in rad/m², C is complex polarized flux **per bin** in Jy/beam, sigma is the standard deviation of each real and imaginary noise component, and epsilon is complex measurement error. The depth grid is phi = -600 + 3 j rad/m². No bin width factor multiplies the discrete sum. Phases use the weighted reference wavelength t0, rather than zero wavelength. There is no unknown overall phase, gain, translation or magnitude convention to fit during evaluation. The channel centre approximation neglects rotation within an individual frequency channel. Q and U are used as absolute fluxes; there is no Stokes I normalization or spectral index correction. Frequency dependent source flux can therefore appear as extra structure in this selected model.

The limited wavelength interval leaves many depth distributions consistent with the measurements. Return a sparse component representation and a finite resolution representation in **one** file, `output/reconstruction.npz`, relative to the task workspace:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 401) | complex128 | Restored complex polarized flux in Jy/beam, axes (batch, increasing Faraday depth). Real and imaginary parts retain the t0 Stokes Q/U phase convention. The finite resolution response is a unit peak Gaussian of FWHM W = 3.8/(max(t)-min(t)) rad/m², with remaining inverse response residual included. No unit area normalization. |
| components | (1, 401) | complex128 | Sparse polarized flux per depth bin in Jy/beam, same axes and phase convention; its forward sum predicts the channel data. |

Evaluation version 2 checks the noise-compatible sparse representation rather than exact agreement with a finite CLEAN trajectory. Require sum(abs(model-observation)^2/sigma^2)/(2 × 288)<=1.209597539639874 and sum_j abs(C_j)<=0.7236795122859186 Jy/beam. The latter is an attainable sparse-flux budget from the validated standard inversion; both real and imaginary components contribute to each modulus. This defines the minimum-total-absolute-flux family under the stated noise discrepancy, without asserting unique physical depth structure. There is no crop, amplitude rescaling, phase rotation, shift or magnitude conversion.

Both outputs must also represent the same conditional solution. Define A by the forward exponent above, dirty spectrum d=A-Hermitian diag(w)P/sum(w), response matrix R=A-Hermitian diag(w)A/sum(w), and G_ij=exp(-4 log(2)((phi_i-phi_j)/W)^2), with W as above and G unit peak. Require reconstruction=GC+d-RC, with complex Euclidean relative error at most 1e-10. Thus a more sparse noise-compatible component estimate is accepted, while a missing or unrelated restored representation fails. The chi-square statistic uses all channels without a fitted-parameter degrees-of-freedom correction.

## Data Description

All visible arrays are in `data/raw_data.npz` and retain original float64 channel values. The complex polarization stores the original Q and U values without averaging or resampling.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frequency | (1, 288) | float64 | Increasing channel centre frequency in Hz; axes (batch, channel). |
| polarization | (1, 288) | complex128 | Q + i U flux density in Jy/beam; axes (batch, channel). |
| noise | (1, 288) | float64 | One standard deviation of each independent Q and U noise component, Jy/beam; axes (batch, channel). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation type | Supplied simulated radio polarization spectrum | — | Existing observations, not a real sky measurement. |
| Channel count | 288 | channels | All distributed frequency channels are retained. |
| Channel centre range | 800000000 to 1088000000 | Hz | Use the supplied frequency values verbatim. |
| Noise standard deviation | 0.1 | Jy/beam | Per real Stokes component; the complex variance is twice its square. |
| Speed of light | 299792458 | m/s | Exact vacuum constant. |
| Polarization sign | P = Q + i U | — | Forward exponent has positive 2 i phi t. |
| Wavelength reference | Weighted mean of squared wavelength | m² | Weights are inverse per component variance. |
| Channel approximation | Monochromatic at channel centre | — | No channel bandwidth integration. |
| Missing channels | None | — | No flags or implicit interpolation. |
| Spatial convention | One independent line of sight | — | Beam integrated spectral observation; no transverse image axes. |
| Depth boundary | Zero component flux outside declared output grid | rad/m² | Finite represented depth support; no periodic physical emission assumption. |

## Method Hints

An inverse variance weighted complex Fourier sum estimates a dirty depth spectrum. Its response to a unit depth component is a rotation measure spread function (RMSF), obtained by applying the same sum to unit channel polarization. This response describes the sidelobes produced by incomplete sampling.

A suitable sparse deconvolution family is Hogbom CLEAN: identify a large complex peak, transfer a fraction of that peak into a component spectrum, and subtract its shifted RMSF. Stop when the residual reaches the expected transformed noise level. Convolve the components with the declared unit peak Gaussian and add the remaining dirty residual to form the restored representation. Use the phase convention consistently in both directions. The optional deeper windowed CLEAN and super resolution variants are distinct choices and need not be introduced for this task.

## References

- Van Eck et al., *RM-Tools: Software for Analyzing Polarized Radio Spectra*, paper version [arXiv:2601.20092v1](https://arxiv.org/abs/2601.20092v1); journal DOI [10.3847/1538-4365/ae3dea](https://doi.org/10.3847/1538-4365/ae3dea). Selected scope: conventional synthesis and single threshold CLEAN.
- Brentjens and de Bruyn, *Faraday rotation measure synthesis*, Astronomy & Astrophysics 441, 1217–1228 (2005), [10.1051/0004-6361:20052990](https://doi.org/10.1051/0004-6361:20052990).
