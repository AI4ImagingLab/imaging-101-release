# Electron Holography of a Material Needle

> Reconstruct the complex electron transmission of an iron needle from a real off-axis hologram and a separately measured vacuum hologram despite carrier overlap and instrumental phase distortions.
> Domain: Chemistry & Materials Science | Keywords: electron imaging, phase retrieval, interferometry | Difficulty: Medium

## Background

Electrons acquire phase as they traverse electric and magnetic potentials in a thin material. Conventional intensity images can obscure this phase information, while electron interference records it through a spatial fringe pattern. Recovering the electron wave provides complementary information about nanoscale material structure, thickness, charge and magnetic fields.

In off-axis electron holography an electron biprism overlaps an object wave with a tilted reference wave. This introduces a carrier that separates the interference information from the average intensity in reciprocal space. An independently acquired vacuum hologram characterizes distortions shared by the illumination, biprism and detector. Applications include nanostructured materials, semiconductor junctions and magnetic devices.

## Problem Description

The data are an experimentally recorded Fe needle with yttrium-oxide nanoparticle inclusions and a separate vacuum exposure. At detector position r=(y,x), the ideal acquisition model is

```math
I(r)=\left|O(r)+R(r)\exp\!\left(2\pi i[k_y y/N_y+k_x x/N_x]\right)\right|^2+\eta(r).
```

Here I is the intensity in detector counts; O is the object electron wave; R is the reference wave; i is the imaginary unit; y and x are zero-based pixel indices; Ny and Nx are the image dimensions; ky and kx are integer carrier bins; and eta represents measurement noise and departures from the ideal model. Wave amplitudes have square-root-count units. In the vacuum exposure O is replaced by the vacuum object-path wave V. The negative-carrier cross term is O times the complex conjugate of R; dividing it by the corresponding vacuum cross term removes the reference distortion. Limited reciprocal-space support, noise and carrier separation limit the spatial frequencies recoverable without mixing interference orders.

Recover the dimensionless, band-limited complex transmission q from these two observations. Its amplitude is the object/vacuum wave-amplitude ratio, and its principal argument in radians is the object-minus-vacuum phase. The coordinate origin, carrier sign and vacuum calibration fix its convention. Absolute electrostatic potential, charge density and specimen thickness are outside this inverse problem; the observations do not uniquely determine them.

Save `output/reconstruction.npz` relative to the task workspace with the single required key:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 512, 512) | complex64 or complex128 | Dimensionless complex transmission q; axes (batch,y,x); real and imaginary components share the vacuum-normalized physical scale; phase is arg(q) in radians. |

Evaluate the full array with complex relative L2 error, defined as the square root of the sum of squared complex absolute errors divided by the square root of the sum of squared reference magnitudes. Also evaluate relative observation-sideband error, namely the L2 norm of q times the vacuum sideband minus the measured object sideband, divided by the object-sideband L2 norm. Both must be at most 1e-5. There is no crop, magnitude conversion, rescaling, conjugation, translation or phase alignment.

For reproducible bandwidth and phase convention, the evaluated sidebands use the full 512-by-512 unshifted DFT, a **positive** roll by (122,454), and a radial aperture W. This roll extracts the negative carrier, rather than the conjugate positive-carrier wave. With f the unshifted radial spatial frequency in nm^-1 and d the pixel size in nm, define

```math
\rho=\tfrac12\sqrt{122^2+58^2},\qquad
f_0=\rho/(512d),\qquad s=0.05 f_0,\qquad
W(f)=\tfrac12\left[1-\tanh\left(\frac{f-f_0}{0.5s}\right)\right].
```

The filtered inverse DFT of each recentered observation defines its sideband. The variables rho, f0 and s are the aperture radius in Fourier pixels, radius in nm^-1 and transition smoothness in nm^-1. This finite-bandwidth output convention defines the observable reconstructed quantity; the tolerance checks reproducibility, not independently measured material accuracy.

## Data Description

`data/raw_data.npz` contains the original repository's already binned/cropped, real acquired intensity images, repackaged without numerical modification. Load with `allow_pickle=False`. Both keys are available at acquisition time; the vacuum image is calibration, not a target reconstruction.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| hologram | (1, 512, 512) | int64 | Object hologram intensity, detector counts; axes (batch,y,x). |
| vacuum | (1, 512, 512) | int64 | Separately recorded vacuum hologram intensity, detector counts; axes (batch,y,x). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Fe needle with YOx nanoparticle inclusions | dimensionless label | Material specimen; YOx indicates yttrium oxide without specified stoichiometry. |
| Instrument | FEI Titan G2 60-300 HOLO at ER-C | dimensionless label | Transmission electron microscope. |
| Beam energy | 300 | keV | Electron kinetic energy. |
| Exposure | 10 | s | Exposure for each image. |
| Biprism voltage | 130 | V | Interference acquisition setting. |
| Stage tilt | 5 | degree | Recorded specimen-stage tilt. |
| Pixel size d | 0.9197516441345215 | nm/pixel | Same square sample-plane spacing in both images. |
| x origin | 0 | nm | Physical x coordinate of column zero. |
| y origin | 183.9503288269043 | nm | Physical y coordinate of row zero. |
| Axis order | y increases with row, x increases with column | dimensionless convention | Retain supplied image orientation. |
| Dimensions Ny,Nx | 512,512 | pixel | No additional cropping or resampling. |
| Observed carrier ky,kx | 122,454 | DFT bin | Peak in the unshifted vacuum DFT; kx=454 is equivalent to -58. |
| Intensity calibration | gain-normalized detector counts | count | No absolute electron-count conversion is provided. |
| Noise model | not independently calibrated | dimensionless description | Retain measured noise; do not assume a supplied Gaussian variance. |
| Boundary convention | periodic DFT extension | dimensionless convention | Fourier filtering uses the complete supplied images. |

The sample's mean inner potential, thickness and charge density are not supplied or required for wave reconstruction. The table reports the binned/cropped images' calibrated axes rather than the original camera frame's pixel spacing.

## Method Hints

Fourier sideband isolation is appropriate because the tilted reference separates a complex cross term from the zero-order intensity. Demodulate the same carrier in the object and vacuum exposures, apply the stated common smooth aperture and divide their reconstructed complex sidebands. A vacuum correction removes phase ramps and reference distortions shared by the two exposures. Preserve the negative-carrier convention: the opposite sideband conjugates the recovered phase. A closed-form spectral reconstruction is sufficient; phase unwrapping is optional for display and does not change the required complex output.

## References

- V. Migunov et al., “Model-independent measurement of the charge density distribution along an Fe atom probe needle using off-axis electron holography without mean inner potential effects,” Journal of Applied Physics 117, 134301 (2015), https://doi.org/10.1063/1.4916609. The present task reconstructs the wave from the supplied holograms; it does not reproduce the paper's complete charge-density experiment.
- C. Boothroyd et al., “FEI Titan G2 60-300 HOLO,” Journal of Large-Scale Research Facilities 2, A44 (2016), https://doi.org/10.17815/jlsrf-2-70.
- E. Prestat et al., HoloSpy v0.3, https://doi.org/10.5281/zenodo.11099344. Experimental hologram data credited to Vadim Migunov; specimen prepared by Andrew London. Dataset archive metadata declares CC BY 4.0; adapted source code is GPL-3.0-or-later. Data attribution: Vadim Migunov; specimen: Andrew London. Source adaptations retain their original copyright and GPL-3.0-or-later terms.
