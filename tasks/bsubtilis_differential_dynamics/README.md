# Bacterial Differential Dynamic Microscopy

> Infer length-dependent relaxation spectra and effective ballistic speed scales from two complete noisy fluorescence movies of motile bacteria.
> Domain: biology | Keywords: dynamic microscopy, fluctuation spectroscopy, parameter inference | Difficulty: Hard

## Background

Microscopy records fluctuations as fluorescent particles move through a spatial intensity pattern. The difference between images separated by a time delay contains a length-dependent signature of this motion. Fourier analysis associates each spatial frequency with a length scale, so many moving objects can be studied without identifying or tracking each cell.

This approach is useful for dense bacterial suspensions and other active biological materials where individual trajectories overlap. Optical contrast and camera noise also contribute to the observed fluctuations. Separating those contributions from the relaxation of the moving material requires multiple time delays and a physically defined intensity normalization. A relaxation-derived speed scale summarizes ensemble dynamics under a motion model; it is not an independently calibrated speed of every bacterium.

## Problem Description

The two complete author recordings show motile B. subtilis in distinct focal conditions, identified by the native filenames `BacCFP_33deg_01umE` and `BacCFP_33deg_16umE`. Preserve this order. Their precise physical depth separation is not independently verified. Let I(t,x,y) denote image intensity in ADU and let F(t,k) be its spatial Fourier transform divided by sqrt(HW), where H and W are the image height and width. For a positive integer delay l, define the image structure function as the mean of |F(t+l,k)-F(t,k)|² over all T-l available frame pairs, then average over the specified radial Fourier ring. T is the number of frames. The effective acquisition model is

```math
D(q,\tau)=A(q)\left[1-\exp\left(-(\Gamma(q)\tau)^{\beta(q)}\right)\right]+B(q),
\qquad \tau=l/f_s.
```

Here q is radial angular spatial frequency in rad/micrometer, tau is delay in seconds, fs is frame rate, A>0 is dynamic contrast in ADU², B is an effective additive background in ADU², Gamma>0 is relaxation rate in s^-1, and beta>=1 is a dimensionless compression exponent. B is a fitted effective offset and can be signed; it is not asserted to be a nonnegative instrument variance. The ensemble ballistic scale v>0 in micrometers/s relates Gamma to vq through an unweighted fit of log(Gamma) to log(vq). Finite recordings, weak contrast, imperfect radial isotropy and coupled parameters make the inverse problem difficult.

Save `output/reconstruction.npz` relative to the task workspace with both required real finite `float64` arrays:

| Key | Shape | Units and axes |
|---|---|---|
| reconstruction | (1, 2, 39, 4) | batch, recording, ascending selected q ring, parameter; last axis is A [ADU²], B [ADU²], Gamma [s^-1], beta [dimensionless]. |
| speed | (1, 2) | Effective ensemble ballistic scale v in micrometers/s, in the same recording order. |

The 39 rings have centers between 0.75 and 4 rad/micrometer, inclusive; their exact construction is specified below. Neither spatial registration nor amplitude rescaling is applied during evaluation. Each parameter and each recording is evaluated separately using relative L2 error, norm(estimate-reference)/norm(reference); speed uses absolute relative error. A direct finite-delay image-difference consistency check also evaluates the model against the measured movies at delays 1,2,5,10,20,50,100,200,500 frames. Every quantity must pass. Agreement is assessed against a standard estimator, not an independently measured biological truth.

## Data Description

The file `data/raw_data.npz` contains both full native movies; no frames or spatial pixels were removed, averaged or resized.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frames | (1, 2, 2000, 128, 128) | uint8 | Native fluorescence intensities in ADU; axes batch, recording, time, downward y, rightward x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Organism | B. subtilis | dimensionless identifier | Author-described motile bacterial sample. |
| Temperature | 33 | degrees Celsius | Author-reported acquisition temperature. |
| Frame rate fs | 100 | Hz | All 2000 frames retained for each condition. |
| Pixel spacing | 75.59/128 = 0.590546875 | micrometers/pixel | Exact calibration used in the supplied analysis; its text rounds this to 0.591. |
| Native dimensions | 128 by 128 | pixels | Finite camera field, 75.59 micrometers on each side. |
| Intensity encoding | unsigned 8 bit | ADU | Native detector values; no photon conversion or measured noise variance supplied. |
| Optical context | point-scanning confocal, 63x oil objective | dimensionless identifier | Author-reported instrument context. |
| Focal conditions | native identifiers 01umE and 16umE | dimensionless identifiers | Two distinct focal conditions; no independently verified metric depth is imposed. |
| Spatial Fourier boundary | periodic discrete basis, no window or padding | pixels | FFT applied to the entire finite field; normalization 1/sqrt(128*128). |
| Wavevector coordinates | kx=2pi*fftfreq(128)[:65]/pixel_spacing; ky=2pi*fftshift(fftfreq(128))/pixel_spacing | rad/micrometer | Real-FFT half-plane, x unshifted and y shifted. |
| Radial edges | 90 equally spaced values from 0 to 90*2pi/75.59 | rad/micrometer | Bin index is left search insertion index of the mode radius in these edges. |
| Ring weights | 2 except x DC/Nyquist columns 1 | dimensionless | Full-plane multiplicity of each stored Fourier mode. |
| Excluded Fourier modes | first x column, central y row, insertion index >=90 | dimensionless indices | Fixed cross mask; no target-dependent mask. |
| Ring center and output selection | weighted mean mode radius; retain centers 0.75 through 4 | rad/micrometer | Ascending selected centers define the 39 output positions. |
| Delay averaging | all T-l overlapping pairs | frame pairs | No temporal wrapping or padded pairs contribute. |

The actual movies are from the authors' PLU-BACTERIA archive, released under CC BY 4.0. Attribution and source are listed below. The archive's `noswarm_depths_at_same_time/*E` names differ from the tutorial's `swarm_depth_sequence/*F` example names; this task analyzes the actual supplied E movies and does not claim to reproduce a particular published F-movie figure.

## Method Hints

Differential dynamic microscopy can form the finite-overlap structure function efficiently using zero-padded temporal Fourier correlation. Retain the exact spatial normalization, radial weights, excluded cross and overlapping-pair denominator. Weighted spatial dispersion can guide nonlinear curve fitting without being interpreted as a calibrated temporal uncertainty.

Fit a compressed exponential relaxation at each selected spatial frequency. The offset and amplitude can be initialized from temporal Fourier variance and high-frequency noise. Neighboring frequency fits can provide stable initial values. The two focal conditions need not have identical compression behavior: a variable exponent for the first recording and fixed beta=1.35 for the second define the selected standard estimator. A logarithmic rate-versus-frequency comparison provides an effective ballistic scale and reveals deviations from a single-scale model.

## References

- Lattuada, Krautgasser, Lavaud, Giavazzi and Cerbino, The Hitchhiker's Guide to Differential Dynamic Microscopy. Journal of Chemical Physics 163,161501 (2025). https://doi.org/10.1063/5.0289471 ; arXiv:2507.05058v2.
- Author-released PLU-BACTERIA observations, Phaidra, CC BY 4.0: https://phaidra.univie.ac.at/detail/o:2170175 .
- fastddm documentation and bacterial-motion tutorial, https://fastddm.readthedocs.io/ . Source-derived numerical implementation is GPL-3.0-or-later.
