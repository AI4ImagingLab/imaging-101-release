# Tin Positron Lifetime Mixture

> Recover three conditional positron lifetime components and their joint instrument-response parameters from six real cumulative pure-tin timing histograms whose short decays overlap the detector response.
> Domain: chemistry & material science | Keywords: positron spectroscopy, inverse kinetics, material characterization | Difficulty: Hard

## Background

Positron annihilation timing probes electronic and microstructural environments in materials. After implantation, a positron survives for a characteristic time before annihilating. Defects, source materials and positronium can introduce additional lifetime components beyond the bulk response.

The measured timing histogram also includes the instrument response and random coincidences. Short material lifetimes and response widths overlap, making decomposition sensitive to the physical model. Such measurements are used to characterize metals, semiconductors, polymers and porous materials, with conditional component estimates distinguished from independently measured microscopic truth.

## Problem Description

The data contain six cumulative acquisitions of annealed4Ntin with a sodium-22source sealed in Kapton, plus an independent cobalt-60prompt spectrum. Use the native counts without background subtraction. Sum each consecutive10native channels to form1000bins of width50ps. Let y(q,b) be these counts for condition q and bin b. Define S(q)=sum of y over bins33through948inclusive and the model amplitude A(q)=S(q)-916B(q), where B is a fitted constant background in counts per50psbin.

For lifetime tau, Gaussian mean m and standard deviation sigma, define the exponentially modified Gaussian cumulative response

```math
H(t;\tau,m,\sigma)=\Phi(z)-\exp\left(-\frac{t-m}{\tau}+\frac{\sigma^2}{2\tau^2}\right)\Phi\left(z-\frac{\sigma}{\tau}\right),\qquad z=(t-m)/\sigma.
```

Phi is the standard normal cumulative distribution. Each condition has three positive lifetimes tau(j) and nonnegative intensity coefficients I(j), and a two-Gaussian response with positive FWHM w(g), relative means m(g), and nonnegative coefficients c(g). Gaussian standard deviation is sigma=w/(2sqrt(2ln2)). With t(b)=50(b-33)ps the expected count is

```math
v_b=A\sum_{g=1}^2\sum_{j=1}^3c_gI_j\left[H(t_b+50;\tau_j,m_g,\sigma_g)-H(t_b;\tau_j,m_g,\sigma_g)\right]+B.
```

This is a bin-integrated response, not a point-sampled exponential. Component intensities I are the original finite-window amplitude coefficients and are not renormalized after fitting. Response weights c have a soft sum-to-one condition. The selected conditional target, independently for each of the six histograms, is

```math
J=\sum_{b=33}^{946}\frac{(y_b-v_b)^2}{y_b+1}+10^8(c_1+c_2-1)^2.
```

The two endpoint bins947and948contribute to S but not the fitted residual, preserving the source histogram-edge convention. All1000predictions are still required. The Poisson-style weighting includes the explicit plus-one safeguard; no unknown noise scale is needed. A physical solution has positive lifetimes/widths, nonnegative intensity/response weights/background and A>0. Joint response estimation is part of the inverse problem: never treat sample-fitted response parameters as known calibration. The independent prompt is provided for response diagnostics; it is a quasi instrument response and does not fix the sample fit's two Gaussian parameters.

Save `output/reconstruction.npz` with every following finite real float64 array. Means use the stated ROI-relative time origin; no shift alignment is applied during evaluation.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 6, 3) | float64 | Lifetimes in ps, ordered shortest to longest within each condition |
| intensity | (1, 6, 3) | float64 | Corresponding dimensionless nonnegative component amplitude coefficients |
| irf_fwhm_ps | (1, 6, 2) | float64 | Response Gaussian FWHMs in ps, ordered narrowest to broadest |
| irf_shift_ps | (1, 6, 2) | float64 | Corresponding Gaussian means in ps relative to t=0 at bin33 |
| irf_weight | (1, 6, 2) | float64 | Corresponding dimensionless nonnegative response coefficients |
| background | (1, 6) | float64 | Constant background counts per50psbin |
| prediction | (1, 6, 1000) | float64 | Expected counts at every rebinned channel |

Axes are batch, acquisition condition, then component or bin as applicable. Evaluation checks all parameters through the complete objective and the consistency of all6000predictions with the forward model. Objective bounds by increasing condition are998.6355536550473,1077.5777890931993,1076.7197227016702,1068.805109303599,1080.0811730058638,1078.3196181547135. Prediction relative L2 error normalized by each observed histogram's norm must be at most1e-9. Different inverse algorithms and Gaussian-label exchanges are allowed through the declared sorted output convention; no strict matching to one finite optimizer's parameters is imposed. These bounds describe a validated regularized/model fit, not independently known material truth.

## Data Description

All arrays are in `data/raw_data.npz`; integer counts and original channel coordinates are preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| counts | (1, 6, 10000) | int64 | Six native measured timing histograms, counts per5pschannel; axes batch, condition, channel |
| prompt_counts | (1, 10000) | int64 | Independently measured cobalt-60prompt counts per5pschannel |
| channel | (1, 10000) | int32 | Native increasing channel indices, dimensionless; native spacing5ps |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample |4Ntin, annealed250degC for5hours | dimensionless | Sodium-22source between two tin plates, Kapton encapsulation |
| Conditions |500k,1000k,1500k,2000k,2500k,2800k | nominal counts | Increasing cumulative stopping statistics; actual totals are sums of raw arrays |
| Native bin width |5 | ps |10000native channels span50ns |
| Analysis binning |10 | native channels | Sum, not mean, giving50psbins |
| Analysis area ROI |33through948inclusive | rebinned index |916bins for A=S-916B |
| Residual ROI |33through946inclusive | rebinned index |914data residuals, original endpoint convention |
| Relative origin |Bin33left edge, native1650ps | ps | Gaussian means measured relative to this origin |
| Source activity |25 | microCi | Sodium-22, source-sample timing acquisition |
| Prompt activity |1 | microCi | Cobalt-60independent quasi response acquisition |
| Nominal setup |Two plastic-scintillator/PMT detectors,180degree arrangement | dimensionless | Semi-analog/digital DRS4 timing acquisition |
| Histogram noise convention |Denominator y+1 | counts | Explicit measured-count weighting |

The six spectra are cumulative, not independent biological or material replicates. The shortest fitted component is associated with bulk tin under this model; other components include source/encapsulation and positronium contributions. Those assignments do not establish uniquely resolved microscopic states. Prompt and sample energy responses need not be identical.

## Method Hints

Fit the bin-integrated exponential/Gaussian mixture jointly with its background using nonlinear least squares. A general Levenberg-Marquardt optimizer is suitable. Keep the weighted residual, soft response normalization, amplitude convention and component permutation symmetry explicit. The independently measured prompt can illustrate response-shape limitations without replacing joint estimation by hidden calibration.

## References

Petschke D., Helm R., Staab T.E.M. Data on pure tin by Positron Annihilation Lifetime Spectroscopy acquired with a semi-analog/digital setup using DDRS4PALS. Data in Brief22,16-29(2019). [Paper and original measurement supplement](https://doi.org/10.1016/j.dib.2018.11.121).

Kirkegaard P., Eldrup M. POSITRONFIT: A versatile program for analysing positron lifetime spectra. Computer Physics Communications3,240-255(1972). [Published analytic convolution method](https://doi.org/10.1016/0010-4655(72)90070-7).
