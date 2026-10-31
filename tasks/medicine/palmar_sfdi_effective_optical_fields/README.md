# Palmar SFDI Effective Optical Fields

> Recover band-specific effective absorption and reduced-scattering fields from calibrated phase-stepped hand reflectance images whose tissue effects overlap across spatial frequencies.
> Domain: medicine | Keywords: diffuse optical imaging, tissue characterization, inverse transport | Difficulty: Hard

## Background

Structured illumination probes tissue at different spatial frequencies. Absorption removes photons while multiple scattering spreads the illumination pattern, reducing contrast. Their different frequency responses provide a noncontact way to map optical properties across a surface.

These measurements are useful for studying superficial tissue contrast and changes in perfusion or hydration. Practical tissue is layered and curved, whereas analytic diffusion models assume simplified geometry. An inferred optical field must therefore be interpreted under its stated transport and calibration assumptions.

## Problem Description

The real hand and independent calibration-phantom measurements contain three phase steps at each of five spatial frequencies. For phase images I1,I2,I3 define DC intensity d and modulation amplitude b:

```math
d=(I_1+I_2+I_3)/3,\qquad
b=\frac{\sqrt{2}}{3}\sqrt{(I_1-I_2)^2+(I_2-I_3)^2+(I_3-I_1)^2}.
```

At zero frequency, calibrated reflectance is the known phantom reflectance times the hand/phantom DC ratio. At each nonzero frequency use the corresponding modulation-amplitude ratio. Average each10 by10 camera-pixel block of calibrated reflectance, yielding50 by40 cells; retain all five frequencies and all cells.

The unknowns are positive absorption $\mu_a$ and reduced scattering $\mu_s'$ in inverse millimetres. The declared homogeneous semi-infinite diffusion model is

```math
\mu_t=\mu_a+\mu_s',\quad
h(f)=\frac{\sqrt{3\mu_a\mu_t+(2\pi f)^2}}{\mu_t},\quad
R(f)=\frac{3A(\mu_s'/\mu_t)}{(h(f)+1)(h(f)+3A)},
```

```math
A=\frac{1-r}{2(1+r)},\qquad
r=-1.440/n^2+0.710/n+0.668+0.0636n.
```

Here f is spatial frequency in cycles/mm, n is the assumed tissue refractive index relative to air, and R is dimensionless diffuse reflectance. The selected conditional target minimizes the sum of squared calibrated-reflectance residuals at the two lowest frequencies, 0 and 0.1 cycles/mm, independently in every cell. These frequencies are fixed from the published diffusion regime before evaluating the new inverse, rather than selected by fit quality. There is no spatial regularization or masked cell. All five observations remain public; the other three frequencies provide held-out model diagnostics and are included in the required forward predictions. The output coefficients are effective, band-specific diffusion parameters, not independently validated absolute tissue constants. The source does not supply the acquisition wavelength; no spectral law or wavelength is used in this forward model.

Save `output/reconstruction.npz` with `reconstruction`, real float64 shape `(1, 2, 50, 40)`, batch, coefficient, row, column; coefficient0 is absorption and coefficient1 reduced scattering, both positive in1/mm. Also save `reflectance`, real float64 shape `(1, 5, 50, 40)`, batch, increasing native frequency, row, column, dimensionless. All entries must be finite. Rows increase down and columns right. The coefficients correspond to10 by10 averages of the supplied ROI, with no interpolation, crop or alignment.

Evaluation checks every cell's selected two-frequency squared residual against1.02 times a validated standard-method residual plus an absolute floor1e-10. It also requires every reported reflectance to agree with the stated forward model with relative L2 error at most1e-9. A better continuous inverse solution can pass without reproducing a finite optimizer path. No optical ground truth is assumed, and fit residual is not tissue-property accuracy.

## Data Description

All arrays are supplied in `data/raw_data.npz`. Raw float32 image precision and float64 calibration values are preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| hand_phase | (1, 5, 3, 500, 400) | float32 | Real hand camera intensity, source arbitrary digital units; axes batch, frequency, phase, row, column |
| phantom_phase | (1, 5, 3, 500, 400) | float32 | Independent calibration-phantom camera intensity, same digital units and axes |
| phantom_reflectance | (1, 5) | float64 | Author-supplied phantom diffuse reflectance for each frequency, dimensionless |
| frequency_mm | (1, 5) | float64 | Native spatial frequencies, cycles/mm |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Spatial frequencies | 0,0.1,0.2,0.3,0.4 | cycles/mm | All native frequency channels |
| Phase steps | 0,120,240 | degree | Three equally spaced illumination phases |
| Tissue index | 1.4 | dimensionless | Explicit model convention relative to air, not a measured calibration |
| Surface model | Homogeneous semi-infinite medium | dimensionless | Diffusion boundary assumption, no surface-profile correction |
| Source ROI | rows400:900, columns300:700 | camera index | Zero-based half-open ROI fixed before fitting from1024 by1024 source frames |
| Block averaging | 10 by10 | pixels | Arithmetic averaging after calibration, all ROI pixels retained |
| Block centers | row404.5+10i, column304.5+10j | source camera index | i=0..49,j=0..39; source pixel centers at integer indices |
| Wavelength | Not supplied | unspecified | A single band; forward model does not require a wavelength |
| Pixel pitch | Not supplied | unspecified | No lateral distance is inferred; coefficient scale comes from known cycles/mm |
| Phantom convention | Supplied reflectance per frequency | dimensionless | Independent author calibration, not estimated from the hand |
| Noise variance | Not supplied | unspecified | Unweighted frequency residuals; no precision claim from assumed noise |

The published diffusion conditions are reduced scattering much greater than absorption and spatial frequency much smaller than the transport coefficient. The paper estimates a maximum frequency around0.25to0.33times the transport coefficient, with tolerances dependent on albedo and calibration. The higher frequencies and some low-scattering edge cells can exceed this regime even though the central tissue is scattering dominated. These cells remain in the selected low-frequency inverse; no fit-dependent mask is used. In fact, the analytic model can predict increasing high-frequency reflectance with increasing absorption, since its high-frequency asymptote is proportional to reduced scattering times the transport coefficient. This approximation artifact further limits interpretation. The dataset also lacks independently measured surface geometry and spectral band information, which limit physiological interpretation.

## Method Hints

Use the independent phantom to remove the instrument transfer response before optical inversion. Fit the analytic frequency-dependent diffusion response with a general nonlinear optimizer. Keep the frequency residuals visible so that transport-model departures remain measurable. A hypothetical change of the assumed refractive index probes model-convention sensitivity; it is not an experimentally measured calibration uncertainty.

## References

Cuccia D.J., Bevilacqua F., Durkin A.J., Ayers F.R., Tromberg B.J. Quantitation and mapping of tissue optical properties using modulated imaging. Journal of Biomedical Optics14,024012(2009). [Published paper](https://doi.org/10.1117/1.3088140).

AppSFDI Matlab1.0, author instructions dated2020-05-14, Sylvain Gioux and contributors: real hand/phantom phase-stepped acquisition and independent phantom reflectance. The selected inverse standard uses the published analytic diffusion model rather than a Monte Carlo inverse lookup table.
