# Cardiac CEST Apparent Pool Contrasts

> Recover spatial maps of six apparent saturation components and resonance shifts from measured mouse cardiac MRI spectra whose overlapping peaks, broad background and field offsets confound interpretation.
> Domain: medicine | Keywords: magnetic resonance imaging, spectral decomposition, molecular contrast | Difficulty: Hard

## Background

Chemical exchange saturation transfer MRI applies radiofrequency saturation at selected frequency offsets and measures the remaining water signal. Saturation can transfer from exchangeable protons to water, making weak molecular environments visible indirectly. Direct water saturation, broad macromolecular effects and nuclear Overhauser effects also alter the measured spectrum.

Frequency-dependent contrast is useful for studying tissue composition and physiological changes in small animals and clinical research. Cardiac measurements are particularly challenging because motion, blood partial volumes and magnetic-field variation coexist with weak, overlapping spectral features. Apparent spectral components are conditional descriptors of an acquisition; their amplitudes are not direct molecular concentrations or chemical exchange rates.

## Problem Description

The input is one complete measured mouse cardiac acquisition reconstructed into62 magnitude images. Repeated far-offset reference images measure unsaturated signal drift. The public support mask selects locations where the first reference exceeds15% of its spatial maximum; retain every supported pixel. This is a signal-support mask, not a myocardial segmentation. There is no independent component ground truth.

For location p and saturation offset omega, the reference-normalized signal Z follows the apparent six-component model

```math
Z(p,\omega)=1-\sum_{k=1}^{6}
\frac{a_k(p)\,w_k(p)^2/4}
{w_k(p)^2/4+[\omega-b(p)-c_k(p)]^2}+\epsilon(p,\omega).
```

Here a is dimensionless peak saturation amplitude, w is full width at half maximum in ppm, c is component center relative to the local water resonance in ppm, b is the local resonance shift in ppm, and epsilon represents measurement noise and model discrepancy. Normalize each nonreference image by the linearly interpolated far-offset reference intensity at its acquisition index. Offsets above15ppm are references, not fitted saturation observations. The interpolation is bounded by the first and last acquired references; there is no extrapolation.

The component order is **water, magnetization-transfer background (MT), amide, creatine, NOE near−3.5ppm, NOE near−1.6ppm**. The names identify apparent spectral components, not independently separated molecular species. Use the following conditional component ranges:

| Component | Amplitude interval, fraction | Width interval, ppm | Center interval, ppm |
|---|---|---|---|
| Water |0.02–1|0.01–10|−0.000001–0.000001|
| MT |0–0.5|30–60|−2.5–0|
| Amide |0–0.3|0.5–5|3.2–4|
| Creatine |0–0.5|0.5–5|1.6–2.6|
| NOE near−3.5ppm |0–0.25|0.5–5|−4–−1.5|
| NOE near−1.6ppm |0–0.25|0.5–5|−1.8–−1.2|

The field shift lies between−10and10ppm. Widths and centers become weakly identifiable when a component amplitude approaches zero. Broad overlapping components and finite frequency sampling permit different parameter combinations to explain similar spectra; reference agreement does not establish biological accuracy.

Save `output/reconstruction.npz`, relative to the task workspace, with **all four** arrays below. Values must be finite, real float32 or float64, and exactly zero outside the supplied mask. No spatial registration, intensity rescaling, component permutation or independent center alignment is permitted.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `amplitude` | (1,192,192,6) | float32 or float64 | Unit-fraction apparent peak amplitudes; axes batch, native row, native column, ordered component;0.1 means10% normalized signal attenuation at an isolated peak |
| `width` | (1,192,192,6) | float32 or float64 | FWHM in ppm, same axes |
| `center` | (1,192,192,6) | float32 or float64 | Center relative to shifted water resonance in ppm, same axes |
| `field_shift` | (1,192,192) | float32 or float64 | Local water-resonance shift b in ppm; axes batch, native row, native column |

Evaluation requires the declared physical ranges and exact zero values outside
the input-only support. It compares the complete six-pool spectrum against all
53 normalized saturation measurements. At every supported pixel its mean
squared residual may exceed the attained standard fit by at most1e-6
(normalized-signal squared). The global normalized-signal RMS must also be at
most0.08586382723233611. The extra voxel budget corresponds to0.1% normalized
signal RMS and is a numerical fit allowance, not a measured noise variance.
All amplitudes, widths, centers and shifts enter this forward check; parameter
combinations with equivalent or better spectra are accepted, including weak
components with nonunique widths. No fitted intensity gain or map alignment is
performed. This defines a conditional bounded spectral estimate, not true pool
concentrations or a unique parameter field.


## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `measurements` | (1,192,192,62) | float64 | Magnitude MRI intensity in arbitrary reconstructed detector units; axes batch, native row, native column, acquisition index |
| `offsets` | (1,62) | float64 | Saturation frequency offsets in ppm, acquisition order including interspersed far-offset references |
| `observation_mask` | (1,192,192) | bool | Dimensionless observation support derived solely from the first reference magnitude; batch,row,column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Mouse cardiac acquisition | dimensionless | Complete native in-vivo scan; support also includes surrounding visible tissue |
| Scanner | Bruker PharmaScan70/16, nominal7T | T | Instrument identity and main magnetic field |
| Working frequency |300.324686903002|MHz|Native proton reference frequency |
| Matrix / field of view |192×192 /25×25|pixel / mm|Native in-plane sampling |
| Slice thickness |0.9|mm|Single acquired slice |
| Echo time / repetition time |2.12390670906575 /8|ms|Native sequence parameters |
| Radial projections / receivers |1000 /4|dimensionless|Measured MRI encoding before magnitude-image formation |
| Saturation peak field |1.1|microtesla|Nominal RF saturation amplitude |
| Saturation pulse / delay / count |13.7 /4.56 /12|ms / ms / dimensionless|Pulsed saturation train |
| Saturation module time |244.866|ms|Native module duration including sequence overhead |
| Offsets |Stored62 values, rounded to0.01ppm upstream|ppm|53 saturation samples and9 references; retain acquisition order |
| Far reference |199.78|ppm|Unsaturated signal reference frames |
| Image formation |Iterative radial reconstruction and receiver root-sum-of-squares|arbitrary intensity|Processed measured observations, not synthetic images or fitted CEST targets |
| Motion/PCA correction |Not applied|dimensionless|Optional corrections absent in this bound acquisition processing |
| Coordinates |Native reconstructed array order|pixel|No subsequent rotation, flipping, cropping or spatial interpolation |
| Boundary / support |Independent pixel spectra; supplied support|dimensionless|All outputs outside support exactly zero |
| Noise |No independent variance calibration|normalized signal fraction|Magnitude noise, motion and partial volumes limit the apparent model |

The images are **author-real acquisition data processed with the original radial reconstruction branch**. No CEST parameter estimate was used to form an observation or calibration.

## Method Hints

Bounded nonlinear least squares on the complete six-pool spectrum gives a
method-independent target. A broad-background fit can initialize the stronger
water and MT components before jointly refining all components. For fixed
widths, centers and water shift, amplitudes form a bounded linear least-squares
subproblem. Multiple initial fits can expose competing explanations. Assess
data residuals and parameter-bound activity; amplitude and width compensate in
weak components, so a finite staged fit is not a uniquely identifiable truth.

## References

- Weigand-Whittier et al. (2026). *Pre-CAT: A web-based, graphical user-interface toolbox for preclinical CEST-MRI data processing and analysis*. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142558/).
- Example measured cardiac acquisition, [Figshare data](https://doi.org/10.6084/m9.figshare.30505685), CC BY4.0.
