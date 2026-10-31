# Implemented calibrated chemical-field inverse

## Binding, acquisition and unknown

The unknown is a dense pH image for every one of 43 real root acquisitions. Native float32 HPTS-ratio observations already include author preprocessing from sequential confocal channels; channel reconstruction is not part of this task. Nine independent pH-5 roots and eight independent pH-7 roots provide the experiment-specific ratio calibration. Target maps never supply those reference levels. The in-vitro four-parameter curve is a separately published calibration, not refit from target roots.

The actual microscopy uses 405/458 nm according to Methods and native LIF XML, while generic ratio labels and the in-vitro plate-reader characterization say 405/465 nm. This discrepancy is preserved explicitly. The inverse uses the supplied author's ratio scale and independent same-experiment references; no invented wavelength correction is applied. Native pixel spacing is 0.18037518037518038 micrometre, from ImageJ TIFF calibration. Images are full 1024-square native z1 planes; separate roots are not a volume.

## Reference normalization and forward response

For each reference root compute the mean of its finite ratio pixels, including finite zeros exactly as ImageJ statistics do. Then take the arithmetic mean of root means, separately for low and high calibration. Each root has equal weight, regardless of pixel count. The validated levels are L=0.3933395155239811 and H=1.875376803966015. These are computed from public calibration images, not hardcoded hidden fit values.

For measured ratio R, normalized response q=(R-L)/(H-L). The chemical response is

```math
q=B+\frac{T-B}{1+10^{h(c-p)}}.
```

Here p is dimensionless pH; T=1.1171, B=-0.0295, h=1.2133 and c=6.2110 are the exact released-code calibration parameters, with c measured on the pH scale. h controls the protonation response slope. Ratio and normalized response are dimensionless. The forward function directly evaluates the power-of-ten response.

## Direct inverse, clipping and output support

Clip normalized response to `[B+epsilon,T-epsilon]`, with epsilon=0.0001 as in the author implementation. For clipped q evaluate

```math
p=c-\frac{\log((T-B)/(q-B)-1)}{h\log(10)}.
```

The logarithm is natural; dividing by log(10) preserves the base-ten physical calibration. This is the selected published closed-form inverse, with no iterative optimizer, regularizer, fitted initial condition or convergence rule. Source `FloatProcessor` stores each calculated pixel as float32; the cleaned method casts after float64 normalization and inverse evaluation.

Only original NaN or exact-zero input ratios are unavailable. Finite negative ratios remain in the computation and can trigger the same asymptote clipping as the original code. Internally the source output would be NaN outside this mask; the benchmark's declared output uses exact-zero exterior for fully finite submissions. No estimated pH, drug label or calibration-range criterion selects pixels. pH 5–7 is a display/calibration range, not a clamp: all out-of-range and saturated values are retained and reported.

## Evaluation and scientific comparisons

Every root is evaluated separately against the unchanged original result on the same observations and independent references. NCC is uncentered cosine and NRMSE is RMS error divided by that root's reference dynamic range on all input-defined valid pixels. Every image must satisfy NCC >=0.999999 and NRMSE <=1e-6. The bounds were frozen before local extraction; the original float32 one-ULP envelope reached 5.695746871895818e-7 NRMSE. There is no independently measured target pH field, and reference agreement is not chemical accuracy.

The notebook preserves the task-relevant scientific comparisons: native spatial drug-condition maps and group distributions; calibration response and inverse sensitivity; calibration-root variation; fresh-versus-original numeric agreement; saturation/out-of-calibration coverage; and actual sensitivity to independent calibration choices. Full-field root statistics deliberately use every measured pixel. Published treatment statistics used manually selected late-meristem cells; those ROI files are absent, so whole-FOV effects are not called a reproduction of the historical ROI effect sizes or significance letters. Any historical values must remain separately labeled.

Independent checks use acid/base equilibrium odds to derive the forward response separately, verify monotonicity and roundtrip through original scalar inverse fixtures, check the natural/base-ten logarithm conversion, ensure equal-root weighting differs from pooled-pixel weighting, and retain saturation/negative-input behavior. No numerical threshold is changed after observing cleaned results.
