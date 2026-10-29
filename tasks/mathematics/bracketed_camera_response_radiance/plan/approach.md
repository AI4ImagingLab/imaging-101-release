# Implemented photometric inverse

The benchmark binds all seven427×640 RGB author JPEG observations; the private conversion merely decodes integers and copies EXIF shutter times. There is no simulated observation generator. The unknowns are a positive relative irradiance E at each pixel/channel and a256-code inverse response h per channel. The exposure model is z≈f(E t), h=f inverse, g=log h. Units are ADU for z,seconds for t,relative exposure for h and relative exposure/s for E. Code128 has h=1, fixing three otherwise independent channel gains. Absolute radiance and a physically calibrated white balance are not identifiable.

## Sample extraction and objective

`sample_stack` uses the reference RegularSpatial strategy with requested256 sites. Set f=round(sqrt(256)+1)=17, horizontal step ceil(640/17)=38 and vertical step ceil(427/17)=26. Start at zero-based(0,0), use all sites through the full image, and order sites by column then row, matching MATLAB column-major flattening. Retain a site when at least one channel decreases strictly as exposure decreases. All channels at retained sites participate. The bound images retain65 sites. This is an observation-only quality rule, not target-based selection.

For each channel solve for q=(g(0),…,g(255),ell(1),…,ell(N)), where ell is sampled log irradiance, N=65 and J=7 exposures. Let w(z)=2min(z/255,1−z/255) and lambda=128. Minimize the unnormalized sum of squares

```math
\sum_{i=1}^{N}\sum_{j=1}^{J}w(z_{ij})^2
 \left(g(z_{ij})-\ell_i-\log t_j\right)^2
 +g(128)^2
 +\lambda^2\sum_{z=1}^{254}w(z)^2
 \left(g(z-1)-2g(z)+g(z+1)\right)^2.
```

There is no division by N or J and no implicit one-half factor. The gauge row has weight1. The discrete curvature is the centered second difference in integer ADU coordinates, with no fictitious outside-code boundary values. `response_system` constructs the original rectangular system including its two unused zero rows. `scipy.linalg.lstsq(...,lapack_driver='gelsd')` directly solves the linear least-squares problem in float64; a rank-deficient system raises an error. No iterative optimizer, initialization, step size or stopping criterion is needed. Set h=exp(g). Monotonicity holds on the validated observations but is not imposed as a different constrained algorithm.

The selected path calls the original `gsolve` directly, then `BuildHDR`. It deliberately retains the paper's midpoint gauge rather than the optional `DebevecCRF` wrapper's chromaticity optimization/global normalization. This selection was fixed before local code. It is a reproduction of the specified published implementation path, not a claim that the entire toolbox demo or1997 executable is unchanged.

## Complete exposure merge

Normalize uint8 codes to float32 v=z/255. Normally weight each code by w. In the shortest exposure, if any channel/pixel exceeds0.9, use identity weight v for that whole exposure. In the longest exposure, if any code falls below0.1, use reverse weight1−v for that whole exposure. These global endpoint switches are preserved from the original source; they are not a new pixel-wise saturation mask.

Nearest-code table lookup gives h(z) in float64. Accumulate numerator S and weight sum W in float32, casting after every exposure exactly as the source's single/double arithmetic requires:

```math
S\leftarrow\operatorname{float32}\left(S+w\,[\log(h(z)+\delta)-\log t]\right),
\qquad W\leftarrow\operatorname{float32}(W+w),\qquad
\delta=1/65535.
```

Compute exp(S/W) in float32, then return float64. For W≤1e−4, the middle exposure selects a bright or dark fallback. The channel mask is expanded to every channel at that pixel; use shortest-exposure h(z)/t for bright pixels and longest exposure for dark pixels. Preserve the source's MATLAB rounded middle-index convention. Reject any remaining nonfinite or nonpositive value. The additive delta, global endpoint weights and float32 accumulation are implementation conventions beyond the ideal logarithmic paper equation; validation records them explicitly.

## Comparisons, robustness and evaluation

The active notebook reruns calibrated reconstruction, uncalibrated linear and gamma2.2 response baselines, response regularization and sample-density comparisons, exposure-subset ablations and labeled added-code-noise perturbations. No perturbed result is called measured input. Baselines differ in assumed photometric response and remain separately labeled. Tone-mapped previews are display-only and cannot replace irradiance/response comparison.

Independent original results supply the method reference, not GT. The Judge independently measures raw-irradiance cosine NCC, log-ratio RMSE for both unknowns, response gauge error and weighted forward ADU residual. All image/code entries enter method agreement; no fitted gain or image alignment is allowed. Forward support uses observed codes8–247 only, weight min(z,255−z), and monotone log-response interpolation with endpoint clipping. The baseline residual7.035289421164854ADU is model mismatch, not a noise estimate. Its boundary7.105642315376502 was frozen before local implementation. Independent original reruns were bit-identical; positive5e−5 log-error tolerances allow float32 backend differences. Such agreement does not prove physical truth.


## Public-contract revision 2026-10-29

V2 replaces sampled-response/finite-fusion array identity with the original measured-code RMS limit7.105642315376502ADU, monotonicity/midpoint gauge and a public log-response second-difference regularity class. No actual irradiance/response truth exists. Independent full-observation Schur-complement photometric LS and log fusion, without source or hidden response in the solve, reaches5.45973770163ADU and curvature.000182705876 versus allowed.001757587274. Both outputs remain evaluated jointly by their forward predictions. Source/output arrays unchanged.
