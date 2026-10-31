# Implemented Approach

This document describes the current corrected mathematical implementation. Complete 200-start validation and research notebook execution remain pending; see `test/validation.md`. Corrections to the upstream discretization are explicit and independently checked, not an unchanged reproduction claim.

## 1. Physical Unknowns And Bound Data

Two complete experimental scans P1/P2 of a gold-coated ZnO microrod are bound to this task. Each has 121 rocking frames and 256 by 256 Timepix pixels. Public `photon_counts` is int32 `(1,2,121,256,256)` and `theta` float64 `(1,2,121)` degrees. No angular or detector plane is cropped, no photon noise is added, and no hidden reconstructed object initializes the CLI. Each position is reconstructed independently before its ensemble is combined.

For one position let `N=(126,256,256)`, `V=prod(N)`, `x` be a complex density on that oblique grid and `k` a nonnegative 16 by16 by16 reciprocal-space coherence kernel. Its mass is one after each completed kernel update. The original observation grid is `(121,256,256)`; the complete selected source preprocessing defines the larger reconstruction grid. Arithmetic promotes to complex128/float64 after the first support projection. Observation units are counts; unscaled Fourier amplitudes are square-root counts. The submitted density retains physical photon scale.

Before inversion, execute complete source preprocessing in order. For counts `I`, compute `t=max(2,0.141*mean(I[I!=0])-3.062)` counts. Set `I<=t` tozero in a float64 array, take its square root, and zero-pad to the smallest even fast FFT size using SciPy's `next_fast_len` followed by repeated odd-length correction. This adds two rocking planes before and three after the original scan, and none on detector axes. Let `P` denote that embedding, `j` its first C-order maximum, `s=floor(N/2)-j`, and `R(s)` a cyclic roll. Fit `a=float32(R(s) P sqrt(where(I<=t,0,I)))`, preserving the source TIFF storage precision after centering. P1/P2 thresholds are2.1296440957926044/2 counts and shifts are `(1,-19,-3)` / `(0,-49,-3)`. Auto threshold is an explicit reproducible released-method choice, not a verified historical author setting. No alien removal, binning, explicit crop or additional shift is configured. Public raw counts remain unchanged.

The peak roll defines the density reciprocal origin and central16-cube PC ROI. For physical predictions, evaluate finite linear blur first, undo the roll, then remove only the five added planes: `Iraw_pred=Q R(-s)(|Fplus(x)|^2*k)`, where `Q` extracts rocking indices2:123. No original measurement is omitted from raw consistency. These operations cannot be exchanged with finite convolution. Ranking and final public-data scale use the thresholded centered126-grid amplitude `a`; the final raw121-grid residual is separately computed against every original count, including counts suppressed internally. The padded planes are part of this selected discretization, not acquired observations.

The nominal wavelength is `1.2398419843320026/9` nm, pixel pitch 55000 nm, detector distance 2.8e9 nm, delta 28.488 degrees, gamma 0.0478 degrees and uniform rocking increment 0.002 degrees. All angular formulas convert degrees to radians. Exact recorded motor arrays are public, but the selected inverse discretization uses the nominal first-order uniform angular grid. Small motor jitter is not silently described as an exactly uniform measurement.

The 2.8 m distance is independently established in the experimental paper; the NeXus string `2800` with an `m` label is wrong. Nominal geometry implies `|q|=22.444636221... rad/nm`, whereas the article reports 22.319 rad/nm. The 0.56% difference is unresolved. The task therefore reports fields in its declared nominal geometry and does not reproduce absolute article strain, refraction correction, electric fields or a full strain tensor.

## 2. Geometry And Dual Sampling Grids

Laboratory coordinates are `(outboard, up, downstream)`. Array indices are `(rocking, detector row down, detector column)`. Set `c=2*pi/lambda`, detector pitch `p`, distance `L`, in-plane angle `d`, out-of-plane angle `g`, and rocking step `t` in radians. Reciprocal basis columns are

```math
b_0=ct(1-\cos d\cos g,\;0,\;\sin d\cos g)^T,
```
```math
b_1=c(p/L)(\sin d\sin g,\;-\cos g,\;\cos d\sin g)^T,\qquad
b_2=c(p/L)(\cos d,\;0,\;-\sin d)^T.
```

Here the sign of `b1` follows the original detector y- convention and the sign of `b2` the x+ convention, whose original orientation lookup factors are +1 and -1 respectively. The nominal reflection vector is

```math
q=c(\cos g\sin d,\;\sin g,\;\cos d\cos g-1)^T.
```

Let `B=[b0 b1 b2]`, rad/nm per reciprocal index. The real-space voxel basis is `D=2*pi*B^(-T)*diag(1/N)`, nm per real-space index. Thus `B.T@D=2*pi*diag(1/N)`. This dual identity and the original beamline transformation are independently tested; a scalar detector pitch estimate alone would not establish oblique geometry correctness. Images must not be relabeled as isotropic 27.4 nm voxels. A Cartesian visualization would require explicitly stated interpolation; the solver itself uses the oblique grid.

## 3. Centered Fourier Acquisition Operator

Let `C=fftshift` and `C^-1=ifftshift`, each on all three axes. The operator and its inverse are

```math
F x=C\;\mathrm{ifftn}(C^{-1}x,\mathrm{norm}=\mathrm{forward}),
\qquad F^{-1}z=C\;\mathrm{fftn}(C^{-1}z,\mathrm{norm}=\mathrm{forward}).
```

Consequently `F` is an unscaled positive-sign DFT, `F^-1` is a negative-sign DFT divided by `V`, and `F^*=V*F^-1`. A constant object has its reciprocal maximum at `floor(N/2)`, including index63 on the126-plane rocking grid. Centered operator routines also support odd diagnostic grids; centered amplitudes and coherence calculations use consistent forward/inverse shifts.

For coherent intensity `g=abs(Fx)^2`, the observation model is `mu=g*k`, where convolution is linear with zero extension and a fixed finite output window. It is not circular convolution. Keeping low-frequency neighbors adjacent in the centered array prevents coherence from losing neighboring intensity at a corner-order array boundary. `full_convolution` uses real FFTs padded to the next fast length at least `N+K-1`, trims the full convolution to exactly that size, then `linear_convolution` crops to the first input's shape. Negative roundoff in an otherwise nonnegative intensity is clipped at zero before square roots.

## 4. Amplitude Constraint And Reported Optimization Error

At an object iteration let `z=Fx`. Before PC activation, predicted amplitude is `b=abs(z)`; afterwards `b=sqrt(max(abs(z)^2*k,0))`. Define the multiplicative ratio to be `a/b` where `b>1e-9` and zero otherwise. The projected wave is `z'=z*(a/b)`, and the object candidate is `p=F^-1 z'`.

The upstream optimization diagnostic is squared relative amplitude error

```math
E=\frac{\sum_j r_j^2}{\sum_j a_j^2},\qquad
r_j=\begin{cases}b_j-a_j,&b_j\ne0,\\0,&b_j=0.\end{cases}
```

The zero case and absolute ratio cutoff are retained explicitly. There is no extra `1/V`, Poisson weighting, background subtraction or arbitrary data normalization. Measured amplitude is float32; its native squared-amplitude reduction preserves float32 source precision, whereas the promoted field/residual arithmetic is float64. Each error is evaluated from the iteration's incoming object, before its wave-amplitude and real-space updates. Thus `errors[-1]` ranks the incoming object at update index1019, whereas the returned density is the result after that final ER update and maximum-magnitude normalization. No extra output-residual evaluation replaces the source ranking. The separately computed thresholded-forward and original-raw residuals use the actual returned/scaled output and need not equal the native final error. In the independent raw metric, take square roots of float32 original counts and accumulate the squared denominator after float64 conversion; this metric convention is separate from the native reduction. This diagnostic ranks runs; it is not a ground-truth accuracy claim. The PC wave-amplitude correction follows the selected published algorithm family and should not be described as an orthogonal projection onto a linear measurement subspace.

## 5. ER, HIO And Adaptive Support

With support indicator `s` and feedback `beta=0.9`, an ER update is `x_new=s*p`. HIO uses `p` where `s>0` and `x-beta*p` elsewhere. No positivity is imposed on complex density or its phase. There is no TV prior, neural network, hidden-domain solver, generic optimizer substitution or amplitude-only reconstruction.

For shrink-wrap, smooth `abs(x)` with a Gaussian of width `sigma` array voxels, half-sample reflect boundaries and truncation radius `floor(4*sigma+0.5)`. The new support is one where the smoothed value is **greater than or equal to** 0.1 times its maximum. The Torch diagnostic backend uses explicitly reflected indices and three separable real convolutions; the validated CuPy reference backend uses the generic reflect-boundary Gaussian primitive with the same convention. It does not use PyTorch reflect padding, whose edge convention differs.

There are five 200-step cycles, each 20 ER followed by 180 HIO, followed by 20 final ER steps: exactly 1020 iterations. Cycle Gaussian widths are 3, 2.5, 2, 1.5 and 1. This is a disclosed cohere-compatible implementation of the article's stated low-to-high-resolution width range, not a verified SI table. Support triggers occur ten steps into each ER/HIO segment and every ten thereafter; in a 200-step cycle these are 10,30,40,...,190, **not** 20. The final 20-ER segment triggers at 1010. The actual local implementation tests this segment boundary explicitly.

## 6. Partial-Coherence Unknown And Exact Finite-ROI Kernel Update

PC is activated at object iteration 400. Kernel updates occur at 400,450,...,1000. At each immediately preceding iteration, save the centered 16-cube ROI of the amplitude after its wave constraint. Let `u` be the current centered Fourier-amplitude ROI and `v` that saved previous ROI. Define `h=(2u-v)^2`, normalized so its sum equals the measured ROI's squared-amplitude sum. This extrapolation and ROI flux normalization are retained from the selected upstream method.

For fixed coherent ROI `h`, measured intensity ROI `y`, kernel `k` and kernel shape `K`, the finite forward operator is

```math
(A_hk)_i=\sum_u h_{i+o-u}k_u,\qquad o_a=\lfloor(K_a-1)/2\rfloor,
```

where out-of-bounds `h` is zero. Its exact transpose is

```math
(A_h^Tr)_u=\sum_i h_{i+o-u}r_i.
```

Compute that transpose as full convolution of `r` with flipped `h`, then crop starting at `R-1-o`, returning `K` values per axis, where `R` is the coherent ROI shape. For even 16, origin is 7 and transpose crop begins at 8. A second `same` convolution begins at 7 and is therefore the wrong transpose. Frozen and independently constructed dense-matrix tests include even kernels.

The inner multiplicative update is

```math
k\leftarrow k\;\frac{A_h^T\!\left(y/\max(A_hk,10^{-30})\right)}{A_h^T1}.
```

Zero sensitivity locations use denominator one. Clamp negative FFT roundoff to zero after the update. Perform exactly 20 inner steps; then take the real nonnegative kernel and normalize its mass to one, following the selected final mass projection convention. This is finite-window Richardson-Lucy followed by explicit mass normalization; it is not claimed to be an exact constrained-likelihood minimizer at each step. A nonconstant sensitivity cannot be omitted merely because a kernel is normalized afterwards.

The original cohere call reverses measured/coherent arguments, uses a miscropped even-kernel transpose, omits sensitivity and convolves corner-ordered intensities. These scientifically consequential corrections are independently validated and archived as an exact private source diff. Historical unmodified outputs remain separately labeled; corrected outputs must not be called unchanged upstream results.

## 7. Initialization, Ensemble And Physical Scale

For each position and start, seed a NumPy default generator with `10000+1000*position+start`, with positions 0/1 and starts 0..199. Draw real/imaginary uniform float32 components into complex64. Scale the random array by its float32 squared norm times the maximum measured amplitude, retaining the reference initialization convention. Multiply in place by a centered half-box support of shape `floor(N/2)`; padding begins at `floor((N-floor(N/2))/2)`. The initial support is float64, and the first real-space multiplication promotes the iterative density to complex128.

The protocol uses all 200 independent starts per position. A shorter `iterations` argument in the per-start numerical helper is only for explicit diagnostics; the CLI supplies none and always uses 1020. There is no early stopping or best-seed shortcut. Rank by final `E`, breaking equal errors by start index, and retain the ten lowest-error objects and kernels. Finishing all starts is required even if early runs have low residuals.

Upstream per-start results are divided by their own maximum magnitude. For complex averaging, align each of the selected nine other objects to the first: compare original and conjugate-flipped candidates, use full complex cyclic cross-correlation to select an integer shift, and multiply by its best constant phase. No subpixel deformation, density rescaling or fitted phase ramp is performed. Average aligned complex objects and the corresponding unit-sum kernels. Density and kernel averages are scientifically different quantities and are not mislabeled as truth.

The shipped complex mean copies the first selected object, adds each subsequently aligned object in the original selected order, then divides by the number selected. It no longer retains a second list of all aligned arrays or creates a ten-object reduction stack. Actual P1 selected10 old-stack mean and this ordered sum were bitwise equal, including alignment and the saved object after the fixed public-amplitude scale. This changes array lifetimes, not the ensemble formula or operation order. The current reconstruction caller spills only its newly computed qualifying selected objects to temporary disk, caches one fixed anchor and loads one candidate at a time during aggregation. It no longer retains ten original arrays1321205760B in RAM. The historical lazy proof preceded this caller change; current disk-backed actual10 proof is separately identified. Final stacking momentarily holds the two result-list arrays264241152B and the new combined stack264241152B simultaneously.

The article Methods establishes 200 random starts, error-based selection and correlation of chosen results, but does not state that ten were selected or specify integer versus subpixel alignment and the averaging formula. Those details cannot be inferred from the inaccessible SI. The ten-object integer/twin/phase complex mean is therefore an explicit postprocessing convention of this released variant. It is not an extracted historical-author averaging routine. In cohere4.5, `dvc_utils.align_arrays_subpixel` registers magnitudes and applies a subpixel shift for genetic breeding, while `Rec.average_operation` accumulates amplitudes within a run; neither is this ensemble calculation. The selected convention averages complex estimates on the declared discrete grid without interpolation, with all symmetries and limitations stated explicitly. Agreement between private and local implementations validates that convention, not the unknown historical postprocessing.

Scientific review must also retain structured count residuals, weak diffraction tails not explained by the model, relative secondary density levels and coherence-kernel complexity. The complete P1 pilot is credible for continued selected-variant benchmark validation, but its stronger secondary lobes and nontrivial inter-start variation do not establish published morphology or true strain. Three-direction measured/predicted log-count projections and signed residuals expose discrepancies without a display-specific scale fit. P2 scientific review and complete fresh two-position execution remain necessary before freezing final acceptance.

Restore photon scale after averaging. Let `b=sqrt(mu(x,k))`; the least-squares amplitude scale is `alpha=sum(a*b)/sum(b^2)` on the thresholded centered126 grid. Save `alpha*x` and the mass-normalized averaged kernel. This scale is computed from public observations, not a reference fit. Raw consistency then uses the saved scale without fitting another scalar. Final arrays are complex128 `(1,2,126,256,256)` and float64 `(1,2,16,16,16)`.

## 8. Relative Projected Strain And Limits

The optional scientific review map is `epsilon=(q dot grad(phi))/abs(q)^2`, with `phi=arg(x)`. Index-space central phase differences use `arg(x[n+1]*conj(x[n-1]))/2`; multiplying by `D^(-T)` converts to a laboratory gradient per nanometer. This requires local phase increments below the wrapped-difference ambiguity; no claim of general three-dimensional defect phase unwrapping is made.

Use magnitude at least `exp(-1)` times the maximum as crystal support, erode by a 3-cube so both neighbors are valid, remove the arithmetic interior mean strain, and set exterior values to zero. The mask and relative offset convention are fixed independently of any hidden reference. Constant complex phase changes no strain. This single-reflection directional strain is not a six-component tensor. The task does not infer electrostatic potential, elastic moduli, absolute hydrostatic strain or refraction effects from the nominal data.

## 9. Validation And Acceptance

The frozen corrected-oracle fixtures cover centered Fourier conventions, amplitude loss, ER/HIO, support, PC forward/kernel updates including actual even16 kernels, and nominal geometry. Independent tests separately establish FFT adjoint scale, finite convolution against direct sums, exact dense forward/transpose/sensitivity, centered odd121 impulse behavior, physical broadening-kernel recovery, and dual geometric sampling. A fixture generated by local source is not accepted as oracle evidence.

The complete-preprocessing unmodified diagnostic executed all 1020 updates and failed numerically during partial coherence; its nonfinite result is archived as a failure. Corrected-source and local full1020 single-start trajectories passed for both positions (captured support parity is available for P1 only). Both independent corrected-source200-start acquisitions and selected10 aggregations completed. The method-reference Judge has16 prospectively frozen, nonzero numerical tolerances where applicable and passed all15 actual bound-reference positive, negative and malformed-input controls; this establishes selected-method consistency, not physical accuracy. Mandatory remaining checks are fresh cleaned full400 execution, complete ensemble parity/quality analysis, active notebook execution and figure inspection. The fresh run's resource guard is currently waiting for its declared start headroom. No unexecuted reconstruction is claimed. Boundaries cannot be relaxed because a fresh result fails. Actual reference reconstruction is not physical ground truth.

## 10. Executed Numerical Conventions And Backend Selection

The validated full-trajectory backend is generic CuPy13.6 using the existing CUDA toolkit; all imaging-specific mathematics remains explicit locally. All squared GPU amplitudes/intensities/errors use `square`, whose executed rounding differs from generic power on this backend. The initial PC kernel is float64: the original GPU `full` wrapper ignores its requested dtype, a recorded source/actual-behavior discrepancy. Finite-window sensitivity uses float64 ones. ROI normalization deliberately preserves square, normalization, sqrt and square operations. Ten-checkpoint parity on the historical 121-plane diagnostic isolated these conventions. On the current complete-preprocessing 126-plane grid, the independently executed 1020-step source/local pair gives final object relative L2 2.19121024248e-16, exact kernel and support, and exact full error history. The Torch backend remains a diagnostic alternative and is not claimed to preserve the identical nonlinear branch. Failed comparisons, historical-grid evidence and current-grid evidence are separately identified in validation.

## 11. Illumination And Morphology Interpretation

The experimental description specifies a nominal 2000 by450nm horizontal/vertical beam after KB focusing and a slit. No independent three-dimensional complex probe array or exact profile is provided. The reconstructed `x` is therefore illuminated/probe-weighted scattering density, not an inferred complete rod shape after probe deconvolution. Interpreting its phase as displacement assumes negligible additional spatially varying illumination phase. The paper reports mean diameter940+/-19nm from fringe spacing and domain927+/-25nm from diffraction FWHM; these are physical context for a scientifically reviewed reconstruction, not arrays used as hidden truth or values imposed by fitting. The article0.6um refers to a selected front-section interval for discussion, not the diameter. Physical extent checks must transform oblique coordinates through the specified basis; stored index widths are not nanometers. The present method does not claim the article orthogonal27.4nm resampling.

## Historical v1 Standalone Reference-Consistency Metrics (superseded by v2 below)

The Judge independently constructs nominal q,D from the README constants. Its fixed phase mask is abs(reference)>=exp(-1)*max(abs(reference)); full3x3x3 zero-boundary erosion fixes the strain interior. It uses the declared integer/twin/globalphase alignment for both metrics, principal angle(candidate*conj(reference)) phase RMS (pi error for predicted zero), and central complex-ratio derivatives divided by2. The nominal scalar strain is q dot inv(D).T dot g / dot(q,q), dimensionless. Reference and candidate means are removed separately on exactly that same interior; RMS of their difference is reported. No submission-dependent support or additional fitted ramp is allowed. This differs intentionally from per-field display masks in the visualization helper. Strict prospective numerical boundaries are frozen privately before new measurements; final acceptance and testing remain pending. Neither metric certifies true or paper strain.

### Current complete-grid trajectory evidence

Both positions now have proper complete1020 local/source parity on actual captured initializations: P1 finalcomplex relativeL2=2.1912102424769778e-16 and P2=2.145917167659144e-16; kernels and all1020errors exact. P1 additionally has captured source support parity; P2 source start0 saved no support, so no P2 support claim is made. The prior55-test suite passed including active P1GPU parity; the added selected-storage unit and related physics/stage subset passed18 tests. Actual selected10 disk-backed aggregation proofs for P1 and P2 are bitwise equal to the previous in-memory objects, kernels, alignments and scale. Both private source200-start ensembles completed and were visually reviewed, with weak-tail/ring discrepancies and count deficits explicitly retained. All15 actual bound-reference controls subsequently passed their expected Judge exit. Fresh400 ensemble parity, independently applied fresh Judge, the final complete test rerun and full notebook execution remain open; single-start parity is not ensemble or physical accuracy proof.


## Current v2 conditional evaluation

V2 removes complex-object/kernel/derived-strain method identity while retaining every required output through the raw physics and compact prior. Each of the two complex fields and corresponding PSFs must meet unchanged full-volume raw amplitude residual limits(.083328729,.096088796), unit PSF mass within1e-12, nonnegative PSF and compact-support volume<=.125. The own-object shrinkwrap is reflect Gaussian sigma1/truncate4, threshold.1of own maximum, integer peak centered, evaluated in direct/twin representatives. Outside-support energy may not exceed attained standard by more than1e-8; no hidden reference mask is used. The.125cap derives from the documented half-box support, not new fitted strain truth. Translation/twin/constantphase equivalent controls pass; wrong amplitude scale fails. Forty-five unit tests passed after replacing the retired identity-direction assertion with a live raw-residual direction assertion. Strain remains a derived, finite diagnostic from nominal q,D and principal central complex phase ratios, dimensionless and mean removed on own eroded amplitude interior. It is not an independent output or a true-strain claim. The joint object/PSF and illumination ambiguity is stated publicly. No arrays changed.
