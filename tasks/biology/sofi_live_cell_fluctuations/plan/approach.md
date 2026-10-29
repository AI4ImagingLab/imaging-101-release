# Implemented mathematics: conventional moment-SOFI

## Bound observations and unknown

Three complete real 200-frame, 450 by 450 movies are source blocks1,10,20 of a live HeLa cell expressing Dronpa-C12-beta-actin. Camera values remain uint16 ADU; no new synthetic benchmark observations are generated. Each block is analyzed independently on native detector pixel centers (column x increasing right, row y increasing down). Absolute object-plane pitch and block timing are not reliably available, so this task does not claim absolute resolution in micrometers or motion speed.

The acquisition model is F(r,t)=sum_j epsilon_j h(r-r_j)b_j(t)+B(r)+eta(r,t). Here r is detector coordinate, t is frame index, j identifies N emitters, epsilon is on-state brightness in ADU, h is normalized intensity PSF shape, r_j is emitter coordinate, b_j is binary blinking, B is stationary background and eta is noise. Independent blinking motivates spatial fluctuation imaging. The selected unknowns are detector-defined moment-SOFI contrast images, not unique concentrations or emitter coordinates.

## Direct inverse statistic

For each pixel, mu=(1/T)sum_t F_t, where T=200. Accumulate mu by sequential float64 addition of the uint16 frames in original frame order. For order n=1,2,3,4, compute m_n=trunc_to_int64[(1/T)sum_t(F_t-mu)^n]. Truncation occurs once after summation and averaging, toward zero, matching original reconstruction.calc_moments exactly. Computation uses float64 before that cast; individual TIFF values and source observations are unchanged. Cast m2 and m4 to float64 for the declared output. No spatial discretization beyond native detector sampling, boundary extension, convolution, interpolation or FFT is involved. The public camera-count moment definition makes truncation explicit; do not replace it by unbiased sample corrections.

The standard entry point returns cumulant2=m2 (ADU squared), and reconstruction=m4 (ADU to fourth power), each (1,3,450,450). The name cumulant2 is retained because the first central moment truncates to zero, making the second moment and second cumulant identical. There is no iterative objective, loss normalization, initialization, regularization, step-size, line search or stopping parameter: this is a direct empirical-moment inverse statistic. The only optional solver argument frames=None uses all frames; integer frames=N restricts each block to its first N observations for controlled review experiments.

## Cumulant comparison and selection rationale

The general cumulant recursion C_n=m_n-sum_{i=1}^{n-1} binom(n-1,i) C_{n-i}m_i is implemented locally in physics_model.cumulants and validated against the original independent execution. For the bound recordings m1=0, so C2=m2 and C4=m4-3m2^2. All signs are preserved in that comparison. The original source's fourth-order moment was captured before extraction together with its cumulants. Actual rendered comparison found fourth cumulants noise dominated for T=200. The main output was therefore fixed to the author's conventional fourth-moment branch before final acceptance, preserving C4 as a visible finite-record failure comparison. This is a documented method selection, not a correction of an upstream formula or a relaxed threshold. Original-reference relative-L2 boundaries remain1e-6, and fresh discrepancies are zero.

Cumulant additivity removes cross-emitter terms under independent stationary blinking. A Gaussian PSF raised to power n has width sigma/sqrt(n). That law describes the ideal cumulant PSF and is not asserted as a measured resolution law for the selected fourth moment, whose mixed terms survive. The method yields spatially informative live-actin contrast with explicit brightness and finite-sampling limitations. Do not interpret higher-order visual contrast alone as quantitative super-resolution truth.

## Active review and evaluation

The notebook actively reconstructs every block, compares original and fresh moment images, recomputes first25/50/100 frame estimates, shows full200-frame results, temporal traces, fixed center spatial profiles, and a separate fourth-cumulant failure map/negative fraction. Display roots are only for matching units on figures; saved arrays retain original ADU powers. No SOFI2.0 Fourier interpolation, noise filter, shrinking-kernel deconvolution, LDRC or cosmetic postprocessing is claimed in this selected conventional method. The original E1 moments/cumulants comparisons are preserved in the review; E11's separate full postprocessing pipeline is explicitly outside scope.

Evaluation uses relative L2 norm per array over every declared entry, no masking/crop/alignment/rescale. There is no independent material truth; the independently executed original is a standard-method reference. Passing means numerical method agreement, not physical accuracy.
