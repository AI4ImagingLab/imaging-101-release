# Implemented polar algebraic Helmholtz baseline

The unknown is complex shear modulus G=Gprime+iGdoubleprime, Pa, independently at80 and100Hz on all128×80×25native voxels. Three phase-derived harmonic channels retain native z,x,y channel order. Public wave axes are batch,frequency,channel,z,y,x; internal axes frequency,x,y,z,channel. Support144637voxels is fixed from author magnitude geometry, not estimated modulus. Zero exterior is an output convention only: derivatives are computed on the complete observed grid before support is applied. Neither inclusion labels nor hardcoded springpot parameters are solver inputs or ground truth.

Acquisition model: rho*omega²*U+G*Delta_xy U≈0, rho1000kg/m³,omega2pi f rad/s, coordinates metres. The source is phase-derived y-gradient harmonic wave under a common1e-5amplitude multiplier. In locally homogeneous constant-Gregions, differentiation commutes with the Helmholtz equation; interfaces, pressure, out-of-plane derivatives and discrete processing violate this approximation. The task does not claim independently calibrated displacement. Positive-imaginary G convention is selected by the original polar arccos branch, not inferred independently from a unique sign of wave propagation.

## Fixed original preprocessing and documented wrapper discrepancy

Original sample phantom_unwrapped_dejittered.mat contains phase_unwrap_noipd and magnitude,80×128×25×8×3×8. Source load_mat metadata reverses MATLAB axes then arranges frequency,t,x,y,z,channel. Original individual author stages are run unchanged with explicit defaults: segment_regions(sigma=.8,threshold280),create_elastogram(),preprocess_wave_image(sigma=.65,truncate3,threshold100,order1). Circular phase smoothing angle(gaussian(exp(i phase))), gradient_y(exp(i phase))*exp(-i phase) imaginary part, forward FFTt,index1,2DButterworth100cycles/morder1perz, uniform1e-5multiplier and source repeated-harmonic averaging produce bound observed U. Selectednative80/100Hzretain full spatialextent. No new synthetic data or spatial resampling is introduced.

Important source bug discovered independently: preprocess(verbose=False) passes its positional False into segment_regions(sigma) and preprocess_wave_image(sigma), giving sigma0rather than intended defaults. The earlier exploratory zero-width run is archived and superseded. This task invokes unchanged individual functions with explicit default arguments, and records the difference; no silently modified source or claim of unchanged wrapper execution. The selected inversion begins from the resulting harmonic observations. Source function unwrap_phase actually computes a phase gradient rather than a phase map; physical terminology reflects executed math.

## Polynomial regularization and Laplacian

Use total polynomial order2and window9×9. Construct integer relative coordinates from -4to4using the original np.meshgrid ordering; monomials x^p y^q with p+q<=2form the local least-squares design A. Rows of pseudoinverse(A), multiplied by p!q!, give convolution kernels for polynomial value and derivatives. NumPy pseudoinverse uses its original default tolerance. Reflect convolution on x/y independently for every z/channel gives smoothed u. Apply second-derivative kernels to this already smoothed u, divide by mean coordinate spacing squared, and sum x/y. This two-stage smoothing is exactly author eval_ahi_baseline(savgol_filter=True,order2,kernel_size9), not a single derivative applied directly to unsmoothed observations. There is no z derivative.

The author's default meshgrid axis convention exchanges individual polynomial x/y derivative kernels relative to array axes; on this isotropic x/y spacing the summed Laplacian is unchanged. Preserve exact kernels/order; do not claim correctness for an unvalidated anisotropic grid. Independent quadratic-field tests show correct planar10Pa-relatedcurvature for2x²+3y² and no z²term. Optional finite-difference comparison uses successive np.gradient calls with actual coordinate arrays, edge_order1, exactly as author xarray.differentiate.

Despeckle the reciprocal Laplacian: let a=1/Lu. For complex a, treat magnitude and principal phase separately. Compute their reflected median with footprint outer([1,2,3,2,1],[1,2,3,2,1])>1on each z/channel plane. Where relative(a−median)/median exceeds+3replace with+3median; where below−3replace with−3median. This is source replacement by threshold times median, not standard clipping of a−median. Recombine abs and angle with exp(i angle), then invert again to obtain Lu. Preserve division and operation order; no target-dependent mask is fitted.

## Direct complex inversion

For each voxel, summing three channels c,

```math
A_G=\frac{\rho\omega^2\sum_c|u_c|}{\sum_c|L u_c|+\epsilon},\qquad
\phi=\arccos\left(\frac{-\sum_c(\Re u_c\Re L u_c+\Im u_c\Im L u_c)}{\sum_c|u_c||L u_c|+\epsilon}\right),\qquad
G=A_G\exp(i\phi).
```

epsilon1e-8is the source absolute denominator stabilizer, in corresponding numerical wave units; preserve it rather than pretending exact arbitrary-scale invariance with finite epsilon. Bound wave/Laplacian amplitudes make this tiny; independent common-phase/amplitude checks quantify invariance at declared tolerance. No cosine clipping is introduced. The original scalar-valued phase projection fixes phi∈[0,pi]; storage may be negative. Loss and storage are both reconstructed and scored. Alternative source nonpolar formula is −rho omega²sum(u)/(sum(Lu)+epsilon), an explicitly different diagnostic; never silently replace the selected polar method with vector least squares.

No optimization objective, iterative initialization, learning rate, regularizer parameter fitting or stopping criterion applies: this closed-form source baseline evaluates polynomial linear solves, deterministic filters and algebraic inversion once. The spatial regularization is fixed polynomial filtering. Independent exact complex Helmholtz pairs recover known G10000+2000iPa with correct omega² and density units; doubling frequency scales recovered Gfourfold, and common complex scaling cancels to epsilon tolerance.

## Evaluation and actual comparisons

Reference is independently executed original selected baseline, not measured truth. Evaluate real storage and imaginary loss separately on support at both frequencies using cosine NCC and reference-dynamic-range NRMSE. All eight component metrics pass independently; no magnitude/conjugation/phase/scale alignment. Two original reruns are bitwise; nonzero1e-7NRMSE/NCC.999999thresholds frozen before local. Additional per-frequency vector residual R=norm(G Lu+rho omega²u)/norm(rho omega²u), on full support/allchannels, uses exactly selected regularized derivatives. Original residuals.2501509234/.2736138916; common boundary1.01max=.2763500306. Residual is approximate-model consistency, not absolute accuracy. No hardcoded springpot map is named ground truth.

Notebook actively reconstructs raw bound harmonic data and compares original/fresh storage/loss, complex phase, wave and derivative structure, support/boundary errors, frequency dispersion and derivative sensitivity. Fresh quadratic windows3/5/7/9/11, direct derivatives, reciprocal-filter omission and source nonpolar formula expose tradeoffs; no variant is substituted for default or omitted because it performs poorly. Comparisons are conditional on same experimental acquisition, not independent biological replicates. Material assumptions may be retained only as explicitly uncertain historical context, not acceptance truth.
