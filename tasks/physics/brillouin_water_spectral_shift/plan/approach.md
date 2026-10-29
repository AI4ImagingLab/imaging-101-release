# Implemented mathematics

The benchmark reproduces the later bmlab implementation of the extraction, Lorentzian peak estimation, nonlinear VIPA calibration and time interpolation described in the 2018 method paper. The paper used MATLAB and a methanol reference; this code uses two independent calibration substances. Do not identify the software example as a paper specimen or imply identical historical code. Actual material truth is unavailable.

## Coordinates, preprocessing and circular extraction

Raw arrays contain all 40 camera frames, uint16 ADU, analysis-oriented clockwise 90° once during packaging. Local runtime does not rotate them. Coordinates are row, column with center indices 0…399. All calculations subsequently use float64. Both ten-frame calibration means pass through SciPy `medfilt2d` with the default 3×3 zero-padded median filter. For mean image M, background b=median(M), candidate pixels satisfy M>b+10 ADU. Detection threshold is h=(mean(M[M>b+10])+b)/2. The binary mask M>h is closed with a Euclidean disk of radius 10 pixels and reflective boundary handling, then labeled with 8-connectivity. Components smaller than 20 pixels are discarded. Each remaining component contributes the maximum of the median-filtered intensity. Only spots within perpendicular distance 50 pixels of the image antidiagonal survive. All constants are the selected source defaults; binning is one.

Fit a circle to the six detected points. Initialization solves the circumcircle of the first, middle and last points. The local extracted `calculate_exact_circle` preserves the source algebra and collinearity adjustment. If more than three points exist, SciPy `minimize` with bounds R≥0 minimizes sum[((row-crow)²+(col-ccol)²-R²)²], with default numerical differentiation and default solver stopping. This is an algebraic circle cost, not radial distance least squares.

Source geometry approximates the circle boundary with 64 polygon edges, then intersects that boundary with the rectangle [0,400]². Local code explicitly constructs the same clockwise 64-segment polygon and computes each segment/rectangle-edge intersection; no geometry library is required. Sort intersections by row and map to polar angles using atan2 modulo 2π. Sample 500 angles between the first and last. The middle trace is c+R e(φ), where e=(cos φ,sin φ). The complete strip is c+R e+d e for d=-2,-1,0,1,2 pixels. Keeping this arithmetic order preserves source floating-point behavior. The physical fitted circle and polygon endpoint approximation are distinct stages.

For a camera image I and strip coordinates a, bilinear `RegularGridInterpolator` uses rows/columns 0…399 and NaN outside the detector. At each angular sample average the finite radial interpolates. The final bundled data happen to yield all finite extracted spectra; no synthetic replacement or crop handles missing values. Interpolate entire strip coordinates linearly between calibration times, held at the nearest endpoint outside the interval. Evaluate the trace at the measurement position's first-frame start time, as upstream; both frames at that position share this trace. Calibrators each use their own trace.

## Peak windows and spectral optimization

On each ten-frame calibration mean spectrum S, find peaks using prominence 15 ADU, height median(S)+15 ADU, and measured peak widths. If fewer than six candidates exist, repeat without minimum height. When exactly six peaks exist, use the mean detector position of candidates 3 and 4 as the center. Otherwise use the background-thresholded center of mass and adjust to ensure three candidates on each side. The two outer peaks are Rayleigh; the four inner peaks form two calibration doublets. A candidate region spans integer-truncated p±4w, where p is detector index and w is the peak-finder width. Only the upper endpoint is clipped to 500, matching source. Merge each side's two Brillouin windows into a doublet window. The test freezes all four resulting regions for both calibrations.

Every regional fit chooses nearest sample indices to its window endpoints and uses the right-exclusive slice. Pixel coordinates are used for calibration fits; frequency in Hz is used for specimen fits. Remove nonfinite pairs before fitting.

The single-peak shape is yhat=B+A h²/[(x-μ)²+h²], h=Γ/2. Initialize μ at the maximum; B=(first+last)/2; A=max(y)-B; Γ=2 times the distance from μ to the first sample above B+A/2. If Γ≤0, use 2 times the region span divided by sample count. Bounds are μ∈[first x,last x], Γ∈[span/N,span], A,B∈[0,2 max y]. `least_squares` uses its standard trust-region reflective method, two-point finite differences, ftol=xtol=gtol=1e-8 and default max_nfev. Crucially the residual passed to that optimizer is (y-yhat)², so its actual objective is 1/2 sum(y-yhat)^4. This behavior is preserved, disclosed and independently checked. It is not silently corrected to L2.

Each calibration doublet has two centers, widths and heights plus one background. Background and height initialization match the single-peak case, each initial width is 10 span/N, and centers are the two peaks with greatest prominence above 1 ADU, sorted by detector position. The same squared residual is passed to `least_squares`, with no parameter bounds, as source. Final six calibration centers are sorted by position. The method's possible poorly constrained calibration doublet parameters are not relabeled physical truth.

## Nonlinear frequency calibration

Define c0=299792458 m/s, λ=780.24e-9 m, f0=c0/λ, cavity index n=1.45367, thickness d=0.006743 m, incidence θ=0.8×2π/360 radians, imaging focal length l=0.2 m and camera pitch s=6.5e-6 m. The nominal free spectral range is F0=c0/(2nd cosθ), resonance order m=round(c0/(λ F0)) and relative order is zero. Instrument coefficients are

```math
 P_1=2\pi nd\cos\theta/c_0,\quad
 P_2=-2\pi nd\tan\theta\sqrt{1-(n\sin\theta)^2}/(c_0l),\quad
 P_3=-\pi d\cos\theta/(c_0l^2).
```

For first fitted calibration peak p0 let r0=p0 s. Initialize a0=(P1+P2 r0+P3 r0²)/(mπ), a1=(P2+2P3 r0)s/(mπ), a2=P3 s²/(mπ), F=F0. Absolute dispersion is V(p)=1/(a0+a1p+a2p²) Hz, relative dispersion v(p)=V(p)-f0. The known independent calibration shifts are q=(0,3.78e9,5.066e9,-5.066e9,-3.78e9,0) Hz; order vector o=(0,0,0,1,1,1).

The calibration objective contains three residual groups. With six ordered fitted centers p[j] and v[j]=v(p[j]), group one is v[j]-q[j]-F o[j]. Group two enforces sideband symmetry: (v1-v0)-(v5-v4), (v2-v0)-(v5-v3). Group three is q1-v1+v0, q2-v2+v0, q2-v5+v3, q1-v5+v4. Minimize the sum of squares of every listed residual using the source `scipy.optimize.fmin` Nelder–Mead defaults (xtol=ftol=1e-4, maxiter=maxfun=200×4=800, no bounds). Scale and stopping are preserved. Some fits reach maxfun and emit the source warning. Do not claim every optimization converged. The finite retained solution is independently assessed through standard-frequency residuals, monotonicity and repeatability; the notebook exposes all calibrator residuals.

Calculate all 500 frequencies for all ten frames in each calibration. Average these frequency axes within each calibration; linearly interpolate the mean axes in time, with constant endpoint extrapolation. Specimen frame times are measurement_time[position]+frame_index×exposure[position]. No unknown specimen peak, nominal target shift or hidden reconstruction contributes to calibration.

## Final inverse outputs and experiments

Fit one peak separately in Brillouin windows [4e9,6e9], [9e9,11e9] Hz and Rayleigh windows [-2e9,2e9], [13e9,17e9] Hz. Retain center, FWHM, height and background for all four regions, both frames and all ten positions. For each Brillouin center μb return min over the two Rayleigh centers μr of |μb-μr|. No averaging is applied to the declared shift output. Return full extracted spectra, time-specific frequency axes, calibrator frequency axes/coefficients and strip geometry as well.

Notebook-only comparisons actively recompute radial widths one and three (3/7 detector samples versus standard five), first-calibration-only frequency assignment, and ordinary L2 specimen peak fitting. The L2 variant initializes at the standard L4 fit, then minimizes unsquared y-yhat with the same bounds/defaults. It leaves calibration doublet estimation unchanged and is explicitly a sensitivity experiment rather than a claimed alternate full calibration. The first-calibration variant still computes both independent calibrations but assigns the first axis to all water frames. It isolates frequency-drift handling while retaining standard geometry interpolation. All observed linewidths remain instrument-broadened; no unsupported intrinsic-viscosity conversion is made.

## Evaluation and independent checks

Actual truth is unavailable. `baseline_reference.npz` is the independent original full-pipeline method result. Euclidean relative error is ||estimate-reference||₂/||reference||₂, independently scored for each mixed-unit parameter component. All metrics must be ≤1e-5, fixed before local extraction; no alignment or rescaling. This nonzero numerical allowance is much smaller than measured calibration/fit variability and does not establish physical truth. Original repeated numerical stages are bit-identical. Upstream stage tolerances are fixed separately: arc absolute 1e-7 pixels, spectral relative 1e-7 with 1e-7 ADU absolute floor, frequency absolute 1e4 Hz, calibration peak relative 1e-5. Analytic tests check circle geometry, bilinear interpolation, Lorentzian half-width, quartic-objective gradient, dispersion derivative and time endpoints. Every frozen stage and every required output has an identified consumer.
