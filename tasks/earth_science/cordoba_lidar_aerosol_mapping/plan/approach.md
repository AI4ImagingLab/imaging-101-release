# Implemented elastic lidar inversion

The selected method is LPP's fixed-lidar-ratio Fernald1984elastic inverse, as described in the2023LPPpaper section2.3. It is not an AERONET-calibrated lidar-ratio estimate or a Raman inversion. Real532nm analog observations cover66complete time integrations. Both returned fields contain66times759heights; extinction is tied to backscatter by a50srprior, so there is one independent aerosol field rather than two unconstrained fields.

## Bound observation and calibration

The original L0reader processes all399native LICEL files for2024-10-02at author LidarPi site(-31.2N,-64.1E,411mASL). Original L1processing takes complete six-record analog means, retaining66groups and discarding the incomplete final3record group. The selected channel is analog532nm,index6. Its5bin trigger shift and25000to30000mmean background correction precede range-squared correction. Gluing and polarization mixing are explicitly disabled because the provided2019configuration does not establish suitable2024interchannel calibration. No dark or overlap correction is fabricated. The selected heights assume full overlap above the excluded lower range. These are explicit known preprocessing assumptions, not target-fitted calibration.

The runtime receives the original calibrated range-corrected return S in analog ADC-count*m², not raw photon counts. Its absolute analog gain is unknown and cancels in reference normalization. Public known molecular beta_m[m^-1sr^-1] and alpha_m[m^-1] come from the source USStandardAtmosphere thermodynamic table and Rayleigh calculation, including the actual L2resampling. L1stored calibration differs slightly from L2's re-evaluated prior; the actual original L2prior was independently captured and bundled. This source calibration is target-independent and constant across all66vertical profiles. No radiosonde or aerosol ground truth is claimed.

Use4096centers r(i)=(i+.5)*7.5m. Reference index812comes from the source's configured6000to7000mASLwindow,indices745to879inclusive. Its center is6093.75mAGL, whereas source index-based printed reference coordinate is6090mAGL/6501mASL. This half-bin labeling distinction is preserved explicitly; the actual public range centers define output coordinates. The reference return is the source linear least-squares fit of observed range-corrected return to known molecular return plus a constant over the configured window. The source replaces exactly the reference signal sample with that fitted value; this calibrated observation is public. Boundary ratio R_ref=beta_total(ref)/beta_m(ref)=1is an aerosol-free prior.

## Forward model and direct inverse

Let beta_a(r,t) be aerosol backscatter, alpha_a=L*beta_a its extinction with L=50sr, beta_m/alpha_m the known molecular fields, and G(t) unknown gain. The elastic equation is S=G(beta_m+beta_a)*exp[-2 integral(alpha_m+alpha_a)dr]. The molecular lidar ratio L_m=8.496628478899588sr satisfies alpha_m=L_m beta_m at valid heights. Subscripts a,m denote aerosol/molecular; t indexes66times, i range bins.

For any y(i), define reference-centered signed trapezoidal integral I_y(ref)=0, I_y(i)=I_y(i-1)+dr/2*(y(i)+y(i-1))going outward, and I_y(i)=I_y(i+1)-dr/2*(y(i+1)+y(i))going inward. Here dr=7.5m. Start index53is round(400/7.5); stop index4095. All integrals and inversion below are evaluated over this complete source interval, including the above-reference neighbors needed by later smoothing.

Set k_a=1/abs(L),k_m=1/L_m, normalized return N(i,t)=S(i,t)/S(ref,t), and

phi(i)=2*(k_m-k_a)/k_a*I_alpha_m(i),

p(i,t)=N(i,t)*exp[-phi(i)],

beta_total(i,t)=p(i,t)/[1/(R_ref*beta_m(ref))-(2/k_a)*I_p(i,t)],

beta_a=beta_total-beta_m, alpha_a=L*beta_a.

These expressions follow by separating known molecular optical depth and using an integrating factor in the elastic equation. The denominator has units m*sr; p is dimensionless, so beta_total has units m^-1sr^-1. The exponent is dimensionless. Backward integration is negative below the reference, producing the physically correct attenuation sign. This is a closed-form integral inverse with no iterative optimizer, gradient, step size, line search or convergence stopping criterion. The only stabilization is the fixed ratio, reference boundary, excluded overlap region and source smoothing.

As in source, bins0through53are filled from bin54after inversion. Beta and alpha are then separately smoothed over11neighbor bins. The source aliases input and output, so earlier cells in the left-to-right moving mean already contain smoothed values; later cells contain unsmoothed values. This is a sequential asymmetric numerical operation, not a symmetric convolution. At array ends count only available neighbors. The implementation preserves accumulation order and aliasing. Applying the same linear operation separately to alpha and beta preserves alpha=L beta within floating-point error. No negative aerosol estimates are clipped. Only the scientifically selected below-reference bins54through812inclusive are returned:408.75to6093.75mAGL. All66time groups and all759selectedheights are retained.

## Experiments and evaluation

The active notebook reconstructs the full selected field with L50, computes additional full inversions at L30and70to expose prior sensitivity, and compares actual smoothed/unsmoothed fields, original/fresh results and forward returns. Height-limited extinction integrals are explicitly distinguished from total-column AOD because the unobserved lower atmosphere is excluded. No uncomputed Raman, AERONET or true aerosol comparison is inserted.

The baseline is the independently executed original C++method, not atmospheric truth. Every sample of both fields is checked by uncentered cosine NCC and RMS/reference dynamic range. A nonzero1e-7NRMSEtolerance was frozen before extraction, and NCCmust exceed.99999999; independently rerun original and I/O-instrumented results are identical, and local dynamic-range-scaled max error is about1.64e-16. These numerical tolerances leave ample arithmetic margin without accommodating a different retrieval.

The standalone Judge independently integrates optical depth with a cumulative forward sum and subtracts its reference value. On the returned height interval it predicts relative return (beta_m+beta_a)/(beta_m(ref)+beta_a(ref))*exp[-2 integral_ref^r(alpha_m+L beta_a)dr]. It does this once from submitted beta and once from submitted alpha/L. Both relative-return RMS errors must be <=1.1times the original smoothed retrieval residual(.10870309474048033). Smoothing/noise and the approximate aerosol-free reference make this residual nonzero; it is not a fabricated photon-sigma test. Require positive total atmospheric backscatter. Negative weak-aerosol estimates remain permitted and visible.

All six metrics must pass. Ground truth has priority if later supplied; otherwise use baseline. No gain fitting, crop, rescaling, translation, masked height selection or alignment occurs during scoring. Independent known constant/linear trapezoids, the analytical molecular-only Beer-Lambert limit, gain invariance and an exact in-place smoothing case validate physics beyond source agreement. Main limitations are fixed L, molecular standard-atmosphere prior, aerosol-free reference, uncertain instrument overlap/offset/background assumptions, incomplete final time group and noise-driven negative aerosol estimates.
