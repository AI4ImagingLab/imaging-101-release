# Implemented approach

## Bound acquisition and ensemble calibration

The task binds 16,625 real KiDS-450 galaxy shapes, with RA/Dec in J2000 degrees. Set RA0=129.0 and Dec0=-0.5 degrees. Gridding coordinates are x=(RA-RA0)cos(Dec0), y=Dec-Dec0, with fixed projected extent x in [-0.5cos(Dec0),0.5cos(Dec0)] and y in [-0.5,0.5]. Transpose the output of histogram2d from (xbin,ybin) to (north,east). There are 64 cells per axis. The FFT uses the original equal-pixel flat-sky approximation; the 0.0038% projected aspect difference is disclosed in README. Published decimal observations are preserved as float64, with no artificial noise.

In each cell accumulate S=sum w(1+m), N1=sum we1 and N2=sum we2. The dimensionless shear components are gammaj=Nj/S where S>0; otherwise both are zero. Here e is lensfit ellipticity, w is its inverse-variance weight, and m is the independent multiplicative calibration. Dividing each galaxy by its own 1+m gives a different, incorrect ensemble weighting. PSF shapes are independent diagnostics; lensfit already corrects the measured galaxy shapes for PSF effects. No factor of two or fitted target-derived calibration is introduced.

## Fourier inverse and resolution choice

At each discrete frequency (fx,fy)=fftfreq(64), set P1=(fx²-fy²)/(fx²+fy²) and P2=2fxfy/(fx²+fy²), with both zero at DC. NumPy's forward FFT has negative sign and inverse normalization 1/4096. The selected KS93 inverse is E=Re IFFT(P1 FFT(gamma1)+P2 FFT(gamma2)), B=Re IFFT(-P2 FFT(gamma1)+P1 FFT(gamma2)). Both maps are dimensionless and have zero mean. This fixes the unobservable uniform mass sheet. B is retained as an output and diagnostic, rather than discarded or forced positive.

This is a closed-form linear inverse: no optimizer, initialization, step size, line search, iterative update or stopping criterion is applicable. Default regularization is gaussian_filter sigma=2.0 pixels, mode='wrap', truncate=4 (SciPy default), applied separately to E and B. This is an explicit fixed spatial-resolution choice after KS93, not a newly asserted probabilistic objective.

The notebook computes sigma=1,2,4 and omission of ensemble calibration. It also computes 32 independent orientation randomizations, rotating each complex ellipticity by exp(2i alpha) with alpha uniform in [0,pi), using private diagnostic seed 20261020. The sample standard deviation (ddof=1) of randomized E maps gives an orientation-null noise estimate. E divided by that estimate is displayed as a diagnostic, not a calibrated detection probability. PSF maps are reconstructed using the independently supplied PSF ellipticities.

## Physical checks and disclosed limitations

For an analytic potential mode with wavevector (3,2), the shear factors are 5/13 and 12/13; tests check that relation, inversion, the DC null space and the real inner-product adjoint identity. On an even grid, mixed Nyquist lines in the original real-part projection are not exactly invertible for arbitrary content. We preserve and disclose that discretization, then display the complete forward-closure residual after smoothing. Gaussian sigma=2 strongly suppresses these frequencies. No exact all-frequency inverse is claimed. The weak-shear approximation uses catalogue ellipticities as shear estimates and does not perform a nonlinear reduced-shear correction.

## Evaluation conventions

The real observations have no independently known projected mass truth. The hidden reference is the independently run original Lenspack result on the exact bound observations. Each required E/B array is scored with uncentered cosine NCC and RMS error divided by reference dynamic range, without crop, alignment or fitted scale. Both must satisfy NCC>=0.999999 and NRMSE<=1e-6. These finite numerical-agreement thresholds were fixed after the oracle and before local implementation; zero output has NCC=0 and fails. Degenerate hidden reference metrics are evaluator errors. Standard-method agreement and the null diagnostics do not establish physical truth.

Independent observation consistency is also required: relative_shear_residual. Original-code value=5.1009808258198334e-07; boundary=0.0001. The Judge reimplements the declared response or shear mathematics without solver imports. The AIA mean is evaluated with the adopted10%+floor error and nominal response; the KiDS norm compares the two-pixel output resolution and accounts for DC degeneracy.

## Quantified DC and even-grid Nyquist limits

A subsequent independent unmodified Lenspack ks93/ks93inv investigation is archived in private kids_nyquist_investigation.json. DC is removed before all following loss measurements. On bound unsmoothed shear, the relative round-trip residual is0.11671448696328429, energy fraction0.013622271467102659; thus the raw arbitrary real-shear map is NOT strictly invertible. On the actual sigma2-pixel output resolution, the residual is5.100980825832628e-7, energy fraction2.602000538551212e-13, maximum component residual5.895273347267249e-8. This small post-filter quantity is distinct from the unobservable DC shear components(-0.002018809126667842,0.0032107191158432307).

A random real shear field likewise has0.11794703832818561 relative raw loss but4.7408767102097243e-7 after sigma2 filtering. An analytic E mode cos(pi*x)*cos(2*pi*3*y/64) gives p1=(32^2-3^2)/(32^2+3^2)=0.9825750242013552; its source-convention E round-trip amplitude is p1^2=0.9654536781842938, agreeing with a3.4546321815706% loss. This is the mixed-Nyquist real-part projection, not the DC mass-sheet degeneracy. Source convention is retained; analytic/adjoint checks do not imply exact inversion for all discrete real modes. B and orientation noise remain comparable to E, and no independently significant dark-matter detection is claimed.
