# Implemented approach

The selected method is the published spherical-cell imaging-FCS estimator from the author pipeline, applied to its existing real20degreeCelsius S.aureus movie. All original observations are retained in the public NPZ. There is no synthetic generator. D,N and the long-lag offset are jointly inferred at52fixed observed-cell apertures. No independent diffusivity truth is available; the hidden method result is explicitly a baseline reference.

## Conditioning and coordinates

The native uint16 movie has44161frames,12rows and80columns. Discard the first1500focus-transient frames, then subtract the single minimum over all42661retained frames and all pixels. Preserve the native sampling interval0.0013096382608637214seconds. No drift registration, zero-frame deletion, clipping, deconvolution or spatial interpolation is performed.

At binning n, each intensity trace is the arithmetic mean over the nonoverlapping nby n camera pixels. Standard n=2uses a=.16n=.32micrometres and produces6by40outputs. An aperture is included if its source labels contain exactly one distinct positive cell ID, even if background is also present. Mixed-cell apertures and all-background apertures are excluded. This geometric public support contains52apertures. It is fixed before fitting; no residual/intensity-based deletion occurs. This differs intentionally from the paper's later population-level80percent intensity and0.015nonlinear-MSE filters: this benchmark reconstructs and evaluates every geometric cell aperture, while the notebook reports quality diagnostics without discarding data.

Image axes are row down,column right. Origin is the upper-left image corner; raw centres are spaced.16micrometres and binned centres.32micrometres. Spatial support is12.8by1.92micrometres. These are image-plane coordinates; no unmeasured3D cell surface is reconstructed.

## Variance-preserving bleaching correction

Let I(j) be an aperture mean in ADU at retained frame index j, s=max I, and z=I/s. Divide the first 42500 normalized samples into 85 blocks of 500 frames and average each; the last 161 samples are omitted from the envelope fit only, not from correlation. The fit abscissae are block centres 250,750,...42250 in frames. Fit

```math
b(j)=A[(1-h)\exp(-j/t_1)+h\exp(-j/t_2)]+c_b.
```

A is normalized amplitude, h is a mixing fraction, t1and t2are decay times in frames and cbis a normalized constant. Initial values are(max block mean, floor(max centre/10),.5,floor(max centre/2),min block mean). Lower bounds are(0,1,0,1,0); upper bounds are(2max block mean,10max centre,1,10max centre,1). Use SciPy1.17.1curve_fit bounded default trust-region reflective optimization: no custom tolerances or weights. Preserve its finite-difference Jacobian, default stopping and numerical backend. This is the original selected procedure, not a newly regularized fit.

Evaluate the fitted b at integer indices0through42660, rather than the fit's half-frame centre convention. For every sample use

```math
I_c(j)=s\left[\frac{z(j)}{\sqrt{b(j)/b(0)}}+b(0)\left(1-\sqrt{b(j)/b(0)}\right)\right].
```

Ic is the corrected intensity in ADU. Scaling deviations by the square root preserves fluctuation variance under the source approximation, unlike a division by the fitted mean. A source envelope failure warns and returns the unchanged trace; standard reconstruction explicitly rejects that failure rather than silently accepting an uncorrected substitute. All52standard fits succeed. There is no inverse-iteration warm start between different apertures.

## Multi-tau observation statistic

Use m=16initial lag intervals and pair-average register compression. Subtract the original corrected-trace mean once; retain that original mean for all normalizations. At level0compute lags0through16, then repeatedly drop a final unpaired sample, average adjacent centered values, and compute register lags9through16with physical time multiplied by2to the current level. Divide each accumulated pair-product sum by the original squared mean and its actual overlap count. Preserve the source final-register rule: when no overlap remains, truncate before the immediately preceding allocated index. Omit zero lag in the returned curve. The selected42661samples yield104positive lag values. This is explicit local code; multipletau is used only in the isolated original oracle.

## Forward model and inverse fit

For lag tau in seconds, define

```math
\mu=\frac{a}{2\sqrt{\sigma^2+fD\tau}},\quad
q(\mu)=\operatorname{erf}(\mu)+\frac{\exp(-\mu^2)-1}{\mu\sqrt{\pi}},\quad
G(\tau;N,D,c)=\frac{1}{N}q(\mu)^2+c.
```

a=.32micrometres, sigma=.19micrometres and f=2.1are fixed author optical/geometric calibrations. N is dimensionless effective molecular occupancy, D is corrected diffusivity in square micrometres/second, c is a signed dimensionless offset, erf is the error function, and q is the integrated1DGaussian/aperture overlap. Independent quadrature tests integrate the triangular aperture autocorrelation against the Gaussian diffusion kernel rather than comparing two copied formulas.

Minimize the unweighted sum over all104lags of squared measured-minus-model covariance residuals, independently for each fixed cell aperture. The source uses unconstrained SciPycurve_fit Levenberg–Marquardt, defaults ftol=xtol=1.49012e-8, gtol=0, finite-difference Jacobian, factor100and default maxfev800for3parameters. There is no regularizer, line search or custom proximal step. Initial N=max(0,1/(3first measured covariance)); initial D=.23squared/4/median lag; initial c=0. The.23micrometre initialization constant is source-specific and distinct from the calibrated PSFsigma. Preserve floating operation order: reciprocal N times squared overlap, and sequential division in the D initial guess. Algebraically equivalent reordered arithmetic perturbed numerical Jacobians enough to fail the original stringent parity tests and was corrected without relaxing them.

The standard52fit results all have positive N,D and finite c. A source fit exception produces explicit(-1,-1,-1)diagnostic sentinels and a warning; standard reconstruction rejects these. Optional native single-pixel experiments retain invalid flags and mask those values in physical plots, never relabel them as real negative diffusion. No clipping or positivity projection changes the source estimator.

The paper describes correcting apparent spherical-cell diffusivity by division by2.1. The selected original spherical model already includes fD, an equivalent parameterization; never divide the fitted D again. f and D are not jointly identifiable from these curves. The result is a calibrated, aperture-scale effective mobility, not absolute viscosity or a direct3D diffusivity measurement.

## Evaluation and experiments

Write batch-first float64 D,N,c as reconstruction,molecule_number,correlation_offset. Values outside the fixed public support are exactly zero. Standard CLI uses the same full raw reconstruction as the active notebook. Evaluate every quantity over the same52apertures using uncentered NCC and dynamic-range NRMSE. The baseline contains identified derived observed ACFs and lags for independent physical forward-consistency scoring; these contain no observation absent from the public movie. The separate residual is global RMS of(model-minus-measured covariance)divided by each aperture's first positive-lag covariance.

Two independently executed original processes and three cleaned reruns are bitwise. Nonzero per-field agreement tolerances are fixed from50times measured float32round-trip NRMSE floors, with minimum1e-7; NCC must exceed1minus1e-10. Forward residual boundary is1.1times the actual original0.09738480767333502. This is numerical method agreement, not proof of physical accuracy. Tests reject constant poor fields, swapped/scaled maps and malformed files. No reference-dependent alignment or evaluation mask is fitted.

The notebook actively runs the full standard benchmark and compares its fresh result with separately labeled original measurements. Additional actual reconstructions study omitted bleaching, curvature f=1, PSFsigma variations and native single-pixel resolution; invalid source fits remain explicit. Cell distributions and source nonlinear-MSE diagnostics describe within-movie variability only, not biological replicates or a new temperature-effect experiment.
