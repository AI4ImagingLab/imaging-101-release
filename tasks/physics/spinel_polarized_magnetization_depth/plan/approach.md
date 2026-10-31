# Explicit polarized layer inversion

All168 measured intensities are used, with original float64 values and diagonal standard deviations. Unknown vector is (I,omega,delta,sigmaB,sigmaT,d,m,rho,h) with units/ranges stated in README. The common intensity and angular offset are constrained by original model bounds. Additional sample broadening omega was a trial-and-error value in the tutorial; it is jointly fitted in both models to avoid treating target-related tuning as independently known calibration. Substrate SLD5.377 follows independently known crystalline composition/density. No film property is fixed to a hidden fit.

## Layer model and units

Incident-to-substrate widths are [0,h-d,d,0], nuclear SLDs [0,rho,rho,5.377], magnetic SLDs [0,m,0,0], roughness [sigmaT,sigmaB,sigmaB]. The uniform comparison drops the suppressed layer, giving [0,h,0]. First/last widths denote semi-infinite media and do not enter finite propagation. Stored SLDs are multiplied by1e-6 to obtain inverse Angstrom squared. Wave numbers use sqrt((Q/2)²-4pi(rho+s m)+4pi i1e-16), the positive-imaginary absorbing branch, for spin s=+1/-1. The1e-16 absorption guard preserves the original magnetic kernel's1e-10 guard in its1e-6 SLD units; it is numerical convention, not physical absorption inferred from data.

Interface amplitude f is Fresnel contrast times exp(-2 k[j]k[j+1] sigma[j]²). Starting from zero reflection in the semi-infinite substrate, accumulate

```math
r_j=\frac{f_j+r_{j+1}\exp(2ik_{j+1}t_{j+1})}
{1+f_jr_{j+1}\exp(2ik_{j+1}t_{j+1})}.
```

t is layer width in Angstrom; r is complex amplitude. Intensity is |r[0]|². This scalar specialization is exact for the original collinear magnetic angles; no spin-flip information is fabricated. Independent Fresnel, zero-magnetism, reversal and integrated-profile tests validate the physics beyond original parity.

## Resolution and smooth profiles

Q=4pi sin(theta+delta)/lambda with degrees converted to radians. Both centre and theory grid shift; dQ uses original angle and adds omega linearly to instrumental angular FWHM before quadrature. FWHM-to-sigma is1/sqrt(8log2). The explicit integration computes each piecewise-linear segment's Gaussian zeroth and first moments using erf and exponentials. Include complete segments crossing ±sqrt(2×6.90775527898213703123) sigma, truncate to available grid and normalize by integrated Gaussian mass. Multiply convolved intensity by I; background stays0 because measured background was already subtracted. No added interpolation or oversampling changes the original84-point operator.

Define B(z;a,s)=0.5[1+erf((z-a)/(sqrt2 s))]. On the fixed depth grid, smooth nuclear SLD=5.377+(rho-5.377)B(z;0,sigmaB)-rho B(z;h,sigmaT), and magnetic SLD=m[B(z;d,sigmaB)-B(z;h,sigmaT)]. z is Angstrom from mean substrate/film boundary. Profile magnetic area is m(h-d); roughness redistributes it without changing its integral. Profiles are not shifted or renormalized by evaluation.

## Inverse updates and stopping

Objective F=0.5 sum over both channels and84 points of [(predicted-observed)/uncertainty]². Parameters outside bounds return infinity. Initial values are [1,.026,0,5,5,10,.1,5,160]; uniform removes index5. The bounded Nelder-Mead simplex starts with one vertex per coordinate displaced by0.15 times its bound range, reversing the displacement at a colliding upper boundary. Reflection1, expansion2, contraction0.5, shrink0.5 are implemented explicitly. Vertices are sorted by objective after each update. Stop only when maximum vertex spread divided by bound range is<=1e-7 and maximum absolute objective spread<=1e-9; maximum15000 iterations. No regularizer, random restart, MCMC or fitted background is used. Upstream's printed status is reversed despite correct numerical convergence; local reporting checks both criteria and raises on failure. Original parameter results and final simplex evidence are archived separately.

## Evaluation and notebook comparisons

Output profiles(1,521,2) and all9 parameters(1,9) are required. Range-normalized parameter difference<=0.0005, each profile dynamic-range NRMSE<=0.002, profile/parameter consistency<=0.0001 of channel range, reduced chi-square<=1.05 times reproduced native dead-layer result. These nonzero tolerances are stricter than the independently frozen0.001 parameter criterion and validated against native reruns/float32 submissions. Baseline is a method result, not truth. Nonlinear model identifiability is conditional on bounds, roughness assumptions, collinearity and omitted covariance. The notebook actively fits both complete models, compares spin asymmetry, SLD profiles, residuals, original outputs and actual nuisance sensitivity. Local curvature diagnostics are conditional linear approximations, not a paper MCMC posterior.
