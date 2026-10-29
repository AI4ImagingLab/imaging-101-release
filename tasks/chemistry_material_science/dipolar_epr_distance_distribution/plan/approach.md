# Implemented mathematics and numerical conventions

Use t=raw_time-raw_time[0]+0.1 microsecond and r=arange(1.5,6,0.01) nm. Phase correction minimizes the summed squared imaginary part: phi=atan2(sum(Re V Im V),sum((Im V)²-(Re V)²)/2)/2+π/2. Rotate by exp(i phi), flip sign if the real mean is negative, then divide by the maximum real signal. No imaginary-to-magnitude substitution occurs.

Let phi_ij=|t_i| D 10^21/r_j³, q_ij=sqrt(6 phi_ij/π), and Fresnel integrals S(q),C(q) use the standard πv²/2 argument. K0=[C(q)cos(phi)+S(q)sin(phi)]/q, with exact limit K0(0,r)=1. Multiply each distance column by trapezoidal r weights. D=mu0*(g muB)²/(2h) using constants copied verbatim from source.

Preserve upstream cubic time interpolation: dt=mean(diff(t)); base time grid is arange(min(t)-0.3-0.048-dt,max(t)+0.3+0.048+dt,dt). Interpolate the weighted kernel to t-t0. The homogeneous background uses |t-t0|, concentration micromol/L times 1e-3 NA and microseconds times 1e-6. The total kernel is B*((1-lambda)*distance_weights+lambda*K0_weighted).

Gaussian columns q_jk=exp(-(r_j-mu_k)²/(2sigma_k²))/(sqrt(2π)sigma_k) share the denominator sum_k trapezoid(q_k,r); this is the upstream common-basis amplitude convention, not separately normalized columns. For nonlinear theta=[lambda,t0,C,mu1,sigma1,mu2,sigma2], solve min_(a>=0)||A(theta)a-y||² with scipy.optimize.nnls, then optimize the unnormalized data residual using scipy.optimize.least_squares, trust-region reflective default, max_nfev=1000, ftol=xtol=gtol=1e-10. Start=[0.01,0.3,50,2.5,0.2,4.5,0.2]. Bounds=[0,0.252,0.01,1,0.05,1,0.05] to [1,0.348,5000,20,2.5,20,2.5]. No regularizer is used for this parametric model.

Normalize the final weighted Gaussian sum to unit trapezoidal integral, but keep the unnormalized amplitudes for the fitted echo and reported amplitude parameters. Component labels are exchangeable, so use increasing fitted mean consistently for the parameter contract and oracle comparison; this is a declared label convention, not a phase/scale alignment. Notebook experiments recompute shortened evolution records and time-calibration perturbations. They do not establish extra molecular species or physical truth.

## Evaluation

No independent object truth exists for these bound observations. The hidden baseline is the freshly executed original selected method. Judge uses strict per-array/per-component relative Euclidean error with boundary 1e-5 fixed before extraction; complex vectors use their Hermitian norm. Original and fresh measurements are identified separately. Do not interpret reference agreement as physical accuracy. Stage, analytic physics and Judge contract tests complement full-pipeline parity.
