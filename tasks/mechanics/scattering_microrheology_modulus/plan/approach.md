# Implemented mechanics and inverse mathematics

The bound observations are three complete experimental repeat curves of author sample patient254, from the author DLSuR replicate example. The solver uses the author's ergodic configuration, particle radius a=250nm, T=310.15K, n=1.333, theta=173 degrees, lambda=633nm and kB=1.38e-23J/K. Disease classification is unavailable. There is no ground truth. The benchmark reproduces a conditional rheological estimator, not a clinical diagnosis.

## Optical forward model and units

For lag t in microseconds, q=4 pi n sin(theta/2)/lambda is in inverse nm. The isotropic Gaussian displacement characteristic function is g1=exp(-q²M/6), where M is the three-dimensional MSD in nm². The Siegert relation gives c=beta*g1². Thus M=-6log(g1)/q²; a positive sign is incorrect. No image discretization, FFT, boundary convolution or spatial mesh is used. Observations are discrete correlations on native nonuniform lags. Noise covariance is unavailable; all fitting residuals are unweighted.

The public correlations are precisely the original reader output: measured c with instrument distribution-fit g1 squared plus the instrument baseline substituted at nearest exported lags. This author processing is identified, not described as raw photon data. No hidden target is used.

## Reliable domain

For each repeat drop first three samples. Let j be the first native index with c<0.05. If argmax returns zero, set j=N-1 as upstream; bound data never starts below threshold. Select indices [3,j). This yields lengths188,145,155. Retain full public curves. Output concatenates these selected domains without interpolation or padding.

## Intercept and early-lag fit

Fit h(t)=b exp(-u*t^v) with b the intercept, u a rate in us^-v and v dimensionless. Initial parameters are [c[1],0.01,1] after domain selection. Nine candidate windows begin at2us and end at40,50,...120us. Endpoints are mapped to nearest lag samples and included in fitting. Each window receives leave-one-out MSE: remove each point, fit remaining data using scipy.optimize.curve_fit (unconstrained Levenberg–Marquardt, maxfev10000, native tolerances), and average squared held-out residuals. Failed fits contribute1000; a failed full-window fit assigns the window score1e6. Refit each window with maxfev100000 and select first minimum CV score. This is the original method, not a replacement optimizer. Standard SciPy LM stopping conditions apply; there is no custom line search or regularizer.

Set beta=b. Replace c between the two nearest selected-window indices, excluding the upper endpoint, with h(t), then g1=sqrt(c/beta). Upstream does the replacement unconditionally despite its comment suggesting a conditional replacement. We preserve that executable behavior. The unused upstream calc_msd_raw helper has the wrong sign; it is not called by the original selected pipeline or this implementation. The validated msd_local_pwr_law path uses the correct negative sign.

## Gaussian local log regression

Compute raw M=-6 log(g1)/q². For negative values only, interpolate in linear t from positive MSD values, as upstream. Zero values are undefined in a log fit and rejected. For x=log(t), y=log(M), design X=[1,x], bandwidth b=0.1 and tau=b*abs(x_first-x_last), use weights w_ij=exp(-(x_j-x_i)²/tau). Note tau is not squared; preserve the upstream convention. At each i, theta_i=(X'W_iX)^-1 X'W_i y. The fitted MSD is exp(theta_i,0+theta_i,1*x_i), and alpha_i=theta_i,1. Matrix operations use float64 and one CPU thread. No positivity clipping of moduli is introduced.

## Mechanical response

omega=1e6/t in s^-1. The complex-modulus magnitude is A=kB*T*1e27/[pi*a*M*Gamma(1+alpha)] in Pa. Return G'=A*cos(pi*alpha/2), G''=A*sin(pi*alpha/2). This is local power-law analytic continuation of the generalized Stokes–Einstein relation. It is not an exact inversion for arbitrary spectra. The Brownian-fluid test follows D=kBT/(6*pi*a*eta), M=6Dt=kBT*t/(pi*a*eta) in SI: G'=0, G''=eta*omega. A constant M gives a purely elastic plateau. Finite lag extent, particle-medium interactions and an ergodicity assumption limit interpretation.

## Comparisons and evaluation

Notebook comparisons actively recompute bandwidths0.05,0.1,0.2, half-lag sampling, and a direct Laplace alternative. For the latter L(s)=trapezoid(M(t)*exp(-st),t), s=1/t; fit local log slope of H(s)=1/[pi*a*s*L(s)] with bandwidth0.01, multiply by kBT*1e27 and continue using cosine/sine. Its finite-domain bias is explicitly displayed; it is not the submitted method. The removed scipy.integrate.trapz is replaced by mathematically equivalent numpy.trapezoid only in this optional comparison.

Each output quantity and both modulus components receive independent standard NCC/NRMSE. Intercept relative error is max(abs(beta/beta_ref-1)). Forward consistency is max over repeats of RMS(beta*exp(-q²M/3)-c). Two original reruns gave identical arrays. Before local extraction, stage tolerance was fixed at rtol1e-8/atol1e-9; Judge NCC>=0.999999, NRMSE<=1e-5, intercept error<=1e-5, and forward RMS<=0.01723522069078028 (1.1 times original worst repeat0.01566838244616389). These tight method-reference bounds measure reproducibility, not physical truth. No threshold was changed after seeing local results.
