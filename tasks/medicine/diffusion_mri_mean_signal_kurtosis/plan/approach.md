# Implemented mathematics: mean-signal diffusion kurtosis

## Geometry, units and observations

Use all496 native magnitude encodings from CFIN human brain, full96x96 native planes at zero-based z8,9,10. Output axes are batch,slice,native-x,native-y; no transpose or resize occurs in numeric reconstruction. Visualization transposes x/y maps solely to label native x horizontally. Recorded NIfTI spacing/affine are given verbatim in README. Diffusion b values are deposited integers represented float64 in s/mm^2; directional vectors are unchanged deposited float64. No new synthetic data, k-space reconstruction, motion correction, noise simulation or denoising is added.

Before fitting define mask M(x)=S(b=0,x)>40 ADU. There is one b=0 volume. This observation-only rule is fixed and public; no fit/ground-truth-derived screening is used. Set D,K,S0 to zero outside M. Negative K and high D inside M remain preserved. No spatial boundary extension or coupling is used.

## Forward model and powder averaging

For each exact shell b, arithmetic-average its measured directional magnitudes to y_b and count its encodings n_b. The bound shells are0,200,...3000, n_0=1 and n_b=33 for all nonzero b. Their exact deposited values already coincide with upstream magnitude-rounded shells. This task does not expose an arbitrary shell-rounding hyperparameter.

The mean-signal cumulant expansion is yhat_b=S0 exp(-bD+b^2 D^2 K/6). D is mean-signal diffusivity in mm^2/s, K is dimensionless signal kurtosis, S0 is extrapolated baseline in ADU. It is an approximate finite-b powder-signal expansion, not a compartment model, angular diffusion tensor or conventional directional mean kurtosis. The expansion can produce negative fitted K; the benchmark does not silently correct this.

## Selected inverse and numerical conventions

Replace y_b below0.0001 ADU by that positive floor before taking logarithms. Set B rows[-b,(1.0/6.0)*b**2,1], parameter p=(D,D^2 K,log S0), and W=diag(n_b y_b^2). Minimize J(p)=sum_b n_b y_b^2 [log y_b-(Bp)_b]^2. There is no1/2 or per-shell normalization factor; any constant positive global factor leaves this closed-form optimum unchanged. The original inverse explicitly evaluates p=pinv(B.T W B) B.T W log y using NumPy's default float64 SVD pseudoinverse, preserving multiplication order and normal equations rather than substituting a better-conditioned but numerically different weighted least-squares factorization. Unpack D=p0, K=p1/(p0^2), S0=exp(p2). There is no regularizer, positivity projection or parameter clipping.

No initialization, custom iteration, update rule, step-size or stopping condition is applicable: one pseudoinverse per included voxel provides the original closed-form estimate. fit_shells weighted=True is the selected standard. The weighted=False review alternative uses identity W. All outputs are realfloat64 (1,3,96,96). The pseudoinverse conditioning, signal floor and preserved invalid-K cases bound physical interpretation; no model output is relabeled as truth.

An initial parity test caught the rounding difference between b*b/6 and original(1/6)*b**2 (relative2e-16 in B). The source was corrected to retain original operation order; the test was not relaxed. After that change the frozen design matrix is exactly identical.

## Controlled comparisons

The active notebook recomputes unweighted all-shell estimates and weighted shells0,600,1000,2000. The latter is an explicitly labeled reduced-acquisition fit, never substituted for full-shell reconstruction. A local lower-order Gaussian baseline minimizes the corresponding weighted log-signal objective with B0=[-b,1], and predicts S0 exp(-bD). The selected model and Gaussian baseline are compared with actual shell means in three fixed coordinate ROIs and across all included voxels of every slice. Negative-K fractions and mask-threshold support counts20/40/60/80 are shown rather than removing difficult voxels.

Compartment-conversion and tensor-DKI outputs discussed in upstream tutorial/review paper are different inverse problems; this task explicitly selects the mean-signal model and does not claim those quantities. No upstream historical plotted metric is presented as a fresh measurement. Relative-L2 error is evaluated separately for D,K,S0 over full output arrays, without physical rescaling, cropping, hidden registration or magnitude conversion. Reference is independent original-method output because no true in-vivo D/K/S0 maps are available.

The independent standalone Judge also evaluates shell-mean forward RMSE in ADU over public b0>40 support, using only submitted D,K,S0 and public observations. Original residual5.002890195639437ADU sets fixed boundary5.503179215203381ADU (1.1times). All required relative-L2 boundaries remain1e-6. Observation consistency supplements standard-method agreement and does not prove a correct tissue model.

## Paper/code weight notation discrepancy

The2021review paper section2.1.6 Equation18 matches the selected signal expansion, but its following prose MathML gives w=Ng*Sbar without defining whether w multiplies residuals or their squares. Actual original msdki.wls_fit_msdki explicitly uses W=diag(Ng*Sbar**2), preserved here. Under independent homoscedastic directional signal noise of variance sigma**2, Var(shellmean)=sigma**2/Ng and the delta method gives Var(log shellmean) approximately sigma**2/(Ng*Sbar**2). Thus the implemented precision weight is physically supported. We reproduce the code-standard inverse, identify the paper prose ambiguity, and do not claim literal identity to that prose weight expression. No upstream source correction or silent method substitution is made.
