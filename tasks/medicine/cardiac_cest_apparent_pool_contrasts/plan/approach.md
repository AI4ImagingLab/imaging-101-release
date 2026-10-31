# Implemented mathematics

The benchmark begins with measured magnitude images from the author's radial MRI branch, not raw k-space. BART and Bruker parsing are private acquisition conversion only. No MRI image is a target quantity here. The unknown is an apparent six-Lorentzian saturation decomposition and local water frequency shift; neither molecular concentration nor exchange rate is inferred.

## Observation normalization

Let I(r,c,j) be stored magnitude intensity and omega(j) its RF offset. Reference indices satisfy omega>15ppm. For every other acquisition index j between adjacent references l,u, set alpha=(j-l)/(u-l) and M(r,c,j)=(1-alpha)I(r,c,l)+alpha I(r,c,u). The normalized observation is Z=I/M. SciPy interpn evaluates the same multilinear interpolation as the original implementation; only acquisition index varies, spatial query points are exact native grid locations. The first/last frames bracket every nonreference sample. IEEE zero divisions outside support are mapped using NumPy nan_to_num, as upstream; all8936 supported spectra are finite. The input-only mask is first reference>0.15 times its maximum. There is no spatial regularizer, boundary extension, image resizing, or support filtering after fitting.

## Apparent forward model

For amplitude a, positive FWHM w and center c, define L(x;a,w,c)=a(w²/4)/(w²/4+(x-c)²). Its peak is a, and L(c±w/2)=a/2. The shifted coordinate is x=omega-b, with fitted water resonance shift b. Predicted normalized signal is1 minus the sum of six L values. Component order is water,MT,amide,creatine,NOE−3.5,NOE−1.6. Amplitudes are fractions, widths/centers/shifts ppm. No alignment or normalization of submitted maps is performed.

## Three fitted stages

1. Reverse descending saturation offsets and spectra into ascending order. Fit Z=1-L(water)-L(MT) over all53points. Initialization is[.8,1.8,0,.15,40,-1], lower bounds[.02,.3,-10,0,30,-2.5], upper[1,10,10,.5,60,0]. The first water center is b. This stage estimates a frequency origin, not the final water/MT fields.
2. Subtract b from every offset. Retain points x<=−4 or x>=4 or −1.4<=x<=1.4ppm for background fitting. Fit the same two-component background with initialization[.8,.2,0,.15,40,-1], lower[.02,.01,−1e-6,0,30,−2.5], upper[1,10,1e-6,.5,60,0]. The near-zero water center fixes the frequency gauge. These fitted water/MT parameters are returned.
3. On all shifted offsets compute d=1−(Z+Lwater+LMT). Fit d to the sum of the four selected weak components. Initial triples in returned order are(.05,1.5,3.5),(.05,.5,2),(.05,1,−3.5),(.05,1,−1.6). Lower triples are(0,.5,3.2),(0,.5,1.6),(0,.5,−4),(0,.5,−1.8); upper triples(.3,5,4),(.5,5,2.6),(.25,5,−1.5),(.25,5,−1.2). Append these four triples to the two background triples.

Every stage minimizes an unweighted sum of squared residuals, without extra normalization or regularization. General SciPy curve_fit uses its bounded trust-region reflective solver and numerical Jacobian. Arguments are xtol1e-10,ftol1e-4,maxfev500; default gtol1e-8,linear loss,two-point derivatives and trust-region decisions are retained. The upstream source exposes these controls;500 replaces its default50 to supply a declared convergence budget before the full original run. Original and local use identical settings and NumPy/SciPy versions. All8936 original fits succeed; no fit failure is replaced with an invented zero result in the local method. There is no randomness. The original two-stage estimator is not a simultaneous global minimum of the full six-component least-squares objective, a distinction shown by the notebook diagnostic.

## Diagnostics and evaluation

The analytic single-Lorentzian Jacobian has amplitude derivative q/(q+d²), width derivative a*w*d²/[2(q+d²)²], and center derivative2*a*q*d/(q+d²)², with q=w²/4,d=x-c. Independent finite differences check it; curve_fit itself retains upstream finite differences for exact parity.

The primary run always enables field correction. The optional correct_field=False flag deliberately keeps b=0 for a diagnostic alternative while retaining the subsequent stages and declared bounds. The notebook's fixed-parameter forward ablation instead keeps all fitted triples and sets only b=0 at reprojection; it is labeled as a forward ablation, not a separately reconstructed method. The shipped joint_refine_spectrum diagnostic refits all18component parameters with the field shift fixed, initialized at the fresh two-step parameters and using the same combined bounds, xtol1e-10,ftol1e-8,max_nfev2000 (SciPy least_squares, bounded trust-region reflective). It minimizes the full normalized-signal squared residual and is run on the three fixed support-order quartiles only; it is not the benchmark method or a full-image joint reconstruction.

All required maps are compared with independently executed original outputs, never truth. Each component has separate absolute RMSE in its own units: amplitude<=1e-5,width<=1e-3ppm,center<=1e-4ppm; field-shift RMSE<=1e-5ppm. These tolerances were frozen before local extraction after a complete exact original rerun. Original whole-observation normalized RMS residual0.07805802475666919 gives a pre-frozen auxiliary boundary0.08586382723233611 (1.1times original). The metric uses every supported pixel and all53fitted offsets with no phase, scale, spatial or component alignment. Source regional RMSE excluding water is preserved separately as a diagnostic and is not relabeled as this all-offset metric.

Zeros outside support are storage sentinels. Inside support all parameter intervals are validated. Weak-component widths/centers can be poorly identified even when numerical reference agreement is excellent. Curves, residuals, active bounds and forward ablation expose this ambiguity; no quantitative biological accuracy is claimed.

The selected middle pixel reaches the unchanged 2000-evaluation joint diagnostic cap. Its actual last iterate and false convergence status are retained. Two other diagnostic pixels terminate successfully. This optional comparison does not replace any primary two-step fit or alter acceptance boundaries.


## Current benchmark v2 acceptance

The v2 Judge independently evaluates every supported voxel and every53-frequency observation under the six-pool model. Each voxel MSE must be no higher than attained baseline MSE+1e-6; original whole-field RMS boundary and all physical ranges/support constraints remain. No ground truth exists. Independent bounded linear-amplitude fitting produces a different improved solution accepted by the Judge.19 unit tests passed. See cest_pytest.log.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.
