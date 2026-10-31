# Implemented mathematical approach

## Fixed scope, units and observation normalization

All15 original ACBP guanidinium chloride samples,3181 measured temperatures each and both330/350nm channels are used:95430 observations. Temperatures and concentrations remain in their original order. No resampling, outlier removal, sample restriction, fitting mask or hidden labels are used. This is the complete original2020 author workflow, not an abbreviated2026 CheMelt hierarchy.

For sample n and channel c, find j(n)=argmin_j|t(n,j)−70| with NumPy first-index tie convention. The normalized public observation is y(c,n,j)=I(c,n,j)/I(c,n,j(n)). This uses only recorded observations. It fixes the optical scale without calibration from fitted thermodynamics. No independently calibrated noise variance is available; residuals are unweighted after normalization and are not claimed independent homoscedastic Gaussian measurements. All original processed observations remain public, including their70°C points.

## Forward model and physical parameters

Let T=t+273.15K, M=t_m+273.15K, s=t−25 (a K interval), D in mol/L, R=.008314kJ/(mol K). The20-element original parameter vector has two7-element optical vectors followed by t_m,H,C,m,m1,m2. Here m1=m2=0 fixed, leaving18 free components. Locally the vector p has exactly18 components: both optical vectors then t_m,H,C,m. No optimization variable is silently omitted.

G=H(1−T/M)+C(T−M−T log(T/M))−mD, in kJ/mol. H in kJ/mol, C in kJ/(mol K), m in kJ/(mol mol/L). The orientation H,C,m>0 identifies folded versus unfolded thermodynamics on this acquisition; all successful native fits have this orientation. Native fitting has no finite bounds, and the local optimizer likewise has none. No artificial prior penalty is introduced.

Set odds=exp(−max(G,−250)/(RT)) and u=odds/(1+odds). The native source clips only energies below−250kJ/mol to prevent numerical overflow. The selected complete fitted domain has G≥−109.014, so the guard never changes the fitted model or reported energy. The energy output itself is the unclipped thermodynamic energy. Independent tests establish u=.5 at G=0, increasing denaturant increases u for positive m, the zero-denaturant thermal midpoint has G=0, dG/dD=−m, and the Gibbs–Helmholtz derivative yields H at T=M.

For each optical channel, F=aF+bFD D+bFT s and U=aU+bUD D+bUT s+bUT2 s². Predicted normalized fluorescence is J=(1−u)F+uU. Optical coefficient units are1,L/mol,K^-1,1,L/mol,K^-1,K^-2. The same u enters both channels, all samples and all temperatures. Flattening is channel first, sample second, temperature frame fastest, matching original merge_samples exactly.

## Entire original observed-data initialization

Native thermodynamic starting values are t_m70°C,H300kJ/mol. For the original empirical residue-count initialization N=90 (an author numerical heuristic, not a newly verified sequence annotation), set C=(−130+16N)/1000 ×(.008314/.001985) and m=(−580+36N)/1000 ×(.008314/.001985), converting the native kcal formulas to kJ with its exact source gas-constant ratio. m1,m2 remain fixed0.

For each channel, choose the lowest-D sample and its first temperature T0. Choose the distinct sample nearest Dmin+1M using the source .01M inward search, and match its temperature nearest T0. Their intensity difference divided by concentration difference gives bFD. Estimate bFT from the same low-D curve using its point nearest T0+5°C (advance one index if tied to the first). Set aF=ymin(T0)+bFT(25−T0)−bFD Dmin.

Choose the highest-D sample and its last temperature Tu. Find the distinct sample nearest Dmax−1M with the same source inward search, match its nearest temperature to Tu, and estimate bUD by the intensity difference divided by concentration difference. Then fit a quadratic to the full highest-D curve y−Dmax bUD versus t−25 using np.polyfit; its constant, linear and quadratic coefficients give aU,bUT,bUT2. The preliminary uncurved source aU/bUT estimates are overwritten by this mandatory whole-curve quadratic stage; directly computing their final identical values is an algebraic elimination of unused intermediate scalars, not dropping a scientific stage. All final initial parameters and complete forward predictions are compared with frozen source values. No source-fitted starts are loaded by local inference.

## Objective, optimizer and stopping

Minimize S(p)=sum_i(J_i(p)−y_i)² across all95430 normalized observations, with no additional normalization, regularization, channel weighting or sample weighting. The selected published family is global nonlinear Levenberg–Marquardt as in the complete original notebook. Local scipy.optimize.leastsq exposes the same general MINPACK least-squares primitive used by LMfit; the acquisition, equilibrium function, residuals, initialization and parameter handling remain explicit local code.

Source default fitting precision was independently improved through the original unmodified public model API before extraction: native start and two full observed-parameter scaled starts1.08/.93 reach objective66.95554159151. Their max componentwise scaled parameter spread1.65e-7 and column-normalized Jacobian minimum singular value.0396 establish a well-defined identifiable physical solution. Source default archived separately, objective66.95554161120. This is original-method precision refinement, not a claim of default-backend bitwise reproduction. Both original precision runs and local use ftol=xtol=gtol=1e-12; local maxfev5000 safeguards failed convergence. MINPACK statuses1–4 are accepted; otherwise fail explicitly. No clipping, reparameterized bounds, random restarts or hidden result is used in standard runtime.

## Outputs, diagnostics and evaluation

The free-energy array retains sample×temperature order at all15×3181 native points. All4 thermodynamic parameters and all14 baseline coefficients are separately required. The evaluator independently recomputes normalized observations and equilibrium fluorescence, scores maximum grid-energy error, componentwise scaled thermodynamic error with denominator max(1,|reference component|), maximum baseline coefficient error, full observed-intensity relative L2, and internal agreement of reported energy and reported thermodynamic parameters. The actual-source pre-extraction tolerances are frozen in metrics.json; they measure selected-method reproducibility, not physical truth.

A source18-column centered-difference Jacobian uses h=max(1,|p_i|)1e-5 and normalizes each column before singular-value/rank analysis, avoiding mere parameter-unit conditioning claims. Original/cleaned Jacobians are checked at the same frozen parameters. Optional notebook diagnostics recompute full global fits from independent public-derived starts, perturb all observations, compare a nested zero-heat-capacity model on the full observations. Those comparisons are labeled and do not replace the standard all-observation model. Curves, denaturant profiles, 2D free-energy/fluorescence surfaces and midpoint D50(t)=G(t,0)/m preserve the scientifically relevant original notebook story. The raw measured intensity is not a ground-truth population; baselines enter any observation-derived fraction estimate. No thermodynamic95% uncertainty coverage is inferred from repeated-point residuals.
