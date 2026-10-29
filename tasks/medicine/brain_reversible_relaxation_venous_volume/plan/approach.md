# Implemented mathematics

The bound input is four complete measured filtered magnitude acquisitions from subject 01. The private conversion averages each group of four adjacent float32 partitions by `((a+b)+c)+d`, then divides by float32 four. This follows the numeric averaging in upstream x.zaverage; source README suggests summation, which multiplies every magnitude by four and leaves both target parameters unchanged. It changes only log amplitude by log four. Native spacing is four times 1.25 = 5 mm; an upstream x.zaverage hardcoded 7.5 mm header assignment is inappropriate to these data and is not propagated. No upstream FSL motion correction, smoothing or anatomical segmentation pipeline was executed.

## Acquisition and model

Let S be magnitude in arbitrary units, tau spin-echo displacement in seconds, r reversible transverse relaxation in s^-1, v apparent deoxygenated venous blood-volume fraction, and a log spin-echo amplitude. At tau zero log S = a. For positive tau > 0.015 s the static-dephasing approximation gives log S = a+v-tau r. Fixed TE means irreversible relaxation is absorbed into a. All 24 measurements are retained, but only zero and 13 times 16 through 64 ms enter the selected original estimator. Negative times and the short-time transition remain measured context, not fabricated predictions.

Each design row is [1,-tau,1] for a long time, preceded by [0,0,1] for the spin echo. With p=[v,r,a]^T and y the corresponding 14 log magnitudes, solve

$$\min_p\sum_{j=1}^{14}(X_jp-y_j)^2.$$

No weighting or objective normalization is applied; normalization would not affect this minimizer. X has rank three, although its first and third columns coincide away from the spin echo. NumPy `linalg.lstsq(X,Y,rcond=None)` uses a general linear least-squares primitive with all voxel curves as right-hand sides. There is no initialization, iteration, learning, regularization, step size or stopping criterion beyond the direct decomposition. The default rank threshold is the numerical primitive's machine-precision convention. It is tested against Octave lscov and a separately derived regression slope/intercept identity.

Convert input magnitudes to float64 before natural logarithms. As in the original, nonfinite log values arising from zero magnitudes become zero. Negative/nonfinite source values are rejected. They do not occur in the bound data. Fit every curve for diagnostics; the task output zeros values outside its fixed public observation mask. Inside, retain signed fits and never select on reconstruction sign. The mask uses the central ellipse, slabs 2 through 7 and condition-specific half-95th-percentile spin-echo threshold exactly stated in README. No target-derived calibration or anatomical atlas is used.

## Diagnostics and limits

Log residual RMS uses all 14 selected observations including spin echo. Oxygen ratios are exploratory notebook diagnostics only. Define gamma=26750 rad/(s Gauss), B=30000 Gauss, susceptibility difference d=.264e-6 dimensionless, haematocrit h=.34 and concentration conversion k=.03. Then OEF=r/[v gamma (4/3) pi d B h] and dHb=r/[v gamma (4/3) pi d B k], in fraction and g/dL respectively. h and k are assumptions, not measured subject calibrations. Zero blood volume produces nonfinite values; near-zero blood volume produces extreme ratios. These values are retained in diagnostic counts rather than clipped into a physiological range. OEF and dHb are not submitted outputs or independent truth.

The scope was fixed after original-only identifiability screening and before extraction: estimate R2-prime and apparent DBV, not stable oxygenation. The paper itself discusses diffusion narrowing, intravascular signal, CSF, macroscopic gradients, myelin and elevated DBV. No anatomical tissue-group claims or paper cohort reproduction are made for this single unregistered subject. Comparison of nulling conditions is descriptive, not a registered voxelwise treatment effect.

## Evaluation and experiments

Evaluate each condition separately with physical-unit RMSE against independent original output on the fixed mask. Boundaries were frozen before extraction: R2-prime 1e-6 s^-1, DBV 1e-8 fraction, outside-region maximum 1e-12. The stricter stage/full-array parity tolerance is absolute and relative 1e-10. The independent original repeat has zero difference. Nonzero tolerances accommodate double-precision decomposition differences; they do not define physiological accuracy. NCC is unnecessary for this signed parameter reproduction. The standard reference is not ground truth.

The active notebook reconstructs all four acquisitions, compares original and fresh fields and metrics, examines signal/model residuals, reports signed fits, and computes OEF/dHb failure counts, quantiles and extrema. Two real-data sensitivity experiments rerun the same local model with cutoff 23 ms and spin-echo magnitude changed by ±1% (the latter explicitly a synthetic perturbation, not new observations). The cutoff comparison probes the selected long-time regime; the perturbation demonstrates intercept/ratio sensitivity. No alternative run replaces the original benchmark method.
