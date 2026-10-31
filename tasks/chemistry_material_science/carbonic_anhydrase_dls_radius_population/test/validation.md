# Provenance and validation

## Scope and independent sources

Physical unknown: two nonnegative intensity-weighted radius-bin mass distributions from measured carbonic-anhydrase light-scattering intensity correlation. The target is explicitly conditional on the source Stokes?Einstein physical conventions and fixed regularized objective. Fine distribution detail is prior dependent; no true molecular-size field is available.

Original paper: Burastero et al., Raynals, an online tool for the analysis of dynamic light scattering, Acta Crystallographica D79,673?683(2023), DOI10.1107/S2059798323004862. Final publication XML captured independently in raynals_paper.xml (PMC10394669); primary publisher https://journals.iucr.org/d/issues/2023/08/00/vo5014/index.html. Source authors link code https://github.com/osvalB/dynamicLightScatteringAnalysis. Exact current remote tree verified:4214be4cb9524a42db47b011cc212ede2b9de590. Complete relevant source/test notebook and source LICENSE are archived privately under sources/osvalB/dynamicLightScatteringAnalysis. Code license EUPL-1.2 retained in task LICENSE with NOTICE attribution.

Dataset: author repository test.csv, headers carbonic anhydrase A1/A2, two real measurements. Source byte SHA256 052a3d4181a3d960c06aa154a48ec2107cb2f6395ea57f81d25ef168c16c18f0. All374 native lag points and two correlations are public, unchanged float64; the source excludes first native lag only during its analysis. The raw NPZ retains it. Storing all source observations does not include cached populations or source session answers. Radius_nm is an independent predetermined discretization coordinate, not recovered weights. No synthetic data are generated.

Private maintainer archive: worker3_material_bio_med under research_batch13_20261030. capture_dls.py imports the original source modules and runs all relevant stages against this exact test.csv: load, q, radius/relaxation discretization, early-lag coherence, Siegert transform, fixed-alpha inversion, all80 L-curve cases, corner selection, selected-alpha inversion and complete predicted correlation. No cleaned code is imported into the oracle. Frozen files dls_capture/upstream_fixed.npz and upstream_lcurve.npz identify fresh original results, not cached paper figures. Existing299 and21blocked semantics were screened; DLS temporal correlation inversion differs from SAXS static angular intensity size inversion.

## Environment and compatibility

Original CPU run uses project Python3.11.11 with private import paths selecting NumPy1.26.4/pandas1.5.3, SciPy1.17.1 and original source. The source triangle function converts size-one arrays into scalar acos; NumPy2 rejects this API, so original code runs unmodified with NumPy1.26.4. This is a validated modern environment execution of the archived implementation, not a claim to reproduce the authors'2023 computer. Private compatibility packages do not enter cleaned requirements or imports. Shared runtime was never modified/upgraded. Cleaned runtime uses NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1. CPU single-thread OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1; no GPU or training checkpoints. Original source warnings about square roots are expected at noisy late lags, and the original valid-prefix convention is preserved explicitly. Public raw observations and output arrays are finite; only a hidden fixture's intermediate field transform retains source NaN values beyond each positive prefix.

## Physical assumptions and discrepancies

Temperature293K, viscosity8.9e-4Pa s and solvent index1.33 come from original dls_experiment defaults; lambda817nm and150degree detection are explicit source demo/instrument settings. The CSV itself has no independently measured solvent temperature/viscosity calibration. Therefore these are declared source analysis/model conventions, not invented acquisition measurements. The README identifies that distinction; actual radius accuracy is not independently established. Do not replace293 by293.15 or alter viscosity to a presumed water value. The source viscosity corresponds to a different common reference condition than exact20degree water; this is a limitation of absolute physical interpretation. Conditional effective radii follow the fully specified source convention. Decay-rate/forward correlations are the actual directly constrained observables.

Paper/code distinctions, independently investigated and preserved openly:

- Paper lower grid radius.1nm versus original notebook.09nm: this task uses the bound author notebook's actual.09..1e6nm200-point grid.
- Paper states exact unit mass and zero endpoint constraints; original code adds sqrt1000 weighted rows, giving soft penalties, not hard constraints. Our measured output sums0.99999598 and0.99999215 demonstrate this difference. No post-fit renormalization is applied.
- Source helper calls nnls on the full regularized augmented matrix and returns its norm as residualNorm; this includes alpha times curvature energy. Independent equality Phi=residualNorm^2 verifies it is not the pure fidelity norm commonly used for a theoretical L curve. We preserve this diagnostic source-version comparison and label its axis augmented residual norm. We do not silently correct it or claim an unchanged corrected theoretical algorithm. The required benchmark uses the separately declared fixed-alpha convex target and does not depend on this diagnostic ambiguity.
- Source alpha.1 is the original getInitialEstimates default, selected before local implementation. It is not claimed to be a paper optimum or tuned after observing cleaned results. The source full L-curve alpha selection(.2,.38073078774315083) is independently rerun and preserved in the notebook.

## Traceability

|Paper/model stage|Original source symbol|Local symbol|Validation evidence|
|---|---|---|---|
|Input and seconds conversion|loadDLSdataHelpers.readWyatFile|preprocessing.load_data|Exact source time/correlation equality, first-lag retention public|
|q and Stokes?Einstein|helpers.get_q, diffusion_from_hydrodynamic_radius, s_inverse_decay_rate|physics_model.relaxation_times|Upstream200 coordinates plus analytic radius/time proportionality|
|Siegert/coherence|helpers.get_beta_prior, g1_from_g2|get_beta through preprocessing.transform|Frozen beta and complete field including valid-prefix convention|
|Laplace acquisition|helpers.g1_first_order_corr, g2_finite_aproximation|kernel,predict|Full source predictions, zero-lag conservation, single-radius analytic decay and adjoint identity|
|Curvature and constraints|helpers.tikhonov_Phillips_reg|curvature_matrix, solvers.solve|Independent fixed-spectrum fixture, objective equality to captured augmented norm|
|Full alpha sweep|dls_experiment.getInitialEstimatesManyAlpha|compare_lcurve|All80 alpha, norms and source selected population parity|
|Triangle selection|helpers.find_Lcurve_corner, cosLawAngle|lcurve_corner|Same selected index/alpha; scalar explicit norm avoids removed array-to-scalar conversion|
|Public solution-class evaluation|Independent source objective measurement|evaluation/judge.py|Independent full objective and forward equations, no solver import|

All formulas/factors/units/signs and source call roles were reviewed. Closed linear inverse is a nonnegative quadratic fit; no artificial iterative physics or domain-library solver is substituted. scipy.optimize.nnls is a permitted general optimization primitive; acquisition kernel, curvature and constraint rows are explicit local numerical code. There are no cross-task imports, hidden-answer public arrays, domain packages, runtime source archive imports or simulator seeds.

## Fixtures and references

Hidden baseline_reference.npz contains the independently reproduced fixed-alpha reconstruction(1,2,200) and correlation(1,2,373), clearly a method reference, not molecular truth. Primary scoring is the continuous objective, not similarity to a finite optimizer path. test/fixtures/upstream.npz preserves source input_time/input_g2 and output_beta/output_s/output_radius/output_g1/output_spectrum/output_prediction/output_residuals; comparing_test consumes them. notebooks/reference_outputs/upstream_lcurve.npz contains input_alpha and output_alpha/norms/spectrum/prediction, consumed by both active notebook and complete L-curve parity. Every fixture/reference file has this documented consumer.

Before local scoring, objective boundaries fixed at1.0001 times each independent original augmented norm squared:0.004642426117226444 and0.007264271807644887. Forward consistency1e-10. These allow independently solved continuous optima and reject zero, incoherent or poor spectra. Every bin affects full curvature/mass/endpoint/data terms and both required prediction arrays must match their submitted spectrum at all373 lags. No undefined NCC/NRMSE on this inverse distribution is substituted; objective and relative forward metric definitions are explicit. Machine reports retain full numeric precision and finite JSON.

## Executed validation

- original: exit0, 1.2224s
- cli: exit0, 0.4902s
- judge: exit0, 0.1671s
- pytest: exit0, 1.5178s
- notebook: exit0, 4.7834s

Final source, local entry point, standalone Judge, pytest and complete active nbconvert all exit0. See dls_final_execution.json and dls_original/dls_cli/dls_judge/dls_pytest/dls_notebook.log in the private archive for exact arguments, wall times, numerical backend and warnings. pytest19 passed; includes malformed files/keys/shapes/dtypes/complex/NaN/Inf/negative mass, poor populations, incorrect predictions, evaluator assets/boundaries, foreign working directory and report-write failure. No threshold was relaxed after a failure. Initial notebook execution exposed a generated string-escaping syntax defect; this was repaired and the complete notebook rerun.

Notebook contains six rendered chapters and display math, actively executes fixed inverse plus every L-curve case, source comparisons and independent Judge. Two result tables report coherence, population mass, mode radius in nm, selected regularization and measured-correlation SSE. Both generated figures were individually opened using view_image: native374-lag observations and four-panel conditional populations/complete L curve/all373 residuals. All200 bins including large-radius aggregate/dust-like tails are visible without omitted ranges. Legends/axes/tables were checked. Aggregation is a model interpretation, not independently certified chemical identity. Author measurements, fresh source run and fresh cleaned-code measurements are separately identified. No archived source result is labeled fresh local inference.

Passed mandatory gates: source/paper/revision/data hashes; exact original and local observations; independent parity and physical checks; preserved constants/units/discretization; local mathematics/minimal dependencies; README completeness and public-only solvability; contract/Judge negatives;19 tests; complete active notebook and actual visual QA; both plans matching final source; licensing and inventory. No physical ground truth is available and no such truth is claimed. Parent independent acceptance remains pending. Temporary executed notebook is archived privately and removed from output; canonical notebook preserves reviewed execution outputs.

Actual-truth precedence is explicit: if ground_truth.npz becomes available, the evaluator requires independently configured truth-relative-L2 tolerance and scores it preferentially; it never silently substitutes the method baseline. A test verifies an altered baseline cannot change this priority. Extreme finite masses leading to overflow are rejected with empty metrics and finite JSON. There is currently no actual molecular ground truth.

Final guards/model-coordinate labeling validation:21pytest passed in1.38s, complete active nbconvert exit0 using research-b13 kernel and120s timeout; final private executed notebook dls_executed_review_final.ipynb. The prior19-test logs identify the historical pre-guard run.

Parent formula review correction: README full physical/objective LaTeX rewritten without control characters; Notebook math fences converted to display math. acceptance/math_review.py ran on all README/plans/NB: 3 formulas, MathJax3.2.2 and KaTeX0.16.22 each0failures. Full active nbconvert repeated exit0; private executed archive dls_math_corrected.ipynb. Algorithms, thresholds and scientific measurements unchanged.

Parent public-only control exposed floating-point cutoff ambiguity. Native raw lag_us has exactly5.0 at index40. Early gate now explicitly lag_us<5.0 in README/local/Judge before unit conversion, preserving original selection; transform signature adds native lag_us and load_data returns it. Positive prefix explicitly stops at first G<1.0, retaining exactly1.0. Regression tests verify exact5excluded and exact1retained. All22tests pass1.37s; full NB dls_native_corrected.ipynb rerun exit0 and dualrender3formulas0failures. No numerical results or acceptance thresholds changed.


## Independent parent acceptance, batch13

Parent reviewed q/D/time units, the Siegert transform, all soft-constraint and curvature rows, native-lag boundary, two plans, original provenance and conditional radius limitations. Parent independently solved the public problem using a scaled polynomial design and BVLS general bounded least squares, without importing task source or hidden arrays. The alternative passes unchanged Judge boundaries. The first control exposed the exact5microsecond floating-point selection ambiguity; native-unit selection is now explicit and regression tested. This is numerical public-only corroboration, not a blind Agent run. Both canonical figures were individually viewed, retaining both replicates, all radius bins, full80case Lcurve and complete residuals.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/carbonic_anhydrase_dls_radius_population/ records commands, reports, actual extracted canonical figures and final hashes.


## Independent parent acceptance, batch13

Parent reviewed q/D/time units, the Siegert transform, all soft-constraint and curvature rows, native-lag boundary, two plans, original provenance and conditional radius limitations. Parent independently solved the public problem using a scaled polynomial design and BVLS general bounded least squares, without importing task source or hidden arrays. The alternative passes unchanged Judge boundaries. The first control exposed the exact5microsecond floating-point selection ambiguity; native-unit selection is now explicit and regression tested. This is numerical public-only corroboration, not a blind Agent run. Both canonical figures were individually viewed, retaining both replicates, all radius bins, full80case Lcurve and complete residuals. Follow-up evaluator guard review rejected empty/missing/extra metric sets, invalid comparison directions and nonnumeric/nonfinite boundaries. Relevant new tests and existing fresh scoring pass; valid scientific boundaries and reconstruction mathematics are unchanged. Previous parent acceptance/hash evidence remains archived rather than silently rewritten.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/carbonic_anhydrase_dls_radius_population/ records commands, reports, actual extracted canonical figures and final hashes.
