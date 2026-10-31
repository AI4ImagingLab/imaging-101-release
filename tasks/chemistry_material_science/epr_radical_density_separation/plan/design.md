# Shipped interfaces and data flow

## Public input and output

The task root contains README.md, requirements.txt and data/raw_data.npz as the solver allowlist. The latter has exactly the four batch-first float64 arrays specified in README. The standard command is python -m src.main. It loads the public NPZ, reconstructs both complete fields and writes output/reconstruction.npz with exactly reconstruction of shape(1,2,100,100), real float64. Output directory creation is its only filesystem side effect.

No runtime source module reads evaluation assets, source fixtures, notebook references or private paths. No domain reconstruction package or upstream checkout is imported. NumPy provides general matrices, exponentials, FFT and random numbers. Matplotlib is used only for review figures. Thread environment variables are set in CLI execution before importing numerical modules; importing src.main does not choose or change a Matplotlib backend.

## preprocessing.py

Constants DELTA_CM=.03, IMAGE_SHAPE=(100,100), NOISE_STD=5000 match acquisition metadata. EXPECTED records the four batch-first shapes.

- load_data(path) returns a dictionary with the batch dimension removed. It requires exact keys/shapes, float64, finite values and increasing field coordinates; it does not normalize, remove noise, resample or alter spectral calibration.
- make_model(data) constructs EPRModel from the three calibration arrays and the fixed spatial spacing/shape. Its observation argument is not needed for calibration.

## physics_model.py

- grad2d(u) accepts one(H,W) real field and returns(2,H,W), forward y/x differences with outward difference zero.
- div2d(p) accepts(2,H,W) and returns(H,W), the negative adjoint.
- EPRModel(field,gradients,spectra,delta=.03,shape=(100,100)) stores field(N,), gradients(2,P), spectra(2,N), all float64. It computes field-frequency masks, irregular dimensionless nodes, template rFFT and a private in-memory phase cache. The cache contains only calibration-derived exponentials.
- phases(shape) returns the y/x centered exponential tables for a requested image/kernel grid. Tables have shapes(number-of-retained-nodes,H/W), complex128.
- type2(u) computes direct nonuniform Fourier coefficients at every retained node from a real image.
- type1(c,shape) computes the conjugate transpose sum from complex retained-node coefficients to the centered mode grid.
- project(u) accepts(2,H,W), uses the declared acquisition quadrature, and returns real(P,N) receiver spectra.
- adjoint(observations) accepts real(P,N) and returns(2,H,W).
- kernels() returns(2,2,2H,2W), real cross-source Toeplitz kernels.
- NormalOperator(kernels,shape) stores their rFFT2 representation. Calling it on(2,H,W) returns the same-shape normal-operator action. The only crop is the documented padded Toeplitz embedding, internal to this operator.

The default model uses about8700 retained nodes. Phase tables are built separately for100² and200² grids; no full48000×20000 system matrix is created. Fine-grid generation is isolated from primary reconstruction and uses a separate model/cache.

## solvers.py

- squared_norm_sequence(u) accumulates source-wise squared norms.
- lipschitz_power_iteration(init,normal,tol=1e-5,nitermax=1000) returns the original1.05 safety-scaled operator bound.
- primal_dual(init,normal,backprojection,Lf,weight,tol=1e-5,nitermax=5000,positivity=False,energy=None) returns reconstruction(2,H,W) and a diagnostics dictionary. The optional energy callback only logs; default is no logging. Diagnostics include iterations, relative_update, first_iterate_inputs/gradients(up to4,2,H,W), history(Q,2), energy_history(R,2), tau and sigma.
- reconstruct(observations,model,lbda=10.,tol=1e-5,nitermax=5000,positivity=False,track_energy=False) computes kernels, normal operator, data backprojection, initializer, Lipschitz bound and normalized TV weight before calling primal_dual. It returns the field and diagnostics augmented with kernels, backprojected_observations, initialization, lipschitz and unnormalized_tv_weight.
- field_metrics(images,truth) returns(2,3) with cosine NCC, dynamic-range NRMSE and relative amount error. It is a review helper, not imported by the Judge.
- review_comparisons(data,primary,diagnostics,truth,clean_observations) computes six method variants, three noise-amplitude cases and three angular cases. It returns a numeric-array dictionary and a list of method labels. Truth/clean observations are supplied explicitly by the maintainer notebook; they never enter primary reconstruction. Returned method/noise/angular metrics are(variant,2,3).

## generate_data.py

This is required because the task uses author-defined synthetic observations and the review makes noise controls.

- physical_truth(shape=(100,100),delta=.03) evaluates the hidden author continuous geometry on a centered nodal grid.
- generate_author_observations(calibration,seed=20261022,noise_std=5000.) explicitly runs the fine-grid forward operator and reproducible receiver-noise generation. It returns clean, noise, observed and truth. It is a validation/regeneration tool; standard main does not call it or replace the exact bound NPZ.
- noise_variants(clean,noise,scales=(.5,1.,2.)) returns three freshly constructed observations using the fixed supplied realization. It generates sensitivity observations, not additional independent acquisitions.

## main.py and evaluation

run(root=None) resolves the task directory from its file when root is omitted, loads public data and returns reconstruct's field/diagnostics. main() parses optional --output, creates its parent and writes the single declared NPZ.

evaluation/judge.py is independent of solver imports. evaluate(prediction,asset_dir=None) returns(report,exitcode), using its own file-relative evaluation directory by default. Actual ground_truth.npz has priority over baseline_reference.npz; malformed truth is an error, not a reason to substitute another reference. It validates JSON numeric types without accepting bool/string coercion. predict_observations implements the Fourier model separately, one projection at a time.

CLI accepts --prediction and --report, resolves these relative to the caller and returns0/1/2 for pass/valid failure/error. Errors have empty metrics. Failed report writing emits a stderr diagnostic and returns2.

## Notebook and figures

The canonical notebook actively executes the standard CLI with the actual kernel executable and verifies its child returncode. It also computes a fresh energy-tracked reconstruction through local modules and verifies equality with the just-written CLI field. Frozen source arrays and actual truth are loaded only for identified comparisons. It independently executes Judge, review ablations and physics controls, then displays and exports six complete figure groups. Chapter order is problem, data, physics, method, results, conclusion.

visualization.py contains review-only figure functions and numeric export helpers. It never selects a backend. Notebook-side export writes paired PNG/SVG/PDF plus numeric NPZ/JSON to the maintainer archive when its explicit directory is provided. No private archive is a runtime dependency.

## Tests and provenance

All tests are under test/. original_numeric_stages.npz has frozen prelocal source kernels, adjoint, initialization, first updates, operator cases and final fields. original_generation_control.npz has independently captured original generation precision cases. Each has a documented consumer in validation.md.

Tests cover function interfaces, strict public loading, separate Judge input/config/reference-priority cases, source numerical parity, analytic point-source sign, units, Euclidean adjoints, TV boundaries, objective finite differences, generation and one complete fresh CLI/foreign-directory Judge. Temporary evaluator assets are copied to an explicitly private pytest base directory. No tests import the upstream package.

## Review helper signatures

review_physics(model,normal,observations) returns the five independent numerical errors and their labels: analytic point-source shift, forward/adjoint dot product, Toeplitz normal action, TV adjoint and directional objective derivative. It uses explicit analytic/random controls rather than reference reconstruction quality.

acquisition_figure(data,reference), field_comparison_figure(truth,source,fresh), method_maps_figure(review,names), bias_and_metrics_figure(review,truth), sensitivity_figure(review), and physics_and_residuals_figure(data,fresh,diagnostics,physics,physics_names,metrics) each return a Matplotlib Figure without writing files or selecting a backend. save_group(fig,directory,name,arrays,metadata) writes name.png, name.svg, name.pdf, name.npz and name.json under the explicitly supplied directory. Numeric arrays and provenance metadata describe the actual plotted measurements.
