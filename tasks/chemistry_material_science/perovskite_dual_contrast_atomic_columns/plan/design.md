# Shipped interfaces and execution

The cleaned runtime uses NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1. General SVD/LU/QR, finite-difference LM, maximum filtering, nearest-neighbor queries and assignment are numerical primitives. The task's Gaussian image formation, local objective preparation, density clustering, plane construction, sequential profile subtraction and reconstruction pipeline are explicit local code. There is no Atomap, HyperSpy, scikit-image, scikit-learn, HDF5 reader or cross-task import in the cleaned runtime. Those packages exist only in the private original-code environment. Existing observations are reused, so no benchmark-data generator exists.

## Preprocessing

`load_data(path)` reads NPZ with allow_pickle=False and validates both required arrays against SHAPE=(1,811,913), real float64, finite and positive values. It returns a dictionary `adf,abf` of independent unbatched float64(811,913) arrays in detector a.u. No denoising, geometric transformation, saved target or intensity normalization occurs while loading. PIXEL_NM is0.012443924564929288; constants are checked against README.

`filter_adf(image,seed=20261022)` returns a float64 image of the same shape, with background removal and rank22 initialization. It uses a private NumPy RandomState; it does not modify global RNG state or inputs. Its normalized output has arbitrary initialization scale only. `find_peaks(image,separation=12)` returns an integer(N,2) array in x,y pixel coordinates, ordered by decreasing filtered intensity, with the stated border and spacing rules.

## Image formation

`gaussian(p,x,y)` takes six parameters `[integrated amplitude,sx,sy,cx,cy,theta]`, broadcasts arbitrary numeric x/y coordinate grids and returns sampled contrast. Coordinates and widths are pixels, theta radians, and the integrated amplitude has contrast×pixel² units. It explicitly evaluates the reference quadratic form.

`physical_columns(parameters,pixel_nm)` takes float64(N,6) internal `[cx,cy,sx,sy,theta,integrated amplitude]` rows and returns float64(N,6) `[cx_nm,cy_nm,Cxx_nm2,Cxy_nm2,Cyy_nm2,peak]`. Peak retains detector a.u. for ADF or a.u.⁻¹ for reciprocal ABF. There is no angle fitting/alignment or per-image normalization in this conversion.

`model_image(parameters,shape)` returns float64 image(shape), adding each profile on its actual5-maximum-width support in parameter order. Boundary clipping uses the provided image shape; it introduces no periodic extension. It has no file or plotting side effects.

## Inverse functions

`fit_crop(image,initial,x_offset,y_offset)` takes an already prepared rectangular image, initial parameters in gaussian() order and integer crop-origin offsets in pixels. It evaluates unweighted model-minus-observation residuals and calls general SciPy leastsq with the original free parameter ordering/defaults. It returns six fitted Gaussian parameters in gaussian() order. It does not estimate background, alter the input, write files or claim an analytic gradient.

`nearest(positions,k=9)` accepts(N,2) x/y pixel positions and returns integer(N,k−1) neighbor identities, excluding self. `center_of_mass(image,position,radius)` takes a full real image, centre(x,y) and circular radius in pixels and returns a float64 x/y position from normalized moments in a zero-padded diagnostic crop. A zero-mass crop raises.

`fit_one(image,parameters,radius,already_fitted=False)` accepts a full image, one internal six-parameter row, radius in pixels and amplitude-initialization status. It explicitly prepares/masks/background-subtracts the source crop, invokes fit_crop, validates profile geometry and runs the original retry/fallback rules. It returns `(parameters,fitted_flag)`; a source fallback sets amplitude zero and retains a diagnostic status rather than pretending the Gaussian succeeded. Inputs are copied, not mutated.

`gaussian_pass(image,parameters,neighbor_indices,fraction,fitted=None)` sequentially fits every row, updating current distances while retaining neighbor identities. It returns copied `(parameters,boolean_flags)`, shapes(N,6) and(N,). A missing fitted array starts all columns as unfitted. No identity or result is dropped.

`refine(image,parameters,fraction,gaussian_repetitions=1,com=False)` freezes8 neighbor identities from the initial positions, optionally performs the selected centre-of-mass pass, then two actual free-rotation Gaussian passes per nominal repetition. It returns all float64(N,6) internal rows. The source's ineffective rotation-disabled flag and retry conventions are documented in approach; they are not silently repaired into a different published-method claim.

`translation_vectors(positions,separation=12)` computes and clusters ordered float32 displacement vectors and returns up to six rounded unique `(dx,dy)` pixel tuples in the source's crystallographic order. Density connectivity is explicitly implemented using general cKDTree neighborhood queries; no clustering API is imported. This binding has fewer than150000 selected displacements; exceeding that validated cap raises rather than silently using an unvalidated sampling convention.

`atomic_planes(positions,vector,separation=12)` walks14 neighbors in the positive and negative vector directions, returns lists of atom indices ordered along each plane, and sorts the planes using the source orthogonal reference point. Cycles raise. `midpoint_lattice(positions,axis_index)` uses the selected translation direction and returns float64(M,2) adjacent-plane midpoints; empty or fragmented required directions raise.

`subtract_columns(image,parameters,neighbor_indices,fraction=.35)` refits the specified sublattice with unfitted auxiliary amplitudes and returns the full float64 residual image. It does not replace the reported cation parameters. Oxygen's B subtraction explicitly passes fraction.30.

`initial_parameters(positions)` returns float64(N,6) internal initial rows with widths1.2pixels, rotation.01rad and provisional amplitude1.

`reconstruct(adf,abf,pixel_nm=.012443924564929288,capture=False)` actively executes the complete selected three-sublattice inverse from both full observations. It returns `(outputs,states)`. Outputs has all three named batch-first float64 arrays: a_columns(1,1009,6), b_columns(1,958,6), oxygen_columns(1,922,6), with the README column conventions. With capture=False, states is empty. With capture=True it contains computed filtered initialization,initial peaks,A/B/O internal parameters,three decomposed images,and B/O initial rows. It reads no files, references or upstream libraries and changes no backend or global RNG. Cardinalities emerge from the geometry, rather than selecting a target number of highest-scoring profiles.

## CLI and notebook flow

`src.main.run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz')` loads the public observations, calls reconstruct, creates the output parent and writes only the three declared keys to compressed NPZ. It returns the output dictionary. `python -m src.main` exposes matching --data/--output arguments. Importing main has no backend/file side effect.

The active notebook separately calls reconstruct(capture=True), writes its fresh output and invokes the independent Judge CLI. Its computed comparisons include all three standard profiles, analytic controls, full-field/zoom structure, ellipticity maps/axes/trend, contrast maps, automatic plane spacing, all six oxygen distance-asymmetry directions, full original parity, actual raw-contrast fits using the same fresh standard initialization, complete alternative randomized initialization, and integer/centre-of-mass refinement. Comparison-only orchestration is explicitly distinct from the fixed standard command. Archived original outputs and independently rerun original-RNG measurements are labelled comparisons, never loaded as fresh output.

## Scientific visualization

`ellipse_properties(columns)` accepts unbatched physical(N,6) rows and returns major/minor width ratios(N,) and principal-axis vectors(N,2). `plane_spacings(positions,axis_index=2)` accepts x/y pixel positions and returns ordered plane-normal mid-coordinates,adjacent spacings and the selected direction, all in pixels. It does not infer a strain-free lattice or manual interface-plane index.

`distance_asymmetries(positions)` returns one `(direction,positions,signed_distance_difference)` record per selected direction. Coordinates/differences are pixels; each difference is preceding-neighbor distance minus following-neighbor distance. Notebook multiplication by PIXEL_NM×1000 converts differences to pm. No cation points with artificial zero displacement are inserted to fill a map.

`export_figure(fig,directory,stem,numeric,caption)` writes paired PNG/SVG/PDF and numeric NPZ plus an identified JSON hash record to the explicitly supplied maintainer export directory; it returns that record. Only this helper writes visualization artifacts. Library modules never choose a Matplotlib backend. The notebook selects inline rendering in its own execution path and uses an explicit private export directory or a temporary default outside the task package.

## Independent evaluator and tests

`evaluation/judge.py` has no solver imports. `read_arrays(path)` validates all required shapes/dtypes/finiteness. `invalid_profiles(a)` counts covariance/peak violations. `compare(a,b)` matches within-sublattice identities and returns maximum/RMS centre errors in pixels, relative covariance/peak L2 errors and invalid-profile count. Assignment changes ordering only. No registration, intensity scaling, output substitution, crop or mask is performed.

`evaluate(prediction,reference_dir=HERE)` resolves its hidden assets relative to the evaluator directory, gives an existing actual truth strict priority, validates nonnegative finite metric boundaries, checks reference profiles, scores every required field and returns `(finite_JSON_report,exit_code)`. Invalid submissions/assets/derived arithmetic give error2 with empty metrics; valid poor estimates give fail1; all gates passed gives pass0. The CLI requires --prediction/--report, retains caller-relative CLI paths, creates the report parent and returns2 on report-write failure with a stderr diagnostic.

Tests consume independently captured source stages and known original Gaussian controls under test/fixtures, plus the public observations. They check source filtering/peak order, original individual fits, full sublattice updates, lattice midpoints, all three subtractions, physical covariance/unit equivalence, analytic image formation/recovery, numerical metric cases, NPZ and hidden-asset failures, truth priority, poor values in each quantity, physical validity, finite-input overflow and CLI/report behavior. The pipeline test runs the complete standard command. Jupyter/pytest are shared development tooling, not task runtime requirements.
