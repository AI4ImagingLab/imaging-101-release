# Provenance and validation

## Fixed scientific scope

Task: spatial total tracer distribution volume V (mL plasma/mL tissue) from real dynamic [11C]PS13 human PET and measured arterial parent plasma. This differs from static PET activity reconstruction: the spatial unknown is reversible tracer distribution volume and acquisition includes the entire time series and blood measurements. Selected published method is the 1990 arterial-input Logan graphical relation, evaluated by the unchanged author weighted normal-equation implementation with k2prime=None. It is an unregularized kinetic baseline, not a reproduction of every reference-tissue model in the source package. No reference ROI, binding-potential subtraction, clipping, post-fit masking or invented truth is used.

## Sources, revisions and data rights

Algorithm paper: Logan et al., “Graphical Analysis of Reversible Radioligand Binding from Time–Activity Measurements Applied to [N-11C-Methyl]-(−)-Cocaine PET Studies in Human Subjects,” J Cereb Blood Flow Metab10(5):740–747 (1990), DOI10.1038/jcbfm.1990.127. Publisher primary PDF https://journals.sagepub.com/doi/pdf/10.1038/jcbfm.1990.127 and PubMed2384545 were verified online. Original arterial relation requires parent-plasma integral divided by tissue concentration; its slope is total distribution volume. Later primary paper https://doi.org/10.1016/j.neuroimage.2008.09.021 explicitly displays both arterial and reference-region formulas, confirming the distinction.

Author implementation: https://github.com/bilgelm/dynamicpet commit92029b5148790593f323823a7612cfd13e9d49bb, MIT, version0.1.5. Exact original checkout is privately archived at ../upstream/dynamicpet-92029b5148790593f323823a7612cfd13e9d49bb relative to this archive directory. Remote main checked live2026-10-22: 92029b5148790593f323823a7612cfd13e9d49bb; full API response remote_verified.json. The selected exact revision is retained regardless of later remote changes. MIT license copied intact; code extraction preserves all selected executed numerical steps.

Data: OpenNeuro ds004868 v1.0.1, DOI10.18112/openneuro.ds004868.v1.0.1, public S3 https://s3.amazonaws.com/openneuro.org/ds004868/ . Native sub-01/ses-baseline/pet/sub-01_ses-baseline_pet.nii.gz with matching PET JSON and recording-manual_blood JSON/TSV. Dataset LICENSE is CC0 Universal1.0 and dataset_description.json declares CC0, independently checked; retained NOTICE credits source. Primary PS13 human kinetics paper Kim et al.2020, DOI10.1007/s00259-020-04855-2, identifies plasma-input distribution volume and widespread COX1 binding; paper identity is context rather than an assertion that this exact released subject appeared in its published figures. No biological ground truth is available.

PET-BIDS specification https://bids-specification.readthedocs.io/en/stable/modality-specific-files/positron-emission-tomography.html verified live. Images/blood share the supplied decay-to-time-zero convention. Native image metadata state ImageDecayCorrected=true, ImageDecayCorrectionTime=0, InjectionStart=0. Blood metadata indicate no dispersion/recovery correction. Do not reapply decay, dose factor or an invented correction. Nonuniform frame timing, blood times, measured plasma and HPLC parent fractions remain explicit inputs. The native float32 NIfTI values are repackaged losslessly into NPZ with all27 frames and complete x/y extent at original z55. Original affine spacing2.0364201068878174 mm x/y,2 mm z; voxel-to-world affine translation shifted exactly to z55. This is one declared central slice, not full-volume reproduction. Mask is fixed from observations before fitting: mean last10frames>=1000 Bq/mL and all starts>=30min positive;5029 voxels. No material/target-derived calibration.

Private archive root: imaging-101-maintainer/research_batch6_20261022/groups/mech_bio_med/pet_ps13 (outside task). The root inventory archive_index.json contains source/data/log/figure/notebook hashes. Raw source JSON/TSV/NIfTI, original checkout, capture scripts, downloaded documents and temporary executed notebook stay private.

| Source or evidence | SHA256 |
|---|---|
| sub-01_ses-baseline_pet.nii.gz | 5cb36e1d1cc27dc9e8628ba6890c25f0115e216e0db8711961cf22ff792c019e |
| sub-01_ses-baseline_pet_sub-01_ses-baseline_pet.json | d111326c6fdadd7142779e8f62ddfe1fb30e2b87cc2d5807552846f29f510484 |
| sub-01_ses-baseline_pet_sub-01_ses-baseline_recording-manual_blood.json | b28ec061bcd8ea11fdb35c367d1410e3c0d9c5b9a7e627280b7bfd1b8003d2ee |
| sub-01_ses-baseline_pet_sub-01_ses-baseline_recording-manual_blood.tsv | c258541b8a6897198b4233be001088794f887b9145b4ba96fe2751009c9c7005 |
| LICENSE | a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499 |
| dataset_description.json | 76b7c0d0a375d44319f19ebb7398e2cde774aecbb77d6cf513d4d7b462760f4d |
| preimplementation_acceptance.json | e239debc1ee37632cdfd5298d59f648273b2c24d1d7bee30a2da504d18d3f093 |
| upstream_fixture.npz | 4f5bff45413b015fc47525e84fe8726393e68769e32a9695a157a44931c6e422 |

Bundled raw_data.npz SHA256 3c44c43140db983422b7083f47bd1cd8992cd1fc85e2d8aea5b451f28dc8f35d. Original standard reference SHA256 7534625ca6e521134d0ee782347190268d99f64666fff75459fea5e2c882ffb2. No ground_truth.npz is created.

## Original execution before extraction

Private capture_pet_original.py and capture_pet_fixtures.py run unchanged dynamicpet TemporalMatrix/TemporalImage/LRTM.fit on exactly the bound5029 TACs, all27 native frame concentrations, same19 blood inputs, minute interpolation, mask, trapz, duration weights, tstar30min, k2prime=None. Original nibabel conversion preserves exact native float32 values and no spatial interpolation. Isolated oracle environment ../asl-oracle-env: Python3.11.11, NumPy2.4.6, SciPy1.17.1, nibabel5.4.2, dynamicpet0.1.5 at exact commit; upstream package is absent from cleaned runtime. CPU single-thread, Windows, numerical libraries pinned and OMP/OPENBLAS/MKL threads1. Original fit runtime 0.202468400s; two additional independent fits had maximum differences [0.0, 0.0]. original_precision.json records environment and full metrics. Initial failed filename-case attempt is original.log, explicitly superseded by successful original_success.log and original_success_exit.json; successful logged confirmation exit0, walltime 1.716132000s. No failed attempt is treated as success.

The preimplementation acceptance file was frozen after original runs and before local extraction: NCC>=0.999999, dynamic-range NRMSE<=1e-7, additional relative weighted transformed residual<=0.06258423596752331. Exact rerun differences0 justify a small nonzero portability tolerance, not zero-error self-comparison. Original residual0.06196459006685477 sets the1.01 multiplier; no threshold was widened after observing local results. Frozen first32 original input curves, midpoint integrals, normal matrices/RHS/coefficients form test/fixtures/upstream_stages.npz. The capture wrapper delegates scipy.linalg.solve unchanged and saves its inputs/outputs; it never replaces numerical execution. The successful logged rerun confirms bitwise equal frozen arrays and does not overwrite the preimplementation criteria.

## Equation/stage traceability and ambiguities

| Paper or numerical stage | Upstream function/file | Local function | Independent evidence |
|---|---|---|---|
| Arterial Logan integrals/tissue normalization; slope V | kineticmodel/logan.py LRTM.fit with k2prime=None | physics_model.transformed_data | Analytic reversible one-tissue model V=K1/k2; frozen design coefficients |
| Frame midpoint integration | temporalobject/temporalmatrix.py and temporalobject/temporalobject.py cumulative_integral | physics_model.cumulative_integral | Frozen original tissue/plasma integrals; known linear-in-time integral |
| Measured parent input interpolation | Private binding of source TemporalMatrix input | preprocessing.parent_input/load_data | Exact measured products, start/duration/BIDS checks and known linear interpolation |
| Weighted normal equations | kineticmodel/logan.py fit, scipy.linalg.solve assume_a=sym | solvers.fit_tacs/reconstruct | Frozen32 normal matrices/RHS/coefficient stages, weighted gradient orthogonality, original full-slice parity |
| Mask and physical unit scale | Native scanner data and fixed observed support | preprocessing.load_data | Native lossless slicing, Bq/kBq joint scale invariance and README constants |
| Standard-method agreement and consistency | Reproduced original arrays, not truth | independent evaluation/judge.py | Known NCC14/sqrt200 and NRMSE0.5, bad submission/CLI/report tests, notebook metric checks |

Discrepancy1: class cites reference-tissue Logan1996 and calls its slope DVR. With arterial parent plasma and k2prime=None, executed column is integral(CP)/C, algebra exactly arterial Logan1990. Only physical label changes to V; no math correction or reference-region fabrication. Discrepancy2: source cumulative-integral docstring suggests frame ends, but executed trapz uses frame midpoints and initial start_time*first_concentration/2. This injection-start acquisition therefore starts integral at0 at first midpoint; omitted half-frame area is documented, preserved and measured as a separate diagnostic. Minute-based interpolation preserved exactly after detecting ~3e-11 Bq/mL rounding differences if interpolating in seconds. Singular-system fallback and np.allclose identical-input special case are retained and tested; no bound voxel triggers fallback. Negative real estimates16/5029 are retained and disclosed, not removed to improve metrics.

## Fresh cleaned method and metrics

Fresh local runtime 0.380502800s in canonical notebook, all5029 voxels. NCC 0.99999999999999989, dynamic-range NRMSE 0 against independently reproduced original; bitwise equal full maps. Fresh transformed residual 0.061964590066854765. Standalone Judge passes; output is real float64(1,1,200,200), absolute physical mL/mL scale, exactly zero exterior. No alignment, crop, fitted scaling or magnitude transform. Metrics use cosine NCC without mean subtraction and RMS error divided by masked reference dynamic range. Observation residual eliminates nuisance intercept independently with public input/calibration, weights in minutes, and integrates all frames before late selection. Same-method parity is not physiological accuracy.

evaluation/baseline_reference.npz is the independent original map used only by Judge. notebooks/reference_outputs/upstream_measurements.npz separately identifies the same original measurement for explicit notebook comparison; no array is presented as truth or substituted for fresh output. test/fixtures/upstream_stages.npz is the stage-fixture consumer. Every hidden file has these documented consumers.

## Active notebook and actual scientific comparisons

Canonical notebook has six self-contained chapters: problem/data/physics/method/results/conclusion, inline definitions and display mathematics, actively imports cleaned modules and reconstructs full selected data, writes fresh NPZ, evaluates independent Judge and executes all sensitivity branches. nbconvert exit0, 22.955940800s, timeout600s; private notebook.log/notebook_exit.json contain command/env. Canonical notebook SHA256 a56e161f031001c52270597c6187634dc3206bea6e60faec68151d5dca819721. No error outputs. Eight actual figures exported PNG/SVG/PDF with numeric sources and scientific_measurements.json privately: observed early/mid/late activity; measured parent-plasma and early bolus; four observed-activity-quantile-selected voxel fits; original/fresh/difference; activity versus kinetic contrast; residual and negative-value diagnostics; four late-start reconstructions; parent/metabolite/quadrature/weight comparisons. All branches freshly computed, no missing method placeholder. Initial-half-frame triangle is explicitly a local diagnostic extension, not an upstream-reproduced algorithm. All late-frame durations equal5min, so equal-weight branch actually coincides. Later tstar lowers residual while medianV shifts from3.17039(20min) to2.94149(50min): late-model uncertainty remains visible. No new PINN/compartment-fit comparison is claimed. Original source package contains models not selected by this task; they are neither silently relabeled nor claimed reproduced.

## Runtime, source and plan review

Clean runtime subset NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, standard library; no nibabel/dynamicpet in shipped requirements/imports. CPU checked. AST imported-symbol and call-role audit symbol_audit.json covers solver, physics, preprocessing, visualization, main, Judge and tests; notebook local imports reviewed. General interpolation and symmetric linear solves are permitted numerical primitives; kinetic quadrature/design/weights/inverse are explicit local code. No cross-task/domain imports, hidden target calibration, backend mutation, opaque solver or generated observation. No generate_data.py because bound observations already exist; analytic test signals do not create benchmark observations.

Plans were semantically checked against actual function signatures/defaults/units/array orders and call paths: minute interpolation, midpoint quadrature, pre-late integration, duration weights (not squared), c=[b,V], real unconstrained V, default tstar30, analytic2x2 solve, no iterative initialization/regularizer/stopping rule. Source fallback, objective normalization and diagnostic extension are explicit. README has every public key, complete output contract, acquisition geometry/noise/timing/decay/boundary conventions, no solver hyperparameter table or hidden assets. Solver-visible README/requirements contain no GitHub URL. No metadata JSON or public target exists.

## Mandatory gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Exact source/paper/data and traceability | passed | Source revision, primary equation check, hashes and mapping above |
| Original exact bundled observations | passed | Original pre-extraction precision/fixtures, successful confirmation log exit0 |
| Independent stages and physics | passed | Original stage/full parity, analytic compartment V/units, known quadrature and normal-equation gradient |
| Constants/units/coordinates | passed | Lossless native slice, supplied timing/affine, README tests |
| Standard CLI and fresh standalone Judge | passed | main fresh pipeline test, canonical notebook, judge.log |
| Judge invalid inputs/poor output/evaluator/report errors | passed | Tests missing/key/extra/shape/dtype/complex/object/NaN/Inf/exterior/broken/poor/missing assets/invalid config/foreign cwd/report failure |
| Public self-contained allowlist/no target leakage | passed | Key tables/calibration provenance, public URL scan, no truth |
| Local mathematics/minimal clean environment | passed | Explicit quadrature/normal equations, call audit, runtime execution |
| Unit/parity/physics/pipeline tests | passed | tests_final.log; 29 passed in 2.897727300s including independent visualization metrics and diagnostic quadrature |
| Active six-chapter executed notebook | passed | nbconvert exit0, eight actual comparisons and source arrays |
| Provenance/metrics/plans semantic consistency | passed | Fresh/reference distinction, known metrics and review above |
| Inventory/licenses/archive | passed | Canonical inventory and private archive index; MIT+CC0 notices |

Public task packaging uses only the authoritative directories and canonical notebook. Temporary notebooks/full upstream sources/archive/downloads remain private. Python caches removed after final verification. This completion claims a validated standard-method kinetic benchmark, not independently known physiological V or whole-volume reproduction.

Final supplemental suite:29 passed, exit0. All eight PNGs inspected for truthful labels, units, zero-neutral signed differences, matched map limits and actual data comparisons. No source changes followed canonical notebook execution.

Final title-layout revision: observed-activity and original/fresh comparison titles split into two lines; complete canonical notebook reexecuted successfully,22.955940800s. Numerical methods unchanged.

Final supplemental suite:29 passed, exit0. All eight PNGs inspected for truthful labels, units, zero-neutral signed differences, matched map limits and actual data comparisons. No source changes followed canonical notebook execution.
