# Provenance and validation evidence

## Fixed scientific binding and semantic novelty

Task: complete dual-contrast projected atomic-column profile inverse for a real LSMO/LFO/STO heterostructure. Both full811×913 calibrated author observations are bundled unchanged numerically, with all1009 A,958 B and922 oxygen columns reconstructed. Required centres,covariances and peak contrasts are separately evaluated in each sublattice. The source API, independent numeric fixtures and gates were frozen at 2026-10-22T11:50:55.816720+00:00 before any local solver implementation, in private `binding_frozen.json`.

This differs from existing atom-probe3D ion-coordinate reconstruction, STEM tilt tomography, electron ptychographic complex-object recovery, electron holographic phase, Bragg/GPA strain and EBSD orientation: its observations are simultaneous ADF/ABF pixel contrast and its unknowns are three projected column-profile point sets. No change of sample/method alone is used to count a new inverse problem.

## Primary paper, exact original code and author data

Paper: Nord,Vullum,MacLaren,Tybell,Holmestad (2017), Advanced Structural and Chemical Imaging3:9, DOI https://doi.org/10.1186/s40679-017-0042-5. The complete12-page published article was obtained legally from the institutional author repository https://eprints.gla.ac.uk/136873/1/136873.pdf, stored privately as paper_published.pdf and read through paper_published.txt; direct publisher-PDF TLS failure is retained as unsuccessful, not fabricated. Published PDF SHA256 `bc4e881acc58773b0b7b4ae6e69ee8479cf21f8d149428ecea23343ce1007233`. The paper covers Gaussian image fitting, column positions/profiles and oxygen structural distortions; its claims are not substituted for measurements in this task.

Official source: https://github.com/atomap-dev/atomap, release commit `274debc5b252f4b30f7ca5aeaa834a3840851a15`, version0.4.3.dev1. Official demonstration data/notebook: https://github.com/atomap-dev/atomap-demos, main commit `394e0b66908bd4ea31b27cb1a459124b8bb90de0`. Git remote/API queries actually verified these selected revisions; no claim that the release commit is the current main HEAD is made. Private source_remote.json,demo_source_remote.json,source_acquisition.json and exact codeload archives identify verified remote revisions, SHA256 and commands. Documentation's stale underscore-name demo URLs returned404 and clone attempts had connection resets; these were not silently treated as available sources. The canonical hyphen-name author repository supplied the exact data.

Private archive root P is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/root/atomap`. Source/paper/download/capture scripts/logs remain there, outside the final task. The maintainer archive index enumerates content hashes; oracle_environment.txt and runtime_environment.json identify original and cleaned environments.

| Private original artifact | Bytes | SHA256 |
|---|---:|---|
| `demo_source/ADF_image.hspy` | 4889108 | `9a5067efab3a54f42f179aefb211262368b280110b8630f1fc40aab8b6c8d705` |
| `demo_source/ABF_image.hspy` | 4677310 | `7676b390f78bf3862e5d450a82a1963a564aabc44091b2868fcbe86d2d203025` |
| `paper_published.pdf` | 3962198 | `bc4e881acc58773b0b7b4ae6e69ee8479cf21f8d149428ecea23343ce1007233` |
| `source.zip` | 1754134 | `8745148131679a10f5f503a8b5bf0e53ebcf4f4b2f5a2dd2e064726b3ab03e8c` |
| `demo_source.zip` | 13856725 | `a5e49662484e535319a3b500ddb4f38ecf99c3bc54833bd4662090f4206d78cc` |

Selected existing observations are full ADF_image.hspy and ABF_image.hspy, not the earlier scripts/test_*_cropped.hspy preflight pair. Both selected full-image axes explicitly carry nm and delta=0.012443924564929288nm/pixel, offsets0. The cropped pair's unit label is absent, so its unrelated scale was never relabelled nm. The full images are authored existing real measurements; no synthetic benchmark observation was generated. The tutorial describes LSMO upper film,LFO middle film,STO substrate and simultaneous channels. It does not provide independent column truth or elemental-occupancy calibration. Source and demo software distribute GPL3 notices; separate microscopy-data-specific permission is not stated. NOTICE preserves this limitation plus source/Gaussian/scikit-learn attributions and full terms. This package is local research material; no undocumented public-data grant is claimed.

## Original execution before local extraction

Private original environment: Python3.11.11, actual selected source on PYTHONPATH, NumPy2.4.6,SciPy1.17.1,HyperSpy2.5.0,scikit-image0.26,scikit-learn1.9.1,Matplotlib3.11.2 and original ancillary packages; complete installed versions are in oracle_environment.txt. This environment is separate from the cleaned runtime. CPU threads1, Windows, noGPU required. run_original.py monitors physical/commit availability and retains commands,seeds,native exits,runtimes,stdout/stderr and samples; required availability floor6GiB. Numerical backend configuration is separately captured, not inferred from package version alone.

Full original command is the selected unmodified public automated API with PerovskiteOxide110(),pixel_separation12,s_image1=full ABF and fixed algorithm RNG20261022. It is not the interactive tutorial's manual refinement sequence, cropping, plane34 selection or (100)-projection synthetic polarization example. The selected API and tutorial acquisition are explicitly associated before implementation. Original Gaussian and reconstruction functions were not patched.

| Full original execution record | Native exit | Wall seconds | Minimum physical GiB |
|---|---:|---:|---:|
| `full_heterostructure_process.json` | 0 | 127.066194100 | 11.6608 |
| `full_passive_process.json` | 0 | 124.056411100 | 9.9530 |
| `full_other_seed_process.json` | 0 | 124.071555500 | 10.9174 |

native_full_heterostructure/original_arrays.npz stores original inputs and all original parameters/images. The active original inverse itself took122.435414s. native_full_passive/source_stages.npz has39 independently captured stages; read-only hooks changed no Gaussian/inverse arithmetic, and every final parameter/image is bitwise identical to that original full run. Independent complete algorithm seed20261023 uses the same observations/configuration and gives a genuine nonzero repeatability envelope, not an observation-generation seed or cached report.

Earlier full cropped unseeded and seeded preflights are retained separately. The first unseeded passive wrapper reconstructed successfully but then failed its own exact-equality assertion because randomized SVD changed ordering/equivalent ellipse gauges; it is not labelled native0. A fixed source seed and the same entry point resolved this capture issue before selecting the full benchmark. HyperSpy deprecated-import and isotropic-angle covariance warnings remain in original stderr. No confidence interval is inferred from singular circle-angle covariance.

## Paper/code convention differences and independent physics

The paper uses a peak-amplitude Gaussian and a rotation-sign convention that differs from the current source's normalized integrated-amplitude quadratic. The local code preserves the current selected source and explicitly converts to covariance/peak output. The current background is the median of the lowest3percent inside each mask, rather than the paper's minimum. The nominal rotation-disabled source pass actually frees rotation inside the fitting routine; the actual selected behavior is preserved, not silently repaired or described as unchanged2017 software.

Before local mathematics, three independent analytic rotated/isotropic Gaussians were defined using a covariance quadratic form rather than the source formula. Original formula relative errors were at most3.04e-16, known-centre errors at most5.48e-9pixel and covariance relative errors at most5.22e-8. The original and cleaned fits both meet the predeclared .005pixel/.02covariance control gates. Independently known physical conversion, ellipse-axis gauge equivalence and metric cases are unit-tested. Agreement of two copied formulas alone is not used as physics evidence. source_physics/criteria_before_run.json,known_original_controls.npz and results.json retain controls, source measurements and preflight qualifications.

## Equation / algorithm / implementation / test traceability

| Required mathematical stage | Exact upstream file/function | Local function | Independent validation |
|---|---|---|---|
| Paper Gaussian image profile, rotated quadratic, amplitude | external/gaussian2d.py::Gaussian2D.function | physics_model.gaussian,physical_columns | known covariance controls; first3 original fits; gauge/unit tests |
| Background removal / rank initialization | atom_finding_refining.py::subtract_average_background,do_pca_on_signal | preprocessing.filter_adf | full original filtered image and peak order |
| Strong column maxima | atom_finding_refining.py::get_atom_positions | preprocessing.find_peaks | full original1009 initial peaks, exact order |
| Mask/background/objective and LM | atom_finding_refining.py::fit_atom_positions_gaussian,_fit_atom_positions_with_gaussian_model | solvers.fit_one,fit_crop | original crop/initial/final Gaussians and all full-sublattice updates |
| Centre-of-mass and sequential neighbor updates | atom_position.py::_get_center_position_com;refine_sublattice | solvers.center_of_mass,refine,gaussian_pass | full original B/O stage fixtures; known-centre recovery |
| Translation direction clustering | sublattice.py::_get_fingerprint,_make_translation_symmetry;tools.Fingerprinter | solvers.translation_vectors | exact source-initialized complete A→B and B→O midpoint sets |
| Atomic planes and missing-sublattice midpoints | sublattice.py::_find_atomic_columns_from_atom,find_missing_atoms_from_zone_vector | solvers.atomic_planes,midpoint_lattice | all958 B and922 O starting coordinates |
| Explicit Gaussian subtraction | tools.py::remove_atoms_from_image_using_2d_gaussian;sublattice.get_model_image | solvers.subtract_columns;physics_model.model_image | all three full subtraction images; preserved .35/.35/.30 fractions |
| Complete selected3-sublattice inverse | main.py::make_atom_lattice_from_image;process_parameters.PerovskiteOxide110 | solvers.reconstruct;main.run | complete CLI and independent Judge on exact bundled observations |
| Structural scientific diagnostics | tutorial spacing,ellipticity,oxygen-distance examples | visualization.ellipse_properties,plane_spacings,distance_asymmetries | actual fresh full-field plots and identified numeric exports |
| Metrics / NPZ / evaluator | independent specification, not solver library | evaluation/judge.py | known numerical cases; every quantity/invalid input/truth priority/error contract |

No analytic-gradient claim is made for finite-difference LM. Randomized range finding explicitly uses general NumPy/SciPy matrix factorizations; density clustering is explicit neighborhood expansion. Neither scikit-learn nor a domain reconstruction API appears in local imports. No cached original positions initialize the cleaned method.

## Frozen tolerances and separately identified measurements

Every required field is evaluated separately in all three sublattices with identity-only assignment and no geometric alignment. Original independent RNG maximum centre discrepancy .076579925pixel,RMS.002410928pixel,covariance relative L2 .000900950,peak relative L2 .0000243183. Before local code, rounded-up3×gates were frozen as .23pixel,.0073pixel,.0028,.000075. Profile violations must be zero. Subpixel centre gates are meaningful relative to the22–31pixel lattice spacings. Half-pixel shift,1percent covariance/peak changes and negative/non-SPD profiles are explicitly rejected. This is a nonzero independently measured tolerance, not source self-comparison1/0 and not an assertion of true structural precision. Full figures compare original variations with a separately computed full local RNG rerun.

Fresh complete local inverse first took30.4147179s. Final pinned Notebook standard took33.432275500s; actual raw-contrast ablations took13.211109500s; a whole alternative randomized initialization took33.740811500s. Its A output is bitwise equivalent to the original; maximum B/O centre discrepancies are8.05697e-9/1.62667e-7pixel and covariance relative errors1.00044e-9/1.32030e-7. Full precision is in output/judge_report.json and private experiment_summary.json. These are fresh local measurements, distinguished from the independent original run and its RNG sensitivity; no paper or historical cached number is labelled fresh.

During extraction, the stage test correctly exposed a1.013percent ABF residual mismatch caused by passing.35 instead of the source B-subtraction fraction.30. Reading the complete configuration resolved it; neither the method nor any frozen gate was altered. All three subtraction parity tests subsequently passed. A separate private diagnostic accidentally omitted its fraction argument and used the upstream default.40; its non-equivalent output remains diagnostic failure, not additional source parity. Initial missing-module stage tests fail before extraction, then progressively pass. Test-driven execution history is retained in the session and summarized here rather than fabricated as upstream fixtures.

## Fixture and reference consumers

test/fixtures/upstream_stages.npz is the independently captured complete original passive run, not generated by local functions. Its initialization,per-fit crop/mask/parameter,sublattice update,subtraction and final-image/parameter keys identify the captured stage. Comparing tests consume filtered image/initial peaks,first3 fits,all3 sublattice updates,initial midpoint sets and3subtractions; unused capture context is explicitly retained for maintainer diagnosis, not claimed as separately tested. known_original_controls.npz contains the3independently specified analytic observations,known parameters and actual original fits; unit tests and Notebook physics comparisons consume it.

evaluation/baseline_reference.npz contains converted original centres/covariances/peaks. notebooks/reference_outputs/upstream_measurements.npz identifies the same original full result for Notebook comparisons. upstream_rng_variation.npz identifies the separately run original seed20261023. cleaned_measurements.npz contains fresh standard,complete fresh alternative RNG,actual raw-contrast alternative B/O fits and actual integer/COM diagnostics, with explicit named consumers in the Notebook. No array is relabelled ground truth. Raw HSPY files,paper,archives and full source stay private.

| Packaged numeric artifact | Bytes | SHA256 |
|---|---:|---|
| `data/raw_data.npz` | 11046145 | `e7dc2c97cc070382b70e855a6ef38f29eaa9a4abf409f629673dc37be6ea8134` |
| `test/fixtures/known_original_controls.npz` | 11930 | `1943b26cc9b37ab7783a12fe18989e9b437c64f78459cae25ec7816d5a89b8e9` |
| `test/fixtures/upstream_stages.npz` | 66363608 | `8a48297ee0ecddab363fc1fed7973fefcfc32a7defb2447b1aa4920d541752f1` |
| `notebooks/reference_outputs/cleaned_measurements.npz` | 376929 | `da20673a166cbce9a664768aca0f659300d917a9f38828487c8b53415b685ac6` |
| `notebooks/reference_outputs/upstream_measurements.npz` | 134667 | `454d613f5ca0eb39b8d33bc5d0ec185647affa06abd80b32c3cf60b7645fe4ff` |
| `notebooks/reference_outputs/upstream_rng_variation.npz` | 134670 | `09d7125e02fe695839fbf3e33045bc2eb76de71c5e2fe1a5600a24a1c02ca804` |
| `evaluation/baseline_reference.npz` | 134667 | `454d613f5ca0eb39b8d33bc5d0ec185647affa06abd80b32c3cf60b7645fe4ff` |

## Notebook execution, scientific preservation and visual review

The complete18-cell six-story Notebook was actually executed with native0/125.817850500s using pinned research Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,CPUthreads1. Private kernel registration is under P/kernel_share; no shared kernel or package was changed. The actual kernel prints its executable and versions in Notebook output. Timeout600s/cell is justified by the122s original and30–34s local full paths plus actual comparisons. The complete full-executed Notebook SHA256 is `910c1522410fe0ba6a0ddb1636b1161a480a748f6aab4532318a7943ace1cda8` at P/atomap_executed.ipynb. It contains12embedded scientific PNG outputs,all code execution counts and no error output.

Visual review found NumPy scalar representations made the6oxygen-direction titles overlap. A genuine presentation-only nbconvert run native0/5.888536300s rerendered only those labels with two-decimal coordinates using the previously computed fresh diagnostic arrays. All12original numeric exports remain unchanged; original fig08 numeric bytes were restored after verifying only1.42e-14pm plotting-unit roundoff. No inverse,observation,threshold,solver or other cell output was changed. Final canonical SHA256 is `cdc8c4b13ddd2e451985ba881869f8f1c119d50aaff3291ecb10f2a83101c643`; it is deliberately distinguished from the full executed SHA, and figure8_presentation/execution.json records both. The original complete execution and revised single-cell presentation execution are separately archived; no cache-preview is counted as active reconstruction.

All12final groups have actually been opened: raw images/FFT,known analytic profiles,active decomposition,full/zoom structure,3-sublattice ellipticity,3-sublattice peak contrast,A principal axes/row trend/plane spacing,all6oxygen distance-asymmetry maps,full original parity,actual contrast ablation,full original/local RNG paired bars and actual integer/COM/Gaussian comparison. The oxygen-map final short titles were also reopened. Units,legends,neutral signed zero and displayed color limits are explicit. Ellipticity is not atom radius; pm structural differences are not independent accuracy. Automatic spacing/oxygen diagnostics preserve the tutorial's scientific comparisons without falsely reusing its manual plane34 or synthetic polarization experiment. All plots have paired numeric NPZ and PNG/SVG/PDF exports in P/figures plus figure_inventory.json; none reside as temporary screenshots/executed notebooks inside the task output directory.

An earlier default-kernel execution0/122.121s is retained under superseded_default_kernel: python3 resolved Anaconda and Agg produced figure text representations rather than embedded PNG. It is not used as cleaned shared-environment completion evidence. The corrected pinned/inline complete execution above supersedes it. This records the error and concrete fix, not an environment compatibility assumption.

## Plan and interface semantic review

Both plans were written from the actual extracted method. Review covered Gaussian cross-term/amplitude/covariance units,background3percent,transpose/projected low-rank convention,all source fractions including oxygen-B.30,sequential neighbor identities,free-parameter ordering,source LM/retries/rotation behavior,5width rendering boundaries,complete3-sublattice calling flow and all output quantities. Design signatures/defaults/shapes/dtypes/returns match the source; selected standard,Notebook comparison paths and private original scripts remain distinct. This semantic review supplements signature checking and independent analytic tests; it does not claim unchanged historical source behavior where the paper/current reference differ.

## Completion gate disposition

Passed with records above: primary paper/exact revisions/data hashes/traceability; full original/repeat and fresh local runs; independently frozen stage parity and known physics; constants/units/conventions; complete declared output and fresh Judge; adversarial evaluator contracts; self-contained public input and no truth/calibration leak; README/output metadata; minimal cleaned CPU runtime without domain imports; local mathematical extraction; initial35-case fullpytest; active complete Notebook/native0 and12actually opened figure groups; identified result/metric provenance; semantic plan/source review; template/NOTICE/private-archive separation. Final independent file/call-role audit,dual formula render and logged parent test rerun remain **unverified until their actual records are appended**, so this working report alone does not mark the task accepted.


## Final actual verification and private acceptance

All previously pending final gates are now passed: complete35-case parent test rerun native0/69.634383800s (B8/qa/perovskite_dual_contrast_atomic_columns_parent_tests.json and stdout); all26 formulas passed KaTeX and MathJax with zero failures; final static/public-input/call-role/file audit errors[]; independent foreign-cwd Judge native0. Parent actually opened all12 final PNG groups individually, including the corrected oxygen-direction titles; all plot-export hashes and all12 embedded PNG outputs were verified. Full original and fresh measurements remain distinct. Full Notebook125.8178505s and display-only5.8885363s executions retain different identified SHA values; canonical differs in exactly one title-only source/output cell, with all other cells unchanged. No reconstruction or scientific array was replaced by a preview. Independent peer /root/batch6_mech_bio_med read the published PDF Eq1 and the original/local code/plans/Judge, confirmed covariance/sign/units/fractions/call flow and non-duplication, and found no blocking issue (P/independent_semantic_review.json). All16 specification gates are passed with the cited evidence and retained limitations. This is private staging acceptance; the20-item batch is not yet released.
