# Provenance and validation evidence

Status: all mandatory gates passed for the selected planar benchmark; final execution evidence is recorded below. No physical ground truth is available.

## Bound scientific problem and sources

Physics: planar gamma emission response from a coded tungsten aperture. Selected published stage: direct planar MURA decoding at a known50mm source plane, retaining the full256虏 measured exposure; Accorsi sparse decoding is the active comparison. This does not claim to reproduce the later3D source-localization fits or optional3D-MLEM experiment. No original notebook was supplied in the selected reconstruction repository, so no original notebook comparison groups were removed. A single fixed-plane map is the benchmark unknown; parameter sweeps reconstruct alternative planes as a sensitivity experiment, not additional submitted samples.

- Method repository: https://github.com/tomeiss/cai-reconstruction , exact remotely verified HEAD `0dcd63a70636395b25f85d28571237c502faf06f`, retrieved2026-10-22.
- Data repository: https://github.com/tomeiss/3d_localization_with_cai , exact remotely verified HEAD `6bb2f59f435e41db71665b366acbad82d029a712`.
- Exact source observation: `detector_images/experimental/x00y00z50_Minipix_Mask_Exp15min.tif`. Full uint32 image, no sample thinning, cropping or changed acquisition. Pixel energy is integrated keV; the PMB2024 experimental image is bound, not its companion MonteCarlo image.
- Papers: Mei脽ner etal., PMB69(2024)165004, DOI10.1088/1361-6560/ad6370, version of record https://publikationen.bibliothek.kit.edu/1000173531/154426760 ; camera/preprocessing EJNMMI Physics11(2024)30 DOI10.1186/s40658-024-00631-5, https://pmc.ncbi.nlm.nih.gov/articles/PMC11266340/ .
- Code/data license: GPL3; original full license preserved in task/LICENSE. Authors' notices retained in local adapted numerical modules. Public NPZ mask is instrument calibration, independent of the unknown source.
- New inverse-problem deduplication: no existing task reconstructs gamma source response from a coded nuclear detector shadow. Existing lensless visible imaging, PET annihilation line tomography and SNSPD photon-return depth use different acquisition and physical unknown conventions; this is not a new sample/method of any of those tasks.

## Maintainer-only archive index

External group archive relative to repository sibling: `../imaging-101-maintainer/research_batch6_20261022/groups/physics_materials/`.

| Artifact | Role |
|---|---|
| cai-reconstruction.zip; cai-reconstruction/ | Complete unchanged original source and license |
| cai-reconstruction_commit.json; cai-reconstruction_repo.json | Live remote exact revision metadata |
| 3d_localization_with_cai_commit.json; 3d_localization_with_cai_tree.json | Data revision and inventory |
| cai-reconstruction/experimental_x00y00z50.tif | Exact source observation |
| gamma_oracle.py; gamma_original_run.log; gamma_original_run.json | Private capture runner, actual log, environment/parameters/timing/hashes |
| gamma_upstream.npz | Independent frozen original measurements |
| gamma_red_tests.log; gamma_core_tests.log | Test-first collection failure and initial staged checks; early cwd/temp-directory errors are explicitly not passing validation |
| gamma_final_tests.log; gamma_nbconvert.log | Final validation evidence after execution |
| gamma_figures/; gamma_experiments.npz; gamma_experiments.json | Actual numerical experiments andPNG/SVG/PDF plot source |

## Original code execution before local extraction

Command: private `oracle-env/Scripts/python.exe gamma_oracle.py`, cwd original checkout. Exit0. Actual runtime 0.433727900s. Python3.11.11, NumPy2.4.6, SciPy1.17.1, CPU, oneOMP/OpenBLAS/MKL thread. TIFF decoding uses private tifffile2023.4.12. General numerical primitives resolve from the existing shared research runtime; original source is only on the oracle process path, not installed in cleaned runtime. No shared packages upgraded.

Original source files are unchanged. Import-only compatibility: an empty tensorflow module supplies only its decorator binding. That decorator replaces the UNUSED optional TensorFlow convolution with a function that immediately raises if called. No TF operation, interpolation or reconstruction is supplied by the shim. Original `main` runs with `mlem3d=False,mura_decoding=True,accorsi_decoding=True`, so it cannot enter any TF numerical branch. `np.in1d=np.isin` restores the NumPy2-removed legacy spelling during unused rank5 dummy-mask construction; the constructor's mask is immediately replaced by the supplied original mask. This is disclosed as an import/API compatibility execution, not an original TensorFlow environment reproduction. A Python tracing hook copies locals at original main return; it does not alter numeric code or results. All exact parameters and captured warnings are in gamma_original_run.json. Runtime warnings count: 0.

Raw detector SHA256 `498a7eac8b375f0f0bcaae84d47802550046227992aa423ffd7802e13cdb5fb0`. Public NPZ SHA256 `089767a22273e5eb4239627f2f9a56d92faf56877a228789cd1d7616898f52e0`. Full source SHA256 inventory is preserved in gamma_original_run.json; original reco.py `8e528c65944642bd30c775dff215655b6e02c65295a48ab19d71db94077cdfb2`. Conversion checks compare NPZ arrays exactly to captured uint32 raw and float32 mask; no quantization error.

## Equation/stage traceability and fixed acceptance

| Mathematical stage | Upstream file/function | Local function | Validation |
|---|---|---|---|
| Robust detector correction | reco.py/main percentile,median,Gaussian block | preprocessing.preprocess | original output_preprocessed exact |
| Signed NTHT decoding period | reco.py/convert_mask_to_decoding_pattern | physics_model.decoding_pattern | original output_decoder exact |
| Magnification1+b/z and126-pixel period | reco.py/main MURA branch | projected_size,nearest_resize,central_crop | frozen original crop and full output; analytic grid test |
| Circular sumFFT inverse | reco.py/main r_tobi expression | cyclic_convolve,reconstruct | independent direct sum,adjoint,flux,shift and frozen original output |
| Sparse/delta comparison | reco.py/main Accorsi branch | solvers.accorsi | original output_accorsi |
| NCC/NRMSE | independent benchmark convention | Judge.score andvisualization.metrics independently | analytic [1,0]vs[1,2] case |

Tolerances were fixed before local results: preprocessing/decoder exact equality; float32 MURA rtol2e-6/atol.05keV and Accorsi rtol2e-6/atol.02keV; analytic float64 convolutions rtol1e-13/atol1e-12. These are stage tolerances, not altered scientific thresholds. Constructor and interpolation roles were audited: core decoding math is local; scipy supplies general filtering/interpolation; NumPy suppliesFFT. No domain solver or upstream imports occur in source, Judge, tests or notebook.

Original results reside in notebooks/reference_outputs/upstream.npz, keys mura/accorsi, consumed only for explicitly labelled notebook original comparisons. Fixtures in test/fixtures/upstream.npz contain output_preprocessed,output_decoder,output_crop,output_mura,output_accorsi and are consumed by test/comparing_test/test_upstream.py. Baseline_reference.npz/reconstruction is the independently captured originalMURA response and is the hidden scoring reference; it is never called truth. The fresh result is computed by src.main and separately identified in notebook execution and gamma_experiments records. There are no cached paper metric values masquerading as this acquisition's results.

The method-reference baseline self-comparison gives NCC1/NRMSE0, so nonzero float32 limits are required. Fixed boundariesNCC>=.999999,NRMSE<=5e-6 are a numerical reproducibility envelope, validated by reruns, precision sensitivity and poor-map rejection. No arbitrary forward consistency is imposed on signed decoding sidelobes; that would incorrectly treat the response image as an exact nonnegative physical activity. Independent physics tests validate the nominal operator but do not establish true scene accuracy. The measured source displacement can include the camera alignment systematics discussed in the publication, so no hidden source coordinate is fitted or treated as a calibration.

## Consistency and completion gates

Mathematical/source review checked FFT signs and backward normalization,NTHT zero filling and central sign,nearest-neighbour pixel centres,crop truncation,keV scale,float32 intermediates,coordinate sign,periodicity and the explicit absence of180degree rotation. Plan/design signatures/defaults are checked against actual source and documented tests verify matching acquisition constants. README has exactly the publicNPZ keys, units and output contract. NoGitHub URLs, truth keys, hidden seeds or target-derived calibration occur in the solver-visible allowlist. All existing tasks remain untouched.

Final gate results, final metrics, notebook runtime, visual inspection and inventory follow in the execution addendum. Mandatory unresolved items must not be interpreted as completion.

## Final execution addendum

`python -m src.main` wrote the exact declared float32(1,126,126) output. The standalone Judge passed the fresh image. `python -m pytest test/ -v --basetemp=<private group archive>/gamma-pytest-review` passed20 tests in the actual shared runtime onCPU. The explicit private pytest temporary directory resolves an existingWindows access-denied defaultTemp directory; this is a test-runner location issue, not a reconstruction exception. Full command output:gamma_final_tests.log. No shared installation or upgrade was made.

The complete notebook was executed with`python -m jupyter nbconvert --to notebook --execute ... --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101-research`. The isolated kernel JSON points explicitly to research-runtime/Scripts/python.exe. The actual full reconstruction paths completed in milliseconds;180s safely covers plotting/kernel startup. The recorded freshMURA time was0.007094299988239072s andAccorsi0.007726399999228306s. The canonical notebook contains all executed code-cell outputs and zero errors. The temporary output/gamma_executed.ipynb was removed after transferring its executed contents to the canonical notebook. NonfatalWindowsZMQ selector-thread/kernelTCP notices are in the nbconvert log; they did not prevent successful completion. All6 figures were exported asPNG/SVG/PDF, and source numerical arrays remain privately archived in gamma_experiments.npz. Actual run measurements, not historical paper values, are in gamma_experiments.json.

Visual inspection checked the six exported scientific panels: native/corrected detector and removed energy; full/signed/projected mask; original/fresh/residual map; MURA/Accorsi plus normalized shape profiles; actual plane sweep; and active preprocessing/smoothing ablations. Titles, independent colorbars, units, axes and legends are readable without overlap. Fresh and original have explicit labels. The gamma source is spatially resolved above signed sidelobes; no zero placeholder or archived result was used as a fresh method. The comparison sampling grids and physical limitations are explained in the notebook.

Independent repeated original output is bitwise equal to the frozen first fixture. Two cleaned reruns are bitwise equal. Float64FFT sensitivity against the standardfloat32 path gives NCC=0.9999999999999775, dynamic-rangeNRMSE=1.1959172908419824e-08, maximum absolute response difference=0.0078125keV. This supports the fixed nonzero5e-6NRMSE tolerance, rather than using a self-error-zero boundary. FreshMURA agrees exactly with the original array; AccorsiNRMSE1.2856621447527002e-7 remains within its prespecified stage tolerance. Analytic independent operator/metric tests and deliberately poor/malformed predictions passed. No threshold was relaxed.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, exact remote revisions, hashes and equation mapping | passed | Source records and traceability table above |
| Original and local run on exact bundled observations | passed | Original-run logs/capture;20-test final run; freshCLI |
| Frozen upstream parity and independent physics | passed | Exact detector/decoder; MURA/Accorsi parity; direct sum,adjoint,flux,shift tests |
| Constants, units, precision and coordinates | passed | Exact raw/mask equality; geometry tests; README/source review |
| Standard NPZ and independent Judge | passed | Fresh output and output/judge_report.json |
| Judge quality rejection and malformed/evaluator failures | passed | Unit contract cases and report-write failure |
| Solver-visible sufficiency and no privileged inputs/URLs | passed | README tables/model/output and allowlist scan |
| Shared runtime and local mathematics | passed | Minimal NumPy/SciPy/Matplotlib; import/call-role audit; CPU execution |
| Complete tests | passed |20 passed, gamma_final_tests.log |
| Active six-chapter notebook and comparisons | passed | Executed canonical notebook;6 inspected figures |
| Provenance/metric/threshold consistency | passed | Original/fresh distinct; precision/repeat experiments; analytic cases |
| Source/plans/README semantic/signature consistency | passed | Equation and actual-flow review recorded above |
| Final inventory, private archive and license | passed | Required task tree only;GPL3 preserved; temporary notebook removed |

The supplied observation is reused, so no synthetic-data generator is required or shipped. No physical ground_truth.npz is invented. The small fixtures and separately identified upstream comparison arrays have documented numerical-test/notebook consumers. Public data contains no extra clean observation or target-derived calibration.

## Parent visual review update

The signed removed-energy and signed fresh-minus-original panels now use symmetric RdBu_r diverging scales with neutral zero. Original/freshMURA share the same intensity range. All numerical experiments were reexecuted via nbconvert after this display-only change; canonical executed outputs and six PNG/SVG/PDF exports were refreshed. The actual kernel prints sys.executable=D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe and NumPy2.4.6. Fresh timings and numeric source remain in the refreshed gamma_experiments.json/.npz. The source version-of-record PDF is privately archived as gamma_PMB2024_version_of_record.pdf, URL https://publikationen.bibliothek.kit.edu/1000173531/154426760, SHA256 b865b3bc1048eaef9e28453c9da95e304f688f395c75d0d8a8aa7ec22116c629;1416845bytes. No PDF is in the final task.
