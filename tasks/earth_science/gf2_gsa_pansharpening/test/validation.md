# Validation and provenance: gf2_gsa_pansharpening

All mandatory cleaning gates below passed on CPU. Method agreement is separated from physical accuracy: these are real observations with no independently known inverse target. No cached paper number is presented as a measurement from this run.

## Sources, exact revisions and licenses

- Repository: https://github.com/codegaj/py_pansharpening
- Exact remotely verified commit: `a1bf9ec0d4fa5694615ac6e9b7ee61a121fc5145` (GitHub commits/tree API queried2026-10-20; not merely a local tracking revision). The downloaded branch ZIP was checked against that commit by recomputing every local git-blob SHA1: 31files checked,0mismatches. See private verified_revisions.json and snapshot_tree_verification.json.
- Paper/version: Aiazzi,Baronti & Selva2007, Improving Component Substitution Pansharpening Through Multivariate Regression of MS+Pan Data, IEEE TGRS45(10),3230–3239, DOI10.1109/TGRS.2007.901007. Archived11-page author manuscript plus IEEE copyright cover from CNR. The source-code implementation and its disclosed conventions are validated, not the paper's different IKONOS/QuickBird images.
- Exact paper URL: https://iris.cnr.it/retrieve/9c4b9c98-49cf-4788-b9a1-a3d05025815b/prod_22743-doc_199666.pdf?download=1
- Archived paper: `aiazzi_2007_author_copy.pdf`, SHA256 `99a7d7254ea0008e1d975c782d018ceaa3c6aa7f30d8f2d6b01b916a99ac0e3e`. Original complete source and paper stay outside the final task.
- Data: Native author-supplied GF2_BJ_mss.npy(1000,1000,4)uint16 and GF2_BJ_pan.npy(4000,4000)uint16. Fixed native registered ROI MS[200:264,200:264],PAN[800:1056,800:1056]. Per-observed-ROI,per-band float32 min-max normalization matches the source demo. No Wald degradation or synthesized PAN is used as the benchmark input. Both native arrays are supplied under the repository MIT distribution with no separate restrictive data notice; preserve original notice. Scene name indicates Beijing; date/geolocation/physical spacing/bandpass/nativePSFs are not documented and are not invented.
- Source license: MIT repository and GSA/utilities; retained SFIM per-file Apache-2.0 notice and complete license also preserved. Task NOTICE preserves source copyright and adaptation information.

## Fixed method, observations and oracle execution

Private archive root: `imaging-101-maintainer/research_batch2_20261020/astro_earth/`, a sibling maintainer area outside this release task. Selected source snapshot: `py_pansharpening-master/`. Capture script: `capture_gsa.py`. It imports only the unmodified private original source, never cleaned src. The cleaned solver/tests/notebook never import that archive. Normal tests consume frozen arrays only.

The first joint capture ran original Lenspack bin2d/ks93, original SFIM and original demregpy dn2dem on exactly the public observations before writing local mathematics. It exited0, measured1.929347800090909seconds total, and saved stage/final arrays. AIA is called48spectra per row to select the original serial path rather than its automatic multiprocessing path at>=200pixels. The original np.int compatibility alias for the old pansharpening interpolation changes only a removed NumPy integer-constructor spelling, not mathematics. Source itself remains unchanged.

GSA was then independently run on exactly the same bound native arrays before its extraction; gsa_acceptance_before_local.json was frozen before select_gsa.py. The later saved capture_gsa.py is a fully reproduced original run with a non-mutating sys.setprofile return-frame observer to capture alpha, gains and intensity. It repeats the earlier final array exactly, takes0.046198399970307946seconds, exits0, and archives gsa_oracle_metadata.json/gsa_oracle_run.log. This later capture is identified as a replay, not falsely dated before the earlier extraction.

Oracle environment: private oracle-env, Windows10.0.26200, CPython3.11.11, NumPy2.4.6, SciPy1.17.1, Astropy8.0.2 FITS reader, threadpoolctl3.7.0 and tqdm4.70.1; OpenCV wheel5.0.0.93 (cv2 reports5.0.0) is used only for original pansharpening execution. Full versions are in oracle_environment_pip_freeze.txt. No GPU or trained weights are required. OMP/OPENBLAS/MKL threads are1. BLAS/backend details and validated cleaned Python/NumPy/SciPy/Matplotlib versions are archived in cleaned_environment.txt. Cleaned runtime uses only its pinned requirements subset; Astropy/OpenCV and upstream domain source are absent from its imports.

Warnings retained honestly: source demregpy snapshot has no generated package-version module and emits a metadata warning, but executes its complete numerical path successfully; real AIA FITS headers contain a floating-image BLANK warning, which Astropy ignores as documented. Windows nbconvert/ZeroMQ emits an event-loop fallback/kernel transport diagnostic. These are recorded in logs; no numeric failure, suppressed solver warning, failed cell or cached reconstruction substitutes for execution. Seeds are absent from real acquisition and deterministic inverse solves; only private notebook shape-orientation diagnostics use the explicit20261020seed.

## Traceability from paper stages through executable code

| Required equation / stage | Original function / source | Local function | Independent validation consumer |
| --- | --- | --- | --- |
| Paper spatial band expansion | utils.py::upsample_interp23 | physics_model.interpolate23 | test_upstream_interpolation_and_quantized_pipeline; test_interpolation_dc_and_phase_insertion |
| PaperSecIV step1 native PAN spatial matching | methods/GSA.py::GSA cv2.resize effective linear operation | solvers.downsample_pan | test_original_pan_resampling_and_sfim_failure_comparison |
| PaperEq3 regression and Eq4 high-resolution synthetic intensity | methods/GSA.py::estimation_alpha,GSA alpha,I,I0 | solvers.regression_intensity | test_frozen_upstream_regression_and_covariance_stages; independent normal-equation stationarity |
| PaperEq2 intensity residual and TableI covariance gains | methods/GSA.py::GSA delta,g,V_hat | solvers.detail_gains,solve | test_frozen_upstream_regression_and_covariance_stages; test_covariance_sample_population_factor_independently |
| Mean preservation, clipping and uint8 quantization | methods/GSA.py::GSA final I_GSA/uint8 return | solvers.solve,reconstruct | test_upstream_interpolation_and_quantized_pipeline; test_fusion_bounds_and_zero_detail_ablation |
| Screened SFIM comparison / physical-radiometric failure investigation | methods/SFIM.py::SFIM | solvers.solve_sfim | test_original_pan_resampling_and_sfim_failure_comparison; active failure plate and metrics |

Stage tolerances were frozen as float64 rtol1e-10/atol1e-12, or relative rtol1e-10 with zero absolute tolerance for physical DEM scales. Original uint8 full-pipeline outputs require exact equality; the original float32 GSA resampler also matches exactly. Independent analytic cases use scale-appropriate tighter tolerances specified directly in tests. A zero-detail GSA recentering comparison permits1e-12 absolute error, matching the frozen float64 floor and avoiding a sub-ULP equality demand. Fixtures never come from the cleaned function being tested.

## Ambiguities, physical checks and baseline quality

SFIM was initially independently run and locally extracted, then rejected as the main baseline after full spatial/radiometric QA. Its histogram-matched PAN has minima[-0.24045,-0.19344,-0.27342,-0.12915]; its five-pixel low-pass is nonpositive at[1721,911,2178,0] pixels by band. Near-zero divisors produce actual source chromatic speckles. Agreement between two copies therefore was insufficient for baseline quality. An independent original GSA run was frozen before selecting/extracting GSA; its native-image spatial contrast is much cleaner, while SFIM remains an actively computed labeled failure comparison. Only one pansharpening task ships.

GSA's source cv2.resize(image_hr,(n,m),cv2.INTER_CUBIC) supplies the constant as dst (third positional argument), so its effective interpolation is default linear. Separate unmodified OpenCV calls confirm equality with linear and max difference0.06815147399902344 with explicit cubic. Local half-pixel difference-form lerp matches original float32 values exactly without an OpenCV runtime dependency. No silent cubic correction is made. The source uses sample covariance(ddof1) divided by population variance(ddof0), a factorN/(N-1) relative to a matched convention; the independent analytic test explicitly verifies that factor. Zero-variance native observations or synthetic intensity are undefined; the cleaned validation guard raises a descriptive error rather than propagating original divide/cast artifacts. On the bound nondegenerate scene the original/local output remains bit-exact.

The screened SFIM float64-input alternative changes201of262144 quantized values by1LSB, NRMSE1.1019833463494301e-4. This changes the input/computation precision convention and is NOT an identical-observation repeat error or justification for changing the frozen1e-6 boundary. Identical-convention original GSA repeats and cleaned GSA are bit-exact. Native PSFs/spectral bandpasses are unavailable; box-degradation and PAN-gradient measurements are disclosed surrogate diagnostics and are not promoted to a calibrated observation-likelihood Judge gate. No unsupported date, ground sampling or color-band wavelengths are invented.

## Data, calibration and reference hashes

Public raw_data.npz SHA256: `0d15c153952ba302ff5fe3e32f166ddf570489a4951b943c12af29c6de16f498`. No hidden inverse target, privileged spectral image, synthetic object seed or target-derived calibration is bundled. All arrays are NPZ and batch-first. Source raw file hashes were captured before normalization/packaging and are listed below.

| Source acquisition file | SHA256 |
| --- | --- |
| py_pansharpening-master\images\GF2_BJ_mss.npy | `db5d9790c83b052320e9926fe0903fd7fbfd359f151b723cb85208d3d383dfeb` |
| py_pansharpening-master\images\GF2_BJ_pan.npy | `cf6f5514c02f2add5e4fca1d5592305def331a934245aebc5d748af8769d3754` |

Hidden reference: `evaluation/baseline_reference.npz` is the original standard-method reconstruction on the bound observations, not physical truth. No ground_truth.npz is invented. If an actual independently validated truth is later provided, the standalone Judge gives it explicit priority.

| Asset | SHA256 | Consumer and provenance |
| --- | --- | --- |
| evaluation\baseline_reference.npz | `c679dd6741ac240cc60fd35f2aec9c99a59a315727b2f6b2558707f1b0a6bc28` | Independent original final result; Judge and original/fresh spatial notebook comparison |
| test\fixtures\sfim_upstream.npz | `4bcd7ca1ca7e4a99ac21571411e3d9dd77661922c7c02b95bfad3a4198f7b4b4` | Original source stage/final fixture; comparing tests |
| test\fixtures\upstream.npz | `d9062313f0d8d11b415110e5bc1350d787e951d7a6f4c396eec83165bc832021` | Original source stage/final fixture; comparing tests |
| notebooks\reference_outputs\fresh_summary.npz | `06af29314fbed687444ae2b121af2f7b68917185f1cf624cc8c8e59f31b6a972` | Fresh actively computed scalar summary; active notebook table, never a reconstruction cache |
| notebooks\reference_outputs\upstream_summary.npz | `06af29314fbed687444ae2b121af2f7b68917185f1cf624cc8c8e59f31b6a972` | Independent original-run scalar summary; active notebook original/fresh summary table |

Every fixture key uses output_* for independently expected outputs. No cleaned result is relabeled as an upstream fixture. The stage/final fixtures and final Judge reference have separate documented consumers; redundant notebook intermediate copies were removed and replaced by small scalar summaries. Complete original capture arrays remain privately indexed.

## Metrics and result chronology

Published/historical reports are context only: none of the paper's numeric tables or cached example reports are used as fresh measurements. Original-code outputs are the frozen reference arrays above. Fresh cleaned measurements come from the current CLI/notebook execution below, using the same data hash, selected method/defaults and declared units. `notebooks/reference_outputs/upstream_summary.npz` and `fresh_summary.npz` separately identify their scalar measurements. Notebook plot arrays and full-precision experimental metrics are private gallery source artifacts.

NCC is cosine similarity between flattened real arrays, without mean subtraction. NRMSE is sqrt(mean((prediction-reference)**2))/(max(reference)-min(reference)), without hidden alignment, crop, magnitude conversion or fitted scale. Both required output modes for KiDS are scored independently. Zero estimate is a valid poor result with NCC0; zero-norm/constant hidden references are evaluator errors. Invalid dtype, shape, key, nonfinite/object content and inaccessible assets return error/status2 with empty metrics. Valid below-boundary predictions return fail/status1. Passing result and report return0; report-write failure returns2 with stderr diagnostic. No solver is imported or executed by the Judge.

Primary boundaries remain exactly the pre-extraction acceptance: NCC>=0.999999, NRMSE<=1e-6. Independent same-convention replays are deterministic, so the finite nonzero1e-6 tolerance is conservative and does not apply the meaningless1.1*0 self-error rule. Disparate precision experiments do not justify relaxing it. Any supplemental forward gate and its later chronology are stated above. The pipeline, physics and parity gates must pass separately from Judge agreement.

Fresh final Judge measurement, full stored precision:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_ncc": 1.0,
    "reconstruction_nrmse": 0.0
  },
  "thresholds": {
    "reconstruction_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "reconstruction_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    }
  },
  "errors": []
}
```

## Final executed commands and evidence

| Command | Exit | Wall seconds | Private log |
| --- | ---: | ---: | --- |
| `python -m src.main` | 0 | 0.821759 | gf2_gsa_pansharpening_main.log |
| `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.163279 | gf2_gsa_pansharpening_judge.log |
| `python -m pytest test/ -v --basetemp <unique private directory>` | 0 | 3.090250 | gf2_gsa_pansharpening_pytest.log |
| `jupyter nbconvert --to notebook --execute notebooks/gf2_gsa_pansharpening.ipynb --output gf2_gsa_pansharpening_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180` | 0 | 10.008596 | gf2_gsa_pansharpening_nbconvert.log |

Pytest summary: 26 passed in 2.76s ==============================. Notebook produced4actual PNG figure displays and0error outputs. The180-second per-cell timeout provides more than10times measured full-notebook execution headroom; it is not evidence that every published large experiment runs in seconds. The private imaging-research kernelspec points explicitly to the validated research-runtime Python, not the default Anaconda kernel. Formal notebook retains outputs of the successful active execution; the temporary executed output copy is archived outside the task.

## Notebook comparisons and visual QA

Four figures are actively generated in every notebook: native observations/main reconstruction, method mechanism, actual sensitivity/ablation experiments and uncertainty or observational diagnostics. The source comparisons relevant to this selected acquisition are preserved; the new task notebook extends a small author example or builds an appropriately scoped review rather than pretending to reproduce every unrelated paper scene or training experiment. Standard result, interpolation/baseline, ablations, tables and original-source comparison are actually computed or explicitly labeled as independently captured original data.

Complete rendered PNGs were visually inspected, with PDF/SVG exports from the same actual Figure objects. Figures use readable DejaVu Sans labels, physical units or explicit normalized units, consistent direct-comparison scales, cividis/viridis/magma sequential palettes and a signed diverging palette for positive/negative fields. Fixed display transforms are used for RGB comparisons. Original/fresh plates, error bands and null diagnostics do not substitute cached results for fresh computation. No clipping or overlapping scientific label remains after the shortened AIA integrated-emission title and corrected physical axis labels.

| QA issue | Resolution | Status |
| --- | --- | --- |
| AIA positive helioprojective longitude mislabeled east in initial draft | Corrected to solar west in README and every final image axis | passed |
| GF2 geographic orientation unknown | Removed east/north labels; use row/column PAN pixels | passed |
| Long AIA integrated-emission title | Shortened after viewing complete PNG; final plate checked | passed |
| SFIM chromatic speckles | Investigated independent low-pass sign; chose original GSA and retained actual failure comparison | passed |
| Unsupported truth/physical-accuracy claims | All references identified as algorithm results; surrogate/noise limitations explicit | passed |

Private gallery: `gallery/gf2_gsa_pansharpening/`, four PNGs, four vector PDFs, four SVGs, `figure_source_arrays.npz` and `experiment_metrics.json`. These trace every exported plot and table to exact numeric arrays. No gallery exports or temporary screenshots are retained in the task package.

## Mandatory gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper, exact remote revision, data hashes and equation/function mapping | passed | Source table, immutable blob-tree verification and traceability table below |
| Original and cleaned methods on exactly bound observations | passed | Private capture scripts/oracle logs; final main and independent frozen parity tests |
| Independent upstream fixtures and applicable physics | passed | Frozen original arrays plus analytic spin-2/adjoint, response/SVD or filter/regression checks |
| Constants, units, precision, coordinates and conventions | passed | README/approach and tests; documented original-code ambiguities below |
| Standard CLI writes exact NPZ and fresh Judge passes | passed | Final command table and output/judge_report.json |
| Judge malformed, poor-quality, missing assets and report failures | passed | Unit cases cover missing/key/shape/dtype/NaN/Inf/complex/object, zero-quality, truth priority, unavailable assets, invalid metric boundaries and report-write failure |
| Solver-visible allowlist sufficient and no hidden-answer leak | passed | README/input/environment review; independent calibration only; no target-derived inputs |
| Public GitHub URL/hidden path/seed leak and metadata JSON audit | passed | No public GitHub URL; acquisition metadata inline; no standalone metadata JSON |
| Minimal cleaned runtime without upstream domain packages | passed | Pinned NumPy/SciPy/Matplotlib; actual final shared-runtime CLI/tests/notebook; AST import audit |
| Core executable mathematics and symbol/call-role checks | passed | Local operators/inverses; final_verification.json inventories all imports/calls; zero forbidden-domain imports |
| Unit, Judge, parity, physics and complete pipeline tests | passed | Final pytest -v log, all cases passed |
| Active notebook and required scientific comparisons | passed | Four real computed figures, fresh reconstruction, independent original comparison, actual variants and scalar table; nbconvert success |
| Metric definitions/result provenance/thresholds | passed | Exact code/reference origins below; original/fresh summaries separated; no paper numbers used as run evidence |
| README and both plans match actual semantics and signatures | passed | Manual mathematical review plus AST signature refresh, explicit units/axes/scaling/defaults/flow |
| Final file inventory, licenses and cleanup | passed | Inventory below; original source/PDFs/logs outside task; executed temporary notebook archived privately; notices retained |

No mandatory gate is failed or unverified. Genuine scientific limitations, source-code discrepancies and unsupported physical quantities are disclosed above instead of being hidden by threshold changes. The legacy evaluation_harness is not claimed compatible with the NPZ task contract.

## Maintainer-only archive index

- Immutable-source provenance: verified_revisions.json, snapshot_tree_verification.json, py_pansharpening-master/, corresponding downloaded ZIPs and source copyright notices.
- Paper: aiazzi_2007_author_copy.pdf, extracted text where available; source URL/version/hash above. The KS93 scan has image-only text, and was reviewed through its published scan/source equations rather than claiming successful text extraction.
- Original execution: capture_oracles.py, oracle_run.log, oracle_exit_status.json, oracle_metadata.json; selected GSA adds capture_gsa.py,gsa_oracle_metadata.json,gsa_oracle_run.log and gf2_method_investigation.json.
- Acceptance chronology: acceptance_before_local.json, gsa_acceptance_before_local.json, initial_local_parity.json; independent_rerun_tolerance.json and independent_rerun_arrays.npz identify different precision conventions separately.
- Environment: oracle-env/, oracle_environment_pip_freeze.txt, cleaned_environment.txt; no upstream-only dependency enters public requirements.
- Final execution: final_verification.json inventories command results, all source/generator/Judge/test import and call roles, public URL scan, notebook output counts and exact metrics; each command has its own log.
- Review exports: gallery/<task>/ numeric sources and PNG/PDF/SVG; executed_notebooks/<task>_executed.ipynb; visual-contract.md and qa-ledger.md.
- Rejected/screened candidates: private SFIM candidate notebook/source/gallery; PRISMA rejected because ASI prohibits redistribution; unrelated weak-lensing simulation/SMPy source was screened but never bundled.

## Final task inventory

```text
NOTICE
README.md
data/raw_data.npz
evaluation/baseline_reference.npz
evaluation/judge.py
evaluation/metrics.json
notebooks/gf2_gsa_pansharpening.ipynb
notebooks/reference_outputs/fresh_summary.npz
notebooks/reference_outputs/upstream_summary.npz
output/judge_report.json
output/reconstruction.npz
plan/approach.md
plan/design.md
requirements.txt
src/__init__.py
src/main.py
src/physics_model.py
src/preprocessing.py
src/solvers.py
src/visualization.py
test/comparing_test/test_pipeline.py
test/comparing_test/test_upstream_physics.py
test/fixtures/sfim_upstream.npz
test/fixtures/upstream.npz
test/unit_test/test_contract.py
test/validation.md
```

All tests and numeric fixtures reside under test/. There are no root wrappers, duplicate tests/report directories, source PDFs, complete upstream checkouts, download archives, metadata JSON, simulator seeds, trained weights or unconsumed array files in this task.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 4 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 26 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. No high-resolution spectral truth or calibrated native sensor PSF; source interpolation convention explicitly preserved.
