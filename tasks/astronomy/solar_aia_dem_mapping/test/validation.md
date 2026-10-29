# Validation and provenance: solar_aia_dem_mapping

All mandatory cleaning gates below passed on CPU. Method agreement is separated from physical accuracy: these are real observations with no independently known inverse target. No cached paper number is presented as a measurement from this run.

## Sources, exact revisions and licenses

- Repository: https://github.com/alasdairwilson/demregpy
- Exact remotely verified commit: `60cbb94eaacb052d5fdde9a5c97642dcbe8bec0c` (GitHub commits/tree API queried2026-10-20; not merely a local tracking revision). The downloaded branch ZIP was checked against that commit by recomputing every local git-blob SHA1: 168files checked,0mismatches. See private verified_revisions.json and snapshot_tree_verification.json.
- Paper/version: Hannah & Kontar2012, arXiv1201.2642v1 (2012-01-12),15pages, and published A&A539,A146, DOI10.1051/0004-6361/201117576. This package reproduces the validated modern Python port, not a newly claimed execution of the original IDL paper code.
- Exact paper URL: https://arxiv.org/pdf/1201.2642v1
- Archived paper: `hannah_kontar_2012_arxiv_v1.pdf`, SHA256 `db8431b746cde64393d930e2d82f1e47b4159d591e3047e113e78c3155b72424`. Original complete source and paper stay outside the final task.
- Data: Real NASA/SDO/AIA2014-01-01 synoptic level1.5 observations distributed in demregpy tests/data/aia. Each channel uses its archived FITS exposure verbatim. The fixed48x48 active-region ROI is rows415:463,columns424:472. Its center was selected from observed171brightness alone; no thermal target was consulted. Nominal independent aia_tresp_en.dat response is separate acquisition calibration. NASA/SDO and AIA science-team attribution is preserved. No synthetic observations are generated.
- Source license: MIT; underlying observations courtesy NASA/SDO/AIA. Task NOTICE preserves source copyright and adaptation information.

## Fixed method, observations and oracle execution

Private archive root: `imaging-101-maintainer/research_batch2_20261020/astro_earth/`, a sibling maintainer area outside this release task. Selected source snapshot: `demregpy-main/`. Capture script: `capture_oracles.py`. It imports only the unmodified private original source, never cleaned src. The cleaned solver/tests/notebook never import that archive. Normal tests consume frozen arrays only.

The first joint capture ran original Lenspack bin2d/ks93, original SFIM and original demregpy dn2dem on exactly the public observations before writing local mathematics. It exited0, measured1.929347800090909seconds total, and saved stage/final arrays. AIA is called48spectra per row to select the original serial path rather than its automatic multiprocessing path at>=200pixels. The original np.int compatibility alias for the old pansharpening interpolation changes only a removed NumPy integer-constructor spelling, not mathematics. Source itself remains unchanged.

The independent source fixtures precede local implementation; acceptance_before_local.json records stage tolerances and reference-agreement boundaries before the first cleaned measurement.

Oracle environment: private oracle-env, Windows10.0.26200, CPython3.11.11, NumPy2.4.6, SciPy1.17.1, Astropy8.0.2 FITS reader, threadpoolctl3.7.0 and tqdm4.70.1; OpenCV wheel5.0.0.93 (cv2 reports5.0.0) is used only for original pansharpening execution. Full versions are in oracle_environment_pip_freeze.txt. No GPU or trained weights are required. OMP/OPENBLAS/MKL threads are1. BLAS/backend details and validated cleaned Python/NumPy/SciPy/Matplotlib versions are archived in cleaned_environment.txt. Cleaned runtime uses only its pinned requirements subset; Astropy/OpenCV and upstream domain source are absent from its imports.

Warnings retained honestly: source demregpy snapshot has no generated package-version module and emits a metadata warning, but executes its complete numerical path successfully; real AIA FITS headers contain a floating-image BLANK warning, which Astropy ignores as documented. Windows nbconvert/ZeroMQ emits an event-loop fallback/kernel transport diagnostic. These are recorded in logs; no numeric failure, suppressed solver warning, failed cell or cached reconstruction substitutes for execution. Seeds are absent from real acquisition and deterministic inverse solves; only private notebook shape-orientation diagnostics use the explicit20261020seed.

## Traceability from paper stages through executable code

| Required equation / stage | Original function / source | Local function | Independent validation consumer |
| --- | --- | --- | --- |
| PaperEq1 optically-thin count response / temperature discretization | demregpy/dn2dem.py::dn2dem log-response interpolation and dlogTfac | physics_model.response_matrix,forward | test_quadrature_units_and_log_response_interpolation; test_upstream_full_dem_errors_resolution_and_forward |
| PaperEq5 whitened constrained quadratic objective and Eq6 spectral inverse | demregpy/demmap.py::_standard_form_svd,_regularised_inverse | solvers._standard_form_svd,_regularised_inverse | test_svd_tikhonov_matches_normal_equations_and_discrepancy |
| PaperEq7 Morozov discrepancy principle | demregpy/demmap.py::_misfit,_discrepancy_lambda | solvers._misfit,_discrepancy_lambda | test_svd_tikhonov_matches_normal_equations_and_discrepancy |
| PaperSec2.1.1 positive solution selection / constraint weights | demregpy/demmap.py::dem_pix self-normalization, EM-loci and positivity loop | solvers.dem_pix,solve | test_upstream_full_dem_errors_resolution_and_forward; actual weighting/target sweep |
| Vertical errors and horizontal response resolution | demregpy/demmap.py::dem_pix edem,kdagk,elogt; dn2dem scaling | solvers.dem_pix,solve | test_upstream_full_dem_errors_resolution_and_forward |
| Full spatial DEM and predicted count rates | original dn2dem called on48spectra per row to keep serial path | solvers.solve,reconstruct | full independent fixture; independent Judge response metric |

Stage tolerances were frozen as float64 rtol1e-10/atol1e-12, or relative rtol1e-10 with zero absolute tolerance for physical DEM scales. Original uint8 full-pipeline outputs require exact equality; the original float32 GSA resampler also matches exactly. Independent analytic cases use scale-appropriate tighter tolerances specified directly in tests. A zero-detail GSA recentering comparison permits1e-12 absolute error, matching the frozen float64 floor and avoiding a sub-ULP equality demand. Fixtures never come from the cleaned function being tested.

## Ambiguities, physical checks and baseline quality

This is the modern demregpy diagonal-constraint SVD/root-finding implementation of the published regularization family. It is not claimed to be a byte-identical historical IDL/GSVD implementation or reproduction of the original paper's different observations. The standard-form diagonal SVD is independently checked against normal equations. The supplied response_log_temperature remains original big-endian float32; no silent float64 rewrite of calibration values occurs. Output DEM and the other measured quantities agree exactly with original Python execution. The author example's 10%+1count error is an adopted discrepancy model, not a full sensor likelihood. No additional time-dependent AIA degradation correction is applied. Positivity target relaxation leads to chi-square/channel ranging0.9999999999661332..11.390624999999993, mean2.9987725151899878. This genuine mismatch is plotted and not hidden. Independent repeat is exact; rounding rate/error to float32 under a different convention yields DEM NRMSE2.745742952335444e-8 and is a sensitivity result, not an identical-input repeat. The mean count-discrepancy Judge gate was added after initial primary parity using the original fixture only: boundary3.2986497667089867=1.1*originalmean. It is additional, not a relaxed reference-agreement threshold. Solar helioprojective longitude increases toward solar west; README and final plots use that correct direction.

## Data, calibration and reference hashes

Public raw_data.npz SHA256: `d0d3749b0742ee272c8d0ce9066f8dd0b0a75e46366ad20cd15f5c5fb1d1fcfb`. No hidden inverse target, privileged spectral image, synthetic object seed or target-derived calibration is bundled. All arrays are NPZ and batch-first. Source raw file hashes were captured before normalization/packaging and are listed below.

| Source acquisition file | SHA256 |
| --- | --- |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_094.fits | `bdbc77705fb8c6f5616f0436e3bd0764d86c3a9a298b35349a378b0a64256b57` |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_131.fits | `18f7b599ed580fa5b56285962dc734b3ae5af438e7e50b495cbdc45c7afcb3c2` |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_171.fits | `3b2969d9de44d055bd11bd5fff57fec93ff49e4dbd0b2f5ac2aaa18723304892` |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_193.fits | `a665e699e724a1504d7d3da2a0706307ffeccd6c9793bc6b019f7b2c290fb525` |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_211.fits | `48c6052218614cd53fe8e34b09a255e052a65f7dcf19142163bf09b660e52176` |
| demregpy-main\demregpy\tests\data\aia\aia_synoptic_2014-01-01T00-00-00_335.fits | `3f17e57803a64c0a691d6774c0158ae231d579468266699e4659eee95c9552a2` |
| Independent nominal response aia_tresp_en.dat | `8ab3cb9da628fc7afa33511f2a767f6e67233ae50256235c7c00a8973df1857c` |

Hidden reference: `evaluation/baseline_reference.npz` is the original standard-method reconstruction on the bound observations, not physical truth. No ground_truth.npz is invented. If an actual independently validated truth is later provided, the standalone Judge gives it explicit priority.

| Asset | SHA256 | Consumer and provenance |
| --- | --- | --- |
| evaluation\baseline_reference.npz | `21c5b72bd2fbc7be0900fe1f52823bd737b23b325c921bf74fe2c46a044d505e` | Independent original final result; Judge and original/fresh spatial notebook comparison |
| test\fixtures\upstream.npz | `7e17203427af139474f7eac114e733922b333d218f3418f5ac95698ea572a17a` | Original source stage/final fixture; comparing tests |
| notebooks\reference_outputs\fresh_summary.npz | `efa12312e33be92cab4ae40880e5bcbbbc2abf5ac576e63792aab56ea9ab98d6` | Fresh actively computed scalar summary; active notebook table, never a reconstruction cache |
| notebooks\reference_outputs\upstream_summary.npz | `efa12312e33be92cab4ae40880e5bcbbbc2abf5ac576e63792aab56ea9ab98d6` | Independent original-run scalar summary; active notebook original/fresh summary table |

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
    "reconstruction_nrmse": 0.0,
    "mean_chi2_per_channel": 2.9987725151899878
  },
  "thresholds": {
    "reconstruction_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "reconstruction_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "mean_chi2_per_channel": {
      "comparison": "<=",
      "boundary": 3.2986497667089867
    }
  },
  "errors": []
}
```

## Final executed commands and evidence

| Command | Exit | Wall seconds | Private log |
| --- | ---: | ---: | --- |
| `python -m src.main` | 0 | 1.648677 | solar_aia_dem_mapping_main.log |
| `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.163966 | solar_aia_dem_mapping_judge.log |
| `python -m pytest test/ -v --basetemp <unique private directory>` | 0 | 4.383312 | solar_aia_dem_mapping_pytest.log |
| `jupyter nbconvert --to notebook --execute notebooks/solar_aia_dem_mapping.ipynb --output solar_aia_dem_mapping_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180` | 0 | 16.196448 | solar_aia_dem_mapping_nbconvert.log |

Pytest summary: 22 passed in 4.09s ==============================. Notebook produced4actual PNG figure displays and0error outputs. The180-second per-cell timeout provides more than10times measured full-notebook execution headroom; it is not evidence that every published large experiment runs in seconds. The private imaging-research kernelspec points explicitly to the validated research-runtime Python, not the default Anaconda kernel. Formal notebook retains outputs of the successful active execution; the temporary executed output copy is archived outside the task.

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

Private gallery: `gallery/solar_aia_dem_mapping/`, four PNGs, four vector PDFs, four SVGs, `figure_source_arrays.npz` and `experiment_metrics.json`. These trace every exported plot and table to exact numeric arrays. No gallery exports or temporary screenshots are retained in the task package.

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

- Immutable-source provenance: verified_revisions.json, snapshot_tree_verification.json, demregpy-main/, corresponding downloaded ZIPs and source copyright notices.
- Paper: hannah_kontar_2012_arxiv_v1.pdf, extracted text where available; source URL/version/hash above. The KS93 scan has image-only text, and was reviewed through its published scan/source equations rather than claiming successful text extraction.
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
notebooks/reference_outputs/fresh_summary.npz
notebooks/reference_outputs/upstream_summary.npz
notebooks/solar_aia_dem_mapping.ipynb
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
test/fixtures/upstream.npz
test/unit_test/test_contract.py
test/validation.md
```

All tests and numeric fixtures reside under test/. There are no root wrappers, duplicate tests/report directories, source PDFs, complete upstream checkouts, download archives, metadata JSON, simulator seeds, trained weights or unconsumed array files in this task.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 4 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 22 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Adopted channel uncertainties and DEM ill-conditioning; inferred baseline is not independent truth.
