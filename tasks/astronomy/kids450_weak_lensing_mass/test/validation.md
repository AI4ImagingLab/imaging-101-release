# Validation and provenance: kids450_weak_lensing_mass

All mandatory cleaning gates below passed on CPU. Method agreement is separated from physical accuracy: these are real observations with no independently known inverse target. No cached paper number is presented as a measurement from this run.

## Sources, exact revisions and licenses

- Repository: https://github.com/CosmoStat/lenspack
- Exact remotely verified commit: `71c22da42a5f470a777fbefedf1de52db05fa90f` (GitHub commits/tree API queried2026-10-20; not merely a local tracking revision). The downloaded branch ZIP was checked against that commit by recomputing every local git-blob SHA1: 44files checked,0mismatches. See private verified_revisions.json and snapshot_tree_verification.json.
- Paper/version: Kaiser & Squires1993, published ApJ404,441–450, DOI10.1086/172297. Exact archived published scan kaiser_squires_1993_published.pdf (filename below uses kaiser).
- Exact paper URL: https://articles.adsabs.harvard.edu/pdf/1993ApJ...404..441K
- Archived paper: `kaiser_squires_1993_published.pdf`, SHA256 `45a8162859aa208ba145a882be6523cec40c52c24b561dcf60f1e41d34091282`. Original complete source and paper stay outside the final task.
- Data: Real KiDS-450/ESO DR3.1, VizieRII/384/kids_shear, Hildebrandt etal.2017, DOI10.1093/mnras/stw2805; input acquisition data DOI10.18727/archive/37. Publisher ReadMe explicitly CC BY4.0. The archived TSV query selects one-degree box around RA129deg,Dec-0.5deg, then exact ROI/quality/z/weight cuts leave16625sources. No simulated catalogue is bundled.
- Source license: MIT; original data CC BY4.0. Task NOTICE preserves source copyright and adaptation information.

## Fixed method, observations and oracle execution

Private archive root: `imaging-101-maintainer/research_batch2_20261020/astro_earth/`, a sibling maintainer area outside this release task. Selected source snapshot: `lenspack-master/`. Capture script: `capture_oracles.py`. It imports only the unmodified private original source, never cleaned src. The cleaned solver/tests/notebook never import that archive. Normal tests consume frozen arrays only.

The first joint capture ran original Lenspack bin2d/ks93, original SFIM and original demregpy dn2dem on exactly the public observations before writing local mathematics. It exited0, measured1.929347800090909seconds total, and saved stage/final arrays. AIA is called48spectra per row to select the original serial path rather than its automatic multiprocessing path at>=200pixels. The original np.int compatibility alias for the old pansharpening interpolation changes only a removed NumPy integer-constructor spelling, not mathematics. Source itself remains unchanged.

The independent source fixtures precede local implementation; acceptance_before_local.json records stage tolerances and reference-agreement boundaries before the first cleaned measurement.

Oracle environment: private oracle-env, Windows10.0.26200, CPython3.11.11, NumPy2.4.6, SciPy1.17.1, Astropy8.0.2 FITS reader, threadpoolctl3.7.0 and tqdm4.70.1; OpenCV wheel5.0.0.93 (cv2 reports5.0.0) is used only for original pansharpening execution. Full versions are in oracle_environment_pip_freeze.txt. No GPU or trained weights are required. OMP/OPENBLAS/MKL threads are1. BLAS/backend details and validated cleaned Python/NumPy/SciPy/Matplotlib versions are archived in cleaned_environment.txt. Cleaned runtime uses only its pinned requirements subset; Astropy/OpenCV and upstream domain source are absent from its imports.

Warnings retained honestly: source demregpy snapshot has no generated package-version module and emits a metadata warning, but executes its complete numerical path successfully; real AIA FITS headers contain a floating-image BLANK warning, which Astropy ignores as documented. Windows nbconvert/ZeroMQ emits an event-loop fallback/kernel transport diagnostic. These are recorded in logs; no numeric failure, suppressed solver warning, failed cell or cached reconstruction substitutes for execution. Seeds are absent from real acquisition and deterministic inverse solves; only private notebook shape-orientation diagnostics use the explicit20261020seed.

## Traceability from paper stages through executable code

| Required equation / stage | Original function / source | Local function | Independent validation consumer |
| --- | --- | --- | --- |
| KiDS ensemble shear calibration / weighted irregular gridding | lenspack/utils.py::bin2d plus published ensemble correction in KiDS ReadMe | solvers.grid_catalogue | test_frozen_upstream_stage_and_full_pipeline; test_calibration_ensemble_is_not_individual_division |
| KS93 Eqs2.1.11?2.1.12 Fourier spin-2 kernels, zero mass-sheet mode | lenspack/image/inversion.py::ks93 | physics_model.kernels,inverse | test_frozen_upstream_stage_and_full_pipeline; test_analytic_potential_spin2_and_mass_sheet |
| Shear forward relation / adjoint | lenspack/image/inversion.py::ks93inv and analytic potential derivative | physics_model.forward | test_adjoint_inner_product; test_analytic_potential_spin2_and_mass_sheet |
| KS93 Sec2.2/Eq2.2.3 Gaussian resolution and shape-noise filtering | original ks93 output followed by scipy.ndimage.gaussian_filter sigma2,wrap | solvers.solve | test_frozen_upstream_stage_and_full_pipeline; independent Judge observation check |
| E/B and finite-field uncertainty diagnostics | KS93 discussion of intrinsic ellipticity noise; real data have no mass truth | visualization.review | 32 actual orientation randomizations, PSF map and sigma1/2/4 computed in notebook |

Stage tolerances were frozen as float64 rtol1e-10/atol1e-12, or relative rtol1e-10 with zero absolute tolerance for physical DEM scales. Original uint8 full-pipeline outputs require exact equality; the original float32 GSA resampler also matches exactly. Independent analytic cases use scale-appropriate tighter tolerances specified directly in tests. A zero-detail GSA recentering comparison permits1e-12 absolute error, matching the frozen float64 floor and avoiding a sub-ULP equality demand. Fixtures never come from the cleaned function being tested.

## Ambiguities, physical checks and baseline quality

The KiDS map is shape-noise limited: the B and randomized-orientation maps have amplitudes comparable to E. No individual peak is advertised as a detected mass concentration. The output is the declared standard-method mass-map estimator, not an independently validated dark-matter image. The even64grid real-part convention loses mixed Nyquist information; the analytic low-frequency and adjoint tests pass, and the full forward residual is retained rather than claiming exact inversion at every frequency. The fixed equal-pixel Fourier aspect approximation differs from the tangent-plane cell aspect by0.0038%. Catalogue decimal precision is preserved. No extra PSF leakage calibration or nonlinear reduced-shear correction is invented. The nominal output resolution sigma2pixels is public. The secondary observation metric was added after initial primary parity, from the independent original fixture: relative shear residual5.100980825819833e-7, boundary1e-4. This only adds a gate; it does not relax the earlier NCC/NRMSE criteria.

## Data, calibration and reference hashes

Public raw_data.npz SHA256: `0f1b96dfc5e7126a9415d20a6c10daf531d6bd7eefab1c1447619b6d99dbbb51`. No hidden inverse target, privileged spectral image, synthetic object seed or target-derived calibration is bundled. All arrays are NPZ and batch-first. Source raw file hashes were captured before normalization/packaging and are listed below.

| Source acquisition file | SHA256 |
| --- | --- |
| kids_roi.tsv | `53a25f7d09e8cc97a3233378a2fa543b186336bf9240505ce07db3f1db02a479` |

Exact catalogue request: https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=II%2F384%2Fkids_shear&-out=RAJ2000%2CDEJ2000%2Ce1%2Ce2%2CWeight%2CMultCal%2CPSFe1%2CPSFe2%2Czbest%2CMask%2CFitClass&-out.max=100000&-c=129.000000-00.500000&-c.bs=3600&-c.u=deg&zbest=0.1..0.9&Mask=0&FitClass=0
The query/table definition and declared CC BY4.0 license are retained in kids_readme.txt; the public calibrated shapes preserve the CDS decimal precision.

Hidden reference: `evaluation/baseline_reference.npz` is the original standard-method reconstruction on the bound observations, not physical truth. No ground_truth.npz is invented. If an actual independently validated truth is later provided, the standalone Judge gives it explicit priority.

| Asset | SHA256 | Consumer and provenance |
| --- | --- | --- |
| evaluation\baseline_reference.npz | `7d8d5b11481bae1f75515cb00527684584203a9f11fec64422bb7a533bcffa93` | Independent original final result; Judge and original/fresh spatial notebook comparison |
| test\fixtures\upstream.npz | `3856422e0edac0002f85b31060854a3f643b2dc1542e2c5c937fec44e9f0feda` | Original source stage/final fixture; comparing tests |
| notebooks\reference_outputs\fresh_summary.npz | `6ea9b5df66acd3be90f9f8f6a4ad9bedd7f12b9564a3c8aea8b982ef9b30ee29` | Fresh actively computed scalar summary; active notebook table, never a reconstruction cache |
| notebooks\reference_outputs\upstream_summary.npz | `e1c257c2a761eed9a4c1917c90f0c2a3bf56949e831ba0efb7f896c6a38511fa` | Independent original-run scalar summary; active notebook original/fresh summary table |

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
    "reconstruction_nrmse": 3.708387311774689e-17,
    "bmode_ncc": 1.0,
    "bmode_nrmse": 4.019121957954137e-17,
    "relative_shear_residual": 5.100980825848082e-07
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
    "bmode_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "bmode_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "relative_shear_residual": {
      "comparison": "<=",
      "boundary": 0.0001
    }
  },
  "errors": []
}
```

## Final executed commands and evidence

| Command | Exit | Wall seconds | Private log |
| --- | ---: | ---: | --- |
| `python -m src.main` | 0 | 0.350303 | kids450_weak_lensing_mass_main.log |
| `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.350199 | kids450_weak_lensing_mass_judge.log |
| `python -m pytest test/ -v --basetemp <unique private directory>` | 0 | 2.345689 | kids450_weak_lensing_mass_pytest.log |
| `jupyter nbconvert --to notebook --execute notebooks/kids450_weak_lensing_mass.ipynb --output kids450_weak_lensing_mass_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180` | 0 | 8.924947 | kids450_weak_lensing_mass_nbconvert.log |

Pytest summary: 24 passed in 2.05s ==============================. Notebook produced4actual PNG figure displays and0error outputs. The180-second per-cell timeout provides more than10times measured full-notebook execution headroom; it is not evidence that every published large experiment runs in seconds. The private imaging-research kernelspec points explicitly to the validated research-runtime Python, not the default Anaconda kernel. Formal notebook retains outputs of the successful active execution; the temporary executed output copy is archived outside the task.

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

Private gallery: `gallery/kids450_weak_lensing_mass/`, four PNGs, four vector PDFs, four SVGs, `figure_source_arrays.npz` and `experiment_metrics.json`. These trace every exported plot and table to exact numeric arrays. No gallery exports or temporary screenshots are retained in the task package.

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

- Immutable-source provenance: verified_revisions.json, snapshot_tree_verification.json, lenspack-master/, corresponding downloaded ZIPs and source copyright notices.
- Paper: kaiser_squires_1993_published.pdf, extracted text where available; source URL/version/hash above. The KS93 scan has image-only text, and was reviewed through its published scan/source equations rather than claiming successful text extraction.
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
notebooks/kids450_weak_lensing_mass.ipynb
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
test/fixtures/upstream.npz
test/unit_test/test_contract.py
test/validation.md
```

All tests and numeric fixtures reside under test/. There are no root wrappers, duplicate tests/report directories, source PDFs, complete upstream checkouts, download archives, metadata JSON, simulator seeds, trained weights or unconsumed array files in this task.

## Quantified DC and even-grid Nyquist limits

A subsequent independent unmodified Lenspack ks93/ks93inv investigation is archived in private kids_nyquist_investigation.json. DC is removed before all following loss measurements. On bound unsmoothed shear, the relative round-trip residual is0.11671448696328429, energy fraction0.013622271467102659; thus the raw arbitrary real-shear map is NOT strictly invertible. On the actual sigma2-pixel output resolution, the residual is5.100980825832628e-7, energy fraction2.602000538551212e-13, maximum component residual5.895273347267249e-8. This small post-filter quantity is distinct from the unobservable DC shear components(-0.002018809126667842,0.0032107191158432307).

A random real shear field likewise has0.11794703832818561 relative raw loss but4.7408767102097243e-7 after sigma2 filtering. An analytic E mode cos(pi*x)*cos(2*pi*3*y/64) gives p1=(32^2-3^2)/(32^2+3^2)=0.9825750242013552; its source-convention E round-trip amplitude is p1^2=0.9654536781842938, agreeing with a3.4546321815706% loss. This is the mixed-Nyquist real-part projection, not the DC mass-sheet degeneracy. Source convention is retained; analytic/adjoint checks do not imply exact inversion for all discrete real modes. B and orientation noise remain comparable to E, and no independently significant dark-matter detection is claimed.

Primary KS93 published scan pages442-444 were rendered and visually read in the private archive (ks93_page_2/3/4.png). The historical quadrupole ellipticity convention includes its own response/sign normalization; this task uses the modern supplied lensfit shear calibration and Lenspack conventions explicitly, not an extra historical ellipticity-response factor or sign flip. The shared spin-2 kernels are mapped to Eq2.1.11 and the practical Gaussian estimator to Sec2.2.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 4 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 24 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Finite-field/DC and real-projection Nyquist loss quantified; no significant mass detection asserted.
