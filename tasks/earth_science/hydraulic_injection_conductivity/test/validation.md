# Validation and provenance

## Fixed physical binding and semantic distinction

Full author-existing simulation:75×100log hydraulic-conductivity cells,30positive injection experiments,870head observations,all native wells and cells. No new observation generation, crop, easier target or changed forcing is substituted. Natural log scale is K/(1m/day), thickness1m; FloPy default DIS time/length units are days/metres. Fixed heads lie on cell centers5/995m, not exterior faces. Q=+25m³/day is injection. Public raw arrays contain only measured heads and known well indices. Independent actual target is original true_logK.txt, never an estimated baseline. Original example explicitly says it is adapted from Section3.1, not a reproduction of the paper.

Full release README search for Darcy/aquifer/groundwater/hydraulic/permeability was checked, with closest actual READMEs read: covercrop_electromagnetic_conductivity recovers electrical layer conductivity from induction coils; australia_magnetotelluric_occam_profile recovers electrical resistivity from EM impedance; mri_transceive_phase_conductivity differentiates RF phase. None has controlled hydraulic injection heads or the water-transmission K field. This binding is not a renamed electrical-conductivity or EIT task.

## Source, paper and rights

Remote-verified GitHub https://github.com/jonghyunharrylee/pyPCGA exact commit d00a1174172da5e734dd586c16c6f43649fb27fb. Source ZIP55,990,039bytes, SHA25623adab85a5eaebd0782506c1aacf7e5e1300af448cac3f63fdc515537bd46d55; exact source is private source/ and original_work/ retains the native example/observations/executables. Root BSD3 copyright2018JonghyunLee is preserved verbatim in LICENSE; no separate data-specific license was found for included author simulation arrays. NOTICE preserves attribution and numerical/UQ qualifications without inventing another data grant. FloPy/MODFLOW exist only in the private oracle environment/archive.

Primary paper: Lee and Kitanidis2014, Large-scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach, WRR50,5410–5427,doi10.1002/2014WR015483, publisher18-pageversion. Full primary PDF text, equations, comparisons and AppendixA were read through https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/2014WR015483 . Actual browser-source response is archived as paper_primary_web_capture.json, SHA2560637be4275bdb4bbb2bffdefa0db44dc82df960dc3687aabddea10c67343ffc2. Two direct binary download attempts returned HTML, not PDFs: paper_author_PCGA2.pdf.failed_response.html and paper_publisher_PCGA2.pdf.failed_response.html are preserved as failures. No successful local PDF download is claimed. Associated foundational paper doi10.1002/2013WR014630 is identified separately; selected executable path follows the supplied source hydraulic example.

Private archive root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch8_20261022\groups\astro_earth_mech\pcga. archive_index.json indexes original logs, commands, source/capture scripts, downloaded/failed responses and their hashes. Data/executable hashes:

```json
{
  "raw_data_sha256": "42c55259c8f0cef35ead415d53283a65cd95957af0846fde7ecd921234f50f08",
  "actual_truth_sha256": "2a216cb16d9b41202c9bf27e9d7128f446808ca188dbf8240679b01a181488be",
  "original_input": {
    "obs.txt": "a78c4e93038ec64ea9f2391690627a65941e58c7a6ed924548b900d1d1880bd3",
    "true_logK.txt": "608eb9d40d76daa7ec7fa103238171dda162eb38bdf2c849ce646800916ffef0"
  },
  "native_executables": {
    "mf2005.exe": "75e7ddc540284839bdb23645ee482dd925638f315f79beb5cf4ae292487cf748",
    "mf2005dbl.exe": "da21a47a621a902dfc5b228d7df7486a8c414fbde3f1b810dc34e8e7728a6593"
  }
}
```

## Original run gates before extraction

Complete unchanged domain code, native mf2005.exe: exit0, runtime986.681466600s. Independent fresh complete source repeat: exit0, runtime1267.618270200s. Author-provided mf2005dbl.exe separate numerical precision diagnostic: exit0, runtime989.218023400s. All use the same exact7500unknowns/870observations/50PC/maximum10iterations. Resource-only sequential CPU1 preserves both native physical-core-dependent proposal counts at24; reducing parallel workers did not silently reduce LM/LS choices. Complete commands/stdout/stderr and backend versions are in original_process.json, selected_native_parameters.json, independent_stage_repeat/process.json and author_double_precision_diagnostic/process.json.

Upstream oracle is isolated oracle_env with NumPy2.3.4/FloPy3.9.5 and read-only research SciPy1.17.1bridge. Shared NumPy2.4.6was not changed. NumPy2.4 legacy scalar-format failure and an initial nested-venv numeric-path failure are retained. Native source np.random.seed1234does not control default ARPACK v0=None; independent repetitions/backend results provide the stability evidence. The first stage-profile attempt spent excessive time resolving file paths on every return; it was explicitly stopped, its true nonzero process status retained under independent_stage_repeat_slow_capture_interrupted. Optimized passive capture did not change domain expressions and completed all5Jacobian/iteration calls and11,430native per-experiment forward calls. Normal tests consume captured numeric fixtures, never oracle imports.

Independent full finite-volume checks on eight captured source7500head arrays have maximum errors below1.354e−5m. Analytic constant-K/recharge profile error6.594e−12m proves signs/boundary placement/units independently. Direct covariance rows verify the captured prior eigenpair residual1.277e−15and orthogonality1.554e−15. Directional Jacobian products have separately measured single/double-backend differences, largest relative.0043941. Independent24candidate co-kriging algebra agrees with original to relative4.7895e−11and residual1.7006e−10. The initial checker formed a float32Gram before mixed-double operations and failed; initial_checker_float32_gram_failure.json is retained. Correcting this own-checker arithmetic preserves source coefficients and does not change any inverse threshold.

## Traceability

| Equation/stage | Original file/function | Local implementation | Independent case |
|---|---|---|---|
| PaperEq24–26 confined flow | example mf.py Model.run_model_single | HydraulicModel.assemble/forward | Eight whole-source head cases; analytic recharge profile |
| Stationary covariance/eigencompression Eq20–22 | covariance/toeplitz.py; PCGA.ComputePriorEig | covariance_operator/prior_eigenpairs | Original eigenpairs; independent dense64rows |
| Objective/quasi-linear inverse | pcga.py ObjectiveFunction/ObjectiveFunctionNoBeta | objective | Original24candidate/fullfield tests |
| Directional forward differences | pcga.py JacVect/JacMat | jacobian_products | Original HX/HZ/Hs; independent author-double diagnostic |
| Co-kriging saddle update | pcga.py IterativeSolve | cokriging_candidates/inverse_data_covariance | All24original candidates before full-local scoring |
| Damping/line search/stopping | pcga.py LinearIteration/LineSearch/GaussNewton | reconstruct | Complete source-field/actual-truth/observation tests |
| Conditional variance Eq17 | pcga.py ComputePosteriorDiagonalEntries | variance_diagnostics | Actual unchanged helper on known small saddle matrix; complete independent inverse |

## Explicit source variance discrepancy

The original helper drops lower-input saddle terms but supplies Xi in its right-hand side. It also clips values above unit prior variance. Unchanged original helper was executed on an independently specified nonsingular small matrix, producing [0.9681826344125222, 0.9238528989872044, 0.8605757951408335]; direct complete inverse gives [1.1584705319829396, 1.1171794821849357, 1.056941063965879], maximum difference0.19636526882504546. See original_variance_known_case.json and prove_original_variance_omission.py. Source result is retained only as source_variance_diagnostic and compared actively; complete unknown-mean covariance is separately computed and allows values above1. No posterior diagnostic is misrepresented as truth or silently dropped. Primary logK inverse and evaluation remain unchanged. The first new known-case unit test exposed an inappropriate inherited upper clip in the independent complete formula; that clip was removed before final notebook/tests, leaving source-form diagnostic unchanged.

## Frozen criteria and fresh results

source_only_quality_freeze.json was written 2026-10-22T08:30:07.097791+00:00 before cleaned implementation/full-quality scoring. First source truth NCC0.9968568173310184, range-NRMSE0.04797944553496118;0.9/1.1rule gives NCC0.8971711355979166, range-NRMSE0.052777390088457304. Full head RMS boundary0.5412737116335221m. Independent source repeat/double field differences fix separate standard-method parity bound0.008721770830491934; source-variance diagnostic relativeL2bound0.005688053250801807. No boundary was relaxed after local measurements. Constant field error.199085and initial error.395682fail the truth range criterion. Source/backends truth errors.047979/.048485/.048941demonstrate meaningful stable quality.

Fresh complete standalone report:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 0.9967930225369562,
    "nrmse": 0.04846226553051235,
    "forward_head_rms": 0.4920785967437012
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.8971711355979166
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 0.052777390088457304
    },
    "forward_head_rms": {
      "comparison": "<=",
      "boundary": 0.5412737116335221
    }
  },
  "errors": []
}
```

Whole-field range-NRMSE uses RMS error divided by actual target max−min; NCC is cosine without centering. No alignment, range clipping, gain fit, magnitude conversion or cell exclusion. Independent truth-observation check gives RMS.498215mconsistent with the stated.5mnoise model, without estimating calibration from target residuals. Source has5updates; local serialized-head backend has7but meets separately frozen fullfield, truth, head and source-variance criteria. Return the best objective field rather than blindly the last iterate.

## Clean runtime, fixtures and execution

```json
{
  "python": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python_version": "3.11.11",
  "platform": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "scipy": "1.17.1",
  "matplotlib": "3.11.1",
  "threads": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}
```

Exact BLAS/compiler configuration is in numerical_backend.txt. Clean runtime imports only standard library/NumPy/SciPy/Matplotlib; no FloPy, MODFLOW, covariance/inversion package or cross-task import. General sparse LU batches the unchanged30right-hand sides; explicit Schur/Woodbury replaces the equivalent generic Krylov linear backend, with independently checked source-stage agreement. No shared package installation/upgrade occurred.

Fixture consumers:

```json
{
  "native_head_cases": "Eight original_forward_case*.npz: all logK, heads, well indices, call IDs, row/column counts consumed by test_full_native_head_fields",
  "prior": "original_prior.npz: eigenpairs/coordinates/X consumed by covariance/geometry and candidate tests",
  "jacobian": "original_jacobian.npz: field, source base, directions and allHX/HZ/Hs consumed by test_source_jacobian_products",
  "candidates": "original_candidates.npz: all24Z/HX/HZ/rhs/alphas/candidate/beta arrays consumed by independent saddle-update test",
  "notebook_reference": "original_reconstruction.npz: source fullfield, all870heads and named source variance diagnostic; original_objective.npz: original printed objective trace"
}
```

Full final test command: ['D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe', '-X', 'utf8', '-m', 'pytest', 'test/', '-v', '--basetemp', 'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\pcga\\pytest_release_temp']; native0, runtime18.339828800s. Logs final_pytest_stdout/stderr retain exact case counts. Standard main native0/14.667600400s, independent Judge native0/0.421975900s, each with its own command/process/log files. Bad file/key/shape/complex/integer/float16/NaN/Inf, malformed numeric configuration, extreme finite metrics, invalid truth without fallback, known metric cases, poor fields, foreign-cwd pass/fail/error and report-write failure are tested.

Final complete active notebook native0/121.105146300s, canonicalSHA256 85cc3a66068ce388d48f704b0f0d5ad9d8bf1a0c50953024d269e1dfe2970d16. Supervisor command is in notebook_process.json; stdout/stderr separate and shell=False. Actual research kernel executable/versions/backend are printed; all code cells executed and six rendered PNG outputs retained. First complete execution122.8104749s was inspected; spatial map aspect ratios and English labels were polished, then the full path executed again. Windows general Proactor/ZMQ warnings are retained, not hidden or misclassified as scientific failure. Temporary executed notebook is private.

All six actual final figure groups were opened: complete survey, eigen-spectrum/convergence, actual truth/source/fresh fields, all870head fits/residuals, original-versus-complete uncertainty and actual rank/noise sensitivity. Every signed scale is symmetric; spatial maps have equal physical aspect. Numeric sources and all PNG/SVG/PDF hashes are in figures/plot_provenance.json, bound to this canonical notebook. Additional diagnostic noise comes only from generate_data.perturb_heads in metres with explicit seed124; it changes no bound observation or geometry. The full primary reconstruction is active, not a cache preview.

Plans semantically checked against actual callable signatures, constants, signs, finite-volume boundary positions, direction/precision, objective normalization,24damping/line-search candidates, stopping/best-selection and output units. Source posterior caveat and complete comparison are explicit. The source example's unavailable paper textbook/tracer arrays are not fabricated or relabeled as computed methods.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/equation mapping | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Original and local exact observations | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Independent source parity and physics | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Constants/units/precision/coordinates | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Standard main and fresh Judge | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Judge poor/adversarial/error contracts | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Public input sufficiency | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| No target/calibration/GitHub leaks | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| README metadata and output contract | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Minimal independent shared runtime | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Explicit local acquisition/inverse mathematics | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Complete unit/parity/physics/pipeline tests | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Active notebook/native execution/visuals | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Metric and result provenance | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Plans/README/signature/math consistency | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |
| Template inventory/private archives/license notices | passed | Evidence and actual commands in the corresponding sections above; machine record final_evidence.json. |

Parent private acceptance is pending; no release move or parent acceptance is claimed.

Final worker audit: qa/audit.json has errors[] and independent foreign-working-directory Judge native0. All23 extracted formulas pass both MathJax3.2.2 and KaTeX0.16.22; scanner issues[]. Full final pytest has32 passed/native0 in18.339828800s. Final equal-aspect figures3/5/6 were opened again after the second complete execution; all six scientific groups reviewed. Parent acceptance remains pending.


## Parent independent private acceptance

Parent independently reran32tests: native0/24.330613700s;23formulas passed both renderers. Parent semantically reviewed README, both plans, full acquisition/inverse/Judge/generator source, provenance and all six notebook story sections, and actually opened all6final scientific PNGs individually. Canonical Notebook is byte-for-byte equal to actual native successful execution0/121.1051463s, SHA256 85cc3a66068ce388d48f704b0f0d5ad9d8bf1a0c50953024d269e1dfe2970d16; every code execution count is populated and no error output exists. Complete7500cell simulation truth is retained, all870heads evaluated, with NCC.9967930/rangeNRMSE.0484623/headRMS.492079m. All source-backend tolerances were frozen before extraction, with the5-versus7iteration difference explicitly retained. The independently proved source posterior omission remains a named source-only diagnostic; the complete unknown-mean saddle covariance is a separate comparison, neither a required recovered field nor true geological uncertainty. No independent real-acquisition claim is made for this author-existing simulation. Separate data license uncertainty remains disclosed. Private staging acceptance only; no batch8release claimed.
