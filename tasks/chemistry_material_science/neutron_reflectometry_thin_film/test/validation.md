# Validation and provenance: neutron reflectometry thin film

## Bound problem and evidence status

This is a new one dimensional inverse scattering task using the complete reduced PLATYPUS curve distributed with refnx. The unknown is the film/oxide depth structure with intensity scale/background nuisance parameters. No independent microscopic truth exists here. The hidden `evaluation/baseline_reference.npz` contains `reconstruction` `(1,821)` and `parameters` `(1,8)` from an independent execution of the standard method, **not ground truth**. No ground_truth.npz is fabricated.

Paper: Nelson and Prescott, Journal of Applied Crystallography 52 (2019), 193–200, DOI 10.1107/S1600576718017296, published version (received 2018-08-31, accepted 2018-12-05). Full text was read as Europe PMC XML https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6362611/fullTextXML and archived as refnx_paper_2019.xml. Publisher PDF returned HTTP403 and PMC HTML download encountered SSL EOF; these unsuccessful requests are not claimed as successful PDF captures. Published XML includes the complete sections. Paper copyright CC-BY; source/data redistribution follows the repository BSD-3-Clause license, preserved in task LICENSE. This reproduces a modern documented deterministic maximum likelihood stage from the accompanying code, not all Bayesian comparisons in the 2019 publication.

Repository: https://github.com/refnx/refnx . Exact remote main commit verified using GitHub commits API on 2026-10-20: `e9bbdff97fb79bdb6cf5ea04f9121fbfe7fd842d`. Exact source archive was downloaded from codeload for that SHA. This is a verified remote revision, not a stale local tracking claim. The initial git clone stalled; the successful archive, not that incomplete clone, supplied execution. Upstream data path: `refnx/analysis/tests/c_PLP0011859_q.txt`, SHA256 `658bdc15a4bb2cc1d65bdbb7838fe32b597520c612ad633954822d1ac575863a`. The four columns are Q, measured reflectivity, its standard error and point dependent resolution FWHM. Data are preserved bit for bit as parsed float64, all 408 positive points, no selection or downcast. The upstream tutorial identifies the instrument dataset and the layer model; no unreleased beamline logs or exact film chemistry are invented.

Bundled raw NPZ SHA256: `bdd3a5c692938925d163a4e1b3f09a39ab50edb6604dd9c9d3e79e3ad34f0fc8`. Frozen oracle NPZ SHA256: `e48ace79f699b2bc73689d991a1f45f066552d995e004d05e56e9dc01c6d6f65`. Seed 2718/31415 control only oracle optimization; no hidden simulator object or target-derived calibration is shipped. Public data contain only q, reflectivity, standard_error, q_fwhm, all batch first. The README defines constants, depth direction, shape, units and complete outputs; original source URLs occur only in maintainer materials. Public requirements use only shared pinned NumPy, SciPy and Matplotlib.

## Independent upstream execution before local extraction

External maintainer archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/optics_astronomy`. Source: `source/refnx-e9bbdff97fb79bdb6cf5ea04f9121fbfe7fd842d`. `capture_neutron.py` imports that unmodified source ahead of an isolated dependency directory and explicitly selects the pure Python backend. The refnx wheel was installed only into `oracle-packages` as an optional investigation aid; **its implementation was not used**. The script prints the source location, and all method imports resolve into the exact source checkout. Oracle-only dependencies: emcee 3.1.6, orsopy 1.2.3, periodictable 2.1.0, PyYAML 6.0.3; installation used --no-deps in the private directory, no shared environment upgrades. Exact source exports were run on the same NumPy 2.4.6/SciPy 1.17.1 CPU backend as the cleaned method. CPU single worker and OMP/OPENBLAS thread limits 1. Arrays are small, no GPU or large allocation required.

Commands (run from project workspace; executable abbreviated as research-python below is `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe`):

```
research-python <archive>/capture_neutron.py
research-python <archive>/capture_neutron.py 31415
```

Both exit status 0. Primary runtime 27.483037300 s, rerun 32.113048200 s. Primary chi squared 919.5863753458957; rerun 919.5863760002351. Independent rerun profile dynamic-range NRMSE = 2.170614302658779e-6. Parameters are archived in both JSON files; no historical paper metric is substituted. Actual oracle logs include SciPy optimizer progress; no unresolved numerical warning was suppressed. The documented tutorial likelihood and acquisition conventions were checked against the paper method sections 2.1–2.3. Optional MCMC, straight line fitting and synthetic multilayer demonstration are explicitly outside the selected benchmark; the notebook retains all selected physical comparisons (data/initial/upstream/fresh/profile/residuals and parameter table).

## Traceability and frozen acceptance criteria

| Published stage | Upstream file/function | Local executable implementation | Independent validation |
|---|---|---|---|
| Four medium slab model, paper section 2.1 | doc/getting_started.ipynb, SLD/Slab setup | physics_model.slabs, constants | exact slab array and README constants test |
| Abeles propagation, paper section 2 | reflect/_reflect.py abeles | physics_model.abeles | initial/final upstream intensities; Fresnel analytic limit, total reflection, zero contrast, layer split invariance |
| Nevot–Croce roughness, section 2.1 | reflect/_reflect.py abeles | explicit rj exponential | initial/final oracle forward parity |
| Resolution FWHM convolution, section 2.1 | reflect/reflect_model.py _smeared_kernel_pointwise | physics_model.forward | oracle initial/final intensities at all 408 points |
| Log transform and Gaussian likelihood, section 2.2/2.3 and likelihood equation (2) | analysis/objective.py Transform, Objective.residuals/logl | residuals, negative_log_likelihood | frozen initial residuals and final objective/chi squared |
| Global maximum likelihood fitting, section 2.3 | analysis/curvefitter.py CurveFitter.fit | solvers.reconstruct with scipy differential_evolution | independent full-pipeline active test and notebook |
| Error function depth representation | reflect/structure.py sld_profile | physics_model.profile | frozen profile on exactly identical Z |
| Evaluation | oracle fit/reference and public uncertainties | standalone evaluation/judge.py | known offset metric, malformed/poor outputs, missing assets, cwd/report failures |

Frozen fixtures were captured from the oracle **before local extraction**. Deterministic parity tolerances were chosen at float64 scale: forward rtol 2e-12/atol 1e-15, residual rtol 2e-11/atol 1e-10, profile rtol 2e-13/atol 1e-13. Fixtures are not generated by cleaned functions. End to end acceptance before inspecting cleaned output: profile NRMSE <=2e-5, parameter maximum normalized difference <=2e-4, observation log chi squared <=930. First two limits are approximately 9.2 and 8.0 times the independently observed numerical rerun variations; nonzero tolerances are justified by a separately seeded full inversion, never by self comparison. The 930 consistency limit retains close standard-method agreement, approximately 1.13 percent above observed chi squared. It is not relaxed to claim reduced chi squared one. All 408 observations are scored. Default NCC is not useful as a sole score for nearly identical nonzero-offset SLD profiles, so explicit profile/parameter/consistency metrics are used and defined in metrics.json.

Suspected discrepancies checked: resolution FWHM is not a sigma; stored SLD is multiplied by 10^-6 only inside wave physics; matrix wavevector branch and reflectivity conjugation are preserved; Gaussian truncated integral is intentionally not renormalized; residual sign and likelihood 1/2 and constant term are preserved; the diagnostic initial vector is not confused with optimizer population initialization. No upstream correction or algorithm substitution was needed. Tiny implementation arithmetic reordering has measured deterministic forward parity. The profile is a conventional visualization of the rough slab model, not an assertion of an exact smooth scattering potential.

## Fresh cleaned results and independent evaluation

`python -m src.main` completed, stdout in neutron_cleaned.log: 38.52659650007263 s, 147 generations, success true, nll -571.1242081121126. The fresh submission passed `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`, exit 0. Full precision measurements:

```json
{
  "profile_nrmse": 5.368655598089066e-11,
  "parameter_error": 4.495999868699604e-10,
  "log_chisqr": 919.5863753459407
}
```

`notebooks/reference_outputs/upstream.npz` contains independently run oracle profile/parameters/prediction for notebook comparisons. `cleaned.npz` contains the first fresh cleaned CLI profile/parameters, explicitly separately identified, not loaded as a fresh result. `test/fixtures/upstream.npz` contains raw numerical input, diagnostic initial parameters, upstream initial model/residual, fitted parameters/prediction/profile/slabs and z; consumers are the parity and active pipeline tests. Hidden baseline only contains the declared reconstruction and parameter arrays. Extra captured arrays privately remain in the oracle archive; no observation is mislabeled as truth.

## Tests, notebook, and visual review

`python -m pytest test/ -v` originally ran numerical parity/physics/pipeline successfully but 12 tmp_path tests encountered Windows WinError5 on the machine's default C:/Users/admin/AppData/Local/Temp/pytest-of-admin. Original failed run is preserved as neutron_pytest.log; it is not counted as a pass. Re-running with an explicit maintainer `--basetemp=<archive>/pytest-neutron` passed all 18 original tests in 39.40 s. Final added README/source isolation test and extra slab/grid assertions are covered by neutron_final_checks.log and neutron_full_final.log. This is a machine temporary-directory ACL workaround, not a missing runtime dependency or skipped test. Unit/Judge contracts include missing file/key, shape, float32/complex, NaN/Inf, zero/poor physical output, evaluator asset loss, output metric known offset, cwd independence and report-write failure. Solver/Judge actual imports are limited to standard library/NumPy/SciPy/Matplotlib; no domain package, cross-task import or submitted solver import enters the evaluator.

The notebook actively reconstructed the complete dataset with runtime 31.88151440001093 s and the same measured scores. It was executed by nbconvert with timeout 180 s (over 4x measured reconstruction time), via a private kernelspec pointing at the validated research Python. Initial default-kernel launch selected machine Anaconda and was interrupted; that launch is not evidence of compatibility. Validated command:

```
research-python -m jupyter nbconvert --to notebook --execute notebooks/neutron_reflectometry_thin_film.ipynb --output neutron_reflectometry_thin_film_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=optics-research
```

Private JUPYTER_PATH points to archive/jupyter/share/jupyter. Validated log neutron_nbconvert_validated.log records successful writing of the executed notebook; its active output confirms NumPy 2.4.6. Routine Windows zmq event loop/TCP warnings do not affect calculation. The executed notebook was archived, its evaluated content retained in the shipped primary notebook with portable python3 kernel metadata, and the temporary output notebook removed. `neutron_review.png` was visually inspected at full figure scale: all three panels rendered, no missing comparisons/zeros, axes/units/legends readable, oracle and fresh profiles overlap as numerically expected, measured fringes and residual systematics remain visible. The figure is archived externally and removed from output.

The nontrivial reduced log chi squared is 2.29896593836485. Reported agreement is only with the selected published method at its specified physical parameterization. No historical result is labeled a fresh metric; no true accuracy is claimed without independent truth.

## Plan consistency and inventory review

Semantically checked the README/approach/design against actual source: all function signatures/defaults, eight parameter order/units, fixed constants, z coordinate and batch axis; Abeles factors/sign/units; FWHM and truncated integral; likelihood sign/normalization; optimizer settings/init/stopping; CLI and notebook data paths; independent Judge behavior. Source function-role audit finds no domain solver shortcut, forbidden import, cross-task import, upstream runtime access, cached-result substitution, or simulator truth leakage. README raw table lists every public NPZ key; no metadata JSON exists and no GitHub URL occurs in solver-visible README/requirements. License notices retained. Source data reused so generate_data.py is correctly omitted.

All final files fit the required task template. Python caches and pytest cache are disposable and removed at handoff; private logs/source/PDF attempts/capture scripts and figure are outside task. No root wrapper or duplicate report directory is created. Archive inventory with SHA256 for each retained evidence file is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/optics_astronomy/neutron_archive_index.json`. Source archive hash `45552c8ed3ffc14e91818652acef4fb39d9a15ff3599b4cff64cf644061b3bc1`, published full text XML hash `b68cd0d4a88f820e3393a0fe491c0f90e48c0621ed932bb90af8e17b957772a0`. Cleaned environment details: neutron_cleaned_environment.json (Windows-10-10.0.26200-SP0, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1), CPU only.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/exact remote revision/data hashes/mapping | passed | published XML, API SHA, hashes and table above |
| Original and cleaned same-data runs | passed | two independent oracle logs, cleaned CLI log |
| Frozen parity and independent physics | passed | comparing_test, upstream fixture, analytical limits |
| Constants/units/precision/conventions | passed | array equality, README/constants/source tests, semantic review |
| Standard entry and fresh Judge | passed | reconstruction.npz and judge_report.json |
| Strict Judge errors/poor output/evaluator failures | passed | unit_test/test_contract.py |
| Public information sufficient | passed | all Q/uncertainties/resolution/calibration constants and full output contract |
| No truth/target calibration/GitHub leak | passed | raw allowlist and public isolation test |
| Minimal dependencies/no domain runtime | passed | isolated oracle only; cleaned imports audited |
| Local executable mathematics/call roles | passed | explicit source/evaluator operations and AST/import audit |
| Full test suite | passed (19 tests, 39.96 s) | neutron_full_final.log: active pipeline plus all contracts |
| Active notebook executed/visual inspection | passed | validated nbconvert log, executed archive, reviewed figure |
| Metric provenance/formulas/thresholds | passed | two upstream seeds, frozen criteria, fresh scores above |
| README/plans match source | passed | semantic review above |
| Inventory/license/archive | passed | task LICENSE, external SHA256 archive index |
# Independent parent acceptance, 2026-10-20

The coordinating parent reran the full task suite in the shared research runtime: **19 passed in40.63s**, exit0. A new standalone Judge invocation from the batch archive working directory passed the existing actively reconstructed output. Parent AST/import/call-role, README/public-array, executed-notebook and file-inventory checks found no errors; the required MIT LICENSE is permitted by AGENTS.md. The parent actually viewed the complete three-panel executed scientific figure (measured/initial/original/fresh reflectivity, original/fresh SLD and weighted residuals) and reviewed the mathematical/interface scope. The source tutorial's maximum-likelihood stage is the selected scope; no unexecuted posterior is represented as obtained. The approximately2.30 reduced log chi-square and systematic residual pattern remain explicit model limitations, not independent microscopic truth. Acceptance evidence is `<batch>/neutron_reflectometry_thin_film-parent-audit.json` and `neutron_reflectometry_thin_film-fresh-independent-judge.json`, where `<batch>` is the private `research_batch_20261020` archive. Old automated suggestions for six numbered notebook headings or at least two separate PNG figures are not AGENTS.md requirements; the scientific story and required comparisons were assessed directly.



## Parent-review contract correction: truth precedence (2026-10-20)

After the independent parent review, the standalone Judge now explicitly uses `evaluation/ground_truth.npz` when that exact file exists, otherwise `evaluation/baseline_reference.npz`. There is no filename/key guessing and no fallback from malformed truth to a baseline. This task still has no actual ground truth and keeps its validated baseline unchanged. A new isolated contract case creates both assets, proves that the truth is selected, and proves that an existing malformed float32 truth produces evaluator error/2 with empty metrics instead of silently using the baseline. This does not change observations, method mathematics, numerical reference, thresholds or the present baseline-branch results.

The entire task suite was rerun with the validated research Python and private basetemp: **20 passed in 75.40s**, exit0. Log: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/optics_astronomy/neutron_contract_final.log`; SHA256 `ba2048f33b2b53fbf77d903206f83a479a622118e9aecaae36f0fcb13e89745e`. Earlier validation evidence and parent review above are preserved; this note supersedes the earlier test count for the final shipped Judge contract. The notebook's current baseline evaluation is numerically unchanged, so no redundant inverse/source rerun is claimed.


Parent independently verified the added actual-truth precedence and malformed-hidden-reference contract test:1passed in0.23s. Combined with the previously parent-verified19 tests, all20 distinct cases have independent parent verification. Source inverse, baseline branch, numerical boundaries and notebook scientific results are unchanged. Agent reran the complete20-case suite after the minimal Judge contract change.
