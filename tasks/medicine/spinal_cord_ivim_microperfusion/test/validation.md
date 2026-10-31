# Validation and provenance

## Bound real observations and method

Lévy et al., Magnetic Resonance in Medicine2020, DOI10.1002/mrm.28195. Original author repository https://github.com/slevyrosetti/ivim-toolbox exact remotely verified revision42809f1681102e654177105fd76d635c60b2d413, MIT license retained. Original measured test_data/dwi_rl.nii.gz and author cord mask/b-values are preserved in the private archive. These are author-processed real human spinal-cord7T MRI observations, not a simulation or independent parameter truth. Native acquisition dimensions128x134x6x11 and all2523 author-mask voxels are retained. No voxel is removed based on fitted values or cleaned performance. Public data are losslessly converted NPZ; NIfTI inputs, full original source, paper evidence and logs remain maintainer-only.

Private archive root: research_batch11_20261027/bio_med/archive/ivim-toolbox relative to maintainer directory. archive_index.json records SHA256 for original code/data/capture scripts/fixtures/logs. remote_metadata.json records the actually verified remote revision. oracle_environment.json records Python3.11.11, NumPy2.4.6, SciPy1.17.1, isolated lmfit1.3.4 and nibabel5.3.2, Windows CPU and per-voxel seeds20261027+native index. No upstream packages are imported by cleaned runtime. Native input SHA256926b8bc637d2eca89a96303644d70faf151d56ac5971af80b15ee6856494c764.

## Independent original execution before extraction

Private oracle.py executes the original joint one-step and fixed-D two-step fitting functions on every bound voxel and captures intermediate/final arrays. Full primary run completed2523voxels in1078.96s without failed fits. The first two-step attempt failed in legacy plotting-only np.linspace APIs; oracle_comparison_posix.log documents the successful original POSIX no-plot branch with unchanged numerical functions. No failed plotting execution is represented as a successful original run. one_step_results.json and two_step_results.json are complete original parameter results; six stage NPZs store initial/updated parameters, predictions, residuals and chi-squares at native indices0,1261,2522.

repeat_check.py/repeat_diagnostics.json retain33-voxel repeated and alternate-seed diagnostics. alternate_full.py then independently repeated all2523 voxels with alternate seeds; interrupted partial attempts remain explicitly separate from the completed alternate_full_restored.log run425.42s. alternate_seed_results.json is the complete independent measurement. Historical matching-seed runs reproduced the original values; this is a provenance observation, not a current seed-to-final-parameter test criterion or proof of physical accuracy. Alternate seeds expose vascular-parameter nonidentifiability.

## Frozen scoring and identifiability

frozen_acceptance.json was written before local extraction. All four quantities S0,D,f,Dstar remain required. Reference-f weights Dstar because the signal derivative vanishes at f=0. Reference (1-exp(-bmax Dstar)) weights fraction because the two compartments coincide as Dstar tends to zero. These are hidden, fixed scoring definitions, not target-derived public calibration. Product fDstar and reprojection of all11 observations are additionally mandatory. All439 original near-zero-f voxels remain in the benchmark. Unweighted Dstar instability is disclosed, not suppressed by a permissive raw-Dstar threshold or a post-hoc mask. The full alternate-seed baseline establishes the frozen weighted bounds; no cleaned output selected them.

## Equation and function traceability

| Stage | Original | Local | Independent evidence |
|---|---|---|---|
| Native measured image/mask conversion | author test_data NIfTI/bvals | preprocessing.load_data | hashes, shape and full support tests |
| Biexponential signal and derivatives | ivim_fitting.py model functions | physics_model | analytic limits and derivative tests |
| High-b tissue initialization | original one/two-step branches | solvers.fit_voxel | six original stage captures |
| Joint bounded fit and fixed-D comparison | original lmfit DE/scalar fitting | local explicit bounded parametrization/objective/updates with general SciPy optimizer | strict deterministic stage0; historical stage1/2 parameters used as fixed signal/residual/chisqr inputs; fresh DE shape/box/objective contracts and unchanged full weighted Judge |
| All native voxels and four parameter arrays | oracle complete results | src.main.run | complete fresh reconstruction evaluated by pre-frozen alternate-seed-supported weighted physical metrics and all-observation residual, not seeded point equality |
| Weighted metrics and all-b consistency | pre-extraction independent repeats | standalone evaluation/judge.py | analytic metric, malformed-input, poor-output tests |

Selected original fitting convention, bounds, transformed parameters, initialization, seed mapping, stopping and units are detailed in plan/approach.md; signatures/caller flow in plan/design.md. The notebook actively reconstructs all2523voxels and independently computes fixed-D and monoexponential alternatives. Archived original one/two-step outputs are labeled original, never substituted for fresh results. Canonical notebook presently contains7 executed code cells and7 generated PNG panels. fresh_measurements.json reports actual primary329.32s and comparison221.68s from that execution, plus all metrics and comparison residuals. No independent true perfusion map exists. Dstar is pseudodiffusion, not directly a measured blood-flow speed.

## Reference consumers and gate status

evaluation/baseline_reference.npz stores original method quantities and fixed evaluation weights, never physical truth. notebooks/reference_outputs/upstream_one_step.npz and upstream_two_step.npz are consumed by notebook comparison panels. fresh_comparisons.npz/fresh_measurements.json identify fresh full runs; consumers are notebook review and validation. Six test/fixtures stage files are consumed by comparing_test/test_parity.py; no cleaned-generated fixture is called upstream parity. Source/generator/solver/Judge/notebook have no cross-task or domain-library imports. No synthetic generation is performed.

| Gate | Status |
|---|---|
| Exact source/data, original full runs and pre-extraction criteria | passed with separately documented original plotting exception |
| Local explicit mathematics and stage parity | existing implementation; independent resumed suite running |
| Full pipeline/standalone Judge and negative/error contracts | fresh stored result passes; resumed suite running |
| Active full notebook | completed canonical execution with actual runtimes and7PNG outputs |
| Final visual, source/plan and release inventory review | pending resumed review and independent parent acceptance |

The task is not declared accepted until pending checks finish. Caches remain private and must be excluded during integration.

## Frozen independent-run measurements and bounds

```json
{
  "independent_full_alternate_seed_metrics": {
    "S0_rmse": 0.3405738877538881,
    "D_rmse": 8.332273081551074e-06,
    "fraction_weighted_rmse": 0.00849507001537094,
    "Dstar_weighted_rmse": 0.0016229982628991188,
    "fDstar_rmse": 0.00011263546582884283,
    "observation_relative_l2": 0.39661039338303
  },
  "boundaries": {
    "S0_rmse": 0.6811477755077762,
    "D_rmse": 1.666454616310215e-05,
    "fraction_weighted_rmse": 0.01699014003074188,
    "Dstar_weighted_rmse": 0.0032459965257982375,
    "fDstar_rmse": 0.00022527093165768567,
    "observation_relative_l2": 0.43627245704530293
  },
  "unweighted_parameter_rmse": [
    0.340573887753888,
    8.332273081551086e-06,
    0.013152722487646259,
    0.008335538612899434
  ],
  "parameter_order": [
    "S0",
    "D",
    "Fivim",
    "Dstar"
  ],
  "zero_fraction_count": 439,
  "stage_atol": 1e-08,
  "stage_rtol": 1e-06,
  "rationale": "Reference f weights Dstar because signal derivative vanishes at f=0; reference 1-exp(-bmax Dstar) weights f because components coincide as Dstar approaches zero. Weights are hidden evaluation definitions, never public calibration. Product fDstar and all-observation reprojection required. Full native support retained. Twice independently observed seed variation plus small floors, frozen before local extraction. Initial unweighted Dstar instability retained in repeat_diagnostics.json."
}
```

## Packaged array hashes

| File | SHA256 |
|---|---|
| `data/raw_data.npz` | `b597ae916ddf944aff7ec13dc36704e6368c57cde52c77a930958f474258cca8` |
| `evaluation/baseline_reference.npz` | `6efbc642b0eb50a55b285df365e411d1946b8afbd0a69a81b928bb1f30af3b1d` |
| `notebooks/reference_outputs/fresh_comparisons.npz` | `5bf7cc507722355500bfb8abc065aa47c960ad8765eee6afe957f4e17b8d7472` |
| `notebooks/reference_outputs/upstream_one_step.npz` | `6efbc642b0eb50a55b285df365e411d1946b8afbd0a69a81b928bb1f30af3b1d` |
| `notebooks/reference_outputs/upstream_two_step.npz` | `304d26a8738d5df21d69bb9518a92ae63b8e843707e576e8f10c61582fda09e9` |
| `output/reconstruction.npz` | `6efbc642b0eb50a55b285df365e411d1946b8afbd0a69a81b928bb1f30af3b1d` |
| `test/fixtures/one_step_stage_0.npz` | `7f81b8fd7e74d8b842779d1f7466a46dec76a9a74688bc6f432b13c8969511b2` |
| `test/fixtures/one_step_stage_1261.npz` | `d2c438795579eed4801fec1b5dee7169a89dd2ff8b85db712a4722ee7b6b7f8d` |
| `test/fixtures/one_step_stage_2522.npz` | `458458d7c805b832c4ff6f380b1a1fdc4e899552774a27ac4107ac1867c274ab` |
| `test/fixtures/two_step_stage_0.npz` | `2562b44f84c99bf14e47e1b3a58d2f7fd59ceca719111ec56717bf92c008926a` |
| `test/fixtures/two_step_stage_1261.npz` | `f99a2adb1f1cd8c660161243661a4860b44bfc49855cd3819587779c89161b93` |
| `test/fixtures/two_step_stage_2522.npz` | `84bb5bff7e8599d18cccc9899b9e67fd5107b11ea08286450c0d103a7c138cd1` |

## Resumed visual/source review

All seven canonical executed PNGs were individually viewed after restoration: six native acquisition slices and mask, all24 parameter maps, both full original/fresh parameter comparisons, three measured voxel spectra, all-observation residual/zero-f coupling panels and original/fresh/mono comparison bars. No placeholder result or visually hidden failed method was found. Coordinates in selected-voxel titles expose NumPy integer type text, a cosmetic limitation. Kernel metadata was made portable python3 without altering executed cells or outputs. Local Minuit sine box transforms, CG/DE objectives and same fixed full-voxel seed indexing match the described interfaces; no truth/import shortcut. Full independent resumed pytest remains running and is not yet claimed passed.

## Resumed full validation result

Full `python -m pytest test -q --basetemp <private archive>/ivim_review_tmp` completed exit0:20passed in354.05s. This includes frozen original six-stage cases, full2523voxel reconstruction and standaloneJudge, plus18 unit/metric/error-contract tests. Unit-only repeat18passed2.48s. All prior resumed-suite pending entries are resolved. Fresh notebook execution was already complete and all7figures have now been inspected. Ready for independent parent acceptance; no parent acceptance is inferred here.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 7 computed PNG outputs, 1 formulas. Private evidence: research_batch11_20261027/acceptance/spinal_cord_ivim_microperfusion/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Testing-tail stochastic compliance correction (2026-10-28)

Root authorized only test/comparing_test/test_parity.py and this validation update. No source, acquisition, observation, method option, metric boundary, reference array, output contract, original executed notebook or accepted inventory was changed. AGENTS Testing forbids using matched random seeds to require exact stochastic outputs. Differential-evolution outputs here are parameter point estimates rather than Monte Carlo posterior samples, but their random-search paths still make seeded final-coordinate equality an inappropriate test oracle. In particular, weakly identifiable Dstar is not measured truth.

The deterministic high-b log-signal stage0 retains the original six-case rtol=1e-6/atol=1e-8 comparisons for parameters, prediction, residual and chi-square. For original stages1/2, the source-recorded parameter vector is now an explicit fixed input to a deterministic forward check: signal, observation-minus-signal residual and sum of squared residuals are recomputed and compared with the independently frozen original outputs at the same unchanged tolerances. These fixed inputs do not replace a fresh fit.

The same six measured cases additionally run fresh one-step/two-step fits with a different diagnostic RNG stream,20261029+native index rather than the original20261027+index. Their DE estimates are checked for float64 shape, finiteness and the actual method-specific box. The second search box is derived from that fresh run's own first estimate, including the original zero-f special case. The fixed-D comparison retains its deterministic D identity. Fresh predictions, residuals and objective values must agree with their own physical parameters and measured observations. No fresh DE coordinate is required to equal a historical seeded estimate, and no new per-voxel quality tolerance was invented.

Scientific quality still requires the existing complete2523voxel pipeline and unchanged standalone Judge. Its independent pre-extraction alternate-seed-supported S0/D errors, physically weighted f/Dstar errors, fDstar product and complete11bvalue observation residual all remain mandatory at exactly the frozen boundaries. Aggregate full-support bounds are not misapplied as individual-voxel limits. All439original near-zero-f voxels remain; no support, difficult method or comparison was removed. Reference agreement and residual quality do not establish histological/perfusion truth.

The six original method_stage case bundles preserve source field names and hashes. Test-local input_observation/input_parameters, config_bvalues/config_lower/config_upper and output_prediction/output_residual/output_chisqr aliases document current call roles. Original stochastic estimates are historical measurements reused as controlled inputs, not exact expectations of the new random search. This avoids falsely renaming a mixed source-case bundle as a pure output fixture.

Independent complete pytest: `======================= 20 passed in 280.58s (0:04:40) ========================`, exit0; subprocess elapsed 280.874287500s. Command: shared Python `-m pytest test/ -v -p no:cacheprovider --basetemp` with the private ivim_testing_compliance/pytest_tmp directory. This really reran the whole2523voxel benchmark through the full-pipeline test. The standalone evaluator then scored that newly generated temporary prediction, exit0, in 0.178902800s; it did not substitute the old stored output. Exact commands, stdout/stderr, fresh-prediction path and report are in maintainer ivim_testing_compliance/execution.json and judge_report.json. Numerical source and the scientific method are unchanged, so no additional notebook execution is claimed or required for this testing-only correction. The canonical seven-cell/seven-figure notebook retains its original measured execution identity. Root separately reviews and updates the frozen inventory.
