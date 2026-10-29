# Validation and provenance

## Binding and realism — passed

Paper: Stone AJ and Blockley NP, *NeuroImage* 147 (2017) 79–88, DOI 10.1016/j.neuroimage.2016.11.057, published article archived as `sqbold_paper.pdf` (ORA full text). Original author implementation: https://github.com/fmriphysiology/streamlined_qbold_tools at remotely verified commit **0778fb40c3446c4c28e04f86e2ec3f71d731d15b**; remote tree archived in `sqbold_tree.json`. Selected function is ase_qbold_3d.m, MIT copyright (c) 2016 fmriphysiology. LICENSE preserved. Not a claim of an unmodified entire FSL preprocessing pipeline.

Data: Stone/Blockley ORA deposit DOI 10.5287/bodleian:E24JbXQwO, https://ora.ox.ac.uk/objects/uuid:177afade-8599-4d9c-959e-26e1426ec486 . Source record `ora_record.json`/`.html`; data download member URL https://ora.ox.ac.uk/objects/uuid:177afade-8599-4d9c-959e-26e1426ec486/files/m9e78c6f8aaa38807e44bfa0bd307a984 . License Open Data Commons Attribution 1.0; attribution preserved in NOTICE. Classification: **verified_real**, author-released filtered magnitude images, not untouched scanner k-space. File-linked dataset README identifies Siemens Magnetom Verio 3T, body transmit/32-channel receive, 24 actual displacements, nulling times and acquisition geometry. No evidence here independently authenticates the vendor reconstruction/filter details; label filtered is retained.

Bound subject-01 complete CSF-null, no-null, GM-null and WM-null acquisitions, each 64×64×40×24 float32. The overall 253 MB ZIP download was intentionally stopped after all subject-01 members were complete. **sqbold_data.zip is a partial archive, not a verified complete ZIP.** Complete individual local ZIP records were deflated and validated against their data-descriptor CRC32 and uncompressed lengths. All original 40-partition NIFTIs are retained privately; final recheck `sqbold_complete_member_checks.json` confirms CRC/length and records SHA256. No truncated member is used. Public NPZ contains all four complete acquisitions after explicitly documented four-partition averaging by this task. Source slope 1/intercept 0, source byte order little endian float32; float32 arithmetic follows sequential author-script sums/division, then lossless promotion to float64 for both original and cleaned inverse.

Closest old tasks: DSC bolus-residue maps estimate contrast-bolus flow/residue, ASL maps use labelled-water delivery, and mouse oxygen-challenge spectra estimate optical hemoglobin mixtures. This binding uses asymmetric spin-echo microscopic dephasing and intercept separation to estimate reversible relaxation/apparent deoxygenated venous volume. A new subject or regularizer is not the basis for distinction.

## Scope, ambiguities and identifiability — passed with documented limitations

Original-only screening identified unstable oxygen ratios. Before local extraction the contract `sqbold_pre_extraction_contract.json` fixed **R2-prime and apparent DBV** as outputs. OEF/dHb are diagnostic derived ratios only. The mask is the public central ellipse/slab/SNR rule, contains no reference-based exclusions and retains negative fitted parameters. Its counts are 4425/4855/3206/1672 for CSF/no/GM/WM nulling. The paper discusses overestimated DBV, nonvascular susceptibility and diffusion; numerical agreement is not physiological accuracy. No independent truth exists for these targets, so baseline_reference is correctly used.

Source x.zaverage averages four partitions but hardcodes a 7.5 mm output slice spacing; these source files have 1.25 mm partitions and documented 5 mm prescribed slices, so actual task spacing is 5 mm. Source data README suggests summation instead of averaging: the factor four changes log amplitude only, independently tested. No mcflirt, BET, FLIRT, FSL smoothing, tissue segmentation or registration was run. Paper cohort tissue statistics therefore cannot be claimed. Native coordinates are retained; condition comparisons are descriptive and not registered paired tissue effects. Source `dChi0` comment says ppm but numeric .264e-6 is the dimensionless susceptibility used with Gauss units; local units explicitly match the formula. Source montage labels DBV/OEF percent despite fractional outputs; local output remains fraction, with plotting multiplication by 100 only where explicitly labelled percent.

## Original execution and independent fixture order — passed

Private archive root is this batch's `bio_med/` outside final task. Original source is byte-identical in `sqbold_oracle/ase_qbold_3d.m`. WSL Ubuntu Octave 8.4.0, x86_64-pc-linux-gnu, system BLAS `-lblas` and LAPACK `-llapack`; OMP/OPENBLAS threads 1, CPU only, no random operations or seeds. Installed Octave control 4.0.1/image 2.14.0/signal 1.4.5 were inspected but not needed/loaded. No package installation or shared runtime modification.

`prepare_sqbold.py` reads archived NIFTI bytes and creates exact averaged float32 observations and losslessly promoted MAT data. The IO-only read_avw adapter loads those arrays, replacing unavailable FSL file reading; a display-only brain_montage no-op disables visualization without changing numeric execution. The original function itself is unchanged and executes on the **exact bundled averages**. Original return fields are captured separately for all four conditions. A second capture function inserts only an intermediate `save` and changes the function name; the saved original X, log S, Y and p are independent stage fixtures. The instrumented repeat agrees with the untouched original exactly. `sqbold_fixture_freeze.json` fixes fixture hashes and repeat difference before local source was written. There is no claim that the adapter is the original NIFTI/FSL implementation.

The original command is `wsl -e bash -lc 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 octave --no-gui --quiet .../bio_med/sqbold_oracle/capture.m'`; exact expanded command, runtime, exit status and task-run commands are in sqbold_validation_records.json. All exit 0. Original logs sqbold_original.log; initial and repeated outputs in sqbold_oracle/*_result.mat and *_repeat.mat. WSL warns about localhost NAT configuration and Qt runtime-directory permissions; no numerical warning. Local logs sqbold_cli.log, sqbold_judge.log, sqbold_tests.log and sqbold_notebook.log.

## Equation / function / test traceability — passed

| Paper stage | Original stage | Local function | Validation |
|---|---|---|---|
| Fixed TE ASE and positive long-time regime, Eq 7a | tau construction, Tc=.015 | preprocessing.TAU; physics_model.design_matrix | Exact original tau/X; README constants test |
| Log signal and spin-echo intercept difference, Eq 8 | log(V), Y, X row [0,0,1] | log_measurements; design_matrix | Original logs/Y; zero handling; independent slope/intercept case |
| Unweighted fitting of r and DBV | p=lscov(X,Y) | fit_signals, NumPy lstsq | Original five-voxel stages per condition and full four-acquisition output; normal equations |
| Magnitude model | X*p in original linear regression | forward_log | Independently specified analytic curve, units and intensity invariance |
| dHb and OEF, Eqs 12–13 | original ratios with Hct=.34 | oxygen_diagnostics | Constants/units tests; notebook original comparisons; signed/nonfinite diagnostics |
| Four nulling comparisons | separate source scans | reconstruct and active notebook | All four kept, residuals/maps/distributions/fit-range sensitivity |

No fake generator, training, opaque solver import, or task-external runtime dependency. Only NumPy lstsq is used as a general linear algebra primitive; task acquisition/design/log/selection/output rules are local executable mathematics. Import/call audit is recorded in sqbold_validation_records.json; source, Judge, tests and notebook inspected. Judge independently loads assets and public mask without solver imports.

## Tolerances and result provenance — passed

Pre-extraction stage/full parity tolerance absolute/relative 1e-10; original deterministic repeat maximum difference zero. Fixed Judge physical-unit RMSE per condition: R2-prime <=1e-6 s^-1, DBV <=1e-8 fraction; outside-region max <=1e-12. Nonzero numerical boundaries are justified by deterministic linear algebra and independent rerun; they are not clinical accuracy thresholds. Signed fields and reference variability make RMSE meaningful. No scaling/alignment or post-hoc fitted-value mask. Each required quantity and condition must pass independently. Standard NCC/NRMSE are not substituted for physical parameter errors.

`evaluation/baseline_reference.npz` is the original masked R2-prime/DBV, not truth. `notebooks/reference_outputs/upstream.npz` contains unmasked original R2-prime/DBV/OEF/dHb fields for notebook comparison. `test/fixtures/upstream_stages.npz` contains original intermediate arrays for comparing tests (five fixed indices spanning background and interior). `notebooks/reference_outputs/cleaned_measurements.npz` is separately identified fresh full cleaned output and residuals from final validation; the notebook actively recomputes rather than loading this cache. Output/reconstruction is the CLI submission. No historical paper numbers are relabelled measurements of this task.

## Tests, notebook and semantic review — passed

19 tests pass: independent original stages, complete pipeline, analytic inverse and normal equations, intensity/time-unit invariance, forward and oxygen units, input/zero-log handling, public mask invariance, independently known RMSE, Judge quality failures for each quantity, missing keys/files/assets, shape/dtype/complex/NaN/Inf rejection, malformed config, foreign CWD and report-write failure. Initial test invocation hit an unrelated inaccessible shared Windows pytest temp directory; rerun uses private --basetemp. Initial scalar oxygen diagnostic divided Python floats at zero; converting scalar inputs to NumPy fixed consistent NaN/Inf semantics and all tests pass. Neither issue changed the bound inverse or tolerance.

Native nbconvert executed every selected reconstruction and sensitivity cell with 180 s timeout (ample relative to observed runtime). All eight figure groups were opened and visually inspected: measured frames/public region; original/fresh/difference R2-prime; original/fresh/difference DBV; actual measured/model curves at median/90th residual quantiles; log residual maps; full-range signed distributions; actual cutoff and labelled ±1% amplitude perturbation; full-range symlog OEF/dHb distributions and failure tables. Initial long titles were split and NumPy scalar coordinate repr shortened, then notebook re-executed and changed figures re-inspected. Required Problem/Data/Physics/Method/Results/Conclusion story is present; every fresh plotted result is computed. No paper comparison group was dropped. Native source notebook keeps executed outputs; duplicate executed notebook moved outside task. Benign frozen-module, Windows zmq selector and local TCP warnings are recorded in notebook log.

README and plans were semantically checked against source: signs, tau seconds, fraction/percent, susceptibility/Gauss constants, 14 rows, objective unweighted/no normalization, direct noniterative solve, mask, no anatomical registration, and all signatures/defaults match. README explains all public arrays/units/acquisition/output constraints, contains no hidden references or GitHub URLs. Requirements only actual pinned NumPy/Matplotlib. Display equations in README use fenced math; notebook uses display dollars with self-contained symbols.

## Completion gates

All mandatory gates passed: primary source and real-data evidence; exact original and cleaned runs; independent stage/full parity and physics; constants/dtypes/units; CLI and standalone multi-output Judge; malformed and poor-output rejection; public allowlist sufficiency/no leakage; dependency/local mathematics audit; tests; active nbconvert and eight visual inspections; provenance distinctions; plans/README consistency; inventory/license cleanup. **Physiological validity, original FSL preprocessing, anatomical registration and stable oxygenation are unverified and not claimed.** Final parent acceptance is separate from this worker validation.

## Exact measured run evidence

```json
{
  "original": {
    "command": [
      "wsl",
      "-e",
      "bash",
      "-lc",
      "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 octave --no-gui --quiet /mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch10_20261027/bio_med/sqbold_oracle/capture.m"
    ],
    "seconds": 0.7929816000396386,
    "exitcode": 0
  },
  "cli": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "seconds": 0.2961566000012681,
    "exitcode": 0
  },
  "judge": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "seconds": 0.22178369999164715,
    "exitcode": 0
  },
  "tests": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp=D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch10_20261027\\bio_med\\pytest_sqbold_final"
    ],
    "seconds": 1.7315721999621019,
    "exitcode": 0
  },
  "notebook": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/brain_reversible_relaxation_venous_volume.ipynb",
      "--output",
      "sqbold_final.ipynb",
      "--output-dir",
      "output",
      "--ExecutePreprocessor.timeout=180"
    ],
    "seconds": 11.562778000021353,
    "exitcode": 0
  },
  "figures": 8,
  "runtime": {
    "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
    "numpy": "2.4.6",
    "matplotlib": "3.11.1"
  }
}
```

Fresh Judge measurements:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "r2prime_rmse_0": 2.357607389099141e-14,
    "dbv_rmse_0": 5.421559953992158e-15,
    "outside_max_0": 0.0,
    "r2prime_rmse_1": 2.5696404719617555e-14,
    "dbv_rmse_1": 6.348489272330082e-15,
    "outside_max_1": 0.0,
    "r2prime_rmse_2": 1.9609717631218863e-14,
    "dbv_rmse_2": 4.816580814932285e-15,
    "outside_max_2": 0.0,
    "r2prime_rmse_3": 2.3124976492920643e-14,
    "dbv_rmse_3": 5.507707585944181e-15,
    "outside_max_3": 0.0
  },
  "thresholds": {
    "r2prime_rmse_0": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "dbv_rmse_0": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "outside_max_0": {
      "comparison": "<=",
      "boundary": 1e-12
    },
    "r2prime_rmse_1": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "dbv_rmse_1": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "outside_max_1": {
      "comparison": "<=",
      "boundary": 1e-12
    },
    "r2prime_rmse_2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "dbv_rmse_2": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "outside_max_2": {
      "comparison": "<=",
      "boundary": 1e-12
    },
    "r2prime_rmse_3": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "dbv_rmse_3": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "outside_max_3": {
      "comparison": "<=",
      "boundary": 1e-12
    }
  },
  "errors": []
}
```

Complete source-member identities:

| Member | Bytes | CRC32 | SHA256 |
|---|---:|---|---|
|streamlined-qBOLD_study/sub-01/func/sub-01_task-csfnull_rec-filtered_ase.nii.gz|3697193|596e06ca|1c8f50440975ce82e4c72fad0c95a17114b9c099b6fe0b452bd06b195e42fdad|
|streamlined-qBOLD_study/sub-01/func/sub-01_task-gmnull_rec-filtered_ase.nii.gz|3583947|91eab150|d68f26ddc074897777a891fa2d12fdbd53177560d5b89b6aba77d32c02445d0b|
|streamlined-qBOLD_study/sub-01/func/sub-01_task-nonull_rec-filtered_ase.nii.gz|4445794|b0e4c766|01125e7dc06dbcf3194865469f43341db126d93be0d6a8216f11ecf9ca76435e|
|streamlined-qBOLD_study/sub-01/func/sub-01_task-wholebrain_rec-filtered_ase.nii.gz|429383|08b54e08|055dfb0d2343c850a78f43e9a7b75b0b7e37d008dc6b1988ab1b506625d07677|
|streamlined-qBOLD_study/sub-01/func/sub-01_task-wmnull_rec-filtered_ase.nii.gz|3791005|2bd09fa3|5b7996306093ce8229c180c0141730b6a2c4a154a48153aeb6be28d367570113|

Private archive/source/reference hash index:

| Path relative to bio_med | SHA256 |
|---|---|
|sqbold_oracle\ase_qbold_3d.m|971a18aba0a69ae3c2b08d5a3e5638c7f4e3813742b292477d1af26006885a4a|
|sqbold_oracle\ase_qbold_capture.m|03d515afcb73b9c2e6d58eff8552ed41acff7c063cc091fc878b3a1a3b2c2fc2|
|sqbold_oracle\brain_montage.m|54b3057b801cb1c42173e2141373d847cbf5a34c49653b85e49d7cfbf6e751d9|
|sqbold_oracle\capture.m|c1f9f0587b4a8d732ac3a728e1084124ffbccd536052acd0e32d360dd3897de8|
|sqbold_oracle\capture_stages.m|29a51a79399712cebd0ecfb8c22f1996a4596ad4edc3cc7a17079fefdacbb726|
|sqbold_oracle\read_avw.m|0b8ef687ae4d5a7abc88e32325333edadde6b1cd30a1569d77ce0320f70fff51|
|sqbold_paper.pdf|94e7f9c124089847539420d1891efb1d71d9d831aa94927bb6f9ad5919d0a7ac|
|sqbold_tree.json|91fe429e1f283c6ec737f91e6e1dfe4c43ba50f1faa81ceddbece2cd59e8255e|
|sqbold_pre_extraction_contract.json|af7aee2b783ec1d08690568da0d5512fcb3f09072301545ee909f10bd83f99f8|
|sqbold_fixture_freeze.json|233ebe25352a4a0f708fc7e485ab069632f41820fb3d922e9f06883038b4ee9a|
|tasks\brain_reversible_relaxation_venous_volume\data\raw_data.npz|063455e9df7dce4e6edcdf99f75862211b8868c1dcfe55ff1da1500c08ec9823|
|tasks\brain_reversible_relaxation_venous_volume\evaluation\baseline_reference.npz|05d2f10c3c359e5ad4cf874f17927f26a07f1e1fed9749e6c6ca4453a98ae982|
|tasks\brain_reversible_relaxation_venous_volume\notebooks\reference_outputs\cleaned_measurements.npz|21e7686b68c3546ce4f5cb6f8d5ff1756bd730b3a25d8bdee444a28f236beb7a|
|tasks\brain_reversible_relaxation_venous_volume\notebooks\reference_outputs\upstream.npz|0a0c8af58234ec4c8bbc1814df67f043def54adb9781e3d282b640c28d2e696a|
|tasks\brain_reversible_relaxation_venous_volume\output\reconstruction.npz|e2698c3c5313f07312b29489b23fdf5c8e5362dcb0e67157e5bf1ac65bfb9118|
|tasks\brain_reversible_relaxation_venous_volume\test\fixtures\upstream_stages.npz|7b1edbdc2cff8b492d9d24d3b4667f1338ec552dc7b1b211485b8aa260761b09|

Original-only screening, frozen before extraction:

```json
{
  "scope": "R2prime and DBV fields for4conditions; OEF/dHb exploratory diagnostics only, dropped as primary before extraction after original identifiability screen",
  "conditions": [
    "csfnull",
    "nonull",
    "gmnull",
    "wmnull"
  ],
  "central": "((x-32)/15)^2+((y-32)/18)^2<1,2<=z<=7",
  "mask": "central AND spin_echo>.5*percentile95(spin_echo) AND all24signals>0",
  "outputs": {
    "reconstruction": "R2prime s^-1 (1,4,64,64,10)",
    "dbv": "fraction (1,4,64,64,10)"
  },
  "parity_atol": 1e-10,
  "parity_rtol": 1e-10,
  "r2prime_rmse_boundary": 1e-06,
  "dbv_rmse_boundary": 1e-08,
  "outside_max_boundary": 1e-12,
  "screen": {
    "csfnull": {
      "valid_count": 4425,
      "log_rms_median": 0.012494647527368364,
      "negative_r2p_fraction": 0.00903954802259887,
      "negative_dbv_fraction": 0.08858757062146892,
      "r2p_median": 3.0673176815386114,
      "dbv_median": 0.04842429315871133,
      "oef_extrema": [
        -888.2336899315792,
        231.2823780168148
      ]
    },
    "nonull": {
      "valid_count": 4855,
      "log_rms_median": 0.013082084776529881,
      "negative_r2p_fraction": 0.01153450051493306,
      "negative_dbv_fraction": 0.026364572605561278,
      "r2p_median": 3.5198802280349213,
      "dbv_median": 0.058543907418432406,
      "oef_extrema": [
        -212.07612141282092,
        153.68661015037384
      ]
    },
    "gmnull": {
      "valid_count": 3206,
      "log_rms_median": 0.028707981360304797,
      "negative_r2p_fraction": 0.07080474111041797,
      "negative_dbv_fraction": 0.13911416094822207,
      "r2p_median": 3.829172952953801,
      "dbv_median": 0.06868736550707788,
      "oef_extrema": [
        -604.8978100610625,
        44.002862609077724
      ]
    },
    "wmnull": {
      "valid_count": 1672,
      "log_rms_median": 0.02757681476979533,
      "negative_r2p_fraction": 0.09150717703349283,
      "negative_dbv_fraction": 0.21650717703349281,
      "r2p_median": 5.095941488885217,
      "dbv_median": 0.07993247899774047,
      "oef_extrema": [
        -2982.35295971451,
        122.18691211669022
      ]
    }
  },
  "frozen_before_local_extraction": true
}
```


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 8 computed PNG outputs, 33 formulas. Private evidence: research_batch10_20261027/acceptance/brain_reversible_relaxation_venous_volume/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
