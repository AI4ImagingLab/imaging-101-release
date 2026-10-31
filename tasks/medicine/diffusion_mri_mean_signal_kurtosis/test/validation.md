# Authoritative provenance and validation evidence

All mandatory gates below passed for the bound real-data task on2026-10-20. This is a newly extracted local standard-method implementation, not a layout-only wrapper. References are independently reproduced method measurements; no reconstruction is relabeled as material ground truth.

## Fixed physical scope, sources, versions and licensing

- Repository: https://github.com/dipy/dipy . Original release1.12.0 is an annotated tag. Remote tag object af546c7338144af9b2f3e2e04b19f183ad8fc667 dereferences to actual commit407e9319cbfe3d05304682d317363ecd09a83fdc. Both API responses are private dipy_tag.json/dipy_commit.json. The installed original msdki.py was normalized-newline text-compared with exact-commit remotely retrieved verified_msdki.py: identical. dipy_source_verification.json records this check. Do not confuse the tag object with a code commit.
- Publication version: Henriques et al., Diffusional Kurtosis Imaging in the Diffusion Imaging in Python Project, Frontiers in Human Neuroscience15:675433(2021), DOI10.3389/fnhum.2021.675433. Fulltext dki-method.xml (PMC8327208), specifically section2.1.6 Equation18 for mean-signal cumulant expansion and WLS description. This is the mean-signal inverse, not the existing DTI tensor or HARDI orientation-distribution tasks, and no axon-fraction conversion is claimed.
- Real author-deposited observations: Hansen/Jespersen human3T multi-b experiment, Scientific Data3:160072(2016), DOI10.1038/sdata.2016.72; fulltext cfin-paper.xml (PMC5004585), specifically Human data and Usage notes. Data DOI10.5061/dryad.9bc43, mirror https://zenodo.org/records/4988044 explicitly CC0(cfin_license_metadata.json). DIPY fetcher identifies identical author filenames and MD5s. Actual files downloaded from https://digital.lib.washington.edu/researchworks/bitstream/handle//1773/38488/ with exact names __DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii/.bval/.bvec. The double slash is necessary; single slash returns HTML. The files' MD5s match original fetcher(nii9daea1d01d68fd0055a3b34f5ffd5f6e,bval3ee44135fde7ea5c9b8c801414bdde2c,bvec948373391de950e7cc1201ba9f696bf0).
- Native source is96x96x19x496, int16 magnitude counts. Predetermined central native planes8,9,10 retain every x/y voxel and encoding; no output-informed selection, resize, denoising, motion correction or new synthetic noise is applied. Public arrays retain original stored values/dtypes. Spacing and affine remain exact NIfTI precision. Paper source values in ms/um² are represented by deposited btable in s/mm²; the deposited numerical table is preserved and its units are explicit publicly. Paper reports an approximate tissue-average SNR39, not per-voxel noise calibration.
- Source code BSD-3-Clause license preserved at task root; data CC0 attribution retained in NOTICE. Upstream dependencies and NIfTI reader exist only in private oracle/conversion environments; cleaned code has no DIPY/nibabel import.

## Original run before extraction

Original DIPY1.12.0 MeanDiffusionKurtosisModel(return_S0_hat=True).fit ran on exactly the visible96x96x3x496 magnitude observations, with the public fixedb0>40mask. Command: private original Python `capture_dki.py`; firstbound run runtime2.0874111000448465seconds, exit0. Independent repeat gave maxparameter difference0. Original shellmeans/counts/design/support and allD/K/S0 were frozen. Later logged repeat(original_repeat.log) again exits0; first and repeat manifests remain distinct. Python3.11.11, NumPy2.4.6. No seed/training/stochastic optimizer is used.
All oracle environments are maintainer-only, outside the task. `dki_environment.log` records full OS/Python/NumPy/SciPy numerical backend configuration, and `dki_freeze.txt` records installed packages. Cleaned environment is Python3.11.11, NumPy2.4.6, Matplotlib3.11.1; CPU1thread via OMP/OPENBLAS/MKL settings. Full cleaned_freeze.txt and cleaned_environment.log are archived. No GPU requirement is introduced; CUDA-tagged Torch wheel was executed onCPU. The shared runtime was not upgraded and contains no selected upstream domain package.

## Independent fixture tolerances and numerical acceptance

Before extraction capture fixed per-stage/output rtol2e-9,atol1e-10. Original design/shellmeans/counts/support are exact once original operation order is preserved. Nominal final maps match exactly on the recorded backend. Independent original one-float64-ULP shellmean rerun gives relativeL2 sensitivity[D1.8936008108594573e-12,K6.2200346720496305e-12,S01.0073988438866727e-12] (dki_precision.json). Required reference-relative boundary1e-6 is a conservative six-significant-digit numeric reproducibility budget for the normal-equationSVD inverse, far below physical data/model uncertainty; it is not a physicalaccuracy claim. No boundary was relaxed after cleaned results. Independent forward-shell consistency is also required: originalRMSE5.002890195639437ADU and boundary1.1times=5.503179215203381ADU. All16shells and the same publicsupport are scored.
Metric definitions: relativeL2=norm(estimate-reference)/norm(reference) per required array, independently checked with knownvectors. No NCC/NRMSE is silently redefined or relabeled. No scale, translation, magnitude conversion, hidden crop or reference filename guessing occurs. Zero-norm references and undefinedconfigurations are rejected. Without independent truth, these boundaries test standard-method reproducibility. A forward residual is meaningful for this approximate shell-signal model and explicitly supplements referenceagreement.

## Equation/stage to executable-source traceability

| Paper equation or algorithm stage | Independently executed original source | Local implementation | Validation case/evidence |
|---|---|---|---|
| Powder-mean signal expansion, paperEq18 | dipy/reconst/msdki.py msdki_prediction and design_matrix | physics_model.predict/design_matrix | exactoriginaldesign; analytic signalrecovery and finite-difference D derivative |
| Shell means and directional measurementcount | msdki.mean_signal_bvalue | preprocessing.shell_average | full frozen means/counts; directionpermutation invariance |
| Public b0-based support | original model.fit(mask=...) API | preprocessing.signal_mask | exact support fixture; observedthreshold checks |
| Signal floor and WLSnormal equations | MeanDiffusionKurtosisModel.fit; wls_fit_msdki | solvers.fit_shells | fulloriginal D/K/S0 parity; multiplicative-gain invariance |
| K normalization by D² and baseline exponential | wls_fit_msdki postprocessing | fit_shells | independently specified analyticD/K/S0 recovery |
| Finite-b model adequacy | selected paperEq18; no compartment conversion | visualization.gaussian_fit/run_experiments | actual Gaussianbaseline/reducedshell/OLS fits; shellforwardresidual; negativeK retained |
| Independent score and forward observation consistency | standalone evaluator implementation | evaluation/judge.py | knownmetriccases, everyquantitybad/missing, malformed/evaluator/report errors |

## Ambiguities, source discrepancies and physical limitations

Background-only voxels make K numerically ill-conditioned as D approaches zero. Before final binding, the original method was rerun with public observation-only b0>40ADUsupport,9752voxels over three fullfields, rather than masking fitted K. All remainingnegativeK(~1%ofsupport depending slice) and partial-volume/model failures are retained and plotted. There is no physical truth map.

The originalpaper section2.1.6 MathML prose saysw=Ng*Sbar, whereas code uses W=diag(Ng*Sbar²). This is recorded as a paper/code weight-notation discrepancy, not silently erased. Under independent equal-variance directional noise sigma², Var(shellmean)=sigma²/Ng, and delta-method Var(log shellmean) approximately sigma²/(Ng*Sbar²), supporting code's precision matrix. The selected code-standard inverse is preserved; literal identity to the prose weight expression is not claimed. A firststage parity failure traced to b*b/6 vs original(1/6)*b² rounding; source arithmetic was corrected, with no test/threshold relaxation. The finaldesign and outputs match exactly.

## Fresh measurements and actual review experiments

The canonical notebook actively recomputes the complete bound benchmark before scoring, without loading any standard result as freshoutput. Original measurements, fresh nominal results and actual alternatives are explicitly labeled. No historicalpaper figure/number is represented as this dataset's measurement. All alternatives in visualization.run_experiments actually run; no missing method is filled withzero.

Fresh measurement record:

```json
{
  "reconstruction_runtime_s": 0.3065068998839706,
  "experiment_runtime_s": 0.6966464000288397,
  "python": "3.11.11",
  "numpy": "2.4.6",
  "metrics": {
    "diffusivity_relative_l2": 0.0,
    "reconstruction_relative_l2": 0.0,
    "baseline_relative_l2": 0.0,
    "forward_shell_rmse": 5.002890195639437
  }
}
```

Final strictJudge report:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "diffusivity_relative_l2": 0.0,
    "reconstruction_relative_l2": 0.0,
    "baseline_relative_l2": 0.0,
    "forward_shell_rmse": 5.002890195639437
  },
  "thresholds": {
    "diffusivity_relative_l2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "reconstruction_relative_l2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "baseline_relative_l2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "forward_shell_rmse": {
      "comparison": "<=",
      "boundary": 5.503179215203381
    }
  },
  "errors": []
}
```

Notebook command and execution: `python -m jupyter nbconvert --to notebook --execute notebooks/diffusion_mri_mean_signal_kurtosis.ipynb --output dki_executed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging-bio-research`. Kernel argv explicitly selects shared research Python, not the host defaultAnaconda kernel. Privatekernelspec is configured only through per-processJUPYTER_PATH. Final subprocess exit0, walltime11.862245899857953seconds; timeout180seconds is more than5times observed total and covers all scientific work, rendering and exports. The original warning about WindowsProactor/ZMQ selector fallback is archived; execution has noerror cells. Canonical notebook retains executedoutputs; temporaryexecutedcopy is private, never in output/.

## Artifact consumers and provenance roles

- data/raw_data.npz: public exact captured observations and legitimate calibration only; loaded by preprocessing, tests and active notebook.
- evaluation/baseline_reference.npz: original standard-method requiredarrays from independentcapture; used by standaloneJudge and whole-pipelineparity. ground_truth.npz is absent because actual materialtruth is unavailable.
- test/fixtures/upstream_stages.npz: frozen original stagevalues, consumed by comparing_test/test_parity.py; nativeaffine also checked againstREADME. No expectedparityarray is generated by cleanedfunctionunder test.
- notebooks/reference_outputs/upstream.npz: only original arrays with active numeric/figurecomparisonconsumers; no redundant unusedfullstage dump remains here.
- notebooks/reference_outputs/cleaned.npz: saved by active notebook from freshlocalruntime, checked by test_numerical_review.py; not independentupstreamevidence.
- notebooks/reference_outputs/experiments.npz: actual alternative/sensitivity sourcevalues, consumed by notebookplots/privatepaperfigureexports and reviewconsistencytest.
- notebooks/reference_outputs/measurement.json: actual runtime/environment/metricmeasurements from notebook, consumed by reviewtest and finalgallerymanifest.
- All originalTIFF/NIfTI/Zarr, rawdownloads, originalsource, papers, capturescripts, originalenvironmentinformation, candidate/rejectedobservations, logs and publicationexports remain in the privatearchive, indexed by archive_index.json. No sourcePDF/upstreamcheckout/metadataJSON leaks to solverallowlist.

## Commands, review and completion gates

Task cwd is diffusion_mri_mean_signal_kurtosis. Sharedruntime `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`; allcommands in final_execution.json ran with CPU1thread.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/version/revision/license/datahash/equationmapping | passed | Source sections and archiveindex; remotely verified fixed revision |
| Original exact-observationexecution before localextraction | passed | Independent originalcapture; originalstages and manifests |
| Independent parity plus applicablephysics | passed | Frozen stage/full-pipeline tests; analytic signal recovery, gain invariance, direction permutation, forward derivative and Gaussian-comparison cases |
| Constants, units, dtypes, coordinates and numericalconventions | passed | Exact unchangedarrays, metadata tests, originalsign/normalization/precisionreview |
| StandardCLI writes declaredNPZ | passed | python -m src.main; dki_cli_final.log exit0 |
| FreshJudge plus malformed/poor/evaluator/report failures | passed | python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; contracttests |
| Allowlist sufficient; no answer/seed/target-calibration/GitHub leaks | passed | README/requirements/raw audit; legitimate independentcalibration only |
| Minimal shareddependency subset and CPU execution | passed | Only actualNumPy/Matplotlib subset; no upstreamdomain import |
| Explicit localmath and call-role audit | passed | import_call_role_audit.json; genericmathprimitives only |
| All tests | passed | python -m pytest test/ -v -p no:cacheprovider --basetemp <privateunique>; 31passed, exit0 |
| Complete active notebookexecution | passed | Explicitresearchkernelspec nbconvert exit0; noerroroutputs |
| Required comparisons and renderedQA | passed | Allfinalplates inspected; nativeunits, sourcevalues, visiblefailures, titles/legends corrected |
| Historical/upstream/fresh measurements and thresholds distinguishable | passed | Original/fresh/alternativeroles, fixedmetricdefinitions and independentprecisionreruns |
| README/plans/source semanticconsistency | passed | Reviewed definitions, formulas, outputdtypes/shapes, signatures/defaults, branches, signs, dimensions and callerflow |
| Final task inventory and license/archivehygiene | passed | Onlytemplatedirs/files, no runtimeupstream/PDF/rootwrapper/metadataJSON/tempnotebook; license/notices preserved |

No mandatory failed or unverified gate remains for this selectedboundtask. Out-of-scope larger sourceexperiments, absolute physicaltruth and unsupportedphysicalcalibrations are not misrepresented as passedchecks.

## Data hashes and archive index

Public raw NPZ SHA256: `935efb04ed96b371a72bbc51f2f875634f5ee3786843c923a37ec01d032725d6`. Everypublicarray is exactly equal to independentoriginalcapture (lossless_packaging.json).

Originaldata/capture SHA256 from firstboundmanifest:

```json
{
  "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.nii": "1081b5b586a5429a6754078653ffe4ab430d348eb712bc7231c8f99919ac3eff",
  "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bval": "508d18044a4e220e28bdfcbe192238f0abb20869e36b0e2c2c4ead85fc75e614",
  "__DTI_AX_ep2d_2_5_iso_33d_20141015095334_4.bvec": "4774696dfb4b20223fd1bbb5e05b90a2c01a8b2b8e6f13ebc807b152cb8b5fdd",
  "dki_bound.npz": "935efb04ed96b371a72bbc51f2f875634f5ee3786843c923a37ec01d032725d6",
  "dki_upstream.npz": "f8a98ce71455235941b0aef12442704b4fe2891eed2a07a027fcbec072040f47"
}
```

Privatearchive: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch2_20261020\bio_med`. Machine-readable `archive_index.json` identifies paths, SHA256, sizes and provenance roles for downloadedsource/data/papers/captures/logs and figures. `final_gallery_manifest.json` names only finalmainfigures and their numericalsources; do not glob rejected2023samples or obsoletecandidateexports. The taskinventory with per-filehashes is private final_task_inventory.json. Fulloriginalcode/data licensing is preserved in taskrootLICENSE/NOTICE and source-depositmetadata.


Final notebook display verification: the figure-creation cell closes Matplotlib managers while retaining the figure objects. Each scientific section explicitly displays its plate once; the complete fresh nbconvert run contains exactly 4 PNG outputs, no automatic duplicate plate outputs, and no error cells. This presentation-only correction leaves the selected inverse method, comparisons and numerical thresholds unchanged.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 4 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 31 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Observation-only b0>40ADU support; negativeK preserved; finite-b signal model, not compartment truth. Paper weight prose ambiguity investigated independently.
