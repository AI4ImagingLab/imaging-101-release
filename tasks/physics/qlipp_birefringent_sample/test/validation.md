# Authoritative provenance and validation evidence

All mandatory gates below passed for the bound real-data task on2026-10-20. This is a newly extracted local standard-method implementation, not a layout-only wrapper. References are independently reproduced method measurements; no reconstruction is relabeled as material ground truth.

## Fixed physical scope, sources, versions and licensing

- Repository: https://github.com/mehta-lab/waveorder . Exact remotely verified commit524926f5922e6486e381194d161b455786529577; wave_branch.json and wave_commit.json independently confirm actual commit identity (tree33353f7f478bf77220d28314a4bd61bd16423a83). Exact source snapshot is privately archived as wave_source.zip and extracted tree. Selected original routines are waveorder/stokes.py and models/inplane_oriented_thick_pol3d.py. BSD-3-Clause source copyright2025Chan Zuckerberg Biohub is preserved at task root.
- Publication versions: Guo et al., Revealing architectural order with quantitative label-free imaging and deep learning, eLife9:e55502(2020), DOI10.7554/eLife.55502, fulltext qlipp-paper-full.xml (PMC7431134), Model of image formation and polarization inverse. Chandler et al., waveOrder: generalist framework for label-agnostic computational microscopy, arXiv2412.09775v1(2024), archived waveorder-paper-v1.html/text. The2026source snapshot is explicitly a modern implementation of the selected Stokes/ADR equations, not a claim that the2020paper used this code revision. Phase-from-defocus and full permittivity-tensor/deep-learning workflows are not selected outputs.
- Real deposited calibration specimen: https://doi.org/10.5281/zenodo.8386856, recOrder_test_data.zip (49,215,386bytes), deposited CC BY4.0 confirmed by private zenodo_wave.json. The selected subdirectory is2022_08_04_recOrder_pytest_20x_04NA, original Kazansky birefringent specimen. All3stagepositions and all16defocus planes at firsttimepoint are retained, full128x256 detector region and4states. It is not a cell and not a simulation. TIFF originals and independent BG/State0..3.tif plus calibration_metadata.txt remain private.
- BG is independently observed empty field, not target-estimated calibration. Both raw detector data and deposited float64BG values are preserved publicly. The original JSON instrument matrix is included unchanged float64; original reconstruction converts it to float32. Measured dark100.58ADU is applied to sample and background. The physical optical power/background exposure are not deposited, so T is explicitly relative intensity. Data have zero65535samples. Hardware page metadata gives20x0.55NA dry,532nm,10ms,z2um,firstz197.5um; calibration's stale MicroscopeParameters60x/NA1.3 and detectorPixelSizeUm0 are not used to invent a pitch. This discrepancy is documented publicly and here.
- An earlier2023recOrder_session sample was screened and rejected for4.115%ADC clipping and sample/background intensity mismatch(T~7-11); it never enters final rawdata, references, plots or metrics. The optional central ROI experiment for SOFI was likewise rejected; final bound observations remain original full nativefields.

## Original run before extraction

Unmodified original stokes.py routines ran the complete instrumentinverse, independentBG Muellerinverse and all4ADR inverses on exactly the visible bound arrays before localextraction. Command: private original Python `capture_qlipp.py`; firstbound runtime3.0724345000926405seconds, exit0, Python3.11.11, NumPy2.4.6, torch2.9.1+cu128. Full originalpipeline wrapper apply_inverse_transfer_function was additionally loaded unchanged with inactive correction/util importshims and verified against originalStokes capture: maximumdifference0 in every quantity (qlipp_original_full_model.json). Neither unused shim is called. Logged independentrepeat exits0; no RNG, training or pretrained weights.
All oracle environments are maintainer-only, outside the task. `qlipp_environment.log` records full OS/Python/NumPy/SciPy numerical backend configuration, and `qlipp_freeze.txt` records installed packages. Cleaned environment is Python3.11.11, NumPy2.4.6, Matplotlib3.11.1,torch2.9.1+cu128 used onCPU; CPU1thread via OMP/OPENBLAS/MKL settings. Full cleaned_freeze.txt and cleaned_environment.log are archived. No GPU requirement is introduced; CUDA-tagged Torch wheel was executed onCPU. The shared runtime was not upgraded and contains no selected upstream domain package.

## Independent fixture tolerances and numerical acceptance

Before extraction capture fixed float32 stage/output rtol5e-5,atol2e-5. Nominal required arrays use realfloat32. Independently rerunning original source after advancing each analysismatrix entry by onefloat32ULP gives relativeL2 changes:delta1.9283500451337183e-7,T8.791601846234786e-8,P1.198598705850827e-7; continuousweighted angularRMSE7.88628023385674e-8rad (qlipp_precision.json). FixedJudge budgets2e-5relativeL2 fordelta/T/P and1e-4rad weightedangle are conservative float32/calibration-numerics tolerances, not claims of materialaccuracy. No threshold was loosened after cleanoutput. Nominal freshcomparison is about4.52e-8retardance relativeL2 and2.37e-8radweightedangle.
Metric definitions: relativeL2=norm(estimate-reference)/norm(reference) per required array, independently checked with knownvectors. Orientation uses pi-periodic wrappedangle and continuous sin(reference retardance)^2weight; no low-signalpixel is removed. No NCC/NRMSE is silently redefined or relabeled. No scale, translation, magnitude conversion, hidden crop or reference filename guessing occurs. Zero-norm references and undefinedconfigurations are rejected. Without independent truth, these boundaries test standard-method reproducibility. No extra physicalaccuracy criterion is fabricated: the supplied measurement/statistic model is not an independently known structural truth; physical validation is provided by independent analytic/Jones cases.

## Equation/stage to executable-source traceability

| Paper equation or algorithm stage | Independently executed original source | Local implementation | Validation case/evidence |
|---|---|---|---|
| Four-state Stokes instrument inverse, QLIPP model | stokes.calculate_intensity_to_stokes_matrix / mmul; actual calibrated A | solvers.reconstruct; physics_model.apply_matrix | frozenanalysis/Stokes; independent complexJones retarder |
| Known camera offset | deposited calibration_metadata.txt BlackLevel100.58 | preprocessing.subtract_dark | publicconstant agreement; frozen sample/background Stokes |
| Measured-background ADR Mueller model | stokes.mueller_from_stokes | physics_model.mueller_matrix/inverse_mueller | frozen independent inverse matrices; matrixinverse identity |
| Weak retardance, pi-periodic slow axis, T,P inverse | stokes.estimate_adr_from_stokes | physics_model.estimate_parameters | independentJones physics, analyticroundtrip/TPnorm; original full4maps |
| Entire calibrated pipeline and default frame convention | models/inplane_oriented_thick_pol3d.apply_inverse_transfer_function | solvers.reconstruct | exact original modelwrapper verification + full4map parity; noflip/rotate |
| Independent empty-field calibration and noise limits | originalmodel cyx_no_sample_data optional | visualization.run_experiments | fresh no-BG and dark80.58/120.58reruns; P>1and full16plane profiles |
| Physical circular-angle evaluation | fixed standalone metric definition | judge.angular_score; visualization.angular_error | pi-invariance knowncase and matching directmetric tests |

## Ambiguities, source discrepancies and physical limitations

The actual optical specimen is a calibration target, not biological tissue. Relative transmission is constrained by independent background intensity but no absolute optical-power/exposure calibration is invented. Retained polarizationP spans about0.9000..1.02282: allnoisy above1values remain in savedoutputs and plot distributions. The weakbranch uses positive corrected s3; near-zero retardance makes orientation uncertain and is handled with continuous sin(delta_ref)²metricweight, not hiddenpixel removal. The original Mueller formula is singular at perfectly isotropic zero-g1/g2 background; actual independentBG is nonsingular and no epsilon correction is silently introduced. Lateral pitch/defaultmetadata discrepancy is public. Modern source wrapper uses inactive correction/util import shims only to load its file privately; with estimatedBG=False neither shim is called and original mathematical routines remain unchanged. Full originalwrapper output matches the earlier original-Stokes capture exactly for all4quantities.

## Fresh measurements and actual review experiments

The canonical notebook actively recomputes the complete bound benchmark before scoring, without loading any standard result as freshoutput. Original measurements, fresh nominal results and actual alternatives are explicitly labeled. No historicalpaper figure/number is represented as this dataset's measurement. All alternatives in visualization.run_experiments actually run; no missing method is filled withzero.

Fresh measurement record:

```json
{
  "reconstruction_runtime_s": 0.1558645002078265,
  "experiment_runtime_s": 0.47529869992285967,
  "python": "3.11.11",
  "numpy": "2.4.6",
  "metrics": {
    "reconstruction_relative_l2": 4.515835926124896e-08,
    "orientation_angular_rmse": 2.3716340610362295e-08,
    "transmittance_relative_l2": 0.0,
    "depolarization_relative_l2": 6.4846591706579635e-09
  }
}
```

Final strictJudge report:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_relative_l2": 4.515835926124896e-08,
    "orientation_angular_rmse": 2.3716340610362295e-08,
    "transmittance_relative_l2": 0.0,
    "depolarization_relative_l2": 6.4846591706579635e-09
  },
  "thresholds": {
    "reconstruction_relative_l2": {
      "comparison": "<=",
      "boundary": 2e-05
    },
    "orientation_angular_rmse": {
      "comparison": "<=",
      "boundary": 0.0001
    },
    "transmittance_relative_l2": {
      "comparison": "<=",
      "boundary": 2e-05
    },
    "depolarization_relative_l2": {
      "comparison": "<=",
      "boundary": 2e-05
    }
  },
  "errors": []
}
```

Notebook command and execution: `python -m jupyter nbconvert --to notebook --execute notebooks/qlipp_birefringent_sample.ipynb --output qlipp_executed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging-bio-research`. Kernel argv explicitly selects shared research Python, not the host defaultAnaconda kernel. Privatekernelspec is configured only through per-processJUPYTER_PATH. Final subprocess exit0, walltime16.04755589994602seconds; timeout180seconds is more than5times observed total and covers all scientific work, rendering and exports. The original warning about WindowsProactor/ZMQ selector fallback is archived; execution has noerror cells. Canonical notebook retains executedoutputs; temporaryexecutedcopy is private, never in output/.

## Artifact consumers and provenance roles

- data/raw_data.npz: public exact captured observations and legitimate calibration only; loaded by preprocessing, tests and active notebook.
- evaluation/baseline_reference.npz: original standard-method requiredarrays from independentcapture; used by standaloneJudge and whole-pipelineparity. ground_truth.npz is absent because actual materialtruth is unavailable.
- test/fixtures/upstream_stages.npz: frozen original stagevalues, consumed by comparing_test/test_parity.py; independent matrix/Stokes16x16stagepatches plusfullrequired references. No expectedparityarray is generated by cleanedfunctionunder test.
- notebooks/reference_outputs/upstream.npz: only original arrays with active numeric/figurecomparisonconsumers; no redundant unusedfullstage dump remains here.
- notebooks/reference_outputs/cleaned.npz: saved by active notebook from freshlocalruntime, checked by test_numerical_review.py; not independentupstreamevidence.
- notebooks/reference_outputs/experiments.npz: actual alternative/sensitivity sourcevalues, consumed by notebookplots/privatepaperfigureexports and reviewconsistencytest.
- notebooks/reference_outputs/measurement.json: actual runtime/environment/metricmeasurements from notebook, consumed by reviewtest and finalgallerymanifest.
- All originalTIFF/NIfTI/Zarr, rawdownloads, originalsource, papers, capturescripts, originalenvironmentinformation, candidate/rejectedobservations, logs and publicationexports remain in the privatearchive, indexed by archive_index.json. No sourcePDF/upstreamcheckout/metadataJSON leaks to solverallowlist.

## Commands, review and completion gates

Task cwd is qlipp_birefringent_sample. Sharedruntime `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`; allcommands in final_execution.json ran with CPU1thread.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/version/revision/license/datahash/equationmapping | passed | Source sections and archiveindex; remotely verified fixed revision |
| Original exact-observationexecution before localextraction | passed | Independent originalcapture; originalstages and manifests |
| Independent parity plus applicablephysics | passed | Frozen stage/full-pipeline tests; independent complex Jones retarder, Stokes identities, circular-angle invariance and background-matrix inverse cases |
| Constants, units, dtypes, coordinates and numericalconventions | passed | Exact unchangedarrays, metadata tests, originalsign/normalization/precisionreview |
| StandardCLI writes declaredNPZ | passed | python -m src.main; qlipp_cli_final.log exit0 |
| FreshJudge plus malformed/poor/evaluator/report failures | passed | python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; contracttests |
| Allowlist sufficient; no answer/seed/target-calibration/GitHub leaks | passed | README/requirements/raw audit; legitimate independentcalibration only |
| Minimal shareddependency subset and CPU execution | passed | Only actualNumPy/Matplotlib/Torch subset; no upstreamdomain import |
| Explicit localmath and call-role audit | passed | import_call_role_audit.json; genericmathprimitives only |
| All tests | passed | python -m pytest test/ -v -p no:cacheprovider --basetemp <privateunique>; 32passed, exit0 |
| Complete active notebookexecution | passed | Explicitresearchkernelspec nbconvert exit0; noerroroutputs |
| Required comparisons and renderedQA | passed | Allfinalplates inspected; nativeunits, sourcevalues, visiblefailures, titles/legends corrected |
| Historical/upstream/fresh measurements and thresholds distinguishable | passed | Original/fresh/alternativeroles, fixedmetricdefinitions and independentprecisionreruns |
| README/plans/source semanticconsistency | passed | Reviewed definitions, formulas, outputdtypes/shapes, signatures/defaults, branches, signs, dimensions and callerflow |
| Final task inventory and license/archivehygiene | passed | Onlytemplatedirs/files, no runtimeupstream/PDF/rootwrapper/metadataJSON/tempnotebook; license/notices preserved |

No mandatory failed or unverified gate remains for this selectedboundtask. Out-of-scope larger sourceexperiments, absolute physicaltruth and unsupportedphysicalcalibrations are not misrepresented as passedchecks.

## Data hashes and archive index

Public raw NPZ SHA256: `e32bc1808db6ad2f82503c43c590ee70d81f435389a1b8415be0dcc47daca361`. Everypublicarray is exactly equal to independentoriginalcapture (lossless_packaging.json).

Originaldata/capture SHA256 from firstboundmanifest:

```json
{
  "raw_bound_sha256": "e32bc1808db6ad2f82503c43c590ee70d81f435389a1b8415be0dcc47daca361"
}
```

Privatearchive: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch2_20261020\bio_med`. Machine-readable `archive_index.json` identifies paths, SHA256, sizes and provenance roles for downloadedsource/data/papers/captures/logs and figures. `final_gallery_manifest.json` names only finalmainfigures and their numericalsources; do not glob rejected2023samples or obsoletecandidateexports. The taskinventory with per-filehashes is private final_task_inventory.json. Fulloriginalcode/data licensing is preserved in taskrootLICENSE/NOTICE and source-depositmetadata.


Final notebook display verification: the figure-creation cell closes Matplotlib managers while retaining the figure objects. Each scientific section explicitly displays its plate once; the complete fresh nbconvert run contains exactly 4 PNG outputs, no automatic duplicate plate outputs, and no error cells. This presentation-only correction leaves the selected inverse method, comparisons and numerical thresholds unchanged.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 4 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 32 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Real calibration specimen, not cells; independent background; T is relative intensity; noisy P>1 preserved; no unsupported lateral metric scale.
