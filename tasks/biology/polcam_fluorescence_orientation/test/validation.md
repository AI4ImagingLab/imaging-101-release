# POLCAM provenance and validation evidence

## Scope and acquisition

Status: ready for independent parent audit; local mandatory gates passed. This task recovers spatial effective Stokes S0/S1/S2, linear-polarization angle and degree from the complete author-deposited real NR4A live T-cell EPI stack. It selects the published diffraction-limited napari-polcam no-interpolation path with nominal offset zero, not single-molecule localization, DSF fitting, 3D dipole orientation or rotational mobility. It is new relative to the 139 frozen tasks: qlipp_birefringent_sample estimates transmitted-light material retardance, retention and slow-axis orientation, while pmhc_single_molecule_localization estimates emitter locations. Neither has this fluorescence-emission acquisition, live T-cell dataset or effective emission-polarization unknown. No existing task was edited.

Paper: Bruggeman et al., *POLCAM: instant molecular orientation microscopy for the life sciences*, Nature Methods 21, 1873-1883, published 2024-10-07, DOI https://doi.org/10.1038/s41592-024-02382-8 . Published PDF inspected at https://eprints.whiterose.ac.uk/id/eprint/219063/1/s41592-024-02382-8.pdf , including equations1,3,4a-c and diffraction-limited Fig6/methods. Exact published PDF archived privately at polcam/paper_published.pdf. This source version was read; no historical paper scalar result is relabeled as this run's measurement.

Code: https://github.com/ezrabru/napari-polcam , exact remotely verified default branch revision **e4cd20f39b6b726ade8e17a4768f2210596e612d** obtained through GitHub repository/tree API on 2026-10-21, archived response ezrabru_napari-polcam_api.json. The codeload exact-revision archive agrees. This is a verified remote revision rather than an unverified local tracking SHA; it is the observed HEAD at retrieval, not a permanent claim of future freshness. Original BSD-3-Clause license is preserved verbatim in task LICENSE and attribution in NOTICE. Original source `_functions.py` SHA256 is **395e74216c2973b405d7edf55a83c5e1c6e4e8bbeb61822210aad7ca9458fe00**. The core source was not modified. The oracle imported this original file directly by path, not a copy of cleaned mathematics.

Data: https://doi.org/10.5281/zenodo.10471496 , version1, author real live T cells, CC BY4.0. The exact record API response is archived as zenodo_10471496.json. Bound file `Tcell_NR4A_EPI_stack.tif`, complete116x298x298 uint16, SHA256 **7efcc784f698775651deab169d4592da0a2eabb4e17ec510fc3e2b2004a6fd47**. It is losslessly converted to `data/raw_data.npz::intensity` by adding one batch axis; no crop, conversion to float32, noise injection, synthetic data or target-derived calibration occurs. Public NPZ SHA256 **28298f0b6e119b7abee30a5426fdf6f10f4934fd40eb2b165d58b6d30f9e96f5**. Raw range is399 through1499 ADU. TIFF conversion used tifffile2026.3.3 only in private maintainer preprocessing; the shipped NPZ runtime does not import or require an image library. The full independent calibration maps from https://doi.org/10.5281/zenodo.10578307 were inspected privately but are not registered to this crop and are neither shipped nor used by the canonical inverse.

## Ambiguities, limitations and explicit decisions

The associated Micro-Manager metadata claim644x644 and ROI808-708-643-643, while the deposited TIFF contains298x298. Thus neither full-sensor crop origin nor physical z step is safely inferred. The complete actual TIFF and plane order are authoritative. No guessed per-pixel gain/dark registration is made. Raw TIFF mosaic is interpreted using original napari-polcam default orientation `[-45 0;90 45]`, expressed explicitly as the task's detector-coordinate convention; acquisition crop parity is not independently certified, so absolute sample-coordinate molecular axes are not claimed. Published microscope3 nominal sample-plane pitch69nm and effective tile spacing138nm are interpretation scales; axes in all exported plots use superpixel or acquired-plane indices. Source metadata z-step100um is incompatible with the biological stack context and is deliberately not reinterpreted as a physical z calibration. This omission is recorded, not silently replaced.

The nominal offset-zero default produces an effective detector field. A uniform added pedestal shifts S0 by twice the pedestal and cancels exactly from S1,S2 and angle. Its effect on degree is measured explicitly in sensitivity figures. This does not correct spatially varying gain/background or establish absolute photon flux. Noise and the use of neighboring pixels as one common field produce a measured nonzero residual. Weak polarized signal makes orientation poorly identifiable; the fixed angular metric weight S1_ref^2+S2_ref^2 expresses this without tuning a spatial mask. No 3D tilt or molecular-wobble parameter is inferred. No new synthetic observations are generated, so src/generate_data.py is correctly omitted.

The upstream GUI's `subtract_bkgnd` can subtract the scalar offset again after constructor subtraction. Our selected original class execution invokes constructor once with offset0 and `convert_unprocessed` directly, hence both paths coincide at the selected convention and no nonzero-offset GUI claim is made. Optional local sensitivity computations use one explicitly stated subtraction and are not declared unmodified upstream GUI runs. Optional upstream cubic interpolation has axis/offset ambiguities for differently sampled channel grids; this task intentionally selects the native no-interpolation option and does not silently correct that unrelated branch.

## Original exact-bound run and frozen success criteria

Private archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/bio_med_mechanics` (B). Complete upstream checkout and exact zip are under B/upstream and B/napari-polcam.zip. Capture command: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/bio_med/oracle-env/Scripts/python.exe B/polcam_oracle.py`. This pre-existing maintainer oracle environment was used read-only; no old task module or source was imported. Original source `_functions.py` was loaded via importlib, then PolarisationCameraImage(raw.astype(float64), 'None', '[-45 0; 90 45]',0.0), convert_unprocessed, stokes_from_intensities, aolp_from_stokes and dolp_from_stokes were executed on every bound pixel and plane. Private B/polcam/oracle.npz contains original input, channel, Stokes, angle and degree stages. B/polcam/oracle_run.json records environment, tolerances and runtime. No stochastic seed applies; the algorithm is deterministic. CPU NumPy uses no GPU or CUDA runtime.

Frozen before local extraction: channels and Stokes require exact equality; angle and degree absolute tolerance1e-14 with rtol0; applicable independent analyzer/adjoint/gradient/pedestal tests use independently known cases. The small task fixture selects original planes0,57,115 and detector top-left12x14, corresponding output6x7; it is a direct subset of frozen upstream output, never generated by cleaned functions. Its consumer is test/comparing_test/test_upstream_and_pipeline.py. The full hidden baseline stores all original output arrays and is independently used by both notebook and full pipeline parity test. A separate array-identical upstream rerun executed from the lossless public NPZ on the shared research runtime, recorded in oracle_rerun.json, demonstrates that no package installed in the oracle environment hides a difference or provides a runtime dependency.

Actual oracle run: 7.100004900006752s, exit0, Python3.11.11, NumPy2.4.6, Windows10.0.26200, CPU, no warnings captured. Original source imports only NumPy and SciPy RectBivariateSpline (the selected no-interpolation branch does not call SciPy). Original rerun: 0.795336999995925s, exit0, all three full arrays elementwise identical. Backend threads were pinned to1 in actual command environment; solver is deterministic float64 elementary arithmetic. The capture script's elapsed time includes TIFF reading and fixture compression, while the rerun excludes TIFF conversion and output compression.

## Equation/source/local/test traceability

| Paper/model stage | Unmodified upstream file/function | Local executable function | Validation case |
|---|---|---|---|
| Four analyzer mosaic, Fig1a and Fig6a | _functions.py::unprocessed_to_unassigned_channels / assign_channel_to_transmission_axis_orientation | preprocessing.split_channels | frozen upstream stage; independent2x2 signed tile |
| Eq4a-c linear Stokes inverse | _functions.py::stokes_from_intensities | solvers.estimate_stokes | frozen/full upstream parity; analytic S=(10,6,8); normal equations |
| Linear analyzer forward law underlying Eq4 | _functions.py::intensities_from_stokes | physics_model.forward / analyzer_matrix | analytic I=(8,9,2,1); A^T A diagonal check |
| Linear transpose and least-squares gradient | Matrix transpose, independently derived; no original special adjoint API | physics_model.adjoint | real inner-product adjoint identity; finite-difference directional objective gradient |
| Eq1 AoLP with correct quadrant | _functions.py::aolp_from_stokes | physics_model.polarization | frozen/full parity; known0.4636476090008061rad; pi-periodicity |
| Eq3 local DoLP / Fig6 effective field | _functions.py::dolp_from_stokes | physics_model.polarization | frozen/full parity; known unit degree; zero/isotropic branch |
| Nominal offset convention | PolarisationCameraImage.__init__ offset0 | preprocessing.split_channels / solvers.reconstruct | independent common-pedestal invariance and degree-bias test |
| Full published diffraction-limited data flow | PolarisationCameraImage None option then Stokes/AoLP/DoLP functions | main.run | full stack parity, CLI and standalone Judge from unrelated cwd |
| Evaluation metric conventions | Independently defined benchmark contract, not a paper-reported score | standalone judge.ncc_nrmse and visualization.ncc_nrmse, separate implementations | independently known cosine and dynamic-range RMSE case |

## References, metric provenance and numerical results

No actual per-voxel biological polarization truth is available. `evaluation/baseline_reference.npz` contains original-method `reconstruction`, `orientation` and `degree` in precisely declared batch-first shapes and float64. It is never called ground truth. `notebooks/reference_outputs/measurements.json` is the sole non-array provenance record there; notebook explicitly reads it to distinguish historical/published scope, reproduced upstream and fresh local measurements. No duplicate full arrays are retained in reference_outputs.

The standard source is its own hidden method reference, giving NCC1 and NRMSE0. Therefore relative10% threshold rules are inappropriate. Array-identical independent original rerun and the condition-number2 analyzer normal matrix justify explicit nonzero float64 tolerances: per-component NCC>=0.9999999999, dynamic-range NRMSE<=1e-10, degree NCC/NRMSE same, weighted periodic angular RMSE<=1e-10rad. No sample normalization, alignment or truncation is fitted. Physical residual bound1.1*upstream residual was fixed before cleaned results. A valid but incorrect constant-shift/angle/degree prediction fails; malformed or unavailable evaluator assets produce statuserror and empty measured metrics. The constant-reference/zero-norm NCC configuration is explicitly rejected rather than returning NaN. JSON uses finite numbers only.

Upstream observation relative residual: **0.016804278834125018**. Boundary: **0.01848470671753752**. Fresh local Stokes arrays agree elementwise with upstream; fresh angle agrees at1e-14 and degree differs only by floating point sqrt versus hypot ordering. Actual fresh report:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "stokes0_ncc": 1.0000000000000002,
    "stokes0_nrmse": 0.0,
    "stokes1_ncc": 1.0000000000000002,
    "stokes1_nrmse": 0.0,
    "stokes2_ncc": 1.0,
    "stokes2_nrmse": 0.0,
    "degree_ncc": 1.0,
    "degree_nrmse": 1.3773962485057759e-17,
    "angular_rmse": 0.0,
    "observation_residual": 0.016804278834125018
  },
  "thresholds": {
    "stokes0_ncc": {
      "comparison": ">=",
      "boundary": 0.9999999999
    },
    "stokes0_nrmse": {
      "comparison": "<=",
      "boundary": 1e-10
    },
    "stokes1_ncc": {
      "comparison": ">=",
      "boundary": 0.9999999999
    },
    "stokes1_nrmse": {
      "comparison": "<=",
      "boundary": 1e-10
    },
    "stokes2_ncc": {
      "comparison": ">=",
      "boundary": 0.9999999999
    },
    "stokes2_nrmse": {
      "comparison": "<=",
      "boundary": 1e-10
    },
    "degree_ncc": {
      "comparison": ">=",
      "boundary": 0.9999999999
    },
    "degree_nrmse": {
      "comparison": "<=",
      "boundary": 1e-10
    },
    "angular_rmse": {
      "comparison": "<=",
      "boundary": 1e-10
    },
    "observation_residual": {
      "comparison": "<=",
      "boundary": 0.018484706717537523
    }
  },
  "errors": []
}
```

## Commands, execution, environment and inspections

All commands use shared Python `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, thread environment OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. Validated runtime is Python3.11.11, NumPy2.4.6, Matplotlib3.11.1 on Windows10.0.26200 CPU. Requirements contain exactly NumPy and Matplotlib pins, with no upstream domain package, image reader, pytest or Jupyter runtime requirement. Jupyter/pytest are development tooling. No installation or shared-package upgrade was needed for this task.

| Stage | Actual runtime (s) | Exit | Command |
|---|---:|---:|---|
| cli | 3.98003100000642 | 0 | `-m src.main` |
| judge | 1.0837809999939054 | 0 | `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` |
| pytest | 15.083906400002888 | 0 | `-m pytest test/ -v -p no:cacheprovider --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch5_20261021\bio_med_mechanics\polcam\pytest-final-temp` |
| notebook | 20.64516949999961 | 0 | `-m nbconvert --to notebook --execute notebooks/polcam_fluorescence_orientation.ipynb --ExecutePreprocessor.timeout=300 --ExecutePreprocessor.kernel_name=imaging101-bio-b5 --output polcam_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch5_20261021\bio_med_mechanics\polcam` |

The initial pytest command without --basetemp encountered a host Temp/pytest-of-admin PermissionError before thirteen tmp_path fixtures could run; twelve numerical tests passed. This was an environment path permission issue, not numerical failure. A private accessible --basetemp was supplied; subsequent full runs passed26 tests. This is transparent environment adaptation, not omitted tests. Private logs polcam/cli.log, judge.log, pytest.log, notebook.log and execution_records.json contain actual exit statuses. The active notebook timeout300s is justified by the full successful execution (~21s), with margin for slower CPU machines. Its complete selected path actively calls cleaned run from raw NPZ, then independent Judge and all five visual groups. The executed canonical notebook includes outputs and execution counts. Temporary executed notebook is private B/polcam/polcam_executed.ipynb, never task/output.

All five exported PNG groups were visually inspected. Groups1-4 show complete valid layouts and appropriate units/angle scales; group5's initial long title clipping was fixed by reducing title font size, regenerated, and re-inspected. Final figures contain no missing-method placeholders. The exact-zero original/local Stokes difference is computed, not a fabricated zero substitute. Figures are output as five PNG+SVG+PDF groups with numeric_sources.npz and a machine-readable SHA256 manifest at B/figures/polcam_fluorescence_orientation. Their metrics.json distinguishes canonical and display-plane sensitivity conditions. The notebook preserves the relevant scientific comparisons for this bound diffraction-limited acquisition; paper single-molecule/DSF/simulation/time-lapse results require different data and are explicitly excluded rather than claimed reproduced.

Source and plan review checked every shipped signature/default, dtype/shape, offset sign, all1/2 factors, signed differences, radian convention, array axes, least-squares normalization and no-iteration flow. Mathematical objective/normal equations were independently checked; signature matching alone was not used as evidence. Audit B/polcam/symbol_role_audit.json lists actual imported symbols and call roles in source, Judge and tests. Allowed calls are NumPy linear algebra/array primitives, Matplotlib plotting, standard file/CLI/test tools. There are no domain-library, cross-task, hidden-reference-in-solver, learned-checkpoint or backend-changing library calls. Notebook calls only local source and independent evaluation; upstream files remain private. Public README/requirements scan found no prohibited GitHub URL; public NPZ contains only original real intensity. No hidden truth, object seed, target-derived calibration or privileged extra measurement is exposed.

## Private archive index and hashes

| Private artifact relative to B | SHA256 |
|---|---|
| polcam/Tcell_NR4A_EPI_stack.tif | 7efcc784f698775651deab169d4592da0a2eabb4e17ec510fc3e2b2004a6fd47 |
| polcam/Tcell_NR4A_EPI_stack_metadata.txt | d4049767dce976302c46812f4e3d355d8bcd39277f747f7fc16e11cb0a79f184 |
| polcam/paper_published.pdf | 4ffe88d2dbced0fe5833a3b623385288c36c8df3a1d371ece68a1227714e935c |
| polcam/oracle.npz | 8ce7e237f841cd6361c14a7ccc376d63645c6c61274de4bc08d7e1b301605739 |
| polcam/oracle_run.json | 6da60b33cefc3bdd2adabe177efa42fd266015b116d805158f9195297cbd3daf |
| polcam/oracle_rerun.json | d880ca99734ef8f1424b20a2d4fc73b0da13e3f33daa915a70af577afddb1b55 |
| napari-polcam.zip | 1b8be4692e260a273d778938209214d370d6be12587927dab946edf3c173d4d7 |
| ezrabru_napari-polcam_api.json | e74948ed829d896063e8a63ee1960e50ea89e67a905debc7e3ec77e5b87ae7fb |
| zenodo_10471496.json | 30e1d7fec2793033290c54064edae0ccc43d8dc163476dd894e60ea80194e353 |

Additional original calibration files inspected but not used are B/polcam/pixelmap_offset.tif, pixelmap_gain.tif and Zenodo10578307 metadata. Exploratory other repositories and rejected candidates stay outside the final task and do not contribute to its reference. Capture/validation/build scripts remain private B/polcam_oracle.py, polcam_oracle_rerun.py, validate_polcam.py and polcam_* builders. Final task inventory contains only README, requirements, LICENSE/NOTICE, raw NPZ, src, output reconstruction/report, plans, active notebook/provenance, hidden evaluation and test fixtures/evidence/tests. There are no PDFs, screenshots, download archives, root wrappers, standalone metadata JSON, duplicate validation reports, temporary executed notebooks or unused reference arrays in the task.

## Mandatory gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes/equation mapping | passed | Published PDF, API revision, exact hashes and mapping above |
| Original and cleaned complete bound-data runs | passed | oracle_run/rerun and CLI records |
| Independent upstream parity and physics | passed | Frozen stages, full arrays, analytic inverse, adjoint/gradient/pedestal tests |
| Constants/units/coordinates/precision preserved | passed | Lossless uint16 NPZ; defaults and source review; public-constant test; explicit metadata limitations |
| Standard CLI and fresh standalone Judge | passed | exit0 CLI and Judge |
| Judge rejects poor and malformed/evaluator cases | passed | every output perturbed; missing/shape/dtype/complex/NaN/Inf/evaluator/report tests |
| Public interface sufficient and leakage absent | passed | self-contained equations/mosaic/output; intensity-only NPZ; URL scan |
| Acquisition/output metadata and no standalone metadata | passed | README explicit tables; no metadata JSON |
| Minimal dependencies and CPU runtime | passed | shared runtime full run; no upstream installed/imported in cleaned runtime |
| Local core mathematics, symbol/call roles | passed | explicit analyzer, inverse and adjoint; AST audit |
| Full unit/parity/physics/pipeline suite | passed |26 passed, no skipped/failed tests |
| Active notebook and scientific comparisons | passed | nbconvert exit0, all cells executed, five inspected groups |
| Result categories and metric/threshold provenance | passed | measurements.json, independent cases and exact-bound upstream |
| README/plans/source mathematical and API consistency | passed | semantic/sign/unit/signature review described above |
| Template inventory/archive/license cleanup | passed | exact final inventory; original BSD3 and CC BY4 attribution |

This ledger records local validation; parent acceptance remains a distinct independent review step.


## Parent acceptance review

Parent independently ran26 tests (26passed in13.13s), foreign-cwd standalone Judge0 and file/call-role audit errors[]. Read the complete public measurement/output contract, explicit four-analyzer forward/adjoint, closed-form Stokes least-squares inverse, periodic angle/degree conventions and both implementation plans. All5 actual scientific groups opened and inspected, including full-stack sections, signed Stokes maps, raw-channel forward residuals and actual pedestal/spatial-averaging sensitivity. Real author-deposited116-plane T-cell fluorescence observations are retained losslessly, reconstructing every plane and all3Stokes components plus both derived maps. This is a new fluorescence-emission inverse relative to old transmitted-light retardance and localization tasks. Original independent exact-data runs and stages are preserved, not biological truth. The nominal zero-offset field and unregistered detector calibration are explicitly limited; no absolute molecular3D angle/wobble, full-sensor crop origin or axial scale is invented. Strict per-output metrics/forward consistency and malformed/poor/evaluator contracts pass. Source defaults, signs, double-angle periodicity, units and actual caller flow agree with plans/README.
