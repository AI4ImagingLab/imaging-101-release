# Validation and provenance: ASL CBF and arterial transit time

## Bound task, source and archive

Physical unknowns are CBF in mL/100g/min and ATT in ms, on the complete author demonstration (5,35,35) spatial grid. The seven shortest-echo difference images, independent M0 and supplied brain mask are the only numerical inputs. No new observations are generated. Data classification: **author-supplied acquisition demonstration**; morphology and author documentation identify ASL images, but native scanner/subject details and an independently validated physiological truth are unavailable. Do not relabel the standard reconstruction as truth.

Selected paper: Buxton, Frank, Wong, Siewert, Warach and Edelman, *A general kinetic model for quantitative perfusion imaging with arterial spin labeling*, published MRM 40:383–396 (1998), DOI 10.1002/mrm.1910400308. The published 14-page PDF was obtained from https://s.mriquestions.com/uploads/3/4/5/7/34572113/buxton_general_kinetic_model.pdf. SHA256 3129804b93d9636948f500cd2ac98d39db84bb4ab68dd18b560a246803bbcc90. Reviewed the general convolution Eq.1, rectangular delivery/clearance Eq.2, and continuous-label expression Eq.5 on published pp.385–386; paper-wide PASL/compartment variants are not claimed as this selected runtime. The local notebook explains the selected physical process independently of equation numbering. The author toolkit is described by Senra Filho and Paschoal, ISMRM Brazilian Chapter 2025; the author README supplies that citation. Public references link papers/documentation and expose no source-host URL.

Original code: https://github.com/LOAMRI/asltk, exact main revision **1487eb865f5e0dd4d578e930f66f11ab988e50b9**, live API verified on 2026-10-22; response archived in `asl/verified_remote_revision.json`. This is actually verified remote HEAD, not merely a local tracking revision. Original complete source is maintained at repository-sibling `imaging-101-maintainer/research_batch5_20261021/bio_med_mechanics/upstream/asltk-1487eb865f5e0dd4d578e930f66f11ab988e50b9/`; it was not modified. The MIT license and copyright 2024 Antonio Carlos da Silva Senra Filho are preserved in task LICENSE. The supplied observations are distributed in the source's MIT-licensed tests/files; there is no additional data-specific license notice in that source archive.

This task's maintainer-only archive is the repository-sibling `imaging-101-maintainer/research_batch6_20261022/groups/mech_bio_med/`. Its `asl/archive_index.json` records hashes, sizes and consumers of the bound input, original outputs, source capture scripts, remote revision response, paper, raw logs, original environment and science numeric sources. Complete upstream source, PDFs, logs and capture scripts stay outside the released task. Prior archive reuse does not turn cached reconstructions into evidence: the full original method was freshly executed here on the exact bound observations.

### Original input hashes and conventions

| Original file | SHA256 | Use |
|---|---|---|
| pcasl_mte.nii.gz | dd0cb7a56a598d46dfb37b2a26a550fc629a35b03a29160657200e004656dafa | First acquired echo only; all seven LD/PLD preparations and all five spatial slices |
| m0.nii.gz | 1dfc3265cac02d845f3ce6876b80129adeed26e843b7dfa2d1d6da33f18a797d | Independent M0 calibration |
| m0_brain_mask.nii.gz | f0171d8e0c1193d21aa741b698d16cf82ab745ab6b7a3653130d92942e42acc3 | Fixed observed tissue mask |

`raw_data.npz` is exactly the original mapper's consumed `pcasl[0]`, M0 and mask, with a leading batch dimension added. The unused seven later echoes are excluded from public data and have no effect on the original CBF mapper. Observation/M0 float32 values are preserved bit for bit. The binary int16 mask is represented as uint8 after exact value equality; this does not change the spatial region. NIfTI header spacing is (1,1,1) mm, RAS diagonal affine (-1,-1,+1), zero origin; output/source arrays use z,y,x. Signal is not normalized or clipped. LD/PLD constants are copied exactly from the author example. The full 5800-voxel mask is used, rather than selecting easy fits after seeing the reconstruction. No hidden target, seed, simulator object, full-sampled clean answer or target-derived calibration is public.

## Original run before local implementation

`capture_asl.py` imports the unmodified original package and calls `ASLData`, `CBFMapping.set_brain_mask` and **`CBFMapping.create_map(cores=1)`**, not a reimplementation or a successful-import surrogate. The point-model fixture calls the original `asl_model_buxton`. The original package loads the complete author source observations but its unchanged numerical worker explicitly consumes only the first echo; these consumed samples are exactly the bound raw NPZ.

Isolated environment: `asl-oracle-env`, Python 3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)], Windows-10-10.0.26200-SP0; NumPy 2.4.6, SciPy 1.17.1, SimpleITK 2.5.6. Complete freeze: `asl/oracle_environment.txt`. It includes original-only I/O dependencies and later candidate-research packages; none are installed into the cleaned shared runtime. CPU execution only; OMP/BLAS/MKL/ITK thread limits are one, no GPU, stochastic seed or stochastic algorithm is used. Original multiprocessing uses one worker.

Command: `asl-oracle-env/Scripts/python.exe capture_asl.py`, captured through `run_asl_oracle_capture.py`. Original capture exit **0**, wall time 36.186 s for two complete reruns; per-run fitting runtimes are 16.928965 s and 18.225473 s. Both processed all 5800 masked voxels and returned nonzero raw-flow estimates at all of them. Captured Python warnings: none. Raw combined stdout/stderr is `asl/oracle_clean_capture.log`. An earlier PowerShell redirection treated logger stderr as NativeCommandError despite completed original computation; this was diagnosed and replaced with subprocess capture recording the actual process exit status. It is not reported as an algorithm failure or silently omitted.

### Independent fixtures and mathematical checks

Acceptance criteria were fixed in `asl/acceptance_criteria.json` after the original run and before local implementation. The initial comparing-test collection failed with missing `src.physics_model`, proving that the independent fixtures preceded implementation. The capture never imports cleaned modules.

| Published stage | Original file/function | Local callable | Frozen/independent validation |
|---|---|---|---|
| Data selection and units | asldata.py ASLData, utils/io.py ImageIO; cbf_mapping.py worker uses pcasl[0] | preprocessing.load_observations | Exact source array equality, dtype/shape/mask contract, README constants; original source hashes above |
| Eq.1 delivered magnetization convolution and Eq.2 single-compartment residue | models/signal_dynamic.py asl_model_buxton | physics_model.buxton_signal | Independent SciPy quadrature of finite bolus integral, including no arrival, delivery and washout |
| Eq.5 continuous-label closed form and effective relaxation | models/signal_dynamic.py asl_model_buxton | buxton_signal, forward_volume | 15 captured original forward cases at three flows and five ATT values; rtol 3e-14, atol 1e-8 scanner units; point/volume consistency |
| Joint nonlinear CBF/ATT inversion | reconstruction/cbf_mapping.py create_map and _cbf_process_slice | solvers.reconstruct | Independent complete original output; rtol 3e-6, atol 1e-5 and map NRMSE below 1e-6, fixed before local judging |
| Positivity, arrival and end-of-bolus continuity | Model assumptions and explicit branches | buxton_signal | Zero-flow/no-arrival cases, onset and offset continuity; integral quadrature rtol 1e-10 |
| Float32 optimizer-result storage and clinical units | cbf_mapping.py Array('f'), raw cbf result | reconstruct final rounding and FLOW_TO_CLINICAL | Original raw-flow fixture conversion equality and independent dimensional 1e-5*100*60000=60 |
| Metric/evaluator contract | Independently defined benchmark convention | visualization.metric_pair; standalone Judge | Known vectors [1,3] and [2,4]: NCC=14/sqrt(200), NRMSE=.5; zero output and undefined constant reference |

No custom optimizer update is invented: the published/reference selection uses a generic bounded optimizer, allowed by the task policy. Its task-specific signal, branches, units, objective and pipeline are explicit local code. Parameters/defaults, finite-difference Jacobian, trust-region method, initialization, float32 storage and stopping conventions agree with the original. Runtime imports are NumPy/SciPy/Matplotlib and standard library; `asl/runtime_audit.json` records imported symbols and confirms no original/domain reconstruction libraries or cross-task imports. The cleaned shared runtime cannot import ASLtk, SimpleITK or nanite; their oracle-only installations are irrelevant to the reference solver.

## Discrepancies and physical limits

1. The original optional `cbf_norm` uses 60*60*1000 and labels it mL/100g/min. Dimensional conversion from the fitted mL/g/ms parameter requires 100*60000=6000000. The benchmark uses the unchanged **raw** original fit and this explicit conversion. It does not claim unchanged reproduction of the mislabeled optional normalized key. A unit test checks the factor independently. There is no fitted gain or retrospective rescaling to improve agreement.
2. `CBFMapping` does not pass its MRIParameters constants into the point-model call. Its actual model defaults are lambda=.98, T1blood=1650 ms, alpha=.85. These executed values, rather than unused object settings, are copied into source/README/plans and tested.
3. M0 is the supplied equilibrium image and is treated as effective blood-equilibrium calibration by the selected model. A separately measured blood/tissue signal correction is unavailable. No independent physiological truth is implied. The shortest echo is used without a separate T2 correction, preserving the selected upstream configuration.
4. The author default flow bound is broad; some noisy voxels have extreme fitted flow or ATT near its bound. Full outputs are retained, counts/distributions and residuals are plotted, and no difficult voxels are removed. Fresh CBF median 68.0607864, 10–90% 6.74127557–188.773757 mL/100g/min; 226 of 5800 voxels exceed 250. ATT median 1051.61377 ms. Native noise variance and physiological multi-compartment effects are unknown. A sizable relative signal residual 0.61462576491045 remains, below zero-signal residual 1 but not evidence of a noise-calibrated physical fit.

## References, metrics and thresholds

No actual truth file is present. `evaluation/baseline_reference.npz` contains independently executed original CBF/ATT fields. `notebooks/reference_outputs/upstream_reconstruction.npz` is a separately identified original-method comparison used by the notebook. `notebooks/reference_outputs/cleaned_metrics.npz` stores fresh metric vector (CBF NCC, CBF NRMSE, ATT NCC, ATT NRMSE), fresh signal residual, and reconstruction runtime; it is consumed by validation/review, not as a reconstruction cache. Fresh reconstruction arrays are the solver submission in output/reconstruction.npz. Frozen stage/pipeline fixtures are consumed by comparing tests, including original raw-flow-to-clinical-unit equality; no unused upstream source or opaque checkpoints ship.

Independent reruns have maximum absolute map differences zero. Additional float32 storage of clinical CBF yields dynamic-range NRMSE 9.10635987963889e-10; original ATT is already float32. Nonzero map NRMSE tolerance 1e-6 covers bounded optimizer/backend/float32-storage roundoff while remaining far from reconstruction-scale error; NCC must be at least .999999. The standard-reference identity is not used to justify a zero-error boundary. Relative observation L2 must be no more than 0.620772022559549, fixed at 1.01 times the independent original value. Every declared metric must pass. Both map metrics use the entire arrays and standard NCC/NRMSE definitions; only the observation residual uses the fixed public mask. No implicit alignment, phase conversion, cropping or normalization occurs. JSON numbers are finite, with machine precision retained.

Fresh Judge measurements:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_ncc": 1.0,
    "reconstruction_nrmse": 0.0,
    "arterial_transit_time_ncc": 1.0,
    "arterial_transit_time_nrmse": 0.0,
    "signal_relative_l2": 0.6146257649104498
  },
  "thresholds": {
    "reconstruction_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "arterial_transit_time_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "reconstruction_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "arterial_transit_time_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "signal_relative_l2": {
      "comparison": "<=",
      "boundary": 0.620772022559549
    }
  },
  "errors": []
}
```

## Execution, notebook and visual review

Cleaned CLI: `python -m src.main`, completed in 16.586 s on CPU and wrote the declared two-key NPZ. `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` returned 0. Shared runtime versions are Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. No shared package installation or upgrade was made. Minimal task requirements contain only these used numerical/visual dependencies; pytest/Jupyter remain development tools.

Full tests: `python -m pytest test/ -v --basetemp <private archive>/asl/pytest-acceptance`, 23 passed in 17.695049 s, exit 0. The machine's default C:/Users/admin/AppData/Local/Temp/pytest-of-admin is inaccessible; an explicit private temporary directory is the environment-only workaround. One earlier README text test used the Windows locale and failed decoding UTF-8; it was corrected to explicit UTF-8 and the whole suite passed. Final strengthened fixture/constant checks are rerun for acceptance below.

The six-chapter canonical notebook actively reconstructs all bound observations, evaluates both quantities and the standalone Judge, and computes three complete sensitivity reruns plus an actual single-delay approximation. Fresh alpha=.75/.95 and initial ATT=500 ms results are kept separate from the default standard fit. No missing method is represented by zero or a historical array. The eight scientific figures cover acquisition, physical transport, both five-slice fields, independent original parity, measured/predicted curves and residuals, calibration/initialization sensitivity, and single-/multi-delay comparisons. Display limits are explicitly labeled and never alter evaluated arrays.

Executed via `python -m nbconvert --to notebook --execute notebooks/asl_cbf_att_maps.ipynb --output asl_cbf_att_maps_executed.ipynb --output-dir <private archive>/asl --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=imaging-101-research`. Exit 0, wall time 110.818914 s. The 600 s timeout is comfortably above four approximately 20-second voxelwise fits and plot/export time; it is not a seconds-only requirement. `python -m jupyter nbconvert` initially delegated through PATH to an unrelated Anaconda frontend and could not discover the research kernel; invoking the shared environment's nbconvert module directly resolved that environment issue. A first notebook attempt exposed an incorrectly escaped table-string literal after six plots; it was corrected and the complete notebook was reexecuted successfully. A visual review found a long sensitivity title overlapping its neighbor; titles were shortened and the entire notebook was reexecuted again, rather than claiming an unexecuted correction.

Nonfatal notebook frontend warnings: Windows Proactor event loop uses a selector thread for ZeroMQ; kernel communication emits an unencrypted-local-TCP notice. These are captured in `asl/notebook_final.log`; no code-cell errors or numerical warnings occurred. All code cells have execution counts and no error outputs. The successfully executed notebook, including outputs, is the sole canonical task notebook; temporary executed copies remain private, not under output/.

All eight exported PNG figures were visually inspected for scientific labels, ranges, clipping and layout. The final sensitivity title correction was re-inspected. PNG/SVG/PDF versions and numerical source arrays are private under `asl/figures`; `asl/figures_manifest.json` identifies each figure. Source arrays in `numerical_sources.npz` distinguish original maps, fresh fields, predictions, residuals, approximation, sensitivity reconstructions and measured tables. No plotted fresh result was substituted from the archived original.

README and plan semantic review: checked finite-bolus signs, timing origins, units, lambda/alpha/T1 defaults, full grid and mask, source signatures, selected SciPy optimizer/default tolerances, actual CBF float32 storage followed by conversion, ATT scale, caller paths, notebook-only comparisons and metric conventions. No algorithm or archived helper is described as the standard runtime unless actually called. This review goes beyond signature matching.

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/version, exact source commit, original data hashes, equation/function/test mapping | passed | Published PDF, remote API response, hashes and traceability above |
| Original and cleaned methods on exact bound observations | passed | Actual unmodified CBFMapping runs, original exit0, CLI output and parity tests |
| Frozen independent original fixtures and physical checks | passed | Captured 15 forward cases, complete original maps, independent integral and continuity tests |
| Constants/units/precision/coordinates preserved and tested | passed | Exact input/timing arrays, point/volume tests, explicit documented normalization discrepancy |
| Standard CLI and standalone Judge on fresh output | passed | output/reconstruction.npz, output/judge_report.json, independent process exit0 |
| Judge quality rejection and invalid/evaluator/report error handling | passed | Zero prediction fails; missing/corrupt/keys/shape/int/float32/complex/NaN/Inf/hidden asset/config/report-write tests |
| Public input sufficiency and no answer/source URL leakage | passed | Self-contained README metadata/output contract; three-key raw NPZ; public URL audit false |
| Minimal shared dependencies and no upstream/domain runtime package | passed | Source import audit and shared-runtime execution; original I/O isolated |
| Explicit forward/inverse mathematics and call-role audit | passed | Local finite-bolus acquisition model and bounded objective/pipeline; no domain solver calls |
| Full unit/parity/physics/pipeline tests | passed | 23 tests; final rerun recorded in acceptance evidence |
| Active notebook complete reconstruction and real comparisons | passed | Complete nbconvert exit0, six chapters, no error outputs, three sensitivity reruns |
| Result/metric provenance and nonvacuous consistent thresholds | passed | Separate original/fresh records, known metric cases, zero prediction rejection and fixed residual boundary |
| README/plans source-consistent mathematics/interfaces | passed | Semantic review above and constants/signature checks |
| File inventory, cleanup and licenses | passed | Standard task tree; canonical notebook only; private archives/plots/logs; task MIT license |

Completion here is task-local and awaits the parent agent's independent acceptance. It does not claim completion of the six-task group or of the user's full batch.

Final acceptance rerun: 23/23 tests passed, pytest reported 17.33 s (process wall 17.695049 s), exit 0; `asl/tests_final.log` and `asl/tests_exit.json`. The final expanded physical contract was tested red first: a 1e-12 exterior value previously passed; the Judge now rejects every negative or nonzero-exterior field before scoring. The final fully executed notebook asserts Float64 observation-residual agreement with the independent Judge to rtol 1e-12, avoiding a previously observed ~3e-8 difference from float32 norm accumulation. The final notebook execution wall time was 110.818914 s, exit 0.

Parent provenance review: the README now describes M0 as author-supplied equilibrium-magnetization calibration, not derived from the hidden kinetic-fit outputs. The repository distributes a separate m0.nii.gz input but does not establish independent acquisition provenance, so that stronger acquisition claim has been removed. This wording change does not affect numerical inputs, model, references, or executed notebook.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- Author documentation: https://asltk.readthedocs.io/en/main/getting_started/; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
