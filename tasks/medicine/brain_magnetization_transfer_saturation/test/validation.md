# Provenance and validation: brain MT saturation

## Binding and measured-data evidence

Domain medicine. Physical unknowns are effective macromolecular MT saturation and longitudinal relaxation rate from three FLASH volumes, distinct from existing T1-only relaxometry and QSM field-to-susceptibility inverse tasks. Parent reserved this acquisition/unknown pair before extraction. Classification **author_real_processed**, not simulated and not a claim of raw k-space access. No independent physical truth is supplied.

Original method repository: https://github.com/qMRLab/qMRLab ; remotely verified tree/commit **c1d75f4ee2a6f8af2ebe831dcf39436d2e7d73c1**. Selective original sources MTSAT_exec.m and src/Models/Magnetization_transfer/mt_sat.m are archived unmodified outside this task. The source license is MIT, copyright2017 NeuroPoly, preserved in NOTICE. This is a modern qMRLab implementation of the published corrected method, not a claim to recreate the2008 software environment.

Paper: Helms et al., MRM60,1396–1407(2008), https://doi.org/10.1002/mrm.21732 ; corrected two-point equations7a/7b in the2010 erratum, MRM64,1856, https://doi.org/10.1002/mrm.22607 . Publisher erratum web text and both equation images were inspected; private GIF/PNG captures indexed below. Full original paper PDF was not downloaded (publisher403), and no claim of a private full-paper archive is made. The task-relevant MT equation is explicitly documented in the original implementation and independently checked by algebraic roundtrip; the corrected R1/amplitude equation images match this implementation. Software paper Karakuzu et al., JOSS2020 DOI10.21105/joss.02343. Author teaching context https://qmrlab.org/mooc/magnetization-transfer-imaging/ distinguishes MTR relaxation confounding and effective MTsat from microscopic qMT.

Data chain: original author qMRLab OSF https://osf.io/c5wdb/download?version=4 , file mt_sat.zip. API https://api.osf.io/v2/files/c5wdb/ records creation2017-11-14, modification2022-05-26, version4 and SHA256. Public node https://osf.io/tmdfu/ is described as the MRI dataset for qMRLab; API license563c1cf88c5e4a3877f9e96a is **CC-By Attribution4.0 International**. The same author distributed ds-mts.zip at https://zenodo.org/records/21696048 (v2,2026-07-30), also explicitly CC-BY4.0 in its API. Original OSF contains fitted outputs too; those were not supplied to solvers or used as upstream-run evidence. The Zenodo package excludes fit outputs.

Primary acquisition evidence: author ISMRM2018 abstract2288, https://archive.ismrm.org/2018/2288.html (archived), figure2 caption identifies the example datasets: “all datasets are from in vivo human brain”, except two specifically named spinal-cord datasets; MTsat is explicitly included. The source file already existed in2017, linking the example acquisition to that period. Dataset conversion provenance https://qmrlab.org/qmrust/getting-started/ documents download of qMRLab OSF examples and lossless BIDS conversion. This author statement is strengthened by **direct numerical identity**: each original signed-int16 OSF PDw/MTw/T1w voxel array equals each Zenodo float64 array exactly, all1572864 voxels per image. No resampling, smoothing, reorientation or value change occurred. Final NPZ is these complete arrays with a leading batch dimension. The source header scaling is slope1/intercept0; integer-to-float64 conversion is exact. Realistic anatomy was visually reviewed, but visual plausibility alone is not the evidence.

Header values: dimensions128×128×96, voxel spacing1.5mm, TE text3.299999952ms. Header repetition time .02800000086426735s repeats in all files, including T1w. The author class configuration and Zenodo sidecar specify T1w .018s with20degrees versus PD/MT .028s with6degrees. **This conflict is retained and disclosed**, not silently harmonized. The selected original function is invoked with its explicit author protocol. A fresh wrong-header-TR ablation quantifies the discrepancy. Exact scanner model, B0 strength, MT pulse parameters, subject identity and preprocessing details cannot be established per-file and are not invented. These are limitations of microscopic/clinical interpretation, not needed constants of the supplied effective-saturation model.

## Original execution, frozen criteria, and numerical investigation

Original MTSAT_exec was run unmodified on all exact complete volumes in WSL Ubuntu Octave8.4.0, with OMP/OPENBLAS threads1. Installed Octave packages control4.0.1,image2.14.0,signal1.4.5 were inspected; no package installed and none needed by this function. A SciPy MAT conversion only replaces NIfTI loading; data identity is checked before execution. Capture source and logs are private. Original initial unmasked function runtime0.1183s; bound masked run0.1491s, independent repetition0difference. Final audited rerun below confirms success. No RNG or seed is used by the inverse.

The complete head includes low-signal voxels with nearly zero inverse denominators; unrestricted outputs can reach10^16 and cannot support meaningful truth-like whole-volume scoring. Before local extraction a public observation-only region was frozen: central ellipsoid and image thresholds plus normalized inverse-denominator threshold. The ellipsoid contains253021 voxels;219351 pass (86.69% of the ellipsoid,13.95% of full volume). It is an analysis region, not an anatomical segmentation. Valid-region MTsat spans−5.486680425206359 to25.5363728260141percent;478 negative estimates are retained. R1 spans0.11387933 to6.17396462s^-1. No fitted target is supplied as calibration; the mask is recomputed from visible observations and fixed public indices.

The original masked code multiplies arrays by0 outside the mask, which can leave NaN because0*NaN isNaN. Initial Judge validation exposed these **outside-domain oracle values**, not a local algorithm mismatch. The private capture explicitly assigns zeros outside the declared region in both original outputs. This is documented output-domain handling; inside-domain original numbers are unchanged. The local code evaluates only the fixed region and allocates zero outside. The first capture and explicit-zero amended capture are described here; all shipped reference arrays use the amended capture, and mandatory finite/shape contract passes. No threshold was relaxed.

`mts_pre_extraction_contract.json` froze mask and tolerances before local extraction. Original independent reruns were bit-identical; a nonzero absolute numerical allowance was fixed (1e-5percentage points MT,1e-6s^-1 R1), cosine.999999, outside1e-12, forward relativeL2 residual1e-6. Stage parity1e-10 absolute and relative is far above observed machine-roundoff but far below tissue variation. Fresh local outputs on the shared runtime equal the original outputs in this run (RMSE0,NCC1 for both). Observation relativeL2 residual1.7025282009375949e-16. These bounds measure numerical reproducibility, not physical accuracy; the3-observation3-unknown fit has no independent residual degrees of freedom.

## Equation/function/test traceability

| Published/reference stage | Original source | Local implementation | Independent validation |
| --- | --- | --- | --- |
| Measured FLASH loader and unit conversion | mt_sat.fit plus MTSAT_exec double conversion | preprocessing.load_data; physics_model.ANGLES/TR | Direct OSF/Zenodo/allNPZ identity and README constants test |
| Corrected Eq7a R1 | Erratum2010 equation image1; MTSAT_exec no-B1 branch | solvers.reconstruct r expression | Frozen original R1 full volume; analytic known R=.8 case |
| Corrected Eq7b amplitude | Erratum2010 equation image2; MTSAT_exec A expression | solvers.reconstruct amplitude | Analytic forward-inverse and all3 observed signal roundtrip |
| MT saturation effective signal loss | Helms2008 MT model; MTSAT_exec MTsat expression | physics_model.flash_signal,solvers.reconstruct delta | Frozen independent original MTsat; monotonic saturation and gain invariance |
| Mask boundary | MTSAT_exec optional Mask plus documented capture zero assignment | preprocessing.analysis_mask; zero allocations in reconstruct | Fixed219351 coverage; Judge zero-outside metric |
| MTR comparison | mt_sat.fit MTR expression | solvers.mtr | Algebraically known ratio test and fresh spatial figure |
| FLASH approximation limitation | Independent Bloch steady state | physics_model.exact_flash_signal | Small-angle short-TR asymptotic test; actual-image discrepancy histogram |
| Metrics | Independent definitions in contract | Judge and visualization.comparison_metrics | Known RMSE/cosine cases; malformed/poor/CLI/error tests |

No adjoint, iterative gradient, proximal update or stochastic solver exists in this closed-form method, so those gates are not applicable. General ndarray arithmetic performs every core inverse step locally. Optional no-B1 branch only is selected because no B1 measurement exists; adding an invented B1 map would change the binding. No prohibited reconstruction/domain library, cross-task import or upstream oracle import occurs in source, Judge, tests or notebook. AST import records are archived and actual call roles were reviewed: NumPy arithmetic/I/O, Matplotlib visualization, standard-library subprocess test execution; no opaque inverse solver.

## Notebook and result provenance

Canonical notebook has explicit Problem/Data/Physics/Method/Results/Conclusion sections and runs full local reconstruction. It compares independent original outputs, plots three measured images, mask/conditioning, spatial original/fresh/error, R1/MTR/MTsat, predicted-observed/residual/Bloch discrepancy, wrong-TR maps, and five1%signal-noise perturbation bars. **All seven figure outputs were visually inspected** via private mts_figure_review PNG captures: axes, scales, colorbars and measured comparisons legible; no missing methods or fabricated zeros. A zero difference image reflects actual equality. Tissue display limits do not clip arrays, and signed extrema/negative counts are printed. The benchmark remains actual measured data; synthetic perturbations are supplementary and explicitly labeled. Perturbation comparisons use the shared accepted mask, counts219165–219199 reported; no seed equality claimed. Wrong-TR RMSE1.2047672795243158percentage points and.38548759144657935s^-1 illustrates the metadata conflict. Reconstruction itself~.07s; timeout180s covers full review including I/O and plots.

No historical paper performance number is reused. Hidden baseline_reference and notebook upstream are **this task's independently executed original outputs**. Notebook cleaned_measurements and output/reconstruction are **fresh local outputs** on exactly the hashed raw data, same nominal protocol/mask. Test fixture keys output_mtsat/output_r1 are consumed by test_original_pipeline. Notebook upstream arrays are consumed by spatial comparisons/metrics; cleaned_measurements records active run results. Evaluation contains no ground_truth file because truth is unavailable. Judge, visual metrics and notebook share explicit formulas with independent known-case tests; RMSE is absolute, not mislabeled dynamic-range NRMSE.

## Runtime and command evidence

Shared cleaned runtime Python3.11.11 Windows10build26200, NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1; CPU single-thread. Source mathematics does not depend on upstream Octave or domain packages. Source/runtime versions and numeric backend inventory are additionally recorded in private runtime_environment.json/runtime_backend.txt from the same validated runtime. RAM/disk headroom checked before large downloads; volume arrays are modest and neither GPU nor shared installation needed.

Native nbconvert invokes the existing runtime through a private JUPYTER_PATH kernel, not a task-packaged absolute environment path. Kernel warnings about frozen debugger modules, Windows zmq selector fallback and local TCP transport were emitted; execution exited0 with no cell error. Pytest uses a private basetemp outside task to avoid the inherited Windows system-temp permission problem previously observed in the same workspace. No permission bypass or library change was needed. Log paths are relative to the private bio_med archive.

| Gate command | Exit | Wall seconds | Private log |
| --- | --- | --- | --- |
| original: `wsl -e bash -lc OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 octave --no-gui --quiet /mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch10_20261027/bio_med/capture_mts_bound.m` | 0 | 1.046647 | mts_original.log |
| cli: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main` | 0 | 0.506328 | mts_cli.log |
| judge: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.431366 | mts_judge.log |
| tests: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v --basetemp=../../pytest_mts_final` | 0 | 5.361923 | mts_tests.log |
| notebook: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/brain_magnetization_transfer_saturation.ipynb --inplace --ExecutePreprocessor.timeout=180` | 0 | 8.774353 | mts_notebook.log |

17 tests passed; final notebook executed0errors. Plans reviewed against source signatures, equations, signs, units, zero boundary and actual notebook flow. README constants and output contract agree with source. Files downloaded privately include originals/results not distributed to solvers. Public README/raw/requirements audited for GitHub URLs and target leakage; only three measured image arrays are visible.

## Private archive index and SHA256

Archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch10_20261027/bio_med/`. Complete downloads, original source, equation-image evidence, capture/conversion/build scripts and logs remain here, outside task. `mt_source_archive/` is original OSF extraction; `mts_archive/` is Zenodo extraction. No cached source FitResults was used as an oracle. `mts_validation_records.json` contains exact commands, runtime records, hashes and import audit; `mts_figure_review/` contains private visual-QA captures.

| Artifact | SHA256 |
| --- | --- |
| MTSAT_exec.m | 5e9fe490f676e13ea414fd042bbcc42a4374423bfc22d27f613fb1a2da9b0295 |
| qMRLab_src_Models_Magnetization_transfer_mt_sat.m | 5071d0409c9ab19e19631e09142393dd94728c6e1cb3f11dc00a57dabb11b812 |
| qMRLab_LICENSE | 656bac472ece46f50c88d32eb9c835ff775e093568f764b50e88e9fc37f3c428 |
| mt_source.zip | 23c212fe7852e77e7b972dd512a68c12be2fb26c3a6e0a229d87fef211784c5e |
| mts.zip | 7c96b08454a2c258b2cf9b5b308fdd22256d4cca35eba7ae47af849ec32b8b3b |
| mts_bound.mat | c8da839d44e6b5027bad3daec14af8e9bf01fca607e2f29bd788728aea468257 |
| mts_bound_oracle.mat | 0e7173d60f06e98e7a4009918fd369a8750e41d02eb39497dc562d1da6674837 |
| mts_pre_extraction_contract.json | 007cf566abbe81857cfc11916a65425a6434737ab504c5b85ebc1ceb28b37f14 |
| mts_observations.npz | 2e87387df3722a4989952e542c1ade83daed325374cb40c10ef98b4835ecc76e |
| capture_mts_bound.m | 49aa56af890716a25cd62462b750ed909e2d0396da987b08b1b704099c51013c |
| mts_erratum_eq1.gif | af4c3071260e7aca7f2e2168f02be7d08eaa7f873a1a838d1e72893fbfb718b1 |
| mts_erratum_eq2.gif | 2c58b265323a4e9f658ef942a02c58b383b542d603660242aa28301e7b2dbae6 |
| ismrm2288.html | 5890133934d97924edcfd51979f5df369328ab160a6158eb3c5b2678310ef77a |
| zenodo_metadata.json | e5dbf803ff91075fe2a89b97046a1df369e868f51089e5c6a4e18afaaa98de39 |
| osf_metadata.json | 9aa3de7461188f2bf67901ad88eeaca84d6f38d91b7444e4b872cfa9dd2a38c5 |
| mts_osf_node.json | 8ae3c5ad93c8df06a6b06cf55962c5c64f8d997410f1d884adbdb5b425b15c4d |
| mts_osf_license.json | 32fb3d7b32be0843803f96424fe7739f69574481211a2d323e16f8da8536c635 |
| task/data/raw_data.npz | 5aa5ecaa633b1f7425918691ab29404af8518e730bdbc44307f4075af442a14a |
| task/evaluation/baseline_reference.npz | 72cbc07cd8fb40103d08b9f591632a8a50b3795c1a002d307f4a0e91e3c1d159 |
| task/notebooks/reference_outputs/upstream.npz | 72cbc07cd8fb40103d08b9f591632a8a50b3795c1a002d307f4a0e91e3c1d159 |
| task/notebooks/reference_outputs/cleaned_measurements.npz | 27978e617897605d3935956b7b811258873b010e22bccce965ea686e4ae2b861 |

## Mandatory gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/revision/data/mapping | passed | Corrected equation images, primary author acquisition evidence, hashes and mapping above |
| Original and cleaned exact-data execution | passed | Original Octave and fresh local logs, zero original-repeat difference |
| Independent parity and physics | passed | Full-volume original fixtures and analytic/Bloch/scale tests |
| Constants/units/coordinates | passed | Lossless integer data equality, explicit TR conflict and concordance test |
| CLI and standalone Judge | passed | Both required arrays scored, valid output passes |
| Judge rejection and errors | passed | Zero poor output, missing/shape/dtype/complex/NaN/Inf/config/report-write tests |
| Public sufficiency and no leakage | passed | Only raw measured images and self-contained public metadata |
| Minimal runtime and explicit math | passed | AST/call-role review; local equations, no domain solver |
| Unit/parity/pipeline tests | passed |17 tests |
| Active full notebook | passed | Native nbconvert,7 figures inspected |
| Reference labeling/metric consistency | passed | No GT claim, original/fresh identified, independent metric cases |
| README/plans consistency | passed | Mathematical and signature review |
| Template/license/cleanup | passed | Required layout only, notices retained, caches archived outside task |

File-specific scanner identification and independent physical truth remain **unavailable**, explicitly labeled as scientific limitations rather than fabricated evidence. No mandatory execution gate is left failed or unverified. This task is ready for parent review, not self-approved final release integration.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 7 computed PNG outputs, 26 formulas. Private evidence: research_batch10_20261027/acceptance/brain_magnetization_transfer_saturation/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Final minimal-dependency documentation amendment

Independent cross-task and parent import/call-role audit confirmed SciPy is unused in the cleaned source, Judge, tests and notebook. Removed its redundant requirements entry and corrected the design runtime description. SciPy remains installed in the shared environment and was used privately for the original MAT adapter; those historical environment records remain accurate. No executable source, code cell, observation, reference, metric, threshold or scientific output changed. Prior complete execution evidence remains applicable; no fresh numerical rerun is claimed for this documentation-only amendment. Private before/after hashes: acceptance/brain_magnetization_transfer_saturation/dependency_amendment.json.
