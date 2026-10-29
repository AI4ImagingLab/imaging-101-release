# Provenance and validation: German seismic scattering and absorption

## Binding, measured acquisition and independence

Status: **author_real_processed**. Observations are actual German Regional Seismic Network broadband traces, not an author-generated model dataset. Author `qopen/example/create_example_files.py` reads waveform records from `eida.bgr.de`, event catalogue from NERIES and station response inventory from ORFEUS, selecting GR.BFO, BUG, CLZ, FUR and TNS. Its acquisition calls, accompanying miniSEED/inventory/catalogue and source metadata identify the real measurement chain. The author decimates to20Hz with `no_filter=True`; this inherited processing limitation is disclosed. Public velocities are captured after native inventory sensitivity division and before detrending/filtering. The calibration inventory was measured independently of inferred transport properties; no fitted target or hidden source spectrum is public calibration.

Events: ['20010623_0000004', '20020722_0000003', '20030222_0000013', '20030322_0000008', '20041205_0000033']. Station mappings in public station_index preserve each event's original input ordering. There are24 acquired records,5 bands,119 usable record/band pairs and25 complete inversions. One event lacks a fifth station; one low-frequency event/station coda fails the observation-derived duration rule. The input includes every acquired record and explicit quality masks, not selectively hidden difficult observations. Onset times preserve ObsPy microsecond precision, while Green times use continuous distance/speed. This distinction was investigated and tested, not silently rounded away.

Physical unknowns: event/frequency-dependent scattering and intrinsic absorption, source spectral energy and relative site response. This differs from older earthquake source localization, travel-time tomography and deformation tasks: it inverts actual multi-component amplitude envelopes for distinct radiative transport coefficients, rather than locations or travel-time velocity. No actual medium truth exists; the hidden baseline is explicitly a reproduced method result.

## Paper, original code and license

Selected method: Eulenfeld & Wegler2016 GJI205744–757, published version DOI https://doi.org/10.1093/gji/ggw035, intrinsic/scattering envelope inversion. Primary publisher paper identifies optimization against acoustic isotropic radiative-transfer energy densities. Foundation: Sens-Schönfelder & Wegler2006 DOI https://doi.org/10.1111/j.1365-246X.2006.03139.x; Paasschens1997 DOI https://doi.org/10.1103/PhysRevE.56.1135. References are method provenance, not a claim to reproduce the paper's full survey. We retain source/site/attenuation comparisons for the entire supplied example acquisition. Brune source-moment fitting is downstream of the selected inverse and is explicitly outside this task.

Original repository https://github.com/trichter/qopen, remotely verified commit `2031565735b4a5120cbadcc5d07916fb45b5efa9`; remote tree response archived in `../../qopen_tree.json`. Selectively archived complete required package/example source in `../../archive/qopen`; original repository MIT license and copyright Tom Eulenfeld preserved as task LICENSE. This author-distributed example data has no separate restrictive notice in the checked package; retain attribution and originating GR-network identification. Full source hash inventory is `../../archive/qopen_source_hashes.json`. No claim of later remote freshness.

## Exact acquisition and frozen fixture hashes

| File | SHA256 |
|---|---|
| `archive\qopen\qopen\example\example_data.mseed` | `8407746f526af2e73251cde299df940870730e4dc396c6d62e47c7ca3988d031` |
| `archive\qopen\qopen\example\example_events.xml` | `1680e0b2adc13f16d0dc2ff8661cf6e3b5aa3e6620c9fa95568709f1b4393e68` |
| `archive\qopen\qopen\example\example_inventory.xml` | `3d9e2a4263a69ea5b7b09979d728fea425c3031b27001c7bef17bc360bd07c5f` |
| `tasks\german_seismic_scattering_absorption\data\raw_data.npz` | `a2c7ace86d04adbe94b1a4761052705a27ecfaa63fae9fe8245d6e67ae2521a1` |
| `tasks\german_seismic_scattering_absorption\test\fixtures\upstream.npz` | `4eff78a64ec2abc11ee662db99d5b7b38e90e884c242770c81725691b041ec7d` |
| `tasks\german_seismic_scattering_absorption\evaluation\baseline_reference.npz` | `374c0478395fd066ecce270b6ee31b287aa006d8bf8c648637ef2082364f0d94` |

Private qopen_bound_velocity.npz contains exact calibrated inputs captured at the unmodified invert_fb boundary. prepare_qopen.py converts these into batched public arrays, preserving float64 without resampling or synthetic perturbation. Frozen fixture upstream.npz is derived from original dump_fitpkl outputs; all pickle files remain private in qopen_dumps. It contains measured processed energy, windows/times, Green densities and original final outputs. Consumers are test/comparing_test/test_oracle.py and standalone Judge parity checks through that test. Baseline arrays are the original output contract quantities. notebooks/reference_outputs/upstream.npz separately identifies original measurements plus weighted_log_error; cleaned.npz contains the fresh active-notebook run. These intentionally distinct consumers/provenance are not claimed independent truth.

## Original execution and success criteria frozen before local extraction

Private runtime `../../oracle-env/Scripts/python.exe`, Python3.11.11, NumPy2.4.6/SciPy1.17.1 plus ObsPy1.5.1, statsmodels0.15.0 and their dependencies. Original code runs only in this oracle environment; the cleaned environment has no Qopen/ObsPy/statsmodels imports. Native source is executed directly, not replaced by the cleaned model. Single-thread OMP/MKL/OPENBLAS=1, CPU Windows; cleaned numerical/backend record `../../archive/qopen_clean_environment.json` describes the shared NumPy/SciPy backend also used by the oracle .pth linkage. Upstream dependency installation is private, with no shared upgrades.

Commands from private archive: `oracle-env/Scripts/python.exe archive/run_qopen.py` ran original default inversion in6.1128593s, exit0, disabling plotting and parallel jobs only. `oracle-env/Scripts/python.exe archive/capture_qopen.py` reran the identical selected25 inversions in3.6277s (exact measurement metadata file retained), exit0, additionally disabling downstream seismic-moment fitting. Its wrapper records input arguments and requests original dump output; original inversion remains unmodified. Original warnings/logs are retained with the example source runtime and capture metadata. Both output JSON files preserve distinct runs. The unused downstream M0 calculation does not alter selected transport inversions.

Independent native repeat: `oracle-env/Scripts/python.exe archive/repeat_qopen.py`, exit0, 3.19815860s; maximum relative change 0.0. Result JSON and stability record are private. Deterministic method uses no random seed. Original outputs are `qopen_original_results.json`, `qopen_bound_results.json`, `qopen_repeat_results.json`; dump fixtures are in `qopen_dumps/`. Capture scripts and logs are outside task packaging.

Before local extraction, qopen_preimplementation_protocol.md froze stage tolerances: energy/bulk rtol1e-10, time atol2e-6seconds, Green rtol2e-6, weighted solution1e-6, full output relative0.005, log residual1e-4 and proposed Judge log-ratio bound0.02. Final bound0.005 tightens that predeclared ceiling. No threshold was enlarged after cleaned results. Full stage measurement: `{"stage": {"energy": 0, "unsmoothed": 0, "tcoda": 0, "Ecoda": 0, "tbulk": 0, "Ebulk": 0}, "shape_errors": [], "final": {"reconstruction": 2.998484234106537e-15, "source_energy": 3.6343559298394874e-15, "site_amplification": 3.7457265500971996e-15}, "seconds": 1.3134905999759212}`. The independent repeat was exactly stable; float32 submission changes were `{"reconstruction_max_abs_log_ratio": 5.215923024021994e-08, "source_energy_max_abs_log_ratio": 4.708436673708229e-08, "site_amplification_max_abs_log_ratio": 5.729535126517141e-08, "forward_error_boundary_ratio": 0.9514970090439773}`. Thus a nonzero0.005 tolerance accommodates numerical search differences while remaining scientifically selective; it is not a self-comparison zero-error threshold.

## Equation/stage traceability

| Mathematical stage | Original file/function | Local function | Independent evidence |
|---|---|---|---|
| Three-component spectral energy, effective two-pass bandwidth | core.py observed_energy/filter_width and invert_fb | preprocessing.observed_energy/filter_width | All119 pairs frozen energy/unsmoothed arrays; quadratic amplitude test |
| Noise, bulk representative time and coda windows | core.py invert_fb | preprocessing.prepare_pair/prepare_band | Frozen119 coda grids and25 bulk vectors; stored timestamp convention test |
| Paasschens3D scattered density/direct pulse | rt.py G/rt3d | physics_model.coda_green/green | Frozen119 Green arrays/25bulk vectors; causality/direct integrated weight/inverse-volume scaling |
| Bartlett smoothed Green and bulk quadrature | core.py invert_fb | physics_model.green_observations | Original Gcoda/Gbulk numerical fixtures |
| Weighted log-energy variable projection and source/site gauge | core.py invert_fb/lstsq using statsmodels WLS | solvers.design_system/weighted_solve/solve_scattering | Original full fits; independent normal-equation stationarity and analytic least-squares case |
| Golden scattering profile under physical bounds | core.py invert_fb optimization | solvers.reconstruct_band | All25 original outputs at frozen0.005 relative bound; actual discrepancies below4e-15 |
| Independent forward consistency and output contract | Original residual error definition | evaluation/judge.py standalone explicit equations | Same native preprocessing/Green fixtures; original per-fit errors; malformed/reference tests |

No suspected formula error is concealed. Important approximations: isotropic homogeneous effective medium; bulk absorption at energy-weighted time rather than inside Green quadrature; first post-arrival coda bin omitted per upstream; correlated coda; no independent GT; gauge-dependent source/site scale. The full paper survey and seismic-moment conversion are not silently represented as completed experiments.

## Evaluation and metric consistency

All three required output quantities are evaluated. Maximum absolute natural log-ratio must be<=0.005 for transport (g,b), source W and valid site R separately. Site invalid padding is exactly0; site geometric-mean gauge log tolerance1e-5. Per-fit forward error boundary is1.05 times the corresponding original weighted log-energy error +0.001, fixed from upstream output before Judge evaluation. Boundaries are nonzero, meaningful and not Gaussian uncertainty claims. Original errors range0.493519–1.025165; noisy regional data/model mismatch is stated, not hidden by rescaling.

Judge is standalone and never imports submitted code. It independently repeats explicit observation reduction and forward transport; tests compare its processed arrays to original fixtures and its errors to original weighted errors. It validates dtype/shape/finite/positivity/padding/gauge, actual-truth precedence, malformed hidden references, missing assets/configuration, external cwd and report-write failure. Invalid input yields empty metrics and exit2; valid poor results exit1. Metric visualization has an independently known log-ratio1/2 test. No NCC/NRMSE is used because mixed physical parameter scales and positive amplitudes make log ratios more interpretable.

## Executed commands, notebook and figure review

From task root, shared `research-runtime/Scripts/python.exe -m src.main`: exit0, all25 fits, output/reconstruction.npz. Standalone `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit0/pass. Full `python -m pytest test/ -q`: **25 passed in11.57s**, covering stages, physical checks, CLI and Judge contracts.

First test pass exposed a mistaken exact equality assumption between continuous distance/speed and microsecond timestamps; corrected test now checks exact rounded timestamp identity, without modifying measurements. A separate CLI child process once raised a SciPy docstring-import TypeError; direct CLI, focused rerun and full final suite passed without package/code changes. The cause is unresolved and is not attributed to a specific environment defect. It does not replace the recorded successful final executions.

Complete active notebook command: `python -m jupyter nbconvert --to notebook --execute notebooks/german_seismic_scattering_absorption.ipynb --output german_seismic_scattering_absorption_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180`, exit0. Kernel explicitly selects shared runtime using private JUPYTER_PATH. Timeout180s comfortably exceeds measured ~seconds reconstruction and all diagnostics. Jupyter emitted Windows Proactor selector-thread fallback and local TCP transport warnings; no cell errors. Canonical notebook contains executed outputs; temporary executed duplicate removed. Numeric printed outputs are archived in qopen_notebook_outputs.txt. All fresh computations use local src; original NPZ is comparison only.

**Four figure groups /16 scientific panels** were actually opened with image viewing: acquisition(4), spectra(4), diagnostics(4), ablation(4). Colorbars/twin axis are not extra scientific panels. Review found crowded frequency tick labels and low-contrast availability counts, corrected with explicit band ticks/white counts and reran full notebook. Waveform/energy/fit context, independent-original crosses, source and site spectra, residual/profile coupling, and actual25-fit fixed-scattering ablation are preserved. Fixed-scattering ablation has one nonphysical negative absorption; it is explicitly counted as an unconstrained sensitivity comparison, never accepted as benchmark output. Figures are private archive/review/qopen_*.png; no screenshots or temporary scripts are shipped.

## Semantic, import and interface review

README specifies all eight public arrays, batch dimensions, real dtype/units, catalogue/station identity, effective filter bandwidth, noise/window/boundary conventions and three output keys. Public inputs contain no GitHub URL or target-derived calibration. README and approach/design match actual functions and flow, including SVD cutoff, residual normalization, source/site gauge and golden stopping. No cross-task imports or domain solver calls exist. Permitted numerical calls: SciPy filters/detrend/Hilbert/frequency response, NumPy linear algebra/convolution, SciPy generic scalar optimizer, Matplotlib plotting. These are mathematical primitives, not hidden reconstruction engines. No generator or training/checkpoint is present.

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/acquisition hashes/stage mapping | passed | Above plus private source/hash archive |
| Exact-bound original and cleaned executions | passed |25 full inversions, original repeat, CLI and notebook |
| Independent stage parity and physical tests | passed | Frozen original arrays; analytic scaling/least-squares tests |
| Constants/units/precision/coordinates | passed | Microsecond timing preservation and explicit conventions |
| Fresh CLI and standalone Judge | passed | output/reconstruction.npz and judge_report.json |
| Negative/error Judge contracts | passed | Missing/dtype/shape/finite/physical/reference/config/report tests |
| Public self-contained contract/no leakage | passed | README/data/requirements review |
| Minimal runtime/no upstream imports | passed | Shared runtime full executions and symbol audit |
| Complete tests | passed |25 final tests |
| Active full notebook and comparison figures | passed | Executed canonical notebook;16 reviewed panels |
| Provenance/metric threshold separation | passed | Original vs fresh vs historical explicitly labeled |
| Plans/source semantic consistency | passed | Formula, signature, units and call flow review |
| Final inventory/license/cleanup | passed | Template files; MIT retained; private archive external |

Root independent acceptance remains pending; this document records worker verification, not a claim of parent approval.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 4 computed PNG outputs, 40 formulas. Private evidence: research_batch10_20261027/acceptance/german_seismic_scattering_absorption/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Portable notebook kernel metadata

Final packaging replaces the maintainer-only kernelspec name batch10-worker with the standard python3 name. All code cells, Markdown cells, execution counts, outputs, scientific arrays and non-kernel metadata are unchanged; parent execution still used the explicitly recorded validated shared interpreter. This is a metadata-only portability amendment, not a new numerical execution. The shared environment must be selected when running the notebook. Private before/after hashes and cell equality evidence: acceptance/german_seismic_scattering_absorption/portable_kernel_amendment.json.
