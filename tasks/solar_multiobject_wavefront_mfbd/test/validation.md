# Provenance and Validation

This is the maintainer-only evidence ledger. Entries marked UNVERIFIED are not completion claims. The public task has no actual physical ground truth; reference agreement is method reproduction.

## Frozen sources and exact observations

- Repository: https://github.com/aasensio/torchmfbd . Remote HEAD `b27a963cdb995cdd73695a17cddd589cb6367421` was actually verified with `git ls-remote`, rather than inferred from a local tracking branch.
- Source archive: `research10/task10/upstream.zip`, extracted exact-revision source under `source/torchmfbd-b27a963cdb995cdd73695a17cddd589cb6367421`. The incomplete history clone `upstream/` was canceled and is not source evidence.
- Paper: arXiv:2505.10639v1, page margin explicitly states v1 and 15 May 2025; manuscript front matter says June 12, 2025. Freeze the actual downloaded bytes rather than claiming the later published typesetting is identical. PDF `root-torchmfbd-arxiv.pdf`, 29,177,301 bytes, SHA256 `6c25bd2643247dc645285133de88c61098f0b546dcc396f6adaba611e1cf2cb3`; extracted text is `root-torchmfbd-arxiv.txt`. Published version DOI `10.1051/0004-6361/202555530` (A&A 703, November 2025) is identified by Crossref and OpenAlex records. Publisher PDF requests produced 769-byte HTML access-denial pages; these are not paper evidence. The selected exact code is newer than the frozen preprint; implementation details are checked against code and discrepancies are listed below.
- Real observations: https://cloud.iac.es/index.php/s/EqMGsqBeyfq6Bnr , file `obs/qs_20190801_081547_8542_npix512_original.h5`. Full download is 25,167,872 bytes; SHA256 `b9294e228b7000be77cf4536befee318023be7e7f8fba02b51531fabc5941c1e`.
- Original HDF5 `im`: `(1,2,12,512,512)`, float32, all values retained exactly as NPZ `observations`; no simulations, new noise, resize, crop, float conversion or exposure selection. The exact conversion equality is tested.
- Public theoretical `kl_mixing`: first 54 one-based KL-index records from upstream `src/torchmfbd/kl/kl_data_npy`, 436 by 3 float32 values, unchanged coefficients. This is a Kolmogorov/Zernike basis calibration independent of the observed Sun, not an estimated wavefront or target-derived calibration.
- All private paths in this ledger resolve under `D:/yan1/agent/AutoSkill/imaging-101-maintainer/` unless explicitly absolute. Full source, paper, archives, original FITS and HDF5, capture scripts and logs remain outside the released task directory.

## Upstream oracle and runtime separation

Oracle environment: `research10/task10/oracle-env`. It is a distinct venv referencing the unchanged common research numerical environment through a `.pth` file. Only this oracle environment contains upstream dependencies such as astropy, scikit-image, einops, PyYAML, scikit-learn, dict-hash and nvitop. The private `oracle_driver.py` imports the unmodified exact-revision library. Neither cleaned solver, tests, Judge nor notebook imports that driver or domain package.

Shared cleaned numerical runtime: `research-runtime/Scripts/python.exe`, Python 3.11, NumPy 2.4.6, SciPy 1.17.1, Torch 2.9.1+cu128, Matplotlib 3.11.1. GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB. Complete OS/driver/backend and package freeze are archived with command evidence. Per-task requirements document the subset; no shared package was installed or upgraded for this task. HDF5 and FITS tools are used only for private source conversion/comparison capture, not the cleaned runtime.

CPU-stage command: `oracle-env/Scripts/python.exe oracle_driver.py --stage cpu`, with OMP, OpenBLAS and MKL threads set to one. This executes original preprocessing and a one-update, full-twelve-frame/two-channel first-region reconstruction, and then captures a nonzero 54-slot PSF/loss/gradient case. It is stage evidence, explicitly not the full benchmark run. Captured `cpu_upstream.npz` is frozen unchanged as `test/fixtures/upstream_stages.npz`; all arrays are consumed by stage parity tests. Seed 0 is recorded, but selected zero initialization and fixed observations use no generated random measurements.

The CPU stage completed successfully before local extraction. Its bound first-patch loss was `22.281057357788086` on the initial two-mode update; the independent nonzero full-mode loss fixture is `23.7374267578125`. Its raw mode variables are uncentered; tests compare their canonical tip/tilt form to local output. Capture environment and numeric backend are distinct from the cleaned source, although intentionally matched in package versions.

Full original command: detached `oracle_driver.py --stage gpu --wait`; it waits for the maintainer's explicit GPU scheduling marker, then executes all 121 overlapping regions simultaneously (`simultaneous_sequences=200`), both channels, all twelve frames and 250 original Adam updates. PASSED: independent complete original runs use the exact observations and selected configuration. The repeat completed in 16.92630386352539 s, exit 0, final reduced loss 8.760294914245605. A separately instrumented original trace completed in 18.73611330986023 s with the same final loss. `gpu_upstream_initial.npz`, `gpu_upstream.npz` and the trace agree exactly in all common final loss, coefficient, object and mosaic arrays. `gpu_oracle.log`, `gpu_oracle_repeat.log`, `gpu_status.json`, `gpu_trace_status.json` and `independent_repeat_evidence.json` are private evidence. The earlier `gpu_status_initial.json` contains a running snapshot, not a completion assertion; complete arrays and final log evidence establish that first run.

## Preselected extraction checks and independent physics

The original numerical fixture was captured before extraction. Acceptance checks retain exact equality for observation conversion, extraction, pupil, phase basis, taper, frequency radius and diffraction mask when identical operations permit it. Noise tolerance is relative `1e-6`, absolute `1e-8`; PSF `rtol=2e-6, atol=1e-8`, complex OTF `rtol=3e-6, atol=1e-9`, loss `rtol=2e-6`, gradient `rtol=3e-5, atol=2e-6`, first update `rtol=2e-6, atol=1e-7` and filtered objects `rtol=2e-6, atol=2e-6`. These are float32 extraction checks, not claimed decimal exact physical truth. CPU stage output currently agrees exactly for basis, pupil, PSF, loss, first update and objects; the compared gradient differs by at most `4.76837158203125e-7` through allowed accumulation ordering.

Noise divergence was investigated rather than tolerated away: an initially contiguous local patch stack produced a covariance-SVD noise difference around `5e-6`. Independent value comparison showed identical observations but different strides. Reproducing upstream's unfold/permute layout removed the difference, yielding identical measured noise. This is why extraction explicitly preserves strides.

Independent mathematical checks, in addition to upstream parity:

| Property | Independent check | Acceptance |
|---|---|---|
| Noll indexing | Analytic first six modes | Exact integer orders |
| PSF physics | Nonnegative PSF, total flux one | float32 `rtol=2e-6`, `atol=2e-7` |
| FFT normalization | OTF DC equals `1/88`, inverse is real | `atol=1e-8` |
| Circular observation operator | Real inner-product adjoint identity in float64 | `rtol=atol=1e-11` |
| Reduced objective derivative | Central directional finite difference, float64, step `1e-5` | `rtol=atol=2e-7` |
| Translation gauge | First-exposure two tip modes vanish; all other coefficients unchanged | Direct array checks |
| Null slot | Altering slot 53 leaves all PSFs invariant | Direct PSF comparison |
| Mode continuation | All nine radial-degree endpoints in monotone 250-update schedule | Exact integer checks |
| Mosaic geometry | Analytic detector-row ramp in 121 patches | Correct `[6:482]` coordinates; epsilon edge attenuation separately identified |

The initial derivative check exposed a float32 accumulator even when testing float64 inputs. The cleaned loss now initializes its accumulator from input precision; float32 benchmark behavior is unchanged, and the independent double-precision derivative check passes. The mosaic test excludes only the analytically identified extremely small-weight edge region when asserting constant-gain ramp reproduction; no benchmark evaluation crop is introduced. Zero Hann endpoints and the preserved `1e-10` denominator explain the outermost zeros and adjacent attenuation.

## Paper equation / upstream stage / local implementation traceability

| Paper equation or stage | Exact upstream function | Local function | Validation consumer |
|---|---|---|---|
| Eq. (1), (5)-(8): local imaging | `Deconvolution.compute_psfs`, convolution in `deconvolve` | `pupil_to_psf`, independent Judge observation calculation | PSF flux/DC and circular adjoint tests |
| Eq. (11)-(12): analytic object / reduced objective | `compute_loss`, `compute_object`, `get_su_s2` | `reduced_loss`, `project_objects` | Frozen loss/gradient/object and directional derivative cases |
| Eq. (20)-(22): pupil phase expansion | `_define_basis`, `KL.precalculate`, `ZernikeNaive.Z_nm`, `compute_psfs` | `noll_indices`, `build_optics`, `canonical_modes`, `pupil_to_psf` | Frozen basis/pupil/PSF, null-slot and gauge checks |
| Eq. (25), Sec. 4.1.1: spectral support | `lofdahl_scharmer_filter`, `compute_diffraction_masks` | `scharmer_filter`, `build_optics`, `project_objects` | Frozen final objects and diffraction-mask equality |
| Sec. 4.1.2: complexity schedule | `compute_annealing`, delayed `n_active` assignment in `deconvolve` | `annealing_schedule`, `optimize_wavefronts` | Exact endpoints and first Adam update |
| Sec. 5: apodization, patches, OLA | `util.apodize`, `Patchify4D.patchify/unpatchify` | `cosine_window`, `apodize`, `extract_patches`, `mosaic` | Frozen taper/preprocessing and coordinate ramp |
| Sec. 6.2: real CRISP burst | original `qs_8542_joint.py` | `load_data`, `normalize_frames`, `reconstruct` | Full bound pipeline and data hash tests |

## Investigated discrepancies and scope

1. The code's telescope logger prints diameter "m", while its formulas and config require cm. The physical model uses 100 cm, not a 100-m telescope. README, local constants and unit tests preserve the formulas' actual unit.
2. Upstream `KL.precalculate` allocates 55 modes, fills KL indices 1 through 54, then removes the piston row, leaving 53 nonzero basis functions and one zero slot. This is retained and declared explicitly. No recovery score is assigned to an arbitrary null coefficient; the output convention fixes it to zero.
3. The pupil grid is integer-centered but KL functions use an inclusive symmetric grid. Preserve both; independent PSF conservation/derivative checks do not pretend those grids coincide.
4. Upstream `compute_loss` excludes both entire frequency axes with `[1:,1:]`, despite its comment saying zero frequency. The plans and tests document the actual exclusion rather than copying the misleading comment.
5. The code constructs but does not step a cosine LR scheduler. The implemented learning rate remains 0.08 for all 250 updates. No local scheduler is activated.
6. Reduced loss uses one scalar mean noise variance per channel. The final object instead uses per-region/per-frame weighting. This is a preserved distinction, not claimed exact optimization of one uniformly weighted objective.
7. Both data and OTF use orthonormal FFTs, but object spectra invert with ordinary inverse FFT. Because OTF DC is `1/88`, the combined formulas recover the correct intensity scale. Independently changing only the inverse normalization would be wrong.
8. Original `qs_8542_joint.py` requests comparison `camXXV_2019-08-01...fits`, absent from the actual public archive. The archive provides `camXIX_2019-08-01...fits`. Identity was independently checked by full-field template correlations: actual `camXX` matches only the broadband mean (NCC 0.93235 vs narrowband 0.12432); `camXIX` matches only the narrowband mean (NCC 0.96878 vs broadband 0.13579). Both best matches share the same translation. These are comparison-file naming/registration issues outside the reconstruction library; the original library is unmodified.
9. The full 512-square input yields a 476-square mosaic after no-padding tiling and six-pixel patch-edge removal. Neither the missing final 24 input rows/columns nor the outer zero-weight mosaic boundary is silently filled or described as a 512-square reconstruction.
10. Upstream current repository root LICENSE contains the MIT license, whereas its pyproject classifier says GPLv3. Preserve the actual license text and both facts rather than silently replacing the license with an inferred classifier. See the full original text below.

## Reference arrays, scientific comparisons and metric provenance

No `ground_truth.npz` is created. `evaluation/baseline_reference.npz` holds the **reproduced original method** output, not physical truth: all declared output quantities, the original PSFs and known optical basis/aperture required by independent evaluation. Array hashes and consumers are indexed below. It comes from `gpu_upstream.npz`, the independent original repeat, and the first two raw phase slots are canonicalized by subtracting the first exposure before packaging; reconstructed intensities are not renamed as truth.

Traditional MOMFBD FITS are archived under `research10/task10/traditional_broadband.fits` and `traditional_camXIX.fits`. Horizontal reversal is the comparison convention in the supplied upstream script. Independently measured translation of the original detector origin is `(183,187)` in the reversed comparison array; hence matching the declared mosaic uses comparison indices `[189:665,193:669]`. Each comparison crop is divided by its own mean for display because the public task normalizes passbands independently. This comparison registration/scaling is explicitly for historical visualization, never Judge alignment. It does not alter any public observation or reference-solver result.

Notebook comparisons retain both passbands, the best measured exposure, reproduced original result, fresh local result, archived traditional MOMFBD, paired RMS contrasts and both passbands' power spectra. The original paper's NMF and other-instrument figures represent different experiments; this task does not fabricate their results or claim full-paper reproduction. High-frequency mosaic power is interpreted with the paper's stated caution.

NCC means uncentered cosine similarity; NRMSE means RMS error divided by the reference dynamic range. Wavefront evaluation compares unit-flux pupil-generated PSFs through a separately named L2-relative error, acknowledging nonunique intensity-only phases. Observation error is L2-relative local forward residual on every raw-data-derived normalized/tapered frame, without hidden target calibration. Thresholds must be set from full original and independent local reruns, not zero-error self-comparison. Final thresholds were fixed before the first cleaned full run. Two failed cleaned runs were rejected rather than accommodated; the corrected method passes the same boundaries. Full measurements and threshold rationale are recorded below.

## Mandatory completion gates

| Gate | Status | Evidence |
|---|---|---|
| Exact paper/source/data provenance and mapping | PASSED | Frozen revision, real HDF5 bytes, preprint version, stage traceability and archive index |
| Original complete bound reconstruction | PASSED | Two independent unmodified full runs and instrumented repeat; all121 regions,2 channels,12 exposures,250 updates |
| Local complete bound reconstruction | PASSED | Full callable run, separate CLI and active notebook; no cached result substituted |
| Independent upstream parity and physics | PASSED | CPU stage and CUDA full-batch gradient fixtures; independent optical/adjoint/derivative tests |
| Constants, units, layout and acquisition contract | PASSED | Exact measured-array bytes; README scalar/constants checks; documented original quirks |
| Standard CLI and independent Judge | PASSED | CLI writes all3 required arrays; notebook invokes Judge; final output/judge_report.json pass |
| Judge poor and malformed inputs | PASSED | Zero predictions fail1; absent/invalid arrays/evaluator assets return2; different cwd and report-write checks |
| Public sufficiency and leak audit | PASSED | Only two public NPZ keys, lossless real observations and theoretical KL calibration; no hidden target or public GitHub URL |
| Clean runtime and device validation | PASSED for CUDA benchmark and CPU stage | Complete validated CUDA path; same-data CPU stage/physics tests; no claim of full250-step CPU numerical parity |
| Tests | PASSED | Full35-test suite including CUDA parity; final post-review36 CPU tests pass with1 deliberate CUDA deselection;37 distinct tests total |
| Active six-section notebook | PASSED | nbconvert exit0,29.356298299971968s,all code cells executed,no errors,5 reviewed figures |
| Historical/upstream/fresh labels and metrics | PASSED | Traditional processing labeled display comparison; independent original and active local results separated |
| Plans/source semantic and callable review | PASSED | Detailed actual signatures, devices, units, normalization, mode schedule, null slot, projection and caller flow reviewed |
| Final inventory and legal notices | PASSED | Exact template; no standalone LICENSE/NOTICE; original MIT text retained below; private archive outside task |

## Original source license and attribution

The source repository carries the MIT license shown below; the pyproject GPLv3 classifier discrepancy is disclosed above. The publicly downloadable observations/FITS archive does not identify a separate explicit data-redistribution license in the retrieved listing and upstream README. Do not silently claim that the software MIT grant licenses the measured data; this local benchmark records the public source and attribution without inventing additional terms. The complete root LICENSE text from the exact revision is preserved below. The upstream Zernike module additionally identifies Alvaro Menduina Fernandez, University of Oxford, January 2018, as its author. The cleaned implementation independently evaluates the same published Zernike formula using SciPy's general Jacobi-polynomial primitive; source attribution is preserved here. No standalone LICENSE or NOTICE is shipped, per the user's instruction.

```text
MIT License

Copyright (c) 2024 Andrés Asensio Ramos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Completed full runs and fixed acceptance thresholds

The original independent repeat, first successful local callable run, CLI and active notebook all use the exact full512-square burst, two channels, twelve frames,121 88-square patches and250 Adam updates. No task reduction, synthetic perturbation, privileged initialization or learned inverse is involved. Cleaned first full runtime was15.246488299919292s; CLI runtime15.258s. The final loss is8.760294914245605. All250 objective values and all canonical phase coefficients match the reproduced original exactly. Maximum object/mosaic difference is7.152557373046875e-7 from final projection/filter arithmetic. Independent original repeats are bitwise identical; fresh callable/CLI/notebook contract arrays are also bitwise identical.

The following bounds were fixed from the original run and extraction-quality policy before the first local full run, then retained unchanged through two failed investigations. Intensity NCC>=0.99999 and dynamic-range NRMSE<=0.002 apply separately to both channels and both intensity quantities. PSF L2-relative error<=0.01 checks every recovered local wavefront through unit-flux optics. These finite nonzero tolerances allow extraction rounding and remain much stricter than a materially different local optimum. They were not obtained by multiplying a zero self-error. Original independent repeat measurements show zero loss/array drift under the validated backend; the practical tolerances are conservative float32 extraction criteria, not an estimate of physical uncertainty. Observation residual bound is1.02 times the independently measured original residual plus1e-8, exactly0.018551577884958616. Mosaic relative error<=2e-6, first-exposure translation coefficient magnitude<=1e-6rad and null-slot magnitude<=1e-7rad check the public conventions. All nine criteria must pass. An identically zero local-object merge uses absolute mosaic RMS instead of dividing by zero; the zero submission remains a valid poor prediction and fails intensity/observation bounds.

Fresh independently evaluated measurements are: reconstruction minimum-channel NCC0.9999999999999917, maximum-channel NRMSE2.990655424569937e-8; local-object NCC0.9999999999999848, NRMSE4.4803136099958475e-8; PSF relative error1.6113642130485458e-7; observation relative residual0.01818781167876641; mosaic error, first-exposure tip/tilt and null slot all0. Full precision is retained in `output/judge_report.json` and private `full_parity_evidence.json`; rounded notebook labels do not replace machine results. `metrics.json` identifies the original baseline and formulas.

## Rejected numerical extractions and actual corrections

1. The first local full run computed the patch mean/taper on CPU. Original `util.apodize` follows the window's GPU device, then `combine_frames` copies the prepared values to a contiguous CPU buffer for FFT. CPU/GPU mean accumulation changed prepared values by up to2.384e-7 and spectra by1.526e-5, which propagated through nonlinear continuation. This failed with mosaic NRMSE0.004340146641524372 and PSF error0.15741067029071804. Preserve its arrays, log, loss and measured failure in private `failed_cpu_apodization/`.
2. Correcting taper device alone gave exact apodization/FFT across all121 patches and exact noise across all121x12 entries, but still failed: mosaic NRMSE0.0043308090367639735 and PSF error0.15245943566611325. Preserve this separately as `failed_gpu_apodization_only/`.
3. Independent original Adam tracing froze same-state gradients. Local objectives were equal but gradients differed about1e-8. Operator/objective cross-checks using the original closure's tensors were exact, eliminating a changed inverse model. The remaining difference was92 transition pixels of the diffraction mask: CPU `torch.cos` versus original GPU `torch.cos` differed by at most1.1920928955078125e-7. `build_optics(...,device=...)` now evaluates the transition on the selected device. The corrected full-batch gradients at original iterations0,1,25,26,27,40,60,79 are bitwise identical. No thresholds, observations, optimizer, mode schedule or physical model were changed.
4. Freeze actual original GPU mask, noise, parameters and gradients at two stages (iterations0 and60, active modes2 and9) in `test/fixtures/upstream_gpu_stages.npz`. Its test reconstructs preprocessing from full public observations and compares the locally computed full-batch gradients, not a fixture generated by the local method. The original trace was produced by external optimizer instrumentation without editing upstream source. CPU-only systems skip this explicitly CUDA-specific parity test; CPU stage and independent physics tests still execute.

Private investigation evidence includes `trace_original.py`, `trace_updates.npz`, `trace_original.log`, `operator_split_diagnostic.json`, `closure_split_diagnostic.json`, `gpu_mask_corrected_gradient_diagnostic.json`, and the two failure directories. Intentional diagnostic stops and early failed notebook dispatch are not reported as successful complete runs.

## Actual commands, environment and review evidence

All numerical runs set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and Torch threads1. The validated runtime is Python3.11.11, Windows build26200, NumPy2.4.6/SciPy1.17.1, Intel oneAPI MKL2025.3, Torch2.9.1+cu128/CUDA12.8, driver610.88, RTX4080SUPER16GB, Matplotlib3.11.1. Complete backend strings and environment details are in private `validated_environment.json`. GPU full reconstruction peak is approximately4GB; diagnostic/production GPU jobs were explicitly scheduled with other maintainers. No paging or OOM was observed. CPU one-region original/local stages and full-scene CPU preprocessing/optics/physics checks pass; full250-step CPU reference parity is not claimed because device-dependent float32 nonconvex trajectories differ. The released reference benchmark is validated on CUDA.

- Original CPU: oracle-env/Scripts/python.exe oracle_driver.py --stage cpu; exit0,2.8890984058380127s, bound first region/all12 frames/both channels.
- Original full: oracle-env/Scripts/python.exe oracle_driver.py --stage gpu --wait; repeat16.92630386352539s, exit0; complete logs and arrays archived.
- Standard local module: research-runtime/Scripts/python.exe -m src.main --device cuda; exit0,15.258s; `local_cli.log` records250 updates and output path. Omitting --device selects CUDA on this validated host via the same complete entry point.
- Judge: research-runtime/Scripts/python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; invoked in the active notebook; exit0, status pass.
- Tests: research-runtime/Scripts/python.exe -m pytest test -v --basetemp <private>/pytest-temp;35 passed in20.68s, `final_tests.log`. After the public immutable-data/metadata tests were added and a display-only DC annulus issue corrected, the two new contract tests and three review-metric tests pass in1.14s (`final_contract_tests.log`). The full scientific runtime is unchanged by that display correction. Final post-review CPU suite command adds `-k "not full_batch_cuda"` and uses private `pytest-final-cpu`:36 passed,1 CUDA test deliberately deselected,in14.89s (`final_cpu_suite.log`). The CUDA full-batch gradient test passed in the earlier full suite on the same scientific source. There are37 distinct passing tests across these recorded runs.
- Notebook: research-runtime/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/solar_multiobject_wavefront_mfbd.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=imaging-101-research --output solar_multiobject_wavefront_mfbd_executed.ipynb --output-dir <private>; exit0,29.356298299971968s. Timeout600s is conservative for the measured full15s inverse plus figures/Judge, not a requirement to simplify the task to a short demo. The executed notebook is copied into its authoritative notebook path; no temporary executed copy is left in task/output.

The first `python -m jupyter nbconvert` dispatch used system Anaconda's command on PATH, which could not find the isolated kernel. Calling the numerical runtime's `python -m nbconvert` resolves its own validated kernel and succeeds. Final notebook warnings are Windows Proactor/ZeroMQ selector-thread fallback and localhost kernel TCP transport notice; neither is a scientific/runtime error. Pytest's default user temporary root was inaccessible on this host; an explicitly private --basetemp avoids that environment permission issue, without changing tested behavior or adding task files.

All five notebook figures were extracted privately and visually inspected. The full observed first/best/mean comparisons show real solar granulation and narrowband fibrils. The2x4 scientific image comparison retains best observation, reproduced original, fresh local and traditional processing for both bands at matched scales. The recovered phases/PSFs show spatial variation; convergence overlays agree; paired contrast bars and both power spectra are retained. A roundoff-scale mean-removed DC annulus was excluded from the plotted power spectrum to avoid an uninformative10^-32 axis range; positive-frequency scientific annuli are retained. Signed difference maps show sub-micro intensity errors. Labels, units, color bars and legends are readable. Original/fresh RMS contrasts are about7.01% broadband and18.99% narrowband, but neither increased contrast nor high-frequency power is called ground-truth accuracy. Figure images remain private, with the executed inline images in the maintainer notebook.

## Asset consumers, hashes and archive index

Private `archive_index.json` records source/paper/data/fixtures/log byte lengths and SHA256; `validated_environment.json`, `import_audit.json`, `independent_repeat_evidence.json` and `lossless_conversion_evidence.json` provide environment, imported-symbol, repeat and independent HDF5-to-NPZ evidence. The early candidate-only `screening.md` and canceled observations.zip/history clone remain explicitly historical/private and are not completion evidence. Publisher-denial PDF-named files and wrong camXXV HTML download are excluded from the task.

| Shipped numeric asset | Provenance | Actual consumer |
|---|---|---|
| data/raw_data.npz observations | Lossless original HDF5 im,25,165,824 raw float32 bytes | Solver, notebook, observation Judge and data tests |
| data/raw_data.npz kl_mixing | Exact theoretical upstream table rows KL<=54 | Local basis construction and optics tests |
| test/fixtures/upstream_stages.npz | Unmodified original CPU run, first full12-frame/two-channel region and nonzero phase case | CPU preprocessing/basis/PSF/loss/gradient/Adam/object parity tests |
| test/fixtures/upstream_gpu_stages.npz | Independent original full-batch optimizer trace at0/60; original mask/noise | CUDA full-batch mode-continuation derivative parity test |
| evaluation/baseline_reference.npz | Reproduced original complete result, canonical phase convention, original PSFs/known optics | Standalone Judge and notebook original image comparison |
| notebooks/reference_outputs/upstream_measurements.npz | Original repeat objective history/runtime | Full-loss test and notebook convergence/runtime comparison |
| notebooks/reference_outputs/fresh_measurements.npz | Identified first successful complete local callable run | Full-loss test and provenance; active notebook computes its own results |
| notebooks/reference_outputs/traditional_comparison.npz | Archived camXX/camXIX FITS, fixed registered crop, mean scaling | Notebook comparison figures only, never Judge/solver |
| output/reconstruction.npz | Last active complete notebook inverse | Final independent Judge and pipeline contract test |
| output/judge_report.json | Independent Judge on that fresh output | Maintainer review; not a solver input |

The measured observations byte SHA256 is8d96c0ac5983f0dbe9273ec8d82ca7da38749d77f624354dbec3295faec73fb9; theoretical mixing bytes SHA256 is0e969bb0960c243f1b86ac5df5ac230bce595f1dea2bb74498a7fea450bf7c36. Isolated h5py equality check confirms the entire HDF5 im equals visible observations, with original float32 dtype and shape. Cleaned runtime never imports h5py, FITS, torchmfbd, skimage, einops or other imaging solvers. AST/call-role review finds only permitted general FFT, convolution/unfold, Jacobi polynomials, covariance SVD, median/label operations, automatic differentiation and Adam; locally implemented acquisition, loss and inverse updates remain explicit. Judge imports no solver modules. Notebook imports only cleaned modules and launches the independent Judge. README and requirements have no prohibited GitHub URLs. Public data has no fitted wavefront, sharp target, hidden seed or reconstructed calibration.

Semantic review checked every shipped callable signature/default against source and actual callers. Plans explicitly cover full-frame normalization, original noncontiguous noise layout, GPU taper/CPU FFT, selected-device diffraction cosine,100-cm unit convention, distinct pupil/basis grids, orthonormal versus ordinary FFT factors, loss-axis exclusion, global versus patch/frame noise use, constant LR/delayed activation, Scharmer support, zero basis slot, translation gauge,76-pixel overlap-add and476-pixel physical coordinates. No optional archived NMF/VAE/training path is described as shipped runtime. The scientific comparisons preserve both passbands and historical/original/fresh identities. No full checkout, PDF, HDF5/FITS, private capture script, download log, extra metadata JSON, duplicate report tree, LICENSE or NOTICE is retained in the task.

### Shipped numeric SHA256

- `data/raw_data.npz`: `07199a8c5594d9e5abc477c3c78b1ae57ff09dbd7aa1f6913e7fa002028bff11` (20954379 bytes).
- `test/fixtures/upstream_stages.npz`: `a064679e59688cf4ae2a4dc266bf3a606ba1326acc345ea77b7f37a8ca3b4247` (4754302 bytes).
- `test/fixtures/upstream_gpu_stages.npz`: `6a02b2eb90f2a8a8daf6711a5cc654cab4a843c6f1a63fe37abab862d05475c8` (130490 bytes).
- `evaluation/baseline_reference.npz`: `bf6c67bee5d8cdef04b3e9932b065861bddbe706f91af1de21b11ed48224a260` (49495428 bytes).
- `notebooks/reference_outputs/upstream_measurements.npz`: `dbfd9aeaa657b4ded15bf3d9cc06d5fbb3980e030e84b4e2f9cc4921b54afda7` (1320 bytes).
- `notebooks/reference_outputs/fresh_measurements.npz`: `9c8963484b84b1f527ec48825ee69e2d19b080a0e2da2d4f502d2091f9734972` (1320 bytes).
- `notebooks/reference_outputs/traditional_comparison.npz`: `dee433101d073d8f9140f901276f55a89d7b0fb97d099b014710dfaee15971fe` (1508971 bytes).
- `output/reconstruction.npz`: `5acb60159221ae0b6140744089008c7039d09df988160ddc7e99c3abb4f09959` (8040464 bytes).

### Frozen upstream source/data SHA256

- `research10/task10/upstream.zip`: `ef17be9e0e8a4743ac62b66a0272b22119ea3188d35338c6ac569591835a1d52` (68041274 bytes).
- `research10/task10/root-torchmfbd-arxiv.pdf`: `6c25bd2643247dc645285133de88c61098f0b546dcc396f6adaba611e1cf2cb3` (29177301 bytes).
- `research10/task10/qs_20190801_081547_8542_npix512_original.h5`: `b9294e228b7000be77cf4536befee318023be7e7f8fba02b51531fabc5941c1e` (25167872 bytes).
- `research10/task10/traditional_broadband.fits`: `06bc8578c541d716f3f43304fbd2ece45a34823c2d1f304a96b70a21ff8b8150` (3767040 bytes).
- `research10/task10/traditional_camXIX.fits`: `cad15bc71ed534da705df00e75891f0763bdb2ddb0a1572d0975ec26e173ec98` (3767040 bytes).
- `research10/task10/source/torchmfbd-b27a963cdb995cdd73695a17cddd589cb6367421/LICENSE`: `a8f3afed8bc6392f52bb2ffb945192f3833ed4f0881568e19d9c39bf5f26323f` (1078 bytes).


### Independent coordinator acceptance

The coordinator independently reviewed the shipped forward and inverse mathematics, callable interfaces and both plans, including orthonormal FFT factors, noise weighting, delayed mode activation, null/translation gauges and the 476-square mosaic convention. All five actual executed-notebook figures were visually inspected: observations, four-method paired-passband comparisons, pupil/PSFs, convergence/contrast/power and fresh-minus-upstream differences. The read-only structural review reported no errors or inventory warnings. A separate invocation of the standalone Judge from the private coordinator directory passed all nine metrics (image NRMSE 2.990655424569937e-8, PSF relative error 1.6113642130485458e-7, observation relative error 0.01818781167876641). Evidence is archived in `research10/coordinator/solar-external-cwd-judge.json` and `research10/coordinator/solar-final-review.json`. This accepts local CUDA benchmark cleaning; the documented CPU full-parity and observational redistribution limitations remain explicit. No scientific accuracy claim is inferred from reference agreement.
