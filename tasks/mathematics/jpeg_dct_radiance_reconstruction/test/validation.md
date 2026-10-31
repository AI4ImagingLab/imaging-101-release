# Validation and provenance

Status: locally validated and ready for independent parent review. This staged task has not been integrated into the release or a public registry.

## Bound source, paper and real observation

The selected executable method is the native `w=0` joint-TV/coefficient-fidelity branch of Victor VdE's GPL-3.0-or-later **jpeg2png v1.01**. Source: https://github.com/victorvde/jpeg2png ; exact revision `04054a2ee78b08cfbe69c71f84f9d84c0aedd9f3`. A fresh GitHub ref API response resolving `refs/tags/v1.01` to that commit is archived as `jpeg_remote_tag.json`. The checked-out source HEAD matches it. This is a remotely verified release revision, not a claim that a tracking branch proves remote freshness. The official Windows binary is https://github.com/victorvde/jpeg2png/releases/download/v1.01/jpeg2png.exe , SHA256 `f485ebc1e46657d966950e4e9413a6d46428b00bfd72d6010eb533c87686b64b`.

The mathematical literature anchor is Bredies and Holler, *A Total-Variation-based JPEG decompression model*, SIAM Journal on Imaging Sciences 5(1):366–393, 2012, DOI `10.1137/110833531`. The actually acquired/read author version is **SFB Report 2011-010, May 2011**, https://imsc.uni-graz.at/mobis/publications/SFB-Report-2011-010.pdf , SHA256 `b5890e1a1d435fc40dcd73520a347e2214416e8b615bf0bf8c4c10098073808b`. The author publication page independently links this report to the published article. Its text was extracted with the existing WSL `pdftotext`; Sections 4–5, especially the discrete DCT, feasibility intervals and projection, were checked against source. An earlier `hollerm/papers/tv_jpeg.pdf` URL returned an HTML personal page; it is not falsely counted as a downloaded paper.

The original program is a practical variant in that literature family. Its native extra coefficient-fidelity term, color coupling and accelerated normalized-subgradient optimizer are preserved and explicitly defined publicly. They are not falsely described as the paper's Algorithm 2 primal-dual solver. The 2013 TGV paper is context for the source's alternative branch; no claim of a literal full-paper TGV reproduction is made.

The exact observation is the already encoded **SpaceX DSCOVR/Falcon 9 launch photograph** distributed by scikit-image v0.19.3 at https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/rocket.jpg . SHA256 `c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c`. Official scikit-image `data.rocket` documentation identifies it as a SpaceX photograph released into the public domain. It is a real photographed scene with an existing JPEG encoding, not a new synthetic degradation. No re-encoding, downsampling, gray-to-RGB conversion, added noise or target-dependent crop was performed. Its 427×640 raster uses 4:4:4 coding and 640×432 coded component grids. Original quantized coefficient integers and positive quantization tables are read by libjpeg and retained as int16/uint16. There is no independently available uncompressed camera image, so **no actual truth is invented**.

## Private archive index and environments

All source checkouts, paper/PDF, original JPEG, binary, C capture drivers, logs, candidate investigations and temporary review images are outside the task, in `research_batch13_20261030/worker1_math_physics_astro/` (denoted W below). The task is staged separately under `_staging/mathematics/jpeg_dct_radiance_reconstruction/`. No archive/source/answer URLs appear in README, raw data or requirements.

| Private artifact | Role and consumer |
|---|---|
| W/jpeg2png/ | Exact original GPL source, including original license; private oracle build only |
| W/jpeg2png.exe | Official native executable; independent original CLI reconstruction |
| W/rocket.jpg | Unmodified original observation; source of public coefficient/calibration arrays |
| W/SFB-Report-2011-010.pdf and .txt | Actually downloaded primary paper version and extracted reading copy |
| W/jpeg_remote_tag.json | Fresh remote release-ref verification |
| W/capture_jpeg.c | Private driver invoking original `read_jpeg`, `decode_coefficients`, `unbox`, `compute` without modifying their source |
| W/capture_tv_gradient.c | Private original-C initial TV-gradient capture at the exact native input |
| W/capture_tv2_boundary.c and W/check_hessian_boundary.py | Original-C analytic-stencil capture and independent mathematical finite differences |
| W/rocket_tv_0.bin, _1.bin, _5.bin, _50.bin, _2000.bin | Frozen independent original numeric stages, from which public integers and hidden numeric fixtures are packaged |
| W/rocket_capture_50.bin | Actually computed original nonzero-TV2 diagnostic result; notebook comparison only |
| W/rocket_tv_2000_rerun.bin | Independently rerun original complete floating result; byte-identical to the first original capture |
| W/oracle_native_tv_2000.json and .log | Native executable command, UTC start, runtime, exit status and log |
| W/jpeg_c_scalar_capture2000.json and .log | Independent final scalar C oracle rerun command/time/status |
| W/jpeg_clean_full_pytest.json and .log | Final complete test command, raw result, timing and exit status |
| W/jpeg_final_judge.json and .log | Final standalone Judge command/time/status |
| W/jpeg_evidence_manifest.json and W/jpeg_task_evidence_manifest.json | Source/observation hashes and complete check/notebook/output manifests |
| W/jpeg_formal_criteria.json | Frozen substantive acceptance criteria; candid exploratory chronology |
| W/jpeg_executed_review_archive.ipynb and W/notebook_figures/figure1–4.png | Exact executed review notebook archive and actually inspected output images |

CPU: Intel Core i9-14900KF. Native oracle uses Windows official executable with `-t 1`; the scalar capture uses existing Ubuntu 24.04 WSL gcc, libjpeg and libm, `-std=c11 -O2 -ffp-contract=off`. There is no upstream solver import in cleaned runtime. Clean execution uses Windows CPython 3.11.11 and the existing research runtime: NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. CPU execution is verified; no GPU is required or used. SciPy DCT calls specify one worker; final test/notebook processes also set numerical thread counts to one. No package was installed or upgraded in the shared clean environment for this task. Pytest/Jupyter are development tools and are not added to task requirements.

## Original execution before extraction

The official native executable successfully reconstructed the exact original JPEG with `jpeg2png.exe rocket.jpg -o rocket_original_tv_2000.png -t 1 -q -i 2000 -w 0 -c rocket_original_tv_2000.csv`. The timed retained rerun took **20.3349815 s**, exit **0**, no reconstruction warning. Original float components were separately captured by the scalar C driver, including complete native padding, at 0, 1, 5, 50 and 2000 updates. The final independent 2000-update C rerun took **20.9025501 s**, exit **0**, and its binary SHA matches the first capture exactly. Seed: **not applicable**, the selected method is deterministic.

An earlier feasibility attempt with an existing author-supplied grayscale JPEG failed with original status 1, `only 3 component jpegs are supported`; it was rejected rather than converted/recompressed. The finally bound SpaceX observation has three native components and passes the actual original implementation.

The numerical capture drivers only perform input/output transfer around original functions; no cleaned function creates expected parity arrays. The normalized objective in the original CSV is retained under its original program meaning: `(TV + probability_distance + TV2)/total_alpha`. It is **not silently labeled as the public unnormalized J**. Notebook and Judge comparisons separately evaluate all stored/fresh floating images with the same float64 criterion and original measurement arrays.

## Traceability and independently checked mathematics

| Paper/model or source stage | Original file/function | Local implementation | Validation |
|---|---|---|---|
| Paper discrete DCT, Eq. 5.6; coefficient set Eqs. 5.5–5.7 | `jpeg.c/read_jpeg`, `decode_coefficients`; Ooura DCT | `physics_model.blocks`, `transform`, `inverse` | Original decode fixture; analytic constant block; transform/transpose identity and inverse round trip |
| Paper projection Eq. 5.14 | `compute.c/compute_projection`, `clamp_dct_c` | `solvers.reconstruct` coefficient clipping | All-coefficient forward consistency and poor-output rejection |
| Paper TV discrete gradient/boundaries; selected joint-color extension | `compute_step_tv_c`, `compute_step_tv_inner_c` | `physics_model.tv_gradient` | Original-C exact-input gradient; independent directional finite differences, boundaries included; constant-field check |
| Native coefficient-fidelity extension, including q² factor | `compute_step_prob_c` | Local inverse-DCT coefficient gradient in `reconstruct`; `objective` | Independent complete-objective directional derivative and known-constant metric case |
| Native acceleration, per-channel normalized steps, initialization, stop | `aux_init`, `compute`, `compute_do_step` | `solvers.reconstruct(coeff,q,iterations=2000)` | Frozen native-stage outputs and complete fresh parity/criterion checks |
| Evaluation | Independently rescored original numeric result | Standalone `evaluation/judge.py` | Known constant array; midpoint/zero negative cases; malformed inputs, evaluator assets, unrelated cwd and report-write failure |
| RGB preview | Standard centered YCbCr display convention | `visualization.rgb_preview` | Complete notebook execution and direct inspection of every generated comparison figure |

Observed original-C versus local gradient discrepancy **at the identical decoded spatial input**: maximum `3.5762787e-7`, RMS `3.962859e-8` intensity-gradient units. Original Ooura versus general SciPy float32 decoding differs by maximum `3.8146973e-5`, RMS `3.7628056e-6` intensity units. The initial decode-induced TV-gradient difference reaches `8.44717e-4` at `(channel=0,row=188,column=536)`, a low-gradient neighborhood with exactly constant chroma components. This controlled change isolates the transform-arithmetic perturbation before nonlinear updates. It is not evidence of a wrong TV adjoint. Full trajectory differences are measured, not assumed away: final RMS is `0.003586214698371803`, or `1.4063587052438442e-5` of the 255-unit nominal full scale; maximum local discrepancy is approximately `0.28994` intensity units. Their objective discrepancy is approximately one part per million.

Formal fixture tolerances are decode max `1e-4`, identical-input gradient max `2e-6`, full native RMS/255 `<5e-5`, relative full objective difference `<2e-5`, float64 operator adjoint `rtol=1e-10`, independent gradient `rtol=1e-5`. Initial source execution preceded the exploratory mathematical port. The formal criteria were fixed **after** that feasibility/diagnostic exploration and before the shipped task package; no claim of preregistration before all exploration is made. They have not been relaxed after the formal tests. This chronology is explicitly available for parent review rather than hidden.

### Confirmed source TV2 boundary defect

For the explicit nondegenerate image `[[0,2,1,5],[3,1,7,2],[4,8,3,6]]`, the original C TV2 stencil produces the stored gradient whose comparison with independent central finite differences (epsilon `1e-5`) gives maximum boundary error **2.0168058630449726**, while interior error is below `1e-9`. The original compiled-C values agree with the analytically transcribed original stencil at float32 scale. Its central `−2*(gxx+gxy+gyy)` boundary accumulation does not account for missing outgoing/initial faces. This is an upstream branch defect, not an error silently corrected and relabeled unchanged reproduction.

The selected first-order-TV branch is explicitly supported by original `-w 0`, is independently differentiated, and preserves its original coefficient-fidelity term. Actual original TV2 with 50 updates remains a visibly labeled notebook diagnostic. No missing method is filled with zeros or replaced by cached output under a fresh label.

## Metric provenance and fair objective target

There is no independently uncompressed camera image. `evaluation/baseline_reference.npz` contains the original centered-channel **method result**, not ground truth. It is used by notebook comparisons; the Judge does **not** require its pixel identity.

All rows/channels/coefficient constraints enter the evaluator. `score` evaluates DCT and criterion in float64, on complete centered arrays, with no cropping, gain fit, RGB conversion or alignment. The source program's mixed float32-DCT diagnostic J and the evaluator's float64 J differ slightly; their values are not conflated:

| Measurement class | Method/data | Public float64 J | Maximum coefficient violation |
|---|---|---:|---:|
| Observable ordinary decoding | Coefficient midpoint | 1338936.4247180317 | 0 |
| Independently reproduced upstream | Original TV, 50 updates | 1072195.2350630593 | 4.196166980818816e-5 |
| Fresh cleaned notebook | Local TV, 50 updates | 1072197.1035155554 | 7.629394519881316e-5 |
| Independently reproduced upstream | Original TV, 2000 updates | 1069561.9721799737 | 4.9591064453125e-5 |
| Fresh cleaned notebook/CLI | Local TV, 2000 updates | 1069560.9183407857 | 6.866455066756316e-5 |

The originally frozen mixed-arithmetic original diagnostic was J=`1069561.9791301962`. The fixed public objective upper boundary remains `1069668.9353281092`, a 0.01% numerical allowance from that captured value. The native float64-versus-mixed diagnostic difference is only `0.0069502225`, far below this allowance; no boundary was moved to accommodate cleaned results. The coefficient boundary is `0.001` raw coefficient units, chosen above measured original float32 transformation residuals, and independently constrained by all public intervals. The trivial midpoint and zero are rejected. The target criterion is fully public, so valid other optimizers are accepted without reproducing a finite solver trajectory. Reference agreement alone is not claimed to prove natural-image accuracy.

## Executed checks and notebook

- `python -m src.main` executed on native bundled coefficients and wrote the declared `(1,3,432,640)` float32 `output/reconstruction.npz`; no hidden inputs were read.
- `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` returned 0 with status `pass` and finite metrics above.
- Final `python -m pytest test/ -v --basetemp <private-W>/pytest-jpeg-final-temp` returned **0**, **18 passed in 107.95 s** (wall-clock wrapper 108.2312356 s). Complete raw command and log are in W. An initial default-temp run had 6 numerical passes and 10 setup errors because Windows denied access to a pre-existing `C:/Users/admin/AppData/Local/Temp/pytest-of-admin` directory. The private basetemp fixes that host ACL issue without weakening a test. All required negative and evaluator-error cases subsequently passed.
- Complete `jupyter nbconvert --to notebook --execute notebooks/jpeg_dct_radiance_reconstruction.ipynb --output jpeg_dct_radiance_reconstruction_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=600` executed with a private kernelspec pointing at the existing clean runtime, exit **0**, no cell errors. The portable final notebook uses `python3` metadata. Active local 2000-update reconstruction took **115.4223954 s**, and the active 50-update comparison took **1.7838234 s**. The 600-second cell timeout is justified by the measured full reconstruction rather than assuming all tasks finish in seconds.
- All **four** actual PNG outputs were extracted privately and individually opened with `view_image`: observation/quantization diagnostics; complete midpoint/fresh/original/TV2 comparison; objective bars plus numerical-difference heatmap/histogram; physical-region details. Labels, units, channel convention, absence of a true-image claim and distinct fresh/original conditions were checked. No blank or placeholder panel was accepted. The executed notebook is retained as the canonical notebook; its temporary duplicate was removed from `output/`.

## Consistency, import roles and inventory

README, plans and code were checked semantically against DCT signs/normalization, centered channel units, full padded boundaries, q² fidelity gradient, native `0.001` coefficient weight, acceleration/step normalization, default 2000 updates, loader signatures and actual caller flow. An implementation detail is disclosed: SciPy DCT replaces the original Ooura primitive, while the acquisition transform and selected inverse updates are preserved and measured within the documented tolerance. Diagnostic plotting crops/clips only RGB display arrays; scoring never does.

The shared clean imports are standard library, NumPy, `scipy.fft.dctn/idctn`, and Matplotlib. Core acquisition, TV, normalized subgradient, projection and public objective are local executable mathematics. Tests and notebook never import the original software, a domain reconstruction library or another task. The Judge imports only standard library, NumPy and a general DCT primitive. It cannot import or execute submitted code. No target-specific weights, private simulator seed, hidden image, target-derived calibration or source URL appears in public input arrays or requirements.

`test/fixtures/upstream_states.npz` consumers are the original-stage and complete-parity tests, plus exact original-integer/calibration equality tests. `notebooks/reference_outputs/upstream.npz` contains only original TV50 and original TV2 diagnostic arrays, consumed by notebook comparisons. The separate baseline stores original TV2000, consumed by the notebook. Raw observations and requirements alone, together with the complete public criterion and geometry in README, suffice to formulate and solve the stated inverse target. No generator is shipped because no observations were generated.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper version, remote source revision, data hashes, equation/function/test mapping | passed | Source/paper/data and traceability sections; archived primary PDF and remote ref |
| Original and cleaned full methods on identical native observations | passed | Native binary and independent C reruns, fresh CLI and notebook |
| Independent original numerical fixtures and mathematical checks | passed | Captured C arrays, 18-test suite, adjoints/FD/interval checks |
| Constants, units, precision, coordinates and numerical conventions | passed | Exact original integer/table tests and explicit padded centered-channel contract |
| Standard entry point and standalone fresh Judge | passed | Fresh output, final independent pass report |
| Poor/malformed/evaluator-error rejection | passed | Midpoint/zero and complete Judge contract tests |
| Public-only solvability | passed | Public convex criterion, all coefficient/step arrays, no hidden target dependency |
| No public truth, calibration, seed or source-URL leak | passed | Native observation-only NPZ and allowlist audit |
| Minimal clean environment and CPU support | passed | Measured native CPU pipeline, general primitives only |
| Local mathematics, independent Judge and no forbidden shortcuts | passed | Import/call-role review and source/fixture separation |
| Full tests | passed | Final 18/18, raw command/log retained |
| Active notebook and actual visual review | passed | nbconvert exit 0, fresh measured runtimes and four individually inspected PNGs |
| Honest historical/upstream/fresh metrics and threshold provenance | passed | Common float64 table, mixed-arithmetic diagnostic disclosed, no fabricated truth |
| README/approach/design consistency | passed | Final semantic/signature/unit/caller review |
| Final task inventory and licenses | passed | Standard layout; GPL/SpaceX notices; no papers/checkouts/screenshots/executed duplicates in task |

Development-order qualification: the acceptance tolerances followed an openly labeled exploratory diagnostic port, as explained above. Independent source execution and numeric fixture capture preceded the shipped extraction. Parent review must assess that disclosed chronology; this document does not rewrite it as a stricter preregistered development history.

## Packaged numerical asset hashes

```json
{
  "data/raw_data.npz": "3327e027fb8b7cabb92d2966ae77934d31688d3c441b8ceb0413982942532383",
  "test/fixtures/upstream_states.npz": "a8b2b0f0ac823307e234ecef5ecbcbc4d2ebedf5a991ce8461eec69758e87689",
  "evaluation/baseline_reference.npz": "1d33e0ac468fdce41cf1640626929a2c104b8c5228e3b525a1cabad0d03c3fcd",
  "notebooks/reference_outputs/upstream.npz": "60ce17fc9c5965a1f6993a3749ebe48a14a24615e39bd52758ccbbaeb3b3b0a7"
}
```

## Parent mathematical documentation review correction

Parent review found damaged TeX despite prior numeric success: EELS README lost macro backslashes in the JS string transport; JPEG notebook plain Python markdown string converted alpha to ASCII BEL. Both are actual documentation defects, now corrected using literal-safe source writing and raw markdown strings. README and plans plus parsed notebook markdown were scanned for control characters and mathematically meaningful symbols. Independent MathJax3.2.2 and KaTeX0.16.22 rendering both passed (JPEG23formulas; EELS72formulas). Rendered formula sheets were opened and visually inspected, including logarithms, Gaussian denominator, convolution and alpha objective. Private math_reviews contains exact TeX, per-formula render records, SVG and actual browser screenshots. Full active nbconvert was rerun after correction (jpeg_parent_math_fix_nbconvert.json and eels_parent_math_fix_nbconvert.json). No numerical method or threshold was changed. EELS energy arrays are validated against fixed nominal origins and spacing by load_observations; README metadata is now tested explicitly. The reference solve uses those same documented constants, not ignored arbitrary energy calibration. Source observations remain unchanged.


## Independent parent acceptance, batch13

Parent reviewed the orthonormal blockDCT, all three complete padded channels, normalized coefficient-fidelity term, jointTV and feasible-interval scoring. This is an explicitly identified original jpeg2png practical variant of the cited TV literature, not the paper authors' primal-dual algorithm. Original supported w=0 branch is selected; the independently demonstrated erroneous TV2 boundary branch remains a labeled actual diagnostic. Parent18tests, full2000updateCLI and foreign Judge exited0. A control-character alpha defect was found in Notebook math, corrected and followed by complete actual reexecution. Parent reviewed all canonical code cells and four actual figure groups, including real photo, all channel measurements, original/fresh/diagnostic panels and difference/detail plots. Arithmetic trajectory discrepancies and exploratory-before-formal-criteria chronology remain candidly documented; no threshold was relaxed. The unknown is encoded intensity rather than calibrated radiance or unavailable exact uncompressed truth.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/jpeg_dct_radiance_reconstruction/ records commands, reports, actual extracted canonical figures and final hashes.

## Authorized Judge configuration guard revision2

Parent authorized reopening only the malformed evaluator-configuration guard after acceptance review. Old Judge, unchanged metrics and old parent acceptance are preserved privately in judge_guard_before_20261030/. JSON number boundaries now exclude booleans and nonnumeric scalars/arrays. Both required metric names must exist and no unscored boundary may be added. Comparisons remain fixed <=; if an optional comparisons declaration exists it must contain exactly those two directions, so an empty or reversed comparison map is invalid. Empty configuration, missing/extra metrics, bool/string/array/emptyarray boundaries, NaN/Inf/negative boundaries and empty/reversed comparison maps are all tested. Every corruption case first proves copied valid evaluator assets pass the same existing fresh reconstruction; missing assets cannot mask a guard failure. Invalid configuration returns2 with empty measured metrics and a descriptive evaluator error.

The scientific metrics.json bytes, numerical score mathematics, valid thresholds, reconstruction and all upstream/parity/physics evidence are unchanged. The before/after valid Judge report dictionaries are exactly equal. Only related Judge contract tests and scoring of the existing fresh result were rerun, as explicitly authorized; no redundant2000-update reconstruction, notebook or upstream run was performed for this guard-only change. JPEG contract25tests passed2.24s; EELS contract28tests passed0.40s. Timed raw commands/exit0 logs are jpeg_config_guard_contract and eels_config_guard_contract; valid-report commands both exit0. Scientific metric version remains1 because definitions/thresholds did not change; this is separately identified configuration guard revision2.


## Independent parent acceptance, batch13

Parent reviewed the orthonormal blockDCT, all three complete padded channels, normalized coefficient-fidelity term, jointTV and feasible-interval scoring. This is an explicitly identified original jpeg2png practical variant of the cited TV literature, not the paper authors' primal-dual algorithm. Original supported w=0 branch is selected; the independently demonstrated erroneous TV2 boundary branch remains a labeled actual diagnostic. Parent18tests, full2000updateCLI and foreign Judge exited0. A control-character alpha defect was found in Notebook math, corrected and followed by complete actual reexecution. Parent reviewed all canonical code cells and four actual figure groups, including real photo, all channel measurements, original/fresh/diagnostic panels and difference/detail plots. Arithmetic trajectory discrepancies and exploratory-before-formal-criteria chronology remain candidly documented; no threshold was relaxed. The unknown is encoded intensity rather than calibrated radiance or unavailable exact uncompressed truth. Follow-up evaluator guard review rejected empty/missing/extra metric sets, invalid comparison directions and nonnumeric/nonfinite boundaries. Relevant new tests and existing fresh scoring pass; valid scientific boundaries and reconstruction mathematics are unchanged. Previous parent acceptance/hash evidence remains archived rather than silently rewritten.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/jpeg_dct_radiance_reconstruction/ records commands, reports, actual extracted canonical figures and final hashes.
