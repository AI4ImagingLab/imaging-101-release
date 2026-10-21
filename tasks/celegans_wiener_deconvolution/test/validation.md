# Provenance and validation: C. elegans Wiener deconvolution

## Fixed task and evidence status

Biology: real processed Airyscan C. elegans intestine restoration under the provider's assumed effective Gaussian PSF. Unknown is one real image, arbitrary intensity units, (1,310,310). Selected method is the framework's **2D SWiener demonstration**, not SPITFIR(e), Richardson-Lucy, or 3D Wiener. No new observations are generated and no biological truth is asserted. The reconstruction reference is explicitly a standard-method output.

## Source revision, paper and rights

- Repository: https://github.com/sylvainprigent/sdeconv , exact `9b774221eb94fea23ae7a5ea6ad84b0b3632fce9`, reports version 1.0.4. Remote `git ls-remote ... HEAD` succeeded on 2026-10-18 and agrees with downloaded codeload ZIP comment. This is an actually verified remote HEAD, not a stale local branch.
- Full ZIP obtained by coordinator from https://codeload.github.com/sylvainprigent/sdeconv/zip/refs/heads/main ; private `../coordinator/sdeconv-discovery.zip`. Its extracted source was copied unchanged to private `upstream/` in this task's archive.
- Data: `sdeconv/data/celegans.tif`. The provider's `sdeconv/data/__init__.py:celegans` describes **2D confocal (Airyscan) C. elegans intestine**, crops `[3:-3,3:-3]` and casts float32. This task preserves exactly that crop. No additional image resampling or synthetic degradation. The image is a processed export, not certified raw detector counts. Instrument, physical pixel pitch, emission wavelength, gain and optics are unavailable; they are not fabricated.
- Code and bundled data are covered by repository `LICENSE`: BSD 3-Clause, copyright 2021 STracking; no separate data license or contradictory third-party notice was found for this TIFF. The complete original license and explicit adaptation/data notice are retained verbatim in the consolidated section below. No claim that this is a measured PSF is made: the supplied Gaussian is the upstream test/quickstart's sigma=(1.5,1.5), shape=(13,13) model, obtained without hidden target information.
- Context paper: Prigent et al., Scientific Reports **13,1489**, published **2023-01-27**, final published version, DOI https://doi.org/10.1038/s41598-022-26178-y , PMCID PMC9883505. Actual full-text XML fetched from https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9883505/fullTextXML and archived `paper.xml`, SHA256 `d084522929c3c8da5c79f0f775662bb21540451a388707b33db2be33ca93f390`. Paper CC-BY-4.0. The Nature HTML returned a JavaScript challenge and was not treated as the full paper. The paper's Introduction distinguishes Wiener/Tikhonov microscopy deconvolution and its noise/bandwidth limitations; its main SPITFIR(e) algorithm is **not** represented as implemented here. This is an explicit selection of its associated framework's standard Wiener method; no SPITFIR(e) results or equations are claimed reproduced.

## Private archive index and reproducible original run

Archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task07_biology_deconvolution/` (maintainer only, outside task).

| Archive item | Role |
| --- | --- |
| upstream/ | Complete unmodified source and original TIFF/data; never imported in task runtime/tests/notebook. |
| oracle-env/ | Separate venv with system-site-packages, using preinstalled CPU Torch; no shared environment altered. |
| capture.py | Original execution and stage capture; loads unchanged selected source submodules to bypass unrelated neural-network package imports. No numerical source patches. |
| capture.json | Exact environment, backend, runtimes, hashes, independent precision run and pre-extraction tolerances. |
| original-run.log | Captured original run output, exit status 0, no warnings. |
| paper.xml, paper.txt, paper.html | Full final paper XML/text; separately identified anti-bot HTML retrieval. |
| reviewed.ipynb, nbconvert.log | Independent active notebook execution archive and conversion log. |
| figure0.png, figure1.png | Private visual inspection captures, not duplicated in released task. |
| pytest.log | Final full validation run. |
| make_notebook.py, finalize.py | Private preparation/provenance utilities, not runtime code. |

Original command (from repository root):

```powershell
& ../imaging-101-maintainer/batch10/task07_biology_deconvolution/oracle-env/Scripts/python.exe ../imaging-101-maintainer/batch10/task07_biology_deconvolution/capture.py
```

Oracle Windows 11 build26200, Python3.14.3, NumPy2.4.3, torch2.11.0+cpu, MKL2025.3, AVX2, single Torch thread, CPU only. Venv explicitly isolates module search and configuration; inherited existing general-purpose packages are documented. SDeconv is loaded solely from archived unchanged checkout. Torch device is explicitly CPU (settings only); no CUDA required. Original `SWiener(psf,beta=0.005,pad=13)(image)` ran on exact packaged observation in **0.03164280008058995 s**, exit0. No seed: direct deterministic inverse. Second original run max absolute difference **0**. A separate image/PSF-double call, preserving original internal float32 kernel convention, gives NRMSE **4.9049516436171566e-08** relative to original float32. Full backend text in capture.json. No cached upstream TIFF is used as the numerical oracle.

Source SHA256:
- `upstream\sdeconv\data\celegans.tif`: `05133eba629a46c932f0a96747967c032052f465ab0ba072df780e75ce80834b`
- `upstream\sdeconv\deconv\wiener.py`: `7d05fc10198ffc36ffb25bbd4b44e553ab1ff7916eebfca7a4266e40901a5f88`
- `upstream\sdeconv\deconv\_utils.py`: `35db1ffbc613fdbcaadbbc0f89f2666ee5441f63e1adf15570e88689c7963752`
- `upstream\sdeconv\psfs\gaussian.py`: `30c0e247b79e64ffcb90c4068301e74b6f50876ee9c7375e0ac7a5a7e824b5df`

## Bound arrays and identified consumers

| File | SHA256 |
| --- | --- |
| `data/raw_data.npz` | `579a37f353c75c8a16208e5c98ea9e200f59cd69bb98c26fde69ff6240c8d2cb` |
| `test/fixtures/upstream_stages.npz` | `719ac9e980595e575d3938c9b56fe9229ab0b41e1ab8765a571e6a4830f50b11` |
| `notebooks/reference_outputs/upstream.npz` | `18e15aaf9d56a280da80b7839465844ac9e47e756710a042386fe65d206a1c93` |
| `evaluation/baseline_reference.npz` | `18e15aaf9d56a280da80b7839465844ac9e47e756710a042386fe65d206a1c93` |

`data/raw_data.npz`: only observation float32 (1,310,310), PSF float32 (1,13,13), physically legitimate assumed effective kernel. No latent target, target-derived PSF, hidden seed or privileged observation. PSF float32 sum differs from1 by2.09e-8; both methods explicitly normalize before embedding.

`test/fixtures/upstream_stages.npz`: frozen **upstream** reflected observation, embedded normalized kernel, complex transfer function, centered Laplacian, float32 output, independent mixed-double output; consumed by stage/full parity tests. `notebooks/reference_outputs/upstream.npz`: reproduced-upstream reconstruction, consumed by notebook comparisons. `evaluation/baseline_reference.npz`: same independently generated result under the reserved evaluator filename, consumed by Judge/contract tests. These distinct copies have defined consumers. No ground_truth.npz exists because no true latent object is available.

## Traceability

| Mathematical or algorithm stage | Upstream file/function | Local function | Independent case |
| --- | --- | --- | --- |
| Gaussian model and original TIFF crop | data.celegans; psfs.gaussian.SPSFGaussian; tests/deconv/test_wiener.py | public immutable arrays; load_data | raw hashes; dimensions/metadata checks |
| Reflect extension; PSF normalization and embedding | deconv/_utils.py:pad_2d,resize_psf_2d | preprocessing.prepare | frozen padded/kernel arrays |
| Fourier convolution and origin | deconv/wiener.py:SWiener._wiener_2d | transfer_function,apply_blur | frozen OTF; asymmetric-kernel adjoint, constant/flux checks |
| Five-point quadratic prior | deconv/wiener.py:laplacian_2d | laplacian_kernel,laplacian | frozen centered stencil; independent spatial normal equation |
| Regularized inverse and central crop | SWiener._wiener_2d; quickstart 2D example | solvers.reconstruct | frozen original and mixed-double final arrays; full-grid stationarity |
| Linear microscopy restoration family | paper Introduction's Wiener/Tikhonov discussion | README/approach explicit derivation | direct normal equation residual independent of copied FFT expression |
| Metrics and no alignment | local benchmark contract | standalone judge.score; visualization.agreement | independent [1,2,3] vs [3,2,1] known case; errors/failures |
| Fresh active comparison | upstream quickstart input/Wiener plots | notebook local reconstruction and visualization.compare | active nbconvert, figure inspection, fresh timing |

The selected 2D Wiener formula has no ambiguity after inspecting origin and centered Laplacian modulus. An independent normal-equation derivation validates factors/signs. The library's 3D Laplacian has an unrelated repeated neighbor assignment; its RL implementation mirrors a rolled kernel by array flip rather than Fourier conjugation (possible pixel-shift issue). Neither path is selected or silently corrected. No claim of full-framework reproduction is made. The limited effective PSF fidelity and unknown processing of Airyscan data remain explicit limitations.

## Pre-extraction tolerances and metric provenance

Frozen before local implementation: padded/laplacian exact; embedded kernel absolute1e-8; OTF absolute3e-7; reconstruction relative2e-5 and absolute0.02a.u. These account for Torch float32 FFT arithmetic. Judge boundaries NCC>=0.999999 and dynamic-range NRMSE<=2e-6 were fixed before extraction using repeated upstream and measured 4.90e-8 precision sensitivity plus conservative cross-backend margin. They are numerical tolerances for method reproduction, not 0.9/1.1 self-comparison thresholds. NCC is uncentered cosine; NRMSE RMS/reference range. No mask/alignment/rescaling/cropping. Zero norm prediction gets NCC0 and fails; constant/zero reference is an evaluator error. No NaN/Infinity JSON values allowed.

Fresh shared CPU run: Python3.11.11, NumPy2.4.6, Matplotlib3.11.1, no Torch/skimage/SDeconv dependency. Standard runtime uses explicit NumPy FFT; only Matplotlib for review. Fresh notebook runtime **0.012976 s**; NCC **0.9999999999999827**, NRMSE **4.899143052145996e-8**. Judge pass. Input-as-output NRMSE **0.017267840724207076** fails by orders of magnitude. Full-grid normal-equation residual relative **2.729766023132876e-16**; flux relative error **0**; relative blur observation residual **0.035350562061759946** (not evaluated as zero). No paper score or old TIFF metric is passed off as newly measured.

## Execution and visual review

Standard CLI and foreign-working-directory Judge passed. Full pytest command with private TEMP/TMP and PYTHONDONTWRITEBYTECODE=1: `shared-env/Scripts/python.exe -m pytest test/ -v -p no:cacheprovider`, **20 passed**, exit0. Initial system TEMP pytest permissions caused setup errors; redirected to private task archive temp, then reran successfully. This is an environment issue, not suppressed test coverage. The final log records the successful run.

`shared-env/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/celegans_wiener_deconvolution.ipynb --output reviewed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=120` passed, exit0. All code cells executed, no errors, two computed image outputs. Timeout120s is comfortably above measured subsecond solve and approximately4s notebook startup; no expensive training path omitted. Original quickstart's observation/Wiener scientific comparison retained, with independent upstream/fresh/difference, profile and quantified metric bar. Visual inspection of both extracted PNGs: same orientation/scale, visible diagonal structures retained, numerical tiny difference colorbar correctly labeled, profile legends and metrics readable. Sharpened ringing is present and not mistaken for ground truth. Only the finalized executed review notebook is shipped; temporary copies/captures are private. Kernel TCP warning is environmental, not a numerical or notebook error.

## Semantic/source and final gates

Manually checked actual signatures/defaults against design, formulas/normalization/signs/units against code, and entrypoint/notebook caller paths. Closed form is executable local math; no domain solver, upstream import, checkpoint or cross-task import in cleaned source, generator (absent), Judge, tests or notebook. Libraries provide general FFT/array/display primitives only. Public materials checked for GitHub URLs and hidden-asset leaks; none. No standalone metadata JSON, root wrappers or duplicate test/report directories. complete license/attribution text preserved below. Only documented fixture/reference consumers shipped.

| Mandatory gate | Status | Evidence |
| --- | --- | --- |
| Published context, selected method, verified revision, hashes and mapping | passed | source/paper/traceability above |
| Original and local exact-observation runs with environments/logs | passed | capture, standard CLI and notebook |
| Independent stage parity and mathematical/physical checks | passed | frozen stages, adjoint, DC/flux, normal equation |
| Constants, coordinates, precision and units | passed | upstream defaults preserved; README/plans and tests |
| Standard CLI output and fresh Judge | passed | output/reconstruction.npz; judge_report.json |
| Judge poor/malformed/error cases | passed | zero/negative; missing/key/shape/dtype/object/NaN/Inf/corrupt; asset/write failures |
| Public input sufficiency and leakage | passed | self-contained README, raw two keys, pinned runtime subset |
| Minimal independent cleaned runtime and CPU | passed | shared no-SDeconv/no-Torch runtime; NumPy explicit math |
| Unit/parity/physics/pipeline suite | passed | 20 tests |
| Active notebook and preserved comparisons | passed | executed notebook, two inspected figures |
| Measurement/reference labels, thresholds and formulas | passed | repeated oracle/precision/fresh results separated |
| Plans match source semantically and by signatures | passed | manual review and normal-equation derivation |
| Final inventory/archive/licensing | passed | template + consolidated license text below; private captures/source outside task |

No failed or unverified mandatory gate is represented as completed. Physical accuracy of an assumed PSF is intentionally not asserted; this task's contract is validated standard-model restoration of real processed observations.


## Independent coordinator review (2026-10-18)

A separate coordinator reran the standard CLI, the standalone Judge from a different working directory, the complete pytest suite, and the active notebook through nbconvert in the shared CPU environment. All four commands returned exit code 0. Command arguments, timing, stdout/stderr, the executed review notebook, and extracted figures are archived at `batch10/coordinator/celegans_wiener_deconvolution/`; the structured check report is `batch10/coordinator/celegans_wiener_deconvolution-review.json`. The coordinator inspected all embedded comparison figures and reviewed the implemented source and mathematical plan. Packaged reference snapshots were preserved around the notebook rerun; any rerun snapshots are separately archived.

The first coordinator run exposed a Windows test portability failure: a child UTF-8 diagnostic was decoded with the parent default GBK encoding. Contract-test subprocesses now explicitly request UTF-8 output/decoding, and text-file reads explicitly specify UTF-8. The final complete rerun passed 20 tests. A separate default-locale rerun with an explicit writable `--basetemp` also passed 20 tests. The machine default pytest temporary directory was inaccessible; coordinator verification uses a writable private TEMP/TMP directory. This is an environment/test fix; solver mathematics did not change.


## Consolidated original license and attribution texts

Task-level license/notice files were consolidated here at user request. The complete original text is retained verbatim below, with its original-byte SHA256 for verification; archived originals are outside the task in revision_docs/task08. These terms continue to apply.

### Original LICENSE

SHA256 `ce7adaf460c662c1f227089603bc52ba76d82d6bd2a7a538d5bfd853fb605066`

```text
BSD 3-Clause License

Copyright (c) 2021, STracking
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Original NOTICE

SHA256 `0a5fb80db5ae7164580892869e76ce38adec3445678220d59eb986c79797d9e6`

```text
This task adapts the 2D Wiener implementation, reflection padding and PSF
embedding conventions from SDeconv 1.0.4, copyright (c) 2021 STracking.
The accompanying LICENSE preserves its BSD 3-Clause terms.

The C. elegans intensity image is redistributed from that repository's
sdeconv/data/celegans.tif under the repository license; no separate data
license or contrary third-party notice accompanies that file. The upstream
loader describes it as a confocal (Airyscan) C. elegans intestine image.
The supplied Gaussian is the example's assumed effective kernel, not an
experimental PSF measurement. The loader crop and float conversion are
preserved; no new acquisition or simulated observation is introduced.

Local adaptations replace the selected Torch FFT arithmetic with explicit
NumPy float64 arithmetic, add independent validation and the benchmark I/O
contract. Neither optical calibration nor independent biological ground
truth is claimed. Source revision and exact source URLs are in private
maintainer provenance.

```


## Document and six-part notebook revision (2026-10-18)

At the user's explicit request, this revision consolidates task-level license/notice files into the full-text legal section in this validation record and removes the separate files. The terms, copyright attribution and data-source notices continue to apply. The full old documents, original notebook, license/notice bytes and all historical reference snapshots were archived before editing at `batch10/revision_docs/task08/celegans_wiener_deconvolution/original/`, outside this task. No skills were used. EHT original's complete approach/design and active comparison notebook were read as depth references; its science and data are not imported into this task.

Both plans now explain the concrete forward/inverse mathematics, dimensions, objective normalization/signs, parameter choices, actual numerical execution and scientific limitations. The design documents every shipped source/Judge callable's actual signature, shapes/dtypes/units, defaults, explicit versus assumed validation, errors, mutation/I/O behavior, module dependencies, distinct CLI/notebook flows and test mapping. No unimplemented optimizer, iterative update or convergence feature was invented for these direct methods.

The primary notebook now has exactly these six ordered H2 sections: `1. Problem`, `2. Data`, `3. Physics`, `4. Method`, `5. Results`, `6. Conclusion`. Data includes every visible array key/shape/dtype/unit, actual source and acquisition table followed by active loading. Physical acquisition, coordinate/unit/boundary conventions and mathematical reconstruction are explained separately. All earlier scientific plots, method groups, independent checks and metric tables are retained and freshly executed through local src functions. There is only one notebook inside the task.

Latest full suite: **20 passed in 2.36 s**, exit 0. Full active `python -m nbconvert --to notebook --execute notebooks/celegans_wiener_deconvolution.ipynb --output executed.ipynb --output-dir <private revision archive> --ExecutePreprocessor.timeout=180` returned exit 0, with 4 executed code cells and **2 image outputs**. Every image was extracted and visually inspected in this revision. Plot labels, comparisons and quantitative scales remain readable; no absent method/result was filled in. The 180 s timeout covers startup and plotting while the measured reconstruction remains short. Private `pytest.log`, `notebook.log`, `verification.json`, `executed.ipynb` and `figure*.png` record commands, timings, outputs and visual evidence; the executed notebook was promoted to the single primary notebook. Notebook warnings are the previously identified local Windows/Jupyter transport advisories; there are no cell errors.

Fresh standalone Judge report: `{"ncc": 0.9999999999999815, "nrmse": 4.899143052145996e-08}`, status `pass`. The fresh output and Judge report are separately archived as `fresh_reconstruction.npz` and `fresh_judge_report.json`; full new timing/diagnostic streams are in verification.json. Historical reference_outputs bytes were snapshotted and verified identical before/after execution, including any old runtime/source-hash measurement records. Those source hashes continue to describe their original measurements, not the current documentation revision. No observation, hidden reference, parity fixture or threshold was changed. Protected-asset before/after SHA256 maps are recorded in verification.json. Runtime algorithms and executable statements are unchanged.

License migration verification: LICENSE: `ce7adaf460c662c1f227089603bc52ba76d82d6bd2a7a538d5bfd853fb605066`; complete original bytes verified within the consolidated section; NOTICE: `0a5fb80db5ae7164580892869e76ce38adec3445678220d59eb986c79797d9e6`; complete original bytes verified within the consolidated section. Historical mentions of an accompanying LICENSE inside the preserved original notice are quoted original text; they now resolve to the consolidated license immediately preceding it, rather than a standalone file.

### Current revised file hashes

| File | SHA256 |
|---|---|
| `README.md` | `9960cf4c8d85263720e3116ae13db8c0f88838f5d136cceb477b12da206a2da2` |
| `plan/approach.md` | `795e38b581205e0f6513d2f33d8b4defe9b6613ca4197b4a9632ef23171bbe90` |
| `plan/design.md` | `bf24b6ca3e506012b56a529314b1129340a15b87b3a4f02fef5dfe4e31b2753a` |
| `notebooks/celegans_wiener_deconvolution.ipynb` | `5d7f375f8ddf8847fa32488ed0623eff11a31f619529eeed66e9848234432fc0` |
| `src/__init__.py` | `c7187ea483877a568ae67f8ca233f3b70519590175cbb6dd3cf7835cf29005eb` |
| `src/main.py` | `1dcf8c3aed41974be190a812117772ffe17d1dc79320af5139f32f9bef9c558b` |
| `src/physics_model.py` | `90b0ae8ab7a70fa36dfee5d2d0fff05143d13c0d502838829e2db5a3d682d5bd` |
| `src/preprocessing.py` | `de805be89ed2581d38f2a7ef89172c170e66b2a0a2b2266940fb0171bb019e5e` |
| `src/solvers.py` | `00dfb33f6cae609309f872ab894c2deeaaffde4fe3e992e5c790ba51dc2bb2bd` |
| `src/visualization.py` | `15bc1c9a0bfeef5081e65a5a8deb4bb72b9f814b9086368084b3680ac201f4b9` |

The source/readme/model distinction remains explicit: real processed Airyscan observations are restored under an assumed Gaussian effective PSF, not measured calibration. The original/fresh/difference image group, central-row profiles and input/fresh NRMSE comparison are retained and inspected; full-grid stationarity/flux/forward diagnostics are freshly evaluated.


### Final Markdown-only editorial polish

The final notebook wording merges repeated Data introductions into one source/selection paragraph followed by the complete tables. Problem, Physics, Method, Results and Conclusion were reviewed together to remove repeated provenance, acquisition or no-iteration statements while retaining task-specific mathematical definitions, processing caveats, comparisons and limitations. Only Markdown cells changed: every code cell, execution count and embedded result/output is byte-equivalent under deterministic JSON serialization to the already executed notebook (SHA256 `708d53b2664f3f0d9779bf7bc1b52687a3f063e6d7e005cf88b4ab87aa27df4f`). The successful numerical execution and 2 inspected image outputs therefore remain applicable; no identical numerical rerun was needed. `revision_docs/task08/celegans_wiener_deconvolution/markdown_final.ipynb` archives this final wording, and the notebook SHA256 in the current revised file table above is updated.
