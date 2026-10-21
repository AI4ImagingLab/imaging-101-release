# Provenance and validation evidence

All mandatory gates below passed on 2026-10-18. This is a new medicine benchmark, not a reformatting of the existing diffusion-tensor task. It reconstructs the full order-eight solid-angle orientation distribution (45 coefficients), not a tensor or scalar FA image. The bound observations are real acquired human data, with depositor motion correction; no new synthetic data generator is applicable.

## Sources, paper versions and licensing

- Original repository: https://github.com/dipy/dipy . Exact release tag 1.12.0 remotely verified using `git ls-remote ... refs/tags/1.12.0`: `af546c7338144af9b2f3e2e04b19f183ad8fc667`. The exact-commit codeload source archive is private `upstream.zip`. The installed 1.12.0 wheel's shm.py was independently text-compared against the remotely downloaded tag file: zero differing lines. Byte hashes differ only through newline encoding (installed 3222f85b133522e82816a1c42f83fe1c29dc1233bbd8c2d7f26537e88dfc1faf; downloaded 8c25ae793f1e8272864039559f87dfacc6ca2229f2dfd93b0c37b211c40027bb). Code is BSD-3-Clause; original license and attribution are retained verbatim in the consolidated section below.
- Selected publication version: Aganj, Lenglet, Sapiro, ISBI 2009 pp1398-1401, DOI10.1109/ISBI.2009.5193327, author manuscript at https://pmc.ncbi.nlm.nih.gov/articles/PMC4360965/ . The fuller MRM2010 author manuscript, DOI10.1002/mrm.22365, is archived from https://pmc.ncbi.nlm.nih.gov/articles/PMC2911516/ . The original model explicitly cites Aganj2009. Both paper versions concern the same task-relevant single-shell CSA formula; no multiple-shell functionality is claimed.
- Actual data: Rokem, Ariel; Yeatman, Jason; Pestilli, Franco; Wandell, Brian (2013), *High angular resolution diffusion MRI*, Stanford Digital Repository https://purl.stanford.edu/yx282xq2090 . Exact files downloaded from `https://stacks.stanford.edu/file/druid:yx282xq2090/` as dwi.nii.gz, dwi.bvals, dwi.bvecs. The repository's fetcher independently identifies these same files and MD5s. Stanford public XML explicitly identifies PDDL1.0 and the real dual-spin-echo 150-direction b2000 acquisition plus ten b0 images. XML is privately archived; license attribution is in the consolidated section below. No identity inference or attempt was made.
- Original data shape (81,106,76,160), spacing 2mm isotropic, affine diag(2,2,2,1) with translation(-80,-120,-60)mm. Bind the predetermined zero-based crop [40:64,40:64,35:36,:]; no output-informed selection, mask, rescaling, interpolation, or volume subsampling. Public arrays hold the full observed 160 magnitudes and original bvalue/direction text values as float64 with batch dimension. NIfTI reading via nibabel occurs privately only. Gradient norms differ up to 7.72112981e-6 because deposited text is rounded; angles use vector normalization as upstream does. These are deposited preprocessed magnitudes rather than scanner raw k-space.

## Independent original run before extraction

The isolated `oracle-env` contains DIPY1.12.0 and its upstream-only nibabel/domain dependencies. `oracle-freeze.txt` records all versions. Python3.11.11, NumPy2.4.6, SciPy1.17.1, Windows10.0.26200-SP0, CPU, NumPy wheel OpenBLAS numerical backend (full np.show_config in upstream.log). No GPU, training, random initialization or seed is used by this deterministic closed-form method. No upstream installation or imports are required by the final task.

Command from the maintainer archive: `oracle-env/Scripts/python.exe capture.py > upstream.log 2>&1`. Successful process exit0. The unchanged original CsaOdfModel was executed on exactly the magnitudes later bundled, order8/smooth0.006/default normalization; runtime 0.010386400041170418 seconds. Repeat maximum absolute difference0; passing float32 rather than float64 source signals also gives0 because upstream explicitly casts normalization to float32. Frozen basis, fit matrix, normalized observations and coefficients were saved before local extraction. Source arrays and command/runtime/environment hashes are in upstream_run.json. The only np.show_config warning recommends optional pyyaml for formatted configuration output; it does not affect reconstruction. Pending-deprecation notices for the intentionally selected legacy SH basis are expected.

Before local implementation the capture manifest fixed stage atol2e-7 and coefficient relative-L2 tolerance1e-6, approximately eight float32 epsilons for the normalization stage. Independent oracle stability command `oracle-env/Scripts/python.exe stability.py` raised every normalized signal value by one float32 ULP and reconstructed with assume_normed=True: relative coefficient L2 sensitivity1.3130226009468278e-7. The already fixed boundary is7.62 times this sensitivity. Independent repeat identity is not misrepresented as an estimate of nonzero physical error.

## Traceability

| Paper equation or algorithm stage | Original file/function | Local callable | Independent case |
|---|---|---|---|
| Fourier signal and solid-angle radial definition; single-shell radial approximation | shm.py CsaOdfModel and cited Aganj2009/2010 | README acquisition model; solvers.reconstruct | Analytic isotropic attenuation and unit solid-angle integral |
| Baseline averaging and double-log attenuation | shm.py normalize_data; CsaOdfModel._get_shm_coef | preprocessing.normalize; solvers.reconstruct | Frozen original normalized array; scale invariance |
| Even real SH convention and angular discretization | shm.py QballBaseModel; real_sh_descoteaux_from_index; core/geometry.py cart2sphere | physics_model.indices/basis | Frozen original B; numerical SH orthogonality and antipodal symmetry |
| Laplace-Beltrami penalty and augmented pseudoinverse | shm.py smooth_pinv | physics_model.fit_operator | Original fit matrix; adjoint identity |
| Funk-Radon P_l(0) and Laplacian factors F L/(8pi) | shm.py CsaOdfModel._set_fit_matrix | physics_model.fit_operator | Frozen original full inverse and full coefficients |
| Unit-integral constant mode | shm.py CsaOdfModel._get_shm_coef | solvers.reconstruct | Independent uniform density 1/(4pi) and sphere integral1 |
| Spatial pipeline and SH sampling | shm.py model.fit, SphHarmFit.odf | main.main; physics_model.sample_odf | Full exact-observation parity; active notebook |

Ambiguities are explicitly preserved: the original default legacy Descoteaux basis differs for negative m from the paper's nonlegacy signed-m definition; output contract specifies the actual original default including abs(m). The method output is the orientation density of water displacement, not an axonal fiber density. The radial approximation, float32 normalization, signal clipping and potential negative SH lobes are preserved. No suspected formula error or correction was silently introduced. q magnitudes and diffusion timing are not deposited, so no absolute radial displacement reconstruction is claimed.

## Reference identity, metrics and fresh results

There is no independently measured orientation truth. `evaluation/baseline_reference.npz` is the independently reproduced original CSA result; `evaluation/ground_truth.npz` is deliberately absent. `notebooks/reference_outputs/upstream.npz` supports notebook original-versus-fresh comparisons. `test/fixtures/upstream_stages.npz` is consumed by test/comparing_test/test_parity.py for stage and complete-pipeline parity. `notebooks/reference_outputs/cleaned.npz` is independently generated during the active notebook from cleaned modules and used to preserve that measured review output and consumed by test_archived_notebook_measurement_consistency; this consistency case is not claimed as independent upstream parity. No publication or cached notebook number is represented as this dataset's measurement.

Fresh local measurements: basis maximum absolute discrepancy3.219646771412954e-15; inverse discrepancy5.898059818321144e-17; normalized-signal discrepancy0; final coefficient maximum absolute error2.4980018054066022e-15; all-coefficient relative L2 error2.8850204102479362e-15. Fresh notebook reconstruction runtime0.003615100053139031s on CPU. Same exact data, SH parameters, environment versions and reference define each comparison. Coefficient evaluation includes all25,920 entries, no mask, crop, alignment, rescaling or magnitude conversion. NCC and dynamic-range NRMSE are not used because signed orthonormal angular coefficients are better evaluated by relative Euclidean norm. `metrics.json` records the explicit metric formula and independently justified nonzero boundary1e-6. A zero reconstruction gives relative error1 and fails.

## Commands and validation gates

Use shared Python `../imaging-101-maintainer/shared-env/Scripts/python.exe`; set PATH to its Scripts directory before launching notebook kernels. Set TEMP/TMP to a writable private temp directory on this Windows installation. No package install or upgrade was needed in shared runtime. Runtime requirements contain only the actual NumPy/SciPy/Matplotlib pinned subset; pytest/Jupyter remain development tooling.

| Gate | Status | Command or evidence |
|---|---|---|
| Exact remote code revision, paper version, data licensing/hashes, equation mapping | passed | Source discussion and archive index below |
| Independent original run and frozen stage/full outputs on exact observations | passed | capture.py, upstream.log, upstream_run.json; exit0 |
| Stage parity and independent physical/mathematical validation | passed | test/comparing_test/test_parity.py and test/unit_test/test_physics.py |
| Constants, units, precision, coordinates and conventions | passed | Gradient/bvalue metadata tests, exact float32 stage parity, independent SH sphere checks |
| Standard CLI writes declared NPZ | passed | `python -m src.main`, exit0 |
| Standalone Judge on fresh result | passed | `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`, exit0; relative_l2=2.8850204102479362e-15 |
| Judge poor/missing/key/shape/dtype/complex/NaN/Inf/evaluator/report failures and foreign cwd | passed | Parametrized test_judge.py cases, metric known case, exit0/1/2 verified |
| Public allowlist self-contained; no hidden/GitHub/truth leaks | passed | README specifies all data keys, physical assumptions, metadata, SH convention and output; public regex/manual audit |
| Minimal runtime without domain library installed | passed | Shared environment has no DIPY; all runtime/test/notebook imports are local modules or NumPy/SciPy/Matplotlib/stdlib |
| Explicit local mathematics and symbol/call-role audit | passed | sph_harm_y/eval_legendre are general special functions; pinv is generic SVD; task-specific CSA factors, objective and normalization are local; no domain imports, pretrained checkpoints or cross-task imports |
| Entire test suite | passed | `python -m pytest test/ -v -p no:cacheprovider`: 24 passed in0.94s, exit0; private pytest.log |
| Active Notebook execution | passed | `python -m jupyter nbconvert --to notebook --execute notebooks/hardi_csa_odf.ipynb --output reviewed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=120`, exit0 |
| Notebook scientific comparisons and visual inspection | passed | Two extracted images privately inspected: b0/anisotropy/error maps and three original/fresh angular distributions; curves overlap; legends/units visible. No missing method substitution. Shipped notebook retains executed outputs |
| Plans/source/README semantic consistency | passed | Reviewed exact SH signs and negative-m convention, 8pi denominator, unit integral, float32 path, signatures/defaults, full evaluator scale and paths |
| Template/inventory/licenses/archive hygiene | passed | No generate_data.py because reused observations; no extra public metadata; original files/paper/source/logs/figures outside task; complete license/attribution text consolidated below |

Notebook timeout120s is justified by a subsecond CPU reconstruction plus kernel startup and plotting (complete execution under5s). The first attempted notebook launch selected the unrelated Anaconda kernel and failed on missing sph_harm_y; the recorded successful run prepends the shared-env Scripts PATH. This was an environment selection issue, not a method change. Successful run logs include Windows Proactor-loop fallback and local Jupyter transport warnings; no cell errors occurred. Initial pytest default temp location was inaccessible; the successful run uses private writable TEMP/TMP. Neither initial attempt is counted as a pass.

The notebook intentionally creates a self-contained single-method reproduction comparison because this is newly authored review material; there was no supplied upstream experiment notebook requiring additional method groups. Polar plots use nonnegative displayed radius for readability and explicitly disclose this; scoring and stored arrays keep signed values. Physical correctness checks establish mathematical identities, not independent tissue accuracy.

## Packaged numeric hashes

| Artifact | SHA256 |
|---|---|
| `data/raw_data.npz` | `26901906305e528d3ed12a9bd0ecadd4a806d402ff057b4888f9dc6b4d3bbee7` |
| `evaluation/baseline_reference.npz` | `bfe5bdcd37efc6cd7b6003d000dca405962fe5abdbe1336d6560ee15d2a5df5d` |
| `notebooks/reference_outputs/cleaned.npz` | `2f304a7ece97119c0bc40738b4cbd13abc50f7398e5b187429d055e8f5da55df` |
| `notebooks/reference_outputs/upstream.npz` | `bfe5bdcd37efc6cd7b6003d000dca405962fe5abdbe1336d6560ee15d2a5df5d` |
| `output/reconstruction.npz` (coordinator rerun) | `e6b3142cac1e117599881dfb133cf65697b222f44a8e9f92d6ceddd31102360e` |
| `test/fixtures/upstream_stages.npz` | `d43939e74c6230c1e096801ede05ddc4d8906d0d52ffb3c616d766d10054676c` |

## Maintainer-only archive index

Archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task08_medicine_diffusion/`. `oracle-env/` is the isolated original environment; pip inventory is oracle-freeze.txt. The original data, exact source archive, private capture/stability scripts, papers, stdout/stderr logs, executed temporary review and extracted figures remain here and are not solver inputs. The original full dataset is retained only here; raw NPZ is the documented crop.

| Archived file | SHA256 |
|---|---|
| `capture.py` | `eea0cfcc77b2e554dc177b1d61371ec0bcf54564a125c3473a5547a9313372f4` |
| `stability.py` | `71aa95a5145be1e62811f7132a25c208af9d048688e6d564156ea2ac3a665159` |
| `stability.json` | `107945722c8c3b026791f8a3b7c92befc20bbcb25365f80e3ea8d943c57f00c3` |
| `upstream_run.json` | `177d6f4e209d4a14dcff2b68f7c661b791ed53fe2b7f154dbfb84d355b9560b3` |
| `upstream.log` | `c7366371eec0bbf75b6e1862e06f843a180f6d8f09715b732cd039bf39d26c8a` |
| `oracle-freeze.txt` | `3a29e64fd462473a6f04533b46a0bf24561fe237d938e710a919fb9b0e20df5b` |
| `install.log` | `ca753c26abdbd918f4aab66eab1ec2e8f55a04b29ef5883e59348907ed3b81c6` |
| `upstream.zip` | `99dfe215e413cb6b38d0eca1a9636011854471fe592fe3c34937db1038c7afde` |
| `shm-v1.12.0.py` | `8c25ae793f1e8272864039559f87dfacc6ca2229f2dfd93b0c37b211c40027bb` |
| `references.bib` | `d77aa495e75b326fa257e369b2a493e73806cd90f33ed1375d8faccd74114353` |
| `LICENSE` | `465f0850d617ed64a3052e20e2bb8d3be641f8eef3605c79ad2d7513cf1768ef` |
| `stanford.xml` | `44b99e68406a969f6fa97601ba7b64c8babee337cc037f95c7ff1bd16ec45602` |
| `paper-isbi2009.html` | `fa79760829b36b9b928d939e54111e1118d81113ddecb723a1690a40350c0dd4` |
| `paper-mrm2010.html` | `c82d9ff2d818c28710bc0b98040ef2bf3e953451791aed2ed75cff38ddf3ad6b` |
| `dwi.nii.gz` | `2c22516114eb4acbedf0052bc4fda6d0549a8c34758da69164a219d39bafd656` |
| `dwi.bvals` | `aa9399094599ddd516603ad81272892bdda85c2d9bee5314086efce8db231e4d` |
| `dwi.bvecs` | `42dd0ed0b39824bac888be7f3d68e9dd4c6a85f1c8bfa26a050653c618111723` |
| `pytest.log` | `3d504b75f6daed69b6b05215e62e2d6b66000d15d9276d0a3dce7d38e74cfbe9` |
| `nbconvert.log` | `70b237d86d46e282c09b829c82f501502e00b7942a93a29fa3bf492ab1f93087` |
| `reviewed.ipynb` | `1cdc30e3e70c3c28f8e0f26b31725e55e57a44617bb729f0a3a55b4a83098e9f` |
| `figure0.png` | `7fe4dcd60af91da162fbb8d749ba6a36bfd6c1ee0a74cb7c22d999abe9d83a8e` |
| `figure1.png` | `73c28c828c37ab99172d65a6a10212c3da76ecac623d20c15cc77a817cecca0e` |


## Independent coordinator review (2026-10-18)

A separate coordinator reran the standard CLI, the standalone Judge from a different working directory, the complete pytest suite, and the active notebook through nbconvert in the shared CPU environment. All four commands returned exit code 0. Command arguments, timing, stdout/stderr, the executed review notebook, and extracted figures are archived at `batch10/coordinator/hardi_csa_odf/`; the structured check report is `batch10/coordinator/hardi_csa_odf-review.json`. The coordinator inspected all embedded comparison figures and reviewed the implemented source and mathematical plan. Packaged reference snapshots were preserved around the notebook rerun; any rerun snapshots are separately archived.

The generated output now identifies the coordinator CLI rerun rather than the worker snapshot. Its maximum absolute difference from the preserved `notebooks/reference_outputs/cleaned.npz` snapshot is 9.43689570931383e-16 (float64 coefficients); the independent Judge and parity tests still pass their predeclared tolerances. The worker snapshot hash remains recorded for the notebook measurement above; these arrays are not asserted bitwise identical across the two executions.


## Consolidated original license and attribution texts

Task-level license/notice files were consolidated here at user request. The complete original text is retained verbatim below, with its original-byte SHA256 for verification; archived originals are outside the task in revision_docs/task08. These terms continue to apply.

### Original LICENSE

SHA256 `465f0850d617ed64a3052e20e2bb8d3be641f8eef3605c79ad2d7513cf1768ef`

```text
Unless otherwise specified by LICENSE.txt files in individual
directories, or within individual files or functions, all code is:

Copyright (c) 2008-2026, dipy developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the dipy developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

### Original NOTICE

SHA256 `9d6e825e14a9968321510f87adaa7f55d49eed2a0e1028e29cf05b96356f244f`

```text
Local SH/CSA implementation adapted from DIPY 1.12.0 under BSD-3-Clause; see LICENSE.
Stanford observations: Rokem, Ariel; Yeatman, Jason; Pestilli, Franco; Wandell, Brian (2013), High angular resolution diffusion MRI, Stanford Digital Repository https://purl.stanford.edu/yx282xq2090 . Data licensed under Open Data Commons Public Domain Dedication and License 1.0, https://opendatacommons.org/licenses/pddl/1-0/ . Preserve privacy and do not attempt identification.

```


## Document and six-part notebook revision (2026-10-18)

At the user's explicit request, this revision consolidates task-level license/notice files into the full-text legal section in this validation record and removes the separate files. The terms, copyright attribution and data-source notices continue to apply. The full old documents, original notebook, license/notice bytes and all historical reference snapshots were archived before editing at `batch10/revision_docs/task08/hardi_csa_odf/original/`, outside this task. No skills were used. EHT original's complete approach/design and active comparison notebook were read as depth references; its science and data are not imported into this task.

Both plans now explain the concrete forward/inverse mathematics, dimensions, objective normalization/signs, parameter choices, actual numerical execution and scientific limitations. The design documents every shipped source/Judge callable's actual signature, shapes/dtypes/units, defaults, explicit versus assumed validation, errors, mutation/I/O behavior, module dependencies, distinct CLI/notebook flows and test mapping. No unimplemented optimizer, iterative update or convergence feature was invented for these direct methods.

The primary notebook now has exactly these six ordered H2 sections: `1. Problem`, `2. Data`, `3. Physics`, `4. Method`, `5. Results`, `6. Conclusion`. Data includes every visible array key/shape/dtype/unit, actual source and acquisition table followed by active loading. Physical acquisition, coordinate/unit/boundary conventions and mathematical reconstruction are explained separately. All earlier scientific plots, method groups, independent checks and metric tables are retained and freshly executed through local src functions. There is only one notebook inside the task.

Latest full suite: **24 passed in 0.94 s**, exit 0. Full active `python -m nbconvert --to notebook --execute notebooks/hardi_csa_odf.ipynb --output executed.ipynb --output-dir <private revision archive> --ExecutePreprocessor.timeout=180` returned exit 0, with 4 executed code cells and **2 image outputs**. Every image was extracted and visually inspected in this revision. Plot labels, comparisons and quantitative scales remain readable; no absent method/result was filled in. The 180 s timeout covers startup and plotting while the measured reconstruction remains short. Private `pytest.log`, `notebook.log`, `verification.json`, `executed.ipynb` and `figure*.png` record commands, timings, outputs and visual evidence; the executed notebook was promoted to the single primary notebook. Notebook warnings are the previously identified local Windows/Jupyter transport advisories; there are no cell errors.

Fresh standalone Judge report: `{"relative_l2": 2.966295310539751e-15}`, status `pass`. The fresh output and Judge report are separately archived as `fresh_reconstruction.npz` and `fresh_judge_report.json`; full new timing/diagnostic streams are in verification.json. Historical reference_outputs bytes were snapshotted and verified identical before/after execution, including any old runtime/source-hash measurement records. Those source hashes continue to describe their original measurements, not the current documentation revision. No observation, hidden reference, parity fixture or threshold was changed. Protected-asset before/after SHA256 maps are recorded in verification.json. Runtime algorithms and executable statements are unchanged.

License migration verification: LICENSE: `465f0850d617ed64a3052e20e2bb8d3be641f8eef3605c79ad2d7513cf1768ef`; complete original bytes verified within the consolidated section; NOTICE: `9d6e825e14a9968321510f87adaa7f55d49eed2a0e1028e29cf05b96356f244f`; complete original bytes verified within the consolidated section. Historical mentions of an accompanying LICENSE inside the preserved original notice are quoted original text; they now resolve to the consolidated license immediately preceding it, rather than a standalone file.

### Current revised file hashes

| File | SHA256 |
|---|---|
| `README.md` | `cb5ab3aa0d04d23cd8ef3b936134d4c7041882f13f4bf545c1f86725a04fce90` |
| `plan/approach.md` | `1092cea928da3c4cb661ac1e9b21e216214c408d7da8c545940aa99af4cbdb16` |
| `plan/design.md` | `a7810da7749c80133e28e4d3bb5ec812eccd608e8192bc0e26a3369fe8e65ca5` |
| `notebooks/hardi_csa_odf.ipynb` | `65f5ea2b0bb0eefb9e0eb5d8e958896599dbe30bb056d640b0557b55365f4104` |
| `src/__init__.py` | `c8f7988881dcc5f9d0da1cc935d5c043a7d3f0f7eebaa2288971bdfc85123201` |
| `src/main.py` | `0a96df1fbba7f4cc622c9ebb5a997abed07f1cdcfe5f735c9a4aad1bc49605d9` |
| `src/physics_model.py` | `6cdccf095862605c176a330f394aca997fa8c25704eb51d410c6671568387504` |
| `src/preprocessing.py` | `d16f7643795f9a75a4efb6d822954f6c5b0209f6916463f8f162f1cf1b509bc1` |
| `src/solvers.py` | `9abeed6262325366fe630ef65c073a351382bdbbe745556b4fb64a0ba36bece9` |
| `src/visualization.py` | `9f6b0d275a1a19def17ac0533dbf0058f4bb5cd9f0eccd09d8da5223048eeb8e` |

The notebook no longer rewrites notebooks/reference_outputs/cleaned.npz during reexecution. Its freshly computed submission is instead retained in output and the private revision archive; the historical cleaned snapshot remains consumed by its repeatability test. NumPy 2.4.6 pinv documentation was inspected to verify the actual default singular cutoff rcond=1e-15; both plans now state it explicitly. The acquired-b0/angular-norm/error maps and all three angular original/fresh comparisons were inspected.


### Final Markdown-only editorial polish

The final notebook wording merges repeated Data introductions into one source/selection paragraph followed by the complete tables. Problem, Physics, Method, Results and Conclusion were reviewed together to remove repeated provenance, acquisition or no-iteration statements while retaining task-specific mathematical definitions, processing caveats, comparisons and limitations. Only Markdown cells changed: every code cell, execution count and embedded result/output is byte-equivalent under deterministic JSON serialization to the already executed notebook (SHA256 `da1eb13680905d009ee86e267b05aa4400a02bcfa9b238e38f6d3fd3c6df3d78`). The successful numerical execution and 2 inspected image outputs therefore remain applicable; no identical numerical rerun was needed. `revision_docs/task08/hardi_csa_odf/markdown_final.ipynb` archives this final wording, and the notebook SHA256 in the current revised file table above is updated.
