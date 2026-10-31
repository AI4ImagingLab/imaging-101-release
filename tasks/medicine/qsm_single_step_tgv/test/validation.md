## Public URL role audit (2026-10-29)

Removed public link `https://www.neuroimaging.at/media/qsm/20170327_qsm2016_recon_challenge.zip` because its role is mixed observation/reference archive with reconstruction/evaluation .m scripts. Original public context: The real-data source is the author-hosted [QSM challenge archive](https://www.neuroimaging.at/media/qsm/20170327_qsm2016_recon_challenge.zip). The challenge publication documents acquisition and preprocessing. The exact protocol field value is tied to this volume by an independent identity check against the corresponding original complex acquisition, rather than estimated from a target susceptibility map.

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Validation and scientific provenance

## Scope and status

This is a full-volume research reconstruction of real human MRI, not a simulated phantom or a simplified two-dimensional inverse. The bound acquisition is the complete 2016 QSM Reconstruction Challenge single-orientation wrapped phase. The implementation extracts the single-step total generalized variation method of Langkammer et al., DOI `10.1016/j.neuroimage.2015.02.041`, through the independent QSM.m reference code. One independently established anisotropic-adjoint error is explicitly corrected below. Accordingly the delivered method is a **documented minimally corrected reproduction**, not an unchanged reproduction of that upstream implementation.

The standard GPU reconstruction, independent original and corrected upstream runs, numerical stage/physics checks, standalone Judge, final 48 tests and active notebook all passed. Full-volume CPU reconstruction was not completed: CPU stage parity is verified, but no claim of complete CPU timing/compatibility is made. The default executable path is CUDA. Public acquisition data have no explicit redistribution grant in the obtained archive; local research use is the current scope, and permission for an external redistribution remains unverified. These two limitations are not disguised as passed checks.

## Private archive and exact sources

All archive-relative names below resolve under maintainer archive `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task08`. This directory is outside the task and never imported by its runtime, Judge, tests or notebook. Complete source checkouts, downloads, PDFs, capture programs and logs are retained only there. Remote revisions were actually queried; these are not inferred solely from local tracking branches.

| Item | Verified identity | Archive |
|---|---|---|
| Reference repository | `https://github.com/kamesy/QSM.m`, commit `f22dc40d81495b37438fbf66f86c9281f8326813` | `qsmm-remote.json`, `qsmm-source.zip`, `qsmm-full/QSM.m-f22dc40d81495b37438fbf66f86c9281f8326813/` |
| Original data repository | `https://github.com/berkinbilgic/qsm-challenge-2016`, commit `ab218740e2ed9c218a4d009ed58d3ae80229a349` | `qsm-challenge-2016-remote.json`, `challenge/data/` |
| Official author-hosted dataset | `https://www.neuroimaging.at/media/qsm/20170327_qsm2016_recon_challenge.zip` | `official-qsm-challenge.zip`, 244982259 bytes, SHA256 `2bc4b345c16d198b65e443df41cfab1f52fc6e0bc75483777284471566fb6d36` |
| Selected paper | Author-hosted `https://www.neuroimaging.at/media/qsm/langkammer2015.pdf`, published NeuroImage 111 (2015) 622-630 | `root-graz-langkammer2015.pdf/.txt`, DOI above |
| Challenge paper | DOI `10.1002/mrm.26830`, MRM 79 (2018), 1661-1673 | `root-graz-qsmchallenge.pdf/.txt` |
| Protocol identity | Corresponding original complex `Img.mat` and script-paired `prot.mat` | `root-fansi-Img.mat`, `upstream/prot.mat`, `root-qsm-protocol-identity-check.json`, `root-fansi-img-acquisition.json` |

The initially obtained Maastricht attachment `root-langkammer-2015.pdf` had a correct cover citation but an unrelated RF-power paper body. It was rejected as methodological evidence. The Graz author copy has the actual QSM body and is the selected paper. This discrepancy is retained instead of silently citing the wrong attachment.

The official archive's phase, magnitude, mask, spatial resolution, COSMOS and STI items were checked against the working source data; all selected bound observation/reference files checked in `official_member_identity.json` are byte-identical. The wrapped-phase MAT SHA256 is `1e5b3c2e85691deeab0a699fd14a724f19860bf5070060905b4daca63b220855`. Full exact upstream-file hashes, downloads and remote-query evidence are in `download_manifest.json`, `screening_hashes.json`, the remote JSON files, and the final hashed appendix below. No observation was generated locally. Batch axes and NPZ containers were added without changing voxel values.

The protocol field `2.89362 T`, echo `0.025 s`, and proton gamma `42.58 MHz/T` are acquisition constants, not fitted to COSMOS/STI. Parent independent checking found the corresponding complex acquisition's angle identical to the bound wrapped phase to less than 1.2e-7 rad voxelwise, and the original loading script explicitly pairs it with the protocol file. Thus `s=19.35383488668318 rad/ppm` is legitimate calibration. Magnitude and brain extraction mask are public acquisition products. No target-derived calibration is exposed. The mask is used unchanged; no tissue segmentation from hidden references is supplied.

## Original execution before the local port

Private success criteria and capture tolerances were fixed in `success_criteria.md`. Original C MEX source was compiled using Octave 8.4.0 and mkoctfile under WSL Ubuntu 24.04, isolated from the cleaned Python runtime. All required original kernels are in `root-oracle-build`; the entire original source and environment are maintainer-only. FFTW threads and OpenMP/BLAS threads were explicitly set to four. The acquisition itself is deterministic and no synthetic seed is used. A seed of 173 applies only to an independent mathematical adjoint test.

`run_original_sstgv.m` runs the actual reference wrapper on the exact full bundled phase/mask/pitches. Persistent `run_oracle_driver.py` captures native WSL exit status, full stdout/stderr and runtime. The command is `wsl -d Ubuntu-24.04 -- bash -lc "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; octave-cli --no-gui --quiet <archive>/run_original_sstgv.m"`. `oracle_full_status.json` records exit 0, 62.0472 s process time; kernel runtime was approximately 58.13 s and 2170 updates with original wrapper defaults. Result: `upstream_full_sstgv.mat`. Evidence: `oracle-full.stdout/.stderr`. A successful import or precomputed upstream notebook was not substituted for this run.

After the independent bug investigation below, `run_canonical.m` ran the full corrected TGV, original TV and unmodified original TGV at the paper-derived pair. Captures `canonical_mode_1.mat`, `_2.mat`, `_3.mat` separate those three algorithms. Corrected and unmodified TGV both stopped at 2480 updates; TV stopped at 2040. `run_joint.m` additionally exposed eta without altering the established updates: `canonical_joint_1.mat` and `_2.mat`. Corrected TGV runtime was 66.67 s. Repeated susceptibility values were bit identical with the prior canonical run. Their process commands, exit 0, environment and timing are in `oracle_canonical_status.json`, `oracle_joint_status.json`, and paired `oracle-canonical.*` / `oracle-joint.*` logs. Source instrumentations only expose existing states; they do not feed any reconstructed truth back into observations.

Private parameter sweeps and 5000-step stricter-stop probes are retained as `run_sweep*.m`, `oracle_sweep*_status.json`, `oracle_sweep_metrics.json`, `run_strict.m`, `oracle_strict_status.json`, `corrected_strict_*.mat` and their logs. The final pair is the documented published physical pair, **not** the best COSMOS-scoring sweep point. Different noise/model error, anisotropy and historical reference processing prevent equating these comparisons with a new challenge leaderboard entry.

## Investigated discrepancy and paper/code conventions

Original `src/inversion/mex/pd_tgv_mex.c` uses each vector component's pitch for every tensor-divergence term in that component. A derivative must instead use its derivative-axis pitch. This distinction vanishes on an isotropic grid but not on the actual `(1.0625,1.0625,1.0714285714285714) mm` grid. An independent Frobenius adjoint calculation with random vector/tensor fields gives lhs 11.788172603891713; original rhs 11.556613298768518 (relative error 0.019643358890651864); corrected rhs 11.788172603891724 (relative error 9.041385288914794e-16). See `independent_anisotropic_adjoint.json`.

Six source sites (three divergence component expressions in float and double branches) were corrected to derivative-axis pitches in `corrected-oracle-build/pd_tgv_mex.c`. All other updates, crop, mask predicate, steps, stopping, initialization and precision conventions remain source-derived. Capture `pd_tgv_capture.c` exposes primal/dual states; TV capture similarly exposes its eta. Original source and original results remain separately preserved. Independent adjoint and energy-directional-derivative tests now verify the correction. Parent review authorized this transparent correction; it is not silently passed off as unmodified upstream parity.

The paper states the full squared discrepancy penalty. QSM.m uses half squared discrepancy. Therefore the paper physical coefficients 0.0015/0.0005 become 0.00075/0.00025 with half fidelity; phase-unit implementation multiplies both by `s`, giving lambda 0.014515376165012384 and ratio 1/3. The wrapper defaults lambda 1e-4, ratio 3 are a distinct configuration, retained only as an explicitly labeled comparison. The delivered discretization is QSM.m's seven-point finite differences, not a claim of byte-identical published executable identity. Source center-hole Mi convention and single-update stopping every ten steps are preserved. The generic roll operators are periodic scratch primitives; masked states and the zero-support crop margin give the actual zero-exterior solver. See both plans for exact updates and caller roles.

## Equation / stage / independent case traceability

The paper's task-relevant mathematical stages are explained directly in approach; equation labels alone are not specifications. Upstream paths below are relative to the exact archived QSM.m source.

| Required stage | Upstream implementation | Local implementation | Independent validation |
|---|---|---|---|
| Acquisition units and phase-to-susceptibility conversion | `sstgv.m` input-unit conventions and acquisition protocol | `preprocessing.PHASE_PER_PPM`, `solvers.reconstruct` | Protocol complex-phase identity; README constant contract test |
| Crop and fixed support | `src/inversion/sstgv.m`, `cropIndices` | `preprocessing.crop_slices`, `physics_model.interior_mask` | Full-data bounding box; explicit center-hole case |
| Wrapped-Laplacian/background elimination | `lapw.m`, `lapw_mex.c` | `physics_model.wrapped_laplacian` | Captured actual patch; independent quadratic/harmonic and 2pi invariance |
| Laplacian/dipole PDE and signs | `pd_tgv_mex.c` dual constraint stage | `laplacian`, `wave_operator`, solver nu stage | Captured nu states; independent transverse Fourier mode; independent Judge stencil |
| Vector/TGV decomposition and Frobenius norm | `update_duald/update_dualf` in `pd_tgv_mex.c` | `symmetric_gradient`, `project_balls`, p/q updates | Independent projection and gradient derivative checks; captured p/q |
| Symmetric-divergence primal update | `update_primald/update_primalf`, corrected sites above | `symmetric_divergence`, w update | Anisotropic Frobenius adjoint and corrected upstream w fixtures |
| Proximal fidelity and primal susceptibility | `update_primald/update_primalf` | explicit chi/eta updates in both solvers | All primal states 1/2/10/50 steps; joint full-output parity |
| Step bound, extrapolation and stopping | `mx_pdd/mx_pdf` loop | `step_sizes`, both primal-dual functions | Source single-update/every-ten test; exact full update counts |
| TV comparison | original `pd_tv_mex.c` | same solver with `ratio=0` | Original C 1/2/10/50 chi captures, full TV parity |
| Final physical maps and measures | full original/corrected runs; independent COSMOS/STI acquisition | `reconstruct`, visualization and independent Judge | Full CUDA vs C arrays, scientific metrics and orthogonal figures |

## Fixture and reference consumers

Every numeric fixture has an active consumer. `test/fixtures/upstream_input.npz` is the measured real-phase patch, mask, physical pitches and original `lapw` capture, consumed by upstream-stage tests. `corrected_step_{1,2,10,50}.npz` contain the separately corrected C chi, eta, w, p, q, nu and Mi, captured **before** local implementation via `capture_states.m`; tests compare every state on NumPy and torch CPU. `tv_step_{1,2,10,50}.npz` are original C TV captures from `capture_tv.m`, consumed by TV stage tests. These files are not generated by local reconstruction functions. Their original MAT captures and compile/instrumentation scripts are archived.

Notebook references: `original_default.npz` consumes `upstream_full_sstgv.mat`; `original_canonical.npz` consumes unmodified `canonical_mode_3.mat`; `canonical_tgv.npz` consumes corrected `canonical_joint_1.mat`; `canonical_tv.npz` consumes original-TV `canonical_joint_2.mat`; `independent_measurements.npz` contains supplied COSMOS and STI chi33 from the real challenge. All are consumed in the active notebook for labeled comparisons. `evaluation/baseline_reference.npz` is the independently corrected C joint method result, consumed only by the evaluator/tests. It is neither COSMOS nor actual truth. There is no `ground_truth.npz`: imperfect independent multi-orientation measurements are not silently relabeled biological ground truth.

## Fresh cleaned measurements and metric provenance

Cleaned environment: Windows, research runtime Python with NumPy 2.4.6, torch 2.9.1+cu128, CUDA 12.8 and Matplotlib 3.11.1, NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9, available memory 17170956288 bytes. GPU peak allocated for this solver was 893033984 bytes. The private runtime is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime`. Its kernel specification resides under `share/jupyter/kernels/imaging-101-research`. No upstream domain package is imported by cleaned code. CPU stages use NumPy/torch CPU float64; full CPU ten-step timing was 41.47 s and the complete 2480-step CPU path was not run. The task therefore declares a validated GPU standard path.

`run_gpu_driver.py`, `gpu_full_status.json`: full local TGV 2480 iterations / 33.3941443 s and full TV 2040 / 13.6960140 s. The full independent-C versus cleaned-CUDA differences in `full_cuda_parity.json` are:

| Quantity | Maximum absolute difference | Relative L2 difference |
|---|---:|---:|
| TGV susceptibility, ppm | 3.885780586188048e-16 | 6.586585284022861e-16 |
| TGV eta, rad | 1.4276774207289122e-15 | 1.5692319984478898e-15 |
| TV susceptibility, ppm | 5.551115123125783e-16 | 6.021502947157964e-16 |
| TV eta, rad | 1.5543122344752192e-15 | 1.386080097947208e-15 |

Small-stage limits were frozen at abs 5e-11 for all solver states and 2e-13 for wrapped Laplacian. Actual CPU/torch CPU errors were below 5e-16, GPU all-state error below 3.9e-16, and wrapped Laplacian below 1.78e-15. Exact captures and initial comparisons are in `initial_stage_parity.json`, `torch_cpu_stage_parity.json`, `gpu_full_status.json`. These are independent numerical checks, not self-generated implementation snapshots.

Cosine NCC is dot product divided by the two norms, without centering; range NRMSE is RMS error divided by reference range. Separately named relative L2 is error norm divided by reference norm; it is never mislabeled NRMSE. Metrics use the same fixed brain support, original sign and scale, no fitted offset or registration. For corrected TGV versus independent COSMOS, NCC is 0.7171843885082954, range NRMSE 0.03781170851995956, and relative L2 0.8851554926127444. TV has NCC 0.7312303121412174 and relative L2 0.7839236756014485. Against STI chi33 the relative errors are approximately 108.1079% and 95.2210%. TGV does **not** outperform TV here. These limitations are visible in notebook figures and conclusion.

The challenge report's historical submission L2 errors range approximately 69.0%-140.9%, median 83.9%; those numbers are historical context, not this run's score. Published postprocessing and reference preprocessing (including LBV/polynomial issues) differ. Single-orientation scalar susceptibility is imperfect for anisotropic tissue; COSMOS and STI differ themselves. The research value here is an explicit validated joint single-step inverse method with a transparent correction, not a claim of exact tissue recovery or state-of-the-art accuracy. A cosine truth threshold of 0.9 times a poor baseline was deliberately not used to hide these limitations.

Instead the Judge evaluates **method reproduction plus measured physics**: both chi and eta must satisfy NCC >=0.999999 and range NRMSE <=1e-5 against independent C. Repeated canonical C captures are bit-identical, and fresh C/CUDA comparisons justify a conservative nonzero float64 numerical floor. These limits were fixed before the full CUDA solve and not relaxed afterward. Their interpretation is validated standard-method agreement, not biological truth. Joint PDE-relative residual must be <=0.004715425872582365 (1.1 times independent 0.004286750793256695); eta RMS <=0.0889638371043339 rad (1.1 times independent 0.08087621554939445). TV comparison PDE residual is 0.0038139922817763804 and eta RMS 0.08681095561261024 rad. Both joint variables are evaluated so a zero chi plus an arbitrary discrepancy potential cannot exploit a constraint-only score. Tests reject zero, wrong-sign, zero-chi-with-reference-eta, and zero-chi-with-raw-phase-eta cases. A zero estimate has defined NCC 0; empty/constant/zero evaluator references are errors. All JSON uses finite numbers.

## Commands, notebook and observed execution

The private `run_verification_driver.py` ran the actual standard `python -m src.main`: exit 0 / 39.9964581 s process time, followed by the exact standalone Judge command: exit 0 / 2.6489506 s. Logs are `cli.log`, `judge.log`, status `verification_status.json`. Fresh output contains both required arrays; the artifact is not the upstream reference copied into output.

The first actual nbconvert attempt exited 1 after 3.9004837 s with observed `NoSuchKernel: imaging-101-research`: the external Jupyter search path omitted the private registered kernel. This failed status remains in `verification_status.json`. The initial complete notebook log was accidentally overwritten by the corrected rerun; that limitation is recorded explicitly in `initial_notebook_failure.md`, rather than claiming its original stderr remains archived.

With `JUPYTER_PATH=<research-runtime>/share/jupyter`, corrected `run_notebook_driver.py` executed `python -m jupyter nbconvert --to notebook --execute notebooks/qsm_single_step_tgv.ipynb --ExecutePreprocessor.timeout=1800 --output executed.ipynb --output-dir <private archive>`. Actual exit 0, 79.3528348 s, `notebook_status.json`; all code cells executed, no error outputs, five PNG figures. The timeout accommodates full reconstruction plus plotting and runtime contention. Windows emitted a zmq Proactor selector-thread warning and an IPKernel TCP-transport warning; neither was a numerical or reconstruction failure. Exact warnings remain in `notebook.log`. The completed reviewed notebook is the canonical task notebook; its temporary executed copy remains outside the task.

All five actual extracted figures `notebook_figure_1.png` through `_5.png` were viewed. Raw wrapped phase/magnitude/mask are nontrivial human data; three orthogonal reconstructions show anatomically coherent vessels and deep gray nuclei, visible model discrepancies, signed scale and matched masks. Original and fresh comparisons use actual computed arrays. The paired bars show TV's better COSMOS agreement. The four-way figure retains both independent reference definitions. Eta/PDE/convergence shows computed discrepancy and nonzero finite residual with actual stopping curves, not fabricated zero panels. No clipped labels, missing methods or empty panels were observed. `notebook_review.json` records code counts and extracted figure identities.

Full command `python -m pytest test/ -v -o cache_dir=<private archive>/pytest-cache`, with private TEMP/TMP and OMP/OPENBLAS/MKL threads 1, completed **47 passed in 36.85 s**. Native shell completion was checked; complete log is `full-tests.log`. Tests cover captured stages, independent physics/adjoints, known metrics, fresh pipeline artifact, malformed keys/shapes/dtypes/NaN/Inf/exterior/gauge, unavailable evaluator assets, foreign cwd, report-write failure and degenerate predictions. The full CLI was actually run separately; the test suite consumes its artifact rather than needlessly rerunning it.

Final independent review found a finite-input overflow case: a mask-contained, zero-mean pair of float64 values +1e200/-1e200 can overflow intermediate norms even though input entries are finite. Judge now checks every computed metric is finite before assigning metrics to the report, returning a descriptive error and empty metrics otherwise. The added actual subprocess test verifies exit 2 and valid UTF-8 error JSON. After this correction the same full command completed **48 passed in 41.61 s**, native exit 0, `full-tests-final.log`. No numerical threshold or solver behavior changed.

## Semantic and package review

Source signatures/defaults were read against design; equations, signs, derivative-axis pitches, Frobenius factors, phase/ppm scaling, objective half factors, source stopping, masks and backend dispatch were reviewed against source and independent cases. README explicitly defines the non-center Mi predicate. Low-level periodic derivatives and actual masked zero boundary are distinguished. Acquisition constants are tested against README. Paper, original code, corrected oracle and fresh results are labeled separately. No full CPU claim is made from a small fixture.

Import/call-role audit: source uses only NumPy, standard library, torch general tensors and Matplotlib plotting. Core forward and inverse stages are local executable arithmetic. Judge imports only standard library and NumPy and never solver modules. Test consumers import local mathematics and frozen numeric fixtures only. Notebook uses local modules and private-in-task reference arrays for labeled review; it never imports archived oracle code or a domain reconstruction package. No simulation generator exists because real observations were reused. README and requirements contain no GitHub URL. Public NPZ has phase/magnitude/mask only, no COSMOS, STI, seed, target object or target-derived calibration. There are no cross-task imports, root wrappers, parallel test directories, standalone metadata JSON, upstream checkout or PDF in the final task.

## Completion gates

| Gate | Status | Evidence / practical scope |
|---|---|---|
| Exact paper/source/data/equation mapping | passed | Source table, hash appendix, traceability and private remote evidence |
| Original and cleaned exact-bound-data executions | passed | Original native exit0; corrected C and fresh full CUDA captures |
| Independent parity and physics | passed | All-state fixtures; full parity; adjoint/Fourier/harmonic checks |
| Constants/units/coordinates/conventions | passed | Protocol identity; README tests; detailed plans |
| Standard CLI and standalone Judge | passed | Actual subprocess logs and output report |
| Invalid and poor submission handling | passed | Metric/contract/error/degenerate tests |
| Public information sufficiency and no leakage | passed | Acquisition metadata, joint output contract, public array inventory |
| Minimal dependency local mathematics | passed | Local operator/update extraction; no domain runtime library |
| Entire test suite | passed | Final 48 passed /41.61s; includes finite-input overflow subprocess |
| Active notebook and scientific comparisons | passed | Full nbconvert exit0 /79.353s; five inspected figures |
| Metric/threshold provenance | passed | Frozen boundaries; C rerun and CUDA errors; no truth claims |
| Plans and semantic review | passed | Actual signatures, factors, masks, units and execution reviewed |
| Final task inventory and required notices | passed | Template files only; MIT preserved below |
| Full-volume CPU execution | unverified, outside validated standard path | Standard is CUDA; CPU stages passed, full CPU was not run |
| External redistribution permission for real data | unverified | Public author archive has no explicit grant; local research scope only |

## Preserved upstream software notice

No standalone LICENSE or NOTICE file is shipped. The complete license required for substantial extracted QSM.m portions is preserved here, with source copyright pointers in the adapted modules:

```text
MIT License

Copyright (c) 2020 kamesy

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

Public dataset availability is not silently interpreted as the above software license extending to the human MRI data. The latter's unresolved redistribution status is recorded separately.

Final cache maintenance: automatic command review rejected recursive cache deletion despite a task-prefix check. A safer, non-destructive operation then moved only this newly created task's six cache directories in full to private `cache_archive_final`, preserving all bytes. Resolved source paths were verified inside this task before moving, and existing archive destinations were forbidden. Exact source/destination records are in `cache_moves.json`. No older task cache was touched. Final task directories contain no `__pycache__` or `.pytest_cache`.


## Exact release and archive hashes

All hashes below are SHA256. The release NPZ manifest was independently computed after execution; original MAT/source/PDF hashes identify the actual privately retained inputs.

| File | Bytes | SHA256 |
|---|---:|---|
| `data/raw_data.npz` | 30300843 | `0935181a3ec1fb34f7d77d9cf975e9f692859091f1ffc44f95772d77f65396a1` |
| `evaluation/baseline_reference.npz` | 14544402 | `6bd6dfa692226e9e66870a321ee582eae819922d73583eb988f571d07a7bca1c` |
| `notebooks/reference_outputs/canonical_tgv.npz` | 14544620 | `827f0e91081808f144cdc2558c3ef86f4454eedee3b905dc22b053b7669143d7` |
| `notebooks/reference_outputs/canonical_tv.npz` | 14539427 | `1189638b3f1025eed023ade7a763d1440d66764083d54b41d41363d374e25ed9` |
| `notebooks/reference_outputs/independent_measurements.npz` | 11402270 | `edfdda91b08afe167a6c3ed6b03b2794f63db28581e7c0f9a3d4548a39a5865d` |
| `notebooks/reference_outputs/original_canonical.npz` | 7272462 | `8781909b8a795d8c2f7c7595a49ef349b4fad01ec6a0afceaf7213b295ac8b1f` |
| `notebooks/reference_outputs/original_default.npz` | 7273912 | `7965750c74baa06e55510a718fca98669343cc816c6e97b1624ac2849d92078c` |
| `output/reconstruction.npz` | 14552703 | `17ecd942f6bae24c30c9ff51a77e98b27e56707f7c3c96772c87f528f8ff0657` |
| `test/fixtures/corrected_step_1.npz` | 15467 | `53369eee30cf49dce38f9186ae3938774df501775acd945316bef2b6e4c20006` |
| `test/fixtures/corrected_step_10.npz` | 57127 | `e7c9a81e8a9bab8946924236a040fc183bf8dbfbb67e94058c03b0c80ae2d9bb` |
| `test/fixtures/corrected_step_2.npz` | 35678 | `bcc0b3961109231436ede133ffbf02de87ed5168b1006771efb76b4bdcf0c93a` |
| `test/fixtures/corrected_step_50.npz` | 57106 | `aaca8fefa58cb1d85872771118db226b649ec08206f24c30b9a80ddee3e29600` |
| `test/fixtures/tv_step_1.npz` | 5195 | `69dd2e55a1b0abfcf8a4a7257ad84e73d405163e94f536fb5fbfec02ecb71f82` |
| `test/fixtures/tv_step_10.npz` | 5227 | `47051caf41b6d6dcdc5903eec972a9364cc4947906de13769622a94745178fcc` |
| `test/fixtures/tv_step_2.npz` | 5213 | `ec0042ba3f616fa927f19da15630f39b706082444a62fda6ce2ad642d7cc7584` |
| `test/fixtures/tv_step_50.npz` | 5219 | `74f7e86d859d85ad57a6a3e871c966a7bb4dc7e69a8113a7e9ccc25b138645f4` |
| `test/fixtures/upstream_input.npz` | 78222 | `d71a53e913b2c4eff4ba6706e92f23b54203239f7fbe4197e994d37a97bcb543` |
| `archive:qsmm-source.zip` | 615145 | `d9d8e0a335e18c1e41a2450955eb9687d2a9e5bfd6a6bd503fd1f4d6fe94f9a5` |
| `archive:root-graz-langkammer2015.pdf` | 1289391 | `f8be114a6e3c109c7364c549ab80c1e081bf2578be7fe79b53bd5b7c2de29f24` |
| `archive:root-graz-qsmchallenge.pdf` | 621652 | `cfdaf643bd4b340d0cbbdb0e7978376c1d28a6dd892cbc5b8aa41a247faaaeaf` |
| `archive:challenge/data/chi_33.mat` | 3885810 | `721a1f4af3a457f2b8af0bbb5bde9cb75c5a1f9cc87ff728a4131a0352ad2645` |
| `archive:challenge/data/chi_cosmos.mat` | 7700581 | `a7affbc1afc71e1bfec6332b82e0929c4ec3c27eae338cdb5b4edf39a0702025` |
| `archive:challenge/data/magn.mat` | 3302824 | `b6608c49ef8d55e7ceade1fdf8e5b8ef51e09e0f0c48ad4d9e63b8d713fc44db` |
| `archive:challenge/data/msk.mat` | 52680 | `72eae42deb00c9c27f00c1ef9229f5fac8b8ae6eedd76db4447b192f9c6d27d7` |
| `archive:challenge/data/phs_tissue.mat` | 7477035 | `63ed90cad351fd8677558c34eba6571f127c6aef114229f08e23a25b001d73be` |
| `archive:challenge/data/phs_wrap.mat` | 27119926 | `1e5b3c2e85691deeab0a699fd14a724f19860bf5070060905b4daca63b220855` |
| `archive:challenge/data/spatial_res.mat` | 197 | `426c07d81fecd4b41ebc2b6102b27cfbda59aca711cf036d814d3713b550094e` |

Actual final environment: {'python': '3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]', 'platform': 'Windows-10-10.0.26200-SP0', 'nvidia_smi': 'NVIDIA GeForce RTX 4080 SUPER, 610.88, 16376 MiB'}. Full symbol/call-role inventory: private `source_call_audit.json`.


## Independent coordinating maintainer acceptance

The coordinating maintainer reviewed the actual operators, NumPy/Torch update paths, public output/metadata contract, independent Judge and both detailed plans. Checks include LD=-W, phase/ppm and full/half-squared scaling, derivative-axis pitches, symmetric Frobenius weights, masked transpose roles, center-hole predicate, crop-only exterior padding, source update order/stopping, zero-reference convention and actual CLI/notebook interfaces. All five actual embedded notebook figures were visually inspected, as was the independent four-way scientific review figure. The retained COSMOS/STI discrepancies and TV advantage are visible and explicitly stated; these measurements are not treated as exact tissue truth.

A standalone Judge invocation from the external coordinator working directory passed the fresh notebook submission. Exact report: private `../coordinator/qsm-root-independent-judge.json`. The coordinating structural/import/public-array/notebook audit found no errors. The final48-test native-exit0 evidence includes the added finite-overflow error-report case; no thresholds or reconstruction mathematics changed during that Judge fix. Required local-cleaning gates are accepted within the documented CUDA/local-research scope. Full CPU reconstruction and external data redistribution permission remain the separately identified limitations above. Final package hashes/inventory are indexed in private `../coordinator/qsm-final-review.json`.


## Public-contract revision 2026-10-29

Version2.0-joint-tgv-primal-witness adds actual source-computed auxiliary_w, physical ppm/mm, shape(1,3,160,160,160). Source primal updates unchanged; reconstruct requests existing return_state and zero-pads/scales its real w. Historical output/baseline/source/interface archived under benchmark_readiness_20261029/qsm_v1_archive. Fresh CUDA main2480updates31.2764s and full actual TGV+TV nbconvert exit0/no cell errors completed with .24process GPU memory cap coordinated with ADJUST. The new independently implemented physical objective is14.571308579816776; bound14.571308725529862 fixed with relative1e-8 numerical allowance, original PDE/eta thresholds unchanged. No actual tissue truth exists. Independent conditional witness descent lowers objective to14.570909040355454 and passes, without changing chi/eta; zero auxiliary and all-zero outputs fail. This control is not a fresh blind full susceptibility solve.
