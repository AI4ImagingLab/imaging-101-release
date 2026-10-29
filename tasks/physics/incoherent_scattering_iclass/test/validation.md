# Provenance, numerical validation and release gates

Validation date: 2026-10-18. Scientific implementation and executed review are
validated locally; **external-data redistribution permission remains unverified**.
Do not interpret working downloads, article access or the repository's code MIT
license as an explicit license to redistribute the separately hosted camera data.
All required local scientific cleaning/execution gates have passed under the
user's authorization for local research work. This separate publication
limitation prevents claiming unrestricted external redistribution permission.
No contact with authors was sent.

## Selected method, real observations and archive

Paper: Gil Weinberg, Elad Sunray and Ori Katz, Noninvasive megapixel fluorescence
microscopy through scattering layers by a virtual incoherent reflection matrix,
Science Advances 10(47), eadl5218, 20 November 2024;
https://doi.org/10.1126/sciadv.adl5218 ; PMC11578164. Full text was independently
downloaded from Europe PMC; supplementary PDF identifies sections S1-S7 and
Figures S1-S7. The article is CC BY-NC 4.0. Publication figures/PDF are retained
privately; the task notebook generates its own scientific figures.

Upstream repository:
https://github.com/Imaging-Lab-HUJI/Fluorescence-Computational-Imaging-Through-Scattering-Layers
Remote revision verified through repository API on this date:
`df2d69a0630905c4d6df12bb643e5354f26120e7`; `iclass_revision.json` archives
the response, distinct from an unverified local branch. Archive checkout folder
ends in this commit. Upstream `ICLASS.py` implements `I_CLASS`; `main.py`
implements `runCLASS`; visualization code implements legacy deconvolution.

Private archive index (absolute local maintainer path, not a runtime input):
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task02/`.
`final_manifest.json` records byte sizes and SHA256 for the authoritative files.
`upstream/Fluorescence-*` is the exact checkout, `iclass.zip` its download,
`paper.xml` the correct article, `paper_supplement/sciadv.adl5218_sm.pdf` the
supplement, and `supplementary.zip` the Europe PMC asset archive.
Earlier `paper_page.html` resolves to an unrelated article and `supplement.pdf`
is an HTML challenge response: neither supplies scientific evidence.

Data collection:
https://drive.google.com/drive/folders/18A_W_JemctYMtloonCAh3RTWsZfOJyJN
Selected Figure 3(D-F) folder:
https://drive.google.com/drive/folders/1OzmGtAwXmoBArWXEEpnbtg44_nUtv2H6
`fig3def_Run1.mat`, file ID `1cws6KfhXO4g9QkCRSmD3wQslD4VSJnNN`, is the
578437289-byte experimental scattering acquisition. `fig3def_Run0.mat`,
ID `1cj00CF-Wt1UQZ_aiaQJ3i3oQvR56phIG`, is the 248632781-byte independent
no-diffuser measurement. The accompanying `fig3def_readme.txt` explicitly maps
Run0 to widefield reference and Run1 to measurement. Archived download pages are
`gdrive_folder.html` and `fig3def.html`. No artificial camera images were generated.

## Bound data and documented discrepancies

Original HDF5 `I_cam` is `(150,1200,1200)` for Run1 and `(100,1200,1200)` for
Run0. HDF5 axis reversal is corrected by transpose `(0,2,1)`, matching the
upstream MATLAB-reader/permutation convention. All 150 complete frames are
retained. Upstream-supported Fourier center crop 1200 to 400 and inverse FFT
retains the full field with threefold coarser sampling. No amplitude compensation
is applied, so the camera DN scale has a factor-nine gain relative to conventional
amplitude-preserving resampling. The float32 bound array is `(1,150,400,400)`.
The raw NPZ contains observations only; no widefield/phase/MTF/truth/hidden seed.

Selected processing uses the paper's symmetric Hann window, sequential y then x,
on the released grid before temporal centering. It is not an assertion that the
paper used these exact unknown crop/processing coordinates. Earlier central
400-pixel spatial-subfield and no-Hann trials are archived under `oracle/` and
`oracle_fullfov/`, explicitly rejected. Strong scattering mixes light across the
field, so silently choosing a central subscene would change the physical problem.

Archived camera-size metadata says 1400 while actual stored images are 1200.
The actual array defines the task. Archived nominal detection wavelength is
520 nm; paper emission filter is 525 nm center/39 nm band. `BEAD_SIZE=1` differs
from the paper's 2-micrometer bead description; neither is used to invent pixel
pitch or claim a resolution. Objective 20x/NA0.6 and 7-mm diffuser separation
describe the reported experiment. Exposure is 0.2 s for Run1 and 0.005 s for Run0.
No absolute object-plane pixel pitch, calibrated sensor variance or sample
irradiance is available. Laser setting 40.2 is recorded as source mW metadata,
not a calibrated object irradiance. These limits are explicit in the public input.

Run0 is a different exposure/illumination measurement, **not truth**. Its temporal
average is full-field Fourier resampled for descriptive notebook comparisons.
No `ground_truth.npz` is manufactured. The hidden `baseline_reference.npz`
contains the validated standard-method image, separately identified below.

## Independent oracle and corrections

Command: `oracle-env/Scripts/python.exe capture_fullfov_hann.py`, working directory
the private archive. It loads the original `ICLASS.I_CLASS`. The original
`runCLASS` function is extracted by Python AST solely to avoid its module-level
demo I/O. A private even-sized `CenterCrop` shim exactly implements the source
frequency slicing without torchvision. No phase update or original output
function is changed. The selected source run executes all 250 iterations on the
exact bound observations, with matching Hann multiplication order. Runtime
15.21381960005965 s, peak CUDA allocation 1547343872 bytes, exit 0, no algorithm
warning. `oracle_fullfov_hann.log`, `oracle_fullfov_hann/run.json` and
`oracle_fullfov_hann/result.npz` retain the original run and legacy visual output.
The deterministic computation uses no randomized initialization (phase all ones);
there is no RNG seed dependency. Six Torch CPU threads were used.

Two source discrepancies were investigated mathematically instead of copied:

1. `ICLASS.RtoO` applies `ifft2` to `(flattened spatial frequency,illumination)`.
   Correct physical spatial inversion restores `(M,H,W)` and transforms H,W.
   A known image/spectrum round trip independently validates this correction.
2. Legacy visualization divides the spectrum of an already-combined RMS image
   by MTF. Supplement S18 requires dividing each frame spectrum before spatial
   inverse and RMS. A separate NumPy finite-case construction validates this
   noncommuting operation. Sigma 0.01 is the supplement's balanced setting.

`frozen/paper_corrected.npz` applies these corrections independently to the
original phase result; it is the standard-method oracle, not actual truth.
The notebook preserves original phase-only and amplitude visualization alongside
fresh physical phase-only/S18 results. We **do not claim unchanged reproduction**.
The source's circular flattened-grid covariance kernel is retained, distinguished
from the supplement's padded derivation. Its dense covariance identity, signs,
roll/reversal, scale factors and positive phase projection are independently
checked, so copying the same erroneous formula is not the only evidence.

Specifically, row packing `ix*H+iy` and cyclic differences operate in Z_(H*W),
not the genuine spatial product group Z_H x Z_W. A displacement that crosses an
iy boundary carries into ix instead of wrapping each 2D coordinate independently.
The dense covariance identity proves the retained release-indexed algebra; it
does not prove a full two-dimensional optical theorem or exact equivalence to
the padded supplement. True 2D convolution/inverse-transform physical checks are
independent of this approximation. Root semantic review identified this point;
the model limitation is now explicit without changing the 250-update core.

An initial development check guessed an unsupported 1.5-fold covariance gain;
the actual oracle gain is 1.1062. The check was corrected to independent oracle
J parity and a positive measured gain. This was a correction of an unjustified
physical-test hypothesis, not relaxation of the Judge or a claim of 1.5-fold gain.
The independently frozen Judge threshold remains 0.95 times original final J.

## Traceability and frozen consumers

| Paper stage | Upstream stage | Local function | Independent validation |
|---|---|---|---|
| Incoherent convolution model | Physical derivation, not a source oracle | `incoherent_forward` | Point PSF and intensity-conservation cases |
| Temporal fluctuations/Hann/Fourier matrix | `main.runCLASS` preprocessing | `prepare_matrix` | `preprocessing.npz` actual columns 0,13,149; parity test |
| Virtual covariance and coherent diagonal sums, S1-S15 | `ICLASS.I_CLASS`, FFT helpers | `circular_convolution`, `circular_correlation`, `phase_direction` | `operators.npz`, independently enumerated complex128 dense covariance |
| Cumulative phase iterations | `ICLASS.I_CLASS` loop | `solve` | `updates.npz` first three original CUDA phases, full 250-step output |
| Relative MTF, S16-S17 | `ICLASS` row power | `estimate_mtf` | Original MTF and independent float64 Judge row powers |
| Physical 2D spatial inverse | Original `RtoO` discrepancy | `spatial_frames` | Known 2D image round trip, independent axes |
| Per-frame regularized amplitude compensation, S18/Fig S1 | Legacy visualization discrepancy | `rms_object` | Independent NumPy finite case and original-phase corrected oracle |
| Evaluation conventions | Not taken from paper metric reports | Standalone Judge | Known metrics, malformed inputs, pass/fail/error CLI, wrong-cwd tests |

`test/fixtures/operators.npz` is consumed by `test_mathematics.py` and freezes
original complex128 operator outputs. `preprocessing.npz` and `updates.npz`
are consumed by `test_upstream_and_pipeline.py`, freezing original real-data
stages. They were captured from original operations, not generated by local
cleaned functions. Tolerances (operators 1e-11 scale, preprocessing rtol2e-6 /
absolute0.02 DN-FFT scale, phases relative L2 2e-4, final arrays rtol2e-4/atol2e-5)
were fixed for stage parity before the final passing run. Inspect the test files
for each exact inequality and its scale. No test imports the oracle checkout.

`notebooks/reference_outputs/upstream_legacy.npz` is the selected original
legacy output plus independent Run0 measurement. `upstream_paper_corrected.npz`
contains independently corrected source-phase output and named widefield.
`cleaned_measurements.npz` contains the active notebook's fresh image/phase/MTF,
physical phase-only and raw fluctuation RMS. The notebook consumes the first two;
the last is an identified numerical record of displayed fresh measurements.
`evaluation/baseline_reference.npz` is the corrected oracle image alone, consumed
by standalone Judge. These are different provenance records, not relabeled truth.

## Environments and execution

Validated runtime: Windows 10 build 26200; Python 3.11.11 MSC1943 AMD64;
NumPy 2.4.6; SciPy 1.17.1; Torch 2.9.1+cu128; Matplotlib 3.11.1.
Development: pytest 9.1.1, nbconvert 7.17.1, nbformat 5.11.1. Runtime subset is
the exact shared research environment, without domain reconstruction packages.
CUDA12.8; NVIDIA RTX4080 SUPER, 17170956288 bytes total GPU memory; driver610.88.
The isolated oracle venv links the same numerical backend through a `.pth`
because Windows venv system-site-packages did not inherit the parent venv.
HDF5 reader is used only in private camera conversion/capture; pypdf6.21.1 only
in private supplement review. Neither is required or imported by cleaned runtime.

Commands from task directory, using the explicit research-runtime Python:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -q --basetemp=<private archive>/pytest-complete
python -m nbconvert --to notebook --execute notebooks/incoherent_scattering_iclass.ipynb --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=imaging-101-research --output reviewed --output-dir <private archive>
```

`JUPYTER_PATH` explicitly selects research-runtime/share/jupyter. Actual kernel
`sys.executable` is printed and verified as research-runtime/Scripts/python.exe;
the system-default Anaconda kernel is not used. Jupyter emitted Windows Proactor
selector-thread and local TCP transport warnings; no cell error or reconstruction
warning. Timeout900s is above measured full CPU reconstruction172.48s and kernel
startup, not an assumption that every research task should finish in seconds.
Final subprocess exit codes and wall times are in `execution_final.json`, with
raw combined logs `cli_final.log`, `judge_final.log`, `nbconvert_final.log`.
All notebook cells were inspected for exceptions and all four scientific figures
were extracted privately. The executed notebook is the shipped primary review;
there is no temporary executed notebook in output/.

Full CPU run: `python -m src.main --device cpu --output <archive>/cpu_reconstruction.npz`.
Runtime172.48182440001983s, final J7.270195960998535. Its independent Judge passes:
NCC0.9999999999441556, range-NRMSE1.3148335831230727e-6,
image consistency3.148097801508709e-7, MTF consistency5.8359521052328144e-8,
J7.270196034011288. `cpu_run.log` and `cpu_judge_report.json` retain evidence.

Final pytest: **32 passed in34.94s**, zero failed, `pytest_final.log`.
Final active CUDA notebook: independent Judge NCC approximately1,
range-NRMSE7.957373654958354e-9, image consistency3.074701668136098e-7,
MTF consistency3.655017697047144e-8, J7.270196452047658.
The raw machine-readable output retains precision; presentation rounding is not
used in acceptance. Minor NCC roundoff above one was corrected by float64 norms
and clipping only final numerical roundoff to [-1,1]; known metric tests pass.

## Metric rationale and scientific review

NCC is flattened cosine similarity without mean subtraction; NRMSE is RMS error
divided by reference dynamic range. The sole image alignment is maximal integer
circular cross-correlation, justified by blind object/PSF translation ambiguity.
The search covers all400x400 shifts; ties use the first row-major maximum,
equivalent to smallest nonnegative row shift followed by column shift. This
matches NumPy argmax/unravel_index and the complete public output contract.
No crop, intensity fit, magnitude coercion, reflection or rotation is hidden.
All three outputs are required. Every declared boundary must pass. Invalid input
or unavailable assets yields empty metrics/error/exit2; poor finite output yields
fail/exit1; passing yields exit0. Cross-working-directory and report-write errors
are covered. JSON never emits NaN/Infinity.

The reference is the standard method, so self-error zero cannot set a zero
tolerance. Independent CPU/GPU reruns measure image NRMSE up to1.32e-6; fixed
NRMSE1e-3 and NCC.999 allow numerical reproducibility without admitting visibly
different reconstructions. Image consistency2e-4 and MTF2e-5 exceed observed
float32-vs-float64 errors while tying submission quantities to public data.
Covariance J boundary6.906686629445275 is 95% of independently measured oracle
7.270196452047658; uncorrected phase J6.572207 fails. This is a covariance model
criterion, not a complete detector likelihood or proof of physical truth accuracy.

Independent numerical rerun envelope: CPU versus source-corrected reference has
NCC0.9999999999441556 and dynamic-range NRMSE1.3148335831230727e-6; fresh CUDA
has NCC0.9999999999999961 and dynamic-range NRMSE7.957373654958354e-9.
Thus the NRMSE boundary1e-3 is about760 times the largest observed discrepancy;
NCC's allowed deficit1e-3 is about1.8e7 times the measured CPU deficit. The
source-corrected reference is a deterministically reproduced standard method,
so these intentionally conservative tolerances support backend differences,
not meaningful alternative-method accuracy. The tighter independent formation
checks further constrain submissions: image L2 observed3.0747e-7(CUDA) to
3.1481e-7(CPU), boundary2e-4 about635 times maximum; MTF L2 observed3.6550e-8
to5.8360e-8, boundary2e-5 about343 times maximum. Earlier source capture with a
different roundoff ordering measured MTF1.0295e-6; even that is19 times below
the fixed boundary. We do not claim these are physical-accuracy confidence
intervals; the covariance threshold additionally rejects the uncorrected phase.

Independent-widefield comparisons use explicit circular shift and individual
peak normalization, separately from Judge metrics. Fresh raw fluctuation RMS
NCC0.43988508, physical phase-only0.47469397, S18 corrected0.64444057;
legacy amplitude visualization0.57661702. Corrected cross-acquisition range-NRMSE
is0.06264631, legacy0.04783397: metrics do not uniformly improve, and neither is
truth accuracy. Corrected percentile contrast(p99.5-p50)/(p99.5+p50)=0.88318393
versus widefield0.94837170. Hann itself suppresses the background, so raw contrast
0.93501095 is not evidence of successful spatial recovery. Covariance gain is
1.10620343. The 2x3 comparison visibly recovers the main bead chain but retains
haze and loses weak/edge beads. Phase, MTF and inferred absolute PSF panels expose
weak optical bands and the broad effective response. PSF is inferred, not a
separately measured calibration. No display background subtraction or invented
missing method improves these comparisons.

## Semantic review and boundary audit

Reviewed source signatures against `plan/design.md`; model, Fourier normalization,
matrix row mapping, conjugation/sign/roll, Hann multiplication order, all-axis
source FFT shift, inverse spatial shift, units, peak normalization, sigma and
250-step stopping against `plan/approach.md`. The physical inverse correction and
S18 operation ordering are explicit deviations, independently checked rather
than silently claimed source parity. README contains every visible key, batch
shape/dtype/unit and required output; its acquisition constants match source and
fixtures. Small scalar acquisition metadata is in README, no metadata JSON.

Symbol/call-role review of source, Judge, tests and notebook finds only general
NumPy/SciPy/Torch/Matplotlib primitives, local inverse updates, FFT and linear
algebra; no domain solver, upstream import, cross-task import or hidden solver
input. Judge is independent of src. Library imports do not call matplotlib.use.
Public README/requirements contain no GitHub domain or hidden target path.
Only observations are solver-visible NPZ inputs. Plans/tests/notebook/evaluation
are maintainer-only. No synthetic generation module is needed for reused real
observations. Data provenance, historical/reproduced/fresh results are distinct.

## Coordinator independent review

The coordinator independently executed the standalone Judge from the private
coordinator directory, using absolute prediction and report paths. It exited 0:
NCC 0.9999999999999961, dynamic-range NRMSE 7.957373654958354e-9,
image consistency 3.074701668136098e-7, MTF consistency
3.655017697047144e-8 and covariance coherence 7.270196452047658.
The private report is `coordinator/iclass-root-independent-judge.json`.

The structural audit found no errors in the required inventory, visible NPZ
contract, six notebook sections, executed cells, independent evaluator boundary
or imported dependencies. All four embedded notebook figures were inspected:
the actual camera observations, six-way legacy/corrected/widefield comparison,
paired comparison metrics, and phase/MTF/inferred-PSF diagnostics. The central
bead-chain recovery and remaining haze/edge losses are visible; no result is
substituted for a fresh computation. Source, plans and metric conventions were
reviewed, including the flattened cyclic approximation and both output fixes.
Documentation wording was clarified for transfer-map frequency axes and the
loader's backing allocation; executable mathematics was unchanged.

## Completion gates

| Required gate | Status | Evidence |
|---|---|---|
| Paper, verified revision, hashes, equation/function/test mapping | passed | Above and hash inventory |
| Original and cleaned methods on exact observations | passed with disclosed output correction | Original phase kernel/legacy output archive; fresh CLI/notebook |
| Independent upstream parity and physics | passed |32 tests; dense covariance and finite physical cases |
| Constants, units, precision, coordinates | passed | Actual-array/Hann/packing fixtures and semantic review |
| Standard CLI and fresh Judge | passed | execution_final.json and report |
| Judge poor/malformed/evaluator failure behavior | passed | Unit and CLI contracts |
| Public input/output and no hidden-data leakage | passed | Boundary audit |
| README metadata and complete NPZ contract | passed | README/source review |
| Minimal clean runtime; CPU/GPU | passed | Explicit environment and CPU/GPU runs |
| Core local executable mathematics and symbol audit | passed | src and independent tests |
| All unit/parity/physics/pipeline tests | passed |32 passed |
| Active six-part notebook and comparisons | passed | nbconvert exit0, four inspected figures |
| Result labels, definitions, thresholds | passed | Distinct references and metrics above |
| Plans match shipped method/signatures | passed | Semantic review above |
| Final template and preservation of code notices | passed | MIT text below, no redundant wrappers/PDF/archive files |

Separate publication limitation: hosted experimental-data redistribution terms
are **unverified**. The author's public download is available, but no explicit
data license was found. This is independent of the passed, authorized local
cleaning/execution gates. It prevents a claim of unrestricted public-release
permission and is not bypassed by interpreting MIT code terms as data terms.

## Preserved upstream code license

The following original notice applies to extracted/adapted source mathematics.
It is embedded here instead of a standalone LICENSE/NOTICE file, as requested.

```text
MIT License

Copyright (c) 2023 Ori Katz

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

## SHA256 inventory

Private paths below are relative to the archive; `task:` paths are task-relative.
No file in the final task needs the private archive during execution.

| Asset | SHA256 |
|---|---|
| `fig3def_Run1.mat` | `d189e8bd00d0238ed9deeae12a39efa5626137e75e575f81bfa0245a3191911e` |
| `fig3def_Run0.mat` | `f01bb7aa1861f654fb453a87fde2b5b7b28a9ae985c6b8b14c5a3a3f1cfc1631` |
| `fig3def_readme.txt` | `3863a87db4edf3f29584075978ffb9e065a364f922a8dbc4f94cc72f5877e776` |
| `paper.xml` | `c78b61516891bcae1c642440f74ea83c2d1bf9c85b72d696a4fc82d9e3afbac6` |
| `paper_supplement/sciadv.adl5218_sm.pdf` | `bcbd4b4503d7a8f888d24253dac5ce882d83839329df0fbce2025b5d398e6546` |
| `iclass_revision.json` | `81b3c997c230e0e6bd35ddd69503c0cbcc44a4b613fc4896bef93f5e55f6de4b` |
| `capture_fullfov_hann.py` | `d8a7dad2208e2b973fdca35d0a94fed0f4827e45c9d309a06b58142e627c3157` |
| `oracle_fullfov_hann/result.npz` | `27d108dc53ef790c71a0d1ffcd65e84d6514b6bcd6bff027c872d43fa19045ef` |
| `frozen/paper_corrected.npz` | `47ac21d7662532d8a2362f6e4bab579bf0aaa09808e1cea51b935dc978b8ac6f` |
| `upstream/Fluorescence-Computational-Imaging-Through-Scattering-Layers-df2d69a0630905c4d6df12bb643e5354f26120e7/ICLASS.py` | `0f6c6e39ca0c00109e87845a07d7128eab94f882a231d39e985fc698f7ef1f6a` |
| `upstream/Fluorescence-Computational-Imaging-Through-Scattering-Layers-df2d69a0630905c4d6df12bb643e5354f26120e7/main.py` | `81164fc93d50da3bc4635faf78b7d9e1a579224e4a701c5cf237892150e1ee37` |
| `upstream/Fluorescence-Computational-Imaging-Through-Scattering-Layers-df2d69a0630905c4d6df12bb643e5354f26120e7/visualize.py` | `38ff068706bf9b106bbfaddb8899c614924cfd447d120547718fbe4ffc173492` |
| `task:data/raw_data.npz` | `93d2eb83c5314c6537edad00c2d2c24261eabbdf78b044d46789f7dafc63c448` |
| `task:evaluation/baseline_reference.npz` | `378530db4c0ce5bd0b714777aaa20764647de0f0bde710e257cd3fc2205df335` |
| `task:test/fixtures/operators.npz` | `a9049a549cd7cc925c654c9658272e414ac1f830c96eb558b35326ba26b9b2a5` |
| `task:test/fixtures/preprocessing.npz` | `989999430e0ef0323f9787bbb273f860c34e40227ea34c72068e98881d77615b` |
| `task:test/fixtures/updates.npz` | `111aa090eb1029077a6d3d68ac7a063f6708a1ccf2306ad0bf3b52bb055f6bd3` |
| `task:notebooks/reference_outputs/cleaned_measurements.npz` | `435fc1600aa175bcd4fc15579da02ac1265f82839c0a4d5a05c2d69548b762ba` |
| `task:notebooks/reference_outputs/upstream_legacy.npz` | `27d108dc53ef790c71a0d1ffcd65e84d6514b6bcd6bff027c872d43fa19045ef` |
| `task:notebooks/reference_outputs/upstream_paper_corrected.npz` | `47ac21d7662532d8a2362f6e4bab579bf0aaa09808e1cea51b935dc978b8ac6f` |


## Current benchmark v2 evaluation, 2026-10-29

The v2 Judge retains independently evaluated image/MTF consistency and the already frozen covariance coherence threshold6.906686629445275, replacing reference-image similarity. No ground truth exists. A distinct smooth-phase candidate with its freshly computed consistent image passes; a random-phase self-consistent image fails covariance quality.31 unit tests passed. Phase-update250 iterations no longer define acceptance. See iclass_pytest.log.

This intentionally changes benchmark acceptance semantics while preserving the validated published reference algorithm and historical parity evidence. No observations, hidden reference NPZ files or stored reconstruction bytes changed. No full fresh blind Agent solve is claimed. Old public/evaluation files are archived privately under benchmark_readiness_20261029 before changes.


## Final public collection-link correction (2026-10-29)

The author GDrive collection URL https://drive.google.com/drive/folders/18A_W_JemctYMtloonCAh3RTWsZfOJyJN is moved from solver-visible README to this maintainer provenance. Existing selected-data records identify fig3def_Run1.mat as the scattering observation, but the same collection also provides fig3def_Run0.mat as an independent no-diffuser widefield target reference. Run0 is a privileged extra observation for this challenge, not public instrument calibration. Retain acquisition attribution, the normal scientific paper and existing rights limitations without providing this target-reference download entry point. Raw Run1 arrays, objective/Judge/thresholds and output are unchanged; no unrelated inverse or notebook was rerun.
