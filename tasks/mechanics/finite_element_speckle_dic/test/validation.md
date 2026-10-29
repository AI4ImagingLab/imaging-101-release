# Provenance and validation

## Bound task and sources

This is the Q4 finite-element modified-Newton inverse displacement problem on two
repository-supplied speckle images. It is a new solid-deformation task, separate
from fluid PIV. There is no new synthetic image generation, no training, and no
independent measured displacement truth.

- Source repository: https://github.com/PolymerGuy/muDIC
- Exact immutable source commit: `c2c54018ed8c9aba2eaf3a82bc79c2cb56878d01`.
  Coordinator actually verified remote HEAD and shallow-cloned this revision on
  2026-10-18; the selected checkout's `git rev-parse HEAD` independently agrees.
  An additional worker `git ls-remote` attempt later failed with connection reset;
  that failed attempt is not represented as a new remote freshness verification.
- Paper: S. N. Olufsen, M. E. Andersen, E. Fagerholt, *muDIC: An open-source toolkit
  for digital image correlation*, SoftwareX 11 (2020) 100391,
  DOI https://doi.org/10.1016/j.softx.2019.100391. Selected paper is the actual
  published version, downloaded and text-read in full. Paper code metadata names
  historical version 0.1.1; this benchmark binds the exact later source revision
  above and does not claim to use that historical release verbatim.
- Publisher requests returned 403/429. The successful institutional mirror is
  NVA/NTNU publication
  https://api.nva.unit.no/publication/0198cc43b845-1e17522b-ea64-459d-bd46-0b38cc97d14c
  with published PDF file `a7d36597-1aef-48d6-9fd8-b55efdb99111`.
  Exact stable download-discovery URL:
  https://api.nva.unit.no/publication/0198cc43b845-1e17522b-ea64-459d-bd46-0b38cc97d14c/filelink/a7d36597-1aef-48d6-9fd8-b55efdb99111
  returned a presigned S3 URL, whose bytes began `%PDF` and totalled 1265902.
  Paper SHA256 `e44847165b4d4a1511c6e54d8bd1e5e0e511ed82cfe77fca3a703a2da235dac1`. CC-BY-4.0; PDF remains private.
- Bound files: `Examples/example_data/ut_151.tif`, SHA256
  `bab64317505b2032a4d277f693456a65cd9c1d2a248a803cca3b8dcd8e3548b8`, and
  `ut_156.tif`, SHA256
  `2f1d431969efc69e492596f0e3d2d8c937f3e149ccf96882ac14b73a74a1b10a`.
  Each is (2050,871) uint8. Upstream `ImageReader.__call__` reverses rows; following
  it, each original is converted by `array[::-1][350:770,290:550].astype(float64)`.
  No blur, resampling, renormalization, or synthetic deformation is introduced.
  Crop origin/mesh selection are acquisition-independent coordinate choices.
- The paper Sec.3.1 describes Docol600L dog-bone tensile experiments, and upstream
  quick_start reads this example directory. However, file-level metadata do not
  independently prove that these exact UT files have that specimen identity.
  Accordingly public materials say repository-supplied speckle observations and
  do not assert specimen material, frame interval or measured strain accuracy.
- Code and example images are distributed under the source tree's MIT license;
  no separate data license/notice was found. Original copyright and full MIT
  notice are retained verbatim in task the consolidated licensing section of this validation record. No claim of a new data license.

## Archive and reproducibility

Private archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task09_solid_dic/`.
Complete unchanged shallow upstream checkout is archived separately at
`../coordinator/mudic-discovery/`. `archive-index.json` records SHA256 for the
paper, metadata/NTNU search, capture script, oracle arrays/logs, tolerance snapshot,
validation logs and relevant unchanged source files. Original PDFs, screenshots,
logs, source checkout, download metadata and capture scripts are outside the task.
No source clone or upstream capture helper is imported by a cleaned module.

Original CPU environment is isolated in `oracle-env/`: Python3.11.11, NumPy1.26.4,
SciPy1.11.4, Matplotlib3.8.4, numba0.59.1, Pillow10.3.0, imageio2.34.0,
scikit-image0.22.0, natsort8.4.0 and dill0.3.8. Full freeze is
`oracle-environment.txt`. NumPy's numerical configuration is in oracle.log
(OpenBLAS/LAPACK on Windows); one thread was requested through OPENBLAS_NUM_THREADS.
No stochastic process or seed is involved. OS Windows10 build26200, CPU execution.
No GPU is required. Optional upstream Perlin dependency `noise` is absent; it prints
an informational message but is never used. NumPy warns that optional pyyaml would
format show_config output; this does not affect reconstruction.

Original command from repository root:
`oracle-env/Scripts/python.exe capture.py` (both paths relative to private archive,
using its absolute capture path from repository cwd). Capture loads the unchanged
muDIC package from the archived checkout and runs original `generate_reference_Q4`
and `correlate_img_to_ref_q4` on the exact images stored in raw_data.npz. This is
the entire selected two-frame reconstruction path, not an import-only check.
It constructs the unchanged original mesher/DICInput and freezes N,B,I0,H inverse,
one update and final nodal displacement. Exit0, converged in10 updates; original
first double run plus exact repeat and reference construction took about0.158s.
The complete repeated capture with tighter stopping and configuration printing
was about0.458s. Independent same-setting repeat maximum difference0 pixels;
changing only stop tolerance from1e-8 to1e-10 changes nodes by at most
5.12358155901893e-10 pixels. Logs preserve the informational first-step
nonconvergence, which is intentional in the one-update capture, not a failed full run.

Before extracting local equations, `predeclared-tolerances.json` froze stage
rtol1e-10/atol1e-9, final displacement atol1e-8 and Judge pixel-RMSE boundary1e-5.
These nonzero numerical allowances are conservative compared with independent
rerun stability and far below pixel sampling. No thresholds were enlarged to
accommodate a poor reconstruction. The optional tighter oracle run later
substantiated numerical stability instead of changing boundaries.

Cleaned environment is the project-level shared environment, Python3.11.11,
NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1.
Task requirements list exactly this runtime subset. pytest/nbconvert are development
packages only. Domain libraries are absent from the shared cleaned environment.
No domain-library symbols are imported by solver, Judge, tests or notebook.

## Frozen arrays and hashes

- data/raw_data.npz: SHA256 `c33576498e25f36257a2160b3f5af3b3de3f1a9885e4688dd654f41344ffa0b9`.
- test/fixtures/upstream_stages.npz: SHA256 `4ab3d03df1755df802bfecf66266cbfc51498d1c28c803e79f0eb2825f4119d9`.
  Consumer: comparing_test/test_upstream.py. Keys output_N0..7, output_B0..7,
  output_I0..7, output_inverse_hessian, output_first_nodes, output_displacement
  were captured from unchanged upstream execution, never from cleaned functions.
  Unused image/node/connectivity copies were removed from this frozen fixture.
- evaluation/baseline_reference.npz: SHA256 `737500e60c59718d35b917690101d9e2ffc9ed945a805a8f277cd3684964ee98`;
  key reconstruction (1,15,2) float64 contains original-method displacement,
  not truth. No ground_truth.npz is created.
- notebooks/reference_outputs/upstream_measurements.npz: original-method
  reconstruction, read for paired scientific comparisons in the notebook.
- notebooks/reference_outputs/cleaned_measurements.npz: fresh notebook-generated
  reconstruction and iteration history; a separately identified current measurement,
  not loaded as a substitute for active reconstruction.
- output/reconstruction.npz: generated fresh solver submission; output/judge_report.json
  generated by standalone Judge. Machine-readable measurements retain full precision.

Public raw arrays are exact grayscale counts, reference mesh coordinates and node
indices. Connectivity is losslessly stored as int64 (original Windows NumPy1.26
mesh uses int32), with identical values. No target-derived calibration, privileged
image, displacement answer, target object/seed or hidden reference is public.
Raw array content SHA256 hashes (C-order bytes): `{"reference_image": "1f3dbd08a40504413a27c225c798d9aabf6b17063e810b6579c7c96df2d39e00", "deformed_image": "5230671295a5fd1c8e0d15d6d7ffd8d9d622fc33623e1b6d198bbb85308ded6f", "nodes": "1020de9992fd405f6440b62d8451c2034a14c3c31715f2597d5398a8f67f8891", "elements": "3b27e4977c828e870f519ab952400f189cbe21cf8bd07e3ae4a68af04b991cb0"}`.

## Equation/stage traceability

| Paper/process | Original source/function | Local implementation | Validation |
|---|---|---|---|
| Eq1-2 basis discretization, Q4 subset explicitly p2 | elements/q4.py Q4.Nn; mesh/meshUtilities.make_grid_Q4 | shape_functions; element_pixels; public known mesh | shape nodal identity/partition; all8 N fixtures |
| Eq3 brightness transport | solver/correlate.correlate_img_to_ref_q4 | sample_element; filtered_frames | original one-update and full-pipeline parity; photometric residual |
| Eq4-5 unnormalized SSD | reference_q4.generate_reference_Q4 and correlate loop | prepare_reference; solvers.reconstruct residual/history | I0/B fixtures; first update; independent analytic gradient |
| Eq6-8 fixed-reference modified Newton | reference_q4.generate_reference_Q4; correlate_img_to_ref_q4 | prepare_reference; reconstruct assembly/inverse/update | all8 B blocks/H inverse, original first_nodes/final displacement |
| Eq9-10 geometric F and right Cauchy-Green, derived Green strain | post/viz.py concept; not copied as hidden truth | green_strain direct geometric derivative | independent affine F and rigid rotation E=0 |
| Paper Fig1 flow | Examples/quick_start.py; DICInput/Q4 correlator | preprocessing -> main -> reconstruct -> NPZ | CLI; foreign-cwd Judge; active notebook |

The paper states cubic/quintic interpolation but repository quick_start sets order4.
This benchmark preserves that explicit source setting. Q4 class docstring says
"Quadratic" although its mathematical shape functions are bilinear; local names and
plans use the actual bilinear formula, verified against the original numerical values.
Paper Eq9's printed reference-Jacobian matrix has ambiguous/repeated entries;
review strain uses the independently derived F=(dx/dparent)(dX/dparent)^-1 with
rigid-motion and affine checks. This diagnostic does not change the inverse solver
or claim parity with upstream postprocessing tensor conventions. Paper Eq11 Hencky
strain and the full-series paper figure are outside the selected two-frame task;
notebook explicitly identifies its new smaller scope and derived Green strain.
No difficult comparison was removed from a supplied notebook: this is a new review
notebook with all selected bound-pair comparisons shown.

## Results, metrics and executed review

The reproduced original and fresh local displacement agree with maximum absolute
difference0 on this platform, while stage tolerances remain finite and predeclared.
NCC1; dynamic-range NRMSE0; displacement RMSE0 pixels; Judge pass with1e-5-pixel
boundary. No claim of physical displacement accuracy follows from self/reference
agreement. The independent geometric tests validate kinematics and gradient signs.
Measured brightness RMSE drops from18.65430566789413 to5.3060448354762935 digital
counts, with all queried coordinates remaining inside the interpolator's valid
interior. This is observation consistency, not a calibrated noise likelihood.
Fresh derived Eyy spans0.004811322938404361 to0.0063014209146108335.

Commands executed from task cwd with shared Python:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v -p no:cacheprovider
python -m jupyter nbconvert --to notebook --execute notebooks/finite_element_speckle_dic.ipynb --ExecutePreprocessor.timeout=180 --output finite_element_speckle_dic_executed.ipynb --output-dir <private archive>
```

All return code0 in subprocess-captured verification. Final pytest result24 passed.
Full command/log evidence is private cli.log, judge.log, pytest.log, nbconvert.log
and verification-runs.json. The default Windows pytest temporary directory had
an ACL access failure during an earlier run (7 passed/11 setup errors); rerunning
with TEMP/TMP set to the writable private tmp directory resolved the environment
issue without changing algorithms or acceptance criteria. Source compiles/imports
and runtime require no upstream domain packages.

The notebook actively reconstructs, saves/evaluates the fresh result, loads upstream
measurements only for comparison, checks independent metric formulas and plots
both observed images, enlarged displacement arrows, paired x/y original/fresh
measurements, derived strain and SSD/increment convergence. nbconvert timeout180s
is generous for the observed subsecond reconstruction and several-second kernel
startup. Typical fresh solver runtime0.0383s,10 updates. Windows ZMQ emits its known
Proactor-selector warning and the local kernel emits a TCP transport warning;
execution finishes without cell errors. Both rendered figure PNGs were inspected
privately; labels/legends are readable, arrows fit the axes and both comparisons
are present. The executed current notebook is shipped under the required name;
temporary execution copies/screenshots are archived externally, not in output/.

## Gate checklist and semantic audit

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/hash/source mapping | passed | Published PDF read; exact checkout; hashes/table above |
| Original and cleaned exact-observation runs | passed | Independent original capture + fresh CLI/notebook |
| Original fixtures and physics checks | passed | Stage/first/final parity; affine/rotation/finite-difference tests |
| Constants/units/precision/coordinate conventions | passed | README test; exact integer counts; x right/y up; Q4 basis |
| CLI output and independent Judge | passed | Both CLI commands exit0; fresh output passes |
| Invalid/poor/evaluator contracts | passed | Missing/key/shape/dtype/nonfinite/complex/object/corrupt/overflow, poor zero, float32 success, missing assets, report-write error, foreign cwd |
| Public input sufficiency/no leaks | passed | Complete README/metadata/output, raw measurements+known mesh; no public source URLs or answer assets |
| Minimal cleaned dependencies/local math | passed | NumPy/SciPy operators and explicit objective/assembly/update; upstream only private oracle |
| Full test suite | passed |24 passed, pytest.log |
| Active complete review notebook | passed | nbconvert exit0, all code executed,2 figure outputs visually inspected |
| Metric/reference identity and thresholds | passed | Actual truth unavailable; method baseline clearly labeled; zero-error tolerance justified |
| Plans match shipped mathematics/interfaces | passed | Manual review of factors/signs, units, shapes, update ordering, stopping, interpolation/crop boundaries and signatures |
| Final inventory/license/private archives | passed | Template paths only; MIT preserved; private paper/source/logs indexed |

Maintainer review confirms the reference-Jacobian linearization is approximate as
intended, the residual/update sign is positive B*(I0-I1), global DOFs interleave
x/y while element B rows use xxxx/yyyy, and edge counting/frame slicing match
source. Every public acquisition constant is checked against arrays or explicit
local defaults; solver hyperparameters remain maintainer-only. No unresolved
mandatory gate remains; file-level specimen identity remains honestly unavailable
and is not required to formulate this pixel-displacement inverse problem.


## Independent coordinator review (2026-10-18)

The coordinator separately reran `python -m src.main`, the standalone Judge from a foreign working directory, all 24 pytest cases, and the full active notebook using nbconvert in the shared CPU environment. All commands exited 0. Both rendered figures were inspected: upstream/local node comparisons, derived strain, photometric loss and convergence are consistent with the implemented mathematics. Source, README and both plans were reviewed for shape-function ordering, gradient/update signs, shared-edge weighting, interpolation boundaries, image orientation, units and the displacement-only output contract. Evidence is archived under `batch10/coordinator/finite_element_speckle_dic/` and `batch10/coordinator/finite_element_speckle_dic-review.json`. Packaged reference snapshots were preserved around notebook execution.


## Consolidated original licenses and attribution

On the user's documentation-layout request, standalone license/notice files were removed after their exact original bytes were archived privately. The complete original texts, including every copyright, condition and disclaimer, are reproduced verbatim below. This changes the notice location, not the governing license.

### Original LICENSE (verbatim)

```text
MIT License



Copyright (c) 2019 PolymerGuy



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


## Documentation revision: six-part active review and consolidated notices

The user requested the exact six ordered notebook parts Problem, Data, Physics,
Method, Results, Conclusion and deeper implemented plans, using the EHT task as
a depth reference. This revision read that reference and the actual source,
Judge, tests, public contract and prior provenance before rewriting. Data tables
now enumerate every visible key with its actual batch-first shape, dtype, units
and legitimate observation/geometry role, and the Data cell actually loads them.
Physics is separated from inverse method, with explicit coordinate and boundary
conventions. Method gives the actual initialization, updates, normalization,
parameters and stopping policy. Every original scientific plot, comparison group
and numeric table remains in Results, which actively reconstructs using src.
Conclusion retains task-specific scientific limitations rather than claiming
method agreement proves physical truth.

The approach plan now explains the implemented derivation, operators/transposes
where relevant, units, normalizations, algorithmic signs, inner/outer stopping,
evaluation and independent validation. The design plan maps every shipped
callable to its exact signature and actual input/output array shape, dtype, units,
defaults, exceptions and state/file side effects, and distinguishes CLI, review
notebook and independent evaluator flows. No executable algorithm, observation,
reference image or metric boundary was changed. Source copyright docstrings that
pointed at removed notice filenames were redirected only to this record.

Standalone license/notice files were first copied byte-for-byte to the private
revision archive; complete original notices and copyright attribution are retained
in the consolidated section above. Current explanatory references point here;
original filename headings inside verbatim records are historical labels.
The governing license is unchanged.

24 tests passed again.

Private revision evidence: `../imaging-101-maintainer/batch10/revision_docs/task09/finite_element_speckle_dic/` (repository-relative sibling archive). Its `before/` preserves the prior plans, notebook, references and notices; `before-hashes.json` freezes the old package. `verification.json` records exact commands, runtimes, return codes and preserved reference hashes. `notebook.log`, `judge.log` and, where rerun, `pytest.log` are actual command output.

The full selected reconstruction notebook was executed through shared-environment `python -m jupyter nbconvert --execute`, using a writable private TEMP/TMP, shared Scripts on PATH and CPU thread count1. Per-cell timeout900s covers the slow blind inverse as well as faster tasks. Notebook exit code: 0; wall time 3.024401000002399 seconds; independent Judge exit code 0. Complete pytest exit code 0. All code cells ran without errors. The 2 extracted figures were visually inspected for readable axes, labels, scientific comparison groups and numerical consistency. The executed output was saved back to the authoritative notebook path.

Before execution all `notebooks/reference_outputs/` bytes were snapshotted. New measurements were archived separately under `new-reference-measurements/`, then every original reference snapshot was restored byte-for-byte; `reference_hashes_preserved=true`. The shipped notebook and output submission are freshly computed, while restored snapshots retain their original measurement identity and hashes. This is an evidence-preservation procedure, not a cache substitute for reconstruction.

Current package hashes, including changed plans/README/notebook/source notice text, are indexed by private `after-hashes.json`; older source/task inventories remain historical records under their original archive rather than being misrepresented as hashes of this revision. Public raw NPZ, baseline and metric-boundary file SHA256 values were checked unchanged. No temporary executed notebook, figure PNG, source checkout or cache was added to the released task.

Follow-up documentation review added the exact muDIC commit, original filenames/shapes and flip/crop expression to notebook Data, and explicitly defined element gather/scatter assembly and the maximum absolute increment stop in Method. The complete notebook was re-executed successfully (3.1422909 s), all 24 tests and the standalone Judge passed again; original reference bytes were restored. The two figure outputs are unchanged scientifically.

Final DIC wording review also removed the ambiguous local-only H=sum(BB^T) shorthand in favor of the explicit gather/scatter formula. The full active notebook and 24-test suite were rerun once more, with notebook wall time3.0717301s and all command exits0; reference snapshots remain byte-identical. Latest command records are in verification.json.
