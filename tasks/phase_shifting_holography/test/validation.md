# Validation ledger: phase-shifting holography

## Pre-implementation binding and acceptance (2026-10-18)

Bound data: the four 1024 x 1024 Fresnel lens frames from pyDHM's
`data/phase-shifting samples/cut_samples/I0_Fresnel_Lens.bmp` through `I3`.
Repository https://github.com/catrujilla/pyDHM ; remote API verified commit
`e8cbb1e1ce5007c0f2b65d9b7ce2a4962915f8cc` on 2026-10-18.
Paper: Castaneda, Trujillo, Doblas (2022), PLoS ONE 17(10):e0275818,
published version https://doi.org/10.1371/journal.pone.0275818 .
Selected method: SOSR quadrature phase-shifting with digital reference search,
the exact example parameters upper=True, wavelength=632.8e-9 m,
dx=dy=6.9e-6 m, s=1 and steps=4. Unknown is the compensated complex
interferometric cross-term, represented by two real channels, not absolute
refractive index, calibrated electric field, or lens height.

Oracle ran before cleaned implementation. The private archive is
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task02_physics`.
`capture.py` imports the unmodified archived phaseShifting.py and executes SOSR
on precisely the grayscale pixels stored in the public NPZ. No upstream code is
imported by the cleaned task. Its `upstream_run.log` records the executed command,
environment and backend details; `upstream_run.json` records SHA256 values,
runtime and three independent calls. First call: 3.2530316999182105 seconds,
exit 0. A repeated float64 call is bitwise identical. An independent float32
input call gives complex RMS error / reference two-channel dynamic range
6.6583485927655845e-9. This deliberately lower-precision run measures portability,
not accuracy against a known object. No stochastic seed is involved.

Acceptance fixed before implementation: stages float64 rtol=1e-11,
atol=1e-10 (FFT atol=1e-7 because an unnormalized FFT sums 1048576 samples);
selected carrier bins and integer search score exact; full field rtol=1e-10,
atol=1e-9. Physics checks: analytic quadrature cancellation, digital plane-wave
phase slope, norm conservation under unit-modulus carrier and quadrature
identity. Judge NRMSE <= 6.6583485927655845e-8 (10 times the independently
measured float32 discrepancy), NCC >= 0.9999999999. These are nonzero numerical
agreement tolerances for a deterministic reference-method task, not thresholds
on physical ground-truth accuracy. Both channels must be scored at fixed scale
and coordinates; no alignment/cropping/phase adjustment. Bad outputs must fail.

Frozen fixtures `upstream_stages.npz` contain mean-centered first hologram,
its centered unnormalized FFT, raw peak, selected carrier and score captured
using Python tracing at the original function's return. They are consumed by
`test/comparing_test/test_parity.py`. The independent final oracle result is
`evaluation/baseline_reference.npz`, consumed by Judge and pipeline parity;
it is explicitly a method reference, not truth. Upstream measurements in
`notebooks/reference_outputs/upstream_measurements.json` are notebook context.

## Source, experimental status and license

The published paper section 3.2 and Fig. 4 explicitly identify the Fresnel lens
as an experimental specimen and the pi/2 shifts as introduced by a liquid lens.
The frozen example `examples_phaseShitting.py` binds exactly these four files
to SOSR(True,632.8e-9,6.9e-6,6.9e-6,1,4). Repository acknowledgments credit
the 3D Imaging & Display Laboratory and Opto-Digital Processing Group for
experimental holograms. This is **real experimental, repository-distributed,
processed image data**, not newly generated observations and not pristine
raw sensor data. No new generator is needed or shipped.

All four .bmp filenames actually contain JPEG RGB encodings. The adjacent
upstream `cut_samples_matlab.m` describes cropping, mat2gray scaling, uint8
export and JPEG writing even under the .bmp name. Its presence documents an
upstream processing route, but exact earlier uncropped originals and crop
coordinates are not distributed, so we do not claim to reconstruct that entire
acquisition/export history. Current input binding is strictly the released
1024-square pixels. Pillow ImageOps.grayscale follows upstream utilities.imageRead
exactly; no new resizing, crop, downcast of higher precision data or synthetic
noise. The source arrays are uint8; computation promotes them to float64.
Grayscale conversion is part of the original sample's input convention.

The repository root MIT license, copyright (c) 2022 Ana Doblas and Carlos
Alejandro Trujillo (EAFIT), is preserved verbatim in the license section below. No separate
per-image license or conflicting image-specific restriction is present in
the checked source tree; redistribution basis is that repository-level grant
for its distributed software and associated example files. This records the
available licensing evidence rather than inventing a separate experimental
institution permission. Source function author attribution also names Raul
Castaneda. Local adaptations split the selected function into explicit
preprocessing, physics and search stages; they preserve its selected method,
coordinates, constants, sign, sum order and finite candidate grid.

## Equation and algorithm traceability

| Published equation/stage | Frozen upstream file/function | Local implementation | Independent or parity validation |
| --- | --- | --- | --- |
| Eq. (3), tilted plane reference | phaseShifting.py SOSR R0..R3 | physics_model.reference_wave | analytic bin-phase slope, SI cancellation, unit norm |
| Eq. (4), optical interference | physical model in paper; SOSR accepts the four measured frames | physics_model.interference (test helper only) | independent analytic four-frame/DC/twin identity |
| Eq. (5), separated Fourier orders | SOSR fftshift/fft2/fftshift and peak | physics_model.centered_spectrum, solvers.find_carrier | captured original spectrum and exact peak; Parseval; strict-upper check |
| Eq. (9), synthetic quadrature sum | SOSR Isynth | physics_model.quadrature | frozen full-pipeline field and analytic 4O*conj(R) identity |
| Mean/background convention | SOSR first four mean subtractions | preprocessing.center_frames | original centered fixture, zero spatial means |
| Fine carrier estimation | SOSR linspace and binary phase count | solvers.reconstruct, physics_model.phase_score | exact original selected bin and score; all 16 fresh candidates |
| Amplitude and wrapped phase | utilities.amplitude / phase, equivalently abs / angle | visualization.comparison_plot and notebook | executed original vs fresh phase/amplitude/error panels |

Paper Eqs. (1)-(2) describe general imaging and propagation; their unknown
optical train is absorbed into the camera-plane object field here. Eqs. (6)-(8)
and (10)-(12) belong to other methods/propagation configurations. They are not
required stages of this selected four-frame slightly-off-axis SOSR task.
Other paper Fig. 4 specimens/algorithms (blind USAF BPS2/3) require different
observations/phase assumptions; they are explicitly outside the bound task.
The notebook preserves the relevant experimental spectrum and Fresnel lens
phase comparison and adds raw-quadrature, amplitude, error and metric views.

## Investigated discrepancies and independent checks

1. The original upper ROI has a negative slice endpoint: on M=1024 it spans
   rows 1..614, slightly beyond the upper half. Retained literal behavior gives
   peak (column 514,row 468). A mathematically independent strict-upper search
   rows 1..511 gives the same peak on the bound observations (notebook assertion).
   Thus no erroneous lower-order selection occurs in this task; no corrected
   method result is silently substituted.
2. Original log(power) emits divide-by-zero at removed DC and produces -Inf.
   Power itself is monotonic with its log, so the local argmax uses power.
   Captured peak/full field agree. This avoids a representational warning,
   not a change in acquisition or inverse physics.
3. Constant-phase min-max scoring is undefined in the original. Local helper
   returns zero in this degenerate case. The bound 16 candidates have nonzero
   phase span; captured score and field confirm unchanged bound execution.
4. The original printed phrase "Synthetic building" concerns the *digital
   reference wave*, not synthetic observation generation. The four bound
   interferograms are experimental; synthetic noise was never added here.
5. Numerical field agreement is exact for this extracted deterministic path,
   but nonzero Judge tolerance was established from a distinct float32 upstream
   call before local implementation. Reference agreement is not physical truth.

Independent analytic tests establish phase sign, factor of four, conjugate/DC
cancellation, half-array origin even for odd-sized toy arrays, physical-unit
cancellation, FFT Parseval normalization, and unit-reference amplitude
conservation. The bound-data amplitude residual is 1.2797180549466623e-16
relative L2. No target-derived calibration or target seed is supplied.

## Environment, commands and observed checks

Cleaned CPU execution: Windows 10 build 26200, Python 3.11.11, NumPy 2.4.6,
Matplotlib 3.11.1. NumPy's built-in pocketfft performs FFTs; NumPy config records
MSVC 19.44, x86-64, scipy-openblas64 0.3.31.188.0. Pillow is used only by private
source-file conversion; it is not imported by runtime, Judge, tests or notebook.
The isolated oracle venv includes OpenCV-headless 5.0.0.93 because the unchanged
phaseShifting.py imports cv2; SOSR itself does not call it. It also imports
SciPy 1.17.1; its optimizer is not called on this selected method. A private
.pth exposes the already pinned shared numerical packages without modifying
the shared installation. Cleaned requirements list only NumPy and Matplotlib;
domain libraries are absent from the shared cleaned environment.

Private capture command:
`oracle-env/Scripts/python.exe capture.py > upstream_run.log 2>&1` from the
archive (capture.py binds absolute task and archived source locations).
Original run exit 0, warnings: log(0), and optional pyyaml configuration-printer
warning; neither changes finite reconstruction. Full backend output is archived.

Cleaned verification commands, from the task root with the pinned shared Python:

```
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
jupyter nbconvert --to notebook --execute notebooks/phase_shifting_holography.ipynb --output phase_shifting_holography_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180
```

The task root is the CLI working directory; all tests and the notebook use only
cleaned modules. Tests: **27 passed in 4.88 seconds**. They include metric known
cases, bad files/keys/shapes/dtypes/complex/NaN/Inf, poor valid zero prediction,
foreign-cwd CLI, report-write failure, missing/bad evaluator references and
invalid thresholds. Initial pytest attempt used an inaccessible default TEMP
directory (11 passed, 12 setup errors); setting per-task TEMP/TMP resolved it,
and all final checks ran in that configuration. No test threshold was relaxed.
Subsequent added evaluator cases brought total to 27.

Main CLI exit 0; fresh Judge status pass, NCC 1.0, dynamic-range NRMSE 0.0.
The executed notebook contains five active code cells, each with an execution
count, no error output, two rendered figures, a paired metric table and fresh
full-data reconstruction (roughly 1.3 seconds). A timeout of 180 seconds allows
over 100 times measured reconstruction runtime plus kernel/plot overhead.
Notebook logs contain Windows zmq selector and local kernel transport warnings,
with successful nbconvert completion. Extracted figures were viewed and checked:
spectral order/candidate score heatmap; measured/raw phase/fresh phase/upstream
phase/amplitude/error panels. Spectrum contrast was adjusted to robust display
percentiles only; all scientific arrays and metrics remain unmodified.
The primary notebook stores executed output; temporary executed output was
copied to private archive and removed from output/. Private figure PNGs are
review evidence, not packaged duplicate assets.

## Semantic review and completion gates

Source/signature review checked every design.md interface and source default;
equations checked against the frozen original and independent interference
expansion. Camera pitch and wavelength preserve the example's supplied SI
values. Array axes, float64 promotion, half-array origin, FFT signs,
unnormalized transform, strict tie ordering, mean removal, quadrature factor,
search grid and stopping path agree. No opaque inverse library, upstream import,
cross-task import or backend mutation exists in runtime/generator/Judge/tests/
notebook. The analytic interference helper is test-only, not a hidden generator.

| Required gate | State | Evidence |
| --- | --- | --- |
| Paper, verified revision, hashes, equation mapping | passed | Source sections, frozen archive and hash inventory |
| Original and cleaned execution on exact bound observations | passed | capture.py/run log, CLI, original-stage/full parity |
| Independent oracle fixtures and physics checks | passed | two comparing tests and analytic physics unit tests |
| Constants, units, precision, coordinates preserved | passed | README constant test, original stages, SI/half-array tests |
| Standard entry point and fresh Judge pass | passed | output/reconstruction.npz and judge_report.json |
| Judge bad input, poor quality and evaluator failures | passed | unit contract cases and foreign-cwd/report-write tests |
| Public allowlist sufficient | passed | README full intensity/complex output convention and raw frames |
| No truth/calibration/seed/URL leakage | passed | public URL test and raw_data sole observation-key inspection |
| README acquisition metadata/output; no metadata JSON | passed | constants test and final inventory |
| Minimal cleaned environment independent of upstream | passed | shared CPU environment run, no cv2/pyDHM imports |
| Explicit local core and symbol/call-role audit | passed | FFT/reference/quadrature/search source and import review |
| Entire unit/parity/physics/pipeline suite | passed | 27 passed; private tests.log |
| Active reconstruction notebook, executed and viewed | passed | five executed cells, two reviewed figures, execution log |
| Provenance/metrics/threshold separation | passed | oracle float32 experiment and separately labeled fresh JSON |
| README and plans mathematical/interface consistency | passed | semantic/signature review recorded above |
| Final inventory, provenance archive and license | passed | full license record below, hash inventory, no final raw image/PDF/oracle code |

Limitations are explicit: processed experimental camera exports rather than
lossless sensor originals; no independent optical-height truth, no absolute
irradiance calibration, and CPU validation only. These are not unexecuted
mandatory checks. No GPU compatibility or legacy harness integration is claimed.


## Frozen archive and packaged-array SHA256 inventory

Complete upstream zip integrity checked with ZipFile.testzip; every captured source and image byte matches its commit archive member. No original capture source was altered. Private archive index: `archive_index.json` under the archive root given above.

| Private archive relative path | Bytes | SHA256 |
| --- | ---: | --- |
| paper.pdf | 4168871 | `787a0957948219a3469855cf00d9ca9bebb454f97c43e453899564dd643a3f1b` |
| paper.xml | 204864 | `be4ee6af04a11475634cf37ff307ddc571c8b3a6b1b0334529de02ac9e52aaa1` |
| paper.html | 269260 | `d164817f739e4d2db36151f382791e956656fbd0484701cd104d2e54304f44c1` |
| upstream_complete.zip | 34847046 | `d512a3f58227fd90503a62790933dc8ac20af399066f2fb69d27a9c136473d75` |
| capture.py | 2636 | `3c75a1cef6ed59e04735d35829a42ae7af2824b7dd5bed114c66e5469a46e499` |
| upstream_run.log | 8978 | `1ea240c5d96d2a775210853c9d2856ae9b6090d7358ea9801cd9715816a09436` |
| upstream_run.json | 977 | `6e1a34887865caf4179652ea66ab817ff974faa8d595b6cdb657424c80df432f` |
| notebook_execution.log | 2428 | `93cd240ad1b312f957b7a8a32fbf0c95cb2cfe921c798fa66a174f5281b15b0b` |
| tests.log | 5522 | `fcc425288ad0018e8818c6f0f95914332b0ea7063328dbaf5b5c8ec263a93604` |
| build_notebook.py | 8317 | `9e38f5d7420f8f0e8b853da967774c4a8cd2fc6535900719a2329fd3b852891a` |
| executed_notebook.ipynb | 1781069 | `c1640054aa5701bc8ab7d2c51c3c8d8d2480837b0893191bb923f0e3800bde15` |
| figure1.png | 119451 | `5f8b0f0b50fa492f79a5cf2990fcd159d0787b38bd1245b293541d037355709a` |
| figure2.png | 1205502 | `8cbd39bfe8df9f2d3143863b3fe7a5f80c70e8377e4d61759f1ba7d33dbc9d68` |
| upstream/cut_samples_matlab.m | 478 | `ebf9c9268a14d2302742552a42608f2377d302d103cb566d603de6a46b33aada` |
| upstream/data/phase-shifting samples/cut_samples/I0_Fresnel_Lens.bmp | 189766 | `e68cf0d4bb3b24b4528b4b050e0c144af6094608b307fec775952668c06f677b` |
| upstream/data/phase-shifting samples/cut_samples/I1_Fresnel_Lens.bmp | 208788 | `2c3a799b50cba14761da923428d6363a24e19ab4dfe99334824b22918215582e` |
| upstream/data/phase-shifting samples/cut_samples/I2_Fresnel_Lens.bmp | 222421 | `96cff1e2f5f872fa9599d92d6a70cd0340ac3ad37a72227fa572e7e82f2912fd` |
| upstream/data/phase-shifting samples/cut_samples/I3_Fresnel_Lens.bmp | 221101 | `d0db25effd5c91672a15ce140fa92569aeaa855f1c9a80c08e7d40f2978c10e4` |
| upstream/docs/index.md | 24192 | `48afa2efdd5a99cf3a95fcf598ca2c7f1551cb41952111f8090ae3629ac20d71` |
| upstream/examples_phaseShitting.py | 4360 | `0905f5af43e1ec0965be665b3710d3e2952b36b55bb161d237b99c3bdd458b4f` |
| upstream/LICENSE | 1104 | `8a69a24ac4e61790ee28c6c0266c690483ff9a527d4b2d2b4bbc8780e61536eb` |
| upstream/pyDHM/phaseShifting.py | 19374 | `fd954ef583f78492d7b4ac2f8d156ad502c6fc72f720ace70b8e4d7d17994857` |
| upstream/pyDHM/utilities.py | 9631 | `6f390b649ba88337e6e57d28ab1accdd116387c35d497e95f3d0f49494f4315b` |
| upstream/README.md | 3319 | `9b64cda8d49470a320892cb4340951acb5edc08893df380ba8445eb58a79dff4` |

| Packaged numeric file | SHA256 |
| --- | --- |
| data/raw_data.npz | `6f1044c71141990dfa8ac03ed7bf6c5f336725f88f97b5fe6b23f00bc6994ea8` |
| evaluation/baseline_reference.npz | `7f177ec8c17fce04b34c1935f48580847c89f87d998985b35a98d3c6a57657d3` |
| output/reconstruction.npz | `7f177ec8c17fce04b34c1935f48580847c89f87d998985b35a98d3c6a57657d3` |
| test/fixtures/upstream_stages.npz | `d91379153e08b413ec9c11ee1aa9ad9ff0bd97b739a63267717ae1ea9bd41d7e` |

Original release inventory (superseded by this documentation revision): README, requirements, former root license; one public observation NPZ; five local source modules plus __init__; two plans; standalone judge/metrics/method reference; one primary executed notebook and separately labeled upstream/fresh measurement JSON; one stage-fixture NPZ; unit/comparing tests and this single validation report; generated reconstruction and judge report. No standalone sensor image, paper, cache, duplicate executed notebook or live upstream import remains.

## Independent coordinator review

On 2026-10-18 the coordinator independently reran `python -m src.main`, the
standalone Judge from a different working directory using absolute submission
and report paths, all 27 tests, and the active notebook through nbconvert.
All four commands exited 0. The coordinator inspected the generated phase,
amplitude, upstream comparison and error panels and reviewed the local physics
functions and shipped interface plan. Public-data and import-role scans passed.
Logs, the separate executed review notebook, extracted figure and structured
review result are archived under `batch10/coordinator/phase_shifting_holography`
and `batch10/coordinator/phase_shifting_holography-review.json` in the private
maintainer archive. These are additional verification evidence, not new truth.


## Preserved full license and attribution record (documentation revision)

The user requested removal of task-root LICENSE/NOTICE files. Their complete original texts and attributions are retained verbatim below, and original bytes are backed up in the external revision archive. This relocates notices without changing their terms, scientific data rights or source copyright. Historical upstream source paths/hashes remain historical evidence. Current source references point to this section.

### Original LICENSE (verbatim)

Original-byte SHA-256: `8a69a24ac4e61790ee28c6c0266c690483ff9a527d4b2d2b4bbc8780e61536eb`.

```text
MIT License

Copyright (c) 2022 Ana Doblas and Carlos Alejandro Trujillo (EAFIT)

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


## Completed documentation revision and revalidation (2026-10-18)

This revision changes explanatory documents, notebook organization and license
placement. Acquisition arrays, inverse mathematics, evaluation assets, thresholds
and packaged scientific reference snapshots are unchanged. The original documents,
notebooks, code and license bytes are retained under the private archive
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/revision_docs/task10/phase_shifting_holography/before/`; removed root notices are additionally retained in
`removed-root-files/`. Full original license/attribution text and original hashes
are reproduced verbatim in the preceding preserved-license record. References to
the former root notice in source comments/module docstrings now identify this
validation record; executable ASTs match the pre-revision source after stripping
docstrings. Necessary README attribution references were updated accordingly.

The approach plan now defines the actual forward discretization, variables,
normalizations, units, initialization, inverse updates/options, termination,
evaluation and scientific limitations. The design plan records every shipped
runtime/Judge top-level signature, parameter/return contract, caller flow, file
consumer and side effect. Signatures were checked against source, and equations,
units, signs and caller roles were reviewed against the unchanged implementation.

The notebook has exactly six H2 sections: Problem, Data, Physics, Method, Results,
Conclusion, numbered 1 through 6. Data includes concrete original filenames,
repository/archive provenance, crop or processing status, visible array contracts
and acquisition metadata. Physics and Method are self-contained. All original
scientific comparison groups remain present, with upstream, fresh and initial or
alternative-condition results labeled explicitly. The default path actively
reconstructs observations with local modules and evaluates the fresh submission;
historical reference snapshots are never rewritten during review.

Revalidation gates: **passed**. The full `python -m pytest test/ -v
-p no:cacheprovider` run passed **27 tests**. Full `python -m jupyter nbconvert
--to notebook --execute notebooks/phase_shifting_holography.ipynb --output review_executed.ipynb
--output-dir <private-task-archive> --ExecutePreprocessor.timeout=240` completed
with exit 0. Every one of **6 code cells** executed without error. All
**2 exported figures** were opened and visually inspected; comparison
panels, annotations, residuals and metric groups remain readable and scientifically
consistent. The shipped notebook contains these executed outputs. Exact absolute
commands, working directory, exit status and wall times are in `commands.json`;
stdout/stderr are in `pytest.log` and `notebook.log`. Fresh reconstruction runtimes
and metric streams are in `new-notebook-streams.json`; no historical runtime was
relabelled as a new measurement.

Byte-level checks passed for all public observations, evaluation files and every
file under `notebooks/reference_outputs/` against the pre-revision snapshot.
The private `before-hashes.json`, `reference-hashes-before.json` and
`reference-hashes-after.json` record this evidence. Figures are indexed by
`figures.json`; the executed review notebook and PNGs stay private. The private
`final-manifest.json` indexes the entire final task with SHA256 hashes, including
this validation record, and records the immutable-reference comparison. No
temporary executed notebook, screenshot, upstream source or capture script is
shipped in the task. No scientific completion gate is replaced by this
documentation review; the original numerical evidence remains authoritative.

Final reviewed document hashes (the validation record is hashed externally to
avoid a self-referential digest):

| File | SHA256 |
|---|---|
| `plan/approach.md` | `211ef9034d46cdf2881e5e154e4b9c11d9d7265fd343931176cad9b67118ad99` |
| `plan/design.md` | `fc93b46757030ab2423db68d973d52ed9764cbb3725079e28aa75df20cd15dad` |
| `notebooks/phase_shifting_holography.ipynb` | `7dd574bac48a1c79f7eea8293170d64a9ad2fd38490b080605fae4c86c2ec258` |
| `README.md` | `f263e9420c6a92dbcadae3ed3fae57a73ed8754b4658be427c4f8eb4f62defbd` |

### Final notebook Markdown readability review

The final prose review corrected concatenated English words and spacing while
preserving scientific meanings and all fenced mathematical formulas. Repeated
source/crop descriptions in the phase-shifting and Spitzer Data sections were
consolidated into their detailed provenance paragraphs. Every code cell was
compared in full against the successfully executed private
`review_executed.ipynb`: source, execution count, outputs and cell metadata are
identical. This Markdown-only change therefore did not rerun unchanged code or
relabel old computations. `final-markdown-proof.json` records the equality
checks and final notebook SHA256; the document-hash table above and private
`final-manifest.json` are updated to the final bytes.
