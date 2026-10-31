# Provenance and validation of the complete measured-RF vector inverse

## Binding, original method and archive

Private archive `B/ultrasound`, with B the external maintainer-only directory
`research_batch7_20261022/groups/astro_earth_mech`. `archive_index.json` and
`final_evidence.json` identify private paper/source/download/run/figure assets
by SHA256. No source archive, paper PDF, capture script or temporary notebook
is shipped in this task. Source, tests, notebook and Judge do not import any
private archive, other task or ultrasound-domain library.

The bound data are the entire **existing real** measured PWI_disk RF ensemble,
334 fast-time samples by128 elements by32 pulses. This is RF/IQ phase inference
and multi-angle vector recovery, not inversion from already supplied radial
velocities. It differs from structural ultrasound imaging and from the existing
dual-radar wind inverse. The output is the entire public-support two-component
spatial velocity field, not a fitted scalar disk angular rate.

The successful oracle is the **official author MATLAB MUST route**, executed
using Octave and explicitly verified general API adapters. The separately
versioned GitHub Python port failed and is not falsely called a successful
unchanged reproduction. The selected author MATLAB domain files themselves
remain byte-identical to the official ZIP. A later public example acquisition
is bound to the published2018 method family; it is not called the identical
historical paper experiment.

| Source | Exact version or identity | SHA256 / archive |
|---|---|---|
| Main paper | Madiena et al., *Color and Vector Flow Imaging in Parallel Ultrasound With Sub-Nyquist Sampling*, IEEE TUFFC65(2018)795–802, DOI10.1109/TUFFC.2018.2817885 | `paper2018_author.pdf`, full8pages, SHA256 `3b57077a1157732520a1d9906f6f7f416517a2565ab73dce35d30cc74298253d` |
| Author full-paper URL | https://www.biomecardio.com/publis/ieeeuffc18a.pdf | Full text, methods, results, figures/captions, limitations and references read in `paper2018_author.txt` |
| Smoothing paper | Garcia, CSDA54(2010)1167–1178, DOI10.1016/j.csda.2009.09.020 | https://www.biomecardio.com/publis/csda10.pdf, `csda10.pdf`, SHA256 `6f7a9f954188c547351ff2040186da997008f5e240950858c6d755284a2d25c0` |
| Corrigendum | Le Tarnec/Garcia, CSDA56(2012)2182, DOI10.1016/j.csda.2011.12.001 | Included as final page of the same downloaded author PDF; corrected positive-definiteness argument explicitly read |
| Vector robust smoothing | Garcia, Experiments in Fluids50(2011)1247–1259, DOI10.1007/s00348-010-0985-y | https://www.biomecardio.com/publis/expfluids11.pdf, SHA256 `8f0872bf42f2f6721b03d2d978493aef15d89e653693a5b9c8ce6757b79e30e4` |
| Successful official code | https://www.biomecardio.com/MUST/functions/MUST.zip | `MUST.zip`, SHA256 `7b133a369e7edaa3d4c1f7a40eb935f7a40951a32c33f942adefdbe2774fe8f1`; unpacked `matlab_source/` |
| MATLAB exact-file version | rf2iq header2023/04, dasmtx header2024/07/03, iq2doppler2020/06/24, smoothn2023/05/11; full `updates.txt` retained | The ZIP has no asserted Git commit. File hashes and exact35-file byte audit establish the actual selected version |
| Separately archived Python port | https://github.com/creatis-ULTIM/PyMUST | Remotely verified `a96be54f19eb1471fe610221d11f7886b4ddd272`, LGPL2.1; source ZIP SHA256 `eb7f02bfad90af1b1f9e071a20fea37300d9f0d7bee1bb2033e7cfe11a83ec8c` |
| Exact real RF and calibration | Official ZIP `PWI_disk.mat`, identical to port `examples/data/PWI_disk.mat` | Both full files SHA256 `c349dc1d677c561434fd0e4a74142c4a0a44b7e6ae0a42d446c078a528ef58c1`; `data_identity.json` verifies entire byte identity |
| Exact author example recipe | https://www.biomecardio.com/MUST/functions/html/iq2doppler_doc.html, example2 | `iq2doppler_doc.html`, selected full RF→two-view→vector→robust recipe |

All three full papers and the appended corrigendum were read, including their
mathematical derivations, boundary conditions, comparisons and limitations;
source filenames and full-text downloads are separately identified. Initial
guessed paper URL404 and HAL HTML/challenge responses are retained as failures,
not counted as paper access. Windows pdftotext was unavailable; WSL pdftotext
successfully extracted the actual complete downloaded PDFs. No citation-only
paper claim substitutes for this evidence.

Official MUST headers and COPYING.LESSER are **LGPL-3.0-or-later**, distinct
from the port's LGPL2.1. Full LGPL3 and incorporated GPL3 terms and source
copyright are retained in LICENSE/NOTICE. The software license is not assigned
to RF data: the authors offer the real example publicly, but do not state a
separate data redistribution license. This is a disclosed local-research
limitation, not an invented CC/GPL data license. No publication/upload approval
is claimed or required by this local cleaning work.

## Original executions and compatibility qualification

Original domain functions are unmodified. The private wrapper implements the
actual official example2 on the exact RF file. Octave8 with signal/image
packages runs only in the isolated upstream verification route, never in the
cleaned Python environment. CPU numerical threads are one. Exact OS, compiler,
Octave/package/BLAS/FFT and cleaned Python/NumPy/SciPy/backend information is
captured in private environment provenance. Shared runtime packages were not
installed or upgraded for this task.

Three general API adapters are explicit:

1. The time-axis334 by128 by32 RF stack is flattened into independent time
   columns for original rf2iq/filtfilt and reshaped back. Octave filtfilt is
   a2D trace API. An independent all-column versus per-trace case differs0.
2. A private imfilter wrapper calls Octave's original real imfilter separately
   on real and imaginary parts. This preserves linear complex filtering and
   the author's replicated boundary. A known complex edge case versus an
   independent nearest-boundary convolution differs3.552713678800501e-15.
3. `all(A,'all')` is mapped to builtin all(A(:)); other calls use builtin
   unchanged. Independent true/false multidimensional cases are verified.

Adapters live only in `octave_compat/`; all actual domain formulas/functions
remain byte-identical. An earlier attempt to rename/copy imfilter failed on
Octave private helper lookup and is retained under `prior_matlab_attempts/`.
The final wrapper captures the original function handle before adding adapters.
No correction to Doppler, beamforming, filtering coefficients or smoothing
domain equations is hidden in these adapters.

The source's toolbox-free smoothn initial guess is actually selected: its
warning that BWDIST is unavailable remains in stderr. Mean-fill/low-frequency
initialization is explicitly reproduced locally. WSL localhost/proxy,
QStandardPaths permissions, source text UTF-8 replacement and built-in-shadow
warnings are also retained, not presented as native failures or suppressed by
an unrequested shared install. Final outputs are finite.

The Python port failure is independently retained in `original_process.json`
and stderr: its nonzero-RX-angle dasmtx attempts `z.tanRX.flatten()` on an
ndarray and raises AttributeError at line557. Static source comparison also
finds flattening of cosRX yielding inappropriate point-by-point broadcasting,
and symmetric convolution boundaries versus official MATLAB replicate edges.
Those latter observations are labeled static discrepancies, not fabricated
successful patched Python runs. An initial wrapper conversion of the author
NumPy scalar t0 to Python float was also corrected as a harness error before
the genuine original attribute failure. No Python source patch is claimed.

## Actual native process evidence

Each authoritative process is captured using Python subprocess.run with an
argument list and `shell=False`, separate stdout/stderr and actual return code.
`PY` is the shared research Python, `B/ultrasound` the private archive.

| Command | Native result | Evidence |
|---|---|---|
| `wsl env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 octave --no-gui --quiet original_matlab.m` | First successful0,20.72989509999752s | `first_successful_original/matlab_original_process.json`, separate logs and both complete original MAT outputs |
| Same unchanged full original command, independent rerun | 0,19.816053900009138s | `matlab_original_process.json`, `independent_original_repeat.json`; all saved stages bitwise equal |
| General API/numerical compatibility cases | 0,.5273105999949621s | `original_evidence.json`, `compat_cases.mat`, `check_compat.m`, separate logs |
| Original full pipeline with explicitly quantized complex64 intermediate IQ | 0,21.075323500001105s | `precision_capture_process.json`, original domain files unchanged, `source_precision_*.mat` |
| Original first smoothing update, DCT and known-phase numeric cases | 0 | `small_stages_process.json`, `source_small_stages.mat`, separate logs |
| `PY -m src.main` | 0,2.4436172999994596s | `local_cli_process.json`, separate logs and output |
| Complete unit/parity/physics/Judge tests | Passed; final count/runtime in final evidence | `final_metadata_contract_process.json` and stdout/stderr |
| Full active Notebook via native nbconvert | Passed; exact final runtime/hash in final evidence | `notebook_process.json`, `notebook_execution/stdout.log`, `stderr.log` |

First complete unmodified original saved IQ(334,128,32), both focused
IQ(251,251,32), Doppler fields/variance, public mask, both full raw vector
components and both full robust-smoothed components. Original final flag is1;
raw ranges are vx[-1.2299046092422572,.6583682779555096],
vz[-.5756656863987553,.5839792966076987]m/s; smoothed ranges
vx[-.5429294607313557,.5077614342345029],
vz[-.5409572485413638,.5840568499440356]m/s on the entire mask.
No cropped or alternate-demo run establishes this gate.

## Pre-extraction independent freeze and meaningful controls

`freeze_criteria.py` ran before any local inverse existed. Its complete
timestamp, paper/data/official ZIP/fixture hashes, original-repeat evidence,
precision results, chosen limits, physical check and poor controls are in
`frozen_criteria.json`. Source stage fixtures include independent complete
selected RF traces, original complex DAS matrix rows, exact filter coefficients,
both phase/variance maps, full raw/smoothed fields and complex correlations.
`source_small_stages.mat` captures original first weighted smoothing update,
an independent small DCT case and a known constant-phase packet before local
implementation. The instrument fixture comes directly from the exact original
MAT calibration; it is not estimated using velocities.

The unchanged independent repeat gives0 at all saved stages. A nonzero
tolerance is based on the original pipeline rerun after explicitly quantizing
its demodulated complex IQ through complex64 and back to double, and separately
serializing final velocities through float32. The precision diagnostic is not
mislabeled as an identical default acquisition or a new measured experiment.

| Quantity | Complex64-intermediate full-pipeline NRMSE | Frozen maximum NRMSE | Frozen minimum NCC |
|---|---|---|---|
| vx | 2.329847190549264e-10 | 3.1279966911092165e-8 | .9999999999999956 |
| vz | 2.7718173354485154e-11 | 3.449385143851913e-8 | .9999999999999956 |

The NRMSE limits are five times the worst measured source precision or
serialization value. NCC includes a20-epsilon float64 cosine floor. NRMSE is
RMS error divided by component reference dynamic range, separately over all
31404 public support pixels. NCC is uncentred cosine. There is no fitted sign,
rotation, gain, permutation, alignment or private quality mask.

Both component metrics and independent physical consistency must pass.
Predict directional velocities from the submitted vector, convert them to
unit phasors using the physical carrier/lag1 Nyquist velocity, and compare
against unit observed complex correlations. Correlation magnitudes weight
the squared complex chord difference. The original weighted phasor RMS is
.05201278063077122; frozen maximum1.1times=.057214058693848345. Exterior
magnitude limit is1e-12m/s. The Judge derives the correlations from public RF,
not a hidden solver output or privileged derived-observation file.

Pre-extraction zero, uniform, wrong-sign, reversed-column and swapped-component
controls fail. Zero has vx/vz NRMSE.25717944947879295/.26999864752007546 and
phasor RMS1.2166443869093133; wrong sign gives.5143588989575859/
.5399972950401509 and1.5624771736618377. Separate corruption of either
component and nonzero exterior are tested. These criteria are nonvacuous.

Frozen stage tolerances are IQ rtol2e-10/atol1e-8, complex operator
rtol2e-10/atol1e-10, radial phase/variance atol2e-9 and smoothing
atol2e-9. Complete derived correlation has an explicit accumulation tolerance;
the independent all-output Judge is much stricter in final-field agreement.
No threshold was changed to admit a failed cleaned numerical result.

## Equation / source / local executable traceability

| Required physical/algorithm stage | Exact official source | Local mathematics | Validation |
|---|---|---|---|
|2018 narrow-band complex downmix and filtering | `rf2iq.m`, down-mixing, butter(5), filtfilt, factor2 | `physics_model.demodulate` | Full selected source IQ traces; independent filter padding/column cases |
| Array/directivity and receive steering | `dasmtx.m`, fnumber criterion and aperture | `automatic_fnumber`, explicit aperture in `beamformer` | Exact source F=3.365971459145058, selected sparse source rows |
| Selected virtual-array TX reader, receive path and coherent focus | `dasmtx.m`, vxdcr/diff2, parabolic distances, linear interpolation and phase | Explicit nearest-parabola formula, travel times, sparse interpolation in `beamformer`, `beamform` | Independent source matrix rows, full RF-to-field parity, complex adjoint test |
|2018 ensemble phase and signed velocity | `iq2doppler.m`, full-packet autocorrelation and Hamming replicate filtering | `doppler_velocity` | Source phase/variance case, exact known phase and alias-period tests |
| Bistatic two-view vector decomposition | Official iq2doppler example2, same2018 geometry family | `invert_receive_projections`, `predicted_radial` | Independent analytic two-component roundtrip/sign test, complete raw source field |
| CSDA2010 Eq.11,18–20 repeating-boundary DCT penalty | `smoothn.m`, Lambda/Gamma and explicit weighted loop | `laplacian_eigenvalues`, `dct_filter`, weighted updates in `robust_smooth` | Source small DCT and first full-grid update; independent small dense repeating-boundary matrix solve |
| Shared-vector robust residual scale and bisquare passes | `smoothn.m: RobustWeights`, vector extension described in ExpFluids2011 | Explicit marginal medians, vector MAD, leverage/studentization/weights | Full original robust fields, measured per-pass stopping evidence |
| Complete selected RF inverse | Official example2 wrapper and unchanged functions | `solvers.reconstruct`, `main.main` | Fresh complete native CLI, complete pipeline comparing test, standalone independent Judge |
|2012 corrected positive-definiteness condition | Corrigendum appended to `csda10.pdf` | Documented repeating Laplacian constant nullspace and positive observation weights | Dense mathematical quadratic/DCT check; no false nonsingular-L assertion |

## Scientific qualifications and resolved checks

The successful official MATLAB code and Python port do not share a falsely
asserted commit or license. The official version is defined by its downloaded
ZIP and exact file hashes. Its data are entirely byte-identical to the
versioned port's example, so using the official method does not silently
switch observations. Source and data rights remain distinct.

The conditional smoother has a fixed finite protocol. Actual local inner
counts100/43/12, with changes.0022206293909106094/
.0009977235547358643/.0009915145713361665, explain its behavior. Source's final
flag asserts final-pass stopping only. A test that incorrectly demanded
convergence of every inner pass was corrected against that actual source
contract; its failed log is preserved in `complete_contract_stdout.log`.
No solver update, field threshold or difficult comparison was dropped.
The first weighted original update and entire final original field are
separately compared, so cached final-field agreement is not the only check.

An earlier loading test recomputed the strict circle using NumPy hypot and
disagreed at floating-point boundary pixels. The fix retains the author's
public boolean support and verifies geometry away from circle roundoff,
rather than modifying data or hiding field errors. Its failed stage log is
preserved in `first_extraction_stdout.log`; final stage tests pass.
Unused zero-denominator far-away TX parabolas were not computed locally;
the selected nearest-parabola coefficients still match the independent source.

Independent physics cases check the complex adjoint, exact two-view inverse
and physical depth sign, phase wrapping/Nyquist scale, finite-packet variance,
known-element directivity symmetry, disjoint alias bands, and DCT equivalence
to an explicit repeating-boundary quadratic solve. The source finite-packet
variance is not renamed true noise. Fitted rigid angular velocity appears
only as a diagnostic of recovered fields, never as public calibration or truth.

The public README defines RF, phase signs, output order/units/support, required
conditional smoothing constants and the parabolic source discretization.
Method parameters are in hints/plans, not confused with instrument metadata.
Tests compare independent original instrument calibration to runtime constants
and actual README table values. All public NPZ keys have batch-first tables.
No hidden velocity/reference, target-fitted calibration or GitHub URL is in
README/raw observations/requirements.

## Fresh measurements, Notebook and scientific assets

Fresh complete local native CLI gives vx NRMSE4.9102137563055166e-15,
vz1.034358962900707e-15, NCC approximately1 for both. Maximum absolute
differences are3.3584246494910985e-14 and4.9960036108132044e-15m/s.
The independent Judge obtains the original phasor RMS.05201278063077122
from public RF; exterior is exactly0. Agreement is numerical reproduction
of a standard estimator, not independent mechanical accuracy.

The canonical notebook actively reads raw RF and reconstructs both complete
fields through local modules. It then independently invokes the standalone
Judge and actually computes all comparisons using fresh local complex beams.
Original references are separately loaded and labeled; there is no preview
mode substituting references for a fresh computation. Full machine precision,
actual kernel sys.executable/Python/NumPy/SciPy/Matplotlib/backend/thread values,
runtime, metrics and per-pass histories are saved in
`notebooks/reference_outputs/fresh_run.json` and private
`figures/experiment_metrics.json`.

Native subprocess nbconvert uses the actual shared research Python kernel,
1800s timeout, separate logs and captured return code. Initial successful
49.34391740000865s run and figures are retained in `prior_notebooks/`.
Final full native rerun after visual fixes returns0; its exact command,
runtime and canonical hash are in `notebook_process.json` and final evidence.
The fixes move geometry labels/arrow scale away from titles, pair sparse
log-metric points with measured limits, label genuine zero bars and use the
complete actual symmetric phase-residual range. They do not alter inverse math.

All six actual final figure groups are privately exported as PNG/SVG/PDF,
with their full-precision numeric NPZ sources and a JSON consumer/hash index:

1. Complete measured RF first pulse, actual all-trace spectrum, known geometry
   and actual sample traces. No unobserved high-rate waveform is invented.
2. Both complete raw and selected velocity components plus speeds/quivers.
   Shared signed display cap±.65m/s explicitly marks227 raw lateral outliers
   with extended colorbars; no numerical field value is clipped.
3. Every source/fresh component value, actual symmetric signed differences,
   complete error histograms and independently frozen reproduction limits.
   Exact zero histogram differences have a disclosed1e-18 display floor only.
4. Both measured Doppler fields, finite-packet correlation ratio, complete
   predicted-minus-observed phase maps and weighted/unweighted residual
   distributions. Full residual color range is common and symmetric, without
   suppressing outliers; scalar metric is a complex chord RMS, not angle RMS.
5. Raw, quadratic and three robust-scale estimates, actual paired vector/phase
   residuals, final-solve weights and a complete signed nonrobust discrepancy.
6. Actually recomputed lags1/2/4 and windows1/5/11/21, alias-sensitive profiles,
   complete component differences, full-field fitted-rigid-motion diagnostic
   and all actual bounded smoothing passes.

Differences from the selected reference are not mislabeled as true-speed error.
Historical2018 high-rate comparisons require unavailable distinct observations
and are explicitly identified as context. The available raw/regularized
comparison is preserved and actively extended, not silently dropped.

## Asset consumers, independent Judge and completion record

`evaluation/baseline_reference.npz` contains the independent original complete
two-component method field, standardized only to the declared public exterior
zero convention. There is no `ground_truth.npz` because no velocity truth is
available. `test/fixtures/upstream_stages.npz` supplies original stage arrays
to comparing tests; `instrument_parameters.npz` supplies actual known-source
calibration to metadata-contract tests. Notebook original arrays/provenance
are in `original_results.npz`/`original_run.json`; active fresh arrays and actual
environment/metrics are in `fresh_results.npz`/`fresh_run.json`. Each has a
documented consumer and original/fresh role, rather than unused opaque files.

The standalone Judge imports no solver and never runs submitted code. It
validates existence, exact key, batch-one shape, real float32/64, finite values,
finite/directional thresholds and nondegenerate references. Assets resolve
relative to its file, while CLI prediction/report paths are caller-relative.
Native exit0/1/2 and pass/fail/error JSON behavior are tested for supported
float32, eight poor complete outputs, malformed/missing/corrupt files,
missing/invalid evaluator assets, shape/dtype/object/NaN/Inf, undefined metrics,
foreign working directory, nested report creation and report-write failure.
Invalid inputs have empty metrics. Each component and physical criterion must
pass; missing output is never replaced with a reconstruction.

Final checks are recorded individually in private `final_evidence.json`:
exact paper/code/data provenance; original exact-bound native runs;
pre-extraction fixtures/freeze; independent physics; local explicit math;
complete native CLI/Judge; adversarial contracts; README/public sufficiency;
minimal dependencies/no domain imports; complete pytest; actual full
nbconvert0; all six actual figure reviews; semantic plans/signatures;
inventory/notices/cache archive. Failed or unverified mandatory checks block
READY. Static audits and two math renderers complement, rather than replace,
the physical derivations and actual numerical/visual reviews.


### Final observed gate results

Final full native pytest: **41 passed**, return code **0**, native process
runtime **31.91879080000217s**. Source-stage, full pipeline,
independent physical/metric, all malformed/poor Judge cases, README instrument
consistency and import-backend contracts pass. Earlier failures are retained
as identified above, with no threshold relaxation.

Final full active native nbconvert: return code **0**, runtime
**46.757693500010646s**, canonical Notebook SHA256
`72a0cea6c5287a62d9292ff8cdee9a0b67467225c0541d3893cd6039e30466e9`. The executed kernel reports the actual shared research Python,
Python3.11.11, NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 with inline backend
and numerical threads1. Process JSON independently captures the command,
kernel path, pre-run physical/commit headroom and real return code. The six
final PNGs were each actually opened after this last execution; paired
SVG/PDF and numeric NPZ are indexed with their consumers and hashes.

Independent readonly audit: **errors=[]**, foreign-working-directory
Judge **exit0**; **81 formulas** have zero extraction errors,
zero MathJax errors and zero KaTeX errors. This includes README, both actual
plans and canonical Notebook mathematics. Source/call-role review confirms
only permitted numerical primitives, explicit acquisition/inverse updates,
independent Judge and no cross-task/domain-library import. Cleaned environment
cannot import pymust. Importing main leaves the backend unchanged.

All mandatory gates are **passed**: full paper/version/data/provenance and
equation mapping; exact-bound independent original and cleaned native runs;
pre-extraction stages/frozen nonzero criteria; independent physics;
constants/units/signs/support; complete native CLI and standalone Judge;
adversarial contracts; public sufficiency/no hidden or GitHub leakage;
minimal environment/local executable math; complete tests; active native0
Notebook/all actual comparisons; distinct historical/original/fresh results;
semantic plan/signature/source review; inventory and legal notices. The
declared qualifications remain explicit: real speed truth is unavailable,
finite robust protocol has a first-pass budget stop, band setting is nominal,
phase ambiguity remains, and a separate data redistribution license is unstated.

Temporary notebooks, paper PDFs, downloads, original domain packages and all
logs/capture scripts stay outside the task. Any generated caches were moved
reversibly into the private archive. Canonical active outputs are retained.
Final inventory, figure provenance, archive index and completion hashes are
in `final_evidence.json`, `figures/plot_provenance.json`, `archive_index.json`
and `completion.json`.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- official author toolbox: https://www.biomecardio.com/MUST/functions/MUST.zip; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
- author vector-Doppler recipe: https://www.biomecardio.com/MUST/functions/html/iq2doppler_doc.html; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
