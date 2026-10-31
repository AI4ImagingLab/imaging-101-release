# Provenance and validation evidence

## Bound scientific task and source selection

This task infers four dominant stellar point-source positions, amplitudes and
constant local skies from real Spitzer GLIMPSE 4.5 micrometre mosaic cutouts.
It is not synthetic star rendering, an image catalog lookup, or a full DAOPHOT
reproduction. Selected method: published Photutils 2.3.0 ImagePSF cubic sampled
calibration model plus additive Const2D sky, with Astropy 8.0.2 TRFLSQFitter
bounded unweighted nonlinear least squares. Its complete forward and objective
mathematics is extracted locally; generic SciPy interpolation and optimizer
primitives are preserved. Stetson (1987), PASP 99, 191, DOI 10.1086/131977,
is the scientific PSF-photometry context, journal version of record verified
through Crossref metadata `doi-paper.json`, rather than a claim to have ported
all DAOPHOT routines or its empirical PSF construction. The full journal PDF
could not be downloaded from ADS (timeouts/404); no unseen paper equations or
historical paper results are represented as validated. All selected mathematical
conventions are directly established by the inspected, archived 2.3.0 source.

Remote repository: https://github.com/astropy/photutils ; tag 2.3.0,
annotated tag 2294e35bb196c555ecc4e0018d57514f9c7e1b53, peeled commit
f060288dd9ece60c3ef11f5b7d45809ee845612d. Both tag and peeled commit were
verified using GitHub API responses saved as `photutils-ref.txt` and `tag.txt`.
The codeload archive for that tag is `photutils-source.zip`; extracted complete
source is archived outside this task under `source/photutils-2.3.0/`.
Original execution used the released unchanged compiled wheel, not edited source.

Data repository: https://github.com/astropy/photutils-datasets , verified remote
commit 8c97b4fa3a6c9e6ea072faeed2d49a20585658ba, API response `datasets-ref.txt`.
Data files:

- https://raw.githubusercontent.com/astropy/photutils-datasets/8c97b4fa3a6c9e6ea072faeed2d49a20585658ba/data/spitzer_example_image.fits
- https://raw.githubusercontent.com/astropy/photutils-datasets/8c97b4fa3a6c9e6ea072faeed2d49a20585658ba/data/irac_ch2_flight.fits

Both commit-pinned downloads were byte-compared against the initial coordinator
downloads and match. This is a verified remote revision, not a local tracking
branch claim. Full FITS files and source archives are maintainer-only assets.

## Observations, calibration and legitimate preprocessing

FITS image shape (513,1025), BUNIT MJy/sr, TELESCOP SPITZER, INSTRUME IRAC,
CHNLNUM 2, GLIMPSE pipeline S13.2.0/1v04, Montage V2.2 mosaic,
FILENAME GLM_01800+0000_mosaic_I2.fits, creation 2006-12-02, mosaic scale
1.2 arcsec, native scale 1.213 arcsec. These are processed real survey data,
not unprocessed sensor counts. Integration-time metadata is not used to invent
a variance map. Four 17-by-17 crops have original zero-based center (row,column):
(294,216), (479,87), (226,788), (104,324), in that order. No rescaling, background
subtraction, injected noise, synthetic source, random seed or resampling is
applied to these observations. Big-endian float32 FITS values are converted
exactly to float64 and stored batch-first in public NPZ.

Selection was performed by the archived `choose_crops.py`: local maxima in a
21-by-21 neighborhood, peak brightness between 200 and 1500 MJy/sr, interior
support and finite patches, followed by selecting the four smallest relative
image residuals under the original one-source model. `selection.json` records
every candidate. This deliberately defines an isolated, bright point-source
benchmark where the stated one-source assumption is useful. It is not evidence
that this method works on all crowded sources or a statistically representative
survey sample; no independent saturation flag was available. Initial arbitrary
crops with severe neighboring-source contamination were rejected before final
observations, tolerances and local implementation were bound. The threshold was
not loosened to pass those contaminated cases.

The independent calibration has shape (81,81), SECPIX 0.30325 arcsec, comments
"drizzled PSF at 4-times oversampling (Megeath 12/03)" and "focus star, IOC/SV
campaign Q", creation 2004-01-02. It is not derived from the four targets.
Its original sum is 0.999999996322208. Calling the mosaic oversampling exactly
four without resampling would be physically wrong: 1.2/0.30325 is not four.
Therefore independently known angular scales define the explicit cubic
resampling to 0.3 arcsec before ImagePSF evaluation. The calibration sum is
then 16, preserving ImagePSF flux conventions. This orchestration is applied
identically around the unchanged original library and local implementation.
No source orientation or additional broadening is secretly calibrated from
the targets. Spatial invariance across distinct epochs/mosaicking is an adopted
approximate model, disclosed publicly; residual morphology shows its limitations.

## Original execution and success criteria frozen before extraction

Private archive root is
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task10_astronomy_photometry/`.
`capture.py` converts exactly the observations above, calls the unchanged
Photutils ImagePSF + Astropy TRFLSQFitter, and saves numeric stages/results.
Command: `oracle-env/Scripts/python.exe capture.py` (absolute paths and cwd
in commands.json). No training, random initialization or random seed is used.
Original environment: Windows 10 build 26200 x86_64, Python 3.11.11,
Photutils 2.3.0, Astropy 8.0.2, NumPy 2.4.6, SciPy 1.17.1, PyERFA 2.0.1.5.
Full installed inventory is `oracle-freeze.log`; `oracle-backend.log` records
NumPy/SciPy build/backend configuration, including OpenBLAS 0.3.31.188.0
64-bit-integer SkylakeX backend for NumPy. CPU execution only; GPU not needed.
The original mathematical fit takes approximately 0.020 seconds, whole Python
process approximately 1.12 seconds. All four terminate successfully by ftol
in 9,9,9,10 function evaluations. Exit status 0, no original scientific warnings.
`original.log` records final execution, `upstream-run.json` numerical parameters
and timing. `first_capture.npz` and `upstream_capture.npz` establish two independent
unchanged original runs on the exact observations: maximum parameter difference 0.

Separate original run `precision.py` rounds the effective PSF to float32 then
casts back to float64 (observations unchanged). It is a sensitivity study, not
the reference method. Max Euclidean position perturbation 1.4987535863424255e-8
pixel, max relative amplitude perturbation 1.1003613781237246e-8, max sky
perturbation 4.0027259196051546e-7 MJy/sr. Full numeric records are
`precision-run.json` and `precision_capture.npz`. Before local extraction,
`pre-extraction-tolerances.json` fixes calibration atol1e-14, initialization
atol1e-10, stage/full rtol1e-6, and Judge boundaries 1e-5 pixel, 1e-5 relative
amplitude, 1e-4 MJy/sr sky. Nonzero boundaries exceed measured finite-precision
sensitivity and are meaningful for reproducing this standard model. They are
not claimed astrophysical uncertainties. The independent observation consistency
boundary 0.12 is approximately 1.1 times the original 0.10814110360050014
relative-L2 residual, rounded upward; no silent alignment, mask or rescaling.

## Equation/stage-to-source-to-validation traceability

| Required mathematics / stage | Inspected upstream implementation | Local implementation | Independent validation consumer |
|---|---|---|---|
| Pixel-scale calibration resampling, sum Q=16 | Physical FITS scales and ImagePSF normalization contract; private capture.py preprocessing | preprocessing.prepare_psf | frozen calibration array vs original orchestration; normalization physics test |
| Exact cubic tensor-product PSF | photutils/psf/image_models.py, ImagePSF.interpolator | physics_model.make_interpolator | frozen original fitted-image evaluation, translation and flux linearity |
| a=4(x-x0)+40, c=4(y-y0)+40; zero outside grid | ImagePSF.evaluate | physics_model.model | original model array parity; synthetic exact recovery; sky linearity |
| Constant local sky | astropy.modeling.models.Const2D | physics_model.model | analytic constant-sky test, original fitted parameters |
| Initialization: max pixel, border median, positive excess | Selected private original run orchestration (not claimed built-in DAOStarFinder) | preprocessing.initialize | frozen original initial arrays; independently known impulse image |
| Residual model-data and unweighted sum squares | astropy.modeling.fitting.Fitter.objective_function / _NLLSQFitter and original model composition | physics_model.residual | original model parity, analytic spline objective-gradient finite differences |
| Bounded TRF optimization, 2-point Jacobian and epsilon | astropy.modeling.fitting._NLLSQFitter._run_fitter, TRFLSQFitter | solvers.reconstruct; explicit SciPy settings | frozen independent full-pipeline parameters and image arrays |
| Position/amplitude/sky metrics | Explicit benchmark definitions, not paper-reported metrics | standalone evaluation/judge.py | analytic known metric case, malformed/poor/CLI/evaluator contract tests |
| Public-data observation residual | Explicit acquisition model above | independent Judge spline calculation | notebook independently computes norm of actual fresh residual and checks same value |

There is no custom inverse update or domain-library wrapper in the cleaned
runtime. The original selected method uses a generic optimizer; replacing
Astropy's adapter with its explicit identical SciPy arguments preserves that
method. Source signatures, signs, factors, units, axis transpose, original
two-point step and defaults, stopping criteria and caller paths were reviewed
against both plans. No suspected upstream equation error or silent correction
was identified. The instrument sampling discrepancy was resolved explicitly
in acquisition preparation rather than concealed as an upstream code bug.

## References, fresh measurements and metric meaning

No actual target truth is available, so no ground_truth.npz is invented.
`evaluation/baseline_reference.npz` contains original independent-run positions,
flux and background. `test/fixtures/upstream.npz` stores original calibration,
initial parameters, final parameters and fitted images, consumed by stage/parity
tests. `notebooks/reference_outputs/upstream.npz` contains identified original
parameters and fitted images used by the notebook. `cleaned.npz` in that folder
contains identified fresh local parameter arrays for provenance; the notebook
always actively reconstructs and does not use this file as a fresh result.
These arrays have distinct documented consumers and are not unlabeled duplicates.

Fresh local output agrees exactly with original double-precision parameters
on the validated environment: position error 0, amplitude relative error 0,
sky error 0; original and fresh global observation relative L2 both
0.10814110360050014. No paper-reported or cached historical numeric value is
presented as reproduced. The metric reference is method output, not physical
truth. No NCC or dynamic-range NRMSE is used for mixed position/amplitude units.
Definitions and nonzero thresholds are explicit in metrics.json and README.

## Cleaned execution, notebook and environment

Shared environment Python is
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/shared-env/Scripts/python.exe`.
Only NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1 are runtime dependencies.
The shared environment has no Photutils or Astropy installed. No new package
was added to the shared environment; the separate oracle has its own packages.
An attempted shared `pip freeze` failed because pip is not installed there;
this optional inventory command is recorded honestly in commands.json/log and
is replaced by importlib.metadata inventory. It does not affect any task gate.

Validated commands from task cwd:

```
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
python -m jupyter nbconvert --to notebook --execute notebooks/spitzer_stellar_psf_photometry.ipynb --output spitzer_stellar_psf_photometry_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=120
```

CLI writes exactly required NPZ. Judge passes. Tests include per-function
physics, independently captured original stages/full pipeline, explicit known
metrics, missing/key/shape/dtype/NaN/Inf/complex/object/corrupt archive handling,
negative amplitude, invalid boundaries, missing evaluator assets, poor valid
submission, foreign-cwd absolute-path CLI and report-write failure code2/stderr.
Initial pytest temp setup hit Windows ACL on the default temp directory;
subsequent validated runs set TEMP/TMP to the private writable archive temp
directory, without changing numerical behavior. Logs preserve the actual
commands and results; no default-directory failed test is relabeled passed.

The active notebook reconstructs all four patches, calls the standalone Judge,
plots independent flight/adopted calibration, all four observation/original/fresh/
residual groups, full parameter tables and paired initialization/original/fresh
observation-residual bars. Fresh CPU mathematics took 0.01596119999885559 s
in the first executed review; 120 s timeout includes Python/kernel/plot startup.
The full nbconvert run exited 0. Windows ZMQ emits a Proactor selector-thread
compatibility warning and an unencrypted local TCP-kernel warning; neither is
a reconstruction error. No cell errors or skipped reconstruction cells.
The executed notebook is the shipped active notebook; the temporary output
copy was removed after archiving. Three extracted figures are stored privately,
not in the task. The 4-by-4 image comparison was visually inspected: source cores
align, original/fresh fits agree, residual structures remain visibly nonzero
and are honestly explained. Calibration and paired-bar plots were also inspected.

## License, source audit and release gates

Photutils code is BSD-3-Clause; original copyright/license is preserved in the license section below
and adaptation identified in physics_model.py and the migrated notice below. The dataset repository has
no separate dataset license file. NASA mission data use is based on NASA's
Images and Media Usage Guidelines (archived `nasa-copyright.txt`), which expressly
permit factual educational use with acknowledgment unless marked third-party
copyright. No such marking was identified in either FITS header. This is not
an assertion that every file in the data repository is BSD or all NASA-hosted
content is public domain. The migrated notice below gives mission, GLIMPSE, IRSA and NASA credit
and policy scope. No endorsement is implied. Full source, full FITS, capture
scripts, original logs and metadata remain only in the private archive.

All source imports and call roles were inspected: local forward spline,
initialization, objective, bounds, inverse optimizer and stopping options are
executable, with no prohibited domain imports, cross-task imports, cached solver
substitution, target-derived calibration, or library backend changes. Judge is
independent of solver code. Public NPZ contains only observations and independent
instrument calibration. Public README/requirements contain no GitHub domains,
hidden references, seed or truth. Every visible key, scalar, coordinate, unit
and output is documented. No metadata JSON or root-level wrapper is shipped.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Versioned source, data hashes, scientific method and stage mapping | passed | Remote tag/commit API, pinned data equality, table and hash inventory |
| Original and local runs on identical bundled observations | passed | Original/fresh logs, frozen original arrays, CLI and active notebook |
| Independent parity and applicable physical/mathematical checks | passed | Frozen stages, gradient, translation, flux/sky linearity, synthetic identifiability tests |
| Constants, units, precision and coordinates preserved | passed | Physical PSF resampling, explicit amplitude units, metadata test and plan review |
| Standard CLI/Judge and poor/malformed/evaluator failures | passed | Contract tests and generated report |
| README/raw/environment self-contained; no privileged leakage | passed | Explicit public forward model, source/dependency/NPZ audits |
| Minimal dependencies and CPU compatibility | passed | Shared environment execution, no upstream packages installed |
| Local forward/inverse mathematics and import/call-role audit | passed | Readable local modules, source review, AST/import inventory |
| Entire test suite | passed | Final test log and summary recorded below |
| Active complete notebook and scientific comparisons | passed | Successful executed review, 3 figures visually checked, fresh metrics |
| Historical/original/fresh and metric provenance distinct | passed | No historical values substituted; independent baseline clearly labeled |
| README/plans match implemented interfaces/math/execution | passed | Semantic source review and signatures listed in design |
| Final template, provenance archive and required notices | passed | Task inventory and external archive index below |

No mandatory gate is waived. Lack of physical truth and imperfect effective PSF
are limitations of the declared benchmark, not claims of independently validated
astrophysical accuracy. Bibliographic full-text retrieval and optional shared
pip inventory failures are separately disclosed rather than presented as success.


## Final measured inventory (SHA-256)

- Private `spitzer_example_image.fits`: `4b9cf15419de311fc683a9c378327a0b28bfee29d01e4fdbb9dc5780417f7ebc`
- Private `irac_ch2_flight.fits`: `8e41f387147b96895a8f7262a5b2fdb38e36c1a8ca76e2717d8f59ccb27a7069`
- Private `photutils-source.zip`: `fe4978f2a806516eb1688141a4e84b4681277cd1548709444ff91d7ba9994a0e`
- Task `data/raw_data.npz`: `7fdf45c9e8c21e7915cf7f7b4af4dfb9f92b5c73d3ed86aaa222f9a8ec6a5522`
- Task `test/fixtures/upstream.npz`: `ba01f8165989ade47c91d66edf54b45286f5cae18768a00888461535dc86d65e`
- Task `evaluation/baseline_reference.npz`: `38ec92818c1242cb26dca30d8d73c037adc7cfd3e1ff965dc61fc9c1257586d7`
- Task `notebooks/reference_outputs/upstream.npz`: `e34ddcf52f641d16eab76308ad365f51d5a5430c1dc7e239d6f12e37ddf54d52`
- Task `notebooks/reference_outputs/cleaned.npz`: `38ec92818c1242cb26dca30d8d73c037adc7cfd3e1ff965dc61fc9c1257586d7`
- Task `output/reconstruction.npz`: `38ec92818c1242cb26dca30d8d73c037adc7cfd3e1ff965dc61fc9c1257586d7`

Final test suite: **21 passed**, `tests-final.log`, exit 0. Final notebook complete execution: exit 0, `notebook-final.log`; final-commands.json records exact commands and wall times. Full shared package inventory: `shared-inventory.json`. All private top-level source/capture/log/figure files are hash-indexed in `archive-index.json`; extracted complete source and isolated oracle environment are identified above.


## Independent coordinator review (2026-10-18)

The coordinator separately reran the standard CLI, standalone Judge from a foreign working directory, all 21 pytest cases, and the complete active notebook via nbconvert in the shared CPU environment. All four commands exited 0. All three rendered comparison figures were inspected, including the independent versus adopted PSF, four observed/upstream/fresh/residual panels, and initial versus original/fresh observation residuals. Source, README and both plans agree on cubic interpolation, physical PSF resampling, amplitude units, patch coordinates, bounds, objective and stopping conventions. Evidence is archived at `batch10/coordinator/spitzer_stellar_psf_photometry/` and `batch10/coordinator/spitzer_stellar_psf_photometry-review.json`. Packaged reference snapshots were preserved around the notebook rerun. These checks establish reproduction of the declared model, not independent astrophysical truth.


## Preserved full license and attribution record (documentation revision)

The user requested removal of task-root LICENSE/NOTICE files. Their complete original texts and attributions are retained verbatim below, and original bytes are backed up in the external revision archive. This relocates notices without changing their terms, scientific data rights or source copyright. Historical upstream source paths/hashes remain historical evidence. Current source references point to this section.

### Original LICENSE (verbatim)

Original-byte SHA-256: `c844ec3c67921ac864041e14732205d0ba436e3ee439b9087d0db94d724ac2d2`.

```text
Copyright (c) 2011-2025, Photutils Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Original NOTICE (verbatim)

Original-byte SHA-256: `88b4554d5c23f1216887970e3b492d427e65c77ea273b0171f2f5e84208d9cc5`.

```text
The local sampled-PSF evaluation is adapted from Photutils 2.3.0 ImagePSF,
Copyright (c) 2011-2025, Photutils Developers, under the BSD-3-Clause terms
reproduced in LICENSE. General optimization uses SciPy, not a vendored
astronomical reconstruction library.

The real observations are Spitzer/IRAC GLIMPSE survey mosaic cutouts, with
an independent IRAC flight focus-star calibration from IOC/SV campaign Q.
Both are distributed as Photutils example data; FITS headers identify the
NASA Spitzer mission, UW Astronomy / GLIMPSE pipeline, and calibration origin.
The data repository does not supply a separate blanket dataset license.
These NASA mission observations are used for factual educational research
under NASA's Images and Media Usage Guidelines, which permit such use of
NASA content unless identified as third-party copyrighted material. No such
restriction was identified in the supplied FITS headers. This is not a
claim that all third-party content hosted by NASA is public domain.
https://www.nasa.gov/nasa-brand-center/images-and-media/

This research has made use of the NASA/IPAC Infrared Science Archive, which
is funded by NASA and operated by the California Institute of Technology.
This work is based on observations made with the Spitzer Space Telescope,
operated by the Jet Propulsion Laboratory, California Institute of Technology
under a contract with NASA. No NASA or institutional endorsement is implied.
```


## Completed documentation revision and revalidation (2026-10-18)

This revision changes explanatory documents, notebook organization and license
placement. Acquisition arrays, inverse mathematics, evaluation assets, thresholds
and packaged scientific reference snapshots are unchanged. The original documents,
notebooks, code and license bytes are retained under the private archive
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/revision_docs/task10/spitzer_stellar_psf_photometry/before/`; removed root notices are additionally retained in
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
-p no:cacheprovider` run passed **21 tests**. Full `python -m jupyter nbconvert
--to notebook --execute notebooks/spitzer_stellar_psf_photometry.ipynb --output review_executed.ipynb
--output-dir <private-task-archive> --ExecutePreprocessor.timeout=240` completed
with exit 0. Every one of **5 code cells** executed without error. All
**3 exported figures** were opened and visually inspected; comparison
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
| `plan/approach.md` | `55a0df5de9b6a5263b888a885a56dd2f19ca6e9c41abaefbbfa7f33f8cc16e28` |
| `plan/design.md` | `fcf003a2be1a062b4db586efb61867ed14380efdae20cd5e2325429415bd7c5e` |
| `notebooks/spitzer_stellar_psf_photometry.ipynb` | `9af6bf7bd817ca70d59099d6cc6a7516ca7b8477767556ed919c9a0214677c39` |
| `README.md` | `6cbc075ae073d2b31033c28b0ff47d439047ea20339859083b876cd4bf75f6f9` |

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


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- source: https://photutils.readthedocs.io/en/stable/api/photutils.psf.ImagePSF.html; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
