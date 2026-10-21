# Provenance and validation evidence

All mandatory gates below passed on 2026-10-18. This is an independently validated local implementation of the selected standard equivalent-source method on real observations, not a claim of geological truth or reproduction of the paper's larger gradient-boosted experiments.

## Selection, sources and licenses

The original GPRPy candidate was rejected before implementation because migration is delegated to external CREWES/irlib with a restrictive noncommercial license. No GPRPy code or data is shipped. The final task is Bushveld real-survey gravity-field continuation.

- Method repository: https://github.com/fatiando/harmonica, release v0.7.0, exact revision `b9f9c00b0999f470dcae971b19add28b055633c9`.
- Remote revision evidence: downloaded https://codeload.github.com/fatiando/harmonica/zip/refs/tags/v0.7.0; archive ZIP comment records that full commit. This is verification of the published tag archive, not a claim that the current branch is fresh. Git ls-remote failed through intermittent network reset and API was rate limited. The unrelated downloaded main archive is not method provenance.
- Unmodified upstream run: PyPI Harmonica 0.7.0 installed privately. Its `_equivalent_sources/cartesian.py` and `_equivalent_sources/utils.py` bytes were separately hashed and are identical to the tag archive. The exact solver stage also uses Verde 1.9.0 `verde.base.least_squares` and scikit-learn Ridge, whose environment is frozen privately. These dependencies never enter the cleaned runtime.
- Paper: Soler, S. R. and Uieda, L. (2021), *Gradient-boosted equivalent sources*, DOI https://doi.org/10.1093/gji/ggab297. Inspected author-accepted manuscript: https://eartharxiv.org/repository/object/2111/download/5366/, DOI https://doi.org/10.31223/X58G7C, accepted 2021 July 27, received 2021 February 21. PDF explicitly states author-accepted, pre-copyedited and CC-BY-4.0. Selected method is Sections 2.1-2.2 equations (1)-(10), the standard full damped solve; subsequent boosting/block-averaging sections are not implemented or claimed. The paper/source provide independent executable mathematics rather than a wrapper.
- Dataset: Bushveld gravity version 1, https://doi.org/10.5281/zenodo.6511942; bytes downloaded from https://zenodo.org/records/6511942/files/bushveld-gravity.csv.xz?download=1.
- Official preparation/documentation: https://raw.githubusercontent.com/fatiando-data/bushveld-gravity/v1/README.md, archived as data-readme.md. The redistributed data are CC-BY-4.0, distinct from the original NOAA NCEI/ETOPO1 public-domain sources. the consolidated attribution section of this validation record credits both levels. Method extraction preserves BSD-3-Clause copyright in modules and the consolidated licensing section of this validation record.
- Survey source has 3877 real stations. Inclusive geographic crop 27.5 <= longitude <= 29 and -25.5 <= latitude <= -24.5 retains all 306 stations in source order. No synthetic observations or noise are generated, so there is no src/generate_data.py. The source observation is `gravity_disturbance_mgal`, derived upstream by normal-gravity subtraction. It is not Bouguer-corrected and not raw gravimeter counts.
- Source station coordinates were projected privately using pyproj 3.7.2 from EPSG:4326 to EPSG:32735 with always_xy=True, then horizontal offsets 600000 m and 7200000 m were subtracted. Source `height_geometric_m` is retained; the distributor formed it from supplied mean-sea-level heights plus interpolated geoid heights. It is not target-derived calibration. The target plane is a requested geometry, not a hidden field measurement. Projection has no runtime dependency.

## Bound observations, hidden references and consumers

`data/raw_data.npz` SHA256: `cf2219ca2d9a7432684fd585c433074bef01a4894b719078f96d3813b5affaa1`.

The released raw archive contains only batch-first station_coordinates (1,306,3), gravity_disturbance (1,306), image_coordinates (1,48,64,3), all float64. Station and grid geometry are legitimate observation/output information. No source coefficients, image target, fitted calibration, truth, hidden seeds or GitHub links are released to the solver. Model depth/damping are not acquisition metadata. The standard public README, raw arrays and shared runtime subset suffice for independent solving.

There is no actual ground truth at the continuation plane. `evaluation/baseline_reference.npz` contains `reconstruction`, float64 (1,48,64), produced by the original unmodified standard method. It is never described as truth. `evaluation/ground_truth.npz` is intentionally absent. Judge gives true-reference precedence if a maintainer later supplies a valid actual truth, tested independently.

`test/fixtures/upstream_stages.npz` contains reconstruction, coefficients, kernel, source_coordinates, station_prediction from the original run; each is consumed by stage/full parity. `notebooks/reference_outputs/upstream.npz` contains original reconstruction and station_prediction, consumed by notebook comparisons and independent Judge subprocess contract checks. `notebooks/reference_outputs/cleaned.npz` contains freshly computed reconstruction, station_prediction, weak_reconstruction and weak_station_prediction; these are identified notebook-review measurements, not reference truth. Their measured comparison/runtime table is `cleaned_measurements.json`. No unused copies or external source trees are present in the task. Private repeat/single arrays are used only to establish numeric variability and tolerances.

## Original run before extraction

The maintainer-only archive is `../imaging-101-maintainer/batch10/task05_earth_gpr/`, relative to repository root. `capture.py` prepared the exact packaged observations and called original `harmonica.EquivalentSources(depth=6000,damping=0.1,parallel=False,dtype=...)`, then original `.fit`, `.jacobian`, `.predict` on both the original stations and the exact target coordinates. It never imports local src. Freeze fixtures/tolerances completed before writing cleaned mathematics. No upstream methods or frozen values were fabricated by a cleaned function.

Command from repository root:

    ../imaging-101-maintainer/batch10/task05_earth_gpr/oracle-env/Scripts/python.exe ../imaging-101-maintainer/batch10/task05_earth_gpr/capture.py

Exit status 0. CPU Windows 10 build26200, Python3.11.11; Harmonica0.7.0, Verde1.9.0, NumPy2.4.6, SciPy1.17.1, scikit-learn1.9.1, Numba0.68.0, pyproj3.7.2. Backend and full pinned package inventory are archived in oracle-backend.txt and oracle-freeze.txt. Original capture sets OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1, uses serial Numba and no GPU; no seed or RNG is used. These are real observations and a deterministic solve.

The first float64 upstream run took 0.6015742999734357 s (including relevant first-call compilation), repeat 0.025565800024196506 s, and independent float32-input run 0.5716448000166565 s. All original warnings lists were empty. Initial stdout numeric evidence is the capture_measurements.json archive; it records runtime, environment and data hash directly during original execution. No separate initial console transcript was retained, so this numeric capture is the original execution record rather than a fabricated transcript. Full original source/capture and unchanged numeric fixtures remain privately reproducible.

Repeated float64 output was bit-identical (max difference0, repeat NPZ hash equal). Float32 internal solve gave dynamic-range NRMSE 2.6443191016598888e-5 against float64. The float64 station RMSE was 0.636177032346923 mGal; float32 station RMSE0.6362874538222755 mGal. No historical paper metric has been relabeled as this run's result.

Frozen BEFORE local extraction in frozen_tolerances.json: kernel rtol3e-15; coefficient relative L2<1e-8; station and final image rtol1e-9/atol1e-8 mGal; Judge NRMSE max(1e-7,10 times float32 sensitivity)=0.0002644319101659889; NCC>=0.9999999999. The finite NCC tolerance was chosen to accommodate numerical cosine roundoff in deterministic double-precision replication, not to accept full single-precision inverse solves. A float32 file containing the standard estimate is separately tested and passes after float64 metric arithmetic. Tests are not relaxed based on local divergence.

## Mathematical and software traceability

| Paper stage | Upstream file/function | Local function | Validation |
|---|---|---|---|
| Sec2.1 eq1-4 generalized harmonic expansion | harmonica/_equivalent_sources/cartesian.py greens_func_cartesian and EquivalentSources.jacobian; utils.py Jacobian kernel | physics_model.greens_matrix, forward | Frozen original kernel/full image; independent 3-4-5 kernel and coordinate scaling |
| Source placement beneath each station, Sec2.4 baseline placement | EquivalentSources._build_points with numeric depth | physics_model.source_coordinates | Exact original source-coordinate fixture |
| Eq6 population column standard deviations | verde.base.least_squares, StandardScaler(with_mean=False) | solvers.solve_coefficients | Frozen original coefficients; independent augmented least-squares solve |
| Eq7-8 scaled objective | Verde least_squares and sklearn Ridge(fit_intercept=False) | solvers.solve_coefficients | Independent normal-equation gradient and normalization |
| Eq9-10 positive-definite solve and undo scaling | Ridge(alpha=0.1) and scale reversal | solvers.solve_coefficients | Coefficient relative L2 parity<1e-8 |
| Sec2.1 target prediction and continuation | EquivalentSources.predict | physics_model.forward, solvers.reconstruct | Full original image and station parity; independent harmonic Laplacian and adjoint |
| File/output boundary | Original capture of exact data and `.predict` | preprocessing.load_observations, main.main | Standard CLI, geometry/README constants, independent Judge |
| Independent metrics/evaluator | No solver or domain imports | evaluation/judge.py | Analytically known NCC/NRMSE; malformed/reference/config/report failure tests |

Physical checks are independent of copied equations: 3-4-5 Euclidean distance has known inverse response, doubling positions halves the kernel, a finite-difference Laplacian vanishes away from sources, the matrix adjoint identity holds, and an augmented SVD least-squares system agrees with the chosen normal-equation solve. The original kernel matrix is not modified during capture because the original `.jacobian` is called separately after `.fit`; Verde's internal in-place scaling cannot contaminate this fixture.

## Ambiguities and scientific review

The source code implements generalized 1/r basis functions even when observations are gravity disturbances, not Newtonian vertical point-mass acceleration. This is the explicitly selected published harmonic representation, not a substitution of a simpler acquisition model. Coefficients have no unique physical density meaning. The paper Eq5 prose says unscaled damping has the same units as Jacobian elements, which is dimensionally inconsistent with its squared objective; the implemented dimensionless scaled Eq8-10 avoids that ambiguity exactly as original software does. No unscaled damping formula was silently corrected or claimed reproduced.

Paper Eq8 changes regularization to the scaled parameters m; it is not merely a numerically preconditioned solution of identical unscaled lambda||c||^2. The approach and local code explicitly preserve this distinction. Paper baseline source placement, numeric depth and unweighted fit are retained; no default automatic depth, block reduction or boosting is substituted.

Signature/semantic review checked defaults6000m and0.1, population ddof0 scales, noncentering, positive residual sign, sum objective with no N factor, metres/mGal coefficient units, axis order, projection offsets, source below station sign and target height, and actual CLI/notebook/Judge paths. Plans describe only shipped runtime. Figures preserve shared scientific comparisons and label every newly computed weak-damping result. There was no upstream notebook with additional comparison groups to omit.

## Fresh cleaned results and notebook

Cleaned runtime: shared Python3.11.11; NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. Requirements declare only these validated dependencies. `cleaned-backend.txt` records NumPy SIMD/BLAS/LAPACK data; its configuration printer reports an optional pyyaml pretty-print warning only, not a solver warning. No domain or projection packages are installed or imported in the shared cleaned runtime. Source uses explicit inverse-distance kernel and scaled objective; SciPy positive-definite linear solve is the sole general inverse primitive.

Fresh full-image max absolute difference from original: 1.5149659304825036e-11 mGal. Coefficient relative L2 difference5.219290428931572e-12. Station max absolute difference1.6830092874897673e-11 mGal. Judge NCC0.9999999999999999, dynamic-range NRMSE3.8676581162572825e-14. Fresh station RMSE0.6361770323473235 mGal. These are actual new measurements, separate from original and from paper history.

Notebook active standard reconstruction runtime: 0.038662999984808266 s. It independently calls cleaned modules, writes the submission and invokes standalone Judge. Its weaker-damping comparison is freshly computed with lambda0.0001; station RMSE0.17826481067404384 mGal and coefficient L2 magnitude10107171.8989579 mGal m, versus standard2116633.8642859757 mGal m. This illustrates a fit/stability tradeoff and is not a different hidden reference. All three plots were extracted and visually inspected: actual irregular station map, shared-scale original/fresh image comparison and fitted-observed scatter; numerical-difference image; separately computed weak-damping and difference panels. Titles, scales, axes and comparison labels are visible and scientifically consistent.

No Matplotlib backend selection occurs in any library module, including import of src.main (unit-tested). Active notebook contains all executed cells with three image outputs and no execution errors. Complete nbconvert timeout120 s is justified by measured sub-second inversions; total notebook execution completed in several seconds. The archived/executed notebook was copied into the primary task notebook, and no temporary executed notebook remains in output/.

Commands from task directory using shared Python executable (its Scripts directory added to PATH for kernel discovery):

    python -m src.main
    python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
    python -m pytest test/ -v -p no:cacheprovider
    python -m jupyter nbconvert --to notebook --execute notebooks/bushveld_gravity_continuation.ipynb --output bushveld_gravity_continuation_executed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=120

All exit0. Pytest: 28 passed in1.30s. Logs archived cleaned-cli.log, judge-cli.log, pytest.log, nbconvert.log. Windows machine's preexisting system pytest temp directory denied access on the first test invocation; rerunning with TEMP/TMP set to the task's private archive temp folder resolved this environment issue, without modifying task behavior or skipping tests. Notebook emitted Windows Proactor-to-selector compatibility and local kernel TCP diagnostic warnings; it completed without error. These are explicitly recorded, not suppressed numerical failures.

Judge validates file/key/shape/float32-or64/finite values before scoring, loads NPZ with pickle disabled, and imports no solver. Score arithmetic converts validated real numeric arrays to float64 to avoid float32 reduction artifacts. All-zero predictions receive NCC0 and fail as valid poor predictions; constant or zero-norm references are evaluator errors. No hidden alignment, crop, normalization, magnitude, reference substitution or filename guessing occurs. Tests cover known metrics, poor predictions, missing file/key, wrong shape, integer/complex/object dtype, NaN/Inf, missing/malformed/constant references, invalid boundaries, truth precedence, foreign cwd, report-directory creation and report-write failure. Invalid/evaluator cases have empty metrics, descriptive errors and code2; valid poor arrays return1; passes return0. Report values serialize with allow_nan=False.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes/equation mapping | passed | Sources and traceability; archive indices below |
| Exact bound original and cleaned runs | passed | capture_measurements.json; fresh submission and CLI log |
| Frozen upstream stage parity and physics | passed | test/comparing_test, 6 scientific tests |
| Constants, units, precision, coordinates | passed | Geometry/README checks; source parity; independent units and harmonic checks |
| Standard CLI and standalone fresh Judge | passed | output/reconstruction.npz and judge_report.json |
| Judge adversarial and evaluator failures | passed | test/unit_test; return-code/report contracts |
| Public inputs sufficient without hidden assets | passed | main reads raw only; README complete geometry/contract |
| No public truth, calibration or GitHub leakage | passed | Public key inventory; README/requirements scan |
| Acquisition metadata/output contract/no metadata JSON | passed | README; inventory |
| Shared minimal CPU runtime without domains | passed | requirements and explicit imported/called-symbol audit |
| Explicit executable acquisition and inverse math | passed | physics_model and solvers; no domain shortcut |
| Complete test suite | passed | 28 passed; pytest.log |
| Active notebook and visual comparison review | passed | nbconvert.log; 3 inspected figures; executed primary notebook |
| Original/fresh/history and metrics distinction | passed | Separate NPZ/measurement identification above |
| Plans/signatures/mathematical semantics | passed | Explicit source and paper semantic review above |
| Final inventory/cleanup/licenses | passed | Only template permitted files; the consolidated licensing section of this validation record/the consolidated attribution section of this validation record retained; private archive outside task |

No mandatory gate is failed or unverified. The absence of actual image-plane truth is explicit and handled through a validated method reference, not by inventing ground truth. Numerical parity and successful Judge are not claimed to prove field accuracy.

## Archive index and hashes

Maintainer archive directory: `../imaging-101-maintainer/batch10/task05_earth_gpr/`. `archive-index.json` lists the following evidence assets and SHA256 values; upstream complete source and isolated environment are under harmonica-0.7.0/ and oracle-env/. Unused candidate downloads remain solely private. `task-file-hashes.json` indexes final package assets before adding this self-referential report. No complete checkouts, PDF, capture script, original logs or screenshots are redistributed inside the task.

| Private asset | SHA256 |
|---|---|
| harmonica-v0.7.0.zip | `38fef8cfbef2a409a8f2465e141447960b6876d6991a64681a8d744db920a7e4` |
| paper.pdf | `a4b7e9f1487b6fe0398d1754ee33f9dee5295ba5653b9eac1205e6d45575ff3a` |
| paper.txt | `634552b82aa93369f6be80832f9f2d14817cbfeb918a66450f888695dac9acf4` |
| paper-page.html | `362600356398e0818feabdc317b560701a5693fa0f76c568ba4032e25ead42ab` |
| preprint-page.html | `6aef7cb44f3a67b68532cfb570afad1792998be60ad28f10edfb35bc88f2e155` |
| data-readme.md | `6eddedc1095eb5810b84122d925f36137dfca26c11ce092ab9df64dee2a945cb` |
| bushveld-gravity.csv.xz | `3fc1daf74a2fc3bcc3cf7f72a632518f8c6b6e306ce12fddf4055d7cb44945c8` |
| capture.py | `666e5e3945443fe656197daa2b50038fb069022bdc1a21002a6cec077ae60203` |
| capture_measurements.json | `3bb05cfbefc5540a02071070a22200486d5d4790a5ea5fb69c0ba52806dcf737` |
| frozen_tolerances.json | `aec55d83be344d9fcdfb774d78341f61791d452188697c975b563588c0d16223` |
| upstream.npz | `95e27ba0f76b73e796189630c1bb9f2d0d492e938da3f551b480687da8242ef2` |
| repeat.npz | `95e27ba0f76b73e796189630c1bb9f2d0d492e938da3f551b480687da8242ef2` |
| single.npz | `2f712b9250907e6cf32c8cd2411633edd3d0a02935e3976086f0183db8b88ae0` |
| oracle-freeze.txt | `7f929edeae065995d389a1a600d22016ec1d23e7121370a91dfe255873b9a40a` |
| cleaned-backend.txt | `05c78a302d7229b6e95903bbc03ca1e03283b667fd1c73b83901f4420d298229` |
| cleaned-cli.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| judge-cli.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| pytest.log | `a89870913728fe4219ea942a5a01a24e09affff788848155d61595b073fdf8c9` |
| nbconvert.log | `5af45f265923961a8310a1cfb521dbddc245a8b433946bbbfcc8a07d25353038` |
| bushveld_gravity_continuation_executed.ipynb | `b6d84aeaedf23b2121447ef878aee047a33a5fa46249d2c6d0ffd3ff4630b67e` |
| figure0.png | `0afb26202821b1267f6557ade79f3f05e0ccbd7ccebac22e8a504ed2ff39731c` |
| figure1.png | `e0b1c321a8255ad1e2552afe4ad7301f1fbd296d28a507099a936697dddc2e25` |
| figure2.png | `bab28c5fb8770cb5aab081a49bc0725b644abea64783ff60433a96e6f5541c4a` |
| task-file-hashes.json | `b2358984796c1dba026e7b78123122953bb1d8a575f09ccea89113ff76504a53` |
| oracle-backend.txt | `605c3073ab7d7a107fbf0542c810f0efe71a3ca363d6667a70c057b80779bcc1` |

Selected task-asset SHA256 values:

| Task asset | SHA256 |
|---|---|
| data/raw_data.npz | `cf2219ca2d9a7432684fd585c433074bef01a4894b719078f96d3813b5affaa1` |
| evaluation/baseline_reference.npz | `0e1cf6855b42ab009f462ba97b8cdb9819134e6dae84eca6cdbe9c2f3ebb2af7` |
| test/fixtures/upstream_stages.npz | `95e27ba0f76b73e796189630c1bb9f2d0d492e938da3f551b480687da8242ef2` |
| notebooks/reference_outputs/upstream.npz | `855cefe07fdc5c0f3b8eab866662bae7ea48f6789664c2d3133107009e5903a2` |
| notebooks/reference_outputs/cleaned.npz | `92248192b442e75a0ab1034e16081eaf5ea4db3522569c1f69c61dc048f15669` |
| notebooks/reference_outputs/cleaned_measurements.json | `d7b37d6a68d38db175d8904929a4f514cff4223ba5038e1364a03764a1963ac7` |

## Independent coordinator review

On 2026-10-18 the coordinator independently executed the standard reconstruction,
the standalone Judge from another working directory, all 28 tests, and the full
active notebook through nbconvert. Each command exited 0. The explicit kernel,
scaled objective, solve, units and approach plan were reviewed. The coordinator
inspected the station distribution, upstream/fresh continuation images and
numerical-error panels; public input and import scans passed. Review logs,
executed notebook and figures are privately archived under
`batch10/coordinator/bushveld_gravity_continuation`, with the structured result
in sibling `bushveld_gravity_continuation-review.json`. Notebook-generated
remeasurements are archived separately and the original packaged measurement
snapshots are retained, so their recorded hashes remain valid.


## Consolidated original licenses and attribution

On the user's documentation-layout request, standalone license/notice files were removed after their exact original bytes were archived privately. The complete original texts, including every copyright, condition and disclaimer, are reproduced verbatim below. This changes the notice location, not the governing license.

### Original LICENSE (verbatim)

```text
Copyright (c) 2018 The Harmonica Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

### Original NOTICE (verbatim)

```text
Method extraction: Copyright (c) 2018 The Harmonica Developers, BSD-3-Clause (LICENSE).

Data: Bushveld gravity version 1, Fatiando a Terra data collection, DOI 10.5281/zenodo.6511942, CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). This task crops the redistributed survey and projects its station coordinates. Original gravity and topography sources are NOAA NCEI and ETOPO1, originally public domain. Redistribution preparation and derived disturbances are credited to the Fatiando a Terra data contributors.

The included observation arrays retain real survey gravity disturbances and measured/converted station heights; the continuation image is an inverse estimate, not a measured density or ground truth.


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

28 tests passed again.

Private revision evidence: `../imaging-101-maintainer/batch10/revision_docs/task09/bushveld_gravity_continuation/` (repository-relative sibling archive). Its `before/` preserves the prior plans, notebook, references and notices; `before-hashes.json` freezes the old package. `verification.json` records exact commands, runtimes, return codes and preserved reference hashes. `notebook.log`, `judge.log` and, where rerun, `pytest.log` are actual command output.

The full selected reconstruction notebook was executed through shared-environment `python -m jupyter nbconvert --execute`, using a writable private TEMP/TMP, shared Scripts on PATH and CPU thread count1. Per-cell timeout900s covers the slow blind inverse as well as faster tasks. Notebook exit code: 0; wall time 3.0302964999573305 seconds; independent Judge exit code 0. Complete pytest exit code 0. All code cells ran without errors. The 3 extracted figures were visually inspected for readable axes, labels, scientific comparison groups and numerical consistency. The executed output was saved back to the authoritative notebook path.

Before execution all `notebooks/reference_outputs/` bytes were snapshotted. New measurements were archived separately under `new-reference-measurements/`, then every original reference snapshot was restored byte-for-byte; `reference_hashes_preserved=true`. The shipped notebook and output submission are freshly computed, while restored snapshots retain their original measurement identity and hashes. This is an evidence-preservation procedure, not a cache substitute for reconstruction.

Current package hashes, including changed plans/README/notebook/source notice text, are indexed by private `after-hashes.json`; older source/task inventories remain historical records under their original archive rather than being misrepresented as hashes of this revision. Public raw NPZ, baseline and metric-boundary file SHA256 values were checked unchanged. No temporary executed notebook, figure PNG, source checkout or cache was added to the released task.
