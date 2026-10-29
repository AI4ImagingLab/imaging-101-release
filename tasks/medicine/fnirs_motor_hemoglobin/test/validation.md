# Provenance and validation: complete FRESH motor hemoglobin task

Status: local cleaning gates passed; parent independent acceptance passed (see Independent parent acceptance). This task is new (relative chromophore concentrations from two-wavelength CW optical intensity), distinct from CT/MRI, calcium, FLIM and polarization tasks. No generated observations or invented biological truth are used.

## Fixed scientific binding and sources

Unknown: all signed effective relative HbO/HbR concentrations, 68 pairs by4860 samples. Acquisition: two nominal wavelengths760/850nm, native8.928571428571429Hz, full sub-01/ses-left2s run, no crop/downcast. Method: original MNE optical_density plus modified Beer–Lambert spectral pseudoinverse; local explicit inverse is the exact full-rank two-wavelength specialization. PPF6.0 is fixed, not fitted from target.

Dataset: Rickson C. Mesquita, FRESH Motor Dataset v1.0.0, https://doi.org/10.18112/openneuro.ds005963.v1.0.0. Author repository https://github.com/OpenNeuroDatasets/ds005963, remotely verified tree f9f9571b257168ffe641b6c3d16316af73570724. Author dataset_description explicitly says CC0. Source data URL https://s3.amazonaws.com/openneuro.org/ds005963/sub-01/ses-left2s/nirs/sub-01_ses-left2s_task-FRESHMOTOR_nirs.snirf. SHA2566bc256d5117d63b8a81b765da8c3ff0096fd30ea66a7e48453b506fb3d8b2765,5703280bytes. All136 channels and4860 samples are retained. The author file is read privately using MNE; public NPZ contains exactly reshaped native float64 intensity and independent calibration, not hidden estimates. Spectral coefficient matrix is independent Prahl hemoglobin tabulation, not fitted from this participant. Midpoints are measured optode geometry and used only for plotting. Per-array exact byte hashes are frozen in test/fixtures/input_hashes.json.

Code: https://github.com/mne-tools/mne-python/tree/5a67897d193a00b63ba92f2b01f4a6eb098918db, tag1.12.1 resolves to this actual code commit (verified remote commits API, not an annotated tag SHA). Archive mne-python-v1.12.1.zip includes full original source and BSD-3-Clause license. Original wheel version1.12.1 is independently installed in private oracle-env. Selected paper background: Gramfort et al., Front.Neurosci.7:267(2013), published version doi:10.3389/fnins.2013.00267; practical fNIRS scientific context Luke et al., Neurophotonics8:025008(2021), published version doi:10.1117/1.NPh.8.2.025008. No claim of reproducing Luke's separate auditory dataset or reported group statistics is made. No historical paper numbers are used as this run's results.

## Archive and reproducible independent original run

Maintainer-only archive is `research_batch3_20261020/bio_med/` beside the release repository, outside all task directories. `fnirs_archive_index.json` indexes exact file sizes and SHA256 for full original source, native data, private capture, environments, logs, stage arrays, acceptance criteria and executed notebook archive. Author metadata/readme, `mne_remote_revision.json`, and `ds005963_tree.json` preserve retrieval evidence. `oracle_environment.txt` is the complete original environment, including NumPy2.4.6/SciPy1.17.1/MNE1.12.1/h5py3.16.0. `clean_numerical_backend.txt` records the actually executed CPU backend and machine. Both use Python3.11.11/Windows; original and clean runs use one OMP/OPENBLAS/MKL thread; no GPU or >2GiB allocation is required. Shared runtime has no MNE import and no domain package was installed there.

Original command, from archive: `oracle-env/Scripts/python.exe capture_fnirs.py`, exit0,0.517s first successful run. `fnirs_oracle.log` records all channels, zero timing jitter, complete sample interval and no numerical warnings. A preceding metadata serialization attempt failed after calculations because raw.n_times was np.int64; the capture was fixed and run again, without touching original functions or observations. `fnirs_original_rerun.log` and `fnirs_rerun_measurements.json` record another independent original execution, exit0 and maximum original-rerun difference0M. No simulator seed is involved. All stages and the full reference were frozen before any local implementation. `fnirs_acceptance.json` fixed stage rtol1e-12/atol1e-16 and Judge thresholds before local extraction.

Public raw NPZ archive SHA256f068b10fba13f3ab8ba358cd1c84e4cb9d51add59b13efa06340ba9e7a8f2ea1. These exact observation arrays are what original MNE read before packaging: pair reshape preserves all values, channel ordering and native precision. No calibration changed after the original run.

## Equation/source/local/test traceability

| Mathematical stage | Original source/function | Local function | Independent validation |
|---|---|---|---|
| z=-ln(I/mean(I)) | mne/preprocessing/nirs/_optical_density.py optical_density | preprocessing.optical_density | complete frozen output_od; multiplicative-scale analytic case |
| Nominal760/850nm E, original0.2303 factor | _beer_lambert_law.py _load_absorption | public extinction, physics_model.absorption_matrix | original frozen calibration bytes, tests input hashes |
| A=1000 E d f, meters, M/mM conversion | _beer_lambert_law.py beer_lambert_law EL/iEL | absorption_matrix, forward | diagonal analytic60-factor case, inverse distance scaling, signed values |
| Full-rank two-species least squares | beer_lambert_law pseudoinverse@OD*1e-3 | solvers.invert_density explicit cofactors | complete frozen output_hemoglobin; analytic inverse, full native pipeline |
| Euclidean transpose | mathematical transpose, independent of original solver | physics_model.adjoint | independent random adjoint identity |
| Canonical public-to-output path | original read_raw_snirf/OD/MBLL capture | main.run | full exact-observation parity, output roundtrip, CLI/Judge |
| Cosine NCC/dynamic-range NRMSE | independently specified benchmark definition | standalone Judge; visualization.metrics | known [0,1,2] case and reverse-vector case |

Ambiguities: effective partial pathlength is assumed, not calibrated for this individual; the model has no depth discrimination. MNE extinction multiplication uses approximate0.2303, which is preserved verbatim rather than recomputing ln(10)/10. Coordinates have original head-frame geometry; there is no invented anatomy overlay. Relative c from arithmetic mean intensity does not have exactly zero temporal mean. The two-wavelength system exactly fits OD and cannot validate absolute physiology. The local explicit inverse differs from SVD pseudoinverse only at roundoff; no upstream equation was corrected or substituted.

## Reference, fixture and metric consumers

`evaluation/baseline_reference.npz` contains key reconstruction(1,68,2,4860), original MNE relative HbO/HbR in M; consumed by Judge and notebook. It is an identified method result, not ground truth; ground_truth.npz is absent. `test/fixtures/upstream_stages.npz` contains output_od and output_hemoglobin, independently captured on the full recording before cleaning; full-stage comparing test consumes both. This deliberate full-stage fixture enables separate preprocessing/inverse diagnosis. `test/fixtures/input_hashes.json` verifies preservation of every public array's native bytes. `notebooks/reference_outputs/upstream_measurements.json` records original source/data/environment/timing; `fresh_measurements.json` records actual fresh complete reconstruction and every controlled experiment. No cached array is plotted as fresh.

Numerical thresholds are nonzero because this is a method-reference task. Independent original reruns give zero difference; cross-implementation SVD-versus-cofactor maximum difference is3.3881317890172014e-21M. A conservative float64 conditioning allowance (condition number3.206208353438045; order1e4*epsilon*condition) motivates NRMSE<=1e-9, NCC>=0.999999999 and relative OD residual<=1e-10, fixed before extraction. These boundaries test numerical reproduction, not physical truth. Every metric must pass for both species and forward consistency. The all-zero prediction fails; malformed arrays produce error/empty metrics. No alignment, crop, rescale or magnitude convention exists. NCC may exceed1 by floating-point roundoff (measured1.0000000000000029), with no material semantic implication.

Actual fresh results: HbO NRMSE8.725454284175138e-18, HbR NRMSE1.4751889331396248e-17, independent Judge OD residual1.8509418462827703e-16. Notebook forward implementation residual1.5687220066700879e-16 differs only in arithmetic operation order. PPF4.8/5.4/6/6.6/7.2 gives relative concentration changes0.25/0.11111111111111119/0/0.09090909090909098/0.16666666666666718. Removing a wavelength produces relative two-species error0.8861644380317752. Four short channels are retained; their regression changes concentrations by0.4401179361201338. Every plotted comparison was computed; regularization and sample-reduction numbers are in fresh JSON without presentation rounding.

## Commands and checks

From task directory using validated shared interpreter (NumPy2.4.6/Matplotlib3.11.1), with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1:

- `python -m src.main`: pass, complete native output, approximately0.15s including compressed write.
- `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: pass, exit0.
- `python -m pytest test/ -v -p no:cacheprovider --basetemp=<private unique fnirs directory>`:11 passed in2.69s, `fnirs_pytest.log`.
- `python -m nbconvert --to notebook --execute notebooks/fnirs_motor_hemoglobin.ipynb --output fnirs_motor_hemoglobin_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180`: pass, `fnirs_nbconvert.log`. Explicit private bio-med-research kernelspec selects shared interpreter. Canonical notebook retains executed outputs; the temporary output notebook is archived outside task. Four figures, each exactly one PNG display output (output count checked programmatically). Windows Jupyter emits its standard Proactor/selector-thread and local TCP transport warnings; no cell errors.

Judge contracts tested: valid reference, poor zeros, missing file/key, wrong shape, float32, complex, NaN, Inf, unavailable evaluator assets, actual-truth priority, foreign caller cwd, creation of report parent, report-write failure, strict finiteJSON and exit0/1/2. Physics tests: analytic inverse/unit/sign, pathlength scale, logarithmic baseline gauge, adjoint identity. All public input arrays checked bytewise. Metric tests independently verify known numeric cases and notebook/visualization formula consistency.

## Visual and semantic review

Private visual contract: fnirs_visual_contract.md. Final figures are only `final_figures/fnirs_motor_hemoglobin/{recording,parity_identifiability,sensitivity_channel_maps,regularization_short_regression}.{png,svg,pdf}`; plot_sources.npz/fresh_measurements.json contain exact plotted arrays/numbers and are listed in fnirs_figure_manifest.json. All four actual PNGs opened and inspected: labels/legends/colorbars readable, no clipping/glyph loss, signed maps share symmetric full-data range, RMS maps share sequential zero-origin scale, residual is signed, units and projected head-coordinate limitations are visible. No biological image manipulation or arbitrary spatial crop. SVG/PDF share the reviewed Matplotlib figures.

Source/plan semantic review passed: natural log sign and temporal baseline, original0.2303 constant, distances in meters, M/mM factor1000, spectral/species ordering, signed unconstrained two-by-two inverse, no regularizer in canonical run, every pair/time preserved, no iteration/stopping invented, library imports do not mutate Matplotlib backend. README five required sections and exact raw table; acquisition metadata/output contract are self-contained and contain no GitHub domain or hidden reference key/path. Source signatures/defaults match design and flow. Source/generator(absent)/Judge/tests/notebook import/call-role audit finds only permitted NumPy/Matplotlib math and standard library; no MNE/domain solver import or cross-task import. Raw positivity model and full-rank assumptions are independently checked.

## Completion gates

| Mandatory gate | Status/evidence |
|---|---|
| Exact paper/source/data/license and equation mapping | passed; sections above/archive index |
| Independent original and fresh complete bound observation execution | passed; original logs/frozen stages/full parity |
| Independent parity and applicable physics | passed;11 tests, declared prior tolerances |
| Constants/units/precision/coordinates | passed; native bytes and analytic scaling |
| CLI NPZ and fresh Judge | passed; complete output/report |
| Judge invalid/poor/evaluator/report failures | passed; standalone contract suite |
| Solver-visible inputs sufficient; no leakage/GitHub URLs | passed; public allowlist audit |
| README acquisition/output, no metadata JSON | passed |
| Minimal shared dependencies/CPU without domain packages | passed; shared CLI/tests/nbconvert |
| Explicit local forward/inverse mathematics and symbol/call roles | passed |
| Unit/metric/Judge/parity/physics/pipeline tests | passed;11 tests |
| Active complete executed notebook, preserved scientific comparisons | passed; four unique output figures |
| Original vs fresh labels, formulas and justified boundaries | passed |
| README/plans/source semantic consistency | passed |
| Final template inventory and license, private archive | passed; no extra root entry/tests/PDF/PNG/source archive |
| Parent independent final acceptance | passed (see Independent parent acceptance); local owner does not self-accept |

Parent review correction: all three time-axis heatmaps now use half-sample outer boundaries [-dt/2,tlast+dt/2] so original sample centers are preserved. The fourth regularization/short-regression figure is explicitly notebook-owned and documented in design; the visualization module returns the first three. Notebook and all final formats regenerated after this change.


## Independent parent acceptance

Passed: parent independently ran 11 tests with the shared research interpreter, then the standalone Judge from a foreign working directory. Parent source/call-role, public-allowlist, exact output, dependency, executed-notebook and file-inventory audit reported no errors. Reviewed modified Beer-Lambert matrix scaling, natural-log optical-density convention, explicit cofactor inverse, all two chromophores, source calibration independence, fixed PPF limitations, independently known metric cases and all four actual executed figures. Pixel-boundary time coordinates corrected and notebook/export re-executed before final audit. Parent audit and separate evidence are archived at `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\fnirs_motor_hemoglobin-parent-audit.json`. Final provenance hashes are refreshed after this parent-only acceptance note.
