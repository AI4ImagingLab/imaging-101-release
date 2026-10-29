# Provenance and validation

## Bound scientific scope and immutable sources

Task: direct-wave total focusing of the author's **real measured** aluminium contact-notch half-matrix. This is a selected published imaging component, not the full multiview immersion experiment. There is no independently known reflectivity truth. `evaluation/baseline_reference.npz` key reconstruction stores the original independently executed method envelope; it must not be relabelled truth.

Paper: Budyn et al., IEEE TUFFC2019, DOI10.1109/TUFFC.2019.2909988. Paper version is the2019 journal method cited by upstream; article metadata was remotely verified. Upstream repository https://github.com/ndtatbristol/arim, remotely queried commit0cee9de26ae995ca80e00bc67801ffb3b1188e44 on2026-10-20, codeload archive `arim.zip`. MIT copyright2016–2025 arim contributors is retained in NOTICE. Bound source file `examples/example-datasets/contact_notch_aluminium.mat` SHA256 c58d25ccaf3081348c392d98a45a5e28809c80c900782a9b60f62d1a7ef2c76f. Do not substitute the differently parameterized tests/data/brain acquisition description.

Public NPZ SHA256 578d9bb06c29a2509abdddda38b476dab474714112d6bb246447725e32fc0ab3. Frozen parity fixture SHA256 dc7a28dde58ecfc4ed2f9d06dc740ce7ee76a7477ee168adbc5e4809cec24b5d. All raw values retain exact original float64 amplitudes/positions/time and integer pair indices; transpose, zero-based indexing and leading batch axis are the only interface changes. Raw calibration comes from stored source data, not the reconstruction. No hidden seed, object or target-derived calibration is public.

## Maintainer archive index

Archive root: `imaging-101-maintainer/research_batch2_20261020/material_mechanics/` outside this task. `arim.zip` and expanded exact-commit source preserve upstream; `capture_ultrasonic.py` records executable oracle capture; `ultrasonic_oracle.npz` and `ultrasonic_oracle.log` retain original measured arrays/intermediates/results/runtime/hash; `oracle-env/` is the isolated oracle environment; `ultrasonic_executed.ipynb` is the temporary executed notebook retained privately; `ultrasonic_cell*.png` are private render QA captures. Builder scripts and private figure exports are indexed here and never imported by shipped code. No original logs, PDFs or checkouts are packaged in this task.

## Original run before extraction

Command from archive: `oracle-env/Scripts/python.exe capture_ultrasonic.py`. Exit0, original contact_tfm executed on exact bound observations before local implementation. Original runtime1.7945842000190169s including focusing, CPU one thread. Original filtering is upstream Hilbert()+ButterworthBandpass(order5,0.5e6,6e6); original grid step0.2mm is a deliberate benchmark resolution choice within the example's same x/z limits (upstream demo uses0.15mm). Every local grid element exactly matches the frozen original grid. Original source functions are unmodified.

Environment: Windows x64, CPython3.11.11, NumPy2.4.6, SciPy1.17.1, numba0.68.0/llvmlite0.50.0; CPU execution, NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1. Oracle-only domain package is archived arim; cleaned shared interpreter is separate and has no arim/numba installed. Fresh notebook runtime2.6031156999524683s. No GPU used. Modern-library port fidelity is numerically checked; this is not a claim to reproduce the entire2019 software environment.

## Equation/function/test traceability

| Mathematical stage | Upstream function | Local function | Validation |
| --- | --- | --- | --- |
| Stored HMC acquisition and units | io.load_expdata | preprocessing.load_data | exact all-array equality against original fixture |
| Bandpass and analytic transform | signal.ButterworthBandpass, Hilbert | preprocessing.analytic_filter | frozen complex samples rtol2e-12 atol1e-14;2MHz unit-amplitude analytic test |
| Homogeneous path delay norm(r-p)/c | im.tfm.contact_tfm + geometry.distance_pairwise | physics_model.travel_times | independent3-4-5mm analytic triangle |
| Reciprocal weighting1/2 | ut.default_timetrace_weights | physics_model.reciprocity_weights | analytic three-pair weights and normalization |
| Nearest-time coherent sum / recorded pairs | im.das.delay_and_sum_numba_noamp | solvers.delay_and_sum | valid constant traces, zero outside record, full original parity |
| Envelope | abs(original complex result) | solvers.reconstruct | max absolute difference9.43689570931383e-16 |
| Reference agreement | independent known arrays | Judge + visualization.agreement | hand-computable metric cases and malformed/poor output contracts |

Parity tolerances were fixed from float64 stage requirements before accepting local results: filter rtol2e-12/atol1e-14 and full image rtol1e-10/atol2e-14. No thresholds were relaxed to accept divergence. All local numerical conventions match the source (including division by recorded count rather than weight sum). No upstream mathematical correction was required.

## Result provenance and thresholds

No historical paper accuracy numbers are reused. Independently reproduced upstream and fresh local refer to identical acquisition, parameters and reference. Fresh NCC1.0, dynamic-range NRMSE7.99205690381781e-17. Separate rerun is deterministic to machine precision; nonzero NRMSE boundary1e-7 and NCC0.999999 allow numerical variation. Zero reconstruction is a valid prediction that fails. Pulse-echo NCC0.38944913581657725/NRMSE0.08262852270936352; every-second-element NCC0.9692052689568728/NRMSE0.015397808528126737; speed+3% NCC0.36170547411514176/NRMSE0.0698119645457603. These compare to the standard method, not actual truth. Full precision remains in notebook output/source arrays. Metrics use no mean subtraction, crop, mask or fitted scale.

## Mandatory gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| Exact source/paper/data/equation mapping | passed | immutable provenance and table above |
| Original and cleaned exact-observation runs | passed | original log and fresh pipeline/parity |
| Independent numeric fixtures and physics | passed | frozen stage arrays; path/weight/boundary/analytic sinusoid checks |
| Constants, units, precision and conventions | passed | exact original-array/grid equality and README constant test |
| Standard entry point and Judge | passed | fresh output; standalone Judge pass report |
| Judge malformed/poor/evaluator/report failures | passed | unit_test/test_contract.py |
| Solver-visible self-contained contract/no leaks | passed | public arrays+README, no GitHub URLs or truth |
| Shared dependencies and no domain solver | passed | NumPy/SciPy/Matplotlib only; separate oracle interpreter |
| Unit/parity/physics/pipeline tests | passed |14passed, independent parent rerun4.46s; command below |
| Active complete notebook via nbconvert | passed | private ultrasonic_executed.ipynb; research kernel;180s timeout; three active figures |
| Required comparisons and fresh labels | passed | source acquisition/filter/dB comparison retained; additional sensitivity computed |
| README/plans semantic review | passed | source signatures/defaults/shapes and normalization reviewed; backwall peak limitation stated |
| Final inventory/licence/private archive | passed | authoritative structure, NOTICE, no runtime-generated figures outside notebook |

Commands: from task, shared Python `-m src.main`; `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `-m pytest test/ -q --basetemp=<private unique directory>`; `-m jupyter nbconvert --to notebook --execute notebooks/ultrasonic_notch_tfm.ipynb --ExecutePreprocessor.kernel_name=research --ExecutePreprocessor.timeout=180 --output ultrasonic_executed.ipynb --output-dir=<private archive>`. JUPYTER_PATH points to the private archive's explicit shared-Python kernel. nbconvert completed exit0; its launcher emitted a Windows Proactor fallback warning, with no notebook execution errors.

Visual QA: all three rendered notebook figures inspected at native resolution: clear mm/time/amplitude units, perceptual magma dB scale, common comparison limits, diverging numerical-difference scale, no clipping. White-space around narrow depth panels is intentional. No quantitative physical-truth claim is inferred from visual agreement.

Final coordinate QA: imshow extent uses pixel/cell boundaries (sample endpoints plus/minus half spacing), with displayed limits set to declared sample-centre endpoints; Fourier-bin extents likewise use half-bin boundaries. Canonical notebook and final exports are reexecuted after this display-only correction.

Final raw-array interface audit identified native tx/rx dtype uint32 (not int64). README/design now explicitly preserve/document uint32, with a public-table-versus-NPZ dtype check; no observation array was recast.


## Independent parent acceptance

Parent reviewed the implemented equations, numerical conventions, shipped signatures/plans, independent upstream evidence, physics checks and all 3 actual scientific figure groups. Parent strict inventory/public-input/import-role audit returned no errors and independently executed the standalone Judge from a foreign working directory (exit 0). Parent verified 15 distinct passing test cases; evidence is archived in the second-batch parent audit/test records. Backwall is strongest peak; pixel-centre display coordinates corrected; no independent reflectivity truth.
