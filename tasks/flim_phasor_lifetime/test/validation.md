# Authoritative provenance and validation — flim_phasor_lifetime

Validation date: 2026-10-20. Scope is a complete independently executed selected standard method on the exact bound real observations. All results below distinguish upstream measurements from fresh local measurements; there is no physical ground truth and no reconstruction is relabeled as truth.

## Frozen selection and source identity

Selected implementation: PhasorPy v0.7; repository https://github.com/phasorpy/phasorpy; exact executed revision `2d426ecbf14a46740b53054d66a522c692d89a89`. Remote identity was verified with a successful `git ls-remote` check (a later network retry reset and is not used to assert newer freshness) (PhasorPy v0.7 tag or qmrpy HEAD) and matched the archived checkout. This identifies an actually checked remote revision on the validation date, rather than merely a local tracking pointer. The private checkout is retained; no freshness beyond that check is claimed.

Paper and version: Digman et al., Biophysical Journal 94(2), L14–L16, issue 2008-01-15; DOI 10.1529/biophysj.107.120154. The three-page version of record was inspected through the author-deposited UC Irvine eScholarship PDF (https://escholarship.org/uc/item/9rg395f7); a downloadable publisher reprint is archived from https://qbi.uq.edu.au/files/40841/phasor-approach-to-lifetime.pdf as digman_2008_uq.pdf. Phase/modulation lifetime conventions are standard frequency-domain summaries, e.g. Weber (1981), DOI 10.1021/j150608a006, and are selected here through the exact modern PhasorPy implementation. The Digman letter explicitly omits component lifetime calculation on L16; this task does not misattribute the apparent-lifetime extension to it.

Data and redistribution: https://zenodo.org/records/15007900 (DOI 10.5281/zenodo.15007900), Alessandro Rossetta / FLIM LABS, with Advanced Bioimaging Unit, Institut Pasteur de Montevideo as curator. Record API zenodo15007900.json establishes CC BY 4.0. Full native 256×256×256 sample histogram is decoded with unmodified upstream signal_from_flimlabs_json; no crop or spatial binning. The separate fluorescein image is pooled spatially in uint64 before phasor normalization. Both original JSON files are archived under data/. No synthetic benchmark data were created.

Code license is MIT; original source copyright/license notices are preserved verbatim in task NOTICE, including qMRLab notices where applicable. Data attribution and conversion changes are also recorded there. Paper PDFs, complete checkouts, original observations, capture scripts and original logs are maintainer-only and are not distributed with this task.

## Maintainer-only archive index

The archive root is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/bio_medical`. Relevant original checkout is `phasorpy/`. Independent capture command is `oracle/Scripts/python.exe oracle_flim.py`; command was run from the archive with the same process interpreter resolved absolutely. The complete SHA256/size archive index is `archive_inventory.json`. Capture source, full stdout/stderr log `flim_oracle.log`, machine record `flim_oracle_record.json`, and raw downloads/data files are retained there. Source capture is independent of all cleaned modules; failing preimplementation import/collection evidence is `flim_red.log`.

Original independent source file hashes:

| Archived source | SHA256 |
| --- | --- |
| phasorpy/src/phasorpy/phasor.py | 1da35b6ee819ae018493293e1016f053f57afefe9a381fad6b5165629af6e97a |
| phasorpy/src/phasorpy/lifetime.py | 94ff660fd43a2ea93084b22dfa3fb68ef4cf04019933584b83b121e6dc35d56e |
| phasorpy/src/phasorpy/_phasorpy.pyx | 23878c7a37e44bfef174683321737ef856402609ecd9771bb2a0512274fd5c9d |

`runtime_environment.json` records the cleaned runtime, platform and complete NumPy backend; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle/` contains isolated upstream-only packages. `upstream_reruns.json` and `upstream_reruns.log` record a later independent original-source run on the exact NPZ arrays. `flim_phasor_lifetime_check_0.log` through `_check_3.log` retain all final entrypoint/Judge/pytest/notebook stdout and stderr. `flim_phasor_lifetime_verification.json` records their subprocess exit codes and wall runtimes. Extracted notebook PNGs are private visual-review evidence, listed in that JSON. No private script is imported by source, tests, Judge or notebook.

## Original upstream run, inputs and numeric capture

This is an actual source execution on the bound observations, not an import check, cached notebook or paper metric. No random seed affects this deterministic method. Observation source hashes and capture environment/runtime are:

```json
{
  "runtime_seconds": 5.362451799912378,
  "frequency_mhz": 40.00017089948058,
  "laser_period_ns": 24.99989318828099,
  "valid_pixels": 16395,
  "python": "3.11.11",
  "numpy": "2.4.6",
  "code_commit": "2d426ecbf14a46740b53054d66a522c692d89a89",
  "inputs": {
    "Convallaria_m2_1740751781_phasor_ch1.json": "a8bf0179f352ab2c6c78d0bd399545ab1fb6d5537f23dc06e5f12a7ef5af6615",
    "Fluorescein_Calibration_m2_1740751189_imaging.json": "aeebb074dbea6bff7578f409c7622b2f7f173bb23e5475d1436adedca7fc2eed"
  },
  "raw_sha256": "f3ae66619e8e847f1abba816b70dde12bed6008172d018dd3a2adf1d5c183f82"
}
```

Capture uses Python 3.11.11 and NumPy 2.4.6 on Windows AMD64 with single-thread OMP/OpenBLAS/MKL settings. PhasorPy runs its published v0.7 Windows compiled wheel whose matching source tag is archived; qmrpy runs directly from the exact original archived source. Upstream-only IO packages and dependencies are isolated in oracle and never become cleaned dependencies. Capture exit status is 0; no numerical warnings occurred on supported voxels. Published/historical example outputs in original download archives are retained only for provenance and were not treated as this capture's result.

## Equation, stage and independent test traceability

| Required published/algorithm stage | Original source or independent derivation | Local callable | Validation case |
| --- | --- | --- | --- |
| Normalized sine/cosine representation; Digman main text L14, supplementary transform referenced there | phasor.py:phasor_from_signal / _phasorpy.pyx | physics_model.normalized_phasor | test_upstream_stages; delta-harmonic sign/count test |
| Instrument phasor division and independent lifetime standard | lifetime.py:phasor_calibrate; phasor.py:phasor_center | physics_model.calibrate; single_exponential_phasor | test_upstream_stages; independent standard calibration test |
| Phase/modulation apparent summaries, modern extension explicitly distinguished from component resolution | lifetime.py:phasor_to_apparent_lifetime; _phasorpy.pyx:_phasor_to_apparent_lifetime | physics_model.lifetimes | test_upstream_stages; analytic single-exponential identity |
| Periodic convolution acquisition / transfer multiplication | published periodic harmonic model and convolution theorem | physics_model.normalized_phasor | test_circular_convolution_transfer |
| Full bound native image and fixed quality support | oracle_flim.py uses upstream transforms on exact released arrays | solvers.reconstruct | test_pipeline; test_metadata_and_no_spatial_binning |

No equation number is invented where the original source does not provide one. The task-specific full equations, signs, units, normalization and branches are self-contained in README and plan/approach.md.

## Ambiguities, adaptations and scientific limits

The first independent original-source capture, which preceded local extraction and fixed stage tolerances, pooled 4×4 spatial pixels. The final native-resolution selection was then independently recaptured with the same acceptance criteria before full native comparison; no cleaned function generated expected fixtures. Before final selection, that exploratory dataset was replaced with complete native 256×256 observations at root review; it is not the released task. The final fixed photon threshold is 200 (not the exploratory 2000). Native support has 16,395 valid pixels. The threshold and calibrated-domain G>0,S>0,G²+S²<1 use observations plus independent dye only. Internal upstream modulation lifetime returns zero for phasor radius>1, whereas a naive square root gives NaN; local branches were corrected after native stage parity exposed this difference. These cases are excluded by support, and no valid lifetime is clipped. Zero intensity gives undefined phasors, which remain internally NaN until masked. Exact frequency is 40.00017089948058 MHz from the stored period, not rounded to 40 MHz. Three-harmonic IO tutorials demonstrate additional library features; the selected published first-harmonic task intentionally evaluates phase/modulation summaries and retains the relevant phasor scatter, lifetime histograms and images, without claiming FRET/component decomposition or reproduction of unrelated original-paper cell experiments.

## Frozen fixtures and reference consumers

upstream_stages.npz contains 16×16 regularly sampled native locations (every sixteenth y,x coordinate), their original delay histograms, the pooled calibration histogram, and corresponding independently captured upstream mean/raw/calibrated phasors and lifetimes. Sampling only reduces fixture duplication; the actual bound reconstruction and Judge use every native pixel. Consumer: test/comparing_test/test_parity.py.

`evaluation/baseline_reference.npz` is the independently captured standard-method reconstruction used only by standalone Judge and full-pipeline tests; actual ground truth is unavailable, so `ground_truth.npz` is deliberately absent. `notebooks/reference_outputs/upstream.npz` identifies the corresponding independent upstream comparison used in the active review notebook. Its purpose is separately labeled notebook comparison, not a fresh reconstruction. `notebooks/reference_outputs/fresh_metrics.json` identifies fresh cleaned-code source/data/reference hashes, the generated submission and exact full-precision measured metrics; the notebook actively checks this ledger against a new reconstruction. Fresh arrays live in `output/reconstruction.npz`, avoiding a redundant fresh-array copy. Every frozen fixture/reference file has an explicit consumer.

## Tolerances, repeatability and metric provenance

Stage acceptance tolerances were fixed before local extraction: rtol=1e-10, atol=1e-12 for transforms/calibration/lifetimes. Exact copied inputs/constants are not validated by decimal rounding. Analytic tests independently exercise physical equations and units; agreement between source and local implementations alone is not treated as physical proof.

Original-source repeatability evidence on exact NPZ data:

```json
{
  "seconds": 0.06148789986036718,
  "upstream_repeat_max_abs": {
    "reconstruction": 0.0,
    "modulation_lifetime": 0.0
  },
  "bitwise_repeat_equal": {
    "reconstruction": true,
    "modulation_lifetime": true
  }
}
```

Repeated upstream execution is bitwise deterministic on this backend. Independent original and cleaned arithmetic pathways give the fresh errors below; the nonzero one-part-per-million reference-range tolerance is a conservative float64 numerical-equivalence allowance, fixed before Judge scoring. It is not inferred mechanically from the zero self-comparison error and is not a physical-accuracy claim. NCC is uncentered cosine similarity (no mean subtraction); NRMSE is RMS/reference dynamic range. Numerical scoring uses scaling internally to avoid norm overflow without changing these definitions. All-zero estimates have NCC=0 and fail, while degenerate zero/constant references are evaluator errors. Every required quantity must pass, and the independent forward/harmonic consistency metric must also pass.

Fresh cleaned Judge result, retaining full numeric precision:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_ncc": 0.9999999999999999,
    "reconstruction_nrmse": 1.6017386406817876e-16,
    "modulation_lifetime_ncc": 1.0,
    "modulation_lifetime_nrmse": 4.0853927644471627e-16,
    "observation_nrmse": 1.4988866383914643e-16
  },
  "thresholds": {
    "reconstruction_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "reconstruction_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "modulation_lifetime_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "modulation_lifetime_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "observation_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    }
  },
  "errors": []
}
```

There are no paper-reported scalar performance values silently compared against these numbers. Upstream repeat records, upstream on-bound arrays, and fresh cleaned metrics have distinct identities and paths. Judge and visualization metric functions are checked with the independent numerical example estimate [0,1,2], reference [0,2,4]: NCC=1, dynamic-range NRMSE=sqrt(5/3)/4.

## Runtime, tests and active notebook execution

The cleaned runtime is Python 3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]; NumPy 2.4.6; Matplotlib 3.11.1; platform Windows-10-10.0.26200-SP0. NumPy backend details are in the private environment JSON. Only NumPy and Matplotlib are required by shipped source/review plotting. SciPy is available in shared environment but not declared because these selected closed-form tasks do not import it. Domain libraries are absent from the cleaned interpreter. CPU execution is validated; GPU is not required. Maximum working arrays are well below 1.5 GiB; all notebook/kernel operations run serially with single-thread numeric settings.

Actual final commands and subprocess evidence:

- Exit 0; 0.258217 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m src.main`
- Exit 0; 0.284529 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`
- Exit 0; 2.117772 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m pytest test/ -v --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch_20261020\bio_medical\tmp_final_flim_phasor_lifetime`
- Exit 0; 4.565769 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m jupyter nbconvert --to notebook --execute notebooks/flim_phasor_lifetime.ipynb --output flim_phasor_lifetime_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101-research`

Pytest: **24 passed**, including per-function, original-source parity, independent physics, full pipeline, numeric metric and standalone Judge contracts. Missing files/keys, wrong shape/dtype, complex, NaN/Inf, zero and poor predictions, missing evaluator assets, caller-independent hidden paths, nested report creation and report-write failure are covered. The Windows default pytest temporary root denied access in this session; the recorded `--basetemp` points into the private archive and does not change tests or mathematical acceptance. This is a host temporary-directory exception, not a task runtime dependency.

Notebook timeout is 180 seconds, comfortably exceeding measured complete reconstruction and rendering on this CPU. Default system `python3` initially selected an unrelated Anaconda kernel; those exploratory runs are not environment validation. Final nbconvert explicitly used an isolated `imaging101-research` kernelspec via private JUPYTER_PATH with argv pointing to the validated research runtime. The executed notebook prints actual sys.executable and NumPy version. Benign Windows ZMQ selector-thread and debugger/frozen-module warnings are preserved in logs; there are no notebook error outputs. The final ledger-aware notebook was separately rerun with subprocess exit 0 in 4.535783 s, and actual research-runtime executable/NumPy2.4.6 plus active ledger agreement were programmatically verified. Canonical notebook contains the executed review outputs; the temporary `output/*_executed.ipynb` was archived and removed.

Visual review: Inspected reference/fresh/difference panels for both lifetimes, full-field count image, calibrated G/S scatter and universal semicircle, overlapping phase/modulation histograms and metric bars. Full image coverage and heterogeneous cell-wall structure are retained; lifetime differences are biologically interpretable harmonic summaries, not fitted component truth. Differences are around 10⁻¹⁴ ns and plot labels/colorbars are readable.

## Semantic plan/source and visibility audit

Reviewed every shipped callable against plan/design.md, and equations against actual numerical call roles. Checked conversion units, angle/frequency sign, normalization, branches, fixed support, parameter defaults, caller flow, file paths, side effects and output order. The exact Bloch diagnostic, where present, is explicitly distinguished from runtime inverse. The current runtime contains no unimplemented historical method. AST import audit is archived as `flim_phasor_lifetime_symbol_audit.json`; all local acquisition/inverse steps are readable NumPy mathematics, and there is no domain-solver API or vendored reconstruction framework. Tests only execute local entrypoint/Judge and consume frozen upstream numeric fixtures; they do not import oracle source. No cross-task import or hidden-answer-derived calibration exists.

The solver-visible allowlist consists solely of README, raw NPZ and requirements/shared environment information. README includes every NPZ key, acquisition metadata, units, axes and full output contract, with no hidden reference paths or GitHub domain URL. No metadata JSON exists. Original source repositories appear only in this maintainer document/NOTICE, never in the solver allowlist. Data are observations and legitimately available independent calibration/support, not a hidden target or privileged extra measurement.

## Mandatory completion gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/version, exact source revision, data hashes and stage mapping | passed | Frozen selection, hashes, table above; inaccessible optional paper PDFs explicitly distinguished |
| Original and cleaned code executed on exact bound observations | passed | Original capture log/record; repeated original source run; final main exit0 |
| Independent original-source parity and physical checks | passed | Comparing tests and independently analytic/Bloch/harmonic checks |
| Constants, units, precision, coordinates and conventions | passed | Source constants, README metadata and per-function/metadata tests |
| Standard entrypoint and fresh standalone Judge | passed | Final main/Judge subprocess exit0 and full report above |
| Invalid/poor submission and evaluator/report failure handling | passed | Contract tests, all failures measured with required exit codes |
| Public observations/environment suffice | passed | Complete forward model and declared acquisition/support in README |
| No truth, privileged calibration, seed/object or URL leakage | passed | Allowlist audit, real independent observation provenance |
| Complete acquisition metadata/output contract, no metadata JSON | passed | README key/output/metadata tables and inventory |
| Minimal shared dependencies and CPU execution without domain libs | passed | Two pinned required packages; separate oracle only |
| Local executable forward/inverse math and call-role audit | passed | Local NumPy code and archived AST/symbol review |
| Unit, metric, Judge, parity, physics and pipeline tests | passed | 24 passed in recorded final pytest subprocess |
| Active complete notebook reconstruction/execution and visual inspection | passed | Final research-kernel nbconvert exit0; reviewed private PNGs |
| Distinct metric/result provenance and justified boundaries | passed | Upstream arrays, fresh measurement ledger, reruns, config/report |
| README/plans match source signatures, units and execution flow | passed | Semantic source review and metadata/forward tests |
| Template inventory, cleanup and original license notices | passed | Inventory/hashes below; NOTICE; private-only original artifacts |

No mandatory gate is marked failed or unverified. Scientific limitations above remain limitations of the selected method, not unexecuted validation claims.

## Numeric artifact hashes

| Task artifact | SHA256 |
| --- | --- |
| data/raw_data.npz | f3ae66619e8e847f1abba816b70dde12bed6008172d018dd3a2adf1d5c183f82 |
| evaluation/baseline_reference.npz | 1af4705b2af4489bda780821e57d0b2a8a78a39e43b3d21c53f6a9152e15f796 |
| evaluation/metrics.json | b660b047a744863bdc639c0755a7d4410b1cc9d1845105de14882ed398f98699 |
| notebooks/reference_outputs/fresh_metrics.json | ced10a817dd0bd75e785dbcab7695a01a790fa02ae16e70989dd8576c9852c4c |
| notebooks/reference_outputs/upstream.npz | 1af4705b2af4489bda780821e57d0b2a8a78a39e43b3d21c53f6a9152e15f796 |
| output/judge_report.json | d6377b06b13bf046808d7e1f563b8d5af743e908f9667d7b2ed954d37112e092 |
| output/reconstruction.npz | af87b553f846f7311287165a0238bdce791549c686eb87c6f3ca59de2ec74fc3 |
| test/fixtures/upstream_stages.npz | a6d24f5cd636aee56b10e53a365cc3b511d9590de8b013368fb975514fc1016c |

## Final review inventory

- `data/raw_data.npz`
- `evaluation/baseline_reference.npz`
- `evaluation/judge.py`
- `evaluation/metrics.json`
- `notebooks/flim_phasor_lifetime.ipynb`
- `notebooks/reference_outputs/fresh_metrics.json`
- `notebooks/reference_outputs/upstream.npz`
- `NOTICE`
- `output/judge_report.json`
- `output/reconstruction.npz`
- `plan/approach.md`
- `plan/design.md`
- `README.md`
- `requirements.txt`
- `src/__init__.py`
- `src/main.py`
- `src/physics_model.py`
- `src/preprocessing.py`
- `src/solvers.py`
- `src/visualization.py`
- `test/comparing_test/test_parity.py`
- `test/fixtures/upstream_stages.npz`
- `test/unit_test/test_contract.py`
- `test/unit_test/test_physics.py`
- `test/validation.md`


## Independent parent acceptance (2026-10-20)

Parent independently reran all 24 tests with exit 0, ran the standalone Judge from a foreign working directory (exit 0), reviewed source mathematics, plans, output and public-input contracts, and inspected all 3 actual notebook figures. After display-only label/unit fixes, the complete active notebook was re-executed successfully and updated comparisons were inspected. No numerical source changed after the parent tests. Structural/import/public-input audit found no errors. See private batch parent-audit JSON and test logs. Accepted for this ten-task batch; method-reference agreement is not independent physical truth.
