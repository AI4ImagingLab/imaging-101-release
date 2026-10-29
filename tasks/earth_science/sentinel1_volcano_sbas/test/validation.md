# Validation and provenance: real Fernandina SBAS

This maintainer-only record documents a numerical port of selected published MintPy stages on the **complete distributed real 450 x 600 scene, 98 acquisitions and 288 interferograms**. It does not claim independent geophysical truth or reproduction of all experiments in the paper. No synthetic observations, simulated eruptions or target-generated calibration are used. The full scene and five-quantity contract were fixed before cleaned implementation. No cross-task imports or skills were used.

## 1. Original sources and bound observations

| Item | Exact identity and role |
|---|---|
| Original repository | https://github.com/insarlab/MintPy |
| Exact revision | `9e0d46b2593483ba89452102efd21c1575662190`, actually verified against remote HEAD during this cleaning; not a claim based only on a local tracking revision |
| Published paper | Yunjun, Fattahi and Amelung (2019), *Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction*, Computers & Geosciences133,104331; https://doi.org/10.1016/j.cageo.2019.104331 |
| Read paper version | EarthArXiv postprint version3,2019-08-07,95pages including supplement; https://doi.org/10.31223/osf.io/9sz6m ; https://eartharxiv.org/repository/view/750/ ; PDF https://eartharxiv.org/repository/object/750/download/1650/ |
| Real dataset | https://zenodo.org/records/3952953 ; https://doi.org/10.5281/zenodo.3952953 ; provider's Fernandina Sentinel-1 descending-track128 observations |
| Download verification | `Fernandina.complete.tar.xz`,750610904bytes; provider MD5 `089fd092ce705a8aec718d36a2949d5b` verified before use |
| Acquisition | Real Sentinel-1 A/B,2014-12-13 to2018-06-19;98dates/288ISCE-topsStack interferograms; complete distributed450azimuth x600range scene, source offset row1150/column1070 |
| Independent external weather | Original provider `mintpy/inputs/ERA5.h5` LOS atmospheric product, originally referenced to20180619; re-referenced to first date and source spatial reference before subtraction |
| Licenses | Original code GPL-3.0-or-later; dataset CC BY4.0. Original code notices remain in adapted source. Complete legal texts are included below instead of standalone task LICENSE/NOTICE |

The paper illustrates a475edge940triangle network, whereas this independently distributed real tutorial has288edges/286triangles. This task preserves the complete latter network. It does not claim the paper's acquisition or reproduce unavailable GNSS, other atmospheric products, bridging experiments or broader geographic acquisitions. The selected algorithm stages and actual comparisons are explicitly mapped below.

The original phase is the second band of little-endian float32 BIL `(450,2,600)[:,1]`; incidence is the first band of the LOS geometry raster. Coherence float32, connected labels uint8 and geolocation float64 retain original values/precision. Per-date perpendicular baseline is the mean of the two supplied swath-average values relative to20141213. Lossless original-to-NPZ conversion adds one leading batch axis. No additional cropping, downsampling, synthetic noise or temporal selection is performed.

The public all-edge-valid mask is exactly the conjunction of original positive connected-component labels across288edges (159762pixels). It is not derived from reconstruction or hidden truth. The independent supplied weather product is not estimated from the unknown deformation. The private original-code run constructs an HDF5 container of these same original raster values so original numerical functions can execute without requiring ISCE in the cleaned runtime. This changes the container, not the acquisition or source mathematics. Raw sample equality and the complete mask formula are tested.

## 2. Private archive and environment

All archive-relative paths below resolve under `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task05`, outside the released task. No cleaned source, Judge, test or notebook imports private capture code or upstream source. `archive_inventory.json` indexes extracted upstream files; `final-evidence-ledger.json` records final important evidence hashes. Historical screening records are superseded by this completed validation and final execution statuses.

| Private path | Documented consumer and role |
|---|---|
| `MintPy.zip`, `MintPy-9e0d46b2593483ba89452102efd21c1575662190/` | Exact unmodified original checkout/archive, isolated oracle imports and original full license |
| `Fernandina.complete.tar.xz`, `data-source/FernandinaSenDT128/` | Verified provider archive and extracted exact input data for original capture/public packaging |
| `Yunjun2019-preprint.pdf`, `paper-extracted.txt`, `paper-crossref.json` | Selected paper, extracted mathematical text and independently retrieved bibliographic metadata |
| `CC-BY-4.0.html`, `zenodo-record.json` | Original downloaded full legalcode and dataset metadata |
| `oracle_capture.py`, `oracle_capture.log`, `oracle/result.json` | Original full selected pipeline execution and original measured stage outputs |
| `oracle/stack.h5`, `oracle/network.npz`, `oracle/final.npz`, `oracle/geometry.npz`, `oracle/closure_regions.npz` | Original lossless observation/closure container, full network/correction outputs and independently captured geometry/regions |
| `oracle_rerun.py`, `oracle_rerun.json` | Original numerical rerun on9real pixels, original LASSO sample capture and expanded fixture evidence |
| `oracle_uniform.py`, `oracle_uniform.json` | Original unweighted minimum-velocity SBAS on all98dates/all159762common pixels |
| `package_bound.py`, `bound_data_manifest.json` | Original public/reference packaging and initial hashes; expanded9pixel fixtures use final hash ledger |
| `cleaned_driver.py`, `cleaned_status.json`, `cli.log`, `judge.log` | Actual fresh standard CLI/Judge commands, runtimes, exits and logs |
| `oracle/local_closure_probe.npz`, `full_closure_parity.json` | Fresh cleaned full spatial repair compared independently against original HDF5 values |
| `audit_lasso.py`, `lasso_discrepancy.json`, `lasso_nullspace_check.json`, `lasso_equivalence_audit.json` | Investigated backend difference, objective/KKT/dual certificates, actual closure-image difference and independently constructed null directions |
| `pytest.log`, `pytest-final.log`, `pytest-measurements.log` | Initial stronger assertion failure retained, final full suite and fresh-summary consumer verification |
| `build_notebook.py`, `notebook_driver.py`, `executed.ipynb`, `notebook.log`, `notebook_status.json`, `figure-*.png` | Notebook authoring/actual complete nbconvert capture and rendered figure inspection, outside final task |
| `cleaned-pip-freeze.txt`, `oracle-pip-freeze.txt`, `cleaned-numerical-environment.txt`, `oracle-numerical-environment.txt` | Actual interpreter/package/backend provenance |

Validated CPU system: Windows-10-10.0.26200-SP0(build26200), Python3.11.11 MSCv.1943 AMD64; NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. BLAS/LAPACK/compiler versions are preserved by `numpy.show_config()` in the environment records. All numeric jobs fix `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; network inversion uses4Python row workers. No GPU validation is claimed.

Cleaned interpreter is project-maintainer shared `research-runtime/Scripts/python.exe`. The private `oracle-env` shares the validated numerical packages through a private site path and additionally installs upstream-only CVXOPT, h5py, scikit-image, cartopy and pyresample. Their actual versions are in the freeze record. Only the original isolated oracle imports MintPy. Task runtime requirements are only NumPy/SciPy/Matplotlib; pytest/Jupyter/IPython are shared development tooling. No upstream domain dependency is installed into the cleaned runtime as part of this task, and no untested platform compatibility is asserted.

Reproducible commands (full interpreter paths, working directories and exits are preserved in status records):

```powershell
# Private archive working directory; original source selected by private capture sys.path
oracle-env/Scripts/python.exe oracle_capture.py
oracle-env/Scripts/python.exe oracle_rerun.py
oracle-env/Scripts/python.exe oracle_uniform.py
# Task working directory, cleaned shared environment
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
jupyter nbconvert --to notebook --execute notebooks/sentinel1_volcano_sbas.ipynb --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=imaging-101-research --output executed.ipynb --output-dir <private-archive>
```

The private oracle fixes `np.random.default_rng` to seed20261018 at the capture boundary only while the unmodified original region sampler executes, then restores it. This is deterministic estimator sampling, not synthetic acquisition generation or a source mathematics patch. It runs actual original weighted `estimate_timeseries` for every pixel, original spatial deramp and original DEM estimator. Import screening was followed by this actual complete exact-data run; an import was not treated as completion.

## 3. Equation / implementation / test traceability

| Paper mathematics or selected stage | Original file/function | Local executable function | Independent evidence |
|---|---|---|---|
| Pair incidence/integrated-velocity design, Eq1-2 | `objects/stack.py:ifgramStack.get_design_matrix4timeseries` | `physics_model.network_matrices` | Frozen exact A/B/dt; independently known constant velocity integration |
| Triangle closure/integer cycles, Eq8-9 | `get_design_matrix4triplet`; `unwrap_error_phase_closure.py` | `network_matrices`, `closure_integers` | Frozen286x288C; graph identity and known integer-cycle/noise test |
| Common/equal-label spatial regions | `objects/conncomp.py`; `get_common_region_int_ambiguity` | `component_labels`, `clean_components`, `repair_closure` | Original coordinates/medians; independent unequal-adjacent-label, area/erosion tests |
| Sparse closure objective, Eq10-11,alpha.01 | `utils/solvers/l1regls.py:l1regls` generic CVXOPT | `lasso_ambiguities`, explicit ADMM/soft threshold | Original100continuous samples; objective/KKT, independently derived null-space alternatives and exact region medians; discrepancy below |
| Spatial2pi correction propagation | `correct_unwrap_error_phase_closure` | `repair_closure` | All77760000phase values bitwise identical; both-region frozen samples |
| Coherence-weighted minimum-velocity inversion, Eq2/5 | `ifgram_inversion.estimate_timeseries`, `simulation/decorrelation.coherence2weight` | `estimate_pixel`, `invert_network` |9real bound pixels/all98dates; independent original rerun; complete network outputNRMSE0 |
| Temporal residual coherence, Eq3 | `estimate_timeseries` | `estimate_pixel` | Original quality fixtures, insufficient-network tests and full-frame uniform comparison |
| LOS and independent atmosphere, Eq4 | Original wavelength convention and suppliedERA5 product | `preprocessing.reference_atmosphere`, `reconstruct` | Space/time gauge/no-mutation tests; original correction-stage snapshots/full final arrays |
| Reference-anchored linear spatial plane | `objects/ramp.py:deramp` | `deramp_timeseries` | Analytic physical-plane test and original full-frame four-date ramp/deramped maps |
| DEM/deformation nuisance fit | `dem_error.estimate_dem_error`, `utils/time_func.get_design_matrix4time_func` | `dem_design`, `temporal_model`, `correct_dem` | Original98x6G; known coefficient/units test; acceleration and strict event-boundary tests; stage snapshots |
| Scientific unweighted comparison | Original `estimate_timeseries(weight_sqrt=None,min_norm_velocity=True)` | `unweighted_common_baseline` | Original complete159762pixel/all98date run; full-frame identified maps/quality/traces |
| Submission/independent evaluation | Benchmark-specific contract | `main.save_submission`, independent `evaluation/judge.py` | Fresh full CLI, independent known metrics, all5quantities, poor/malformed/error/other-CWD tests |

Scene-mean DEM geometry follows original selected convention: mean sin(incidence)=.5953686237335205, mean range810012.77671583m. The dimensionless baseline geometry column multiplies elevation error in metres. Network design uses float32 elapsed days/365.25; deformation basis uses calendar-year+(day-of-year-1)/365.25 in float64 before float32 basis storage. These conventions are deliberately not silently unified. Quadratic column tau-squared/2 makes its coefficient acceleration; event indicators use strictly `date > event`. Combined G is float64. Weighted SVD uses square root of coherence floored at.05, source missing-data/redundancy rules and rcond1e-5; DEM fit uses cond1e-8. Ramp uses float32 row/column/one, source `nobs==288 & quality>=.7`, pinv rcond1e-15, reference anchoring and source zero-value preservation. These factors/signs/units are checked semantically as well as by signatures.

## 4. Investigated LASSO backend discrepancy

The initial expanded suite had33passed/1failed because it required every rounded individual coefficient to equal the generic original CVXOPT backend. That initial failure remains in `pytest.log`. Investigation preserved observations, selected objective, region/final outputs and all final Judge thresholds.

Both backends minimize `sum(((-C)@u-q)^2)+.01*sum(abs(u))`. The graph is rank deficient, and the paper explicitly identifies nonuniqueness. Local ADMM converges in380iterations. Nine of28800sample coefficients round differently across4samples. Local-minus-original coefficient Frobenius norm is.7920480220340274; closure-image Frobenius norm.0050718090030433295, maximum absolute.00037046764519235503, maximum per-sample L2norm.0007510775185427522. **The actual backend difference is not exactly a null-space vector.**

Maximum proximal fixed-point KKT residual is6.979500516597625e-8(local),6.045493026786353e-4(original). Feasible-dual-certified maximum primal-dual gaps are2.2263055603488624e-6(local),1.887738682382345e-5(original). Local objective is lower for every sampled RHS; maximum local-minus-original objective is-7.920037176090133e-7. Original stopping precision therefore also contributes. This is not described as merely roundoff at exact half integers.

An independent active-support null-space construction gives support/nullity62/9,102/21,38/1,102/21 for samples25,28,30,32. Nonzero sign-preserving perturbations retain closure predictions within1.2e-16 and L1norm within3.6e-15, establishing genuine equal-objective alternatives. Final tests check those independent directions, convex objective/KKT and exact scientific region medians; they do not demand CVXOPT's arbitrary continuous representative or trajectory.

All288median integer offsets match in both regions: reference region155840pixels/zerooffsets, second region2779pixels/28nonzeroedges. Full spatially repaired phase matches bitwise across77760000values (`full_closure_parity.json`). Five full final output arrays haveNRMSE0. These checks establish numerical equivalence on this acquisition, not unknown physical truth or equality of all intermediates. The nine coefficients and their distances to half-integer boundaries are appended below directly from the comprehensive audit.

## 5. Measurements and fixed evaluation boundaries

| Measurement class | Exact condition | Observed evidence |
|---|---|---|
| Historical paper | Different475edge940triangle network and wider experiments | Context only; no historical metric relabelled as this run |
| Reproduced original selected pipeline | Exact full bound observations, original unmodified functions/revision, original environment, deterministic sampler boundary | Exit0;290.12101650238037s; final global range[-.24134954810142517,.4597126245498657]m; meanquality.6597128510475159;159447plane-fit pixels |
| Independent original repeatability |9real pixels incl. both common regions and invalid corner | Nine original float32 LOS differences0; original LASSO capture3.293489456176758s |
| Original unweighted comparison | Same corrected observations, all98dates/all159762common pixels | Exit0;14.73247742652893s; full common-region histories and identified comparison maps/quality/traces |
| Fresh cleaned CLI | Exact publicNPZ, local selected pipeline, original full hidden reference | Exit0;442.6107802391052s; Judge exit0/10.128461360931396s; every5quantityNRMSE0, NCCapproximately1 |
| Fresh notebook | Complete local reconstruction and uniform comparison run again, identified original comparisons | Final actual execution/figure evidence appended after completion; cache does not replace active reconstruction |

There is **no ground_truth.npz**: actual independent deformation/DEM truth is unavailable. `evaluation/baseline_reference.npz` contains the independently run original selected method and is never relabelled truth. Original numeric fixtures/stages were frozen before extraction of local mathematics.

Before first cleaned full run, `define_metrics.py` fixed each quantity's NCC>=.99999 and dynamic-rangeNRMSE<=1e-5. Original self-reference gives1/0, so generic multiplicative default boundaries are unsuitable. Independent original repeatability produced zero float32 error; nonzero1e-5range tolerance allows ordinary backend variation. It was not relaxed after cleaned results. It measures method agreement, not external geophysical accuracy.

Additional circular observation coherence checks the apparent `network_timeseries`, not the corrected movie with atmosphere/DEM/ramp removed. Original reference-pixel phase differences and predicted LOS graph differences use the declared phase sign; complex residuals are averaged across288edges then magnitudes across the fixed public common mask. Integer2pi corrections do not change this check. Original value.994020419696998 determines prespecified.99relative boundary.984080196. This check does not establish atmosphere/deformation identifiability.

NCC is flattened uncentred cosine similarity. NRMSE is RMS error divided by reference dynamic range. There is no crop, magnitude conversion, scaling fit or alignment. All98dates and5parameter planes are scored on159762publicly known common pixels. Exact declared float32 shapes and finiteness are required throughout the arrays. The Judge imports no solver, rejects malformed files/keys/shapes/dtypes/complex/NaN/Inf and unavailable assets, and returns2 for report-write/evaluator errors. Valid all-zero output fails with1. Other-working-directory tests verify reference/user path resolution. Known analytic metric cases validate both notebook helper and independent Judge; degenerate zero-norm/range cases yield explicit errors, neverNaNJSON.

## 6. Tests and semantic review

Final full `python -m pytest test/ -v` passed **35tests in90.03seconds** (`pytest-final.log`): unit physics, raw/constants parity, original frozen intermediates, LASSO mathematical equivalence, full spatial repair/uniform comparison, six correction-stage snapshots, all5fresh saved outputs and strict Judge contracts. A compact fresh CLI summary is additionally consumed by the full-output test and checked in a targeted rerun (`pytest-measurements.log`). Analytic synthetic arrays used only for known-coefficient unit tests are not benchmark data or fabricated ground truth.

The notebook has exactly six ordered H2s: Problem, Data, Physics, Method, Results, Conclusion. It actively runs complete `reconstruct`, actively recomputes the uniform comparison from fresh repaired phase, saves/judges the fresh outputs and displays data, graph, ambiguity, stages, parameters, traces and weighted/uniform comparisons. No absent method is filled with zeros. First successful479.9357588291168s execution had no embedded PNG because inherited Agg backend rendered figure representations; artifact inspection caught this. Notebook-only `%matplotlib inline` was added and full reconstruction genuinely executed again. Final capture/inspection evidence is appended below. Library imports do not select a backend; `src.main` import preserves notebook backend, explicitly tested.1800sper-cell timeout is justified by measured approximately7minute full CPU reconstruction and concurrent execution overhead.

Semantic review checked actual callable signatures/shapes/returns/defaults/paths against design and the forward signs, cycle orientation/rounding, LASSO normalization, velocity integration, LOS conversion, time coordinates, DEM geometry/units, quadratic factor, event strictness, ramp/missing-data rules and metric conventions against paper/source. Plans document current shipped flow, not unimplemented historical experiments. Explicit ADMM is a disclosed general optimizer backend substitution for the same convex objective, with its nonunique coefficient discrepancy quantified.

Import/call-role audit covers source, Judge, test and notebook: local executable mathematics constructs acquisition operators, cycle correction, explicit ADMM, weighted inversion, ramp and nuisance updates using only permitted general NumPy/SciPy operations. There are no MintPy/domain solvers, opaque vendored frameworks, private oracle imports or cross-task imports in the cleaned runtime. No generator is supplied because no observations are generated. PublicREADME/requirements contain noGitHubURLs or hidden reference paths/keys. Metadata and full output contract are self-contained; maintainer references are separate.

## 7. Assets and completion gates

Hash tables below identify original data/fixtures and fresh results separately. `upstream_measurements.npz` is consumed by six-stage parity tests; `upstream_uniform_sbas.npz` by tests and notebook; `cleaned_measurements.npz` by the full-output test. Expanded9pixel `upstream_stages.npz` supports raw/network/basis tests; regions and100sample continuous LASSO fixtures support respective stage tests. None is unused upstream copied source. Original traces retained with stage maps are identified reference measurements, not fresh results.

| Mandatory gate | State | Evidence |
|---|---|---|
| Exact paper/revision/data hashes/stage mapping | Passed | Sections1-3 and final ledger |
| Original and cleaned full bound executions | Passed | Original290.121s and CLI442.611s,exit0 |
| Independent parity and physics | Passed |35tests/full phase/full final quantities; intermediate discrepancy investigated explicitly |
| Constants/units/precision/coordinates | Passed | Original fixtures, analytic conventions and README tests |
| Standard CLI and fresh Judge | Passed | Five declared quantities/eleven metrics,exit0 |
| Judge invalid/poor/evaluator/report-write contracts | Passed | Strict standalone tests |
| Public information sufficient/no privileged leak | Passed | Source observations, independent weather and observation mask; public schema audit |
| Metadata/output contract/no metadataJSON | Passed | README exact sections/tables and constant checks |
| Minimal runtime independent of domain packages | Passed | Shared numeric environment and import/call audit |
| Local executable inverse/acquisition mathematics | Passed | Traceability and semantic review |
| Full unit/parity/physics/pipeline tests | Passed |35passed/90.03s and summary-consumer rerun |
| Active six-section executed notebook/rendered comparisons | Passed | Actual nbconvert495.015826s,10executed cells,7inspected embedded PNGs,0errors |
| Consistent metric and result provenance | Passed | Historical/original/fresh distinguished; thresholds fixed before fresh result |
| README/plans match code/math | Passed | Full signature and semantic review; disclosed LASSO discrepancy |
| Final inventory/legal/temporary cleanup | Passed | No standalone LICENSE/NOTICE, complete legal texts below; temporary execution/render assets remain private |

## 8. Original copyright and complete legal texts

Adapted numerical stages originate in MintPy, Copyright(c)2013-2026 Zhang Yunjun, Heresh Fattahi and MintPy contributors, GPL version3 or later. Changes extract local numerical functions, convert interfaces/storage, implement explicit ADMM for the same closure objective and add tests/docs. Original attribution remains in adapted source. Dataset attribution: Fernandina Sentinel-1 descending-track128 observations, Zenodo3952953, DOI10.5281/zenodo.3952953, CC BY4.0. Changes are documented lossless container conversion and referencing, not invented physical observations.

Complete upstream GPL and downloaded CC BY4.0 legalcode text follow. They are embedded here to honor source notices while shipping no standalone task LICENSE/NOTICE.

### Quantified nine coefficient discrepancies

| Edge | Sample | Local | Original | Local distance to half integer | Original distance |
|---|---|---|---|---|---|
| 8 | 28 | -0.483250850550605 | -0.6078238434857363 | 0.016749149449394984 | 0.10782384348573626 |
| 12 | 28 | -0.42596115135538787 | -0.5456092252213625 | 0.07403884864461213 | 0.04560922522136246 |
| 78 | 25 | -0.5013510674702415 | -0.47845091614410346 | 0.001351067470241496 | 0.021549083855896545 |
| 86 | 25 | 0.580326692028775 | 0.4816117136243423 | 0.08032669202877496 | 0.018388286375657703 |
| 91 | 25 | -0.5452287097416348 | -0.482572643314617 | 0.04522870974163484 | 0.017427356685382978 |
| 95 | 32 | -0.5582583416632417 | -0.49603705209827764 | 0.05825834166324173 | 0.003962947901722358 |
| 264 | 30 | -0.499681818181855 | -0.5001581189172031 | 0.00031818181814502466 | 0.00015811891720307614 |
| 266 | 28 | 0.5093095797760171 | 0.45883759279279307 | 0.009309579776017096 | 0.04116240720720693 |
| 268 | 30 | -0.500136363636327 | -0.4996589349153058 | 0.0001363636363269638 | 0.00034106508469422536 |

### Fresh CLI metrics

| Metric | Value |
|---|---|
| `reconstruction_ncc` | 0.9999999999999998 |
| `reconstruction_nrmse` | 0.0 |
| `dem_error_ncc` | 1.0 |
| `dem_error_nrmse` | 0.0 |
| `deformation_parameters_ncc` | 1.0000000000000002 |
| `deformation_parameters_nrmse` | 0.0 |
| `temporal_coherence_ncc` | 1.0 |
| `temporal_coherence_nrmse` | 0.0 |
| `network_timeseries_ncc` | 1.0 |
| `network_timeseries_nrmse` | 0.0 |
| `observation_coherence` | 0.994020419696998 |

### Final data/reference/fixture hashes

| Task path | Bytes | SHA256 |
|---|---|---|
| `data/raw_data.npz` | 581983494 | `00b96aabb032e4e6b06ca062a541d29529c62ca841a3931136ff34720e3ce1fd` |
| `evaluation/baseline_reference.npz` | 170626618 | `be4158650f2eabddab561b21e7387d136592ce1fb6d5d5dbed592c32a279a4c6` |
| `notebooks/reference_outputs/cleaned_measurements.npz` | 827 | `3f9f70b20ba43a79e2716c56557b4a6884e469da9e8cc6018768b2b5aecc8e4f` |
| `notebooks/reference_outputs/upstream_measurements.npz` | 16924118 | `f0bab3f728f0b829a36400e616b7361224a10e513dd01d7041ebd9b234dcefa0` |
| `notebooks/reference_outputs/upstream_uniform_sbas.npz` | 2203736 | `8e58481ca6ec8fafa9ded78153d0c71e52f01cdc0dcae48a90d264903a69d3b9` |
| `test/fixtures/upstream_lasso.npz` | 35146 | `2676558c218d369cae0c7d410d7540a2954ff260e0df8560cc93b2983eb9b37b` |
| `test/fixtures/upstream_regions.npz` | 8834 | `640bb8c260fbf141bb6105899e9d3654a3ee7730751103501c8162851b5bdfd9` |
| `test/fixtures/upstream_stages.npz` | 602506 | `7a7654efb6e81c29060dac740ea230b4274a763d1cb7af194599708625278c04` |
| `output/judge_report.json` | 1566 | `000fac8df5b79d362dfd14e4320a2996f36e14338c3ea860b77e6f3319baec8f` |
| `output/reconstruction.npz` | 170626618 | `be4158650f2eabddab561b21e7387d136592ce1fb6d5d5dbed592c32a279a4c6` |

### Complete original GNU GPL text

```text
Copyright (c) 2016, Zhang Yunjun and Heresh Fattahi
Copyright (c) 2013, Heresh Fattahi

This is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

A copy of the GNU General Public License is included below.
For further information, see <http://www.gnu.org/licenses/>.
---------------------------------------------------------------------------

                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<http://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<http://www.gnu.org/philosophy/why-not-lgpl.html>.

```

### Complete CC BY4.0 legalcode text

Retrieved from https://creativecommons.org/licenses/by/4.0/legalcode.en ; original HTML is archived. Entire readable text follows.

```text
Legal Code - Attribution 4.0 International - Creative Commons

 Skip to content

 Languages available

 Bahasa Indonesia

 Basque

 dansk

 Deutsch

 eesti

 English

 español

 français

 frysk

 Hrvatski

 italiano

 latviešu

 Lietuviškai

 Māori

 Nederlands

 norsk

 polski

 Português

 Română

 slovensky

 Slovenščina

 suomi

 svenska

 Türkçe

 česky

 Ελληνικά

 Русский

 Українська

 العربيّة

 日本語

 简体中文

 繁體中文

 한국어

Creative Commons
 Menu

 Who We Are
 Expand

Strategic Plan

Team

Governance

Opportunities

 Annual Reports & Financials

History

Press

 What We Do
 Expand

 Build 

 Open Infrastructure

 Expand

CC Licenses

CC Signals

Public Domain

Chooser

FAQs

 Implement 

 the Commons

 Expand

Where CC Makes An Impact

Resources

 Search the Commons

 Engage 

 the People

 Expand

 Training + Webinars

Advocacy

Community

Events

Blog

 Support Us
 Expand

 Make a Gift

Open Infrastructure Circle

Donor FAQ

 Donate

Attribution 4.0 International

 CC BY 4.0

Legal Code

 Attribution 4.0 International

 Section 1 – Definitions. 

 Section 2 – Scope. 

 Section 3 – License Conditions. 

 Section 4 – Sui Generis Database Rights. 

 Section 5 – Disclaimer of Warranties and Limitation of
 Liability.

 Section 6 – Term and Termination. 

 Section 7 – Other Terms and Conditions. 

 Section 8 – Interpretation. 

Canonical URL
 https://creativecommons.org/licenses/by/4.0/

Other formats

Plain Text

RDF/XML

 See the deed

 Version 4.0 • See the
 errata page for any corrections and
 the date of change

 About the license and Creative Commons

 Creative Commons Corporation ("Creative Commons") is not a law firm
 and does not provide legal services or legal advice. Distribution of
 Creative Commons public licenses does not create a lawyer-client or
 other relationship. Creative Commons makes its licenses and related
 information available on an "as-is" basis. Creative Commons gives no
 warranties regarding its licenses, any material licensed under their
 terms and conditions, or any related information. Creative Commons
 disclaims all liability for damages resulting from their use to the
 fullest extent possible.

 Using Creative Commons Public Licenses

 Creative Commons public licenses provide a standard set of terms and
 conditions that creators and other rights holders may use to share
 original works of authorship and other material subject to copyright
 and certain other rights specified in the public license below. The
 following considerations are for informational purposes only, are
 not exhaustive, and do not form part of our licenses.

 Considerations for licensors

 Our public licenses are intended for use by those authorized to give
 the public permission to use material in ways otherwise restricted
 by copyright and certain other rights. Our licenses are irrevocable.
 Licensors should read and understand the terms and conditions of the
 license they choose before applying it. Licensors should also secure
 all rights necessary before applying our licenses so that the public
 can reuse the material as expected. Licensors should clearly mark
 any material not subject to the license. This includes other
 CC-licensed material, or material used under an exception or
 limitation to copyright.
 More considerations for licensors.

 Considerations for the public

 By using one of our public licenses, a licensor grants the public
 permission to use the licensed material under specified terms and
 conditions. If the licensor’s permission is not necessary for any
 reason–for example, because of any applicable exception or
 limitation to copyright–then that use is not regulated by the
 license. Our licenses grant only permissions under copyright and
 certain other rights that a licensor has authority to grant. Use of
 the licensed material may still be restricted for other reasons,
 including because others have copyright or other rights in the
 material. A licensor may make special requests, such as asking that
 all changes be marked or described. Although not required by our
 licenses, you are encouraged to respect those requests where
 reasonable.
 More considerations for the public.

Attribution 4.0 International

 By exercising the Licensed Rights (defined below), You accept and
 agree to be bound by the terms and conditions of this Creative
 Commons Attribution 4.0 International Public License ("Public
 License"). To the extent this Public License may be interpreted as a
 contract, You are granted the Licensed Rights in consideration of
 Your acceptance of these terms and conditions, and the Licensor
 grants You such rights in consideration of benefits the Licensor
 receives from making the Licensed Material available under these
 terms and conditions.

Section 1 – Definitions.

 Adapted Material
 means material subject to Copyright and Similar Rights that is
 derived from or based upon the Licensed Material and in which the
 Licensed Material is translated, altered, arranged, transformed,
 or otherwise modified in a manner requiring permission under the
 Copyright and Similar Rights held by the Licensor. For purposes of
 this Public License, where the Licensed Material is a musical
 work, performance, or sound recording, Adapted Material is always
 produced where the Licensed Material is synched in timed relation
 with a moving image.

 Adapter's License
 means the license You apply to Your Copyright and Similar Rights
 in Your contributions to Adapted Material in accordance with the
 terms and conditions of this Public License.

 Copyright and Similar Rights
 means copyright and/or similar rights closely related to copyright
 including, without limitation, performance, broadcast, sound
 recording, and Sui Generis Database Rights, without regard to how
 the rights are labeled or categorized. For purposes of this Public
 License, the rights specified in Section
 2(b)(1)-(2) are not Copyright and Similar
 Rights.

 Effective Technological Measures
 means those measures that, in the absence of proper authority, may
 not be circumvented under laws fulfilling obligations under
 Article 11 of the WIPO Copyright Treaty adopted on December 20,
 1996, and/or similar international agreements.

 Exceptions and Limitations
 means fair use, fair dealing, and/or any other exception or
 limitation to Copyright and Similar Rights that applies to Your
 use of the Licensed Material.

 Licensed Material
 means the artistic or literary work, database, or other material
 to which the Licensor applied this Public License.

 Licensed Rights
 means the rights granted to You subject to the terms and
 conditions of this Public License, which are limited to all
 Copyright and Similar Rights that apply to Your use of the
 Licensed Material and that the Licensor has authority to license.

 Licensor means the
 individual(s) or entity(ies) granting rights under this Public
 License.

 Share means to
 provide material to the public by any means or process that
 requires permission under the Licensed Rights, such as
 reproduction, public display, public performance, distribution,
 dissemination, communication, or importation, and to make material
 available to the public including in ways that members of the
 public may access the material from a place and at a time
 individually chosen by them.

 Sui Generis Database Rights
 means rights other than copyright resulting from Directive 96/9/EC
 of the European Parliament and of the Council of 11 March 1996 on
 the legal protection of databases, as amended and/or succeeded, as
 well as other essentially equivalent rights anywhere in the world.

 You means the
 individual or entity exercising the Licensed Rights under this
 Public License. Your has a corresponding meaning.

Section 2 – Scope.

 License grant.

 Subject to the terms and conditions of this Public License,
 the Licensor hereby grants You a worldwide, royalty-free,
 non-sublicensable, non-exclusive, irrevocable license to
 exercise the Licensed Rights in the Licensed Material to:

 reproduce and Share the Licensed Material, in whole or in
 part; and

 produce, reproduce, and Share Adapted Material.

 Exceptions and Limitations. For the avoidance
 of doubt, where Exceptions and Limitations apply to Your use,
 this Public License does not apply, and You do not need to
 comply with its terms and conditions.

 Term. The term of this Public License is
 specified in Section 6(a).

 Media and formats; technical modifications allowed. The Licensor authorizes You to exercise the Licensed Rights
 in all media and formats whether now known or hereafter
 created, and to make technical modifications necessary to do
 so. The Licensor waives and/or agrees not to assert any right
 or authority to forbid You from making technical modifications
 necessary to exercise the Licensed Rights, including technical
 modifications necessary to circumvent Effective Technological
 Measures. For purposes of this Public License, simply making
 modifications authorized by this Section
 2(a)(4) never produces Adapted Material.

 Downstream recipients.

 Offer from the Licensor – Licensed Material. Every recipient of the Licensed Material automatically
 receives an offer from the Licensor to exercise the
 Licensed Rights under the terms and conditions of this
 Public License.

 No downstream restrictions. You may not offer or impose any additional or different
 terms or conditions on, or apply any Effective
 Technological Measures to, the Licensed Material if doing
 so restricts exercise of the Licensed Rights by any
 recipient of the Licensed Material.

 No endorsement. Nothing in this Public License constitutes or may be
 construed as permission to assert or imply that You are, or
 that Your use of the Licensed Material is, connected with, or
 sponsored, endorsed, or granted official status by, the
 Licensor or others designated to receive attribution as
 provided in Section 3(a)(1)(A)(i).

 Other rights.

 Moral rights, such as the right of integrity, are not licensed
 under this Public License, nor are publicity, privacy, and/or
 other similar personality rights; however, to the extent
 possible, the Licensor waives and/or agrees not to assert any
 such rights held by the Licensor to the limited extent
 necessary to allow You to exercise the Licensed Rights, but
 not otherwise.

 Patent and trademark rights are not licensed under this Public
 License.

 To the extent possible, the Licensor waives any right to
 collect royalties from You for the exercise of the Licensed
 Rights, whether directly or through a collecting society under
 any voluntary or waivable statutory or compulsory licensing
 scheme. In all other cases the Licensor expressly reserves any
 right to collect such royalties.

Section 3 – License Conditions.

 Your exercise of the Licensed Rights is expressly made subject to
 the following conditions.

 Attribution.

 If You Share the Licensed Material (including in modified
 form), You must:

 retain the following if it is supplied by the Licensor
 with the Licensed Material:

 identification of the creator(s) of the Licensed
 Material and any others designated to receive
 attribution, in any reasonable manner requested by the
 Licensor (including by pseudonym if designated);

a copyright notice;

 a notice that refers to this Public License;

 a notice that refers to the disclaimer of warranties;

 a URI or hyperlink to the Licensed Material to the
 extent reasonably practicable;

 indicate if You modified the Licensed Material and retain
 an indication of any previous modifications; and

 indicate the Licensed Material is licensed under this
 Public License, and include the text of, or the URI or
 hyperlink to, this Public License.

 You may satisfy the conditions in Section
 3(a)(1) in any reasonable manner based on
 the medium, means, and context in which You Share the Licensed
 Material. For example, it may be reasonable to satisfy the
 conditions by providing a URI or hyperlink to a resource that
 includes the required information.

 If requested by the Licensor, You must remove any of the
 information required by Section
 3(a)(1)(A) to the extent reasonably
 practicable.

 If You Share Adapted Material You produce, the Adapter's
 License You apply must not prevent recipients of the Adapted
 Material from complying with this Public License.

Section 4 – Sui Generis Database Rights.

 Where the Licensed Rights include Sui Generis Database Rights that
 apply to Your use of the Licensed Material:

 for the avoidance of doubt, Section
 2(a)(1) grants You the right to extract,
 reuse, reproduce, and Share all or a substantial portion of the
 contents of the database;

 if You include all or a substantial portion of the database
 contents in a database in which You have Sui Generis Database
 Rights, then the database in which You have Sui Generis Database
 Rights (but not its individual contents) is Adapted Material; and

 You must comply with the conditions in Section
 3(a) if You Share all or a substantial portion
 of the contents of the database.

 For the avoidance of doubt, this Section
 4 supplements and does not replace Your
 obligations under this Public License where the Licensed Rights
 include other Copyright and Similar Rights.

 Section 5 – Disclaimer of Warranties and Limitation of Liability.

 Unless otherwise separately undertaken by the Licensor, to the
 extent possible, the Licensor offers the Licensed Material as-is
 and as-available, and makes no representations or warranties of
 any kind concerning the Licensed Material, whether express,
 implied, statutory, or other. This includes, without limitation,
 warranties of title, merchantability, fitness for a particular
 purpose, non-infringement, absence of latent or other defects,
 accuracy, or the presence or absence of errors, whether or not
 known or discoverable. Where disclaimers of warranties are not
 allowed in full or in part, this disclaimer may not apply to
 You.

 To the extent possible, in no event will the Licensor be liable
 to You on any legal theory (including, without limitation,
 negligence) or otherwise for any direct, special, indirect,
 incidental, consequential, punitive, exemplary, or other losses,
 costs, expenses, or damages arising out of this Public License
 or use of the Licensed Material, even if the Licensor has been
 advised of the possibility of such losses, costs, expenses, or
 damages. Where a limitation of liability is not allowed in full
 or in part, this limitation may not apply to You.

 The disclaimer of warranties and limitation of liability provided
 above shall be interpreted in a manner that, to the extent
 possible, most closely approximates an absolute disclaimer and
 waiver of all liability.

Section 6 – Term and Termination.

 This Public License applies for the term of the Copyright and
 Similar Rights licensed here. However, if You fail to comply with
 this Public License, then Your rights under this Public License
 terminate automatically.

 Where Your right to use the Licensed Material has terminated
 under Section 6(a), it reinstates:

 automatically as of the date the violation is cured, provided
 it is cured within 30 days of Your discovery of the violation;
 or

upon express reinstatement by the Licensor.

 For the avoidance of doubt, this Section
 6(b) does not affect any right the Licensor
 may have to seek remedies for Your violations of this Public
 License.

 For the avoidance of doubt, the Licensor may also offer the
 Licensed Material under separate terms or conditions or stop
 distributing the Licensed Material at any time; however, doing so
 will not terminate this Public License.

 Sections 1, 5,
 6, 7, and
 8 survive termination of this Public License.

Section 7 – Other Terms and Conditions.

 The Licensor shall not be bound by any additional or different
 terms or conditions communicated by You unless expressly agreed.

 Any arrangements, understandings, or agreements regarding the
 Licensed Material not stated herein are separate from and
 independent of the terms and conditions of this Public License.

Section 8 – Interpretation.

 For the avoidance of doubt, this Public License does not, and
 shall not be interpreted to, reduce, limit, restrict, or impose
 conditions on any use of the Licensed Material that could lawfully
 be made without permission under this Public License.

 To the extent possible, if any provision of this Public License is
 deemed unenforceable, it shall be automatically reformed to the
 minimum extent necessary to make it enforceable. If the provision
 cannot be reformed, it shall be severed from this Public License
 without affecting the enforceability of the remaining terms and
 conditions.

 No term or condition of this Public License will be waived and no
 failure to comply consented to unless expressly agreed to by the
 Licensor.

 Nothing in this Public License constitutes or may be interpreted
 as a limitation upon, or waiver of, any privileges and immunities
 that apply to the Licensor or You, including from the legal
 processes of any jurisdiction or authority.

About Creative Commons

 Creative Commons is not a party to its public licenses.
 Notwithstanding, Creative Commons may elect to apply one of its
 public licenses to material it publishes and in those instances will
 be considered the "Licensor." The text of the Creative Commons
 public licenses is dedicated to the public domain under the
 CC0 Public Domain Dedication.
 Except for the limited purpose of indicating that material is shared
 under a Creative Commons public license or as otherwise permitted by
 the Creative Commons policies published at
 creativecommons.org/policies, Creative
 Commons does not authorize the use of the trademark "Creative
 Commons" or any other trademark or logo of Creative Commons without
 its prior written consent including, without limitation, in
 connection with any unauthorized modifications to any of its public
 licenses or any other arrangements, understandings, or agreements
 concerning use of licensed material. For the avoidance of doubt,
 this paragraph does not form part of the public licenses.

 Creative Commons may be contacted at
 creativecommons.org.

 Creative Commons is the nonprofit behind the open licenses and other
 legal tools that allow creators to share their work. Our legal tools
 are free to use.

Learn more about our work

 Learn more about CC Licensing

Support our work

Use the license for your own material.

Licenses List

Public Domain List

 Creative Commons

 submit

Who we are

What we do

Blog

Support us

 Store

Contact

Privacy

Policies

Terms

Contact Us

 Creative Commons 

 PO Box 1866, Mountain View, 

 CA 94042

 info@creativecommons.org

 Bluesky

 Mastodon

 LinkedIn

Subscribe to our newsletter
 Subscribe

 Except where otherwise noted, content
 on this site is licensed under a
 Creative Commons Attribution 4.0 International license. Icons by
 Font Awesome.
```

## 9. Final notebook execution and artifact inspection

Final `nbconvert --execute` returned0 after495.01582622528076s with10ordered non-null execution counts1-10,7embeddedPNGfigures and no error outputs. The actual kernel executable printed in the notebook is the shared `research-runtime/Scripts/python.exe`, Python3.11.11/NumPy2.4.6. Fresh complete local reconstruction took423.731932100025s; active independent uniform comparison took14.499489100067876s. The saved output was again judged successfully. Shared Jupyter's nbconvert launcher resolves through Anaconda, while the explicit research kernel runs the recorded shared interpreter. The Windows Proactor/zmq selector and local TCP transport warnings are archived in the log; they did not interrupt execution or change numerical results.

All7extractedPNGfigures were visually inspected individually: original phase/components/coherence/mask/incidence/weather, complete temporal graph/baselines, raw/repaired triangle counts and original/fresh integer paired bars, six full-frame correction/reference/error panels, full-frame quality/DEM/velocity/acceleration/eruption maps, original/fresh traces and residual diagnostics, and active weighted/uniform difference maps. Labels, units, reference/fresh identities and colorbars are visible, with no blank/clipped panels. Exact-zero fresh-minus-original is explicitly shown on a nonzero display scale, not fabricated noise. Full-frame masking is the declared observation-derived mask; the separately shown temporal-model residual is explicitly not ground-truth error.

Weighted and uniform mean temporal coherence on the common observation mask are.994020402431488 and.9939572215080261. Fresh weighted-minus-uniform apparent-LOS RMS difference is.0003745961163351199m. These are same-acquisition numerical comparisons, not independent physical accuracy measurements. The first no-PNG execution was superseded by the full rerun, not by fabricated figure outputs. Canonical notebook retains the actual complete executed outputs. Temporary executed copies/inspection PNGs/logs remain outside final task.

The final summary-consumer test passed1test in9.63s after the35test full suite. This checks the separate fresh eleven-metric summary against independently reevaluated output. No code changes followed these successful tests; subsequent changes are documentation/provenance only.

### Original source and execution evidence hashes

| Archive-relative file | Bytes | SHA256 |
|---|---|---|
| `MintPy.zip` | 1303495 | `ec8ede62bdc0adb1fe5173f20fc121e4b69ddd7a178feaf36b9618044c1afddd` |
| `Fernandina.complete.tar.xz` | 750610904 | `51086497da5cda0add780fc2858db51645693c9c53b1fb8339aac847f348d84c` |
| `Yunjun2019-preprint.pdf` | 15004489 | `b81729a8846b87dfd82397d852755e1349b055541f152acdf763aae63d7afe65` |
| `CC-BY-4.0.html` | 49836 | `3219ecae225e999e633f740a98895dfb9885766770bfefdf94a7cc903d5e6a85` |
| `oracle_capture.py` | 7362 | `4104b6564faebec29774383a2dcdba932493277cb3db569e89f171e1e91271c1` |
| `oracle_capture.log` | 31056 | `fa036a3093d68175f61806bf6eb21dc4c80319cd6501c437b2e7bce9860853b3` |
| `oracle/result.json` | 1325 | `e4f250f38f1763e45f34c44fba6e033ca131cea290f6de71ef5e5d6bd4a23850` |
| `oracle_rerun.json` | 395 | `ca31baf57c50965bbe4b67224039065107be7f33dc3a97ce8c457568534f948a` |
| `oracle_uniform.json` | 355 | `9dd787bfff29039801ef0e743e7bcd0462b4dc266051ee178ea935561796205c` |
| `lasso_equivalence_audit.json` | 21828 | `b7516441ede60bdf58c48e55335b665e445d911e9ef5577d880ae6f9e7d0ebb6` |
| `full_closure_parity.json` | 367 | `fdc00c02fd19edfd54a2cb0150c98585bcc34a325f9e9a141629d87855e95367` |
| `cleaned_status.json` | 757 | `c8d1fddefc848677b60a948265015f3226d57e889dbd11cbe5e7b8ad6a5df512` |
| `pytest-final.log` | 8134 | `1438a296ce1464948717bc145068bf9dfc55ea74133a07d6bf919128f171a40a` |
| `pytest-measurements.log` | 1200 | `b8846a0573f8c72c351034e13211723ddb7612484e3ee01752003ca9666a26e7` |
| `notebook_status.json` | 831 | `dd25970250b7aa19170af4caa079b9d9a70b5219e13c08dec8aeafc385ffb6a3` |
| `notebook.log` | 837 | `1e0e680f9c5636e42d2bd5c40e9b6b7fe3668cdf7ee814841b9c95e6a40f980d` |


## 10. Independent maintainer review

The coordinating maintainer reviewed the implemented preprocessing, acquisition/design matrices, closure correction, weighted inversion, atmosphere/ramp/DEM correction, submitted-array conventions, independent Judge, and their README/plan interfaces. The review checked signs, spatial/time reference conventions, units, acceleration factor, strict event inequalities, original numerical precision and selected regularization rather than signatures alone. The nine individual rounded LASSO discrepancies remain explicitly disclosed; complete propagated phase and final submitted arrays match the original run.

A fresh standalone Judge invocation from outside the task working directory passed all five declared quantities and the public observation-consistency criterion. The report is privately archived as `../coordinator/insar-root-independent-judge.json`. The structural review found no errors. All seven actual embedded notebook figures were inspected: observation/geometry/mask, complete network, closure correction, correction-stage maps, fitted physical parameters, temporal traces/residuals, and weighted/uniform comparison. Figure scales/units and labels distinguish original reproduced results, fresh results, model residual and absence of independent ground truth. No numerical thresholds were changed by this review. The final inventory/hashes are privately indexed in `../coordinator/insar-final-review.json`.
